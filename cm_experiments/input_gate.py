"""Serving-time input gate g_input and deployed latency (Tables S23, S37; Figure S16).

The released lifecycle has no serving-time input gate: its admission lock
records an exact prepared-schema check and the deployed rule executes on every
input. This runner adds g_input outside the runtime, as main Eq. (4) and
Algorithm S2 describe it -- a static rule frozen before any evaluation label is
read, using only pre-execution compatibility information -- and measures what
the gated rule q~_M(x) = g_input(x) q_M(x) + [1 - g_input(x)] q_C(x) does.

Registered gate (declared in the plan; an implementation choice, since the
manuscript does not define the compatibility variables):

* unit: a complete graph of an independent-graph benchmark, or a buffered core
  of a single-graph benchmark (prepared/clusters.npz; Table S37's deployment
  episodes); a core's gated prediction mixes the full-graph q_M and q_C rows;
* compatibility vector: log1p of the unit's node count, weighted degree volume,
  mean and maximum weighted degree, followed by the initial label-free measurement
  y_0 of the declared sensor profile (pooled over the unit, mean over the
  ensemble members), computed from H_0 before any controlled transition;
* envelope: per-coordinate median and 1.4826 MAD over the D_explore units; the
  score is the largest absolute robust z-score; g_input = 1 iff the score is at
  most the (1 - alpha) empirical quantile of the D_explore scores.

On the evaluation role the runner executes q_M and q_C with the released
``execute_policy`` and measures, for every unit, the gate decision, the
synchronized wall time of the gate, controller and comparator after warm-up,
peak GPU memory of the deployed path, and the paired loss of q_M against q_C.
It reports the input-fallback probability, the deployed wall-time overhead
over q_C, peak GPU and host memory, and the sequence-level false-safe and
harm-miss rates of Table S37. G_admit is read from the parent's admission: the
parent admitted the ungated rule, so the gated rule's admission is the parent's
decision, not a new admission test.
"""
from __future__ import annotations

import math
import resource
import sys
import time
from pathlib import Path

import numpy as np

from .artifacts import sha256, write_json
from .predictability import (ExperimentError, ParentRun, RunDirectory, create_plan, export_record_set, open_plan,
                             quantile, read_csv, read_json, role_labels, role_units, summary_row, write_csv)

GATE_SCHEMA = "cm-static-input-gate-1"
STATIC_NAMES = ("log1p_nodes", "log1p_degree_volume", "log1p_mean_degree", "log1p_max_degree")
GIB = float(2**30)
RAW_EXPORTS = ("raw/gate.json", "raw/development_units.csv", "raw/deployment_units.csv", "raw/memory.json",
               "raw/timing.csv", "raw/parity.json")
DEFINITIONS = {
    "fallback_probability": "fraction of evaluation units (graphs, or buffered cores of a single graph) returned to "
                            "q_C by the static input gate g_input (a numerical controller failure served by q_C is "
                            "not a gate event)",
    "deployed_wall_overhead_percent": "100 x (sum over units of the gated rule's synchronized wall time, gate "
                                      "included / sum of q_C's wall time - 1), post-warm-up means",
    "gpu_memory_gib": "peak allocated GPU memory of the deployed path over the evaluation units (GiB)",
    "cpu_memory_gib": "peak resident host memory of the serving process (ru_maxrss; GiB)",
    "false_safe_rate": "units passed by g_input whose paired complete-policy loss of q_M exceeds q_C's, over the "
                       "units passed by g_input",
    "harm_miss_rate": "harmful units (q_M loss above q_C's) passed by g_input, over all harmful units",
    "expected_work_overhead_percent": "100 x (sum of the gated rule's deployed multiply-add products, gate included / "
                                      "sum of q_C's - 1)",
    "stepwise_support_rejection": "fraction of q_M's decision sites on gate-passed units resolved by a recovery origin",
    "development_fallback": "in-sample fraction of D_explore units outside the frozen envelope",
}


# ----------------------------------------------------------------------------- the gate (NumPy only)
def fit_gate(vectors, names, *, alpha, unit_ids):
    v = np.asarray(vectors, dtype=np.float64)
    if v.ndim != 2 or len(v) < 2 or v.shape[1] != len(names) or not np.isfinite(v).all():
        raise ExperimentError("the gate needs at least two finite development compatibility vectors")
    if not 0 < alpha < 1:
        raise ExperimentError("the gate's development rejection level must lie in (0, 1)")
    center = np.median(v, axis=0)
    mad = 1.4826 * np.median(np.abs(v - center), axis=0)
    scale = np.maximum(mad, 1e-6 * np.maximum(1., np.abs(center)))
    scores = np.max(np.abs(v - center) / scale, axis=1)
    threshold = quantile(scores, 1 - alpha)
    return {"schema": GATE_SCHEMA, "names": list(names), "center": center.tolist(), "scale": scale.tolist(),
            "threshold": threshold, "alpha": float(alpha), "development_units": [str(u) for u in unit_ids],
            "development_scores": scores.tolist(),
            "score": "max_j |v_j - median_j| / max(1.4826 MAD_j, 1e-6 max(1, |median_j|))",
            "decision": "g_input = 1 iff score <= threshold; threshold = (1 - alpha) empirical quantile (higher) of "
                        "the D_explore scores"}


def gate_decision(gate, vector):
    v = np.asarray(vector, dtype=np.float64)
    if v.shape != (len(gate["names"]),):
        raise ExperimentError("compatibility vector does not match the frozen gate")
    if not np.isfinite(v).all():
        return float("inf"), 0
    score = float(np.max(np.abs(v - np.asarray(gate["center"])) / np.asarray(gate["scale"])))
    return score, int(score <= gate["threshold"])


def gate_identity(gate):
    from .artifacts import canonical
    return canonical(gate)


# ----------------------------------------------------------------------------- compatibility vectors (PyTorch)
def measurement_bank(graph, model, config, profile, mask=None):
    """The runtime's MeasurementBank with sensor_bank's exact arguments, plus a pooling mask."""
    from cross_morse.core import MeasurementBank
    from cross_morse.tasks import task_spec
    s = config["sensor"]
    return MeasurementBank(graph, model.hidden_dim, bands=s["bands"], degree=s["degree"], sketch_dim=s["sketch_dim"],
                           seed=config["seed"], profile=profile, mask=mask,
                           task=getattr(model, "task", task_spec(config)), plant=model,
                           sensing_period=s.get("sensing_period", 1))


def compatibility_vectors(models, x, graph, config, *, profile="full", cores=None, bank_factory=None):
    """({unit: vector}, names) from graph statistics and the initial measurement; no label, no control step."""
    import torch
    factory = bank_factory or measurement_bank
    L = int(config["model"]["depth"])
    degree = graph.degree.double()
    if cores is None:
        units = {"graph": None}
    else:
        units = {int(c): torch.as_tensor(np.asarray(cores) == c, device=graph.device) for c in np.unique(cores)}
    measured, schema = {u: [] for u in units}, None
    for model in models:
        model.bind_graph(graph)
        h0 = model.encode(x)
        logits0 = model.logits(h0)
        for unit, mask in units.items():
            bank = factory(graph, model, config, profile, mask)
            bank.prepare(h0, logits0)
            measured[unit].append(bank.measure(h0, logits0, 0, L).double().cpu().numpy())
            schema = list(bank.names)
    vectors = {}
    for unit, mask in units.items():
        d = degree if mask is None else degree[mask]
        statics = [math.log1p(len(d)), math.log1p(float(d.sum())), math.log1p(float(d.mean())), math.log1p(float(d.max()))]
        vectors[unit] = np.concatenate([statics, np.mean(np.stack(measured[unit]), axis=0)])
    return vectors, list(STATIC_NAMES) + [f"y0_{name}" for name in schema]


def unit_cores(data, episode, collection):
    """None for a complete graph; the prepared core of every node of a single transductive graph."""
    return None if collection else np.asarray(data.cluster_ids)


def fit_development_gate(data, models, config, device, *, profile, alpha, max_graphs=0, bank_factory=None):
    """Freeze g_input on D_explore units before any evaluation label is read."""
    import torch
    from cross_morse.common import precision_context
    from cross_morse.runtime import graph_operator, is_graph_collection
    collection = is_graph_collection(data)
    vectors, ids, names = [], [], None
    for eid, episode in role_units(data, "explore", max_graphs):
        graph = graph_operator(episode, device)
        with torch.no_grad(), precision_context(config, torch.device(device)):
            found, schema = compatibility_vectors(models, episode.x.to(device), graph, config, profile=profile,
                                                  cores=unit_cores(data, episode, collection),
                                                  bank_factory=bank_factory)
        if names is not None and schema != names:
            raise ExperimentError("compatibility vector schema changed between development units")
        names = schema
        for unit, vector in sorted(found.items(), key=lambda item: str(item[0])):
            vectors.append(vector)
            ids.append(f"graph:{eid}" if collection else f"core:{unit}")
    if not vectors:
        raise ExperimentError("D_explore holds no development unit for the gate")
    return fit_gate(np.stack(vectors), names, alpha=alpha, unit_ids=ids)


# ----------------------------------------------------------------------------- execution helpers (PyTorch)
CONTROLLED_ROUTES = ("phase", "fixed_sequence", "mpc_ol", "mpc_rh", "one_step", "radius")


def policy_context(parent, candidate):
    """Members, controller and reference exactly as Pipeline.predict_candidate binds them."""
    from cross_morse.learning import Controller
    profile = candidate.get("profile", "full")
    profile = "full" if profile == "none" else profile
    controller = (Controller.load(parent.root / f"fit/{profile}/controller.pt", "cpu")
                  if candidate["route"] in CONTROLLED_ROUTES else None)
    return {"candidate": candidate, "controller": controller,
            "members": parent.seeds[:candidate["ensemble"]]}


def ensemble_execute(models, x, graph, config, context, reference):
    """Ensemble mean of member executions; a numerical policy failure returns None with its trace."""
    from cross_morse.policies import PolicyExecutionError, execute_policy
    from .protocol import deployed_work
    probabilities, traces = [], []
    for seed in context["members"]:
        try:
            p, trace = execute_policy(models[seed], x, graph, config, context["candidate"], context["controller"],
                                      reference)
        except PolicyExecutionError as error:
            return None, {"failure": error.reason, "traces": traces + [error.trace]}
        probabilities.append(np.asarray(p, dtype=np.float64))
        traces.append(trace)
    work = sum(deployed_work(t) for t in traces)
    peak = max((t.get("peak_cuda_allocated_bytes") or 0 for t in traces), default=0)
    return (np.mean(np.stack(probabilities), axis=0).astype(np.float32),
            {"traces": traces, "work": work, "peak_cuda_bytes": peak})


def synchronized(device):
    import torch
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def host_peak_bytes():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(usage if sys.platform == "darwin" else usage * 1024)


def unit_losses(probabilities, labels, ids, unit_of_row, config):
    """Mean task loss of each unit's labelled rows (graph or core)."""
    from cross_morse.statistics import cluster_summary
    from cross_morse.tasks import sample_losses, task_spec
    values = sample_losses(np.asarray(probabilities)[ids], labels, task_spec(config))
    valid = np.isfinite(values)
    means, units, _ = cluster_summary(values[valid], np.asarray(unit_of_row)[valid])
    return {int(u): float(m) for u, m in zip(units, means)}


# ----------------------------------------------------------------------------- plan / run
def plan_input_gate(parent, output, *, role="test", alpha=.05, profile="full", warmup=3, repetitions=10,
                    max_graphs=0, allow_fixture=False):
    parent = parent if isinstance(parent, ParentRun) else ParentRun(parent)
    if parent.synthetic and not allow_fixture:
        raise ExperimentError("a synthetic fixture parent requires --allow-fixture")
    checked = parent.check("admit")
    parent.require_role(role)
    if profile not in parent.config["sensor"]["profiles"]:
        raise ExperimentError(f"gate profile {profile!r} is not a sensor profile of the parent")
    if not 0 < alpha < 1 or type(warmup) is not int or warmup < 1 or type(repetitions) is not int or repetitions < 3:
        raise ExperimentError("declare 0 < alpha < 1, at least one warm-up and three measured repetitions (E17)")
    if type(max_graphs) is not int or max_graphs < 0:
        raise ExperimentError("max_graphs must be a nonnegative integer")
    admission = parent.admission()
    candidates = parent.candidates()
    body = {"evaluation_role": role, "gate": {"alpha": float(alpha), "profile": profile, "fit_role": "explore",
                                              "variables": "graph statistics and the initial measurement y_0"},
            "confirmed": candidates[admission["confirmed_index"]], "comparator": candidates[admission["comparator_index"]],
            "admission": admission, "timing": {"warmup": warmup, "repetitions": repetitions,
                                               "protocol": "E17: synchronized, alternating order, post-warm-up means"},
            "max_graphs": max_graphs, "parent_check": checked, "definitions": DEFINITIONS}
    return create_plan("input_gate", parent, output, body)


def run_input_gate(plan_path, *, device="cpu", trust=False):
    import torch
    from cross_morse.common import precision_context
    from cross_morse.data import load_data
    from cross_morse.provenance import sha256_state_dict
    from cross_morse.runtime import graph_operator, is_graph_collection
    from cross_morse.training import load_plant

    plan, parent = open_plan(plan_path, "input_gate")
    parent.verify("evaluate" if plan["evaluation_role"] == "test" else "admit")
    run = RunDirectory(plan_path, plan, parent, device=device, trust=trust)
    root, config = parent.root, parent.config
    data = load_data(root).protect_labels()
    collection = is_graph_collection(data)
    confirmed, comparator = policy_context(parent, plan["confirmed"]), policy_context(parent, plan["comparator"])
    seeds = sorted(set(confirmed["members"]) | set(comparator["members"]))
    models = {s: load_plant(data, config, root / "train", s, device) for s in seeds}
    frozen = {s: sha256_state_dict(m) for s, m in models.items()}
    gate_models = [models[s] for s in confirmed["members"]]
    reference = dict(np.load(root / "explore/nominal_reference.npz", allow_pickle=False))
    gate = fit_development_gate(data, gate_models, config, device, profile=plan["gate"]["profile"],
                                alpha=plan["gate"]["alpha"], max_graphs=plan["max_graphs"])
    write_json(run.raw("gate.json"), dict(gate, frozen_before_evaluation_labels=True, parent=parent.name))
    write_csv(run.raw("development_units.csv"),
              [{"unit_id": u, "score": repr(s), "g_input": int(s <= gate["threshold"])}
               for u, s in zip(gate["development_units"], gate["development_scores"])])
    role, timing = plan["evaluation_role"], plan["timing"]
    units, timings, predictions = [], [], {}
    ensemble_seeds = "+".join(map(str, confirmed["members"]))
    for eid, episode in role_units(data, role, plan["max_graphs"]):
        ids = np.asarray(episode.splits[role], dtype=np.int64)
        if not len(ids):
            continue
        graph = graph_operator(episode, device)
        x = episode.x.to(device)
        cores = unit_cores(data, episode, collection)

        def gate_pass():
            with torch.no_grad(), precision_context(config, torch.device(device)):
                vectors, _ = compatibility_vectors(gate_models, x, graph, config, profile=plan["gate"]["profile"],
                                                   cores=cores)
            return {u: gate_decision(gate, v) for u, v in vectors.items()}

        def controller_pass():
            return ensemble_execute(models, x, graph, config, confirmed, reference)

        def comparator_pass():
            return ensemble_execute(models, x, graph, config, comparator, reference)

        # One untimed deployed pass fixes predictions, decisions, work and memory.
        if str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(torch.device(device))
        decisions = gate_pass()
        gate_peak = torch.cuda.max_memory_allocated(torch.device(device)) if str(device).startswith("cuda") else None
        p_m, m_trace = controller_pass()
        p_c, c_trace = comparator_pass()
        if p_c is None:
            raise ExperimentError(f"the locked comparator failed on input {eid}; the gate cannot fall back")
        gate_work = _gate_work(gate_models, x, graph, config, plan["gate"]["profile"], cores)
        seconds = {"gate": [], "controller": [], "comparator": []}
        calls = {"gate": gate_pass, "controller": controller_pass, "comparator": comparator_pass}
        for repeat in range(timing["warmup"] + timing["repetitions"]):
            order = ("gate", "controller", "comparator") if repeat % 2 == 0 else ("comparator", "controller", "gate")
            for name in order:
                synchronized(device)
                began = time.perf_counter()
                calls[name]()
                synchronized(device)
                elapsed = time.perf_counter() - began
                timings.append({"input_id": eid, "repeat": repeat, "warmup": int(repeat < timing["warmup"]),
                                "branch": name, "seconds": repr(elapsed)})
                if repeat >= timing["warmup"]:
                    seconds[name].append(elapsed)
        mean = {k: float(np.mean(v)) for k, v in seconds.items()}
        # g_input of every prediction row; a numerical controller failure serves q_C for the whole input.
        row_core = None if collection else np.asarray(cores)
        if p_m is None:
            served = np.zeros(len(p_c), dtype=bool)
        elif collection:
            served = np.full(len(p_c), bool(decisions["graph"][1]))
        else:
            served = np.asarray([bool(decisions[int(c)][1]) for c in row_core])
        gated = np.where(served[:, None], p_m if p_m is not None else p_c, p_c).astype(np.float32)
        any_pass, any_fallback = bool(served.any()), not bool(served.all())
        deployed_seconds = (mean["gate"] + (mean["controller"] if any_pass else 0.)
                            + (mean["comparator"] if any_fallback else 0.))
        deployed_work = gate_work + (m_trace.get("work", 0.) if any_pass else 0.) + (c_trace["work"] if any_fallback else 0.)
        peaks = [v for v in (gate_peak, m_trace.get("peak_cuda_bytes") if any_pass else None,
                             c_trace.get("peak_cuda_bytes") if any_fallback else None) if v is not None]
        sites = recoveries = 0
        if p_m is not None:
            for trace in m_trace["traces"]:
                sites += len(trace["origins"])
                recoveries += sum(str(o).startswith("recovery") for o in trace["origins"])
        # Every prediction of this input exists before its evaluation labels are read.
        labels = role_labels(data, episode, role, ids)
        row_units = np.full(len(ids), eid) if collection else row_core[ids]
        loss_c = unit_losses(p_c, labels, ids, row_units, config)
        loss_m = unit_losses(p_m, labels, ids, row_units, config) if p_m is not None else {}
        loss_d = unit_losses(gated, labels, ids, row_units, config)
        global_ids = ids if episode.global_sample_ids is None else np.asarray(episode.global_sample_ids)[ids]
        for name, value in (("observation_id", global_ids), ("unit_id", row_units), ("y_true", labels),
                            ("qC", p_c[ids]), ("qD_gated", gated[ids]),
                            ("qM", p_m[ids] if p_m is not None else np.full_like(p_c[ids], np.nan))):
            predictions.setdefault(name, []).append(np.asarray(value))
        share = 1. / len(decisions)
        for unit, (score, g) in sorted(decisions.items(), key=lambda item: str(item[0])):
            key = eid if collection else unit
            rows = np.ones(len(p_c), dtype=bool) if collection else row_core == unit
            g_effective = int(bool(served[rows].all()))
            units.append({
                "dataset": parent.dataset, "seed": ensemble_seeds,
                "unit_id": f"graph:{eid}" if collection else f"core:{unit}", "input_id": eid,
                "g_input": int(g), "served_controller": g_effective, "gate_score": repr(score),
                "comparator_work": repr(c_trace["work"] * share), "controller_work": repr(m_trace.get("work", 0.) * share),
                "gate_work": repr(gate_work * share), "actual_work": repr(deployed_work * share),
                "fallback_predictions_equal": "" if g_effective else int(np.array_equal(gated[rows], p_c[rows])),
                "comparator_loss": repr(loss_c[key]) if key in loss_c else "",
                "controller_loss": repr(loss_m[key]) if key in loss_m else "",
                "deployed_loss": repr(loss_d[key]) if key in loss_d else "",
                "harm": repr(loss_m[key] - loss_c[key]) if key in loss_m and key in loss_c else "",
                "controller_failure": "" if p_m is not None else m_trace.get("failure", ""),
                "gate_seconds": repr(mean["gate"] * share), "controller_seconds": repr(mean["controller"] * share),
                "comparator_seconds": repr(mean["comparator"] * share),
                "deployed_seconds": repr(deployed_seconds * share),
                "peak_gpu_bytes": max(peaks) if peaks else "",
                "decision_sites": sites if g_effective else 0, "recovery_sites": recoveries if g_effective else 0})
    if not units:
        raise ExperimentError("the evaluation role holds no scored unit")
    if any(sha256_state_dict(m) != frozen[s] for s, m in models.items()):
        raise RuntimeError("a frozen predictor changed during serving measurements")
    parent.verify("evaluate" if role == "test" else "admit")
    write_csv(run.raw("deployment_units.csv"), units)
    write_csv(run.raw("timing.csv"), timings)
    stacked = {k: np.concatenate(v) for k, v in predictions.items()}
    np.savez_compressed(run.raw("predictions.npz"), **stacked)
    parity = {"scope": "the served comparator branch against the parent's sealed test-role q_C predictions"}
    sealed = root / "final/qC_probabilities.npy"
    if role == "test" and sealed.is_file():
        stored = np.load(sealed, mmap_mode="r", allow_pickle=False)
        parity["qC_max_abs_deviation"] = float(np.max(np.abs(np.asarray(stored[stacked["observation_id"]], dtype=np.float64)
                                                             - stacked["qC"])))
        parity["sealed_qC_sha256"] = sha256(sealed)
    write_json(run.raw("parity.json"), parity)
    write_json(run.raw("memory.json"), {"host_peak_bytes": host_peak_bytes(), "device": str(device),
                                        "host_scope": "ru_maxrss of the serving process, data loading included",
                                        "gpu_scope": "max over units of the peak allocated bytes of the gate and the "
                                                     "executed branches (per-member peaks from the runtime traces)"})
    summary = summarize_input_gate([run.root])
    summary["truncated"] = bool(plan["max_graphs"])
    return run.finish(summary)


def _gate_work(models, x, graph, config, profile, cores):
    """Instrumented multiply-add products of one gate evaluation (encoder, head and sensing)."""
    import torch
    from cross_morse.common import precision_context
    banks = []

    def counting(graph_, model, config_, profile_, mask):
        bank = measurement_bank(graph_, model, config_, profile_, mask)
        banks.append(bank)
        return bank
    for model in models:
        model.reset_counters()
    graph.reset_counters()
    with torch.no_grad(), precision_context(config, x.device):
        compatibility_vectors(models, x, graph, config, profile=profile, cores=cores, bank_factory=counting)
    return float(graph.sparse_multiply_adds + sum(m.counters["dense_multiply_adds"] for m in models)
                 + sum(int(b.counters.get("dense_multiply_adds", 0)) for b in banks))


def summarize_input_gate(dirs):
    """Recompute the input-gate values from the sealed unit table (no PyTorch)."""
    rows, notes = [], {}
    pooled = {}
    for d in dirs:
        d = Path(d)
        exp, plan = read_json(d / "experiment.json"), read_json(d / "plan.json")
        entry = pooled.setdefault(exp["parent"]["dataset"], {"units": [], "development": [], "memory": [],
                                                             "admitted": [], "device": set()})
        entry["units"] += read_csv(d / "raw/deployment_units.csv")
        entry["development"] += read_csv(d / "raw/development_units.csv")
        memory = read_json(d / "raw/memory.json")
        entry["memory"].append(memory)
        entry["device"].add(memory["device"])
        entry["admitted"].append(bool((plan.get("admission") or {}).get("passed")))
    for dataset, entry in sorted(pooled.items()):
        keys = {"dataset": dataset}
        units = entry["units"]
        g = np.array([int(u["g_input"]) for u in units])
        served = np.array([int(u.get("served_controller", u["g_input"])) for u in units])
        deployed = np.array([float(u["deployed_seconds"]) for u in units])
        reference = np.array([float(u["comparator_seconds"]) for u in units])
        harm = np.array([float(u["harm"]) if u["harm"] != "" else np.nan for u in units])
        work = np.array([float(u["actual_work"]) for u in units])
        reference_work = np.array([float(u["comparator_work"]) for u in units])
        admitted = all(entry["admitted"])
        notes[dataset] = {"units": len(units), "admitted": admitted, "runs_pooled": len(entry["admitted"]),
                          "devices": sorted(entry["device"])}
        measured = {"cpu_memory_gib": max(m["host_peak_bytes"] for m in entry["memory"]) / GIB}
        gpu = [float(u["peak_gpu_bytes"]) for u in units if u["peak_gpu_bytes"] != ""]
        if gpu:
            measured["gpu_memory_gib"] = max(gpu) / GIB
        if admitted:
            measured.update(fallback_probability=float(np.mean(1 - g)),
                            deployed_wall_overhead_percent=100. * (deployed.sum() / reference.sum() - 1.),
                            expected_work_overhead_percent=100. * (work.sum() / reference_work.sum() - 1.))
            passed = g == 1
            scored = np.isfinite(harm)
            harmful = scored & (harm > 0)
            if (passed & scored).any():
                measured["false_safe_rate"] = float(np.mean(harmful[passed & scored]))
                sites = sum(int(u["decision_sites"]) for u, p in zip(units, served) if p)
                if sites:
                    measured["stepwise_support_rejection"] = sum(int(u["recovery_sites"]) for u, p in zip(units, served) if p) / sites
            if harmful.any():
                measured["harm_miss_rate"] = float(np.mean(passed[harmful]))
        else:
            notes[dataset]["status"] = ("GLOBAL_RETURN: G_admit = 0 deploys q_C; the gate's fallback probability and the "
                                        "deployed overhead are not reported")
        development = np.array([int(u["g_input"]) for u in entry["development"]])
        measured["development_fallback"] = float(np.mean(1 - development))
        units_of = {"fallback_probability": "fraction", "deployed_wall_overhead_percent": "percent",
                    "gpu_memory_gib": "GiB", "cpu_memory_gib": "GiB", "false_safe_rate": "fraction",
                    "harm_miss_rate": "fraction", "expected_work_overhead_percent": "percent",
                    "stepwise_support_rejection": "fraction", "development_fallback": "fraction"}
        for name, value in measured.items():
            rows.append(summary_row(keys, name, value, units_of[name], DEFINITIONS[name]))
    return {"rows": rows, "notes": notes}


def export_records(results_dirs, **kwargs):
    protocol = ("static g_input frozen on D_explore (graph statistics and the initial label-free measurement; robust "
                "envelope at the declared development rejection level); q_M and q_C executed by the released runtime on "
                "the evaluation role; synchronized post-warm-up wall times, gate included; paired losses for the "
                "Table S37 rates (Tables S23, S37; Figure S16)")
    return export_record_set("input_gate", results_dirs, summarize=summarize_input_gate,
                             raw_exports=RAW_EXPORTS, protocol=protocol, **kwargs)
