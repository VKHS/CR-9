"""Negative controls and unsupported regimes (Supplementary Table S19; Figure S16d).

Four constructions in which no useful transport exists or the sensors are
corrupted are applied to the evaluation-role inputs of one sealed parent run.
The frozen confirmed policy q_M and comparator q_C are executed on the
constructed inputs by the released ``execute_policy``, behind the static input
gate g_input of ``cm_experiments.input_gate`` (frozen on unmodified D_explore
inputs before any evaluation label is read). Per evaluation unit (a complete
graph, or a buffered core of a single graph) the runner records the gate
decision, whether the served controller executed any non-nominal action, and
the paired loss of q_M against q_C before the fallback.

Registered constructions (declared in the plan; implementation choices, since
Table S19 names the constructions only):

``graph irrelevant``  Y independent of A given X on independent realizations: every
                      input's edges are replaced by a degree-preserving configuration-
                      model rewiring drawn independently of features and labels;
``misleading graph``  edges anti-aligned with labels: every edge joining two labelled
                      rows of the same class is rewired to a uniformly drawn labelled row
                      of another class (multiclass node-label tasks only; the labels shape
                      the input and are never shown to a policy);
``sensor bias``       offset/heteroskedastic innovation corruption: every sensed coordinate
                      y_j of every measurement (the gate's, the controller's and the
                      comparator's) becomes y_j + s_j (delta + sigma xi (1 + |z_j|)), s_j and
                      z_j the fitted observer's robust scale and z-score, xi standard normal;
``support shift``     unseen degree strata: uniformly drawn new edges multiply the edge
                      count by the declared factor.

The record reports, per construction: residual action activity (the fraction of
units served by the controller that executed a non-nominal action), the input-
fallback fraction and the worst pre-fallback harm (the largest paired loss of
the ungated q_M minus q_C, nats).
"""
from __future__ import annotations

import contextlib
import dataclasses
from pathlib import Path

import numpy as np

from .artifacts import sha256, verify, write_json
from .predictability import (ExperimentError, ParentRun, RunDirectory, create_plan, export_record_set, open_plan,
                             read_csv, read_json, role_labels, role_units, summary_row, write_csv)

CONTROLS = ("graph irrelevant", "misleading graph", "sensor bias", "support shift")
RAW_EXPORTS = ("raw/units.csv", "raw/constructions.json", "raw/gate.json")
HISTORY_COORDINATES = ("relative_depth", "normalized_mass", "normalized_hops", "spectral_age_fraction")
DEFINITIONS = {
    "residual_activity": "fraction of evaluation units (x realizations) served by the controller (g_input = 1) "
                         "on which some member executed a non-nominal action",
    "input_fallback": "fraction of evaluation units (x realizations) returned to q_C by g_input (a numerical "
                      "controller failure served by q_C is not a gate event)",
    "worst_pre_fallback_harm": "largest paired loss of the ungated confirmed policy q_M minus q_C over evaluation "
                               "units (x realizations), before the input fallback (nats)",
}


# ----------------------------------------------------------------------------- constructions (NumPy only)
def undirected_pairs(edge_index, weights=None):
    """Unique (i < j) pairs of a symmetric edge list, with the weight of the i->j entry."""
    e = np.asarray(edge_index, dtype=np.int64)
    keep = e[0] < e[1]
    pairs = e[:, keep].T
    w = None if weights is None else np.asarray(weights, dtype=np.float64)[keep]
    codes = pairs[:, 0] * (int(e.max(initial=0)) + 1) + pairs[:, 1]
    _, first = np.unique(codes, return_index=True)
    first = np.sort(first)
    return pairs[first], (None if w is None else w[first])


def _normalize(a, b, n):
    keep = a != b
    lo, hi = np.minimum(a[keep], b[keep]), np.maximum(a[keep], b[keep])
    codes = np.unique(lo.astype(np.int64) * n + hi)
    return np.stack([codes // n, codes % n], axis=1)


def configuration_rewire(pairs, n, rng):
    """Degree-preserving random graph; self-loops and repeated pairs of the stub matching are dropped."""
    degree = np.bincount(pairs.ravel(), minlength=n)
    stubs = np.repeat(np.arange(n, dtype=np.int64), degree)
    rng.shuffle(stubs)
    if len(stubs) % 2:
        stubs = stubs[:-1]
    return _normalize(stubs[0::2], stubs[1::2], n)


def anti_align(pairs, labels, rng, max_rounds=64):
    """Rewire every same-class labelled pair to a labelled row of another class.

    ``labels`` is an [n] integer array with -1 for rows outside the evaluation role.
    """
    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels)
    labelled = np.flatnonzero(labels >= 0)
    a, b = pairs[:, 0].copy(), pairs[:, 1].copy()
    same = (labels[a] >= 0) & (labels[b] >= 0) & (labels[a] == labels[b])
    todo = np.flatnonzero(same)
    rewired = 0
    for _ in range(max_rounds):
        if not len(todo) or len(np.unique(labels[labelled])) < 2:
            break
        draw = labelled[rng.integers(0, len(labelled), size=len(todo))]
        ok = labels[draw] != labels[a[todo]]
        b[todo[ok]] = draw[ok]
        rewired += int(ok.sum())
        todo = todo[~ok]
    result = _normalize(a, b, n)
    remaining = (labels[result[:, 0]] >= 0) & (labels[result[:, 0]] == labels[result[:, 1]])
    return result, {"same_class_pairs_before": int(same.sum()), "rewired": rewired, "unresolved": int(len(todo)),
                    "same_class_pairs_after": int(remaining.sum())}


def inflate(pairs, n, factor, rng, max_rounds=64):
    """Add uniformly drawn absent pairs until the pair count is round(factor x original)."""
    target = min(int(round(factor * len(pairs))), n * (n - 1) // 2)
    codes = set((pairs[:, 0].astype(np.int64) * n + pairs[:, 1]).tolist())
    added = []
    for _ in range(max_rounds):
        need = target - len(codes)
        if need <= 0:
            break
        i, j = rng.integers(0, n, size=2 * need), rng.integers(0, n, size=2 * need)
        for lo, hi in zip(np.minimum(i, j).tolist(), np.maximum(i, j).tolist()):
            if lo != hi and lo * n + hi not in codes and len(codes) < target:
                codes.add(lo * n + hi)
                added.append((lo, hi))
    extra = np.asarray(added, dtype=np.int64).reshape(-1, 2)
    return np.concatenate([pairs, extra]), {"pairs_before": int(len(pairs)), "pairs_added": int(len(extra)),
                                            "target_pairs": int(target)}


def construct(control, pairs, weights, n, rng, *, labels=None, degree_factor=3.):
    """(pairs, weights, record) of one construction; weights of new pairs are drawn from the original ones."""
    if control == "graph irrelevant":
        new, record = configuration_rewire(pairs, n, rng), {"construction": "degree-preserving configuration model"}
    elif control == "misleading graph":
        new, record = anti_align(pairs, labels, rng)
        record["construction"] = "same-class labelled pairs rewired to another class"
    elif control == "support shift":
        new, record = inflate(pairs, n, degree_factor, rng)
        record["construction"] = f"uniform new pairs up to {degree_factor} x the pair count"
    elif control == "sensor bias":
        return pairs, weights, {"construction": "graph unchanged; sensor corruption"}
    else:
        raise ExperimentError(f"unknown control {control!r}")
    if weights is None or not len(weights):
        new_weights = None
    else:
        lookup = dict(zip((pairs[:, 0] * n + pairs[:, 1]).tolist(), weights.tolist()))
        codes = (new[:, 0] * n + new[:, 1]).tolist()
        drawn = rng.choice(weights, size=len(codes))
        new_weights = np.asarray([lookup.get(c, d) for c, d in zip(codes, drawn)], dtype=np.float64)
    degree = np.bincount(new.ravel(), minlength=n)
    record.update(pairs=int(len(new)), mean_degree=float(degree.mean()), max_degree=int(degree.max(initial=0)),
                  isolated_rows=int((degree == 0).sum()))
    return new, new_weights, record


# ----------------------------------------------------------------------------- sensor bias (PyTorch at call time)
class BiasedBank:
    """A runtime MeasurementBank whose sensed coordinates are corrupted after measurement."""

    def __init__(self, bank, *, center, scale, offset, noise, rng):
        self.bank, self.offset, self.noise, self.rng = bank, float(offset), float(noise), rng
        sensed = np.asarray([name not in HISTORY_COORDINATES for name in bank.names], dtype=np.float64)
        width = len(bank.names)
        self.center = np.asarray(center, dtype=np.float64)[:width]
        self.scale = np.asarray(scale, dtype=np.float64)[:width] * sensed

    def prepare(self, h0, logits0):
        return self.bank.prepare(h0, logits0)

    def measure(self, *args, **kwargs):
        import torch
        y = self.bank.measure(*args, **kwargs)
        value = y.double().cpu().numpy()
        z = np.where(self.scale > 0, (value - self.center) / np.where(self.scale > 0, self.scale, 1.), 0.)
        xi = self.rng.standard_normal(len(value))
        corrupted = value + self.scale * (self.offset + self.noise * xi * (1. + np.abs(z)))
        return torch.as_tensor(corrupted, dtype=y.dtype, device=y.device)

    def __getattr__(self, name):
        return getattr(self.bank, name)


@contextlib.contextmanager
def runtime_sensor(factory):
    """Swap cross_morse.policies.sensor_bank for the duration of a corrupted execution."""
    import cross_morse.policies as policies
    original = policies.sensor_bank
    policies.sensor_bank = factory(original)
    try:
        yield
    finally:
        policies.sensor_bank = original


# ----------------------------------------------------------------------------- plan / run
def plan_negative_controls(parent, output, *, role="test", controls=CONTROLS, realizations=1, seed=None,
                           max_graphs=0, gate_alpha=.05, gate_profile="full", gate_from=None,
                           bias_offset=1.5, bias_noise=.5, degree_factor=3., allow_fixture=False):
    parent = parent if isinstance(parent, ParentRun) else ParentRun(parent)
    if parent.synthetic and not allow_fixture:
        raise ExperimentError("a synthetic fixture parent requires --allow-fixture")
    checked = parent.check("admit")
    parent.require_role(role)
    controls = list(controls)
    if not controls or not set(controls) <= set(CONTROLS) or len(set(controls)) != len(controls):
        raise ExperimentError(f"controls must be distinct members of {CONTROLS}")
    if type(realizations) is not int or realizations < 1 or type(max_graphs) is not int or max_graphs < 0:
        raise ExperimentError("realizations must be positive and max_graphs nonnegative integers")
    if not (0 < gate_alpha < 1 and degree_factor > 1 and bias_noise >= 0):
        raise ExperimentError("invalid gate level, degree factor or sensor noise")
    gate = None
    if gate_from is not None:
        source = Path(gate_from).resolve()
        verify(source)
        executed = read_json(source / "experiment.json")
        if (executed.get("experiment") != "input_gate"
                or executed["parent"]["config_sha256"] != parent.identity()["config_sha256"]
                or executed["parent"]["source_sha256"] != parent.identity()["source_sha256"]):
            raise ExperimentError("the gate must come from a sealed input_gate run of the same parent")
        gate = {"from": source.name, "gate_sha256": sha256(source / "raw/gate.json"),
                "path": str(source / "raw/gate.json")}
    admission = parent.admission()
    candidates = parent.candidates()
    body = {"evaluation_role": role, "controls": controls, "realizations": realizations,
            "seed": int(parent.config["seed"] if seed is None else seed), "max_graphs": max_graphs,
            "gate": {"alpha": float(gate_alpha), "profile": gate_profile, "fit_role": "explore",
                     "fit_inputs": "unmodified D_explore units", "reused": gate},
            "sensor_bias": {"offset": float(bias_offset), "noise": float(bias_noise),
                            "scale": "fitted observer's robust per-coordinate scale (fit/full/controller.pt)",
                            "history_coordinates_unchanged": list(HISTORY_COORDINATES)},
            "support_shift": {"degree_factor": float(degree_factor)},
            "confirmed": candidates[admission["confirmed_index"]], "comparator": candidates[admission["comparator_index"]],
            "admission": admission, "parent_check": checked, "definitions": DEFINITIONS}
    return create_plan("negative_controls", parent, output, body)


def run_negative_controls(plan_path, *, device="cpu", trust=False):
    import torch
    from cross_morse.common import precision_context
    from cross_morse.data import load_data
    from cross_morse.learning import Controller
    from cross_morse.provenance import sha256_state_dict
    from cross_morse.runtime import graph_operator, is_graph_collection
    from cross_morse.tasks import task_spec
    from cross_morse.training import load_plant
    from .input_gate import (compatibility_vectors, ensemble_execute, fit_development_gate, gate_decision,
                             measurement_bank, policy_context, unit_cores, unit_losses)

    plan, parent = open_plan(plan_path, "negative_controls")
    role = plan["evaluation_role"]
    parent.verify("evaluate" if role == "test" else "admit")
    run = RunDirectory(plan_path, plan, parent, device=device, trust=trust)
    root, config = parent.root, parent.config
    data = load_data(root).protect_labels()
    collection = is_graph_collection(data)
    spec = task_spec(config, data.metadata)
    confirmed, comparator = policy_context(parent, plan["confirmed"]), policy_context(parent, plan["comparator"])
    seeds = sorted(set(confirmed["members"]) | set(comparator["members"]))
    models = {s: load_plant(data, config, root / "train", s, device) for s in seeds}
    frozen = {s: sha256_state_dict(m) for s, m in models.items()}
    gate_models = [models[s] for s in confirmed["members"]]
    reference = dict(np.load(root / "explore/nominal_reference.npz", allow_pickle=False))
    if plan["gate"]["reused"] is not None:
        gate_path = Path(plan["gate"]["reused"]["path"])
        if sha256(gate_path) != plan["gate"]["reused"]["gate_sha256"]:
            raise ExperimentError("the reused gate changed since planning")
        gate = read_json(gate_path)
    else:
        gate = fit_development_gate(data, gate_models, config, device, profile=plan["gate"]["profile"],
                                    alpha=plan["gate"]["alpha"], max_graphs=plan["max_graphs"])
    write_json(run.raw("gate.json"), dict(gate, frozen_before_evaluation_labels=True))
    observer = Controller.load(root / "fit/full/controller.pt", "cpu").observer
    bias = plan["sensor_bias"]
    multiclass_nodes = spec.task_type == "multiclass" and spec.prediction_level == "node"
    units, constructions, counter = [], [], 0
    for c_index, control in enumerate(plan["controls"]):
        if control == "misleading graph" and not multiclass_nodes:
            constructions.append({"control": control, "status": "NOT_APPLICABLE",
                                  "reason": "label anti-alignment is defined for multiclass node labels only"})
            continue
        for realization in range(plan["realizations"]):
            for eid, episode in role_units(data, role, plan["max_graphs"]):
                ids = np.asarray(episode.splits[role], dtype=np.int64)
                if not len(ids):
                    continue
                rng = np.random.default_rng([plan["seed"], c_index, realization, int(eid)])
                # Only the misleading construction needs labels to shape its input; the
                # other constructions read them after every prediction exists.
                labels = role_labels(data, episode, role, ids) if control == "misleading graph" else None
                n = int(episode.num_nodes)
                weights = None if episode.edge_weight is None else episode.edge_weight.numpy()
                pairs, pair_weights = undirected_pairs(episode.edge_index.numpy(), weights)
                node_labels = None
                if control == "misleading graph":
                    node_labels = np.full(n, -1, dtype=np.int64)
                    node_labels[ids] = np.asarray(labels, dtype=np.int64)
                new_pairs, new_weights, record = construct(control, pairs, pair_weights, n, rng, labels=node_labels,
                                                           degree_factor=plan["support_shift"]["degree_factor"])
                edges = np.concatenate([new_pairs.T, new_pairs.T[::-1]], axis=1)
                changed = dataclasses.replace(
                    episode, edge_index=torch.as_tensor(edges, dtype=torch.long),
                    edge_weight=None if new_weights is None else torch.as_tensor(
                        np.concatenate([new_weights, new_weights]), dtype=torch.float32),
                    edge_attr=None, edge_ptr=None if episode.edge_ptr is None else np.array([0, edges.shape[1]]))
                graph = graph_operator(changed, device)
                x = changed.x.to(device)
                cores = unit_cores(data, episode, collection)

                def biased(original, purpose):
                    stream = np.random.default_rng([plan["seed"], c_index, realization, int(eid), purpose])
                    def factory(graph_, model, config_, profile="full", mask=None):
                        bank = (original(graph_, model, config_, profile) if mask is None and purpose != 0
                                else measurement_bank(graph_, model, config_, profile, mask))
                        return BiasedBank(bank, center=observer.median, scale=observer.scale, offset=bias["offset"],
                                          noise=bias["noise"], rng=stream)
                    return factory
                corrupt = control == "sensor bias"
                with torch.no_grad(), precision_context(config, torch.device(device)):
                    vectors, _ = compatibility_vectors(gate_models, x, graph, config, profile=plan["gate"]["profile"],
                                                       cores=cores,
                                                       bank_factory=biased(measurement_bank, 0) if corrupt else None)
                decisions = {u: gate_decision(gate, v) for u, v in vectors.items()}
                with (runtime_sensor(lambda original: biased(original, 1)) if corrupt else contextlib.nullcontext()):
                    p_m, m_trace = ensemble_execute(models, x, graph, config, confirmed, reference)
                with (runtime_sensor(lambda original: biased(original, 2)) if corrupt else contextlib.nullcontext()):
                    p_c, c_trace = ensemble_execute(models, x, graph, config, comparator, reference)
                if p_c is None:
                    raise ExperimentError(f"the locked comparator failed on a constructed input ({control}, {eid})")
                if labels is None:
                    labels = role_labels(data, episode, role, ids)
                active = p_m is not None and any(a != 0 for t in m_trace["traces"] for a in t["actions"])
                nonnominal = 0 if p_m is None else sum(a != 0 for t in m_trace["traces"] for a in t["actions"])
                row_units = np.full(len(ids), eid) if collection else np.asarray(cores)[ids]
                loss_c = unit_losses(p_c, labels, ids, row_units, config)
                loss_m = unit_losses(p_m, labels, ids, row_units, config) if p_m is not None else {}
                constructions.append({"control": control, "realization": realization, "input_id": eid,
                                      "status": "EXECUTED", **record,
                                      "controller_failure": None if p_m is not None else m_trace.get("failure")})
                for unit, (score, g) in sorted(decisions.items(), key=lambda item: str(item[0])):
                    key = eid if collection else unit
                    if key not in loss_c:
                        continue
                    served = int(bool(g) and p_m is not None)
                    harm = loss_m[key] - loss_c[key] if key in loss_m else None
                    units.append({"dataset": parent.dataset, "seed": realization,
                                  "unit_id": f"graph:{eid}" if collection else f"core:{unit}",
                                  "condition": control, "realization": realization, "input_id": eid,
                                  "comparator_loss": repr(loss_c[key]),
                                  "deployed_loss": repr(loss_m[key] if served else loss_c[key]),
                                  "controller_loss": repr(loss_m[key]) if key in loss_m else "",
                                  "fallback": 1 - int(g), "g_input": int(g), "served_controller": served,
                                  "gate_score": repr(score),
                                  "active": int(served and active), "nonnominal_actions": nonnominal,
                                  "harm": "" if harm is None else repr(harm)})
                counter += 1
    if not units:
        raise ExperimentError("no constructed evaluation unit was scored")
    if any(sha256_state_dict(m) != frozen[s] for s, m in models.items()):
        raise RuntimeError("a frozen predictor changed during the negative controls")
    parent.verify("evaluate" if role == "test" else "admit")
    write_csv(run.raw("units.csv"), units)
    write_json(run.raw("constructions.json"), {"inputs": constructions, "executed_inputs": counter,
                                               "labels": "evaluation-role labels shape the misleading graph and score "
                                                         "every construction; no policy receives a label"})
    summary = summarize_negative_controls([run.root])
    summary["truncated"] = bool(plan["max_graphs"])
    return run.finish(summary)


def summarize_negative_controls(dirs):
    """Recompute the Table S19 values from the sealed unit table (no PyTorch)."""
    pooled, notes, datasets = {}, {}, set()
    for d in dirs:
        d = Path(d)
        datasets.add(read_json(d / "experiment.json")["parent"]["dataset"])
        for row in read_csv(d / "raw/units.csv"):
            pooled.setdefault(row["condition"], []).append(row)
        for record in read_json(d / "raw/constructions.json")["inputs"]:
            if record.get("status") == "NOT_APPLICABLE":
                notes.setdefault(record["control"], {"status": "NOT_APPLICABLE", "reason": record["reason"]})
    if len(datasets) > 1:
        raise ExperimentError("the negative-control record has no dataset key; export one dataset at a time")
    rows = []
    for control in CONTROLS:
        group = pooled.get(control)
        if not group:
            continue
        keys = {"control": control}
        harms = [float(r["harm"]) for r in group if r["harm"] != ""]
        rows.append(summary_row(keys, "residual_activity", float(np.mean([int(r["active"]) for r in group])),
                                "fraction", DEFINITIONS["residual_activity"]))
        rows.append(summary_row(keys, "input_fallback", float(np.mean([int(r["fallback"]) for r in group])),
                                "fraction", DEFINITIONS["input_fallback"]))
        if harms:
            rows.append(summary_row(keys, "worst_pre_fallback_harm", max(harms), "nats",
                                    DEFINITIONS["worst_pre_fallback_harm"]))
        notes[control] = {"units": len(group), "dataset": sorted(datasets)[0]}
    return {"rows": rows, "notes": notes}


def export_records(results_dirs, **kwargs):
    protocol = ("Table S19 constructions applied to the evaluation-role inputs of one sealed run; q_M and q_C executed by "
                "the released runtime behind the static g_input frozen on unmodified D_explore inputs; residual "
                "activity, input fallback and worst pre-fallback paired harm per construction")
    return export_record_set("negative_controls", results_dirs, summarize=summarize_negative_controls,
                             raw_exports=RAW_EXPORTS, protocol=protocol, **kwargs)
