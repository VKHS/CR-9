"""E12/E17: the internal rows of Supplementary Tables S15 and S15b (record set ``baselines``).

Tables S15 and S15b print, for the internal methods of every dataset, rows of the ``baselines`` record set that
no lifecycle stage writes: the per-seed official scores and GPU-hours of every internal route, and the E17
serial cost and wall time of the comparator q_C and of the deployed rule. This runner measures them on the
seven sealed lifecycle runs (Roman-empire: its campaign of ten split runs, one job per split run); the export is
``scripts/export_records.py baselines --internal <output>`` (also ``scripts/run_internal_baselines.py export``),
which adds these rows to the rows of the pinned-upstream external studies.

Workflow (``scripts/run_internal_baselines.py``):

    plan    parents and settings -> <output>/plan.json: per run the method rows, the configuration of every
            method (with the ranking that chose it), the timed inputs and the stage timing of the GPU-hours;
            torch-free
    run     one job per run (--job K for an array task) -> <output>/jobs/<dataset>/<run>/: scores/<method>/
            seed<s>.json, timing.csv, executions.json, session.log, gpu_hours.json, environment.json, result.json
            and ARTIFACT_MANIFEST.json (the seal, written last). An interrupted job resumes its scores; the timing
            sessions always run in the allocation that completes the job, whose scheduler key the rows carry.
    export  scripts/export_records.py baselines --internal <output> (verifies the seals, the plan identity and
            the parent identities and adds the rows; the timing rows need the sacct export of the allocations)

Definitions (the runner's choices where the paper gives none; recorded in the plan and the provenance):

* Methods: the internal rows of Table S15 as ``paper/build/displays/records_campaigns.method_rows`` builds them
  (``cm_campaigns.baselines.internal_methods``): the routes of S15_ROUTES plus the comparator's and the deployed
  rule's route, keyed '<route> (qC)', '<route> (deployed)' (absent when every run returned its comparator) and
  by route name. q_C route, q_D route, 'returned' and E follow ``run_ledger``: the modal comparator route and
  ensemble size over a campaign's split runs, the modal deployed route over the splits that did not return
  their comparator, returned when every split returned it.
* Configuration: q_C and q_D are each run's own selected comparator (select/decision.json qC_index) and deployed
  rule (admit/decision.json deployed_index). Every other route takes the configuration with the smallest
  counts-weighted mean D_select unit loss (select/unit_scores.npz) among the route's candidates at the
  comparator's ensemble size E that executed without failure, ties broken by the registered order of
  ``cross_morse.statistics._candidate_key``: the family representative of the run's comparator rule
  (``cross_morse.statistics._reconstruct_comparator``) restricted to the route. The plan checks that this rule,
  applied to the comparator's own route and feasibility, reproduces the run's comparator.
* Scores: every configuration is executed once per registered plant checkpoint (config training.seeds) as a
  one-member ensemble on D_test, labels read in the evaluate stage's label scope, and scored with the official
  metric (``cross_morse.evaluation.score_observations``) in the set's units: percent, AP native for
  Peptides-func. Roman-empire: per seed, the mean over the ten split runs.
* Timing (E17, Table S15b): a ``cm_campaigns.timing.TimingSession`` per timed method (q_C and the deployed rule)
  and input alternates one nominal execution (the registered nominal E = 1 candidate: one plant member at full
  depth with the seed of the method's first ensemble member) and the method's execution (its E members in seed
  order, ensemble mean on the host) on the same input. work = the method's deployed multiply-add count divided
  by the nominal execution's (the cost accounting of select/candidates.json), nominal_work = 1; a session whose
  executions differ in work is refused. Inputs: node-level datasets the full graph; graph collections the test
  graphs at edge-count quantiles {0, 0.5, 1} (``cm_experiments.timing_records.graph_instances``).
* GPU-hours (Supplementary Table S41 accounting on the run's CUDA device): open-loop routes, the full-depth plant
  and q_C: backbone training of the configuration's E plant checkpoints (sum of seconds_this_attempt in
  train/manifest.json) / 3600; feedback routes and the deployed rule add the qualification increment, the summed
  seconds_this_attempt of the explore, fit, select, confirm and admit stages (stages/<stage>.json). Roman-empire:
  the sum over the ten split runs. A parent whose recorded device is not CUDA is refused (``allow_fixture``
  accepts CPU fixtures as a software check; their sets are partial), and a missing stage time is refused.

Nothing in ``cross_morse`` is modified: the runtime is imported and called (``ParentContext`` of
``cm_experiments.ensemble_grid``). Planning, statistics and the export are torch-free; execution imports the
runtime lazily.
"""
from __future__ import annotations

import math
import shutil
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from cm_campaigns.baselines import (DEPLOYED, FEEDBACK_ROUTES, QC, ROUTE, internal_methods, method_directory)
from cm_campaigns.contract import METRICS

from .artifacts import seal, sha256, verify, write_json
from .ensemble_grid import (STAGES, ParentContext, RunnerError, SealedRun, candidate_identity, check_identity,
                            check_role, create_plan, environment_record, json_safe, load_plan, member_runs,
                            read_json, runtime_source_sha256, selected_jobs, slug, utc_now)
from .timing_records import (DEFAULT_GRAPH_QUANTILES, DEFAULT_REPETITIONS, DEFAULT_WARMUP, PREDICTION_ENCODING,
                             _dump, _write_bytes, graph_instances, member_record, npy_sha256, plant_layout,
                             read_timing_rows, sealed_output, timing_job_key, timing_means, write_timing_rows)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_FILES = ("cm_experiments/internal_baselines.py", "scripts/run_internal_baselines.py",
                    "cm_campaigns/baselines.py")
RUNNER = "internal_baselines"
RESULT_SCHEMA = "cm-internal-baselines-run-1"
SCORE_SCHEMA = "cm-internal-baselines-score-1"
GPU_SCHEMA = "cm-internal-baselines-gpu-hours-1"
MANIFEST = "ARTIFACT_MANIFEST.json"
QUALIFICATION_STAGES = ("explore", "fit", "select", "confirm", "admit")
DEFINITIONS = {
    "methods": (
        "the internal rows of Supplementary Table S15 as paper/build/displays/records_campaigns.method_rows builds "
        "them (cm_campaigns.baselines.internal_methods): the routes of S15_ROUTES plus the comparator's and the "
        "deployed rule's route, keyed '<route> (qC)', '<route> (deployed)' (absent when every run returned its "
        "comparator) and by route name; q_C route, q_D route, returned and E with run_ledger semantics (modal route "
        "and ensemble size over a campaign's split runs, the deployed route over the splits that did not return "
        "their comparator; returned when every split returned it)"),
    "configuration": (
        "q_C and q_D: each run's own comparator (select/decision.json qC_index) and deployed rule "
        "(admit/decision.json deployed_index); every other route: the configuration with the smallest "
        "counts-weighted mean D_select unit loss (select/unit_scores.npz) among the route's feasible candidates at "
        "the comparator's ensemble size E that executed without failure (select/candidates.json feasible true, "
        "execution_failure false), ties broken by the registered order of cross_morse.statistics._candidate_key: "
        "the family representative of the run's comparator rule (cross_morse.statistics._reconstruct_comparator) "
        "restricted to the route. A route without a feasible configuration at E takes its best executed "
        "configuration and records feasible false"),
    "score": (
        "official test metric (cross_morse.evaluation.score_observations) of the configuration executed as a "
        "one-member ensemble with each registered plant checkpoint (config training.seeds) on D_test, labels read "
        "in the evaluate stage's label scope; percent, AP native for Peptides-func; Roman-empire: per seed the mean "
        "over the ten split runs"),
    "timing": (
        "cm_campaigns.timing.TimingSession per timed method (q_C and the deployed rule) and input: one nominal "
        "execution (the registered nominal E = 1 candidate, one plant member at full depth with the seed of the "
        "method's first ensemble member) alternating with the method's execution (its E members in seed order, "
        "ensemble mean on the host) on the same input; work = the method's deployed multiply-add count divided by "
        "the nominal execution's (the cost accounting of select/candidates.json), nominal_work = 1; inputs: the "
        "full graph of a node-level dataset, the test graphs at the declared edge-count quantiles of a collection"),
    "gpu_hours": (
        "Supplementary Table S41 accounting on the run's CUDA device: open-loop routes, the full-depth plant and "
        "q_C: backbone training of the configuration's E plant checkpoints (sum of seconds_this_attempt in "
        "train/manifest.json) / 3600; feedback routes and the deployed rule: plus the qualification increment, the "
        "summed seconds_this_attempt of the explore, fit, select, confirm and admit stages (stages/<stage>.json); "
        "Roman-empire: the sum over the ten split runs"),
}


# ============================================================================= shared helpers
def experiment_source_identity() -> dict:
    return {name: sha256(ROOT / name) for name in EXPERIMENT_FILES if (ROOT / name).is_file()}


def reported_score(metric: float, metric_name: str) -> tuple[float, str]:
    """The official metric in the units of the baselines set: percent, AP native."""
    return (float(metric), "AP") if metric_name == "ap" else (100.0 * float(metric), "%")


def load_internal_plan(output) -> dict:
    plan = load_plan(output, "baselines")
    if plan.get("runner") != RUNNER:
        raise RunnerError(f"{output}/plan.json is not a plan of scripts/run_internal_baselines.py")
    return plan


def job_directory(output, entry: dict) -> Path:
    return Path(output) / "jobs" / entry["slug"] / slug(entry["member"])


def _mode(values):
    """The modal value, ties in the order first encountered (run_ledger's Counter.most_common)."""
    return Counter(values).most_common(1)[0][0]


def _finite(value, where: str) -> float:
    if value is None or isinstance(value, bool):
        raise RunnerError(f"{where}: value is missing")
    value = float(value)
    if not math.isfinite(value):
        raise RunnerError(f"{where}: value is not finite")
    return value


# ============================================================================= planning (torch-free)
def selection_risk(candidates: list[dict], loss, counts, *, where: str) -> np.ndarray:
    """Counts-weighted mean D_select unit loss of every executed candidate (NaN for an execution failure).

    The statistic of the comparator rule's family representative: (w @ L) / sum(w) over the executed columns
    (cross_morse.statistics._reconstruct_comparator with the registered unit weights of select/unit_scores.npz).
    """
    loss, counts = np.asarray(loss, dtype=np.float64), np.asarray(counts, dtype=np.float64)
    if loss.ndim != 2 or loss.shape[1] != len(candidates) or counts.shape != (loss.shape[0],) or not len(counts):
        raise RunnerError(f"{where}: select/unit_scores.npz does not hold one loss per selection unit and candidate")
    if not np.isfinite(counts).all() or (counts <= 0).any():
        raise RunnerError(f"{where}: the selection unit weights are not positive")
    observed = [j for j, c in enumerate(candidates) if not c.get("execution_failure")]
    if not observed or not np.isfinite(loss[:, observed]).all():
        raise RunnerError(f"{where}: the executed candidates carry nonfinite selection losses")
    risk = np.full(len(candidates), np.nan)
    risk[observed] = (counts @ loss[:, observed]) / counts.sum()
    return risk


def rank(candidates: list[dict], risk, ids) -> list[int]:
    """Candidates ordered by selection risk, ties by the registered order (cross_morse.statistics._argmin)."""
    from cross_morse.statistics import _candidate_key
    return sorted(ids, key=lambda j: (float(risk[j]), _candidate_key(candidates[j], j)))


def lifecycle_timing(run: SealedRun) -> dict:
    """Stage times of the qualification stages and the backbone training time of every plant checkpoint."""
    stages = {}
    for stage in QUALIFICATION_STAGES:
        run.require(stage)
        relative = f"stages/{stage}.json"
        marker = read_json(run.root / relative)
        manifest = run.root / "provenance" / "stages" / f"{STAGES.index(stage):02d}_{stage}.json"
        if (marker.get("stage") != stage or marker.get("status") != "COMPLETE"
                or marker.get("manifest_sha256") != sha256(manifest)):
            raise RunnerError(f"{run.root.name}: {relative} is not the completion marker of the sealed {stage} stage")
        value = marker.get("seconds_this_attempt")
        if value is None or isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0:
            raise RunnerError(f"{run.root.name}: {relative} records no stage time (seconds_this_attempt); the "
                              "qualification GPU-hours are not filled in")
        stages[stage] = {"seconds": float(value), "file": relative, "sha256": sha256(run.root / relative)}
    digest = sealed_output(run, "train", "train/manifest.json")
    checkpoints = {}
    for item in read_json(run.root / "train" / "manifest.json").get("checkpoints") or []:
        if item.get("baseline"):
            continue
        seed = int(item["seed"])
        if seed in checkpoints:
            raise RunnerError(f"{run.root.name}: train/manifest.json lists plant seed {seed} twice")
        checkpoints[seed] = item.get("seconds_this_attempt")
    seeds = [int(s) for s in run.config["training"]["seeds"]]
    untimed = [s for s in seeds if checkpoints.get(s) is None or isinstance(checkpoints[s], bool)
               or not math.isfinite(float(checkpoints[s])) or float(checkpoints[s]) < 0]
    if untimed:
        raise RunnerError(f"{run.root.name}: train/manifest.json records no training time of plant seeds {untimed}; "
                          "the backbone GPU-hours are not filled in")
    return {"stages": stages, "checkpoints": {str(s): float(checkpoints[s]) for s in seeds},
            "train_manifest": {"file": "train/manifest.json", "sha256": digest}}


def member_selection(run: SealedRun, *, allow_fixture: bool) -> dict:
    """The run's comparator, deployed rule, registry and selection statistic, read from its sealed records."""
    check_role(run, "test")
    sealed_output(run, "prepare", "environment.json")
    environment = read_json(run.root / "environment.json")
    device = str(environment.get("device") or "")
    cuda = device.startswith("cuda")
    if not cuda and not allow_fixture:
        raise RunnerError(f"{run.root.name}: the lifecycle ran on {device or 'an unrecorded device'}, not on CUDA; its "
                          "stage times are not GPU-hours (--allow-fixture accepts CPU fixtures as a software check)")
    for stage, relative in (("select", "select/candidates.json"), ("select", "select/decision.json"),
                            ("select", "select/unit_scores.npz"), ("admit", "admit/decision.json"),
                            ("admit", "policy_lock.json"), ("evaluate", "final/results_summary.json")):
        sealed_output(run, stage, relative)
    candidates = read_json(run.root / "select" / "candidates.json")
    decision = read_json(run.root / "select" / "decision.json")
    admission = read_json(run.root / "admit" / "decision.json")
    lock = read_json(run.root / "policy_lock.json")
    qc, qd = decision.get("qC_index"), admission.get("deployed_index")
    valid = all(isinstance(i, int) and not isinstance(i, bool) and 0 <= i < len(candidates) for i in (qc, qd))
    if (not valid or decision.get("comparator_index") != qc or lock.get("comparator_index") != qc
            or lock.get("deployed_index") != qd):
        raise RunnerError(f"{run.root.name}: the comparator and deployed indices of select/decision.json, "
                          "admit/decision.json and policy_lock.json disagree")
    with np.load(run.root / "select" / "unit_scores.npz", allow_pickle=False) as scores:
        loss, counts = scores["loss"], scores["counts"]
    risk = selection_risk(candidates, loss, counts, where=run.root.name)
    route = candidates[qc]["route"]
    family = [j for j, c in enumerate(candidates) if c.get("route") == route and bool(c.get("is_open_loop", False))
              and bool(c.get("feasible", True)) and not c.get("execution_failure")]
    reproduced = rank(candidates, risk, family)[0] if family else None
    if reproduced != qc:
        raise RunnerError(f"{run.root.name}: the selection statistic recomputed from select/unit_scores.npz ranks "
                          f"candidate {reproduced} first in the comparator's route, the run selected {qc}; the route "
                          "configurations cannot be derived from these records")
    nominal = next((i for i, c in enumerate(candidates) if c.get("route") == "nominal"
                    and int(c.get("ensemble", 0)) == 1 and not c.get("execution_failure")), None)
    if nominal is None:
        raise RunnerError(f"{run.root.name}: the registry holds no executed nominal E = 1 candidate")
    return {"candidates": candidates, "qc": qc, "qd": qd, "returned": qd == qc, "risk": risk, "nominal": nominal,
            "device": device, "cuda": cuda, "summary": read_json(run.root / "final" / "results_summary.json")}


def dataset_ledger(members: list[dict]) -> dict:
    """q_C route, q_D route, returned and E of a dataset with run_ledger semantics."""
    def candidate(member, key):
        return member["candidates"][member[key]]
    deployed = [m for m in members if not m["returned"]] or members
    return {"qc_route": _mode(candidate(m, "qc")["route"] for m in members),
            "E": _mode(int(candidate(m, "qc")["ensemble"]) for m in members),
            "qd_route": _mode(candidate(m, "qd")["route"] for m in deployed),
            "E_qd": _mode(int(candidate(m, "qd")["ensemble"]) for m in deployed),
            "returned": all(m["returned"] for m in members)}


def method_configuration(run: SealedRun, member: dict, method: dict, ledger: dict) -> dict:
    candidates, risk = member["candidates"], member["risk"]
    ranking = None
    if method["role"] == QC:
        index, rule = member["qc"], "the run's comparator q_C (select/decision.json qC_index)"
    elif method["role"] == DEPLOYED:
        index, rule = member["qd"], ("the run's deployed rule (admit/decision.json deployed_index; q_C when the run "
                                     "returned its comparator)")
    else:
        ids = [j for j, c in enumerate(candidates) if c.get("route") == method["route"]
               and int(c.get("ensemble", 0)) == ledger["E"] and not c.get("execution_failure")]
        if not ids:
            raise RunnerError(f"{run.root.name}: route {method['route']} has no configuration at E = {ledger['E']} that "
                              "executed without failure; Table S15 has no configuration to measure")
        feasible = [j for j in ids if bool(candidates[j].get("feasible", True))]
        ordered = rank(candidates, risk, feasible or ids)
        index, rule = ordered[0], ("smallest counts-weighted mean D_select loss among the route's feasible "
                                   "configurations at E (definitions)" if feasible else
                                   "no feasible configuration of the route at E: smallest counts-weighted mean "
                                   "D_select loss among its executed configurations (definitions)")
        ranking = [{"registry_index": j, "candidate_id": candidates[j].get("candidate_id"),
                    "selection_risk": float(risk[j]), "feasible": bool(candidates[j].get("feasible", True)),
                    "cost": candidates[j].get("cost")} for j in ordered]
    chosen = dict(candidates[index], registry_index=index)
    seeds = [int(s) for s in run.config["training"]["seeds"]]
    ensemble = int(chosen["ensemble"])
    if not 1 <= ensemble <= len(seeds):
        raise RunnerError(f"{run.root.name}: configuration {index} has {ensemble} members and the run {len(seeds)} seeds")
    risk_value = risk[index]
    return {"key": method["key"], "route": method["route"], "role": method["role"],
            "configuration_route": chosen["route"], "registry_index": index, "candidate": candidate_identity(chosen),
            "ensemble": ensemble, "member_seeds": seeds[:ensemble], "rule": rule,
            "feasible": bool(chosen.get("feasible", True)), "cost": chosen.get("cost"),
            "selection_risk": float(risk_value) if np.isfinite(risk_value) else None, "ranking": ranking,
            "qualification": method["role"] == DEPLOYED or (method["role"] == ROUTE and method["route"] in FEEDBACK_ROUTES),
            "timed": method["role"] in (QC, DEPLOYED)}


def timing_inputs(run: SealedRun, quantiles) -> tuple[list[dict], dict]:
    """The timed inputs: the full graph of a node-level dataset, test graphs at edge-count quantiles otherwise."""
    sealed_output(run, "prepare", "prepared/metadata.json")
    metadata = read_json(run.root / "prepared" / "metadata.json")
    layout = plant_layout(run.config, metadata, run.dataset)
    sealed_output(run, "prepare", "prepared/graph.npz")
    with np.load(run.root / "prepared" / "graph.npz", allow_pickle=False) as graph:
        if layout["independent_graphs"]:
            sealed_output(run, "prepare", "prepared/graph_splits.npz")
            with np.load(run.root / "prepared" / "graph_splits.npz", allow_pickle=False) as roles:
                test = roles["test"]
            instances = graph_instances(graph["graph_ptr"], graph["edge_ptr"], test, quantiles)
        else:
            instances = [{"instance": "full", "kind": "full", "nodes": int(metadata["num_nodes"]),
                          "edge_entries": int(graph["edge_index"].shape[1])}]
    return instances, layout


def plan_internal_baselines(parents, output, *, warmup=DEFAULT_WARMUP, repetitions=DEFAULT_REPETITIONS,
                            graph_quantiles=None, allow_fixture=False) -> Path:
    if not parents:
        raise RunnerError("give at least one --parent run")
    if int(warmup) < 1 or int(repetitions) < 3:
        raise RunnerError("protocol E17 needs at least one warm-up repeat and three measured repeats")
    quantiles = tuple(sorted({float(q) for q in (graph_quantiles or DEFAULT_GRAPH_QUANTILES)}))
    if any(not 0 <= q <= 1 for q in quantiles):
        raise RunnerError("graph quantiles must lie in [0, 1]")
    datasets, jobs = [], []
    for parent in parents:
        runs = [SealedRun(m) for m in member_runs(parent)]
        dataset = runs[0].dataset
        if len({r.dataset for r in runs}) != 1 or any(d["dataset"] == dataset for d in datasets):
            raise RunnerError(f"{parent}: one parent (run or campaign) per dataset")
        if METRICS.get(dataset) is None:
            raise RunnerError(f"{dataset} is not a dataset of the baselines record set")
        if len({tuple(int(s) for s in r.config["training"]["seeds"]) for r in runs}) != 1:
            raise RunnerError(f"{parent}: the split runs register different plant seeds")
        members = [member_selection(run, allow_fixture=allow_fixture) for run in runs]
        ledger = dataset_ledger(members)
        try:
            methods = internal_methods(ledger["qc_route"], ledger["qd_route"], ledger["returned"])
        except ValueError as exc:
            raise RunnerError(f"{dataset}: {exc}") from None
        if len({method_directory(m["key"]) for m in methods}) != len(methods):
            raise RunnerError(f"{dataset}: two method keys share a directory name")
        for run, member in zip(runs, members):
            configurations = [method_configuration(run, member, method, ledger) for method in methods]
            inputs, layout = timing_inputs(run, quantiles)
            nominal = dict(member["candidates"][member["nominal"]], registry_index=member["nominal"])
            timing = lifecycle_timing(run)
            jobs.append({"job": len(jobs), "dataset": dataset, "slug": slug(dataset), "member": run.root.name,
                         "run": str(run.root), "split": run.root.name if len(runs) > 1 else "official",
                         "identity": run.identity(), "seeds": [int(s) for s in run.config["training"]["seeds"]],
                         "device": member["device"], "cuda": member["cuda"], "methods": configurations,
                         "nominal": candidate_identity(nominal), "inputs": inputs, "layout": layout,
                         "lifecycle_timing": timing, "comparator_index": member["qc"],
                         "deployed_index": member["qd"], "returned": member["returned"],
                         "parent_files_read": run.inputs()})
        datasets.append({"dataset": dataset, "slug": slug(dataset), "parent": str(Path(parent).resolve()),
                         "members": [r.root.name for r in runs], "synthetic": runs[0].synthetic,
                         "metric": METRICS[dataset], **ledger, "methods": methods,
                         "aggregation": ("per seed, the mean of the score over the official split runs; GPU-hours summed "
                                         "over them; timing rows pooled" if len(runs) > 1 else "single run")})
    return create_plan(output, {
        "experiment": "E12, E17 internal rows of Supplementary Tables S15 and S15b", "record_set": "baselines",
        "runner": RUNNER, "datasets": datasets, "jobs": jobs, "warmup": int(warmup), "repetitions": int(repetitions),
        "graph_quantiles": list(quantiles), "allow_fixture": bool(allow_fixture), "definitions": DEFINITIONS,
        "experiment_code_sha256": experiment_source_identity(),
        "protocol": ("each method's configuration executed on D_test once per plant checkpoint as a one-member "
                     "ensemble and scored with the official metric; E17 TimingSession of q_C and the deployed rule "
                     "against the nominal execution on the same input; GPU-hours from the sealed stage and "
                     "training times (definitions)")})


# ============================================================================= execution (torch-agnostic)
def planned_candidate(run: SealedRun, identity: dict) -> dict:
    """The registry entry the plan names, checked against its recorded identity."""
    candidates = run.candidates()
    index = identity["registry_index"]
    candidate = dict(candidates[index], registry_index=index)
    if candidate_identity(candidate) != identity:
        raise RunnerError(f"{run.root.name}: registry entry {index} differs from the planned configuration")
    return candidate


def verify_planned_inputs(run: SealedRun, entry: dict) -> None:
    """The records the plan was derived from are unchanged (the sealed ones are covered by the identity)."""
    check_identity(run, entry["identity"])
    check_role(run, "test")
    if lifecycle_timing(run) != entry["lifecycle_timing"]:
        raise RunnerError(f"{run.root.name}: the stage or training times changed after planning")
    for stage, relative in (("select", "select/candidates.json"), ("select", "select/unit_scores.npz"),
                            ("admit", "policy_lock.json"), ("evaluate", "final/results_summary.json")):
        sealed_output(run, stage, relative)


def _clear_attempt(directory: Path) -> None:
    """Keep the per-seed scores of an interrupted attempt; everything else is executed again."""
    for path in sorted(directory.iterdir()) if directory.is_dir() else []:
        if path.name == "scores" and path.is_dir():
            for stale in path.rglob("*.tmp"):
                stale.unlink()
            continue
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()


def sealed_counterpart(summary: dict, method: dict) -> str | None:
    """The evaluate stage's model with the same configuration and members (qC, qD or qN), if any."""
    candidate = method["candidate"]
    comparator, deployed = summary.get("comparator") or {}, summary.get("deployed") or {}
    if method["role"] == QC and candidate.get("candidate_id") == comparator.get("candidate_id"):
        return "qC"
    if method["role"] == DEPLOYED and candidate.get("candidate_id") == deployed.get("candidate_id"):
        return "qD"
    if (candidate.get("route") == "nominal" and method["role"] == ROUTE
            and int(candidate.get("ensemble", 0)) == int(comparator.get("ensemble", -1))):
        return "qN"
    return None


def reproduction(ctx, summary: dict, name: str, ensemble: np.ndarray, labels, units) -> dict:
    """The job's E-member ensemble against the sealed test predictions of the same model (informational)."""
    from cross_morse.evaluation import score_observations
    sealed = np.load(ctx.run.root / "final" / f"{name}_probabilities.npy", mmap_mode="r", allow_pickle=False)
    rows = np.asarray(sealed[ctx.role_ids("test")], dtype=np.float32)
    score = score_observations(ensemble, labels, units, ctx.config)
    sealed_loss = ((summary.get("models") or {}).get(name) or {}).get("loss_nats")
    return {"computed": True, "sealed_model": name, "sealed_loss_nats": sealed_loss, "loss_nats": score["node_ce"],
            "abs_difference": None if sealed_loss is None else abs(float(sealed_loss) - score["node_ce"]),
            "sealed_prediction_sha256": npy_sha256(rows), "prediction_sha256": npy_sha256(ensemble),
            "identical": bool(np.array_equal(rows, ensemble))}


def score_method(ctx, plan_sha: str, entry: dict, method: dict, directory: Path, *, labels, units, summary,
                 failures=(), boundary=None) -> dict:
    """Per-seed official scores of one configuration, one execution per plant checkpoint (resumable)."""
    from cross_morse.evaluation import score_observations
    run = ctx.run
    candidate = planned_candidate(run, method["candidate"])
    controller = ctx.controller_for(candidate)
    folder = directory / "scores" / method_directory(method["key"])
    counterpart = sealed_counterpart(summary, method)
    replayed = folder / "reproduction.json"
    aggregate, collected, resumed = None, 0, False
    records = []
    for seed in entry["seeds"]:
        path = folder / f"seed{seed}.json"
        if path.is_file():
            record = read_json(path)
            if (record.get("schema") != SCORE_SCHEMA or record.get("plan_sha256") != plan_sha
                    or record.get("method") != method["key"] or record.get("seed") != seed
                    or record.get("candidate") != method["candidate"]):
                raise RunnerError(f"{path}: the score belongs to another plan, method or seed")
            resumed = resumed or seed in method["member_seeds"]
        else:
            checkpoint = run.root / "train" / f"plant_seed{seed}_best.pt"
            record = {"schema": SCORE_SCHEMA, "dataset": entry["dataset"], "member": entry["member"],
                      "method": method["key"], "role": method["role"], "seed": int(seed), "plan_sha256": plan_sha,
                      "candidate": method["candidate"], "checkpoint_sha256": sha256(checkpoint), "execution_role": "test"}
            model = ctx.plant(seed)
            try:
                rows, totals = ctx.predict(model, candidate, controller, "test")
            except failures as exc:
                record.update(status="FAILED", failure_reason=str(getattr(exc, "reason", exc)))
                rows = None
            del model
            if rows is not None:
                score = score_observations(rows, labels, units, ctx.config)
                if score["metric_name"] != METRICS.get(entry["dataset"]):
                    raise RunnerError(f"{entry['dataset']}: the run scores {score['metric_name']}, the baselines set "
                                      f"{METRICS.get(entry['dataset'])}")
                reported, unit = reported_score(score["metric"], score["metric_name"])
                record.update(status="COMPLETE", metric_name=score["metric_name"], metric=float(score["metric"]),
                              score_reported=reported, unit=unit, loss_nats=float(score["node_ce"]),
                              loss_name=score["loss_name"], units=len(score["core_ids"]), rows=int(rows.shape[0]),
                              prediction_sha256=npy_sha256(rows), prediction_encoding=PREDICTION_ENCODING,
                              execution={k: totals.get(k) for k in ("deployed_multiply_add_work", "multiply_add_work",
                                                                    "wall_seconds", "feasible", "support_violations",
                                                                    "graphs")})
                if counterpart and not replayed.is_file() and seed in method["member_seeds"] and not resumed:
                    aggregate = np.array(rows, dtype=np.float64) if aggregate is None else aggregate + rows
                    collected += 1
            elif seed in method["member_seeds"]:
                resumed = True          # a failed member: the sealed ensemble cannot be rebuilt
            record["finished_utc"] = utc_now()
            _dump(path, record)
        records.append(record)
        if counterpart and aggregate is not None and collected == len(method["member_seeds"]):
            ensemble = (aggregate / collected).astype(np.float32)        # the runtime's ensemble mean
            _dump(replayed, reproduction(ctx, summary, counterpart, ensemble, labels, units))
            aggregate = None
        if boundary is not None:
            boundary()
    if counterpart and replayed.is_file():
        replay = read_json(replayed)
    elif counterpart:
        replay = {"computed": False, "sealed_model": counterpart,
                  "reason": "the ensemble members were executed in an earlier attempt or failed"}
    else:
        replay = {"computed": False, "reason": "the evaluate stage sealed no model with this configuration"}
    complete = [r for r in records if r["status"] == "COMPLETE"]
    return {"key": method["key"], "route": method["route"], "role": method["role"], "candidate": method["candidate"],
            "ensemble": method["ensemble"], "member_seeds": method["member_seeds"],
            "scores": [{"seed": r["seed"], "status": r["status"], "score_reported": r.get("score_reported"),
                        "unit": r.get("unit"), "prediction_sha256": r.get("prediction_sha256"),
                        "file": f"scores/{method_directory(method['key'])}/seed{r['seed']}.json",
                        "sha256": sha256(folder / f"seed{r['seed']}.json")} for r in records],
            "failed_seeds": [r["seed"] for r in records if r["status"] != "COMPLETE"],
            "score_mean": (math.fsum(r["score_reported"] for r in complete) / len(complete)) if complete else None,
            "score_sd": statistics.stdev(r["score_reported"] for r in complete) if len(complete) > 1 else None,
            "reproduction": replay}


def time_method(entry: dict, plan: dict, method: dict, instance: dict, *, job_key: str, execute_nominal,
                execute_members) -> dict:
    """The E17 session of one method on one input (torch-agnostic orchestration).

    ``execute_nominal()`` executes the nominal member and returns its member record; ``execute_members()`` returns
    an iterator that executes the method's members one at a time, yielding (probabilities, member record). One
    untimed counting execution of each precedes the session; the k-th record of each belongs to repeat k.
    """
    from cm_campaigns.timing import TimingSession
    records = {"nominal": [], "method": []}

    def nominal_call():
        records["nominal"].append(execute_nominal())

    def method_call():
        aggregate, members = None, []
        for probabilities, record in execute_members():
            aggregate = np.array(probabilities, dtype=np.float64) if aggregate is None else aggregate + probabilities
            members.append(record)
        if aggregate is None:
            raise RunnerError(f"{method['key']} has no member")
        ensemble = (aggregate / len(members)).astype(np.float32)          # the runtime's ensemble mean
        records["method"].append({"members": members, "finite": bool(np.isfinite(ensemble).all())})

    nominal_call()          # untimed counting execution: work, operator size, lazy initialisation
    method_call()
    counted = {kind: values.pop() for kind, values in records.items()}
    nominal_work = int(counted["nominal"]["deployed_multiply_add_work"])
    method_work = sum(int(m["deployed_multiply_add_work"]) for m in counted["method"]["members"])
    if nominal_work <= 0:
        raise RunnerError(f"{entry['dataset']}/{instance['instance']}: the nominal execution counts no work")
    ratio = method_work / nominal_work
    session = TimingSession(dataset=entry["dataset"], method=method["key"], seed=int(method["member_seeds"][0]),
                            split=entry["split"], job_key=job_key, log="session.log")
    started = utc_now()
    rows = session.measure(input_id=instance["instance"], nominal=nominal_call, method=method_call, nominal_work=1,
                           method_work=ratio, warmup=plan["warmup"], repetitions=plan["repetitions"])
    finished = utc_now()
    executions = [counted["method"], *records["method"]]
    if not all(e["finite"] for e in executions):
        raise RunnerError(f"{entry['dataset']}/{method['key']}/{instance['instance']}: nonfinite ensemble output")
    works = {sum(int(m["deployed_multiply_add_work"]) for m in e["members"]) for e in executions}
    nominal_works = {int(n["deployed_multiply_add_work"]) for n in [counted["nominal"], *records["nominal"]]}
    if works != {method_work} or nominal_works != {nominal_work}:
        raise RunnerError(f"{entry['dataset']}/{method['key']}/{instance['instance']}: the executed work differs "
                          "between the executions of the session; its rows carry one work value")
    return {"rows": rows, "records": records, "counted": counted, "work_ratio": ratio, "method_work": method_work,
            "nominal_work": nominal_work, "started_utc": started, "finished_utc": finished}


def time_methods(ctx, plan: dict, entry: dict, *, job_key: str, execute) -> list[dict]:
    """Every timed method on every planned input; ``execute(model, candidate, controller, instance)``."""
    nominal = planned_candidate(ctx.run, entry["nominal"])
    sessions = []
    for method in entry["methods"]:
        if method["timed"]:
            sessions += _time_sessions(ctx, plan, entry, method, nominal, job_key=job_key, execute=execute)
    return sessions


def _time_sessions(ctx, plan: dict, entry: dict, method: dict, nominal: dict, *, job_key: str, execute) -> list[dict]:
    """The sessions of one method; its E members stay loaded for all of its inputs."""
    candidate = planned_candidate(ctx.run, method["candidate"])
    controller = ctx.controller_for(candidate)
    models = [ctx.plant(seed) for seed in method["member_seeds"]]

    def members(instance):
        return (execute(model, candidate, controller, instance) for model in models)

    sessions = []
    for instance in entry["inputs"]:
        session = time_method(entry, plan, method, instance, job_key=job_key,
                              execute_nominal=lambda instance=instance: execute(models[0], nominal, None, instance)[1],
                              execute_members=lambda instance=instance: members(instance))
        sessions.append({"method": method, "instance": instance, **session})
    return sessions


def gpu_hours_record(run: SealedRun, entry: dict) -> dict:
    """GPU-hours of every method of one run from its sealed stage and training times (torch-free)."""
    timing = lifecycle_timing(run)
    if timing != entry["lifecycle_timing"]:
        raise RunnerError(f"{run.root.name}: the stage or training times changed after planning")
    methods = []
    for method in entry["methods"]:
        backbone = [timing["checkpoints"][str(seed)] for seed in method["member_seeds"]]
        qualification = ({stage: timing["stages"][stage]["seconds"] for stage in QUALIFICATION_STAGES}
                         if method["qualification"] else {})
        seconds = math.fsum(backbone) + math.fsum(qualification.values())
        methods.append({"key": method["key"], "route": method["route"], "role": method["role"],
                        "ensemble": method["ensemble"], "member_seeds": method["member_seeds"],
                        "backbone_seconds": backbone, "qualification_seconds": qualification, "seconds": seconds,
                        "gpu_hours": seconds / 3600.0})
    return {"schema": GPU_SCHEMA, "dataset": entry["dataset"], "member": entry["member"],
            "lifecycle_device": entry["device"], "cuda": entry["cuda"], "methods": methods,
            "inputs": {**{f"stages/{s}.json": timing["stages"][s]["sha256"] for s in QUALIFICATION_STAGES},
                       "train/manifest.json": timing["train_manifest"]["sha256"]},
            "definition": DEFINITIONS["gpu_hours"]}


def execute_job(output, plan: dict, entry: dict, directory, *, ctx, device, job_key: str, execute, environment: dict,
                failures=(), boundary=None) -> Path:
    """One run's scores, timing sessions and GPU-hours; the seal (ARTIFACT_MANIFEST.json) is written last."""
    from cm_campaigns.timing import COLUMNS
    directory = Path(directory)
    if (directory / MANIFEST).is_file():
        return directory / "result.json"
    directory.mkdir(parents=True, exist_ok=True)
    _clear_attempt(directory)
    started = utc_now()
    plan_sha = sha256(Path(output) / "plan.json")
    run = ctx.run
    summary = read_json(run.root / "final" / "results_summary.json")
    labels, units = ctx.labels("test"), ctx.units("test")
    methods = [score_method(ctx, plan_sha, entry, method, directory, labels=labels, units=units, summary=summary,
                            failures=failures, boundary=boundary) for method in entry["methods"]]
    sessions = time_methods(ctx, plan, entry, job_key=job_key, execute=execute)
    gpu = gpu_hours_record(run, entry)
    environment_sha = _dump(directory / "environment.json", environment)
    gpu_sha = _dump(directory / "gpu_hours.json", gpu)
    rows = [row for session in sessions for row in session["rows"]]
    timing = None
    if sessions:
        timing_sha = write_timing_rows(directory / "timing.csv", rows)
        executions_sha = _dump(directory / "executions.json", {
            "schema": "cm-internal-baselines-executions-1",
            "sessions": [{"method": s["method"]["key"], "input": s["instance"]["instance"],
                          "counting_execution": s["counted"],
                          "repeats": [{"repeat": i, "nominal": s["records"]["nominal"][i],
                                       "method": s["records"]["method"][i]} for i in range(len(s["rows"]))]}
                         for s in sessions]})
        log = "".join(f"E17 TimingSession (cm_campaigns.timing): {entry['dataset']} / {s['method']['key']} / "
                      f"{s['instance']['instance']}; run {entry['member']}; device {device}; warm-up repeats "
                      f"{plan['warmup']}; measured repeats {plan['repetitions']}; job key {job_key}; started "
                      f"{s['started_utc']}; finished {s['finished_utc']}\n" for s in sessions)
        log_sha = _write_bytes(directory / "session.log", log.encode())
        timing = {"file": "timing.csv", "sha256": timing_sha, "columns": list(COLUMNS),
                  "executions": {"file": "executions.json", "sha256": executions_sha},
                  "session_log": {"file": "session.log", "sha256": log_sha},
                  "sessions": [{"method": s["method"]["key"], "input": s["instance"]["instance"],
                                "definition": s["instance"], "work_ratio": s["work_ratio"],
                                "method_work": s["method_work"], "nominal_work": s["nominal_work"],
                                "ledger_cost": s["method"]["cost"], **timing_means(s["rows"]),
                                "started_utc": s["started_utc"], "finished_utc": s["finished_utc"]}
                               for s in sessions]}
    write_json(directory / "result.json", json_safe({
        "schema": RESULT_SCHEMA, "experiment": plan["experiment"], "record_set": "baselines",
        "dataset": entry["dataset"], "member": entry["member"], "split": entry["split"], "device": str(device),
        "job_key": job_key, "seeds": entry["seeds"], "methods": methods, "timing": timing,
        "gpu_hours": {"file": "gpu_hours.json", "sha256": gpu_sha},
        "environment": {"file": "environment.json", "sha256": environment_sha}, "definitions": DEFINITIONS,
        "identities": {"plan_sha256": plan_sha, "runtime_source_sha256": runtime_source_sha256(),
                       "parent": entry["identity"], "parent_files_read": run.inputs(),
                       "experiment_code_sha256": experiment_source_identity()},
        "started_utc": started, "finished_utc": utc_now()}))
    seal(directory)
    return directory / "result.json"


# ============================================================================= execution (torch)
def run_internal_baselines(output, *, job=None, device="cuda", trust=False, job_key=None) -> list[Path]:
    if not trust:
        raise RunnerError("loading local checkpoints requires explicit trust (--trust-local-checkpoints)")
    output = Path(output).resolve()
    plan = load_internal_plan(output)
    key = timing_job_key(job_key)
    done = []
    for entry in selected_jobs(plan, job):
        directory = job_directory(output, entry)
        if not (directory / MANIFEST).is_file():
            _run_job(output, plan, entry, directory, device, key)
        done.append(directory / "result.json")
    return done


def _run_job(output, plan, entry, directory, device, job_key):
    import gc
    import torch
    from cross_morse.common import stop_at_boundary
    from cross_morse.policies import PolicyExecutionError, execute_policy
    from cross_morse.runtime import graph_operator, predict_dataset
    run = SealedRun(entry["run"])
    verify_planned_inputs(run, entry)
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    ctx = ParentContext(run, device, through="evaluate")
    environment = environment_record(device)
    data, config, reference = ctx.data, ctx.config, ctx.reference
    tests = set(np.asarray(data.graph_splits["test"]).tolist()) if ctx.collection else set()

    def execute(model, candidate, controller, instance):
        """One member execution on a planned input: the runtime's per-input path (timing_records)."""
        with torch.no_grad():
            if instance["kind"] == "graph":
                graph = int(instance["graph_id"])
                if graph not in tests:
                    raise RunnerError(f"graph {graph} is not a test-role graph of this collection")
                if (int(data.graph_ptr[graph + 1] - data.graph_ptr[graph]),
                        int(data.edge_ptr[graph + 1] - data.edge_ptr[graph])) != (instance["nodes"], instance["edge_entries"]):
                    raise RunnerError(f"graph {graph} differs from its registered size")
                episode = data.graph(graph)
                operator = graph_operator(episode, device)
                x = episode.x.to(device)
                probabilities, trace = execute_policy(model, x, operator, config, candidate, controller, reference)
                edges, nodes = operator.num_edges, operator.num_nodes
                del x, operator, episode
            else:
                if ctx.collection or (data.num_nodes, int(data.edge_index.shape[1])) != (
                        instance["nodes"], instance["edge_entries"]):
                    raise RunnerError("the prepared graph differs from its registered size")
                probabilities, trace = predict_dataset(model, data, ctx.graph, config, candidate, controller,
                                                       reference, device=device)
                edges, nodes = ctx.graph.num_edges, ctx.graph.num_nodes
        return probabilities, member_record(trace, edges, nodes)

    try:
        execute_job(output, plan, entry, directory, ctx=ctx, device=device, job_key=job_key, execute=execute,
                    environment=environment, failures=(PolicyExecutionError,), boundary=stop_at_boundary)
    except PolicyExecutionError as exc:
        raise RunnerError(f"{entry['dataset']}/{entry['member']}: a timed rule failed numerically ({exc.reason}); "
                          "it is not timed") from None


# ============================================================================= collection for the export (torch-free)
def _verify_result(output: Path, plan: dict, plan_sha: str, entry: dict) -> dict:
    directory = job_directory(output, entry)
    where = f"{entry['dataset']}/{entry['member']}"
    if not (directory / MANIFEST).is_file():
        raise RunnerError(f"{where}: job {entry['job']} has not completed")
    try:
        verify(directory)
    except (ValueError, OSError, KeyError) as exc:
        raise RunnerError(f"{where}: the sealed result directory does not verify ({exc})") from None
    result = read_json(directory / "result.json")
    if (result.get("schema") != RESULT_SCHEMA or result.get("dataset") != entry["dataset"]
            or result.get("member") != entry["member"] or result["identities"].get("plan_sha256") != plan_sha):
        raise RunnerError(f"{where}: the result belongs to another plan or run")
    parent = result["identities"].get("parent") or {}
    if any(parent.get(k) != entry["identity"].get(k) for k in ("config_sha256", "runtime_source_sha256",
                                                                 "units_sha256", "stage_manifests")):
        raise RunnerError(f"{where}: executed on another parent run than the planned one")
    if [m["key"] for m in result["methods"]] != [m["key"] for m in entry["methods"]] or result["seeds"] != entry["seeds"]:
        raise RunnerError(f"{where}: the methods or seeds differ from the plan")
    for method, planned in zip(result["methods"], entry["methods"]):
        if method["candidate"] != planned["candidate"]:
            raise RunnerError(f"{where}: {method['key']} executed another configuration than the planned one")
        if [s["seed"] for s in method["scores"]] != entry["seeds"]:
            raise RunnerError(f"{where}: {method['key']} lacks the scores of planned seeds")
        for score in method["scores"]:
            record = read_json(directory / score["file"])
            same = (record.get("schema") == SCORE_SCHEMA and record.get("plan_sha256") == plan_sha
                    and record.get("method") == method["key"] and record.get("seed") == score["seed"]
                    and record.get("candidate") == planned["candidate"] and record.get("status") == score["status"]
                    and record.get("score_reported") == score["score_reported"]
                    and sha256(directory / score["file"]) == score["sha256"])
            if not same:
                raise RunnerError(f"{where}: {score['file']} differs from the result's record")
    gpu = read_json(directory / "gpu_hours.json")
    if sha256(directory / "gpu_hours.json") != result["gpu_hours"]["sha256"]:
        raise RunnerError(f"{where}: gpu_hours.json differs from its recorded SHA-256")
    if [m["key"] for m in gpu["methods"]] != [m["key"] for m in entry["methods"]]:
        raise RunnerError(f"{where}: gpu_hours.json does not cover the planned methods")
    for item, planned in zip(gpu["methods"], entry["methods"]):
        seconds = math.fsum(item["backbone_seconds"]) + math.fsum(item["qualification_seconds"].values())
        expected = [entry["lifecycle_timing"]["checkpoints"][str(s)] for s in planned["member_seeds"]]
        stages = ({s: entry["lifecycle_timing"]["stages"][s]["seconds"] for s in QUALIFICATION_STAGES}
                  if planned["qualification"] else {})
        if (item["backbone_seconds"] != expected or item["qualification_seconds"] != stages
                or not math.isclose(seconds, item["seconds"], rel_tol=1e-12, abs_tol=0.0)
                or not math.isclose(seconds / 3600.0, item["gpu_hours"], rel_tol=1e-12, abs_tol=0.0)):
            raise RunnerError(f"{where}: the GPU-hours of {item['key']} do not follow from the planned stage times")
    rows = []
    timed = [m["key"] for m in entry["methods"] if m["timed"]]
    if timed:
        timing = result.get("timing") or {}
        if sha256(directory / "timing.csv") != timing.get("sha256"):
            raise RunnerError(f"{where}: timing.csv differs from its recorded SHA-256")
        rows = read_timing_rows(directory / "timing.csv")
        expected = len(timed) * len(entry["inputs"]) * (plan["warmup"] + plan["repetitions"])
        sessions = {(r["method"], r["input_id"]) for r in rows}
        if (len(rows) != expected or sessions != {(k, i["instance"]) for k in timed for i in entry["inputs"]}
                or {r["dataset"] for r in rows} != {entry["dataset"]} or {r["job_key"] for r in rows} != {result["job_key"]}):
            raise RunnerError(f"{where}: timing.csv does not hold the planned sessions")
        for session in timing["sessions"]:
            group = [r for r in rows if (r["method"], r["input_id"]) == (session["method"], session["input"])]
            if any(not math.isclose(float(r["work"]), session["work_ratio"], rel_tol=1e-12) for r in group):
                raise RunnerError(f"{where}: the work column differs from the session's work ratio")
    return {"entry": entry, "directory": directory, "result": result, "gpu": gpu, "timing_rows": rows,
            "environment": read_json(directory / "environment.json"),
            "files": sorted(p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file())}


def collect_internal_baselines(output) -> dict:
    """Every planned run's sealed result, verified against the plan, aggregated per dataset and method."""
    output = Path(output).resolve()
    plan = load_internal_plan(output)
    plan_sha = sha256(output / "plan.json")
    datasets, reasons = [], []
    for dataset in plan["datasets"]:
        entries = [j for j in plan["jobs"] if j["dataset"] == dataset["dataset"]]
        if [e["member"] for e in entries] != dataset["members"]:
            raise RunnerError(f"{dataset['dataset']}: the plan's jobs do not cover its runs")
        members = [_verify_result(output, plan, plan_sha, entry) for entry in entries]
        methods = []
        for index, method in enumerate(dataset["methods"]):
            seeds, missing = [], []
            for seed in entries[0]["seeds"]:
                parts = []
                for member in members:
                    score = next(s for s in member["result"]["methods"][index]["scores"] if s["seed"] == seed)
                    if score["status"] != "COMPLETE":
                        missing.append(f"{member['entry']['member']} seed {seed}")
                        break
                    parts.append({"member": member["entry"]["member"], "score_reported": score["score_reported"],
                                  "unit": score["unit"], "file": score["file"], "sha256": score["sha256"]})
                else:
                    seeds.append({"seed": seed, "parts": parts, "unit": parts[0]["unit"],
                                  "value": math.fsum(p["score_reported"] for p in parts) / len(parts)})
            if missing:
                reasons.append(f"{dataset['dataset']}/{method['key']}: the configuration failed numerically on "
                               f"{', '.join(missing)}; those seeds have no score")
            gpu = [{"member": m["entry"]["member"], "index": index, "gpu_hours": m["gpu"]["methods"][index]["gpu_hours"]}
                   for m in members]
            methods.append({**method, "seeds": seeds, "gpu_parts": gpu,
                            "gpu_hours": math.fsum(g["gpu_hours"] for g in gpu)})
        for member in members:
            if not member["gpu"]["cuda"]:
                reasons.append(f"{dataset['dataset']}/{member['entry']['member']}: the lifecycle ran on "
                               f"{member['gpu']['lifecycle_device'] or 'an unrecorded device'}; its stage times are not "
                               "GPU-hours")
            if member["timing_rows"] and not str(member["result"]["device"]).startswith("cuda"):
                reasons.append(f"{dataset['dataset']}/{member['entry']['member']}: timed on {member['result']['device']}, "
                               "not on the accelerator")
        datasets.append({"dataset": dataset["dataset"], "slug": dataset["slug"], "synthetic": dataset["synthetic"],
                         "members": members, "methods": methods, "ledger": {k: dataset[k] for k in (
                             "qc_route", "qd_route", "returned", "E", "E_qd")}})
    return {"output": output, "plan": plan, "plan_sha256": plan_sha, "datasets": datasets, "reasons": reasons,
            "synthetic": any(d["synthetic"] for d in datasets)}
