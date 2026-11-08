"""Cross-backbone substitutions (Supplementary Table S25; record set ``cross_backbone``).

Each regime compares the paper's frozen plant (model A) with a substitute frozen backbone
(model B) behind the same adapter interface (main Definition 1):

    arxiv / homophily                 residual MLP4-512 (shared transport)  vs  frozen GCNII
    Roman / heterophily               heterophily encoder4-384              vs  frozen GCNII
    Pascal / independent long range   GraphGPS-style 6x128                  vs  frozen GINE/GRAMA-style

Every row is one complete lifecycle per run (a Roman-empire row is the ten official split
runs) executed by the released runtime.  Model A reuses the sealed parent run (or reruns its
configuration with ``--rerun-model-a``); model B is the parent configuration with the
backbone keys replaced (``plan`` writes every variant configuration).  The record holds

    effect_nats         the run's sealed paired test effect q_D - q_C (final/results_summary.json;
                        mean over split runs)
    route_stable_folds  outer folds whose leave-one-outer-fold-out re-application of the registered
                        selection rule (cross_morse.statistics.nested_selection on the sealed
                        D_select unit scores, with the registered inner partitions) selects the
                        reported route (the route of the run's Eq. (19) selection)
    outer_folds         outer folds for which that re-selection could be evaluated

Before any fold is left out, the same call on all units must reproduce the sealed
select/decision.json exactly; otherwise collection stops.

The Pascal substitute is expressible through configuration keys (``backbone=linear_gine``).
GCNII is not a backbone of the released runtime: ``GCNIIPlant`` below subclasses the
runtime's FrozenPlant (initial residual plus identity mapping, one weight matrix per
control site, aggregation = the registry's A0 lazy random walk so that A0 reproduces the
native aggregation exactly), and ``python -B -m cm_experiments.backbones <run_cross_morse.py
arguments>`` runs the unmodified lifecycle with this class installed in the runtime's plant
factory.  The variant configuration names the class and the SHA-256 of this file, so the
run's configuration hash -- part of its identity -- covers the substitute code, and the
factory refuses a configuration whose recorded source differs.
"""
from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from .artifacts import canonical, sha256, write_json
from .ensemble_grid import (ROOT, STAGES, RunnerError, SealedRun, check_identity, code_files, create_plan, json_safe,
                            load_plan, member_runs, public_plan, read_json, selected_jobs, slug, utc_now,
                            write_record_set)

REGIME_DATASETS = {"arxiv / homophily": "ogbn-arxiv", "Roman / heterophily": "Roman-empire",
                   "Pascal / independent long range": "PascalVOC-SP"}
MODEL_BACKBONES = {"residual MLP4-512": "shared_transport", "heterophily encoder4-384": "heterophily",
                   "GraphGPS-style 6x128": "gps", "frozen GCNII": "gcnii", "frozen GINE/GRAMA-style": "linear_gine"}
MODEL_A = ("residual MLP4-512", "heterophily encoder4-384", "GraphGPS-style 6x128")
DEFAULT_BACKBONES = {"ogbn-proteins": "edge_weighted_transport", "Roman-empire": "heterophily",
                     "PascalVOC-SP": "gps", "COCO-SP": "gps", "Peptides-func": "linear_gine"}
GCNII_BACKBONE = "gcnii"
SUBSTITUTE_CLASS = "GCNIIPlant"


def coverage_rows() -> list[tuple[str, str]]:
    from cm_campaigns.contract import RECORD_SETS
    return [tuple(row) for row in RECORD_SETS["cross_backbone"]["coverage_rows"]]


def plant_source_sha256() -> str:
    return sha256(Path(__file__).resolve())


def substitute_identity() -> dict:
    return {"module": "cm_experiments.backbones", "class": SUBSTITUTE_CLASS, "source_sha256": plant_source_sha256()}


def backbone_of(config: dict) -> str:
    dataset = config.get("benchmark_dataset") or config.get("dataset")
    return config["model"].get("backbone", DEFAULT_BACKBONES.get(dataset, "shared_transport"))


def default_patch(model: str) -> dict:
    """Configuration keys replaced in the parent configuration for a substitute backbone (JSON pointers)."""
    if model == "frozen GCNII":
        return {"/model/backbone": GCNII_BACKBONE, "/model/gcnii_lambda": 0.5,
                "/model/substitute_plant": substitute_identity()}
    if model == "frozen GINE/GRAMA-style":
        return {"/model/backbone": "linear_gine", "/model/native_layers": 8, "/model/feedforward_hidden": None}
    return {}


def apply_patch(config: dict, patch: dict) -> dict:
    """Replace registered keys; new keys are accepted only inside the model section."""
    result = copy.deepcopy(config)
    for pointer, value in patch.items():
        if not pointer.startswith("/"):
            raise RunnerError(f"configuration pointer must start with '/': {pointer!r}")
        tokens = [t.replace("~1", "/").replace("~0", "~") for t in pointer[1:].split("/")]
        node = result
        for token in tokens[:-1]:
            if not isinstance(node, dict) or token not in node:
                raise RunnerError(f"configuration has no section {pointer}")
            node = node[token]
        if tokens[-1] not in node and tokens[0] != "model":
            raise RunnerError(f"configuration has no key {pointer}; only model keys may be added")
        node[tokens[-1]] = copy.deepcopy(value)
    return result


def validate_variant(config: dict) -> None:
    """The structural checks of cross_morse.core.build_plant, without importing torch."""
    model = config["model"]
    backbone = backbone_of(config)
    blocks = (model.get("feedforward_hidden") is not None, model.get("block_mlp_hidden") is not None,
              int(model.get("block_mlp_layers", 2)) != 2)
    if backbone == GCNII_BACKBONE:
        if model.get("substitute_plant") != substitute_identity():
            raise RunnerError("a GCNII variant must name this module's substitute plant and its current SHA-256")
        if any(blocks) or not float(model.get("gcnii_lambda", 0.5)) > 0:
            raise RunnerError("GCNII takes no block-width settings and a positive gcnii_lambda")
    elif backbone in ("gps", "linear_gine"):
        from .attribution import designated_sites
        designated_sites(config)
        if backbone == "linear_gine" and blocks[0]:
            raise RunnerError("feedforward_hidden applies only to gps")
        if backbone == "gps" and (blocks[1] or blocks[2]):
            raise RunnerError("block_mlp settings apply only to linear_gine")
    elif backbone in ("shared_transport", "edge_weighted_transport", "heterophily"):
        if any(blocks):
            raise RunnerError("block-width settings require a gps or linear_gine backbone")
    else:
        raise RunnerError(f"unknown backbone {backbone!r}")


def resolved(path: Path) -> dict:
    """The configuration exactly as run_cross_morse.py will resolve it (standard library only)."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from run_cross_morse import resolve_configuration
    try:
        return resolve_configuration(config_path=path)
    except (OSError, ValueError, TypeError) as exc:
        raise RunnerError(f"{path.name}: {exc}") from None


# ----------------------------------------------------------------------------- planning (torch-free)
def plan_cross_backbone(parents, output, *, rows=None, patches=None, rerun_model_a=False, reps=None) -> Path:
    """``parents``: lifecycle runs or campaigns, one per regime dataset; ``patches``: {"regime|model": patch}."""
    by_dataset = {}
    for parent in parents:
        runs = [SealedRun(m) for m in member_runs(parent)]
        dataset = runs[0].dataset
        if len({r.dataset for r in runs}) != 1 or dataset in by_dataset:
            raise RunnerError(f"{parent}: one parent (run or campaign) per dataset")
        by_dataset[dataset] = (Path(parent).resolve(), runs)
    wanted = [tuple(r) for r in (rows or coverage_rows())]
    unknown = set(wanted) - set(coverage_rows())
    if unknown:
        raise RunnerError(f"rows outside the S25 coverage contract: {sorted(unknown)}")
    patches = dict(patches or {})
    output = Path(output).expanduser().resolve()
    planned_rows, jobs, configs = [], [], {}
    for index, (regime, model) in enumerate(wanted):
        dataset = REGIME_DATASETS[regime]
        if dataset not in by_dataset:
            raise RunnerError(f"{regime}: give the sealed {dataset} parent with --parent")
        parent, runs = by_dataset[dataset]
        key = f"{regime}|{model}"
        reuse = model in MODEL_A and not rerun_model_a and key not in patches
        patch = patches.get(key, default_patch(model))
        row_slug = slug(f"{dataset}_{model}")
        row = {"row": index, "regime": regime, "model": model, "dataset": dataset, "slug": row_slug,
               "expected_backbone": MODEL_BACKBONES[model], "source": "parent" if reuse else "lifecycle",
               "patch": patch, "parent": str(parent), "runs": []}
        for run in runs:
            if reuse:
                for stage in ("select", "evaluate"):
                    run.require(stage)
                if backbone_of(run.config) != row["expected_backbone"]:
                    raise RunnerError(f"{run.root.name}: backbone {backbone_of(run.config)!r} is not {model}")
                row["runs"].append({"name": run.root.name, "run": str(run.root), "identity": run.identity(),
                                    "synthetic": run.synthetic})
                continue
            config = apply_patch(run.config, patch)
            validate_variant(config)
            if backbone_of(config) != row["expected_backbone"]:
                raise RunnerError(f"{key}: the patch gives backbone {backbone_of(config)!r}")
            relative = f"configs/{row_slug}/{slug(run.root.name)}.json"
            configs[relative] = config
            target = output / "runs" / row_slug / slug(run.root.name)
            jobs.append({"job": len(jobs), "row": index, "name": run.root.name, "config": str(output / relative),
                         "run": str(target), "config_sha256": canonical(config),
                         "substitute_plant": backbone_of(config) == GCNII_BACKBONE,
                         "parent_member": str(run.root), "parent_identity": run.identity()})
            row["runs"].append({"name": run.root.name, "run": str(target), "job": jobs[-1]["job"],
                                "synthetic": run.synthetic})
        planned_rows.append(row)
    synthetic = {r["synthetic"] for row in planned_rows for r in row["runs"]}
    if len(synthetic) != 1:
        raise RunnerError("synthetic and benchmark parents cannot share one record set")
    with tempfile.TemporaryDirectory() as probe:
        for relative, config in configs.items():
            path = Path(probe) / slug(relative)
            path.write_text(json.dumps(config), encoding="utf-8")
            if canonical(resolved(path)) != canonical(config):
                raise RunnerError(f"{relative}: run_cross_morse.py would resolve a different configuration")
    plan_path = create_plan(output, {
        "experiment": "S25 cross-backbone substitutions", "record_set": "cross_backbone", "rows": planned_rows,
        "jobs": jobs, "synthetic": synthetic.pop(),
        "parent_runs": [str(p) for p, _ in by_dataset.values()],
        "settings": {"reps": None if reps is None else int(reps), "rerun_model_a": bool(rerun_model_a),
                     "substitute_plant": substitute_identity()},
        "protocol": ("one complete lifecycle per run of every (regime, model) row with the released runtime; effect = "
                     "sealed paired test effect q_D - q_C; route stability = leave-one-outer-fold-out re-application of "
                     "the registered nested selection on the sealed D_select unit scores")})
    for relative, config in configs.items():
        write_json(output / relative, config)
    return plan_path


# ----------------------------------------------------------------------------- lifecycle execution
def lifecycle_command(job: dict, *, data_root, device: str, threads: int, python: str = sys.executable) -> list[str]:
    entry = ["-m", "cm_experiments.backbones"] if job["substitute_plant"] else [str(ROOT / "run_cross_morse.py")]
    return [python, "-B", *entry, "--config", job["config"], "--output-dir", job["run"],
            "--data-root", str(data_root), "--device", device, "--threads", str(threads),
            "--stage", "all", "--resume"]


def sealed_run(path) -> bool:
    path = Path(path)
    if not (path / "run_state.json").is_file():
        return False
    state = json.loads((path / "run_state.json").read_text())
    return state.get("status") == "COMPLETE" and state.get("last_complete_stage") == "evaluate"


def run_cross_backbone(output, *, job=None, data_root=None, device="cuda", threads=16, python=sys.executable) -> dict:
    output = Path(output).resolve()
    plan = load_plan(output, "cross_backbone")
    executed = []
    if plan["jobs"] and data_root is None:
        raise RunnerError("--data-root is required to execute the substitution lifecycles")
    for entry in (selected_jobs(plan, job) if plan["jobs"] else []):
        if canonical(read_json(entry["config"])) != entry["config_sha256"]:
            raise RunnerError(f"{entry['config']} changed after planning")
        if not sealed_run(entry["run"]):
            command = lifecycle_command(entry, data_root=data_root, device=device, threads=threads, python=python)
            code = subprocess.run(command, cwd=ROOT, check=False).returncode
            if code == 75:
                raise InterruptedError(f"lifecycle {entry['run']} checkpointed at a safe boundary")
            if code != 0:
                raise RunnerError(f"lifecycle {entry['run']} failed with exit code {code} (see its failure_summary.json)")
        executed.append(entry["run"])
    collected = collect_cross_backbone(output, require_complete=False)
    return {"executed_runs": executed, "collected_rows": [str(p) for p in collected]}


# ----------------------------------------------------------------------------- collection (torch-free)
def _metric_evidence(observations: dict, units, weights, candidate_count: int, config: dict):
    from cross_morse.evaluation import MetricEvidence
    sample_units = observations["sample_units"]
    mask = np.isin(sample_units, units)
    template = next(observations[k] for k in sorted(observations) if k.startswith("candidate_"))
    predictions = [observations[f"candidate_{j}"][mask] if f"candidate_{j}" in observations
                   else np.full((int(mask.sum()),) + template.shape[1:], np.nan, dtype=template.dtype)
                   for j in range(candidate_count)]
    return MetricEvidence(predictions, observations["labels"][mask], sample_units[mask], units, weights, config)


def route_stability(run: SealedRun, *, reps=None) -> dict:
    """Leave-one-outer-fold-out re-application of the registered selection rule (Eq. (19), Algorithm S4)."""
    from cross_morse.statistics import nested_selection
    run.require("select")
    decision = run.json("select/decision.json")
    candidates = run.candidates()
    arrays = run.npz("select/unit_scores.npz")
    loss, counts, folds, inner = arrays["loss"], arrays["counts"], arrays["outer_fold"], arrays["inner_fold"]
    accuracy, units = arrays.get("accuracy"), arrays["core_ids"]
    observations = run.npz("select/metric_observations.npz") if accuracy is None else None
    selection = run.config["selection"]
    seed = int(decision.get("bootstrap_seed", int(run.config["seed"]) + 4))

    def select(keep, inner_ids, replicates):
        evidence = (None if observations is None else
                    _metric_evidence(observations, units[keep], counts[keep], len(candidates), run.config))
        return nested_selection(loss[keep], None if accuracy is None else accuracy[keep], candidates, folds[keep],
                                inner_fold_ids=inner_ids, reps=replicates, alpha=selection["alpha"],
                                epsilon_r=selection["epsilon_r"], epsilon_eq=selection["epsilon_eq"],
                                epsilon_metric=selection["epsilon_metric"], seed=seed, weights=counts[keep],
                                metric_evaluator=evidence,
                                max_t_family=decision.get("max_t_family", "executed"))

    replay = select(np.ones(len(folds), dtype=bool), inner, int(decision["bootstrap_replicates"]))
    if replay["provisional_index"] != decision["provisional_index"] or replay["U"] != decision["U"]:
        raise RunnerError(f"{run.root.name}: the sealed D_select arrays do not reproduce select/decision.json")
    reported = candidates[decision["provisional_index"]]
    outer = [int(k) for k in np.unique(folds)]
    replicates = int(reps or selection["bootstrap_reps"])
    rows = []
    for k in outer:
        keep = folds != k
        if inner.ndim == 2:
            inner_ids = np.stack([inner[outer.index(j)][keep] for j in outer if j != k] + [inner[outer.index(k)][keep]])
        else:   # historical archives: one inner partition shared by every outer fold
            inner_ids = inner[keep]
        try:
            result = select(keep, inner_ids, replicates)
        except ValueError as exc:
            rows.append({"outer_fold": k, "evaluated": False, "units": int(keep.sum()), "reason": str(exc)})
            continue
        chosen = candidates[result["provisional_index"]]
        rows.append({"outer_fold": k, "evaluated": True, "units": int(keep.sum()),
                     "selected_index": int(result["provisional_index"]), "selected_route": chosen["route"],
                     "selected_candidate_id": chosen["candidate_id"], "route_matches": chosen["route"] == reported["route"],
                     "eligible_count": int(result["eligible_count"])})
    evaluated = [r for r in rows if r["evaluated"]]
    return {"reported_route": reported["route"], "reported_index": int(decision["provisional_index"]),
            "reported_candidate_id": reported["candidate_id"],
            "route_stable_folds": sum(r["route_matches"] for r in evaluated), "outer_folds": len(evaluated),
            "registered_outer_folds": len(outer), "folds": rows, "replay_reproduces_decision": True,
            "bootstrap_replicates": replicates, "bootstrap_seed": seed,
            "selection_effect_nats": decision["mean"][decision["provisional_index"]],
            "rule": "nested_selection on the units outside one outer fold, with the registered inner partitions of "
                    "the remaining outer folds and of that outer-training set"}


def run_measurements(run: SealedRun, *, reps=None) -> dict:
    run.require("evaluate")
    summary = run.json("final/results_summary.json")
    contrast = summary["paired_qD_minus_qC_loss"]
    inventory = run.json("train/model_inventory.json")
    stability = route_stability(run, reps=reps)
    model = run.config["model"]
    return {"name": run.root.name, "identity": run.identity(), "backbone": backbone_of(run.config),
            "architecture": inventory.get("architecture"), "architecture_module": inventory.get("architecture_module"),
            "architecture_widths": inventory.get("architecture_widths"),
            "substitute_plant": model.get("substitute_plant"),
            "substitute_plant_matches_current_source": (None if model.get("substitute_plant") is None
                                                        else model["substitute_plant"] == substitute_identity()),
            "effect_nats": contrast["mean"], "effect_interval_nats": [contrast.get("lower"), contrast.get("upper")],
            "admitted": summary.get("admitted"), "deployed_route": summary["deployed"]["route"],
            "comparator_route": summary["comparator"]["route"], **stability, "files_read": run.inputs()}


def collect_cross_backbone(output, *, require_complete=True) -> list[Path]:
    output = Path(output).resolve()
    plan = load_plan(output, "cross_backbone")
    written = []
    for row in plan["rows"]:
        path = output / "results" / f"{row['slug']}.json"
        runs = [SealedRun(r["run"]) if sealed_run(r["run"]) else None for r in row["runs"]]
        if any(run is None for run in runs):
            if require_complete:
                raise RunnerError(f"{row['regime']} / {row['model']}: not every run is sealed through evaluate")
            continue
        for planned, run in zip(row["runs"], runs):
            if "identity" in planned:
                check_identity(run, planned["identity"])
            else:
                job = plan["jobs"][planned["job"]]
                if canonical(run.config) != job["config_sha256"]:
                    raise RunnerError(f"{run.root}: the executed configuration differs from the planned variant")
            if backbone_of(run.config) != row["expected_backbone"]:
                raise RunnerError(f"{run.root}: backbone differs from {row['model']}")
        if not path.is_file():
            measured = [run_measurements(run, reps=plan["settings"]["reps"]) for run in runs]
            write_json(path, json_safe({
                "schema": "cm-cross-backbone-row-1", "regime": row["regime"], "model": row["model"],
                "dataset": row["dataset"], "source": row["source"], "patch": row["patch"],
                "effect_nats": math.fsum(m["effect_nats"] for m in measured) / len(measured),
                "route_stable_folds": sum(m["route_stable_folds"] for m in measured),
                "outer_folds": sum(m["outer_folds"] for m in measured),
                "reported_routes": sorted({m["reported_route"] for m in measured}), "runs": measured,
                "aggregation": "effect: mean over runs (official split runs for Roman-empire); folds: sums over runs",
                "plan_sha256": sha256(output / "plan.json"), "collected_utc": utc_now(),
                "run_period": [min(_created(run, "prepare") for run in runs),
                               max(_created(run, "evaluate") for run in runs)]}))
        written.append(path)
    return written


def _created(run: SealedRun, stage: str) -> str:
    return read_json(run.root / "provenance" / "stages" / f"{STAGES.index(stage):02d}_{stage}.json")["created_utc"]


# ----------------------------------------------------------------------------- export (torch-free)
def export_cross_backbone(output, *, root=None, replace=False, allow_partial=False, hardware=None, period=None,
                          command=None) -> Path:
    output = Path(output).resolve()
    plan = load_plan(output, "cross_backbone")
    paths = collect_cross_backbone(output, require_complete=True)
    files, rows, results, inputs = code_files(output), [], [], []
    for job in plan["jobs"]:
        files[f"code/{Path(job['config']).relative_to(output).as_posix()}"] = Path(job["config"])
    for row, path in zip(plan["rows"], paths):
        result = read_json(path)
        base = f"raw/{row['slug']}"
        files[f"{base}/row.json"] = path
        for run in row["runs"]:
            files[f"{base}/{slug(run['name'])}/results_summary.json"] = Path(run["run"]) / "final" / "results_summary.json"
            files[f"{base}/{slug(run['name'])}/environment.json"] = Path(run["run"]) / "environment.json"
        for quantity, unit in (("effect_nats", "nats"), ("route_stable_folds", "folds"), ("outer_folds", "folds")):
            rows.append({"regime": row["regime"], "model": row["model"], "quantity": quantity, "value": result[quantity],
                         "unit": unit, "source_file": f"{base}/row.json", "locator": f"json:/{quantity}"})
        results.append(result)
        inputs.append({"file": str(path.relative_to(output)), "sha256": sha256(path)})
    environments = [read_json(Path(r["run"]) / "environment.json") for row in plan["rows"] for r in row["runs"]]
    hardware = hardware or " | ".join(sorted({f"{e.get('gpu') or e.get('machine', '?') + ' CPU'}; torch {e.get('torch')}; "
                                              f"python {e.get('python')}" for e in environments}))
    period = period or (f"{min(r['run_period'][0] for r in results)}/{max(r['run_period'][1] for r in results)}")
    provenance = {
        "experiment": "Cross-backbone substitutions (Supplementary Table S25)",
        "produced_by": "scripts/run_experiment.py cross-backbone plan/run/export (cm_experiments/backbones.py); "
                       "lifecycles by run_cross_morse.py (GCNII rows through python -m cm_experiments.backbones)",
        "hardware": hardware, "period": period,
        "protocol": ("complete lifecycle per run with the frozen substitute backbone behind the same adapter; effect = "
                     "sealed paired test effect q_D - q_C (mean over official split runs for Roman-empire); route "
                     "stability = outer folds whose leave-one-outer-fold-out re-selection under the registered rule "
                     "selects the reported route, over the folds evaluated"),
        "synthetic": plan["synthetic"], "plan_sha256": sha256(output / "plan.json"),
        "substitute_plant": plan["settings"]["substitute_plant"]}
    return write_record_set("cross_backbone", rows, files, provenance, root=root, replace=replace,
                            allow_partial=allow_partial, command=command, inputs=inputs,
                            generated={"code/plan.json": public_plan(output)})


# ----------------------------------------------------------------------------- GCNII substitute plant (torch)
_GCNII_CLASS = None


def gcnii_plant_class():
    """GCNII-style frozen plant behind the CROSS-MORSE adapter (defined lazily: torch is optional here).

    H_{l+1} = act( (1 - beta_l) S_l + beta_l S_l W_l ),  S_l = (1 - alpha) M_l + alpha H_0,
    beta_l = log(lambda / l + 1) for layer l = site + 1, one bias-free W_l per control site.
    The native message M_l is the registry's A0 lazy random walk (1 - 0.70) H + 0.70 P H with
    alpha = 0.25, so adapter A0 reproduces the native step; action a supplies M_l = T_a H and
    alpha_a.  A8/A9 are the runtime's direct resets.  Encoder and head are FrozenPlant's.
    """
    global _GCNII_CLASS
    if _GCNII_CLASS is not None:
        return _GCNII_CLASS
    from torch import nn
    from cross_morse.core import FrozenPlant, _action, apply_transport

    class GCNIIPlant(FrozenPlant):
        def __init__(self, *args, control_depth, gcnii_lambda=0.5, **kwargs):
            super().__init__(*args, **kwargs)
            del self.shared
            del self.norm
            self.backbone = GCNII_BACKBONE
            self.control_depth = int(control_depth)
            self.gcnii_lambda = float(gcnii_lambda)
            if self.control_depth < 1 or not self.gcnii_lambda > 0:
                raise ValueError("GCNII needs a positive depth and lambda")
            self.layers = nn.ModuleList(nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
                                        for _ in range(self.control_depth))
            self.betas = tuple(math.log(self.gcnii_lambda / (layer + 1) + 1.0) for layer in range(self.control_depth))

        def architecture_widths(self):
            widths = super().architecture_widths()
            widths.update(gcnii_layers=self.control_depth, gcnii_lambda=self.gcnii_lambda)
            return widths

        def instrumentation_metadata(self):
            meta = super().instrumentation_metadata()
            meta.update(native_aggregation="registry A0 lazy random walk (1-0.70)H + 0.70 PH",
                        native_update="act(((1-alpha)M + alpha H0)((1-beta_l)I + beta_l W_l)); beta_l = log(lambda/l + 1)",
                        substitute_plant=substitute_identity())
            return meta

        def _site(self, site):
            if site is None or int(site) != site or not 0 <= int(site) < self.control_depth:
                raise ValueError("Controlled site index outside the registered trajectory")
            return int(site)

        def _update(self, h0, message, alpha, site):
            support = (1.0 - alpha) * message + alpha * h0
            beta = self.betas[site]
            mapped = (1.0 - beta) * support + beta * self._linear(self.layers[site], support)
            self.counters["dense_updates"] += 1
            return self._activation(mapped)

        def native_step(self, h, h0, graph, site=None):
            site = self._site(site)
            message = (1.0 - 0.70) * h + 0.70 * graph.apply(h)
            return self._update(h0, message, 0.25, site)

        def step(self, h, h0, graph, action_id, last_safe=None, site=None):
            site = self._site(site)
            a = _action(action_id)
            if a.mode != "transport":
                self.counters["resets"] += 1
                return (h0 if a.mode == "source" or last_safe is None else last_safe).clone()
            return self._update(h0, apply_transport(graph, h, a), a.alpha, site)

    GCNIIPlant.__module__ = __name__
    GCNIIPlant.__qualname__ = SUBSTITUTE_CLASS
    _GCNII_CLASS = GCNIIPlant
    return GCNIIPlant


def build_substitute_plant(input_dim, num_outputs, model_config, task=None):
    from cross_morse.tasks import TaskSpec, task_spec
    m = model_config
    if m.get("substitute_plant") != substitute_identity():
        raise ValueError("The configuration's substitute plant differs from cm_experiments/backbones.py "
                         "(class or source SHA-256); plan the substitution again")
    if (m.get("feedforward_hidden") is not None or m.get("block_mlp_hidden") is not None
            or int(m.get("block_mlp_layers", 2)) != 2):
        raise ValueError("GCNII takes no block-width settings")
    spec = TaskSpec(num_outputs=num_outputs) if task is None else task_spec(task)
    return gcnii_plant_class()(
        input_dim=input_dim, hidden_dim=int(m["hidden_dim"]), num_classes=num_outputs,
        encoder_layers=int(m["encoder_layers"]), dropout=float(m.get("dropout", .1)),
        activation=m.get("activation", "relu"), task=spec, dense_chunk_size=int(m.get("dense_chunk_size", 0)),
        encoder_hidden=m.get("encoder_hidden"), head_hidden=m.get("head_hidden", 0),
        control_depth=int(m["depth"]), gcnii_lambda=float(m.get("gcnii_lambda", 0.5)))


def install_substitute_plants() -> None:
    """Add the GCNII backbone to the runtime's plant factory for this process only."""
    import cross_morse.core as core
    import cross_morse.training as training
    original = core.build_plant
    if getattr(original, "substitute_plants", False):
        return

    def build_plant(input_dim, num_outputs, model_config, task=None):
        if model_config.get("backbone") == GCNII_BACKBONE:
            return build_substitute_plant(input_dim, num_outputs, model_config, task)
        return original(input_dim, num_outputs, model_config, task)

    build_plant.substitute_plants = True
    core.build_plant = build_plant
    training.build_plant = build_plant


def lifecycle_main(argv=None) -> int:
    """run_cross_morse.py (same arguments) with the substitute plants installed."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    install_substitute_plants()
    import run_cross_morse
    return run_cross_morse.main(list(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from cm_experiments.backbones import lifecycle_main as _lifecycle_main  # the importable module, not __main__
    raise SystemExit(_lifecycle_main())
