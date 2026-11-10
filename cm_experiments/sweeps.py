"""Sensitivity sweeps and module ablations derived from sealed lifecycle runs.

Each sweep fills one record contract of ``records/specs.json``:

``horizon_beam_sweep``  MPC-RH planning horizon and beam width (Supplementary Table S29, Figure S5)
``sensitivity_grids``   support thresholds, selection margins and clip level (Table S18)
``module_ablations``    matched module ablations (Table S17): three modules from sealed contrasts, the
                        other seven from the variant lifecycles of cm_experiments/ablation_lifecycle.py
                        attached to the plan (``ablations``)
``buffer_sensitivity``  cluster-buffer sensitivity of descriptive admission (Table S27): the sealed per-core
                        paired losses regrouped into the resampling units of each buffer
                        (cm_experiments/buffered_admission.py, which states the unit definition)

A quantity is recomputed from the parent run's sealed outputs whenever that is
exact: main Eq. (19) on the sealed simultaneous upper limits for the selection
margins, ``cross_morse.statistics.admission`` with the stage's seed for the clip
level and for the buffered resampling units, the registered support rule on the
logged per-cell statistics for the support masks, and confirmation contrasts
already sealed in ``confirm/``.
Settings that change execution (planner horizon and beam width, the support
thresholds' effect on routes) are measured by new lifecycle runs: the parent's
configuration with one registered key changed, executed by the released
``run_cross_morse.py`` from ``prepare`` through ``select`` in a fresh output
directory. The runtime is never modified, and nothing is filled in: a missing
output stops the collection.

Every collected value is written into a small computed JSON file together with
the SHA-256 of each sealed file it was read from; ``results.json`` lists the
values with their location, and ``scripts/export_sweep_records.py`` turns it
into a record set.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import statistics as stats
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPECS_PATH = ROOT / "records" / "specs.json"
SWEEPS = ("horizon_beam_sweep", "sensitivity_grids", "module_ablations", "buffer_sensitivity")
LIFECYCLE_THROUGH_SELECT = ("prepare", "train", "explore", "fit", "select")
PLAN_SCHEMA = "cm-sweep-plan-1"
RESULTS_SCHEMA = "cm-sweep-results-1"
SUPPORT_PARAMETERS = ("p_min", "n_min", "max_normalized_weight")
MARGIN_PARAMETERS = ("epsilon_r", "epsilon_eq")
# two-sided 95% Student t quantiles by degrees of freedom (paper/build/evidence.py uses the same table)
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
UNITS = {
    "selection_effect_nats": "nats", "expected_serial_cost": "nominal executions",
    "oracle_headroom_bound_nats": "nats", "realized_fraction": "fraction",
    "qualified_fraction": "fraction", "strata_without_transport": "strata", "eligible_count": "candidates",
    "selected_complexity": "complexity", "route_changed": "indicator", "route_simpler": "indicator",
    "clipped_fraction": "fraction", "clipped_mean_nats": "nats", "admission_upper_nats": "nats",
    "raw_maximum_harm_nats": "nats", "admitted": "indicator",
    "loss_change_nats": "nats", "retained_effect_fraction": "fraction", "cost_fraction": "fraction",
    "warning_lead_change_depths": "depths", "false_safe_rate_change": "fraction", "risk_rmse_change": "nats",
    "action_entropy_change": "nats", "innovation_rmse": "nats", "clean_score_change": "points",
    "worst_harm_change_nats": "nats",
    "clipped_effect_nats": "nats", "half_width_nats": "nats", "upper_nats": "nats", "resampling_units": "units",
}


class SweepError(ValueError):
    """The inputs do not support a complete, traceable sweep."""


# ----------------------------------------------------------------------------- small helpers
def load_spec(name: str) -> dict:
    sets = json.loads(SPECS_PATH.read_text(encoding="utf-8"))["sets"]
    if name not in SWEEPS or name not in sets:
        raise SweepError(f"{name} is not a sweep record contract of records/specs.json")
    return sets[name]


def read_json(path: Path):
    path = Path(path)
    if not path.is_file():
        raise SweepError(f"missing input file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
                          .encode()).hexdigest()


def number(value, where: str) -> float:
    if value is None or isinstance(value, bool):
        raise SweepError(f"{where}: value is missing")
    value = float(value)
    if not math.isfinite(value):
        raise SweepError(f"{where}: value is not finite")
    return value


def safe_label(text) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(text)).strip("_")


# ----------------------------------------------------------------------------- runs
class SealedRun:
    """Read access to one sealed lifecycle run; records the hash of every file read."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.config = read_json(self.root / "config.json")
        self.read: dict[str, str] = {"config.json": sha256_file(self.root / "config.json")}

    @property
    def dataset(self) -> str:
        return self.config.get("benchmark_dataset") or self.config["dataset"]

    @property
    def synthetic(self) -> bool:
        return self.config.get("dataset") == "synthetic" or self.config.get("scientific_run") is False

    def require(self, stage: str) -> None:
        marker = self.root / "stages" / f"{stage}.json"
        if not marker.is_file() or read_json(marker).get("status") != "COMPLETE":
            raise SweepError(f"{self.root.name}: stage {stage} is not sealed")

    def json(self, relative: str):
        path = self.root / relative
        value = read_json(path)
        self.read[relative] = sha256_file(path)
        return value

    def npz(self, relative: str) -> dict:
        path = self.root / relative
        if not path.is_file():
            raise SweepError(f"missing input file: {path}")
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        self.read[relative] = sha256_file(path)
        return arrays

    def inputs(self) -> list[dict]:
        return [{"file": name, "sha256": digest} for name, digest in sorted(self.read.items())]


def member_runs(root: Path) -> list[Path]:
    """The split runs of a Roman-empire campaign directory, otherwise the run itself."""
    root = Path(root).resolve()
    if (root / "config.json").is_file():
        return [root]
    splits = sorted(p for p in root.glob("split_*") if (p / "config.json").is_file())
    if not splits:
        raise SweepError(f"{root} is neither a run nor a campaign of split runs")
    return splits


def task_loss_name(config: dict) -> str:
    from cross_morse.tasks import task_spec
    return task_spec(config).loss_name


# ----------------------------------------------------------------------------- configuration variants
def _pointer_tokens(pointer: str) -> list[str]:
    if not pointer.startswith("/"):
        raise SweepError(f"configuration pointer must start with '/': {pointer!r}")
    return [t.replace("~1", "/").replace("~0", "~") for t in pointer[1:].split("/")]


def config_value(config: dict, pointer: str):
    node = config
    for token in _pointer_tokens(pointer):
        if not isinstance(node, dict) or token not in node:
            raise SweepError(f"configuration has no key {pointer}")
        node = node[token]
    return node


def variant_config(parent: dict, patch: dict) -> dict:
    """The parent configuration with registered keys replaced; new keys are refused."""
    config = copy.deepcopy(parent)
    for pointer, value in patch.items():
        tokens = _pointer_tokens(pointer)
        node = config
        for token in tokens[:-1]:
            if not isinstance(node, dict) or token not in node:
                raise SweepError(f"configuration has no key {pointer}")
            node = node[token]
        if not isinstance(node, dict) or tokens[-1] not in node:
            raise SweepError(f"configuration has no key {pointer}")
        node[tokens[-1]] = copy.deepcopy(value)
    return config


def lifecycle_command(config_path: Path, output: Path, stage: str, *, data_root: Path, device: str,
                      threads: int, python: str = sys.executable) -> list[str]:
    """One released lifecycle stage on a variant configuration (resume-safe)."""
    return [python, "-B", str(ROOT / "run_cross_morse.py"), "--config", str(config_path),
            "--output-dir", str(output), "--data-root", str(data_root), "--device", device,
            "--threads", str(threads), "--stage", stage, "--resume"]


# ----------------------------------------------------------------------------- main Eq. (19)
def eq19(decision: dict, candidates: list[dict], epsilon_r: float, epsilon_eq: float) -> dict:
    """The selection rule of cross_morse.statistics.nested_selection on sealed upper limits.

    The simultaneous upper limits U and the joint (resource, support, metric)
    feasibility do not depend on the two margins, so re-applying the rule to the
    sealed arrays gives exactly the selection the stage would make at other margins.
    """
    from cross_morse.statistics import _candidate_key
    upper, joint = decision["U"], decision["joint_feasible"]
    if len(upper) != len(candidates) or len(joint) != len(candidates):
        raise SweepError("select/decision.json and select/candidates.json differ in length")
    feedback = [j for j, c in enumerate(candidates)
                if not c.get("is_open_loop", False) and joint[j] and upper[j] is not None]
    minimum_upper = min(0.0, min(float(upper[j]) for j in feedback)) if feedback else 0.0
    eligible = [j for j in feedback if upper[j] < -epsilon_r and upper[j] <= minimum_upper + epsilon_eq]
    selected = (min(eligible, key=lambda j: _candidate_key(candidates[j], j)) if eligible
                else int(decision["comparator_index"]))
    return {"eligible": eligible, "selected": int(selected), "minimum_upper": minimum_upper}


def route_identity(candidate: dict) -> dict:
    return {k: candidate.get(k) for k in ("route", "profile", "horizon", "kappa", "ensemble", "candidate_id")}


def selection_outcome(run: SealedRun, epsilon_r: float | None = None, epsilon_eq: float | None = None) -> dict:
    run.require("select")
    decision = run.json("select/decision.json")
    candidates = run.json("select/candidates.json")
    registered = eq19(decision, candidates, decision["epsilon_r"], decision["epsilon_eq"])
    if (registered["eligible"] != list(decision["Q_s"])
            or registered["selected"] != int(decision["provisional_index"])):
        raise SweepError(f"{run.root.name}: select/decision.json is not reproduced by main Eq. (19)")
    outcome = eq19(decision, candidates,
                   decision["epsilon_r"] if epsilon_r is None else epsilon_r,
                   decision["epsilon_eq"] if epsilon_eq is None else epsilon_eq)
    return {**outcome, "candidates": candidates, "registered": registered}


def route_quantities(selected: dict, reference: dict) -> dict:
    from cross_morse.statistics import _candidate_key
    simpler = _candidate_key(selected, 0)[:3] < _candidate_key(reference, 0)[:3]
    return {"selected_complexity": float(selected["complexity"]),
            "route_changed": float(selected["hash"] != reference["hash"]),
            "route_simpler": float(simpler)}


# ----------------------------------------------------------------------------- support masks
def support_summary(support: dict, p_min: float, n_min: float, omega: float) -> dict:
    """The registered qualification rule of cross_morse.learning._support on logged cell statistics."""
    cells = support["cells"]
    if not cells:
        raise SweepError("the support audit has no cells")
    qualified = {}
    for key, cell in cells.items():
        qualified[key] = bool(cell["min_propensity"] >= p_min - 1e-12 and cell["ess"] >= n_min - 1e-9
                              and cell["max_normalized_weight"] <= omega + 1e-12)
    strata = sorted({int(c["stratum"]) for c in cells.values()})
    without = [s for s in strata if not any(qualified.get(f"{s}:{a}", False) for a in range(1, 10))]
    return {"qualified_fraction": sum(qualified.values()) / len(qualified),
            "strata_without_transport": float(len(without)), "strata": len(strata), "cells": len(cells)}


def check_support_rule(run: SealedRun, support: dict) -> None:
    """The recomputed rule must reproduce the sealed mask at the registered thresholds."""
    t = support["thresholds"]
    for key, cell in support["cells"].items():
        again = (cell["min_propensity"] >= t["p_min"] - 1e-12 and cell["ess"] >= t["n_min"] - 1e-9
                 and cell["max_normalized_weight"] <= t["max_normalized_weight"] + 1e-12)
        if again != bool(cell["qualified"]):
            raise SweepError(f"{run.root.name}: support cell {key} is not reproduced by the registered rule")


# ----------------------------------------------------------------------------- admission replay
def admission_replay(run: SealedRun, clip: float) -> dict:
    """cross_morse.statistics.admission on the sealed paired units with the stage's own seed."""
    from cross_morse.evaluation import independent_graphs
    from cross_morse.statistics import admission
    run.require("admit")
    decision = run.json("admit/decision.json")
    units = run.npz("admit/paired_units.npz")
    config = run.config["admission"]
    loss = decision["loss"]
    replay = admission(units["candidate_loss"] - units["comparator_loss"], weights=units["counts"],
                       reps=int(loss["bootstrap_replicates"]), alpha=config["alpha"],
                       epsilon_a=config["epsilon_a"], epsilon_metric=config["epsilon_metric"], clip=clip,
                       seed=int(loss["bootstrap_seed"]), metric_summary=decision["metric"],
                       independent_graphs=independent_graphs(run.config),
                       execution_valid=decision.get("execution_gate"))
    graph = independent_graphs(run.config)
    upper = replay["empirical_bernstein"]["upper"] if graph else replay["loss"]["upper"]
    return {"clipped_fraction": replay["clipping_fraction"], "clipped_mean_nats": replay["loss"]["mean"],
            "admission_upper_nats": upper, "raw_maximum_harm_nats": replay["raw_maximum_harm"],
            "admitted": float(replay["passed"]), "statistic": "empirical_bernstein" if graph else "cluster_bootstrap",
            "metric_gate": replay["metric_gate"], "epsilon_a": config["epsilon_a"],
            "_replay": replay, "_decision": decision}


def check_admission_replay(run: SealedRun) -> None:
    clip = run.config["admission"]["clip"]
    replay = admission_replay(run, clip)
    sealed, again = replay["_decision"], replay["_replay"]
    if (again["loss"]["mean"] != sealed["loss"]["mean"] or again["loss"]["upper"] != sealed["loss"]["upper"]
            or again["passed"] != sealed["passed"]):
        raise SweepError(f"{run.root.name}: admit/decision.json is not reproduced by the admission replay")


def repeated_split_upper(values: list[float]) -> float:
    """Small-sample t upper limit over official splits (the Roman-empire admission statistic)."""
    if len(values) < 2:
        raise SweepError("a repeated-split interval needs at least two splits")
    se = stats.stdev(values) / math.sqrt(len(values))
    return stats.fmean(values) + T975.get(len(values) - 1, 1.96) * se


# ----------------------------------------------------------------------------- collected rows
class Collector:
    """Computed files with their input hashes, and summary rows pointing into them."""

    def __init__(self, output: Path, record_set: str):
        self.output = Path(output)
        self.record_set = record_set
        self.spec = load_spec(record_set)
        self.rows: list[dict] = []
        self.files: list[str] = []
        self.replicates: list[dict] = []
        self.deposits: list[dict] = []      # files of attached experiments copied under raw/ by the exporter

    def add(self, name: str, keys: dict, values: dict, detail: dict, runs: list[SealedRun]) -> None:
        if set(keys) != set(self.spec["keys"]):
            raise SweepError(f"{self.record_set}: row keys {sorted(keys)} differ from {self.spec['keys']}")
        relative = f"computed/{self.record_set}/{safe_label(name)}.json"
        clean = {}
        for quantity, value in values.items():
            if quantity not in self.spec["quantities"]:
                raise SweepError(f"{self.record_set}: {quantity} is not a quantity of the contract")
            if value is None:
                continue
            clean[quantity] = number(value, f"{name}/{quantity}")
        record = {"record_set": self.record_set, "keys": {k: str(v) for k, v in keys.items()},
                  "values": clean, "detail": detail,
                  "inputs": [{"run": str(run.root), **item} for run in runs for item in run.inputs()]}
        write_json(self.output / relative, record)
        self.files.append(relative)
        for quantity, value in clean.items():
            self.rows.append({"keys": record["keys"], "quantity": quantity, "value": value,
                              "unit": UNITS[quantity], "source_file": relative,
                              "locator": f"json:/values/{quantity}"})


# ----------------------------------------------------------------------------- plan
def plan_sweep(sweep: str, parents: list[Path], output: Path, *, horizon: int | None = None,
               horizons=None, beams=None, profile: str | None = None, ensemble: int | None = None,
               beam: int | None = None, grids: dict | None = None, variants: bool = True,
               replicate: bool = True, allow_source_change: bool = False, ablations=None,
               buffers=None) -> Path:
    """Write ``plan.json`` and the variant configurations of one sweep.

    ``horizon``, ``horizons`` and ``beams`` replace the registered horizon and the two
    grids of the horizon/beam contract, and ``grids`` the Table S18 grids, for
    example to exercise the sweep on the synthetic smoke runs; ``variants=False``
    plans only the rows that are recomputed from the parents.

    ``beam`` is the beam width that the horizon sweep holds when it is not the
    parent's own (the registered 16 on a parent planned with another width): every
    point that the parent's planner registry does not hold is then a variant, and
    no replicate of the parent's configuration is planned, since the registered
    configuration is not the parent's.

    ``ablations`` (module_ablations only) attaches collected experiment directories of
    ``scripts/run_ablation.py modules`` (results/module_ablations.json): their seven
    lifecycle-measured modules join the sealed-contrast modules in one record set.

    A variant lifecycle run retrains the plants and refits the controllers, so its
    outcome mixes the changed setting with retraining variation. ``replicate`` adds
    one lifecycle run of each parent's unchanged configuration, whose outcome is
    reported next to the parent's. Variants must execute the runtime source that
    sealed the parent; ``allow_source_change`` records a mismatch instead of refusing it.

    ``buffer_sensitivity`` plans one measurement job per sealed parent (arxiv,
    proteins) on the buffer grid of the contract, or on ``buffers``; it executes no
    lifecycle (cm_experiments/buffered_admission.py).
    """
    spec = load_spec(sweep)
    if ablations and sweep != "module_ablations":
        raise SweepError("only the module_ablations sweep takes ablation experiments")
    if buffers is not None and sweep != "buffer_sensitivity":
        raise SweepError("only the buffer_sensitivity sweep takes a buffer grid")
    output = Path(output).resolve()
    if (output / "plan.json").exists():
        raise SweepError(f"{output}/plan.json exists; use a new sweep directory")
    if not parents:
        raise SweepError("give at least one --parent run")
    entries, planned, settings = [], [], {}
    for parent in parents:
        members = member_runs(parent)
        runs = [SealedRun(m) for m in members]
        if len({r.dataset for r in runs}) != 1:
            raise SweepError(f"{parent}: split runs of different datasets")
        entries.append({"dataset": runs[0].dataset, "root": str(Path(parent).resolve()),
                        "members": [str(m) for m in members], "synthetic": runs[0].synthetic,
                        "config_sha256": [canonical_sha256(r.config) for r in runs],
                        "runtime_source_sha256": sorted({read_json(m / "run_state.json")["source_sha256"]
                                                         for m in members})})
    if sweep == "horizon_beam_sweep":
        if len(entries) != 1 or len(entries[0]["members"]) != 1:
            raise SweepError("the horizon and beam sweep takes exactly one sealed run")
        run = SealedRun(Path(entries[0]["members"][0]))
        settings = horizon_beam_settings(run, spec, horizon=horizon, horizons=horizons, beams=beams,
                                         profile=profile, ensemble=ensemble, beam=beam)
        for point in settings["points"]:
            if point["source"] == "variant" and variants:
                planned.append(_variant(output, run, f"H{point['horizon']}_B{point['beam_width']}",
                                        point["patch"], {"horizon": point["horizon"],
                                                         "beam_width": point["beam_width"]}))
        if planned and replicate and settings["registered_in_parent"]:
            planned.append(_variant(output, run, "replicate", {}, {"horizon": settings["registered_horizon"],
                                                                   "beam_width": settings["registered_beam_width"],
                                                                   "replicate": True}))
    elif sweep == "sensitivity_grids":
        settings = {"grids": grids or {}}
        if variants:
            for entry in entries:
                for member in entry["members"]:
                    run = SealedRun(Path(member))
                    for parameter in SUPPORT_PARAMETERS:
                        registered = config_value(run.config, f"/support/{parameter}")
                        for value in parameter_grid(parameter, run, grids):
                            if value == registered:
                                continue
                            name = f"{safe_label(run.dataset)}_{run.root.name}_{parameter}_{value:g}"
                            planned.append(_variant(output, run, name, {f"/support/{parameter}": value},
                                                    {"parameter": parameter, "setting": value}))
                    if replicate:
                        planned.append(_variant(output, run, f"{safe_label(run.dataset)}_{run.root.name}_replicate",
                                                {}, {"parameter": "replicate", "setting": None, "replicate": True}))
    elif sweep == "module_ablations":
        settings = {"modules": [m["module"] for m in spec["modules"]], "ablations": attach_ablations(ablations)}
    elif sweep == "buffer_sensitivity":
        from .buffered_admission import plan_settings
        settings = plan_settings(entries, spec, buffers)
    current = _runtime_sha256()
    mismatch = sorted({s for e in entries for s in e["runtime_source_sha256"]} - {current})
    if planned and mismatch and not allow_source_change:
        raise SweepError(f"the parent runs were sealed by runtime source {', '.join(mismatch)} but this checkout is "
                         f"{current}; run the sweep from the parent's source snapshot (variant and parent must "
                         "execute the same runtime) or pass allow_source_change")
    plan = {"schema": PLAN_SCHEMA, "sweep": sweep, "record_set": sweep, "parents": entries,
            "settings": settings, "variants": planned, "stages": list(LIFECYCLE_THROUGH_SELECT),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "runtime_source_sha256": current, "parent_runtime_source_mismatch": mismatch if planned else []}
    write_json(output / "plan.json", plan)
    return output / "plan.json"


def _runtime_sha256() -> str:
    from cross_morse.provenance import current_source_sha256
    return current_source_sha256()


def _variant(output: Path, parent: SealedRun, name: str, patch: dict, keys: dict) -> dict:
    identifier = safe_label(name)
    config = variant_config(parent.config, patch)
    write_json(output / "variants" / identifier / "config.json", config)
    return {"id": identifier, "dataset": parent.dataset, "parent": str(parent.root), "patch": patch,
            "keys": keys, "parent_config_sha256": canonical_sha256(parent.config),
            "config_sha256": canonical_sha256(config),
            "config": f"variants/{identifier}/config.json", "output": f"runs/{identifier}"}


def parameter_grid(parameter: str, run: SealedRun, grids: dict | None) -> list[float]:
    """Registered grid of one Table S18 parameter for this run's loss."""
    if grids and parameter in grids:
        return [float(v) for v in grids[parameter]]
    constants = json.loads((ROOT / "paper" / "registered_constants.json").read_text(encoding="utf-8"))
    pointer = load_spec("sensitivity_grids")["parameters"][parameter]["grid"]
    grid = constants
    for token in pointer.split("/"):
        grid = grid[token]
    if isinstance(grid, dict):
        grid = grid["mean_BCE" if task_loss_name(run.config) == "bce" else "CE"]
    return [float(v) for v in grid]


# ----------------------------------------------------------------------------- run
def run_plan(plan_path: Path, *, data_root: Path, device: str, threads: int = 16,
             python: str = sys.executable, only: list[str] | None = None, dry_run: bool = False,
             runner=subprocess.run) -> list[list[str]]:
    """Execute every planned variant with the released lifecycle, stage by stage.

    A buffer_sensitivity plan has no lifecycle variants: its measurement jobs run
    here (``only`` selects job identifiers) and no command is returned.
    """
    plan_path = Path(plan_path).resolve()
    plan = read_json(plan_path)
    if plan.get("schema") != PLAN_SCHEMA:
        raise SweepError("not a sweep plan")
    if plan["sweep"] == "buffer_sensitivity":
        if not dry_run:
            from .buffered_admission import run_buffer_jobs
            run_buffer_jobs(plan_path, only=only)
        return []
    base = plan_path.parent
    commands = []
    for variant in plan["variants"]:
        if only and variant["id"] not in only:
            continue
        config_path = base / variant["config"]
        if canonical_sha256(read_json(config_path)) != variant["config_sha256"]:
            raise SweepError(f"{variant['id']}: variant configuration changed after planning")
        output = base / variant["output"]
        log = base / "logs" / f"{variant['id']}.log"
        for stage in plan["stages"]:
            marker = output / "stages" / f"{stage}.json"
            if marker.is_file() and read_json(marker).get("status") == "COMPLETE":
                continue
            command = lifecycle_command(config_path, output, stage, data_root=Path(data_root), device=device,
                                        threads=threads, python=python)
            commands.append(command)
            if dry_run:
                continue
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a", encoding="utf-8") as handle:
                handle.write(f"$ {' '.join(command)}\n")
                handle.flush()
                result = runner(command, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT, check=False)
            if result.returncode != 0:
                raise SweepError(f"{variant['id']}: stage {stage} exited with {result.returncode}; see {log}")
    return commands


# ----------------------------------------------------------------------------- collect
def collect(plan_path: Path) -> Path:
    """Measure every planned row from the parents and the completed variant runs."""
    plan_path = Path(plan_path).resolve()
    plan = read_json(plan_path)
    base = plan_path.parent
    collector = Collector(base, plan["record_set"])
    sweep = plan["sweep"]
    if sweep == "horizon_beam_sweep":
        collect_horizon_beam(plan, base, collector)
    elif sweep == "sensitivity_grids":
        collect_sensitivity(plan, base, collector)
    elif sweep == "module_ablations":
        collect_module_ablations(plan, collector)
    elif sweep == "buffer_sensitivity":
        from .buffered_admission import collect_buffer
        collect_buffer(plan_path, plan, collector)
    else:
        raise SweepError(f"{sweep} has no collector")
    synthetic = {p["synthetic"] for p in plan["parents"]}
    attached = {a["synthetic"] for a in plan["settings"].get("ablations") or []}
    if len(synthetic | attached) != 1:
        raise SweepError("synthetic examples and benchmark runs cannot share one sweep")
    measured = {r["keys"].get("module") for r in collector.rows}
    unsupported = [{"module": m["module"], "runtime": m["runtime"]} for m in collector.spec.get("modules", [])
                   if m["runtime"].startswith("unsupported") and m["module"] not in measured]
    not_measured = [{"module": m["module"], "runtime": m["runtime"]} for m in collector.spec.get("modules", [])
                    if m["module"] not in measured]
    results = {"schema": RESULTS_SCHEMA, "sweep": sweep, "record_set": plan["record_set"],
               "keys": collector.spec["keys"], "rows": collector.rows, "computed": collector.files,
               "plan_sha256": sha256_file(plan_path), "synthetic": synthetic.pop(),
               "parents": plan["parents"], "variants": plan["variants"], "unsupported": unsupported,
               "not_measured": not_measured if sweep == "module_ablations" else [],
               "deposits": collector.deposits, "ablations": plan["settings"].get("ablations") or [],
               "replicates": collector.replicates,
               "parent_runtime_source_mismatch": plan.get("parent_runtime_source_mismatch", []),
               "runtime_source_sha256": _runtime_sha256(),
               "collected_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    write_json(base / "results.json", results)
    return base / "results.json"


def _variant_run(base: Path, variant: dict) -> SealedRun:
    if not (base / variant["output"] / "config.json").is_file():
        raise SweepError(f"variant {variant['id']} has not been run (run_sweep.py run --plan ...)")
    run = SealedRun(base / variant["output"])
    if canonical_sha256(run.config) != variant["config_sha256"]:
        raise SweepError(f"{variant['id']}: the run's configuration is not the planned variant")
    return run


# ----------------------------------------------------------------------------- horizon and beam (S29, Fig. S5)
def reference_candidate(run: SealedRun) -> dict:
    """The parent's deployed MPC-RH rule (or its provisional MPC-RH route)."""
    candidates = run.json("select/candidates.json")
    lock = run.json("policy_lock.json") if (run.root / "policy_lock.json").is_file() else {}
    for index in (lock.get("deployed_index"), run.json("select/decision.json")["provisional_index"]):
        if index is not None and candidates[index]["route"] == "mpc_rh":
            return candidates[index]
    raise SweepError(f"{run.root.name}: neither the deployed nor the provisional rule is MPC-RH; "
                     "name the profile and ensemble of the swept configuration explicitly")


def horizon_beam_settings(run: SealedRun, spec: dict, *, horizon=None, horizons=None, beams=None,
                          profile=None, ensemble=None, beam=None) -> dict:
    """The points of the horizon and beam sweep on one parent run.

    The horizon sweep holds the registered beam width: the parent's own, or ``beam``
    when given (a parent planned with another width; the registered configuration is
    then itself a variant, since the parent's planner never ran it)."""
    controller = run.config["controller"]
    depth = int(run.config["model"]["depth"])
    if profile is None or ensemble is None:
        reference = reference_candidate(run)
        profile = reference["profile"] if profile is None else profile
        ensemble = reference["ensemble"] if ensemble is None else int(ensemble)
    registered_h = int(spec["registered"]["horizon"] if horizon is None else horizon)
    parent_b = int(controller["beam_width"])
    registered_b = parent_b if beam is None else int(beam)
    horizon_grid = [int(h) for h in (spec["grid"]["horizon"] if horizons is None else horizons)]
    beam_grid = [int(b) for b in (spec["grid"]["beam_width"] if beams is None else beams)]
    if beams is None and registered_b != int(spec["registered"]["beam_width"]):
        raise SweepError(f"{run.root.name}: beam width {registered_b} is not the registered "
                         f"{spec['registered']['beam_width']}")
    if registered_b not in beam_grid or registered_h not in horizon_grid:
        raise SweepError("the registered horizon and beam width must lie on their grids")
    parent_horizons = sorted({min(int(h), depth) for h in controller["horizons"]})
    registered_in_parent = registered_b == parent_b and min(registered_h, depth) in parent_horizons
    if beam is None and not registered_in_parent:
        raise SweepError(f"{run.root.name}: the registered horizon is not in the parent's registry")
    points = []
    wanted = [(h, registered_b) for h in horizon_grid] + [(registered_h, b) for b in beam_grid]
    for h, b in dict.fromkeys(wanted):
        inside = b == parent_b and min(h, depth) in parent_horizons
        patch = {}
        if b != parent_b:
            patch["/controller/beam_width"] = b
        if min(h, depth) not in parent_horizons:
            patch["/controller/horizons"] = sorted(set(int(x) for x in controller["horizons"]) | {h})
        points.append({"horizon": h, "beam_width": b, "source": "parent" if inside else "variant",
                       "patch": patch})
    return {"profile": profile, "ensemble": int(ensemble), "registered_horizon": registered_h,
            "registered_beam_width": registered_b, "parent_beam_width": parent_b,
            "registered_in_parent": registered_in_parent, "depth": depth, "points": points}


def _mpc_rh(run: SealedRun, horizon: int, profile: str, ensemble: int) -> tuple[int, dict, float]:
    run.require("select")
    candidates = run.json("select/candidates.json")
    decision = run.json("select/decision.json")
    depth = int(run.config["model"]["depth"])
    found = [j for j, c in enumerate(candidates) if c["route"] == "mpc_rh" and c["profile"] == profile
             and c["ensemble"] == ensemble and c.get("horizon") == min(horizon, depth)]
    if len(found) != 1:
        raise SweepError(f"{run.root.name}: expected one MPC-RH candidate at H={horizon}, {profile}, "
                         f"E={ensemble}; found {len(found)}")
    j = found[0]
    if candidates[j].get("execution_failure"):
        raise SweepError(f"{run.root.name}: the MPC-RH candidate at H={horizon} failed execution; "
                         "it has no selection effect")
    return j, candidates[j], number(decision["mean"][j], f"{run.root.name} selection effect")


def collect_horizon_beam(plan: dict, base: Path, collector: Collector) -> None:
    parent = SealedRun(Path(plan["parents"][0]["members"][0]))
    settings = plan["settings"]
    headroom = parent.json("select/entry_complete.json")["headroom"]
    bound = headroom.get("upper")
    variants = {(v["keys"]["horizon"], v["keys"]["beam_width"]): v for v in plan["variants"]
                if not v["keys"].get("replicate")}
    for point in settings["points"]:
        h, b = point["horizon"], point["beam_width"]
        if point["source"] == "parent":
            run = parent
        elif (h, b) in variants:
            run = _variant_run(base, variants[(h, b)])
        else:
            raise SweepError(f"H={h}, beam {b} needs a variant run that the plan does not contain")
        j, candidate, effect = _mpc_rh(run, h, settings["profile"], settings["ensemble"])
        cost = number(candidate["cost"], f"{run.root.name} cost")
        values = {"selection_effect_nats": effect, "expected_serial_cost": cost,
                  "oracle_headroom_bound_nats": bound,
                  "realized_fraction": effect / bound if bound is not None and bound < 0 else None}
        detail = {"source": point["source"], "run": str(run.root), "candidate_index": j,
                  "candidate": route_identity(candidate), "beam_width_executed": run.config["controller"]["beam_width"],
                  "headroom": headroom, "headroom_run": str(parent.root),
                  "realized_fraction_definition": "selection effect / U(Delta_oracle) of the parent run"}
        runs = [run] if run is parent else [run, parent]
        collector.add(f"{run.dataset}_H{h}_B{b}", {"dataset": run.dataset, "horizon": h, "beam_width": b},
                      values, detail, runs)
    for variant in plan["variants"]:
        if variant["keys"].get("replicate"):
            run = _variant_run(base, variant)
            h, b = variant["keys"]["horizon"], variant["keys"]["beam_width"]
            j, candidate, effect = _mpc_rh(run, h, settings["profile"], settings["ensemble"])
            _, _, parent_effect = _mpc_rh(parent, h, settings["profile"], settings["ensemble"])
            collector.replicates.append({"variant": variant["id"], "horizon": h, "beam_width": b,
                                         "selection_effect_nats": effect, "parent_selection_effect_nats": parent_effect,
                                         "expected_serial_cost": candidate["cost"], "inputs": run.inputs()})


# ----------------------------------------------------------------------------- Table S18 grids
def collect_sensitivity(plan: dict, base: Path, collector: Collector) -> None:
    grids = plan["settings"].get("grids") or {}
    variants = {}
    for variant in plan["variants"]:
        if variant["keys"].get("replicate"):
            parent, run = SealedRun(Path(variant["parent"])), _variant_run(base, variant)
            reference, again = selection_outcome(parent), selection_outcome(run)
            collector.replicates.append({
                "variant": variant["id"], "parent": variant["parent"],
                **route_quantities(again["candidates"][again["selected"]],
                                   reference["candidates"][reference["selected"]]),
                "eligible_count": len(again["eligible"]), "parent_eligible_count": len(reference["eligible"]),
                "inputs": run.inputs()})
            continue
        variants[(variant["parent"], variant["keys"]["parameter"], float(variant["keys"]["setting"]))] = variant
    for entry in plan["parents"]:
        members = [SealedRun(Path(m)) for m in entry["members"]]
        dataset = entry["dataset"]
        first = members[0]
        for parameter in SUPPORT_PARAMETERS:
            for value in parameter_grid(parameter, first, grids):
                rows, used = [], list(members)
                for run in members:
                    support = run.json("fit/full/audit.json")["support"]
                    check_support_rule(run, support)
                    thresholds = dict(support["thresholds"])
                    thresholds[parameter] = value
                    row = support_summary(support, thresholds["p_min"], thresholds["n_min"],
                                          thresholds["max_normalized_weight"])
                    reference = selection_outcome(run)
                    registered = run.config["support"][parameter]
                    if value == registered:
                        routed = run
                    else:
                        key = (str(run.root), parameter, float(value))
                        routed = _variant_run(base, variants[key]) if key in variants else None
                    if routed is not None:
                        outcome = selection_outcome(routed)
                        chosen = outcome["candidates"][outcome["selected"]]
                        base_route = reference["candidates"][reference["selected"]]
                        row.update(eligible_count=float(len(outcome["eligible"])),
                                   **route_quantities(chosen, base_route), route=route_identity(chosen))
                        if routed is not run:
                            used.append(routed)
                    rows.append({"split": run.root.name, **row})
                collector.add(f"{dataset}_{parameter}_{value:g}",
                              {"parameter": parameter, "setting": f"{value:g}", "dataset": dataset},
                              _mean_quantities(rows), {"per_run": rows, "aggregation": _aggregation(members)},
                              used)
        for parameter in MARGIN_PARAMETERS:
            for value in parameter_grid(parameter, first, grids):
                rows = []
                for run in members:
                    reference = selection_outcome(run)
                    outcome = selection_outcome(run, **{parameter: value})
                    chosen = outcome["candidates"][outcome["selected"]]
                    rows.append({"split": run.root.name, "eligible_count": float(len(outcome["eligible"])),
                                 **route_quantities(chosen, reference["candidates"][reference["selected"]]),
                                 "route": route_identity(chosen)})
                collector.add(f"{dataset}_{parameter}_{value:g}",
                              {"parameter": parameter, "setting": f"{value:g}", "dataset": dataset},
                              _mean_quantities(rows), {"per_run": rows, "aggregation": _aggregation(members)},
                              members)
        for value in parameter_grid("clip", first, grids):
            rows = []
            for run in members:
                check_admission_replay(run)
                replay = admission_replay(run, value)
                rows.append({"split": run.root.name, **{k: v for k, v in replay.items() if not k.startswith("_")}})
            values = _mean_quantities(rows)
            if len(members) > 1:
                means = [r["clipped_mean_nats"] for r in rows]
                values["admission_upper_nats"] = repeated_split_upper(means)
                values["raw_maximum_harm_nats"] = max(r["raw_maximum_harm_nats"] for r in rows)
                epsilon = rows[0]["epsilon_a"]
                values["admitted"] = float(values["clipped_mean_nats"] <= -epsilon and values["admission_upper_nats"] < 0
                                           and all(r["metric_gate"] for r in rows))
            collector.add(f"{dataset}_clip_{value:g}", {"parameter": "clip", "setting": f"{value:g}", "dataset": dataset},
                          values, {"per_run": rows, "aggregation": _aggregation(members)}, members)


def _aggregation(members: list[SealedRun]) -> str:
    if len(members) == 1:
        return "single sealed run"
    return ("mean over the official split runs; repeated-split admission upper limit = small-sample t "
            "limit over the per-split clipped effects; decision = descriptive rule on those two statistics")


def _mean_quantities(rows: list[dict]) -> dict:
    names = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)
             and k in UNITS]
    out = {}
    for name in names:
        values = [r[name] for r in rows if name in r]
        if len(values) == len(rows):
            out[name] = stats.fmean(values)
    return out


# ----------------------------------------------------------------------------- Table S17 modules
def attach_ablations(paths) -> list[dict]:
    """Collected module studies of cm_experiments.ablation_lifecycle, identified by their results' SHA-256."""
    from .ablation_lifecycle import module_results
    from .ensemble_grid import RunnerError
    attached = []
    for path in paths or ():
        experiment = Path(path).resolve()
        results_path = experiment / "results" / "module_ablations.json"
        try:
            results = module_results(results_path)
        except (RunnerError, OSError) as exc:
            raise SweepError(f"{experiment}: not a collected modules study ({exc}); run "
                             "scripts/run_ablation.py modules export first") from None
        if results.get("truncated") and not results["synthetic"]:
            raise SweepError(f"{experiment}: the modules study was truncated by --max-graphs (a software check), "
                             "not a measurement of the held-out role")
        attached.append({"experiment": str(experiment), "results": "results/module_ablations.json",
                         "results_sha256": sha256_file(results_path), "plan_sha256": results["plan_sha256"],
                         "synthetic": bool(results["synthetic"]), "role": results["role"],
                         "truncated": bool(results.get("truncated"))})
    names = [safe_label(Path(a["experiment"]).name) for a in attached]
    if len(set(names)) != len(names):
        raise SweepError("attached ablation experiments need distinct directory names")
    return attached


def collect_ablation_rows(attached: dict, collector: Collector) -> None:
    """The lifecycle-measured modules of one attached study; its files are deposited under raw/ablations/."""
    from .ablation_lifecycle import module_results
    from .ensemble_grid import RunnerError
    try:
        results = module_results(Path(attached["experiment"]) / attached["results"], attached["results_sha256"])
    except (RunnerError, OSError) as exc:
        raise SweepError(f"{attached['experiment']}: {exc}") from None
    name, base = safe_label(Path(attached["experiment"]).name), Path(results["base"])
    deposited = {"plan.json", attached["results"]}
    for row in results["rows"]:
        keys = {"module": row["module"], "arm": row["arm"], "dataset": row["dataset"]}
        detail = {"source": "variant lifecycles of cm_experiments/ablation_lifecycle.py", "experiment": name,
                  "results": {"file": f"raw/ablations/{name}/{attached['results']}", "sha256": attached["results_sha256"]},
                  "files": [{"file": f"raw/ablations/{name}/{f['file']}", "sha256": f["sha256"]} for f in row["files"]],
                  "role": results["role"], "complete_arm": results["complete"], **row["detail"]}
        collector.add(f"{row['module']}_{row['arm']}_{row['dataset']}", keys, row["values"], detail, [])
        deposited |= {f["file"] for f in row["files"]}
    for relative in sorted(deposited):
        collector.deposits.append({"path": f"ablations/{name}/{relative}", "from": str(base / relative),
                                   "sha256": sha256_file(base / relative)})


def collect_module_ablations(plan: dict, collector: Collector) -> None:
    for attached in plan["settings"].get("ablations") or []:
        collect_ablation_rows(attached, collector)
    for entry in plan["parents"]:
        members = [SealedRun(Path(m)) for m in entry["members"]]
        dataset = entry["dataset"]
        for module, arm, function in (("radius_uncertainty", "one_step", _radius_contrast),
                                      ("sequence_risk", "sequence_deleted", _sequence_deletion),
                                      ("full_lean_profile", "lean", _full_lean)):
            rows = [dict(split=run.root.name, **found) for run in members
                    if (found := function(run)) is not None]
            if not rows:
                continue
            values = {k: stats.fmean(r[k] for r in rows) for k in rows[0]
                      if k in collector.spec["quantities"]}
            collector.add(f"{module}_{arm}_{dataset}", {"module": module, "arm": arm, "dataset": dataset},
                          values, {"per_run": rows, "runs_with_contrast": len(rows), "runs": len(members),
                                   "aggregation": "mean over the runs that hold the contrast"}, members)


def _confirmed_contrast(run: SealedRun, kind: str) -> tuple[dict, dict] | None:
    run.require("confirm")
    decision = run.json("confirm/decision.json")
    provisional = decision["provisional_index"]
    rows = [c for c in decision["contrasts"] if c["kind"] == kind and c["candidate_index"] == provisional
            and c.get("available", True) and c.get("mean") is not None]
    return (rows[0], decision) if rows else None


def _radius_contrast(run: SealedRun) -> dict | None:
    found = _confirmed_contrast(run, "radius")
    if found is None:
        return None
    contrast, _ = found
    return {"loss_change_nats": -float(contrast["mean"]), "contrast": contrast,
            "definition": "minus the D_confirm mean of radius minus one-step"}


def _deployed(run: SealedRun) -> tuple[int, list[dict]]:
    run.require("admit")
    lock = run.json("policy_lock.json")
    return int(lock["deployed_index"]), run.json("select/candidates.json")


def _sequence_deletion(run: SealedRun) -> dict | None:
    deployed, candidates = _deployed(run)
    rule = candidates[deployed]
    if rule["route"] != "mpc_rh":
        return None
    decision = run.json("select/decision.json")
    one_step = [j for j, c in enumerate(candidates) if c["route"] == "one_step"
                and c["profile"] == rule["profile"] and c["ensemble"] == rule["ensemble"]]
    if len(one_step) != 1 or decision["mean"][one_step[0]] is None or decision["mean"][deployed] is None:
        raise SweepError(f"{run.root.name}: the matched one-step candidate has no selection effect")
    return {"loss_change_nats": float(decision["mean"][one_step[0]]) - float(decision["mean"][deployed]),
            "deployed_index": deployed, "one_step_index": one_step[0],
            "definition": "nested D_select effect of one-step minus that of the deployed MPC-RH rule"}


def _full_lean(run: SealedRun) -> dict | None:
    found = _confirmed_contrast(run, "full")
    if found is None:
        return None
    contrast, decision = found
    candidates = run.json("select/candidates.json")
    full = candidates[contrast["candidate_index"]]
    lean = candidates[contrast["predecessor_index"]]
    if full["profile"] != "full" or lean["profile"] != "lean":
        raise SweepError(f"{run.root.name}: the Full/Lean contrast does not pair a Full rule with its Lean predecessor")
    comparator = candidates[decision["comparator_index"]]["cost"]
    added = full["cost"] - comparator
    if added <= 0:
        raise SweepError(f"{run.root.name}: the Full rule adds no cost over q_C")
    return {"retained_effect_fraction": 1.0 - float(contrast["mean"]),
            "cost_fraction": (lean["cost"] - comparator) / added, "contrast": contrast,
            "costs": {"full": full["cost"], "lean": lean["cost"], "q_C": comparator},
            "definition": "Lean retains one minus the Full-minus-Lean retained-fraction contrast on D_confirm; "
                          "cost fraction = (C_Lean - C_qC) / (C_Full - C_qC)"}
