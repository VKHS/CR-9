"""Equal-budget 2^3 component factorials (Supplementary Tables S16, S16b; record set ``factorials``).

    ogbn-arxiv   (S16)   actions (A), observer (O), risk-controlled execution (X)
    PascalVOC-SP (S16b)  spectral information (S), conditioned observer (O), risk-controlled
                         execution (X), in the receding-horizon regime (MPC-RH)

Every cell is the confirmed feedback route q_M of a sealed parent run, executed with the
parent's frozen plant members and fitted controllers -- one fitting budget shared by all
cells -- with the cell's components switched on or off.  A switch is executed with the
released runtime and a thin wrapper of its fitted Controller:

    profile       S: "full" | "lean"                the runtime's two fitted sensor profiles (the Lean
                                                    profile zeros the spectral and reachability coordinates)
    observer      O: "conditioned" | "memoryless"   the fitted conditioned observer, or the decision state
                                                    re-initialised from each new measurement (no filtering)
    support       X: "registered" | "unrestricted"  the registered support masks with their recovery, or every
                                                    registered action admissible in every stratum
    action_bank   A: "registered" | "nominal"       the controller's choice set, or A0 only (recovery unchanged)

The factor-to-switch table of a plan (``FACTOR_SWITCHES``, overridable) is a declared
reconstruction: the manuscript names the components but not their executable definitions.

Cell protocol (the six quantities of the record set):
    ld_integrated_loss       log-depth-integrated CE of main Eq. (6) on the registered depth grid
                             {1, 2, 4, ..., L} (cross_morse.training.audit_training's grid): the loss of the
                             prediction head after the first d executed actions (exact replay of the cell's
                             own action and anchor trace with cross_morse.policies.execute_policy), integrated
                             per unit by the trapezoid rule in log2 depth, then unit-weighted
    interval_low/high        central 95% paired whole-unit bootstrap of that mean (the registered
                             cross_morse.statistics.paired_bootstrap); one resampling stream shared by the
                             eight cells, so the joint draws give every contrast's interval (raw result)
    official_metric          official metric of the cell's ensemble output, in percent / F1 points
    activity                 share of executed actions other than A0, over sites, members and inputs
    relative_time            synchronized wall time of the cell's executions over cell 000's
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np

from .artifacts import sha256, write_json
from .ensemble_grid import (FEEDBACK_ROUTES, ParentContext, RunnerError, SealedRun, candidate_identity, check_identity,
                            check_role, code_files, create_plan, ensemble_probabilities, environment_record,
                            job_directory, json_safe, load_plan, member_runs, provenance_hardware, provenance_period,
                            read_json, runtime_source_sha256, save_array, save_arrays, selected_jobs, slug, utc_now,
                            write_record_set)

CELLS = ("000", "100", "010", "001", "110", "101", "011", "111")
DESIGNS = {"ogbn-arxiv": {"table": "S16", "factors": ("A", "O", "X"), "route": None},
           "PascalVOC-SP": {"table": "S16b", "factors": ("S", "O", "X"), "route": "mpc_rh"}}
FACTOR_SWITCHES = {"A": ("action_bank", ("nominal", "registered")),
                   "S": ("profile", ("lean", "full")),
                   "O": ("observer", ("memoryless", "conditioned")),
                   "X": ("support", ("unrestricted", "registered"))}
SWITCH_VALUES = {"action_bank": ("registered", "nominal"), "profile": ("full", "lean"),
                 "observer": ("conditioned", "memoryless"), "support": ("registered", "unrestricted")}
METRIC_UNITS = {"accuracy": "%", "macro_f1": "F1 points", "rocauc": "AUC points", "ap": "AP points"}


# ----------------------------------------------------------------------------- definitions (torch-free)
def depth_grid(depth: int) -> list[int]:
    """The geometric grid of main Eq. (6) as cross_morse.training.audit_training builds it."""
    depth = int(depth)
    return sorted(set([1, depth] + [2 ** i for i in range(depth.bit_length()) if 2 ** i <= depth]))


def log_depth_integral(values, depths) -> np.ndarray:
    """Main Eq. (6): trapezoid rule in log2 depth over [d_1, d_K], normalized by log2(d_K / d_1).

    ``values`` has one row per depth (axis 0) and one column per unit.
    """
    values = np.asarray(values, dtype=np.float64)
    x = np.log2(np.asarray(depths, dtype=np.float64))
    if len(x) == 1:
        return values[0]
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return integrate(values, x, axis=0) / (x[-1] - x[0])


def cell_switches(cell: str, factors, base_profile: str, table=None) -> dict:
    table = table or FACTOR_SWITCHES
    switches = {"profile": base_profile, "observer": "conditioned", "support": "registered", "action_bank": "registered"}
    for factor, bit in zip(factors, cell):
        name, values = table[factor]
        switches[name] = values[int(bit)]
    for name, value in switches.items():
        if value not in SWITCH_VALUES[name]:
            raise RunnerError(f"switch {name}={value!r} is not executable; choose one of {SWITCH_VALUES[name]}")
    return switches


def profile_counterpart(candidates: list[dict], candidate: dict, profile: str) -> dict:
    """The registered candidate with the same route, configuration and ensemble at another sensor profile."""
    if candidate.get("profile") == profile:
        return candidate
    keys = ("route", "ensemble", "horizon", "kappa", "depth")
    for index, other in enumerate(candidates):
        if other.get("profile") == profile and all(other.get(k) == candidate.get(k) for k in keys):
            return dict(other, registry_index=index)
    raise RunnerError(f"no registered {candidate['route']} candidate at the {profile} profile")


def base_policy(run: SealedRun, design: dict, *, allow_fixture: bool) -> tuple[dict, dict]:
    confirmed = run.confirmed()
    wanted = design["route"]
    if confirmed["route"] in FEEDBACK_ROUTES and (wanted is None or confirmed["route"] == wanted):
        return confirmed, {"source": "confirmed feedback route q_M", "fixture_override": False}
    if not (allow_fixture and run.synthetic):
        raise RunnerError(f"{run.root.name}: the confirmed policy is {confirmed['route']}; the factorial needs a "
                          f"confirmed {wanted or 'feedback'} route")
    comparator = run.comparator()
    for index, c in enumerate(run.candidates()):
        if (c["route"] == (wanted or "radius") and c["ensemble"] == comparator["ensemble"]
                and c.get("profile") == "full"):
            return dict(c, registry_index=index), {"source": "registered feedback candidate (synthetic fixture)",
                                                   "fixture_override": True}
    raise RunnerError(f"{run.root.name}: no registered {wanted or 'radius'} candidate for the fixture")


# ----------------------------------------------------------------------------- planning
def plan_factorial(parents, output, *, role="test", reps=None, warmup=1, max_graphs=0, allow_fixture=False,
                   switches=None) -> Path:
    if not parents:
        raise RunnerError("give the sealed ogbn-arxiv and/or PascalVOC-SP parent run(s)")
    table = dict(FACTOR_SWITCHES)
    for factor, mapping in (switches or {}).items():
        if factor not in table or len(mapping) != 2:
            raise RunnerError(f"switch override must map a factor to [switch, [off, on]]: {factor}")
        table[factor] = (mapping[0], tuple(mapping[1]))
    if int(warmup) < 0 or int(max_graphs) < 0:
        raise RunnerError("warm-up and max_graphs must be nonnegative")
    jobs, datasets = [], []
    for parent in parents:
        runs = [SealedRun(m) for m in member_runs(parent)]
        if len(runs) != 1 or runs[0].dataset not in DESIGNS:
            raise RunnerError(f"{parent}: the factorials run on one ogbn-arxiv or PascalVOC-SP run each")
        run = runs[0]
        if any(d["dataset"] == run.dataset for d in datasets):
            raise RunnerError(f"{run.dataset}: one parent run per dataset")
        check_role(run, role)
        design = DESIGNS[run.dataset]
        base, origin = base_policy(run, design, allow_fixture=allow_fixture)
        candidates = run.candidates()
        cells = []
        for cell in CELLS:
            values = cell_switches(cell, design["factors"], base.get("profile", "full"), table)
            candidate = profile_counterpart(candidates, base, values["profile"])
            cells.append({"cell": cell, "factors": dict(zip(design["factors"], map(int, cell))),
                          "switches": values, "candidate": candidate})
        seeds = [int(s) for s in run.config["training"]["seeds"][:int(base["ensemble"])]]
        jobs.append({"job": len(jobs), "dataset": run.dataset, "slug": slug(run.dataset), "member": run.root.name,
                     "run": str(run.root), "identity": run.identity(), "table": design["table"],
                     "factors": list(design["factors"]), "base": candidate_identity(base), "base_origin": origin,
                     "cells": cells, "member_seeds": seeds, "depth_grid": depth_grid(run.config["model"]["depth"]),
                     "bootstrap": {"reps": int(reps or run.config["admission"]["bootstrap_reps"]),
                                   "seed": int(run.config["seed"]) + 17}})
        datasets.append({"dataset": run.dataset, "slug": slug(run.dataset), "synthetic": run.synthetic})
    return create_plan(output, {
        "experiment": "Equal-budget 2^3 component factorials", "record_set": "factorials", "role": role,
        "warmup": int(warmup), "max_graphs": int(max_graphs), "datasets": datasets, "jobs": jobs,
        "factor_switches": {k: [v[0], list(v[1])] for k, v in table.items()},
        "cell_definition_status": ("declared reconstruction: the manuscript names the components but not their "
                                   "executable definitions; confirm the factor-to-switch table before execution"),
        "fitting_budget": "the parent's fitted controllers and frozen members, shared by all eight cells",
        "protocol": ("eight cells of the confirmed feedback route with components switched; per-depth head losses by "
                     "exact replay of each cell's action and anchor trace; Eq. (6) log-depth integration per unit; "
                     "one paired bootstrap stream shared by the cells")})


# ----------------------------------------------------------------------------- statistics (torch-free)
def joint_bootstrap(unit_values: dict, weights, *, reps: int, seed: int) -> np.ndarray:
    """Weighted unit means of every cell under one resampling stream ([reps, cells]).

    The draws are those of cross_morse.statistics.paired_bootstrap with the same seed, so each
    column's quantiles are that function's marginal interval.
    """
    names = list(unit_values)
    matrix = np.stack([np.asarray(unit_values[c], dtype=np.float64) for c in names], axis=1)
    w = np.ones(len(matrix)) if weights is None else np.asarray(weights, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty((reps, len(names)), dtype=np.float64)
    for start in range(0, reps, 1024):
        ids = rng.integers(0, len(matrix), size=(min(1024, reps - start), len(matrix)))
        bw = w[ids]
        draws[start:start + len(ids)] = np.einsum("ru,ruc->rc", bw, matrix[ids]) / bw.sum(axis=1)[:, None]
    return draws


def cell_statistics(unit_dil: dict, weights, factors, *, reps: int, seed: int) -> tuple[dict, list[dict], np.ndarray]:
    """Cell means with their registered marginal intervals, and every contrast with its joint interval."""
    from cm_theory.factorial import all_effects, anchored_difference
    from cross_morse.statistics import paired_bootstrap
    if list(unit_dil) != list(CELLS):
        raise RunnerError(f"the factorial needs the cells {list(CELLS)} in order")
    cells = {}
    for cell, values in unit_dil.items():
        boot = paired_bootstrap(np.asarray(values, dtype=np.float64), weights=weights, reps=reps, alpha=0.025, seed=seed)
        cells[cell] = {"ld_integrated_loss": boot["mean"], "interval_low": boot["lower"], "interval_high": boot["upper"],
                       "bootstrap_se": boot["se"], "units": boot["units"]}
    draws = joint_bootstrap(unit_dil, weights, reps=reps, seed=seed)
    point = {c: cells[c]["ld_integrated_loss"] for c in CELLS}
    effects = all_effects(point)
    replicate_effects = [all_effects(dict(zip(CELLS, row))) for row in draws]
    contrasts = []
    for positions, value in effects.items():
        series = np.array([r[positions] for r in replicate_effects])
        contrasts.append({"contrast": ":".join(factors[p] for p in positions), "effect_coded_nats": value,
                          "interval_low": float(np.quantile(series, 0.025)),
                          "interval_high": float(np.quantile(series, 0.975)),
                          "baseline_anchored_nats": anchored_difference(point, positions)})
    return cells, contrasts, draws


# ----------------------------------------------------------------------------- controller switches (torch)
class MemorylessObserver:
    """O = 0: each measurement re-initialises the decision state (no conditioned filtering or dynamics)."""

    def __init__(self, controller):
        self._controller = controller

    def __getattr__(self, name):
        return getattr(self._controller, name)

    def observe(self, state, y, depth, previous_action):
        base = self._controller
        base.counters["observer_updates"] += 1
        base.counters["approx_dense_multiply_adds"] += len(y) * base.observer.dimension
        return base.observer.start(y, previous_action, depth)


class NominalActionBank:
    """A = 0: the controller may choose only A0; recovery actions remain under the support switch."""

    def __init__(self, controller):
        self._controller = controller

    def __getattr__(self, name):
        return getattr(self._controller, name)

    def supported(self, y, depth):
        return [0] if 0 in self._controller.supported(y, depth) else []


def unrestricted_controller(path):
    """The fitted controller with every registered (stratum, action) cell admissible (X = 0)."""
    import torch
    from cross_morse.learning import Controller
    payload = dict(torch.load(Path(path) / "controller.pt", map_location="cpu", weights_only=False))
    support = copy.deepcopy(payload["support"])
    controller = payload["config"]["controller"]
    strata = int(controller.get("depth_bins", 4)) * (len(controller.get("entropy_bins", [1 / 3, 2 / 3])) + 1)
    for stratum in range(strata):
        for action in range(10):
            cell = support["cells"].setdefault(f"{stratum}:{action}", {"stratum": stratum, "action": action})
            cell["qualified"] = True
    support["switch"] = "unrestricted: every registered action admissible in every stratum (factorial X = 0)"
    payload["support"] = support
    return Controller(payload, "cpu")


def cell_controller(ctx: ParentContext, candidate: dict, switches: dict, cache: dict):
    profile = candidate.get("profile", "full")
    key = (profile, switches["support"])
    if key not in cache:
        cache[key] = (ctx.controller(profile) if switches["support"] == "registered"
                      else unrestricted_controller(ctx.run.root / "fit" / ("full" if profile == "none" else profile)))
    controller = cache[key]
    if switches["observer"] == "memoryless":
        controller = MemorylessObserver(controller)
    if switches["action_bank"] == "nominal":
        controller = NominalActionBank(controller)
    return controller


# ----------------------------------------------------------------------------- execution (torch)
def run_factorial(output, *, job=None, device="cuda", trust=False) -> list[Path]:
    if not trust:
        raise RunnerError("loading local checkpoints requires explicit trust (--trust-local-checkpoints)")
    output = Path(output).resolve()
    plan = load_plan(output, "factorials")
    done = []
    for entry in selected_jobs(plan, job):
        directory = job_directory(output, entry)
        if not (directory / "factorial.json").is_file():
            _run_factorial_job(plan, entry, directory, device)
        done.append(directory / "factorial.json")
    return done


def _execute(model, x, graph, config, candidate, controller, reference, depths, rows):
    """One member execution and the head probabilities (``rows`` only) after the first d executed actions."""
    from cross_morse.policies import execute_policy
    final, trace = execute_policy(model, x, graph, config, candidate, controller, reference)
    actions, anchors = list(trace["actions"]), list(trace["safe_depths"])
    by_depth = []
    for d in depths:
        if d >= len(actions):
            by_depth.append(np.asarray(final[rows]))  # a stopped policy's output holds at every later depth
        else:
            replay, _ = execute_policy(model, x, graph, config, candidate, None, reference,
                                       replay_actions=actions[:d], replay_safe_depths=anchors[:d])
            by_depth.append(np.asarray(replay[rows]))
    return by_depth, trace


def _run_factorial_job(plan, entry, directory, device):
    from cross_morse.tasks import sample_losses, task_spec
    run = SealedRun(entry["run"])
    check_identity(run, entry["identity"])
    role = plan["role"]
    check_role(run, role)
    started = utc_now()
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / "environment.json").exists():
        write_json(directory / "environment.json", environment_record(device))
    ctx = ParentContext(run, device, through="evaluate" if role == "test" else "confirm")
    config, spec, depths = ctx.config, task_spec(ctx.config), entry["depth_grid"]
    models = {seed: ctx.plant(seed) for seed in entry["member_seeds"]}
    cache = {}
    cells = [(c, c["candidate"], cell_controller(ctx, c["candidate"], c["switches"], cache)) for c in entry["cells"]]
    labels, units = ctx.labels(role), ctx.units(role)
    unit_ids = np.unique(units)
    unit_index = np.searchsorted(unit_ids, units)
    sums = {c["cell"]: np.zeros((len(depths), len(unit_ids))) for c, _, _ in cells}
    counts = np.zeros(len(unit_ids))
    terminal = {c["cell"]: None for c, _, _ in cells}
    ledger = {c["cell"]: {"wall_seconds": 0.0, "deployed_work": 0.0, "actions": 0, "active": 0, "inputs": 0,
                          "feasible": True, "terminal_depths": []} for c, _, _ in cells}
    first_seed = entry["member_seeds"][0]
    for count, (_, x, graph, rows, positions) in enumerate(ctx.episodes(role)):
        if plan["max_graphs"] and count >= plan["max_graphs"]:
            break
        if count == 0:
            for _ in range(plan["warmup"]):          # discarded: lazy initialisation outside the timed cells
                for _, candidate, controller in cells:
                    _execute(models[first_seed], x, graph, config, candidate, controller, ctx.reference,
                             depths[-1:], rows)
        y, index = labels[positions], unit_index[positions]
        valid = None
        for cell, candidate, controller in cells:
            name = cell["cell"]
            per_member = []
            for seed in entry["member_seeds"]:
                by_depth, trace = _execute(models[seed], x, graph, config, candidate, controller, ctx.reference,
                                           depths, rows)
                per_member.append(by_depth)
                actions = list(trace["actions"])
                ledger[name]["wall_seconds"] += float(trace["wall_seconds"])
                ledger[name]["deployed_work"] += float(trace.get("deployed_multiply_add_work", trace["multiply_add_work"]))
                ledger[name]["actions"] += len(actions)
                ledger[name]["active"] += sum(a != 0 for a in actions)
                ledger[name]["feasible"] &= bool(trace.get("feasible", True))
                ledger[name]["terminal_depths"].append(int(trace["terminal_depth"]))
            ledger[name]["inputs"] += 1
            for k in range(len(depths)):
                ensemble = ensemble_probabilities(member[k] for member in per_member)
                values = sample_losses(ensemble, y, spec)
                if valid is None:
                    valid = np.isfinite(values)
                    np.add.at(counts, index[valid], 1)
                np.add.at(sums[name][k], index[valid], values[valid])
                if k == len(depths) - 1:
                    if terminal[name] is None:
                        terminal[name] = np.full((len(units), ensemble.shape[1]), np.nan, dtype=np.float32)
                    terminal[name][positions] = ensemble
    write_factorial_result(directory, plan, entry, sums=sums, counts=counts, unit_ids=unit_ids, ledger=ledger,
                           terminal=terminal, labels=labels, config=config, parent_files=run.inputs(), started=started)


def write_factorial_result(directory, plan, entry, *, sums, counts, unit_ids, ledger, terminal, labels, config,
                           parent_files, started, environment=None) -> Path:
    """Cell statistics from the accumulated per-unit depth losses, and the job's raw outputs (factorial.json last).

    ``sums[cell]`` is [depths, units] summed sample losses, ``counts`` the scored samples per unit,
    ``terminal[cell]`` the ensemble output on the role rows and ``ledger[cell]`` the execution totals.
    """
    from cross_morse.evaluation import independent_graphs
    from cross_morse.tasks import metric_score, task_spec
    directory = Path(directory)
    if environment is not None and not (directory / "environment.json").exists():
        write_json(directory / "environment.json", json_safe(environment))
    spec, depths, role = task_spec(config), entry["depth_grid"], plan["role"]
    scored = counts > 0
    weights = None if independent_graphs(config) else counts[scored]
    unit_dil, arrays = {}, {"unit_id": unit_ids[scored], "unit_weight": counts[scored], "depth_grid": np.asarray(depths)}
    for name in CELLS:
        means = sums[name][:, scored] / counts[scored]
        unit_dil[name] = log_depth_integral(means, depths)
        arrays[f"cell{name}_unit_depth_loss"] = means
        arrays[f"cell{name}_unit_dil"] = unit_dil[name]
    statistics, contrasts, draws = cell_statistics(unit_dil, weights, entry["factors"],
                                                   reps=entry["bootstrap"]["reps"], seed=entry["bootstrap"]["seed"])
    arrays["joint_bootstrap_cell_means"] = draws
    truncated = bool(plan["max_graphs"])
    rows = []
    baseline = ledger["000"]["wall_seconds"]
    for cell in entry["cells"]:
        name = cell["cell"]
        rows_mask = np.isfinite(terminal[name]).all(axis=1)
        if not truncated and not rows_mask.all():
            raise RunnerError(f"cell {name}: role rows without a prediction")
        prediction_sha = save_array(directory / "predictions" / f"cell{name}.npy", terminal[name])
        metric = metric_score(terminal[name][rows_mask], labels[rows_mask], spec)
        item = ledger[name]
        rows.append({"cell": name, "factors": cell["factors"], "switches": cell["switches"],
                     "candidate": candidate_identity(cell["candidate"]), **statistics[name],
                     "official_metric": 100.0 * metric, "metric_name": spec.metric,
                     "activity": item["active"] / item["actions"] if item["actions"] else 0.0,
                     "relative_time": item["wall_seconds"] / baseline, "wall_seconds": item["wall_seconds"],
                     "deployed_work": item["deployed_work"], "inputs": item["inputs"], "feasible": item["feasible"],
                     "integration_domain": {"depth_grid": depths, "rule": "Eq. (6) trapezoid in log2 depth",
                                            "terminal_depths": sorted(set(item["terminal_depths"])),
                                            "after_stop": "the stopped policy's output at every later depth"},
                     "terminal_predictions": {"file": f"predictions/cell{name}.npy", "sha256": prediction_sha}})
    unit_file = save_arrays(directory / "unit_dil.npz", **arrays)
    write_json(directory / "factorial.json", json_safe({
        "schema": "cm-factorial-1", "dataset": entry["dataset"], "table": entry["table"], "member": entry["member"],
        "factors": entry["factors"], "role": role, "cells": rows, "contrasts": contrasts,
        "member_seeds": entry["member_seeds"], "base": entry["base"], "base_origin": entry["base_origin"],
        "factor_switches": plan["factor_switches"], "truncated": truncated, "max_graphs": plan["max_graphs"],
        "warmup": plan["warmup"], "bootstrap": entry["bootstrap"],
        "unit_dil": {"file": "unit_dil.npz", "sha256": unit_file},
        "identities": {"runtime_source_sha256": runtime_source_sha256(), "parent": entry["identity"],
                       "plan_sha256": sha256(directory.parents[2] / "plan.json"), "parent_files_read": parent_files},
        "units": {"official_metric": METRIC_UNITS[spec.metric], "relative_time": "cell 000 = 1"},
        "started_utc": started, "finished_utc": utc_now()}))
    return directory / "factorial.json"


# ----------------------------------------------------------------------------- export (torch-free)
QUANTITY_UNITS = {"ld_integrated_loss": "nats", "interval_low": "nats", "interval_high": "nats",
                  "activity": "fraction", "relative_time": "cell 000 = 1"}


def export_factorial(output, *, root=None, replace=False, allow_partial=False, hardware=None, period=None,
                     command=None) -> Path:
    output = Path(output).resolve()
    plan = load_plan(output, "factorials")
    files, rows, results, environments, inputs = code_files(output), [], [], [], []
    for entry in plan["jobs"]:
        directory = job_directory(output, entry)
        path = directory / "factorial.json"
        if not path.is_file():
            raise RunnerError(f"{entry['dataset']}: job {entry['job']} has not completed")
        result = read_json(path)
        if [c["cell"] for c in result["cells"]] != list(CELLS) or result["role"] != plan["role"]:
            raise RunnerError(f"{path}: cells or role differ from the plan")
        if result["truncated"] and not allow_partial:
            raise RunnerError(f"{entry['dataset']}: a run truncated by max_graphs is not a complete measurement")
        base = f"raw/{entry['slug']}"
        files[f"{base}/factorial.json"] = path
        files[f"{base}/unit_dil.npz"] = directory / "unit_dil.npz"
        files[f"{base}/environment.json"] = directory / "environment.json"
        for i, cell in enumerate(result["cells"]):
            for quantity in ("ld_integrated_loss", "interval_low", "interval_high", "official_metric", "activity",
                             "relative_time"):
                unit = result["units"]["official_metric"] if quantity == "official_metric" else QUANTITY_UNITS[quantity]
                if cell[quantity] is None:
                    raise RunnerError(f"{entry['dataset']} cell {cell['cell']}: {quantity} is not estimable")
                rows.append({"dataset": entry["dataset"], "cell": cell["cell"], "quantity": quantity,
                             "value": cell[quantity], "unit": unit, "source_file": f"{base}/factorial.json",
                             "locator": f"json:/cells/{i}/{quantity}"})
        results.append(result)
        environments.append(read_json(directory / "environment.json"))
        inputs.append({"file": str(path.relative_to(output)), "sha256": sha256(path)})
    synthetic = {d["synthetic"] for d in plan["datasets"]}
    if len(synthetic) != 1:
        raise RunnerError("synthetic and benchmark parents cannot share one record set")
    provenance = {
        "experiment": "Equal-budget 2^3 component factorials (Supplementary Tables S16, S16b)",
        "produced_by": "scripts/run_experiment.py factorial plan/run/export (cm_experiments/factorial.py)",
        "hardware": hardware or provenance_hardware(environments), "period": period or provenance_period(results),
        "protocol": (f"eight cells of the confirmed feedback route on the {plan['role']} role with the parent's fitted "
                     "controllers and frozen members; components switched as in factor_switches; Eq. (6) "
                     "log-depth-integrated CE per unit from exact replays of each cell's action/anchor trace; central 95% "
                     "paired whole-unit bootstrap; official metric of the ensemble output; activity = non-A0 share; "
                     "relative time = synchronized wall time over cell 000"),
        "synthetic": synthetic.pop(), "role": plan["role"], "factor_switches": plan["factor_switches"],
        "cell_definition_status": plan["cell_definition_status"], "plan_sha256": sha256(output / "plan.json")}
    return write_record_set("factorials", rows, files, provenance, root=root, replace=replace,
                            allow_partial=allow_partial, command=command, inputs=inputs)
