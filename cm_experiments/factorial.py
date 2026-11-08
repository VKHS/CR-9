"""Equal-budget 2^3 component factorials (Supplementary Tables S16, S16b; record set ``factorials``).

    ogbn-arxiv   (S16)   actions (A), observer (O), risk-controlled execution (X)
    PascalVOC-SP (S16b)  spectral information (S), conditioned observer (O), risk-controlled
                         execution (X), in the receding-horizon regime (MPC-RH), the action bank held fixed

Every cell executes the confirmed feedback route q_M of a sealed parent run with the parent's frozen plant
members.  The manuscript names the components and describes cell 010 ("the nominal schedule under a read-only
observer; the activity and prediction change of that cell come from the changed stopping depth under the
risk-controlled execution domain, which the execution record of the cell identifies"); the executable definitions
below are declared (``cell_definition``), bit 1 = component present:

A  actions            1: the transport actions A1-A7 are admissible in feedback decisions; 0: they are not (the
                      decisions choose among A0 and, under X = 1, the resets A8/A9).  PascalVOC-SP: fixed at 1.
O  observer           1: the runtime's conditioned observer, read by the stopping rule below (a read-only observer:
                      it never changes an action); 0: the memoryless observer (the decision state re-initialised from
                      every measurement) and no stopping.
X  risk-controlled    1: the registered support masks, the resets A8/A9 admissible in feedback decisions and MPC-RH
   execution          planning, the registered recovery (A0 if supported, else A9 when a last-safe anchor exists,
                      else A8) and stops only inside the registered support domain; 0: every (stratum, action) cell
                      qualified, A8/A9 never admissible, every recovery fallback A0, stops in any stratum.
S  spectral           1: the Full sensor profile; 0: the registered Lean counterpart of the same configuration
   information        (spectral and reachability coordinates zeroed), fitted in the same lifecycle.

Stopping rule of the observer cells (O = 1).  At every controlled depth d = 1, ..., L-1, after the runtime's observer
update with the measurement y_d (the state the cell's decisions read), the state is eligible when it is finite and
valid (NIS within the fitted threshold) and, under X = 1, A0 is a qualified action of the measurement's support
stratum (the risk-controlled execution domain).  At an eligible state the rule evaluates T_d, the cell's fitted
terminal-value model: the predicted remaining nominal CE change from d to L (the runtime's ``terminal_delta``
target).  Propagation stops at the first eligible d with T_d > 1e-7 nats (the runtime's nominal-relative identity
tolerance): no later site is executed and the cell's prediction is the head at depth d.  The online execution ends
there (the rule interrupts the runtime's own call: ``ReadOnlyStop``); the stopped execution's output and trace are
the runtime executing the same rule to depth d (``dict(candidate, depth=d)``: the identical first d decisions,
checked action by action and anchor by anchor against the online pass).  Every (input, member) execution's stopping
depth (L without a stop), the terminal value at the stop and the evaluated depths form the cell's execution record
(``executions.npz``; summarized per cell in factorial.json).

The eight cells of each design are distinct by construction: A changes the admissible transport actions, O the
observer and the stopping rule, X the support, the resets, the recovery and the stopping domain, S the sensor
profile; planning refuses a switch table under which two cells coincide (``check_distinct``).  Cell 000 of S16
executes the nominal trajectory (A0 at every site, no stop: the read-only observer on the nominal trajectory
reproduces it) and cell 010 the nominal trajectory under the read-only observer, which changes only the stopping
depth.  A measurement in which two cells nevertheless give identical outputs (e.g. an observer cell whose rule stopped
no execution) names them in factorial.json (``coinciding_cells``).

Two modes declare where each cell's controller comes from:

``fitted`` (the paper's equal-budget design)  every cell's controller is FITTED separately with the cell's
          components: one variant lifecycle per distinct set of cell operations (cm_experiments/ablation_lifecycle.py:
          explore -> fit (-> select -> confirm) re-executed from byte copies of the parent's prepare/train outputs,
          with the same exploration size, folds and epochs, i.e. an equal fitting budget), executed here with the
          cell's variant installed.  Operations (``cell_operations``): O = 0 ``observer: memoryless`` (critic and
          consequence models fitted on its features), A = 0 ``action_bank`` (A0, A8, A9), X = 0 ``risk_control: off``
          (unrestricted support sealed at fit time; no resets; nominal recovery); S = 0 is the registered Lean
          profile of the same lifecycle.  Plan it with ``scripts/run_ablation.py factorial-fit plan``.
``switch`` (a cheaper reconstruction)  the parent's fitted controllers -- one fitting shared by all cells -- with the
          cell's decision operations installed for its executions (the same process-local ``action_bank`` and
          ``risk_control`` patches) and the memoryless observer as a run-time wrapper of the conditioned fit.  It
          cannot reproduce cells that differ only in fitting (a conditioned critic driven by a memoryless state).

    switch        factor   (component off, on)
    action_bank   A        "no_transport" | "registered"
    observer      O        "memoryless"   | "conditioned"       (conditioned: with the stopping rule)
    execution     X        "unrestricted" | "risk_controlled"
    profile       S        "lean"         | "full"

Cell protocol (the six quantities of the record set):
    ld_integrated_loss       log-depth-integrated CE of main Eq. (6) on the registered depth grid
                             {1, 2, 4, ..., L} (cross_morse.training.audit_training's grid): the loss of the
                             prediction head after the first d executed actions (exact replay of the cell's own action
                             and anchor trace with cross_morse.policies.execute_policy; a stopped execution's output
                             holds at every grid depth from its stop on), integrated per unit by the trapezoid rule in
                             log2 depth, then unit-weighted
    interval_low/high        central 95% paired whole-unit bootstrap of that mean (the registered
                             cross_morse.statistics.paired_bootstrap); one resampling stream shared by the
                             eight cells, so the joint draws give every contrast's interval (raw result)
    official_metric          official metric of the cell's ensemble output, in percent / F1 points
    activity                 share of executed actions other than A0, over executed sites, members and inputs
    relative_time            wall time of the cell's online executions (device-synchronized around the runtime's
                             call; a stopped execution is timed up to its stop) over cell 000's
"""
from __future__ import annotations

import contextlib
import json
import math
import time
from pathlib import Path

import numpy as np

from .artifacts import canonical, sha256, write_json
from .ensemble_grid import (FEEDBACK_ROUTES, ParentContext, RunnerError, SealedRun, candidate_identity, check_identity,
                            check_role, code_files, create_plan, ensemble_probabilities, environment_record,
                            job_directory, json_safe, load_plan, member_runs, provenance_hardware, provenance_period,
                            public_plan, read_json, runtime_source_sha256, save_array, save_arrays, selected_jobs,
                            slug, utc_now, write_record_set)

CELLS = ("000", "100", "010", "001", "110", "101", "011", "111")
MODES = ("switch", "fitted")
ABLATION_FILES = ("cm_experiments/ablation_lifecycle.py", "scripts/run_ablation.py")
DESIGNS = {"ogbn-arxiv": {"table": "S16", "factors": ("A", "O", "X"), "route": None},
           "PascalVOC-SP": {"table": "S16b", "factors": ("S", "O", "X"), "route": "mpc_rh"}}
FACTOR_SWITCHES = {"A": ("action_bank", ("no_transport", "registered")),
                   "S": ("profile", ("lean", "full")),
                   "O": ("observer", ("memoryless", "conditioned")),
                   "X": ("execution", ("unrestricted", "risk_controlled"))}
SWITCH_VALUES = {"action_bank": ("registered", "no_transport"), "profile": ("full", "lean"),
                 "observer": ("conditioned", "memoryless"), "execution": ("risk_controlled", "unrestricted")}
TRANSPORT_ACTIONS, RESET_ACTIONS = tuple(range(1, 8)), (8, 9)
STOP_THRESHOLD = 1e-7       # nats: the runtime's nominal-relative identity tolerance (cross_morse.learning.fit_controller)
STOP_FIRST_DEPTH = 1        # depth zero is excluded, as in the runtime's learned-stop family
STOP_RULE = ("read-only observer stop (O = 1 cells): at every controlled depth d = 1..L-1, after the observer update "
             "with y_d, an eligible state (finite, valid NIS; under X = 1 also A0 qualified in the measurement's "
             "support stratum) whose terminal-value model predicts a remaining nominal CE change above 1e-7 nats stops "
             "propagation; the prediction is the head at d and no later site is executed")
MEASUREMENT_SCOPE = {
    "wall_seconds": ("device-synchronized online call; for a stopped observer cell the timer ends at the stop "
                     "exception before the replay that produces final probabilities, so this is a partial "
                     "execution latency rather than complete inference latency"),
    "deployed_work": ("runtime product count of execution to the stopping depth; for stopped cells this "
                      "excludes stop-rule evaluations and the stopping-site measurement"),
    "relative_time": "the above wall_seconds divided by cell 000 wall_seconds; retains the stopped-cell scope",
}
METRIC_UNITS = {"accuracy": "%", "macro_f1": "F1 points", "rocauc": "AUC points", "ap": "AP points"}
EXECUTION_FIELDS = ("input", "member", "terminal_depth", "stopped", "stop_value", "evaluated_depths", "wall_seconds",
                    "actions", "active")


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
    switches = {"profile": base_profile, "observer": "conditioned", "execution": "risk_controlled",
                "action_bank": "registered"}
    used = set()
    for factor, bit in zip(factors, cell):
        name, values = table[factor]
        if name in used:
            raise RunnerError(f"two factors switch {name!r}; every factor needs a component of its own")
        used.add(name)
        switches[name] = values[int(bit)]
    for name, value in switches.items():
        if value not in SWITCH_VALUES[name]:
            raise RunnerError(f"switch {name}={value!r} is not executable; choose one of {SWITCH_VALUES[name]}")
    return switches


def cell_definition(switches: dict) -> dict:
    """What a cell's execution admits and stops on (the declared executable definition of its components)."""
    controlled = switches["execution"] == "risk_controlled"
    actions = ([0] + (list(TRANSPORT_ACTIONS) if switches["action_bank"] == "registered" else [])
               + (list(RESET_ACTIONS) if controlled else []))
    stopping = None
    if switches["observer"] == "conditioned":
        stopping = {"rule": "read-only observer", "model": "terminal-value model of the cell's fitted controller",
                    "threshold_nats": STOP_THRESHOLD, "first_depth": STOP_FIRST_DEPTH,
                    "domain": "registered" if controlled else "unrestricted"}
    return {"profile": switches["profile"], "observer": switches["observer"], "admissible_actions": actions,
            "support": "registered" if controlled else "unrestricted",
            "recovery": "registered" if controlled else "nominal", "stopping": stopping}


def cell_operations(switches: dict) -> list[dict]:
    """The cell's operations of cm_experiments.ablation_lifecycle (fitted mode: fitted and installed; switch mode: the
    decision operations installed and the memoryless observer as a wrapper)."""
    from .ablation_lifecycle import NO_TRANSPORT
    ops = []
    if switches["observer"] == "memoryless":
        ops.append({"op": "observer", "kind": "memoryless"})
    if switches["action_bank"] == "no_transport":
        ops.append({"op": "action_bank", "allowed": list(NO_TRANSPORT)})
    if switches["execution"] == "unrestricted":
        ops.append({"op": "risk_control", "mode": "off"})
    return ops


def check_distinct(cells: list[dict]) -> None:
    """Every cell of a factorial differs from every other by construction (its executable definition)."""
    seen = {}
    for cell in cells:
        key = canonical(cell["definition"])
        if key in seen:
            raise RunnerError(f"cells {seen[key]} and {cell['cell']} have the same executable definition; the "
                              "factorial needs eight distinct cells")
        seen[key] = cell["cell"]


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
                   switches=None, mode="switch", fitted=None) -> Path:
    """``mode="fitted"`` takes every cell's controller from the variant lifecycles of the ablation plan
    ``fitted`` (``scripts/run_ablation.py factorial-fit plan`` writes both plans)."""
    if not parents:
        raise RunnerError("give the sealed ogbn-arxiv and/or PascalVOC-SP parent run(s)")
    if mode not in MODES or (mode == "fitted") != (fitted is not None):
        raise RunnerError("mode is 'switch', or 'fitted' with the ablation plan of the separately fitted cells")
    if mode == "fitted" and switches:
        raise RunnerError("a fitted factorial takes its components from the cell lifecycles, not a switch table")
    from .ablation_lifecycle import variant_spec
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
        fitted_cells = None
        if mode == "fitted":
            from .ablation_lifecycle import factorial_cells
            fitted_cells = factorial_cells(fitted, run.root.name)
        for cell in CELLS:
            values = cell_switches(cell, design["factors"], base.get("profile", "full"), table)
            candidate = profile_counterpart(candidates, base, values["profile"])
            item = {"cell": cell, "factors": dict(zip(design["factors"], map(int, cell))), "switches": values,
                    "definition": cell_definition(values), "operations": cell_operations(values),
                    "candidate": candidate}
            if fitted_cells is not None:
                if cell not in fitted_cells or fitted_cells[cell]["operations"] != item["operations"]:
                    raise RunnerError(f"{run.root.name}: the fitted plan holds no lifecycle with cell {cell}'s components")
                item["fitted"] = fitted_cells[cell]
            else:
                decision = [op for op in item["operations"] if op["op"] != "observer"]
                item["switch_variant"] = (variant_spec(f"factorial-switch-{cell}", decision, study="factorial")
                                          if decision else None)
            cells.append(item)
        check_distinct(cells)
        seeds = [int(s) for s in run.config["training"]["seeds"][:int(base["ensemble"])]]
        jobs.append({"job": len(jobs), "dataset": run.dataset, "slug": slug(run.dataset), "member": run.root.name,
                     "run": str(run.root), "identity": run.identity(), "table": design["table"],
                     "factors": list(design["factors"]), "base": candidate_identity(base), "base_origin": origin,
                     "cells": cells, "member_seeds": seeds, "depth_grid": depth_grid(run.config["model"]["depth"]),
                     "bootstrap": {"reps": int(reps or run.config["admission"]["bootstrap_reps"]),
                                   "seed": int(run.config["seed"]) + 17}})
        datasets.append({"dataset": run.dataset, "slug": slug(run.dataset), "synthetic": run.synthetic})
    fitted_plan = None
    if mode == "fitted":
        fitted_plan = {"path": str(Path(fitted).resolve()), "plan_sha256": sha256(Path(fitted) / "plan.json")}
    return create_plan(output, {
        "experiment": "Equal-budget 2^3 component factorials", "record_set": "factorials", "role": role,
        "mode": mode, "fitted_plan": fitted_plan,
        "warmup": int(warmup), "max_graphs": int(max_graphs), "datasets": datasets, "jobs": jobs,
        "factor_switches": {k: [v[0], list(v[1])] for k, v in table.items()},
        "stopping_rule": {"description": STOP_RULE, "threshold_nats": STOP_THRESHOLD, "first_depth": STOP_FIRST_DEPTH},
        "cell_definitions": "eight distinct executable definitions per design (checked at planning)",
        "cell_definition_status": (
            "declared reconstruction: A = transport actions A1-A7 admissible; O = conditioned observer with the "
            "read-only stopping rule (memoryless observer, no stop, when off); X = registered support, resets A8/A9 "
            "and registered recovery, stops in the registered support domain (unrestricted support, no resets, "
            "nominal recovery when off); S = Full/Lean profile; confirm the definitions before execution"
            + ("" if mode == "fitted" else "; switch mode: the parent's fitted controllers, decision operations "
                                           "installed at execution, memoryless observer as a run-time wrapper")),
        "fitting_budget": ("the parent's fitted controllers and frozen members, shared by all eight cells"
                           if mode == "switch" else
                           "one separately fitted controller per cell variant (same exploration size, folds and "
                           "epochs: equal budget); the parent's frozen members"),
        "protocol": ("eight cells of the confirmed feedback route with components "
                     + ("switched" if mode == "switch" else "fitted separately") + "; observer cells stopped by the "
                     "read-only stopping rule; per-depth head losses by exact replay of each cell's action and anchor "
                     "trace; Eq. (6) log-depth integration per unit; one paired bootstrap stream shared by the cells")})


# ----------------------------------------------------------------------------- statistics (torch-free)
def joint_bootstrap(unit_values: dict, weights, *, reps: int, seed: int) -> np.ndarray:
    """Weighted unit means of every cell under one resampling stream ([reps, cells]).

    The draws are those of cross_morse.statistics.paired_bootstrap with the same seed, so each
    column's quantiles are that function's marginal interval.
    """
    columns = [np.asarray(unit_values[c], dtype=np.float64) for c in unit_values]
    n = len(columns[0])
    w = np.ones(n, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty((reps, len(columns)), dtype=np.float64)
    for start in range(0, reps, 1024):
        ids = rng.integers(0, n, size=(min(1024, reps - start), n))
        bw = w[ids]
        for j, d in enumerate(columns):     # the same arithmetic as paired_bootstrap, column by column
            draws[start:start + len(ids), j] = (bw * d[ids]).sum(axis=1) / bw.sum(axis=1)
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


def execution_arrays(cell: str, rows: list[dict]) -> dict:
    """The cell's execution record as arrays (one entry per (input, member) execution)."""
    out = {}
    for field in EXECUTION_FIELDS:
        values = [row.get(field) for row in rows]
        if field in ("stop_value", "wall_seconds"):
            out[f"cell{cell}_{field}"] = np.asarray([np.nan if v is None else float(v) for v in values], dtype=np.float64)
        else:
            out[f"cell{cell}_{field}"] = np.asarray([0 if v is None else int(v) for v in values], dtype=np.int64)
    return out


def stopping_summary(definition, rows: list[dict], control_depth: int) -> dict:
    """Stopping depths of a cell's executions: every execution's terminal depth (L without a stop) and the stops."""
    terminal = np.asarray([row["terminal_depth"] for row in rows], dtype=np.float64)
    stopped = np.asarray([bool(row.get("stopped")) for row in rows], dtype=bool)

    def quantiles(values):
        if not len(values):
            return None
        return {name: float(np.quantile(values, q)) for name, q in
                (("min", 0.0), ("q25", 0.25), ("median", 0.5), ("q75", 0.75), ("max", 1.0))}
    rule = (definition or {}).get("stopping")
    return {"rule": None if rule is None else STOP_RULE, "domain": None if rule is None else rule["domain"],
            "control_depth": int(control_depth), "executions": int(len(rows)), "stopped": int(stopped.sum()),
            "stopped_fraction": float(stopped.mean()) if len(rows) else None,
            "terminal_depth": quantiles(terminal), "stopping_depth": quantiles(terminal[stopped]),
            "file": "executions.npz"}


# ----------------------------------------------------------------------------- the read-only stopping observer (torch)
class StopPropagation(Exception):
    """Raised by the read-only stopping observer inside the runtime's execution loop: a stop, never a failure."""

    def __init__(self, depth, actions, anchors, value):
        super().__init__(f"read-only observer stop at depth {depth}")
        self.depth, self.value = int(depth), float(value)
        self.actions, self.anchors = [int(a) for a in actions], [int(s) for s in anchors]


class ReadOnlyStop:
    """The cell's controller with its observer read by the stopping rule (no decision changes).

    The runtime calls ``start`` with the depth-0 measurement and ``observe`` once per controlled depth d >= 1; after
    the controller's own update the rule checks the state at d (``StopPropagation`` ends the online execution before
    the decision at d).  It keeps the executed actions (``previous_action`` of every update) and the last-safe anchors
    of the runtime's rule (cross_morse.recovery.label_free_safe), so the first d actions and anchors replay exactly.
    """

    def __init__(self, controller, rule: dict, control_depth: int):
        self._controller = controller
        self._stop_rule, self._stop_depth = dict(rule), int(control_depth)
        self._stop_actions, self._stop_anchors, self._stop_values = [], [0], []
        self._stop_checked = 0

    def __getattr__(self, name):
        return getattr(self._controller, name)

    def start(self, y, previous_action=-1):
        self._stop_actions, self._stop_anchors, self._stop_values, self._stop_checked = [], [0], [], 0
        return self._controller.start(y, previous_action)

    def _eligible(self, state, y, depth) -> bool:
        if not (bool(np.isfinite(np.asarray(state.z, dtype=np.float64)).all()) and bool(state.valid)):
            return False
        if self._stop_rule["domain"] == "registered":
            return 0 in self._controller.supported(y, depth)
        return True

    def observe(self, state, y, depth, previous_action):
        from cross_morse.recovery import label_free_safe
        result = self._controller.observe(state, y, depth, previous_action)
        depth = int(depth)
        self._stop_actions.append(int(previous_action))
        if int(self._stop_rule["first_depth"]) <= depth < self._stop_depth:
            self._stop_checked += 1
            if self._eligible(result, y, depth):
                value = float(self._controller.terminal_value(result))
                self._stop_values.append((depth, value))
                if math.isfinite(value) and value > float(self._stop_rule["threshold_nats"]):
                    raise StopPropagation(depth, self._stop_actions, self._stop_anchors, value)
        self._stop_anchors.append(depth if label_free_safe(y) else self._stop_anchors[-1])
        return result

    def summary(self, stop=None) -> dict:
        values = [v for _, v in self._stop_values if math.isfinite(v)]
        return {"stopping_depth": None if stop is None else stop.depth,
                "value_at_stop": None if stop is None else stop.value,
                "checked_depths": int(self._stop_checked), "evaluated_depths": len(self._stop_values),
                "largest_value": max(values) if values else None}


def stop_rule(cell: dict, controller):
    """The cell's stopping rule (None for O = 0 cells); it needs the cell's fitted terminal-value model."""
    stopping = (cell.get("definition") or {}).get("stopping")
    if stopping is None:
        return None
    if getattr(controller, "terminal_available", True) is False:
        raise RunnerError(f"cell {cell['cell']}: the read-only observer's stopping rule needs the cell's fitted "
                          "terminal-value model, and the controller has none (no finite terminal_delta targets)")
    return stopping


# ----------------------------------------------------------------------------- controllers (torch)
class MemorylessObserver:
    """Switch mode O = 0: each measurement re-initialises the decision state (no conditioned filtering or dynamics)."""

    def __init__(self, controller):
        self._controller = controller

    def __getattr__(self, name):
        return getattr(self._controller, name)

    def observe(self, state, y, depth, previous_action):
        base = self._controller
        base.counters["observer_updates"] += 1
        base.counters["approx_dense_multiply_adds"] += len(y) * base.observer.dimension
        return base.observer.start(y, previous_action, depth)


def cell_controller(ctx: ParentContext, candidate: dict, switches: dict, cache: dict):
    """Switch mode: the parent's fitted controller of the cell's profile (memoryless wrapper when O = 0); the cell's
    decision operations are installed around its executions (``cell_scope``)."""
    profile = candidate.get("profile", "full")
    if profile not in cache:
        cache[profile] = ctx.controller_for(candidate)
    controller = cache[profile]
    if switches["observer"] == "memoryless":
        controller = MemorylessObserver(controller)
    return controller


def cell_scopes(plan: dict, entry: dict, run: SealedRun) -> dict:
    """{cell: (run root, installed variant)}: the cell's lifecycle variant (fitted mode, each lifecycle verified) or
    its decision operations on the parent (switch mode; None when the cell has none)."""
    if plan.get("mode", "switch") != "fitted":
        return {c["cell"]: (run.root, c.get("switch_variant")) for c in entry["cells"]}
    from .ablation_lifecycle import open_fitted_cell
    if sha256(Path(plan["fitted_plan"]["path"]) / "plan.json") != plan["fitted_plan"]["plan_sha256"]:
        raise RunnerError("the ablation plan of the fitted cells changed after planning")
    return {c["cell"]: open_fitted_cell(c["fitted"], run) for c in entry["cells"]}


def cell_scope(scopes, cell: str):
    """The cell's variant installed into the runtime for the duration of a block."""
    root, variant = (scopes or {}).get(cell, (None, None))
    if variant is None:
        return contextlib.nullcontext()
    from .ablation_lifecycle import installed
    return installed(variant, run_dir=root)


def fitted_cell_controller(ctx: ParentContext, cell: dict, root):
    """The controller fitted by the cell's own lifecycle (the parent's for a cell planned on the parent)."""
    from cross_morse.learning import Controller
    candidate = cell["candidate"]
    if cell["fitted"]["source"] == "parent":
        return ctx.controller_for(candidate)
    profile = candidate.get("profile", "full")
    controller = Controller.load(Path(root) / "fit" / ("full" if profile == "none" else profile), "cpu")
    if candidate["route"].startswith("mpc") and not controller.sequence_models_available:
        raise RunnerError("MPC configuration requires fitted transition/stage/terminal models")
    return controller


def _public_fitted(fitted):
    if fitted is None:
        return None
    return {k: (Path(v).name if k == "run" else v) for k, v in fitted.items() if k != "variant"} | (
        {"variant_id": fitted["variant"]["id"], "operations": fitted["variant"]["operations"]}
        if fitted.get("variant") else {})


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


def _execute(model, x, graph, config, candidate, controller, reference, depths, rows, stopping=None):
    """One member execution of a cell and the head probabilities (``rows`` only) after the first d executed actions.

    The online execution is timed around the runtime's call with device synchronization.  In an observer cell it
    carries the read-only stopping observer; a stop at depth d* ends it, and the stopped execution is the runtime
    executing the same rule to depth d* (identical first d* decisions, checked against the online pass). The
    replay producing final probabilities is outside the timer; MEASUREMENT_SCOPE records this limitation.
    """
    from cross_morse.common import synchronize
    from cross_morse.policies import execute_policy
    watcher = None if stopping is None else ReadOnlyStop(controller, stopping, int(config["model"]["depth"]))
    stop = None
    synchronize(x.device)
    begin = time.perf_counter()
    try:
        final, trace = execute_policy(model, x, graph, config, candidate,
                                      controller if watcher is None else watcher, reference)
    except StopPropagation as signal:
        stop = signal
    synchronize(x.device)
    wall = time.perf_counter() - begin
    if stop is not None:
        final, trace = execute_policy(model, x, graph, config, dict(candidate, depth=stop.depth), controller, reference)
    actions, anchors = [int(a) for a in trace["actions"]], [int(s) for s in trace["safe_depths"]]
    if stop is not None and (actions != stop.actions or anchors != stop.anchors):
        raise RunnerError("the execution to the stopping depth did not reproduce the online decisions; the stopped "
                          "output is not identified")
    by_depth = []
    for d in depths:
        if d >= len(actions):
            by_depth.append(np.asarray(final[rows]))  # a stopped or complete execution's output holds at later depths
        else:
            replay, _ = execute_policy(model, x, graph, config, candidate, None, reference,
                                       replay_actions=actions[:d], replay_safe_depths=anchors[:d])
            by_depth.append(np.asarray(replay[rows]))
    record = {"actions": actions, "wall_seconds": wall,
              "runtime_wall_seconds": None if stop is not None else float(trace["wall_seconds"]),
              "deployed_work": float(trace.get("deployed_multiply_add_work", trace["multiply_add_work"])),
              "work_scope": ("the runtime's execution to the stopping depth (stop-rule evaluations and the stopping "
                             "site's measurement excluded)" if stop is not None else "the online execution"),
              "feasible": bool(trace.get("feasible", True)), "terminal_depth": int(trace["terminal_depth"]),
              "stopped": stop is not None, "stop": None if watcher is None else watcher.summary(stop)}
    return by_depth, record


def _ledger() -> dict:
    return {"wall_seconds": 0.0, "deployed_work": 0.0, "actions": 0, "active": 0, "inputs": 0, "feasible": True,
            "terminal_depths": [], "executions": [], "action_counts": [0] * 10}


def _book(ledger: dict, record: dict, input_id, seed) -> None:
    """One member execution into the cell's ledger and execution record."""
    actions = [int(a) for a in record["actions"]]
    active = sum(a != 0 for a in actions)
    stop = record.get("stop") or {}
    ledger["wall_seconds"] += float(record["wall_seconds"])
    ledger["deployed_work"] += float(record["deployed_work"])
    ledger["actions"] += len(actions)
    ledger["active"] += active
    for action in actions:
        ledger["action_counts"][action] += 1
    ledger["feasible"] &= bool(record["feasible"])
    ledger["terminal_depths"].append(int(record["terminal_depth"]))
    ledger["executions"].append({"input": int(input_id), "member": int(seed),
                                 "terminal_depth": int(record["terminal_depth"]), "stopped": bool(record.get("stopped")),
                                 "stop_value": stop.get("value_at_stop"),
                                 "evaluated_depths": int(stop.get("evaluated_depths") or 0),
                                 "wall_seconds": float(record["wall_seconds"]), "actions": len(actions),
                                 "active": int(active)})


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
    scopes = cell_scopes(plan, entry, run)
    fitted = plan.get("mode", "switch") == "fitted"
    ctx = ParentContext(run, device, through="evaluate" if role == "test" else "confirm")
    config, spec, depths = ctx.config, task_spec(ctx.config), entry["depth_grid"]
    models = {seed: ctx.plant(seed) for seed in entry["member_seeds"]}
    cache = {}
    cells = []
    for c in entry["cells"]:
        with cell_scope(scopes, c["cell"]):
            controller = (fitted_cell_controller(ctx, c, scopes[c["cell"]][0]) if fitted
                          else cell_controller(ctx, c["candidate"], c["switches"], cache))
        cells.append((c, c["candidate"], controller, stop_rule(c, controller)))
    labels, units = ctx.labels(role), ctx.units(role)
    unit_ids = np.unique(units)
    unit_index = np.searchsorted(unit_ids, units)
    sums = {c["cell"]: np.zeros((len(depths), len(unit_ids))) for c, *_ in cells}
    counts = np.zeros(len(unit_ids))
    terminal = {c["cell"]: None for c, *_ in cells}
    ledger = {c["cell"]: _ledger() for c, *_ in cells}
    first_seed = entry["member_seeds"][0]
    for count, (input_id, x, graph, rows, positions) in enumerate(ctx.episodes(role)):
        if plan["max_graphs"] and count >= plan["max_graphs"]:
            break
        if count == 0:
            for _ in range(plan["warmup"]):          # discarded: lazy initialisation outside the timed cells
                for cell, candidate, controller, stopping in cells:
                    with cell_scope(scopes, cell["cell"]):
                        _execute(models[first_seed], x, graph, config, candidate, controller, ctx.reference,
                                 depths[-1:], rows, stopping=stopping)
        y, index = labels[positions], unit_index[positions]
        valid = None
        for cell, candidate, controller, stopping in cells:
            name = cell["cell"]
            per_member = []
            for seed in entry["member_seeds"]:
                with cell_scope(scopes, name):
                    by_depth, record = _execute(models[seed], x, graph, config, candidate, controller, ctx.reference,
                                                depths, rows, stopping=stopping)
                per_member.append(by_depth)
                _book(ledger[name], record, input_id, seed)
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
    ``terminal[cell]`` the ensemble output on the role rows and ``ledger[cell]`` the execution totals with the
    per-execution records (``executions``).
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
    control_depth = int(config["model"]["depth"])
    executions = {}
    for name in CELLS:
        executions.update(execution_arrays(name, ledger[name].get("executions") or []))
    execution_file = save_arrays(directory / "executions.npz", replace=True, **executions)
    rows = []
    baseline = ledger["000"]["wall_seconds"]
    for cell in entry["cells"]:
        name = cell["cell"]
        rows_mask = np.isfinite(terminal[name]).all(axis=1)
        if not truncated and not rows_mask.all():
            raise RunnerError(f"cell {name}: role rows without a prediction")
        prediction_sha = save_array(directory / "predictions" / f"cell{name}.npy", terminal[name], replace=True)
        metric = metric_score(terminal[name][rows_mask], labels[rows_mask], spec)
        item = ledger[name]
        rows.append({"cell": name, "factors": cell["factors"], "switches": cell["switches"],
                     "definition": cell.get("definition"), "fitted": _public_fitted(cell.get("fitted")),
                     "candidate": candidate_identity(cell["candidate"]), **statistics[name],
                     "official_metric": 100.0 * metric, "metric_name": spec.metric,
                     "activity": item["active"] / item["actions"] if item["actions"] else 0.0,
                     "action_counts": item.get("action_counts"),
                     "relative_time": item["wall_seconds"] / baseline, "wall_seconds": item["wall_seconds"],
                     "deployed_work": item["deployed_work"], "inputs": item["inputs"], "feasible": item["feasible"],
                     "integration_domain": {"depth_grid": depths, "rule": "Eq. (6) trapezoid in log2 depth",
                                            "terminal_depths": sorted(set(item["terminal_depths"])),
                                            "after_stop": "the stopped execution's output at every later grid depth"},
                     "stopping": stopping_summary(cell.get("definition"), item.get("executions") or [], control_depth),
                     "terminal_predictions": {"file": f"predictions/cell{name}.npy", "sha256": prediction_sha}})
    coinciding = [[a, b] for i, a in enumerate(CELLS) for b in CELLS[i + 1:]
                  if np.array_equal(arrays[f"cell{a}_unit_depth_loss"], arrays[f"cell{b}_unit_depth_loss"])
                  and np.array_equal(terminal[a], terminal[b], equal_nan=True)]
    unit_file = save_arrays(directory / "unit_dil.npz", replace=True, **arrays)   # factorial.json: marker
    write_json(directory / "factorial.json", json_safe({
        "schema": "cm-factorial-2", "dataset": entry["dataset"], "table": entry["table"], "member": entry["member"],
        "factors": entry["factors"], "role": role, "mode": plan.get("mode", "switch"), "cells": rows,
        "contrasts": contrasts, "coinciding_cells": coinciding,
        "coinciding_note": ("pairs of cells with identical per-unit depth losses and terminal predictions in this "
                            "measurement; their executable definitions differ (planning checks it), e.g. an observer "
                            "cell whose stopping rule stopped no execution"),
        "stopping_rule": plan.get("stopping_rule"),
        "measurement_scope": MEASUREMENT_SCOPE,
        "member_seeds": entry["member_seeds"], "base": entry["base"], "base_origin": entry["base_origin"],
        "factor_switches": plan["factor_switches"], "truncated": truncated, "max_graphs": plan["max_graphs"],
        "warmup": plan["warmup"], "bootstrap": entry["bootstrap"],
        "unit_dil": {"file": "unit_dil.npz", "sha256": unit_file},
        "executions": {"file": "executions.npz", "sha256": execution_file, "fields": list(EXECUTION_FIELDS),
                       "definition": "one entry per (input, member) execution of every cell: terminal (stopping) "
                                     "depth, L without a stop; the terminal value at the stop (NaN without one); "
                                     "eligible depths at which the rule evaluated the terminal-value model; online "
                                     "wall seconds; executed and non-A0 actions"},
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
    definitions = {}
    for entry in plan["jobs"]:
        directory = job_directory(output, entry)
        path = directory / "factorial.json"
        if not path.is_file():
            raise RunnerError(f"{entry['dataset']}: job {entry['job']} has not completed")
        result = read_json(path)
        if [c["cell"] for c in result["cells"]] != list(CELLS) or result["role"] != plan["role"]:
            raise RunnerError(f"{path}: cells or role differ from the plan")
        identities = result.get("identities") or {}
        if identities.get("plan_sha256") != sha256(output / "plan.json") or identities.get("parent") != entry["identity"]:
            raise RunnerError(f"{path}: the measurement belongs to another plan or parent")
        if result["truncated"] and not allow_partial:
            raise RunnerError(f"{entry['dataset']}: a run truncated by max_graphs is not a complete measurement")
        if not (directory / "executions.npz").is_file():
            raise RunnerError(f"{entry['dataset']}: job {entry['job']} has no execution record (executions.npz)")
        for field, relative in (("unit_dil", "unit_dil.npz"), ("executions", "executions.npz")):
            if (result.get(field, {}).get("file") != relative or not (directory / relative).is_file()
                    or sha256(directory / relative) != result[field].get("sha256")):
                raise RunnerError(f"{entry['dataset']}: {relative} differs from its measured artifact hash")
        base = f"raw/{entry['slug']}"
        files[f"{base}/factorial.json"] = path
        files[f"{base}/unit_dil.npz"] = directory / "unit_dil.npz"
        files[f"{base}/executions.npz"] = directory / "executions.npz"
        files[f"{base}/environment.json"] = directory / "environment.json"
        definitions[entry["dataset"]] = {c["cell"]: c.get("definition") for c in entry["cells"]}
        for i, cell in enumerate(result["cells"]):
            planned = entry["cells"][i]
            if cell["candidate"] != candidate_identity(planned["candidate"]) or cell.get("definition") != planned.get("definition"):
                raise RunnerError(f"{entry['dataset']}: cell {cell['cell']} differs from its planned policy or definition")
            relative = f"predictions/cell{cell['cell']}.npy"
            prediction = cell.get("terminal_predictions") or {}
            if (prediction.get("file") != relative or not (directory / relative).is_file()
                    or sha256(directory / relative) != prediction.get("sha256")):
                raise RunnerError(f"{entry['dataset']}: cell {cell['cell']} prediction is absent or differs from its "
                                  "measured artifact hash")
            files[f"{base}/{relative}"] = directory / relative
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
    mode = plan.get("mode", "switch")
    generated = {"code/plan.json": public_plan(output)}
    if mode == "fitted":             # the cell lifecycles: the ablation plan and each lifecycle's completion record
        from .ensemble_grid import ROOT
        from .ablation_lifecycle import controller_artifacts
        files.update({f"code/{name}": ROOT / name for name in ABLATION_FILES if (ROOT / name).is_file()})
        fitted = Path(plan["fitted_plan"]["path"])
        if sha256(fitted / "plan.json") != plan["fitted_plan"]["plan_sha256"]:
            raise RunnerError("the ablation plan of the fitted cells changed after planning")
        generated["code/ablation_plan.json"] = public_plan(fitted)
        for job in read_json(fitted / "plan.json")["jobs"]:
            record = fitted / "jobs" / job["slug"] / "lifecycle.json"
            if not record.is_file():
                raise RunnerError(f"cell lifecycle {job['slug']} has no completion record (jobs/.../lifecycle.json)")
            files[f"raw/fitted/{job['slug']}/lifecycle.json"] = record
            assets, manifest = controller_artifacts(job, SealedRun(job["parent"]))
            base = f"raw/fitted/{job['slug']}/controller_package"
            files.update({f"{base}/{relative}": path for relative, path in assets.items()})
            generated[f"{base}/CONTROLLERS.json"] = json.dumps(json_safe(manifest), indent=2, sort_keys=True) + "\n"
    else:                            # the installer of the switch mode's decision operations
        from .ensemble_grid import ROOT
        files["code/cm_experiments/ablation_lifecycle.py"] = ROOT / "cm_experiments" / "ablation_lifecycle.py"
    provenance = {
        "experiment": "Equal-budget 2^3 component factorials (Supplementary Tables S16, S16b)",
        "produced_by": ("scripts/run_experiment.py factorial plan/run/export (cm_experiments/factorial.py)"
                        if mode == "switch" else
                        "scripts/run_ablation.py factorial-fit plan/run/export (cm_experiments/ablation_lifecycle.py: "
                        "separately fitted cell lifecycles; cm_experiments/factorial.py: cell execution and export)"),
        "hardware": hardware or provenance_hardware(environments), "period": period or provenance_period(results),
        "protocol": (f"eight cells of the confirmed feedback route on the {plan['role']} role with "
                     + ("the parent's fitted controllers and frozen members; components switched as in factor_switches"
                        if mode == "switch" else
                        "the parent's frozen members and one separately fitted controller per cell variant (equal "
                        "fitting budget), executed with the cell's variant installed")
                     + "; observer cells stopped by the read-only observer's stopping rule (stopping_rule); Eq. (6) "
                     "log-depth-integrated CE per unit from exact replays of each cell's action/anchor trace, a stopped "
                     "output held at later depths; central 95% paired whole-unit bootstrap; official metric of the "
                     "ensemble output; activity = non-A0 share of the executed actions; relative time = synchronized "
                     "online wall time over cell 000; every execution's stopping depth in raw/<dataset>/executions.npz"),
        "synthetic": synthetic.pop(), "role": plan["role"], "mode": mode, "factor_switches": plan["factor_switches"],
        "cell_definitions": definitions, "stopping_rule": plan.get("stopping_rule"),
        "measurement_scope": MEASUREMENT_SCOPE,
        "fitted_plan": None if mode == "switch" else {"plan_sha256": plan["fitted_plan"]["plan_sha256"]},
        "cell_definition_status": plan["cell_definition_status"], "plan_sha256": sha256(output / "plan.json")}
    return write_record_set("factorials", rows, files, provenance, root=root, replace=replace,
                            allow_partial=allow_partial, command=command, inputs=inputs, generated=generated)
