"""Decision-log audits of the deployed rules: action gaps and switching (Supplementary Table S21, Figure S17f;
read again by Tables S33-S34 and Figure S7), executed decision origins and within-depth action entropy (main
Figure 3c-d), and reliability, harm and fallback diagnostics (Supplementary Figure S16; E7, E8).

One instrumented serving-time execution of every sealed lifecycle run (a Roman-empire campaign: every split run)
feeds the three record sets ``action_gap_audit``, ``decision_traces`` and ``reliability_diagnostics``.

Execution
---------
The released runtime executes the deployed rule exactly as at deployment: ``cross_morse.policies.execute_policy``
per member on the evaluation role (one complete graph at a time for the independent-graph benchmarks, the
transductive graph otherwise), unmodified. The only hook is the one the negative-controls runner already uses:
``cross_morse.policies.sensor_bank`` is wrapped for the duration of a call (``negative_controls.runtime_sensor``)
so that a copy of every measurement vector the runtime consumed is kept; the returned tensors are untouched.
From those measurements and the runtime's own decision log everything else is recomputed outside the runtime and
checked against the log site by site (support set, observer validity and NIS, anchor update, one-step/radius
argmin and ``score_gap``, the MPC-RH planned sequence): the observer is replayed with the runtime's
``Controller.start``/``observe``; a mismatch stops the job. Executed actions are compared with the sealed member
traces (transductive runs) or the sealed per-graph action digest (independent-graph runs) and the parity is
recorded.

Registered perturbation (the manuscript names a "state perturbation" without defining it; declared here)
------------------------------------------------------------------------------------------------------
At every executed decision site the measurement y_l that the rule received is replaced by
y~_l = y_l + delta_l, one draw of the deployed observer's frozen measurement-noise model expressed in measurement
units: delta_l = c * diag(s) B L_R xi, xi ~ N(0, I_k), with R = L_R L_R^T the observer's registered measurement
covariance (cross_morse.learning: R = 0.25 Q + 1e-3 I in its latent space), B its frozen PCA basis, s its robust
per-coordinate scale and c the declared factor (1). The transport-history and spectral-age coordinates
(bookkeeping, not sensed) are not perturbed, as in the negative-controls sensor-bias construction. The decision is
recomputed with the runtime's own calls on the perturbed measurement: support set, last-safe anchor rule, observer
update from the realized previous state, one-step/radius scores or the registered beam (``beam_plan``) and the
registered recovery logic. The perturbation is site-local: the realized trajectory is kept, so a switch at one site
does not propagate (main Proposition 3 is a statement about one site). Draws are seeded by
(run seed + declared offset, input, member, depth).

Gaps (main Proposition 3)
-------------------------
* one-step and radius rules: the runtime's ``score_gap``, second-smallest minus smallest finite supported score
  (recomputed and checked against the log);
* MPC-RH: the planner's own root-action scores, J(a) = best final objective (Eq. 17) of a beam sequence starting
  with a, from the registered beam (``ranked_beam`` mirrors ``beam_plan`` and must return the executed plan);
  m_l = J(best other root) - J(executed plan). When pruning removed every other root from the final beam, the
  registered beam is rerun from the same state restricted to the planner's other root candidates (its top-k root
  expansion). A negative value (a better plan with another first action exists outside the executed beam: a
  search inversion) is kept in the raw file and enters the quantiles as 0; inversions are counted;
* MPC-OL deployed rules: the depth-0 planning decision, by the same rule;
* audited sites: executed sites whose origin is ``policy`` with at least two finite candidates; recovery sites and
  single-candidate sites are counted separately. Open-loop deployed rules make no scored decision and have no row.

Decision origins (main Figure 3d): each executed decision on an evaluation unit (an independent graph, or a graph
core holding evaluation rows) is weighted 1/E of its serving ensemble, so that every (unit, depth) carries unit
mass. Served by q_C because of a global return (the deployed rule is the comparator), g_input = 0 or a numerical
failure of the deployed rule: input fallback; otherwise the runtime origin: fixed_schedule/nominal -> schedule or
nominal, policy of one_step/radius -> state conditioned, policy of mpc_rh -> exclusive sequence, recovery_* ->
recovery. g_input is the static gate of ``cm_experiments.input_gate`` (frozen on D_explore; refitted with the
declared settings, which is deterministic, or reused from a sealed input_gate run).

Reliability (Figure S16)
------------------------
* coverage, level 1: executed decisions; realized one-action consequence Delta_l(a_l) = l_{l+h}(a_l -> pi_N) -
  l_{l+h}(pi_N) on the evaluation rows (h = exploration.critic_horizon, the critic's own target) against the
  critic mean with the registered calibrated radius r(stratum, a_l) of the deployed controller;
  levels 4, 8, 16, 32: executed sequences; realized loss change l_{l+h} - l_l along the realized trajectory against
  the frozen consequence models' rollout of the executed actions from the observer state at l (sum of stage-model
  predictions along transition-model states, the MPC planner's model), with an h-step radius calibrated by the
  runtime's own rule (per-trajectory maximum residual, higher (1 - alpha) quantile, (stratum, action) cells with at
  least ``calibration_min_count`` trajectories, action-pooled otherwise) on the D_explore radius-fit partition of
  the sealed calibration split (cross_morse.learning.calibration_partition role 1) with the sealed controller;
  pooled over the audited feedback rules of all datasets; 95% interval: stratified percentile bootstrap over the
  registered statistical units (graph cores of a transductive run, graphs of an independent-graph run, strata =
  runs);
* observer: E8 variants on the D_explore trajectories of each run: raw (latest latent measurement), EMA (declared
  weight), global KF (the conditioned observer's latent space with one global transition and its residual Q,
  R = 0.25 Q + 1e-3 I), conditioned KF (the sealed observer). For every variant the action critic is retrained with
  the runtime's recipe (``cross_morse.learning._train_network``, the final-fit seed, model-fit rows and internal
  validation groups of the sealed fit) and evaluated on the untouched assessment partition (role 2). risk_rmse is
  the held-out RMSE of that one-step risk prediction (the measurement module_ablations calls innovation_rmse);
  nis_coverage the fraction of assessment-row innovations (observer updates) whose NIS lies in the two-sided
  (1 - alpha) chi-square region with the latent dimension as degrees of freedom; pooled over datasets with equal
  weights (sqrt of the mean of the per-dataset mean squared errors; mean of the per-dataset fractions);
* harm, per dataset: executed decisions of the deployed feedback rule with a finite risk prediction; predicted
  safe iff the critic's calibrated upper limit mu + r <= 0 (flagged otherwise); harmful iff Delta_l > tol;
  false-safe = safe and harmful / executed decisions; harm-miss = safe and harmful / harmful; the point-prediction
  variant (mu <= 0) is kept in the raw summary;
* stress, level clean: evaluation units of the declared stress dataset (the negative-controls reference): input
  fallback = share with g_input = 0; residual activity = share served by the confirmed policy q_M (g_input = 1,
  no numerical failure) on which some member executed a non-nominal action (negative_controls' definitions).

Tolerance ``IDENTITY_TOLERANCE`` (1e-7 nats, the runtime's nominal-relative identity tolerance of
``fit_controller``) applies to "harmful" and to the coverage comparisons, so that the exact A0 identity is not
broken by floating-point noise.

Workflow (``scripts/run_decision_audit.py plan|run|export``): plan -> <output>/plan.json; run -> one job per
lifecycle run, <output>/jobs/<dataset>/<run>/ with three resumable phases (calibration.json, observers.json,
execution outputs) and audit.json as completion marker; export -> records/<set> for each of the three sets.
Planning, statistics and export are torch-free.
"""
from __future__ import annotations

import contextlib
import copy
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from .artifacts import sha256
from .ensemble_grid import (ROLE_STAGES, RunnerError, SealedRun, check_identity, check_role, candidate_identity,
                            create_plan, environment_record, job_directory, json_safe, load_plan, member_runs,
                            public_plan, registered_constants, runtime_source_sha256, save_arrays, selected_jobs,
                            slug, utc_now)

ROOT = Path(__file__).resolve().parents[1]
PLAN_KIND = "decision_audit"
SETS = ("action_gap_audit", "decision_traces", "reliability_diagnostics")
AUDIT_FILES = ("cm_experiments/decision_audit.py", "scripts/run_decision_audit.py")
REUSED_FILES = ("cm_experiments/ensemble_grid.py", "cm_experiments/input_gate.py",
                "cm_experiments/negative_controls.py", "cm_experiments/predictability.py",
                "cm_experiments/artifacts.py", "cm_experiments/budget.py")
DATASETS = ("ogbn-arxiv", "ogbn-proteins", "ogbn-products", "Roman-empire", "PascalVOC-SP", "COCO-SP",
            "Peptides-func")
FEEDBACK_ROUTES = ("one_step", "radius", "mpc_rh")
PLANNING_ROUTES = ("mpc_rh", "mpc_ol")
CATEGORIES = ("schedule_nominal", "state_conditioned", "exclusive_sequence", "recovery", "input_fallback")
OBSERVER_LEVELS = ("raw", "EMA", "global KF", "conditioned KF")
MODULE_ARMS = {"raw": "raw", "EMA": "ema", "global KF": "global", "conditioned KF": "conditioned"}
ORIGINS = ("policy", "fixed_schedule", "nominal", "recovery_no_support", "recovery_observer", "recovery_budget",
           "recovery_no_finite_score", "recovery_support_mismatch")
GAP_KINDS = {0: "not audited", 1: "one-step/radius score gap", 2: "beam root gap (final beam)",
             3: "beam root gap (restricted re-plan)"}
IDENTITY_TOLERANCE = 1e-7
FINAL_FIT_SEED = 7919            # cross_morse.learning.fit_controller: seed of the final model fit
HISTORY_COORDINATES = ("relative_depth", "normalized_mass", "normalized_hops", "spectral_age_fraction")
SUMMARY_TAIL = ("quantity", "value", "unit", "source_file", "locator")
JOB_SCHEMA = "cm-decision-audit-job-1"
SUMMARY_SCHEMA = "cm-decision-audit-summary-1"
BLOCK = 500

UNITS = {
    "median_gap_nats": "nats", "p5_gap_nats": "nats", "near_zero_threshold_nats": "nats",
    "near_zero_fraction": "fraction", "switch_fraction": "fraction", "recovery_index_stability": "fraction",
    "beam_sequence_stability": "fraction", "decision_sites": "sites",
    "same_depth_action_entropy": "nats", **{f"fraction_{c}": "fraction" for c in CATEGORIES},
    "coverage": "fraction", "coverage_ci_low": "fraction", "coverage_ci_high": "fraction", "risk_rmse": "nats",
    "nis_coverage": "fraction", "false_safe_rate": "fraction", "false_safe_ci_low": "fraction",
    "false_safe_ci_high": "fraction", "harm_miss_rate": "fraction", "harm_miss_ci_low": "fraction",
    "harm_miss_ci_high": "fraction", "input_fallback": "fraction", "residual_activity": "fraction",
}
DEFINITIONS = {
    "action_gap_audit": {
        "median_gap_nats": "median over the audited sites (executed policy decisions with at least two finite "
                           "candidates) of the gap m_l between the best and second-best supported scores; MPC root "
                           "gaps from the registered beam, search inversions entering as 0 (numpy median)",
        "p5_gap_nats": "5th percentile of the same gaps (numpy linear interpolation)",
        "near_zero_threshold_nats": "declared near-zero threshold (default: the registered selective-sensing "
                                    "certificate floor tau of cm_experiments.budget.Certificate)",
        "near_zero_fraction": "fraction of audited sites whose gap is below the threshold",
        "switch_fraction": "fraction of audited sites whose decision under the registered perturbation (one draw "
                           "of the observer's measurement-noise model, site-local) is another action",
        "recovery_index_stability": "fraction of executions (input x member) of the deployed feedback rule in "
                                    "which no site's last-safe anchor update changes under the perturbation",
        "beam_sequence_stability": "fraction of planning sites (MPC-RH policy sites, MPC-OL depth 0) whose planned "
                                   "sequence is unchanged under the perturbation; 1 for rules that do not plan",
        "decision_sites": "number of audited sites",
    },
    "decision_traces": {
        "same_depth_action_entropy": "entropy (nats) of the executed actions at each controlled depth over the "
                                     "weighted D_test decisions, averaged over depths with their decision weights",
        **{f"fraction_{c}": f"weighted share of D_test decisions of origin {c} (each evaluation unit and depth "
                            "carries unit mass, shared by the members of its serving ensemble)" for c in CATEGORIES},
    },
    "reliability_diagnostics": {
        "coverage": "level 1: share of executed decisions whose realized one-action consequence lies within the "
                    "registered calibrated radius of the critic prediction; level h > 1: share of executed "
                    "sequences whose realized h-step loss change lies within the D_explore-calibrated h-step radius "
                    "of the consequence-model rollout",
        "coverage_ci_low": "2.5% point of the stratified unit bootstrap of the coverage",
        "coverage_ci_high": "97.5% point of the same bootstrap",
        "risk_rmse": "held-out (assessment partition) RMSE of the one-step risk prediction of the critic retrained "
                     "on the observer variant's features; equal-weight pool over datasets",
        "nis_coverage": "share of assessment-row observer updates whose NIS lies in the two-sided (1 - alpha) "
                        "chi-square region (dof = latent dimension); equal-weight pool over datasets",
        "false_safe_rate": "executed decisions predicted safe (mu + r <= 0) whose one-action paired loss increased, "
                           "over executed decisions with a finite risk prediction",
        "false_safe_ci_low": "2.5% point of the stratified unit bootstrap", "false_safe_ci_high": "97.5% point",
        "harm_miss_rate": "executed decisions with a one-action paired loss increase that were predicted safe, over "
                          "harmful executed decisions",
        "harm_miss_ci_low": "2.5% point of the stratified unit bootstrap", "harm_miss_ci_high": "97.5% point",
        "input_fallback": "share of evaluation units of the stress dataset returned to q_C by g_input",
        "residual_activity": "share of evaluation units served by q_M on which some member executed a non-nominal "
                             "action",
    },
}


def code_identity() -> dict:
    return {name: sha256(ROOT / name) for name in AUDIT_FILES + REUSED_FILES if (ROOT / name).is_file()}


def registered_horizons() -> list[int]:
    return [int(h) for h in registered_constants()["observer_eq12_table_S9"]["horizons"]]


def certificate_floor() -> float:
    """Registered acquisition threshold tau: the floor of the selective-sensing certificate (Table S34, Fig. S7)."""
    import dataclasses
    from .budget import Certificate
    return float(next(f.default for f in dataclasses.fields(Certificate) if f.name == "floor"))


def _dump(path, value) -> str:
    """Atomic JSON write that replaces an interrupted attempt; returns the file's SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        Path(temporary).write_text(json.dumps(json_safe(value), indent=1, sort_keys=True, allow_nan=False) + "\n",
                                   encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return sha256(path)


def _read(path):
    path = Path(path)
    if not path.is_file():
        raise RunnerError(f"missing input file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path, rows, fields) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        with open(temporary, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fields})
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return sha256(path)


def _read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# ============================================================================= statistics (torch-free)
def origin_category(origin: str, route: str, *, fallback: bool = False) -> str:
    """The Figure 3d category of one executed decision of the runtime's decision log."""
    if fallback:
        return "input_fallback"
    if origin in ("fixed_schedule", "nominal"):
        return "schedule_nominal"
    if origin.startswith("recovery"):
        return "recovery"
    if origin == "policy" and route in ("one_step", "radius"):
        return "state_conditioned"
    if origin == "policy" and route == "mpc_rh":
        return "exclusive_sequence"
    raise RunnerError(f"decision origin {origin!r} of route {route!r} has no registered category")


class DecisionMass:
    """Weighted executed decisions: totals per origin category and per (depth, action)."""

    def __init__(self):
        self.categories = {c: 0.0 for c in CATEGORIES}
        self.depth_actions: dict[tuple[int, int], float] = {}

    def add(self, category, depth, action, weight):
        if category not in self.categories:
            raise RunnerError(f"unknown decision category {category!r}")
        if not weight > 0:
            raise RunnerError("decision weights must be positive")
        self.categories[category] += float(weight)
        key = (int(depth), int(action))
        self.depth_actions[key] = self.depth_actions.get(key, 0.0) + float(weight)

    def add_trace(self, actions, origins, route, weight, *, fallback):
        if len(actions) != len(origins):
            raise RunnerError("a decision trace needs one origin per executed action")
        for depth, (action, origin) in enumerate(zip(actions, origins)):
            self.add(origin_category(origin, route, fallback=fallback), depth, action, weight)

    def merge(self, other: "DecisionMass") -> "DecisionMass":
        for c, w in other.categories.items():
            self.categories[c] += w
        for key, w in other.depth_actions.items():
            self.depth_actions[key] = self.depth_actions.get(key, 0.0) + w
        return self

    @property
    def total(self) -> float:
        return math.fsum(self.categories.values())

    def fractions(self) -> dict:
        total = self.total
        if not total > 0:
            raise RunnerError("no executed decision to classify")
        return {c: self.categories[c] / total for c in CATEGORIES}

    def entropy(self) -> float:
        return same_depth_entropy([(d, a, w) for (d, a), w in self.depth_actions.items()])

    def to_json(self) -> dict:
        return {"categories": dict(self.categories),
                "depth_actions": [[d, a, w] for (d, a), w in sorted(self.depth_actions.items())]}

    @classmethod
    def from_json(cls, value) -> "DecisionMass":
        mass = cls()
        for c, w in value["categories"].items():
            if c not in mass.categories:
                raise RunnerError(f"unknown decision category {c!r}")
            mass.categories[c] = float(w)
        for d, a, w in value["depth_actions"]:
            mass.depth_actions[(int(d), int(a))] = float(w)
        return mass


def same_depth_entropy(depth_actions) -> float:
    """sum_l W_l H_l / sum_l W_l with H_l the entropy (nats) of the weighted actions at depth l."""
    by_depth: dict[int, dict[int, float]] = {}
    for depth, action, weight in depth_actions:
        if weight < 0:
            raise RunnerError("negative decision weight")
        by_depth.setdefault(int(depth), {})
        by_depth[int(depth)][int(action)] = by_depth[int(depth)].get(int(action), 0.0) + float(weight)
    total, accumulated = 0.0, 0.0
    for actions in by_depth.values():
        w = np.asarray([v for v in actions.values() if v > 0], dtype=np.float64)
        if not len(w):
            continue
        p = w / w.sum()
        accumulated += float(w.sum()) * float(-np.sum(p * np.log(p)))
        total += float(w.sum())
    if not total > 0:
        raise RunnerError("no executed decision for the within-depth entropy")
    return accumulated / total


def gap_summary(gaps, threshold) -> dict:
    g = np.asarray(gaps, dtype=np.float64)
    if g.ndim != 1 or not len(g) or not np.isfinite(g).all():
        raise RunnerError("the gap audit needs a nonempty finite gap vector")
    if (g < 0).any():
        raise RunnerError("audited gaps are nonnegative (search inversions enter as 0)")
    return {"median": float(np.median(g)), "p5": float(np.quantile(g, 0.05)),
            "near_zero_fraction": float(np.mean(g < float(threshold))), "sites": int(len(g)),
            "quantiles": "numpy median and linear-interpolation quantile"}


def chi_square_region(dof: int, alpha: float) -> tuple[float, float]:
    from scipy.stats import chi2
    if int(dof) < 1 or not 0 < alpha < 1:
        raise RunnerError("the chi-square region needs positive degrees of freedom and 0 < alpha < 1")
    return float(chi2.ppf(alpha / 2, int(dof))), float(chi2.ppf(1 - alpha / 2, int(dof)))


def perturbation_operator(r, basis, scale, names, factor=1.0) -> np.ndarray:
    """Measurement-unit map of the observer's noise model: delta = factor diag(s) B L_R xi, history rows zero."""
    r, basis, scale = (np.asarray(v, dtype=np.float64) for v in (r, basis, scale))
    k = r.shape[0]
    if r.shape != (k, k) or basis.ndim != 2 or basis.shape[1] != k or scale.shape != (basis.shape[0],) \
            or len(names) != basis.shape[0]:
        raise RunnerError("observer noise model, basis, scale and measurement schema do not match")
    if not math.isfinite(factor) or factor < 0:
        raise RunnerError("the perturbation factor must be finite and nonnegative")
    chol = np.linalg.cholesky((r + r.T) / 2 + 1e-12 * np.eye(k))
    operator = float(factor) * (basis @ chol) * scale[:, None]
    for i, name in enumerate(names):
        if name in HISTORY_COORDINATES:
            operator[i] = 0.0
    return operator


def perturbation_draw(operator, seed_parts) -> np.ndarray:
    rng = np.random.default_rng([int(p) for p in seed_parts])
    return np.asarray(operator) @ rng.standard_normal(np.asarray(operator).shape[1])


def higher_quantile(values, level):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return float(np.quantile(values, min(1.0, max(0.0, level)), method="higher"))


def sequence_radii(errors, units, strata, actions, *, alpha, minimum) -> dict:
    """The runtime's calibration rule (cross_morse.learning._calibrate) for an h-step target, without the A0
    identity (an absolute loss change of a sequence starting with A0 is not zero)."""
    errors, units = np.asarray(errors, dtype=np.float64), np.asarray(units)
    strata, actions = np.asarray(strata, dtype=np.int64), np.asarray(actions, dtype=np.int64)

    def entry(rows):
        if not len(rows):
            return None
        unique, inverse = np.unique(units[rows], return_inverse=True)
        per_unit = np.full(len(unique), -np.inf)
        np.maximum.at(per_unit, inverse, errors[rows])
        return {"radius": higher_quantile(per_unit, 1 - alpha), "trajectory_count": int(len(unique)),
                "row_count": int(len(rows)), "small_sample": bool(len(unique) < minimum)}
    finite = np.isfinite(errors)
    cells, by_action = {}, {}
    for a in range(10):
        value = entry(np.flatnonzero(finite & (actions == a)))
        if value is not None:
            by_action[str(a)] = value
    for st in np.unique(strata[finite]):
        for a in range(10):
            rows = np.flatnonzero(finite & (strata == st) & (actions == a))
            if len(np.unique(units[rows])) >= minimum:
                cells[f"{int(st)}:{a}"] = entry(rows)
    return {"cells": cells, "actions": by_action}


def radius_lookup(radii, stratum, action) -> float:
    entry = radii["cells"].get(f"{int(stratum)}:{int(action)}") or radii["actions"].get(str(int(action)))
    value = None if entry is None else entry.get("radius")
    return float(value) if value is not None else math.inf


def harm_indicators(delta, mu, radius, valid, tolerance=IDENTITY_TOLERANCE) -> dict:
    """{statistic: (numerator mask, denominator mask)} of the one-action harm and coverage statistics.

    ``delta``: realized one-action consequences, [S] or [S, B] (bootstrap draws in columns)."""
    d = np.asarray(delta, dtype=np.float64)
    d = d[:, None] if d.ndim == 1 else d
    mu = np.asarray(mu, dtype=np.float64)[:, None]
    r = np.asarray(radius, dtype=np.float64)[:, None]
    valid = np.broadcast_to(np.asarray(valid, dtype=bool)[:, None] & np.isfinite(mu) & ~np.isnan(r), d.shape)
    valid = valid & np.isfinite(d)
    with np.errstate(invalid="ignore"):
        harmful = valid & (d > tolerance)
        safe = valid & (mu + r <= 0)
        safe_point = valid & (mu <= 0)
        covered = valid & (np.abs(d - mu) <= r + tolerance)
    return {"false_safe": (safe & harmful, valid), "harm_miss": (safe & harmful, harmful),
            "false_safe_point": (safe_point & harmful, valid), "harm_miss_point": (safe_point & harmful, harmful),
            "coverage_1": (covered, valid)}


def sequence_indicators(realized, prediction, radius, tolerance=IDENTITY_TOLERANCE):
    x = np.asarray(realized, dtype=np.float64)
    x = x[:, None] if x.ndim == 1 else x
    p = np.asarray(prediction, dtype=np.float64)[:, None]
    r = np.asarray(radius, dtype=np.float64)[:, None]
    evaluable = np.broadcast_to(np.isfinite(p) & ~np.isnan(r), x.shape) & np.isfinite(x)
    with np.errstate(invalid="ignore"):
        covered = evaluable & (np.abs(x - p) <= r + tolerance)
    return covered, evaluable


def statistic_names(horizons) -> list[str]:
    return ["false_safe", "harm_miss", "false_safe_point", "harm_miss_point", "coverage_1"] + \
           [f"sequence_{h}" for h in horizons]


class SharedUnits:
    """A transductive run: every audited site is scored on the same statistical units (graph cores)."""

    def __init__(self, arrays, tolerance=IDENTITY_TOLERANCE):
        self.counts = np.asarray(arrays["unit_counts"], dtype=np.float64)
        self.units = len(self.counts)
        self.mu, self.radius = np.asarray(arrays["mu"]), np.asarray(arrays["radius1"])
        self.valid = np.asarray(arrays["valid"], dtype=bool)
        self.delta = np.nan_to_num(np.asarray(arrays["delta_sums"], dtype=np.float64))
        self.horizons = [int(h) for h in arrays["horizons"]]
        self.prediction = np.asarray(arrays["prediction"], dtype=np.float64)
        self.radius_h = np.asarray(arrays["radius_h"], dtype=np.float64)
        self.realized = np.asarray(arrays["realized_sums"], dtype=np.float64)
        self.tolerance = tolerance
        if self.delta.shape != (len(self.mu), self.units) or self.realized.shape != (len(self.horizons),) + self.delta.shape:
            raise RunnerError("transductive reliability arrays have inconsistent shapes")

    def _mean(self, sums, c):
        return (sums @ c.T) / (self.counts @ c.T)[None, :]

    def totals(self, statistic, c):
        c = np.asarray(c, dtype=np.float64)
        if statistic.startswith("sequence_"):
            k = self.horizons.index(int(statistic.split("_")[1]))
            realized = self._mean(np.nan_to_num(self.realized[k]), c)
            realized[~np.isfinite(self.realized[k]).all(axis=1)] = np.nan
            covered, evaluable = sequence_indicators(realized, self.prediction[k], self.radius_h[k], self.tolerance)
            return covered.sum(axis=0).astype(float), evaluable.sum(axis=0).astype(float)
        delta = self._mean(self.delta, c)
        numerator, denominator = harm_indicators(delta, self.mu, self.radius, self.valid, self.tolerance)[statistic]
        return numerator.sum(axis=0).astype(float), denominator.sum(axis=0).astype(float)


class OwnUnits:
    """An independent-graph run: each graph carries its own sites; unit totals are resampled whole."""

    def __init__(self, arrays):
        self.table = {k: np.asarray(v, dtype=np.float64) for k, v in arrays.items() if k.endswith(("_num", "_den"))}
        self.units = len(np.asarray(arrays["unit_ids"]))

    def totals(self, statistic, c):
        c = np.asarray(c, dtype=np.float64)
        try:
            return c @ self.table[f"{statistic}_num"], c @ self.table[f"{statistic}_den"]
        except KeyError:
            raise RunnerError(f"{statistic} is not among the unit totals") from None


def own_unit_totals(sites, horizons, tolerance=IDENTITY_TOLERANCE) -> dict:
    """Per-graph numerator/denominator columns from per-site scalar consequences (independent-graph runs)."""
    unit = np.asarray(sites["unit"], dtype=np.int64)
    ids = np.unique(unit)
    index = np.searchsorted(ids, unit)
    out = {"unit_ids": ids}
    indicators = harm_indicators(sites["delta"], sites["mu"], sites["radius1"], sites["valid"], tolerance)
    for k, h in enumerate(horizons):
        indicators[f"sequence_{h}"] = sequence_indicators(sites["realized"][k], sites["prediction"][k],
                                                          sites["radius_h"][k], tolerance)
    for name, (numerator, denominator) in indicators.items():
        out[f"{name}_num"] = np.bincount(index, weights=numerator[:, 0].astype(float), minlength=len(ids))
        out[f"{name}_den"] = np.bincount(index, weights=denominator[:, 0].astype(float), minlength=len(ids))
    return out


def rate_interval(strata, statistic, *, reps, seed) -> dict | None:
    """Pooled rate over strata and its stratified percentile bootstrap 95% interval (units resampled per stratum)."""
    if not strata:
        return None
    numerator = denominator = 0.0
    for s in strata:
        n, d = s.totals(statistic, np.ones((1, s.units)))
        numerator, denominator = numerator + float(n[0]), denominator + float(d[0])
    if not denominator > 0:
        return None
    rng = np.random.default_rng([int(v) for v in np.atleast_1d(seed)])
    nums, dens = np.zeros(int(reps)), np.zeros(int(reps))
    for s in strata:
        for start in range(0, int(reps), BLOCK):
            size = min(BLOCK, int(reps) - start)
            draws = rng.multinomial(s.units, np.full(s.units, 1.0 / s.units), size=size).astype(np.float64)
            n, d = s.totals(statistic, draws)
            nums[start:start + size] += n
            dens[start:start + size] += d
    defined = dens > 0
    if defined.sum() < 0.5 * len(dens):
        low = high = None
    else:
        low, high = (float(v) for v in np.quantile(nums[defined] / dens[defined], [0.025, 0.975]))
    return {"value": numerator / denominator, "low": low, "high": high, "numerator": numerator,
            "denominator": denominator, "replicates": int(reps), "undefined_replicates": int((~defined).sum()),
            "seed": [int(v) for v in np.atleast_1d(seed)], "strata": len(strata),
            "interval": "stratified percentile bootstrap over statistical units (strata = runs)"}


# ============================================================================= planning (torch-free)
def _deployment(run: SealedRun, fixture_route=None, rule="deployed") -> dict:
    run.require("admit")
    candidates = run.candidates()
    lock = run.json("policy_lock.json")
    admission = run.json("admit/decision.json")
    deployed = dict(candidates[lock["deployed_index"]], registry_index=lock["deployed_index"])
    comparator = dict(candidates[lock["comparator_index"]], registry_index=lock["comparator_index"])
    confirmed = dict(candidates[lock["confirmed_index"]], registry_index=lock["confirmed_index"])
    result = {"deployed": deployed, "comparator": comparator, "confirmed": confirmed,
              "admitted": bool(admission.get("passed")), "rule": "deployed",
              "global_return": lock["deployed_index"] == lock["comparator_index"], "fixture_override": None}
    if rule == "provisional":
        index = int(run.json("select/decision.json")["provisional_index"])
        result.update(deployed=dict(candidates[index], registry_index=index), rule="provisional",
                      global_return=index == lock["comparator_index"])
    elif rule != "deployed":
        raise RunnerError(f"the audited rule is 'deployed' (q_D) or 'provisional' (q_s of Eq. 19), not {rule!r}")
    if fixture_route is not None:
        if not run.synthetic:
            raise RunnerError("a fixture rule override is reserved for synthetic smoke parents")
        options = sorted((dict(c, registry_index=i) for i, c in enumerate(candidates)
                          if c["route"] == fixture_route and c["ensemble"] == comparator["ensemble"]
                          and not c.get("execution_failure")),
                         key=lambda c: (c.get("profile") != "full", c.get("horizon") or 0, c.get("kappa") or 0,
                                        c["hash"]))
        if not options:
            raise RunnerError(f"{run.root.name}: no registered {fixture_route} candidate at E={comparator['ensemble']}")
        result.update(deployed=options[0], confirmed=options[0], admitted=True, global_return=False,
                      fixture_override={"route": fixture_route, "registry_index": options[0]["registry_index"],
                                        "scope": "software check only: a registered feedback candidate replaces "
                                                 "the deployed rule of a synthetic parent"})
    return result


def _evaluation_inputs(run: SealedRun, role: str) -> list[int]:
    prepared = run.root / "prepared" / "graph_splits.npz"
    if run.dataset in ("PascalVOC-SP", "COCO-SP", "Peptides-func") and prepared.is_file():
        return sorted(int(g) for g in run.npz("prepared/graph_splits.npz")[role])
    return [0]


def plan_decision_audit(parents, output, *, role="test", reps=None, seed_offset=13, perturbation_factor=1.0,
                        near_zero_threshold=None, audit_inputs=0, audit_seed=0, gate_from=None, gate_alpha=.05,
                        gate_profile="full", ema_weight=.5, stress_dataset=None, calibration_trajectories=0,
                        observer_study=True, fixture_route=None, provisional=()) -> Path:
    """Declare the audit of every parent run (a Roman-empire campaign: every split run); nothing is executed.

    ``provisional``: datasets whose audited rule is the provisional selection q_s of main Eq. (19) instead of the
    deployed rule q_D (e.g. a rule returned to q_C after confirmation); every output then names that rule."""
    if role not in ROLE_STAGES:
        raise RunnerError(f"evaluation role must be one of {sorted(ROLE_STAGES)}; decision roles are never reused")
    if not parents:
        raise RunnerError("give at least one --parent run")
    routes = fixture_route.values() if isinstance(fixture_route, dict) else [fixture_route]
    if any(r is not None and r not in FEEDBACK_ROUTES for r in routes):
        raise RunnerError(f"a fixture rule override names a feedback route of {FEEDBACK_ROUTES}")
    threshold = certificate_floor() if near_zero_threshold is None else float(near_zero_threshold)
    if not threshold > 0 or not 0 < ema_weight <= 1 or not 0 < gate_alpha < 1 or not perturbation_factor >= 0:
        raise RunnerError("declare a positive near-zero threshold, 0 < EMA weight <= 1, 0 < gate alpha < 1 and a "
                          "nonnegative perturbation factor")
    if type(audit_inputs) is not int or audit_inputs < 0 or type(calibration_trajectories) is not int \
            or calibration_trajectories < 0:
        raise RunnerError("audit_inputs and calibration_trajectories are nonnegative integers (0 = all)")
    gate_from = {str(k): Path(v).resolve() for k, v in (gate_from or {}).items()}
    horizons = registered_horizons()
    datasets, jobs, seen = [], [], set()
    for parent in parents:
        runs = [SealedRun(m) for m in member_runs(parent)]
        dataset = runs[0].dataset
        if len({r.dataset for r in runs}) != 1 or dataset in seen:
            raise RunnerError(f"{parent}: one parent (run or campaign) per dataset")
        seen.add(dataset)
        members = []
        for run in runs:
            check_role(run, role)
            found = _deployment(run, fixture_route.get(dataset) if isinstance(fixture_route, dict) else fixture_route,
                                "provisional" if dataset in set(provisional or ()) else "deployed")
            deployed = found["deployed"]
            route = deployed["route"]
            profile = "full" if deployed.get("profile") in (None, "none") else deployed["profile"]
            inputs = _evaluation_inputs(run, role)
            audited = None
            if audit_inputs and len(inputs) > audit_inputs:
                chosen = np.random.default_rng([int(audit_seed), len(jobs)]).choice(len(inputs), audit_inputs,
                                                                                    replace=False)
                audited = sorted(inputs[i] for i in chosen)
            key = f"{dataset}/{run.root.name}" if f"{dataset}/{run.root.name}" in gate_from else dataset
            reused = None
            if key in gate_from:
                reused = _reused_gate(gate_from[key], run)
            seed = int(run.config["seed"])
            jobs.append({
                "job": len(jobs), "dataset": dataset, "slug": slug(dataset), "member": run.root.name,
                "run": str(run.root), "identity": run.identity(), "synthetic": run.synthetic,
                "deployed": candidate_identity(deployed), "comparator": candidate_identity(found["comparator"]),
                "confirmed": candidate_identity(found["confirmed"]), "route": route, "profile": profile,
                "admitted": found["admitted"], "global_return": found["global_return"], "rule": found["rule"],
                "fixture_override": found["fixture_override"], "feedback": route in FEEDBACK_ROUTES,
                "plans": route in PLANNING_ROUTES, "seeds": [int(s) for s in run.config["training"]["seeds"]],
                "evaluation_inputs": len(inputs), "audited_inputs": audited,
                "perturbation_seed": seed + int(seed_offset),
                "gate": {"needed": (not found["global_return"]) or dataset == stress_dataset,
                         "alpha": reused["alpha"] if reused else float(gate_alpha),
                         "profile": reused["profile"] if reused else gate_profile, "reused": reused},
                "stress": dataset == stress_dataset,
                "observer_study": bool(observer_study)})
            members.append(run.root.name)
        datasets.append({"dataset": dataset, "slug": slug(dataset), "parent": str(Path(parent).resolve()),
                         "members": members, "synthetic": runs[0].synthetic,
                         "aggregation": "pooled over the official split runs" if len(runs) > 1 else "single run"})
    if stress_dataset is not None and stress_dataset not in seen:
        raise RunnerError(f"the stress dataset {stress_dataset!r} is not among the planned parents")
    first = SealedRun(member_runs(parents[0])[0])
    return create_plan(output, {
        "experiment": "decision-log audits of the deployed rules (Tables S21, S33-S34; Figures S7, S16, S17; "
                      "main Figure 3c-d; E7, E8)",
        "record_set": PLAN_KIND, "record_sets": list(SETS), "role": role, "horizons": horizons,
        "near_zero_threshold_nats": threshold,
        "perturbation": {"factor": float(perturbation_factor), "seed_offset": int(seed_offset),
                         "definition": "one draw of the deployed observer's frozen measurement-noise model R "
                                       "(latent N(0, R) mapped through its PCA basis and robust scale), "
                                       "history coordinates unchanged, applied site-locally to the received "
                                       "measurement"},
        "audit": {"inputs": int(audit_inputs), "seed": int(audit_seed),
                  "scope": "per-site gap, perturbation and consequence analysis; decision traces use every input"},
        "observer": {"levels": list(OBSERVER_LEVELS), "ema_weight": float(ema_weight), "study": bool(observer_study),
                     "region": "two-sided (1 - calibration_alpha) chi-square, dof = latent dimension"},
        "calibration": {"trajectories": int(calibration_trajectories), "seed": int(audit_seed)},
        "bootstrap": {"reps": int(reps or first.config["admission"]["bootstrap_reps"]),
                      "seed": int(first.config["seed"]) + 17},
        "tolerance_nats": IDENTITY_TOLERANCE, "stress_dataset": stress_dataset,
        "datasets": datasets, "jobs": jobs, "audit_code_sha256": code_identity(),
        "definitions": DEFINITIONS})


def _reused_gate(directory: Path, run: SealedRun) -> dict:
    from .artifacts import verify
    verify(directory)
    executed = _read(directory / "experiment.json")
    identity = run.identity()
    if executed.get("experiment") != "input_gate" or executed["parent"]["config_sha256"] != identity["config_sha256"] \
            or executed["parent"]["source_sha256"] != identity["runtime_source_sha256"]:
        raise RunnerError(f"{directory}: the gate must come from a sealed input_gate run of {run.root.name}")
    declared = _read(directory / "plan.json")["gate"]
    return {"from": directory.name, "path": str(directory / "raw" / "gate.json"),
            "gate_sha256": sha256(directory / "raw" / "gate.json"), "profile": declared["profile"],
            "alpha": float(declared["alpha"])}


# ============================================================================= execution (PyTorch)
class RecordingBank:
    """A runtime MeasurementBank whose measurements are copied as they are returned (values untouched)."""

    def __init__(self, bank, entries):
        self.bank, self.entries = bank, entries

    def measure(self, h, logits, depth, max_depth, *args, **kwargs):
        value = self.bank.measure(h, logits, depth, max_depth, *args, **kwargs)
        self.entries.append((int(depth), value.detach().float().cpu().numpy().copy()))
        return value

    def __getattr__(self, name):
        return getattr(self.bank, name)


@contextlib.contextmanager
def recording_sensors(log):
    """``cross_morse.policies.sensor_bank`` wrapped for one execution; every bank's measurements go to ``log``."""
    from .negative_controls import runtime_sensor

    def factory(original):
        def make(graph, model, config, profile="full"):
            entries = []
            log.append(entries)
            return RecordingBank(original(graph, model, config, profile), entries)
        return make
    with runtime_sensor(factory):
        yield


def execute_member(model, x, graph, config, candidate, controller, reference, *, record):
    """(probabilities, trace, measurements) of one member, or (None, failure record, None)."""
    from cross_morse.policies import PolicyExecutionError, execute_policy
    log = []
    try:
        with (recording_sensors(log) if record else contextlib.nullcontext()):
            probabilities, trace = execute_policy(model, x, graph, config, candidate, controller, reference)
    except PolicyExecutionError as error:
        return None, {"failure": error.reason, "actions": list(error.trace.get("actions", []))}, None
    if record and len(log) > 1:
        raise RunnerError("one execution created more than one sensor bank")
    return np.asarray(probabilities), trace, (log[0] if log else [])


def ranked_beam(controller, state, horizon, config, *, open_loop=False, root_actions=None):
    """``cross_morse.policies.beam_plan`` (same arithmetic, ties and pruning) returning its final ranking.

    ``root_actions`` restricts the root expansion; None is returned when the restriction leaves no finite root."""
    from cross_morse.core import ACTIONS, site_transport_accounting
    from cross_morse.policies import recovery_action
    c = config["controller"]

    def finite_cost(value, component):
        if not np.isfinite(value):
            raise FloatingPointError(f"Nonfinite {component} in learned policy planning")
        return float(value)
    beams, roots, ranked = [(0., [], copy.deepcopy(state))], None, []
    for offset in range(int(horizon)):
        expanded = []
        for cost, sequence, simulated in beams:
            supported = controller.supported(simulated.y, simulated.depth)
            if open_loop:
                supported = [a for a in supported if a != 9]
            restricted = offset == 0 and root_actions is not None
            if restricted:
                supported = [a for a in supported if a in root_actions]
            finite_actions, scores = [], None
            if supported:
                scores = controller.score(simulated, supported, kind="radius", kappa=1., lambda_rho=c["lambda_rho"])
                finite_actions = [a for a in supported if np.isfinite(scores[a]["score"])]
            if not finite_actions:
                if restricted:
                    return None
                a = recovery_action(supported, last_safe_available=not open_loop)
                nxt = controller.predict(simulated, a)
                value = finite_cost(controller.stage_cost(simulated, a), "recovery stage cost")
                expanded.append((cost + c["gamma"] ** offset * value, sequence + [a], nxt))
                if offset == 0:
                    roots = [a]
                continue
            ordered = sorted(finite_actions, key=lambda a: (scores[a]["score"], a))[:c["top_k"]]
            if offset == 0:
                roots = list(ordered)
            for a in ordered:
                stage = controller.stage_cost(simulated, a)
                transport = site_transport_accounting(a, simulated.depth, config)
                stage += scores[a]["sigma"] + c["lambda_rho"] * ACTIONS[a].rho + c["lambda_transport"] * transport["hops"]
                stage = finite_cost(stage, "stage cost")
                nxt = controller.predict(simulated, a)
                expanded.append((cost + c["gamma"] ** offset * stage, sequence + [a], nxt))
        if not expanded:
            return {"plan": [], "ranked": [], "roots": roots}
        ranked = []
        for cost, sequence, simulated in expanded:
            tail = finite_cost(controller.terminal_value(simulated), "terminal value")
            ranked.append((cost + c["gamma"] ** len(sequence) * c["lambda_terminal"] * tail, cost, sequence, simulated))
        ranked.sort(key=lambda item: (item[0], tuple(item[2])))
        beams = [(cost, sequence, simulated) for _, cost, sequence, simulated in ranked[:c["beam_width"]]]
    return {"plan": [int(a) for a in beams[0][1]], "ranked": [(float(o), [int(a) for a in s]) for o, _, s, _ in ranked],
            "roots": roots}


def beam_gap(controller, state, horizon, config, ranked, *, open_loop=False):
    """(gap, kind, inversion) of a planning decision: J(best other root) - J(executed plan)."""
    plan, roots = ranked["plan"], ranked["roots"] or []
    if not plan or len(roots) < 2:
        return math.nan, 0, False
    best, first = ranked["ranked"][0][0], plan[0]
    others = [objective for objective, sequence in ranked["ranked"] if sequence[0] != first]
    if others:
        return min(others) - best, 2, False
    dedicated = ranked_beam(controller, state, horizon, config, open_loop=open_loop,
                            root_actions=[a for a in roots if a != first])
    if dedicated is None or not dedicated["ranked"]:
        return math.nan, 0, False
    gap = dedicated["ranked"][0][0] - best
    return gap, 3, bool(gap < 0)


def counterfactual_decision(route, controller, state, supported, depth, config, candidate, *, exhausted=False):
    """The runtime's feedback decision at one site (``_execute_policy``), from a given state and support set."""
    from cross_morse.policies import PlanningBudgetExceeded, beam_plan, recovery_action
    L, c = int(config["model"]["depth"]), config["controller"]
    if not supported:
        return recovery_action(supported), None, "recovery_no_support"
    if not state.valid:
        return recovery_action(supported), None, "recovery_observer"
    if exhausted:
        return recovery_action(supported), None, "recovery_budget"
    plan = None
    if route == "mpc_rh":
        try:
            plan = [int(a) for a in beam_plan(controller, state, min(int(candidate["horizon"]), L - depth), config,
                                              {"calls": 0})]
        except PlanningBudgetExceeded:
            return recovery_action(supported), None, "recovery_budget"
        a, origin = (plan[0] if plan else recovery_action(supported)), "policy"
    else:
        scores = controller.score(state, supported, kind=route, kappa=float(candidate.get("kappa", 1.)),
                                  lambda_rho=c["lambda_rho"])
        finite = [a for a in supported if np.isfinite(scores[a]["score"])]
        a = min(finite, key=lambda a: (scores[a]["score"], a)) if finite else recovery_action(supported)
        origin = "policy" if finite else "recovery_no_finite_score"
    if a not in supported and a not in (8, 9):
        a, origin = recovery_action(supported), "recovery_support_mismatch"
    return int(a), plan, origin


def replay_states(controller, ys, actions):
    """The observer states of ``_execute_policy``: start at depth 0, then observe with the previous action."""
    states = [controller.start(ys[0])]
    for depth in range(1, len(actions)):
        states.append(controller.observe(states[-1], ys[depth], depth, actions[depth - 1]))
    return states


def _same(a, b, rel=1e-9, absolute=1e-12):
    """Equality of a replayed value and its logged counterpart (the runtime's jsonable logs nonfinite as null)."""
    a = None if a is None or not math.isfinite(float(a)) else float(a)
    b = None if b is None or not math.isfinite(float(b)) else float(b)
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=rel, abs_tol=absolute)


def audit_feedback_execution(controller, candidate, config, trace, measurements, operator, seed_parts):
    """Per-site gap, perturbation and stability records of one feedback execution, and the replayed states."""
    from cross_morse.recovery import label_free_safe
    route = candidate["route"]
    L, c = int(config["model"]["depth"]), config["controller"]
    actions = [int(a) for a in trace["actions"]]
    origins, decisions = list(trace["origins"]), trace["decisions"]
    if [d for d, _ in measurements] != list(range(len(actions))):
        raise RunnerError("the recorded measurements do not cover every executed site exactly once")
    ys = [y for _, y in measurements]
    states = replay_states(controller, ys, actions)
    rows, exhausted = [], False
    for depth, (y, state, decision, origin) in enumerate(zip(ys, states, decisions, origins)):
        supported = controller.supported(y, depth)
        where = f"depth {depth}"
        if (list(supported) != [int(a) for a in decision["supported_actions"]]
                or bool(state.valid) != bool(decision.get("observer_valid"))
                or bool(label_free_safe(y)) != bool(decision.get("anchor_updated"))
                or not _same(state.nis, decision.get("observer_nis"))):
            raise RunnerError(f"{where}: the replayed controller differs from the runtime decision log")
        if origin == "recovery_budget":
            exhausted = True
        kind, gap, inversion, plan = 0, math.nan, False, None
        if origin == "policy" and route in ("one_step", "radius"):
            scores = controller.score(state, supported, kind=route, kappa=float(candidate.get("kappa", 1.)),
                                      lambda_rho=c["lambda_rho"])
            finite = sorted((scores[a]["score"], a) for a in supported if np.isfinite(scores[a]["score"]))
            if not finite or finite[0][1] != actions[depth]:
                raise RunnerError(f"{where}: the replayed scores do not select the executed action")
            if len(finite) > 1:
                kind, gap = 1, float(finite[1][0] - finite[0][0])
            if decision.get("score_gap") is not None and not _same(decision["score_gap"], gap, 1e-9, 1e-12):
                raise RunnerError(f"{where}: the replayed score gap differs from the runtime's score_gap")
        elif origin == "policy" and route == "mpc_rh" and decision.get("planned_sequence"):
            plan = [int(a) for a in decision["planned_sequence"]]
            horizon = min(int(candidate["horizon"]), L - depth)
            ranked = ranked_beam(controller, state, horizon, config)
            if ranked["plan"] != plan:
                raise RunnerError(f"{where}: the replayed beam does not return the executed plan")
            gap, kind, inversion = beam_gap(controller, state, horizon, config, ranked)
        perturbed = (np.asarray(y, dtype=np.float64) + perturbation_draw(operator, [*seed_parts, depth])).astype(np.float32)
        state_t = controller.start(perturbed) if depth == 0 else \
            controller.observe(states[depth - 1], perturbed, depth, actions[depth - 1])
        supported_t = controller.supported(perturbed, depth)
        action_t, plan_t, origin_t = counterfactual_decision(route, controller, state_t, supported_t, depth, config,
                                                             candidate, exhausted=exhausted)
        rows.append({"depth": depth, "action": actions[depth], "origin": ORIGINS.index(origin), "kind": kind,
                     "gap": gap, "inversion": int(inversion), "perturbed_action": action_t,
                     "perturbed_origin": ORIGINS.index(origin_t), "switch": int(action_t != actions[depth]),
                     "plan_stable": -1 if plan is None else int(plan_t == plan),
                     "anchor_flip": int(bool(label_free_safe(perturbed)) != bool(label_free_safe(y)))})
    return rows, states


def audit_open_loop_plan(controller, candidate, config, trace, measurements, operator, seed_parts):
    """The depth-0 planning decision of an MPC-OL rule (gap, perturbed plan); no anchor is measurement-selected."""
    from cross_morse.policies import PlanningBudgetExceeded, beam_plan
    L = int(config["model"]["depth"])
    if not measurements or measurements[0][0] != 0:
        raise RunnerError("the MPC-OL initial measurement was not recorded")
    y = measurements[0][1]
    horizon = min(int(candidate["horizon"]), L)
    state = controller.start(y)
    ranked = ranked_beam(controller, state, horizon, config, open_loop=True)
    plan = ranked["plan"]
    if (plan + [0] * (L - len(plan)))[:len(trace["actions"])] != [int(a) for a in trace["actions"]]:
        raise RunnerError("the replayed MPC-OL plan differs from the executed schedule")
    gap, kind, inversion = beam_gap(controller, state, horizon, config, ranked, open_loop=True)
    perturbed = (np.asarray(y, dtype=np.float64) + perturbation_draw(operator, [*seed_parts, 0])).astype(np.float32)
    try:
        plan_t = [int(a) for a in beam_plan(controller, controller.start(perturbed), horizon, config, {"calls": 0},
                                            open_loop=True)]
    except PlanningBudgetExceeded:
        plan_t = []
    first, first_t = (plan[0] if plan else 0), (plan_t[0] if plan_t else 0)
    return [{"depth": 0, "action": int(trace["actions"][0]), "origin": ORIGINS.index("fixed_schedule"), "kind": kind,
             "gap": gap, "inversion": int(inversion), "perturbed_action": first_t,
             "perturbed_origin": ORIGINS.index("fixed_schedule"), "switch": int(first_t != first),
             "plan_stable": int(plan_t == plan), "anchor_flip": 0}]


def risk_and_rollouts(controller, states, actions, config, horizons):
    """Critic prediction of every executed action and consequence-model rollouts of the executed sequence."""
    lam = config["controller"]["lambda_rho"]
    T, H = len(actions), max(horizons)
    mu, sigma, radius = (np.full(T, np.nan) for _ in range(3))
    stratum = np.zeros(T, dtype=np.int64)
    prediction = np.full((len(horizons), T), np.nan)
    for depth, state in enumerate(states):
        action = int(actions[depth])
        values = controller.score(state, [action], kind="radius", kappa=1., lambda_rho=lam)[action]
        mu[depth], sigma[depth], radius[depth] = values["mean"], values["sigma"], values["radius"]
        stratum[depth] = controller.stratum(state.y, state.depth)
        if not controller.sequence_models_available:
            continue
        total, current = 0.0, state
        for k in range(min(H, T - depth)):
            total += float(controller.stage_cost(current, actions[depth + k]))
            current = controller.predict(current, actions[depth + k])
            if k + 1 in horizons:
                prediction[horizons.index(k + 1), depth] = total
    return {"mu": mu, "sigma": sigma, "radius1": radius, "stratum": stratum, "prediction": prediction}


def unit_loss_sums(model, rows, labels, unit_index, n_units, spec, device):
    """A function h -> per-unit sums of the per-row task loss on the evaluation rows (labels read by the caller)."""
    import torch
    from cross_morse.tasks import loss as task_loss
    targets = torch.as_tensor(np.asarray(labels), device=device)
    rows = None if rows is None else torch.as_tensor(np.asarray(rows, dtype=np.int64), device=device)

    def sums(h):
        logits = model.logits(h)
        logits = logits if rows is None else logits[rows]
        values = task_loss(logits.float(), targets, spec, reduction="none").detach().cpu().numpy().astype(np.float64)
        finite = np.isfinite(values)
        return np.bincount(unit_index[finite], weights=values[finite], minlength=n_units)
    return sums


def replay_consequences(model, x, graph, config, actions, safe_depths, sums):
    """Realized losses along the executed trajectory and the one-action consequences of the critic's target."""
    L, horizon = int(config["model"]["depth"]), int(config["exploration"]["critic_horizon"])
    model.bind_graph(graph)
    h0 = model.encode(x)
    h, last_safe, last = h0, h0, 0
    losses, deltas = [sums(h)], []
    for depth, action in enumerate(actions):
        anchor = int(safe_depths[depth])
        if anchor == depth:
            last_safe, last = h.clone(), depth
        elif anchor != last:
            raise RunnerError("the replayed last-safe anchors differ from the executed trace")
        following = model.step(h, h0, graph, int(action), last_safe, site=depth)
        nominal, alternate = model.native_step(h, h0, graph, site=depth), following
        for offset in range(1, min(horizon, L - depth)):
            nominal = model.native_step(nominal, h0, graph, site=depth + offset)
            alternate = model.native_step(alternate, h0, graph, site=depth + offset)
        deltas.append(sums(alternate) - sums(nominal))
        h = following
        losses.append(sums(h))
    return np.stack(losses), np.stack(deltas), model.logits(h)


# ----------------------------------------------------------------------------- D_explore phases (PyTorch)
def _partition(records, controller, run_root, profile):
    from cross_morse.learning import calibration_partition
    roles, declaration = calibration_partition(records, controller.config)
    sealed = _read(Path(run_root) / "fit" / profile / "calibration_partition.json")
    if json_safe(declaration["groups"]) != json_safe(sealed["groups"]):
        raise RunnerError("the recomputed calibration partition differs from the sealed one")
    return roles


def trajectory_states(observer, records, rows):
    """Observer states of one logged trajectory, exactly as cross_morse.learning._encode_records builds them."""
    from cross_morse.learning import _profile_y
    states, state = [], None
    for i in rows:
        y = np.asarray(records["y"])[i]
        depth, previous = int(np.asarray(records["depth"])[i]), int(np.asarray(records["previous_action"])[i])
        if state is None:
            state = observer.start(y, previous, depth)
        elif depth != state.depth or not np.array_equal(state.y, _profile_y(y, observer.profile)):
            state = observer.observe(state, y, depth, previous)
        states.append(state)
    return states


def _trajectories(records, rows):
    from cross_morse.learning import _groups
    groups, depths = _groups(records), np.asarray(records["depth"])
    rows = np.asarray(rows, dtype=np.int64)
    out = {}
    for i in rows:
        out.setdefault(int(groups[i]), []).append(int(i))
    return {g: sorted(r, key=lambda i: int(depths[i])) for g, r in sorted(out.items())}


def calibration_phase(run_root, controller, profile, config, horizons, *, max_trajectories, seed):
    """h-step radii of the consequence-model rollout on the D_explore radius-fit partition (runtime rule)."""
    from cross_morse.exploration import load_records
    from cross_morse.learning import _holdout_groups, _strata
    records = load_records(Path(run_root) / "explore")
    roles = _partition(records, controller, run_root, profile)
    trajectories = _trajectories(records, np.flatnonzero(roles == 1))
    keys = list(trajectories)
    if max_trajectories and len(keys) > max_trajectories:
        keys = sorted(np.random.default_rng(seed).choice(keys, max_trajectories, replace=False).tolist())
    if not controller.sequence_models_available:
        return {"status": "NO_SEQUENCE_MODELS", "horizons": horizons, "radii": {}}
    units, strata = _holdout_groups(records), _strata(np.asarray(records["y"]), np.asarray(records["depth"]), config)
    actions, stage = np.asarray(records["action"]).astype(int), np.asarray(records["stage_delta"], dtype=np.float64)
    error = {h: [] for h in horizons}
    rows_used = {h: [] for h in horizons}
    for key in keys:
        rows = trajectories[key]
        states = trajectory_states(controller.observer, records, rows)
        cumulative = np.concatenate([[0.0], np.cumsum(stage[rows])])
        for j, state in enumerate(states):
            total, current = 0.0, state
            for k in range(min(max(horizons), len(rows) - j)):
                total += float(controller.stage_cost(current, int(actions[rows[j + k]])))
                current = controller.predict(current, int(actions[rows[j + k]]))
                if k + 1 in horizons:
                    error[k + 1].append(abs(cumulative[j + k + 1] - cumulative[j] - total))
                    rows_used[k + 1].append(rows[j])
    alpha = float(config["controller"].get("calibration_alpha", 0.05))
    minimum = int(config["controller"].get("calibration_min_count", 20))
    radii = {}
    for h in horizons:
        r = np.asarray(rows_used[h], dtype=np.int64)
        radii[str(h)] = sequence_radii(error[h], units[r], strata[r], actions[r], alpha=alpha, minimum=minimum)
    return {"status": "CALIBRATED", "horizons": horizons, "alpha": alpha, "minimum_trajectories_for_cell": minimum,
            "trajectories": len(keys), "available_trajectories": len(trajectories),
            "rows": {str(h): len(rows_used[h]) for h in horizons}, "radii": radii,
            "target": "absolute h-step loss change sum of logged stage_delta (D_explore labels)",
            "prediction": "sum of stage-model predictions along transition-model states from the observer state",
            "partition": "radius-fit trajectories of the sealed calibration partition (role 1)"}


class LatentObserver:
    """raw (weight 1) or EMA (weight beta) estimate of the conditioned observer's latent measurement, updated at
    the registered observer cadence; constant innovation covariance estimated on the model-fit rows."""

    def __init__(self, base, weight, covariance):
        self.base, self.weight = base, float(weight)
        self.cov = np.asarray(covariance, dtype=np.float64)
        self.inverse = np.linalg.inv(self.cov)
        self.scale = np.sqrt(np.maximum(np.diag(self.cov), 1e-12))
        self.dimension, self.max_depth = base.dimension, base.max_depth
        self.period, self.profile = base.period, base.profile

    def start(self, y, previous_action=-1, depth=0):
        from cross_morse.learning import ControllerState, _profile_y
        raw = _profile_y(y, self.profile)
        return ControllerState(self.base.transform(raw), self.cov.copy(), 0.0, 0.0, 0.0, raw, int(depth),
                               int(previous_action), bool(np.isfinite(raw).all()), np.zeros(self.dimension))

    def observe(self, state, y, depth, previous_action):
        from cross_morse.learning import ControllerState, _profile_y
        raw = _profile_y(y, self.profile)
        finite = bool(np.isfinite(raw).all())
        if depth < state.depth:
            raise ValueError("Observer depth cannot move backwards")
        if depth > state.depth + 1:
            result = self.start(raw, previous_action, depth)
            result.valid = False
            return result
        if depth % self.period:
            return ControllerState(state.z, state.cov, state.nis, state.bias, state.whiteness, raw, int(depth),
                                   int(previous_action), finite and state.valid, state.innovation)
        error = self.base.transform(raw) - state.z
        nis = max(0.0, float(error @ self.inverse @ error))
        previous = state.innovation if state.innovation is not None else np.zeros_like(error)
        whiteness = float(error @ previous / max(np.linalg.norm(error) * np.linalg.norm(previous), 1e-12))
        z = state.z + self.weight * error
        return ControllerState(z, self.cov.copy(), nis, float(np.mean(error / self.scale)), whiteness, raw,
                               int(depth), int(previous_action), finite and bool(np.isfinite(z).all()), error)


def global_kf(observer, records, rows):
    """The conditioned observer's latent space with one global transition and its own residual Q, R = 0.25 Q + 1e-3 I."""
    from cross_morse.learning import ConditionedObserver, _low_rank_diagonal, _profile_y
    payload = copy.deepcopy(observer.payload)
    median, scale = np.asarray(payload["median"]), np.asarray(payload["scale"])
    center, basis = np.asarray(payload["center"]), np.asarray(payload["basis"])

    def normalized(values):
        values = _profile_y(np.asarray(values)[rows], payload["profile"])
        values = np.where(np.isfinite(values), values, median)
        return np.clip((values - median) / scale, -5, 5)
    now, following = normalized(records["y"]), normalized(records["y_next"])
    z, z_next = (now - center) @ basis, (following - center) @ basis
    design = np.column_stack((z, now[:, 5:8], np.ones(len(z))))
    residual = z_next - design @ np.asarray(payload["global_coef"])
    k, rank = basis.shape[1], int(payload["rank"])
    q = _low_rank_diagonal(residual.T @ residual / max(len(residual) - 1, 1) + np.eye(k) * 1e-4, rank)
    payload.update(conditioned={}, q=q.tolist(), r=_low_rank_diagonal(0.25 * q + np.eye(k) * 1e-3, rank).tolist(),
                   nis_threshold=1e12, identification="global ridge transition; R=0.25Q+1e-3I (decision audit E8)")
    return ConditionedObserver(payload)


def innovation_events(observer, records, rows):
    """(NIS, innovation) at every observer update of the given rows' complete trajectories."""
    nis, innovations = [], []
    for trajectory in _trajectories(records, rows).values():
        for j, state in enumerate(trajectory_states(observer, records, trajectory)):
            if j and state.depth % observer.period == 0:
                nis.append(float(state.nis))
                innovations.append(np.asarray(state.innovation, dtype=np.float64))
    return np.asarray(nis), (np.stack(innovations) if innovations else np.zeros((0, observer.dimension)))


def train_critic(observer, records, fitting, fit_rows, validation_rows, config, directory, device, name):
    """The runtime's critic recipe of the final fit (cross_morse.learning._fit_one) on another observer's features."""
    import torch
    from cross_morse.common import stop_requested
    from cross_morse.learning import ActionCritic, ResidualModel, _encode_records, _fingerprint, _train_network, \
        _with_actions
    features, _, _, _ = _encode_records(observer, records, fitting)
    x = _with_actions(features, np.asarray(records["action"])[fitting].astype(int))
    torch.manual_seed(FINAL_FIT_SEED)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(FINAL_FIT_SEED)
    width = features.shape[1]
    models = [ActionCritic(width + 10), ResidualModel(width + 10, observer.dimension), ResidualModel(width + 10, 1),
              ResidualModel(width, 1)]          # creation order of _fit_one: identical initial RNG stream
    critic = models[0]
    scaler, audit = _train_network(critic, x, np.asarray(records["delta"])[fitting, None], fit_rows, validation_rows,
                                   config, Path(directory) / f"critic_{slug(name)}.last.pt",
                                   _fingerprint(records, config) + ":" + name, str(device), stop_requested,
                                   FINAL_FIT_SEED, heteroscedastic=True)
    if scaler is None:
        raise RunnerError(f"{name}: the critic has no finite fitting targets")
    return critic, scaler, audit


def critic_predictions(critic, scaler, observer, records, rows, device):
    import torch
    from cross_morse.learning import _encode_records, _with_actions
    features, _, _, _ = _encode_records(observer, records, rows)
    actions = np.asarray(records["action"])[rows].astype(int)
    x = _with_actions(features, actions)
    out = []
    with torch.no_grad():
        for start in range(0, len(x), 65536):
            mean, _ = critic(torch.as_tensor(x[start:start + 65536], dtype=torch.float32, device=str(device)))
            out.append(mean.cpu().numpy().reshape(-1))
    mean = np.concatenate(out) * scaler["scale"][0] + scaler["mean"][0]
    mean[actions == 0] = 0.0
    return mean


def observer_phase(run_root, controller, profile, directory, device, *, ema_weight):
    """E8: risk RMSE and NIS consistency of the four observer variants on the D_explore assessment partition."""
    from cross_morse.exploration import load_records
    from cross_morse.learning import _holdout_groups
    records = load_records(Path(run_root) / "explore")
    config = controller.config
    roles = _partition(records, controller, run_root, profile)
    fitting, assessment = np.flatnonzero(roles == 0), np.flatnonzero(roles == 2)
    groups = _holdout_groups(records)
    unique = np.unique(groups[fitting])
    shuffled = np.random.default_rng(FINAL_FIT_SEED).permutation(unique)
    count = min(len(unique) - 1, max(1, int(math.ceil(0.10 * len(unique)))))
    validation_groups = set(shuffled[:count].tolist())
    sealed_groups = controller.payload.get("audit", {}).get("final_fit", {}).get("internal_validation_groups")
    if sealed_groups is not None and sorted(int(g) for g in sealed_groups) != sorted(int(g) for g in validation_groups):
        raise RunnerError("the replicated internal validation groups differ from the sealed final fit")
    mask = np.asarray([g in validation_groups for g in groups[fitting]])
    fit_rows, validation_rows = np.flatnonzero(~mask), np.flatnonzero(mask)
    observer_rows = fitting[fit_rows]
    base = controller.observer
    variants = {}
    for name, weight in (("raw", 1.0), ("EMA", float(ema_weight))):
        _, innovations = innovation_events(LatentObserver(base, weight, np.eye(base.dimension)), records, observer_rows)
        if not len(innovations):
            raise RunnerError("no observer update among the model-fit rows")
        second = innovations.T @ innovations / len(innovations) + 1e-6 * np.eye(base.dimension)
        variants[name] = LatentObserver(base, weight, second)
    variants["global KF"] = global_kf(base, records, observer_rows)
    variants["conditioned KF"] = base
    alpha = float(config["controller"].get("calibration_alpha", 0.05))
    low, high = chi_square_region(base.dimension, alpha)
    target = np.asarray(records["delta"], dtype=np.float64)[assessment]
    result, arrays = {}, {"assessment_rows": assessment, "delta": target}
    for name in OBSERVER_LEVELS:
        observer = variants[name]
        critic, scaler, audit = train_critic(observer, records, fitting, fit_rows, validation_rows, config, directory,
                                             device, name)
        prediction = critic_predictions(critic, scaler, observer, records, assessment, device)
        nis, innovations = innovation_events(observer, records, assessment)
        inside = (nis >= low) & (nis <= high)
        squared = float(np.sum((target - prediction) ** 2))
        result[name] = {"assessment_rows": int(len(assessment)), "squared_error_sum": squared,
                        "mse": squared / len(assessment), "rmse": math.sqrt(squared / len(assessment)),
                        "nis_events": int(len(nis)), "nis_in_region": int(inside.sum()),
                        "nis_coverage": float(inside.mean()) if len(nis) else None,
                        "innovation_rms": float(np.sqrt(np.mean(innovations ** 2))) if len(innovations) else None,
                        "training": {k: v for k, v in audit.items() if k != "history"},
                        "module_ablations_arm": MODULE_ARMS[name]}
        arrays[f"prediction_{MODULE_ARMS[name]}"] = prediction
        arrays[f"nis_{MODULE_ARMS[name]}"] = nis
    sealed = Path(run_root) / "fit" / profile / "calibration_assessment.npz"
    parity = None
    if sealed.is_file():
        with np.load(sealed, allow_pickle=False) as z:
            if np.array_equal(np.asarray(z["row_index"]), assessment):
                sealed_mean = np.asarray(z["mean"], dtype=np.float64)
                parity = {"sealed_critic_rmse": float(np.sqrt(np.mean((target - sealed_mean) ** 2))),
                          "max_abs_difference_retrained_conditioned": float(np.max(np.abs(
                              sealed_mean - arrays["prediction_conditioned"])))}
    return {"status": "MEASURED", "profile": profile, "dimension": int(base.dimension), "period": int(base.period),
            "alpha": alpha, "region": [low, high], "ema_weight": float(ema_weight),
            "rows": {"model_fit": int(len(fitting)), "assessment": int(len(assessment)),
                     "internal_validation_groups": len(validation_groups)},
            "variants": result, "conditioned_parity": parity,
            "innovation_covariance": "second moment of model-fit innovations (raw, EMA)"}, arrays


# ----------------------------------------------------------------------------- the execution phase (PyTorch)
def _resolve(run: SealedRun, identity: dict) -> dict:
    candidate = dict(run.candidates()[identity["registry_index"]], registry_index=identity["registry_index"])
    if candidate_identity(candidate) != identity:
        raise RunnerError(f"{run.root.name}: a planned rule differs from the sealed registry")
    return candidate


def _sealed_member(run: SealedRun, candidate, seed, role, collection):
    folder = run.root / "member_predictions" / (role if collection else "")
    path = folder / f"{candidate['behavior_id']}_seed{seed}.json"
    return _read(path) if path.is_file() else None


def execution_phase(plan, entry, directory, ctx, run, device):
    import torch
    from cross_morse.common import precision_context
    from cross_morse.provenance import sha256_state_dict
    from cross_morse.tasks import probabilities as to_probabilities, task_spec, valid_sample_mask
    from .input_gate import compatibility_vectors, fit_development_gate, gate_decision, unit_cores
    role, config, horizons = plan["role"], ctx.config, [int(h) for h in plan["horizons"]]
    rules = {name: _resolve(run, entry[name]) for name in ("deployed", "comparator", "confirmed")}
    deployed = rules["deployed"]
    route = deployed["route"]
    seeds = entry["seeds"]
    members = {name: seeds[:c["ensemble"]] for name, c in rules.items()}
    controllers = {name: ctx.controller_for(c) for name, c in rules.items()}
    models = {}

    def plant(seed):
        if seed not in models:
            models[seed] = ctx.plant(seed)
        return models[seed]
    for seed in sorted(set(members["deployed"]) | set(members["comparator"]) | set(members["confirmed"])):
        plant(seed)
    frozen = {s: sha256_state_dict(m) for s, m in models.items()}
    analyse = entry["feedback"] or entry["plans"]
    calibration = _read(directory / "calibration.json") if entry["feedback"] else None
    gate = None
    if entry["gate"]["needed"]:
        reused = entry["gate"]["reused"]
        if reused is not None:
            if sha256(reused["path"]) != reused["gate_sha256"]:
                raise RunnerError("the reused gate changed since planning")
            gate = _read(reused["path"])
        else:
            gate = fit_development_gate(ctx.data, [plant(s) for s in members["confirmed"]], config, device,
                                        profile=entry["gate"]["profile"], alpha=entry["gate"]["alpha"])
        _dump(directory / "gate.json", dict(gate, frozen_before_evaluation_labels=True))
    operator, deployed_runs_schema = None, None
    observer = controllers["deployed"].observer if analyse else None
    spec = task_spec(config, ctx.data.metadata)
    labels = ctx.labels(role) if entry["feedback"] else None
    collection = ctx.collection
    role_units = None if collection else np.asarray(ctx.units(role))      # graph cores of the evaluation rows
    audited = None if entry["audited_inputs"] is None else set(int(g) for g in entry["audited_inputs"])
    mass = DecisionMass()
    executions, unit_rows, site_rows, reliability = [], [], [], {"shared": [], "own": []}
    digests = {(name, s): hashlib.sha256() for name in rules for s in members[name]}
    executed_inputs = {name: 0 for name in rules}
    probability_parity = []
    for graph_id, x, graph, rows, positions in ctx.episodes(role):
        input_id = int(graph_id)
        if collection:
            units = [input_id]
        else:
            units = sorted(int(u) for u in np.unique(role_units))
        decisions = {}
        if gate is not None:
            with torch.no_grad(), precision_context(config, torch.device(device)):
                vectors, _ = compatibility_vectors([plant(s) for s in members["confirmed"]], x, graph, config,
                                                   profile=entry["gate"]["profile"],
                                                   cores=unit_cores(ctx.data, None, collection))
            decisions = {(input_id if collection else int(u)): gate_decision(gate, v) for u, v in vectors.items()}
        record = analyse and (audited is None or input_id in audited)
        results = {}

        def run_rule(name, measure=False):
            if name in results:
                return results[name]
            if name != "deployed" and rules[name]["hash"] == deployed["hash"] and "deployed" in results:
                results[name] = results["deployed"]
                return results[name]
            outcome = []
            for seed in members[name]:
                p, trace, measurements = execute_member(plant(seed), x, graph, config, rules[name],
                                                        controllers[name], ctx.reference, record=measure)
                outcome.append((seed, p, trace, measurements))
                if p is None:
                    break
            executed_inputs[name] += 1
            results[name] = outcome
            return outcome
        deployed_runs = run_rule("deployed", measure=record)
        deployed_ok = all(p is not None for _, p, _, _ in deployed_runs)
        served = {u: (not entry["global_return"]) and deployed_ok and (gate is None or decisions[u][1] == 1)
                  for u in units}
        fallback = [u for u in units if not served[u]]
        if any(served.values()):
            weight = sum(served.values()) / len(members["deployed"])
            for _, _, trace, _ in deployed_runs:
                mass.add_trace(trace["actions"], trace["origins"], route, weight, fallback=False)
        if fallback:
            source = "deployed" if entry["global_return"] else "comparator"
            outcome = run_rule(source)
            if not all(p is not None for _, p, _, _ in outcome):
                raise RunnerError(f"the locked comparator failed on input {input_id}; the fallback cannot be served")
            weight = len(fallback) / len(members[source])
            for _, _, trace, _ in outcome:
                mass.add_trace(trace["actions"], trace["origins"], rules[source]["route"], weight, fallback=True)
        stress = {}
        if entry["stress"]:
            outcome = run_rule("confirmed")
            ok = all(p is not None for _, p, _, _ in outcome)
            active = ok and any(int(a) != 0 for _, _, trace, _ in outcome for a in trace["actions"])
            for u in units:
                passed = decisions[u][1] == 1
                stress[u] = {"stress_served": int(passed and ok), "stress_active": int(passed and ok and active)}
        for u in units:
            unit_rows.append({"unit_id": f"graph:{u}" if collection else f"core:{u}", "input_id": input_id,
                              "g_input": "" if u not in decisions else decisions[u][1],
                              "gate_score": "" if u not in decisions else repr(decisions[u][0]),
                              "served_by": "deployed" if served[u] else ("comparator" if not entry["global_return"]
                                                                          else "deployed (global return)"),
                              **stress.get(u, {})})
        for name, outcome in results.items():
            for seed, p, trace, _ in outcome:
                executions.append({"input_id": input_id, "rule": name, "member": seed, "route": rules[name]["route"],
                                   "status": "EXECUTED" if p is not None else "FAILED",
                                   "terminal_depth": len(trace.get("actions", [])),
                                   "nonnominal_actions": sum(int(a) != 0 for a in trace.get("actions", [])),
                                   "recovery_sites": sum(str(o).startswith("recovery") for o in trace.get("origins", [])),
                                   "failure": trace.get("failure", "") if p is None else ""})
                if p is not None and collection:
                    digests[(name, seed)].update(json.dumps({"graph_id": input_id, "actions": trace.get("actions", []),
                                                             "safe_depths": trace.get("safe_depths", [])},
                                                            sort_keys=True).encode())
                if p is not None and not collection:
                    sealed = _sealed_member(run, rules[name], seed, role, False)
                    if sealed is not None:
                        stored = run.root / "member_predictions" / f"{rules[name]['behavior_id']}_seed{seed}.npy"
                        deviation = float(np.max(np.abs(np.load(stored, mmap_mode="r") - p))) if stored.is_file() else None
                        probability_parity.append({"rule": name, "member": seed,
                                                   "actions_equal": sealed.get("actions") == trace["actions"]
                                                   and sealed.get("safe_depths") == trace["safe_depths"],
                                                   "max_abs_probability_deviation": deviation})
        if not (record and deployed_ok):
            continue
        # per-site analysis of the deployed rule on this input
        controller = controllers["deployed"]
        if operator is None:
            deployed_runs_schema = list(deployed_runs[0][2]["sensor_configuration"]["schema"])
            operator = perturbation_operator(observer.r, observer.basis, observer.scale, deployed_runs_schema,
                                             plan["perturbation"]["factor"])
        if entry["feedback"]:
            task = getattr(plant(members["deployed"][0]), "task", spec)   # the exploration targets' task spec
            if collection:
                unit_index, n_units = np.zeros(len(positions), dtype=np.int64), 1
                episode_labels, row_index = labels[positions], None
            else:
                unique_units, unit_index = np.unique(role_units, return_inverse=True)
                n_units, episode_labels, row_index = len(unique_units), labels, rows
            counts = np.bincount(unit_index[np.asarray(valid_sample_mask(episode_labels, task))], minlength=n_units)
        for seed, p, trace, measurements in deployed_runs:
            seed_parts = [entry["perturbation_seed"], input_id, seed]
            if entry["feedback"]:
                sites, states = audit_feedback_execution(controller, deployed, config, trace, measurements, operator,
                                                         seed_parts)
            else:
                sites, states = audit_open_loop_plan(controller, deployed, config, trace, measurements, operator,
                                                     seed_parts), None
            for row in sites:
                row.update(input_id=input_id, member=seed)
            site_rows.extend(sites)
            if not entry["feedback"]:
                continue
            actions = [int(a) for a in trace["actions"]]
            risk = risk_and_rollouts(controller, states, actions, config, horizons)
            model = plant(seed)
            sums = unit_loss_sums(model, row_index, episode_labels, unit_index, n_units, task, device)
            with torch.no_grad(), precision_context(config, torch.device(device)):
                losses, deltas, logits = replay_consequences(model, x, graph, config, actions, trace["safe_depths"],
                                                             sums)
                replayed = to_probabilities(logits.float(), model.task).cpu().numpy()
            probability_parity.append({"rule": "deployed (consequence replay)", "member": seed, "input_id": input_id,
                                       "max_abs_probability_deviation": float(np.max(np.abs(replayed - p)))})
            T = len(actions)
            radius_h = np.full((len(horizons), T), np.nan)
            realized = np.full((len(horizons), T, n_units), np.nan)
            for k, h in enumerate(horizons):
                radii = calibration["radii"].get(str(h))
                for depth in range(T):
                    if radii is not None:
                        radius_h[k, depth] = radius_lookup(radii, risk["stratum"][depth], actions[depth])
                    if depth + h <= T:
                        realized[k, depth] = losses[depth + h] - losses[depth]
            block = {"mu": risk["mu"], "radius1": risk["radius1"], "valid": np.isfinite(risk["mu"]),
                     "prediction": risk["prediction"], "radius_h": radius_h}
            if collection:
                total = float(counts[0])
                reliability["own"].append({**block, "unit": np.full(T, input_id), "delta": deltas[:, 0] / total,
                                           "realized": realized[:, :, 0] / total})
            else:
                reliability["shared"].append({**block, "delta_sums": deltas, "realized_sums": realized,
                                              "unit_counts": counts, "unit_ids": unique_units})
    if any(sha256_state_dict(m) != frozen[s] for s, m in models.items()):
        raise RuntimeError("a frozen predictor changed during the audit")
    parity = {"probabilities": probability_parity, "digests": {},
              "decision_log": {"replayed_sites": len(site_rows), "mismatches": 0,
                               "rule": "every replayed site matched the runtime decision log (support set, observer "
                                       "validity and NIS, anchor update, argmin and score_gap, MPC-RH plan); a "
                                       "mismatch stops the job"}}
    if collection:
        for (name, seed), digest in digests.items():
            sealed = _sealed_member(run, rules[name], seed, role, True)
            complete = executed_inputs[name] == entry["evaluation_inputs"]
            parity["digests"][f"{name}/{seed}"] = {
                "replayed": digest.hexdigest() if complete else None,
                "sealed": None if sealed is None else sealed.get("actions_sha256"),
                "equal": None if sealed is None or not complete else digest.hexdigest() == sealed.get("actions_sha256")}
    files = {}
    files["decisions.json"] = _dump(directory / "decisions.json", {
        "schema": JOB_SCHEMA, "route": route, "global_return": entry["global_return"], "role": role,
        "mass": mass.to_json(), "units": len(unit_rows),
        "weighting": "each (evaluation unit, depth) carries unit mass shared by its serving ensemble's members"})
    files["units.csv"] = _write_csv(directory / "units.csv", unit_rows,
                                    ["unit_id", "input_id", "g_input", "gate_score", "served_by", "stress_served",
                                     "stress_active"])
    files["executions.csv"] = _write_csv(directory / "executions.csv", executions,
                                         ["input_id", "rule", "member", "route", "status", "terminal_depth",
                                          "nonnominal_actions", "recovery_sites", "failure"])
    files["parity.json"] = _dump(directory / "parity.json", parity)
    if operator is not None:
        source = run.root / "fit" / entry["profile"] / "controller.pt"
        files["perturbation.json"] = _dump(directory / "perturbation.json", {
            "definition": plan["perturbation"]["definition"], "factor": plan["perturbation"]["factor"],
            "seed": entry["perturbation_seed"], "seed_parts": "[seed, input id, member seed, depth]",
            "measurement_schema": deployed_runs_schema, "history_coordinates": list(HISTORY_COORDINATES),
            "operator": operator, "observer_source": f"fit/{entry['profile']}/controller.pt",
            "observer_source_sha256": sha256(source) if source.is_file() else None})
    if analyse:
        files.update(_site_files(directory, entry, site_rows, reliability, horizons))
    return files


def _site_files(directory, entry, site_rows, reliability, horizons):
    files = {}
    if not site_rows:
        raise RunnerError("no executed site was audited")
    columns = ("input_id", "member", "depth", "action", "origin", "kind", "gap", "inversion", "perturbed_action",
               "perturbed_origin", "switch", "plan_stable", "anchor_flip")
    sites = {c: np.asarray([r[c] for r in site_rows], dtype=np.float64 if c == "gap" else np.int64) for c in columns}
    files["sites.npz"] = save_arrays(directory / "sites.npz", replace=True, **sites)
    audited = sites["kind"] > 0
    trajectories = {}
    for r in site_rows:
        key = (r["input_id"], r["member"])
        trajectories[key] = trajectories.get(key, True) and not r["anchor_flip"]
    planning = sites["plan_stable"] >= 0
    # The deposited per-site file (compact types; the exported values are recomputed from exactly these arrays).
    files["gaps.npz"] = save_arrays(
        directory / "gaps.npz", replace=True,
        gap=sites["gap"][audited].astype(np.float32), kind=sites["kind"][audited].astype(np.int8),
        switch=sites["switch"][audited].astype(np.int8), inversion=sites["inversion"][audited].astype(np.int8),
        input_id=sites["input_id"][audited].astype(np.int32), member=sites["member"][audited].astype(np.int32),
        depth=sites["depth"][audited].astype(np.int16), plan_stable=sites["plan_stable"][planning].astype(np.int8),
        anchor_stable=np.asarray([int(v) for v in trajectories.values()], dtype=np.int8),
        excluded_origin=sites["origin"][~audited].astype(np.int8),
        route=np.asarray(entry["route"]), feedback=np.asarray(int(entry["feedback"])))
    if entry["feedback"]:
        h = np.asarray(horizons, dtype=np.int64)
        if reliability["shared"]:
            parts = reliability["shared"]
            if any(not np.array_equal(p["unit_ids"], parts[0]["unit_ids"]) for p in parts):
                raise RunnerError("transductive executions were scored on different units")
            files["reliability.npz"] = save_arrays(
                directory / "reliability.npz", replace=True, mode=np.asarray("shared"), horizons=h,
                unit_ids=parts[0]["unit_ids"], unit_counts=parts[0]["unit_counts"],
                mu=np.concatenate([p["mu"] for p in parts]), radius1=np.concatenate([p["radius1"] for p in parts]),
                valid=np.concatenate([p["valid"] for p in parts]),
                prediction=np.concatenate([p["prediction"] for p in parts], axis=1),
                radius_h=np.concatenate([p["radius_h"] for p in parts], axis=1),
                delta_sums=np.concatenate([p["delta_sums"] for p in parts]),
                realized_sums=np.concatenate([p["realized_sums"] for p in parts], axis=1))
        else:
            parts = reliability["own"]
            merged = {"unit": np.concatenate([p["unit"] for p in parts]),
                      "delta": np.concatenate([p["delta"] for p in parts]),
                      "mu": np.concatenate([p["mu"] for p in parts]),
                      "radius1": np.concatenate([p["radius1"] for p in parts]),
                      "valid": np.concatenate([p["valid"] for p in parts]),
                      "prediction": np.concatenate([p["prediction"] for p in parts], axis=1),
                      "radius_h": np.concatenate([p["radius_h"] for p in parts], axis=1),
                      "realized": np.concatenate([p["realized"] for p in parts], axis=1)}
            files["reliability_sites.npz"] = save_arrays(directory / "reliability_sites.npz", replace=True, **merged)
            files["reliability.npz"] = save_arrays(directory / "reliability.npz", replace=True, mode=np.asarray("own"),
                                                   horizons=h, **own_unit_totals(merged, horizons))
    return files


# ----------------------------------------------------------------------------- run
def run_decision_audit(output, *, job=None, device="cuda", trust=False) -> list[Path]:
    if not trust:
        raise RunnerError("loading local checkpoints requires explicit trust (--trust-local-checkpoints)")
    output = Path(output).resolve()
    plan = load_plan(output, PLAN_KIND)
    done = []
    for entry in selected_jobs(plan, job):
        directory = job_directory(output, entry)
        if not (directory / "audit.json").is_file():
            _run_job(plan, entry, directory, device, output)
        done.append(directory / "audit.json")
    return done


def _run_job(plan, entry, directory, device, output):
    from .ensemble_grid import ParentContext
    run = SealedRun(entry["run"])
    check_identity(run, entry["identity"])
    check_role(run, plan["role"])
    started = utc_now()
    directory.mkdir(parents=True, exist_ok=True)
    files = {"environment.json": _dump(directory / "environment.json", environment_record(device))}
    ctx = ParentContext(run, device, through="evaluate" if plan["role"] == "test" else "confirm")
    deployed = _resolve(run, entry["deployed"])
    profile = entry["profile"]
    controller = ctx.controller(profile)
    horizons = [int(h) for h in plan["horizons"]]
    if entry["feedback"]:
        if not (directory / "calibration.json").is_file():
            result = calibration_phase(run.root, controller, profile, controller.config, horizons,
                                       max_trajectories=plan["calibration"]["trajectories"],
                                       seed=[plan["calibration"]["seed"], entry["job"]])
            _dump(directory / "calibration.json", result)
        files["calibration.json"] = sha256(directory / "calibration.json")
    if entry["observer_study"]:
        if not (directory / "observers.json").is_file():
            result, arrays = observer_phase(run.root, controller, profile, directory, device,
                                            ema_weight=plan["observer"]["ema_weight"])
            save_arrays(directory / "observers.npz", replace=True, **arrays)
            _dump(directory / "observers.json", result)
        files["observers.json"] = sha256(directory / "observers.json")
        files["observers.npz"] = sha256(directory / "observers.npz")
    files.update(execution_phase(plan, entry, directory, ctx, run, device))
    if (directory / "gate.json").is_file():
        files["gate.json"] = sha256(directory / "gate.json")
    check_identity(run, entry["identity"])
    _dump(directory / "audit.json", {
        "schema": JOB_SCHEMA, "dataset": entry["dataset"], "member": entry["member"], "role": plan["role"],
        "route": deployed["route"], "deployed": entry["deployed"], "global_return": entry["global_return"],
        "fixture_override": entry["fixture_override"], "files": files,
        "identities": {"runtime_source_sha256": runtime_source_sha256(), "parent": entry["identity"],
                       "plan_sha256": sha256(output / "plan.json"), "audit_code_sha256": code_identity(),
                       "parent_files_read": run.inputs()},
        "started_utc": started, "finished_utc": utc_now()})


# ============================================================================= export (torch-free)
def collect_jobs(output, plan, *, allow_partial=False) -> list[dict]:
    """Completed jobs with their files verified against the recorded SHA-256; missing jobs refuse a complete set."""
    found, missing = [], []
    for entry in plan["jobs"]:
        directory = job_directory(output, entry)
        marker = directory / "audit.json"
        if not marker.is_file():
            missing.append(f"{entry['dataset']}/{entry['member']} (job {entry['job']})")
            continue
        audit = _read(marker)
        for name, digest in audit["files"].items():
            if not (directory / name).is_file() or sha256(directory / name) != digest:
                raise RunnerError(f"{directory}/{name} differs from its recorded SHA-256")
        found.append({"entry": entry, "directory": directory, "audit": audit})
    if missing and not allow_partial:
        raise RunnerError("jobs have not completed: " + ", ".join(missing))
    return found


def _by_dataset(plan, jobs):
    out = {}
    for d in plan["datasets"]:
        members = [j for j in jobs if j["entry"]["dataset"] == d["dataset"]]
        if members:
            out[d["dataset"]] = members
    return out


def _npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def parity_summary(members) -> dict:
    """Sealed-action parity of the instrumented executions of one dataset (from the jobs' parity.json)."""
    out = {"transductive_executions_compared": 0, "transductive_actions_equal": 0, "graph_digests_compared": 0,
           "graph_digests_equal": 0, "decision_log_sites_replayed": 0}
    for j in members:
        parity = _read(j["directory"] / "parity.json")
        for p in parity.get("probabilities", []):
            if "actions_equal" in p:
                out["transductive_executions_compared"] += 1
                out["transductive_actions_equal"] += int(bool(p["actions_equal"]))
        for d in parity.get("digests", {}).values():
            if d.get("equal") is not None:
                out["graph_digests_compared"] += 1
                out["graph_digests_equal"] += int(bool(d["equal"]))
        out["decision_log_sites_replayed"] += parity.get("decision_log", {}).get("replayed_sites", 0)
    return out


def _row(keys, quantity, value):
    value = float(value)
    if not math.isfinite(value):
        raise RunnerError(f"{keys}/{quantity}: nonfinite value")
    return {"keys": {k: str(v) for k, v in keys.items()}, "quantity": quantity, "value": value,
            "unit": UNITS[quantity]}


def summarize_action_gaps(plan, jobs) -> dict:
    threshold = float(plan["near_zero_threshold_nats"])
    rows, details = [], {}
    for dataset, members in _by_dataset(plan, jobs).items():
        parts = [_npz(j["directory"] / "gaps.npz") for j in members if (j["directory"] / "gaps.npz").is_file()]
        routes = sorted({j["entry"]["route"] for j in members})
        if not parts:
            details[dataset] = {"status": "NO_SCORED_DECISIONS", "routes": routes,
                                "reason": "the deployed rule is open loop and does not plan"}
            continue
        gap = np.concatenate([p["gap"] for p in parts])
        if not len(gap):
            details[dataset] = {"status": "NO_SCORED_DECISIONS", "routes": routes,
                                "reason": "no executed policy decision with two finite candidates"}
            continue
        used = np.maximum(gap, 0.0)
        summary = gap_summary(used, threshold)
        switch = np.concatenate([p["switch"] for p in parts])
        plans = np.concatenate([p["plan_stable"] for p in parts])
        anchors = np.concatenate([p["anchor_stable"] for p in parts])
        keys = {"dataset": dataset}
        values = {"median_gap_nats": summary["median"], "p5_gap_nats": summary["p5"],
                  "near_zero_threshold_nats": threshold, "near_zero_fraction": summary["near_zero_fraction"],
                  "switch_fraction": float(np.mean(switch)), "decision_sites": float(len(gap))}
        planning = any(r in PLANNING_ROUTES for r in routes)
        if planning and len(plans):
            values["beam_sequence_stability"] = float(np.mean(plans))
        elif not planning:
            values["beam_sequence_stability"] = 1.0
        if len(anchors):
            values["recovery_index_stability"] = float(np.mean(anchors))
        rows += [_row(keys, q, values[q]) for q in DEFINITIONS["action_gap_audit"] if q in values]
        details[dataset] = {"status": "AUDITED", "routes": routes, "audited_sites": int(len(gap)),
                            "search_inversions": int(np.concatenate([p["inversion"] for p in parts]).sum()),
                            "gap_kinds": {GAP_KINDS[k]: int(np.sum(np.concatenate([p["kind"] for p in parts]) == k))
                                          for k in (1, 2, 3)},
                            "excluded_sites_by_origin": {
                                ORIGINS[o]: int(np.sum(np.concatenate([p["excluded_origin"] for p in parts]) == o))
                                for o in range(len(ORIGINS))},
                            "planning_sites": int(len(plans)), "trajectories": int(len(anchors)),
                            "runs": [j["entry"]["member"] for j in members], "quantiles": summary["quantiles"],
                            "sealed_parity": parity_summary(members)}
    complete = {d["dataset"] for d in plan["datasets"]} >= set(DATASETS) and \
        len({j["entry"]["dataset"] for j in jobs}) == len(plan["datasets"]) and \
        all(len([r for r in rows if r["keys"]["dataset"] == d]) == len(DEFINITIONS["action_gap_audit"])
            for d, v in details.items() if v["status"] == "AUDITED")
    return {"rows": rows, "details": details, "complete": complete}


def summarize_decision_traces(plan, jobs) -> dict:
    rows, details = [], {}
    if plan["role"] != "test":
        raise RunnerError("decision traces describe D_test decisions; plan the audit on the test role")
    for dataset, members in _by_dataset(plan, jobs).items():
        mass = DecisionMass()
        for j in members:
            mass.merge(DecisionMass.from_json(_read(j["directory"] / "decisions.json")["mass"]))
        fractions = mass.fractions()
        keys = {"dataset": dataset}
        rows.append(_row(keys, "same_depth_action_entropy", mass.entropy()))
        rows += [_row(keys, f"fraction_{c}", fractions[c]) for c in CATEGORIES]
        details[dataset] = {"decision_weight": mass.total, "category_weights": mass.categories,
                            "runs": [j["entry"]["member"] for j in members],
                            "routes": sorted({j["entry"]["route"] for j in members}),
                            "global_return": sorted({j["entry"]["global_return"] for j in members}),
                            "sealed_parity": parity_summary(members)}
    complete = set(details) >= set(DATASETS)
    return {"rows": rows, "details": details, "complete": complete}


def reliability_strata(jobs) -> list:
    strata = []
    for j in sorted(jobs, key=lambda j: (j["entry"]["dataset"], j["entry"]["member"])):
        path = j["directory"] / "reliability.npz"
        if not path.is_file():
            continue
        arrays = _npz(path)
        strata.append((j, SharedUnits(arrays) if str(arrays["mode"]) == "shared" else OwnUnits(arrays)))
    return strata


def summarize_reliability(plan, jobs, *, stress_dataset=None) -> dict:
    reps, seed = plan["bootstrap"]["reps"], plan["bootstrap"]["seed"]
    horizons = [int(h) for h in plan["horizons"]]
    strata = reliability_strata(jobs)
    rows, details = [], {"coverage": {}, "observer": {}, "harm": {}, "stress": {}}
    for k, h in enumerate(horizons):
        statistic = "coverage_1" if h == 1 else f"sequence_{h}"
        interval = rate_interval([s for _, s in strata], statistic, reps=reps, seed=[seed, 1, k])
        details["coverage"][str(h)] = interval
        if h == 1:
            details["coverage"]["1_consequence_model"] = rate_interval([s for _, s in strata], "sequence_1",
                                                                       reps=reps, seed=[seed, 1, 99])
        if interval is None or interval["low"] is None:
            continue
        keys = {"diagnostic": "coverage", "level": str(h)}
        rows += [_row(keys, "coverage", interval["value"]), _row(keys, "coverage_ci_low", interval["low"]),
                 _row(keys, "coverage_ci_high", interval["high"])]
    observed = {}
    for j in jobs:
        path = j["directory"] / "observers.json"
        if path.is_file():
            observed.setdefault(j["entry"]["dataset"], []).append(_read(path))
    for level in OBSERVER_LEVELS:
        per_dataset = {}
        for dataset, results in observed.items():
            v = [r["variants"][level] for r in results]
            events = sum(x["nis_events"] for x in v)
            per_dataset[dataset] = {"mse": math.fsum(x["squared_error_sum"] for x in v) / sum(x["assessment_rows"] for x in v),
                                    "nis_coverage": sum(x["nis_in_region"] for x in v) / events if events else None,
                                    "runs": len(v)}
        details["observer"][level] = per_dataset
        if not per_dataset or any(p["nis_coverage"] is None for p in per_dataset.values()):
            continue
        keys = {"diagnostic": "observer", "level": level}
        rows += [_row(keys, "risk_rmse", math.sqrt(np.mean([p["mse"] for p in per_dataset.values()]))),
                 _row(keys, "nis_coverage", float(np.mean([p["nis_coverage"] for p in per_dataset.values()])))]
    for d_index, dataset in enumerate(DATASETS):
        chosen = [s for j, s in strata if j["entry"]["dataset"] == dataset]
        if not chosen:
            continue
        found = {name: rate_interval(chosen, name, reps=reps, seed=[seed, 2, d_index, i])
                 for i, name in enumerate(("false_safe", "harm_miss", "false_safe_point", "harm_miss_point"))}
        details["harm"][dataset] = found
        keys = {"diagnostic": "harm", "level": dataset}
        for name, prefix in (("false_safe", "false_safe"), ("harm_miss", "harm_miss")):
            r = found[name]
            if r is None or r["low"] is None:
                continue
            rows += [_row(keys, f"{prefix}_rate", r["value"]), _row(keys, f"{prefix}_ci_low", r["low"]),
                     _row(keys, f"{prefix}_ci_high", r["high"])]
    stress_dataset = stress_dataset or plan.get("stress_dataset")
    if stress_dataset is not None:
        units = []
        for j in jobs:
            if j["entry"]["dataset"] == stress_dataset and j["entry"]["stress"]:
                units += _read_csv(j["directory"] / "units.csv")
        if units:
            fallback = float(np.mean([1 - int(u["g_input"]) for u in units]))
            activity = float(np.mean([int(u["stress_active"]) for u in units]))
            keys = {"diagnostic": "stress", "level": "clean"}
            rows += [_row(keys, "input_fallback", fallback), _row(keys, "residual_activity", activity)]
            details["stress"] = {"dataset": stress_dataset, "units": len(units),
                                 "definitions": "negative_controls: units with evaluation rows; q_M behind g_input"}
    expected = {("coverage", str(h)) for h in horizons} | {("observer", v) for v in OBSERVER_LEVELS} | \
        {("stress", "clean")} | {("harm", j["entry"]["dataset"]) for j, _ in strata}
    present = {}
    for r in rows:
        present.setdefault((r["keys"]["diagnostic"], r["keys"]["level"]), set()).add(r["quantity"])
    quantities = {"coverage": 3, "observer": 2, "harm": 6, "stress": 2}
    complete = {d["dataset"] for d in plan["datasets"]} >= set(DATASETS) and \
        all(key in present and len(present[key]) == quantities[key[0]] for key in expected)
    return {"rows": rows, "details": details, "complete": complete}


SUMMARIZERS = {"action_gap_audit": summarize_action_gaps, "decision_traces": summarize_decision_traces,
               "reliability_diagnostics": summarize_reliability}
DEPOSITED = {
    "action_gap_audit": ("gaps.npz", "perturbation.json", "executions.csv", "parity.json", "audit.json",
                         "environment.json"),
    "decision_traces": ("decisions.json", "units.csv", "executions.csv", "parity.json", "audit.json",
                        "environment.json"),
    "reliability_diagnostics": ("reliability.npz", "calibration.json", "observers.json", "units.csv", "parity.json",
                                "audit.json", "environment.json"),
}


def write_audit_set(set_name, rows, document, files, provenance, *, root=None, replace=False, allow_partial=False,
                    complete=True, command=None, plan_text=None) -> Path:
    """records/<set> as scripts/records.py verifies it (cf. ensemble_grid.write_record_set, whose coverage table of
    cm_campaigns.contract.RECORD_SETS does not list these sets): summary.csv traced into raw/summary.json, the
    deposited job files under raw/, the audit code under code/, provenance.json and MANIFEST.json."""
    from scripts.export_records import fmt, git_head
    from scripts.record_mapping import _private_scan, install_verified_set
    from scripts.records import RECORDS, is_official_root, load_specs, verify_set, write_manifest
    specs = load_specs(RECORDS, RECORDS / "specs.json")
    if set_name not in specs["sets"]:
        raise RunnerError(f"{set_name} is not a record set of records/specs.json")
    spec = specs["sets"][set_name]
    root = Path(root or RECORDS).resolve()
    synthetic = provenance.get("synthetic") is True
    if is_official_root(root) and (synthetic or allow_partial or not complete):
        raise RunnerError("records/ holds complete measured sets only; write synthetic or partial sets to another root")
    if not complete and not allow_partial:
        raise RunnerError(f"{set_name}: the audit does not cover every required dataset and level; pass "
                          "--allow-partial to write the measured part to another root")
    for field in specs["provenance_fields"]:
        if not str(provenance.get(field) or "").strip():
            raise RunnerError(f"provenance field {field!r} is empty; pass it explicitly")
    if not rows:
        raise RunnerError(f"{set_name}: no measured value to export")
    keys, seen = list(spec["keys"]), set()
    for row in rows:
        if set(row["keys"]) != set(keys) or row["quantity"] not in spec["quantities"]:
            raise RunnerError(f"{set_name}: row {row['keys']}/{row['quantity']} does not fit the contract")
        identity = tuple(row["keys"][k] for k in keys) + (row["quantity"],)
        if identity in seen:
            raise RunnerError(f"{set_name}: duplicate row {identity}")
        seen.add(identity)
    if (root / set_name).exists() and not replace:
        raise RunnerError(f"{root / set_name} exists; pass --replace to rebuild it")
    root.mkdir(parents=True, exist_ok=True)
    staging_parent = ROOT.parent if root.is_relative_to(ROOT) else root.parent
    staging = Path(tempfile.mkdtemp(prefix=f"cm-record-{set_name}-", dir=staging_parent))
    try:
        target = staging / set_name
        (target / "raw").mkdir(parents=True)
        (target / "code").mkdir()
        copied = []
        for relative, source in sorted(files.items()):
            parts = Path(relative).parts
            if Path(relative).is_absolute() or ".." in parts or parts[0] != "raw" or relative == "raw/summary.json":
                raise RunnerError(f"deposited paths lie under raw/: {relative}")
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            if sha256(destination) != sha256(source):
                raise RunnerError(f"{source} changed while it was copied")
            copied.append({"path": relative, "sha256": sha256(destination)})
        (target / "raw" / "summary.json").write_text(json.dumps(json_safe(dict(document, rows=rows)), indent=1,
                                                                sort_keys=True, allow_nan=False) + "\n",
                                                     encoding="utf-8")
        for name in AUDIT_FILES + REUSED_FILES:
            if (ROOT / name).is_file():
                destination = target / "code" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, destination)
        if plan_text is not None:
            (target / "code" / "plan.json").write_text(plan_text, encoding="utf-8")
        (target / "code" / "SOURCE.json").write_text(json.dumps({
            "schema": "cross-morse-record-source-1", "exporter": "cm_experiments/decision_audit.py:export_decision_audit",
            "command": [Path(a).name if Path(a).is_absolute() else a for a in (command or [])],
            "runtime_source_sha256": runtime_source_sha256(), "audit_code_sha256": code_identity(),
            "repository_commit": git_head(), "deposited_files": copied}, indent=1, sort_keys=True) + "\n",
            encoding="utf-8")
        with (target / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(keys + list(SUMMARY_TAIL))
            for index, row in enumerate(rows):
                writer.writerow([row["keys"][k] for k in keys] + [row["quantity"], fmt(row["value"]), row["unit"],
                                                                  "raw/summary.json", f"json:/rows/{index}/value"])
        (target / "provenance.json").write_text(json.dumps(json_safe(dict(provenance, coverage_complete=bool(complete),
                                                                          synthetic=synthetic)),
                                                           indent=1, sort_keys=True, allow_nan=False) + "\n",
                                                encoding="utf-8")
        private = _private_scan(target)
        if private:
            raise RunnerError("redact private infrastructure identifiers before depositing:\n  " + "\n  ".join(private[:10]))
        write_manifest(target)
        verify_set(target, spec, specs)
        return install_verified_set(staging, target, root, specs, replace=replace)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _hardware(jobs):
    seen = set()
    for j in jobs:
        env = _read(j["directory"] / "environment.json")
        seen.add(f"{env.get('gpu') or (env.get('machine', '?') + ' CPU')}; torch {env.get('torch', '?')}; "
                 f"python {env.get('python', '?')}")
    if not seen:
        raise RunnerError("the jobs carry no environment record; pass --hardware")
    return " | ".join(sorted(seen))


def _period(jobs):
    starts = [j["audit"]["started_utc"] for j in jobs]
    ends = [j["audit"]["finished_utc"] for j in jobs]
    if not starts:
        raise RunnerError("the jobs carry no execution timestamps; pass --period")
    return f"{min(starts)}/{max(ends)}"


def export_decision_audit(output, *, sets=SETS, root=None, replace=False, allow_partial=False, hardware=None,
                          period=None, command=None, stress_dataset=None) -> list[Path]:
    output = Path(output).resolve()
    plan = load_plan(output, PLAN_KIND)
    jobs = collect_jobs(output, plan, allow_partial=allow_partial)
    if not jobs:
        raise RunnerError("no completed job to export")
    synthetic = {j["entry"]["synthetic"] for j in jobs}
    if len(synthetic) != 1:
        raise RunnerError("synthetic and benchmark parents cannot share one record set")
    synthetic = synthetic.pop() or any(j["entry"]["fixture_override"] for j in jobs)
    written = []
    for name in sets:
        if name not in SETS:
            raise RunnerError(f"unknown record set {name!r}; expected one of {SETS}")
        summary = (summarize_reliability(plan, jobs, stress_dataset=stress_dataset) if name == "reliability_diagnostics"
                   else SUMMARIZERS[name](plan, jobs))
        files, inputs = {}, []
        for j in jobs:
            base = f"raw/{j['entry']['slug']}/{slug(j['entry']['member'])}"
            for file in DEPOSITED[name]:
                path = j["directory"] / file
                if path.is_file():
                    files[f"{base}/{file}"] = path
                    inputs.append({"file": f"{base}/{file}", "sha256": sha256(path)})
        document = {"schema": SUMMARY_SCHEMA, "record_set": name, "definitions": DEFINITIONS[name],
                    "details": summary["details"], "inputs": inputs, "plan_sha256": sha256(output / "plan.json"),
                    "recomputed_by": f"cm_experiments/decision_audit.py:{SUMMARIZERS[name].__name__}",
                    "perturbation": plan["perturbation"], "near_zero_threshold_nats": plan["near_zero_threshold_nats"],
                    "bootstrap": plan["bootstrap"], "tolerance_nats": plan["tolerance_nats"]}
        provenance = {
            "experiment": {"action_gap_audit": "Action-gap and switching audit of the deployed rules (Table S21, "
                                               "Fig. S17f)",
                           "decision_traces": "Executed decision origins and within-depth action entropy (main Fig. "
                                              "3c-d)",
                           "reliability_diagnostics": "Reliability, harm and fallback diagnostics (Fig. S16; E7, E8)"}[name],
            "produced_by": "python -B scripts/run_decision_audit.py plan/run/export (cm_experiments/decision_audit.py)",
            "hardware": hardware or _hardware(jobs), "period": period or _period(jobs),
            "protocol": ("instrumented serving-time execution of the deployed rule of each sealed run on the "
                         f"{plan['role']} role with the released runtime (recording sensor hook only); decision log "
                         "replayed and checked site by site; registered perturbation: one draw of the observer's "
                         "measurement-noise model per site; see raw/summary.json for every definition"),
            "role": plan["role"], "plan_sha256": sha256(output / "plan.json"),
            "parents": [j["entry"]["identity"] for j in jobs], "synthetic": bool(synthetic),
            "fixture_override": sorted({json.dumps(j["entry"]["fixture_override"], sort_keys=True)
                                        for j in jobs if j["entry"]["fixture_override"]})}
        written.append(write_audit_set(name, summary["rows"], document, files, provenance, root=root, replace=replace,
                                       allow_partial=allow_partial, complete=summary["complete"], command=command,
                                       plan_text=public_plan(output)))
    return written
