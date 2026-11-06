"""Work accounting: Algorithm S3 and Lemma 3, the Note S19 cost model and the Table S44 tail model.

Costs are in units of one dense update N d^2 (Note S19) or of one nominal
execution (Tables S23 and S44), as each function states. The budget-aware rule
is verified against the executable implementation used by experiment E6,
``cm_experiments.budget.run_budget_episode``, and against the independent
restatement :func:`algorithm_s3_reference` below.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

COST_FIELDS = ("initialization", "cheap_measurement", "score", "acquisition", "action", "prediction", "comparator")


# ---------------------------------------------------------------------------- Note S19
def sparse_to_dense_ratio(edges: float, nodes: float, width: int) -> float:
    """r = E / (N d): one sparse product costs E d multiply-adds, one dense update N d^2."""
    if min(edges, nodes, width) <= 0:
        raise ValueError("edges, nodes and width must be positive")
    return float(edges) / (float(nodes) * float(width))


def sensing_share(r: float, k_over_d: float = 1.0, period: int = 1) -> float:
    """Degree-8 sensing as a share of one nominal step (1 + r) N d^2: 8 r (k/d) / (P_s (1 + r))."""
    if r <= 0 or not 0 < k_over_d <= 1 or period < 1:
        raise ValueError("need r > 0, 0 < k/d <= 1 and a positive sensing period")
    return 8.0 * r * k_over_d / (period * (1.0 + r))


def per_member_multiplier(r: float, s: float, f_res: float, order: int) -> float:
    """m(T) = (1 + r + s + f_res r (T - 1)) / (1 + r), per controlled depth relative to the nominal step."""
    if r <= 0 or s < 0 or not 0 <= f_res <= 1 or order < 1:
        raise ValueError("need r > 0, s >= 0, 0 <= f_res <= 1 and T >= 1")
    return (1.0 + r + s + f_res * r * (order - 1)) / (1.0 + r)


def admissible_truncation_order(m: float, r: float, s: float, f_res: float) -> float:
    """Largest real T with m(T) <= m: T <= 1 + [m (1 + r) - (1 + r + s)] / (f_res r)."""
    if r <= 0 or f_res <= 0:
        raise ValueError("need r > 0 and f_res > 0")
    return 1.0 + (m * (1.0 + r) - (1.0 + r + s)) / (f_res * r)


def max_resolvent_fraction(m: float, r: float, s: float, order: int) -> float:
    """Largest executed resolvent fraction compatible with a measured multiplier at order T."""
    if r <= 0 or order < 2:
        raise ValueError("need r > 0 and T >= 2")
    return (m * (1.0 + r) - (1.0 + r + s)) / (r * (order - 1))


def admissible_sketch_ratio(m: float, r: float) -> float:
    """Largest (k/d)/P_s whose sensing share fits the controller overhead: min(1, (m-1)(1+r)/(8r))."""
    if r <= 0:
        raise ValueError("need r > 0")
    return float(min(1.0, (m - 1.0) * (1.0 + r) / (8.0 * r)))


def sketch_relative_sd_bound(k: int) -> float:
    """Note S19: a Gaussian sketch of width k estimates a band power with relative SD at most sqrt(2/k)."""
    if k < 1:
        raise ValueError("sketch width must be positive")
    return math.sqrt(2.0 / k)


# ---------------------------------------------------------------------------- Table S44
def resolvent_increment(r: float, order: int = 8) -> float:
    """v = r (T - 1) / (1 + r): extra cost of one resolvent step over a nominal step."""
    return r * (order - 1) / (1.0 + r)


def fixed_overhead(o_ctrl: float, f_bar: float, v: float) -> float:
    """o_fix = o_ctrl - f_bar v, so that the expected controller cost is C_qC (1 + o_ctrl)."""
    return o_ctrl - f_bar * v


def tail_cost(C_qC: float, o_fix: float, f_res: float, v: float) -> float:
    """C_ctrl(x) = C_qC [1 + o_fix + f_res(x) v] at a per-input resolvent share f_res(x)."""
    return C_qC * (1.0 + o_fix + f_res * v)


def tail_scenarios(C_qC: float, o_ctrl: float, f_bar: float, r: float, order: int = 8,
                   shifts: Sequence[float] = (0.10, 0.20, 0.35)) -> dict:
    """The low, central and high 99th-percentile scenarios f99 = f_bar + shift of Table S44."""
    v = resolvent_increment(r, order)
    o_fix = fixed_overhead(o_ctrl, f_bar, v)
    f99 = [f_bar + shift for shift in shifts]
    costs = [tail_cost(C_qC, o_fix, f, v) for f in f99]
    return {"v": v, "o_fix": o_fix, "f99": f99, "C_099": costs,
            "overhead_pct": [100.0 * (c / C_qC - 1.0) for c in costs]}


# ---------------------------------------------------------------------------- Algorithm S3
@dataclass(frozen=True)
class Episode:
    """One input for the budget-aware rule: declared bounds, observed work and decision events.

    ``supported[d]`` is the supported set at depth d (possibly empty),
    ``certified[d]`` whether the selective-sensing certificate holds there, and
    ``actual`` the observed work of each operation as a fraction of its declared bound.
    """
    budget: float
    costs: dict
    depth: int
    supported: tuple
    certified: tuple
    actual: dict


def random_episode(rng: np.random.Generator, depth: int | None = None) -> Episode:
    depth = int(rng.integers(1, 7)) if depth is None else depth
    comparator = float(rng.uniform(1.0, 4.0))
    costs = {"initialization": float(rng.uniform(0.0, 0.6)), "cheap_measurement": float(rng.uniform(0.0, 0.2)),
             "score": float(rng.uniform(0.0, 0.2)), "acquisition": float(rng.uniform(0.0, 1.0)),
             "action": float(rng.uniform(0.05, 1.2)), "prediction": float(rng.uniform(0.0, 1.0)) * comparator,
             "comparator": comparator}
    # Budgets from infeasible through tight to generous, with exact boundary cases.
    kind = rng.integers(5)
    if kind == 0:
        budget = float(rng.uniform(0.0, comparator))
    elif kind == 1:
        budget = comparator + costs["initialization"] * float(rng.uniform(0.0, 1.0))
    elif kind == 2:
        budget = comparator + costs["initialization"]
    else:
        budget = comparator + float(rng.uniform(0.0, 1.0 + 2.0 * depth))
    supported = tuple(tuple(int(a) for a in np.flatnonzero(rng.random(8) < 0.5)) for _ in range(depth))
    certified = tuple(bool(rng.random() < 0.4) for _ in range(depth))
    actual = {name: float(rng.choice([1.0, rng.uniform(0.0, 1.0)])) for name in COST_FIELDS}
    return Episode(budget, costs, depth, supported, certified, actual)


class _Backend:
    """Synthetic paid backend: every callback reports work no larger than its declared bound."""

    def __init__(self, episode: Episode):
        self.e = episode

    def _work(self, name):
        return self.e.costs[name] * self.e.actual[name]

    def initialize(self):
        return None, self._work("initialization")

    def measure(self, d):
        return None, self._work("cheap_measurement")

    def score(self, d):
        supported = list(self.e.supported[d])
        best = supported[0] if supported else 8
        return {"supported": supported, "action": best, "recovery": 8, "gap": 1.0}, self._work("score")

    def acquire(self, d):
        supported = list(self.e.supported[d])
        return (supported[-1] if supported else 8), self._work("acquisition")

    def act(self, action, d):
        return None, self._work("action")

    def predict(self):
        return "controller", self._work("prediction")

    def restart(self):
        return "comparator", self._work("comparator")


def run_runtime_episode(episode: Episode) -> dict:
    """Execute ``cm_experiments.budget.run_budget_episode`` on a synthetic backend."""
    from cm_experiments.budget import Certificate, run_budget_episode

    def certificate(d, score):
        holds = episode.certified[d]
        return Certificate(gap=1.0, lipschitz_bound=0.0, radius=0.0, floor=0.0 if holds else 2.0,
                           support_invariant=True, recovery_invariant=True, search_invariant=True,
                           bound_source="synthetic")

    prediction, trace = run_budget_episode(_Backend(episode), depth=episode.depth, budget=episode.budget,
                                           costs=dict(episode.costs), certificate=certificate)
    rows = trace["operations"]
    total = float(rows[-1]["cumulative_work"]) if rows else 0.0
    return {"status": trace["status"], "prediction": prediction, "total": total, "rows": rows,
            "operations": [row["operation"] for row in rows],
            "action_ids": [row["action_id"] for row in rows if row["operation"] == "action"],
            "decisions": {key: trace[key] for key in
                          ("acquisitions", "certified_skips", "unaffordable_acquisitions", "recoveries")}}


def algorithm_s3_reference(episode: Episode) -> dict:
    """Independent restatement of Supplementary Algorithm S3 with the runtime's cost schema.

    ``acquisition`` includes the rescore and ``action`` the largest supported
    action plus the dense update, as the implementation declares them. Returns
    the status, total work, paid-operation order, executed action IDs and decision
    counts. Action-specific checks matter even when different actions have the
    same declared work. This reference covers the synthetic backend above; it
    does not construct valid certificates for learned controllers.
    """
    c, W = episode.costs, episode.budget
    reserve = c["comparator"]
    work = lambda name: c[name] * episode.actual[name]  # noqa: E731 - observed work of one operation
    ledger: list[float] = []
    operations, actions = [], []
    decisions = {"acquisitions": 0, "certified_skips": 0, "unaffordable_acquisitions": 0, "recoveries": 0}

    def result(status, total, prediction):
        return {"status": status, "prediction": prediction, "total": total, "ledger": ledger,
                "operations": operations, "action_ids": actions, "decisions": decisions}

    if W < c["comparator"]:
        operations.append("none")
        return result("INFEASIBLE", 0.0, None)
    if c["initialization"] + reserve > W + 1e-10:
        operations.append("comparator_restart")
        return result("STATIC_FALLBACK", work("comparator"), "comparator")
    w = work("initialization")
    operations.append("initialization")
    ledger.append(w)
    for d in range(episode.depth):
        if w + c["cheap_measurement"] + c["score"] + reserve > W + 1e-10:
            operations.append("comparator_restart")
            return result("LATE_FALLBACK", w + work("comparator"), "comparator")
        w += work("cheap_measurement")
        operations.append("cheap_measurement")
        ledger.append(w)
        w += work("score")
        operations.append("score")
        ledger.append(w)
        action = 8  # the synthetic backend's registered recovery
        if not episode.supported[d]:
            decisions["recoveries"] += 1
        elif episode.certified[d]:
            action = episode.supported[d][0]
            decisions["certified_skips"] += 1
        elif w + c["acquisition"] + c["action"] + reserve <= W + 1e-10:
            w += work("acquisition")
            operations.append("acquisition")
            ledger.append(w)
            action = episode.supported[d][-1]
            decisions["acquisitions"] += 1
        else:
            decisions["unaffordable_acquisitions"] += 1
            decisions["recoveries"] += 1
        if w + c["action"] + reserve > W + 1e-10:
            operations.append("comparator_restart")
            return result("LATE_FALLBACK", w + work("comparator"), "comparator")
        w += work("action")
        operations.append("action")
        actions.append(action)
        ledger.append(w)
    operations.append("prediction")
    return result("COMPLETED", w + work("prediction"), "controller")


def lemma3_invariants(episode: Episode, result: dict) -> dict:
    """The claims of Lemma 3 for one executed episode."""
    W, reserve = episode.budget, episode.costs["comparator"]
    status, total = result["status"], result["total"]
    ledger = result.get("ledger")
    if ledger is None:  # runtime trace: cumulative work after each non-terminal paid operation
        ledger = [float(row["cumulative_work"]) for row in result["rows"]
                  if row.get("operation") not in ("prediction", "comparator_restart", "none")]
    invariant = all(w + reserve <= W + 1e-9 for w in ledger)
    if status == "INFEASIBLE":
        holds = total == 0.0 and result.get("prediction") is None and W < reserve
    else:
        holds = total <= W + 1e-9 and invariant
    return {"holds": bool(holds), "invariant": bool(invariant), "status": status, "total": total, "budget": W}


def reserve_released_counterexample() -> dict:
    """Supplementary Figure S11: releasing the reserve before the final site can exceed W.

    W = 3, C_qC = W_C = 2: a rule that spends the reserve on one more action
    (w = 2.5) and then aborts must pay the restart, 2.5 + 2 = 4.5 > W.
    """
    W, reserve, spent = 3.0, 2.0, 2.5
    return {"budget": W, "work_with_released_reserve": spent + reserve, "exceeds": spent + reserve > W}
