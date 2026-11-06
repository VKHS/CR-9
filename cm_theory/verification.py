"""Numerical verification battery for every statement of :mod:`cm_theory.statements`.

Each check draws randomized instances of a statement's hypotheses from a seeded
generator and tests its conclusion; identities are tested to floating-point
precision and inequalities with a declared slack. Monte Carlo coverage checks
compare the observed miss rate with the nominal level plus three binomial
standard errors. ``run_battery(quick=True)`` takes seconds and runs inside
``make verify``; ``quick=False`` enlarges every sample.

A battery that passes shows that the implementations, the printed constants and
the stated hypotheses agree on the instances drawn. It does not replace the
proofs of the supplementary notes and makes no claim about predictive benefit.
"""
from __future__ import annotations

import math
import time
from typing import Callable
import zlib

import numpy as np

from . import admission as adm
from . import budget, contract, control, economics, inference, selection, spectral, stability
from . import operators as op
from ._util import Check, fro, graph_family, laplacian_eigensystem, norm_inf2, passed, random_graph, random_walk
from .statements import STATEMENTS

CheckFunction = Callable[[np.random.Generator, bool], list]
CHECKS: dict[str, CheckFunction] = {}
CHECK_STATEMENT: dict[str, str] = {}
DEFAULT_SEED = 20261104
TOL = 1e-12


def _register(statement_key: str):
    def wrap(function: CheckFunction) -> CheckFunction:
        CHECKS[function.__name__] = function
        CHECK_STATEMENT[function.__name__] = statement_key
        return function
    return wrap


def _rng(seed: int, name: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(name.encode())])


def _mc_limit(level: float, reps: int) -> float:
    return level + 3.0 * math.sqrt(level * (1.0 - level) / reps)


def _row_stochastic(rng: np.random.Generator, n: int, sparse: bool = True) -> np.ndarray:
    m = rng.random((n, n))
    if sparse:
        m *= rng.random((n, n)) < 0.4
        m[np.arange(n), rng.integers(0, n, n)] += 0.1
    return m / m.sum(axis=1, keepdims=True)


def _transport(registry=None):
    return op.transport_actions(registry)


# ============================================================================ Definition 1
@_register("definition-1")
def definition1_nominal_identity(rng, quick):
    graphs = graph_family(rng, 8 if quick else 30)
    a0 = op.load_registry()[0]
    worst = max(float(np.max(np.abs(op.transport_matrix(random_walk(w), a0) - op.nominal_shared_matrix(random_walk(w)))))
                for w in graphs)
    native = (a0.rho, a0.nu, a0.alpha) == (0.0, 0.70, 0.25)
    return [Check("Definition 1", "registry A0 equals the native shared aggregation 0.3 I + 0.7 P", passed(worst <= TOL),
                  worst, TOL, len(graphs)),
            Check("Definition 1", "native (rho, nu, alpha) of A0 is (0, 0.70, 0.25) (Table S5b)", passed(native),
                  (a0.rho, a0.nu, a0.alpha), (0.0, 0.70, 0.25))]


@_register("definition-1")
def definition1_inserted_sites(rng, quick):
    actions = _transport()
    bypass, consensus, count = 0.0, 0.0, 0
    for w in graph_family(rng, 6 if quick else 20):
        P = random_walk(w)
        n = len(P)
        H, H0 = rng.normal(size=(n, 5)), rng.normal(size=(n, 5))
        bypass = max(bypass, float(np.max(np.abs(op.inserted_site(H, H0, P, actions[0]) - H))))
        common = np.tile(rng.normal(size=(1, 5)), (n, 1))
        for a in actions[1:]:
            consensus = max(consensus, float(np.max(np.abs(op.inserted_site(common, common, P, a) - common))))
            count += 1
    return [Check("Definition 1", "A0 at an inserted site is a literal bypass", passed(bypass == 0.0), bypass, 0.0),
            Check("Definition 1", "every inserted-site map V_a preserves a common node state (affine weights sum to one)",
                  passed(consensus <= 1e-12), consensus, 1e-12, count)]


# ============================================================================ Lemma 1
@_register("lemma-1")
def lemma1_random_mixtures(rng, quick):
    trials, ok = (200 if quick else 2000), 0
    for _ in range(trials):
        n, m = int(rng.integers(2, 12)), int(rng.integers(1, 5))
        kernels = [_row_stochastic(rng, n, bool(rng.integers(2))) for _ in range(m)]
        w = rng.dirichlet(np.ones(m) * float(rng.uniform(0.2, 2.0)))
        nu = float(rng.choice([0.0, 1.0, rng.uniform()]))
        ok += op.is_row_stochastic(op.mixture_transport(kernels, w, nu))
    return [Check("Lemma 1", "convex kernel mixtures with lazy mass are nonnegative and row stochastic",
                  passed(ok == trials), ok, trials, trials)]


@_register("lemma-1")
def lemma1_registered_polynomials(rng, quick):
    actions = _transport()
    coefficients = [op.operator_polynomial(a) for a in actions]
    nonneg = min(float(c.min()) for c in coefficients)
    unit = max(abs(float(c.sum()) - 1.0) for c in coefficients)
    stochastic = all(op.is_row_stochastic(op.transport_matrix(random_walk(w), a))
                     for w in graph_family(rng, 6 if quick else 25) for a in actions)
    return [Check("Lemma 1", "every registered operator polynomial has nonnegative coefficients summing to one",
                  passed(nonneg >= 0 and unit <= 1e-14), {"min_coefficient": nonneg, "row_sum_error": unit}, 1e-14, len(actions)),
            Check("Lemma 1", "every registered transport matrix is nonnegative and row stochastic on random graphs",
                  passed(stochastic), stochastic, True, len(actions))]


# ============================================================================ Lemma 2
@_register("lemma-2")
def lemma2_resolvent_bound(rng, quick):
    worst_ratio, worst_excess, worst_identity, valid, count = 0.0, -math.inf, 0.0, True, 0
    etas = [0.08, 0.2, 0.5, 1.0] + list(rng.uniform(0.01, 1.0, 2 if quick else 8))
    for w in graph_family(rng, 5 if quick else 20):
        P = random_walk(w)
        for eta in etas:
            for order in (0, 1, 2, 4, 8, 12):
                r = op.resolvent_truncation_error(P, float(eta), order)
                valid &= op.is_row_stochastic(op.exact_resolvent(P, eta), 1e-10) and \
                    op.is_row_stochastic(op.truncated_resolvent(P, eta, order), 1e-12)
                worst_excess = max(worst_excess, r["error"] - r["bound"])
                if r["bound"] > 1e-10:  # below this the comparison is floating-point roundoff
                    worst_ratio = max(worst_ratio, r["error"] / r["bound"])
                worst_identity = max(worst_identity, r["identity_residual"])
                count += 1
    registered = {f"eta={eta}": round(2 * (1 - eta) ** 9, 4) for eta in (0.5, 0.2, 0.08)}
    return [Check("Lemma 2", "K_eta and K~_eta,T are nonnegative and row stochastic", passed(valid), valid, True, count),
            Check("Lemma 2", "||K_eta - K~_eta,T||_inf <= 2 (1-eta)^(T+1)", passed(worst_excess <= 1e-13 and worst_ratio <= 1 + 1e-10),
                  worst_ratio, 1.0, count, f"largest error/bound ratio (bounds above 1e-10); largest excess {worst_excess:.1e}"),
            Check("Lemma 2", "K_eta - K~_eta,T = (1-eta)^(T+1) (P^(T+1) K_eta - I) exactly", passed(worst_identity <= 1e-10),
                  worst_identity, 1e-10, count),
            Check("Lemma 2", "registered truncation bounds 2(1-eta)^9 at T = 8", "INFO", registered, None, 3)]


# ============================================================================ Eq. (10)
@_register("eq-10")
def eq10_registry_moments(rng, quick):
    actions = _transport()
    range_err = max(abs(op.executed_range(a) - op.executed_range_closed_form(a)) for a in actions)
    ident_err = max(abs(op.identity_coefficient(a) - op.identity_coefficient_closed_form(a)) for a in actions)
    reach = {a.id: op.maximum_reach(a) for a in actions}
    reach_ok = all(reach[a.id] == (a.order if a.is_resolvent else 1) for a in actions)
    a0, a7 = actions[0], actions[7]
    sentence = (round(op.executed_range(a0), 2), round(op.identity_coefficient(a0), 2),
                round(op.executed_range(a7), 2), round(op.identity_coefficient(a7), 2))
    return [Check("Eq. (10)", "executed range equals the first moment of the operator polynomial", passed(range_err <= 1e-14),
                  range_err, 1e-14, len(actions)),
            Check("Eq. (10)", "identity coefficient equals the P^0 coefficient", passed(ident_err <= 1e-14), ident_err, 1e-14,
                  len(actions)),
            Check("Eq. (10)", "maximum reach is T for resolvent actions and one otherwise", passed(reach_ok), reach, None,
                  len(actions)),
            Check("Eq. (10)", "main text: A0 moves 0.70 hops with identity 0.30; A7 moves 0.60 hops with identity 0.57",
                  passed(sentence == (0.70, 0.30, 0.60, 0.57)), sentence, (0.70, 0.30, 0.60, 0.57))]


# ============================================================================ Proposition 1
@_register("proposition-1")
def proposition1_replay(rng, quick):
    actions = _transport()
    trials, identical, switching = (10 if quick else 60), 0, 0
    for _ in range(trials):
        n, k = int(rng.integers(6, 16)), int(rng.integers(3, 7))
        P = random_walk(random_graph(n, rng, density=float(rng.uniform(0.2, 0.6))))
        H0 = rng.normal(size=(n, k))
        W, b, head = rng.normal(size=(k, k)) / math.sqrt(k), rng.normal(size=k) * 0.1, rng.normal(size=(k, 3))
        weights = rng.normal(size=(len(actions), 3))
        depth = int(rng.integers(4, 12))
        closed = control.closed_loop(P, H0, W, b, head, actions, weights, depth)
        replay = control.open_loop(P, H0, W, b, head, actions, closed["actions"])
        identical += all(np.array_equal(x, y) for x, y in zip(closed["states"], replay["states"])) and \
            np.array_equal(closed["logits"], replay["logits"])
        switching += len(set(closed["actions"])) > 1
    return [Check("Proposition 1", "open-loop replay of the realized closed-loop sequence is bitwise identical",
                  passed(identical == trials), identical, trials, trials),
            Check("Proposition 1", "instances in which the feedback rule switched actions", "INFO", switching, None, trials)]


# ============================================================================ Proposition 2
def _tube_instance(rng):
    actions = _transport()
    n, k = int(rng.integers(6, 14)), int(rng.integers(3, 6))
    w = random_graph(n, rng, density=0.4, weighted=True)
    noise = np.triu(rng.uniform(0, 0.05, size=w.shape) * (w > 0), 1)
    w_tilde = w + noise + noise.T
    P, Pt = random_walk(w), random_walk(w_tilde)
    depth = int(rng.integers(3, 9))
    chosen = [actions[int(i)] for i in rng.integers(0, len(actions), depth)]
    W = rng.normal(size=(k, k)) / math.sqrt(k) * float(rng.uniform(0.6, 1.3))
    b = rng.normal(size=k) * 0.1
    H0 = rng.normal(size=(n, k))
    H0t = H0 + rng.normal(size=(n, k)) * 1e-2
    tube = stability.trajectory_tube([op.transport_matrix(P, a) for a in chosen], [op.transport_matrix(Pt, a) for a in chosen],
                                     H0, H0t, W, b, [a.alpha for a in chosen])
    return tube, k


@_register("proposition-2")
def proposition2_tube(rng, quick):
    trials, step_ok, unrolled_ok, recursion_err = (20 if quick else 150), 0, 0, 0.0
    for _ in range(trials):
        tube, _ = _tube_instance(rng)
        e, rhs = tube["errors"], tube["one_step_rhs"]
        step_ok += bool(np.all(e[1:] <= rhs * (1 + 1e-12) + 1e-15))
        unrolled_ok += bool(e[-1] <= tube["unrolled"] * (1 + 1e-12) + 1e-15)
        recursion_err = max(recursion_err, abs(tube["recursion"][-1] - tube["unrolled"]) / max(1.0, tube["unrolled"]))
    return [Check("Proposition 2", "every realized step satisfies e_{l+1} <= beta_l [e_l + M_l eps_l] + R_l e_ref",
                  passed(step_ok == trials), step_ok, trials, trials),
            Check("Proposition 2", "the unrolled product-sum bound holds at the final depth", passed(unrolled_ok == trials),
                  unrolled_ok, trials, trials),
            Check("Proposition 2", "iterated recursion equals the product-sum form", passed(recursion_err <= 1e-12),
                  recursion_err, 1e-12, trials)]


@_register("proposition-2")
def proposition2_logit_bound(rng, quick):
    trials, ok, worst = (20 if quick else 150), 0, 0.0
    gradient = 0.0
    for _ in range(trials):
        tube, k = _tube_instance(rng)
        H, Ht = tube["final_states"]
        head = rng.normal(size=(k, 4))
        labels = rng.integers(0, 4, len(H))
        change = np.abs(stability.cross_entropy(H @ head, labels) - stability.cross_entropy(Ht @ head, labels))
        bound = stability.logit_loss_bound(head, norm_inf2(H - Ht))
        ok += bool(change.max() <= bound * (1 + 1e-12) + 1e-15)
        worst = max(worst, float(change.max() / bound) if bound > 0 else 0.0)
        logits = rng.normal(size=(50, 4)) * 5
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        y = rng.integers(0, 4, 50)
        p[np.arange(50), y] -= 1.0
        gradient = max(gradient, float(np.max(np.linalg.norm(p, axis=1))))
    return [Check("Proposition 2", "frozen-head CE change per node <= sqrt(2) ||W_head||_2 e_L", passed(ok == trials),
                  ok, trials, trials, f"largest change/bound ratio {worst:.3f}"),
            Check("Proposition 2", "||grad_z CE||_2 <= sqrt(2)", passed(gradient <= math.sqrt(2) + 1e-12), gradient,
                  math.sqrt(2), trials * 50)]


# ============================================================================ Proposition 3
def _linear_scores(rng, actions: int, dim: int):
    c = rng.normal(size=actions) * 0.05
    g = rng.normal(size=(actions, dim)) * 0.02
    return c, g, float(np.max(np.linalg.norm(g, axis=1)))


@_register("proposition-3")
def proposition3_gap_stability(rng, quick):
    trials, ok = (2000 if quick else 20000), 0
    for _ in range(trials):
        c, g, L = _linear_scores(rng, int(rng.integers(2, 10)), 4)
        s = rng.normal(size=4)
        scores = c + g @ s
        gap = stability.score_gap(scores)
        direction = rng.normal(size=4)
        distance = float(rng.uniform(0, 1)) * gap / (2 * L)
        s_tilde = s + distance * direction / np.linalg.norm(direction)
        ok += int(np.argmin(c + g @ s_tilde) == np.argmin(scores)) if stability.stable_by_gap(gap, L, distance) else 1
    # Two scores with slopes +1 and -1: gap m at s = 0, the minimizer flips just beyond m/2 = m/(2L).
    m, eps = 0.01, 1e-6
    flips = (0.0 + (m / 2 + eps)) > (m - (m / 2 + eps))
    return [Check("Proposition 3", "the minimizer is unchanged whenever 2 L_J ||s - s~|| < m(s)", passed(ok == trials), ok,
                  trials, trials),
            Check("Proposition 3", "the constant 2 is attained: a perturbation of m/(2L) + eps flips the minimizer",
                  passed(flips), flips, True)]


# ============================================================================ Theorem 1 and Corollary 1
def _family(rng, size: int):
    feedback = rng.random(size) < 0.6
    feedback[0] = True
    feasible = rng.random(size) < 0.9
    feasible[0] = True
    truth = rng.uniform(-0.012, 0.004, size)
    corr = 0.5
    sd = rng.uniform(0.0006, 0.0018, size)
    cov = (corr * np.ones((size, size)) + (1 - corr) * np.eye(size)) * np.outer(sd, sd)
    complexity = [(int(i),) for i in rng.permutation(size)]
    return feedback, feasible, truth, cov, complexity


@_register("theorem-1")
def theorem1_selection(rng, quick):
    configs, reps = (6 if quick else 25), (300 if quick else 2000)
    eps_r, eps_eq, delta = 0.003, 0.002, 0.025
    covered_draws, violations, coverage = 0, 0, []
    for config in range(configs):
        size = int(rng.integers(4, 12))
        feedback, feasible, truth, cov, complexity = _family(rng, size)
        radii = selection.max_t_radii(cov, delta, rng, draws=20000 if quick else 60000)
        candidate_scope = "feedback_only" if config % 2 == 0 else "all_feasible"
        hits = 0
        for _ in range(reps):
            estimate = rng.multivariate_normal(truth, cov)
            chosen = selection.eq19_select(estimate + radii, feasible, feedback, complexity, eps_r, eps_eq,
                                           candidate_scope=candidate_scope)
            family = np.zeros(size, dtype=bool)
            family[chosen["effective_candidates"]] = True
            if not selection.covers(estimate[family], truth[family], radii[family]):
                continue
            hits += 1
            covered_draws += 1
            for member in chosen["Q_s"]:  # the confirmation map may return any member of Q_s
                result = selection.theorem1_guarantee(truth, member, radii, family, eps_r, eps_eq)
                violations += not result["holds"]
        coverage.append(hits / reps)
    return [Check("Theorem 1", "on the coverage event every member of Q_s improves by eps_R and is near-best within "
                  "eps_eq + 2 r_max in the declared candidate family", passed(violations == 0), violations, 0, covered_draws,
                  "feedback_only and all_feasible reference scopes; coverage and guarantee use the returned effective family"),
            Check("Theorem 1", "max-t radii cover the family simultaneously (nominal 0.975)",
                  passed(min(coverage) >= 1 - _mc_limit(delta, reps)), min(coverage), 1 - delta, configs * reps,
                  "lowest per-configuration coverage on the declared effective family; reference scopes alternate")]


@_register("corollary-1")
def corollary1_lifecycle(rng, quick):
    reps, J = (1500 if quick else 12000), 1200
    eps_r, eps_eq, eps_a, delta_s, delta_a, B = 0.003, 0.002, 0.003, 0.025, 0.025, 0.10
    failures, deployed, selected = 0, 0, 0
    size = 6
    feedback, feasible, truth, cov, complexity = _family(rng, size)
    truth[:3] = [-0.0062, -0.0048, -0.0038]
    radii = selection.max_t_radii(cov, delta_s, rng, draws=20000)
    candidate_scope = "feedback_only"
    for trial in range(reps):
        estimate = rng.multivariate_normal(truth, cov)
        chosen = selection.eq19_select(estimate + radii, feasible, feedback, complexity, eps_r, eps_eq,
                                       candidate_scope=candidate_scope)
        choice = chosen["selected"]
        if choice is None:
            continue
        selected += 1
        mean = truth[choice]
        # Exercise both successful low-variance admission and high-variance rejection.
        # Both laws have exactly the selected population mean and stay in [-B, B].
        if trial % 2:
            units = mean + rng.uniform(-0.001, 0.001, J)
        else:
            p = (mean + B) / (2 * B)
            units = np.where(rng.random(J) < p, B, -B)
        upper = adm.admission_upper(float(units.mean()), float(units.var(ddof=1)), J, B, delta_a)
        if upper > -eps_a:
            continue
        deployed += 1
        family = np.zeros(size, dtype=bool)
        family[chosen["effective_candidates"]] = True
        clause = selection.theorem1_guarantee(truth, choice, radii, family, eps_r, eps_eq)["holds"]
        failures += not (clause and mean <= -eps_a)
    limit = _mc_limit(delta_s + delta_a, reps)
    return [Check("Corollary 1", "deployed and the joint selection-admission statement false: rate <= delta_s + delta_a",
                  passed(failures / reps <= limit and 0 < deployed < selected), failures / reps, delta_s + delta_a, reps,
                  f"{deployed} deployed, {selected - deployed} rejected at admission, "
                  f"{reps - selected} returned the comparator at selection; nonzero deployment and rejection required; "
                  f"candidate_scope={candidate_scope}, guarantee uses the returned effective family")]


@_register("theorem-1")
def theorem1_runtime_parity(rng, quick):
    """Check the actual nested selector's terminal rule, not its coverage.

    The full runtime reconstructs its comparator and computes intervals. Only
    then are its observed intervals, joint feasibility and fixed tie order fed
    to the independent reference. Open-loop comparisons remain diagnostics.
    """
    claim = "runtime nested selection agrees with the explicit feedback_only reference on its terminal Eq. (19) rule"
    try:
        from cross_morse.statistics import nested_selection
    except Exception as exc:
        return [Check("Theorem 1", claim, "FAIL", f"{type(exc).__name__}: {exc}", None, 0,
                      "Required runtime parity was not executed; the base selection module must be importable.")]
    scenarios = ("benefit", "resource_exclusion", "metric_exclusion", "no_benefit", "failed_observation", "no_feedback")
    outcomes, agree = [], 0
    n = 40
    for scenario in scenarios:
        candidates = [
            {"route": "nominal", "complexity": 0, "is_open_loop": True, "ensemble": 1, "cost": 1., "hash": "a"},
            {"route": "fixed_sequence", "complexity": 4, "is_open_loop": True, "ensemble": 1, "cost": 1., "hash": "b"},
            {"route": "one_step", "complexity": 6, "is_open_loop": False, "ensemble": 1, "cost": 1., "hash": "c"},
            {"route": "radius", "complexity": 7, "is_open_loop": False, "ensemble": 2, "cost": 2., "hash": "d"},
            {"route": "radius", "complexity": 7, "is_open_loop": False, "ensemble": 1, "cost": 1., "hash": "e"},
            {"route": "mpc_rh", "complexity": 8, "is_open_loop": False, "ensemble": 1, "cost": 3., "hash": "f"},
        ]
        offsets = np.array([0., -.001, -.006, -.010, -.010, -.011])
        if scenario == "no_benefit":
            offsets[2:] = .001
        losses = rng.uniform(.4, .6, (n, 1)) + offsets[None, :]
        metrics = np.full_like(losses, .7)
        if scenario == "resource_exclusion":
            candidates[4]["feasible"] = False
        elif scenario == "metric_exclusion":
            metrics[:, 4] = .6
        elif scenario == "failed_observation":
            candidates[4].update(feasible=False, execution_failure=True)
            losses[:, 4] = metrics[:, 4] = np.nan
        elif scenario == "no_feedback":
            for candidate in candidates[2:]:
                candidate["feasible"] = False
        runtime = nested_selection(losses, metrics, candidates, np.arange(n) % 5, np.arange(n) % 4,
                                   reps=30 if quick else 100, seed=int(rng.integers(2**30)),
                                   weights=1. + np.arange(n) % 3)
        observed = np.flatnonzero(runtime["observations_available"])
        complexity = [(candidates[j]["complexity"], candidates[j]["ensemble"], candidates[j]["cost"],
                       candidates[j]["hash"]) for j in observed]
        reference = selection.eq19_select(
            [runtime["upper"][j] for j in observed], [runtime["joint_feasible"][j] for j in observed],
            [not candidates[j]["is_open_loop"] for j in observed], complexity, .003, .002,
            candidate_scope="feedback_only")
        eligible = [int(observed[j]) for j in reference["Q_s"]]
        chosen = (runtime["qC_index"] if reference["selected"] is None
                  else int(observed[reference["selected"]]))
        expected = 1 if scenario in ("no_benefit", "no_feedback") else 4 if scenario == "benefit" else 3
        same = (eligible == runtime["Q_s"] == runtime["eligible_indices"]
                and chosen == runtime["provisional_index"] == expected
                and abs(reference["U_min"] - runtime["minimum_upper"]) <= 1e-14
                and runtime["open_loop_contrasts_are_diagnostic"] is True)
        agree += same
        outcomes.append({"scenario": scenario, "selected": runtime["provisional_index"],
                         "reference_selected": chosen, "expected": expected, "Q_s": eligible})
    return [Check("Theorem 1", claim, passed(agree == len(scenarios)), agree, len(scenarios), len(scenarios),
                  "Synthetic fixed predictions; actual nested resampling and terminal rule executed. "
                  "No assertion of real-data coverage or resolution of the Eq. (3)/S11 domain ambiguity. " + str(outcomes))]


# ============================================================================ Theorem 2 and Corollary 2
def _bounded_samples(rng, kind: str, mean: float, B: float, shape) -> np.ndarray:
    """I.i.d. units supported in [-B, B] with population mean ``mean`` (no clipping needed)."""
    if kind == "two-point":
        return np.where(rng.random(shape) < (mean + B) / (2 * B), B, -B)
    if kind == "uniform":
        half = B - abs(mean)
        return rng.uniform(mean - half, mean + half, size=shape)
    if kind == "rare-harm":  # frequent small gains in [-0.2B, 0], rare harms in [0.5B, B]
        q = (mean + 0.1 * B) / (0.75 * B + 0.1 * B)
        harm = rng.random(shape) < q
        return np.where(harm, rng.uniform(0.5 * B, B, size=shape), rng.uniform(-0.2 * B, 0.0, size=shape))
    raise ValueError(kind)


@_register("theorem-2")
def theorem2_coverage(rng, quick):
    B, delta, reps = 0.10, 0.025, (4000 if quick else 40000)
    worst, cells = 0.0, 0
    for kind in ("two-point", "uniform", "rare-harm"):
        for J in (20, 100, 400):
            true_mean = -0.004
            x = _bounded_samples(rng, kind, true_mean, B, (reps, J))
            upper = np.array([adm.admission_upper(float(mean), float(variance), J, B, delta)
                              for mean, variance in zip(x.mean(axis=1), x.var(axis=1, ddof=1))])
            worst = max(worst, float(np.mean(true_mean > upper)))
            cells += 1
    return [Check("Theorem 2", "Pr(mu_clip > U_a) <= delta_a on bounded i.i.d. units", passed(worst <= _mc_limit(delta, reps)),
                  worst, delta, cells * reps, "largest observed miss rate over distributions and J")]


@_register("theorem-2")
def theorem2_clipping_identity(rng, quick):
    trials, ok, identity = (500 if quick else 5000), 0, 0.0
    B = 0.10
    for _ in range(trials):
        J = int(rng.integers(2, 300))
        raw = rng.standard_t(3, size=J) * float(rng.uniform(0.01, 0.12))
        r = adm.admission_upper_from_units(raw, B)
        bound = adm.clipped_fraction_bound(r["D_bar"], r["S2"], J, B)
        clipped = np.clip(raw, -B, B)
        identity = max(identity, abs(float(np.mean(clipped ** 2)) - ((J - 1) * r["S2"] / J + r["D_bar"] ** 2)))
        ok += r["clipped_fraction"] <= bound + 1e-12
    return [Check("Theorem 2", "clipped fraction f <= [(J-1) S^2/J + D_bar^2]/B^2 deterministically", passed(ok == trials), ok,
                  trials, trials),
            Check("Theorem 2", "J^-1 sum D_j^2 = (J-1) S^2/J + D_bar^2", passed(identity <= 1e-15), identity, 1e-15, trials)]


@_register("theorem-2")
def theorem2_runtime_parity(rng, quick):
    try:
        from cross_morse.statistics import empirical_bernstein
    except Exception as exc:  # pragma: no cover - the runtime module is NumPy-only
        return [Check("Theorem 2", "mandatory runtime empirical_bernstein parity", "FAIL",
                      f"{type(exc).__name__}: {exc}", None, 0,
                      "The required base-repository runtime could not be imported; parity was not executed.")]
    worst, trials = 0.0, (200 if quick else 2000)
    for _ in range(trials):
        J = int(rng.integers(2, 500))
        raw = rng.normal(-0.004, float(rng.uniform(0.001, 0.08)), J)
        mine = adm.admission_upper_from_units(raw, 0.10, 0.025)
        runtime = empirical_bernstein(raw, clip=0.10, alpha=0.025, independent_graphs=True)
        errors = (abs(mine["U_a"] - runtime["upper"]), abs(mine["clipped_fraction"] - runtime["clipping_fraction"]))
        if not np.isfinite(errors).all():
            raise ValueError("runtime admission parity produced non-finite comparison values")
        worst = max(worst, *errors)
    return [Check("Theorem 2", "cm_theory admission bound equals cross_morse.statistics.empirical_bernstein",
                  passed(worst <= 1e-15), worst, 1e-15, trials)]


@_register("corollary-2")
def corollary2_necessity(rng, quick):
    trials, violations, admitted = (20000 if quick else 200000), 0, 0
    for _ in range(trials):
        D = -float(rng.uniform(0, 0.02))
        S2 = float(rng.uniform(0, 0.0008))
        J = int(rng.integers(2, 4000))
        eps = float(rng.choice([0.003, 0.0015]))
        if adm.admission_upper(D, S2, J) <= -eps:
            admitted += 1
            violations += not J >= adm.minimum_admission_threshold(D, eps)
    constant = adm.corollary2_constant()
    terms = tuple(round(adm.bernstein_terms(0.0, J, 0.10, 0.025)["fixed_term"], 3) for J in (10, 32, 64))
    note = (math.ceil(adm.minimum_admission_threshold(-0.0090, 0.003) - 1), math.ceil(adm.minimum_admission_threshold(-0.0045, 0.003) - 1))
    return [Check("Corollary 2", "every admitted configuration satisfies J >= 1 + 14 B log(2/delta)/(3(|D_bar| - eps_a))",
                  passed(violations == 0), violations, 0, admitted),
            Check("Corollary 2", "numerator constant 14 x 0.1 x log 80 / 3 = 2.045 nats", passed(round(constant, 3) == 2.045),
                  constant, 2.045),
            Check("Corollary 2", "Note S16: second penalty term 0.227, 0.066, 0.032 nats at J = 10, 32, 64",
                  passed(terms == (0.227, 0.066, 0.032)), terms, (0.227, 0.066, 0.032)),
            Check("Corollary 2", "Note S16: J - 1 > 341 at |D_bar| = 0.0090 and 1364 at 0.0045 (eps_a = 0.003)",
                  passed(note == (341, 1364)), note, (341, 1364))]


# ============================================================================ Proposition 4
@_register("proposition-4")
def proposition4_oracle(rng, quick):
    trials, ok, equality, ratio_ok, ratio_cases = (300 if quick else 3000), 0, 0, 0, 0
    for _ in range(trials):
        n, family = int(rng.integers(10, 80)), int(rng.integers(2, 8))
        losses = rng.gamma(2.0, 0.2, size=(n, family))
        comparator = int(rng.integers(family))
        oracle = selection.oracle_gap(losses, losses[:, comparator])
        choices = rng.integers(0, family, n)
        gap = selection.policy_gap(losses, choices, losses[:, comparator])
        ok += gap >= oracle - 1e-15
        equality += abs(selection.policy_gap(losses, np.argmin(losses, axis=1), losses[:, comparator]) - oracle) <= 1e-15
        if gap <= 0 and oracle < 0:
            ratio_cases += 1
            ratio_ok += 0 <= gap / oracle <= 1 + 1e-12
    return [Check("Proposition 4", "Delta(pi) >= Delta_oracle for every policy inside the supported family",
                  passed(ok == trials), ok, trials, trials),
            Check("Proposition 4", "the per-input argmin attains the oracle", passed(equality == trials), equality, trials, trials),
            Check("Proposition 4", "Delta(q_D)/Delta_oracle lies in [0, 1] when Delta(q_D) <= 0", passed(ratio_ok == ratio_cases),
                  ratio_ok, ratio_cases, ratio_cases)]


# ============================================================================ Proposition 5
@_register("proposition-5")
def proposition5_break_even(rng, quick):
    trials, ok = (5000 if quick else 50000), 0
    for _ in range(trials):
        l_inf, D_inf = float(rng.uniform(0.2, 2.0)), float(rng.uniform(0.001, 0.08))
        E, o = float(rng.choice([1, 2, 4, 8, 12])), float(rng.uniform(0.01, 2.0))
        delta = -float(rng.uniform(0, 0.01))
        threshold = economics.break_even_effect(D_inf, o, E)
        if abs(abs(delta) - threshold) < 1e-12:
            ok += 1
            continue
        equal_cost = economics.ensemble_loss(l_inf, D_inf, E * (1 + o))
        better = economics.ensemble_loss(l_inf, D_inf, E) + delta < equal_cost
        ok += better == (abs(delta) > threshold)
    return [Check("Proposition 5", "q_D beats the equal-compute comparator iff |Delta| > D_inf o/(E(1+o))",
                  passed(ok == trials), ok, trials, trials)]


@_register("proposition-5")
def proposition5_fit(rng, quick):
    grid = np.array([1, 2, 4, 8, 12, 16, 20], dtype=float)
    exact = economics.fit_ensemble_law(grid, 0.87 + 0.0305 / grid)
    within, trials = 0, (200 if quick else 2000)
    for _ in range(trials):
        fit = economics.fit_ensemble_law(grid, 0.87 + 0.0305 / grid + rng.normal(0, 0.0004, len(grid)))
        within += abs(fit["D_inf"] - 0.0305) <= 4 * fit["se_D_inf"]
    return [Check("Proposition 5", "least squares recovers (l_inf, D_inf) exactly from noiseless E14 grid data",
                  passed(abs(exact["D_inf"] - 0.0305) <= 1e-12 and abs(exact["l_inf"] - 0.87) <= 1e-12),
                  (exact["l_inf"], exact["D_inf"]), (0.87, 0.0305)),
            Check("Proposition 5", "noisy fits fall within four standard errors", passed(within >= 0.98 * trials), within,
                  trials, trials)]


# ============================================================================ Definition 2
@_register("definition-2")
def definition2_economics(rng, quick):
    trials, ok_price, ok_horizon = (5000 if quick else 50000), 0, 0
    for _ in range(trials):
        delta, added = float(rng.uniform(1e-4, 0.01)), float(rng.uniform(0.05, 3.0))
        rate = economics.exchange_rate(delta, 4.0 + added, 4.0)
        price = float(rng.uniform(0, 2 * rate))
        if abs(price - rate) > 1e-12:
            ok_price += (economics.net_benefit(delta, price, added) > 0) == (price < rate)
        else:
            ok_price += 1
        c_qual, theta, t_nom = float(rng.uniform(0.5, 3)), float(rng.uniform(0.01, 0.5)), float(10 ** rng.uniform(-4, 1.5))
        horizon = economics.amortization_horizon(c_qual, theta, added, t_nom)
        M = float(horizon * rng.uniform(0.2, 5.0))
        if abs(M - horizon) > 1e-9 * horizon:
            ok_horizon += (3600 * c_qual / (M * t_nom) <= theta * added) == (M >= horizon)
        else:
            ok_horizon += 1
    return [Check("Definition 2", "net benefit |Delta| - lambda dEC > 0 iff lambda < lambda*", passed(ok_price == trials),
                  ok_price, trials, trials),
            Check("Definition 2", "amortized qualification <= theta dEC iff M >= M*", passed(ok_horizon == trials), ok_horizon,
                  trials, trials)]


# ============================================================================ Proposition 6
def _spectral_instances(rng, quick):
    for w in graph_family(rng, 5 if quick else 18):
        yield w, rng.normal(size=(len(w), int(rng.integers(1, 6))))


def _pairs(actions, rng, extra: int):
    pairs = [(a, actions[0]) for a in actions[1:]]
    for _ in range(extra):
        i, j = rng.choice(len(actions), 2, replace=False)
        pairs.append((actions[int(i)], actions[int(j)]))
    return pairs


@_register("proposition-6")
def proposition6_band_bound(rng, quick):
    actions = _transport()
    ok_sym, ok_rs, frame_err, count = 0, 0, 0.0, 0
    for w, H in _spectral_instances(rng, quick):
        d = w.sum(axis=1)
        lam, U = laplacian_eigensystem(w)
        for a, b in _pairs(actions, rng, 4):
            r = spectral.displacement_bound(w, H, a, b)
            ok_sym += r["symmetric_displacement"] <= r["symmetric_bound"] * (1 + 1e-12) + 1e-14
            ok_rs += r["displacement"] <= r["bound"] * (1 + 1e-12) + 1e-14
            P = random_walk(w)
            g = spectral.action_response(a, lam) - spectral.action_response(b, lam)
            similar = (1 / np.sqrt(d))[:, None] * ((U * g) @ U.T @ (np.sqrt(d)[:, None] * H))
            frame_err = max(frame_err, float(np.max(np.abs((op.transport_matrix(P, a) - op.transport_matrix(P, b)) @ H - similar))))
            count += 1
    return [Check("Proposition 6(i)", "||(g_a - g_a')(L) X||_F <= [sum_b s_aa'b^2 P_b(X)]^(1/2)", passed(ok_sym == count),
                  ok_sym, count, count),
            Check("Proposition 6(iv)", "row-stochastic realization obeys the bound with d_min^(-1/2)", passed(ok_rs == count),
                  ok_rs, count, count),
            Check("Proposition 6(iv)", "T_a H = D^(-1/2) g_a(L_sym) D^(1/2) H", passed(frame_err <= 1e-10), frame_err, 1e-10, count)]


@_register("proposition-6")
def proposition6_windows(rng, quick):
    coeff = spectral.chebyshev_window_coefficients()
    ok, recurrence, count = 0, 0.0, 0
    epsilons = []
    for w, H in _spectral_instances(rng, quick):
        lam, U = laplacian_eigensystem(w)
        X = np.sqrt(w.sum(axis=1))[:, None] * H
        energy = np.sum((U.T @ X) ** 2, axis=1)
        members = spectral.band_membership(lam)
        for band in range(len(coeff)):
            eps_b = spectral.window_error(coeff[band], band)
            ideal = float(energy[members == band].sum())
            windowed = float(np.linalg.norm(spectral.window_matrix(w, coeff[band]) @ X) ** 2)
            ok += abs(ideal - windowed) <= eps_b * float(np.sum(energy)) * (1 + 1e-12) + 1e-12
            dense = spectral.window_matrix(w, coeff[band]) @ X
            recurrence = max(recurrence, float(np.max(np.abs(spectral.window_recurrence(w, coeff[band], X) - dense))) /
                             max(1.0, float(np.max(np.abs(dense)))))
            count += 1
    epsilons = [round(spectral.window_error(coeff[b], b), 3) for b in range(len(coeff))]
    return [Check("Proposition 6(ii)", "|P_b(X) - ||G_b X||^2| <= eps_b ||X||^2 for the registered degree-8 windows",
                  passed(ok == count), ok, count, count),
            Check("Proposition 6", "the NumPy reference three-term recurrence on -S equals the dense window g_b(L)",
                  passed(recurrence <= 1e-10), recurrence, 1e-10, count),
            Check("Proposition 6(ii)", "window approximation errors eps_b of the six registered windows", "INFO", epsilons, None,
                  len(coeff))]


@_register("proposition-6")
def proposition6_update_and_head(rng, quick):
    actions = _transport()
    ok_update, ok_head, count = 0, 0, 0
    for w, H in _spectral_instances(rng, quick):
        P, k = random_walk(w), H.shape[1]
        W = rng.normal(size=(k, k))
        head = rng.normal(size=(k, 3))
        labels = rng.integers(0, 3, len(P))
        for a, b in _pairs(actions, rng, 3):
            Ta, Tb = op.transport_matrix(P, a), op.transport_matrix(P, b)
            Fa, Fb = np.maximum(Ta @ H @ W, 0), np.maximum(Tb @ H @ W, 0)
            ok_update += fro(Fa - Fb) <= np.linalg.norm(W, 2) * fro((Ta - Tb) @ H) * (1 + 1e-12) + 1e-14
            change = abs(stability.cross_entropy(Fa @ head, labels).mean() - stability.cross_entropy(Fb @ head, labels).mean())
            ok_head += change <= math.sqrt(2.0 / len(P)) * np.linalg.norm(head, 2) * fro(Fa - Fb) * (1 + 1e-12) + 1e-14
            count += 1
    return [Check("Proposition 6(iii)", "an L_U-Lipschitz update bounds the state displacement by L_U ||(T_a - T_a')H||_F",
                  passed(ok_update == count), ok_update, count, count),
            Check("Proposition 6(iii)", "mean CE over N nodes changes by at most sqrt(2/N) ||W_head||_2 ||dH||_F",
                  passed(ok_head == count), ok_head, count, count)]


@_register("proposition-6")
def proposition6_retention_terms(rng, quick):
    actions = _transport()
    ok, count = 0, 0
    for w, H in _spectral_instances(rng, quick):
        P, k = random_walk(w), H.shape[1]
        W = rng.normal(size=(k, k)) / math.sqrt(k)
        Ra, Rb = rng.normal(size=H.shape), rng.normal(size=H.shape)
        for a, b in _pairs(actions, rng, 4):
            Ta, Tb = op.transport_matrix(P, a), op.transport_matrix(P, b)
            fa, fb = np.maximum(Ta @ H @ W, 0), np.maximum(Tb @ H @ W, 0)
            Fa, Fb = (1 - a.alpha) * fa + a.alpha * Ra, (1 - b.alpha) * fb + b.alpha * Rb
            bound = ((1 - a.alpha) * np.linalg.norm(W, 2) * fro((Ta - Tb) @ H) + abs(a.alpha - b.alpha) * fro(Rb - fb)
                     + a.alpha * fro(Ra - Rb))
            ok += fro(Fa - Fb) <= bound * (1 + 1e-12) + 1e-14
            count += 1
    return [Check("Proposition 6(iv')", "retention and reference differences add |alpha_a - alpha_a'| ||R_a' - f_a'|| + "
                  "alpha_a ||R_a - R_a'||", passed(ok == count), ok, count, count)]


@_register("proposition-6")
def proposition6_alignment_and_frame(rng, quick):
    coeff = spectral.chebyshev_window_coefficients()
    worst, ok_frame, count = 0.0, 0, 0
    for w, H in _spectral_instances(rng, quick):
        H0 = rng.normal(size=H.shape)
        lam, _ = laplacian_eigensystem(w)
        total = sum(spectral.window_response(coeff[b], lam) ** 2 for b in range(len(coeff)))
        c_min, c_max = float(total.min()), float(total.max())
        R = rng.normal(size=H.shape)
        windowed = sum(float(np.linalg.norm(spectral.window_matrix(w, coeff[b]) @ R) ** 2) for b in range(len(coeff)))
        energy = float(np.linalg.norm(R) ** 2)
        ok_frame += windowed / c_max <= energy * (1 + 1e-10) and energy <= windowed / c_min * (1 + 1e-10)
        for b in range(len(coeff)):
            G = spectral.window_matrix(w, coeff[b])
            for eps in (0.0, 1e-12, 1e-3):
                worst = max(worst, spectral.alignment_identity_residual(G, H, H0, eps))
        count += 1
    return [Check("Proposition 6(v)", "alignment identity (6.2) holds exactly at every registered and test epsilon",
                  passed(worst <= 1e-12), worst, 1e-12, count * len(coeff) * 3),
            Check("Proposition 6(v)", "frame inequality c_max^-1 sum_b ||G_b R||^2 <= ||R||^2 <= c_min^-1 sum_b ||G_b R||^2",
                  passed(ok_frame == count), ok_frame, count, count)]


@_register("proposition-6")
def proposition6_sketch(rng, quick):
    trials, ok = (300 if quick else 3000), 0
    for _ in range(trials):
        X = rng.normal(size=(int(rng.integers(5, 40)), int(rng.integers(1, 30))))
        k = int(rng.integers(1, 128))
        ok += spectral.sketch_relative_sd(X, k) <= spectral.sketch_relative_sd_bound(k) * (1 + 1e-12)
    X = rng.normal(size=(30, 12))
    k, draws = 16, (4000 if quick else 40000)
    target = float(np.linalg.norm(X) ** 2)
    est = np.array([np.linalg.norm(X @ (rng.normal(size=(12, k)) / math.sqrt(k))) ** 2 for _ in range(draws)])
    exact_sd = spectral.sketch_relative_sd(X, k)
    mc_sd = float(est.std() / target)
    unbiased = abs(est.mean() / target - 1) <= 4 * exact_sd / math.sqrt(draws)
    printed = (round(spectral.sketch_relative_sd_bound(16), 2), round(spectral.sketch_relative_sd_bound(64), 2))
    return [Check("Note S19", "exact relative SD of a Gaussian sketch is at most sqrt(2/k)", passed(ok == trials), ok, trials,
                  trials),
            Check("Note S19", "sketched band power is unbiased and its Monte Carlo SD matches the exact formula",
                  passed(unbiased and abs(mc_sd / exact_sd - 1) <= 0.1), (est.mean() / target, mc_sd, exact_sd), None, draws),
            Check("Note S19", "sqrt(2/k) = 0.35 at k = 16 and 0.18 at k = 64", passed(printed == (0.35, 0.18)), printed,
                  (0.35, 0.18))]


@_register("proposition-6")
def proposition6_reach_score(rng, quick):
    actions = _transport()
    coeff = spectral.chebyshev_window_coefficients()
    ok, count = 0, 0
    for w, H in _spectral_instances(rng, quick):
        lam, U = laplacian_eigensystem(w)
        X = np.sqrt(w.sum(axis=1))[:, None] * H
        eps = np.array([spectral.window_error(coeff[b], b) for b in range(len(coeff))])
        for a in actions[1:]:
            s = spectral.band_sensitivity(a, actions[0])
            score = math.sqrt(sum(s[b] ** 2 * np.linalg.norm(spectral.window_matrix(w, coeff[b]) @ X) ** 2 for b in range(len(coeff))))
            g = spectral.action_response(a, lam) - spectral.action_response(actions[0], lam)
            displacement = float(np.linalg.norm((U * g) @ U.T @ X))
            ok += displacement <= score + math.sqrt(float(np.sum(s ** 2 * eps))) * float(np.linalg.norm(X)) + 1e-12
            count += 1
    return [Check("Proposition 6", "reach score: ||H(a) - H(A0)|| <= s^_a(H) + [sum_b s_b^2 eps_b]^(1/2) ||H|| (L_U = 1)",
                  passed(ok == count), ok, count, count)]


# ============================================================================ Proposition 7 and Eq. (7)
@_register("proposition-7")
def proposition7_regret(rng, quick):
    trials, ok = (5000 if quick else 50000), 0
    for _ in range(trials):
        A = int(rng.integers(2, 11))
        truth = rng.normal(0, 0.01, A)
        truth[0] = 0.0  # A0: Delta_l,1(A0) = 0 by definition of the nominal-relative target
        e = rng.uniform(0.0005, 0.02, A)
        estimate = truth + rng.uniform(-1, 1, A) * e
        chosen = int(np.argmin(estimate))
        ok += truth[chosen] <= truth.min() + 2 * e.max() + 1e-15 and truth[chosen] <= 2 * e.max() + 1e-15
    return [Check("Proposition 7", "on the accuracy event Delta(pi_one) <= min_a Delta + 2 max e <= 2 max e (A0 supported)",
                  passed(ok == trials), ok, trials, trials)]


@_register("proposition-7")
def proposition7_radius_coverage(rng, quick):
    """Two-point units at +-B (the largest variance in the range) with a different mean per action."""
    B, A, delta, reps = 0.10, 10, 0.05, (2000 if quick else 20000)
    results = {}
    for n in (64, 200):
        p = rng.uniform(0.3, 0.7, A)
        truth = 2 * B * p - B
        x = np.where(rng.random((reps, A, n)) < p[None, :, None], B, -B)
        avg = x.mean(axis=2)
        e_h = adm.hoeffding_radius(n, B, A, delta)
        miss_h = float(np.mean(np.any(np.abs(avg - truth) > e_h, axis=1)))
        e_b = np.array([adm.bernstein_radius(n, float(sd), B, A, delta)
                        for sd in x.std(axis=2, ddof=1).ravel()]).reshape(reps, A)
        miss_b = float(np.mean(np.any(np.abs(avg - truth) > e_b, axis=1)))
        results[n] = (miss_h, miss_b)
    limit = _mc_limit(delta, reps)
    worst = max(max(v) for v in results.values())
    return [Check("Proposition 7", "Hoeffding and empirical-Bernstein radii cover all actions simultaneously at 1 - delta",
                  passed(worst <= limit), results, delta, 2 * reps, "simultaneous miss rates (Hoeffding, Bernstein) per n")]


@_register("proposition-7")
def proposition7_constants(rng, quick):
    main = round(2 * adm.hoeffding_radius(64), 3)
    table = tuple(round(2 * adm.hoeffding_radius(n), 4) for n in (64, 74, 91, 100, 142))
    needed = 8 * 0.10 ** 2 * math.log(2 * 10 / 0.05) / 0.003 ** 2
    crossover = adm.bernstein_crossover(0.007)
    bernstein_n = adm.units_for_bernstein_regret(0.003, 0.007)
    printed = (0.0865, 0.0805, 0.0726, 0.0692, 0.0581)
    return [Check("Proposition 7", "main text: the regret bound 2e at n_min = 64 is 0.087 nats", passed(main == 0.087), main, 0.087),
            Check("Proposition 7", "Table S28b Hoeffding column 2e(n) at n = 64, 74, 91, 100, 142", passed(table == printed),
                  table, printed, len(printed)),
            Check("Proposition 7", "2e(n) <= 0.003 needs about 53,000 units (8 B^2 log(2|A|/delta)/0.003^2)",
                  passed(53000 <= needed <= 53500), needed, 53000),
            Check("Proposition 7", "empirical-Bernstein radius falls below Hoeffding's for n >~ 95 (sigma_1 = 0.007)",
                  passed(90 <= crossover <= 100), crossover, 95),
            Check("Proposition 7", "2 e_B(n) <= 0.003 at n ~ 3,000 for sigma_1 = 0.007", passed(2900 <= bernstein_n <= 3100),
                  bernstein_n, 3000)]


@_register("proposition-7")
def eq7_support_identity(rng, quick):
    trials, ok = (5000 if quick else 50000), 0
    for _ in range(trials):
        w = rng.gamma(float(rng.uniform(0.2, 5.0)), 1.0, size=int(rng.integers(2, 400)))
        ok += adm.max_normalized_weight(w) <= adm.weight_cap_from_ess(adm.effective_sample_size(w)) + 1e-15
    caps = (round(adm.weight_cap_from_ess(64), 3), round(adm.weight_cap_from_ess(32), 3))
    return [Check("Eq. (7)", "omega_max <= 1/sqrt(ESS) on one weight vector", passed(ok == trials), ok, trials, trials),
            Check("Eq. (7)", "ESS >= 64 gives omega_max <= 0.125; ESS >= 32 gives <= 0.177 (Tables S7, S18)",
                  passed(caps == (0.125, 0.177)), caps, (0.125, 0.177))]


# ============================================================================ Propositions 8 and 9
@_register("proposition-8")
def proposition8_search(rng, quick):
    trials, ok = (5000 if quick else 50000), 0
    for _ in range(trials):
        S = int(rng.integers(2, 40))
        J = rng.normal(0, 0.01, S)
        e_h, zeta = float(rng.uniform(0, 0.005)), float(rng.uniform(0, 0.003))
        J_hat = J + rng.uniform(-1, 1, S) * e_h
        candidates = np.flatnonzero(J_hat <= J_hat.min() + zeta)
        chosen = int(rng.choice(candidates))
        ok += J[chosen] - J.min() <= stability.search_regret_bound(e_h, zeta) + 1e-15
    e = 0.004
    tight = (np.array([0.0, 2 * e]) + np.array([e, -e]))  # J_hat ties; picking the second gives regret 2 e
    return [Check("Proposition 8", "J(a^) - min J <= 2 e_H + zeta_H", passed(ok == trials), ok, trials, trials),
            Check("Proposition 8", "the bound is attained when J^ ties a good and a bad sequence", passed(tight[0] == tight[1]),
                  tight.tolist(), None)]


@_register("proposition-9")
def proposition9_certified_ball(rng, quick):
    trials, certified, ok = (2000 if quick else 20000), 0, 0
    for _ in range(trials):
        c, g, L = _linear_scores(rng, int(rng.integers(2, 10)), 3)
        center = rng.normal(size=3)
        gap = stability.score_gap(c + g @ center)
        radius = float(rng.uniform(0, 1)) * gap / (2 * L) * 1.5
        if not stability.certified_ball(gap, L, radius):
            continue
        certified += 1
        points = rng.normal(size=(50, 3))
        points = center + points / np.linalg.norm(points, axis=1, keepdims=True) * radius * rng.uniform(0, 1, (50, 1))
        best = np.argmin(c[None, :] + points @ g.T, axis=1)
        ok += bool(np.all(best == np.argmin(c + g @ center)))
    return [Check("Proposition 9", "gap > 2 L_J r certifies one minimizer on the whole ball", passed(ok == certified), ok,
                  certified, certified)]


# ============================================================================ Lemma 3 (Algorithm S3)
@_register("lemma-3")
def lemma3_work_bound(rng, quick):
    episodes = (1500 if quick else 15000)
    holds, statuses = 0, {}
    for _ in range(episodes):
        e = budget.random_episode(rng)
        result = budget.run_runtime_episode(e)
        verdict = budget.lemma3_invariants(e, result)
        holds += verdict["holds"]
        statuses[verdict["status"]] = statuses.get(verdict["status"], 0) + 1
    every_branch = set(statuses) == {"INFEASIBLE", "STATIC_FALLBACK", "LATE_FALLBACK", "COMPLETED"}
    from cm_experiments.budget import WorkBoundViolation
    episode = budget.random_episode(rng, depth=2)
    episode.actual["action"] = 1.5  # observed work above the declared bound
    try:
        budget.run_runtime_episode(budget.Episode(episode.budget + 20, episode.costs, 2, ((0,), (0,)), (False, False),
                                                  episode.actual))
        refused = False
    except WorkBoundViolation:
        refused = True
    counter = budget.reserve_released_counterexample()
    return [Check("Lemma 3", "cm_experiments.budget: total work <= W on every branch with a prediction; infeasible pays nothing",
                  passed(holds == episodes), holds, episodes, episodes, f"branches {statuses}"),
            Check("Lemma 3", "all four terminal statuses are exercised (not full decision-event coverage)", passed(every_branch), sorted(statuses), None, episodes),
            Check("Lemma 3", "observed work above a declared bound is refused rather than credited", passed(refused), refused, True),
            Check("Lemma 3", "Figure S11: releasing the reserve before the final site can exceed W", "INFO", counter, None)]


@_register("lemma-3")
def lemma3_reference_agreement(rng, quick):
    episodes, agree = (1500 if quick else 15000), 0
    for _ in range(episodes):
        e = budget.random_episode(rng)
        runtime, reference = budget.run_runtime_episode(e), budget.algorithm_s3_reference(e)
        same_events = all(runtime[key] == reference[key]
                          for key in ("status", "prediction", "operations", "action_ids", "decisions"))
        agree += same_events and abs(runtime["total"] - reference["total"]) <= 1e-12
    return [Check("Lemma 3", "the executable rule and the reference agree on paid operations, actions, decision counts and total work",
                  passed(agree == episodes), agree, episodes, episodes,
                  "Synthetic callbacks with declared work bounds and supplied certificate premises; not learned-controller certification.")]


# ============================================================================ Proposition 10
@_register("proposition-10")
def proposition10_boundaries(rng, quick):
    trials, ok = (5000 if quick else 50000), [0, 0, 0, 0]
    for _ in range(trials):
        g, phi, r_b = float(rng.uniform(0.001, 0.01)), float(rng.uniform(0, 1.3)), float(rng.uniform(0.85, 1.0))
        o = float(rng.uniform(0.05, 0.4))
        o_dec, o_b = o * float(rng.uniform(0, 1)), o * float(rng.uniform(0, 1))
        D_inf, E = float(rng.uniform(0.005, 0.08)), float(rng.choice([4, 8]))
        R_C = 1.0
        ok[0] += ((R_C - phi * g) < (R_C - g)) == (phi > 1) and ((R_C - phi * g) < (R_C - r_b * g)) == (phi > r_b)
        bound = economics.decision_boundary(g, r_b, o, o_dec, o_b, D_inf, E)
        pred = R_C - phi * g - bound["b_predictor"]
        ok[1] += (pred < R_C - g) == (phi > bound["phi_FB_ensemble"]) or abs(phi - bound["phi_FB_ensemble"]) < 1e-12
        ok[2] += (pred < R_C - r_b * g - bound["b_budget_aware"]) == (phi > bound["phi_B_ensemble"]) or \
            abs(phi - bound["phi_B_ensemble"]) < 1e-12
        cost_fb, cost_pred = 4 * (1 + o), 4 * (1 + o_dec)
        if cost_fb > cost_pred + 1e-9:
            lam_star = economics.priced_crossover(phi, g, cost_fb, cost_pred)
            lam = float(rng.uniform(0, 2 * abs(lam_star) + 1e-6))
            advantage = (1 - phi) * g - lam * (cost_fb - cost_pred)
            ok[3] += (advantage > 0) == (lam < lam_star) or abs(lam - lam_star) < 1e-12
        else:
            ok[3] += 1
    labels = ("(i) contract crossovers 1 and r_B", "(ii) ensemble-spend crossover against q_FB",
              "(ii) ensemble-spend crossover against q_B", "(iii) priced-work crossover lambda*_pred")
    return [Check("Proposition 10", label, passed(count == trials), count, trials, trials) for label, count in zip(labels, ok)]


# ============================================================================ Proposition 11
@_register("proposition-11")
def proposition11_advantage(rng, quick):
    trials, worst = (4 if quick else 20), 0.0
    for t in range(trials):
        process = control.random_process(rng, states=3, actions=3, horizon=3 if quick else 4)
        policy = control.random_history_policy(rng, 3)
        if t % 3 == 0:
            comparator = control.random_history_policy(rng, 3)
        elif t % 3 == 1:
            plan = rng.integers(0, 3, process.horizon)
            comparator = lambda h, plan=plan: np.eye(3)[plan[(len(h) - 1) // 2]]  # noqa: E731 - open-loop schedule
        else:
            comparator = lambda h: np.eye(3)[0]  # noqa: E731 - constant nominal action
        r = control.advantage_identity(process, policy, comparator)
        worst = max(worst, r["residual"])
    return [Check("Proposition 11", "V^pi_0 - V^C_0 = E_pi sum_l A^C_l(h_l, a_l) (history-dependent, open-loop and nominal "
                  "comparators)", passed(worst <= 1e-12), worst, 1e-12, trials)]


# ============================================================================ Eqs. (5)-(6), (32)-(33), Note S19, Note S24, Methods
@_register("eq-5-6")
def eq6_dil(rng, quick):
    trials, worst = (500 if quick else 5000), 0.0
    for _ in range(trials):
        K = int(rng.integers(2, 10))
        depths = np.cumprod(np.r_[float(rng.integers(1, 4)), rng.uniform(1.2, 3.0, K - 1)])
        losses = rng.uniform(0.2, 2.0, K)
        x = np.log2(depths)
        integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        reference = float(integrate(losses, x) / (x[-1] - x[0]))
        worst = max(worst, abs(contract.log_depth_integrated_loss(depths, losses) - reference))
    trials_gate, agree = (2000 if quick else 20000), 0
    t = contract.ENTRY_THRESHOLDS
    for _ in range(trials_gate):
        row = {"U_Delta_plant": float(rng.uniform(-0.008, 0.0)), "PDD_pp": float(rng.uniform(0, 1.5)),
               "DIL_gap_nats": float(rng.uniform(0, 0.03)), "phi_registry": float(rng.uniform(0.7, 1.0)),
               "selected_strata": float(rng.choice([1.0, 0.95])), "U_Delta_oracle": float(rng.uniform(-0.006, 0.0))}
        direct = (row["U_Delta_plant"] < -t["epsilon_plant_nats"] and row["PDD_pp"] <= t["tau_PDD_percentage_points"]
                  and row["DIL_gap_nats"] <= t["tau_DIL_nats"] and row["phi_registry"] >= t["tau_sup"]
                  and row["selected_strata"] == 1.0 and row["U_Delta_oracle"] < -t["epsilon_headroom_nats"])
        agree += contract.entry_contract(row)["passes"] == direct
    return [Check("Eq. (6)", "log-depth-integrated loss equals the NumPy reference trapezoid rule in log2 depth",
                  passed(worst <= 1e-12), worst, 1e-12, trials),
            Check("Eq. (5)", "the six registered gates and thresholds decide entry", passed(agree == trials_gate), agree,
                  trials_gate, trials_gate)]


@_register("eq-32-33")
def eq33_branch_accounting(rng, quick):
    trials, within, reduce_err = (40 if quick else 300), 0, 0.0
    draws = 20000
    for _ in range(trials):
        C_g, C_C0, C_S, C_A, C_C1 = rng.uniform(0, 0.5), rng.uniform(1, 8), rng.uniform(1, 10), rng.uniform(0, 3), rng.uniform(1, 8)
        p0, p1 = float(rng.uniform(0, 0.3)), float(rng.uniform(0, 0.2))
        static = rng.random(draws) < p0
        late = rng.random(draws) < p1
        cost = C_g + np.where(static, C_C0, np.where(late, C_A + C_C1, C_S))
        expected = economics.expected_cost_general(C_g, p0, C_C0, p1, C_S, C_A, C_C1)
        within += abs(cost.mean() - expected) <= 5 * cost.std() / math.sqrt(draws) + 1e-12
        reduce_err = max(reduce_err, abs(economics.expected_cost_general(0.0, p0, C_C0, 0.0, C_S, 0.0, 0.0)
                                         - economics.expected_cost(C_S, C_C0, p0)))
    return [Check("Eq. (33)", "branch-weighted mean cost converges to the accounting identity", passed(within == trials), within,
                  trials, trials * draws),
            Check("Eq. (32)", "with C_g = 0 and p1 = 0 Eq. (33) reduces to (1 - p) C_controller + p C_qC",
                  passed(reduce_err <= 1e-12), reduce_err, 1e-12, trials)]


@_register("note-s19")
def note_s19_cost_model(rng, quick):
    trials, inverse_err, fraction_err = (2000 if quick else 20000), 0.0, 0.0
    for _ in range(trials):
        r, s, f, T = float(10 ** rng.uniform(-3, 0)), float(rng.uniform(0, 0.2)), float(rng.uniform(0.05, 1)), int(rng.integers(2, 64))
        m = budget.per_member_multiplier(r, s, f, T)
        inverse_err = max(inverse_err, abs(budget.admissible_truncation_order(m, r, s, f) - T))
        fraction_err = max(fraction_err, abs(budget.max_resolvent_fraction(m, r, s, T) - f))
    share = budget.sensing_share(0.583) / 2 - budget.sensing_share(0.583, 1.0, 2)
    reach = {a.id: op.maximum_reach(a) for a in _transport() if a.is_resolvent}
    return [Check("Note S19", "the admissible truncation order inverts the per-member multiplier", passed(inverse_err <= 1e-9),
                  inverse_err, 1e-9, trials),
            Check("Note S19", "the admissible resolvent fraction inverts the per-member multiplier", passed(fraction_err <= 1e-12),
                  fraction_err, 1e-12, trials),
            Check("Note S19", "sensing at period 2 costs half of full-state sensing at every site", passed(abs(share) <= 1e-15),
                  share, 0.0),
            Check("Note S19", "registered resolvent polynomials have maximum power T (runtime sparse-work counting is separate)",
                  passed(set(reach.values()) == {8}), reach, 8, len(reach))]


@_register("note-s24")
def note_s24_variance_components(rng, quick):
    reps, J, S = (300 if quick else 3000), 40, 12
    su, sv, se = 0.0004 ** 2, 0.002 ** 2, 0.003 ** 2
    truth = inference.variance_of_mean(su, sv, se, J, S)
    estimates, identity = [], 0.0
    for _ in range(reps):
        D = 0.001 + rng.normal(0, math.sqrt(su), (J, 1)) + rng.normal(0, math.sqrt(sv), (1, S)) + rng.normal(0, math.sqrt(se), (J, S))
        r = inference.variance_components(D)
        ms = r["mean_squares"]
        identity = max(identity, abs(r["var_mean_unbiased"] - (ms["graph"] + ms["seed"] - ms["residual"]) / (J * S)))
        estimates.append(r["var_mean_unbiased"])
    estimates = np.array(estimates)
    unbiased = abs(estimates.mean() - truth) <= 4 * estimates.std() / math.sqrt(reps)
    bound_ok = 0
    for _ in range(1000):
        a, b, c = rng.uniform(0, 1e-4, 3)
        Jx, Sx = int(rng.integers(5, 6000)), int(rng.integers(2, 20))
        v_test = a / Jx + b / Sx + c / (Jx * Sx)
        v_graph = (a + c) / Jx
        bound_ok += v_test - v_graph <= b / Sx + 1e-18
    return [Check("Note S24", "Var^(D_bar) = (MS_graph + MS_seed - MS_e)/(J S)", passed(identity <= 1e-18), identity, 1e-18, reps),
            Check("Note S24", "the untruncated two-way moment estimate of Var D_bar is unbiased", passed(unbiased), (estimates.mean(), truth), None, reps),
            Check("Note S24", "V_test - V_graph <= sigma_v^2/S, so 1 - rho^-2 lower-bounds the seed share", passed(bound_ok == 1000),
                  bound_ok, 1000, 1000)]


@_register("note-s24")
def methods_power_and_equivalence(rng, quick):
    multipliers = (round(inference.minimum_detectable_effect(), 2), round(inference.minimum_detectable_effect(family_size=4), 2))
    reps, n, margin = (4000 if quick else 40000), 20, 0.002
    x = rng.normal(margin, 0.003, size=(reps, n))  # boundary of the equivalence null
    se = x.std(axis=1, ddof=1) / math.sqrt(n)
    mean = x.mean(axis=1)
    p = np.maximum([inference.student_t_sf(v, n - 1) for v in (mean + margin) / se],
                   [inference.student_t_sf(v, n - 1) for v in (margin - mean) / se])
    size = float(np.mean(p < 0.05))
    worst = 0.0
    try:
        from cross_morse.statistics import holm_adjust as runtime_holm
        from cm_campaigns.numeric import holm as campaign_holm
        for _ in range(200):
            pv = rng.uniform(0, 0.2, int(rng.integers(1, 12)))
            mine = inference.holm_adjust(pv)
            errors = (float(np.max(np.abs(mine - runtime_holm(pv)))),
                      float(np.max(np.abs(mine - np.asarray(campaign_holm(list(pv)))))))
            if not np.isfinite(errors).all():
                raise ValueError("runtime Holm parity produced non-finite comparison values")
            worst = max(worst, *errors)
        holm = Check("Methods", "Holm adjustment agrees with cross_morse.statistics and cm_campaigns.numeric",
                     passed(worst <= 1e-15), worst, 1e-15, 200)
    except Exception as exc:  # a missing or broken required runtime cannot establish parity
        holm = Check("Methods", "mandatory Holm adjustment parity", "FAIL",
                     f"{type(exc).__name__}: {exc}", None, 0,
                     "The required base-repository runtime could not be imported or checked; parity was not established.")
    return [Check("Methods", "minimum detectable effects 2.80 SE (two-sided 0.05) and 3.34 SE (first Holm step of four)",
                  passed(multipliers == (2.80, 3.34)), multipliers, (2.80, 3.34)),
            Check("Methods", "two one-sided tests keep their size at the margin boundary", passed(size <= _mc_limit(0.05, reps)),
                  size, 0.05, reps),
            holm]


# ============================================================================ runner
def run_battery(*, quick: bool = True, seed: int = DEFAULT_SEED, only: list[str] | None = None) -> dict:
    known = set(CHECKS) | set(CHECK_STATEMENT.values())
    unknown = set(only or ()) - known
    if unknown:
        raise KeyError(f"no check or statement named {sorted(unknown)}")
    names = list(CHECKS) if not only else [n for n in CHECKS if n in set(only) or CHECK_STATEMENT[n] in set(only)]
    rows, timings = [], {}
    for name in names:
        start = time.perf_counter()
        try:
            checks = CHECKS[name](_rng(seed, name), quick)
        except Exception as exc:  # a crashed check is a failure, never a silent pass
            checks = [Check(CHECK_STATEMENT[name], f"{name} raised", "FAIL", f"{type(exc).__name__}: {exc}", None, 0)]
        timings[name] = round(time.perf_counter() - start, 3)
        for c in checks:
            rows.append({"check": name, "statement_key": CHECK_STATEMENT[name], **c.to_dict()})
    counts = {status: sum(r["status"] == status for r in rows) for status in ("PASS", "FAIL", "ADVISORY", "INFO")}
    covered = {CHECK_STATEMENT[n] for n in names}
    missing = [s.key for s in STATEMENTS if s.key not in covered] if not only else []
    return {"schema_version": "cm-theory-battery-1", "seed": seed, "mode": "quick" if quick else "full",
            "counts": counts, "statement_groups_selected": len(covered), "check_generators_selected": len(names),
            "assertion_rows": counts["PASS"] + counts["FAIL"],
            "scope": "Sampled reference checks and explicitly executed runtime parity only; not proofs or benchmark reproduction.",
            "statements_without_checks": missing, "rows": rows, "seconds": timings}
