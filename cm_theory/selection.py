"""Finite-family supported selection (main Eq. 19, Theorem 1) and Proposition 4.

:func:`eq19_select` defaults to the feedback-plus-comparator domain depicted
in main Eq. (3) and Supplementary Note S1, preserving the proposal's default.
The optional ``all_feasible`` scope is an unrestricted reading of Eqs. (18)-(19)
for diagnosis. The relation between that feasible-set definition, Eq. (3), and
the MPC-OL predecessor clause in Supplementary Table S11 remains unresolved.
The current ``cross_morse.statistics.nested_selection`` uses the feedback-only
domain. Its terminal-rule parity check is separate from the synthetic theorem
simulation and does not establish coverage of the real nested bootstrap.

The returned effective candidate indices identify the family to use for both
simultaneous coverage and Theorem 1's guarantee. The comparator's zero contrast
is included implicitly in either scope; its selection is represented by None.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def eq19_select(upper: Sequence[float], feasible: Sequence[bool], feedback: Sequence[bool],
                complexity: Sequence[tuple], epsilon_r: float, epsilon_eq: float, *,
                candidate_scope: str = "feedback_only") -> dict:
    """Return Q_s and its effective family (selected index, or None for q_C).

    ``feedback_only`` intersects the feasibility and feedback masks, following
    the lifecycle domain in Eq. (3). ``all_feasible`` uses the feasibility mask
    alone as a diagnostic alternative. A guarantee checked against a different
    family does not establish Theorem 1 for the selected scope.
    """
    if not isinstance(candidate_scope, str) or candidate_scope not in ("feedback_only", "all_feasible"):
        raise ValueError("candidate_scope must be 'feedback_only' or 'all_feasible'")
    U = np.asarray(upper, dtype=np.float64)
    ok = np.asarray(feasible, dtype=bool)
    route_type = np.asarray(feedback, dtype=bool)
    if U.ndim != 1 or not np.isfinite(U).all() or ok.shape != U.shape or route_type.shape != U.shape \
            or len(complexity) != len(U) or not np.isfinite([epsilon_r, epsilon_eq]).all() \
            or epsilon_r < 0 or epsilon_eq < 0:
        raise ValueError("need aligned finite candidate vectors and nonnegative finite margins")
    if candidate_scope == "feedback_only":
        ok = ok & route_type
    effective_candidates = np.flatnonzero(ok).tolist()
    u_min = min(0.0, float(np.min(U[ok]))) if ok.any() else 0.0
    eligible = [j for j in range(len(U)) if ok[j] and U[j] < -epsilon_r and U[j] <= u_min + epsilon_eq]
    choice = min(eligible, key=lambda j: (tuple(complexity[j]), j)) if eligible else None
    return {"Q_s": eligible, "selected": choice, "U_min": u_min,
            "candidate_scope": candidate_scope, "effective_candidates": effective_candidates}


def theorem1_guarantee(true_effects: Sequence[float], selected: int | None, radii: Sequence[float],
                       feasible: Sequence[bool], epsilon_r: float, epsilon_eq: float) -> dict:
    """Theorem 1's conclusion for a selection made on intervals that covered.

    Either the comparator is returned, or Delta(q_M) < -epsilon_R and
    Delta(q_M) <= min_{q in Q_feas} Delta(q) + epsilon_eq + 2 r_max, where Q_feas
    contains the comparator itself with Delta(q_C) = 0. The ``feasible`` mask
    must identify the selector's effective candidate family, after any declared
    scope restriction; it is not necessarily the original feasibility mask.
    """
    delta = np.asarray(true_effects, dtype=np.float64)
    if selected is None:
        return {"holds": True, "returned_comparator": True, "slack": None}
    ok = np.asarray(feasible, dtype=bool)
    best = min(0.0, float(np.min(delta[ok]))) if ok.any() else 0.0
    r_max = float(np.max(np.asarray(radii, dtype=np.float64)[ok]))
    limit = best + epsilon_eq + 2 * r_max
    improves = bool(delta[selected] < -epsilon_r)
    near_best = bool(delta[selected] <= limit + 1e-15)
    return {"holds": improves and near_best, "returned_comparator": False,
            "improves": improves, "near_best": near_best, "slack": float(limit - delta[selected])}


def max_t_radii(covariance: np.ndarray, alpha: float, rng: np.random.Generator, draws: int = 20000) -> np.ndarray:
    """Simultaneous radii r_q = c * sd_q, with c the (1-alpha) quantile of max_q |Z_q|/sd_q."""
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1] or not len(covariance) \
            or not np.isfinite(covariance).all() or not np.allclose(covariance, covariance.T) \
            or not 0 < alpha < 1 or not isinstance(draws, (int, np.integer)) or draws < 1:
        raise ValueError("need a finite symmetric covariance, 0 < alpha < 1 and positive integer draws")
    eigenvalues = np.linalg.eigvalsh(covariance)
    if np.min(np.diag(covariance)) < 0 or eigenvalues.min() < -1e-12 * max(1.0, np.linalg.norm(covariance, 2)):
        raise ValueError("covariance must be positive semidefinite")
    sd = np.sqrt(np.diag(covariance))
    active = sd > 0
    if not active.any():
        return np.zeros_like(sd)
    z = rng.multivariate_normal(np.zeros(len(sd)), covariance, size=draws, check_valid="raise")
    critical = float(np.quantile(np.max(np.abs(z[:, active]) / sd[active], axis=1), 1 - alpha))
    return critical * sd


def covers(estimates: np.ndarray, truth: np.ndarray, radii: np.ndarray) -> bool:
    return bool(np.all(np.abs(estimates - truth) <= radii))


# ---------------------------------------------------------------------------- Proposition 4
def oracle_gap(losses: np.ndarray, comparator_losses: np.ndarray) -> float:
    """Delta_oracle = E[min over the supported family] - E[loss of q_C] (<= 0 when q_C is in the family)."""
    return float(np.mean(np.min(losses, axis=1)) - np.mean(comparator_losses))


def policy_gap(losses: np.ndarray, choices: np.ndarray, comparator_losses: np.ndarray) -> float:
    """Delta_sel of a policy whose realized sequence per input is ``choices`` (indices into the family)."""
    return float(np.mean(losses[np.arange(len(losses)), choices]) - np.mean(comparator_losses))
