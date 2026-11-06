"""Deployment economics: Eqs. (32)-(33), Definition 2, Propositions 5 and 10.

Costs are in units of one nominal full-depth execution; effects are paired loss
differences in nats (negative favours the deployed rule).
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------- Eqs. (32)-(33)
def expected_cost(controller: float, comparator: float, p_fallback: float) -> float:
    """Eq. (32): C_expected = (1 - p_fallback) C_controller + p_fallback C_qC."""
    if not 0 <= p_fallback <= 1:
        raise ValueError("p_fallback must be a probability")
    return (1 - p_fallback) * controller + p_fallback * comparator


def expected_cost_general(C_g: float, p0: float, C_C0: float, p1: float, C_S: float,
                          C_A: float, C_C1: float) -> float:
    """Eq. (33): E C = C_g + p0 C_{C,0} + (1-p0)[(1-p1) C_S + p1 (C_A + C_{C,1})]."""
    if not (0 <= p0 <= 1 and 0 <= p1 <= 1):
        raise ValueError("branch probabilities must lie in [0, 1]")
    return C_g + p0 * C_C0 + (1 - p0) * ((1 - p1) * C_S + p1 * (C_A + C_C1))


def assurance_cost(controller: float, comparator: float) -> float:
    """Two-branch assurance configuration: both branches execute."""
    return controller + comparator


# ---------------------------------------------------------------------------- Definition 2
def exchange_rate(abs_delta: float, expected: float, comparator: float) -> float:
    """lambda* = |Delta| / (E C(q_D) - C(q_C)), nats per nominal execution of added work."""
    added = expected - comparator
    if added <= 0:
        raise ValueError("the exchange rate is defined for positive added work")
    return abs_delta / added


def amortization_horizon(qualification_gpu_h: float, theta: float, added_work: float, t_nom: float) -> float:
    """M* = 3600 C_qual / [theta (E C(q_D) - C(q_C)) t_nom] served inputs."""
    if min(theta, added_work, t_nom) <= 0:
        raise ValueError("theta, added work and t_nom must be positive")
    return 3600.0 * qualification_gpu_h / (theta * added_work * t_nom)


def net_benefit(abs_delta: float, price: float, added_work: float) -> float:
    """U(q) = |Delta| - lambda (E C(q_D) - C(q_C)); positive iff lambda < lambda*."""
    return abs_delta - price * added_work


# ---------------------------------------------------------------------------- Proposition 5
def fit_ensemble_law(ensemble_sizes: Sequence[float], losses: Sequence[float]) -> dict:
    """Least-squares fit of l(E) = l_inf + D_inf / E (Supplementary Table S26b), with standard errors."""
    E = np.asarray(ensemble_sizes, dtype=np.float64)
    y = np.asarray(losses, dtype=np.float64)
    if E.shape != y.shape or len(E) < 3 or (E <= 0).any():
        raise ValueError("need at least three positive ensemble sizes with one loss each")
    X = np.column_stack([np.ones_like(E), 1.0 / E])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    residual = y - X @ coef
    dof = len(E) - 2
    sigma2 = float(residual @ residual / dof) if dof > 0 else float("nan")
    cov = sigma2 * np.linalg.inv(X.T @ X) if dof > 0 else np.full((2, 2), np.nan)
    return {"l_inf": float(coef[0]), "D_inf": float(coef[1]), "se_l_inf": float(math.sqrt(cov[0, 0])),
            "se_D_inf": float(math.sqrt(cov[1, 1])), "residual_sd": float(math.sqrt(sigma2)),
            "residuals": residual.tolist()}


def ensemble_loss(l_inf: float, D_inf: float, E: float) -> float:
    return l_inf + D_inf / E


def break_even_effect(D_inf: float, overhead: float, E: float) -> float:
    """Proposition 5: q_D improves at equal compute iff |Delta| > D_inf o / (E (1 + o))."""
    if E <= 0 or overhead < 0:
        raise ValueError("E must be positive and the overhead nonnegative")
    return D_inf * overhead / (E * (1 + overhead))


def equal_compute_gain(D_inf: float, overhead: float, E: float) -> float:
    """Loss reduction of the comparator ensemble grown to the same serial work, E (1 + o) members."""
    return D_inf / E - D_inf / (E * (1 + overhead))


# ---------------------------------------------------------------------------- Proposition 10
def spare_budget_gain(D_inf: float, E: float, spare_overhead: float) -> float:
    """b(o') = D_inf o' / [E (1 + o')]: gain of spending spare serial budget on ensemble members."""
    return break_even_effect(D_inf, spare_overhead, E)


def decision_boundary(g: float, r_B: float, o: float, o_dec: float, o_B: float,
                      D_inf: float | None = None, E: float | None = None) -> dict:
    """Crossover retained shares of Proposition 10.

    (i) contract: the initial-input policy beats q_FB iff phi > 1 and beats q_B iff phi > r_B;
    (ii) ensemble spend (first order): phi*_FB = 1 - b(o - o_dec)/g and
         phi*_B = r_B - [b(o - o_dec) - b(o - o_B)]/g.
    """
    if g <= 0:
        raise ValueError("g = R(q_C) - R(q_FB) must be positive")
    out = {"phi_FB_contract": 1.0, "phi_B_contract": float(r_B)}
    if D_inf is not None and E is not None:
        b_dec = spare_budget_gain(D_inf, E, o - o_dec)
        b_B = spare_budget_gain(D_inf, E, o - o_B)
        out.update(phi_FB_ensemble=1.0 - b_dec / g, phi_B_ensemble=r_B - (b_dec - b_B) / g,
                   b_predictor=b_dec, b_budget_aware=b_B)
    return out


def priced_crossover(phi: float, g: float, cost_feedback: float, cost_predictor: float) -> float:
    """lambda*_pred = (1 - phi) g / [E C(q_FB) - E C(q_pred)]; feedback is preferred iff lambda < lambda*_pred."""
    added = cost_feedback - cost_predictor
    if added <= 0:
        raise ValueError("feedback must cost more than the initial-input policy")
    return (1 - phi) * g / added
