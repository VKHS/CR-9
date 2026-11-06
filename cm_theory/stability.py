"""Perturbation and stability statements: main Propositions 2, 3, 8 and 9.

Proposition 2 is checked on the layer-normalization-free member of the plant
class, H_{l+1} = (1-alpha) phi(T_l H_l W + b) + alpha H_ref with phi = ReLU, for
which L_LN = L_phi = 1 hold exactly and beta_l = (1-alpha) ||W||_2. Layer
normalization enters the manuscript bound only through its local Lipschitz
factor L_LN,l, which depends on the state and is a separate assumption.
"""
from __future__ import annotations

import math

import numpy as np

from ._util import norm_inf, norm_inf2


# ---------------------------------------------------------------------------- Proposition 2
def perturbation_recursion(beta, M, eps_op, R, e_ref, e0: float) -> np.ndarray:
    """e_{l+1} <= beta_l [e_l + M_l eps_op_l] + R_l e_ref_l, iterated from e_0."""
    beta, M, eps_op, R, e_ref = (np.asarray(v, dtype=np.float64) for v in (beta, M, eps_op, R, e_ref))
    out = np.empty(len(beta) + 1)
    out[0] = e0
    for l in range(len(beta)):
        out[l + 1] = beta[l] * (out[l] + M[l] * eps_op[l]) + R[l] * e_ref[l]
    return out


def unrolled_bound(beta, M, eps_op, R, e_ref, e0: float) -> float:
    """Product-sum form: e_L <= e_0 prod beta + sum_j (beta_j M_j eps_j + R_j e_ref_j) prod_{k>j} beta_k."""
    beta, M, eps_op, R, e_ref = (np.asarray(v, dtype=np.float64) for v in (beta, M, eps_op, R, e_ref))
    L = len(beta)
    total = e0 * float(np.prod(beta))
    for j in range(L):
        total += (beta[j] * M[j] * eps_op[j] + R[j] * e_ref[j]) * float(np.prod(beta[j + 1:]))
    return float(total)


def relu_retention_step(T: np.ndarray, H: np.ndarray, W: np.ndarray, b: np.ndarray,
                        alpha: float, H_ref: np.ndarray) -> np.ndarray:
    return (1 - alpha) * np.maximum(T @ H @ W + b, 0.0) + alpha * H_ref


def trajectory_tube(operators, perturbed, H0, H0_perturbed, W, b, alphas) -> dict:
    """Run both trajectories and the Proposition 2 recursion with the realized M_l.

    Returns the realized errors e_l = ||H_l - H~_l||_{inf,2}, the one-step
    right-hand sides and the unrolled bound; every realized step must satisfy
    its one-step inequality.
    """
    w2 = float(np.linalg.norm(W, 2))
    H, Ht = H0.copy(), H0_perturbed.copy()
    e = [norm_inf2(H - Ht)]
    rhs, betas, Ms, eps, Rs, refs = [], [], [], [], [], []
    e_ref = norm_inf2(H0 - H0_perturbed)
    for T, Tt, alpha in zip(operators, perturbed, alphas):
        beta = (1 - alpha) * w2
        M = norm_inf2(Ht)
        eps_op = norm_inf(T - Tt)
        rhs.append(beta * (e[-1] + M * eps_op) + alpha * e_ref)
        betas.append(beta); Ms.append(M); eps.append(eps_op); Rs.append(alpha); refs.append(e_ref)
        H = relu_retention_step(T, H, W, b, alpha, H0)
        Ht = relu_retention_step(Tt, Ht, W, b, alpha, H0_perturbed)
        e.append(norm_inf2(H - Ht))
    return {"errors": np.array(e), "one_step_rhs": np.array(rhs),
            "unrolled": unrolled_bound(betas, Ms, eps, Rs, refs, e[0]),
            "recursion": perturbation_recursion(betas, Ms, eps, Rs, refs, e[0]),
            "final_states": (H, Ht)}


def cross_entropy(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    return np.log(np.exp(z).sum(axis=1)) - z[np.arange(len(labels)), labels]


def logit_loss_bound(head: np.ndarray, e_L: float) -> float:
    """Frozen-head CE-with-logits change per node: at most sqrt(2) ||W_head||_2 e_L."""
    return float(np.sqrt(2.0) * np.linalg.norm(head, 2) * e_L)


# ---------------------------------------------------------------------------- Propositions 3 and 9
def score_gap(scores: np.ndarray) -> float:
    """m(s): gap between the best and second-best supported scores (infinite for one action)."""
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 1 or not len(s) or not np.isfinite(s).all():
        raise ValueError("scores must be a nonempty finite vector; empty support requires recovery")
    s = np.sort(s)
    return float("inf") if len(s) == 1 else float(s[1] - s[0])


def stable_by_gap(gap: float, lipschitz: float, distance: float) -> bool:
    """Proposition 3: the minimizer is unchanged whenever 2 L_J ||s - s~|| < m(s)."""
    if math.isnan(gap) or gap < 0 or not math.isfinite(lipschitz) or lipschitz < 0 \
            or not math.isfinite(distance) or distance < 0:
        raise ValueError("gap must be nonnegative; Lipschitz bound and distance must be finite and nonnegative")
    return 2.0 * lipschitz * distance < gap


def certified_ball(gap_at_center: float, lipschitz: float, radius: float) -> bool:
    """Proposition 9: one minimizer on the whole ball B(s_bar, r) when gap > 2 L_J r."""
    return stable_by_gap(gap_at_center, lipschitz, radius)


# ---------------------------------------------------------------------------- Proposition 8
def search_regret_bound(e_H: float, zeta: float) -> float:
    """J(a_hat) - min J <= 2 e_H + zeta for a zeta-approximate minimizer of a uniformly e_H-accurate J_hat."""
    return 2.0 * e_H + zeta
