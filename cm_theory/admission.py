"""Independent-graph admission (Theorem 2, Corollary 2) and one-step regret (Proposition 7).

Theorem 2 is the Maurer-Pontil empirical-Bernstein bound on [0, 1] rescaled to
clipped paired differences on [-B, B] (range 2B):

    U_a = D_bar + sqrt(2 S^2 log(2/delta)/J) + 14 B log(2/delta) / (3 (J - 1)).

``cross_morse.statistics.empirical_bernstein`` executes the same bound in the
runtime; these functions take summary statistics so the manuscript's printed
moments can be checked directly.
"""
from __future__ import annotations

import math
from numbers import Integral

import numpy as np


# ---------------------------------------------------------------------------- Theorem 2
def bernstein_terms(S2: float, J: int, B: float, delta: float) -> dict:
    if isinstance(J, bool) or not isinstance(J, Integral) or J < 2 or not math.isfinite(S2) or S2 < 0 \
            or not math.isfinite(B) or B <= 0 or not math.isfinite(delta) or not 0 < delta < 1:
        raise ValueError("Theorem 2 needs J >= 2, S^2 >= 0, B > 0 and 0 < delta < 1")
    log_term = math.log(2.0 / delta)
    variance = math.sqrt(2.0 * S2 * log_term / J)
    fixed = 14.0 * B * log_term / (3.0 * (J - 1))
    return {"variance_term": variance, "fixed_term": fixed, "penalty": variance + fixed}


def admission_upper(D_bar: float, S2: float, J: int, B: float = 0.10, delta: float = 0.025) -> float:
    if not math.isfinite(D_bar):
        raise ValueError("the clipped paired mean must be finite")
    return float(D_bar + bernstein_terms(S2, J, B, delta)["penalty"])


def admission_upper_from_units(differences: np.ndarray, B: float = 0.10, delta: float = 0.025) -> dict:
    """Clip each paired unit difference to [-B, B], then apply Eq. (27).

    A unit counts as clipped when |D_j| > B, the convention of
    ``cross_morse.statistics.empirical_bernstein``.
    """
    raw = np.asarray(differences, dtype=np.float64)
    if raw.ndim != 1 or len(raw) < 2 or not np.isfinite(raw).all():
        raise ValueError("admission requires a finite one-dimensional vector of at least two paired units")
    d = np.clip(raw, -B, B)
    J = len(d)
    S2 = float(np.var(d, ddof=1))
    return {"D_bar": float(d.mean()), "S2": S2, "J": J,
            "U_a": admission_upper(float(d.mean()), S2, J, B, delta),
            "clipped_fraction": float(np.mean(np.abs(raw) > B))}


def clipped_fraction_bound(D_bar: float, S2: float, J: int, B: float) -> float:
    """f B^2 <= mean(D_j^2) = (J-1) S^2 / J + D_bar^2, since every clipped unit has |D_j| = B."""
    return float(((J - 1) * S2 / J + D_bar ** 2) / B ** 2)


# ---------------------------------------------------------------------------- Corollary 2
def corollary2_constant(B: float = 0.10, delta: float = 0.025) -> float:
    return 14.0 * B * math.log(2.0 / delta) / 3.0


def minimum_admission_threshold(D_bar: float, epsilon_a: float, B: float = 0.10, delta: float = 0.025) -> float:
    """Necessary zero-variance threshold: J >= 1 + 14 B log(2/delta)/(3 (|D_bar|-epsilon_a)).

    Equality can admit when S^2 = 0; the manuscript prints a strict inequality,
    which is necessary only when the variance penalty is positive. This is not
    a sufficient admission test when S^2 > 0: use the full admission_upper.
    """
    bernstein_terms(0.0, 2, B, delta)  # validate the common bound parameters
    if not math.isfinite(D_bar) or not math.isfinite(epsilon_a) or epsilon_a < 0:
        raise ValueError("D_bar and nonnegative epsilon_a must be finite")
    gap = abs(D_bar) - epsilon_a
    if D_bar >= 0 or gap <= 0:
        return math.inf
    return 1.0 + corollary2_constant(B, delta) / gap


def minimum_admission_units(D_bar: float, epsilon_a: float, B: float = 0.10, delta: float = 0.025) -> int | float:
    """Smallest integer allowed by the zero-variance necessary condition (not a sufficient test)."""
    threshold = minimum_admission_threshold(D_bar, epsilon_a, B, delta)
    return math.inf if math.isinf(threshold) else max(2, int(math.ceil(threshold)))


# ---------------------------------------------------------------------------- Proposition 7
def hoeffding_radius(n: int, B: float = 0.10, actions: int = 10, delta: float = 0.05) -> float:
    """e(n) = 2B sqrt(log(2|A|/delta) / (2n)): simultaneous over |A| actions, two-sided."""
    if n < 1:
        raise ValueError("n must be positive")
    return 2.0 * B * math.sqrt(math.log(2.0 * actions / delta) / (2.0 * n))


def bernstein_radius(n: int, sigma: float, B: float = 0.10, actions: int = 10, delta: float = 0.05) -> float:
    """e_B(n) = sqrt(2 sigma^2 log(4|A|/delta)/n) + 7 (2B) log(4|A|/delta) / (3 (n-1))."""
    if n < 2:
        raise ValueError("n must be at least two")
    log_term = math.log(4.0 * actions / delta)
    return math.sqrt(2.0 * sigma ** 2 * log_term / n) + 7.0 * (2.0 * B) * log_term / (3.0 * (n - 1))


def regret_bound(radius: float) -> float:
    """Proposition 7: Delta(pi_one) <= min_a Delta(a) + 2 max_a e(a) (<= 2 max e when A0 is supported)."""
    return 2.0 * radius


def units_for_hoeffding_regret(target: float, B: float = 0.10, actions: int = 10, delta: float = 0.05) -> int:
    """Smallest n with 2 e(n) <= target: n >= 8 B^2 log(2|A|/delta) / target^2."""
    return int(math.ceil(8.0 * B ** 2 * math.log(2.0 * actions / delta) / target ** 2))


def bernstein_crossover(sigma: float, B: float = 0.10, actions: int = 10, delta: float = 0.05,
                        n_max: int = 10 ** 6) -> int:
    """Smallest n for which the empirical-Bernstein radius falls below the Hoeffding radius."""
    for n in range(2, n_max):
        if bernstein_radius(n, sigma, B, actions, delta) < hoeffding_radius(n, B, actions, delta):
            return n
    raise ValueError("no crossover below n_max")


def units_for_bernstein_regret(target: float, sigma: float, B: float = 0.10, actions: int = 10,
                               delta: float = 0.05, n_max: int = 10 ** 7) -> int:
    """Smallest n with 2 e_B(n) <= target (bisection on the monotone radius)."""
    lo, hi = 2, n_max
    if 2 * bernstein_radius(hi, sigma, B, actions, delta) > target:
        raise ValueError("target not reached below n_max")
    while lo < hi:
        mid = (lo + hi) // 2
        if 2 * bernstein_radius(mid, sigma, B, actions, delta) <= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


# ---------------------------------------------------------------------------- support identity (Eq. 7)
def effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64)
    return float(w.sum() ** 2 / np.sum(w ** 2))


def max_normalized_weight(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64)
    return float(w.max() / w.sum())


def weight_cap_from_ess(ess: float) -> float:
    """omega_max <= 1/sqrt(ESS) on the same weight vector, so ESS >= 64 forces omega_max <= 0.125."""
    return 1.0 / math.sqrt(ess)
