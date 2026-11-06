"""Inference the manuscript's Methods and Supplementary Table S42 rely on.

* two one-sided tests (TOST) for equivalence at a prespecified margin
  (registered: "equivalence claims use two one-sided tests at the prespecified
  margin"), from unit differences or from a printed estimate and interval;
* minimum detectable effect and power of a paired contrast, including the
  first Holm step of a declared family (2.80 and 3.34 standard errors);
* Holm step-down adjustment (independent restatement of the runtime rule);
* the two-way graph-seed variance decomposition of the main text,
  Var D_bar = sigma_u^2/J + sigma_v^2/S + sigma_e^2/(J S), estimated from a
  balanced [graphs x seeds] effect array (experiment E16, Note S24).

The crossed seed-by-unit bootstrap and paired sign-flip tests used for the
budget-matched arms already live in ``cm_campaigns.numeric.paired_effect``.
"""
from __future__ import annotations

import math
from statistics import NormalDist
from typing import Sequence

import numpy as np

_NORMAL = NormalDist()
_TINY = 1e-300


# ---------------------------------------------------------------------------- Student t (standard library)
def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Lentz evaluation of the continued fraction of the incomplete beta function."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > _TINY else _TINY)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > _TINY else _TINY)
        c = 1.0 + aa / c if abs(c) > _TINY else 1.0 + aa / _TINY
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > _TINY else _TINY)
        c = 1.0 + aa / c if abs(c) > _TINY else 1.0 + aa / _TINY
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            return h
    raise ArithmeticError("incomplete beta continued fraction did not converge")


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """I_x(a, b) for a, b > 0 and 0 <= x <= 1."""
    if a <= 0 or b <= 0 or not 0.0 <= x <= 1.0:
        raise ValueError("need a, b > 0 and 0 <= x <= 1")
    if x in (0.0, 1.0):
        return x
    log_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                 + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(log_front) * _beta_continued_fraction(a, b, x) / a
    return 1.0 - math.exp(log_front) * _beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_sf(x: float, df: float) -> float:
    """Upper tail Pr(T_df > x); the normal tail when df is infinite."""
    if not math.isfinite(df):
        return 1.0 - _NORMAL.cdf(x)
    if df <= 0:
        raise ValueError("degrees of freedom must be positive")
    tail = 0.5 * regularized_incomplete_beta(df / 2.0, 0.5, df / (df + x * x))
    return tail if x >= 0 else 1.0 - tail


def student_t_quantile(p: float, df: float) -> float:
    """Inverse of the Student t distribution function by bracketed bisection."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie strictly between zero and one")
    if not math.isfinite(df):
        return _NORMAL.inv_cdf(p)
    if p < 0.5:
        return -student_t_quantile(1.0 - p, df)
    lo, hi = 0.0, max(1.0, _NORMAL.inv_cdf(p))
    while 1.0 - student_t_sf(hi, df) < p:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 1.0 - student_t_sf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-13 * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


def _t_sf(x: float, df: float) -> float:
    return student_t_sf(x, df)


def z_quantile(p: float) -> float:
    return _NORMAL.inv_cdf(p)


# ---------------------------------------------------------------------------- Holm
def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted P values in the original contrast order."""
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("P values must be a nonempty vector in [0, 1]")
    order = np.argsort(p, kind="stable")
    multipliers = len(p) - np.arange(len(p))
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(multipliers * p[order]))
    return adjusted


# ---------------------------------------------------------------------------- TOST
def tost(differences: Sequence[float], margin: float, alpha: float = 0.05) -> dict:
    """Paired TOST of |mu| < margin with the Student-t reference on n - 1 degrees of freedom.

    H0_low: mu <= -margin and H0_high: mu >= margin; equivalence is declared when
    both are rejected at alpha, i.e. when max(p_low, p_high) < alpha.
    """
    d = np.asarray(differences, dtype=np.float64)
    if d.ndim != 1 or len(d) < 2 or margin <= 0:
        raise ValueError("TOST needs at least two paired differences and a positive margin")
    n = len(d)
    se = float(np.std(d, ddof=1) / math.sqrt(n))
    return tost_from_summary(float(d.mean()), se, margin, alpha, df=n - 1)


def tost_from_summary(estimate: float, se: float, margin: float, alpha: float = 0.05,
                      df: float = math.inf) -> dict:
    if se <= 0 or margin <= 0:
        raise ValueError("TOST needs a positive standard error and margin")
    p_low = _t_sf((estimate + margin) / se, df)
    p_high = _t_sf((margin - estimate) / se, df)
    p = max(p_low, p_high)
    return {"estimate": estimate, "se": se, "margin": margin, "df": df, "p_lower": p_low,
            "p_upper": p_high, "p_tost": p, "equivalent": bool(p < alpha), "alpha": alpha,
            "reference": "Student t" if math.isfinite(df) else "normal"}


def tost_from_interval(estimate: float, lower: float, upper: float, margin: float,
                       level: float = 0.95, alpha: float = 0.05) -> dict:
    """Normal-approximation TOST from a printed central interval (se = width / (2 z))."""
    if not lower <= estimate <= upper:
        raise ValueError("the estimate must lie inside its interval")
    z = z_quantile(0.5 + level / 2)
    result = tost_from_summary(estimate, (upper - lower) / (2 * z), margin, alpha)
    result["interval_inside_margin"] = bool(-margin < lower and upper < margin)
    return result


# ---------------------------------------------------------------------------- power
def critical_value(alpha: float = 0.05, family_size: int = 1, holm_step: int = 1, two_sided: bool = True) -> float:
    """z critical value at the Holm step's level alpha / (m - step + 1)."""
    if not 1 <= holm_step <= family_size:
        raise ValueError("holm_step must lie in 1..family_size")
    level = alpha / (family_size - holm_step + 1)
    return z_quantile(1 - level / 2) if two_sided else z_quantile(1 - level)


def minimum_detectable_effect(se: float = 1.0, alpha: float = 0.05, power: float = 0.80,
                              family_size: int = 1, holm_step: int = 1, two_sided: bool = True) -> float:
    """(z_{1-a'} + z_power) se: 2.80 se at two-sided 0.05; 3.34 se at the first Holm step of four."""
    return (critical_value(alpha, family_size, holm_step, two_sided) + z_quantile(power)) * se


def power_of_contrast(effect: float, se: float, alpha: float = 0.05, family_size: int = 1,
                      holm_step: int = 1, two_sided: bool = True) -> float:
    """Normal power Phi(|effect|/se - z) plus the (negligible) opposite tail for two-sided tests."""
    z = critical_value(alpha, family_size, holm_step, two_sided)
    shift = abs(effect) / se
    tail = _NORMAL.cdf(-z - shift) if two_sided else 0.0
    return float(_NORMAL.cdf(shift - z) + tail)


# ---------------------------------------------------------------------------- graph-seed hierarchy
def variance_components(effects: np.ndarray) -> dict:
    """Two-way random-effects decomposition of a balanced [graphs J x seeds S] paired-effect array.

    D_js = mu + u_j + v_s + e_js. Mean squares give unbiased components
    sigma_u^2 = (MS_graph - MS_e)/S, sigma_v^2 = (MS_seed - MS_e)/J,
    sigma_e^2 = MS_e, truncated at zero for reporting, and the exact identity
    Var_hat_unbiased(D_bar) = (MS_graph + MS_seed - MS_e) / (J S). This
    untruncated moment estimate can be negative. Reporting fields var_mean,
    se_mean and seed_share consistently use the nonnegative component estimates;
    they are not unbiased. The signed estimate is retained as var_mean_unbiased.
    """
    D = np.asarray(effects, dtype=np.float64)
    if D.ndim != 2 or min(D.shape) < 2 or not np.isfinite(D).all():
        raise ValueError("need a finite [graphs, seeds] array with at least two of each")
    J, S = D.shape
    grand = D.mean()
    rows, cols = D.mean(axis=1), D.mean(axis=0)
    ms_graph = S * np.sum((rows - grand) ** 2) / (J - 1)
    ms_seed = J * np.sum((cols - grand) ** 2) / (S - 1)
    residual = D - rows[:, None] - cols[None, :] + grand
    ms_e = np.sum(residual ** 2) / ((J - 1) * (S - 1))
    su, sv, se = (ms_graph - ms_e) / S, (ms_seed - ms_e) / J, ms_e
    var_mean_unbiased = su / J + sv / S + se / (J * S)
    var_mean = max(su, 0.0) / J + max(sv, 0.0) / S + se / (J * S)
    return {"graphs": J, "seeds": S, "mean": float(grand),
            "sigma_u2": float(max(su, 0.0)), "sigma_v2": float(max(sv, 0.0)), "sigma_e2": float(se),
            "sigma_u2_unbiased": float(su), "sigma_v2_unbiased": float(sv),
            "var_mean": float(var_mean), "se_mean": float(math.sqrt(var_mean)),
            "var_mean_unbiased": float(var_mean_unbiased),
            "se_mean_unbiased": float(math.sqrt(var_mean_unbiased)) if var_mean_unbiased >= 0 else None,
            "variance_reporting": "nonnegative component estimates; not unbiased",
            "seed_share": float(max(sv, 0.0) / S / var_mean) if var_mean > 0 else None,
            "mean_squares": {"graph": float(ms_graph), "seed": float(ms_seed), "residual": float(ms_e)}}


def variance_of_mean(sigma_u2: float, sigma_v2: float, sigma_e2: float, J: int, S: int) -> float:
    """Main text: Var D_bar = sigma_u^2/J + sigma_v^2/S + sigma_e^2/(J S)."""
    return sigma_u2 / J + sigma_v2 / S + sigma_e2 / (J * S)
