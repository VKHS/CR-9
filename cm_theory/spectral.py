"""Cross-spectral quantities: windows, band powers and main Proposition 6.

For P = D^{-1} W the similarity P = D^{-1/2} (I - L) D^{1/2} with L = L_sym gives
T_a = D^{-1/2} g_a(L) D^{1/2}, where g_a(lambda) = sum_t c_t (1 - lambda)^t uses
the coefficients of :func:`cm_theory.operators.operator_polynomial`. Proposition 6
bounds the displacement between two actions band by band through the sensitivity
s_{aa'b} = max_{lambda in I_b} |g_a(lambda) - g_a'(lambda)| and the ideal band
power P_b(X) = ||1_{I_b}(L) X||_F^2 of the degree-weighted state X = D^{1/2} H.
"""
from __future__ import annotations

import math

import numpy as np

from ._util import fro, laplacian_eigensystem
from .operators import Action, operator_polynomial, transport_matrix


def response(coefficients: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """g(lambda) = sum_t c_t (1 - lambda)^t on eigenvalues of L_sym."""
    p = 1.0 - np.asarray(lam, dtype=np.float64)
    return np.polynomial.polynomial.polyval(p, coefficients)


def action_response(action: Action, lam: np.ndarray) -> np.ndarray:
    return response(operator_polynomial(action), lam)


def band_edges(windows: int = 6) -> list[tuple[float, float]]:
    return [(2.0 * b / windows, 2.0 * (b + 1) / windows) for b in range(windows)]


def band_membership(lam: np.ndarray, windows: int = 6) -> np.ndarray:
    """Band index of each eigenvalue for the equal partition of [0, 2] (last band closed)."""
    return np.minimum((np.asarray(lam) * windows / 2.0).astype(int), windows - 1)


# ---------------------------------------------------------------------------- Chebyshev windows
def chebyshev_window_coefficients(degree: int = 8, windows: int = 6) -> np.ndarray:
    """NumPy port of ``cross_morse.core.chebyshev_window_coefficients``.

    Equal-interval indicator coefficients on x = lambda - 1 with Jackson damping
    (angle pi/(degree+2), literal c0 convention). The runtime stores float32;
    this port keeps float64.
    """
    if degree < 1 or windows < 1:
        raise ValueError("positive polynomial degree and window count required")
    coeff = np.empty((windows, degree + 1), dtype=np.float64)
    angle = math.pi / (degree + 2)
    for b in range(windows):
        left, right = 2.0 * b / windows, 2.0 * (b + 1) / windows
        theta_left = math.acos(max(-1.0, min(1.0, left - 1.0)))
        theta_right = math.acos(max(-1.0, min(1.0, right - 1.0)))
        coeff[b, 0] = (theta_left - theta_right) / math.pi
        for k in range(1, degree + 1):
            raw = 2.0 * (math.sin(k * theta_left) - math.sin(k * theta_right)) / (math.pi * k)
            damping = ((degree - k + 2) * math.cos(k * angle) + math.sin(k * angle) / math.tan(angle)) / (degree + 2)
            coeff[b, k] = raw * damping
    return coeff


def window_response(coefficients: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """g_b(lambda) = sum_k c_k T_k(lambda - 1)."""
    x = np.clip(np.asarray(lam, dtype=np.float64) - 1.0, -1.0, 1.0)
    return np.polynomial.chebyshev.chebval(x, coefficients)


def window_matrix(adjacency: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """G_b = g_b(L_sym) evaluated through the eigendecomposition (dense reference)."""
    lam, U = laplacian_eigensystem(adjacency)
    return (U * window_response(coefficients, lam)) @ U.T


def window_recurrence(adjacency: np.ndarray, coefficients: np.ndarray, X: np.ndarray) -> np.ndarray:
    """G_b X by the runtime's three-term recurrence on L - I = -S (one product per degree)."""
    from ._util import symmetric_normalized
    S = symmetric_normalized(adjacency)
    t0 = X.astype(np.float64)
    out = coefficients[0] * t0
    t1 = -S @ t0
    out = out + coefficients[1] * t1
    for k in range(2, len(coefficients)):
        t2 = -2.0 * (S @ t1) - t0
        out = out + coefficients[k] * t2
        t0, t1 = t1, t2
    return out


def _grid_upper_bound(points: np.ndarray, values: np.ndarray, lipschitz: float) -> float:
    """Sample maximum plus a covering-radius remainder for a continuous scalar function.

    Between successive sorted samples every point is within half their gap of a
    sample. The mean-value theorem supplies a global upper bound when `lipschitz`
    bounds the absolute derivative. The final padding accounts for ordinary
    float64 arithmetic; this is not an interval-arithmetic certificate.
    """
    spacing = float(np.max(np.diff(points))) if len(points) > 1 else 0.0
    maximum = float(np.max(values))
    bound = maximum + lipschitz * spacing / 2.0
    return float(bound + 64 * np.finfo(float).eps * max(1.0, abs(bound), lipschitz))


def window_error(coefficients: np.ndarray, band: int, windows: int = 6, grid: int = 8193,
                 extra_points: np.ndarray | None = None) -> float:
    """Conservative bound for sup |1_{I_b}(lambda) - g_b(lambda)^2| on [0, 2].

    Each side of an indicator discontinuity is evaluated separately, including
    its one-sided limit. A global derivative bound for the squared Chebyshev
    polynomial supplies the grid-to-continuum remainder. The estimate therefore
    does not rely on supplying a tested graph's eigenvalues. Finite float64
    arithmetic remains separate from a rigorous interval certificate.
    """
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.ndim != 1 or not len(coefficients) or not np.isfinite(coefficients).all() \
            or not isinstance(windows, (int, np.integer)) or windows < 1 \
            or not isinstance(band, (int, np.integer)) or not 0 <= band < windows \
            or not isinstance(grid, (int, np.integer)) or grid < 2:
        raise ValueError("need finite coefficients, a valid band, and integer grid >= 2")
    extra = np.array([], dtype=float) if extra_points is None else np.asarray(extra_points, dtype=float)
    if extra.ndim != 1 or not np.isfinite(extra).all() or np.any((extra < 0) | (extra > 2)):
        raise ValueError("extra points must be a finite vector in [0, 2]")
    squared = np.polynomial.chebyshev.chebmul(coefficients, coefficients)
    # |T'_k(x)| <= k^2 for x in [-1,1], and x = lambda - 1.
    lipschitz = float(np.dot(np.arange(len(squared), dtype=float) ** 2, np.abs(squared)))
    lo, hi = band_edges(windows)[band]
    segments = [(lo, hi, 1.0)]
    if lo > 0:
        segments.append((0.0, lo, 0.0))
    if hi < 2:
        segments.append((hi, 2.0, 0.0))
    base = np.linspace(0.0, 2.0, grid)
    bounds = []
    for left, right, indicator in segments:
        points = np.unique(np.r_[left, right, base[(base >= left) & (base <= right)],
                                 extra[(extra >= left) & (extra <= right)]])
        values = np.abs(indicator - window_response(coefficients, points) ** 2)
        bounds.append(_grid_upper_bound(points, values, lipschitz))
    return max(bounds)


# ---------------------------------------------------------------------------- Proposition 6
def band_sensitivity(a: Action, b: Action, windows: int = 6, grid: int = 4097,
                     extra_points: np.ndarray | None = None) -> np.ndarray:
    """Conservative bandwise upper bounds on |g_a - g_b|.

    A sampled maximum is augmented by a derivative bound times half the largest
    grid gap. This covers off-grid extrema instead of labeling a sample maximum
    as the supremum. Bounds are numerical float64 references, not interval proofs.
    """
    if not isinstance(windows, (int, np.integer)) or windows < 1 \
            or not isinstance(grid, (int, np.integer)) or grid < 2:
        raise ValueError("windows and grid must be positive integers with grid >= 2")
    ca, cb = operator_polynomial(a), operator_polynomial(b)
    difference = np.polynomial.polynomial.polysub(ca, cb)
    # x = 1-lambda in [-1,1], so |d/dlambda sum c_t x^t| <= sum t |c_t|.
    lipschitz = float(np.dot(np.arange(len(difference)), np.abs(difference)))
    extra = np.array([], dtype=float) if extra_points is None else np.asarray(extra_points, dtype=float)
    if extra.ndim != 1 or not np.isfinite(extra).all() or np.any((extra < 0) | (extra > 2)):
        raise ValueError("extra points must be a finite vector in [0, 2]")
    out = np.zeros(windows)
    for k, (lo, hi) in enumerate(band_edges(windows)):
        lam = np.unique(np.r_[np.linspace(lo, hi, grid), extra[(extra >= lo) & (extra <= hi)]])
        out[k] = _grid_upper_bound(lam, np.abs(response(difference, lam)), lipschitz)
    return out


def displacement_bound(adjacency: np.ndarray, H: np.ndarray, a: Action, b: Action,
                       windows: int = 6) -> dict:
    """Main Proposition 6(i) for the transport part, in both frames.

    Returns the realized row-stochastic displacement ||(T_a - T_b) H||_F, its
    symmetric-frame counterpart ||(g_a - g_b)(L) X||_F with X = D^{1/2} H, the
    band-limited bound [sum_b s_b^2 P_b(X)]^{1/2} and the row-stochastic bound
    d_min^{-1/2} times that quantity.
    """
    d = adjacency.sum(axis=1)
    P = adjacency / d[:, None]
    lam, U = laplacian_eigensystem(adjacency)
    X = np.sqrt(d)[:, None] * H
    coordinates = U.T @ X
    energy = np.sum(coordinates ** 2, axis=1)
    diff = action_response(a, lam) - action_response(b, lam)
    s = band_sensitivity(a, b, windows)
    members = band_membership(lam, windows)
    powers = np.array([energy[members == k].sum() for k in range(windows)])
    symmetric_bound = float(np.sqrt(np.sum(s ** 2 * powers)))
    return {"displacement": fro((transport_matrix(P, a) - transport_matrix(P, b)) @ H),
            "symmetric_displacement": float(np.sqrt(np.sum(diff ** 2 * energy))),
            "symmetric_bound": symmetric_bound,
            "bound": symmetric_bound / math.sqrt(float(np.min(d))),
            "band_powers": powers, "sensitivities": s}


def alignment(GH: np.ndarray, GH0: np.ndarray, eps: float) -> float:
    """c_lb of Eq. (11) at the registered epsilon."""
    return float(np.sum(GH * GH0) / (np.linalg.norm(GH) * np.linalg.norm(GH0) + eps))


def alignment_identity_residual(G: np.ndarray, H: np.ndarray, H0: np.ndarray, eps: float = 1e-12) -> float:
    """Main Proposition 6(ii): ||G(H0-H)||^2 = ||GH0||^2 + ||GH||^2 - 2 c (||GH0|| ||GH|| + eps) exactly."""
    GH, GH0 = G @ H, G @ H0
    c = alignment(GH, GH0, eps)
    lhs = np.linalg.norm(G @ (H0 - H)) ** 2
    rhs = np.linalg.norm(GH0) ** 2 + np.linalg.norm(GH) ** 2 - 2 * c * (np.linalg.norm(GH0) * np.linalg.norm(GH) + eps)
    return float(abs(lhs - rhs) / max(1.0, lhs))


# ---------------------------------------------------------------------------- sketch
def sketch_relative_sd(X: np.ndarray, k: int) -> float:
    """Exact relative SD of ||X Omega||_F^2 as an estimate of ||X||_F^2, Omega ~ N(0, 1/k).

    With M = X^T X the estimator is unbiased with variance 2 tr(M^2)/k, so the
    relative SD is sqrt(2 tr(M^2)/k)/tr(M) <= sqrt(2/k) (equality iff rank one).
    """
    if k < 1:
        raise ValueError("sketch width must be positive")
    M = X.T @ X
    trace = float(np.trace(M))
    if trace == 0:
        return 0.0
    return float(math.sqrt(2.0 * float(np.sum(M * M)) / k) / trace)


def sketch_relative_sd_bound(k: int) -> float:
    return math.sqrt(2.0 / k)
