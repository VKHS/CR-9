"""Two-level factorial analysis of Supplementary Tables S16 and S16b.

Cells are keyed by strings such as ``"101"`` (factor j is "on" when character j is
"1"). Standard effect coding reports each effect as the difference between the
mean of the cells where the product of the involved factors' +-1 codes is +1 and
the mean where it is -1. Baseline-anchored finite differences are the
inclusion-exclusion contrasts from the all-off cell, e.g.
I_AO = y_110 - y_100 - y_010 + y_000.
"""
from __future__ import annotations

from itertools import combinations
import math
from typing import Mapping

import numpy as np


def _validate(cells: Mapping[str, float]) -> int:
    keys = list(cells)
    k = len(keys[0]) if keys else 0
    if k < 1 or len(keys) != 2 ** k or any(len(key) != k or set(key) - {"0", "1"} for key in keys):
        raise ValueError("cells must cover every 0/1 string of one length")
    return k


def effect(cells: Mapping[str, float], factors: tuple[int, ...]) -> float:
    """Effect-coded main effect or interaction of the given factor positions."""
    k = _validate(cells)
    plus, minus = [], []
    for key, value in cells.items():
        sign = 1
        for j in factors:
            sign *= 1 if key[j] == "1" else -1
        (plus if sign > 0 else minus).append(float(value))
    if len(plus) != 2 ** (k - 1):
        raise ValueError("balanced design required")
    return float(np.mean(plus) - np.mean(minus))


def all_effects(cells: Mapping[str, float]) -> dict[tuple[int, ...], float]:
    k = _validate(cells)
    return {factors: effect(cells, factors) for order in range(1, k + 1) for factors in combinations(range(k), order)}


def anchored_difference(cells: Mapping[str, float], factors: tuple[int, ...]) -> float:
    """Inclusion-exclusion finite difference over the named factors, other factors off."""
    k = _validate(cells)
    total = 0.0
    for r in range(len(factors) + 1):
        for subset in combinations(factors, r):
            key = "".join("1" if j in subset else "0" for j in range(k))
            total += (-1) ** (len(factors) - r) * float(cells[key])
    return float(total)


def full_minus_none(cells: Mapping[str, float]) -> float:
    k = _validate(cells)
    return float(cells["1" * k]) - float(cells["0" * k])


def effect_half_width(cell_half_widths: Mapping[str, float], z: float = 1.959963984540054) -> float:
    """Half-width of any effect under independent cell estimates.

    An effect is (sum of 2^{k-1} cells - sum of 2^{k-1} cells) / 2^{k-1}, so
    SE = sqrt(sum_c SE_c^2) / 2^{k-1} with SE_c = half_width_c / z.
    """
    k = _validate(cell_half_widths)
    se = math.sqrt(sum((float(h) / z) ** 2 for h in cell_half_widths.values())) / 2 ** (k - 1)
    return z * se
