"""Training-only entry contract: main Eqs. (5) and (6).

``cross_morse.training`` computes DIL with ``np.trapezoid`` over log2 depths and
uses the feature-only reference q0 repeated on the depth grid as q_ref in the
DIL gap; :func:`log_depth_integrated_loss` restates Eq. (6) independently.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

ENTRY_THRESHOLDS = {"epsilon_plant_nats": 0.004, "tau_PDD_percentage_points": 1.0, "tau_DIL_nats": 0.020,
                    "tau_sup": 0.80, "epsilon_headroom_nats": 0.003}


def log_depth_integrated_loss(depths: Sequence[float], losses: Sequence[float]) -> float:
    """Eq. (6): trapezoid rule in log2 depth, normalized by log2(d_K / d_1)."""
    d = np.asarray(depths, dtype=np.float64)
    y = np.asarray(losses, dtype=np.float64)
    if d.shape != y.shape or len(d) < 2 or (d <= 0).any() or np.any(np.diff(d) <= 0):
        raise ValueError("need increasing positive depths with one loss each")
    x = np.log2(d)
    return float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(x)) / (x[-1] - x[0]))


def peak_depth_deterioration(metrics: Sequence[float]) -> float:
    """PDD_L = max_d M_d - M_L (the last entry is full depth)."""
    m = np.asarray(metrics, dtype=np.float64)
    return float(m.max() - m[-1])


def entry_contract(row: Mapping[str, float], thresholds: Mapping[str, float] = ENTRY_THRESHOLDS) -> dict:
    """Evaluate the Eq. (5) gates on one Supplementary Table S4 row."""
    gates = {
        "nominal_graph_value": row["U_Delta_plant"] < -thresholds["epsilon_plant_nats"],
        "peak_depth_deterioration": row["PDD_pp"] <= thresholds["tau_PDD_percentage_points"],
        "depth_integrated_gap": row["DIL_gap_nats"] <= thresholds["tau_DIL_nats"],
        "registry_support": row["phi_registry"] >= thresholds["tau_sup"],
        "selected_strata": row["selected_strata"] == 1.0,
        "residual_headroom": row["U_Delta_oracle"] < -thresholds["epsilon_headroom_nats"],
    }
    margins = {
        "nominal_graph_value": -thresholds["epsilon_plant_nats"] - row["U_Delta_plant"],
        "peak_depth_deterioration": thresholds["tau_PDD_percentage_points"] - row["PDD_pp"],
        "depth_integrated_gap": thresholds["tau_DIL_nats"] - row["DIL_gap_nats"],
        "registry_support": row["phi_registry"] - thresholds["tau_sup"],
        "residual_headroom": -thresholds["epsilon_headroom_nats"] - row["U_Delta_oracle"],
    }
    scale = {"nominal_graph_value": thresholds["epsilon_plant_nats"],
             "peak_depth_deterioration": thresholds["tau_PDD_percentage_points"],
             "depth_integrated_gap": thresholds["tau_DIL_nats"], "registry_support": thresholds["tau_sup"],
             "residual_headroom": thresholds["epsilon_headroom_nats"]}
    relative = {name: margins[name] / scale[name] for name in margins}
    return {"passes": all(gates.values()), "gates": gates, "margins": margins,
            "relative_margins": relative, "tightest_gate": min(relative, key=relative.get)}
