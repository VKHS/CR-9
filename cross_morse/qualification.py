"""Training-only support qualification for the controller entry contract.

Main Eq. (5) and Supplementary Table S3 require selected-strata support of
1.00. Table S35 convention C28 supplies the operational definition used by
the S4 reconstruction: A0 belongs to the supported set of every registered
stratum. These records describe the full controller's D_explore model-fit
support; they do not inspect selection, confirmation, admission or test labels.
"""
from __future__ import annotations


SELECTED_STRATA_SCHEMA = "cross-morse-selected-strata-support-1"
SELECTED_STRATA_FRACTION = 1.0
NOMINAL_ACTION_ID = 0


def selected_strata_support(support: dict) -> dict:
    """Audit A0 support without recomputing or weakening cell qualification.

    ``support`` is the full-profile audit emitted by ``learning._support``.
    Its cell qualification already applies the propensity, effective-sample-
    size and maximum-normalized-weight thresholds. Missing nominal cells fail
    the gate, and an empty set of registered strata has an undefined fraction
    and never passes. Malformed identifiers or non-boolean qualifications are
    rejected rather than silently accepted by Python's truth-value coercion.

    The caller records whether the entry contract was enforced. This helper
    only evaluates it; it must not be used to claim that an older run enforced
    a gate absent from that run's original, source-bound entry record.
    """
    if not isinstance(support, dict):
        raise ValueError("selected-strata support must be an audit object")
    strata = support.get("registered_strata")
    if (not isinstance(strata, list)
            or any(type(s) is not int or s < 0 for s in strata)
            or len(set(strata)) != len(strata)):
        raise ValueError("registered_strata must contain unique nonnegative integer IDs")
    cells = support.get("cells")
    if not isinstance(cells, dict):
        raise ValueError("selected-strata support cells must be an object")
    strata = sorted(strata)
    qualified, failed, missing = [], [], []
    for stratum in strata:
        key = f"{stratum}:{NOMINAL_ACTION_ID}"
        if key not in cells:
            missing.append(key)
            failed.append(stratum)
            continue
        cell = cells[key]
        if not isinstance(cell, dict) or type(cell.get("qualified")) is not bool:
            raise ValueError(f"support cell {key} must have a boolean qualified flag")
        for field, expected in (("stratum", stratum), ("action", NOMINAL_ACTION_ID)):
            if field in cell and (type(cell[field]) is not int or cell[field] != expected):
                raise ValueError(f"support cell {key} has an inconsistent {field}")
        (qualified if cell["qualified"] else failed).append(stratum)
    fraction = len(qualified) / len(strata) if strata else None
    return {
        "schema": SELECTED_STRATA_SCHEMA,
        "profile": "full",
        "support_source": "fit/full/audit.json#/support",
        "data_role": "D_explore model-fit trajectories",
        "definition": "fraction of registered strata with qualified A0; Supplementary Table S35, C28",
        "nominal_action": NOMINAL_ACTION_ID,
        "required_fraction": SELECTED_STRATA_FRACTION,
        "observed_fraction": fraction,
        "registered_strata": strata,
        "qualified_strata": qualified,
        "failed_strata": failed,
        "missing_cells": missing,
        "passed": fraction == SELECTED_STRATA_FRACTION,
    }
