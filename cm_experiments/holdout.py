"""Bind arm evaluation to the holdout sealed before parent model fitting."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from cross_morse.split_contract import (
    DATA_ROLE_PROTOCOL, RERUN_GUIDANCE, require_reserved_comparison,
    role_identity, validate_reserved_roles,
)
from .artifacts import sha256

ARM_SCHEMA = "cm-budget-arm-experiment-3"
CAMPAIGN_SCHEMA = "cm-budget-matched-3"
HOLDOUT_SCHEMA = "cm-arm-comparison-holdout-1"


def comparison_contract(data, parent):
    """Read the existing sealed reservation; never create or split observations."""
    roles = require_reserved_comparison(data.metadata, data.splits)
    parent = Path(parent)
    preparation = parent / "provenance/stages/00_prepare.json"
    if not preparation.is_file():
        raise ValueError("The pre-fitting preparation manifest is missing. " + RERUN_GUIDANCE)
    return {
        "schema": HOLDOUT_SCHEMA, "evaluation_role": "compare", "label_role": "D_compare",
        "data_role_protocol": DATA_ROLE_PROTOCOL, "reserved_at": "prepare",
        "observation_count": len(roles["compare"]),
        "observation_ids_sha256": role_identity(roles["compare"]),
        "parent_data_fingerprint": data.metadata["fingerprint"],
        "base_prepare_manifest_sha256": sha256(preparation),
    }


def require_arm_contract(experiment, roles, *, results=None, observations=None):
    """Reject old/wrong-role studies before any scores can be accepted."""
    if experiment.get("schema") != ARM_SCHEMA or experiment.get("evaluation_role") != "D_compare":
        raise ValueError("Arm studies must evaluate the reserved D_compare role. " + RERUN_GUIDANCE)
    partition = validate_reserved_roles(roles)
    contract = experiment.get("comparison_holdout", {})
    if (contract.get("schema") != HOLDOUT_SCHEMA or contract.get("evaluation_role") != "compare" or
            contract.get("label_role") != "D_compare" or contract.get("data_role_protocol") != DATA_ROLE_PROTOCOL or
            contract.get("reserved_at") != "prepare" or
            contract.get("observation_count") != len(partition["compare"]) or
            contract.get("observation_ids_sha256") != role_identity(partition["compare"])):
        raise ValueError("Arm holdout does not match the preallocated comparison roles. " + RERUN_GUIDANCE)
    for field in ("parent_data_fingerprint", "base_prepare_manifest_sha256"):
        value = contract.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Missing parent preparation identity. " + RERUN_GUIDANCE)
    if results is not None and results.get("evaluation_role") != "D_compare":
        raise ValueError("Arm results were not evaluated on D_compare")
    if observations is not None:
        ids = np.asarray(observations)
        expected = partition["compare"]
        if (ids.ndim != 1 or ids.dtype.kind not in "iu" or len(np.unique(ids)) != len(ids) or
                not set(ids.tolist()) <= set(expected.tolist())):
            raise ValueError("Arm predictions contain observations outside D_compare")
        # Explicitly truncated software studies remain inspectable; campaigns
        # separately require full coverage and max_graphs == 0.
        if not experiment.get("max_graphs", 0) and not np.array_equal(np.sort(ids), np.sort(expected)):
            raise ValueError("Arm predictions do not cover the reserved D_compare holdout")
    return contract
