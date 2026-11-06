"""Predeclared label roles shared by preparation and independent arm checks.

The comparison holdout is allocated before any predictor or controller is fit.
It is not a subdivision of an already inspected confirmation or admission set.
This module has no model/checkpoint dependency so archive checks can use it.
"""
from __future__ import annotations

import hashlib
import numpy as np

PREPARED_SCHEMA_VERSION = 3
DATA_ROLE_PROTOCOL = "fit60-explore20-select10-confirm5-compare5-v1"
TRAIN_CUTS = (0.60, 0.80, 0.90, 0.95)
ROLE_NAMES = ("fit_train", "fit_val", "fit", "explore", "select", "confirm", "compare", "admit", "test")
LEAF_ROLES = ("fit_train", "fit_val", "explore", "select", "confirm", "compare", "admit", "test")
RERUN_GUIDANCE = (
    "An independent D_compare holdout must be reserved during preparation, before fitting. "
    "Rerun preparation and the complete lifecycle in a new output directory with the "
    "60/20/10/5/5 protocol; never repartition a previously confirmed run."
)


def validate_reserved_roles(splits, *, size=None):
    """Require nonempty disjoint leaves and an exact, unmodified fit union."""
    if "compare" not in splits:
        raise ValueError(RERUN_GUIDANCE)
    if set(splits) != set(ROLE_NAMES):
        raise ValueError("Unexpected prepared roles. " + RERUN_GUIDANCE)
    rows = {}
    for role in ROLE_NAMES:
        values = np.asarray(splits[role])
        if values.ndim != 1 or values.dtype.kind not in "iu" or not len(values):
            raise ValueError(f"Label role {role} must contain a nonempty integer index vector")
        if np.any(values < 0) or len(np.unique(values)) != len(values):
            raise ValueError(f"Label role {role} contains invalid or duplicate indices")
        rows[role] = values.astype(np.int64, copy=False)
    leaves = np.concatenate([rows[role] for role in LEAF_ROLES])
    if len(np.unique(leaves)) != len(leaves):
        raise ValueError("Comparison and other leaf label roles overlap")
    expected_size = len(leaves) if size is None else int(size)
    if len(leaves) != expected_size or not np.array_equal(np.sort(leaves), np.arange(expected_size)):
        raise ValueError("Prepared leaf roles do not cover the complete observation index")
    if not np.array_equal(np.sort(rows["fit"]), np.sort(np.r_[rows["fit_train"], rows["fit_val"]])):
        raise ValueError("fit must be exactly fit_train plus fit_val")
    return rows


def role_identity(indices):
    """Order-sensitive identity of an observation-ID vector in a portable type."""
    values = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def require_reserved_comparison(metadata, splits):
    if (metadata.get("schema_version") != PREPARED_SCHEMA_VERSION or
            metadata.get("data_role_protocol") != DATA_ROLE_PROTOCOL or
            metadata.get("comparison_holdout_reserved_at") != "prepare"):
        raise ValueError(RERUN_GUIDANCE)
    return validate_reserved_roles(splits, size=metadata.get("num_samples"))
