"""Source-declared analytical inputs for Tables S28/S28b and Figure S17g.

The PDF's reference counts and effect scales define a mathematical scenario.
They are separate from support minima and effects measured in a lifecycle run.
Only input coordinates are read here; bounds and required counts are computed
with ``cm_theory.admission`` and never read from the printed-value targets.
"""
from __future__ import annotations

import math

from cm_theory import admission, operators

from .framework import DATASETS, DisplayInputError, DisplayValidationError

S28_INPUTS = "regret_bound_table_S28"
S28B_INPUTS = "regret_bound_table_S28b"
S28_TARGETS = "supplementary_table_S28_one_step_regret_bound"
S28B_TARGETS = "supplementary_table_S28b_variance_aware_regret_bound"


def _group(ctx, name):
    group = ctx.constants.get(name)
    if not isinstance(group, dict):
        raise DisplayInputError(f"registered constant {name} is required for the regret reference scenario")
    return group


def _positive(value, name, *, integer=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or value <= 0 or (integer and int(value) != value)):
        raise DisplayValidationError(f"{name} must be a finite positive {'integer' if integer else 'number'}")
    return int(value) if integer else float(value)


def parameters(ctx):
    """B and |A| from the implementation; delta_sup from SI Table S28."""
    group = _group(ctx, S28_INPUTS)
    if "delta_sup" not in group:
        raise DisplayInputError(f"registered constant {S28_INPUTS}.delta_sup is required")
    delta = _positive(group["delta_sup"], f"{S28_INPUTS}.delta_sup")
    if delta >= 1:
        raise DisplayValidationError("the regret scenario delta_sup must lie strictly between zero and one")
    B = _positive(ctx.constants["admission_eq27_theorem2_table_S13"]["B"], "admission clipping B")
    actions = len(operators.load_registry())
    if actions < 1:
        raise DisplayValidationError("the regret scenario needs at least one runtime action")
    return {"B": B, "actions": actions, "delta": delta}


def reference_support(ctx):
    """The listed S28 scenario, not a reduction over observed support cells."""
    rows = _group(ctx, S28_INPUTS).get("reference_support")
    if not isinstance(rows, dict):
        raise DisplayInputError(f"registered constant {S28_INPUTS}.reference_support is required")
    if set(rows) != set(DATASETS):
        raise DisplayValidationError("the Table S28 reference scenario must contain exactly the seven datasets")
    result = {}
    for dataset in DATASETS:
        row = rows[dataset]
        if not isinstance(row, dict) or set(row) != {"minimum_count", "reference_ess"}:
            raise DisplayValidationError(f"{dataset}: the Table S28 reference count and ESS must both be declared")
        result[dataset] = {key: _positive(value, f"{dataset}.{key}", integer=True) for key, value in row.items()}
    return result


def variance_grid(ctx):
    """Explicit abscissas and target scales of the two blocks of SI Table S28b."""
    group = _group(ctx, S28B_INPUTS)
    for key in ("sigma1_grid", "count_grid", "effect_targets"):
        if key not in group:
            raise DisplayInputError(f"registered constant {S28B_INPUTS}.{key} is required")
    if not isinstance(group["sigma1_grid"], list) or len(group["sigma1_grid"]) != 3:
        raise DisplayValidationError("Table S28b requires its three declared sigma_1 values")
    sigmas = [_positive(v, "sigma_1") for v in group["sigma1_grid"]]
    if sigmas != sorted(set(sigmas)):
        raise DisplayValidationError("Table S28b sigma_1 values must be distinct and increasing")
    if not isinstance(group["count_grid"], list) or not group["count_grid"]:
        raise DisplayValidationError("Table S28b count_grid must be a nonempty list")
    counts = [_positive(v, "reference count", integer=True) for v in group["count_grid"]]
    if min(counts) < 2 or counts != sorted(set(counts)):
        raise DisplayValidationError("Table S28b counts must be distinct increasing integers of at least two")
    if not isinstance(group["effect_targets"], list) or not group["effect_targets"]:
        raise DisplayValidationError("Table S28b effect_targets must be a nonempty list")
    targets = []
    for row in group["effect_targets"]:
        if (not isinstance(row, dict) or set(row) != {"id", "label", "target_nats"}
                or not isinstance(row["id"], str) or not row["id"]
                or not isinstance(row["label"], str) or not row["label"].strip()):
            raise DisplayValidationError("Table S28b targets need an id, label and positive target_nats")
        targets.append({**row, "target_nats": _positive(row["target_nats"], f"{row['id']}.target_nats")})
    if len({t["id"] for t in targets}) != len(targets) or targets[0]["id"] != "smallest_admitted_effect":
        raise DisplayValidationError("Table S28b targets must be unique and start with smallest_admitted_effect")
    if targets[0]["target_nats"] != min(t["target_nats"] for t in targets):
        raise DisplayValidationError("the Table S28b smallest analytical target must be the smallest declared scale")
    return sigmas, counts, targets


def hoeffding_equality(target, params):
    """Real n at equality. Printed tables round this; a sufficient integer uses ceil."""
    target = _positive(target, "regret target")
    return 8 * params["B"] ** 2 * math.log(2 * params["actions"] / params["delta"]) / target ** 2


def target_requirements(target, sigmas, params):
    real_n = hoeffding_equality(target, params)
    return {"hoeffding_n_at_equality": real_n, "hoeffding_n_rounded": round(real_n),
            "hoeffding_minimum_integer_n": admission.units_for_hoeffding_regret(target, **params),
            "bernstein_n": [admission.units_for_bernstein_regret(target, sigma, **params) for sigma in sigmas]}


def record_parameters(out, params, *, prefix="", variance=False):
    out.data(f"{prefix}bound_parameters", ["B_nats", "action_count", "delta_sup"],
             [[params["B"], params["actions"], params["delta"]]])
    groups = [S28_INPUTS, "admission_eq27_theorem2_table_S13.B"]
    if variance:
        groups.append(S28B_INPUTS)
    out.uses("repository", "paper/registered_constants.json", groups)
    out.uses("repository", "cross_morse/core.py", ["ACTIONS"])
