#!/usr/bin/env python3
"""Check printed mechanism contrasts against the registered confirmation rule.

The report separates the policy printed in Table 1 from the registry's canonical
lifecycle resolution. It checks the available contrasts at the registered alpha
and practical margins; it does not reconstruct selection, predecessor eligibility
or admission from rounded manuscript values. Those need the original run records.
A retained route cannot be authorized by a failed contrast. A fallback predecessor
cannot be declared eligible merely because its name differs from the failed route.

Default exit: 0 consistent, 1 discrepancy or undetermined, 2 invalid input.
--warn-only makes findings advisory, while invalid input still exits 2.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUTE_CONTRAST = {"radius": "radius minus one-step", "MPC-RH": "MPC-RH minus MPC-OL"}
PREDECESSOR = {"radius": "one-step", "MPC-RH": "MPC-OL"}


def parse_p(value) -> float:
    """Return a numeric P value, or the upper bound in '< ...'; reject invalid values."""
    if isinstance(value, bool):
        raise ValueError("P value must be numeric, not boolean")
    text = str(value).strip().replace("−", "-")
    match = re.fullmatch(r"<\s*([0-9.eE+-]+)", text)
    number = float(match.group(1) if match else text)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"P value is outside [0, 1]: {value!r}")
    return number


def policy(text) -> str | None:
    if not isinstance(text, str):
        return None
    text = text.strip()
    if text in {"same phase", "return_qC", "qC", "q_C", "qC returned", "q_C returned"}:
        return "qC"
    match = re.fullmatch(r"(radius|MPC-RH|MPC-OL|one-step)(?:/\d+)?", text)
    return match.group(1) if match else None


def provisional_route(text: str) -> str | None:
    return policy(text)


def contrast_result(contrast: dict, *, margin: float, alpha: float, beneficial_sign: int) -> dict:
    effect, printed_margin = float(contrast["effect"]), float(contrast["margin"])
    if not all(math.isfinite(x) for x in (effect, printed_margin, margin)) or min(printed_margin, margin) < 0:
        raise ValueError("Effects and margins must be finite; margins must be nonnegative")
    p = parse_p(contrast["holm_P"])
    # Match the strict p < alpha rule in statistics.confirmation and the
    # canonical paper view. A printed '< alpha' proves that strict inequality;
    # a numeric p == alpha does not. A bound above alpha is undetermined.
    bounded = isinstance(contrast["holm_P"], str) and contrast["holm_P"].strip().startswith("<")
    significant = (True if p <= alpha else None) if bounded else p < alpha
    beyond = beneficial_sign * effect > margin
    passed = False if not beyond or significant is False else True if significant is True else None
    return {"kind": contrast["kind"], "effect": effect, "margin": margin, "printed_margin": printed_margin,
            "margin_matches_registered": math.isclose(printed_margin, margin, rel_tol=0, abs_tol=1e-12),
            "holm_P": contrast["holm_P"], "significant_at_alpha": significant,
            "beyond_margin": beyond, "confirmation_pass": passed}


def evaluate(registry: dict, constants: dict | None = None) -> dict:
    if constants is None:
        constants = json.loads((ROOT / "paper" / "registered_constants.json").read_text(encoding="utf-8"))
    rule = constants["confirmation_eq20_table_S11"]
    alpha = float(rule["holm_alpha"])
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("Registered Holm alpha must lie in (0, 1)")
    table1 = registry["main_table_1_policy_and_deployment_ledger"]
    contrasts = registry["supplementary_table_S14_mechanism_contrasts_submitted_protocol"]
    if not isinstance(table1, dict) or not isinstance(contrasts, list):
        raise ValueError("Table 1 must be an object and Table S14 a list")
    rows = []
    for dataset, entry in table1.items():
        if dataset.startswith("_"):
            continue
        route = provisional_route(entry.get("provisional"))
        deployed = policy(entry.get("deployed_rule"))
        resolution = policy(entry.get("lifecycle_resolution"))
        recorded_pass = entry.get("registered_confirmation_pass")
        if recorded_pass is not None and type(recorded_pass) is not bool:
            raise ValueError(f"{dataset}: registered_confirmation_pass must be boolean")
        row = {"dataset": dataset, "provisional": route, "deployed": entry.get("deployed_rule"),
               "registry_resolution": entry.get("lifecycle_resolution"),
               "registry_confirmation_pass": recorded_pass, "evaluation_status": entry.get("evaluation_status"),
               "checks": [], "issues": [], "unresolved": []}
        allowed, expected_pass = set(), None
        if route == "qC":
            allowed = {"qC"}
            row["rule_outcome"] = "qC"
            row["reason"] = "Comparator selected; no feedback mechanism is being retained."
        elif route in ROUTE_CONTRAST:
            required = ROUTE_CONTRAST[route]
            selected = [c for c in contrasts if c.get("dataset") == dataset and c.get("kind") == required]
            if len(selected) != 1:
                row["unresolved"].append("Required mechanism contrast absent or ambiguous in Table S14")
            else:
                margin = float(rule["contrasts"][route]["practical_margin_nats"])
                row["checks"].append(contrast_result(selected[0], margin=margin, alpha=alpha, beneficial_sign=-1))
            # Only the state contrasts actually registered for this dataset are examined.
            state = [c for c in contrasts if c.get("dataset") == dataset and c.get("kind") == "conditional destruction minus intact"]
            if len(state) > 1:
                row["unresolved"].append("State contrast is ambiguous in Table S14")
            elif state:
                margin = float(rule["contrasts"]["state-conditioned route"]["practical_margin_nats"])
                row["checks"].append(contrast_result(state[0], margin=margin, alpha=alpha, beneficial_sign=1))
            for check in row["checks"]:
                if not check["margin_matches_registered"]:
                    row["issues"].append(f"{check['kind']}: printed margin differs from registered rule")
            passes = [check["confirmation_pass"] for check in row["checks"]]
            expected_pass = False if False in passes else None if row["unresolved"] or None in passes or not passes else True
            if expected_pass is True:
                # Confirmation permits a route; the later admission/input gates
                # may still return qC. This check does not infer their decisions.
                allowed = {route, "qC"}
                row["rule_outcome"] = route
                row["reason"] = "The available required contrasts pass Holm and the registered point-estimate margin."
            elif expected_pass is False:
                allowed = {"qC"}
                row["rule_outcome"] = f"{PREDECESSOR[route]} if eligible, otherwise qC"
                row["reason"] = "At least one required contrast does not retain the provisional mechanism."
            else:
                row["rule_outcome"] = "UNDETERMINED"
                row["reason"] = "Printed contrasts do not determine the registered confirmation outcome."
        else:
            row["rule_outcome"] = "UNDETERMINED"
            row["reason"] = "Provisional route is not recognized by this check."
            row["unresolved"].append(row["reason"])
        row["computed_confirmation_pass"] = expected_pass
        if recorded_pass is not None and expected_pass is not None and recorded_pass != expected_pass:
            row["issues"].append("Registry confirmation flag disagrees with the printed contrasts and registered rule")

        def compare_policy(actual, label):
            if actual is None:
                return "NOT_RECORDED" if label == "registry" and entry.get("lifecycle_resolution") is None else "UNDETERMINED"
            if actual in allowed:
                return "CONSISTENT"
            if expected_pass is False and actual == PREDECESSOR.get(route):
                # Eligibility comes from selection records, not the name of the fallback.
                return "UNDETERMINED"
            if row["rule_outcome"] == "UNDETERMINED":
                return "UNDETERMINED"
            return "DISCREPANCY"

        row["manuscript_status"] = compare_policy(deployed, "manuscript")
        row["registry_status"] = compare_policy(resolution, "registry")
        if row["manuscript_status"] == "DISCREPANCY":
            row["issues"].append("Table 1 policy differs from the outcome permitted by the printed confirmation contrasts")
        if row["registry_status"] == "DISCREPANCY":
            row["issues"].append("Registry lifecycle resolution differs from the registered confirmation rule")
        if entry.get("G_admit") == 0 and deployed not in {None, "qC"}:
            row["issues"].append("Table 1 retains a controller although its printed G_admit is zero")
        if row["manuscript_status"] == "UNDETERMINED" or row["registry_status"] == "UNDETERMINED":
            row["unresolved"].append("Policy identity or predecessor eligibility needs the original selection/lifecycle record")
        row["status"] = "DISCREPANCY" if row["issues"] else "UNDETERMINED" if row["unresolved"] else "CONSISTENT"
        rows.append(row)
    if not rows:
        raise ValueError("Table 1 contains no dataset rows")
    discrepancies = [r["dataset"] for r in rows if r["status"] == "DISCREPANCY"]
    unresolved = [r["dataset"] for r in rows if r["status"] == "UNDETERMINED"]
    return {"schema": "cross-morse-decision-consistency-1", "alpha": alpha, "rows": rows,
            "discrepancies": discrepancies, "undetermined": unresolved,
            "status": "DISCREPANCY" if discrepancies else "UNDETERMINED" if unresolved else "CONSISTENT",
            "scope": "Printed confirmation contrasts, Table 1 policy labels and optional canonical registry fields; "
                     "selection eligibility, admission and the original observations require archived run records."}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry", type=Path, default=ROOT / "paper" / "reported_results.json")
    parser.add_argument("--constants", type=Path, default=ROOT / "paper" / "registered_constants.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warn-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = evaluate(json.loads(args.registry.read_text(encoding="utf-8")),
                          json.loads(args.constants.read_text(encoding="utf-8")))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"{'dataset':14s} {'provisional':11s} {'Table 1':15s} {'canonical registry':19s} status")
    for r in report["rows"]:
        print(f"{r['dataset']:14s} {str(r['provisional']):11s} {r['manuscript_status']:15s} {r['registry_status']:19s} {r['status']}")
        for issue in r["issues"] + r["unresolved"]:
            print(f"  {issue}")
    print(f"{report['status']}: registered confirmation rule at alpha = {report['alpha']}; {report['scope']}")
    return 0 if report["status"] == "CONSISTENT" or args.warn_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
