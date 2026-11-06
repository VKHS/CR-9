"""Recompute the derived entries of the supplementary tables from their printed inputs.

    python -B scripts/check_si_identities.py                       # summary; exit 1 on FAIL
    python -B scripts/check_si_identities.py --json-output r.json --markdown r.md
    python -B scripts/check_si_identities.py --strict              # ADVISORY findings also fail

Standard-library only. The check reads paper/reported_results.json and
paper/registered_constants.json (transcribed from the manuscript) and the short
list of in-text statements in TEXT_CLAIMS below. It checks the explicitly enumerated
identities against the stated formula; it is not an exhaustive inventory of cells:
the data-role sizes (Table S2), entry-contract decisions (S4), the
support extrema (S7), the registered selection rule on the representative rows
(S12, S12c, S20), the clipping, sample-size and two-unit statements (S13,
Corollary 2, Note S24), the Holm family, margin tests and confirmation-size
rescaling (S14, S14b), both factorials (S16, S16b), the tube audit (S22), the
amortization horizon (S23b), the ensemble law (S26b), the retained-share and
power models (S34, S42, S46, S46b, S47), the attribution share (S45), the tail
and cost models (S44, Note S19) and the derived Table 2 columns. It complements
scripts/check_paper_constants.py, which covers the registry, the configurations,
Tables 1-2 against S23/S41/S13 and the single-graph Corollary 2 counts.

Explicit decimal strings preserve the last printed digit; bare numeric registry
values can lose trailing-zero precision and require an explicit half-unit override.
A value PASSes when the stated formula, evaluated over the supplied bounds, reaches the
printed value at its printed precision. ADVISORY marks a source-text interpretation discrepancy, a printed value that
the stated formula does not reproduce but an alternative reading does (the note
names it), or an inconsistent rounding convention; FAIL marks a value that no
documented reading reproduces. INFO records derived quantities and patterns
without a criterion, including inverse fits of unprinted parameters. Check counts
are not counts of distinct manuscript cells or a source-coverage percentage.
Passing arithmetic does not verify the underlying
measurements, which need the run archives and record sets.
"""
from __future__ import annotations

import argparse
from itertools import product
from decimal import Decimal, InvalidOperation
import json
import math
from pathlib import Path
from statistics import NormalDist
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._paper_common import (BCE_DATASETS, DATASETS, MULTI_GRAPH, load_registered,  # noqa: E402
                                   load_reported, printed_half_unit)

NORMAL = NormalDist()
Z975 = NORMAL.inv_cdf(0.975)
STATUSES = ("PASS", "ADVISORY", "FAIL", "INFO")
SHORT = {"ogbn-arxiv": "arxiv", "ogbn-proteins": "proteins", "ogbn-products": "products", "Roman-empire": "Roman",
         "PascalVOC-SP": "Pascal", "COCO-SP": "COCO", "Peptides-func": "Peptides"}

# In-text statements that the registry does not transcribe as table cells. Each entry names its location in the
# manuscript (main text) or the supplementary information (SI) and is recomputed below from registry values.
TEXT_CLAIMS = {
    "S4_ranges": {"where": "main text, Section 3 'Target qualification'",
                  "U_Delta_plant": [-0.006, -0.015], "PDD_pp": [0.18, 0.78], "DIL_gap_nats": [0.007, 0.019],
                  "phi_registry": [0.81, 0.92]},
    "S4_boundary": {"where": "main text, Section 3; SI Table S4", "boundary_dataset": "Peptides-func", "gate": "residual headroom"},
    "S12_reproduced": {"where": "main text, Section 3 'Selection and deployment'", "datasets_reproduced": 4},
    "S12_family": {"where": "SI Table S12, pp. 8-9, one representative row per route and dataset",
                   "routes": ["qC", "best-depth", "learned-stop", "phase", "fixed-sequence", "MPC-OL",
                              "one-step", "radius", "MPC-RH"]},
    "S12c_gaps": {"where": "SI Table S12c caption", "Roman-empire": "0.00110", "PascalVOC-SP": "0.00085", "COCO-SP": "0.00101"},
    "S14_family": {"where": "SI Table S14, p. 10, nine prespecified Holm contrasts",
                   "contrasts": ["arxiv/state", "arxiv/radius", "proteins/state", "proteins/radius", "products/state",
                                 "Roman/RH", "Pascal/RH", "COCO/RH", "Peptides/radius"]},
    "clipped_fraction_pct": {"where": "main text, Theorem 2", "PascalVOC-SP": "2.8", "COCO-SP": "3.4", "Peptides-func": "1.4"},
    "clipped_fraction_pct_SI": {"where": "SI Note S24", "PascalVOC-SP": "2.84", "COCO-SP": "3.44", "Peptides-func": "1.36"},
    "corollary2_counts": {"where": "main text, Corollary 2; SI Figure S17a", "COCO-SP": 342, "Roman-empire": 602,
                          "ogbn-proteins": 853, "ogbn-arxiv": 1279},
    "note_s24": {"where": "SI Note S24",
                 "graph_half_width": {"PascalVOC-SP": 0.00076, "COCO-SP": 0.00045, "Peptides-func": 0.00046},
                 "test_half_width": {"PascalVOC-SP": 0.00436, "COCO-SP": 0.00363, "Peptides-func": 0.00171},
                 "ratio": {"PascalVOC-SP": 5.75, "COCO-SP": 8.08, "Peptides-func": 3.75},
                 "seed_share_lower_bound": {"PascalVOC-SP": "0.970", "COCO-SP": "0.985", "Peptides-func": "0.929"},
                 "seed_sd": {"PascalVOC-SP": "0.0077", "COCO-SP": "0.0064", "Peptides-func": "0.0030"}, "seeds": 12},
    "mechanism_numbers": {"where": "main text, Section 3 'Headroom and mechanism'", "arxiv/state": "0.0057",
                          "proteins/state": "0.0049"},
    "note_s17_hoeffding": {"where": "SI Note S17 (proof of Proposition 7)", "2e(64)": "0.087", "2e(74)": "0.081", "2e(142)": "0.058"},
    "note_s19": {"where": "SI Note S19", "s": 0.06, "f_res": 0.5, "order": 8,
                 "T_max": {"ogbn-arxiv": 18, "PascalVOC-SP": 13, "COCO-SP": 14, "Peptides-func": 22},
                 "T_min_exceeded": {"Roman-empire": 64}, "proteins_T_max": "1.9", "proteins_f_res_max": "0.063"},
    "power_ranges": {"where": "main text, Section 4", "registered_units_pct": [65, 100], "outside_Roman_pct": [84, 100],
                     "seed_paired_pct": [76, 100]},
    "effect_range": {"where": "main text, Discussion", "nats": [0.003, 0.009]},
}


class Audit:
    def __init__(self):
        self.rows: list[dict] = []

    def add(self, group, check, status, computed=None, printed=None, note=""):
        if status not in STATUSES:
            raise ValueError(status)
        self.rows.append({"group": group, "check": check, "status": status, "computed": _plain(computed),
                          "printed": _plain(printed), "note": note})
        return status

    def compare(self, group, check, point, printed, *, low=None, high=None, half=None, decimals=None, mismatch="FAIL",
                note="", alternative=None):
        """PASS when ``point`` or any value in [low, high] rounds to ``printed``; else ``alternative`` or ``mismatch``.

        ``decimals`` states the printed precision of a table column whose trailing zeros the JSON registry drops
        (0.350 is stored as 0.35); a string ``printed`` keeps its own digits.
        """
        shown = printed
        if isinstance(printed, str):
            printed, half = _text_value(printed)
        if decimals is not None:
            half = 0.5 * 10.0 ** (-decimals)
        half = printed_half_unit(printed) if half is None else half
        bounds = [point] + [v for v in (low, high) if v is not None]
        if not all(math.isfinite(float(v)) for v in bounds + [printed, half]) or half < 0:
            raise ValueError("comparisons require finite values and a nonnegative half-unit")
        lo, hi = min(bounds), max(bounds)
        target_lo, target_hi = float(printed) - half - 1e-12, float(printed) + half + 1e-12
        if target_lo <= point <= target_hi:
            return self.add(group, check, "PASS", point, shown, note)
        if hi >= target_lo and lo <= target_hi:
            return self.add(group, check, "PASS", point, shown, (note + "; " if note else "") +
                            f"reproduced within the rounding of the printed inputs [{_fmt(lo)}, {_fmt(hi)}]")
        if alternative is not None:
            value, reading = alternative
            if float(printed) - half - 1e-12 <= value <= float(printed) + half + 1e-12:
                return self.add(group, check, "ADVISORY", point, shown,
                                f"the stated formula gives {_fmt(point)} (range {_fmt(lo)} to {_fmt(hi)}); "
                                f"the printed value follows from {reading} ({_fmt(value)})" + (f"; {note}" if note else ""))
        return self.add(group, check, mismatch, point, shown,
                        (note + "; " if note else "") + f"computed {_fmt(point)}, range {_fmt(lo)} to {_fmt(hi)}")

    def equal(self, group, check, computed, printed, note="", mismatch="FAIL"):
        return self.add(group, check, "PASS" if computed == printed else mismatch, computed, printed, note)


def _text_value(text: str) -> tuple[float, float]:
    """Preserve decimal precision, including trailing zeros and scientific notation."""
    try:
        value = Decimal(text.strip())
    except InvalidOperation as exc:
        raise ValueError(f"not a printed decimal: {text!r}") from exc
    if not value.is_finite():
        raise ValueError("printed decimal must be finite")
    return float(value), float(Decimal(1).scaleb(value.as_tuple().exponent) / 2)


def _sequence(value, length: int, label: str):
    """Reject incomplete arrays before zip/all can silently omit required entries."""
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{label} must contain exactly {length} entries")
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value):
        raise ValueError(f"{label} must contain finite numeric entries")
    return value


def _require_keys(value, expected, label: str):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{label} must contain exactly {sorted(expected)}")


def _require_records(rows, key: str, expected, label: str):
    """Check the declared source family before omissions or duplicates change an audit."""
    if not isinstance(rows, (list, tuple)) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{label} must be a sequence of records")
    identifiers = [row[key] for row in rows]
    if len(identifiers) != len(expected) or len(set(identifiers)) != len(identifiers) or set(identifiers) != set(expected):
        raise ValueError(f"{label} must contain each of {sorted(expected)} exactly once")


def _interval_contains_estimate(audit, group, check, interval, estimate):
    lo, hi = _sequence(interval, 2, check)
    if not isinstance(estimate, (int, float)) or not math.isfinite(estimate):
        raise ValueError(f"{check}: estimate must be finite")
    if lo > hi:
        raise ValueError(f"{check}: interval endpoints are reversed")
    # A bootstrap interval need not be centred at the estimate. An interval that
    # excludes it needs inspection, not an unsupported normal-centering assertion.
    inside = lo <= estimate <= hi
    audit.add(group, check, "PASS" if inside else "ADVISORY", estimate, interval,
              "" if inside else "the printed estimate is outside its interval; verify the bootstrap record and interval "
              "semantics rather than validating only the width")


def _plain(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    return value


def _fmt(value) -> str:
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def span(function, *args):
    """Point value and range of ``function`` over the rounding box of the printed arguments.

    An argument given as (value, half) is printed with that half-unit; a bare
    number is exact. Every corner of the box is evaluated; the formulas audited
    here are monotone in each argument, so the corners bound the range.
    """
    points, boxes = [], []
    for arg in args:
        if isinstance(arg, tuple):
            value, half = arg
            points.append(value)
            boxes.append((value - half, value + half))
        else:
            points.append(arg)
            boxes.append((arg, arg))
    values = [function(*corner) for corner in product(*boxes)]
    return function(*points), min(values), max(values)


def P(value, half=None):
    """A printed input: (value, half-unit)."""
    return (float(value), printed_half_unit(value) if half is None else half)


def two_sided(z: float) -> float:
    return 2.0 * (1.0 - NORMAL.cdf(abs(z)))


def holm(p: list[float]) -> list[float]:
    order = sorted(range(len(p)), key=lambda i: p[i])
    adjusted, running = [0.0] * len(p), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def parse_p(value):
    """Return (low, high, printed), retaining printed decimal precision."""
    shown = value
    if isinstance(value, str):
        text = value.replace(" ", "")
        if text.startswith("<"):
            bound, _ = _text_value(text[1:])
            if not 0 < bound <= 1:
                raise ValueError("P-value upper bound must lie in (0, 1]")
            return 0.0, bound, shown
        value, half = _text_value(text)
    else:
        half = printed_half_unit(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("P value must lie in [0, 1]")
    return max(0.0, value - half), min(1.0, value + half), shown


def p_matches(computed_low, computed_high, printed) -> bool:
    if not (math.isfinite(computed_low) and math.isfinite(computed_high) and computed_low <= computed_high):
        raise ValueError("computed P range must be finite and ordered")
    low, high, _ = parse_p(printed)
    if isinstance(printed, str) and printed.replace(" ", "").startswith("<"):
        return computed_low < high
    return computed_high >= low - 1e-12 and computed_low <= high + 1e-12


# ============================================================================ S2, S37, S39
def check_roles(audit, registered, reported):
    counts = registered["data_roles_table_S2"]["multi_graph_counts"]
    t1 = reported["main_table_1_policy_and_deployment_ledger"]
    s13 = reported["supplementary_table_S13_independent_graph_admission"]
    s23 = reported["supplementary_table_S23_assurance_and_expected_compute"]
    s37 = reported["supplementary_table_S37_input_fallback_counts"]
    for d in MULTI_GRAPH:
        n = counts[d]["official_train"]
        conf = (95 * n) // 100 - (90 * n) // 100
        compare = n - (95 * n) // 100
        audit.equal("S2", f"{SHORT[d]}: n_conf = floor(0.95 n) - floor(0.90 n)", conf, counts[d]["D_confirm"])
        audit.equal("S2", f"{SHORT[d]}: n_compare = n - floor(0.95 n)", compare, counts[d]["D_compare"])
        audit.equal("S1/S13", f"{SHORT[d]}: admission units equal the official validation split", s13[d]["J"], counts[d]["D_admit"])
        audit.equal("T1/S13", f"{SHORT[d]}: Table 1 admission J", t1[d]["admission_J"], s13[d]["J"])
        row = s37[d]
        audit.equal("S37", f"{SHORT[d]}: fallback denominator is the official test split", row["denominator"], counts[d]["D_test"])
        audit.compare("S37", f"{SHORT[d]}: input-fallback rate numerator/denominator", row["numerator"] / row["denominator"], row["rate"],
                      decimals=3)
        audit.equal("S37/T1/S23", f"{SHORT[d]}: one fallback fraction in Table 1, S23 and S37",
                    (t1[d]["input_fallback_fraction"], s23[d]["fallback_probability"]), (row["rate"], row["rate"]))


# ============================================================================ S4
def check_entry_contract(audit, registered, reported):
    t = registered["entry_contract_eq5_table_S3"]
    s4 = reported["supplementary_table_S4_target_qualification"]
    margins = {}
    for d in DATASETS:
        row = s4[d]
        gates = {"nominal graph value": row["U_Delta_plant"] < -t["epsilon_plant_nats"],
                 "peak-depth deterioration": row["PDD_pp"] <= t["tau_PDD_percentage_points"],
                 "depth-integrated gap": row["DIL_gap_nats"] <= t["tau_DIL_nats"],
                 "registry support": row["phi_registry"] >= t["tau_sup"],
                 "selected strata": row["selected_strata"] == t["selected_strata_fraction"],
                 "residual headroom": row["U_Delta_oracle"] < -t["epsilon_headroom_nats"]}
        audit.equal("S4", f"{SHORT[d]}: every Eq. (5) gate passes as the printed decision states", all(gates.values()),
                    row["decision"] in ("pass", "boundary pass"))
        margins[d] = {"nominal graph value": (-t["epsilon_plant_nats"] - row["U_Delta_plant"]) / t["epsilon_plant_nats"],
                      "peak-depth deterioration": (t["tau_PDD_percentage_points"] - row["PDD_pp"]) / t["tau_PDD_percentage_points"],
                      "depth-integrated gap": (t["tau_DIL_nats"] - row["DIL_gap_nats"]) / t["tau_DIL_nats"],
                      "registry support": (row["phi_registry"] - t["tau_sup"]) / t["tau_sup"],
                      "residual headroom": (-t["epsilon_headroom_nats"] - row["U_Delta_oracle"]) / t["epsilon_headroom_nats"]}
    claims = TEXT_CLAIMS["S4_ranges"]
    for key, label in (("U_Delta_plant", "U(Delta_plant)"), ("PDD_pp", "PDD"), ("DIL_gap_nats", "DIL gap"),
                       ("phi_registry", "registry support")):
        values = [s4[d][key] for d in DATASETS]
        observed = sorted([max(values), min(values)], key=abs) if key == "U_Delta_plant" else [min(values), max(values)]
        audit.equal("S4/main", f"{label} range printed in {claims['where']}", observed, claims[key])
    boundary = TEXT_CLAIMS["S4_boundary"]
    headroom = {SHORT[d]: round(margins[d]["residual headroom"], 3) for d in DATASETS}
    tightest = {SHORT[d]: min(margins[d], key=margins[d].get) for d in DATASETS}
    closer = [SHORT[d] for d in DATASETS if margins[d]["residual headroom"] < margins[boundary["boundary_dataset"]]["residual headroom"]]
    audit.add("S4", "relative margin of the residual-headroom gate per dataset", "INFO", headroom, None,
              f"tightest relative gate per dataset: {tightest}")
    if closer:
        audit.add("S4", f"{SHORT[boundary['boundary_dataset']]} is labelled the boundary pass on the {boundary['gate']} gate",
                  "INFO", closer, boundary["boundary_dataset"],
                  f"{', '.join(closer)} pass the {boundary['gate']} gate with a smaller relative margin "
                  f"(U(Delta_oracle) {', '.join(str(s4[d]['U_Delta_oracle']) for d in DATASETS if SHORT[d] in closer)} against -0.003); "
                  f"the Peptides-func margin is {headroom[SHORT[boundary['boundary_dataset']]]:.3f} of the threshold and its tightest "
                  f"relative gate is {tightest[SHORT[boundary['boundary_dataset']]]}; the label needs its criterion stated")


# ============================================================================ S7
def check_support(audit, registered, reported):
    s7 = reported["supplementary_table_S7_support_audit"]
    p_min = registered["support_eq7"]["p_min"]
    for d in DATASETS:
        row = s7[d]
        cap = 1.0 / math.sqrt(row["per_cell_minimum_ESS"])
        audit.add("S7", f"{SHORT[d]}: per-cell extrema obey omega_max <= 1/sqrt(ESS)",
                  "PASS" if row["per_cell_maximum_normalized_weight"] - printed_half_unit(row["per_cell_maximum_normalized_weight"]) <= cap
                  else "FAIL", row["per_cell_maximum_normalized_weight"], round(cap, 4))
        audit.add("S7", f"{SHORT[d]}: minimum propensity >= p_min", "PASS" if row["minimum_propensity"] >= p_min else "FAIL",
                  row["minimum_propensity"], p_min)
    audit.compare("S7/S18", "ESS >= 64 implies omega_max <= 0.125", 1 / math.sqrt(64), "0.125")
    audit.compare("S7/S18", "ESS >= 32 implies omega_max <= 0.177", 1 / math.sqrt(32), "0.177")


# ============================================================================ S12, S12c, S20
FEEDBACK = ("one-step", "radius", "MPC-RH")


def _reading(rows, eps_r, eps_eq, on: str, *, candidate_scope="feedback_only"):
    """Replay on an explicit feasible domain; default to the lifecycle Eq. (3) family.

    All representative rows in the current printed S12 have pass/pass/pass gates.
    The caller must prefilter a future ledger by its feasibility mask and declared
    candidate domain. The default feedback-only family is supported by main
    Eq. (3) and reproduces S12c. The opt-in all-feasible reading is a diagnostic:
    Eqs. (18)-(19) and S11 leave the ambient domain ambiguous when all
    pass/pass/pass S12 rows are supplied. It does not change the registered rule.
    """
    if on not in ("U", "Delta"):
        raise ValueError("reading must use U or Delta")
    if candidate_scope not in ("feedback_only", "all_feasible"):
        raise ValueError("candidate_scope must be feedback_only or all_feasible")
    candidates = [r for r in rows if candidate_scope == "all_feasible" or r["route"] in FEEDBACK]
    eligible = [r for r in candidates if r["U_s"] < -eps_r]
    if not eligible:
        return [], "qC", None
    key = "U_s" if on == "U" else "Delta_s_hat"
    u_min = min(0.0, min(r[key] for r in eligible))
    near = [r for r in eligible if r[key] <= u_min + eps_eq + 1e-12]
    choice = min(near, key=lambda r: r["rank"])["route"]
    return [r["route"] for r in sorted(near, key=lambda r: r["rank"])], choice, u_min


def _parse_reading(text: str):
    """'radius, MPC-RH -> radius' gives (['radius', 'MPC-RH'], 'radius'); a bare 'qC' gives ([], 'qC')."""
    if "->" not in text:
        return [], text.strip()
    left, right = [part.strip() for part in text.split("->")]
    members = [] if left in ("", "\u2014", "-") else [m.strip() for m in left.split(",")]
    return members, right


def check_selection(audit, registered, reported):
    rule = registered["selection_eq19_algorithm_S4"]
    s12 = reported["supplementary_table_S12_selection_ledger"]
    s12c = reported["supplementary_table_S12c_selection_rule_audit"]
    s20 = reported["supplementary_table_S20_selection_theorem_instantiation"]
    t1 = reported["main_table_1_policy_and_deployment_ledger"]
    reproduced = 0
    gaps = TEXT_CLAIMS["S12c_gaps"]
    for label, table in (("S12", s12), ("S12c", s12c), ("S20", s20)):
        _require_keys(table, DATASETS, f"{label} dataset rows")
    for d in DATASETS:
        eps_r = rule["epsilon_R_nats"]["mean_multilabel_BCE" if d in BCE_DATASETS else "graph_level_CE"]
        rows = s12[d]
        _require_records(rows, "route", TEXT_CLAIMS["S12_family"]["routes"], f"S12 {SHORT[d]} representative routes")
        audit.add("S12", f"{SHORT[d]}: complete representative route family", "PASS", len(rows),
                  len(TEXT_CLAIMS["S12_family"]["routes"]),
                  "Source-row completeness only; the full configuration ledger is a separate run artifact.")
        worst_mid, widest = 0.0, 0.0
        for r in rows:
            if r["route"] == "qC":
                continue
            worst_mid = max(worst_mid, abs((r["L_s"] + r["U_s"]) / 2 - r["Delta_s_hat"]))
            widest = max(widest, (r["U_s"] - r["L_s"]) / 2)
        audit.add("S12", f"{SHORT[d]}: simultaneous intervals are centred on the selection effect",
                  "PASS" if worst_mid <= 1e-5 + 1e-12 else "FAIL", worst_mid, 1e-5)
        audit.add("S12/S20", f"{SHORT[d]}: representative radii do not exceed r_max", "PASS" if widest <= s20[d]["r_max"] + 5e-6 else "FAIL",
                  round(widest, 6), s20[d]["r_max"])
        members, choice, u_min = _reading(rows, eps_r, rule["epsilon_eq_nats"], "U")
        printed_members, printed_choice = _parse_reading(s12c[d]["registered_reading_eps_eq_0.002_on_U_s"])
        printed = (printed_members, printed_choice)
        audit.equal("S12c", f"{SHORT[d]}: Eq. (3) feedback-domain representative reading",
                    (members, choice), printed)
        all_members, all_choice, all_min = _reading(
            rows, eps_r, rule["epsilon_eq_nats"], "U", candidate_scope="all_feasible")
        audit.add("S12c", f"{SHORT[d]}: all-feasible candidate-domain sensitivity", "INFO",
                  {"near_best": all_members, "choice": all_choice, "U_min": all_min,
                   "differs_from_feedback_domain": (all_members, all_choice) != (members, choice)}, printed,
                  "All current S12 rows print pass/pass/pass. Eqs. (18)-(19) use Q_feas and S11 permits MPC-OL in Q_s, "
                  "while main Eq. (3) and the SI lifecycle diagram restrict selection to Q_FB plus qC. "
                  "This alternate-domain calculation does not amend the rule or prove an executed-selection error. "
                  "Authors must reconcile the candidate domain and confirmation map; the complete ledger is unavailable.")
        if s12c[d]["U_min"] is not None:
            audit.compare("S12c", f"{SHORT[d]}: U_min", u_min, s12c[d]["U_min"], decimals=5)
        for eps_eq, key in ((0.002, "point_estimate_reading_eps_eq_0.002"), (0.001, "point_estimate_reading_eps_eq_0.001")):
            m, c, _ = _reading(rows, eps_r, eps_eq, "Delta")
            audit.equal("S12c", f"{SHORT[d]}: feedback-domain point-estimate transcription, eps_eq = {eps_eq}", (m, c),
                        _parse_reading(s12c[d][key]))
        deployed = "qC" if t1[d]["deployed_rule"].startswith("qC") else t1[d]["deployed_rule"]
        reproduced += choice == deployed
        if d in gaps:
            radius = next(r for r in rows if r["route"] == "radius")
            audit.compare("S12c", f"{SHORT[d]}: radius upper limit above U_min (caption)", radius["U_s"] - u_min, gaps[d])
        slack = rule["epsilon_eq_nats"] + 2 * s20[d]["r_max"]
        audit.compare("S20", f"{SHORT[d]}: slack eps_eq + 2 r_max", slack, s20[d]["eps_eq_plus_2r_max"], decimals=4)
    audit.equal("S12/main", "datasets on which the Eq. (3) feedback-domain reading reproduces the deployed route", reproduced,
                TEXT_CLAIMS["S12_reproduced"]["datasets_reproduced"])
    audit.equal("S20", "products: empty eligible set and comparator return", (s20["ogbn-products"]["Q_s_size"],
                _reading(s12["ogbn-products"], 0.003, 0.002, "U")[1]), (0, "qC"))


# ============================================================================ S13, Corollary 2, Note S24, S30
def check_admission(audit, registered, reported):
    s13 = reported["supplementary_table_S13_independent_graph_admission"]
    t1 = reported["main_table_1_policy_and_deployment_ledger"]
    s30 = reported["supplementary_table_S30_oracle_dispersion"]
    adm = registered["admission_eq27_theorem2_table_S13"]
    constant = 14 * adm["B"] * math.log(2 / adm["delta_a"]) / 3
    note = TEXT_CLAIMS["note_s24"]
    for d in MULTI_GRAPH:
        r = s13[d]
        f = lambda D, S2: ((r["J"] - 1) * S2 / r["J"] + D * D) / r["B"] ** 2 * 100  # noqa: E731
        point, lo, hi = span(f, P(r["D_bar"]), P(r["S2"]))
        audit.compare("Theorem 2/S24", f"{SHORT[d]}: clipped-fraction bound (SI, percent)", point,
                      TEXT_CLAIMS["clipped_fraction_pct_SI"][d], low=lo, high=hi)
        audit.compare("Theorem 2", f"{SHORT[d]}: clipped-fraction bound (main text, percent)", point,
                      TEXT_CLAIMS["clipped_fraction_pct"][d], low=lo, high=hi)
        threshold = 1 + constant / (abs(r["D_bar"]) - r["epsilon_a"])
        necessary = r["J"] > threshold if r["S2"] > 0 else r["J"] >= threshold
        audit.add("Cor 2", f"{SHORT[d]}: J meets the necessary Corollary 2 lower bound", "PASS" if necessary else "FAIL",
                  round(threshold, 2), r["J"],
                  "J >= threshold is necessary; S^2 > 0 makes it strict. This is not sufficient for admission.")
        hw_graph = lambda S2: Z975 * math.sqrt(S2 / r["J"])  # noqa: E731
        point, lo, hi = span(hw_graph, P(r["S2"]))
        audit.compare("S24", f"{SHORT[d]}: graph-unit 95% half-width 1.96 S/sqrt(J)", point, note["graph_half_width"][d], low=lo, high=hi)
        lo_t, hi_t = t1[d]["loss_effect_interval_nats"]
        point, lo, hi = span(lambda a, b: (b - a) / 2, P(lo_t), P(hi_t))
        audit.compare("S24", f"{SHORT[d]}: Table 1 test-stage half-width", point, note["test_half_width"][d], low=lo, high=hi)
        point, lo, hi = span(lambda t, g: t / g, P(note["test_half_width"][d]), P(note["graph_half_width"][d]))
        audit.compare("S24", f"{SHORT[d]}: ratio rho of the two half-widths", point, note["ratio"][d], low=lo, high=hi)
        point, lo, hi = span(lambda rho: 1 - rho ** -2, P(note["ratio"][d]))
        audit.compare("S24", f"{SHORT[d]}: seed-share lower bound 1 - rho^-2", point, note["seed_share_lower_bound"][d], low=lo, high=hi)
        S = note["seeds"]
        stated = lambda t, g: math.sqrt(S) * math.sqrt(max(t * t - g * g, 0.0)) / Z975  # noqa: E731
        point, lo, hi = span(stated, P(note["test_half_width"][d]), P(note["graph_half_width"][d]))
        alternative = math.sqrt(S) * note["test_half_width"][d] / Z975
        audit.compare("S24", f"{SHORT[d]}: implied seed-level SD sqrt(S) (hw_test^2 - hw_graph^2)^(1/2) / 1.96", point,
                      note["seed_sd"][d], low=lo, high=hi, mismatch="FAIL",
                      alternative=(alternative, "sqrt(S) hw_test / 1.96, without the graph term"))
        audit.compare("S30", f"{SHORT[d]}: per-input SD of the clipped effect equals sqrt(S^2)", math.sqrt(r["S2"]),
                      s30[d]["per_input_SD_clipped_effect"], decimals=4, low=math.sqrt(r["S2"] - printed_half_unit(r["S2"])),
                      high=math.sqrt(r["S2"] + printed_half_unit(r["S2"])))
    # Corollary 2 counts: one rounding convention for all four printed values.
    counts = TEXT_CLAIMS["corollary2_counts"]
    eps = adm["epsilon_a_nats"]
    thresholds = {}
    counts = {d: v for d, v in counts.items() if d != "where"}
    for d in counts:
        D = s13[d]["D_bar"] if d in s13 else t1[d]["admission_D_bar"]
        thresholds[d] = 1 + constant / (abs(D) - eps["mean_multilabel_BCE" if d in BCE_DATASETS else "graph_level_CE"])
    # The variance-independent necessary bound is J >= threshold. A strict
    # integer display J > n therefore uses n = ceil(threshold) - 1; at an exact
    # integer threshold zero variance can attain equality.
    minimal = {SHORT[d]: math.ceil(v) for d, v in thresholds.items()}
    floors = {k: v - 1 for k, v in minimal.items()}
    printed = {SHORT[d]: v for d, v in counts.items()}
    audit.add("Cor 2", "Corollary 2 thresholds 1 + 2.045/(|D_bar| - eps_a)", "INFO",
              {SHORT[d]: round(v, 2) for d, v in thresholds.items()}, None, "smallest integer not excluded by the variance-independent necessary bound: " + str(minimal))
    if printed == floors:
        audit.add("Cor 2", "printed counts follow one rounding convention", "PASS", printed, floors)
    else:
        mixed = [k for k in printed if printed[k] != floors[k]]
        audit.add("Cor 2", "printed counts follow one rounding convention", "ADVISORY", floors, printed,
                  f"'J > n' requires n = ceil(threshold) - 1 for the variance-independent necessary bound; "
                  f"the printed {', '.join(mixed)} count does not use this form. The smallest integers not excluded are "
                  f"{minimal} (necessary only, not admission guarantees); "
                  f"the strict form is {', '.join(f'J > {floors[k]}' for k in mixed)}, or state 'J >= n' with n = "
                  f"{', '.join(str(minimal[k]) for k in printed)}")


# ============================================================================ S14, S14b
def check_mechanism(audit, registered, reported):
    rows = reported["supplementary_table_S14_mechanism_contrasts_submitted_protocol"]
    _require_records(rows, "contrast", TEXT_CLAIMS["S14_family"]["contrasts"], "S14 Holm family")
    audit.add("S14", "complete prespecified nine-contrast Holm family", "PASS", len(rows),
              len(TEXT_CLAIMS["S14_family"]["contrasts"]),
              "The multiplicity correction uses this full family, including nonsignificant contrasts.")
    lows, highs = [], []
    for r in rows:
        lo, hi, _ = parse_p(r["raw_P"])
        lows.append(lo)
        highs.append(hi)
    holm_lo, holm_hi = holm(lows), holm(highs)
    for i, r in enumerate(rows):
        ok = p_matches(holm_lo[i], holm_hi[i], r["holm_P"])
        audit.add("S14", f"{r['contrast']}: Holm-adjusted P from the nine raw P values", "PASS" if ok else "FAIL",
                  [round(holm_lo[i], 5), round(holm_hi[i], 5)], r["holm_P"])
        lo, hi = r["interval"]
        effect, margin = r["effect"], r["margin"]
        if r["P_vs_margin"] is not None:
            f = lambda e, a, b: 1 - NORMAL.cdf((abs(e) - margin) / ((b - a) / (2 * Z975)))  # noqa: E731
            point, plo, phi = span(f, P(effect), P(lo), P(hi))
            ok = p_matches(plo, phi, r["P_vs_margin"])
            audit.add("S14", f"{r['contrast']}: one-sided P against the margin from the 95% interval (normal)",
                      "PASS" if ok else "FAIL", [round(plo, 5), round(phi, 5)], r["P_vs_margin"])
        f = lambda e, a, b: two_sided(e / ((b - a) / (2 * Z975)))  # noqa: E731
        point, plo, phi = span(f, P(effect), P(lo), P(hi))
        if "bootstrap" in r["test"]:
            audit.add("S14", f"{r['contrast']}: raw P against zero from the 95% interval (normal)",
                      "PASS" if p_matches(plo, phi, r["raw_P"]) else "FAIL", [round(plo, 6), round(phi, 6)], r["raw_P"])
        else:
            audit.add("S14", f"{r['contrast']}: raw P against zero ({r['test']})", "INFO", round(point, 6), r["raw_P"],
                      "the printed P comes from the unit-level test, not from a normal reading of the interval")
    products = next(r for r in rows if r["contrast"] == "products/state")
    lo, hi = products["interval"]
    se = (hi - lo) / (2 * Z975)
    p_tost = max(1 - NORMAL.cdf((products["effect"] + products["margin"]) / se),
                 1 - NORMAL.cdf((products["margin"] - products["effect"]) / se))
    z90 = NORMAL.inv_cdf(0.95)
    ci90 = [products["effect"] - z90 * se, products["effect"] + z90 * se]
    audit.add("S14", "products/state: two one-sided tests at +-0.002 from the printed interval (normal reading)", "INFO",
              round(p_tost, 5), products["margin"],
              f"90% interval [{ci90[0]:.5f}, {ci90[1]:.5f}] lies inside +-{products['margin']}; the table states that the "
              "two one-sided tests were not performed, so equivalence is not claimed; cm_theory.inference.tost runs them on the "
              "cluster differences")
    claims = TEXT_CLAIMS["mechanism_numbers"]
    for contrast, printed in claims.items():
        if contrast == "where":
            continue
        effect = next(r["effect"] for r in rows if r["contrast"] == contrast)
        audit.compare("S14/main", f"{contrast}: state destruction effect printed in the main text", effect, printed)
    abstract = reported["abstract_and_section_claims"]["feedback_vs_plan_once_improvement_nats"]
    rh = [abs(r["effect"]) for r in rows if r["contrast"].endswith("/RH")]
    audit.compare("S14/main", "smallest replanning improvement (abstract range)", min(rh), abstract[0], decimals=4)
    audit.compare("S14/main", "largest replanning improvement (abstract range)", max(rh), abstract[1], decimals=4)
    for r in reported["supplementary_table_S14b_separated_data_sensitivity"]:
        lo_old, hi_old = r["interval_old"]
        g = r["graph_variance_share"]
        f_lo = lambda a, b, share: r["effect"] - Z975 * (b - a) / (2 * Z975) * math.sqrt((1 - share) + share * r["J_old"] / r["J_new"])  # noqa: E731
        f_hi = lambda a, b, share: r["effect"] + Z975 * (b - a) / (2 * Z975) * math.sqrt((1 - share) + share * r["J_old"] / r["J_new"])  # noqa: E731
        for side, f, printed in (("lower", f_lo, r["interval"][0]), ("upper", f_hi, r["interval"][1])):
            point, lo, hi = span(f, P(lo_old), P(hi_old), P(g, 0.0005))
            audit.compare("S14b", f"{r['contrast']}: {side} end at J_new under the two-way variance model", point, printed,
                          low=lo, high=hi, decimals=5)


# ============================================================================ S16, S16b
def _cells(table):
    _require_keys(table["cells"], {"".join(bits) for bits in product("01", repeat=3)}, "factorial cells")
    return {k: v["log_depth_CE"] for k, v in table["cells"].items()}


def _effect(cells, factors):
    plus, minus = [], []
    for key, value in cells.items():
        sign = 1
        for j in factors:
            sign *= 1 if key[j] == "1" else -1
        (plus if sign > 0 else minus).append(value)
    return sum(plus) / len(plus) - sum(minus) / len(minus)


def _anchored(cells, factors):
    from itertools import combinations
    total = 0.0
    for size in range(len(factors) + 1):
        for subset in combinations(factors, size):
            key = "".join("1" if j in subset else "0" for j in range(3))
            total += (-1) ** (len(factors) - size) * cells[key]
    return total


def _independent_half_width(table):
    se = [((c["interval"][1] - c["interval"][0]) / 2) / Z975 for c in table["cells"].values()]
    return Z975 * math.sqrt(sum(s * s for s in se)) / 4


def check_factorials(audit, reported):
    rounding = 8 * 0.00005 / 4  # every effect averages four cells against four cells printed to four decimals
    compare = lambda *args, **kw: audit.compare(*args, decimals=4, **kw)  # noqa: E731 - S16/S16b print four decimals
    s16 = reported["supplementary_table_S16_arxiv_factorial"]
    cells = _cells(s16)
    names = ("actions", "observer", "risk_controlled_execution")
    for j, name in enumerate(names):
        compare("S16", f"arxiv main effect: {name}", _effect(cells, (j,)), s16["effect_coded_main_effects_nats"][name],
                      low=_effect(cells, (j,)) - rounding, high=_effect(cells, (j,)) + rounding)
    _sequence(s16["pairwise_interactions_nats"], 3, "S16 pairwise interactions")
    for (a, b), printed in zip(((0, 1), (0, 2), (1, 2)), s16["pairwise_interactions_nats"]):
        value = _effect(cells, (a, b))
        compare("S16", f"arxiv effect-coded interaction {'AOX'[a]}{'AOX'[b]}", value, printed, low=value - rounding, high=value + rounding)
    value = _effect(cells, (0, 1, 2))
    compare("S16", "arxiv effect-coded three-way interaction", value, s16["three_way_interaction_nats"],
                  low=value - rounding, high=value + rounding)
    anchored = s16["baseline_anchored_finite_differences"]
    for label, factors in (("I_AO", (0, 1)), ("I_AX", (0, 2)), ("I_OX", (1, 2)), ("I_AOX", (0, 1, 2))):
        value = _anchored(cells, factors)
        compare("S16", f"arxiv baseline-anchored {label}", value, anchored[label],
                      low=value - 0.00005 * 2 ** len(factors), high=value + 0.00005 * 2 ** len(factors))
    compare("S16", "arxiv full system minus no component", cells["111"] - cells["000"], s16["full_system_minus_no_component_nats"])
    compare("S16", "arxiv effect half-width under independent cell estimates", _independent_half_width(s16),
                  s16["effect_coded_main_effects_nats"]["half_width_95"])
    s16b = reported["supplementary_table_S16b_pascal_factorial"]
    cells = _cells(s16b)
    names = ("spectral_information", "conditioned_observer", "risk_controlled_execution")
    for j, name in enumerate(names):
        value = _effect(cells, (j,))
        compare("S16b", f"Pascal main effect: {name}", value, s16b["effect_coded_main_effects_nats"][name],
                      low=value - rounding, high=value + rounding)
    compare("S16b", "Pascal effect half-width under independent cell estimates", _independent_half_width(s16b),
                  s16b["effect_coded_main_effects_nats"]["half_width_95"],
                  note="the caption uses the joint resampling covariance; independent cells give the same half-width")
    interactions = {"SO": (0, 1), "SX": (0, 2), "OX": (1, 2), "SOX": (0, 1, 2)}
    values = {k: _effect(cells, f) for k, f in interactions.items()}
    beyond = {k: round(v, 6) for k, v in values.items() if abs(v) > 1e-4 + 1e-12}
    if not beyond:
        audit.add("S16b", "Pascal interactions within 1e-4 nats of zero at the tabulated precision", "PASS",
                  {k: round(v, 6) for k, v in values.items()}, 1e-4)
    else:
        audit.add("S16b", "Pascal interactions within 1e-4 nats of zero at the tabulated precision", "ADVISORY",
                  {k: round(v, 6) for k, v in values.items()}, 1e-4,
                  f"{', '.join(f'{k} = {v:+.6f}' for k, v in beyond.items())} from the displayed cell means. "
                  "At four-decimal precision this rounds within 1e-4, consistent with that reading of the caption. "
                  f"Cell rounding permits +-{rounding:.0e} in the unrounded interaction. A 1.3e-4 bound applies to "
                  "the calculation from displayed central values, not to the unavailable unrounded effects or bootstrap records.")


# ============================================================================ S22, S23b
def check_tube_and_amortization(audit, reported):
    for d, row in reported["supplementary_table_S22_trajectory_tube_audit"].items():
        audit.compare("S22", f"{SHORT[d]}: violation rate = 1 - mode-stable coverage", 1 - row["mode_stable_coverage"], row["violation_rate"],
                      decimals=3)
    s23b = reported["supplementary_table_S23b_nominal_execution_time_and_amortization"]
    for d, row in s23b.items():
        ms = "training_time_anchor_ms" in row
        unit = 1e-3 if ms else 1.0
        anchor = row["training_time_anchor_ms" if ms else "training_time_anchor_s"]
        if "shares the PascalVOC-SP anchor" in row.get("analytical_reference", ""):
            anchor = s23b["PascalVOC-SP"]["training_time_anchor_ms"]
        t_nom = row["t_nom_measured_ms" if ms else "t_nom_measured_s"]
        if row.get("throughput_anchor_s"):
            f = lambda a, b, c: math.sqrt(math.sqrt(a * b) * c)  # noqa: E731
            point, lo, hi = span(f, P(anchor[0]), P(anchor[1]), P(row["throughput_anchor_s"]))
            label = "geometric mean of the throughput anchor and the training-time range"
        else:
            point, lo, hi = span(lambda a, b: math.sqrt(a * b), P(anchor[0]), P(anchor[1]))
            label = "geometric mean of the training-time range"
        audit.compare("S23b", f"{SHORT[d]}: t_nom as the {label}", point, t_nom, low=lo, high=hi, decimals=2)
        mt = row["M_star_t_nom_thousand_input_s"]
        f = lambda m, t: m * 1e3 / (t * unit)  # noqa: E731
        point, lo, hi = span(f, P(mt), P(t_nom))
        audit.compare("S23b", f"{SHORT[d]}: M* = (M* t_nom)/t_nom (two significant digits)", point, row["M_star_inputs"],
                      low=lo, high=hi, half=_sig_half(row["M_star_inputs"]))
        t_range = row["t_nom_reference_range_ms" if ms else "t_nom_reference_range_s"]
        for side, t_end, printed in (("upper", t_range[0], row["M_star_range"][1]), ("lower", t_range[1], row["M_star_range"][0])):
            point, lo, hi = span(f, P(mt), P(t_end))
            audit.compare("S23b", f"{SHORT[d]}: M* range {side} end", point, printed, low=lo, high=hi, half=_sig_half(printed))


def _sig_half(value: float, digits: int = 2) -> float:
    exponent = math.floor(math.log10(abs(value)))
    return 0.5 * 10 ** (exponent - digits + 1)


# ============================================================================ Table 2, S44, Note S19
def _ratio(registered, d):
    stats = registered["dataset_statistics_table_S6c_S23b"][d]
    width = registered["backbones_table_S5"][d]["hidden"]
    if "nodes" in stats:
        return stats["reported_edges"] / (stats["nodes"] * width)
    return stats["mean_reported_edges_per_graph"] / (stats["mean_nodes_per_graph"] * width)


def check_costs(audit, registered, reported):
    t1 = reported["main_table_1_policy_and_deployment_ledger"]
    t2 = reported["main_table_2_gain_and_cost_ledger"]
    s23 = reported["supplementary_table_S23_assurance_and_expected_compute"]
    s44 = reported["supplementary_table_S44_per_input_cost_model"]
    for d in DATASETS:
        if d == "ogbn-products":
            continue
        added = s23[d]["expected_serial_cost"] - s23[d]["comparator_cost"]
        lo, hi = t1[d]["loss_effect_interval_nats"]
        for side, end, printed in (("lower", hi, t2[d]["lambda_star_range"][0]), ("upper", lo, t2[d]["lambda_star_range"][1])):
            point, plo, phi = span(lambda e, a: abs(e) / a * 1e3, P(end), P(added, 0.0005))
            audit.compare("T2", f"{SHORT[d]}: lambda* range {side} end from the 95% interval of Delta (mnats)", point, printed,
                          low=plo, high=phi, decimals=1)
        # Table 1 prints four-decimal native AP for Peptides. The registry stores
        # 100 * AP in official_gain_pp, so its 0.00005 half-unit also scales by
        # 100, to 0.005 pp. All other Table 1 official gains print two-decimal pp.
        metric_half, metric_note = 0.005, ""
        if d == "Peptides-func":
            audit.compare("T1/T2", "Peptides: native AP change converted to percentage points", 
                          100 * t1[d]["official_gain_native"], t1[d]["official_gain_pp"], decimals=2)
            metric_note = ("Table 1 prints native AP; the registry converts both the gain and its print precision to "
                           "percentage points. Table 2's 0.79 follows percentage points per work; in native AP per work "
                           "the corresponding value is about 0.0079. This check makes the unit reading explicit.")
        point, plo, phi = span(lambda g, a: g / a, P(t1[d]["official_gain_pp"], metric_half), P(added, 0.0005))
        audit.compare("T2", f"{SHORT[d]}: official-metric percentage points per unit of added work", point,
                      t2[d]["lambda_star_metric_rate"], low=plo, high=phi, decimals=2, note=metric_note)
        row = s44[d]
        for key in ("f99_low_central_high", "C_099_low_central_high", "overhead_pct_low_central_high"):
            _sequence(row[key], 3, f"S44 {SHORT[d]} {key}")
        r = _ratio(registered, d)
        order = TEXT_CLAIMS["note_s19"]["order"]
        v = r * (order - 1) / (1 + r)
        audit.compare("S44", f"{SHORT[d]}: v = r (T - 1)/(1 + r) with r from the dataset statistics", v, row["v"], decimals=3)
        o_ctrl = s23[d]["controller_cost"] / s23[d]["comparator_cost"] - 1
        audit.compare("S44", f"{SHORT[d]}: o_ctrl = C_controller/C_qC - 1", o_ctrl, row["o_ctrl"], decimals=3)
        o_fix = o_ctrl - row["f_bar"] * v
        audit.compare("S44", f"{SHORT[d]}: o_fix = o_ctrl - f_bar v", o_fix, row["o_fix"], decimals=3,
                      low=o_fix - printed_half_unit(row["f_bar"]) * v, high=o_fix + printed_half_unit(row["f_bar"]) * v)
        C = s23[d]["comparator_cost"]
        for i, (shift, f99, cost, pct) in enumerate(zip((0.10, 0.20, 0.35), row["f99_low_central_high"],
                                                        row["C_099_low_central_high"], row["overhead_pct_low_central_high"])):
            label = ("low", "central", "high")[i]
            audit.compare("S44", f"{SHORT[d]}: f99 = f_bar + {shift:.2f} ({label})", row["f_bar"] + shift, f99, decimals=2)
            f = lambda fb: C * (1 + o_ctrl - fb * v + (fb + shift) * v)  # noqa: E731
            point, plo, phi = span(f, P(row["f_bar"]))
            audit.compare("S44", f"{SHORT[d]}: C_0.99 ({label}) = C_qC [1 + o_fix + f99 v]", point, cost, low=plo, high=phi,
                          decimals=2)
            audit.compare("S44", f"{SHORT[d]}: C_0.99/C_qC - 1 ({label}, percent)", 100 * (point / C - 1), pct,
                          low=100 * (plo / C - 1), high=100 * (phi / C - 1))
        audit.compare("T2/S44", f"{SHORT[d]}: Table 2 C_0.99 is the central scenario", row["C_099_low_central_high"][1], t2[d]["C_099"],
                      decimals=1)
        audit.equal("T2/S44", f"{SHORT[d]}: Table 2 tail-scenario range is the low and high scenarios",
                    [round(row["C_099_low_central_high"][0] + 1e-9, 1), round(row["C_099_low_central_high"][2] + 1e-9, 1)],
                    t2[d]["C_099_range"])
        audit.compare("T2/S44", f"{SHORT[d]}: Table 2 C_0.99 overhead", row["overhead_pct_low_central_high"][1], t2[d]["C_099_overhead_pct"])
    note = TEXT_CLAIMS["note_s19"]
    s, f_res, order = note["s"], note["f_res"], note["order"]
    for d, printed in note["T_max"].items():
        r = _ratio(registered, d)
        m = s23[d]["controller_cost"] / s23[d]["comparator_cost"]
        t_max = 1 + (m * (1 + r) - (1 + r + s)) / (f_res * r)
        audit.equal("Note S19", f"{SHORT[d]}: admissible truncation order at s = {s}, f_res = {f_res}", math.floor(t_max), printed,
                    note=f"T <= {t_max:.2f}")
    for d, printed in note["T_min_exceeded"].items():
        r = _ratio(registered, d)
        m = s23[d]["controller_cost"] / s23[d]["comparator_cost"]
        t_max = 1 + (m * (1 + r) - (1 + r + s)) / (f_res * r)
        audit.add("Note S19", f"{SHORT[d]}: admissible truncation order exceeds {printed}", "PASS" if t_max > printed else "FAIL",
                  round(t_max, 1), printed)
    r = _ratio(registered, "ogbn-proteins")
    m = s23["ogbn-proteins"]["controller_cost"] / s23["ogbn-proteins"]["comparator_cost"]
    audit.compare("Note S19", "proteins: admissible truncation order at f_res = 0.5", 1 + (m * (1 + r) - (1 + r + s)) / (f_res * r),
                  note["proteins_T_max"])
    audit.compare("Note S19/S6", "proteins: largest resolvent fraction compatible with T = 8",
                  (m * (1 + r) - (1 + r + s)) / (r * (order - 1)), note["proteins_f_res_max"])
    executed = reported["supplementary_table_S6c_sensing_cost_identity"]["ogbn-proteins"]["executed_resolvent_fraction_f_res"]
    audit.add("Note S19/S6c", "proteins: executed resolvent fraction within the admissible fraction",
              "PASS" if executed <= float(note["proteins_f_res_max"]) else "FAIL", executed, note["proteins_f_res_max"])


# ============================================================================ S26b
def check_ensemble_law(audit, reported):
    s26b = reported["supplementary_table_S26b_ensemble_law_and_break_even"]
    s23 = reported["supplementary_table_S23_assurance_and_expected_compute"]
    s41 = reported["supplementary_table_S41_qualification_cost_components"]
    t1 = reported["main_table_1_policy_and_deployment_ledger"]
    kappas = {}
    for d, row in s26b.items():
        kappa = row["D_inf_hat"] / row["l_plant_nats"]
        kappas[SHORT[d]] = round(kappa, 4)
        audit.compare("S26b", f"{SHORT[d]}: kappa = D_inf/l_plant", kappa, row["kappa_hat"], decimals=3,
                      low=row["D_inf_hat"] / (row["l_plant_nats"] + 0.005), high=row["D_inf_hat"] / (row["l_plant_nats"] - 0.005))
        for i, k in enumerate((0.03, 0.04)):
            audit.compare("S26b", f"{SHORT[d]}: D_inf band end {k} l_plant", k * row["l_plant_nats"], row["D_inf_band"][i], decimals=4)
        E = t1[d]["deployed_E"]
        o_exp = s23[d]["expected_serial_cost"] / s23[d]["comparator_cost"] - 1
        o_ass = s23[d]["assurance_serial_cost"] / s23[d]["comparator_cost"] - 1
        for label, o, printed in (("expected", o_exp, row["break_even_expected"]), ("assurance", o_ass, row["break_even_assurance"])):
            for i, D in enumerate(row["D_inf_band"]):
                audit.compare("S26b", f"{SHORT[d]}: break-even D_inf o/(E(1+o)) at the {label} overhead, band end {i + 1}",
                              D * o / (E * (1 + o)), printed[i], decimals=4)
        audit.compare("S26b", f"{SHORT[d]}: measured |Delta| equals Table 1", abs(t1[d]["loss_effect_nats"]), row["measured_abs_Delta"],
                      decimals=4)
        audit.compare("S26b", f"{SHORT[d]}: GPU-hours of eight further checkpoints",
                      row["extra_checkpoints_E16_20"] * s41[d]["backbone_training_GPU_h_per_seed"], row["extra_GPU_h"])
    rounded = {round(row["D_inf_hat"] / row["l_plant_nats"], 3) for row in s26b.values()}
    audit.add("S26b", "fitted kappa = D_inf/l_plant across datasets", "INFO", kappas, None,
              f"rounds to {rounded.pop():.3f} on all {len(s26b)} datasets, the centre of the registered band [0.03, 0.04]"
              if len(rounded) == 1 else "")


# ============================================================================ S34, S42, S45, S46, S46b, S47
def _phi_grid(reported, d):
    row = reported["supplementary_table_S30_oracle_dispersion"][d]
    return row["phi_x_grid"][0], row["phi_x_central"], row["phi_x_grid"][1]


def check_budget_matched(audit, reported):
    t1 = reported["main_table_1_policy_and_deployment_ledger"]
    s23 = reported["supplementary_table_S23_assurance_and_expected_compute"]
    s34 = reported["supplementary_table_S34_budget_aware_rule"]
    s42 = reported["supplementary_table_S42_power_of_budget_matched_test"]
    s45 = reported["supplementary_table_S45_inserted_site_attribution"]
    s46 = reported["supplementary_table_S46_budget_matched_frontier"]
    s46b = reported["supplementary_table_S46b_arm_level_outcomes"]
    s47 = reported["supplementary_table_S47_decision_boundary_and_power"]
    expected = set(DATASETS) - {"ogbn-products"}
    for label, table in (("S34", s34), ("S42", s42), ("S46", s46), ("S46b", s46b), ("S47", s47)):
        _require_keys(table, expected, label)
    z_holm = NORMAL.inv_cdf(1 - 0.05 / 4 / 2)
    model_equal, model_cells, implied_ol = 0, 0, {}
    powers = {}
    for d in s46:
        g = abs(t1[d]["loss_effect_nats"]) * 1e3  # mnats
        o = s23[d]["expected_serial_cost"] / s23[d]["comparator_cost"] - 1
        row34 = s34[d]
        point, lo, hi = span(lambda skip, switch: 1 - skip * switch, P(row34["skip_rate"]), P(row34["switch_fraction"]))
        audit.compare("S34", f"{SHORT[d]}: r_B = 1 - skip x switch", point, row34["retained_fraction_r_B"], low=lo, high=hi, decimals=3)
        audit.compare("S34", f"{SHORT[d]}: present overhead equals the expected overhead of S23 (percent)", 100 * o,
                      row34["present_overhead_pct"], decimals=1)
        # S42: minimum detectable effects, margin and power
        r42 = s42[d]
        se = r42["paired_SE_nats"] * 1e3
        se_half = 0.000005 * 1e3  # S42 prints five decimal places, including Pascal's 0.00050
        for label, z, printed in (("0.05", Z975, r42["MDE_mnats_0.05"]), ("Holm step", z_holm, r42["MDE_mnats_holm"])):
            multiplier = z + NORMAL.inv_cdf(0.8)
            point, lo, hi = span(lambda s: multiplier * s, (se, se_half))
            audit.compare("S42", f"{SHORT[d]}: minimum detectable effect ({label}, mnats)", point, printed, low=lo, high=hi, decimals=2)
        phi_lo, phi_c, phi_hi = _phi_grid(reported, d)
        g_half = 0.000005 * 1e3  # Table 1 paired loss prints five decimals, including 0.00450/0.00380
        audit.compare("S42", f"{SHORT[d]}: margin FB - E3 = (1 - phi) g at the central phi", (1 - phi_c) * g,
                      r42["reported_margin_mnats"], low=(1 - phi_c) * (g - g_half), high=(1 - phi_c) * (g + g_half), decimals=2)
        for i, phi in enumerate((phi_hi, phi_lo)):
            audit.compare("S42", f"{SHORT[d]}: margin range end {i + 1}", (1 - phi) * g, r42["margin_range"][i],
                          low=(1 - phi) * (g - g_half), high=(1 - phi) * (g + g_half), decimals=2)
        power = lambda m, s, z: 100 * (NORMAL.cdf(m / s - z) + NORMAL.cdf(-m / s - z))  # noqa: E731
        for label, margin, printed, z in (("central", r42["reported_margin_mnats"], r42["power_at_unit_pct"], Z975),
                                          ("range low", r42["margin_range"][0], r42["power_range"][0], Z975),
                                          ("range high", r42["margin_range"][1], r42["power_range"][1], Z975),
                                          ("Holm step", r42["reported_margin_mnats"], r42["power_holm_step_pct"], z_holm)):
            point, lo, hi = span(lambda m, s: power(m, s, z), P(margin), (se, se_half))
            audit.compare("S42", f"{SHORT[d]}: power at the registered unit ({label}, percent)", point, printed, low=lo, high=hi)
        powers[d] = r42["power_at_unit_pct"]
        # S46: retained-share model at the common ceiling
        r46 = s46[d]
        audit.compare("S46", f"{SHORT[d]}: common ceiling 1 + o", 1 + o, r46["ceiling"], decimals=2)
        audit.compare("S46", f"{SHORT[d]}: feedback effect (mnats)", -g, r46["feedback_effect_mnats"])
        audit.compare("S46", f"{SHORT[d]}: present overhead (percent)", 100 * o, r46["present_overhead_pct"])
        rb = row34["retained_fraction_r_B"]
        model = {"E3": -phi_c * g, "E4": -0.9 * phi_c * g, "E6": -rb * g}
        ranges = {"E3": (-phi_lo * g, -phi_hi * g), "E4": (-0.9 * phi_lo * g, -0.9 * phi_hi * g)}
        mpc_ol = t1[d]["qC_identity"].startswith("MPC-OL")
        if mpc_ol:
            model["E5"] = -0.08 * g
            ranges["E5"] = (-0.03 * g, -0.15 * g)
        printed = {"E3": r46["E3_predictor"], "E4": r46["E4_distilled"], "E5": r46["E5_no_observation_replanning"],
                   "E6": r46["E6_qB_effect"]}
        printed_ranges = {"E3": r46["E3_range"], "E4": r46["E4_range"], "E5": r46["E5_range"]}
        for arm, values in printed_ranges.items():
            _sequence(values, 2, f"S46 {SHORT[d]} {arm} range")
        for arm, value in model.items():
            audit.compare("S46", f"{SHORT[d]}: {arm} effect of the retained-share model (mnats)", value, printed[arm],
                          low=value * (g - g_half) / g, high=value * (g + g_half) / g)
        for arm, (a, b) in ranges.items():
            for i, value in enumerate((a, b)):
                audit.compare("S46", f"{SHORT[d]}: {arm} grid range end {i + 1}", value, printed_ranges[arm][i],
                              low=value * (g - g_half) / g, high=value * (g + g_half) / g)
        if not mpc_ol:
            # fixed-sequence regimes: E5 = -(rho + phi_OL) g with rho in {0.03, 0.08, 0.15}; phi_OL is not printed
            bounds = []
            for rho, value in zip((0.03, 0.08, 0.15), (printed_ranges["E5"][0], printed["E5"], printed_ranges["E5"][1])):
                h = 0.05
                bounds.append(((abs(value) - h) / g - rho, (abs(value) + h) / g - rho))
            lo_ol, hi_ol = max(b[0] for b in bounds), min(b[1] for b in bounds)
            implied_ol[SHORT[d]] = [lo_ol, hi_ol] if lo_ol <= hi_ol else "inconsistent"
        audit.compare("S46/S34", f"{SHORT[d]}: E6 overhead equals the S34 q_B overhead (percent)", row34["qB_overhead_pct"],
                      r46["E6_overhead_pct"])
        # S46b: arm-level outcomes against the frozen teacher
        arms = s46b[d]["arms"]
        _require_keys(arms, {"E3", "E4", "E5", "E6"}, f"S46b {SHORT[d]} arms")
        raw = {}
        for arm, a in arms.items():
            audit.compare("S46b", f"{SHORT[d]}: {arm} effect vs q_M = effect vs q_C - feedback effect",
                          a["effect_vs_qC_mnats"] + g, a["effect_vs_qM_mnats"], low=a["effect_vs_qC_mnats"] + g - 0.05 - g_half,
                          high=a["effect_vs_qC_mnats"] + g + 0.05 + g_half)
            _interval_contains_estimate(audit, "S46b", f"{SHORT[d]}: {arm} estimate and bootstrap interval are compatible",
                                        a["interval_vs_qM_mnats"], a["effect_vs_qM_mnats"])
            width = (a["interval_vs_qM_mnats"][1] - a["interval_vs_qM_mnats"][0]) / 2
            audit.compare("S46b", f"{SHORT[d]}: {arm} interval half-width equals 1.96 x the S42 paired SE (mnats)", Z975 * se,
                          width, low=Z975 * (se - se_half), high=Z975 * (se + se_half), half=0.1)
            model_cells += 1
            model_equal += abs(a["effect_vs_qC_mnats"] - printed[arm]) < 1e-9
            raw[arm] = a
        names = list(raw)
        lows = holm([two_sided((abs(raw[n]["effect_vs_qM_mnats"]) + 0.05) / (se - se_half)) for n in names])
        highs = holm([two_sided(max(abs(raw[n]["effect_vs_qM_mnats"]) - 0.05, 0.0) / (se + se_half)) for n in names])
        for i, n in enumerate(names):
            audit.add("S46b", f"{SHORT[d]}: {n} Holm P within the four-contrast family (normal reading with the S42 SE)",
                      "PASS" if p_matches(lows[i], highs[i], raw[n]["holm_P"]) else "FAIL",
                      [round(lows[i], 5), round(highs[i], 5)], raw[n]["holm_P"])
        # S47: decision boundaries and power
        r47 = s47[d]
        _sequence(r47["phi_FB_ensemble"], 2, f"S47 {SHORT[d]} phi_FB_ensemble")
        _sequence(r47["phi_B_ensemble"], 2, f"S47 {SHORT[d]} phi_B_ensemble")
        audit.equal("S47", f"{SHORT[d]}: contract crossover against full feedback", r47["phi_FB_contract"], 1)
        audit.compare("S47", f"{SHORT[d]}: contract crossover against q_B equals r_B", rb, r47["phi_B_contract"], decimals=3)
        audit.compare("S47/S42", f"{SHORT[d]}: margin FB - E3", r42["reported_margin_mnats"], r47["margin_FB_E3_mnats"])
        audit.compare("S47/S42", f"{SHORT[d]}: minimum detectable effect at the unit", r42["MDE_mnats_0.05"], r47["MDE_unit"])
        audit.equal("S47/S42", f"{SHORT[d]}: power at the unit", (r42["power_at_unit_pct"], r42["power_range"]),
                    (r47["power_unit_pct"], r47["power_unit_range"]))
        # S47: ensemble-spend crossovers of Proposition 10(ii); o_dec is not printed, so search for one that fits
        s26b = reported["supplementary_table_S26b_ensemble_law_and_break_even"][d]
        E = t1[d]["deployed_E"]
        o_b = row34["qB_overhead_pct"] / 100
        g_n = g / 1e3
        b = lambda D, spare: D * spare / (E * (1 + spare))  # noqa: E731
        fitting = []
        for step in range(0, 2001):
            o_dec = o * step / 2000
            fb = sorted(1 - b(D, o - o_dec) / g_n for D in s26b["D_inf_band"])
            bb = sorted(rb - (b(D, o - o_dec) - b(D, o - o_b)) / g_n for D in s26b["D_inf_band"])
            if all(abs(x - y) <= 0.005 + 1e-9 for x, y in zip(fb, r47["phi_FB_ensemble"])) and \
                    all(abs(x - y) <= 0.0005 + 0.0005 + 1e-9 for x, y in zip(bb, r47["phi_B_ensemble"])):
                fitting.append(o_dec)
        if fitting:
            audit.add("S47", f"{SHORT[d]}: ensemble-spend crossovers compatible with an inferred decoding overhead o_dec", "INFO",
                      [round(min(fitting), 4), round(max(fitting), 4)], {"phi_FB": r47["phi_FB_ensemble"], "phi_B": r47["phi_B_ensemble"]},
                      "o_dec is not printed: this inverse-fit grid is a compatibility diagnostic, not reproduction from "
                      "prespecified inputs. The range lists sampled values that fit both printed crossover ranges.")
        else:
            audit.add("S47", f"{SHORT[d]}: ensemble-spend crossovers reproduced by one decoding overhead o_dec", "ADVISORY",
                      None, {"phi_FB": r47["phi_FB_ensemble"], "phi_B": r47["phi_B_ensemble"]},
                      "no single o_dec in [0, o] reproduces both printed ranges; print o_dec and o_B with Table S47")
        # S45: native-only share of the deployed gain
        if d in s45:
            r45 = s45[d]
            point, lo, hi = span(lambda a, b: a / b, P(r45["arm1prime_native_feedback"]), P(r45["arm3_inserted_feedback_deployed"]))
            audit.compare("S45", f"{SHORT[d]}: retained share of the native-only feedback arm", point, r45["retained_share_arm1prime"],
                          low=lo, high=hi, decimals=2)
    if implied_ol:
        valid = [v for v in implied_ol.values() if v != "inconsistent"]
        common = [max(v[0] for v in valid), min(v[1] for v in valid)] if valid else [1.0, 0.0]
        all_fit = len(valid) == len(implied_ol) and common[0] <= common[1]
        audit.add("S46", "phi_OL of the no-observation arm in fixed-sequence regimes (not printed)", "INFO", implied_ol, None,
                  "interval of phi_OL values reproducing the printed E5 entries"
                  + (f"; one value in [{common[0]:.6g}, {common[1]:.6g}] fits all {len(implied_ol)} datasets" if all_fit
                     else "; no common value was established across all fixed-sequence datasets")
                  + "; Note S25 names phi_OL without a value")
    audit.add("S46b", "arm effects against q_C equal the retained-share model at the central phi", "INFO",
              f"{model_equal} of {model_cells}", None,
              "Table S46b reports executed arms; equality with the S33/S46 model cells in every arm and dataset is worth a "
              "sentence in the caption")
    claims = TEXT_CLAIMS["power_ranges"]
    registered_units = [min(powers.values()), max(powers.values())]
    outside = [v for d, v in powers.items() if d != "Roman-empire"]
    seeds = [v for d in s42 for v in s42[d]["power_seed_paired_pct"]]
    audit.equal("S42/main", "power at the registered units (range)", registered_units, claims["registered_units_pct"])
    audit.equal("S42/main", "power outside the Roman unit (range)", [min(outside), max(outside)], claims["outside_Roman_pct"])
    audit.equal("S42/main", "power with twelve paired seeds (range)", [min(seeds), max(seeds)], claims["seed_paired_pct"])


def check_main_text(audit, reported):
    t1 = reported["main_table_1_policy_and_deployment_ledger"]
    effects = [abs(t1[d]["loss_effect_nats"]) for d in DATASETS if t1[d]["loss_effect_nats"]]
    claim = TEXT_CLAIMS["effect_range"]["nats"]
    audit.equal("main", "feedback improvements on six datasets (Discussion, nats)", [round(min(effects), 3), round(max(effects), 3)],
                claim)
    note = TEXT_CLAIMS["note_s17_hoeffding"]
    for key, printed in note.items():
        if key == "where":
            continue
        n = int(key[3:-1])
        value = 2 * 2 * 0.10 * math.sqrt(math.log(2 * 10 / 0.05) / (2 * n))
        four = round(value, 4)
        audit.compare("Note S17", f"Hoeffding regret bound {key}", value, printed,
                      alternative=(four, f"rounding the four-decimal value {four} of Table S28b again"))


GROUPS = (check_roles, check_entry_contract, check_support, check_selection, check_admission, check_mechanism,
          lambda a, reg, rep: check_factorials(a, rep), lambda a, reg, rep: check_tube_and_amortization(a, rep),
          check_costs, lambda a, reg, rep: check_ensemble_law(a, rep), lambda a, reg, rep: check_budget_matched(a, rep),
          lambda a, reg, rep: check_main_text(a, rep))


def run(registered: dict | None = None, reported: dict | None = None) -> dict:
    registered = load_registered() if registered is None else registered
    reported = load_reported() if reported is None else reported
    audit = Audit()
    for group in GROUPS:
        try:
            group(audit, registered, reported)
        except (KeyError, TypeError, ValueError, ZeroDivisionError, StopIteration) as exc:
            audit.add("input", getattr(group, "__name__", "group"), "FAIL", None, None, f"{type(exc).__name__}: {exc}")
    counts = {s: sum(r["status"] == s for r in audit.rows) for s in STATUSES}
    return {"schema_version": "si-identities-1", "counts": counts, "rows": audit.rows}


def render_markdown(result: dict) -> str:
    lines = ["# Supplementary identities", "",
             ", ".join(f"{k} {v}" for k, v in result["counts"].items()), "",
             "| status | group | check | computed | printed | note |", "|---|---|---|---|---|---|"]
    order = {"FAIL": 0, "ADVISORY": 1, "INFO": 2, "PASS": 3}
    for r in sorted(result["rows"], key=lambda r: order[r["status"]]):
        cells = [r["status"], r["group"], r["check"], json.dumps(r["computed"])[:80], json.dumps(r["printed"])[:60], r["note"]]
        lines.append("| " + " | ".join(str(c).replace("|", "/") for c in cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json-output", type=Path, help="write every row as JSON")
    parser.add_argument("--markdown", type=Path, help="write a Markdown report")
    parser.add_argument("--strict", action="store_true", help="ADVISORY findings also fail")
    parser.add_argument("--quiet", action="store_true", help="print findings and the summary only")
    args = parser.parse_args(argv)
    result = run()
    groups: dict[str, list[int]] = {}
    for r in result["rows"]:
        tally = groups.setdefault(r["group"].split("/")[0], [0, 0, 0, 0])
        tally[STATUSES.index(r["status"])] += 1
    if not args.quiet:
        print(f"{'group':12s} {'pass':>5s} {'advis':>5s} {'fail':>5s} {'info':>5s}")
        for group, (ok, advisory, fail, info) in groups.items():
            print(f"{group:12s} {ok:5d} {advisory:5d} {fail:5d} {info:5d}")
    for r in result["rows"]:
        if r["status"] in ("FAIL", "ADVISORY"):
            print(f"{r['status']} [{r['group']}] {r['check']}: computed={r['computed']!r} printed={r['printed']!r} {r['note']}")
    c = result["counts"]
    print(f"{c['PASS']} of {c['PASS'] + c['FAIL'] + c['ADVISORY']} enumerated identity checks passed; "
          f"{c['ADVISORY']} advisory, {c['FAIL']} failed, {c['INFO']} derived values reported")
    for path, text in ((args.json_output, json.dumps(result, indent=2) + "\n"),
                       (args.markdown, render_markdown(result) if args.markdown else None)):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    return 1 if c["FAIL"] or (args.strict and c["ADVISORY"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
