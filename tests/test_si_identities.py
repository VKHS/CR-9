"""Supplementary derived-value audit (scripts/check_si_identities.py); standard library only."""
from __future__ import annotations

import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts import check_si_identities as si  # noqa: E402
from scripts._paper_common import load_registered, load_reported  # noqa: E402

# Expected findings from the current PDF and the complete committed registry.
# Reconcile changes with the source review and registry; do not bless new counts
# merely because many unrelated checks still pass.
KNOWN_ADVISORIES = {
    ("S24", "Pascal: implied seed-level SD sqrt(S) (hw_test^2 - hw_graph^2)^(1/2) / 1.96"),
    ("S24", "Peptides: implied seed-level SD sqrt(S) (hw_test^2 - hw_graph^2)^(1/2) / 1.96"),
    ("Cor 2", "printed counts follow one rounding convention"),
    ("S16b", "Pascal interactions within 1e-4 nats of zero at the tabulated precision"),
    ("Note S17", "Hoeffding regret bound 2e(74)"),
}


class CommittedRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = si.run()

    def test_no_failures_and_broad_coverage(self):
        failures = [r for r in self.result["rows"] if r["status"] == "FAIL"]
        self.assertEqual(failures, [])
        # Counts include cross-references and diagnostics; a large PASS total is
        # not an inventory of unique manuscript cells or a completeness check.
        keys = [(r["group"], r["check"]) for r in self.result["rows"]]
        self.assertTrue(keys)
        self.assertEqual(len(keys), len(set(keys)))
        groups = {r["group"].split("/")[0] for r in self.result["rows"]}
        for group in ("S2", "S4", "S7", "S12", "S12c", "Theorem 2", "Cor 2", "S14", "S14b", "S16", "S16b", "S22",
                      "S23b", "S26b", "S34", "S42", "S44", "S45", "S46", "S46b", "S47", "Note S19", "T2"):
            self.assertIn(group, groups)

    def test_known_advisories_are_exactly_reported(self):
        advisories = {(r["group"], r["check"]) for r in self.result["rows"] if r["status"] == "ADVISORY"}
        self.assertEqual(advisories, KNOWN_ADVISORIES)
        for r in self.result["rows"]:
            if r["status"] == "ADVISORY":
                self.assertTrue(r["note"], r["check"])

    def test_inferred_parameters_and_boundary_label_are_diagnostics(self):
        inferred = [r for r in self.result["rows"] if "inferred decoding overhead" in r["check"]]
        self.assertEqual(len(inferred), 6)
        self.assertTrue(all(r["status"] == "INFO" for r in inferred))
        boundary = [r for r in self.result["rows"] if "labelled the boundary pass" in r["check"]]
        self.assertEqual([r["status"] for r in boundary], ["INFO"])

    def test_every_comparison_arm_has_an_interval_diagnostic(self):
        rows = [r for r in self.result["rows"] if "estimate and bootstrap interval are compatible" in r["check"]]
        self.assertEqual(len(rows), 6 * 4)
        self.assertTrue(all(r["status"] == "PASS" for r in rows))

    def test_text_claims_name_their_location(self):
        for key, claim in si.TEXT_CLAIMS.items():
            self.assertIn("where", claim, key)

    def test_rows_are_json_serializable(self):
        json.dumps(self.result)


class CorruptionTests(unittest.TestCase):
    def audit(self, mutate):
        reported, registered = copy.deepcopy(load_reported()), copy.deepcopy(load_registered())
        mutate(reported, registered)
        return si.run(registered, reported)

    def failed(self, result, fragment):
        return [r for r in result["rows"] if r["status"] == "FAIL" and fragment in r["check"]]

    def test_changed_tail_cost_fails(self):
        def mutate(rep, reg):
            rep["supplementary_table_S44_per_input_cost_model"]["ogbn-arxiv"]["C_099_low_central_high"][1] = 4.90
        self.assertTrue(self.failed(self.audit(mutate), "C_0.99 (central)"))

    def test_changed_holm_value_fails(self):
        def mutate(rep, reg):
            rep["supplementary_table_S14_mechanism_contrasts_submitted_protocol"][0]["holm_P"] = 0.02
        self.assertTrue(self.failed(self.audit(mutate), "arxiv/state: Holm-adjusted P"))

    def test_changed_factorial_cell_fails(self):
        def mutate(rep, reg):
            rep["supplementary_table_S16_arxiv_factorial"]["cells"]["111"]["log_depth_CE"] = 0.8500
        self.assertTrue(self.failed(self.audit(mutate), "arxiv"))

    def test_changed_role_size_fails(self):
        def mutate(rep, reg):
            reg["data_roles_table_S2"]["multi_graph_counts"]["COCO-SP"]["D_confirm"] = 5665
        self.assertTrue(self.failed(self.audit(mutate), "COCO: n_conf"))

    def test_missing_open_loop_representative_cannot_disappear_from_selection_audit(self):
        def mutate(rep, reg):
            table = rep["supplementary_table_S12_selection_ledger"]
            table["Roman-empire"] = [row for row in table["Roman-empire"] if row["route"] != "MPC-OL"]
        result = self.audit(mutate)
        self.assertTrue(any(row["status"] == "FAIL" and "S12 Roman representative routes" in row["note"]
                            for row in result["rows"]))

    def test_duplicate_representative_route_is_not_a_complete_family(self):
        def mutate(rep, reg):
            rows = rep["supplementary_table_S12_selection_ledger"]["ogbn-arxiv"]
            rows.append(copy.deepcopy(rows[0]))
        result = self.audit(mutate)
        self.assertTrue(any(row["status"] == "FAIL" and "S12 arxiv representative routes" in row["note"]
                            for row in result["rows"]))

    def test_missing_contrast_fails_even_if_the_smaller_holm_family_is_recomputed(self):
        def mutate(rep, reg):
            key = "supplementary_table_S14_mechanism_contrasts_submitted_protocol"
            rep[key] = [row for row in rep[key] if row["contrast"] != "arxiv/radius"]
            raw = [sum(si.parse_p(row["raw_P"])[:2]) / 2 for row in rep[key]]
            for row, adjusted in zip(rep[key], si.holm(raw)):
                row["holm_P"] = adjusted
        result = self.audit(mutate)
        self.assertTrue(any(row["status"] == "FAIL" and "S14 Holm family" in row["note"]
                            for row in result["rows"]))

    def test_duplicate_confirmation_identifier_cannot_replace_a_contrast(self):
        def mutate(rep, reg):
            rows = rep["supplementary_table_S14_mechanism_contrasts_submitted_protocol"]
            rows[1]["contrast"] = rows[0]["contrast"]
        result = self.audit(mutate)
        self.assertTrue(any(row["status"] == "FAIL" and "S14 Holm family" in row["note"]
                            for row in result["rows"]))

    def test_native_ap_and_percentage_point_gains_cannot_diverge(self):
        def mutate(rep, reg):
            rep["main_table_1_policy_and_deployment_ledger"]["Peptides-func"]["official_gain_native"] = 0.58
        self.assertTrue(self.failed(self.audit(mutate), "native AP change converted to percentage points"))

    def test_native_ap_precision_scales_with_the_percentage_point_conversion(self):
        # A nearby work denominator makes this check depend on the gain's source
        # precision. Native AP 0.0058 +/- 0.00005 becomes 0.58 +/- 0.005 pp;
        # using +/- 0.00005 pp would incorrectly reject the printed 0.79 rate.
        reported, registered = copy.deepcopy(load_reported()), load_registered()
        reported["supplementary_table_S23_assurance_and_expected_compute"]["Peptides-func"]["expected_serial_cost"] = 4.725
        audit = si.Audit()
        si.check_costs(audit, registered, reported)
        row = next(row for row in audit.rows if row["check"] ==
                   "Peptides: official-metric percentage points per unit of added work")
        self.assertEqual(row["status"], "PASS", row)
        self.assertIn("rounding of the printed inputs", row["note"])

    def test_empty_crossover_ranges_fail_instead_of_vacuous_match(self):
        def mutate(rep, reg):
            row = rep["supplementary_table_S47_decision_boundary_and_power"]["ogbn-arxiv"]
            row["phi_FB_ensemble"] = []
            row["phi_B_ensemble"] = []
        result = self.audit(mutate)
        self.assertTrue(any(r["status"] == "FAIL" and "phi_FB_ensemble" in r["note"] for r in result["rows"]))

    def test_missing_arm_fails_instead_of_shrinking_holm_family(self):
        def mutate(rep, reg):
            del rep["supplementary_table_S46b_arm_level_outcomes"]["ogbn-arxiv"]["arms"]["E4"]
        result = self.audit(mutate)
        self.assertTrue(any(r["status"] == "FAIL" and "S46b arxiv arms" in r["note"] for r in result["rows"]))

    def test_shifted_interval_is_flagged_even_if_width_is_unchanged(self):
        def mutate(rep, reg):
            row = rep["supplementary_table_S46b_arm_level_outcomes"]["ogbn-arxiv"]["arms"]["E3"]
            row["interval_vs_qM_mnats"] = [v + 100.0 for v in row["interval_vs_qM_mnats"]]
        result = self.audit(mutate)
        self.assertTrue(any(r["status"] == "ADVISORY" and "arxiv: E3 estimate and bootstrap interval" in r["check"]
                            for r in result["rows"]))

    def test_missing_tail_scenario_fails(self):
        def mutate(rep, reg):
            rep["supplementary_table_S44_per_input_cost_model"]["ogbn-arxiv"]["f99_low_central_high"] = [0.3]
        result = self.audit(mutate)
        self.assertTrue(any(r["status"] == "FAIL" and "f99_low_central_high" in r["note"] for r in result["rows"]))

    def test_malformed_registry_is_a_failure_not_a_crash(self):
        def mutate(rep, reg):
            del rep["supplementary_table_S42_power_of_budget_matched_test"]
        result = self.audit(mutate)
        self.assertTrue([r for r in result["rows"] if r["group"] == "input" and r["status"] == "FAIL"])


class PrimitiveTests(unittest.TestCase):
    def test_compare_semantics(self):
        audit = si.Audit()
        self.assertEqual(audit.compare("g", "exact", 0.12346, 0.1235), "PASS")
        self.assertEqual(audit.compare("g", "box", 0.1239, 0.1235, low=0.1230, high=0.1240), "PASS")
        self.assertEqual(audit.compare("g", "alt", 0.1, 0.2, alternative=(0.2, "another reading")), "ADVISORY")
        self.assertEqual(audit.compare("g", "miss", 0.1, 0.2), "FAIL")
        self.assertEqual(audit.compare("g", "digits", 0.00299, "0.0030"), "PASS")
        self.assertEqual(audit.compare("g", "digits", 0.00291, "0.0030"), "FAIL")
        self.assertEqual(audit.compare("g", "decimals", 0.3504, 0.35, decimals=3), "PASS")
        self.assertEqual(audit.compare("g", "decimals", 0.3540, 0.35, decimals=3), "FAIL")

    def test_scientific_notation_and_printed_trailing_zeros(self):
        self.assertEqual(si._text_value("1e-4"), (0.0001, 0.00005))
        self.assertEqual(si._text_value("1.0e-4"), (0.0001, 0.000005))
        self.assertEqual(si._text_value("0.00010"), (0.0001, 0.000005))
        lo, hi, _ = si.parse_p("0.0030")
        self.assertAlmostEqual(lo, 0.00295)
        self.assertAlmostEqual(hi, 0.00305)
        self.assertEqual(si.Audit().compare("g", "scientific", 0.2, "1e-4"), "FAIL")
        for value in ("NaN", "Infinity", "-Infinity"):
            with self.assertRaises(ValueError):
                si._text_value(value)

    def test_reversed_rounding_endpoints_are_not_collapsed(self):
        audit = si.Audit()
        self.assertEqual(audit.compare("g", "decreasing", -1.60, "-1.70", low=-1.4, high=-1.8), "PASS")

    def test_incomplete_sequences_and_arms_are_rejected(self):
        for value in ([], [1.0], [1.0, 2.0, 3.0], [float("nan"), 1.0]):
            with self.assertRaises(ValueError):
                si._sequence(value, 2, "range")
        with self.assertRaises(ValueError):
            si._require_keys({"E3": {}}, {"E3", "E4", "E5", "E6"}, "arms")

    def test_bootstrap_interval_need_not_be_symmetric(self):
        audit = si.Audit()
        si._interval_contains_estimate(audit, "g", "asymmetric", [-1.0, 3.0], 0.0)
        self.assertEqual(audit.rows[-1]["status"], "PASS")
        si._interval_contains_estimate(audit, "g", "shifted", [99.0, 103.0], 0.0)
        self.assertEqual(audit.rows[-1]["status"], "ADVISORY")

    def test_p_values_and_holm(self):
        self.assertEqual(si.parse_p("< 1e-4")[:2], (0.0, 1e-4))
        for invalid in ("< 2", "< Infinity", "< NaN", "< 0"):
            with self.subTest(printed=invalid), self.assertRaises(ValueError):
                si.p_matches(0.01, 0.02, invalid)
        low, high, _ = si.parse_p(0.0014)
        self.assertAlmostEqual(low, 0.00135)
        self.assertAlmostEqual(high, 0.00145)
        self.assertEqual(si.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        self.assertTrue(si.p_matches(0.0005, 0.0007, "< 0.001"))
        self.assertFalse(si.p_matches(0.002, 0.003, "< 0.001"))

    def test_selection_exposes_candidate_domain_without_changing_default(self):
        # Current S12 Roman values. All three printed configurations are feasible;
        # MPC-OL clears -0.003 and lies within -0.00499 + 0.002.
        rows = [
            {"route": "MPC-OL", "U_s": -0.00326, "Delta_s_hat": -0.00472, "rank": 5},
            {"route": "radius", "U_s": -0.00389, "Delta_s_hat": -0.00539, "rank": 7},
            {"route": "MPC-RH", "U_s": -0.00499, "Delta_s_hat": -0.00664, "rank": 8},
        ]
        default = si._reading(rows, 0.003, 0.002, "U")
        all_rows = si._reading(rows, 0.003, 0.002, "U", candidate_scope="all_feasible")
        self.assertEqual(default[:2], (["radius", "MPC-RH"], "radius"))
        self.assertEqual(all_rows[:2], (["MPC-OL", "radius", "MPC-RH"], "MPC-OL"))
        self.assertEqual(si._reading(rows, 0.003, 0.002, "Delta")[1], "radius")
        self.assertEqual(si._reading(rows, 0.003, 0.002, "Delta", candidate_scope="all_feasible")[1], "MPC-OL")
        with self.assertRaises(ValueError):
            si._reading(rows, 0.003, 0.002, "U", candidate_scope="unspecified")

    def test_span_bounds_a_monotone_function(self):
        point, lo, hi = si.span(lambda a, b: a / b, (1.0, 0.05), (2.0, 0.05))
        self.assertAlmostEqual(point, 0.5)
        self.assertAlmostEqual(lo, 0.95 / 2.05)
        self.assertAlmostEqual(hi, 1.05 / 1.95)


class CommandLineTests(unittest.TestCase):
    def run_main(self, *args):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = si.main(list(args))
        return code, buffer.getvalue()

    def test_default_and_strict_exit_codes(self):
        self.assertEqual(self.run_main("--quiet")[0], 0)
        self.assertEqual(self.run_main("--quiet", "--strict")[0], 1)

    def test_reports_are_written(self):
        with tempfile.TemporaryDirectory(prefix="cm-si-") as tmp:
            code, out = self.run_main("--quiet", "--json-output", f"{tmp}/si.json", "--markdown", f"{tmp}/si.md")
            self.assertEqual(code, 0, out)
            self.assertEqual(json.loads(Path(tmp, "si.json").read_text())["schema_version"], "si-identities-1")
            self.assertIn("| ADVISORY |", Path(tmp, "si.md").read_text())

    def test_runs_without_site_packages(self):
        result = subprocess.run([sys.executable, "-S", "-B", str(PROJECT / "scripts" / "check_si_identities.py"), "--quiet"],
                                capture_output=True, text=True, cwd=PROJECT)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("enumerated identity checks passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
