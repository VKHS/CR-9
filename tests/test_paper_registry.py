"""The manuscript registry: executable constants equal registered constants, and the reported ledgers are arithmetically coherent.

These tests need only the standard library. They are the mechanised form of the
checks a reviewer performs by hand: that the action registry in ``core.py`` is
the registry of Supplementary Table S6, that its formulas reproduce Table S6b and
Figure S15b, that the seven full profiles carry the registered thresholds, and
that main Tables 1 and 2 follow from the supplementary ledgers they summarize.
When PyTorch is available, the registry formulas are additionally compared with
``cross_morse.core`` itself.
"""
from __future__ import annotations

import json
import math
import copy
import csv
import hashlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import check_paper_constants
from scripts import render_paper_results
from scripts._paper_common import (BCE_DATASETS, DATASETS, PAPER, REGISTERED, REPORTED, ROOT, STEMS, band_sensitivity,
                                   bernstein_penalty, executed_hops, executed_identity_coefficient, full_config,
                                   identity_mass, load_registered, load_reported, minimum_admission_J,
                                   parse_actions_from_core, printed_half_unit, runtime_source_fingerprint, peptides_confirmation_caption,
                                   resolved_policy_view, evaluated_arm_binding_targets)

try:  # optional: the torch-backed registry
    from cross_morse import core as torch_core
except Exception:  # pragma: no cover - torch absent
    torch_core = None


class RegistryFilesTests(unittest.TestCase):
    def test_registry_files_exist_and_name_their_provenance(self):
        for path in (REGISTERED, REPORTED):
            self.assertTrue(path.is_file(), path)
            value = json.loads(path.read_text(encoding="utf-8"))
            provenance = value["_provenance"]
            self.assertEqual(provenance["repository"], "https://github.com/VKHS/CR-9")
            self.assertIn("transcription_rule", provenance)
            self.assertIn("Scientific Reports", provenance["venue"])

    def test_every_dataset_has_rows_in_the_main_ledgers(self):
        reported = load_reported()
        for key in ("main_table_1_policy_and_deployment_ledger", "main_table_2_gain_and_cost_ledger",
                    "supplementary_table_S4_target_qualification", "supplementary_table_S12_selection_ledger",
                    "supplementary_table_S15_method_comparison", "supplementary_table_S20_selection_theorem_instantiation",
                    "supplementary_table_S23_assurance_and_expected_compute", "supplementary_table_S41_qualification_cost_components"):
            self.assertEqual({k for k in reported[key] if not k.startswith("_")}, set(DATASETS), key)

    def test_optional_registry_reports_are_generated_and_stale_copies_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "reference reports"
            command = [sys.executable, "-B", str(ROOT / "scripts/render_paper_results.py"),
                       "--output-dir", str(output)]
            result = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((output / "REGISTRY_SUMMARY.md").is_file())
            self.assertTrue((output / "paper_summary.csv").is_file())
            result = subprocess.run(command + ["--check"], capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            (output / "paper_summary.csv").write_text("manually changed\n")
            result = subprocess.run(command + ["--check"], capture_output=True, text=True, cwd=ROOT)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Stale generated view", result.stderr)

    def test_bindings_reference_registry_values(self):
        reported = load_reported()
        t1 = reported["main_table_1_policy_and_deployment_ledger"]
        for dataset in DATASETS:
            path = PAPER / "bindings" / f"{STEMS[dataset]}_full.json"
            self.assertTrue(path.is_file(), path)
            binding = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(binding["schema_version"], "result-binding-1")
            self.assertEqual(binding["status"], "pending_run")
            self.assertEqual(binding["run_identities"]["dataset"], dataset)
            for entry in binding["results"]:
                self.assertTrue(entry["json_pointer"].startswith("/"))
                self.assertIsInstance(entry["absolute_tolerance"], float)
                # an exact zero (the comparator-return case) binds with zero tolerance; everything else with its printed half-unit
                self.assertGreaterEqual(entry["absolute_tolerance"], 0)
                self.assertIsNone(entry["reported_value"])
                self.assertEqual(entry["absolute_tolerance"] == 0, entry.get("reference_value", entry.get("manuscript_value")) == 0)
            pointers = {e["json_pointer"]: e.get("reference_value", e.get("manuscript_value")) for e in binding["results"]}
            view = resolved_policy_view(reported, dataset)
            self.assertEqual(pointers["/paired_qD_minus_qC_loss/mean"], view["loss_effect_nats"])
            self.assertEqual(pointers["/paired_qD_minus_qC_metric_pp/mean"], view["official_gain_pp"])
            if dataset == "Peptides-func":
                self.assertEqual(len(binding["results"]), 9)
                self.assertEqual(len(binding["unavailable_evaluated_arm_targets"]), 7)
                self.assertTrue(all(row["target_role"] == "served_policy" for row in binding["results"]))


class ResolvedPolicyTests(unittest.TestCase):
    def setUp(self):
        self.reported, self.registered = load_reported(), load_registered()

    def contrast(self):
        return next(row for row in self.reported["supplementary_table_S14_mechanism_contrasts_submitted_protocol"]
                    if row["contrast"] == "Peptides/radius")

    def test_failed_confirmation_resolves_to_identity_without_mutating_observations(self):
        original = copy.deepcopy(self.reported)
        view = resolved_policy_view(self.reported, "Peptides-func", self.registered)
        self.assertEqual(self.reported, original)
        self.assertEqual(view["deployed_rule"], "qC returned")
        self.assertFalse(view["decision_basis"]["predecessor_eligible"])
        self.assertEqual(view["G_admit"], 0)
        self.assertEqual(view["admission_D_bar"], 0)
        self.assertGreater(view["admission_U_a"], 0)
        self.assertEqual(view["loss_effect_nats"], 0)
        self.assertEqual(view["official_gain_pp"], 0)
        self.assertIsNone(view["loss_effect_interval_nats"])
        self.assertEqual(view["scores"]["qD"]["official_score"], view["scores"]["qC"]["official_score"])
        self.assertEqual(view["costs"]["C_expected"], 4.0)
        self.assertEqual(view["costs"]["C_assure"], 4.0)
        self.assertIsNone(view["costs"]["C_099"])
        self.assertIsNone(view["costs"]["lambda_star_mnats"])
        self.assertEqual(view["reported_arm"]["loss_effect_nats"], -0.00303)
        self.assertEqual(view["reported_arm"]["official_score"], .720)
        self.assertEqual(view["reported_arm"]["costs"]["C_expected"], 4.73)

    def test_resolution_uses_numeric_confirmation_and_strict_boundaries(self):
        self.contrast()["holm_P"] = .01
        self.assertEqual(resolved_policy_view(self.reported, "Peptides-func", self.registered)["deployed_rule"], "radius")
        self.contrast()["holm_P"] = self.registered["confirmation_eq20_table_S11"]["holm_alpha"]
        self.assertEqual(resolved_policy_view(self.reported, "Peptides-func", self.registered)["deployed_rule"], "qC returned")
        self.contrast().update(holm_P=.01, effect=-.0015)
        self.assertEqual(resolved_policy_view(self.reported, "Peptides-func", self.registered)["deployed_rule"], "qC returned")

    def test_eligible_downgrade_needs_its_own_admission_and_measurements(self):
        one_step = next(row for row in self.reported["supplementary_table_S12_selection_ledger"]["Peptides-func"]
                        if row["route"] == "one-step")
        one_step["U_s"] = -.0022
        view = resolved_policy_view(self.reported, "Peptides-func", self.registered)
        self.assertTrue(view["decision_basis"]["predecessor_eligible"])
        self.assertEqual(view["confirmed_rule"], "one-step")
        self.assertIsNone(view["deployed_rule"])
        self.assertIsNone(view["loss_effect_nats"])
        self.assertIsNone(view["scores"]["qD"]["official_score"])
        self.assertIsNone(view["costs"]["C_expected"])

    def test_analytical_sensitivity_cannot_supply_confirmation(self):
        for row in self.reported["supplementary_table_S14b_separated_data_sensitivity"]:
            row.update(effect=-.5, holm_P=.0001)
        self.assertEqual(resolved_policy_view(self.reported, "Peptides-func", self.registered)["deployed_rule"], "qC returned")

    def test_evaluated_arm_targets_preserve_source_values_without_fabricated_artifacts(self):
        targets = evaluated_arm_binding_targets(self.reported, "Peptides-func", self.registered)
        self.assertEqual(len(targets), 7)
        for row in targets:
            value = self.reported
            for part in row["reference_pointer"].strip("/").split("/"):
                value = value[int(part)] if isinstance(value, list) else value[part]
            self.assertEqual(value, row["reported_value"])
            self.assertNotIn("artifact", row)
            self.assertNotIn("json_pointer", row)
        self.assertEqual(evaluated_arm_binding_targets(self.reported, "ogbn-arxiv", self.registered), [])


class ActionRegistryTests(unittest.TestCase):
    def setUp(self):
        self.registered = load_registered()
        self.code = {a["id"]: a for a in parse_actions_from_core()}
        self.table = {r["id"]: r for r in self.registered["action_registry_table_S6"]["actions"]}

    def test_ten_actions_with_the_registered_parameters(self):
        self.assertEqual(sorted(self.code), list(range(10)))
        for identifier, row in self.table.items():
            code = self.code[identifier]
            self.assertAlmostEqual(code["probability"], row["exploration_probability"], places=12, msg=f"A{identifier}")
            if row["mode"] == "direct":
                self.assertIn(code["mode"], ("source", "last_safe"), f"A{identifier}")
                continue
            self.assertEqual(code["mode"], "transport", f"A{identifier}")
            self.assertEqual((code["rho"], code["nu"], code["alpha"]), (row["rho"], row["nu"], row["alpha"]), f"A{identifier}")
            self.assertEqual(code["eta"], row["eta"], f"A{identifier}")
            if row["eta"] is not None:
                self.assertEqual(code["order"], row["T"], f"A{identifier}")
        self.assertAlmostEqual(sum(a["probability"] for a in self.code.values()), 1.0, places=12)

    def test_formulas_reproduce_table_S6b(self):
        derived = self.registered["action_registry_table_S6"]["derived_table_S6b"]
        for key, row in derived.items():
            code = self.code[int(key[1:])]
            self.assertAlmostEqual(executed_hops(code["rho"], code["nu"], code["eta"], code["order"]), row["executed_range_hops_per_step"], delta=0.0005, msg=key)
            self.assertAlmostEqual(executed_identity_coefficient(code["rho"], code["nu"], code["eta"], code["order"]), row["executed_identity_coefficient"], delta=0.0005, msg=key)
            mass = identity_mass(code["eta"], code["order"])
            if row["kernel_identity_mass"] is None:
                self.assertIsNone(mass, key)
            else:
                self.assertAlmostEqual(mass, row["kernel_identity_mass"], delta=0.00005, msg=key)

    def test_band_sensitivities_reproduce_figure_S15b(self):
        figure = self.registered["action_registry_table_S6"]["band_sensitivity_figure_S15b"]
        for key, printed in figure.items():
            if not key.startswith("A"):
                continue
            values = band_sensitivity(self.code[int(key[1:])], self.code[0])
            for b, (value, target) in enumerate(zip(values, printed)):
                self.assertAlmostEqual(value, target, delta=0.00501, msg=f"{key} band {b}")

    def test_local_actions_have_closed_form_sensitivity(self):
        # For one-step kernels g_a(lambda) = 1 - nu lambda, so s_ab = (nu_A0 - nu_a) * right band edge.
        for identifier in (1, 2):
            values = band_sensitivity(self.code[identifier], self.code[0])
            for b, value in enumerate(values):
                self.assertAlmostEqual(value, abs(self.code[0]["nu"] - self.code[identifier]["nu"]) * 2.0 * (b + 1) / 6.0, delta=1e-9)

    @unittest.skipIf(torch_core is None, "PyTorch is not installed; the torch-backed registry is checked in the full test suite")
    def test_pure_python_formulas_agree_with_core(self):
        import torch
        for action in torch_core.ACTIONS[:8]:
            self.assertAlmostEqual(torch_core.executed_hops(action), executed_hops(action.rho, action.nu, action.eta, action.order), places=12)
            self.assertAlmostEqual(torch_core.executed_identity_coefficient(action), executed_identity_coefficient(action.rho, action.nu, action.eta, action.order), places=12)
            self.assertAlmostEqual(torch_core.identity_mass(action), identity_mass(action.eta, action.order) or 0.0, places=12)
            grid = torch.linspace(0.0, 2.0, 41, dtype=torch.float64)
            response = torch_core.action_operator_response(action, grid)
            for lam, value in zip(grid.tolist(), response.tolist()):
                self.assertAlmostEqual(value, torch_core.action_operator_response(action, torch.tensor([lam], dtype=torch.float64)).item(), places=12)
                from scripts._paper_common import spectral_response
                self.assertAlmostEqual(value, spectral_response(action.rho, action.nu, action.eta, lam, action.order), places=9)


class FullProfileTests(unittest.TestCase):
    def test_registered_margins_depend_on_the_loss_type(self):
        for dataset in DATASETS:
            c = full_config(dataset)
            expected = 0.0015 if dataset in BCE_DATASETS else 0.003
            self.assertEqual(c["selection"]["epsilon_r"], expected, dataset)
            self.assertEqual(c["admission"]["epsilon_a"], expected, dataset)
            self.assertEqual(c["selection"]["epsilon_eq"], 0.002, dataset)
            self.assertEqual(c["admission"]["clip"], 0.10, dataset)
            self.assertEqual((c["selection"]["alpha"], c["admission"]["alpha"]), (0.025, 0.025), dataset)
            self.assertEqual(c["selection"]["bootstrap_reps"], 10000, dataset)

    def test_check_paper_constants_passes(self):
        exit_code = check_paper_constants.main(["--quiet"])
        self.assertEqual(exit_code, 0)


class BackboneReferenceTests(unittest.TestCase):
    def setUp(self):
        self.registered = load_registered()
        self.reported = load_reported()

    def check_references(self):
        report = check_paper_constants.Report()
        check_paper_constants.check_backbone_references(report, self.registered, self.reported)
        return report

    def test_adapter_dimensions_include_encoded_peptides_features(self):
        self.assertEqual(check_paper_constants.source_dimensions(), {
            "ogbn-arxiv": (128, 40), "ogbn-proteins": (8, 112), "ogbn-products": (100, 47),
            "Roman-empire": (300, 18), "PascalVOC-SP": (14, 21), "COCO-SP": (14, 81),
            "Peptides-func": (174, 10)})

    def test_released_counts_agree_with_table_s5_at_the_printed_precision(self):
        released = self.registered["backbones_source_status"]["released_full_profiles"]
        targets = self.registered["backbones_table_S5"]
        for dataset in DATASETS:
            with self.subTest(dataset=dataset):
                target = targets[dataset]["parameters_M"]
                tolerance = check_paper_constants.printed_half_unit(target) * 1_000_000
                self.assertLessEqual(abs(released[dataset]["parameter_count"] - target * 1_000_000), tolerance)
                self.assertEqual(sum(released[dataset]["parameter_components"].values()), released[dataset]["parameter_count"])
        self.assertEqual(self.check_references().failures, [])

    def test_released_count_outside_the_printed_precision_is_reported(self):
        released = self.registered["backbones_source_status"]["released_full_profiles"]
        released["ogbn-arxiv"]["parameter_count"] = 1_138_216  # the default-width plant of Table S5d
        released["ogbn-arxiv"]["parameter_components"] = {"encoder": 854016, "shared": 262656, "norm": 1024, "head": 20520}
        self.reported["supplementary_table_S5d_model_inventory"]["ogbn-arxiv"]["full_profile_parameters"] = 1_138_216
        failures = self.check_references().failures
        self.assertTrue(failures)
        self.assertTrue(all(failure["group"] == "S5 count" for failure in failures))
        self.assertTrue(any("agrees with Table S5" in failure["check"] for failure in failures))

    def test_each_s15_plant_reference_must_match_its_s5_target(self):
        original = self.reported["supplementary_table_S15_method_comparison"]
        for dataset in DATASETS:
            with self.subTest(dataset=dataset):
                self.reported = load_reported()
                self.reported["supplementary_table_S15_method_comparison"][dataset]["plant_params_M"] += 0.1
                failures = self.check_references().failures
                self.assertEqual(len(failures), 1)
                self.assertEqual(failures[0]["group"], "S5/S15")
                self.assertIn(STEMS[dataset], failures[0]["check"])
                self.assertEqual(failures[0]["expected"], original[dataset]["plant_params_M"])

    def test_repeated_method_plant_references_are_checked_but_total_params_are_separate(self):
        table = self.reported["supplementary_table_S15_method_comparison"]["ogbn-arxiv"]
        for row in table["rows"]:
            row["plant_params_M"] = table["plant_params_M"]
            row["params_M"] = 1000.0  # A method/ensemble total is not a one-plant count.
        self.assertEqual(self.check_references().failures, [])
        table["rows"][-1]["plant_params_M"] += 0.1
        failures = self.check_references().failures
        self.assertEqual(len(failures), 1)
        self.assertIn(f"rows[{len(table['rows']) - 1}].plant_params_M", failures[0]["check"])

    def test_s5d_keeps_released_and_benchmark_count_references_consistent(self):
        for field, wrong_value in (("released_default_parameters", 1_138_217), ("benchmark_parameters_M", 3.2),
                                   ("full_profile_parameters", 3_116_633)):
            with self.subTest(field=field):
                self.reported = load_reported()
                self.reported["supplementary_table_S5d_model_inventory"]["ogbn-arxiv"][field] = wrong_value
                failures = self.check_references().failures
                self.assertEqual(len(failures), 1)
                self.assertIn("S5d", failures[0]["group"])

    def test_released_references_require_every_dataset(self):
        released = self.registered["backbones_source_status"]["released_full_profiles"]
        del released["COCO-SP"]
        failures = self.check_references().failures
        self.assertTrue(any("dataset coverage" in row["check"] for row in failures))
        self.assertTrue(any("coco_sp: released reference present" == row["check"] for row in failures))

    def test_peptides_cannot_use_raw_or_wrong_encoded_width(self):
        row = self.registered["backbones_source_status"]["released_full_profiles"]["Peptides-func"]
        for width in (9, 173, 175):
            with self.subTest(width=width):
                row["input_dim"] = width
                failures = self.check_references().failures
                self.assertTrue(any(failure["check"] == "peptides_func: adapter input/output dimensions"
                                    for failure in failures))

    def test_invalid_count_or_untraceable_source_metadata_fails(self):
        for field, value in (("parameter_count", True), ("parameter_count", 0), ("buffer_elements", -1),
                             ("input_dim", 128.0), ("num_outputs", 41), ("dimension_source", "  "),
                             ("configuration", "configs/arxiv_smoke.json")):
            with self.subTest(field=field, value=value):
                self.registered = load_registered()
                self.registered["backbones_source_status"]["released_full_profiles"]["ogbn-arxiv"][field] = value
                self.assertTrue(self.check_references().failures)

    def test_constant_checks_work_without_site_packages(self):
        result = subprocess.run([sys.executable, "-S", "-B", str(ROOT / "scripts/check_paper_constants.py"), "--quiet"],
                                capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_same_count_cannot_hide_source_width_or_component_corruption(self):
        for field, value in (("architecture_widths", {"hidden_dim": 512}),
                             ("parameter_components", None), ("parameter_components", {"encoder": "3116632"})):
            with self.subTest(field=field, value=value):
                self.registered = load_registered()
                self.registered["backbones_source_status"]["released_full_profiles"]["ogbn-arxiv"][field] = value
                self.assertTrue(self.check_references().failures)

    def test_upper_half_unit_boundary_is_not_count_agreement(self):
        row = self.registered["backbones_source_status"]["released_full_profiles"]["ogbn-arxiv"]
        difference = 3_150_000 - row["parameter_count"]
        row["parameter_count"] += difference
        row["parameter_components"]["encoder"] += difference
        failures = self.check_references().failures
        self.assertTrue(any("agrees with Table S5" in item["check"] for item in failures))


class DatasetIdentityTests(unittest.TestCase):
    def setUp(self):
        self.registered, self.reported = load_registered(), load_reported()

    def check_identities(self):
        report = check_paper_constants.Report()
        check_paper_constants.check_dataset_identities(report, self.registered, self.reported)
        return report

    def test_published_statistics_reproduce_the_identities(self):
        self.assertEqual(self.check_identities().failures, [])

    def test_missing_dataset_and_independent_population_drift_fail(self):
        for dataset, field, value in (("ogbn-arxiv", "nodes", 169344),
                                     ("Peptides-func", "graphs", 15534), ("COCO-SP", "graphs", True)):
            with self.subTest(dataset=dataset, field=field):
                self.registered = load_registered()
                self.registered["dataset_statistics_table_S6c_S23b"][dataset][field] = value
                self.assertTrue(any("matches adapter" in row["check"] for row in self.check_identities().failures))
        self.registered["dataset_statistics_table_S6c_S23b"].pop("Roman-empire")
        self.assertTrue(any("coverage" in row["check"] for row in self.check_identities().failures))

    def test_invalid_edge_counts_fail_cleanly_and_sources_remain_explicit(self):
        for value in (0, -1, True, float("nan"), float("inf"), "1166243"):
            with self.subTest(edges=value):
                self.registered = load_registered()
                self.registered["dataset_statistics_table_S6c_S23b"]["ogbn-arxiv"]["reported_edges"] = value
                self.assertTrue(any("positive finite edges" in row["check"] for row in self.check_identities().failures))
        self.registered = load_registered()
        self.registered["dataset_statistics_table_S6c_S23b"]["ogbn-arxiv"].pop("edge_count_convention")
        self.assertTrue(any("edge convention" in row["check"] for row in self.check_identities().failures))

    def test_nominal_work_uses_three_significant_digits_not_half_percent_tolerance(self):
        self.reported["supplementary_table_S23b_nominal_execution_time_and_amortization"]["ogbn-arxiv"]["W_nom_MAC"] = 1.148e13
        self.assertTrue(any(row["group"] == "S23b" for row in self.check_identities().failures))

    def test_changed_edge_statistic_is_recomputed_against_both_tables(self):
        self.registered["dataset_statistics_table_S6c_S23b"]["ogbn-proteins"]["reported_edges"] *= 2
        groups = {row["group"] for row in self.check_identities().failures}
        self.assertIn("S6c", groups)
        self.assertIn("S23b", groups)


class ReportedArithmeticTests(unittest.TestCase):
    def setUp(self):
        self.reported = load_reported()
        self.registered = load_registered()

    def test_admission_bound_arithmetic_table_S13(self):
        for dataset, r in self.reported["supplementary_table_S13_independent_graph_admission"].items():
            penalty = bernstein_penalty(r["S2"], r["J"], r["B"], r["delta_a"])
            self.assertAlmostEqual(penalty, r["penalty"], delta=5.1e-6, msg=dataset)
            self.assertAlmostEqual(r["D_bar"] + penalty, r["U_a"], delta=5.1e-6, msg=dataset)
            self.assertEqual(int(r["U_a"] <= -r["epsilon_a"]), r["G_admit"], dataset)

    def test_corollary_2_minimum_sample_sizes(self):
        t1 = self.reported["main_table_1_policy_and_deployment_ledger"]
        for dataset, expected in (("ogbn-arxiv", 1279), ("ogbn-proteins", 853), ("Roman-empire", 602)):
            eps = 0.0015 if dataset in BCE_DATASETS else 0.003
            # The main text prints the integer part of the threshold: "J > 602 (Roman), 853 (proteins) and 1279 (arxiv)".
            self.assertEqual(math.floor(minimum_admission_J(t1[dataset]["admission_D_bar"], eps)), expected, dataset)

    def test_table_2_follows_from_tables_S23_and_S41(self):
        t1 = self.reported["main_table_1_policy_and_deployment_ledger"]
        t2 = self.reported["main_table_2_gain_and_cost_ledger"]
        s23 = self.reported["supplementary_table_S23_assurance_and_expected_compute"]
        s41 = self.reported["supplementary_table_S41_qualification_cost_components"]
        for dataset in DATASETS:
            if dataset == "ogbn-products":
                continue
            expected, comparator = s23[dataset]["expected_serial_cost"], s23[dataset]["comparator_cost"]
            self.assertAlmostEqual(expected, t2[dataset]["C_expected"], delta=0.005, msg=dataset)
            self.assertAlmostEqual(1000 * abs(t1[dataset]["loss_effect_nats"]) / (expected - comparator), t2[dataset]["lambda_star_mnats"], delta=0.01, msg=dataset)
            amortization = 3600 * s41[dataset]["total_qualification_increment_GPU_h"] / (0.1 * (expected - comparator)) / 1000
            self.assertAlmostEqual(amortization, t2[dataset]["M_star_t_nom_thousand_input_seconds"], delta=0.5, msg=dataset)

    def test_table_1_gains_match_or_preserve_documented_discrepancy(self):
        t1 = self.reported["main_table_1_policy_and_deployment_ledger"]
        s15 = self.reported["supplementary_table_S15_method_comparison"]
        for dataset in DATASETS:
            deployed = next(r for r in s15[dataset]["rows"] if r["family"] == "CROSS-MORSE deployment")
            comparator = next(r for r in s15[dataset]["rows"] if "(qC" in r["method"])
            scale = 100.0 if s15[dataset]["score_unit"].startswith("AP") else 1.0
            gain = (deployed["official_score"] - comparator["official_score"]) * scale
            if dataset == "Peptides-func":
                # S26 prints four-decimal scores and explicitly rejects rounding
                # alone as an explanation for this 0.0002 native-AP difference.
                note = self.reported["_interpretation_notes"]["peptides_score_discrepancy"]
                self.assertAlmostEqual((gain - t1[dataset]["official_gain_pp"]) / 100,
                                       note["difference_minus_main_paired_gain_AP"], places=12)
                self.assertGreater(note["difference_minus_main_paired_gain_AP"], 0.0001)
            else:
                self.assertAlmostEqual(gain, t1[dataset]["official_gain_pp"], delta=0.011, msg=dataset)

    def test_printed_half_unit(self):
        self.assertEqual(printed_half_unit(-0.0045), 0.00005)
        self.assertEqual(printed_half_unit(73.22), 0.005)
        self.assertEqual(printed_half_unit(112), 0.5)
        self.assertEqual(printed_half_unit(0.0), 0.0)


class RegistryFigureTests(unittest.TestCase):
    def test_registry_figures_render(self):
        try:
            import matplotlib  # noqa: F401
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib/numpy are not installed")
        import tempfile
        from scripts import plot_registry_figures
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(plot_registry_figures.main(["--output-dir", tmp]), 0)
            names = sorted(p.name for p in Path(tmp).glob("registry_*.png"))
            self.assertEqual(names, ["registry_admission_arithmetic.png", "registry_deployment_economics.png", "registry_gain_and_cost.png",
                                     "registry_mechanism_contrasts.png", "registry_selection_geometry.png", "registry_spectral_fingerprint.png"])


    def test_confirmation_caption_tracks_numeric_values_and_resolved_rule(self):
        reported, registered = copy.deepcopy(load_reported()), copy.deepcopy(load_registered())
        contrast = next(row for row in reported["supplementary_table_S14_mechanism_contrasts_submitted_protocol"]
                        if row["contrast"] == "Peptides/radius")
        contrast["holm_P"] = 0.0125
        registered["confirmation_eq20_table_S11"]["holm_alpha"] = 0.02
        caption = peptides_confirmation_caption(reported, registered)
        self.assertIn("Holm P = 0.0125", caption)
        self.assertIn("α = 0.02", caption)
        self.assertIn("evaluated radius arm", caption)
        self.assertIn("rule: retain radius", caption)
        # Passing Holm alone does not satisfy equality at the practical margin.
        contrast["effect"] = -.0015
        caption = peptides_confirmation_caption(reported, registered)
        self.assertIn("evaluated radius arm", caption)
        self.assertIn("rule: return qC", caption)

    def test_summary_svg_uses_registry_caption_values(self):
        try:
            import matplotlib  # noqa: F401
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib/numpy are not installed")
        from scripts import plot_paper_summary
        reported, registered = copy.deepcopy(load_reported()), copy.deepcopy(load_registered())
        contrast = next(row for row in reported["supplementary_table_S14_mechanism_contrasts_submitted_protocol"]
                        if row["contrast"] == "Peptides/radius")
        contrast["holm_P"] = 0.0125
        registered["confirmation_eq20_table_S11"]["holm_alpha"] = 0.02
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(plot_paper_summary, "load_reported", return_value=reported), \
             patch.object(plot_paper_summary, "load_registered", return_value=registered):
            plot_paper_summary.render(Path(tmp))
            svg = (Path(tmp) / "reported_results.svg").read_text()
        self.assertIn("Holm P = 0.0125", svg)
        self.assertIn("α = 0.02", svg)
        self.assertIn("evaluated radius arm", svg)
        self.assertIn("rule: retain radius", svg)
        self.assertNotIn("Holm P = 0.070", svg)

    def test_mechanism_title_tracks_scope_and_preserves_analytical_label(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.figure import Figure
        except ImportError:
            self.skipTest("matplotlib is not installed")
        from scripts import plot_registry_figures
        reported, registered = copy.deepcopy(load_reported()), copy.deepcopy(load_registered())
        contrast = next(row for row in reported["supplementary_table_S14_mechanism_contrasts_submitted_protocol"]
                        if row["contrast"] == "Peptides/radius")
        contrast["holm_P"] = 0.018
        registered["confirmation_eq20_table_S11"]["holm_alpha"] = 0.025
        reported["main_table_1_policy_and_deployment_ledger"]["Peptides-func"].update(
            registered_confirmation_pass=True, evaluation_status="confirmed_arm", lifecycle_resolution="retain_radius")
        captured = {}
        def capture(figure, *args, **kwargs):
            captured["title"] = figure.axes[0].get_title()
            captured["label"] = figure.axes[0].get_xlabel()
        with patch.object(Figure, "savefig", new=capture):
            plot_registry_figures.mechanism_contrasts(plt, registered, reported, Path("unused"))
        self.assertIn("Holm P = 0.018", captured["title"])
        self.assertIn("α = 0.025", captured["title"])
        self.assertIn("evaluated radius arm", captured["title"])
        self.assertIn("rule: retain radius", captured["title"])
        self.assertIn("analytical sensitivity", captured["label"])


class SourceIdentityTests(unittest.TestCase):
    def test_runtime_fingerprint_matches_the_validation_record(self):
        record = json.loads((ROOT / "docs" / "VALIDATION.json").read_text(encoding="utf-8"))
        current = runtime_source_fingerprint(ROOT)
        self.assertEqual(record["source_sha256"], current)
        for run in record["smoke_runs"]:
            with self.subTest(dataset=run["dataset"]):
                recorded = run["identities"]["source_sha256"]
                if recorded != current:
                    # A smoke run executed on an earlier runtime must say so.
                    self.assertEqual(run["status"], "PASS_ON_PREVIOUS_SOURCE")
                    self.assertIn(recorded[:12], run["note"])
                else:
                    self.assertEqual(run["status"], "PASS")


class PaperSourceProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.reported = load_reported()
        self.index = json.loads((PAPER / "sources.json").read_text(encoding="utf-8"))

    def test_recorded_manuscript_identities_cover_every_table_group(self):
        result = render_paper_results.validate_sources(ROOT, self.reported, self.index)
        # The identities are those recorded in sources.json (scripts/register_manuscript_sources.py --register
        # rewrites them for the final PDFs); a registration record, when present, must name the same bytes.
        self.assertEqual(result["pdf_sha256"], {role: entry["sha256"] for role, entry in self.index["documents"].items()})
        for digest in result["pdf_sha256"].values():
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
        record = PAPER / "transcription_check.json"
        if record.is_file():
            self.assertEqual(json.loads(record.read_text(encoding="utf-8"))["pdf_sha256"], result["pdf_sha256"])
        # References retain source identities without distributing manuscript PDFs.
        self.assertEqual(result["pdf_bytes_verified"], {"main": False, "supplementary": False})
        self.assertEqual(result["verification_mode"], "reference_metadata_only")
        for entry in self.index["documents"].values():
            self.assertIsNone(entry["bundled_path"])
            self.assertFalse((ROOT / "paper" / "manuscript.pdf").exists())
        self.assertEqual(result["table_groups_checked"], len([k for k in self.reported if not k.startswith("_")]))

    def test_a_local_copy_is_verified_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = copy.deepcopy(self.index)
            copies = {}
            for name, entry in index["documents"].items():
                payload = b"%PDF-1.7\nlocal copy fixture\n" + name.encode()
                copies[name] = root / f"{name}.pdf"
                copies[name].write_bytes(payload)
                entry["byte_size"] = len(payload)
                entry["sha256"] = hashlib.sha256(payload).hexdigest()
            result = render_paper_results.validate_sources(ROOT, self.reported, index, copies)
            self.assertEqual(result["pdf_bytes_verified"], {"main": True, "supplementary": True})
            copies["main"].write_bytes(b"%PDF-1.7\nanother version\n")
            with self.assertRaisesRegex(ValueError, "content identity changed"):
                render_paper_results.validate_sources(ROOT, self.reported, index, copies)

    def temporary_sources(self, root):
        """Small byte fixtures exercise validation without copying research PDFs."""
        index = copy.deepcopy(self.index)
        (root / "paper").mkdir()
        for name, entry in index["documents"].items():
            payload = b"%PDF-1.7\nprovenance byte fixture\n" + name.encode()
            entry["bundled_path"] = f"paper/{name}.pdf"
            path = root / entry["bundled_path"]
            path.write_bytes(payload)
            entry["byte_size"] = len(payload)
            entry["sha256"] = hashlib.sha256(payload).hexdigest()
        return index

    def test_changed_pdf_is_rejected_even_when_manifest_shape_is_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = self.temporary_sources(root)
            render_paper_results.validate_sources(root, self.reported, index)
            (root / index["documents"]["main"]["bundled_path"]).write_bytes(b"%PDF-1.7\nchanged content\n")
            with self.assertRaisesRegex(ValueError, "content identity changed"):
                render_paper_results.validate_sources(root, self.reported, index)

    def test_declared_bundled_pdf_must_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = self.temporary_sources(root)
            (root / index["documents"]["main"]["bundled_path"]).unlink()
            with self.assertRaisesRegex(ValueError, "Missing PDF"):
                render_paper_results.validate_sources(root, self.reported, index)

    def test_table_provenance_cannot_silently_omit_a_group(self):
        index = copy.deepcopy(self.index)
        del index["table_groups"]["main_table_1_policy_and_deployment_ledger"]
        with self.assertRaisesRegex(ValueError, "coverage differs"):
            render_paper_results.validate_sources(ROOT, self.reported, index)

    def test_page_reference_must_be_inside_the_named_source(self):
        index = copy.deepcopy(self.index)
        index["table_groups"]["main_table_1_policy_and_deployment_ledger"]["pages"] = [21]
        with self.assertRaisesRegex(ValueError, "Invalid source page"):
            render_paper_results.validate_sources(ROOT, self.reported, index)

    def test_pdf_path_cannot_escape_repository(self):
        index = copy.deepcopy(self.index)
        index["documents"]["main"]["bundled_path"] = "../manuscript.pdf"
        with self.assertRaisesRegex(ValueError, "Invalid relative PDF path"):
            render_paper_results.validate_sources(ROOT, self.reported, index)

    def test_numeric_registry_rejects_nonfinite_values(self):
        reported = copy.deepcopy(self.reported)
        reported["main_table_1_policy_and_deployment_ledger"]["ogbn-arxiv"]["loss_effect_nats"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            render_paper_results.validate_sources(ROOT, reported, self.index)

    def test_measurements_are_distinguished_from_sensitivity_models(self):
        groups = self.index["table_groups"]
        self.assertEqual(groups["supplementary_table_S14_mechanism_contrasts_submitted_protocol"]["evidence_kind"],
                         "author_reported_measurements")
        for key in ("supplementary_table_S14b_separated_data_sensitivity", "supplementary_table_S42_power_of_budget_matched_test",
                    "supplementary_table_S44_per_input_cost_model"):
            self.assertEqual(groups[key]["evidence_kind"], "analytical_sensitivity")
        self.assertEqual(groups["supplementary_table_S46_budget_matched_frontier"]["evidence_kind"],
                         "author_reported_measurements")

    def test_missing_absolute_scores_and_reported_support_extrema_remain_distinct(self):
        availability = self.reported["_availability"]
        for key in ("q0_absolute_test_loss", "q0_absolute_official_test_score", "qN_qC_qD_absolute_test_losses"):
            self.assertEqual(availability[key]["status"], "not_tabulated")
            self.assertIsNone(availability[key]["value"])
        self.assertEqual(availability["per_cell_support_ESS_and_max_weight"]["status"], "author_reported_summary")
        for row in self.reported["supplementary_table_S7_support_audit"].values():
            self.assertGreater(row["per_cell_minimum_ESS"], 0)
            self.assertGreater(row["per_cell_maximum_normalized_weight"], 0)
            self.assertIn("failed cells", row["per_cell_weight_summary_status"])
        peptides = next(row for row in self.reported["supplementary_table_S14_mechanism_contrasts_submitted_protocol"] if row["contrast"] == "Peptides/radius")
        self.assertEqual(peptides["holm_P"], 0.070)
        alpha = load_registered()["confirmation_eq20_table_S11"]["holm_alpha"]
        self.assertGreater(peptides["holm_P"], alpha)
        effect = self.reported["main_table_1_policy_and_deployment_ledger"]["Peptides-func"]
        self.assertFalse(effect["registered_confirmation_pass"])
        self.assertEqual(effect["evaluation_status"], "evaluated_arm")
        self.assertEqual(effect["lifecycle_resolution"], "return_qC")

    def test_csv_preserves_missing_values_and_exact_comparator_return(self):
        rows = list(csv.DictReader(io.StringIO(render_paper_results.render_csv(self.reported, self.index))))
        self.assertEqual([r["dataset"] for r in rows], list(DATASETS))
        by_dataset = {r["dataset"]: r for r in rows}
        self.assertTrue(all(r["q0_score"] == "" for r in rows))
        products = by_dataset["ogbn-products"]
        self.assertEqual(float(products["paired_qD_minus_qC_loss_nats"]), 0.0)
        self.assertEqual(products["loss_CI95_lower"], "")
        self.assertEqual(products["loss_CI95_upper"], "")
        peptides = by_dataset["Peptides-func"]
        self.assertEqual(float(peptides["metric_gain_pp"]), 0)
        self.assertEqual(float(peptides["paired_qD_minus_qC_loss_nats"]), 0)
        self.assertEqual(float(peptides["qD_score"]), .714)
        self.assertEqual(peptides["qD_origin"], "derived_comparator_identity")
        self.assertEqual(peptides["loss_CI95_lower"], "")
        self.assertEqual(peptides["modelled_C099"], "")
        self.assertAlmostEqual(float(peptides["evaluated_arm_metric_gain_pp"]), 100 * self.reported["main_table_1_policy_and_deployment_ledger"]["Peptides-func"]["official_gain_native"])
        self.assertEqual(float(peptides["evaluated_arm_score"]), .720)
        self.assertEqual(float(peptides["evaluated_arm_expected_cost"]), 4.73)
        self.assertEqual(float(peptides["qN_score"]), 0.704)

    def test_overview_and_csv_render_from_the_canonical_values(self):
        reported = copy.deepcopy(self.reported)
        reported["main_table_1_policy_and_deployment_ledger"]["ogbn-arxiv"]["loss_effect_nats"] = -0.00451
        overview = render_paper_results.render_overview(reported, self.index)
        self.assertIn("−0.00451", overview)
        rows = list(csv.DictReader(io.StringIO(render_paper_results.render_csv(reported, self.index))))
        self.assertEqual(float(rows[0]["paired_qD_minus_qC_loss_nats"]), -0.00451)

    def test_metadata_validation_does_not_require_or_create_generated_summaries(self):
        outputs = (ROOT / "paper/REGISTRY_SUMMARY.md", ROOT / "results/paper_summary.csv")
        self.assertFalse(any(path.exists() for path in outputs))
        result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/render_paper_results.py"), "--validate-only"],
                                capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("reference metadata only; no PDF bytes verified", result.stdout)
        self.assertIn("no reports written", result.stdout)
        self.assertFalse(any(path.exists() for path in outputs))


if __name__ == "__main__":
    unittest.main()
