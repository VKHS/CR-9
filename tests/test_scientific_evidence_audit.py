"""Independent numerical counterexamples and real tiny-lifecycle audit checks."""
from __future__ import annotations

import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from cross_morse.split_contract import DATA_ROLE_PROTOCOL
from scripts.scientific_evidence_audit import (ROOT, audit_run, binding_coverage, build_report, check_campaign_binding,
                                              main, reference_selector, support_row, validate_record_coverage,
                                              validate_split_evidence)


class SupportEvidenceTests(unittest.TestCase):
    def test_impossible_reported_ess_and_maximum_weight_are_flagged_per_cell(self):
        # A maximum normalized weight of .2 alone contributes .04 to sum w²,
        # so ESS cannot exceed 25, irrespective of how many other units exist.
        row = {"ess": 64., "max_normalized_weight": .2, "trajectory_count": 128,
               "sum_normalized_weights": 1., "sum_squared_normalized_weights": 1 / 64}
        result = support_row(row)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("ess_exceeds_inverse_max_weight_squared", result["flags"])
        self.assertIn("sum_squared_weights_below_max_weight_squared", result["flags"])
        self.assertAlmostEqual(result["ess_upper_bound_from_max_weight"], 25.)

    def test_heterogeneous_exact_weights_pass_and_false_qualification_fails(self):
        # The three weights .5,.25,.25 give ESS=8/3, not the count of three.
        row = {"ess": 8 / 3, "max_normalized_weight": .5, "trajectory_count": 3,
               "sum_normalized_weights": 1., "sum_squared_normalized_weights": 3 / 8,
               "min_propensity": .1, "qualified": False}
        thresholds = {"p_min": .05, "n_min": 3, "max_normalized_weight": .6}
        self.assertEqual(support_row(row, thresholds)["status"], "PASS")
        wrong = support_row({**row, "qualified": True}, thresholds)
        self.assertIn("qualification_differs_from_thresholds", wrong["flags"])

    def test_plausible_legacy_summary_cannot_establish_the_weight_identity(self):
        result = support_row({"ess": 4., "max_normalized_weight": .25, "trajectory_count": 4})
        self.assertEqual(result["status"], "PENDING")
        self.assertIn("sum_squared_normalized_weights", result["missing"])

    def test_empty_support_cell_is_explicitly_unqualified(self):
        row = {"ess": 0., "max_normalized_weight": 1., "trajectory_count": 0,
               "sum_normalized_weights": 0., "sum_squared_normalized_weights": 0., "qualified": False}
        self.assertEqual(support_row(row)["status"], "PASS")
        self.assertEqual(support_row({**row, "qualified": True})["status"], "FAIL")


class SplitEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.metadata = {"schema_version": 3, "data_role_protocol": DATA_ROLE_PROTOCOL,
                         "comparison_holdout_reserved_at": "prepare", "num_samples": 140}
        self.official = {"train": np.arange(100), "valid": np.arange(100, 120), "test": np.arange(120, 140)}
        self.roles = {"fit_train": np.arange(50), "fit_val": np.arange(50, 60), "fit": np.arange(60),
                      "explore": np.arange(60, 80), "select": np.arange(80, 90), "confirm": np.arange(90, 95),
                      "compare": np.arange(95, 100), "admit": np.arange(100, 120), "test": np.arange(120, 140)}

    def test_exact_coverage_and_preparation_protocol_pass(self):
        result = validate_split_evidence(self.metadata, self.roles, self.official)
        self.assertEqual(result["sizes"], {"fit": 60, "explore": 20, "select": 10, "confirm": 5, "compare": 5})

    def test_old_schema_cannot_be_relabelled_as_preallocated_comparison(self):
        with self.assertRaisesRegex(ValueError, "never repartition"):
            validate_split_evidence({**self.metadata, "schema_version": 2}, self.roles, self.official)

    def test_disjoint_but_reassigned_validation_is_rejected(self):
        changed = copy.deepcopy(self.roles)
        changed["compare"][0], changed["admit"][0] = changed["admit"][0], changed["compare"][0]
        with self.assertRaisesRegex(ValueError, "official train"):
            validate_split_evidence(self.metadata, changed, self.official)

    def test_confirmation_comparison_overlap_is_rejected(self):
        changed = copy.deepcopy(self.roles)
        changed["compare"][0] = changed["confirm"][0]
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_split_evidence(self.metadata, changed, self.official)


class PublicationEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = Path(self.temporary.name)
        for name in ("paper/reported_results.json", "paper/registered_constants.json", "paper/sources.json", "records/specs.json"):
            target = self.fixture / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)

    def tearDown(self):
        self.temporary.cleanup()

    def auxiliary_rows(self, name):
        from cm_campaigns.contract import RECORD_SETS
        specs = json.loads((self.fixture / "records/specs.json").read_text())
        spec, contract = specs["sets"][name], RECORD_SETS[name]
        return [{**dict(zip(contract["coverage_keys"], key)), "quantity": quantity,
                 "value": "0.1", "unit": "unit-test fixture", "source_file": "raw/fixture.json",
                 "locator": "json:/value"}
                for key in contract["coverage_rows"] for quantity in spec["quantities"]]

    def write_auxiliary_fixture(self, name, rows=None, *, synthetic=False):
        """Ephemeral validator fixture; never written into the repository's records."""
        from scripts.records import write_manifest
        path = self.fixture / "records" / name
        if path.exists():
            shutil.rmtree(path)
        (path / "raw").mkdir(parents=True)
        (path / "raw/fixture.json").write_text(json.dumps({"value": .1}))
        specs = json.loads((self.fixture / "records/specs.json").read_text())
        spec = specs["sets"][name]
        with (path / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=spec["keys"] + specs["summary_columns"])
            writer.writeheader()
            writer.writerows(self.auxiliary_rows(name) if rows is None else rows)
        (path / "provenance.json").write_text(json.dumps({
            **{key: "unit-test fixture only, not experiment evidence" for key in specs["provenance_fields"]},
            "synthetic": synthetic}))
        write_manifest(path)
        return path

    def test_control_and_backbone_coverage_uses_their_actual_row_keys(self):
        for name in ("negative_controls", "cross_backbone"):
            self.write_auxiliary_fixture(name)
        result = build_report(self.fixture)
        self.assertEqual(result["auxiliary_records"]["validated"], 2)
        coverage = result["auxiliary_records"]["sets"]
        self.assertEqual(coverage["negative_controls"]["coverage"],
                         {"keys": ["control"], "rows": 4, "quantities_per_row": 3})
        self.assertEqual(coverage["cross_backbone"]["coverage"],
                         {"keys": ["regime", "model"], "rows": 6, "quantities_per_row": 3})
        self.assertFalse(result["publication_ready"])
        self.assertEqual(result["official_archives"]["validated"], 0)

    def test_custom_record_root_uses_the_checkout_contract(self):
        source = self.write_auxiliary_fixture("negative_controls")
        external = self.fixture / "external-records"
        external.mkdir()
        shutil.move(str(source), str(external / source.name))
        # Evidence may live elsewhere, but it cannot supply a weaker contract.
        (external / "specs.json").write_text(json.dumps({"sets": {}}))
        result = build_report(self.fixture, records_root=external)
        self.assertEqual(result["auxiliary_records"]["validated"], 1)
        self.assertIn("negative_controls", result["auxiliary_records"]["sets"])
        self.assertEqual(result["inputs"]["records_root"], str(external))
        self.assertEqual(build_report(self.fixture)["auxiliary_records"]["validated"], 0)

    def test_custom_run_index_resolves_archives_relative_to_its_parent(self):
        external = self.fixture / "external-runs"
        (external / "arxiv").mkdir(parents=True)
        index = external / "index.json"
        index.write_text(json.dumps({"schema": "cross-morse-runs-index-1", "runs": [
            {"dataset": "ogbn-arxiv", "path": "arxiv", "kind": "run", "verified": True,
             "archive_sha256": "a" * 64}]}))
        with patch("scripts.scientific_evidence_audit.audit_run", return_value={"status": "PASS"}) as audit:
            result = build_report(self.fixture, index_path=index)
        audit.assert_called_once_with(external / "arxiv", official_dataset="ogbn-arxiv", manuscript_root=self.fixture)
        self.assertEqual(result["inputs"]["run_index"], str(index))
        # Missing manuscript bindings still prevent publication readiness.
        self.assertFalse(result["publication_ready"])
        self.assertTrue(any(item["code"] == "manuscript_numbers_unlinked" for item in result["blockers"]))

    def test_missing_control_or_substitution_and_missing_quantity_are_blocking(self):
        from cm_campaigns.contract import RECORD_SETS
        for name in ("negative_controls", "cross_backbone"):
            for corruption in ("row", "quantity", "unexpected"):
                with self.subTest(name=name, corruption=corruption):
                    rows = self.auxiliary_rows(name)
                    keys = RECORD_SETS[name]["coverage_keys"]
                    if corruption == "row":
                        first = tuple(rows[0][k] for k in keys)
                        rows = [r for r in rows if tuple(r[k] for k in keys) != first]
                    elif corruption == "quantity":
                        rows = rows[1:]
                    else:
                        rows[0][keys[0]] = "unregistered construction or regime"
                    self.write_auxiliary_fixture(name, rows)
                    result = build_report(self.fixture)
                    failures = [r for r in result["blockers"] if r["code"] == "invalid_auxiliary_set"]
                    self.assertTrue(any(r["location"] == name for r in failures))
                    self.assertNotIn(name, result["auxiliary_records"]["sets"])

    def test_named_coverage_does_not_relax_raw_integrity_or_synthetic_refusal(self):
        path = self.write_auxiliary_fixture("negative_controls")
        (path / "raw/fixture.json").write_text(json.dumps({"value": .2}))
        self.write_auxiliary_fixture("cross_backbone", synthetic=True)
        result = build_report(self.fixture)
        self.assertEqual(result["auxiliary_records"]["validated"], 0)
        failures = {r["location"]: r["detail"] for r in result["blockers"] if r["code"] == "invalid_auxiliary_set"}
        self.assertIn("changed after indexing", failures["negative_controls"])
        self.assertIn("Synthetic records", failures["cross_backbone"])

    def test_dataset_record_sets_still_require_exact_dataset_coverage(self):
        from cm_campaigns.contract import RECORD_SETS
        spec = json.loads((self.fixture / "records/specs.json").read_text())["sets"]["input_gate"]
        with self.assertRaisesRegex(ValueError, "dataset coverage"):
            validate_record_coverage("input_gate", [{"dataset": "ogbn-arxiv"}], spec, RECORD_SETS["input_gate"])

    def test_representative_rows_read_radius_on_three_datasets(self):
        result = reference_selector(ROOT)
        self.assertEqual(set(result["selection_trace_required_datasets"]), {"Roman-empire", "PascalVOC-SP", "COCO-SP"})
        self.assertFalse(result["measured_rerun"])
        for row in result["rows"]:
            if not row["representative_rows_match_reported_choice"]:
                self.assertEqual(row["registered_choice"], "radius")
                self.assertIsNotNone(row["note"])
                self.assertEqual(row["complete_ledger"], "selector trace of the run archive (E15)")

    def test_repository_with_source_and_reference_tables_is_not_publication_ready(self):
        # A valid reference registry plus record specs is not measured evidence.
        result = build_report(self.fixture)
        self.assertFalse(result["publication_ready"])
        self.assertEqual(result["official_archives"]["validated"], 0)
        self.assertEqual(result["auxiliary_records"]["validated"], 0)
        self.assertEqual(sum(r["code"] == "missing_official_archive" for r in result["blockers"]), 7)
        self.assertEqual(sum(r["code"] == "missing_auxiliary_set" for r in result["blockers"]), 10)
        self.assertEqual(result["manuscript_documents"]["verification_mode"], "reference_metadata_only")
        self.assertFalse(any(result["manuscript_documents"]["pdf_bytes_verified"].values()))
        self.assertFalse(any(r["code"] == "manuscript_document_identity" for r in result["blockers"]))
        self.assertEqual(result["confirmation_family"]["status"], "PENDING")
        self.assertIsNone(result["inputs"]["confirmation_campaign"])
        self.assertTrue(any(r["code"] == "confirmation_campaign_missing" for r in result["blockers"]))

    def test_standalone_confirmation_pass_cannot_replace_official_run_linkage(self):
        campaign = self.fixture / "confirmation"
        with patch("cm_campaigns.confirmation.check_campaign", return_value={"status": "PASS", "family_size": 9}):
            result = build_report(self.fixture, confirmation_campaign=campaign)
        self.assertEqual(result["inputs"]["confirmation_campaign"], str(campaign))
        self.assertEqual(result["confirmation_family"]["status"], "PENDING")
        self.assertEqual(len(result["confirmation_family"]["official_run_linkage"]["missing_datasets"]), 7)
        self.assertTrue(any(r["code"] == "confirmation_official_run_linkage" for r in result["blockers"]))
        self.assertFalse(result["publication_ready"])

    def test_synthetic_confirmation_is_a_scientific_failure(self):
        from cm_campaigns.confirmation import template
        campaign = self.fixture / "confirmation"
        campaign.mkdir()
        manifest = template(self.fixture)
        manifest.update(status="author_completed", synthetic=True, evidence_kind="original_confirmation_tests")
        (campaign / "campaign.json").write_text(json.dumps(manifest))
        result = build_report(self.fixture, confirmation_campaign=campaign)
        self.assertEqual(result["confirmation_family"]["status"], "FAIL")
        self.assertIn("Synthetic", result["confirmation_family"]["error"])

    def test_cli_writes_machine_readable_failure_and_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            out = Path(tmp) / "report.json"
            code = main(["--root", str(self.fixture), "--output", str(out)])
            data = json.loads(out.read_text())
        self.assertEqual(code, 1)
        self.assertFalse(data["publication_ready"])

    def test_legacy_budget_campaign_schema_is_not_accepted_from_a_pass_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = Path(tmp)
            (campaign / "study.json").write_text(json.dumps({"schema": "cm-budget-matched-2", "passed": True}))
            result = build_report(self.fixture, budget_campaign=campaign)
        self.assertEqual(result["budget_invariants"]["status"], "FAIL")
        self.assertIn("Unrecognized campaign schema", result["budget_invariants"]["error"])

    def test_single_split_number_cannot_fill_an_aggregate_binding(self):
        binding = json.loads((ROOT / "paper/bindings/roman_empire_full.json").read_text())
        self.assertEqual(binding_coverage(binding), [])
        self.assertEqual(len(binding_coverage(binding, campaign=True)), 9)


class EvaluatedArmBindingCoverageTests(unittest.TestCase):
    def targets(self):
        from scripts._paper_common import evaluated_arm_binding_targets
        return evaluated_arm_binding_targets(json.loads((ROOT / "paper/reported_results.json").read_text()),
                                             "Peptides-func", json.loads((ROOT / "paper/registered_constants.json").read_text()))

    def binding(self, *, include_evaluated=False):
        from scripts.scientific_evidence_audit import BINDING_POINTERS
        from scripts._paper_common import resolved_policy_view
        view = resolved_policy_view(json.loads((ROOT / "paper/reported_results.json").read_text()), "Peptides-func")
        result = {"run_identities": {"dataset": "Peptides-func"}, "results": [
            {"artifact": "final/results_summary.json", "json_pointer": pointer,
             "reported_value": 0., "absolute_tolerance": 0., "target_role": "served_policy",
             "target_kind": "derived_comparator_identity"}
            for pointer in sorted(BINDING_POINTERS)]}
        for row in result["results"]:
            if row["json_pointer"].startswith("/models/"):
                model = row["json_pointer"].split("/")[2]
                row.update(reported_value=100 * view["scores"][model]["official_score"], absolute_tolerance=.05,
                           target_kind="derived_comparator_identity" if model == "qD" else "author_reported_score")
        if include_evaluated:
            # These are coverage declarations only. check_binding independently
            # requires the named artifact's actual sealed bytes and values.
            result["results"] += [{**target, "artifact": "evaluated_arms/radius/summary.json",
                                  "json_pointer": f"/values/{i}"} for i, target in enumerate(self.targets())]
        return result

    def test_comparator_only_final_results_do_not_cover_the_reported_radius_arm(self):
        binding = self.binding()
        binding["unavailable_evaluated_arm_targets"] = self.targets()
        missing = binding_coverage(binding)
        self.assertEqual(len(missing), 7)
        self.assertTrue(all(row.startswith("evaluated_arm:") for row in missing))

    def test_all_separate_arm_targets_complete_the_coverage_declaration(self):
        self.assertEqual(binding_coverage(self.binding(include_evaluated=True)), [])
        targets = self.targets()
        native_ap = next(t for t in targets if t["reference_pointer"].endswith("official_score"))
        gain = next(t for t in targets if t["reference_pointer"].endswith("official_gain_pp"))
        self.assertEqual(native_ap["reported_value"], .720)
        self.assertEqual(gain["reported_value"], .58)

    def test_final_comparator_fields_cannot_be_relabelled_as_the_evaluated_arm(self):
        binding = self.binding(include_evaluated=True)
        for row in binding["results"]:
            if row.get("target_role") == "evaluated_arm":
                row.update(artifact="final/results_summary.json", json_pointer="/paired_qD_minus_qC_loss/mean")
        self.assertEqual(len(binding_coverage(binding)), 7)

    def test_final_targets_must_follow_the_frozen_comparator_resolution(self):
        for pointer, value, tolerance in (("/paired_qD_minus_qC_loss/mean", -.00303, 0.),
                ("/paired_qD_minus_qC_loss/mean", 1e-13, 0.),
                ("/paired_qD_minus_qC_metric_pp/upper", 1.07, 0.),
                ("/models/qD/metric_percent", 72., .05),
                ("/paired_qD_minus_qC_loss/mean", 0., .01)):
            with self.subTest(pointer=pointer, value=value, tolerance=tolerance):
                binding = self.binding(include_evaluated=True)
                row = next(r for r in binding["results"] if r["json_pointer"] == pointer)
                row.update(reported_value=value, absolute_tolerance=tolerance)
                self.assertIn("served_policy:" + pointer, binding_coverage(binding))

    def test_missing_duplicate_invented_or_loose_arm_declarations_fail(self):
        for corruption in ("missing", "duplicate", "value", "tolerance", "role", "kind"):
            with self.subTest(corruption=corruption):
                binding = self.binding(include_evaluated=True)
                row = binding["results"][-1]
                if corruption == "missing":
                    binding["results"].pop()
                elif corruption == "duplicate":
                    binding["results"].append(copy.deepcopy(row))
                elif corruption == "value":
                    row["reported_value"] = 0.
                elif corruption == "tolerance":
                    row["absolute_tolerance"] = 1.
                elif corruption == "role":
                    row["target_role"] = "served_policy"
                else:
                    row["target_kind"] = "derived_comparator_identity"
                self.assertTrue(binding_coverage(binding))


class CampaignBindingTests(unittest.TestCase):
    def setUp(self):
        self.paths = [Path(f"split_{i:02d}") for i in range(10)]
        self.identities = [{"split_index": i, "source_sha256": "a" * 64, "config_sha256": str(i) * 64,
                            "unit_sha256": str(i) * 64, "dataset": "Roman-empire"} for i in range(10)]
        self.inspected = [{"numeric_replay": {"paired_qD_minus_qC_loss": {"mean": i},
                            "paired_qD_minus_qC_metric_pp": {"mean": 2 * i},
                            "models": {name: {"metric_percent": 70 + i} for name in ("qN", "qC", "qD")}}}
                          for i in range(10)]
        self.binding = {"schema_version": "result-binding-1", "status": "author_completed",
                        "aggregation": "paired_official_split_t95", "manuscript_identifier": "test fixture",
                        "relationship_to_manuscript": "post_result_revision",
                        "run_identities": {"dataset": "Roman-empire", "splits": self.identities},
                        "results": [{"artifact": "derived/roman_split_aggregate.json",
                                    "json_pointer": "/paired_qD_minus_qC_loss/mean", "manuscript_location": "fixture",
                                    "reported_value": 4.5, "absolute_tolerance": 1e-12}]}
        identities = self.identities
        class Reader:
            def __init__(self, path):
                self.index = int(path.name[-2:])
                self.identities = {k: v for k, v in identities[self.index].items() if k != "split_index"}
            def json(self, path):
                return {"data": {"split_index": self.index}}
        self.reader = Reader

    def test_campaign_uses_all_split_effects_and_the_small_sample_interval(self):
        with patch("scripts.scientific_evidence_audit.NumericEvidence", self.reader):
            result = check_campaign_binding(self.paths, self.inspected, self.binding)
        self.assertEqual(result["status"], "MATCH")
        # For 0,...,9, sample variance is 55/6, SE=sqrt(11/12).
        expected = 4.5 + 2.2621571628540993 * np.sqrt(11 / 12)
        self.assertAlmostEqual(result["aggregate"]["paired_qD_minus_qC_loss"]["upper"], expected)
        self.assertEqual(result["aggregate"]["models"]["qD"]["metric_percent"], 74.5)

    def test_relabelled_split_identity_cannot_validate_matching_numbers(self):
        changed = copy.deepcopy(self.binding)
        changed["run_identities"]["splits"][9]["unit_sha256"] = "f" * 64
        with patch("scripts.scientific_evidence_audit.NumericEvidence", self.reader), \
             self.assertRaisesRegex(ValueError, "actual split identities"):
            check_campaign_binding(self.paths, self.inspected, changed)


class ActualRunEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from cross_morse.pipeline import Pipeline
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        directory = Path(cls.temporary.name)
        cls.run_root = directory / "run"
        cfg = json.loads((ROOT / "configs/arxiv_smoke.json").read_text())
        cfg["training"].update(epochs=1, patience=1)
        cfg["controller"].update(epochs=1, patience=1)
        cfg["selection"]["bootstrap_reps"] = 20
        cfg["admission"]["bootstrap_reps"] = 20
        with contextlib.redirect_stdout(io.StringIO()):
            Pipeline(cfg, directory / "data", cls.run_root, "cpu").run()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_complete_audit_reconstructs_weights_and_selection_without_unpickling(self):
        with patch("torch.load", side_effect=AssertionError("Evidence audit must not unpickle checkpoints")):
            result = audit_run(self.run_root)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["evidence_kind"], "synthetic_software_validation")
        self.assertTrue(result["numeric_replay"]["selection_replayed"])
        for profile in result["support"].values():
            self.assertTrue(profile["raw_weight_reconstruction"])
            self.assertTrue(all(cell["status"] == "PASS" for cell in profile["cells"].values()))

    def test_even_a_complete_fixture_cannot_count_as_an_official_archive(self):
        with self.assertRaisesRegex(ValueError, "official full-profile archive is required"):
            audit_run(self.run_root, official_dataset="ogbn-arxiv")


if __name__ == "__main__":
    unittest.main()
