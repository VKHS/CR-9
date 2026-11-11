"""Ephemeral numerical and integrity fixtures; never benchmark observations."""
from __future__ import annotations

import contextlib
import csv
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cm_campaigns.common import canonical, read_json, sha256
from cm_campaigns.contract import ADMITTED, ARMS
from paper.build.supplement import FACTORIAL_CELLS, ROOT, build_supplement
from scripts.build_supplement import main
from scripts.records import write_manifest, verify_set


class SupplementBuildTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.records = self.base / "records"
        self.records.mkdir()
        self.index = self.records / "index.json"
        self.specs = read_json(ROOT / "records/specs.json")
        self.entries = []
        self.index.write_bytes(canonical({"schema": "cross-morse-records-index-1", "sets": []}))

    def record(self, name, values, *, synthetic=False, **provenance):
        path = self.records / name
        (path / "raw").mkdir(parents=True, exist_ok=True)
        data = {str(index): item[2] for index, item in enumerate(values)}
        (path / "raw/fixture.json").write_bytes(canonical(data))
        (path / "provenance.json").write_bytes(canonical({
            **{field: "Ephemeral validator fixture, not an original experiment" for field in self.specs["provenance_fields"]},
            "synthetic": synthetic, **provenance}))
        with (path / "summary.csv").open("w", newline="") as handle:
            columns = self.specs["sets"][name]["keys"] + self.specs["summary_columns"]
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for index, (keys, quantity, value, unit) in enumerate(values):
                writer.writerow({**keys, "quantity": quantity, "value": repr(value), "unit": unit,
                                 "source_file": "raw/fixture.json", "locator": f"json:/{index}"})
        write_manifest(path)
        entry = verify_set(path, self.specs["sets"][name], self.specs)
        self.entries = [row for row in self.entries if row["set"] != name] + [entry]
        self.index.write_bytes(canonical({"schema": "cross-morse-records-index-1", "sets": self.entries}))
        return path

    def factorial_values(self):
        values = []
        for dataset in ("ogbn-arxiv", "PascalVOC-SP"):
            for cell in FACTORIAL_CELLS:
                a, o, x = map(int, cell)
                loss = 1 + .03*a + .02*o - .04*x + .012*a*o - .008*a*o*x
                if dataset == "PascalVOC-SP":
                    loss += .4
                quantities = {"ld_integrated_loss": (loss, "nats"), "interval_low": (loss-.01, "nats"),
                              "interval_high": (loss+.01, "nats"), "official_metric": (72.5+.1*(a+o+x), "percent"),
                              "activity": (.1+.05*(a+o+x), "fraction"), "relative_time": (1+.1*(a+o+x), "relative to cell 000")}
                values.extend(({"dataset": dataset, "cell": cell}, quantity, value, unit) for quantity, (value, unit) in quantities.items())
        return values

    def ensemble_values(self):
        return [({"dataset": dataset, "ensemble": str(ensemble)}, "loss_nats", .5 + .01*index + (.08+.01*index)/ensemble, "nats")
                for index, dataset in enumerate(ADMITTED) for ensemble in (1, 2, 4, 8, 12, 16, 20)]

    def assert_output_integrity(self, report, root):
        ids = [output["id"] for output in report["outputs"]]
        self.assertEqual(len(ids), len(set(ids)))
        for output in report["outputs"]:
            self.assertFalse(output["synthetic"])
            for relative in output["files"]:
                self.assertEqual(sha256(root / relative), output["sha256"][relative])
        self.assertEqual(read_json(root / "BUILD.json"), report)

    def test_empty_index_is_pending_without_invented_figures(self):
        out = self.base / "empty"
        report = build_supplement(out, records_index=self.index)
        self.assertEqual(report["status"], "PENDING")
        self.assertEqual(report["outputs"], [])
        self.assertEqual(report["errors"], [])
        self.assertEqual(len(report["skipped"]), 4)
        self.assertFalse((out / "figures").exists())

    def test_factorial_cells_and_contrasts_reconstruct_distinct_effect_conventions(self):
        self.record("factorials", self.factorial_values())
        out = self.base / "factorial"
        report = build_supplement(out, records_index=self.index)
        self.assertEqual(report["status"], "PASS", report["errors"])
        self.assertEqual(len(report["outputs"]), 5)
        for table in ("S16", "S16b"):
            with (out / f"tables/{table}.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["cell"] for row in rows], list(FACTORIAL_CELLS))
            self.assertEqual(len(rows), 8)
            self.assertTrue(all(float(row["interval_low"]) < float(row["ld_integrated_loss"]) < float(row["interval_high"]) for row in rows))
        with (out / "tables/factorial_point_contrasts.csv").open() as handle:
            contrasts = {(row["dataset"], row["contrast"]): row for row in csv.DictReader(handle)}
        main = contrasts["ogbn-arxiv", "A"]
        interaction = contrasts["ogbn-arxiv", "A:O"]
        triple = contrasts["ogbn-arxiv", "A:O:X"]
        self.assertAlmostEqual(float(main["effect_coded_high_minus_low_nats"]), .034)
        self.assertAlmostEqual(float(interaction["effect_coded_high_minus_low_nats"]), .004)
        self.assertAlmostEqual(float(interaction["baseline_anchored_difference_nats"]), .012)
        self.assertAlmostEqual(float(triple["baseline_anchored_difference_nats"]), -.008)
        self.assertTrue(all(not row["interval"] for row in contrasts.values()))
        self.assertTrue(all(item["coverage"] != "full_numerical_display" for item in report["outputs"]))
        self.assert_output_integrity(report, out)

    def test_missing_factorial_cell_is_an_error_not_a_partial_success(self):
        values = [row for row in self.factorial_values() if not (row[0]["dataset"] == "ogbn-arxiv" and row[0]["cell"] == "111")]
        self.record("factorials", values)
        report = build_supplement(self.base / "missing", records_index=self.index)
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("incomplete registered row", report["errors"][0]["error"])
        self.assertEqual(report["outputs"], [])

    def test_interval_and_units_are_validated_before_plotting(self):
        for fault in ("interval", "metric_units"):
            values = self.factorial_values()
            for index, (keys, quantity, value, unit) in enumerate(values):
                if fault == "interval" and quantity == "interval_high":
                    values[index] = keys, quantity, .01, unit
                    break
                if fault == "metric_units" and quantity == "official_metric":
                    values[index] = keys, quantity, value/100, "fraction"
                    break
            self.record("factorials", values)
            report = build_supplement(self.base / fault, records_index=self.index)
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["outputs"], [])

    def test_stale_raw_data_and_stale_index_are_refused(self):
        path = self.record("ensemble_grid", self.ensemble_values())
        (path / "raw/fixture.json").write_text('{}\n')
        first = build_supplement(self.base / "stale_raw", records_index=self.index)
        self.assertIn("changed after indexing", first["errors"][0]["error"])
        self.record("ensemble_grid", self.ensemble_values())
        index = read_json(self.index)
        index["sets"][0]["manifest_sha256"] = "0"*64
        self.index.write_bytes(canonical(index))
        second = build_supplement(self.base / "stale_index", records_index=self.index)
        self.assertIn("stale index manifest", second["errors"][0]["error"])

    def test_examples_and_advisory_record_classifications_are_refused(self):
        for tag, fields in (("synthetic", {"synthetic": True}), ("planning", {"data_kind": "model_based_planning"}),
                            ("toy", {"example": "toy CPU worked example"})):
            self.record("ensemble_grid", self.ensemble_values(), **fields)
            report = build_supplement(self.base / tag, records_index=self.index)
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["outputs"], [])

    def test_the_demonstration_builds_from_its_fixture_sets_and_labels_every_output(self):
        self.record("factorials", self.factorial_values(), synthetic=True)
        self.record("ensemble_grid", self.ensemble_values(), synthetic=True)
        refused = build_supplement(self.base / "paper", records_index=self.index)
        self.assertEqual(refused["status"], "FAIL")
        self.assertEqual(refused["outputs"], [])
        out = self.base / "demonstration"
        report = build_supplement(out, records_index=self.index, demonstration=True)
        self.assertEqual(report["status"], "PASS", report["errors"])
        self.assertTrue(report["demonstration"] and report["synthetic"] and report["synthetic_examples"])
        self.assertTrue(report["outputs"])
        for output in report["outputs"]:
            self.assertTrue(output["synthetic"] and output["demonstration"])
            self.assertEqual(output["input_kind"], "synthetic_examples")
            self.assertTrue(output["note"].startswith("CPU execution on public data"))
        self.assertIn("CPU execution on public data", (out / "tables/S16.md").read_text())
        self.assertIn("CPU execution on public data", (out / "figures/ensemble_fit.svg").read_text())
        self.assertEqual(sorted((row["record_set"], row["input_kind"], row["synthetic"]) for row in report["inputs"]
                                if "record_set" in row),
                         [("ensemble_grid", "cpu_execution_record_set", True), ("factorials", "cpu_execution_record_set", True)])
        # an example set stays refused in the demonstration, and the paper's build carries no banner afterwards
        self.record("ensemble_grid", self.ensemble_values(), synthetic=True, example="toy CPU worked example")
        report = build_supplement(self.base / "example", records_index=self.index, demonstration=True)
        self.assertIn("ensemble_grid", [row["input"] for row in report["errors"]])
        self.record("factorials", self.factorial_values())
        report = build_supplement(self.base / "after", records_index=self.index)
        self.assertNotIn("CPU execution on public data", (self.base / "after/tables/S16.md").read_text())

    def test_ensemble_law_is_fit_to_original_points_with_exact_csv_plot_sources(self):
        self.record("ensemble_grid", self.ensemble_values())
        out = self.base / "ensemble"
        report = build_supplement(out, records_index=self.index)
        self.assertEqual(report["status"], "PASS", report["errors"])
        with (out / "tables/S26b_fit.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 6)
        self.assertAlmostEqual(float(rows[0]["l_inf"]), .5)
        self.assertAlmostEqual(float(rows[0]["D_inf"]), .08)
        self.assertLess(float(rows[0]["max_abs_residual"]), 1e-14)
        with (out / "data/ensemble_grid.csv").open() as handle:
            points = list(csv.DictReader(handle))
        self.assertEqual(len(points), 42)
        self.assertEqual(float(points[5]["loss_nats"]), self.ensemble_values()[5][2])
        self.assertTrue(all(item["display_ids"] == ["supplementary:Table S26b"] for item in report["outputs"]))
        self.assert_output_integrity(report, out)

    def test_missing_ensemble_sizes_and_invalid_index_do_not_pass(self):
        values = [row for row in self.ensemble_values() if row[0]["ensemble"] != "20"]
        self.record("ensemble_grid", values)
        report = build_supplement(self.base / "incomplete_ensemble", records_index=self.index)
        self.assertEqual(report["status"], "FAIL")
        self.index.write_bytes(canonical({"schema": "wrong", "sets": []}))
        report = build_supplement(self.base / "invalid_index", records_index=self.index)
        self.assertEqual(report["errors"][0]["input"], "records_index")

    def test_output_preserves_existing_files_and_cannot_modify_inputs(self):
        out = self.base / "existing"
        out.mkdir()
        sentinel = out / "keep.txt"
        sentinel.write_text("unchanged")
        with self.assertRaisesRegex(ValueError, "fresh or empty"):
            build_supplement(out)
        self.assertEqual(sentinel.read_text(), "unchanged")
        with self.assertRaisesRegex(ValueError, "separate from all input"):
            build_supplement(self.records / "generated", records_index=self.index)

    def test_the_demonstration_reconstruction_reports_a_supplementary_input_it_cannot_build(self):
        from scripts.reconstruct import reconstruct, verify_workspace
        values = [row for row in self.factorial_values() if not (row[0]["dataset"] == "ogbn-arxiv" and row[0]["cell"] == "111")]
        self.record("factorials", values, synthetic=True)
        self.record("ensemble_grid", self.ensemble_values(), synthetic=True)
        with contextlib.redirect_stdout(io.StringIO()):
            paper = reconstruct(self.base / "paper_workspace", records_index=self.index, label="paper")
            demonstration = reconstruct(self.base / "demonstration_workspace", records_index=self.index,
                                        label="demonstration", demonstration=True, scope_note="CPU execution.")
            # the scope opens the report inside the verified workspace, which a second build replaces
            report = (self.base / "demonstration_workspace/RECONSTRUCTION_REPORT.md").read_text()
            self.assertTrue(report.startswith("> **CPU execution.** CPU execution."))
            verify_workspace(self.base / "demonstration_workspace")
            reconstruct(self.base / "demonstration_workspace", records_index=self.index, label="demonstration",
                        demonstration=True, scope_note="CPU execution.")
        # the paper's reconstruction does not build supplementary displays from synthetic sets
        self.assertIsNone(paper["supplementary_build"])
        errors = demonstration["display_generators"]["errors"]
        self.assertIn("incomplete registered row", errors["supplement:factorials"])
        self.assertNotIn("supplement:ensemble_grid", errors)
        self.assertTrue(any(row["id"] == "ensemble_law_fit" and row["input_kind"] == "synthetic_examples"
                            for row in demonstration["outputs"]))

    def test_record_symlinks_cannot_supply_values(self):
        path = self.record("ensemble_grid", self.ensemble_values())
        original = path / "raw/fixture.json"
        other = self.base / "external.json"
        original.replace(other)
        original.symlink_to(other)
        report = build_supplement(self.base / "symlink", records_index=self.index)
        self.assertIn("Symlink", report["errors"][0]["error"])

    def test_original_joint_s14_outputs_remain_nine_tests_not_local_families(self):
        from tests.test_s14_confirmation import S14CampaignTests
        fixture = S14CampaignTests("test_complete_family_reconstructs_effects_and_returns_failed_peptides_to_comparator")
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        out = self.base / "s14"
        report = build_supplement(out, confirmation_campaign=fixture.root, manuscript_root=fixture.manuscript)
        self.assertEqual(report["status"], "PASS", report["errors"])
        with (out / "tables/S14_joint.csv").open() as handle:
            rows = {row["contrast"]: row for row in csv.DictReader(handle)}
        self.assertEqual(len(rows), 9)
        self.assertEqual(rows["Peptides/radius"]["retained"], "false")
        self.assertEqual(rows["products/state"]["tested_policy_selected"], "false")
        self.assertAlmostEqual(float(rows["Peptides/radius"]["joint_Holm_P_zero"]), .07)
        self.assertFalse(report["inputs"][0]["original_test_p_values_recomputed"])
        self.assert_output_integrity(report, out)
        fixture.change_result("COCO/RH", p_value_kind="locally_Holm_adjusted")
        rejected = build_supplement(self.base / "not_original", confirmation_campaign=fixture.root, manuscript_root=fixture.manuscript)
        self.assertEqual(rejected["status"], "FAIL")
        self.assertEqual(rejected["outputs"], [])

    def budget_fixture(self):
        summary = []
        for dataset in ADMITTED:
            for index, arm in enumerate(ARMS):
                effect = (.001, .002, .003, .0003)[index]
                summary.append({"dataset": dataset, "arm": arm, "effect": effect, "effect_vs_qC": effect-.005,
                                "ci_low": effect-.002, "ci_high": effect+.002, "p_holm": .04,
                                "work_ratio": 1.08 if arm == "E6" else 1.2, "reference_work_ratio": 1.18, "ceiling": 1.2,
                                "seeds": 2, "units": 10, "inference_scope": "fixture only"})
        return {"passed": True, "summary": summary, "historical_table_reproduction": False, "scope": "validator fixture only",
                "reserve_check": {"reserve_trace_passed": True}}

    def test_s14_files_cannot_change_between_validation_and_final_input_hash(self):
        from cm_campaigns.confirmation import check_campaign, input_identity
        from tests.test_s14_confirmation import S14CampaignTests
        fixture = S14CampaignTests("test_complete_family_reconstructs_effects_and_returns_failed_peptides_to_comparator")
        fixture.setUp()
        self.addCleanup(fixture.tearDown)

        def mutate_after_validation(path, **kwargs):
            result = check_campaign(path, **kwargs)
            original = fixture.root / fixture.manifest["tests"][0]["result"]["path"]
            original.write_text(original.read_text() + "\n")
            result["input_identity"] = input_identity(fixture.root)
            return result

        with patch("cm_campaigns.confirmation.check_campaign", side_effect=mutate_after_validation):
            report = build_supplement(self.base / "changed_campaign", confirmation_campaign=fixture.root,
                                      manuscript_root=fixture.manuscript)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["outputs"], [])
        self.assertIn("changed during verification", report["errors"][0]["error"])

    def test_budget_plots_use_strict_replay_and_work_over_ceiling(self):
        campaign = self.base / "campaign"
        campaign.mkdir()
        study = {"fixture_only": True}
        (campaign / "study.json").write_bytes(canonical(study))
        out = self.base / "budget"
        with patch("cm_campaigns.arm_campaign.check_campaign", return_value=self.budget_fixture()) as check:
            report = build_supplement(out, budget_campaign=campaign)
        check.assert_called_once_with(campaign, study, strict=True)
        self.assertEqual(report["status"], "PASS", report["errors"])
        with (out / "tables/S46b_comparison.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 24)
        e6 = next(row for row in rows if row["arm"] == "E6")
        self.assertAlmostEqual(float(e6["work_fraction_of_ceiling"]), .9)
        self.assertAlmostEqual(float(e6["teacher_effect_vs_qC_nats"]), -.005)
        self.assertTrue(all(item["historical_table_reproduction"] is False for item in report["outputs"]))
        self.assert_output_integrity(report, out)
        # A real call to the validator refuses this non-campaign immediately.
        rejected = build_supplement(self.base / "bad_campaign", budget_campaign=campaign)
        self.assertEqual(rejected["status"], "FAIL")
        self.assertEqual(rejected["outputs"], [])
        # the CPU execution's campaign is replayed without the strict qualification, and labelled
        with patch("cm_campaigns.arm_campaign.check_campaign", return_value=self.budget_fixture()) as check:
            report = build_supplement(self.base / "budget_demonstration", budget_campaign=campaign, demonstration=True)
        check.assert_called_once_with(campaign, study, strict=False)
        self.assertEqual(report["status"], "PASS", report["errors"])
        self.assertTrue(all(item["synthetic"] and item["input_kind"] == "synthetic_examples"
                            and item["source_input_kind"] == "strict_budget_campaign" for item in report["outputs"]))
        self.assertEqual([row["input_kind"] for row in report["inputs"]], ["cpu_execution_budget_campaign"])
        with patch("cm_campaigns.arm_campaign.check_campaign", side_effect=ModuleNotFoundError("No module named 'torch'")):
            report = build_supplement(self.base / "budget_without_runtime", budget_campaign=campaign, demonstration=True)
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("recorded CPU environment", report["errors"][0]["error"])
        short = dict(self.budget_fixture(), summary=self.budget_fixture()["summary"][:20])
        with patch("cm_campaigns.arm_campaign.check_campaign", return_value=short):
            report = build_supplement(self.base / "budget_short", budget_campaign=campaign, demonstration=True)
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("24 replayed contrasts", report["errors"][0]["error"])

    def test_missing_campaign_paths_are_errors_but_omitted_inputs_are_skipped(self):
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(["--confirmation-campaign", str(self.base / "missing"), "--output-dir", str(self.base / "cli")])
        self.assertEqual(code, 1)
        result = read_json(self.base / "cli/BUILD.json")
        self.assertEqual(result["errors"][0]["input"], "confirmation_campaign")
        self.assertEqual(len(result["skipped"]), 3)


if __name__ == "__main__":
    unittest.main()
