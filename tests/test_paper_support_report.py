"""Current-paper source binding, complete display coverage and result semantics."""
from __future__ import annotations

import copy
import csv
import io
import json
from pathlib import Path
import unittest

from scripts import paper_support_report as report

ROOT = Path(__file__).resolve().parents[1]


class PaperSupportTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads((ROOT / "paper/display_manifest.json").read_text())

    def test_current_inventory_covers_all_reviewed_displays(self):
        report.validate_manifest(ROOT, self.manifest)
        self.assertEqual(len(self.manifest["displays"]), 90)
        self.assertEqual({r["id"] for r in self.manifest["displays"]}, report.expected_displays())

    def test_missing_and_duplicate_displays_are_rejected(self):
        for change in (lambda rows: rows.pop(), lambda rows: rows.append(copy.deepcopy(rows[0]))):
            manifest = copy.deepcopy(self.manifest)
            change(manifest["displays"])
            with self.assertRaises(ValueError):
                report.validate_manifest(ROOT, manifest)

    def test_a_changed_manuscript_or_registry_cannot_reuse_reviewed_mapping(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["documents"]["main"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "current PDFs"):
            report.validate_manifest(ROOT, manifest)
        manifest = copy.deepcopy(self.manifest)
        manifest["registry_sha256"]["reported_results.json"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "changed reported_results"):
            report.validate_manifest(ROOT, manifest)

    def test_unsupported_evidence_status_and_dangling_paths_fail(self):
        mutations = (
            ("coverage", "all_measurements_verified"),
            ("implementation", ["scripts/missing_original_analysis.py"]),
            ("commands", ["make nonexistent-target"]),
            ("registry_groups", ["nonexistent_result_group"]),
            ("record_sets", ["invented_observations"]),
            ("pages", [999]),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                manifest = copy.deepcopy(self.manifest)
                manifest["displays"][0][field] = value
                with self.assertRaises(ValueError):
                    report.validate_manifest(ROOT, manifest)

    def test_report_retains_measured_arm_and_separate_frozen_rule_outcome(self):
        outputs = report.build(ROOT)
        rows = {row["dataset"]: row for row in csv.DictReader(io.StringIO(outputs["paper/RESULTS.csv"]))}
        peptides = rows["Peptides-func"]
        self.assertEqual(float(peptides["paired_qD_minus_qC_loss_nats"]), 0.0)
        self.assertEqual(float(peptides["evaluated_arm_loss_nats"]), -0.00303)
        self.assertEqual(float(peptides["evaluated_arm_metric_gain_pp"]), 0.58)
        self.assertEqual(peptides["loss_name"], "BCE")
        self.assertEqual(peptides["lifecycle_resolution"], "return_qC")
        self.assertIn("radius, E=4 (evaluated arm)", outputs["paper/RESULTS.md"])
        self.assertIn("0.0058 in native AP units", outputs["paper/RESULTS.md"])
        self.assertIn("not a measured maximum", outputs["paper/RESULTS.md"])

    def test_registered_counts_do_not_infer_evidence_from_manuscript_values(self):
        source = json.loads((ROOT / "paper/sources.json").read_text())
        state = {"official": {}, "records": {"verified": []}}
        text = report.render_support(self.manifest, source, state)
        self.assertIn("`runs/index.json` and `records/index.json`", text)
        self.assertIn("supplied files from generated products", text)
        self.assertNotIn("0/7", text)
        self.assertNotIn("0/10", text)
        self.assertIn("Original analysis required", text)
        self.assertIn("Related plots / partial panels", text)

    def test_generated_reports_are_current(self):
        for relative, content in report.build(ROOT).items():
            self.assertEqual((ROOT / relative).read_text(encoding="utf-8"), content, relative)


if __name__ == "__main__":
    unittest.main()
