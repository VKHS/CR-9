"""Reporting regressions; these constructed values are not benchmark data."""
import copy
import csv
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import reproduce_examples as re


def report_fixture():
    runs = []
    for stem, dataset in re.TASKS:
        models = {p: {"metric": .1 * (i + 1), "loss_name": "cross_entropy", "loss_nats": 1.25 - i * .1,
                      "test_samples": 32, "test_units": 8} for i, p in enumerate(re.PREDICTORS)}
        runs.append({"stem": stem, "benchmark_dataset": dataset,
            "prediction_level": "node", "deployed_route": "radius", "output_dimensions": 4,
            "identities": {"source_sha256": "a"*64, "config_sha256": "b"*64, "unit_sha256": "c"*64},
            "replay": {"models": models, "task": {"metric": "accuracy"}, "admitted": True,
                       "paired_qD_minus_qC_loss": {"mean": -.1, "lower": -.2, "upper": 0.},
                       "paired_qD_minus_qC_metric_pp": {"mean": 10.}}})
    return {"schema_version": re.SCHEMA, "scope": re.SCOPE, "runtime_source_sha256": "a"*64, "runs": runs}


class ExampleReproductionTests(unittest.TestCase):
    def test_table_keeps_native_scores_and_all_predictors(self):
        report = report_fixture()
        report["runs"][-1]["replay"]["task"]["metric"] = "ap"
        rows = re.table_rows(report)
        self.assertEqual(len(rows), len(re.TASKS) * len(re.PREDICTORS))
        self.assertEqual(rows[-1]["metric_fraction"], .4)
        self.assertEqual(rows[-1]["metric_name"], "ap")
        self.assertEqual(rows[-1]["data_kind"], "synthetic")
        views = re.rendered_tables(report)
        self.assertIn("not official benchmark scores", views["metrics.md"])
        self.assertEqual(json.loads(views["replay.json"]), report)

    def test_refresh_is_idempotent_and_preserves_unrelated_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/"report"
            report = report_fixture()
            re.write_reports(report, out)
            original = {p.name: p.read_bytes() for p in out.iterdir()}
            (out/"reader_notes.txt").write_text("keep")
            re.write_reports(report, out)
            self.assertEqual(original, {name: (out/name).read_bytes() for name in original})
            self.assertEqual((out/"reader_notes.txt").read_text(), "keep")
            re.write_reports(report, out, check=True)

    def test_user_edited_report_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/"report"
            re.write_reports(report_fixture(), out)
            (out/"metrics.csv").write_text("a manual correction")
            with self.assertRaisesRegex(re.EvidenceError, "changed outside"):
                re.write_reports(report_fixture(), out)
            self.assertEqual((out/"metrics.csv").read_text(), "a manual correction")

    def test_unowned_collisions_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/"report"
            out.mkdir()
            (out/"metrics.md").write_text("existing unrelated report")
            with self.assertRaisesRegex(re.EvidenceError, "no generator manifest"):
                re.write_reports(report_fixture(), out)
            self.assertFalse((out/"metrics.csv").exists())

    def test_check_detects_changed_recomputed_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/"report"
            first = report_fixture()
            re.write_reports(first, out)
            second = copy.deepcopy(first)
            second["runs"][0]["replay"]["models"]["qD"]["metric"] = .75
            with self.assertRaisesRegex(re.EvidenceError, "differ or are missing"):
                re.write_reports(second, out, check=True)

    def test_check_accepts_cross_blas_roundoff_without_rewriting_saved_views(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report"
            first = report_fixture()
            re.write_reports(first, out)
            original = {p.name: p.read_bytes() for p in out.iterdir()}
            second = copy.deepcopy(first)
            metric = second["runs"][-1]["replay"]["models"]["qC"]
            metric["metric"] = math.nextafter(metric["metric"], math.inf)
            re.write_reports(second, out, check=True)
            self.assertEqual(original, {p.name: p.read_bytes() for p in out.iterdir()})

    def test_check_rejects_changed_decisions_identities_counts_and_report_shape(self):
        changes = {
            "admission": lambda r: r["runs"][0]["replay"].update(admitted=False),
            "boolean_as_integer": lambda r: r["runs"][0]["replay"].update(admitted=1),
            "source": lambda r: r["runs"][0]["identities"].update(source_sha256="d" * 64),
            "count": lambda r: r["runs"][0]["replay"]["models"]["qC"].update(test_samples=33),
            "count_type": lambda r: r["runs"][0]["replay"]["models"]["qC"].update(test_samples=32.0),
            "extra_field": lambda r: r.update(unregistered=True),
            "missing_run": lambda r: r["runs"].pop(),
            "material_numeric_change": lambda r: r["runs"][0]["replay"]["models"]["qC"].update(metric=.30000001),
            "nonfinite": lambda r: r["runs"][0]["replay"]["models"]["qC"].update(metric=math.nan),
        }
        for name, change in changes.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "report"
                report = report_fixture()
                re.write_reports(report, out)
                saved = copy.deepcopy(report)
                change(saved)
                (out / "replay.json").write_text(json.dumps(saved))
                with self.assertRaisesRegex(re.EvidenceError, "replay.json"):
                    re.write_reports(report, out, check=True)

    def test_check_rejects_csv_corruption_and_reports_numeric_delta(self):
        for cell_value in ("nan", "inf", "0.30000001", "not-a-number"):
            with self.subTest(cell=cell_value), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "report"
                report = report_fixture()
                re.write_reports(report, out)
                rows = list(csv.reader(io.StringIO((out / "metrics.csv").read_text())))
                rows[3][rows[0].index("metric_fraction")] = cell_value
                stream = io.StringIO(newline="")
                csv.writer(stream, lineterminator="\n").writerows(rows)
                (out / "metrics.csv").write_text(stream.getvalue())
                with self.assertRaisesRegex(re.EvidenceError, r"metrics.csv\[row 4\].metric_fraction") as result:
                    re.write_reports(report, out, check=True)
                if cell_value == "0.30000001":
                    self.assertIn("absolute difference=", str(result.exception))

    def test_check_rejects_duplicate_json_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report"
            report = report_fixture()
            re.write_reports(report, out)
            path = out / "replay.json"
            path.write_text(path.read_text().replace('"admitted": true', '"admitted": false, "admitted": true', 1))
            with self.assertRaisesRegex(re.EvidenceError, "duplicate JSON field"):
                re.write_reports(report, out, check=True)

    def test_destination_guards_and_readonly_committed_views(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/"source"
            source.mkdir()
            runs = source/"examples/runs"
            runs.mkdir(parents=True)
            with patch.object(re, "SOURCE", source):
                with self.assertRaises(re.EvidenceError):
                    re.safe_destination(source/"examples/generated", runs)
                self.assertEqual(re.safe_destination(source/"examples/generated", runs, readonly=True), source/"examples/generated")
                with self.assertRaises(re.EvidenceError):
                    re.safe_destination(runs/"arxiv/report", runs, readonly=True)
                with self.assertRaises(re.EvidenceError):
                    re.safe_destination(Path(tmp), runs)

    def test_missing_runs_fail_before_creating_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/"reports"
            self.assertEqual(re.main(["--runs-root", str(Path(tmp)/"missing"), "--output-dir", str(out)]), 1)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
