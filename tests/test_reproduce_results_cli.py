"""The public numerical-replay command uses the run's exact runtime source.

The bundled example contains synthetic observations. These checks establish
source routing, historical software replay and report handling, not paper results.
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from cross_morse.review import evidence_markdown
from scripts import reproduce_results
from scripts._paper_common import runtime_source_fingerprint


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/runs/arxiv"


class ReproduceResultsCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def invoke(self, *arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = reproduce_results.main(list(map(str, arguments)))
        return code, stdout.getvalue(), stderr.getvalue()

    def historical_replay(self, *arguments):
        expected = json.loads((EXAMPLE / "provenance/stages/00_prepare.json").read_text())["source_sha256"]
        self.assertNotEqual(expected, runtime_source_fingerprint(ROOT), "fixture must exercise historical routing")
        original_run = subprocess.run
        with mock.patch("scripts._source_snapshots.subprocess.run", wraps=original_run) as spawned:
            code, stdout, stderr = self.invoke(EXAMPLE, *arguments)
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stderr, "")
        # The actual replay subprocess must import the preserved source. A
        # direct call to the currently imported review code does not satisfy it.
        spawned.assert_called_once()
        command = spawned.call_args.args[0]
        self.assertEqual(command[1:4], ["-I", "-B", "-c"])
        self.assertEqual(Path(command[-4]), ROOT / "examples/source_snapshots" / expected)
        self.assertEqual(Path(command[-3]), EXAMPLE)
        self.assertEqual(command[-1], expected)
        report = json.loads(stdout)
        self.assertEqual(report["source_sha256"], expected)
        self.assertEqual(report["status"], "PASS")
        self.assertFalse(report["scientific_run"])
        return report, command

    def test_historical_runtime_is_executed_and_report_formats_are_preserved(self):
        output = self.root / "reports/replay.json"
        entry = EXAMPLE / "select/entry_complete.json"
        before = entry.read_bytes()
        report, command = self.historical_replay("--output", output)
        self.assertEqual(command[-2], "true")
        self.assertTrue(report["selection_replayed"])
        self.assertTrue(report["confirmation_replayed"])
        self.assertTrue(report["admission_replayed"])
        self.assertEqual(output.read_text(), json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        self.assertEqual(output.with_suffix(".md").read_text(), evidence_markdown(report))
        self.assertEqual(entry.read_bytes(), before)

    def test_skip_selection_is_forwarded_to_the_historical_runtime(self):
        report, command = self.historical_replay("--skip-selection-replay")
        self.assertEqual(command[-2], "false")
        self.assertFalse(report["selection_replayed"])
        self.assertTrue(report["confirmation_replayed"])
        self.assertTrue(report["admission_replayed"])

    def test_unresolvable_source_is_rejected_before_execution_or_report_creation(self):
        run = self.root / "run"
        stage = run / "provenance/stages/00_prepare.json"
        stage.parent.mkdir(parents=True)
        stage.write_text(json.dumps({"source_sha256": "f" * 64}))
        output = self.root / "unavailable/replay.json"
        with mock.patch("scripts._source_snapshots.subprocess.run") as spawned:
            code, stdout, stderr = self.invoke(run, "--output", output)
        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        failure = json.loads(stderr)
        self.assertEqual(failure["status"], "FAIL")
        self.assertIn("exact runtime source snapshot", failure["error"])
        spawned.assert_not_called()
        self.assertFalse(output.exists())
        self.assertFalse(output.with_suffix(".md").exists())

    def test_existing_reports_and_run_contents_are_protected_before_replay(self):
        run = self.root / "run"
        run.mkdir()
        existing = self.root / "existing.json"
        existing.write_text("original report")
        markdown = self.root / "markdown-only.md"
        markdown.write_text("original markdown")
        for output in (run / "report.json", existing, markdown.with_suffix(".json")):
            with self.subTest(output=output), mock.patch.object(reproduce_results, "replay_results_exact") as replay:
                code, stdout, stderr = self.invoke(run, "--output", output)
                self.assertEqual(code, 1)
                self.assertEqual(stdout, "")
                failure = json.loads(stderr)
                self.assertEqual(failure["status"], "FAIL")
                self.assertIn("new and outside the immutable run", failure["error"])
                replay.assert_not_called()
        self.assertEqual(existing.read_text(), "original report")
        self.assertEqual(markdown.read_text(), "original markdown")
        self.assertFalse((run / "report.json").exists())


if __name__ == "__main__":
    unittest.main()
