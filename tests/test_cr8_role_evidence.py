"""A numerical replay archive is not automatically fresh-graph evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cross_morse.role_evidence import audit_role_isolation

ROOT = Path(__file__).resolve().parents[1]


class RoleEvidenceTests(unittest.TestCase):
    def test_legacy_independent_examples_do_not_pass_the_new_evidence_gate(self):
        for name in ("peptides_func",):
            with self.subTest(dataset=name):
                root = ROOT / "examples" / "runs" / name
                summary = root / "final" / "results_summary.json"
                original = summary.read_bytes()
                report = audit_role_isolation(root)
                self.assertTrue(report["applicable"])
                self.assertFalse(report["passed"])
                self.assertIn("new lifecycle run", " ".join(report["errors"]))
                self.assertEqual(original, summary.read_bytes())

    def test_transductive_scope_never_becomes_an_independent_graph_assertion(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.json").write_text(json.dumps({"dataset": "ogbn-arxiv"}))
            report = audit_role_isolation(directory)
            self.assertTrue(report["passed"])
            self.assertFalse(report["applicable"])
            self.assertIn("no independent-graph admission claim", report["reason"])

    def test_missing_independent_archive_is_a_failed_report(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.json").write_text(json.dumps({"dataset": "Peptides-func"}))
            report = audit_role_isolation(directory)
            self.assertTrue(report["applicable"])
            self.assertFalse(report["passed"])
            self.assertTrue(report["errors"])

    def test_audit_does_not_need_a_checkpoint_loader(self):
        with patch.dict("sys.modules", {"torch": None}):
            report = audit_role_isolation(ROOT / "examples" / "runs" / "peptides_func")
        self.assertFalse(report["passed"])
        self.assertTrue(report["errors"])


if __name__ == "__main__":
    unittest.main()
