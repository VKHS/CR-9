"""The analytical-statement battery, its registry of statements and docs/THEORY.md; NumPy only."""
from __future__ import annotations

import importlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from cm_theory import STATEMENTS, Check, statement  # noqa: E402
from cm_theory import verification  # noqa: E402
from scripts import verify_theory  # noqa: E402


class StatementRegistryTests(unittest.TestCase):
    def test_keys_are_unique_and_every_statement_has_checks(self):
        keys = [s.key for s in STATEMENTS]
        self.assertEqual(len(keys), len(set(keys)))
        checked = set(verification.CHECK_STATEMENT.values())
        self.assertEqual(set(keys) - checked, set())
        self.assertEqual(checked - set(keys), set())
        for s in STATEMENTS:
            self.assertEqual(sorted(s.checks), sorted(n for n, k in verification.CHECK_STATEMENT.items() if k == s.key), s.key)

    def test_main_text_statements_are_all_registered(self):
        labels = {s.label for s in STATEMENTS}
        expected = {"Definition 1", "Definition 2", "Lemma 1", "Lemma 2", "Theorem 1", "Theorem 2", "Corollary 1", "Corollary 2"}
        expected |= {f"Proposition {i}" for i in range(1, 12)}
        self.assertEqual(expected - labels, set())

    def test_reference_implementations_resolve(self):
        for s in STATEMENTS:
            for dotted in s.theory:
                module, name = dotted.rsplit(".", 1)
                self.assertTrue(callable(getattr(importlib.import_module(module), name)), dotted)

    def test_runtime_counterparts_name_existing_files(self):
        for s in STATEMENTS:
            for entry in s.runtime:
                path = entry.split(":")[0].split(" (")[0].strip()
                self.assertTrue((PROJECT / path).exists(), f"{s.key}: {path}")

    def test_lookup(self):
        self.assertEqual(statement("theorem-2").label, "Theorem 2")
        with self.assertRaises(KeyError):
            statement("theorem-3")


class BatteryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = verification.run_battery(quick=True)

    def test_quick_battery_passes_and_covers_every_statement(self):
        failures = [r for r in self.result["rows"] if r["status"] == "FAIL"]
        self.assertEqual(failures, [])
        self.assertEqual(self.result["statements_without_checks"], [])
        self.assertGreater(self.result["counts"]["PASS"], 80)

    def test_rows_are_json_serializable_and_typed(self):
        text = json.dumps(self.result)
        self.assertIn("cm-theory-battery-1", text)
        for row in self.result["rows"]:
            self.assertIn(row["status"], ("PASS", "FAIL", "ADVISORY", "INFO"))
            self.assertIn(row["statement_key"], {s.key for s in STATEMENTS})

    def test_another_seed_passes(self):
        result = verification.run_battery(quick=True, seed=7, only=["lemma-2", "theorem-1", "theorem-2", "lemma-3"])
        self.assertEqual(result["counts"]["FAIL"], 0)

    def test_a_crashing_check_is_a_failure(self):
        def broken(rng, quick):
            raise RuntimeError("boom")
        with mock.patch.dict(verification.CHECKS, {"eq6_dil": broken}):
            result = verification.run_battery(quick=True, only=["eq6_dil"])
        self.assertEqual(result["counts"]["FAIL"], 1)
        self.assertIn("RuntimeError", result["rows"][0]["observed"])

    def test_a_false_conclusion_is_detected(self):
        # Corrupt Lemma 2's bound to (1-eta)^(T+1): the periodic test graphs exceed it.
        from cm_theory import operators
        original = operators.resolvent_truncation_error

        def halved(P, eta, order):
            out = original(P, eta, order)
            return {**out, "bound": out["bound"] / 2}
        with mock.patch.object(operators, "resolvent_truncation_error", halved):
            result = verification.run_battery(quick=True, only=["lemma2_resolvent_bound"])
        self.assertGreater(result["counts"]["FAIL"], 0)

    def test_check_status_vocabulary(self):
        with self.assertRaises(ValueError):
            Check("x", "y", "OK")

    def test_doc_is_current(self):
        self.assertEqual((PROJECT / "docs" / "THEORY.md").read_text(encoding="utf-8"), verify_theory.render_doc(self.result))


class CommandLineTests(unittest.TestCase):
    def run_cli(self, *args):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = verify_theory.main(list(args))
        return code, buffer.getvalue()

    def test_subset_writes_json_and_markdown(self):
        with tempfile.TemporaryDirectory(prefix="cm-theory-") as tmp:
            code, out = self.run_cli("--only", "corollary-2", "proposition-8", "--json-output", f"{tmp}/t.json",
                                     "--markdown", f"{tmp}/t.md", "--quiet")
            self.assertEqual(code, 0, out)
            data = json.loads(Path(tmp, "t.json").read_text())
            self.assertEqual({r["statement_key"] for r in data["rows"]}, {"corollary-2", "proposition-8"})
            self.assertIn("| PASS |", Path(tmp, "t.md").read_text())

    def test_unknown_statement_is_a_usage_error(self):
        with redirect_stdout(io.StringIO()), mock.patch("sys.stderr", io.StringIO()):
            self.assertEqual(verify_theory.main(["--only", "no-such-statement"]), 2)

    def test_doc_options_refuse_non_default_runs(self):
        with self.assertRaises(SystemExit), mock.patch("sys.stderr", io.StringIO()):
            verify_theory.main(["--write-doc", "--full"])


if __name__ == "__main__":
    unittest.main()
