"""Confirmation input propagation and saved-review freshness.

All campaign files and positive audit summaries here are private temporary test
fixtures. They test report integrity, not any official result or confirmation.
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts import readiness, review
from tests.test_uplift_tools import write_complete_reports

PROJECT = Path(__file__).resolve().parents[1]


class ConfirmationReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cm-confirmation-review-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.campaign = self.root / "confirmation"
        (self.campaign / "raw").mkdir(parents=True)
        (self.campaign / "code").mkdir()
        (self.campaign / "campaign.json").write_text('{"fixture": true}\n')
        (self.campaign / "raw/paired.csv").write_text("unit,loss_a,loss_b\n0,0.1,0.2\n")
        (self.campaign / "code/original_test.py").write_text("# Private identity fixture.\n")

    def options(self):
        return {"index": PROJECT / "runs/index.json", "records": PROJECT / "records",
                "confirmation_campaign": self.campaign}

    def saved_fixture(self):
        invocation = self.root / "invocation"
        invocation.mkdir()
        write_complete_reports(invocation)
        audit_path = invocation / "scientific-evidence-audit.json"
        audit = json.loads(audit_path.read_text())
        audit["inputs"] = {"repository_root": str(PROJECT), "run_index": str(self.options()["index"]),
                           "records_root": str(self.options()["records"]),
                           "confirmation_campaign": str(self.campaign)}
        audit_path.write_text(json.dumps(audit))
        steps = review.plan(invocation, offline=False, **self.options())
        saved = {"schema": "cross-morse-review-1", "offline": False, "verdict": "EVIDENCE COMPLETE",
                 "inputs_unchanged": True, "input_identity": review.input_identity(**self.options()),
                 "artifact_directory": str(invocation), "artifact_sha256": review.artifact_identities(invocation),
                 "steps": [{"name": step.name, "group": step.group, "status": "PASS", "exit": 0} for step in steps]}
        path = self.root / "review.json"
        path.write_text(json.dumps(saved))
        return path, invocation, saved

    def test_raw_observations_and_nested_dependencies_invalidate_saved_review(self):
        path, _, _ = self.saved_fixture()
        self.assertEqual(review.validate_saved_review(path, **self.options())["verdict"], "EVIDENCE COMPLETE")
        for member in (self.campaign / "raw/paired.csv", self.campaign / "code/original_test.py"):
            original = member.read_bytes()
            member.write_bytes(original + b"changed\n")
            with self.subTest(member=member.name), self.assertRaisesRegex(ValueError, "stale"):
                review.validate_saved_review(path, **self.options())
            member.write_bytes(original)
        self.assertEqual(review.validate_saved_review(path, **self.options())["verdict"], "EVIDENCE COMPLETE")

    def test_campaign_names_additions_removals_and_budget_payload_are_bound(self):
        before = review._directory_identity(self.campaign)
        self.assertEqual(before["files"], 3)
        original, renamed = self.campaign / "raw/paired.csv", self.campaign / "raw/renamed.csv"
        original.rename(renamed)
        self.assertNotEqual(review._directory_identity(self.campaign), before)
        renamed.rename(original)
        extra = self.campaign / "new_protocol.json"
        extra.write_text("{}")
        self.assertNotEqual(review._directory_identity(self.campaign), before)
        extra.unlink()
        self.assertEqual(review._directory_identity(self.campaign), before)
        options = {**self.options(), "budget_campaign": self.campaign}
        identity = review.input_identity(**options)
        original.write_text("different original paired observations")
        after = review.input_identity(**options)
        self.assertNotEqual(after["campaign"], identity["campaign"])
        self.assertNotEqual(after["confirmation_campaign"], identity["confirmation_campaign"])

    def test_campaign_symlink_dependency_is_rejected_even_when_dangling(self):
        link = self.campaign / "raw/linked.csv"
        link.symlink_to(self.root / "absent.csv")
        with self.assertRaisesRegex(ValueError, "symlink"):
            review.input_identity(**self.options())

    def test_audit_with_a_different_campaign_cannot_supply_saved_verdict(self):
        path, invocation, saved = self.saved_fixture()
        audit_path = invocation / "scientific-evidence-audit.json"
        audit = json.loads(audit_path.read_text())
        audit["inputs"]["confirmation_campaign"] = None
        audit_path.write_text(json.dumps(audit))
        saved["artifact_sha256"] = review.artifact_identities(invocation)
        path.write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError, "different selected inputs"):
            review.validate_saved_review(path, **self.options())

    def test_evidence_plan_passes_confirmation_path_to_scientific_audit(self):
        steps = review.plan(self.root / "reports", offline=False, **self.options())
        audit = next(step for step in steps if step.group == "audit")
        self.assertEqual(audit.argv[audit.argv.index("--confirmation-campaign") + 1], str(self.campaign))

    def test_evidence_plan_passes_campaigns_to_canonical_availability_check(self):
        budget = self.root / "budget"
        for budget_path, confirmation_path in ((None, None), (budget, None),
                                               (None, self.campaign), (budget, self.campaign)):
            with self.subTest(budget=budget_path, confirmation=confirmation_path):
                options = {"index": self.root / "selected/index.json", "records": self.root / "selected/records",
                           "budget_campaign": budget_path, "confirmation_campaign": confirmation_path}
                steps = review.plan(self.root / "reports", offline=False, **options)
                availability = next(step for step in steps if step.argv[0] == "scripts/availability.py")
                expected = ["scripts/availability.py", "--check"]
                if budget_path is not None:
                    expected += ["--budget-campaign", str(budget_path)]
                if confirmation_path is not None:
                    expected += ["--confirmation-campaign", str(confirmation_path)]
                self.assertEqual(availability.argv, expected)
                # The checkout claim retains canonical registrations, independent of selected audit roots.
                self.assertNotIn("--index", availability.argv)
                self.assertNotIn("--records", availability.argv)
                offline = review.plan(self.root / "offline", offline=True, **options)
                offline_availability = next(step for step in offline if step.argv[0] == "scripts/availability.py")
                self.assertEqual(offline_availability.argv, ["scripts/availability.py", "--check"])

    def test_readiness_passes_campaigns_to_canonical_availability_subprocess(self):
        budget = self.root / "budget"
        for budget_path, confirmation_path in ((None, None), (budget, None),
                                               (None, self.campaign), (budget, self.campaign)):
            with self.subTest(budget=budget_path, confirmation=confirmation_path):
                args = SimpleNamespace(review_report=None, index=self.root / "selected/index.json",
                                       records=self.root / "selected/records", budget_campaign=budget_path,
                                       confirmation_campaign=confirmation_path)
                with mock.patch.object(readiness, "_quiet", return_value=0) as quiet:
                    readiness.repository_items(args)
                expected = ["scripts/availability.py", "--check"]
                if budget_path is not None:
                    expected += ["--budget-campaign", str(budget_path)]
                if confirmation_path is not None:
                    expected += ["--confirmation-campaign", str(confirmation_path)]
                self.assertEqual(quiet.call_args_list[0].args[0], expected)
                self.assertEqual(quiet.call_args_list[1].args[0], ["scripts/claims_ledger.py", "--check"])
                self.assertEqual(quiet.call_args_list[2].args[0], ["scripts/manuscript_availability.py", "--check"])

    def test_missing_confirmation_gate_prevents_positive_verdict(self):
        _, invocation, _ = self.saved_fixture()
        data = review.reports(invocation)
        del data["audit"]["confirmation_family"]
        category, reasons = review.verdict([], data, offline=False)
        self.assertEqual(category, "EVIDENCE INCOMPLETE")
        self.assertTrue(any("S14" in reason for reason in reasons))

    def test_offline_confirmation_and_reports_inside_campaign_are_refused(self):
        for options in (["--offline"], []):
            report_root = self.campaign / "report"
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                review.main(["--report-root", str(report_root), "--confirmation-campaign", str(self.campaign),
                             "--dry-run", *options])
            self.assertEqual(exc.exception.code, 2)
            self.assertFalse(report_root.exists())

    def test_readiness_does_not_write_output_into_confirmation_inputs(self):
        output = self.campaign / "readiness.json"
        with contextlib.redirect_stderr(io.StringIO()):
            result = readiness.main(["--confirmation-campaign", str(self.campaign), "--output", str(output)])
        self.assertEqual(result, 2)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
