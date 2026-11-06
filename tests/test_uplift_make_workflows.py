"""Exercise publication switches and reviewer paths without network or Docker."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("make"), "GNU Make is required")
class UpliftMakeWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / "checkout with spaces"
        self.project.mkdir()
        shutil.copyfile(PROJECT / "Makefile", self.project / "Makefile")
        self.capture = self.root / "calls.jsonl"
        self.interpreter = self.root / "recording python"
        self.interpreter.write_text(f"#!{sys.executable}\nimport json, os, sys\n"
                                    "with open(os.environ['CM_MAKE_CAPTURE'], 'a') as stream:\n"
                                    "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n")
        self.interpreter.chmod(0o755)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        shutil.copyfile(self.interpreter, self.bin / "docker")
        (self.bin / "docker").chmod(0o755)

    def invoke(self, target, *, succeeds=True, **variables):
        self.capture.unlink(missing_ok=True)
        result = subprocess.run(["make", "--no-print-directory", target, f"PYTHON={self.interpreter}",
                                 *(f"{key}={value}" for key, value in variables.items())],
                                cwd=self.project, text=True, capture_output=True, timeout=30,
                                env=dict(os.environ, CM_MAKE_CAPTURE=str(self.capture),
                                         PATH=str(self.bin) + os.pathsep + os.environ.get("PATH", "")))
        if succeeds:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = [json.loads(line) for line in self.capture.read_text().splitlines()] if self.capture.exists() else []
        return result, calls

    def test_zero_flags_never_request_publication_or_sandbox(self):
        for target in ("deposit", "deposit-code"):
            with self.subTest(target=target):
                _, calls = self.invoke(target, ARCHIVE="/deposits/archive.tar.gz", PUBLISH=0, SANDBOX=0)
                self.assertEqual(len(calls), 1)
                self.assertNotIn("--publish", calls[0])
                self.assertNotIn("--sandbox", calls[0])

    def test_only_explicit_one_requests_publication(self):
        for target in ("deposit", "deposit-code"):
            with self.subTest(target=target):
                _, calls = self.invoke(target, ARCHIVE="/deposits/archive.tar.gz", PUBLISH=1, SANDBOX=1)
                self.assertIn("--publish", calls[0])
                self.assertIn("--sandbox", calls[0])
                _, calls = self.invoke(target, ARCHIVE="/deposits/archive.tar.gz")
                self.assertNotIn("--publish", calls[0])

    def test_invalid_publication_switches_fail_before_any_client_is_called(self):
        for variable, value in (("PUBLISH", "true"), ("PUBLISH", "0 1"), ("SANDBOX", "false")):
            for target in ("deposit", "deposit-code"):
                with self.subTest(target=target, variable=variable, value=value):
                    result, calls = self.invoke(target, succeeds=False, ARCHIVE="/deposits/archive.tar.gz",
                                                **{variable: value})
                    self.assertIn(f"{variable} must be 0 or 1", result.stderr)
                    self.assertEqual(calls, [])

    def test_archive_license_is_explicit_and_not_applied_to_the_software_snapshot(self):
        for target in ("deposit-plan", "deposit"):
            with self.subTest(target=target):
                _, calls = self.invoke(target, ARCHIVE="/deposits/archive.tar.gz", DEPOSIT_LICENSE="CC-BY-4.0")
                self.assertEqual(calls[0][calls[0].index("--license") + 1], "CC-BY-4.0")
        _, calls = self.invoke("deposit-code", DEPOSIT_LICENSE="CC-BY-4.0")
        self.assertNotIn("--license", calls[0])

    def test_reviewer_fetch_does_not_rewrite_author_manifests_or_statements(self):
        _, calls = self.invoke("fetch-deposits")
        self.assertEqual([call[1] for call in calls], ["scripts/fetch_deposits.py"])

    def test_author_record_index_refreshes_all_generated_evidence_statements(self):
        _, calls = self.invoke("records-index")
        self.assertEqual([call[1] for call in calls], ["scripts/records.py", "scripts/availability.py",
                                                    "scripts/claims_ledger.py", "scripts/manuscript_availability.py"])

    def test_review_preserves_evidence_paths_and_zero_switches(self):
        variables = {"BUDGET_CAMPAIGN": self.root / "budget campaign",
                     "MANUSCRIPT_PDF": self.root / "main paper.pdf",
                     "SUPPLEMENTARY_PDF": self.root / "supplementary paper.pdf",
                     "RECORDS_ROOT": self.root / "record sets", "RUNS_INDEX": self.root / "run index.json",
                     "DEPOSITS_DIR": self.root / "verified deposits"}
        _, calls = self.invoke("review", REVIEW_OFFLINE=0, REQUIRE_EVIDENCE=0, **variables)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("--offline", calls[0])
        self.assertNotIn("--require-evidence", calls[0])
        for key, flag in (("BUDGET_CAMPAIGN", "--budget-campaign"), ("MANUSCRIPT_PDF", "--manuscript-pdf"),
                          ("SUPPLEMENTARY_PDF", "--supplementary-pdf"), ("RECORDS_ROOT", "--records"),
                          ("RUNS_INDEX", "--index"), ("DEPOSITS_DIR", "--deposits-dir")):
            self.assertEqual(calls[0][calls[0].index(flag) + 1], str(variables[key]))
        _, calls = self.invoke("review", REVIEW_OFFLINE=1, REQUIRE_EVIDENCE=1)
        self.assertIn("--offline", calls[0])
        self.assertIn("--require-evidence", calls[0])

    def test_scientific_audit_receives_campaign_and_manuscript_sources(self):
        variables = {"BUDGET_CAMPAIGN": self.root / "budget campaign",
                     "MANUSCRIPT_PDF": self.root / "main paper.pdf",
                     "SUPPLEMENTARY_PDF": self.root / "supplementary paper.pdf",
                     "RECORDS_ROOT": self.root / "record sets", "RUNS_INDEX": self.root / "run index.json"}
        _, calls = self.invoke("scientific-audit", **variables)
        self.assertEqual(len(calls), 1)
        for key, flag in (("BUDGET_CAMPAIGN", "--budget-campaign"), ("MANUSCRIPT_PDF", "--manuscript-pdf"),
                          ("SUPPLEMENTARY_PDF", "--supplementary-pdf"), ("RECORDS_ROOT", "--records"),
                          ("RUNS_INDEX", "--index")):
            self.assertEqual(calls[0][calls[0].index(flag) + 1], str(variables[key]))

    def test_decision_advisory_mode_is_explicit(self):
        _, calls = self.invoke("decision-check", DECISION_WARN_ONLY=0)
        self.assertNotIn("--warn-only", calls[0])
        _, calls = self.invoke("decision-check", DECISION_WARN_ONLY=1)
        self.assertIn("--warn-only", calls[0])

    def test_container_verification_runs_full_tests_without_network(self):
        _, calls = self.invoke("docker-verify")
        self.assertEqual(calls, [["build", "-t", "cross-morse:verify", "."],
                                 ["run", "--rm", "--network", "none", "cross-morse:verify", "make", "verify", "test", "PYTHON=python"]])

    def test_container_review_uses_a_writable_host_owned_output_directory(self):
        reports = self.root / "review reports"
        _, calls = self.invoke("docker-review", REPORT_ROOT=reports)
        command = calls[-1]
        self.assertTrue(reports.is_dir())
        self.assertEqual(command[command.index("--network") + 1], "none")
        self.assertEqual(command[command.index("--user") + 1], f"{os.getuid()}:{os.getgid()}")
        self.assertEqual(command[command.index("-v") + 1], f"{reports}:/reports")
        self.assertIn("REVIEW_OFFLINE=1", command)


if __name__ == "__main__":
    unittest.main()
