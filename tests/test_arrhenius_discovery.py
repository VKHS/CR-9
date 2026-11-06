"""Private HPC discovery routing and failure propagation; no scheduler is called."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "hpc/discover_evidence_arrhenius.sh"


@unittest.skipUnless(shutil.which("bash"), "bash is required")
class ArrheniusDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cm-hpc-discovery-")
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.exports = self.work / "exports"
        self.exports.mkdir()
        self.evidence = self.work / "private evidence"
        self.profile = self.work / "profile.env"
        self.profile.write_text("# Trusted empty test profile\n", encoding="utf-8")
        self.source = self.work / "source checkout"
        (self.source / "scripts").mkdir(parents=True)
        (self.source / "scripts/source_fingerprint.py").write_text(
            "import os, sys\n"
            "if os.environ.get('CM_TEST_FINGERPRINT_FAIL'):\n"
            "    raise SystemExit(9)\n"
            "print('a' * 64)\n", encoding="utf-8")
        (self.source / "scripts/evidence_pipeline.py").write_text(
            "import json, os, pathlib, sys\n"
            "pathlib.Path(os.environ['CM_TEST_CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n"
            "status = int(os.environ.get('CM_TEST_STATUS', '0'))\n"
            "if status in (0, 1):\n"
            "    report = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
            "    plan = pathlib.Path(sys.argv[sys.argv.index('--plan-output') + 1])\n"
            "    for path in (report, report.with_suffix('.md'), plan):\n"
            "        with path.open('x') as stream:\n"
            "            stream.write('diagnostic\\n')\n"
            "raise SystemExit(status)\n", encoding="utf-8")
        self.capture = self.work / "arguments.json"

    def environment(self, **changes):
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("CM_", "SLURM_", "SBATCH_"))}
        env.update(CM_SOURCE_ROOT=str(self.source), CM_ENV_FILE=str(self.profile),
                   CM_EVIDENCE_ROOT=str(self.exports), CM_EVIDENCE_DIR=str(self.evidence),
                   CM_PYTHON=sys.executable, CM_TEST_CAPTURE=str(self.capture), SLURM_JOB_ID="12345")
        env.update(changes)
        return env

    def run_script(self, **changes):
        return subprocess.run(["bash", str(SCRIPT)], cwd=self.work, env=self.environment(**changes),
                              capture_output=True, text=True, timeout=90)

    def test_required_environment_is_an_invalid_setup(self):
        result = self.run_script(CM_SOURCE_ROOT="")
        self.assertEqual(result.returncode, 2)
        self.assertIn("CM_SOURCE_ROOT", result.stderr)
        self.assertFalse(self.evidence.exists())

    def test_cli_status_and_limits_are_preserved_and_outputs_are_private(self):
        before = {p.relative_to(self.source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in self.source.rglob("*") if p.is_file()}
        for status in (0, 1, 2, 7):
            with self.subTest(status=status):
                private = self.work / f"evidence-{status}"
                result = self.run_script(CM_EVIDENCE_DIR=str(private), CM_TEST_STATUS=str(status),
                                         CM_DISCOVERY_MAX_DEPTH="9", CM_DISCOVERY_MAX_DIRECTORIES="321")
                self.assertEqual(result.returncode, status, result.stdout + result.stderr)
                args = json.loads(self.capture.read_text())
                self.assertEqual(args[0], "discover")
                for flag, value in (("--root", str(self.exports)), ("--max-depth", "9"),
                                    ("--max-directories", "321"), ("--deposits-dir", str(private / "deposits"))):
                    self.assertEqual(args[args.index(flag) + 1], value)
                self.assertEqual(stat.S_IMODE(private.stat().st_mode), 0o700)
                if status in (0, 1):
                    for path in private.iterdir():
                        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
                    self.assertEqual(len(list(private.iterdir())), 3)
                else:
                    self.assertEqual(list(private.iterdir()), [])
        self.assertEqual(before, {p.relative_to(self.source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in self.source.rglob("*") if p.is_file()})

    def test_unsafe_or_shared_destinations_are_rejected_before_creation(self):
        link = self.work / "source-link"
        link.symlink_to(self.source, target_is_directory=True)
        shared = self.work / "shared-evidence"
        shared.mkdir(mode=0o755)
        shared.chmod(0o755)
        for path in (self.source / "new-private", self.exports / "new-private", link / "new-private", shared):
            with self.subTest(path=path):
                result = self.run_script(CM_EVIDENCE_DIR=str(path))
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertFalse(self.capture.exists())
                if path != shared:
                    self.assertFalse(path.exists())
        result = self.run_script(CM_EVIDENCE_DIR="relative-output")
        self.assertEqual(result.returncode, 2)
        self.assertFalse((self.work / "relative-output").exists())

    def test_fingerprint_failure_does_not_start_discovery_or_create_outputs(self):
        result = self.run_script(CM_TEST_FINGERPRINT_FAIL="1")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("Source identity could not be verified", result.stderr)
        self.assertFalse(self.capture.exists())
        self.assertFalse(self.evidence.exists())

    def test_expected_runtime_identity_is_checked_before_discovery(self):
        result = self.run_script(CM_SOURCE_ROOT=str(ROOT), CM_EXPECT_SOURCE_SHA256="0" * 64)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("MISMATCH", result.stderr)
        self.assertFalse(self.evidence.exists())

    def test_existing_job_reports_are_never_overwritten(self):
        result = self.run_script(CM_TEST_STATUS="1")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        before = {path.name: path.read_bytes() for path in self.evidence.iterdir()}
        self.capture.unlink()
        result = self.run_script()
        self.assertEqual(result.returncode, 2)
        self.assertIn("already exists", result.stderr)
        self.assertFalse(self.capture.exists())
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.evidence.iterdir()})

    def test_bad_limits_and_unsafe_job_identifiers_fail_before_creation(self):
        for changes in ({"CM_DISCOVERY_MAX_DEPTH": "-1"}, {"CM_DISCOVERY_MAX_DIRECTORIES": "0"},
                        {"CM_DISCOVERY_MAX_DIRECTORIES": "unknown"}, {"SLURM_JOB_ID": "../elsewhere"}):
            with self.subTest(changes=changes):
                result = self.run_script(**changes)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertFalse(self.capture.exists())
                self.assertFalse(self.evidence.exists())

    def test_real_empty_export_preserves_reports_and_no_candidate_status(self):
        result = self.run_script(CM_SOURCE_ROOT=str(ROOT))
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        report = json.loads((self.evidence / "discovery-12345.json").read_text())
        plan = json.loads((self.evidence / "plan-12345.json").read_text())
        self.assertEqual(report["status"], "NO_CANDIDATES")
        self.assertTrue(report["scan_complete"])
        self.assertTrue(all(row["run_root"] is None for row in plan["datasets"].values()))
        self.assertIsNone(plan["relationship_to_manuscript"])
        self.assertFalse((self.evidence / "deposits").exists())
        for path in self.evidence.iterdir():
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)


if __name__ == "__main__":
    unittest.main()
