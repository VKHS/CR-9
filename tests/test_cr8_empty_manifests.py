"""An explicitly empty official inventory is a status, never executed evidence."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from paper.build.evidence import EvidenceError
from scripts.build_paper import from_index
from scripts.run_manifests import main, manifest_runs_from_index


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "cross-morse-runs-index-1"


class EmptyOfficialManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.index = self.root / "index.json"
        self.output = self.root / "manifests"
        self.write_index({"schema": SCHEMA, "runs": []})

    def write_index(self, value):
        self.index.write_text(json.dumps(value), encoding="utf-8")

    def invoke(self, *extra):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(["runs", "--index", str(self.index), "--label", "official",
                         "--output-root", str(self.output), *extra])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_explicit_empty_index_reports_status_without_fabricating_manifests(self):
        for mode in ([], ["--check"], ["--with-indices"]):
            with self.subTest(mode=mode):
                code, stdout, stderr = self.invoke("--allow-empty", *mode)
                self.assertEqual(code, 0, stderr)
                self.assertIn("0 official runs; no executed manifests", stdout)
                self.assertFalse(self.output.exists())
        (self.output / "official").mkdir(parents=True)
        code, _, stderr = self.invoke("--allow-empty", "--check")
        self.assertEqual(code, 0, stderr)
        self.assertEqual(list((self.output / "official").iterdir()), [])

    def test_default_and_paper_index_reader_remain_strict(self):
        code, stdout, stderr = self.invoke()
        self.assertEqual(code, 2)
        self.assertIn("no verified runs listed", stderr)
        self.assertNotIn("0 official runs", stdout)
        with self.assertRaisesRegex(EvidenceError, "no verified runs listed"):
            from_index(self.index)

    def test_malformed_index_or_entry_never_becomes_empty_success(self):
        values = [[], {}, {"schema": "unknown", "runs": []}, {"schema": SCHEMA},
                  {"schema": SCHEMA, "runs": None}, {"schema": SCHEMA, "runs": {}},
                  *[{"schema": SCHEMA, "runs": [entry]} for entry in (
                      None, "run", {}, {"verified": "true", "path": "run"},
                      {"verified": 1, "path": "run"}, {"verified": True, "path": 7},
                      {"verified": True, "path": " "},
                      {"verified": True, "path": "run", "stem": "../run"})]]
        for value in values:
            with self.subTest(value=value):
                self.write_index(value)
                code, stdout, stderr = self.invoke("--allow-empty", "--check")
                self.assertEqual(code, 2, (stdout, stderr))
                self.assertIn("ERROR:", stderr)
                self.assertNotIn("0 official runs", stdout)
                self.assertFalse(self.output.exists())

    def test_nonempty_missing_or_unverified_runs_are_not_allowed_empty(self):
        for verified in (True, False):
            with self.subTest(verified=verified):
                self.write_index({"schema": SCHEMA, "runs": [{"verified": verified, "path": "missing"}]})
                code, stdout, stderr = self.invoke("--allow-empty")
                self.assertEqual(code, 2)
                self.assertIn("no completed run or campaign" if verified else "no verified runs listed", stderr)
                self.assertNotIn("0 official runs", stdout)

    def test_valid_nonempty_index_still_uses_completed_run_discovery(self):
        final = self.root / "existing" / "final"
        final.mkdir(parents=True)
        (final / "results_summary.json").write_text("{}", encoding="utf-8")
        self.write_index({"schema": SCHEMA, "runs": [{"verified": True, "path": "existing", "stem": "run"}]})
        self.assertEqual(manifest_runs_from_index(self.index, allow_empty=True),
                         [("run", final.parent.resolve())])
        self.write_index({"schema": SCHEMA, "runs": [
            {"verified": True, "path": "existing", "stem": "run"},
            {"verified": True, "path": "missing", "stem": "missing"}]})
        with self.assertRaisesRegex(EvidenceError, "no completed run or campaign"):
            manifest_runs_from_index(self.index, allow_empty=True)

    def test_nonempty_index_copies_real_membership_arrays_into_new_output_directory(self):
        example = ROOT / "examples" / "runs" / "arxiv"
        self.write_index({"schema": SCHEMA, "runs": [
            {"verified": True, "path": str(example), "stem": "arxiv"}]})
        code, stdout, stderr = self.invoke("--allow-empty", "--with-indices")
        self.assertEqual(code, 0, stderr)
        self.assertIn("wrote 1 run manifests", stdout)
        official = self.output / "official"
        self.assertEqual((official / "arxiv_splits.npz").read_bytes(),
                         (example / "prepared" / "splits.npz").read_bytes())
        manifest = json.loads((official / "arxiv.json").read_text())
        self.assertIn("arxiv_splits.npz", manifest["index_arrays_copied"])
        code, _, stderr = self.invoke("--allow-empty", "--check")
        self.assertEqual(code, 0, stderr)

    def test_empty_index_rejects_and_preserves_stale_manifests_and_arrays(self):
        official = self.output / "official"
        (official / "nested").mkdir(parents=True)
        stale = {official / "arxiv.json": b'{"old": true}',
                 official / "nested" / "arxiv_splits.npz": b"old membership bytes"}
        for path, content in stale.items():
            path.write_bytes(content)
        for mode in ([], ["--check"]):
            with self.subTest(mode=mode):
                code, stdout, stderr = self.invoke("--allow-empty", *mode)
                self.assertEqual(code, 2)
                self.assertIn("stale executed manifest files", stderr)
                self.assertNotIn("0 official runs", stdout)
                for path, content in stale.items():
                    self.assertEqual(path.read_bytes(), content)

    @unittest.skipUnless(shutil.which("make"), "GNU Make is not available")
    def test_only_inventory_make_targets_allow_empty_official_index(self):
        for target in ("manifests", "manifests-check", "ingest-run", "paper"):
            with self.subTest(target=target):
                result = subprocess.run(["make", "--dry-run", "--no-print-directory", target,
                                         f"RUNS_INDEX={self.index}", "ARCHIVE=fixture.tar.gz"],
                                        cwd=ROOT, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.count("--allow-empty"),
                                 int(target in ("manifests", "manifests-check")))


if __name__ == "__main__":
    unittest.main()
