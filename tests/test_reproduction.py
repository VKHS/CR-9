"""scripts/reproduction.py: identity, progress, accounting and the paper-comparison report."""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts import reproduction as rp  # noqa: E402

FAKE_COMPARE = r'''import json, sys
from pathlib import Path
args = sys.argv[1:]
out = Path(args[args.index("--output") + 1])
rows = [{"section": "Table 1", "quantity": "paired CE effect qD - qC (nats), mean", "run_value": -0.0046,
         "reported_value": -0.0045, "status": "interval"},
        {"section": "Table S15", "quantity": "official score", "run_value": 71.9, "reported_value": 71.9, "status": "exact"}]
comparison = {"dataset": "ogbn-arxiv", "run": args[0], "rows": rows,
              "levels": {"exact": 1, "interval": 1, "outside": 0, "match": 0, "differs": 0, "missing": 0}}
if "--roman-campaign" in args:
    comparison["campaign_summary"] = {"paired_qD_minus_qC_loss": {"mean": -0.0057, "lower": -0.009, "upper": -0.0025},
                                      "admitted_split_count": 10}
out.write_text(json.dumps(comparison))
out.with_suffix(".md").write_text("# comparison\n")
sys.exit(1 if "--fail" in args else 0)
'''


class IdentityAndReferenceTests(unittest.TestCase):
    def test_runtime_fingerprint_matches_the_release_record_and_the_cli(self):
        ident = rp.identity(PROJECT)
        self.assertTrue(ident["matches_release"])
        from scripts._paper_common import runtime_source_fingerprint
        self.assertEqual(ident["runtime_sha256"], runtime_source_fingerprint(PROJECT))

    def test_compute_reference_uses_the_table_s15_increment_for_products(self):
        reference = rp.compute_reference(PROJECT, ["ogbn-products", "ogbn-arxiv"])
        self.assertEqual(reference["ogbn-products"], {"training_GPU_h_per_seed": 4.8, "qualification_increment_GPU_h": 2.28})
        self.assertEqual(reference["ogbn-arxiv"]["qualification_increment_GPU_h"], 1.92)

    def test_dataset_selection_is_canonical_and_ordered(self):
        self.assertEqual(rp.validate_datasets(["Peptides-func", "ogbn-arxiv"]), ["ogbn-arxiv", "Peptides-func"])
        self.assertEqual(rp.validate_datasets([]), list(rp.DATASETS))
        with self.assertRaises(rp.SuiteError):
            rp.validate_datasets(["arxiv"])
        configs = rp.validate_configs(PROJECT, "full", ["Roman-empire"])
        self.assertEqual(configs["Roman-empire"]["data"]["split_index"], 0)


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_run_state_precedence(self):
        run = self.root / "run"
        (run / "launcher").mkdir(parents=True)
        self.assertEqual(rp.run_state(run)["state"], "NOT_STARTED")
        (run / "run_state.json").write_text(json.dumps({"status": "RUNNING", "last_complete_stage": "train"}))
        self.assertEqual(rp.run_state(run)["state"], "IN_PROGRESS")
        (run / "interruption.json").write_text("{}")
        self.assertEqual(rp.run_state(run)["state"], "CHECKPOINTED")
        (run / "failure_summary.json").write_text(json.dumps({"type": "ValueError", "reason": "bad"}))
        self.assertEqual(rp.run_state(run)["state"], "FAILED")
        (run / "launcher/terminal.json").write_text(json.dumps({"status": "INCOMPLETE", "failure_class": "TIME_BUDGET", "reason": "x"}))
        self.assertEqual(rp.run_state(run)["detail"], "TIME_BUDGET: x")
        (run / "run_state.json").write_text(json.dumps({"status": "COMPLETE", "last_complete_stage": "evaluate"}))
        self.assertEqual(rp.run_state(run)["state"], "COMPLETE")

    def test_scheduler_queries_and_accounting(self):
        suite = self.root / "suite"
        (suite / "000_MANIFEST").mkdir(parents=True)
        (suite / "000_MANIFEST/job_ids.tsv").write_text(
            "role\tkey\tjob_id\tsubmitted_utc\nsetup\t00_setup\t11\tx\nrun\t01_arxiv\t12\tx\nrun\t04_roman_empire\t13\tx\n"
            "report\t99_report\t14\tx\ncontinuation\t01_arxiv\t15\tx\n")
        squeue = "12 COMPLETING\n13_[3-9] PENDING\n13_2 RUNNING\n14 RUNNING\n99 RUNNING\n"
        self.assertEqual(rp.parse_squeue(squeue)["13"], "RUNNING")
        self.assertEqual(rp.active_jobs(suite, squeue), ["12", "13"])
        self.assertEqual(rp.active_jobs(suite, squeue, exclude_ids=("12",)), ["13"])
        sacct = ("JobIDRaw|JobID|JobName|ElapsedRaw|AllocTRES\n"
                 "12|12|cmr0001_01_arxiv|3600|cpu=16,gres/gpu=1\n"
                 "13_0|13_0|cmr0001_04_roman_empire|1800|cpu=16,gres/gpu:gh200=2\n"
                 "77|77|unrelated|99999|gres/gpu=4\n"
                 "11|11|cmr0001_setup|360|cpu=16\n")
        self.assertEqual(rp.gpu_hours_from_sacct(sacct, "CMR_0001"),
                         {"01_arxiv": 1.0, "04_roman_empire": 1.0, "setup": 0.0})
        self.assertEqual(rp.gpu_hours_from_sacct("", "CMR_0001"), {})


class SuiteAndReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cm report ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.saved = dict(os.environ)
        os.environ.update(CM_ACCOUNT="private-allocation", CM_PARTITION="gpu", CM_DATASET="ignored")
        self.addCleanup(self._restore)

    def _restore(self):
        os.environ.clear()
        os.environ.update(self.saved)

    def create(self, profile, datasets):
        return rp.create_suite(outputs=self.root / "outputs", state_dir=self.root / "state", source=PROJECT,
                               profile=profile, data_root=self.root / "data", runtime_shell=PROJECT / "run_cross_morse.sh",
                               datasets=datasets)

    def complete(self, run_dir: Path, example="arxiv"):
        source = PROJECT / "examples/runs" / example
        for name in ("run_state.json", "config.json", "environment.json"):
            shutil.copy2(source / name, run_dir / name)
        shutil.copytree(source / "final", run_dir / "final", dirs_exist_ok=True)
        (run_dir / "train").mkdir(exist_ok=True)
        (run_dir / "train/plant_seed1_best.pt").write_bytes(b"checkpoint")

    def test_suite_creation_numbers_freezes_and_records_settings(self):
        first = self.create("smoke", ["Peptides-func"])
        second = self.create("smoke", ["ogbn-arxiv"])
        self.assertEqual((first.name, second.name), ("CROSS_MORSE_REPRODUCTION_0001", "CROSS_MORSE_REPRODUCTION_0002"))
        manifest = rp.load_suite(second)
        self.assertEqual(manifest["settings"]["CM_ACCOUNT"], "private-allocation")
        self.assertNotIn("CM_DATASET", manifest["settings"])
        self.assertTrue((second / "runs/01_arxiv/launcher/source_sha256.json").is_file())
        self.assertEqual((self.root / "outputs/CROSS_MORSE_REPRODUCTION_LATEST").resolve(), second.resolve())
        self.assertIn("export CM_ACCOUNT=private-allocation", (second / "000_MANIFEST/suite.env").read_text())
        with self.assertRaises(rp.SuiteError):
            rp.create_suite(outputs=PROJECT / "outputs-inside-source", state_dir=self.root / "state", source=PROJECT,
                            profile="smoke", data_root=self.root / "data", runtime_shell=PROJECT / "run_cross_morse.sh",
                            datasets=["ogbn-arxiv"])
        self.assertFalse((PROJECT / "outputs-inside-source").exists())

    def test_full_profile_report_places_runs_beside_the_paper(self):
        suite = self.create("full", ["ogbn-arxiv", "Roman-empire", "Peptides-func"])
        manifest = rp.load_suite(suite)
        arxiv, roman, peptides = (Path(run["path"]) for run in manifest["runs"])
        self.complete(arxiv)
        for index in range(10):
            split = roman / f"split_{index:02d}"
            split.mkdir()
            self.complete(split)
        (peptides / "launcher").mkdir(exist_ok=True)
        (peptides / "launcher/terminal.json").write_text(json.dumps({"status": "FAILED", "failure_class": "RUNTIME", "reason": "rc 1"}))
        for run_dir in (arxiv, roman):
            (run_dir / "source/scripts/compare_with_manuscript.py").write_text(FAKE_COMPARE)
        result = rp.build_report(suite, sacct_text="JobIDRaw|JobName|ElapsedRaw|AllocTRES\n1|cmr0001_01_arxiv|7200|gres/gpu=1\n")
        self.assertEqual((result["complete_runs"], result["planned_runs"]), (2, 3))
        report = (suite / "REPRODUCTION_REPORT.md").read_text()
        self.assertIn("Regenerated results beside the paper (main Table 1)", report)
        self.assertIn("| ogbn-arxiv | learned_stop, E=2 (q_C returned) / radius, E=4 |", report)
        self.assertIn("| -0.00450 [-0.00750, -0.00150] | interval | exact 1, interval 1 |", report)
        self.assertIn("| Roman-empire | ten-split campaign / MPC-RH, E=4 | -0.00570 [-0.00900, -0.00250] |", report)
        self.assertIn("| Peptides-func | not complete (FAILED) / radius, E=4 (evaluated arm) |", report)
        self.assertIn("GPU-hours (Slurm accounting) | 2.0", report)
        status = json.loads((suite / "SUITE_STATUS.json").read_text())
        self.assertEqual(status["runs"][0]["comparison"]["levels"]["interval"], 1)
        self.assertTrue(Path(status["runs"][0]["comparison"]["markdown"]).is_file())
        with zipfile.ZipFile(result["results_zip"]) as archive:
            names = archive.namelist()
            self.assertIn("999_REPORT/comparisons/01_arxiv.json", names)
            self.assertIn("runs/04_roman_empire/split_09/final/results_summary.json", names)
            self.assertFalse([name for name in names if name.endswith(".pt")])
            self.assertNotIn("private-allocation", archive.read("000_MANIFEST/suite.json").decode())

    def test_the_report_rebuilds_the_paper_tables_from_the_suite(self):
        suite = self.create("smoke", ["ogbn-arxiv", "Peptides-func"])
        for run, example in zip(rp.load_suite(suite)["runs"], ("arxiv", "peptides_func")):
            shutil.copytree(PROJECT / "examples/runs" / example, run["path"], dirs_exist_ok=True)
        result = rp.build_report(suite)
        self.assertEqual(result["complete_runs"], 2)
        self.assertIn("T1", result["paper_tables"]["tables"])
        report = (suite / "REPRODUCTION_REPORT.md").read_text()
        self.assertIn("## Paper tables rebuilt from this suite", report)
        self.assertIn("### Policy and deployment ledger (main Table 1)", report)
        self.assertIn("### Gain and cost ledger (main Table 2)", report)
        self.assertTrue((suite / "999_REPORT/paper/tables/T1.tex").is_file())
        self.assertEqual(result["paper_tables"]["manifests"], ["arxiv.json", "peptides_func.json"])
        self.assertIn("records the data-role membership, seeds and environment", report)
        self.assertIn("| Peptides-func | phase, E=2 / phase, E=2 | False |", report)
        tools = json.loads((suite / "000_MANIFEST/source_sha256.json").read_text())
        self.assertIn("paper/build/evidence.py", tools)
        self.assertNotIn("configs/arrhenius.env.example", tools)

    def test_a_refused_comparison_is_reported_not_hidden(self):
        suite = self.create("full", ["ogbn-arxiv"])
        run_dir = Path(rp.load_suite(suite)["runs"][0]["path"])
        self.complete(run_dir)
        result = rp.build_report(suite)
        comparison = result["runs"][0]["comparison"]
        self.assertIsNone(comparison["json"])
        self.assertEqual(comparison["error"], "comparison refused: EvidenceError: Eight distinct sealed lifecycle stages are required")
        self.assertEqual(comparison["exit_code"], 2)
        self.assertIn("comparison refused: EvidenceError", (suite / "REPRODUCTION_REPORT.md").read_text())

    def test_status_cli_reads_the_latest_suite(self):
        suite = self.create("pilot", ["COCO-SP"])
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            self.assertEqual(rp.main(["status", str(suite)]), 0)
        self.assertIn("06_coco_sp", printed.getvalue())
        status = rp.suite_status(suite, "")
        self.assertEqual(status["runs"][0]["state"], "NOT_STARTED")
        self.assertEqual(status["planned"], 1)


if __name__ == "__main__":
    unittest.main()
