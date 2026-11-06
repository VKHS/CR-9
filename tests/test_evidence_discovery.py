"""Discovery safety and pipeline handoff; fixtures are never benchmark results.

The positive fixture uses the bundled software example's complete byte structure
with explicitly resealed test-only scope metadata. It tests candidate discovery,
not the truth of a benchmark declaration or the later scientific preflight.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import unittest
from unittest import mock

from cross_morse import provenance as pv
from scripts import evidence_discovery as discovery
from scripts import evidence_pipeline as pipeline

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=1) + "\n", encoding="utf-8")


def reseal_fixture(root):
    """Reseal only a temporary test fixture, never a submitted experiment."""
    config = json.loads((root / "config.json").read_text())
    config_hash = pv._json_hash(config)
    cache = {}
    def artifact(relative):
        if relative not in cache:
            cache[relative] = pv._artifact(root, relative)
        return cache[relative]
    for stage in pv.STAGES:
        path = pv.stage_manifest_path(root, stage)
        row = json.loads(path.read_text())
        row.update(dataset=config["dataset"], config_sha256=config_hash,
                   claim_scope="Unit-test structural fixture, not scientific evidence")
        row["inputs"] = [artifact(value["path"]) for value in row["inputs"]]
        row["outputs"] = [artifact(value["path"]) for value in row["outputs"]]
        if row["parent_manifest"] is not None:
            row["parent_manifest"] = pv._artifact(root, row["parent_manifest"]["path"])
        write_json(path, row)
        marker = root / "stages" / f"{stage}.json"
        if marker.is_file():
            value = json.loads(marker.read_text())
            value.update(manifest_sha256=pv.sha256_file(path), config_sha256=config_hash)
            write_json(marker, value)
    state = json.loads((root / "run_state.json").read_text())
    state.update(dataset=config["dataset"], profile="full", scientific_run=True, config_sha256=config_hash)
    state["task"]["dataset"] = config["dataset"]
    write_json(root / "run_state.json", state)


def official_scope_fixture(root, *, dataset="ogbn-arxiv", split=None, source=None):
    shutil.copytree(source or ROOT / "examples/runs/arxiv", root)
    for relative in ("config.json", "prepared/metadata.json", "final/results_summary.json"):
        path = root / relative
        value = json.loads(path.read_text())
        value.update(dataset=dataset, benchmark_dataset=dataset, scientific_run=True)
        if relative != "prepared/metadata.json":
            value["profile"] = "full"
        else:
            value["is_benchmark_data"] = True
            value["source"] = "unit-test fixture; not original benchmark measurements"
        if relative == "config.json" and split is not None:
            value["data"]["split_index"] = split
        if "task" in value:
            value["task"]["dataset"] = dataset
        write_json(path, value)
    reseal_fixture(root)
    return root


class EvidenceDiscoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_directory = tempfile.TemporaryDirectory(prefix="cm-discovery-fixture-")
        cls.base = official_scope_fixture(Path(cls.base_directory.name) / "base")

    @classmethod
    def tearDownClass(cls):
        cls.base_directory.cleanup()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="cm-discovery-test-")
        self.addCleanup(self.directory.cleanup)
        self.work = Path(self.directory.name)
        self.source = self.work / "export"
        self.source.mkdir()

    def run_fixture(self, name="run"):
        return Path(shutil.copytree(self.base, self.source / name))

    def row(self, report, dataset="ogbn-arxiv"):
        return next(row for row in report["datasets"] if row["dataset"] == dataset)

    def test_unique_candidate_is_deterministic_and_keeps_historical_identity(self):
        run = self.run_fixture()
        before = {p.relative_to(run).as_posix(): pv.sha256_file(p) for p in run.rglob("*") if p.is_file()}
        report = discovery.discover(self.source)
        self.assertEqual(report, discovery.discover(self.source))
        self.assertEqual(report["status"], "CANDIDATES_FOUND")
        self.assertEqual(self.row(report)["proposed_run_root"], str(run))
        candidate = report["candidates"][0]
        state = json.loads((run / "run_state.json").read_text())
        self.assertEqual(candidate["identities"]["source_sha256"], state["source_sha256"])
        self.assertFalse(candidate["source_resolution"]["run_artifacts_rewritten"])
        self.assertFalse(candidate["verification_scope"]["checkpoint_deserialization"])
        self.assertFalse(candidate["verification_scope"]["statistical_replay"])
        from scripts.evidence_preflight import stage_content_sha256
        self.assertEqual(candidate["sealed_content_sha256"], stage_content_sha256(run))
        self.assertEqual(before, {p.relative_to(run).as_posix(): pv.sha256_file(p)
                                  for p in run.rglob("*") if p.is_file()})

    def test_identical_duplicate_runs_are_ambiguous_not_ranked_or_deduplicated(self):
        self.run_fixture("a-old")
        self.run_fixture("z-new")
        report = discovery.discover(self.source)
        self.assertEqual(report["status"], "REVIEW_REQUIRED")
        self.assertEqual(self.row(report)["status"], "AMBIGUOUS")
        plan = discovery.proposed_plan(report, self.work / "deposits")
        self.assertIsNone(plan["datasets"]["ogbn-arxiv"]["run_root"])
        self.assertEqual(len(self.row(report)["candidates"]), 2)

    def test_synthetic_and_pilot_are_rejected_before_expensive_inventory(self):
        for name, dataset, profile in (("synthetic", "synthetic", "full"), ("pilot", "ogbn-arxiv", "pilot")):
            root = self.source / name
            write_json(root / "run_state.json", {"status": "COMPLETE"})
            write_json(root / "config.json", {"dataset": dataset, "benchmark_dataset": "ogbn-arxiv",
                                               "profile": profile, "scientific_run": True})
        with mock.patch("scripts.artifact_inventory.build_inventory", side_effect=AssertionError("must not inventory rejected scope")):
            report = discovery.discover(self.source)
        self.assertEqual(report["status"], "NO_CANDIDATES")
        self.assertEqual([row["status"] for row in report["candidates"]], ["REJECTED", "REJECTED"])
        reasons = str([row["reasons"] for row in report["candidates"]])
        self.assertIn("pilot", reasons)
        self.assertIn("synthetic", reasons)

    def test_complete_bookkeeping_cannot_replace_a_missing_evaluation_seal(self):
        run = self.run_fixture()
        (run / "provenance/stages/07_evaluate.json").unlink()
        report = discovery.discover(self.source)
        self.assertEqual(report["status"], "NO_CANDIDATES")
        self.assertIn("07_evaluate.json", str(report["candidates"][0]["reasons"]))

    def test_bare_configuration_is_reported_without_hiding_nested_runs(self):
        wrapper = self.source / "wrapper"
        write_json(wrapper / "config.json", {"dataset": "ogbn-arxiv", "profile": "full", "scientific_run": True})
        run = Path(shutil.copytree(self.base, wrapper / "actual-run"))
        report = discovery.discover(self.source)
        self.assertEqual(report["candidates"][0]["status"], "REJECTED")
        self.assertIn("run_state.json", str(report["candidates"][0]["reasons"]))
        self.assertEqual(self.row(report)["proposed_run_root"], str(run))

    def test_changed_sealed_bytes_are_rejected_without_repair(self):
        run = self.run_fixture()
        summary = run / "final/results_summary.json"
        changed = summary.read_bytes() + b" "
        summary.write_bytes(changed)
        report = discovery.discover(self.source)
        self.assertEqual(report["candidates"][0]["status"], "REJECTED")
        self.assertIn("hash changed", str(report["candidates"][0]["reasons"]))
        self.assertEqual(summary.read_bytes(), changed)

    def test_missing_checkpoint_is_not_an_analysis_only_candidate(self):
        run = self.run_fixture()
        (run / "train/plant_seed1_best.pt").unlink()
        report = discovery.discover(self.source)
        self.assertEqual(report["candidates"][0]["status"], "REJECTED")
        self.assertIn("plant_seed1_best.pt", str(report["candidates"][0]["reasons"]))

    def test_malformed_npz_does_not_abort_valid_neighbor_or_erase_diagnostics(self):
        malformed = self.run_fixture("a-malformed")
        ancillary = malformed / "ancillary.npz"
        ancillary.write_bytes(b"not a ZIP archive")
        valid = self.run_fixture("z-valid")
        output, plan = self.work / "report.json", self.work / "plan.json"
        with contextlib.redirect_stdout(io.StringIO()):
            code = pipeline.main(["discover", "--root", str(self.source), "--output", str(output),
                                  "--plan-output", str(plan), "--deposits-dir", str(self.work / "deposits")])
        self.assertEqual(code, 0)
        report = json.loads(output.read_text())
        candidates = {row["path"]: row for row in report["candidates"]}
        self.assertEqual(candidates[str(malformed)]["status"], "REJECTED")
        self.assertIn("BadZipFile", str(candidates[str(malformed)]["reasons"]))
        self.assertEqual(candidates[str(valid)]["status"], "CANDIDATE")
        self.assertEqual(self.row(report)["proposed_run_root"], str(valid))
        self.assertEqual(pipeline.load_plan(plan)["datasets"]["ogbn-arxiv"]["run_root"], str(valid))
        self.assertIn("BadZipFile", output.with_suffix(".md").read_text())
        self.assertEqual(ancillary.read_bytes(), b"not a ZIP archive")

    def test_malformed_descriptor_does_not_abort_valid_neighbor(self):
        malformed = self.run_fixture("a-malformed")
        manifest = malformed / "provenance/stages/00_prepare.json"
        descriptor = json.loads(manifest.read_text())
        descriptor["outputs"][0] = None
        write_json(manifest, descriptor)
        altered = manifest.read_bytes()
        valid = self.run_fixture("z-valid")
        report = discovery.discover(self.source)
        candidates = {row["path"]: row for row in report["candidates"]}
        self.assertEqual(report["status"], "CANDIDATES_FOUND")
        self.assertEqual(candidates[str(malformed)]["status"], "REJECTED")
        self.assertIn("Malformed prepare.outputs[0] artifact descriptor", str(candidates[str(malformed)]["reasons"]))
        self.assertEqual(candidates[str(valid)]["status"], "CANDIDATE")
        self.assertEqual(self.row(report)["proposed_run_root"], str(valid))
        self.assertEqual(manifest.read_bytes(), altered)

    def test_roman_requires_all_ten_splits_as_one_campaign(self):
        campaign = self.source / "roman"
        (campaign / "split_00").mkdir(parents=True)
        report = discovery.discover(self.source)
        self.assertEqual(len(report["candidates"]), 1)
        self.assertEqual(report["candidates"][0]["kind"], "campaign")
        self.assertEqual(report["candidates"][0]["status"], "REJECTED")
        self.assertIn("all ten", str(report["candidates"][0]["reasons"]))
        self.assertIsNone(self.row(report, "Roman-empire")["proposed_run_root"])

    def test_roman_common_source_and_configuration_are_required(self):
        campaign = self.source / "roman"
        campaign.mkdir()
        for split in range(10):
            official_scope_fixture(campaign / f"split_{split:02d}", dataset="Roman-empire", split=split, source=self.base)
        report = discovery.discover(self.source)
        self.assertEqual(self.row(report, "Roman-empire")["status"], "UNIQUE_CANDIDATE", report["candidates"])
        self.assertEqual(len(report["candidates"]), 1)
        self.assertEqual(len(report["candidates"][0]["identities"]), 10)
        changed = campaign / "split_09"
        config = json.loads((changed / "config.json").read_text())
        config["seed"] += 1
        write_json(changed / "config.json", config)
        reseal_fixture(changed)
        report = discovery.discover(self.source)
        self.assertEqual(report["candidates"][0]["status"], "REJECTED")
        self.assertIn("share source and non-split configuration", str(report["candidates"][0]["reasons"]))

    def test_limits_block_every_proposal_instead_of_assuming_uniqueness(self):
        self.run_fixture("a-complete")
        (self.source / "z-wrapper/hidden").mkdir(parents=True)
        for limits in ({"max_depth": 1}, {"max_directories": 1}):
            with self.subTest(limits=limits):
                report = discovery.discover(self.source, **limits)
                self.assertFalse(report["scan_complete"])
                self.assertEqual(report["status"], "INCOMPLETE_SCAN")
                self.assertTrue(all(row["proposed_run_root"] is None for row in report["datasets"]))

    def test_symlink_directory_is_never_followed(self):
        (self.source / "linked-run").symlink_to(self.base, target_is_directory=True)
        with mock.patch.object(discovery, "inspect_candidate", side_effect=AssertionError("must not follow link")):
            report = discovery.discover(self.source)
        self.assertEqual(report["status"], "NO_CANDIDATES")
        self.assertEqual(len(report["excluded"]), 1)
        with self.assertRaises(discovery.DiscoveryError):
            discovery.discover(self.source / "linked-run")
        with mock.patch.object(discovery, "_object", side_effect=AssertionError("must not inspect a rejected symlink")):
            rejected = discovery.inspect_candidate(self.source / "linked-run", "run")
        self.assertEqual(rejected["status"], "REJECTED")
        self.assertIn("symlink", str(rejected["reasons"]))

    def test_unreadable_branch_blocks_uniqueness(self):
        original = discovery.os.walk
        def unreadable(root, **kwargs):
            error = PermissionError("injected inaccessible export")
            error.filename = str(self.source / "unreadable")
            kwargs["onerror"](error)
            return original(root, **kwargs)
        self.run_fixture()
        with mock.patch.object(discovery.os, "walk", side_effect=unreadable):
            report = discovery.discover(self.source)
        self.assertEqual(report["status"], "INCOMPLETE_SCAN")
        self.assertIsNone(self.row(report)["proposed_run_root"])

    def test_private_plan_loads_without_attestation_doi_or_registration(self):
        self.run_fixture()
        report = discovery.discover(self.source)
        output, plan_path = self.work / "reports/discovery.json", self.work / "plan.json"
        deposits = self.work / "deposits"
        discovery.write_outputs(report, output, plan_output=plan_path, deposits=deposits)
        plan = pipeline.load_plan(plan_path)
        for field in ("license", "code_doi", "relationship_to_manuscript", "budget_campaign"):
            self.assertIsNone(plan[field])
        self.assertTrue(all(row == {"exporter": None, "mapping": None} for row in plan["record_sets"].values()))
        self.assertEqual(pipeline.selected(plan, ["ogbn-arxiv"], need_root=True), ["ogbn-arxiv"])
        self.assertFalse(deposits.exists())
        for path in (output, output.with_suffix(".md"), plan_path):
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_outputs_never_overwrite_or_write_into_the_source(self):
        report = discovery.discover(self.source)
        existing = self.work / "plan.json"
        existing.write_text("keep this plan")
        output = self.work / "report.json"
        with self.assertRaisesRegex(discovery.DiscoveryError, "overwrite"):
            discovery.write_outputs(report, output, plan_output=existing, deposits=self.work / "deposits")
        self.assertEqual(existing.read_text(), "keep this plan")
        self.assertFalse(output.exists())
        with self.assertRaisesRegex(discovery.DiscoveryError, "outside"):
            discovery.write_outputs(report, self.source / "report.json")
        with self.assertRaisesRegex(discovery.DiscoveryError, "outside"):
            discovery.proposed_plan(report, self.source / "deposits")
        with self.assertRaisesRegex(discovery.DiscoveryError, "distinct"):
            discovery.write_outputs(report, output, plan_output=output, deposits=self.work / "deposits")

    def test_cli_writes_diagnostic_reports_and_preserves_nonzero_status(self):
        output = self.work / "report.json"
        plan = self.work / "plan.json"
        with contextlib.redirect_stdout(io.StringIO()):
            code = pipeline.main(["discover", "--root", str(self.source), "--output", str(output),
                                  "--plan-output", str(plan), "--deposits-dir", str(self.work / "deposits")])
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(output.read_text())["status"], "NO_CANDIDATES")
        self.assertTrue(all(row["run_root"] is None for row in json.loads(plan.read_text())["datasets"].values()))
        with contextlib.redirect_stderr(io.StringIO()):
            code = pipeline.main(["discover", "--root", str(self.source), "--output", str(self.work / "other.json"),
                                  "--plan-output", str(self.work / "other-plan.json")])
        self.assertEqual(code, 2)
        self.assertFalse((self.work / "other.json").exists())


if __name__ == "__main__":
    unittest.main()
