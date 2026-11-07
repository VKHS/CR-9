"""End-to-end output coverage and integrity, without training or downloading data."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from scripts import reconstruct as rc

ROOT = Path(__file__).resolve().parents[1]


class ReconstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.work = Path(cls.temporary.name)
        cls.output = cls.work / "example-build"
        cls.result = rc.reconstruct(cls.output, runs_root=ROOT / "examples/runs",
                                    records_index=ROOT / "examples/records/index.json", label="examples")
        from scripts.archive_run import pack
        from scripts.ingest_run import ingest
        archive = cls.work / "arxiv.tar.gz"
        pack(ROOT / "examples/runs/arxiv", archive, campaign=False)
        cls.registered = cls.work / "registered"
        cls.entry = ingest(archive, doi=None, runs_root=cls.registered,
                           index_path=cls.registered / "index.json", replace=False)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def copied_workspace(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "workspace"
        shutil.copytree(self.output, path)
        return path

    def test_real_build_has_every_display_and_traceable_outputs(self):
        self.assertEqual(len(self.result["displays"]), 90)
        self.assertEqual(len({r["id"] for r in self.result["displays"]}), 90)
        ids = {row["id"] for row in self.result["outputs"]}
        self.assertIn("run:T1", ids)
        self.assertIn("run:F3", ids)
        self.assertIn("record:nominal_time", ids)
        for entry in self.result["outputs"]:
            for name in entry["files"]:
                self.assertEqual(rc.sha256(self.output / name), entry["file_sha256"][name])
        inputs = [row for row in self.result["inputs"] if row["kind"] == "sealed_run"]
        self.assertEqual(len(inputs), 2)
        self.assertTrue(all(len(row["stage_manifests"]) == 8 for row in inputs))
        self.assertIn("scripts/reconstruct.py", self.result["software_sources"])
        self.assertEqual(rc.verify_workspace(self.output), self.result)

    def test_examples_and_partial_panels_never_claim_complete_paper(self):
        self.assertTrue(self.result["synthetic_examples"])
        self.assertFalse(self.result["all_displays_numerically_reconstructed"])
        self.assertFalse(self.result["historical_experiment_identity_verified"])
        rows = {row["id"]: row for row in self.result["displays"]}
        self.assertEqual(rows["main:Table 1"]["coverage"], ["columns"])
        self.assertEqual(rows["main:Figure 2"]["coverage"], ["partial_panels"])
        self.assertFalse(rows["main:Figure 2"]["full_numerical_display"])
        self.assertEqual(rows["supplementary:Figure S5"]["generated_outputs"], [])
        self.assertIn("Synthetic software examples", (self.output / "RECONSTRUCTION_REPORT.md").read_text())

    def test_changed_table_is_detected(self):
        path = self.copied_workspace()
        (path / "runs/tables/T1.csv").write_text("manually changed result\n")
        with self.assertRaisesRegex(rc.ReconstructionError, "artifact changed"):
            rc.verify_workspace(path)

    def test_missing_or_extra_output_is_detected(self):
        for changed in ("missing", "extra"):
            with self.subTest(changed=changed):
                path = self.copied_workspace()
                if changed == "missing":
                    (path / "runs/tables/T1.csv").unlink()
                else:
                    (path / "unregistered-result.csv").write_text("1\n")
                with self.assertRaisesRegex(rc.ReconstructionError, "file set differs"):
                    rc.verify_workspace(path)

    def test_symlinks_and_parent_traversal_are_rejected(self):
        path = self.copied_workspace()
        (path / "linked").symlink_to(self.output, target_is_directory=True)
        with self.assertRaisesRegex(rc.ReconstructionError, "symlinks"):
            rc.verify_workspace(path)
        for name in ("../runs/T1.csv", "/tmp/T1.csv", "runs/../T1.csv", "runs//T1.csv"):
            with self.subTest(name=name), self.assertRaises(rc.ReconstructionError):
                rc.relative_file(self.output, name)

    def test_failed_rebuild_preserves_previous_report(self):
        path = self.copied_workspace()
        before = rc.sha256(path / "FILES.json")
        with mock.patch.object(rc.build_paper, "write_all", side_effect=ValueError("builder failed")):
            with self.assertRaisesRegex(ValueError, "builder failed"):
                rc.reconstruct(path, runs_root=ROOT / "examples/runs", label="examples")
        self.assertEqual(rc.sha256(path / "FILES.json"), before)
        rc.verify_workspace(path)

    def test_modified_input_cannot_be_used_as_a_sealed_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "arxiv"
            shutil.copytree(ROOT / "examples/runs/arxiv", run)
            config = rc.read_json(run / "config.json")
            config["profile"] = "full"
            rc.write_json(run / "config.json", config)
            with self.assertRaisesRegex(ValueError, "Configuration content changed"):
                rc.reconstruct(Path(temporary) / "output", runs_root=run)

    def test_nonmanaged_directory_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "notes"
            output.mkdir()
            note = output / "keep.txt"
            note.write_text("author notes\n")
            with self.assertRaisesRegex(rc.ReconstructionError, "not a managed"):
                rc.reconstruct(output)
            self.assertEqual(note.read_text(), "author notes\n")

    def test_a_bookkeeping_flag_cannot_relabel_synthetic_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "arxiv"
            shutil.copytree(ROOT / "examples/runs/arxiv", run)
            state = rc.read_json(run / "run_state.json")
            state["scientific_run"] = True
            rc.write_json(run / "run_state.json", state)
            with self.assertRaisesRegex(rc.ReconstructionError, "scope differs"):
                rc.reconstruct(Path(temporary) / "output", runs_root=run)

    def test_input_directory_is_never_an_output_directory(self):
        with self.assertRaisesRegex(rc.ReconstructionError, "replace source or input"):
            rc.reconstruct(ROOT)
        with self.assertRaisesRegex(rc.ReconstructionError, "outside the input"):
            rc.reconstruct(ROOT / "examples/runs/reconstruction", runs_root=ROOT / "examples/runs")
        with self.assertRaisesRegex(rc.ReconstructionError, "outside the input evidence"):
            rc.reconstruct(ROOT / "examples/records/reconstruction", records_index=ROOT / "examples/records/index.json")

    def test_empty_index_creates_useful_inventory_without_a_success_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            result = rc.reconstruct(output, index=ROOT / "runs/index.json", records_index=ROOT / "records/index.json")
            self.assertEqual(result["build_status"], "INPUTS_REQUIRED")
            # Only the displays generated from the repository's own sources (protocol and analytical tables).
            self.assertTrue(result["outputs"])
            self.assertEqual({o["input_kind"] for o in result["outputs"]}, {"repository_sources"})
            self.assertEqual(len(result["displays"]), 90)
            self.assertFalse(result["all_displays_numerically_reconstructed"])
            self.assertEqual(rc.verify_workspace(output), result)

    def test_registered_missing_bytes_remain_unavailable(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            index = directory / "index.json"
            rc.write_json(index, {"schema": "cross-morse-runs-index-1", "runs": [self.entry]})
            result = rc.reconstruct(directory / "output", index=index)
            self.assertEqual(len(result["unavailable_inputs"]), 1)
            self.assertEqual(result["build_status"], "INPUTS_REQUIRED")
            self.assertEqual({o["input_kind"] for o in result["outputs"]}, {"repository_sources"})

    def test_duplicate_run_would_not_silently_replace_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            index = directory / "index.json"
            rc.write_json(index, {"schema": "cross-morse-runs-index-1", "runs": [self.entry, self.entry]})
            with self.assertRaisesRegex(rc.ReconstructionError, "Duplicate run-index"):
                rc.reconstruct(directory / "output", index=index)

    def test_registered_archive_roundtrip_is_verified_against_its_declaration(self):
        found, inputs, unavailable = rc._run_inputs(None, self.registered / "index.json")
        self.assertEqual(len(found), 1)
        self.assertEqual(unavailable, [])
        self.assertIn("run_registration", {row["kind"] for row in inputs})
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "registered"
            shutil.copytree(self.registered, directory)
            index = rc.read_json(directory / "index.json")
            index["runs"][0]["run_identities"]["config_sha256"] = "0" * 64
            rc.write_json(directory / "index.json", index)
            with self.assertRaisesRegex(ValueError, "identities differ from the registration"):
                rc._run_inputs(None, directory / "index.json")

    def test_unknown_display_output_is_rejected(self):
        entry = copy.deepcopy(self.result["outputs"][0])
        entry["display_ids"] = ["main:Figure 999"]
        with self.assertRaisesRegex(rc.ReconstructionError, "unknown manuscript display"):
            rc.add_output([], self.output, entry, {r["id"] for r in self.result["displays"]})

    def test_roman_campaign_uses_nested_split_ids_and_requires_all_ten(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign = Path(temporary) / "roman"
            for split in range(10):
                run = campaign / f"split_{split:02d}"
                rc.write_json(run / "run_state.json", {"profile": "full", "scientific_run": True,
                              "task": {"dataset": "Roman-empire"}})
                rc.write_json(run / "final/results_summary.json", {"benchmark_dataset": "Roman-empire",
                              "scientific_run": True})
            def verified_fixture(run):
                split = int(run.name.removeprefix("split_"))
                config = {"dataset": "Roman-empire", "profile": "full", "scientific_run": True,
                          "data": {"split_index": split}}
                return run, config, {}, {}, {}, [], {}, {}
            with mock.patch.object(rc, "verify_run_bytes", side_effect=verified_fixture):
                found, inputs, unavailable = rc._run_inputs(campaign, None)
                self.assertEqual(len(found), 1)
                self.assertEqual(len(inputs), 10)
                self.assertFalse(unavailable)
                shutil.rmtree(campaign / "split_09")
                with self.assertRaisesRegex(rc.ReconstructionError, "ten official split IDs"):
                    rc._run_inputs(campaign, None)

    def test_arbitrary_split_directories_cannot_claim_roman_aggregation(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign = Path(temporary) / "campaign"
            shutil.copytree(ROOT / "examples/runs/arxiv", campaign / "split_00")
            with self.assertRaisesRegex(rc.ReconstructionError, "Only Roman-empire"):
                rc._run_inputs(campaign, None)

    def test_strict_cli_fails_when_display_coverage_is_partial(self):
        self.assertEqual(rc.main(["--verify", str(self.output)]), 0)
        self.assertEqual(rc.main(["--verify", str(self.output), "--require-all-displays"]), 1)


if __name__ == "__main__":
    unittest.main()
