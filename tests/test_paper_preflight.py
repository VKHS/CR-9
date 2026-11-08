"""Local publication assessment using sealed software fixtures, never paper evidence.

The real input verifiers run on bundled synthetic examples. Isolated display
fixtures exercise exit-status gates; one test performs the actual reconstruction.
No fixture bypass is exposed by the command-line interface.
"""
from __future__ import annotations

from contextlib import ExitStack, redirect_stderr
import errno
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from scripts import artifact_inventory, paper_artifacts as artifacts, paper_preflight as preflight
from scripts import _source_snapshots as snapshots

ROOT = Path(__file__).resolve().parents[1]


class LocalPublicationPreflight(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="cm-preflight-")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.sources = self.base / "originals"
        self.sources.mkdir()
        self.map = self.base / "paper-artifacts.map.json"
        self.work = self.base / "assessment"
        self.contracts = artifacts.required_artifacts()
        self.source_paths = {}

    def inputs(self, *, run=True, record=False, environment=False):
        choices = []
        if run:
            choices.append(("run:ogbn-arxiv", ROOT / "examples/runs/arxiv", self.sources / "arxiv"))
        if record:
            choices.append(("records:nominal_time", ROOT / "examples/records/nominal_time",
                            self.sources / "nominal_time"))
        for identifier, source, target in choices:
            shutil.copytree(source, target)
            self.source_paths[identifier] = target
        if environment:
            target = self.sources / "gpu"
            target.mkdir()
            (target / "fixture.json").write_text('{"software_fixture": true}\n')
            self.source_paths["environment:gpu"] = target
        self.write_map()
        return self.source_paths

    def write_map(self, relationships=True):
        value = {key: str(path) for key, path in self.source_paths.items()}
        value["_relationships"] = ({key: "post_result_revision" for key in self.source_paths}
                                   if relationships else {})
        self.map.write_text(json.dumps(value))

    def complete_displays(self):
        # Explicit arithmetic-free fixture for the artifact/status gate, using
        # every registered display ID rather than shrinking the paper contract.
        identifiers = json.loads((ROOT / "paper/display_manifest.json").read_text())["displays"]
        return {"displays": [{"id": row["id"], "full_numerical_display": True, "content_complete": True}
                             for row in identifiers],
                "display_generators": {"comparison_with_printed_values": {
                    "exact": 1, "differs": 0, "not_comparable": 0, "no_reference": 0, "missing": 0},
                    "comparison_coverage_complete": True, "errors": {}, "module_errors": {}}}

    def assess(self, *, mode="hardlink", real_build=False, required=None, effect=None, scientific=False):
        contract = ([row for row in self.contracts if row["id"] in self.source_paths]
                    if required is None else [row for row in self.contracts if row["id"] in required])
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(artifacts, "required_artifacts", return_value=contract))
            if not real_build:
                result = self.complete_displays()
                stack.enter_context(mock.patch.object(preflight, "reconstruct", return_value=result,
                                                      side_effect=effect))
                stack.enter_context(mock.patch.object(preflight, "verify_workspace", return_value=result))
            return preflight.assess(self.map, self.work, copy_mode=mode, _software_fixture=not scientific)

    def test_real_legacy_inputs_reconstruct_without_claiming_publication(self):
        paths = self.inputs(record=True)
        original_hash = artifacts.sha256_file(artifacts.MANIFEST)
        state = json.loads((paths["run:ogbn-arxiv"] / "run_state.json").read_text())
        self.assertNotEqual(state["source_sha256"], snapshots.runtime_source_fingerprint(ROOT))
        report = self.assess(real_build=True)
        self.assertEqual(report["exit_code"], 1)
        self.assertEqual(report["verdict"], "INCOMPLETE")
        self.assertFalse(report["public_availability_verified"])
        self.assertFalse(report["availability_statement_fulfilled"])
        self.assertFalse(report["historical_identity_authenticated"])
        self.assertTrue(report["software_fixture"])
        self.assertEqual(artifacts.sha256_file(artifacts.MANIFEST), original_hash)
        self.assertTrue((self.work / "inputs/records/index.json").is_file())
        self.assertTrue((self.work / "reconstruction/RECONSTRUCTION_REPORT.md").is_file())
        rebuilt = preflight.verify_workspace(self.work / "reconstruction")
        self.assertTrue(rebuilt["synthetic_examples"])
        self.assertTrue(any(row["kind"] == "sealed_run" for row in rebuilt["inputs"]))
        self.assertTrue(report["reconstruction"]["workspace_verified_after_inputs"])
        for row in report["artifacts"]:
            self.assertEqual(row["status"], "VERIFIED")
            self.assertTrue(row["source_unchanged_after_reconstruction"])
            self.assertTrue(row["materialization_unchanged_after_reconstruction"])
            self.assertTrue((self.work / row["inventory_file"]).is_file())

    def test_shared_hardlink_bytes_keep_original_file_modes(self):
        source = self.inputs()["run:ogbn-arxiv"] / "config.json"
        os.chmod(source, 0o440)
        before = source.stat()
        report = self.assess()
        linked = self.work / "inputs/runs/arxiv/config.json"
        self.assertEqual(report["exit_code"], 0)
        self.assertEqual(report["verdict"], "SOFTWARE_FIXTURE_COMPLETE")
        self.assertFalse(report["public_availability_verified"])
        self.assertEqual((source.stat().st_dev, source.stat().st_ino), (linked.stat().st_dev, linked.stat().st_ino))
        self.assertEqual(source.stat().st_mode, before.st_mode)
        self.assertIn("not an independent immutable copy", (self.work / "REPORT.md").read_text())

    def test_copy_materializes_separate_verified_bytes(self):
        source = self.inputs()["run:ogbn-arxiv"] / "config.json"
        report = self.assess(mode="copy")
        copied = self.work / "inputs/runs/arxiv/config.json"
        self.assertEqual(report["exit_code"], 0)
        self.assertNotEqual((source.stat().st_dev, source.stat().st_ino), (copied.stat().st_dev, copied.stat().st_ino))
        self.assertEqual(source.read_bytes(), copied.read_bytes())

    def test_environment_is_required_even_when_every_display_fixture_is_complete(self):
        self.inputs()
        report = self.assess(required={"run:ogbn-arxiv", "environment:gpu"})
        self.assertEqual(report["exit_code"], 1)
        environment = next(row for row in report["artifacts"] if row["id"] == "environment:gpu")
        self.assertEqual(environment["status"], "MISSING")
        self.assertEqual(report["reconstruction"]["problems"], [])

    def test_missing_relationship_cannot_be_inferred_from_complete_local_bytes(self):
        self.inputs()
        self.write_map(relationships=False)
        report = self.assess()
        self.assertEqual(report["exit_code"], 1)
        self.assertIsNone(report["artifacts"][0]["relationship"])
        self.assertTrue(any("_relationships" in row["message"] for row in report["issues"]))
        self.assertTrue((self.work / "inputs/runs/arxiv/config.json").is_file())

    def test_scientific_default_rejects_synthetic_runs_and_has_no_cli_bypass(self):
        self.inputs()
        report = self.assess(scientific=True)
        self.assertEqual(report["exit_code"], 2)
        self.assertFalse(report["software_fixture"])
        self.assertEqual(report["artifacts"][0]["status"], "INVALID")
        self.assertIn("not a full scientific run", report["artifacts"][0]["error"])
        self.assertFalse((self.work / "inputs/runs/arxiv").exists())
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            preflight.main(["--map", str(self.map), "--work", str(self.base / "bypass"),
                            "--software-fixture"])
        self.assertEqual(error.exception.code, 2)

    def test_tampered_run_is_invalid_but_missing_folder_is_incomplete(self):
        source = self.inputs()["run:ogbn-arxiv"]
        (source / "config.json").write_text((source / "config.json").read_text() + " ")
        report = self.assess()
        self.assertEqual(report["exit_code"], 2)
        self.assertEqual(report["artifacts"][0]["status"], "INVALID")
        self.source_paths["run:ogbn-arxiv"] = self.base / "absent-original"
        self.write_map()
        self.work = self.base / "missing"
        report = self.assess()
        self.assertEqual(report["exit_code"], 1)
        self.assertEqual(report["artifacts"][0]["status"], "MISSING")

    def test_workspace_cannot_overlap_inputs_checkout_or_existing_directory(self):
        source = self.inputs()["run:ogbn-arxiv"]
        for output in (source / "new", self.sources, ROOT / "new-preflight", self.base):
            with self.subTest(output=output), self.assertRaisesRegex(preflight.PreflightError, "overlaps|already exists"):
                preflight.assess(self.map, output, _software_fixture=True)
        self.work.mkdir()
        sentinel = self.work / "keep"
        sentinel.write_text("original contents")
        with self.assertRaisesRegex(preflight.PreflightError, "already exists"):
            preflight.assess(self.map, self.work, _software_fixture=True)
        self.assertEqual(sentinel.read_text(), "original contents")

    def test_symlinked_source_or_output_and_relative_input_paths_are_rejected(self):
        source = self.inputs()["run:ogbn-arxiv"]
        linked = self.base / "linked-original"
        linked.symlink_to(source, target_is_directory=True)
        self.source_paths["run:ogbn-arxiv"] = linked
        self.write_map()
        with self.assertRaisesRegex(preflight.PreflightError, "symlink"):
            preflight.assess(self.map, self.work, _software_fixture=True)
        self.source_paths["run:ogbn-arxiv"] = Path("relative/run")
        self.write_map()
        with self.assertRaisesRegex(preflight.PreflightError, "absolute"):
            preflight.assess(self.map, self.work, _software_fixture=True)
        self.source_paths["run:ogbn-arxiv"] = source
        self.write_map()
        output_parent = self.base / "linked-output"
        output_parent.symlink_to(self.sources, target_is_directory=True)
        with self.assertRaisesRegex(preflight.PreflightError, "symlink"):
            preflight.assess(self.map, output_parent / "new", _software_fixture=True)
        self.assertFalse(self.work.exists())

    def test_cross_filesystem_hardlink_failure_requires_explicit_copy(self):
        self.inputs()
        with mock.patch.object(preflight.os, "link", side_effect=OSError(errno.EXDEV, "cross-device link")):
            report = self.assess()
        self.assertEqual(report["exit_code"], 2)
        self.assertTrue(any("--copy-mode copy" in row["message"] for row in report["issues"]))
        self.assertFalse((self.work / "inputs/runs/arxiv").exists())
        self.work = self.base / "copy-after-cross-filesystem"
        self.assertEqual(self.assess(mode="copy")["exit_code"], 0)

    def test_original_bytes_are_reverified_after_reconstruction(self):
        source = self.inputs()["run:ogbn-arxiv"] / "config.json"

        def change_original(*args, **kwargs):
            source.write_text(source.read_text() + " ")
            return self.complete_displays()

        report = self.assess(effect=change_original)
        self.assertEqual(report["exit_code"], 2)
        self.assertTrue(any("Original bytes or sealed identity changed" in row["message"] for row in report["issues"]))
        self.assertFalse(report["public_availability_verified"])

    def test_copied_bytes_are_also_reverified(self):
        source = self.inputs()["run:ogbn-arxiv"] / "config.json"
        before = source.read_bytes()

        def change_copy(*args, **kwargs):
            target = self.work / "inputs/runs/arxiv/config.json"
            target.write_text(target.read_text() + " ")
            return self.complete_displays()

        report = self.assess(mode="copy", effect=change_copy)
        self.assertEqual(report["exit_code"], 2)
        self.assertEqual(source.read_bytes(), before)
        self.assertTrue(report["artifacts"][0]["source_unchanged_after_reconstruction"])
        self.assertTrue(any("Materialized input bytes changed" in row["message"] for row in report["issues"]))

    def test_map_and_builder_changes_cannot_receive_a_ready_verdict(self):
        self.inputs()

        def change_map(*args, **kwargs):
            value = json.loads(self.map.read_text())
            value["_instructions"] = "changed after validation"
            self.map.write_text(json.dumps(value))
            return self.complete_displays()

        report = self.assess(effect=change_map)
        self.assertEqual(report["exit_code"], 2)
        self.assertTrue(any(row["scope"] == "input map" for row in report["issues"]))
        self.work = self.base / "changed-builder"
        with mock.patch.object(preflight, "source_inventory", side_effect=[{"builder.py": "before"},
                                                                          {"builder.py": "after"}]):
            report = self.assess()
        self.assertEqual(report["exit_code"], 2)
        self.assertTrue(any(row["scope"] == "builder" for row in report["issues"]))

    def test_mismatch_is_incomplete_and_generator_failure_is_invalid(self):
        self.inputs()
        for name, key in (("mismatch", "differs"), ("uncomputed", "missing")):
            self.work = self.base / name
            result = self.complete_displays()
            result["display_generators"]["comparison_with_printed_values"][key] = 1
            self.assertEqual(self.assess(effect=lambda *args, value=result, **kwargs: value)["exit_code"], 2,
                             "a reconstruction file manifest inconsistent with its summary is invalid")
            # Supply a matching output report so this assertion isolates the
            # scientific comparison gate, not the final workspace verifier.
            self.work = self.base / (name + "-consistent")
            with mock.patch.object(self, "complete_displays", return_value=result):
                self.assertEqual(self.assess()["exit_code"], 1)
        self.work = self.base / "generator-failure"
        result = self.complete_displays()
        result["display_generators"]["errors"] = {"fixture": "test failure"}
        with mock.patch.object(self, "complete_displays", return_value=result):
            self.assertEqual(self.assess()["exit_code"], 2)

    def test_parent_source_snapshot_is_preserved_without_touching_the_original(self):
        source = self.inputs()["run:ogbn-arxiv"]
        expected = json.loads((source / "run_state.json").read_text())["source_sha256"]
        archived = snapshots.resolve_runtime_source(expected, ROOT)
        shared = source.parent / "source"
        shutil.copytree(archived, shared)
        unrelated = self.base / "unrelated-checkout"
        (unrelated / "cross_morse").mkdir(parents=True)
        (unrelated / "cross_morse/pipeline.py").write_text("# different software fixture\n")
        resolve = lambda digest, root, **kw: snapshots.resolve_runtime_source(digest, unrelated, **kw)
        with mock.patch.object(artifacts, "ROOT", unrelated), \
             mock.patch.object(artifact_inventory, "SOURCE", unrelated), \
             mock.patch.object(preflight, "resolve_runtime_source", side_effect=resolve):
            report = self.assess()
        self.assertEqual(report["exit_code"], 0)
        target_source = self.work / "inputs/runs/arxiv/source"
        self.assertEqual(snapshots.runtime_source_fingerprint(target_source), expected)
        self.assertFalse((source / "source").exists())
        self.assertEqual(snapshots.runtime_source_fingerprint(shared), expected)
        dependency = report["artifacts"][0]["runtime_sources"][0]
        self.assertEqual(dependency["source"], str(shared))
        self.assertEqual(dependency["additional_materialization"], "source")
        self.assertFalse(report["historical_identity_authenticated"])

    def test_duplicate_map_keys_are_rejected_without_creating_workspace(self):
        self.map.write_text('{"_relationships": {}, "run:ogbn-arxiv": "", "run:ogbn-arxiv": ""}')
        with self.assertRaisesRegex(preflight.PreflightError, "Duplicate JSON key"):
            preflight.assess(self.map, self.work, _software_fixture=True)
        self.assertFalse(self.work.exists())

    def test_unbundled_parent_snapshot_inside_checkout_is_still_materialized(self):
        original = self.inputs()["run:ogbn-arxiv"]
        expected = json.loads((original / "run_state.json").read_text())["source_sha256"]
        archived = snapshots.resolve_runtime_source(expected, ROOT)
        checkout = self.base / "checkout"
        run = checkout / "local-originals/run"
        shutil.copytree(original, run)
        shared = run.parent / "source"
        shutil.copytree(archived, shared)
        identity = artifacts._verify_source("run:ogbn-arxiv", run, scientific_only=False)
        with mock.patch.object(preflight, "ROOT", checkout):
            dependencies = preflight._runtime_sources("run:ogbn-arxiv", run, identity)
        self.assertEqual(dependencies[0]["source"], str(shared))
        self.assertEqual(dependencies[0]["additional_materialization"], "source")
        self.assertTrue(dependencies[0]["files"])


if __name__ == "__main__":
    unittest.main()
