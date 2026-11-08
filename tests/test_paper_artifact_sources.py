"""Portable publication of historical runtimes beside sealed synthetic runs.

The checkout used by each test has no matching current or bundled runtime.
Packing and relocated verification therefore have to preserve the actual
run/campaign snapshot rather than accidentally depending on this repository.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from scripts import artifact_inventory
from scripts import paper_artifacts as pa
from scripts import _source_snapshots as snapshots

ROOT = Path(__file__).resolve().parents[1]


class PortableRuntimePackingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.run = self.base / "campaign/run"
        shutil.copytree(ROOT / "examples/runs/arxiv", self.run)
        self.expected = json.loads((self.run / "run_state.json").read_text())["source_sha256"]
        self.archived = snapshots.resolve_runtime_source(self.expected, ROOT)
        self.snapshot = self.run.parent / "source"
        shutil.copytree(self.archived, self.snapshot)
        self.checkout = self.base / "unrelated-checkout"
        (self.checkout / "cross_morse").mkdir(parents=True)
        (self.checkout / "cross_morse/pipeline.py").write_text("# unrelated runtime fixture\n")
        self.patch_root = mock.patch.object(pa, "ROOT", self.checkout)
        self.patch_root.start()
        self.addCleanup(self.patch_root.stop)
        self.out = self.base / "packed"

    def pack(self, out=None):
        return pa.pack_one("run:ogbn-arxiv", self.run, out or self.out,
                           relationship="exact_result_producing_run", scientific_only=False)

    def unpack(self, entry, tier, *, suffix=""):
        destination = self.base / f"relocated-{tier}{suffix}"
        destination.mkdir()
        top = pa._safe_extract(self.out / entry["tiers"][tier]["archive"], destination)
        run = destination / top
        pa.verify_unpacked(run)
        omitted = pa.analysis_omissions(run) or {}
        identities, _ = pa.verify_analysis_run(run, omitted)
        self.assertEqual(identities["source_sha256"], self.expected)
        self.assertEqual(snapshots.resolve_runtime_source(self.expected, self.checkout, run_root=run), run / "source")
        if tier == "complete":
            with mock.patch.object(artifact_inventory, "SOURCE", self.checkout):
                artifact_inventory.verify_run_bytes(run)
        return run

    def test_parent_runtime_is_in_both_tiers_without_copying_run_or_extra_trees(self):
        # An unrelated file outside the runtime identity must not be swept into
        # the source overlay, even if a launcher put it beside the source files.
        unrelated = self.snapshot / "checkpoints/not-runtime.pt"
        unrelated.parent.mkdir()
        unrelated.write_bytes(b"not a checkpoint; source-only staging sentinel")
        before_run, before_source = pa._inventory(self.run), pa._inventory(self.snapshot)
        original_copy = shutil.copyfile
        with mock.patch.object(pa.shutil, "copyfile", wraps=original_copy) as copied:
            entry = self.pack()
        expected_paths = {p.relative_to(self.snapshot).as_posix() for p in snapshots.runtime_files(self.snapshot)}
        self.assertEqual({Path(c.args[0]).relative_to(self.snapshot).as_posix() for c in copied.call_args_list},
                         expected_paths)
        for tier in ("complete", "analysis"):
            with self.subTest(tier=tier):
                run = self.unpack(entry, tier)
                inventory = pa.read_json(run / pa.INVENTORY_NAME)
                source_rows = [row for row in inventory["files"] if row["path"].startswith("source/")]
                self.assertEqual({row["path"][len("source/"):] for row in source_rows}, expected_paths)
                self.assertEqual(snapshots.runtime_source_fingerprint(run / "source"), self.expected)
                self.assertEqual(inventory["runtime_source_additions"][0]["source_sha256"], self.expected)
                self.assertFalse((run / "source/checkpoints").exists())
                self.assertTrue(all(not pa.is_heavy(row["path"]) for row in source_rows))
                for row in source_rows:
                    path = run / row["path"]
                    self.assertEqual(path.stat().st_size, row["bytes"])
                    self.assertEqual(pa.sha256_file(path), row["sha256"])
        self.assertEqual(pa._inventory(self.run), before_run)
        self.assertEqual(pa._inventory(self.snapshot), before_source)
        self.assertFalse((self.run / "source").exists())

    def test_repacking_external_runtime_produces_identical_inventories_and_assets(self):
        first = self.pack()
        second = self.pack(self.base / "packed-again")
        self.assertEqual(first["inventory_sha256"], second["inventory_sha256"])
        self.assertEqual(first["tiers"], second["tiers"])
        self.assertEqual(first["files"], second["files"])
        self.assertEqual(first["bytes"], second["bytes"])

    def test_existing_own_runtime_is_inventoried_without_staging_or_rewriting_it(self):
        self.snapshot.rename(self.run / "source")
        before = pa._inventory(self.run)
        with mock.patch.object(pa.shutil, "copyfile") as copied:
            entry = self.pack()
        copied.assert_not_called()
        for tier in ("complete", "analysis"):
            run = self.unpack(entry, tier)
            self.assertEqual(pa._inventory(run / "source"), pa._inventory(self.run / "source"))
        self.assertEqual(pa._inventory(self.run), before)

    def test_unbundled_snapshot_inside_checkout_is_still_shipped(self):
        campaign = self.checkout / "work/campaign"
        campaign.parent.mkdir()
        self.run.parent.rename(campaign)
        self.run, self.snapshot = campaign / "run", campaign / "source"
        entry = self.pack()
        for tier in ("complete", "analysis"):
            run = self.unpack(entry, tier)
            self.assertEqual(snapshots.runtime_source_fingerprint(run / "source"), self.expected)

    def test_runtime_change_during_packing_preserves_previous_assets_and_entry(self):
        self.pack()
        before = pa._inventory(self.out)
        path = self.snapshot / "cross_morse/pipeline.py"
        original = path.read_bytes()
        original_tar = pa._tar
        calls = 0

        def changed_after_archive(*args, **kwargs):
            nonlocal calls
            original_tar(*args, **kwargs)
            calls += 1
            if calls == 1:
                path.write_bytes(original + b"\n# changed during pack\n")

        try:
            with mock.patch.object(pa, "_tar", side_effect=changed_after_archive):
                with self.assertRaisesRegex(pa.ArtifactError, "Frozen runtime source changed"):
                    self.pack()
            self.assertEqual(pa._inventory(self.out), before)
            self.assertFalse(list(self.out.glob(".pack-*")))
        finally:
            path.write_bytes(original)

    def test_conflicting_own_source_is_not_overwritten_by_parent_snapshot(self):
        own = self.run / "source"
        own.mkdir()
        (own / "retain.txt").write_text("unresolved original source")
        with self.assertRaisesRegex(pa.ArtifactError, "Conflicting frozen source folder"):
            self.pack()
        self.assertEqual((own / "retain.txt").read_text(), "unresolved original source")
        self.assertFalse(self.out.exists())

    def test_input_output_and_parent_snapshot_symlinks_are_rejected(self):
        alias = self.base / "run-alias"
        alias.symlink_to(self.run, target_is_directory=True)
        with self.assertRaisesRegex(pa.ArtifactError, "symlink"):
            pa.pack_one("run:ogbn-arxiv", alias, self.out, relationship="post_result_revision", scientific_only=False)
        target = self.base / "actual-output"
        target.mkdir()
        out_alias = self.base / "output-alias"
        out_alias.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(pa.ArtifactError, "symlink"):
            self.pack(out_alias)
        actual_source = self.base / "actual-source"
        self.snapshot.rename(actual_source)
        self.snapshot.symlink_to(actual_source, target_is_directory=True)
        with self.assertRaisesRegex(pa.ArtifactError, "symlink"):
            self.pack()
        self.assertFalse(self.out.exists())
        self.assertEqual(list(target.iterdir()), [])

    def test_packing_output_cannot_overlap_the_external_snapshot(self):
        for out in (self.snapshot, self.snapshot / "publication"):
            with self.subTest(out=out), self.assertRaisesRegex(pa.ArtifactError, "separate from the external runtime source"):
                self.pack(out)
        self.assertFalse((self.snapshot / "publication").exists())


if __name__ == "__main__":
    unittest.main()
