"""Historical examples must resolve to byte-exact original runtime source."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from scripts._paper_common import runtime_source_fingerprint
from scripts import _source_snapshots as snapshots, artifact_inventory as inventory
from scripts._source_snapshots import resolve_runtime_source, source_resolution
from scripts.artifact_inventory import _verified_test_caches, InventoryError


class SourceSnapshotTests(unittest.TestCase):
    def source(self, root, text):
        (root / 'cross_morse').mkdir(parents=True)
        (root / 'cross_morse/pipeline.py').write_text(text)
        (root / 'run_cross_morse.py').write_text('# entry point fixture\n')
        return runtime_source_fingerprint(root)

    def test_old_identity_requires_the_matching_original_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'checkout'
            current = self.source(root, '# current\n')
            original = Path(directory) / 'original'
            old = self.source(original, '# original\n')
            with self.assertRaisesRegex(ValueError, 'exact runtime source snapshot'):
                resolve_runtime_source(old, root)
            archive = root / 'examples/source_snapshots' / old
            shutil.copytree(original, archive)
            self.assertEqual(resolve_runtime_source(old, root), archive)
            self.assertEqual(resolve_runtime_source(current, root), root)
            self.assertFalse(source_resolution(old, root)['run_artifacts_rewritten'])
            (archive / 'cross_morse/pipeline.py').write_text('# different bytes\n')
            with self.assertRaisesRegex(ValueError, 'exact runtime source snapshot'):
                resolve_runtime_source(old, root)

    def test_missing_or_extra_runtime_file_changes_the_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = self.source(root, '# original\n')
            (root / 'cross_morse/new_module.py').write_text('# extra\n')
            with self.assertRaises(ValueError):
                resolve_runtime_source(original, root)
            (root / 'cross_morse/new_module.py').unlink()
            (root / 'run_cross_morse.py').unlink()
            with self.assertRaises(ValueError):
                resolve_runtime_source(original, root)

    def test_symlinked_historical_runtime_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'checkout'
            self.source(root, '# current\n')
            original = Path(directory) / 'original'
            old = self.source(original, '# original\n')
            parent = root / 'examples/source_snapshots'
            parent.mkdir(parents=True)
            (parent / old).symlink_to(original, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'symlink'):
                resolve_runtime_source(old, root)

    def test_original_run_and_campaign_snapshots_resolve_without_rewriting(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            checkout, run = base / 'checkout', base / 'campaign/run'
            self.source(checkout, '# current\n')
            own = run / 'source'
            old = self.source(own, '# original\n')
            self.assertEqual(resolve_runtime_source(old, checkout, run_root=run), own)
            resolved = source_resolution(old, checkout, run_root=run)
            self.assertEqual(resolved['source_location'], str(own))
            self.assertFalse(resolved['run_artifacts_rewritten'])
            shared = run.parent / 'source'
            own.rename(shared)
            self.assertEqual(resolve_runtime_source(old, checkout, run_root=run), shared)
            (shared / 'cross_morse/pipeline.py').write_text('# changed bytes\n')
            with self.assertRaisesRegex(ValueError, 'exact runtime source snapshot'):
                resolve_runtime_source(old, checkout, run_root=run)

    def test_own_source_cannot_escape_through_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            checkout, run = base / 'checkout', base / 'run'
            self.source(checkout, '# current\n')
            external = base / 'external'
            old = self.source(external, '# original\n')
            run.mkdir()
            (run / 'source').symlink_to(external, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'symlink'):
                resolve_runtime_source(old, checkout, run_root=run)
            (run / 'source').unlink()
            shutil.copytree(external, run / 'source')
            (run / 'source/cross_morse/linked.py').symlink_to(external / 'cross_morse/pipeline.py')
            with self.assertRaisesRegex(ValueError, 'symlink'):
                resolve_runtime_source(old, checkout, run_root=run)
            linked_run = base / 'linked-run'
            linked_run.symlink_to(run, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'symlink'):
                resolve_runtime_source(old, checkout, run_root=linked_run)

    def test_complete_original_run_uses_its_own_exact_source_snapshot(self):
        # Actual sealed synthetic software run; no result values or provenance
        # are rewritten to make it agree with the current implementation.
        original_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            checkout, run = base / 'checkout', base / 'original-run'
            self.source(checkout, '# unrelated current source\n')
            shutil.copytree(original_root / 'examples/runs/arxiv', run)
            before = {p.relative_to(run).as_posix(): p.read_bytes() for p in run.rglob('*') if p.is_file()}
            old = json.loads((run / 'run_state.json').read_text())['source_sha256']
            archived = resolve_runtime_source(old, original_root)
            shutil.copytree(archived, run / 'source')
            with mock.patch.object(inventory, 'SOURCE', checkout):
                checked = inventory.build_inventory(run)
            self.assertEqual(checked['status'], 'PASS')
            self.assertEqual(checked['source_resolution']['source_location'], str(run / 'source'))
            self.assertEqual(checked['identities']['source_sha256'], old)
            self.assertNotEqual(runtime_source_fingerprint(checkout), old)
            self.assertTrue(all((run / path).read_bytes() == data for path, data in before.items()))

    def test_replay_rejects_unverified_source_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            checkout, run = base / 'checkout', base / 'run'
            self.source(checkout, '# current\n')
            old = self.source(run / 'source', '# original\n')
            (run / 'provenance/stages').mkdir(parents=True)
            (run / 'provenance/stages/00_prepare.json').write_text(json.dumps({'source_sha256': old}))
            (run / 'source/cross_morse/pipeline.py').write_text('# corrupt\n')
            with mock.patch.object(snapshots.subprocess, 'run') as execute:
                with self.assertRaisesRegex(ValueError, 'exact runtime source snapshot'):
                    snapshots.replay_results_exact(run, root=checkout)
            execute.assert_not_called()

    def test_optional_test_cache_must_match_the_sealed_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'final').mkdir()
            (root / 'predictions/test').mkdir(parents=True)
            array = root / 'predictions/test/policy.npy'
            array.write_bytes(b'opaque bytes for hash-only fixture')
            record = {'candidate': {'candidate_id': 'policy'}, 'execution_role': 'test',
                      'prediction_sha256': hashlib.sha256(array.read_bytes()).hexdigest(), 'members': []}
            (root / 'predictions/test/policy.json').write_text(json.dumps(record))
            (root / 'final/results_summary.json').write_text(json.dumps({
                'prediction_storage_scope': 'test_role', 'test_role_execution': {'policy': record}}))
            self.assertEqual(_verified_test_caches(root), {
                'predictions/test/policy.npy', 'predictions/test/policy.json'})
            array.write_bytes(array.read_bytes() + b'changed')
            with self.assertRaisesRegex(InventoryError, 'hash differs'):
                _verified_test_caches(root)


if __name__ == '__main__':
    unittest.main()
