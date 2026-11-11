"""Offline submission-gate and preservation tests; never publish or train models."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from scripts import submission_package as sp


def status():
    return {
        'displays': 2, 'complete': 2, 'values_exact': 3,
        'values_differ': 0, 'values_not_comparable': 0, 'values_no_reference': 0,
        'values_missing': 0, 'incomplete': [], 'display_content_incomplete': [],
        'artifacts_awaiting_publication': [], 'generator_errors': {}, 'module_errors': {},
        'comparison_coverage_complete': True,
        'artifact_materialization': {
            'requested_tier': 'complete', 'not_registered': [], 'not_published': [],
            'missing_local': [], 'incomplete_tiers': [],
            'required_artifacts': ['run:a'], 'verified_tiers': {'run:a': 'complete'},
        },
    }


class Gate(unittest.TestCase):
    def test_complete_report_passes(self):
        self.assertEqual(sp.reconstruction_problems(status(), 2, {'run:a'}), [])

    def test_missing_field_is_not_a_pass(self):
        for key in status():
            doc = status(); del doc[key]
            with self.subTest(key=key):
                self.assertTrue(sp.reconstruction_problems(doc, 2, {'run:a'}))

    def test_counts_and_booleans_fail_closed(self):
        for key, values in {
            'displays': [0, 1, 3, True, '2'], 'complete': [0, 1, 3, True, '2'],
            'values_exact': [0, -1, True, '3'], 'values_differ': [1, False, '0'],
            'comparison_coverage_complete': [None, False, 1, 'true'],
        }.items():
            for value in values:
                doc = status(); doc[key] = value
                with self.subTest(key=key, value=value):
                    self.assertTrue(sp.reconstruction_problems(doc, 2, {'run:a'}))

    def test_zero_of_zero_cannot_pass(self):
        doc = status(); doc['displays'] = doc['complete'] = 0
        self.assertTrue(sp.reconstruction_problems(doc, 0, set()))

    def test_alternate_channel_does_not_satisfy_manuscript_gate(self):
        doc = status(); doc['channel'] = 'demonstration'
        self.assertTrue(sp.reconstruction_problems(doc, 2, {'run:a'}))

    def test_every_canonical_artifact_is_required(self):
        doc = status()
        self.assertTrue(sp.reconstruction_problems(doc, 2, {'run:a', 'run:b'}))
        doc['artifact_materialization']['required_artifacts'] = ['run:a', 'run:a']
        self.assertTrue(sp.reconstruction_problems(doc, 2, {'run:a'}))

    def test_no_analysis_cache_or_missing_bytes(self):
        for key, value in (
            ('requested_tier', 'analysis'), ('verified_tiers', {'run:a': 'analysis'}),
            ('not_registered', ['run:b']), ('not_published', ['run:a']),
            ('missing_local', ['run:a']), ('incomplete_tiers', [{'id': 'run:a'}]),
        ):
            doc = status(); doc['artifact_materialization'][key] = value
            with self.subTest(key=key):
                self.assertTrue(sp.reconstruction_problems(doc, 2, {'run:a'}))

    def test_errors_and_incomplete_content_fail(self):
        for key, value in (('generator_errors', {'x': 'failure'}), ('module_errors', {'y': 'failure'}),
                           ('incomplete', ['figure']), ('display_content_incomplete', ['figure']),
                           ('values_missing', 2), ('values_no_reference', 2), ('values_not_comparable', 2)):
            doc = status(); doc[key] = value
            self.assertTrue(sp.reconstruction_problems(doc, 2, {'run:a'}))

    def test_check_does_not_mutate_status(self):
        doc = status(); before = copy.deepcopy(doc)
        sp.reconstruction_problems(doc, 2, {'run:a'})
        self.assertEqual(doc, before)


class Preservation(unittest.TestCase):
    def test_invalid_json_is_not_silently_normalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'map.json'
            for text in ('{"x":1,"x":2}', '{"x": NaN}', '{"x": Infinity}'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    sp.read_json(path)

    def test_nested_and_parent_workspaces_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); data = base / 'data'; data.mkdir()
            for work in (base, data, data / 'work'):
                with self.assertRaises(sp.PackageError):
                    sp.require_separate(work, [data])
            sp.require_separate(base / 'work', [data])

    def test_symlink_paths_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'real').mkdir(); (root / 'link').symlink_to(root / 'real')
            with self.assertRaises(sp.PackageError):
                sp.safe_path(root / 'link' / 'new')

    def test_existing_report_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / 'old'; work.mkdir(); report = work / 'report.json'; report.write_text('preserve')
            self.assertNotEqual(sp.main(['inventory', '--work', str(work)]), 0)
            self.assertEqual(report.read_text(), 'preserve')

    def test_output_inside_map_input_refused_before_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = root / 'data'; data.mkdir()
            mapping = root / 'map.json'; mapping.write_text(json.dumps({'run:a': str(data)}))
            work = data / 'work'
            self.assertNotEqual(sp.main(['finalize', '--map', str(mapping), '--work', str(work)]), 0)
            self.assertFalse(work.exists())

    def test_display_manifest_checks_duplicates_and_empty_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'paper').mkdir()
            path = root / 'paper/display_manifest.json'
            for rows in ([], [{'id': 'x'}, {'id': 'x'}], [{'id': None}]):
                path.write_text(json.dumps({'displays': rows}))
                with self.assertRaises(sp.PackageError):
                    sp.expected_displays(root)
            path.write_text(json.dumps({'displays': [{'id': 'x'}, {'id': 'y'}]}))
            self.assertEqual(sp.expected_displays(root), ['x', 'y'])

    def test_failed_command_stops_sequence_and_preserves_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); report = {'steps': []}
            with self.assertRaises(sp.PackageError):
                sp.run_step('check', [sys.executable, '-c', "print('failure detail'); exit(4)"], root, root, report)
            self.assertEqual(report['steps'][0]['exit_code'], 4)
            self.assertIn('failure detail', (root / '00-check.log').read_text())

    def test_empty_descriptors_are_not_an_absence_claim_about_original_data(self):
        issues = sp.descriptor_problems({'status': 'awaiting_measured_outputs'})
        self.assertIn('Complete-tier archive URLs and checksums are not connected', issues)
        self.assertNotIn('Data do not exist', issues)

    def test_ready_descriptor_requires_identity_and_complete_parts(self):
        good = {'status': 'published', 'identity': {'run': 'a'}, 'inventory_sha256': 'a' * 64,
                'tiers': {'complete': {'parts': [{'url': 'https://example.test/a', 'sha256': 'b' * 64}]}}}
        self.assertEqual(sp.descriptor_problems(good), [])
        for key in ('identity', 'tiers', 'inventory_sha256'):
            doc = copy.deepcopy(good); doc.pop(key)
            self.assertTrue(sp.descriptor_problems(doc))
        good['scientific_only_verification'] = False
        self.assertTrue(sp.descriptor_problems(good))


class Finalization(unittest.TestCase):
    def test_preflight_without_publish_does_not_upload_or_change_branch(self):
        import scripts
        fake = types.SimpleNamespace(required_artifacts=lambda root: [{'id': 'run:a'}],
                                     RELATIONSHIPS={'exact_result_producing_run'})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'repo'; root.mkdir()
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            subprocess.run(['git', 'config', 'user.email', 'test@example.invalid'], cwd=root, check=True)
            subprocess.run(['git', 'config', 'user.name', 'Test'], cwd=root, check=True)
            (root / 'source.txt').write_text('untouched')
            subprocess.run(['git', 'add', '.'], cwd=root, check=True)
            subprocess.run(['git', 'commit', '-qm', 'fixture'], cwd=root, check=True)
            head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root)
            data = Path(tmp) / 'data'; data.mkdir(); (data / 'observation').write_text('original')
            mapping = Path(tmp) / 'map.json'
            mapping.write_text(json.dumps({'run:a': str(data), '_relationships': {'run:a': 'exact_result_producing_run'}}))
            work = Path(tmp) / 'work'; work.mkdir(); report = {'steps': []}; executed = []
            def run(name, argv, cwd, target, result):
                executed.append(name)
                if name == 'isolated-checkout':
                    subprocess.run(argv, cwd=cwd, check=True, capture_output=True)
                elif name == 'original-input-preflight':
                    (work / 'preflight').mkdir()
                    (work / 'preflight/report.json').write_text(json.dumps({
                        'verdict': 'READY_FOR_PUBLICATION', 'software_fixture': False}))
                else:
                    self.fail('Unrequested action: ' + name)
            with mock.patch.object(scripts, 'paper_artifacts', fake, create=True), mock.patch.object(sp, 'run_step', side_effect=run):
                sp.finalize(root, work, mapping, False, 'copy', report)
            self.assertEqual(executed, ['isolated-checkout', 'original-input-preflight'])
            self.assertEqual(report['verdict'], 'LOCAL_INPUTS_READY_FOR_PUBLICATION')
            self.assertFalse(report['full_reconstruction_verified'])
            self.assertEqual((data / 'observation').read_text(), 'original')
            self.assertEqual((root / 'source.txt').read_text(), 'untouched')
            self.assertEqual(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root), head)
            self.assertFalse(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root))

    def test_fixture_map_rejected_before_worktree(self):
        import scripts
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); mapping = root / 'map.json'; mapping.write_text('{"_fixture": true}')
            with mock.patch.object(scripts, 'paper_artifacts', types.SimpleNamespace(), create=True), \
                    mock.patch.object(sp, 'run_step') as run, self.assertRaises(sp.PackageError):
                sp.finalize(root, root / 'work', mapping, True, 'copy', {'steps': []})
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
