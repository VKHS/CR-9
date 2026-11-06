"""Registration descriptors on fresh clones, safe replacement, and HPC failure paths.

All runs and DOI strings here are offline synthetic fixtures, not benchmark deposits.
"""
from contextlib import redirect_stderr, redirect_stdout
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from scripts import ingest_run, run_manifests
from scripts.archive_run import pack, STAGES
from tests.test_deposit_transport import dump, make_run, row


ROOT = Path(__file__).resolve().parents[1]


def fixture(root, dataset="Peptides-func", split=0, profile="smoke"):
    make_run(root, dataset=dataset, split=split)
    if profile != "smoke":
        config = json.loads((root / "config.json").read_text())
        config["profile"] = profile
        dump(root / "config.json", config)
        state = json.loads((root / "run_state.json").read_text())
        state["profile"] = profile
        state["config_sha256"] = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        dump(root / "run_state.json", state)
        parent = None
        for i, stage in enumerate(STAGES):
            rel = f"provenance/stages/{i:02d}_{stage}.json"
            value = json.loads((root / rel).read_text())
            value.update(config_sha256=state["config_sha256"], parent_manifest=parent)
            value["outputs"] = [row(root, output["path"]) for output in value["outputs"]]
            dump(root / rel, value)
            parent = row(root, rel)
    # Minimal display inputs for the manifest generator. The fixture is explicitly
    # non-scientific and cannot pass the benchmark evidence audit.
    for rel in ("final/results_summary.json", "select/decision.json", "admit/decision.json"):
        dump(root / rel, {})


class RegistrationCloneSafetyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.runs = self.root / "runs"
        self.index = self.runs / "index.json"
        self.out = self.root / "manifests"
        self.source = self.root / "source"
        fixture(self.source)
        self.archive = self.root / "peptides.tar.gz"
        pack(self.source, self.archive, campaign=False)
        self.entry = ingest_run.ingest(self.archive, doi="10.5281/zenodo.4242", runs_root=self.runs,
                                       index_path=self.index, replace=False)
        self.payload = self.runs / "peptides_func"
        self.assertEqual(self.manifests("--with-indices")[0], 0)
        self.manifest = self.out / "official/peptides_func.json"
        self.original_manifest = self.manifest.read_bytes()

    def write_index(self, entries):
        dump(self.index, {"schema": ingest_run.SCHEMA, "runs": entries})

    def manifests(self, *flags):
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            code = run_manifests.main(["runs", "--index", str(self.index), "--label", "official",
                                       "--output-root", str(self.out), *flags])
        return code, output.getvalue()

    def check(self, *, absent=True):
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            code = ingest_run.check(self.runs, self.index, allow_deposited_absent=absent)
        return code, output.getvalue()

    def test_valid_absent_descriptor_defers_evidence_verification_explicitly(self):
        shutil.rmtree(self.payload)
        code, text = self.check()
        self.assertEqual(code, 0, text)
        self.assertIn("NOT VERIFIED", text)
        self.assertIn("0 registered run(s) verify", text)
        self.assertEqual(self.check(absent=False)[0], 1)
        self.assertEqual(self.manifests("--check", "--allow-deposited-absent")[0], 0)
        # Even a valid offline descriptor does not make the strict paper reader
        # discover a run that is absent.
        from scripts.build_paper import from_index
        from paper.build.evidence import EvidenceError
        with self.assertRaises(EvidenceError):
            from_index(self.index)

    def test_absent_shortcut_needs_archive_and_complete_identity_binding(self):
        shutil.rmtree(self.payload)
        variants = [dict(self.entry, archive_sha256="f" * 64), dict(self.entry, archive_bytes=self.entry["archive_bytes"] + 1),
                    dict(self.entry, deposit_doi="10.5281/zenodo.4243"), dict(self.entry, profile="full"),
                    {k: v for k, v in self.entry.items() if k != "archive_name"},
                    {k: v for k, v in self.entry.items() if k != "run_identities"}]
        for key in ingest_run.IDENTITY_KEYS:
            changed = copy.deepcopy(self.entry)
            changed["run_identities"][key] = "f" * 64
            variants.append(changed)
        for entry in variants:
            with self.subTest(entry=entry):
                self.write_index([entry])
                self.assertEqual(self.check()[0], 1)
                self.assertNotEqual(self.manifests("--check", "--allow-deposited-absent")[0], 0)
        self.write_index([self.entry])
        dump(self.manifest, {"schema": "cross-morse-run-manifest-1", "dataset": "Peptides-func",
                             "identities": {"source_sha256": self.entry["run_source_sha256"][0]}})
        self.assertEqual(self.check()[0], 1)
        self.assertNotEqual(self.manifests("--check", "--allow-deposited-absent")[0], 0)

    def test_dangling_links_and_escape_paths_cannot_be_called_absent(self):
        shutil.rmtree(self.payload)
        self.payload.symlink_to(self.root / "missing")
        self.assertEqual(self.check()[0], 1)
        self.assertNotEqual(self.manifests("--check", "--allow-deposited-absent")[0], 0)
        self.payload.unlink()
        self.write_index([dict(self.entry, path="../missing")])
        self.assertEqual(self.check()[0], 1)
        self.assertNotEqual(self.manifests("--check", "--allow-deposited-absent")[0], 0)

    def test_copied_memberships_and_unregistered_manifests_remain_checked(self):
        shutil.rmtree(self.payload)
        split = self.out / "official/peptides_func_splits.npz"
        split.write_bytes(b"changed membership bytes")
        self.assertEqual(self.check()[0], 1)
        self.assertNotEqual(self.manifests("--check", "--allow-deposited-absent")[0], 0)
        shutil.copyfile(self.source / "prepared/splits.npz", split)
        dump(self.out / "official/arxiv.json", {})
        code, text = self.manifests("--check", "--allow-deposited-absent")
        self.assertEqual(code, 2, text)
        self.assertIn("stale executed manifest files", text)

    def test_duplicate_or_unverified_entries_cannot_hide_beside_an_absent_entry(self):
        shutil.rmtree(self.payload)
        for entries in ([self.entry, self.entry], [self.entry, dict(self.entry, verified=False)]):
            self.write_index(entries)
            self.assertEqual(self.check()[0], 1)
            self.assertNotEqual(self.manifests("--check", "--allow-deposited-absent")[0], 0)

    def test_present_payloads_and_index_identities_are_rechecked(self):
        changed = copy.deepcopy(self.entry)
        changed["run_identities"]["unit_sha256"] = "f" * 64
        self.write_index([changed])
        self.assertEqual(self.check()[0], 1)
        self.assertNotEqual(self.manifests("--check", "--allow-deposited-absent")[0], 0)
        self.write_index([self.entry])
        (self.payload / "unarchived.txt").write_text("not in the sealed archive inventory")
        self.assertEqual(self.check()[0], 1)
        self.assertNotEqual(self.manifests("--check", "--allow-deposited-absent")[0], 0)

    def test_failed_replacement_keeps_previous_payload_and_index(self):
        original_index = self.index.read_bytes()
        original_manifest = (self.payload / "ARCHIVE_MANIFEST.json").read_bytes()
        with mock.patch.object(ingest_run, "save_index", side_effect=OSError("simulated full filesystem")):
            with self.assertRaisesRegex(OSError, "simulated full filesystem"):
                ingest_run.ingest(self.archive, doi="10.5281/zenodo.4243", runs_root=self.runs,
                                  index_path=self.index, replace=True)
        self.assertEqual(self.index.read_bytes(), original_index)
        self.assertEqual((self.payload / "ARCHIVE_MANIFEST.json").read_bytes(), original_manifest)
        self.assertEqual(self.check(absent=False)[0], 0)

    def test_custom_index_controls_default_payload_location(self):
        custom = self.root / "separate output/index.json"
        result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/ingest_run.py"), str(self.archive),
                                 "--index", str(custom)], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((custom.parent / "peptides_func/final/results_summary.json").is_file())
        self.assertEqual(json.loads(custom.read_text())["runs"][0]["run_identities"], self.entry["run_identities"])

    def test_pilot_archives_remain_usable_without_relaxing_publication_profiles(self):
        from scripts.deposit_zenodo import verified_run_manifest
        source = self.root / "pilot source"
        fixture(source, profile="pilot")
        archive = self.root / "pilot.tar.gz"
        pack(source, archive, campaign=False)
        entry = ingest_run.ingest(archive, doi=None, runs_root=self.root / "pilot runs",
                                 index_path=self.root / "pilot runs/index.json", replace=False)
        self.assertEqual(entry["profile"], "pilot")
        self.assertFalse(entry["scientific_run"])
        with self.assertRaisesRegex(ValueError, "Invalid archive profile"):
            verified_run_manifest(archive)

    def test_one_roman_split_cannot_register_as_the_full_benchmark_campaign(self):
        entry = dict(self.entry, dataset="Roman-empire", stem="roman_empire", path="roman_empire",
                     profile="full", scientific_run=True)
        with self.assertRaisesRegex(ValueError, "all ten splits"):
            ingest_run.registration_descriptor(entry)


class CampaignRegistrationTests(unittest.TestCase):
    def test_all_ten_identity_and_membership_records_are_required_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = root / "campaign"
            for i in range(10):
                fixture(campaign / f"split_{i:02d}", dataset="Roman-empire", split=i)
            archive = root / "roman.tar.gz"
            pack(campaign, archive, campaign=True)
            runs, out = root / "runs", root / "manifests"
            index = runs / "index.json"
            entry = ingest_run.ingest(archive, doi="10.5281/zenodo.4242", runs_root=runs, index_path=index, replace=False)
            flags = ["runs", "--index", str(index), "--label", "official", "--output-root", str(out)]
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(run_manifests.main([*flags, "--with-indices"]), 0)
                self.assertEqual(run_manifests.main([*flags, "--check"]), 0)
                self.assertEqual(ingest_run.check(runs, index), 0)
            self.assertEqual(set(entry["run_identities"]), set(ingest_run.ROMAN_SPLITS))
            shutil.rmtree(runs / "roman_empire")
            manifest = out / "official/roman_empire.json"
            value = json.loads(manifest.read_text())
            self.assertEqual(len(value["index_arrays_copied"]), 10)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(ingest_run.check(runs, index, allow_deposited_absent=True), 0)
                # Before this fix an empty campaign loop returned success.
                self.assertEqual(ingest_run.check(runs, index), 1)
                (runs / "roman_empire").mkdir()
                self.assertEqual(ingest_run.check(runs, index, allow_deposited_absent=True), 1)
                (runs / "roman_empire").rmdir()
            for key in ("source_sha256", "config_sha256", "unit_sha256"):
                changed = copy.deepcopy(value)
                changed["campaign"]["per_split"][9]["identities"][key] = "f" * 64
                dump(manifest, changed)
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    self.assertEqual(ingest_run.check(runs, index, allow_deposited_absent=True), 1)
                    self.assertNotEqual(run_manifests.main([*flags, "--check", "--allow-deposited-absent"]), 0)
            changed = copy.deepcopy(value)
            changed["campaign"]["per_split"].pop()
            dump(manifest, changed)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(ingest_run.check(runs, index, allow_deposited_absent=True), 1)


class AvailabilityRegistrationTests(unittest.TestCase):
    def test_first_full_registration_refreshes_derived_scope_without_enabling_claims(self):
        from scripts.availability import main
        from tests.test_availability import minimal_tree
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            minimal_tree(root, official=["ogbn-arxiv"])
            path = root / "release/release_scope.json"
            scope = json.loads(path.read_text())
            scope["full_official_dataset_runs_verified"] = []
            dump(path, scope)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main(["--root", str(root)]), 0)
                self.assertEqual(main(["--root", str(root), "--check"]), 0)
            updated = json.loads(path.read_text())
            self.assertEqual(updated, dict(scope, full_official_dataset_runs_verified=["ogbn-arxiv"]))
            self.assertFalse(updated["original_result_archives_included"])
            self.assertFalse(updated["one_command_original_tables_reproduction_claimed"])
            # The writer also preserves an author's affirmative claims. Their
            # validity remains the separate scientific gate; it cannot erase a
            # false claim merely to make verification green.
            updated.update(original_result_archives_included=True, one_command_original_tables_reproduction_claimed=True)
            dump(path, updated)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main(["--root", str(root)]), 0)
                self.assertEqual(main(["--root", str(root), "--check"]), 1)
            self.assertEqual(json.loads(path.read_text()), updated)


@unittest.skipUnless(shutil.which("bash"), "bash is required")
class HpcPlanFailureTests(unittest.TestCase):
    def test_invalid_plan_is_failure_and_unconfigured_dataset_is_explicit_noop(self):
        from scripts.evidence_pipeline import template
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = root / "plan.json"
            profile = root / "profile.env"
            profile.write_text(f"CM_PYTHON={sys.executable}\n")
            env = {k: v for k, v in os.environ.items() if not k.startswith("CM_")}
            env.update(CM_SOURCE_ROOT=str(ROOT), CM_ENV_FILE=str(profile), CM_EVIDENCE_PLAN=str(plan), CM_DATASET_INDEX="0")
            plan.write_text("{broken json")
            command = ["bash", str(ROOT / "hpc/pack_evidence_arrhenius.sh")]
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Cannot validate", result.stderr)
            self.assertNotIn("nothing to do", result.stdout)
            value = template()
            value["deposits_dir"] = str(root / "deposits")
            dump(plan, value)
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("no run_root", result.stdout)
            self.assertFalse((root / "deposits/evidence-state.json").exists())


if __name__ == "__main__":
    unittest.main()
