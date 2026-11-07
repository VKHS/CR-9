"""Typed availability and artifact-identity checks, using temporary artificial fixtures.

No test creates or claims a scientific observation. The fixtures exercise path,
registration, provenance-label and report-freshness boundaries.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from scripts import artifact_catalog as catalog
from scripts._paper_common import DATASETS
from scripts.ingest_run import registration_descriptor
from scripts.records import cmd_index
from tests.test_records import baseline_set

PROJECT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture_tree(root: Path) -> dict:
    for directory in ("release", "docs", "runs", "records", "paper", "data", "cross_morse", "scripts"):
        (root / directory).mkdir(parents=True)
    (root / "run_cross_morse.py").write_text("# Fixture entry point; must never be executed.\n")
    (root / "cross_morse/__init__.py").write_text("# Fixture source identity.\n")
    (root / "scripts/artifact_catalog.py").write_text("# Fixture catalog identity.\n")
    write_json(root / "release/release_scope.json", {"bundled_example_run_roots": [], "original_result_archives_included": False})
    write_json(root / "runs/index.json", {"schema": "cross-morse-runs-index-1", "runs": []})
    write_json(root / "records/index.json", {"schema": "cross-morse-records-index-1", "sets": []})
    shutil.copyfile(PROJECT / "records/specs.json", root / "records/specs.json")
    write_json(root / "paper/sources.json", {"fixture": True})
    write_json(root / "paper/display_manifest.json", {"fixture": True})
    write_json(root / "data/sources.json", {
        "schema_version": "cross-morse-data-sources-1",
        "datasets": [{"dataset": dataset, "provider_raw_data_bundled": False,
                      "source_documentation": "https://example.org/fixture", "download_url": None,
                      "statistical_unit": "fixture"} for dataset in DATASETS],
        "loader_file_sha256": {"cross_morse/__init__.py": hashlib.sha256((root / "cross_morse/__init__.py").read_bytes()).hexdigest()},
        "role_split_rule": {"fixture": True}, "checksum_policy": "Fixture only",
    })
    spec = {"schema": catalog.SPEC_SCHEMA, "commands": {
        "produce": {"mode": "execute", "argv": ["python", "-B", "run_cross_morse.py"],
                    "entrypoints": ["run_cross_morse.py"], "scope": "Fixture recipe, never executed", "requires": []}},
        "entries": [{"id": name, "title": name, "scope": "Fixture catalog contract",
                     "bundled": [{"artifact_class": "implementation", "patterns": ["run_cross_morse.py"]}],
                     "generated": [{"root": "run", "patterns": ["final/results.json"], "producers": ["produce"],
                                    "scope": "Fixture output contract", "identity_fields": list(catalog.IDENTITY_KEYS)}],
                     "commands": ["produce"], "contracts": ["data/sources.json"],
                     "provider_data": name == "source_data", "artifact_indexes": ["runs/index.json"]}
                    for name in sorted(catalog.CLAIMS)]}
    write_json(root / catalog.SPEC, spec)
    return spec


def registered_reference(root: Path) -> dict:
    identities = dict(zip(catalog.IDENTITY_KEYS, ("1" * 64, "2" * 64, "3" * 64)))
    entry = {"dataset": "ogbn-arxiv", "stem": "arxiv", "path": "arxiv", "verified": True,
             "kind": "run", "profile": "full", "scientific_run": True, "archive_name": "fixture.tar.gz",
             "archive_sha256": "4" * 64, "archive_bytes": 123, "files": 2,
             "deposit_doi": "10.5281/zenodo.123456", "run_source_sha256": [identities["source_sha256"]],
             "run_identities": identities}
    write_json(root / "runs/index.json", {"schema": "cross-morse-runs-index-1", "runs": [entry]})
    manifest = {"schema": "cross-morse-run-manifest-1", "registration": registration_descriptor(entry),
                "dataset": entry["dataset"], "profile": "full", "scientific_run": True,
                "identities": identities, "seeds": {}, "source_data": {}, "policy_lock": {}, "environment": {},
                "splits": {"arrays": {"splits.npz": {"sha256": identities["unit_sha256"], "bytes": 5}}}}
    write_json(root / "manifests/official/arxiv.json", manifest)
    return entry


class ArtifactCatalogTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="cm-catalog-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.spec = fixture_tree(self.root)

    def test_recipes_are_not_executed_and_generated_products_are_not_bundled(self):
        with mock.patch("subprocess.run", side_effect=AssertionError("producer executed")):
            report = catalog.build(self.root)
        self.assertEqual({row["id"] for row in report["entries"]}, catalog.CLAIMS)
        self.assertFalse(report["scientific_evidence_verified"])
        self.assertFalse(report["all_paper_displays_reproduced"])
        self.assertEqual(report["registered_runs"], [])
        self.assertFalse((self.root / "final").exists())
        for row in report["entries"]:
            self.assertEqual(row["generated"][0]["artifact_class"], "new_execution_contract")
            self.assertFalse(row["generated"][0]["materialized_by_catalog"])
        self.assertTrue(all(row["network_checked"] is False for row in report["provider_datasets"]))

    def test_missing_duplicate_and_unknown_availability_items_fail(self):
        variants = []
        removed = copy.deepcopy(self.spec)
        removed["entries"].pop()
        variants.append(removed)
        duplicate = copy.deepcopy(self.spec)
        duplicate["entries"].append(duplicate["entries"][0])
        variants.append(duplicate)
        unknown = copy.deepcopy(self.spec)
        unknown["entries"][0]["id"] = "everything_is_reproduced"
        variants.append(unknown)
        for value in variants:
            with self.subTest(ids=[row["id"] for row in value["entries"]]), self.assertRaisesRegex(catalog.CatalogError, "exactly once"):
                catalog.validate_spec(self.root, value)

    def test_synthetic_and_reported_references_cannot_be_promoted(self):
        for pattern in ("examples/runs/fixture/final/results.json", "paper/reported_results.json"):
            spec = copy.deepcopy(self.spec)
            spec["entries"][0]["bundled"][0]["patterns"] = [pattern]
            with self.subTest(pattern=pattern), self.assertRaisesRegex(catalog.CatalogError, "remain synthetic|reported reference"):
                catalog.validate_spec(self.root, spec)

    def test_bad_paths_and_symlinked_bundled_inputs_fail(self):
        for value in ("../outside", "/absolute", "a/../outside", "a\\b", "./run_cross_morse.py"):
            with self.subTest(value=value), self.assertRaises(catalog.CatalogError):
                catalog.safe_path(self.root, value)
        (self.root / "linked.py").symlink_to(self.root / "run_cross_morse.py")
        spec = copy.deepcopy(self.spec)
        spec["entries"][0]["bundled"][0]["patterns"] = ["linked.py"]
        with self.assertRaisesRegex(catalog.CatalogError, "symlink"):
            catalog.build(self.root, spec=spec)
        spec["entries"][0]["bundled"][0]["patterns"] = ["absent.py"]
        with self.assertRaisesRegex(catalog.CatalogError, "matches nothing"):
            catalog.build(self.root, spec=spec)

    def test_unknown_producers_or_unlisted_entrypoints_fail(self):
        spec = copy.deepcopy(self.spec)
        spec["entries"][0]["generated"][0]["producers"] = ["imagined_generator"]
        with self.assertRaisesRegex(catalog.CatalogError, "Unknown producer"):
            catalog.validate_spec(self.root, spec)
        spec = copy.deepcopy(self.spec)
        spec["commands"]["produce"]["argv"][-1] = "different.py"
        with self.assertRaisesRegex(catalog.CatalogError, "entry point"):
            catalog.validate_spec(self.root, spec)

    def test_dataset_registry_cannot_silently_change_loader_identity(self):
        (self.root / "cross_morse/__init__.py").write_text("# Changed provider loader\n")
        with self.assertRaisesRegex(catalog.CatalogError, "loader source identity changed"):
            catalog.build(self.root)

    def test_verified_flag_does_not_create_an_archive_or_public_download(self):
        write_json(self.root / "runs/index.json", {"schema": "cross-morse-runs-index-1",
                   "runs": [{"dataset": "ogbn-arxiv", "verified": True}]})
        with self.assertRaises(ValueError):
            catalog.registered_runs(self.root)
        registered_reference(self.root)
        rows = catalog.registered_runs(self.root)
        self.assertEqual(rows[0]["materialization"], "deposit_reference_metadata_verified")
        self.assertFalse(rows[0]["public_download_verified"])
        self.assertEqual(rows[0]["manuscript_relationship"]["artifact_class"], "benchmark_relationship_unspecified")
        manifest = self.root / "manifests/official/arxiv.json"
        changed = json.loads(manifest.read_text())
        changed["registration"]["archive_sha256"] = "9" * 64
        write_json(manifest, changed)
        with self.assertRaisesRegex(ValueError, "differs from the index"):
            catalog.registered_runs(self.root)

    def test_original_and_new_run_relationships_require_matching_identities(self):
        entry = registered_reference(self.root)
        path = self.root / "paper/bindings/arxiv_full.json"
        binding = {"schema_version": "result-binding-1", "status": "author_completed", "relationship_to_manuscript": "post_result_revision",
                   "run_identities": {**entry["run_identities"], "dataset": entry["dataset"]}}
        write_json(path, binding)
        self.assertEqual(catalog.binding_relationship(self.root, entry)["artifact_class"], "declared_new_reproduction")
        binding["relationship_to_manuscript"] = "exact_result_producing_run"
        write_json(path, binding)
        result = catalog.binding_relationship(self.root, entry)
        self.assertEqual(result["artifact_class"], "declared_original_benchmark")
        self.assertTrue(result["identity_checked"])
        self.assertFalse(result["historical_origin_authenticated"])
        binding["run_identities"]["config_sha256"] = "9" * 64
        write_json(path, binding)
        with self.assertRaisesRegex(catalog.CatalogError, "identity mismatch"):
            catalog.binding_relationship(self.root, entry)

    def test_registered_measurements_recheck_raw_files_and_index_identity(self):
        path = baseline_set(self.root / "records")
        with contextlib.redirect_stdout(io.StringIO()):
            cmd_index(self.root / "records")
        records = catalog.registered_records(self.root)
        self.assertEqual(records[0]["materialization"], "local_payload_and_value_trace_verified")
        self.assertEqual(records[0]["artifact_class"], "measurement_records_origin_unspecified")
        (path / "raw/fixture_timing.log").write_text("changed value\n")
        with self.assertRaisesRegex(ValueError, "changed after indexing"):
            catalog.registered_records(self.root)

    def test_catalog_check_detects_changed_bundled_bytes(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(catalog.main(["--root", str(self.root)]), 0)
            self.assertEqual(catalog.main(["--root", str(self.root), "--check"]), 0)
        (self.root / "run_cross_morse.py").write_text("# Changed producer source\n")
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(catalog.main(["--root", str(self.root), "--check"]), 2)

    def test_duplicate_and_nonfinite_json_are_rejected(self):
        path = self.root / "invalid.json"
        for content in ('{"a": 1, "a": 2}', '{"a": NaN}'):
            path.write_text(content)
            with self.subTest(content=content), self.assertRaises(catalog.CatalogError):
                catalog.read_json(path)

    def test_committed_spec_covers_statement_without_promising_all_figures(self):
        spec = catalog.read_json(PROJECT / catalog.SPEC)
        self.assertEqual({row["id"] for row in spec["entries"]}, catalog.CLAIMS)
        self.assertTrue(any("display" in row["scope"].lower() for row in spec["entries"]))
        for name in ("reconstruct", "reconstruction_example", "supplement"):
            self.assertEqual(spec["commands"][name]["mode"], "build")


if __name__ == "__main__":
    unittest.main()
