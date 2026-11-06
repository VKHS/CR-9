"""Regressions for stale manuscripts and empty official comparisons."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.check_manuscript_sources import inspect_sources
from scripts.compare_registered_runs import inspect_index


class ManuscriptIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "paper/documents").mkdir(parents=True)
        self.source = {"schema_version": "paper-sources-1", "documents": {},
                       "table_groups": {"test": {"source": "main", "pages": [1],
                                                 "evidence_kind": "author_reported_measurements",
                                                 "label": "Test table", "note": "Synthetic source-index fixture."}}}
        for role in ("main", "supplementary"):
            raw = ("%PDF-1.7\nfixture-" + role).encode()
            path = f"paper/documents/{role}.pdf"
            (self.root / path).write_bytes(raw)
            self.source["documents"][role] = {"bundled_path": path, "sha256": hashlib.sha256(raw).hexdigest(), "byte_size": len(raw), "pages": 2}
        for name in ("reported_results.json", "registered_constants.json"):
            value = {"_provenance": {"main_text_pages": 2, "supplementary_pages": 2}}
            if name == "reported_results.json":
                value["test"] = {}
            (self.root / "paper" / name).write_text(json.dumps(value))
        self.save()

    def save(self):
        (self.root / "paper/sources.json").write_text(json.dumps(self.source))

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_source_identity_passes_and_modified_pdf_fails(self):
        report = inspect_sources(self.root)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["verification_mode"], "pdf_bytes_verified")
        self.assertTrue(all(report["pdf_bytes_verified"].values()))
        with (self.root / "paper/documents/main.pdf").open("ab") as handle:
            handle.write(b"changed")
        self.assertEqual(inspect_sources(self.root)["status"], "FAIL")

    def test_old_page_references_cannot_describe_shorter_revision(self):
        self.source["table_groups"]["test"]["pages"] = [20]
        self.save()
        self.assertEqual(inspect_sources(self.root)["status"], "FAIL")

    def test_external_pdf_path_is_rejected(self):
        self.source["documents"]["main"]["bundled_path"] = "../../somewhere.pdf"
        self.save()
        self.assertEqual(inspect_sources(self.root)["status"], "FAIL")

    def test_reference_only_sources_do_not_claim_pdf_byte_verification(self):
        for role, record in self.source["documents"].items():
            (self.root / record["bundled_path"]).unlink()
            record["bundled_path"] = None
        self.save()
        report = inspect_sources(self.root)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["verification_mode"], "reference_metadata_only")
        self.assertEqual(report["pdf_bytes_verified"], {"main": False, "supplementary": False})
        self.assertEqual(report["table_groups_checked"], 1)

    def test_missing_declared_pdf_is_an_error_even_with_a_matching_local_copy(self):
        path = self.root / "paper/documents/main.pdf"
        local = self.root / "external-main.pdf"
        path.rename(local)
        report = inspect_sources(self.root, {"main": local})
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("Missing PDF", " ".join(report["issues"]))

    def test_reference_only_mode_requires_an_explicit_distribution_declaration(self):
        del self.source["documents"]["main"]["bundled_path"]
        self.save()
        report = inspect_sources(self.root)
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("distribution declaration", " ".join(report["issues"]))

    def test_local_copy_adds_verification_and_modified_copy_fails(self):
        copies = {}
        for role, record in self.source["documents"].items():
            copies[role] = self.root / record["bundled_path"]
            record["bundled_path"] = None
        self.save()
        report = inspect_sources(self.root, copies)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["verification_mode"], "pdf_bytes_verified")
        copies["main"].write_bytes(b"%PDF-1.7\nother version")
        self.assertEqual(inspect_sources(self.root, copies)["status"], "FAIL")

    def test_invalid_reference_identity_fails_without_a_distributed_pdf(self):
        self.source["documents"]["main"].update(bundled_path=None, sha256="unrecorded")
        self.save()
        report = inspect_sources(self.root)
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("Invalid SHA-256", " ".join(report["issues"]))


class OfficialCoverageTests(unittest.TestCase):
    def test_missing_and_empty_indexes_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            self.assertTrue(inspect_index(path)[1])
            path.write_text(json.dumps({"schema": "cross-morse-runs-index-1", "runs": []}))
            entries, problems = inspect_index(path)
            self.assertEqual(entries, [])
            self.assertTrue(problems)

    def test_synthetic_entry_never_counts_even_with_verified_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "example").mkdir()
            row = {"dataset": "ogbn-arxiv", "path": "example", "verified": True,
                   "profile": "smoke", "scientific_run": False}
            path = root / "index.json"
            path.write_text(json.dumps({"schema": "cross-morse-runs-index-1", "runs": [row]}))
            self.assertEqual(inspect_index(path)[0], [])

    def test_index_path_escape_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            row = {"dataset": "ogbn-arxiv", "path": "../outside", "verified": True,
                   "profile": "full", "scientific_run": True}
            path.write_text(json.dumps({"schema": "cross-morse-runs-index-1", "runs": [row]}))
            entries, problems = inspect_index(path)
            self.assertEqual(entries, [])
            self.assertTrue(any("unsafe" in problem for problem in problems))

    def test_index_cannot_relabel_a_different_dataset_as_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run").mkdir()
            (root / "run/config.json").write_text(json.dumps({"dataset": "ogbn-proteins", "profile": "full", "scientific_run": True}))
            row = {"dataset": "ogbn-arxiv", "path": "run", "verified": True, "profile": "full", "scientific_run": True}
            path = root / "index.json"
            path.write_text(json.dumps({"schema": "cross-morse-runs-index-1", "runs": [row]}))
            entries, problems = inspect_index(path)
            self.assertEqual(entries, [])
            self.assertTrue(any("differs from index" in problem for problem in problems))


if __name__ == "__main__":
    unittest.main()
