"""Registration fault paths: contextual reviews, replayed derivations and fresh report scope.

All measurements/documents/services below are private temporary test fixtures.
These checks do not create or certify any official experiment or deposit.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from scripts import readiness, record_mapping, records, register_manuscript_sources as sources, review

PROJECT = Path(__file__).resolve().parents[1]


class TemporaryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cm-registration-safety-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)


class ContextualSourceTests(TemporaryTest):
    def setUp(self):
        super().setUp()
        paper = self.root / "paper"
        paper.mkdir()
        self.paths = {role: self.root / f"{role}.pdf" for role in ("main", "supplementary")}
        for role, path in self.paths.items():
            path.write_bytes(b"%PDF-private-test-fixture-" + role.encode())
        self.source = {"documents": {role: {"sha256": "a" * 64, "byte_size": 1, "pages": 1}
                                     for role in self.paths},
                       "table_groups": {"table": {"source": "main", "pages": [1], "label": "Table 1"}}}
        (paper / "sources.json").write_text(json.dumps(self.source))
        (paper / "reported_results.json").write_text(json.dumps({"table": {"effect": -0.25, "AP": 0.58}}))
        (paper / "registered_constants.json").write_text(json.dumps({"threshold": 0.05}))
        for symbol, name in (("SOURCES", "sources.json"), ("REPORTED", "reported_results.json"),
                             ("CONSTANTS", "registered_constants.json"), ("CHECK", "transcription_check.json")):
            patch = mock.patch.object(sources, symbol, paper / name)
            patch.start()
            self.addCleanup(patch.stop)

        class ExtractorFixture:
            kind = "test-fixture"
            def __init__(self, path):
                self.path = path
            def pages(self):
                return 1
            def text(self, page):
                return "Table 1. Independent quantities elsewhere on the page: -0.25; 0.58."
        patch = mock.patch.object(sources, "Extractor", ExtractorFixture)
        patch.start()
        self.addCleanup(patch.stop)

    def inspected(self):
        return sources.report(self.paths["main"], self.paths["supplementary"])

    def reviewed(self, result):
        notes = sources.review_template(result)
        notes["reviewed_by"] = "Unit-test author"
        for value in notes["groups"].values():
            value.update(review_status="reviewed", note="Fixture contextual review; AP units and table meaning checked.")
        return notes

    def register(self, result=None, notes=None):
        result = result or self.inspected()
        notes = notes or self.reviewed(result)
        sources.register(result, pdf_paths=self.paths, review_notes=notes)

    def test_page_occurrence_is_signed_and_not_a_transcription_certificate(self):
        self.assertEqual(sources.numbers("1,138,216"), {1138216.0})
        self.assertFalse(sources.present(-0.25, {0.25}))
        result = self.inspected()
        self.assertEqual(result["spot_check"]["coverage"], 1.0)
        self.assertIs(result["spot_check"]["verifies_transcription"], False)
        self.assertIn("not a transcription accuracy rate", sources.markdown(result))
        notes = sources.review_template(result)
        notes["reviewed_by"] = "Unit-test author"
        with self.assertRaisesRegex(sources.SourceError, "contextual review"):
            self.register(result, notes)
        self.assertFalse(sources.CHECK.exists())

    def test_different_pdf_bytes_cannot_use_old_review_notes(self):
        result = self.inspected()
        notes = self.reviewed(result)
        original_sources = sources.SOURCES.read_bytes()
        self.paths["main"].write_bytes(self.paths["main"].read_bytes() + b" revised")
        with self.assertRaisesRegex(sources.SourceError, "inputs changed"):
            self.register(result, notes)
        self.assertEqual(sources.SOURCES.read_bytes(), original_sources)
        self.assertFalse(sources.CHECK.exists())
        with self.assertRaisesRegex(sources.SourceError, "review notes are stale"):
            self.register(self.inspected(), notes)

    def test_modified_registry_or_page_count_invalidates_an_author_review(self):
        self.register()
        self.assertEqual(sources.check_review_record(self.root)["reviewed_by"], "Unit-test author")
        old = sources.REPORTED.read_bytes()
        sources.REPORTED.write_bytes(old + b"\n")
        with self.assertRaisesRegex(sources.SourceError, "stale"):
            sources.check_review_record(self.root)
        sources.REPORTED.write_bytes(old)
        document = json.loads(sources.SOURCES.read_text())
        document["documents"]["main"]["pages"] = 2
        sources.SOURCES.write_text(json.dumps(document))
        with self.assertRaisesRegex(sources.SourceError, "stale"):
            sources.check_review_record(self.root)

    def test_source_pair_write_failure_restores_the_previous_record(self):
        self.register()
        old_source, old_check = sources.SOURCES.read_bytes(), sources.CHECK.read_bytes()
        original_write = sources._atomic_write
        def write(path, data):
            if path == sources.SOURCES:
                raise OSError("simulated source write failure")
            original_write(path, data)
        with mock.patch.object(sources, "_atomic_write", side_effect=write):
            with self.assertRaisesRegex(OSError, "source write failure"):
                self.register()
        self.assertEqual(sources.SOURCES.read_bytes(), old_source)
        self.assertEqual(sources.CHECK.read_bytes(), old_check)

    def test_legacy_search_only_record_does_not_count_as_contextual_review(self):
        sources.CHECK.write_text(json.dumps({"schema": "cross-morse-transcription-check-1",
                                            "pdf_sha256": {r: d["sha256"] for r, d in self.source["documents"].items()},
                                            "spot_check": None}))
        with self.assertRaisesRegex(sources.SourceError, "predates"):
            sources.check_review_record(self.root)

    def test_output_cannot_replace_a_pdf_or_completed_notes(self):
        with self.assertRaises(sources.SourceError):
            sources._output_path(self.paths["main"], list(self.paths.values()))
        notes = self.root / "review.json"
        notes.write_text("keep this review")
        with self.assertRaisesRegex(sources.SourceError, "already exists"):
            sources._output_path(notes, [], new=True)
        self.assertEqual(notes.read_text(), "keep this review")


class MappingIntegrityTests(TemporaryTest):
    def setUp(self):
        super().setUp()
        self.records = self.root / "records"
        self.records.mkdir()
        shutil.copyfile(PROJECT / "records/specs.json", self.records / "specs.json")
        self.raw = self.root / "outputs"
        self.raw.mkdir()
        datasets = ["ogbn-arxiv", "ogbn-proteins", "Roman-empire", "PascalVOC-SP", "COCO-SP", "Peptides-func"]
        rows = []
        for index, dataset in enumerate(datasets):
            (self.raw / f"{index}.json").write_text(json.dumps({"numerator": 6.0 + index, "denominator": 3.0, "seconds": 1.25 + index}))
            rows.append({"keys": {"dataset": dataset}, "quantity": "t_nom_seconds", "unit": "seconds",
                         "source_file": f"raw/{index}.json", "locator": "json:/seconds"})
        rows[0] = {"keys": {"dataset": datasets[0]}, "quantity": "t_nom_seconds", "unit": "seconds",
                   "derive": {"op": "ratio", "inputs": [{"source_file": "raw/0.json", "locator": "json:/numerator"},
                                                        {"source_file": "raw/0.json", "locator": "json:/denominator"}]}}
        self.mapping = {"schema": record_mapping.MAPPING_SCHEMA, "set": "nominal_time",
                        "raw_files": [{"from": "outputs", "to": "raw"}], "rows": rows,
                        "provenance": {key: "private unit-test fixture" for key in
                                       ("experiment", "produced_by", "hardware", "period", "protocol")}}
        self.path = self.root / "mapping.json"

    def build(self, *, replace=False):
        self.path.write_text(json.dumps(self.mapping))
        with contextlib.redirect_stdout(io.StringIO()):
            return record_mapping.build(self.path, records_root=self.records, replace=replace)

    def test_relative_inputs_resolve_beside_mapping_and_originals_are_unchanged(self):
        original = {p.name: p.read_bytes() for p in self.raw.glob("*.json")}
        target = self.build()
        for name, data in original.items():
            self.assertEqual((target / "raw" / name).read_bytes(), data)
            self.assertEqual((self.raw / name).read_bytes(), data)
        self.assertNotIn(str(self.raw), (target / "code/MAPPING.json").read_text())

    def test_standalone_records_cli_replays_mapped_derivations(self):
        self.build()
        result = subprocess.run([os.sys.executable, "-B", str(PROJECT / "scripts/records.py"), "check",
                                 "--root", str(self.records), "--specs", str(self.records / "specs.json")],
                                cwd=self.root, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_generated_paths_and_inline_values_are_refused(self):
        (self.raw / "derived").mkdir()
        (self.raw / "derived/row_00000.json").write_text("original file")
        with self.assertRaisesRegex(record_mapping.MappingError, "reserved"):
            self.build()
        shutil.rmtree(self.raw / "derived")
        self.mapping["rows"][1]["value"] = 99
        with self.assertRaisesRegex(record_mapping.MappingError, "inline values"):
            self.build()
        self.assertFalse((self.records / "nominal_time").exists())

    def test_symlinked_input_directory_member_is_refused(self):
        (self.raw / "alias.json").symlink_to(self.raw / "0.json")
        with self.assertRaisesRegex(record_mapping.MappingError, "symlinks"):
            self.build()

    def test_reindex_cannot_bless_an_altered_derived_value_and_matching_summary(self):
        target = self.build()
        derived = target / "raw/derived/row_00000.json"
        value = json.loads(derived.read_text())
        value["value"] = 7
        derived.write_text(json.dumps(value))
        with (target / "summary.csv").open(newline="") as stream:
            reader = csv.DictReader(stream)
            names, rows = reader.fieldnames, list(reader)
        rows[0]["value"] = "7"
        with (target / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=names)
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(record_mapping.MappingError, "does not recompute"):
            records.cmd_index(self.records, self.records / "specs.json")

    def test_reindex_rechecks_input_hashes_and_the_declared_operation(self):
        target = self.build()
        raw = target / "raw/0.json"
        original = raw.read_bytes()
        raw.write_bytes(original + b"\n")
        with self.assertRaisesRegex(record_mapping.MappingError, "input bytes"):
            records.cmd_index(self.records, self.records / "specs.json")
        raw.write_bytes(original)
        derived = target / "raw/derived/row_00000.json"
        doc = json.loads(derived.read_text())
        doc["op"] = "sum"
        derived.write_text(json.dumps(doc))
        with self.assertRaisesRegex(record_mapping.MappingError, "mapping declaration"):
            records.cmd_index(self.records, self.records / "specs.json")

    def test_replacement_rolls_back_if_index_commit_fails(self):
        target = self.build()
        before = {p.relative_to(target): p.read_bytes() for p in target.rglob("*") if p.is_file()}
        index_before = (self.records / "index.json").read_bytes()
        (self.raw / "0.json").write_text(json.dumps({"numerator": 12, "denominator": 3, "seconds": 1.25}))
        replace = os.replace
        def replace_with_fault(src, dst):
            if Path(dst) == self.records / "index.json":
                raise OSError("simulated index commit failure")
            return replace(src, dst)
        with mock.patch.object(record_mapping.os, "replace", side_effect=replace_with_fault):
            with self.assertRaisesRegex(OSError, "index commit failure"):
                self.build(replace=True)
        after = {p.relative_to(target): p.read_bytes() for p in target.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual((self.records / "index.json").read_bytes(), index_before)
        self.assertEqual(list(self.root.glob("cm-record-*")), [])

    def test_public_root_refuses_partial_or_private_override(self):
        self.path.write_text(json.dumps(self.mapping))
        with mock.patch.object(record_mapping, "ROOT", self.root):
            for flag in ("allow_partial", "allow_private"):
                with self.subTest(flag=flag), self.assertRaisesRegex(record_mapping.MappingError, "external staging"):
                    record_mapping.build(self.path, records_root=self.records, **{flag: True})


class ReviewFreshnessTests(TemporaryTest):
    def options(self):
        return {"index": PROJECT / "runs/index.json", "records": PROJECT / "records"}

    def complete_fixture(self):
        from tests.test_uplift_tools import write_complete_reports
        invocation = self.root / "invocation"
        invocation.mkdir()
        write_complete_reports(invocation)
        audit_path = invocation / "scientific-evidence-audit.json"
        audit = json.loads(audit_path.read_text())
        audit["inputs"] = {"repository_root": str(PROJECT), "run_index": str(self.options()["index"]),
                           "records_root": str(self.options()["records"]), "confirmation_campaign": None}
        audit_path.write_text(json.dumps(audit))
        steps = review.plan(invocation, offline=False, **self.options())
        saved = {"schema": "cross-morse-review-1", "offline": False, "verdict": "EVIDENCE COMPLETE",
                 "inputs_unchanged": True, "input_identity": review.input_identity(**self.options()),
                 "artifact_directory": str(invocation), "artifact_sha256": review.artifact_identities(invocation),
                 "steps": [{"name": step.name, "group": step.group, "status": "PASS", "exit": 0} for step in steps]}
        path = self.root / "review.json"
        path.write_text(json.dumps(saved))
        return path, invocation, saved

    def test_current_fixture_verdict_recomputes_and_blank_verdict_cannot_pass(self):
        path, _, _ = self.complete_fixture()
        self.assertEqual(review.validate_saved_review(path, **self.options())["verdict"], "EVIDENCE COMPLETE")
        path.write_text(json.dumps({"verdict": "EVIDENCE COMPLETE"}))
        with self.assertRaisesRegex(ValueError, "evidence-mode"):
            review.validate_saved_review(path, **self.options())

    def test_changed_artifact_and_manually_rehashed_failed_audit_are_rejected(self):
        path, invocation, saved = self.complete_fixture()
        audit_path = invocation / "scientific-evidence-audit.json"
        audit = json.loads(audit_path.read_text())
        audit["publication_ready"] = False
        audit_path.write_text(json.dumps(audit))
        with self.assertRaisesRegex(ValueError, "artifacts.*changed"):
            review.validate_saved_review(path, **self.options())
        saved["artifact_sha256"] = review.artifact_identities(invocation)
        path.write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError, "do not support"):
            review.validate_saved_review(path, **self.options())

    def test_report_for_other_roots_or_missing_steps_cannot_pass(self):
        path, _, saved = self.complete_fixture()
        with self.assertRaisesRegex(ValueError, "other selected inputs"):
            review.validate_saved_review(path, index=self.root / "other/index.json", records=PROJECT / "records")
        saved["steps"] = saved["steps"][:-1]
        path.write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError, "step records"):
            review.validate_saved_review(path, **self.options())

    def test_tool_builder_and_environment_changes_invalidate_fingerprint(self):
        for name in ("paper/build/table.py", "environment/requirements-cpu.lock", "Makefile"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("original")
        args = {"index": self.root / "runs/index.json", "records": self.root / "records", "root": self.root}
        before = review.input_identity(**args)
        for name in ("paper/build/table.py", "environment/requirements-cpu.lock", "Makefile"):
            path = self.root / name
            path.write_text("revised")
            self.assertNotEqual(review.input_identity(**args)["tooling_sha256"], before["tooling_sha256"])
            path.write_text("original")
        self.assertEqual(review.input_identity(**args), before)

    def test_advisory_patterns_do_not_override_missing_evidence(self):
        warning = readiness.item("baselines", "WARN", "repeated values", "inspect original records")
        passed = readiness.item("records", "PASS", "traced")
        self.assertTrue(readiness.is_ready([warning, passed]))
        self.assertFalse(readiness.is_ready([warning, readiness.item("records", "TODO", "missing")]))

    def test_a_study_json_is_not_a_campaign_validation(self):
        campaign = self.root / "campaign"
        campaign.mkdir()
        (campaign / "study.json").write_text("{}")
        args = argparse.Namespace(index=PROJECT / "runs/index.json", records=PROJECT / "records", budget_campaign=campaign)
        result = next(row for row in readiness.evidence_items(args) if row["item"] == "campaign")
        self.assertEqual(result["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
