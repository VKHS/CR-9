"""Missing evidence must stay visible; joint uncertainty must use paired units.

All numerical inputs here are temporary software fixtures, never manuscript data.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from paper.build.displays import framework as fw
from paper.build.displays.factorial_data import CELLS, joint_factorial_contrasts
from paper.build.displays.testing import fake_records
from scripts import reconstruct_paper as command

ROOT = Path(__file__).resolve().parents[1]


class ComparisonIntegrity(unittest.TestCase):
    def build(self, generator, registry, *, display_id="main:Table 1", scope="measurement"):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        ctx = fw.DisplayContext()
        spec = fw.Generator(display_id, "fixture", "Temporary software fixture", ("static",), generator, scope)
        with mock.patch.object(fw, "_load_generators"), mock.patch.dict(fw.GENERATORS, {display_id: spec}):
            return fw.build_displays(ctx, Path(temporary.name), registry=registry, only=[display_id])["displays"][0]

    def test_omitted_required_comparison_remains_in_denominator(self):
        def partial(ctx, out):
            out.table(["value"], [[1]])
            out.compare("main_table_1_fixture", ("a",), 1)
        row = self.build(partial, {"main_table_1_fixture": {"a": 1, "b": 2}})
        self.assertEqual(row["status"], "built")
        self.assertTrue(row["content_complete"])
        self.assertFalse(row["comparison_complete"])
        self.assertEqual(row["comparison"]["missing"], 1)
        self.assertEqual(row["comparison_coverage"]["expected_values"], 2)

    def test_none_is_missing_but_explicit_not_applicable_is_not(self):
        def supplied(ctx, out):
            out.table(["value", "not applicable"], [[None, fw.not_applicable("printed dash")]])
            out.compare("main_table_1_fixture", ("value",), None)
            out.compare("main_table_1_fixture", ("na",), None)
        row = self.build(supplied, {"main_table_1_fixture": {"value": 1, "na": None}})
        self.assertFalse(row["content_complete"])
        self.assertEqual(len(row["missing_content"]), 1)
        self.assertEqual(len(row["not_applicable_cells"]), 1)
        self.assertEqual(row["comparison"]["missing"], 1)
        self.assertEqual(row["comparison"]["exact"], 1)

    def test_missing_reference_is_distinct_from_explicit_null(self):
        rows = fw.compare([fw.Comparison("g", ("null",), None), fw.Comparison("g", ("absent",), None)],
                          {"g": {"null": None}})
        self.assertEqual([row["status"] for row in rows], ["exact", "no_reference"])

    def test_partial_array_and_nonfinite_value_do_not_satisfy_comparison(self):
        def invalid(ctx, out):
            out.table(["value"], [[float("nan")]])
            out.compare("main_table_1_fixture", ("interval",), [1])
        row = self.build(invalid, {"main_table_1_fixture": {"interval": [1, 2]}})
        self.assertFalse(row["content_complete"])
        self.assertFalse(row["comparison_complete"])
        self.assertEqual(row["comparison"]["not_comparable"], 1)

    def test_related_table_checks_do_not_certify_a_whole_figure(self):
        extra, coverage = fw.comparison_coverage("main:Figure 2", [fw.Comparison("main_table_1_fixture", ("a",), 1)],
                                                 {"main_table_1_fixture": {"a": 1}}, {}, scope="measurement")
        self.assertFalse(coverage["complete"])
        self.assertEqual(extra[0]["status"], "missing")

    def test_typed_precision_preserves_trailing_zeroes(self):
        self.assertEqual(fw.compare_value(0.0204, {"$value": 0.020, "$decimals": 3}), "exact")
        self.assertEqual(fw.compare_value(0.024, {"$value": 0.020, "$decimals": 3}), "differs")

    def test_reference_sidecar_must_bind_current_pdf_and_display(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "paper").mkdir()
            documents = {"main": {"sha256": "a" * 64}}
            files = {"reported_results.json": {}, "sources.json": {"documents": documents},
                     "display_manifest.json": {"displays": [{"id": "main:Table 1", "source": "main", "pages": [2]}]},
                     "printed_display_values.json": {"schema": fw.PRINTED_SCHEMA, "documents": {"main": "a" * 64},
                        "groups": {"main_table_1_fixture": {"display_id": "main:Table 1",
                            "source": {"document": "main", "pages": [2]}, "values": {"p": {"$value": .02, "$decimals": 3}, "na": None}}}}}
            for name, value in files.items():
                (root / "paper" / name).write_text(json.dumps(value))
            values, metadata, hashes = fw.load_references(root)
            self.assertEqual(len(hashes), 2)
            self.assertEqual(metadata["main_table_1_fixture"]["display_id"], "main:Table 1")
            bad = files["printed_display_values.json"]
            bad["documents"]["main"] = "b" * 64
            (root / "paper/printed_display_values.json").write_text(json.dumps(bad))
            with self.assertRaisesRegex(fw.DisplayValidationError, "current registered PDFs"):
                fw.load_references(root)


class JointFactorialIntegrity(unittest.TestCase):
    def setUp(self):
        from cm_experiments.factorial import cell_statistics
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        unit_noise = np.arange(12) * .002
        # Common noise cancels from every contrast. Independence would invent
        # positive-width intervals even though these paired contrasts are fixed.
        self.units = {cell: .5 + unit_noise - .004 * int(cell[0]) - .002 * int(cell[1])
                      - .001 * int(cell[2]) for cell in CELLS}
        stats, contrasts, draws = cell_statistics(self.units, np.ones(12), ("A", "O", "X"), reps=128, seed=7)
        np.savez(self.root / "unit_dil.npz", unit_id=np.arange(12), unit_weight=np.ones(12),
                 joint_bootstrap_cell_means=draws, **{f"cell{cell}_unit_dil": v for cell, v in self.units.items()})
        self.raw = {"schema": "cm-factorial-2", "dataset": "ogbn-arxiv", "factors": ["A", "O", "X"],
                    "cells": [{"cell": cell, **stats[cell]} for cell in CELLS], "contrasts": contrasts,
                    "bootstrap": {"reps": 128, "seed": 7},
                    "unit_dil": {"file": "unit_dil.npz", "sha256": hashlib.sha256((self.root / "unit_dil.npz").read_bytes()).hexdigest()}}
        (self.root / "factorial.json").write_text(json.dumps(self.raw))
        self.cells = stats
        self.records = fake_records("factorials", [{"dataset": "ogbn-arxiv", "cell": c,
            "quantity": "ld_integrated_loss", "value": stats[c]["ld_integrated_loss"], "source_file": "factorial.json"}
            for c in CELLS], root=self.root)

    def test_joint_draws_preserve_covariance_in_intervals(self):
        checked = joint_factorial_contrasts(self.records, "ogbn-arxiv", self.cells)
        effect = checked["contrasts"][(0,)]
        self.assertAlmostEqual(effect["effect"], -.004)
        self.assertLess(effect["high"] - effect["low"], 1e-12)
        self.assertGreater(self.cells["000"]["interval_high"] - self.cells["000"]["interval_low"], .001)
        self.assertEqual(checked["replicates"], 128)

    def test_changed_raw_draws_are_rejected_even_if_the_hash_is_updated(self):
        with np.load(self.root / "unit_dil.npz") as source:
            arrays = {key: source[key] for key in source.files}
        arrays["joint_bootstrap_cell_means"][0, 0] += .01
        np.savez(self.root / "unit_dil.npz", **arrays)
        self.raw["unit_dil"]["sha256"] = hashlib.sha256((self.root / "unit_dil.npz").read_bytes()).hexdigest()
        (self.root / "factorial.json").write_text(json.dumps(self.raw))
        with self.assertRaisesRegex(fw.DisplayValidationError, "shared resampling stream"):
            joint_factorial_contrasts(self.records, "ogbn-arxiv", self.cells)

    def test_summary_only_inputs_do_not_create_contrast_intervals(self):
        self.records.root = None
        self.assertIsNone(joint_factorial_contrasts(self.records, "ogbn-arxiv", self.cells))

    def test_older_tabular_raw_source_retains_point_only_fallback(self):
        (self.root / "cells.csv").write_text("cell,mean\n000,0.5\n")
        for row in self.records.rows:
            row["source_file"] = "cells.csv"
        self.assertIsNone(joint_factorial_contrasts(self.records, "ogbn-arxiv", self.cells))

    def test_summary_change_is_not_hidden_by_a_valid_raw_bundle(self):
        self.cells["000"]["ld_integrated_loss"] += .001
        with self.assertRaisesRegex(fw.DisplayValidationError, "differs from its summary"):
            joint_factorial_contrasts(self.records, "ogbn-arxiv", self.cells)


class OneCommandComparisonGate(unittest.TestCase):
    def result(self):
        return {"displays": [{"id": "main:Table 1", "full_numerical_display": True, "content_complete": True}],
                "display_generators": {"comparison_with_printed_values": {"exact": 1},
                    "comparison_coverage_complete": True, "errors": {}, "module_errors": {}}}

    def run_command(self, result):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(command.artifacts, "load_manifest", return_value={"artifacts": []}), \
                 mock.patch.object(command.artifacts, "required_artifacts", return_value=[]), \
                 mock.patch.object(command.artifacts, "verify_fetched", return_value={"artifacts": [], "awaiting": []}) as verify, \
                 mock.patch.object(command, "reconstruct", return_value=result):
                code = command.main(["--offline", "--artifacts", str(root / "artifacts"), "--output-dir", str(root / "paper")])
                verify.assert_called_once()
                return code

    def test_every_incomplete_comparison_state_stops_success(self):
        self.assertEqual(self.run_command(self.result()), 0)
        for state in ("differs", "not_comparable", "no_reference", "missing"):
            with self.subTest(state=state):
                result = self.result()
                result["display_generators"]["comparison_with_printed_values"][state] = 1
                self.assertEqual(self.run_command(result), 1)

    def test_missing_coverage_metadata_and_missing_content_stop_success(self):
        result = self.result()
        result["display_generators"].pop("comparison_coverage_complete")
        self.assertEqual(self.run_command(result), 1)
        result = self.result()
        result["displays"][0]["content_complete"] = False
        self.assertEqual(self.run_command(result), 1)

    def test_optional_pending_artifact_does_not_block_required_reconstruction(self):
        status = command.summarize(self.result(), {"artifacts": [{"id": "optional", "required": False, "status": "pending"}]})
        self.assertEqual(status["artifacts_awaiting_publication"], [])


if __name__ == "__main__":
    unittest.main()
