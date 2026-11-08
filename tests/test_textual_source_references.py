"""Printed SI text is distinct from preserved repository explanations.

Expected wording was visually checked against the registered supplement on
pages 10, 11, 15 and 25. These are source-text checks, not empirical validation.
"""
from __future__ import annotations

import json
from pathlib import Path
import unittest

from paper.build.displays.framework import _reference_leaves


ROOT = Path(__file__).resolve().parents[1]


class TextualSourceReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reported = json.loads((ROOT / "paper/reported_results.json").read_text())

    def assert_context_is_metadata(self, row, field, old_text):
        metadata_key = f"_repository_{field}_note"
        self.assertIn("metadata rather than printed source text", row[metadata_key])
        self.assertTrue(row[metadata_key].endswith(old_text))
        targets = dict(_reference_leaves(row, include_text=True))
        self.assertIn((field,), targets)
        self.assertNotIn((metadata_key,), targets)

    def test_s14b_peptides_preserves_the_printed_holm_qualification(self):
        rows = self.reported["supplementary_table_S14b_separated_data_sensitivity"]
        row = next(row for row in rows if row["contrast"] == "Peptides/radius")
        self.assertEqual(
            row["interpretation"],
            "interval excludes zero; Holm P above 0.05 under this model; "
            "the confirmation record decides the label (Note S26)",
        )
        self.assert_context_is_metadata(
            row, "interpretation",
            "Analytical confirmation-size sensitivity; frozen-rule outcome follows the S14 confirmation record",
        )

    def test_s14b_all_six_unchanged_cells_use_the_printed_text(self):
        rows = self.reported["supplementary_table_S14b_unchanged_contrasts"]
        self.assertEqual({row["contrast"] for row in rows}, {
            "arxiv/state", "arxiv/radius", "proteins/state", "proteins/radius", "products/state", "Roman/RH",
        })
        self.assertEqual(len(rows), 6)
        for row in rows:
            with self.subTest(contrast=row["contrast"]):
                self.assertEqual(row["interpretation"], "unchanged")
                old = ("unchanged (not tested against the margin)" if row["contrast"] == "products/state"
                       else "unchanged (Holm value re-formed)")
                self.assert_context_is_metadata(row, "interpretation", old)

    def test_s20_peptides_uses_the_printed_boundary_description(self):
        row = self.reported["supplementary_table_S20_selection_theorem_instantiation"]["Peptides-func"]
        self.assertEqual(row["interpretation"], "boundary graph-level selection")
        self.assert_context_is_metadata(row, "interpretation", "fragile graph-level selection")

    def test_s42_comparison_units_preserve_added_counts_only_as_metadata(self):
        rows = self.reported["supplementary_table_S42_power_of_budget_matched_test"]
        for dataset, count in (("PascalVOC-SP", 425), ("COCO-SP", 5665), ("Peptides-func", 544)):
            with self.subTest(dataset=dataset):
                self.assertEqual(rows[dataset]["unit"], "comparison graphs (fixed ensemble)")
                self.assert_context_is_metadata(
                    rows[dataset], "unit",
                    f"{count} D_compare graphs (fixed ensemble; analytical power model)",
                )


if __name__ == "__main__":
    unittest.main()
