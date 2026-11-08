"""Independent printed-value contracts of Supplementary Figures S9 and S13.

Every input here is a temporary software fixture, never benchmark evidence. Where a fixture
equals the printed values, every comparison of the dedicated figure contract is exact; the
remaining tests show that a dropped comparison, a changed input or an inconsistent label stays
visible against the independently registered targets.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

from cm_campaigns.confirmation import FAMILY
from paper.build.displays import run_audits
from paper.build.displays.framework import (DATASETS, RunView, compare_value, comparison_coverage,
                                           load_references)
from paper.build.displays.testing import build_one, context, fake_records, fake_run

ROOT = Path(__file__).resolve().parents[1]
REGISTRY, REFERENCE_METADATA, _ = load_references(ROOT)
S9_ID, S13_ID = "supplementary:Figure S9", "supplementary:Figure S13"
S9, S13 = run_audits.S9_FIGURE_GROUP, run_audits.S13_FIGURE_GROUP


def data_rows(out_dir, stem, name):
    with (Path(out_dir) / "data" / stem / f"{name}.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def own(rows, group):
    return {row["path"]: row for row in rows if row["group"] == group}


class ContractRegistrationTests(unittest.TestCase):
    def test_groups_are_bound_to_their_figure_pages_and_document_their_sources(self):
        sidecar = json.loads((ROOT / "paper/printed_display_values.json").read_text(encoding="utf-8"))["groups"]
        for group, display_id, page in ((S9, S9_ID, 33), (S13, S13_ID, 34)):
            with self.subTest(group=group):
                self.assertEqual(REFERENCE_METADATA[group], {"display_id": display_id,
                                                             "source": {"document": "supplementary", "pages": [page]}})
                entry = sidecar[group]
                self.assertTrue(entry["supporting_sources"] and entry["evidence_kind"] and entry["note"])
                # The coverage denominator is the dedicated contract alone.
                _, coverage = comparison_coverage(display_id, [], REGISTRY, REFERENCE_METADATA, scope="measurement")
                self.assertEqual(coverage["expected_values"], {S9: 97, S13: 83}[group])
                self.assertFalse(coverage["complete"])


class FigureS9ContractTests(unittest.TestCase):
    # Table S14b at J_old for the three multigraph contrasts, with the graph shares of its caption.
    SENSITIVITY = {"Pascal/RH": {"units_old": 1428, "effect_nats": -0.00902, "ci_low_nats": -0.01355,
                                 "ci_high_nats": -0.00449, "graph_variance_share": 0.030},
                   "COCO/RH": {"units_old": 5000, "effect_nats": -0.00962, "ci_low_nats": -0.01372,
                               "ci_high_nats": -0.00552, "graph_variance_share": 0.015},
                   "Peptides/radius": {"units_old": 2331, "effect_nats": -0.00172, "ci_low_nats": -0.00316,
                                       "ci_high_nats": -0.00028, "graph_variance_share": 0.071}}

    def setUp(self):
        from tests.test_s14_confirmation import S14CampaignTests
        name = "test_complete_family_reconstructs_effects_and_returns_failed_peptides_to_comparator"
        self.campaign = S14CampaignTests(name)
        self.campaign.setUp()
        self.addCleanup(self.campaign.tearDown)
        self.builds = 0

    def build(self, registry=REGISTRY, **changes):
        rows = []
        for contrast, values in self.SENSITIVITY.items():
            # The record contract also carries holm_p and the label-survival probability; the figure reads neither.
            values = {**values, "holm_p": 0.5, "label_survival_probability": 0.5, **changes.get(contrast, {})}
            rows += [{"contrast": contrast, "quantity": quantity, "value": value} for quantity, value in values.items()]
        ctx = context(campaigns={"confirmation": self.campaign.root},
                      records={"confirmation_sensitivity": fake_records("confirmation_sensitivity", rows)})
        self.builds += 1
        out = self.campaign.base / f"displays_{self.builds}"
        writer, compared = build_one(S9_ID, ctx, out, registry=registry)
        return writer, compared, out

    def test_every_drawn_element_is_a_registered_target_and_printed_inputs_are_exact(self):
        writer, compared, out = self.build()
        missing, coverage = comparison_coverage(S9_ID, writer.comparisons, REGISTRY, REFERENCE_METADATA,
                                                scope="measurement")
        self.assertEqual(missing, [])
        self.assertTrue(coverage["complete"])
        self.assertEqual(coverage["expected_values"], 97)
        rows = own(compared, S9)
        self.assertEqual({row["status"] for row in rows.values()}, {"exact"})
        self.assertEqual({row["status"] for row in compared}, {"exact"})     # S14b table checks agree too
        contract = REGISTRY[S9]["panel_a"]["rows"]
        self.assertEqual([row["label"] for row in contract], list(FAMILY))
        self.assertEqual([row["label"] for row in contract if row["rescaled"] is not None],
                         ["Pascal/RH", "COCO/RH", "Peptides/radius"])
        self.assertEqual(contract[7]["holm_P"], "<0.001")                    # a printed inequality stays text
        self.assertEqual(rows["panel_a/rows/7/holm_P"]["computed"], "< 0.001")
        self.assertEqual(len(contract[4]["margin_ticks_nats"]), 2)          # products/state: ±0.002
        self.assertEqual(rows["panel_a/rows/5/rescaled"]["computed"], None)  # no blue interval is drawn
        self.assertEqual(rows["panel_a/rows/6/annotation"]["computed"], "J: 1428 → 425; Holm P 0.0013")
        self.assertIn("figures/FigS9.png", writer.files)

    def test_dropping_any_element_family_leaves_coverage_incomplete(self):
        writer, _, _ = self.build()
        paths = [("panel_a", "label"), ("panel_a", "title"), ("panel_a", "x_label"), ("panel_a", "legend"),
                 ("panel_a", "zero_line_nats"), ("panel_a", "rows", 0, "label"), ("panel_a", "rows", 1, "J_old"),
                 ("panel_a", "rows", 6, "J_new"), ("panel_a", "rows", 2, "effect_nats"),
                 ("panel_a", "rows", 3, "original_interval_nats"), ("panel_a", "rows", 5, "rescaled"),
                 ("panel_a", "rows", 6, "rescaled", "interval_nats"), ("panel_a", "rows", 8, "rescaled", "effect_nats"),
                 ("panel_a", "rows", 7, "holm_P"), ("panel_a", "rows", 4, "margin_ticks_nats"),
                 ("panel_a", "rows", 8, "annotation")]
        for path in paths:
            with self.subTest(path=path):
                retained = [item for item in writer.comparisons if not (item.group == S9 and item.path == path)]
                self.assertEqual(len(retained), len(writer.comparisons) - 1)
                missing, coverage = comparison_coverage(S9_ID, retained, REGISTRY, REFERENCE_METADATA,
                                                        scope="measurement")
                self.assertFalse(coverage["complete"])
                self.assertTrue(missing and all(row["group"] == S9 and row["status"] == "missing" for row in missing))
                self.assertTrue(all(row["path"].startswith("/".join(map(str, path))) for row in missing))

    def test_rescaled_intervals_follow_the_two_way_model_not_the_campaign_interval(self):
        # The supplied campaign's own D_confirm interval is separate source data, not the plotted sensitivity.
        self.campaign.change_result("Pascal/RH", interval_nats=[-0.0150, -0.0030])
        _, compared, out = self.build()
        rows = own(compared, S9)
        self.assertEqual(rows["panel_a/rows/6/rescaled/interval_nats"]["status"], "exact")
        pascal = data_rows(out, "FigS9", "panel_a")[6]
        self.assertEqual((float(pascal["campaign_low_nats"]), float(pascal["campaign_high_nats"])), (-0.015, -0.003))
        # A different graph share changes the rescaled interval only.
        _, compared, _ = self.build(**{"Pascal/RH": {"graph_variance_share": 0.30}})
        rows = own(compared, S9)
        self.assertEqual(rows["panel_a/rows/6/rescaled/interval_nats"]["status"], "differs")
        self.assertEqual(rows["panel_a/rows/6/original_interval_nats"]["status"], "exact")
        self.assertEqual(rows["panel_a/rows/6/rescaled/effect_nats"]["status"], "exact")
        # A changed J_old interval moves both families of the row.
        _, compared, _ = self.build(**{"Peptides/radius": {"ci_low_nats": -0.0040}})
        rows = own(compared, S9)
        self.assertEqual(rows["panel_a/rows/8/original_interval_nats"]["status"], "differs")
        self.assertEqual(rows["panel_a/rows/8/rescaled/interval_nats"]["status"], "differs")
        self.assertEqual(rows["panel_a/rows/7/rescaled/interval_nats"]["status"], "exact")

    def test_margin_ticks_sit_at_the_registered_margin_with_both_sides_for_equivalence(self):
        _, _, out = self.build()
        source = {row["contrast"]: row for row in data_rows(out, "FigS9", "panel_a")}
        self.assertEqual(source["products/state"]["margin_ticks_nats"], "-0.002 0.002")
        self.assertEqual(source["arxiv/radius"]["margin_ticks_nats"], "0.0015")
        self.assertEqual(source["Pascal/RH"]["margin_ticks_nats"], "0.002")
        # The beneficial-direction threshold remains in the source data.
        self.assertEqual(float(source["arxiv/radius"]["signed_margin_nats"]), -0.0015)
        rows = REGISTRY[S9]["panel_a"]["rows"]
        self.assertEqual(compare_value([-0.0015], rows[1]["margin_ticks_nats"]), "differs")
        self.assertEqual(compare_value([0.002], rows[1]["margin_ticks_nats"]), "differs")
        self.assertEqual(compare_value([0.002], rows[4]["margin_ticks_nats"]), "not_comparable")
        self.assertEqual(compare_value([-0.002, 0.002], rows[4]["margin_ticks_nats"]), "exact")

    def test_targets_are_never_generator_inputs(self):
        _, with_targets, first = self.build()
        _, without, second = self.build(registry={})
        self.assertEqual({row["status"] for row in with_targets}, {"exact"})
        self.assertEqual({row["status"] for row in without}, {"no_reference"})
        self.assertEqual([(row["path"], row["computed"]) for row in with_targets],
                         [(row["path"], row["computed"]) for row in without])
        self.assertEqual((first / "data/FigS9/panel_a.csv").read_bytes(), (second / "data/FigS9/panel_a.csv").read_bytes())


# main Table 1: deployed route (None: q_C returned), ensemble, paired effect and 95% interval (nats)
TABLE_1 = {"ogbn-arxiv": ("radius", 4, -0.0045, (-0.0075, -0.0015)),
           "ogbn-proteins": ("radius", 4, -0.0038, (-0.0068, -0.0008)),
           "ogbn-products": (None, 2, 0.0, (0.0, 0.0)),
           "Roman-empire": ("mpc_rh", 4, -0.00572, (-0.0089, -0.0026)),
           "PascalVOC-SP": ("mpc_rh", 8, -0.00842, (-0.0128, -0.0041)),
           "COCO-SP": ("mpc_rh", 8, -0.00905, (-0.0127, -0.0054)),
           "Peptides-func": ("radius", 4, -0.00303, (-0.0047, -0.0013))}
ROMAN_SPLITS = 10
T_9 = 2.262            # two-sided 95% t quantile at nine degrees of freedom (paper.build.evidence)


def run_files(route, ensemble, mean, low, high):
    """Only the files Figure S13 reads: candidate ledger, comparator, deployed index and paired summary."""
    candidates = [{"route": "fixed_sequence", "ensemble": ensemble, "is_open_loop": True},
                  {"route": route or "radius", "ensemble": ensemble, "is_open_loop": False}]
    deployed = 0 if route is None else 1
    return {"select/candidates.json": candidates,
            "select/decision.json": {"comparator_index": 0, "provisional_index": deployed},
            "admit/decision.json": {"deployed_index": deployed},
            "final/results_summary.json": {"paired_qD_minus_qC_loss": {
                "mean": mean, "lower": low, "upper": high, "method": "software fixture"}}}


def printed_runs(base: Path, **override) -> dict:
    """Seven runs whose deployed rules and paired effects equal main Table 1 (Roman: ten-split campaign)."""
    runs = {}
    for dataset in DATASETS:
        route, ensemble, mean, (low, high) = override.get(dataset, TABLE_1[dataset])
        if dataset == "Roman-empire":
            # Ten split means whose small-sample t interval is mean ± 0.00315, i.e. [-0.0089, -0.0026] printed.
            sd = 0.00315 * math.sqrt(ROMAN_SPLITS) / T_9
            step = sd * math.sqrt((ROMAN_SPLITS - 1) / ROMAN_SPLITS)
            splits = [fake_run(base / "roman" / f"split_{k:02d}", dataset,
                               run_files(route, ensemble, mean + (step if k % 2 else -step), low, high))
                      for k in range(ROMAN_SPLITS)]
            runs[dataset] = RunView(dataset, base / "roman", splits=splits)
        else:
            runs[dataset] = fake_run(base / dataset, dataset, run_files(route, ensemble, mean, low, high))
    return runs


class FigureS13ContractTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.base = Path(directory.name)
        self.builds = 0

    def build(self, runs):
        self.builds += 1
        return build_one(S13_ID, context(runs=runs), self.base / f"out_{self.builds}", registry=REGISTRY)

    def test_structural_and_measured_panels_are_covered_and_printed_inputs_are_exact(self):
        writer, compared = self.build(printed_runs(self.base / "printed"))
        missing, coverage = comparison_coverage(S13_ID, writer.comparisons, REGISTRY, REFERENCE_METADATA,
                                                scope="measurement")
        self.assertEqual(missing, [])
        self.assertTrue(coverage["complete"])
        self.assertEqual(coverage["expected_values"], 83)
        rows = own(compared, S13)
        self.assertEqual({row["status"] for row in rows.values()}, {"exact"})
        self.assertEqual({row["status"] for row in compared}, {"exact"})    # main Table 1 checks agree too
        self.assertEqual(rows["panel_c/rows/2/label"]["computed"], ["products", "returns q_C"])
        self.assertIsNone(rows["panel_c/rows/2/interval_nats"]["computed"])
        self.assertEqual(rows["panel_b/families/1/title"]["computed"], "feedback valuable (six datasets)")
        self.assertIs(rows["panel_b/families/0/common_risk_minimizer"]["computed"], True)
        self.assertIs(rows["panel_b/families/1/common_risk_minimizer"]["computed"], False)
        self.assertTrue({"figures/FigS13.png", "data/FigS13/panel_a.csv", "data/FigS13/panel_b.csv",
                         "data/FigS13/panel_b_markers.csv", "data/FigS13/panel_c.csv"} <= set(writer.files))

    def test_dropping_one_comparison_in_any_panel_leaves_coverage_incomplete(self):
        writer, _ = self.build(printed_runs(self.base / "printed"))
        paths = [("panel_a", "label"), ("panel_a", "title"), ("panel_a", "settings", 0, "name"),
                 ("panel_a", "settings", 1, "description"), ("panel_a", "settings", 2, "backbone_box"),
                 ("panel_a", "settings", 2, "adapter_box"),
                 ("panel_b", "label"), ("panel_b", "x_label"), ("panel_b", "families", 0, "title"),
                 ("panel_b", "families", 0, "x_tick_labels"), ("panel_b", "families", 1, "y_label"),
                 ("panel_b", "families", 1, "annotations"), ("panel_b", "families", 1, "common_risk_minimizer"),
                 ("panel_c", "label"), ("panel_c", "title"), ("panel_c", "legend"), ("panel_c", "x_label"),
                 ("panel_c", "zero_line_nats"), ("panel_c", "rows", 0, "category"), ("panel_c", "rows", 6, "label"),
                 ("panel_c", "rows", 3, "effect_nats"), ("panel_c", "rows", 4, "interval_nats"),
                 ("panel_c", "rows", 2, "interval_nats")]
        for path in paths:
            with self.subTest(path=path):
                retained = [item for item in writer.comparisons if not (item.group == S13 and item.path == path)]
                self.assertEqual(len(retained), len(writer.comparisons) - 1)
                missing, coverage = comparison_coverage(S13_ID, retained, REGISTRY, REFERENCE_METADATA,
                                                        scope="measurement")
                self.assertFalse(coverage["complete"])
                self.assertTrue(missing and all(row["group"] == S13 for row in missing))
                self.assertTrue(all(row["path"].startswith("/".join(map(str, path))) for row in missing))

    def test_family_titles_follow_the_measured_outcomes(self):
        # Peptides returning q_C must move it into the schedule-adequate family, not only change panel c.
        runs = printed_runs(self.base / "peptides_returns", **{"Peptides-func": (None, 4, 0.0, (0.0, 0.0))})
        _, compared = self.build(runs)
        rows = own(compared, S13)
        self.assertEqual(rows["panel_b/families/0/title"]["computed"], "schedule adequate (products, Peptides)")
        self.assertEqual(rows["panel_b/families/1/title"]["computed"], "feedback valuable (five datasets)")
        for path in ("panel_b/families/0/title", "panel_b/families/1/title", "panel_c/rows/6/label",
                     "panel_c/rows/6/category", "panel_c/rows/6/effect_nats"):
            self.assertEqual(rows[path]["status"], "differs", path)
        self.assertEqual(rows["panel_c/rows/6/interval_nats"]["status"], "missing")
        self.assertTrue(all(row["status"] == "exact" for path, row in rows.items() if path.startswith("panel_a")))

    def test_synthetic_lifecycle_runs_keep_complete_coverage_and_report_their_differences(self):
        from tests.test_displays_run_audits import Fixture
        runs = Fixture(self).runs(comparator=("ogbn-products",))
        writer, compared = self.build(runs)
        _, coverage = comparison_coverage(S13_ID, writer.comparisons, REGISTRY, REFERENCE_METADATA,
                                          scope="measurement")
        self.assertTrue(coverage["complete"])               # every target compared ...
        rows = own(compared, S13)
        self.assertEqual(rows["panel_c/rows/0/label"]["computed"], ["arxiv", "radius, E=1"])
        self.assertEqual(rows["panel_c/rows/0/label"]["status"], "differs")     # ... but synthetic E differs
        self.assertEqual(rows["panel_c/rows/3/effect_nats"]["status"], "differs")
        self.assertEqual(rows["panel_c/rows/2/interval_nats"]["status"], "exact")   # comparator identity
        self.assertTrue(all(row["status"] == "exact" for path, row in rows.items() if path.startswith("panel_a")))
        self.assertNotIn("no_reference", {row["status"] for row in compared})


if __name__ == "__main__":
    unittest.main()
