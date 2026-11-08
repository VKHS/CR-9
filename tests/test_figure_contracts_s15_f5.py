"""Independent full-figure reference contracts of Supplementary Figure S15 and main Figure 5.

The contracts in paper/printed_display_values.json were transcribed and evaluated from the PDFs, not from a
generator. These tests build each figure from software fixtures whose inputs equal the printed values
(Tables S4, S6, S6b, S15, S23, S23b, S26, S41 and main Tables 1-2) and check that every registered target is
compared, that an omitted comparison cannot read as complete coverage, and that changed inputs are detected.
Every fixture value is a software input, never a benchmark observation.
"""
from __future__ import annotations

import copy
import csv
import json
import math
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

from cm_campaigns.contract import EXTERNAL
from paper.build.displays import analytical, main_paper
from paper.build.displays.framework import (DATASETS, RunView, comparison_coverage, compare,
                                            load_references)
from paper.build.displays.testing import build_one, context, fake_records

import tests.test_displays_analytical as analytical_fixture
import tests.test_displays_main_paper as main_fixture

ROOT = Path(__file__).resolve().parents[1]
REGISTRY, REFERENCE_METADATA, _ = load_references(ROOT)
G_S15 = "supplementary_figure_S15_spectral_fingerprint_and_headroom"
G_F5 = "main_figure_5_deployment_economics"

# ---------------------------------------------------------------------------- printed-value fixture for Figure 5
# Official scores in points (Table S15; Peptides AP x 100, q_C and deployed from Table S26). A returned
# comparator deploys q_C, so its deployed score is the q_C score.
SCORES = {"ogbn-arxiv": (72.93, 73.15, 73.22), "ogbn-proteins": (87.02, 87.15, 87.26),
          "ogbn-products": (87.59, 87.74, 87.74), "Roman-empire": (91.76, 92.37, 92.63),
          "PascalVOC-SP": (46.89, 47.61, 48.21), "COCO-SP": (31.60, 32.38, 33.03),
          "Peptides-func": (70.40, 71.40, 72.00)}
RETRAINED = {("ogbn-arxiv", "AMP"): 73.06, ("ogbn-arxiv", "ADMP-GNN"): 73.02, ("ogbn-proteins", "AMP"): 87.05,
             ("ogbn-proteins", "GRAMA"): 87.03, ("ogbn-products", "ADMP-GNN"): 87.57, ("ogbn-products", "GRAMA"): 87.49,
             ("Roman-empire", "Early-Exit GNN"): 92.24, ("Roman-empire", "ADMP-GNN"): 92.18,
             ("PascalVOC-SP", "Graph Mamba"): 47.41, ("PascalVOC-SP", "GRAMA"): 47.36,
             ("COCO-SP", "Graph Mamba"): 32.58, ("COCO-SP", "GRAMA"): 32.51,
             ("Peptides-func", "Graph Mamba"): 71.4, ("Peptides-func", "GRAMA"): 71.3}
# Timing anchors of Table S23b (seconds): throughput, training-time low and high. COCO-SP carries its own
# quotient; the printed bracket is PascalVOC-SP's, which the generator takes by the shared-anchor convention.
ANCHORS = {"ogbn-arxiv": (14.89, 4.32, 8.64), "ogbn-proteins": (9.22, 7.20, 14.40), "Roman-empire": (1.40, 0.68, 1.35),
           "PascalVOC-SP": (None, 0.52e-3, 1.04e-3), "COCO-SP": (None, 0.05e-3, 0.09e-3),
           "Peptides-func": (None, 0.40e-3, 0.79e-3)}


def printed_plan() -> dict:
    """The main-paper fixture plan with unrounded interval ends inside the printed Table 1 rounding that
    reproduce the printed Table 2 ranges (the ten-split Roman interval is represented by two splits)."""
    plan = copy.deepcopy(main_fixture.PLAN)
    plan["ogbn-arxiv"]["effect"] = (-0.0045, -0.00752, -0.0015)
    plan["Peptides-func"]["effect"] = (-0.00303, -0.00474, -0.0013)
    half = 0.00316 / 12.706                           # t_1 x |a - b| / 2 = 0.00316: interval [-0.00888, -0.00256]
    plan["Roman-empire"]["splits"] = [
        dict(effect=(-0.00572 - half, -0.0090, -0.0029), gain=(0.20, 0.05, 0.35), admission=(-0.0060, -0.0030, 1)),
        dict(effect=(-0.00572 + half, -0.0085, -0.0024), gain=(0.32, 0.15, 0.49), admission=(-0.0068, -0.0036, 1))]
    return plan


def set_scores(run: RunView, dataset: str) -> None:
    for member in run.members():
        path = member.root / "final/results_summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        for name, score in zip(("qN", "qC", "qD"), SCORES[dataset]):
            summary["models"][name].update(metric=score / 100.0, metric_percent=score)
        path.write_text(json.dumps(summary, indent=1), encoding="utf-8")


def printed_runs(base: Path) -> dict:
    plan, runs = printed_plan(), {}
    for dataset, entry in plan.items():
        if "splits" in entry:
            camp = base / "roman"
            splits = [main_fixture.synthetic_run(camp / f"split_{i:02d}", dataset, entry, split, i)
                      for i, split in enumerate(entry["splits"])]
            runs[dataset] = RunView(dataset, camp, splits=splits)
        else:
            runs[dataset] = main_fixture.synthetic_run(base / dataset, dataset, entry)
        set_scores(runs[dataset], dataset)
    return runs


def printed_baselines():
    rows = []
    for dataset in DATASETS:
        ap = main_fixture.PLAN[dataset]["metric"] == "ap"
        for method in EXTERNAL[dataset]:
            score = RETRAINED[dataset, method]
            rows.append({"dataset": dataset, "method": method, "seed": "", "quantity": "serial_cost",
                         "value": main_fixture.PLAN[dataset]["E"] * 1.3, "unit": "nominal executions"})
            rows += [{"dataset": dataset, "method": method, "seed": str(seed), "quantity": "score",
                      "value": score / 100.0 if ap else score, "unit": "AP" if ap else "%"} for seed in (1, 2, 3)]
    return fake_records("baselines", rows)


def timing_anchors():
    rows = []
    for dataset, (through, low, high) in ANCHORS.items():
        values = {"training_anchor_low_seconds": low, "training_anchor_high_seconds": high}
        if through is not None:
            values["throughput_anchor_seconds"] = through
        rows += [{"dataset": dataset, "quantity": q, "value": v, "unit": "s"} for q, v in values.items()]
    return fake_records("timing_anchors", rows)


def leaf_counts(group: str) -> Counter:
    from paper.build.displays.framework import _reference_leaves
    return Counter(path[0] for path, _ in _reference_leaves(REGISTRY[group], include_text=True))


def read_rows(writer, name):
    with (writer.out_dir / "data" / writer.stem / f"{name}.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class ContractRegistration(unittest.TestCase):
    def test_groups_are_bound_to_their_printed_pages(self):
        self.assertEqual(REFERENCE_METADATA[G_S15], {"display_id": "supplementary:Figure S15",
                                                     "source": {"document": "supplementary", "pages": [35]}})
        self.assertEqual(REFERENCE_METADATA[G_F5], {"display_id": "main:Figure 5",
                                                    "source": {"document": "main", "pages": [12]}})
        self.assertEqual(analytical.G_FS15, G_S15)
        self.assertEqual(main_paper.G_F5, G_F5)
        self.assertEqual(leaf_counts(G_S15), {"panel_a": 84, "panel_b": 57, "panel_c": 42, "panel_d": 28})
        self.assertEqual(leaf_counts(G_F5), {"panel_a": 129, "panel_b": 31, "panel_c": 36, "panel_d": 72})

    def test_every_series_and_family_is_enumerated(self):
        s15, f5 = REGISTRY[G_S15], REGISTRY[G_F5]
        self.assertEqual(list(s15["panel_a"]["curves"]), [f"A{i}" for i in range(8)])
        self.assertTrue(all(len(c["response"]) == 7 for c in s15["panel_a"]["curves"].values()))
        self.assertEqual([s15["panel_a"]["curves"][a]["zero_crossing_lambda"] for a in ("A6", "A7")], [None, None])
        self.assertEqual({a: len(row) for a, row in s15["panel_b"]["sensitivity"].items()},
                         {f"A{i}": 6 for i in range(1, 8)})
        self.assertEqual(set(s15["panel_c"]["actions"]), {f"A{i}" for i in range(8)})
        self.assertEqual(s15["panel_d"]["dataset_order"], list(DATASETS))
        self.assertEqual(set(f5["panel_a"]["datasets"]), set(DATASETS))
        for dataset in DATASETS:
            self.assertEqual(list(f5["panel_a"]["datasets"][dataset]["retrained"]), list(EXTERNAL[dataset]))
        self.assertTrue(f5["panel_a"]["datasets"]["ogbn-products"]["deployed"]["returned_comparator"])
        shown = [d for d in DATASETS if d != "ogbn-products"]
        self.assertEqual(f5["panel_b"]["dataset_order"], shown)
        self.assertEqual(list(f5["panel_c"]["expected_cost"]), shown)
        self.assertEqual(list(f5["panel_c"]["assurance_cost"]), list(main_paper.ASSURANCE_ROWS))
        self.assertEqual(list(f5["panel_d"]["curves"]), shown)
        self.assertEqual(list(f5["panel_d"]["operating_points"]), shown)

    def test_generators_compare_but_never_read_the_targets(self):
        import inspect
        for function in (analytical.figure_s15, main_paper.figure_5):
            source = inspect.getsource(function)
            for token in ("printed_display_values", "load_references", "reported_results", "REGISTRY"):
                self.assertNotIn(token, source, f"{function.__name__} reads {token}")

    def test_formula_targets_restate_the_printed_functions(self):
        # Spot checks by hand, independent of cm_theory: g_A0 = 1 - 0.7 lambda; g_a(1) = identity coefficient;
        # A7 executed range 0.5[(1 - 0.25) + 0.25 sum_t t 0.08 0.92^t].
        s15 = REGISTRY[G_S15]
        self.assertAlmostEqual(s15["panel_a"]["curves"]["A0"]["response"]["lambda_5_3"]["$value"], 1 - 0.7 * 5 / 3, 8)
        self.assertAlmostEqual(s15["panel_a"]["curves"]["A0"]["zero_crossing_lambda"]["$value"], 1 / 0.7, 8)
        for action, values in s15["panel_c"]["actions"].items():
            self.assertAlmostEqual(s15["panel_a"]["curves"][action]["response"]["lambda_1"]["$value"],
                                   values["identity_coefficient"]["$value"], 8)
        hops = sum(t * 0.08 * 0.92 ** t for t in range(9))
        self.assertAlmostEqual(s15["panel_c"]["actions"]["A7"]["executed_range"]["$value"], 0.5 * (0.75 + 0.25 * hops), 8)
        # Definition 2 at the printed inputs: M* t_nom = 3600 C_qual / (theta dEC), Pascal on S41 and S23.
        f5 = REGISTRY[G_F5]
        k = 3600 * 1.59 / (0.1 * (10.298 - 8.0))
        self.assertAlmostEqual(f5["panel_d"]["curves"]["PascalVOC-SP"]["log10_M_star_at_domain_start"]["$value"],
                               math.log10(k / 1e-4), 8)
        self.assertAlmostEqual(f5["panel_c"]["assurance_cost"]["ogbn-arxiv"]["zero_crossing_lambda"]["$value"],
                               4.5 / 4.7, 8)


# ============================================================================ Supplementary Figure S15
S15_REPRESENTATIVE = {
    "panel_a": [("panel_a", "curves", "A3", "response", "lambda_4_3"), ("panel_a", "curves", "A6", "zero_crossing_lambda"),
                ("panel_a", "anchor_lambdas"), ("panel_a", "series"), ("panel_a", "truncation_order"),
                ("panel_a", "zero_line")],
    "panel_b": [("panel_b", "sensitivity", "A7"), ("panel_b", "band_edges"), ("panel_b", "reference_action")],
    "panel_c": [("panel_c", "actions", "A5", "identity_label"), ("panel_c", "actions", "A0", "max_reach"),
                ("panel_c", "nominal_level")],
    "panel_d": [("panel_d", "datasets", "PascalVOC-SP", "unrealized_lower_bound_mnats"),
                ("panel_d", "datasets", "ogbn-products", "realized_effect_mnats"), ("panel_d", "dataset_order")],
}


class FigureS15Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.ctx = analytical_fixture.build_context(root / "inputs", complete=True)
        cls.writer, cls.rows = build_one("supplementary:Figure S15", cls.ctx, root / "out", registry=REGISTRY)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def coverage(self, comparisons):
        return comparison_coverage("supplementary:Figure S15", comparisons, REGISTRY, REFERENCE_METADATA,
                                   scope="measurement")

    def test_every_target_is_compared_and_reproduced_at_printed_precision(self):
        missing, coverage = self.coverage(self.writer.comparisons)
        self.assertEqual(missing, [])
        self.assertTrue(coverage["complete"])
        self.assertEqual(coverage["expected_values"], 211)
        mine = [row for row in self.rows if row["group"] == G_S15]
        self.assertEqual(len(mine), 144)
        self.assertEqual({row["status"] for row in self.rows}, {"exact"})

    def test_dropping_one_comparison_per_panel_breaks_coverage(self):
        for panel, paths in S15_REPRESENTATIVE.items():
            for path in paths:
                with self.subTest(panel=panel, path=path):
                    retained = [c for c in self.writer.comparisons if not (c.group == G_S15 and c.path == path)]
                    self.assertEqual(len(retained), len(self.writer.comparisons) - 1)
                    missing, coverage = self.coverage(retained)
                    self.assertFalse(coverage["complete"])
                    self.assertTrue(any(row["group"] == G_S15 and row["path"].startswith("/".join(path))
                                        for row in missing))

    def test_changed_response_function_fails_its_independent_curve_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            shifted = analytical.spectral.action_response
            with mock.patch.object(analytical.spectral, "action_response",
                                   side_effect=lambda action, lam: shifted(action, lam) + 0.001):
                _, rows = build_one("supplementary:Figure S15", self.ctx, Path(tmp), registry=REGISTRY)
        differing = {row["path"] for row in rows if row["group"] == G_S15 and row["status"] == "differs"}
        self.assertEqual(differing, {f"panel_a/curves/A{i}/response/{key}" for i in range(8)
                                     for key in analytical.FS15_ANCHORS})

    def test_headroom_bars_follow_the_measured_bound_and_effect(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = analytical_fixture.build_context(Path(tmp) / "inputs", complete=True)
            ctx.require_run("Roman-empire").evidence["final"]["paired_loss"]["mean"] = -0.0040
            ctx.require_run("COCO-SP").evidence["entry"]["headroom"]["upper"] = -0.0300
            _, rows = build_one("supplementary:Figure S15", ctx, Path(tmp) / "out", registry=REGISTRY)
        status = {row["path"]: row["status"] for row in rows if row["group"] == G_S15}
        self.assertEqual(status["panel_d/datasets/Roman-empire/realized_effect_mnats"], "differs")
        self.assertEqual(status["panel_d/datasets/Roman-empire/unrealized_lower_bound_mnats"], "differs")
        self.assertEqual(status["panel_d/datasets/Roman-empire/headroom_bound_mnats"], "exact")
        self.assertEqual(status["panel_d/datasets/COCO-SP/headroom_bound_mnats"], "differs")
        self.assertEqual(status["panel_d/datasets/COCO-SP/unrealized_lower_bound_mnats"], "differs")
        self.assertTrue(all(s == "exact" for p, s in status.items() if not p.startswith("panel_d/datasets/")))

    def test_anchor_evaluations_are_exported_beside_the_rendered_curve(self):
        anchors = read_rows(self.writer, "panel_a_anchor_response")
        self.assertEqual(len(anchors), 8 * 7)
        crossings = {row["action"]: row["zero_crossing_lambda"] for row in anchors}
        self.assertEqual((crossings["A6"], crossings["A7"]), ("", ""))
        self.assertAlmostEqual(float(crossings["A2"]), 1 / 0.85)
        self.assertEqual(len(read_rows(self.writer, "panel_a_response")), 401)


# ============================================================================ main Figure 5
F5_REPRESENTATIVE = {
    "panel_a": [("panel_a", "datasets", "ogbn-products", "deployed", "returned_comparator"),
                ("panel_a", "datasets", "COCO-SP", "retrained", "GRAMA", "relative_cost_range"),
                ("panel_a", "datasets", "Peptides-func", "plant_score_points"),
                ("panel_a", "marker_roles", "retrained")],
    "panel_b": [("panel_b", "rows", "Peptides-func", "metric_points_per_unit"), ("panel_b", "dataset_order")],
    "panel_c": [("panel_c", "assurance_cost", "PascalVOC-SP", "zero_crossing_lambda"),
                ("panel_c", "expected_cost", "COCO-SP", "net_benefit_at_lambda_10_mnats"),
                ("panel_c", "price_domain_mnats")],
    "panel_d": [("panel_d", "curves", "Roman-empire", "log10_M_star_at_domain_end"),
                ("panel_d", "operating_points", "COCO-SP", "label_datasets"),
                ("panel_d", "operating_points", "Peptides-func", "t_nom_anchor_bracket_seconds"), ("panel_d", "theta")],
}
BRACKET_FIELDS = ("t_nom_anchor_bracket_seconds", "log10_M_star_at_anchor_bracket")


class Figure5Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.tmp.name)
        cls.runs = printed_runs(cls.base / "runs")
        cls.records = {"input_gate": main_fixture.input_gate(), "baselines": printed_baselines(),
                       "nominal_time": main_fixture.nominal_time(), "timing_anchors": timing_anchors()}
        cls.writer, cls.rows = cls.build(cls.records, "complete")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def build(cls, records, name):
        ctx = context(runs=cls.runs, records=records)
        return build_one("main:Figure 5", ctx, cls.base / "out" / name, registry=REGISTRY)

    def coverage(self, comparisons):
        return comparison_coverage("main:Figure 5", comparisons, REGISTRY, REFERENCE_METADATA, scope="measurement")

    def test_every_target_is_compared_and_reproduced_at_printed_precision(self):
        missing, coverage = self.coverage(self.writer.comparisons)
        self.assertEqual(missing, [])
        self.assertTrue(coverage["complete"])
        self.assertEqual(coverage["expected_values"], 268)
        mine = [row for row in self.rows if row["group"] == G_F5]
        self.assertEqual(len(mine), 227)
        self.assertEqual([(row["path"], row["computed"]) for row in mine if row["status"] != "exact"], [])
        self.assertEqual(self.writer.missing_content, [])
        # The related-table checks that the generator already made agree with the same fixture.
        related = [row for row in self.rows if row["group"] in (main_paper.T2_GROUP, main_paper.S15_GROUP,
                                                                 main_paper.S23B_GROUP)]
        self.assertTrue(related)
        self.assertEqual({row["status"] for row in related}, {"exact"})

    def test_dropping_one_comparison_per_panel_breaks_coverage(self):
        for panel, paths in F5_REPRESENTATIVE.items():
            for path in paths:
                with self.subTest(panel=panel, path=path):
                    retained = [c for c in self.writer.comparisons if not (c.group == G_F5 and c.path == path)]
                    self.assertEqual(len(retained), len(self.writer.comparisons) - 1)
                    missing, coverage = self.coverage(retained)
                    self.assertFalse(coverage["complete"])
                    self.assertTrue(any(row["group"] == G_F5 and row["path"].startswith("/".join(path))
                                        for row in missing))

    def test_absent_timing_anchors_leave_the_printed_brackets_missing(self):
        records = {k: v for k, v in self.records.items() if k != "timing_anchors"}
        writer, rows = self.build(records, "no_anchors")
        missing = {row["path"] for row in rows if row["group"] == G_F5 and row["status"] == "missing"}
        shown = [d for d in DATASETS if d != "ogbn-products"]
        self.assertEqual(missing, {f"panel_d/operating_points/{d}/{field}" for d in shown for field in BRACKET_FIELDS})
        self.assertEqual({row["status"] for row in rows if row["group"] == G_F5 and row["path"] not in missing}, {"exact"})
        self.assertEqual([item["location"] for item in writer.missing_content], ["data/Fig5/panel_d_t_nom_brackets"])
        # The display-level comparison state counts a missing comparison as incomplete.
        states = Counter(row["status"] for row in compare(writer.comparisons, REGISTRY))
        self.assertEqual(states["missing"], 2 * len(shown))

    def test_executed_timing_is_compared_with_the_s23b_anchor_not_substituted(self):
        timing = fake_records("nominal_time", [{"dataset": d, "quantity": "t_nom_seconds",
                                                "value": 6.0 if d == "ogbn-arxiv" else t, "unit": "s"}
                                               for d, t in main_fixture.T_NOM.items()])
        _, rows = self.build({**self.records, "nominal_time": timing}, "measured_timing")
        differing = {row["path"] for row in rows if row["group"] == G_F5 and row["status"] == "differs"}
        self.assertEqual(differing, {"panel_d/operating_points/ogbn-arxiv/t_nom_seconds",
                                     "panel_d/operating_points/ogbn-arxiv/M_star_label_mantissa"})

    def test_changed_measured_inputs_change_their_dependent_targets(self):
        gate = fake_records("input_gate", [{"dataset": d, "quantity": "fallback_probability",
                                            "value": 0.30 if d == "PascalVOC-SP" else p}
                                           for d, p in main_fixture.FALLBACK.items()])
        _, rows = self.build({**self.records, "input_gate": gate}, "changed_fallback")
        differing = {row["path"] for row in rows if row["group"] == G_F5 and row["status"] == "differs"}
        self.assertIn("panel_a/datasets/PascalVOC-SP/deployed/expected_relative_cost", differing)
        self.assertIn("panel_b/rows/PascalVOC-SP/lambda_star_mnats", differing)
        self.assertIn("panel_c/expected_cost/PascalVOC-SP/zero_crossing_lambda", differing)
        self.assertIn("panel_d/curves/PascalVOC-SP/log10_M_star_at_domain_start", differing)
        self.assertFalse([p for p in differing if "PascalVOC-SP" not in p])
        # The two-branch assurance configuration does not depend on the fallback probability.
        self.assertNotIn("panel_c/assurance_cost/PascalVOC-SP/zero_crossing_lambda", differing)
        self.assertNotIn("panel_a/datasets/PascalVOC-SP/deployed/assurance_relative_cost", differing)

    def test_returned_comparator_is_drawn_as_the_deployed_rule_without_new_deployed_rows(self):
        panel_a = read_rows(self.writer, "panel_a")
        self.assertEqual(Counter(row["policy"] for row in panel_a),
                         {"qC": 7, "deployed": 6, "retrained": 14, "returned": 1})
        products = {row["policy"]: row for row in panel_a if row["dataset"] == "ogbn-products"}
        for column in ("relative_cost", "score_gain_points"):
            self.assertAlmostEqual(float(products["returned"][column]), float(products["qC"][column]))
        self.assertAlmostEqual(float(products["returned"]["relative_cost_high"]), 1.05)
        brackets = {row["dataset"]: row for row in read_rows(self.writer, "panel_d_t_nom_brackets")}
        self.assertEqual(brackets["COCO-SP"]["anchor_reference_dataset"], "PascalVOC-SP")
        self.assertAlmostEqual(float(brackets["ogbn-arxiv"]["t_nom_high_seconds"]), 14.89)


if __name__ == "__main__":
    unittest.main()
