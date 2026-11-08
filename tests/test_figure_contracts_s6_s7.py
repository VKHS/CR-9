"""Independent printed-value contracts of supplementary Figures S6 and S7 (supplementary p. 32).

The display inputs are the synthetic fixtures of tests/test_displays_analytical.py, whose measured
inputs (Table S23 costs, Table S21 action-gap quantiles) equal the printed values. The reference
targets are re-derived here from inputs typed from the paper (Note S19, Tables S5, S21, S23 and S32,
main Proposition 9) and from the provider dataset statistics that the paper's r = E/(N d) uses; no
generator output serves as a reference. Passing these checks establishes software behavior on
synthetic fixtures, not a reproduction of the original figures.
"""
from __future__ import annotations

import itertools
import json
import math
import tempfile
import unittest
from pathlib import Path
from statistics import NormalDist
from unittest import mock

from paper.build.displays import analytical
from paper.build.displays.framework import ROOT, _reference_leaves, compare_value, comparison_coverage
from paper.build.displays.testing import build_one
from tests.test_displays_analytical import REFERENCE_METADATA, REGISTRY, build_context, data_rows

S6, S7 = "supplementary:Figure S6", "supplementary:Figure S7"
G6, G7 = analytical.G_FS6, analytical.G_FS7
ADMITTED = analytical.ADMITTED
SIDECAR = json.loads((ROOT / "paper/printed_display_values.json").read_text(encoding="utf-8"))

# ----------------------------------------------------------------------------- printed inputs, typed by hand
# Provider statistics (N, E) and the Table S5 hidden width d (SI p. 3).
DIMENSIONS = {"ogbn-arxiv": (169343, 1166243, 512), "ogbn-proteins": (132534, 39561252, 512),
              "Roman-empire": (22662, 32927, 384), "PascalVOC-SP": (479.40, 2710.48, 128),
              "COCO-SP": (476.88, 2693.67, 128), "Peptides-func": (150.94, 307.30, 128)}
S23_COSTS = {"ogbn-arxiv": (4.700, 4.000), "ogbn-proteins": (4.800, 4.000), "Roman-empire": (5.400, 4.000),
             "PascalVOC-SP": (10.500, 8.000), "COCO-SP": (10.800, 8.000), "Peptides-func": (4.900, 4.000)}
# Figure S6b labels (25 %, 50 %), Note S19 s, registered T and the largest order shown.
PRINTED_LABELS = {"ogbn-arxiv": ("36", "18"), "ogbn-proteins": ("3", "2"), "Roman-empire": (">64", ">64"),
                  "PascalVOC-SP": ("25", "13"), "COCO-SP": ("29", "15"), "Peptides-func": ("43", "22")}
S, T_REG, T_AXIS, SHARES = 0.06, 8, 64, (0.25, 0.50)
COST_HALF_UNIT = 0.0005
# Table S21 / Figure S7b legend (mnats), Table S32 threshold (0.002 nats), one-decimal legend rounding.
S21_MNATS = {"ogbn-arxiv": (6.2, 1.1), "ogbn-proteins": (5.7, 1.0), "Roman-empire": (4.9, 0.8),
             "PascalVOC-SP": (5.4, 0.9), "COCO-SP": (5.6, 1.0), "Peptides-func": (3.6, 0.5)}
SHORT = {"ogbn-arxiv": "arxiv", "ogbn-proteins": "proteins", "Roman-empire": "Roman", "PascalVOC-SP": "Pascal",
         "COCO-SP": "COCO", "Peptides-func": "Peptides"}
TAU_MNATS, GAP_HALF_UNIT = 2.0, 0.05
Z95 = NormalDist().inv_cdf(0.95)


def sparse_ratio(dataset, digits=None):
    nodes, edges, width = DIMENSIONS[dataset]
    r = edges / (nodes * width)
    return r if digits is None else round(r, digits)


def multiplier(r, f, t):
    return (1 + r + S + f * r * (t - 1)) / (1 + r)


def admissible(m, r, f):
    return 1 + (m * (1 + r) - (1 + r + S)) / (f * r)


def fraction(m, r, t):
    return (m * (1 + r) - (1 + r + S)) / (r * (t - 1))


def lognormal(median, p5):
    """mu in log nats and sigma of the log-normal through the median and the 5th percentile (both in mnats)."""
    return math.log(median / 1e3), (math.log(median) - math.log(p5)) / Z95


def log_gap_density(x, median, p5):
    """Density per unit natural-log gap at x (mnats)."""
    mu, sigma = lognormal(median, p5)
    return math.exp(-0.5 * ((math.log(x / 1e3) - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


def cost_shift(dataset, function):
    c, q = S23_COSTS[dataset]
    nominal = function(c / q)
    return max(abs(function((c + dc) / (q + dq)) - nominal)
               for dc, dq in itertools.product((-COST_HALF_UNIT, COST_HALF_UNIT), repeat=2))


def gap_shift(dataset, function, steps=40):
    median, p5 = S21_MNATS[dataset]
    nominal = function(median, p5)
    offsets = [GAP_HALF_UNIT * (2 * i / steps - 1) for i in range(steps + 1)]
    return max(abs(function(median + a, p5 + b) - nominal) for a in offsets for b in offsets)


def statuses(rows, group):
    return {row["path"]: row["status"] for row in rows if row["group"] == group}


class FigureS6S7Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.ctx = build_context(root / "inputs", complete=True)
        cls.out = root / "displays"
        cls.results = {display_id: build_one(display_id, cls.ctx, cls.out, registry=REGISTRY) for display_id in (S6, S7)}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def coverage(self, display_id, comparisons):
        return comparison_coverage(display_id, comparisons, REGISTRY, REFERENCE_METADATA, scope="measurement")

    def assertTightest(self, reference, shift, label):
        """The stored precision covers the propagated input rounding and is the tightest decimal that does."""
        decimals = reference["$decimals"]
        self.assertGreaterEqual(0.5 * 10.0 ** -decimals, shift, label)
        if decimals < 6:
            self.assertLess(0.5 * 10.0 ** -(decimals + 1), shift, label)

    def build(self, display_id, ctx, name):
        return build_one(display_id, ctx, Path(self.tmp.name) / name, registry=REGISTRY)

    # -- registration and coverage ---------------------------------------------------------------
    def test_each_figure_has_one_dedicated_page_bound_group(self):
        for display_id, group in ((S6, G6), (S7, G7)):
            with self.subTest(display=display_id):
                self.assertEqual([name for name, meta in REFERENCE_METADATA.items() if meta["display_id"] == display_id],
                                 [group])
                self.assertEqual(REFERENCE_METADATA[group]["source"], {"document": "supplementary", "pages": [32]})
                entry = SIDECAR["groups"][group]
                for key in ("supporting_sources", "evidence_kind", "note", "source_inputs", "formulas", "precision"):
                    self.assertTrue(entry.get(key), key)
        self.assertEqual(SIDECAR["groups"][G7]["supporting_sources"][0], {"document": "main", "pages": [8],
                                                                           "label": "Proposition 9: retain when the gap "
                                                                                    "exceeds 2 L_J r_ℓ"})

    def test_coverage_is_complete_with_the_expected_target_counts(self):
        expected = {S6: (G6, 78, {"model": 1, "panel_a": 27, "panel_b": 27, "panel_c": 23}),
                    S7: (G7, 60, {"panel_a": 11, "panel_b": 49})}
        for display_id, (group, total, panels) in expected.items():
            with self.subTest(display=display_id):
                writer, _ = self.results[display_id]
                missing, coverage = self.coverage(display_id, writer.comparisons)
                self.assertEqual(missing, [])
                self.assertTrue(coverage["complete"])
                self.assertEqual((coverage["expected_values"], coverage["covered_values"]), (total, total))
                self.assertEqual({panel: sum(1 for _ in _reference_leaves(value, include_text=True))
                                  for panel, value in REGISTRY[group].items()}, panels)

    def test_fixture_comparisons_are_exact(self):
        related = {S6: analytical.G_S6C, S7: analytical.G_S21}
        for display_id, group, rows_expected in ((S6, G6, 75), (S7, G7, 57)):
            with self.subTest(display=display_id):
                _, rows = self.results[display_id]
                self.assertEqual({row["status"] for row in rows}, {"exact"})
                self.assertEqual(len([row for row in rows if row["group"] == group]), rows_expected)
                self.assertEqual({row["group"] for row in rows}, {group, related[display_id]})

    def test_every_curve_family_member_scenario_and_marker_is_enumerated(self):
        s6 = REGISTRY[G6]
        for family in (s6["panel_a"]["curves"], s6["panel_a"]["measured_multipliers"], s6["panel_b"]["bars"],
                       REGISTRY[G7]["panel_b"]["curves"]):
            self.assertEqual(list(family), list(ADMITTED))
            self.assertNotIn("ogbn-products", family)
        self.assertTrue(all(set(anchors) == {"T1", "T8", "T64"} for anchors in s6["panel_a"]["curves"].values()))
        self.assertTrue(all(set(bars) == {"25", "50"} for bars in s6["panel_b"]["bars"].values()))
        self.assertEqual(list(s6["panel_c"]["points"]), [str(t) for t in range(2, 17)])
        self.assertEqual(set(REGISTRY[G7]["panel_a"]["regions"]), {"retain", "acquire"})

    def test_omitting_one_comparison_per_panel_leaves_a_coverage_gap(self):
        targets = {
            S6: (G6, [("model", "sensing_share_s"),
                      ("panel_a", "resolvent_step_share_percent"), ("panel_a", "truncation_order_domain"),
                      ("panel_a", "curves", "Peptides-func", "T64"), ("panel_a", "curves", "ogbn-proteins", "T8"),
                      ("panel_a", "measured_multipliers", "COCO-SP"),
                      ("panel_b", "resolvent_step_shares_percent"), ("panel_b", "registered_T_line"),
                      ("panel_b", "bars", "Roman-empire", "50", "bar_height_T"),
                      ("panel_b", "bars", "ogbn-arxiv", "25", "label"),
                      ("panel_c", "dataset"), ("panel_c", "registered_reduced_sensor", "sketch_width_k"),
                      ("panel_c", "multiplier_m"), ("panel_c", "registered_T_line"),
                      ("panel_c", "f_res_bound_label"), ("panel_c", "truncation_order_domain"),
                      ("panel_c", "points", "2"), ("panel_c", "points", "16")]),
            S7: (G7, [("panel_a", "boundary_slope"), ("panel_a", "radius_domain"),
                      ("panel_a", "boundary_gap_at_domain_ends"), ("panel_a", "x_axis"), ("panel_a", "y_axis"),
                      ("panel_a", "regions", "retain", "label"), ("panel_a", "regions", "acquire", "side"),
                      ("panel_b", "x_axis"), ("panel_b", "y_axis"), ("panel_b", "p5_standard_normal_quantile"),
                      ("panel_b", "threshold_tau_mnats"), ("panel_b", "threshold_label"),
                      ("panel_b", "gap_domain_mnats"), ("panel_b", "curves", "COCO-SP", "legend_label"),
                      ("panel_b", "curves", "Peptides-func", "sigma_log"),
                      ("panel_b", "curves", "ogbn-arxiv", "density_at_tau")]),
        }
        for display_id, (group, paths) in targets.items():
            writer, _ = self.results[display_id]
            for path in paths:
                with self.subTest(display=display_id, path=path):
                    retained = [c for c in writer.comparisons if not (c.group == group and c.path == path)]
                    self.assertEqual(len(retained), len(writer.comparisons) - 1)
                    missing, coverage = self.coverage(display_id, retained)
                    self.assertFalse(coverage["complete"])
                    self.assertTrue(missing)
                    self.assertTrue(all(row["group"] == group and row["path"].startswith("/".join(path))
                                        for row in missing))

    def test_an_omitted_dataset_or_scenario_cannot_claim_complete_coverage(self):
        cases = ((S6, G6, lambda path: "Peptides-func" in path, 8),           # three anchors, dotted line, four bar targets
                 (S6, G6, lambda path: path[:2] == ("panel_b", "bars") and path[3] == "25", 12),
                 (S6, G6, lambda path: path[:2] == ("panel_c", "points"), 15),
                 (S7, G7, lambda path: "Roman-empire" in path, 7),
                 (S7, G7, lambda path: path[0] == "panel_a", 11))
        for display_id, group, drop, count in cases:
            with self.subTest(display=display_id, count=count):
                writer, _ = self.results[display_id]
                retained = [c for c in writer.comparisons if not (c.group == group and drop(c.path))]
                missing, coverage = self.coverage(display_id, retained)
                self.assertFalse(coverage["complete"])
                self.assertEqual(coverage["missing_values"], count)

    # -- independence and precision of the targets -------------------------------------------------
    def test_s6_targets_are_independent_formula_evaluations_at_justified_precision(self):
        values = REGISTRY[G6]
        inputs = SIDECAR["groups"][G6]["source_inputs"]
        self.assertEqual(values["model"]["sensing_share_s"], {"$value": S, "$decimals": 2})
        self.assertEqual((inputs["sensing_share_s"], inputs["resolvent_step_shares"], inputs["registered_truncation_order_T"],
                          inputs["truncation_axis_T"]), (S, list(SHARES), T_REG, T_AXIS))
        self.assertEqual(values["panel_a"]["truncation_order_domain"], [1, T_AXIS])
        for dataset in ADMITTED:
            with self.subTest(dataset=dataset):
                nodes, edges, width = DIMENSIONS[dataset]
                c, q = S23_COSTS[dataset]
                self.assertEqual(inputs["datasets"][dataset], {"N": nodes, "E": edges, "d": width, "C_ctrl": c, "C_qC": q})
                r, m = sparse_ratio(dataset), c / q
                for key, t in (("T1", 1), ("T8", T_REG), ("T64", T_AXIS)):
                    reference = values["panel_a"]["curves"][dataset][key]
                    self.assertAlmostEqual(reference["$value"], multiplier(r, 0.5, t), places=8)
                    self.assertEqual(reference["$decimals"], 6)
                dotted = values["panel_a"]["measured_multipliers"][dataset]
                self.assertAlmostEqual(dotted["$value"], m, places=9)
                self.assertTightest(dotted, cost_shift(dataset, lambda x: x), dataset)
                for share, label in zip(SHARES, PRINTED_LABELS[dataset]):
                    bar = values["panel_b"]["bars"][dataset][f"{100 * share:.0f}"]
                    self.assertEqual(bar["label"], label)
                    bound = admissible(m, r, share)
                    if bound > T_AXIS:
                        self.assertEqual((bar["bar_height_T"], label), ({"$value": T_AXIS, "$decimals": 6}, f">{T_AXIS}"))
                        continue
                    self.assertEqual(label, f"{bound:.0f}")
                    self.assertAlmostEqual(bar["bar_height_T"]["$value"], bound, places=8)
                    self.assertTightest(bar["bar_height_T"], cost_shift(dataset, lambda x, f=share: admissible(x, r, f)),
                                        f"{dataset}/{share}")
        r = sparse_ratio("ogbn-proteins")
        for t in range(2, 17):
            reference = values["panel_c"]["points"][str(t)]
            self.assertAlmostEqual(reference["$value"], fraction(1.2, r, t), places=8)
            self.assertTightest(reference, cost_shift("ogbn-proteins", lambda x, t=t: fraction(x, r, t)), str(t))
        self.assertEqual(f"{fraction(1.2, r, T_REG):.3f}", "0.063")
        self.assertEqual(values["panel_c"]["f_res_bound_label"], {"$value": 0.063, "$decimals": 3})
        self.assertEqual(values["panel_c"]["multiplier_m"], {"$value": 1.2, "$decimals": 2})
        self.assertEqual(values["panel_c"]["truncation_order_domain"], [2, 2 * T_REG])
        # Caption: "the residual sensing-plus-observer share of the reduced sensor (k = 32, P_s = 4)".
        self.assertEqual(values["panel_c"]["registered_reduced_sensor"], {"sketch_width_k": 32, "sensing_period_P_s": 4})
        self.assertEqual(inputs["panel_c_reduced_sensor"], {"k": 32, "P_s": 4})

    def test_s6_printed_labels_require_the_unrounded_sparse_ratio(self):
        # Note S19 and Table S6c print r to three decimals; that rounded r cannot reproduce the printed bars.
        c, q = S23_COSTS["ogbn-arxiv"]
        rounded = [f"{admissible(c / q, sparse_ratio('ogbn-arxiv', 3), share):.0f}" for share in SHARES]
        self.assertEqual(rounded, ["37", "19"])
        self.assertNotEqual(tuple(rounded), PRINTED_LABELS["ogbn-arxiv"])
        # The figure labels round the real bound; Note S19 quotes the integer part ('T <= 14' for COCO at 50 %).
        c, q = S23_COSTS["COCO-SP"]
        bound = admissible(c / q, sparse_ratio("COCO-SP"), 0.5)
        self.assertEqual((math.floor(bound), PRINTED_LABELS["COCO-SP"][1]), (14, "15"))

    def test_s7_targets_are_independent_formula_evaluations_at_justified_precision(self):
        values = REGISTRY[G7]
        inputs = SIDECAR["groups"][G7]["source_inputs"]
        self.assertEqual(values["panel_a"]["boundary_slope"], {"$value": 2.0, "$decimals": 6})
        self.assertEqual([v["$value"] for v in values["panel_a"]["radius_domain"]], [0.0, 1.0])
        self.assertEqual([v["$value"] for v in values["panel_a"]["boundary_gap_at_domain_ends"]], [0.0, 2.0])
        self.assertEqual({k: v["side"] for k, v in values["panel_a"]["regions"].items()},
                         {"retain": "above", "acquire": "below"})
        panel = values["panel_b"]
        self.assertAlmostEqual(panel["p5_standard_normal_quantile"]["$value"], Z95, places=9)
        self.assertEqual(panel["threshold_tau_mnats"], {"$value": TAU_MNATS, "$decimals": 6})
        self.assertEqual(inputs["tau_nats"], TAU_MNATS / 1e3)
        self.assertEqual([v["$value"] for v in panel["gap_domain_mnats"]], [0.05, 50.0])
        self.assertEqual(panel["y_axis"], "density per log-gap")
        formulas = {"mu_log_nats": lambda a, b: lognormal(a, b)[0], "sigma_log": lambda a, b: lognormal(a, b)[1],
                    "density_at_median": lambda a, b: log_gap_density(a, a, b),
                    "density_at_tau": lambda a, b: log_gap_density(TAU_MNATS, a, b)}
        for dataset in ADMITTED:
            with self.subTest(dataset=dataset):
                median, p5 = S21_MNATS[dataset]
                self.assertEqual(inputs["action_gap_quantiles_mnats"][dataset], {"median_gap_mnats": median,
                                                                                  "p5_gap_mnats": p5})
                curve = panel["curves"][dataset]
                self.assertEqual((curve["median_mnats"], curve["p5_mnats"]),
                                 ({"$value": median, "$decimals": 1}, {"$value": p5, "$decimals": 1}))
                self.assertEqual(curve["legend_label"], f"{SHORT[dataset]}: median {median:.1f}, p5 {p5:.1f}")
                for key, function in formulas.items():
                    self.assertAlmostEqual(curve[key]["$value"], function(median, p5), places=8)
                    self.assertTightest(curve[key], gap_shift(dataset, function), f"{dataset}/{key}")
                # Log-gap convention: the peak 1/(sigma sqrt(2 pi)) lies at the median. A density per unit gap
                # (in mnats or nats) cannot satisfy the registered peak.
                sigma = lognormal(median, p5)[1]
                peak = curve["density_at_median"]
                self.assertAlmostEqual(peak["$value"], 1 / (sigma * math.sqrt(2 * math.pi)), places=8)
                for scale in (median, median / 1e3):
                    self.assertEqual(compare_value(peak["$value"] / scale, peak), "differs")

    # -- input perturbations are detected by the independent targets --------------------------------
    def test_s6_cost_rounding_box_keeps_every_target_exact(self):
        for i, (dc, dq) in enumerate(itertools.product((-COST_HALF_UNIT, COST_HALF_UNIT), repeat=2)):
            with self.subTest(controller_shift=dc, comparator_shift=dq):
                ctx = build_context(Path(self.tmp.name) / f"cost_box_{i}")
                for dataset in ADMITTED:
                    c, q = S23_COSTS[dataset]
                    ctx.require_run(dataset).evidence["cost"].update(C_qD=c + dc, C_qC=q + dq)
                _, rows = self.build(S6, ctx, f"cost_box_out_{i}")
                self.assertEqual(set(statuses(rows, G6).values()), {"exact"})

    def test_s6_measured_cost_change_moves_only_the_cost_dependent_targets(self):
        ctx = build_context(Path(self.tmp.name) / "proteins_cost")
        ctx.require_run("ogbn-proteins").evidence["cost"]["C_qD"] = 5.2
        _, rows = self.build(S6, ctx, "proteins_cost_out")
        differing = {path for path, status in statuses(rows, G6).items() if status != "exact"}
        self.assertEqual(differing, {"panel_a/measured_multipliers/ogbn-proteins",
                                     "panel_b/bars/ogbn-proteins/25/bar_height_T", "panel_b/bars/ogbn-proteins/25/label",
                                     "panel_b/bars/ogbn-proteins/50/bar_height_T", "panel_c/multiplier_m",
                                     "panel_c/f_res_bound_label"} | {f"panel_c/points/{t}" for t in range(2, 17)})
        # The 50 % proteins label stays "2" (1.88 -> 2.42), which is why the bar height is a separate target.
        self.assertEqual(statuses(rows, G6)["panel_b/bars/ogbn-proteins/50/label"], "exact")

    def test_s6_reduced_sensor_is_read_from_the_registered_configuration(self):
        ctx = build_context(Path(self.tmp.name) / "sensor")
        registered = ctx.config

        def config(dataset, profile="full"):
            copy = json.loads(json.dumps(registered(dataset, profile)))
            if dataset == "ogbn-proteins":
                copy["sensor"]["sketch_dim"] = 64
            return copy

        ctx.config = config
        writer, rows = self.build(S6, ctx, "sensor_out")
        found = statuses(rows, G6)
        self.assertEqual({path for path, status in found.items() if status != "exact"},
                         {"panel_c/registered_reduced_sensor/sketch_width_k"})
        self.assertIn({"kind": "configuration", "name": "configs/proteins_full.json",
                       "files": ["sensor.sensing_period", "sensor.sketch_dim"]}, writer.inputs)

    def test_s6_rounded_ratio_or_changed_share_fails_the_analytical_targets(self):
        original = analytical.sparse_ratio
        with mock.patch.object(analytical, "sparse_ratio",
                               side_effect=lambda ctx, dataset: round(original(ctx, dataset), 3)):
            _, rows = self.build(S6, build_context(Path(self.tmp.name) / "rounded_r"), "rounded_r_out")
        found = statuses(rows, G6)
        for dataset in ADMITTED:
            self.assertEqual(found[f"panel_a/curves/{dataset}/T64"], "differs", dataset)
        self.assertEqual((found["panel_b/bars/ogbn-arxiv/25/label"], found["panel_b/bars/ogbn-arxiv/50/label"]),
                         ("differs", "differs"))
        with mock.patch.object(analytical, "SENSING_SHARE", 0.07):
            _, rows = self.build(S6, build_context(Path(self.tmp.name) / "share"), "share_out")
        found = statuses(rows, G6)
        self.assertEqual(found["model/sensing_share_s"], "differs")
        self.assertTrue(all(found[f"panel_a/curves/{d}/T1"] == "differs" for d in ADMITTED))
        self.assertTrue(all(found[f"panel_c/points/{t}"] == "differs" for t in range(2, 17)))
        with mock.patch.object(analytical, "RESOLVENT_STEP_SHARES", (0.25, 0.75)):
            writer, rows = self.build(S6, build_context(Path(self.tmp.name) / "shares"), "shares_out")
        self.assertTrue(any(row["status"] == "no_reference" and "/75/" in row["path"] for row in rows))
        missing, coverage = self.coverage(S6, writer.comparisons)
        self.assertFalse(coverage["complete"])
        self.assertTrue(any(row["path"].startswith("panel_b/bars/ogbn-arxiv/50/") for row in missing))

    def test_s7_quantile_rounding_box_keeps_every_target_exact(self):
        offset = 0.0499 * 1e-3                                  # just inside the one-decimal legend rounding (nats)
        for i, (dm, dp) in enumerate(itertools.product((-offset, offset), repeat=2)):
            with self.subTest(median_shift=dm, p5_shift=dp):
                ctx = build_context(Path(self.tmp.name) / f"gap_box_{i}")
                for row in ctx.require_records(analytical.GAP_SET).rows:
                    if row["quantity"] == "median_gap_nats":
                        row["value"] += dm
                    elif row["quantity"] == "p5_gap_nats":
                        row["value"] += dp
                _, rows = self.build(S7, ctx, f"gap_box_out_{i}")
                self.assertEqual(set(statuses(rows, G7).values()), {"exact"})

    def test_s7_model_threshold_and_quantile_changes_are_detected(self):
        with mock.patch.object(analytical, "Z95", NormalDist().inv_cdf(0.975)):
            _, rows = self.build(S7, build_context(Path(self.tmp.name) / "z"), "z_out")
        found = statuses(rows, G7)
        self.assertEqual(found["panel_b/p5_standard_normal_quantile"], "differs")
        self.assertTrue(all(found[f"panel_b/curves/{d}/density_at_median"] == "differs" for d in ADMITTED))
        self.assertTrue(all(found[f"panel_b/curves/{d}/median_mnats"] == "exact" for d in ADMITTED))
        with mock.patch.object(analytical, "certificate_floor", return_value=0.003):
            _, rows = self.build(S7, build_context(Path(self.tmp.name) / "tau"), "tau_out")
        found = statuses(rows, G7)
        self.assertEqual(found["panel_b/threshold_tau_mnats"], "differs")
        for dataset in ADMITTED:
            reference = REGISTRY[G7]["panel_b"]["curves"][dataset]["density_at_tau"]
            expected = compare_value(log_gap_density(3.0, *S21_MNATS[dataset]), reference)
            self.assertEqual(found[f"panel_b/curves/{dataset}/density_at_tau"], expected, dataset)
        self.assertGreaterEqual(sum(found[f"panel_b/curves/{d}/density_at_tau"] == "differs" for d in ADMITTED), 5)
        ctx = build_context(Path(self.tmp.name) / "p5")
        for row in ctx.require_records(analytical.GAP_SET).rows:
            if row["dataset"] == "Peptides-func" and row["quantity"] == "p5_gap_nats":
                row["value"] = 0.0007
        _, rows = self.build(S7, ctx, "p5_out")
        found = statuses(rows, G7)
        changed = {path.rsplit("/", 1)[1] for path, status in found.items()
                   if status != "exact" and "/Peptides-func/" in path}
        expected = {"p5_mnats", "legend_label"} | {
            key for key, function in (("sigma_log", lambda a, b: lognormal(a, b)[1]),
                                      ("density_at_median", lambda a, b: log_gap_density(a, a, b)),
                                      ("density_at_tau", lambda a, b: log_gap_density(TAU_MNATS, a, b)))
            if compare_value(function(3.6, 0.7), REGISTRY[G7]["panel_b"]["curves"]["Peptides-func"][key]) == "differs"}
        self.assertEqual(changed, expected)
        self.assertIn("density_at_median", changed)
        self.assertEqual({path for path, status in found.items() if status != "exact"},
                         {f"panel_b/curves/Peptides-func/{key}" for key in changed})

    # -- what is plotted is what is compared --------------------------------------------------------
    def test_compared_coordinates_are_the_plotted_source_data(self):
        curve = {(row["dataset"], int(row["T"])): float(row["multiplier_at_half_resolvent_steps"])
                 for row in data_rows(self.out, "FigS6", "panel_a_multiplier")}
        self.assertEqual({t for _, t in curve}, set(range(1, T_AXIS + 1)))
        for dataset in ADMITTED:
            for key, t in (("T1", 1), ("T8", T_REG), ("T64", T_AXIS)):
                self.assertEqual(compare_value(curve[dataset, t], REGISTRY[G6]["panel_a"]["curves"][dataset][key]), "exact")
        for row in data_rows(self.out, "FigS6", "panel_b_admissible_order"):
            bar = REGISTRY[G6]["panel_b"]["bars"][row["dataset"]][f"{100 * float(row['resolvent_step_share']):.0f}"]
            self.assertEqual(compare_value(float(row["plotted_T"]), bar["bar_height_T"]), "exact")
            self.assertEqual(row["label"], bar["label"])
            if row["dataset"] == "Roman-empire":                     # unclipped bound kept as data, not a target
                self.assertGreater(float(row["admissible_T"]), T_AXIS)
        points = {row["T"]: float(row["f_res"]) for row in data_rows(self.out, "FigS6", "panel_c_proteins_resolvent_fraction")}
        self.assertEqual(list(points), list(REGISTRY[G6]["panel_c"]["points"]))
        for t, value in points.items():
            self.assertEqual(compare_value(value, REGISTRY[G6]["panel_c"]["points"][t]), "exact")
        density = [row for row in data_rows(self.out, "FigS7", "panel_b_gap_density") if row["dataset"] == "Peptides-func"]
        self.assertEqual([float(density[0]["gap_mnats"]), float(density[-1]["gap_mnats"])], [0.05, 50.0])
        boundary = data_rows(self.out, "FigS7", "panel_a_certificate")
        self.assertEqual([(float(r["region_radius_over_LJ"]), float(r["gap_over_LJ_boundary"])) for r in boundary],
                         [(0.0, 0.0), (1.0, 2.0)])

    def test_analytical_bound_stays_separate_from_executed_observations(self):
        writer, _ = self.results[S6]
        self.assertEqual(set(REGISTRY[G6]["panel_c"]), {"dataset", "registered_reduced_sensor", "multiplier_m",
                                                         "registered_T_line", "f_res_bound_label",
                                                         "truncation_order_domain", "points"})
        self.assertFalse([c for c in writer.comparisons if any("executed" in str(part) for part in c.path)])
        # Table S6c's executed proteins fraction (0.061) is not a figure target; the 0.063 label is the model bound.
        leaves = [value for _, value in _reference_leaves(REGISTRY[G6], include_text=True)]
        self.assertNotIn({"$value": 0.061, "$decimals": 3}, leaves)

    def test_generators_do_not_read_the_reference_files(self):
        source = Path(analytical.__file__).read_text(encoding="utf-8")
        self.assertNotIn("printed_display_values", source)
        self.assertNotIn("reported_results", source)


if __name__ == "__main__":
    unittest.main()
