"""Independent printed-value contracts of Supplementary Figures S12 and S20.

The two dedicated groups of ``paper/printed_display_values.json`` enumerate every panel of the
analytical figures from printed sources (Tables S42, S47, S23, S26b, S13, S6/S6b, main Table 1,
Note S24 and the SI p. 20 statement of Fig. S20b) and from the stated formulas evaluated with
printed inputs. The fixture inputs of ``tests/test_displays_analytical.py`` equal the printed
inputs, so every dedicated comparison is exact; omitting one comparison leaves a coverage gap;
perturbing an input or an analytical function is detected. Agreement on synthetic fixtures is
software behaviour, not evidence for the paper.
"""
from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from statistics import NormalDist
from unittest import mock

import numpy as np

from paper.build.displays import analytical
from paper.build.displays.framework import ROOT, _reference_leaves, comparison_coverage, load_references
from paper.build.displays.testing import build_one
from tests.test_displays_analytical import build_context

# The sidecar is read here from the repository root, so the new groups are part of this registry.
REGISTRY, REFERENCE_METADATA, _ = load_references(ROOT)
S12, S20 = "supplementary:Figure S12", "supplementary:Figure S20"
GROUPS = {S12: analytical.G_FS12, S20: analytical.G_FS20}
EXPECTED = {S12: (128, {"panel_a": 63, "panel_b": 65}),
            S20: (310, {"panel_a": 64, "panel_b": 78, "panel_c": 13, "panel_d": 155})}
DATASETS = ("ogbn-arxiv", "ogbn-proteins", "Roman-empire", "PascalVOC-SP", "COCO-SP", "Peptides-func")
SINGLE_GRAPH = DATASETS[:3]
ACTIONS = tuple(f"A{i}" for i in range(1, 8))
ANCHORS = ("0.05", "0.1", "0.3", "1", "2")
N = NormalDist()


def value(reference):
    return reference["$value"] if isinstance(reference, dict) else reference


def within(computed, reference):
    return abs(computed - reference["$value"]) <= 0.5 * 10.0 ** -reference["$decimals"] * (1 + 1e-9) + 1e-15


def build(display_id, ctx, out):
    return build_one(display_id, ctx, out, registry=REGISTRY)


def statuses(rows, group):
    return {row["path"]: row["status"] for row in rows if row["group"] == group}


class FigureContractS12S20(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.ctx = build_context(root / "inputs", complete=True)
        cls.results = {display_id: build(display_id, cls.ctx, root / "displays") for display_id in GROUPS}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def rebuilt(self, display_id, ctx):
        with tempfile.TemporaryDirectory() as tmp:
            return build(display_id, ctx, Path(tmp))

    # -- registration and complete coverage ----------------------------------------------------------
    def test_groups_are_bound_to_the_figure_pages_and_enumerate_every_family_member(self):
        for display_id, page in ((S12, 34), (S20, 38)):
            meta = REFERENCE_METADATA[GROUPS[display_id]]
            self.assertEqual((meta["display_id"], meta["source"]["document"], meta["source"]["pages"]),
                             (display_id, "supplementary", [page]))
        s12, s20 = REGISTRY[analytical.G_FS12], REGISTRY[analytical.G_FS20]
        for panel in ("panel_a", "panel_b"):
            self.assertEqual(tuple(s12[panel]["datasets"]), DATASETS)
        self.assertEqual(tuple(s20["panel_a"]["datasets"]), DATASETS)
        self.assertEqual(tuple(s20["panel_a"]["break_even_curves"]), ("E2", "E4", "E8"))
        self.assertEqual(tuple(s20["panel_b"]["actions"]), ACTIONS)
        self.assertTrue(all(tuple(s20["panel_b"]["actions"][a]) == ANCHORS for a in ACTIONS))
        self.assertEqual(tuple(s20["panel_c"]["datasets"]), analytical.GRAPH_LEVEL)
        self.assertEqual(tuple(s20["panel_d"]["datasets"]), DATASETS)
        for dataset in DATASETS:
            for reading in ("conservative_power_percent", "optimistic_power_percent"):
                self.assertEqual(list(s20["panel_d"]["datasets"][dataset][reading]), [str(s) for s in range(1, 13)])

    def test_every_target_is_compared_and_exact_on_printed_fixture_inputs(self):
        for display_id, group in GROUPS.items():
            total, panels = EXPECTED[display_id]
            with self.subTest(display=display_id):
                writer, rows = self.results[display_id]
                missing, coverage = comparison_coverage(display_id, writer.comparisons, REGISTRY,
                                                        REFERENCE_METADATA, scope="analytical")
                self.assertEqual(missing, [])
                self.assertTrue(coverage["complete"])
                self.assertEqual((coverage["expected_values"], coverage["covered_values"]), (total, total))
                counts = {panel: len(list(_reference_leaves(REGISTRY[group][panel], include_text=True)))
                          for panel in REGISTRY[group]}
                self.assertEqual(counts, panels)
                dedicated = [row for row in rows if row["group"] == group]
                self.assertEqual(len({row["path"] for row in dedicated}), len(dedicated))
                self.assertTrue(all(row["status"] == "exact" for row in dedicated))
                self.assertEqual([row for row in rows if row["status"] != "exact"], [])

    def test_dropping_one_comparison_per_panel_leaves_a_coverage_gap(self):
        representatives = {
            S12: [("panel_a", "datasets", "ogbn-arxiv", "legend_label"),
                  ("panel_a", "datasets", "Roman-empire", "power_at_domain_end_percent"),
                  ("panel_a", "datasets", "COCO-SP", "margin_range_mnats"),
                  ("panel_a", "margin_domain_mnats"),
                  ("panel_a", "reference_power_percent"),
                  ("panel_b", "datasets", "Peptides-func", "conservative_margin_at_reference_power_mnats"),
                  ("panel_b", "datasets", "ogbn-proteins", "optimistic_power_at_central_margin_percent"),
                  ("panel_b", "paired_seeds"),
                  ("panel_b", "holm_family_size")],
            S20: [("panel_a", "break_even_curves", "E2", "at_unit_overhead"),
                  ("panel_a", "datasets", "Roman-empire", "whisker_gain_over_D_inf"),
                  ("panel_a", "datasets", "PascalVOC-SP", "break_even_at_assurance_overhead"),
                  ("panel_a", "overhead_domain"),
                  ("panel_a", "ensemble_law_kappa_band"),
                  ("panel_b", "actions", "A5", "0.3", "exact_displacement"),
                  ("panel_b", "actions", "A7", "2", "upper_bound"),
                  ("panel_b", "printed_ratio_extremes", "maximum_at_lambda0_1"),
                  ("panel_b", "e18_audit_densities"),
                  ("panel_b", "spectral_scale_domain"),
                  ("panel_c", "datasets", "COCO-SP", "ratio_label"),
                  ("panel_c", "graph_unit_z_multiplier"),
                  ("panel_d", "datasets", "Roman-empire", "conservative_power_percent", "7"),
                  ("panel_d", "datasets", "COCO-SP", "pairable_share"),
                  ("panel_d", "optimistic_level")]}
        for display_id, paths in representatives.items():
            group = GROUPS[display_id]
            writer, _ = self.results[display_id]
            for path in paths:
                with self.subTest(display=display_id, path=path):
                    retained = [c for c in writer.comparisons if not (c.group == group and c.path == path)]
                    self.assertEqual(len(retained), len(writer.comparisons) - 1)
                    missing, coverage = comparison_coverage(display_id, retained, REGISTRY, REFERENCE_METADATA,
                                                            scope="analytical")
                    self.assertFalse(coverage["complete"])
                    self.assertTrue(missing)
                    self.assertTrue(all(row["group"] == group and row["path"].startswith("/".join(path))
                                        for row in missing))

    def test_omitted_series_cannot_claim_complete_coverage(self):
        series = {S12: [("panel_b", "datasets", "Roman-empire")],
                  S20: [("panel_a", "break_even_curves", "E4"), ("panel_a", "datasets", "COCO-SP"),
                        ("panel_b", "actions", "A1"), ("panel_c", "datasets", "Peptides-func"),
                        ("panel_d", "datasets", "ogbn-proteins", "optimistic_power_percent")]}
        for display_id, prefixes in series.items():
            group = GROUPS[display_id]
            writer, _ = self.results[display_id]
            for prefix in prefixes:
                with self.subTest(display=display_id, series=prefix):
                    retained = [c for c in writer.comparisons
                                if not (c.group == group and c.path[:len(prefix)] == prefix)]
                    self.assertLess(len(retained), len(writer.comparisons))
                    missing, coverage = comparison_coverage(display_id, retained, REGISTRY, REFERENCE_METADATA,
                                                            scope="analytical")
                    self.assertFalse(coverage["complete"])
                    node = REGISTRY[group]
                    for key in prefix:
                        node = node[key]
                    expected = {"/".join(map(str, (*prefix, *path))) for path, _ in
                                _reference_leaves(node, include_text=True)}
                    self.assertEqual({row["path"] for row in missing if row["group"] == group}, expected)
                    self.assertEqual(len(missing), len(expected))

    # -- perturbations are detected against the independent targets ---------------------------------
    def test_changed_paired_standard_error_changes_only_that_datasets_power_curves(self):
        ctx = build_context(Path(tempfile.mkdtemp(dir=self.tmp.name)), complete=True)
        for row in ctx.require_records("budget_matched").rows:
            if row["dataset"] == "Roman-empire" and row["arm"] == "E3":
                row["value"] = 0.002 + 1.2 * (row["value"] - 0.002)
        for display_id, group in GROUPS.items():
            with self.subTest(display=display_id):
                _, rows = self.rebuilt(display_id, ctx)
                found = statuses(rows, group)
                differing = {path for path, status in found.items() if status != "exact"}
                self.assertTrue(differing)
                self.assertTrue(all("/Roman-empire/" in path for path in differing))
                if display_id == S12:
                    self.assertTrue({"panel_a/datasets/Roman-empire/power_at_central_margin_percent",
                                     "panel_a/datasets/Roman-empire/margin_at_reference_power_mnats",
                                     "panel_a/datasets/Roman-empire/power_at_domain_end_percent",
                                     "panel_b/datasets/Roman-empire/conservative_margin_at_reference_power_mnats"}
                                    <= differing)
                    self.assertEqual(found["panel_a/datasets/Roman-empire/power_at_zero_margin_percent"], "exact")
                else:
                    self.assertTrue({f"panel_d/datasets/Roman-empire/conservative_power_percent/{s}"
                                     for s in range(1, 13)} <= differing)

    def test_changed_fallback_probability_moves_only_the_expected_overhead_marker(self):
        ctx = build_context(Path(tempfile.mkdtemp(dir=self.tmp.name)), complete=True)
        for row in ctx.require_records("input_gate").rows:
            if row["dataset"] == "ogbn-arxiv":
                row["value"] = 0.30
        _, rows = self.rebuilt(S20, ctx)
        differing = {path for path, status in statuses(rows, analytical.G_FS20).items() if status != "exact"}
        self.assertEqual(differing, {"panel_a/datasets/ogbn-arxiv/expected_overhead",
                                     "panel_a/datasets/ogbn-arxiv/break_even_at_expected_overhead"})

    def test_perturbed_action_functions_fail_only_their_reach_targets(self):
        original_response = analytical.spectral.action_response
        original_sensitivity = analytical.spectral.band_sensitivity
        nominal = analytical.operators.load_registry()[0]

        def response(action, lam):
            g = original_response(action, lam)
            if action.id != 7:
                return g
            g0 = original_response(nominal, lam)
            return g0 + 0.98 * (g - g0)

        def sensitivity(action, reference, *args, **kwargs):
            s = original_sensitivity(action, reference, *args, **kwargs)
            return 1.02 * s if action.id == 3 else s

        with mock.patch.object(analytical.spectral, "action_response", side_effect=response):
            _, rows = build(S20, self.ctx, Path(tempfile.mkdtemp(dir=self.tmp.name)))
        differing = {path for path, status in statuses(rows, analytical.G_FS20).items() if status != "exact"}
        self.assertEqual(differing, {f"panel_b/actions/A7/{a}/exact_displacement" for a in ANCHORS})
        with mock.patch.object(analytical.spectral, "band_sensitivity", side_effect=sensitivity):
            _, rows = build(S20, self.ctx, Path(tempfile.mkdtemp(dir=self.tmp.name)))
        differing = {path for path, status in statuses(rows, analytical.G_FS20).items() if status != "exact"}
        # A3 holds the smallest printed ratio at lambda0 = 0.1, so the SI p. 20 statement fails with its bound.
        self.assertEqual(differing, {f"panel_b/actions/A3/{a}/upper_bound" for a in ANCHORS}
                         | {"panel_b/printed_ratio_extremes/minimum_at_lambda0_0.1"})

    def test_changed_pairable_share_is_detected_in_both_seed_designs(self):
        with mock.patch.dict(analytical.PAIRABLE_SHARE, {"single_graph": 0.4}):
            _, s12 = build(S12, self.ctx, Path(tempfile.mkdtemp(dir=self.tmp.name)))
            _, s20 = build(S20, self.ctx, Path(tempfile.mkdtemp(dir=self.tmp.name)))
        s12_differing = {p for p, s in statuses(s12, analytical.G_FS12).items() if s != "exact"}
        s20_differing = {p for p, s in statuses(s20, analytical.G_FS20).items() if s != "exact"}
        for dataset in SINGLE_GRAPH:
            path = f"panel_b/datasets/{dataset}/"
            self.assertTrue({path + "pairable_share", path + "conservative_power_at_central_margin_percent",
                             path + "conservative_margin_at_reference_power_mnats"} <= s12_differing)
        self.assertTrue(all(p.startswith("panel_b/datasets/") and p.split("/")[2] in SINGLE_GRAPH
                            and ("conservative" in p or p.endswith("pairable_share")) for p in s12_differing))
        self.assertEqual(s20_differing, {f"panel_d/datasets/{d}/pairable_share" for d in SINGLE_GRAPH}
                         | {f"panel_d/datasets/{d}/conservative_power_percent/{s}" for d in SINGLE_GRAPH
                            for s in range(2, 13)})

    # -- the registered targets agree with independent evaluation and the printed tables ----------------
    def test_printed_targets_agree_with_the_transcribed_tables(self):
        s42 = REGISTRY["supplementary_table_S42_power_of_budget_matched_test"]
        s12, s20 = REGISTRY[analytical.G_FS12], REGISTRY[analytical.G_FS20]
        for d in DATASETS:
            a, b, design = s12["panel_a"]["datasets"][d], s12["panel_b"]["datasets"][d], s20["panel_d"]["datasets"][d]
            self.assertEqual(value(a["central_margin_mnats"]), s42[d]["reported_margin_mnats"])
            self.assertEqual(value(b["marker_margin_mnats"]), s42[d]["reported_margin_mnats"])
            self.assertEqual([value(v) for v in a["margin_range_mnats"]], s42[d]["margin_range"])
            self.assertEqual(value(a["power_at_central_margin_percent"]), s42[d]["power_at_unit_pct"])
            self.assertEqual([value(v) for v in a["power_at_range_endpoints_percent"]], s42[d]["power_range"])
            self.assertEqual(value(a["margin_at_reference_power_mnats"]), s42[d]["MDE_mnats_0.05"])
            self.assertEqual([value(b["conservative_power_at_central_margin_percent"]),
                              value(b["optimistic_power_at_central_margin_percent"])], s42[d]["power_seed_paired_pct"])
            self.assertEqual(a["legend_label"].split(" (")[1].rstrip(")"),
                             s42[d]["unit"].replace(" (fixed ensemble)", ", fixed ensemble"))
            self.assertEqual(value(design["conservative_power_percent"]["1"]), s42[d]["power_holm_step_pct"])
            self.assertEqual(value(design["optimistic_power_percent"]["1"]), s42[d]["power_at_unit_pct"])
            self.assertEqual([value(design["conservative_power_percent"]["12"]),
                              value(design["optimistic_power_percent"]["12"])], s42[d]["power_seed_paired_pct"])
        s47 = REGISTRY["supplementary_table_S47_decision_boundary_and_power"]
        for d in DATASETS:
            b = s12["panel_b"]["datasets"][d]
            self.assertEqual([value(b["conservative_margin_at_reference_power_mnats"]),
                              value(b["optimistic_margin_at_reference_power_mnats"])], s47[d]["MDE_seed_paired"])
        table1 = REGISTRY["main_table_1_policy_and_deployment_ledger"]
        for d in analytical.GRAPH_LEVEL:
            low, high = table1[d]["loss_effect_interval_nats"]
            # Note S24 prints the unrounded half-width; the rounded Table 1 endpoints agree within 0.05 mnats.
            self.assertLessEqual(abs(value(s20["panel_c"]["datasets"][d]["test_stage_half_width_mnats"])
                                     - 500 * (high - low)), 0.05 + 1e-12)

    def test_formula_targets_match_independent_evaluation_of_the_printed_inputs(self):
        s20 = REGISTRY[analytical.G_FS20]
        s23 = REGISTRY["supplementary_table_S23_assurance_and_expected_compute"]
        s26b = REGISTRY["supplementary_table_S26b_ensemble_law_and_break_even"]
        table1 = REGISTRY["main_table_1_policy_and_deployment_ledger"]
        for E in (2, 4, 8):
            curve = s20["panel_a"]["break_even_curves"][f"E{E}"]
            for key, o in (("at_domain_start", 0.08), ("at_unit_overhead", 1.0), ("at_domain_end", 2.0)):
                self.assertTrue(within(o / (E * (1 + o)), curve[key]))
        for d in DATASETS:
            point = s20["panel_a"]["datasets"][d]
            E, g = table1[d]["deployed_E"], abs(table1[d]["loss_effect_nats"])
            q = s23[d]["comparator_cost"]
            o, o_a = s23[d]["expected_serial_cost"] / q - 1, s23[d]["assurance_serial_cost"] / q - 1
            self.assertEqual(point["ensemble_size"], E)
            self.assertTrue(within(o, point["expected_overhead"]) and within(o_a, point["assurance_overhead"]))
            self.assertTrue(within(g / s26b[d]["D_inf_hat"], point["gain_over_D_inf"]))
            band = s26b[d]["D_inf_band"]
            self.assertTrue(within(g / band[1], point["whisker_gain_over_D_inf"][0]))
            self.assertTrue(within(g / band[0], point["whisker_gain_over_D_inf"][1]))
            # The registered curve values at the operating points reproduce every printed S26b threshold.
            for field, printed in (("break_even_at_expected_overhead", s26b[d]["break_even_expected"]),
                                   ("break_even_at_assurance_overhead", s26b[d]["break_even_assurance"])):
                self.assertEqual([round(value(point[field]) * b, 4) for b in band], printed)
        # Panel b from Table S6 by an independent dense-grid evaluation.
        registry = {1: (0.20, 0.55, None, None), 2: (0.20, 0.85, None, None), 3: (0.25, 0.70, 0.50, 8),
                    4: (0.30, 0.65, 0.50, 8), 5: (0.30, 0.70, 0.20, 8), 6: (0.35, 0.60, 0.20, 8),
                    7: (0.25, 0.50, 0.08, 8)}
        lam = np.linspace(0.0, 2.0, 600001)
        x = 1.0 - lam
        g0 = 0.30 + 0.70 * x

        def response(rho, nu, eta, order):
            if eta is None:
                return (1 - nu) + nu * x
            kernel = eta * sum((1 - eta) ** t * x ** t for t in range(order + 1)) + (1 - eta) ** (order + 1)
            return (1 - nu) + nu * ((1 - rho) * x + rho * kernel)

        band = np.minimum((lam * 3).astype(int), 5)
        for a, params in registry.items():
            diff = np.abs(response(*params) - g0)
            s = np.array([diff[band == b].max() for b in range(6)])
            for anchor in ANCHORS:
                l0 = float(anchor)
                mass = np.array([math.exp(-b / 3 / l0) - math.exp(-(b + 1) / 3 / l0) for b in range(6)])
                mass /= 1 - math.exp(-2 / l0)
                density = np.exp(-lam / l0) / (l0 * (1 - math.exp(-2 / l0)))
                exact = math.sqrt(float(np.trapezoid(diff ** 2 * density, lam)))
                targets = s20["panel_b"]["actions"][f"A{a}"][anchor]
                self.assertTrue(within(math.sqrt(float(np.sum(s ** 2 * mass))), targets["upper_bound"]), (a, anchor))
                self.assertTrue(within(exact, targets["exact_displacement"]), (a, anchor))
        # Panel c from Table S13 moments and the Note S24 half-widths.
        s13 = REGISTRY["supplementary_table_S13_independent_graph_admission"]
        for d in analytical.GRAPH_LEVEL:
            bars = s20["panel_c"]["datasets"][d]
            graph = 1.96 * math.sqrt(s13[d]["S2"] / s13[d]["J"]) * 1e3
            rho = value(bars["test_stage_half_width_mnats"]) / graph
            self.assertTrue(within(graph, bars["graph_unit_half_width_mnats"]))
            self.assertTrue(within(rho, bars["ratio_label"]))
            self.assertTrue(within(100 * (1 - rho ** -2), bars["seed_share_lower_bound_percent"]))
        # Panels S12 and S20d: the stated normal power at zero margin is the test level.
        z = N.inv_cdf(1 - 0.05 / 2)
        self.assertAlmostEqual(200 * N.cdf(-z), 5.0)
        self.assertTrue(within(2 * N.cdf(-N.inv_cdf(1 - 0.0125 / 2)), s20["panel_d"]["conservative_level"]))

    def test_declared_density_scenario_stays_distinct_from_the_e18_plant_audit(self):
        actions = REGISTRY[analytical.G_FS20]["panel_b"]["actions"]

        def ratios(anchor):
            return [value(actions[a][anchor]["exact_displacement"]) / value(actions[a][anchor]["upper_bound"])
                    for a in ACTIONS]
        # SI p. 20 states the plotted ratio range, which the registered coordinates reproduce ...
        self.assertEqual((round(min(ratios("0.1")), 2), round(max(ratios("1")), 2)), (0.35, 0.86))
        # ... whereas Table S43's plant-specific E18 audit (0.73 at 0.3; 0.82-0.86 at 1) is not this scenario.
        self.assertEqual(round(max(ratios("0.3")), 2), 0.74)
        self.assertEqual(round(min(ratios("1")), 2), 0.81)
        self.assertEqual(set(REGISTRY[analytical.G_FS20]["panel_b"]["printed_ratio_extremes"]),
                         {"minimum_at_lambda0_0.1", "maximum_at_lambda0_1"})

    def test_generators_read_neither_reference_file(self):
        source = Path(analytical.__file__).read_text(encoding="utf-8")
        self.assertNotIn("printed_display_values", source)
        self.assertNotIn("reported_results", source)


if __name__ == "__main__":
    unittest.main()
