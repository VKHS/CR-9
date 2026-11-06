"""Reference implementations of cm_theory against the printed tables and the runtime's NumPy code; no PyTorch."""
from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from cm_theory import admission, budget, contract, control, economics, factorial, inference, operators, selection, spectral  # noqa: E402
from cm_theory._util import complete_bipartite, path_graph, random_graph, random_walk  # noqa: E402
from scripts import _paper_common as common  # noqa: E402

REPORTED = common.load_reported()
REGISTERED = common.load_registered()


class OperatorTests(unittest.TestCase):
    def test_polynomial_moments_agree_with_the_standard_library_restatement(self):
        literal = {a["id"]: a for a in common.parse_actions_from_core()}
        for a in operators.transport_actions():
            row = literal[a.id]
            args = (row["rho"], row["nu"], row["eta"], row["order"])
            self.assertAlmostEqual(operators.executed_range(a), common.executed_hops(*args), places=14)
            self.assertAlmostEqual(operators.identity_mass(a), common.identity_mass(row["eta"], row["order"]) or 0.0, places=14)
            self.assertAlmostEqual(operators.identity_coefficient(a), common.executed_identity_coefficient(*args), places=14)

    def test_lemma2_bound_is_nearly_attained_on_a_periodic_walk(self):
        P = random_walk(complete_bipartite(3, 5))
        result = operators.resolvent_truncation_error(P, 0.5, 2)
        self.assertLessEqual(result["error"], result["bound"] + 1e-15)
        self.assertGreater(result["error"] / result["bound"], 0.9)

    def test_inserted_site_matches_the_designated_site_formula(self):
        rng = np.random.default_rng(3)
        P = random_walk(random_graph(9, rng))
        H, H0 = rng.normal(size=(9, 4)), rng.normal(size=(9, 4))
        a = operators.transport_actions()[5]
        expected = (1 - a.nu) * H + a.nu * ((1 - a.alpha) * operators.kernel_matrix(P, a) @ H + a.alpha * H0)
        np.testing.assert_allclose(operators.inserted_site(H, H0, P, a), expected, atol=1e-14)


class SpectralTests(unittest.TestCase):
    def test_registered_windows_form_a_partition_of_unity(self):
        coeff = spectral.chebyshev_window_coefficients()
        lam = np.linspace(0, 2, 1001)
        total = sum(spectral.window_response(coeff[b], lam) for b in range(len(coeff)))
        np.testing.assert_allclose(total, np.ones_like(lam), atol=1e-12)

    def test_recurrence_matches_dense_window_on_a_path(self):
        w = path_graph(12)
        coeff = spectral.chebyshev_window_coefficients()
        X = np.random.default_rng(1).normal(size=(12, 3))
        for b in range(len(coeff)):
            np.testing.assert_allclose(spectral.window_recurrence(w, coeff[b], X), spectral.window_matrix(w, coeff[b]) @ X, atol=1e-12)

    def test_sketch_bound_is_attained_by_rank_one_states(self):
        X = np.outer(np.arange(1.0, 6.0), np.ones(4))
        self.assertAlmostEqual(spectral.sketch_relative_sd(X, 16), spectral.sketch_relative_sd_bound(16), places=12)


class AdmissionTests(unittest.TestCase):
    def test_table_s13_arithmetic(self):
        for d, row in REPORTED["supplementary_table_S13_independent_graph_admission"].items():
            terms = admission.bernstein_terms(row["S2"], row["J"], row["B"], row["delta_a"])
            self.assertAlmostEqual(terms["penalty"], row["penalty"], delta=5e-6, msg=d)
            for label, sensitivity in row["sensitivity"].items():
                B = float(label.split("=")[1])
                upper = admission.admission_upper(row["D_bar"], row["S2"], row["J"], B, row["delta_a"])
                self.assertEqual(int(upper <= -row["epsilon_a"]), sensitivity["G_admit"], (d, label))

    def test_corollary2_counts(self):
        self.assertEqual(admission.minimum_admission_units(-0.00901, 0.003), 342)
        self.assertEqual(admission.minimum_admission_units(-0.0046, 0.003), 1280)
        self.assertEqual(math.floor(admission.minimum_admission_threshold(-0.0046, 0.003)), 1279)
        self.assertTrue(math.isinf(admission.minimum_admission_threshold(-0.002, 0.003)))

    def test_runtime_bound_parity(self):
        from cross_morse.statistics import empirical_bernstein
        raw = np.random.default_rng(5).normal(-0.004, 0.05, 300)
        mine = admission.admission_upper_from_units(raw)
        runtime = empirical_bernstein(raw, clip=0.10, alpha=0.025, independent_graphs=True)
        self.assertAlmostEqual(mine["U_a"], runtime["upper"], places=15)
        self.assertAlmostEqual(mine["clipped_fraction"], runtime["clipping_fraction"], places=15)

    def test_table_s28b(self):
        # Supplementary Table S28b: 2e(n) (Hoeffding) and 2e_B(n) at sigma_1 = 0.005, 0.007, 0.010.
        rows = {64: (0.0865, 0.1036, 0.1054, 0.1082), 74: (0.0805, 0.0897, 0.0914, 0.0940),
                91: (0.0726, 0.0732, 0.0747, 0.0770), 100: (0.0692, 0.0667, 0.0681, 0.0703),
                142: (0.0581, 0.0473, 0.0485, 0.0504), 500: (0.0310, 0.0141, 0.0148, 0.0158),
                1000: (0.0219, 0.0074, 0.0079, 0.0086), 3018: (0.0126, 0.0027, 0.0030, 0.0034),
                10000: (0.0069, 0.0010, 0.0011, 0.0014)}
        for n, printed in rows.items():
            computed = (2 * admission.hoeffding_radius(n),) + tuple(2 * admission.bernstein_radius(n, s) for s in (0.005, 0.007, 0.010))
            self.assertEqual(tuple(round(v, 4) for v in computed), printed, n)
        targets = {0.0030: (53257, 2716, 3018, 3529), 0.0045: (23670, 1725, 1881, 2139),
                   0.0084: (6793, 873, 930, 1022), 0.0090: (5917, 810, 861, 944)}
        for target, (hoeffding, *bernstein) in targets.items():
            self.assertEqual(round(8 * 0.01 * math.log(400) / target ** 2), hoeffding, target)
            self.assertEqual([admission.units_for_bernstein_regret(target, s) for s in (0.005, 0.007, 0.010)], bernstein, target)

    def test_table_s28(self):
        # Supplementary Table S28: 2e at the minimum count and the reference ESS; n at the dataset's own effect.
        rows = {"ogbn-arxiv": (128, 91, 0.0612, 0.0726, 23670), "ogbn-proteins": (144, 96, 0.0577, 0.0707, 33194),
                "ogbn-products": (192, 132, 0.0500, 0.0603, None), "Roman-empire": (96, 74, 0.0707, 0.0805, 14650),
                "PascalVOC-SP": (128, 88, 0.0612, 0.0738, 6761), "COCO-SP": (200, 142, 0.0490, 0.0581, 5852),
                "Peptides-func": (150, 104, 0.0565, 0.0679, 52208)}
        support = REPORTED["supplementary_table_S7_support_audit"]
        effects = REPORTED["main_table_1_policy_and_deployment_ledger"]
        for d, (count, ess, at_count, at_ess, n_effect) in rows.items():
            self.assertEqual(count, support[d]["minimum_count"], d)
            self.assertEqual(round(2 * admission.hoeffding_radius(count), 4), at_count, d)
            self.assertEqual(round(2 * admission.hoeffding_radius(ess), 4), at_ess, d)
            if n_effect is not None:
                effect = abs(effects[d]["loss_effect_nats"])
                self.assertEqual(round(8 * 0.01 * math.log(400) / effect ** 2), n_effect, d)


class InferenceTests(unittest.TestCase):
    def test_student_t_matches_scipy_when_available(self):
        try:
            from scipy import stats
        except ImportError:
            self.skipTest("SciPy not installed")
        for df in (1, 3, 9, 31, 200):
            for x in (-3.0, -0.5, 0.0, 1.0, 2.26, 6.0):
                self.assertAlmostEqual(inference.student_t_sf(x, df), float(stats.t.sf(x, df)), places=11)
            with self.subTest(df=df, p=0.975):
                actual = inference.student_t_quantile(0.975, df)
                reference = float(stats.t.ppf(0.975, df))
                # A supported SciPy implementation can return 12.706204736432095
                # at df=1, whereas the exact Cauchy quantile is about
                # 12.706204736174696.  Compare independent implementations with
                # explicit scale-aware tolerance, not ten fixed decimal places.
                # The tighter closed-form tests below do not depend on SciPy.
                self.assertTrue(
                    math.isfinite(actual) and math.isfinite(reference)
                    and math.isclose(actual, reference, rel_tol=5e-11, abs_tol=5e-12),
                    msg=f"Student-t ppf disagreement: df={df}, p=0.975, "
                        f"actual={actual!r}, scipy={reference!r}, "
                        f"absolute_error={abs(actual - reference)!r}",
                )

    def test_student_t_quantile_matches_cauchy_closed_form(self):
        # T_1 is standard Cauchy: F^-1(p) = tan(pi * (p - 1/2)).
        for p in (0.01, 0.025, 0.1, 0.25, 0.75, 0.9, 0.975, 0.99):
            with self.subTest(p=p, df=1):
                expected = math.tan(math.pi * (p - 0.5))
                actual = inference.student_t_quantile(p, 1)
                self.assertTrue(
                    math.isclose(actual, expected, rel_tol=2e-13, abs_tol=2e-13),
                    msg=f"Cauchy closed form: p={p}, actual={actual!r}, expected={expected!r}",
                )

    def test_student_t_quantile_matches_df2_closed_form(self):
        # F_2(x) = 1/2 + x/(2*sqrt(x*x+2)); solve explicitly for x.
        for p in (0.01, 0.025, 0.1, 0.25, 0.75, 0.9, 0.975, 0.99):
            with self.subTest(p=p, df=2):
                expected = (2 * p - 1) / math.sqrt(2 * p * (1 - p))
                actual = inference.student_t_quantile(p, 2)
                self.assertTrue(
                    math.isclose(actual, expected, rel_tol=2e-13, abs_tol=2e-13),
                    msg=f"df=2 closed form: p={p}, actual={actual!r}, expected={expected!r}",
                )

    def test_mde_multipliers(self):
        self.assertEqual(round(inference.minimum_detectable_effect(), 2), 2.80)
        self.assertEqual(round(inference.minimum_detectable_effect(family_size=4), 2), 3.34)

    def test_tost_declares_equivalence_only_inside_the_margin(self):
        rng = np.random.default_rng(2)
        inside = rng.normal(0.0002, 0.001, 64)
        outside = rng.normal(0.0030, 0.001, 64)
        self.assertTrue(inference.tost(inside, 0.002)["equivalent"])
        self.assertFalse(inference.tost(outside, 0.002)["equivalent"])
        printed = inference.tost_from_interval(0.00036, -0.00064, 0.00136, 0.002)
        self.assertTrue(printed["interval_inside_margin"])
        self.assertLess(printed["p_tost"], 0.001)

    def test_holm_parity_with_runtime(self):
        from cross_morse.statistics import holm_adjust
        p = [0.0014, 0.0030, 0.0016, 0.0034, 0.42, 0.0047, 0.00016, 0.00001, 0.035]
        np.testing.assert_allclose(inference.holm_adjust(p), holm_adjust(p), atol=1e-15)

    def test_variance_components_identity(self):
        D = np.random.default_rng(4).normal(size=(30, 12))
        r = inference.variance_components(D)
        ms = r["mean_squares"]
        self.assertAlmostEqual(r["var_mean_unbiased"], (ms["graph"] + ms["seed"] - ms["residual"]) / (30 * 12), places=15)


class CostModelTests(unittest.TestCase):
    def ratio(self, d):
        stats = REGISTERED["dataset_statistics_table_S6c_S23b"][d]
        width = REGISTERED["backbones_table_S5"][d]["hidden"]
        if "nodes" in stats:
            return stats["reported_edges"] / (stats["nodes"] * width)
        return stats["mean_reported_edges_per_graph"] / (stats["mean_nodes_per_graph"] * width)

    def test_note_s19_truncation_orders(self):
        s23 = REPORTED["supplementary_table_S23_assurance_and_expected_compute"]
        expected = {"ogbn-arxiv": 18, "PascalVOC-SP": 13, "COCO-SP": 14, "Peptides-func": 22}
        for d, order in expected.items():
            m = s23[d]["controller_cost"] / s23[d]["comparator_cost"]
            self.assertEqual(math.floor(budget.admissible_truncation_order(m, self.ratio(d), 0.06, 0.5)), order, d)
        m = s23["Roman-empire"]["controller_cost"] / s23["Roman-empire"]["comparator_cost"]
        self.assertGreater(budget.admissible_truncation_order(m, self.ratio("Roman-empire"), 0.06, 0.5), 64)
        r = self.ratio("ogbn-proteins")
        self.assertEqual(round(budget.admissible_truncation_order(1.2, r, 0.06, 0.5), 1), 1.9)
        self.assertEqual(round(budget.max_resolvent_fraction(1.2, r, 0.06, 8), 3), 0.063)

    def test_table_s44_central_scenarios(self):
        s44 = REPORTED["supplementary_table_S44_per_input_cost_model"]
        s23 = REPORTED["supplementary_table_S23_assurance_and_expected_compute"]
        for d, row in s44.items():
            C = s23[d]["comparator_cost"]
            o_ctrl = s23[d]["controller_cost"] / C - 1
            scenarios = budget.tail_scenarios(C, o_ctrl, row["f_bar"], self.ratio(d))
            self.assertEqual([round(c, 2) for c in scenarios["C_099"]], row["C_099_low_central_high"], d)

    def test_reserve_counterexample(self):
        self.assertTrue(budget.reserve_released_counterexample()["exceeds"])

    def test_runtime_and_reference_algorithm_s3_agree(self):
        rng = np.random.default_rng(11)
        for _ in range(300):
            e = budget.random_episode(rng)
            runtime, reference = budget.run_runtime_episode(e), budget.algorithm_s3_reference(e)
            for key in ("status", "prediction", "operations", "action_ids", "decisions"):
                self.assertEqual(runtime[key], reference[key], key)
            self.assertAlmostEqual(runtime["total"], reference["total"], places=12)
            self.assertTrue(budget.lemma3_invariants(e, runtime)["holds"])

    def test_equal_cost_actions_do_not_hide_budget_decision_errors(self):
        costs = dict(initialization=.1, cheap_measurement=.1, score=.1, acquisition=1.,
                     action=.2, prediction=.2, comparator=2.)
        actual = {key: 1. for key in costs}
        cases = (("certified", 5., (0, 3), True, 0, "certified_skips"),
                 ("acquired", 5., (0, 3), False, 3, "acquisitions"),
                 ("empty", 5., (), False, 8, "recoveries"),
                 ("unaffordable", 2.5, (0, 3), False, 8, "unaffordable_acquisitions"))
        for label, cap, support, certified, expected_action, decision in cases:
            with self.subTest(branch=label):
                episode = budget.Episode(cap, costs, 1, (support,), (certified,), actual)
                runtime = budget.run_runtime_episode(episode)
                reference = budget.algorithm_s3_reference(episode)
                self.assertEqual(runtime["status"], "COMPLETED")
                self.assertEqual(runtime["action_ids"], [expected_action])
                self.assertEqual(runtime["decisions"][decision], 1)
                self.assertEqual(runtime["operations"], reference["operations"])
                self.assertEqual(runtime["action_ids"], reference["action_ids"])
                self.assertEqual(runtime["decisions"], reference["decisions"])


class EconomicsTests(unittest.TestCase):
    def test_break_even_of_table_s26b(self):
        row = REPORTED["supplementary_table_S26b_ensemble_law_and_break_even"]["ogbn-arxiv"]
        o = 4.617 / 4.0 - 1
        values = [round(economics.break_even_effect(D, o, 4), 4) for D in row["D_inf_band"]]
        self.assertEqual(values, row["break_even_expected"])

    def test_ensemble_crossovers_of_table_s47_with_a_three_percent_decoding_overhead(self):
        s47 = REPORTED["supplementary_table_S47_decision_boundary_and_power"]["ogbn-arxiv"]
        band = REPORTED["supplementary_table_S26b_ensemble_law_and_break_even"]["ogbn-arxiv"]["D_inf_band"]
        o = 4.617 / 4.0 - 1
        fb, b = [], []
        for D in band:
            r = economics.decision_boundary(0.0045, 0.969, o, 0.03, 0.06, D, 4)
            fb.append(round(r["phi_FB_ensemble"], 2))
            b.append(round(r["phi_B_ensemble"], 3))
        self.assertEqual(sorted(fb), s47["phi_FB_ensemble"])
        self.assertEqual(sorted(b), s47["phi_B_ensemble"])

    def test_definition2_on_table2(self):
        t2 = REPORTED["main_table_2_gain_and_cost_ledger"]["ogbn-arxiv"]
        self.assertEqual(round(1e3 * economics.exchange_rate(0.0045, 4.617, 4.0), 2), t2["lambda_star_mnats"])
        horizon_seconds = economics.amortization_horizon(1.92, 0.1, 0.617, 1.0)
        self.assertEqual(round(horizon_seconds / 1e3), t2["M_star_t_nom_thousand_input_seconds"])


class FactorialAndContractTests(unittest.TestCase):
    def test_table_s16_decomposition(self):
        s16 = REPORTED["supplementary_table_S16_arxiv_factorial"]
        cells = {k: v["log_depth_CE"] for k, v in s16["cells"].items()}
        for label, factors in (("I_AO", (0, 1)), ("I_AX", (0, 2)), ("I_OX", (1, 2)), ("I_AOX", (0, 1, 2))):
            self.assertAlmostEqual(factorial.anchored_difference(cells, factors), s16["baseline_anchored_finite_differences"][label],
                                   delta=5e-5 + 1e-12)
        mains = s16["effect_coded_main_effects_nats"]
        for j, name in enumerate(("actions", "observer", "risk_controlled_execution")):
            self.assertAlmostEqual(factorial.effect(cells, (j,)), mains[name], delta=5e-5 + 1e-12)
        self.assertAlmostEqual(factorial.full_minus_none(cells), s16["full_system_minus_no_component_nats"], places=12)

    def test_s4_rows_pass_the_entry_contract(self):
        for d, row in REPORTED["supplementary_table_S4_target_qualification"].items():
            self.assertTrue(contract.entry_contract(row)["passes"], d)

    def test_dil_is_the_trapezoid_rule_in_log_depth(self):
        depths, losses = [2, 4, 8, 16, 32], [1.0, 0.9, 0.85, 0.86, 0.9]
        x = np.log2(depths)
        expected = float(np.sum((np.array(losses[1:]) + np.array(losses[:-1])) / 2 * np.diff(x)) / (x[-1] - x[0]))
        self.assertAlmostEqual(contract.log_depth_integrated_loss(depths, losses), expected, places=15)


class ControlTests(unittest.TestCase):
    def test_advantage_identity_on_a_small_process(self):
        rng = np.random.default_rng(8)
        process = control.random_process(rng, states=2, actions=2, horizon=3)
        r = control.advantage_identity(process, control.random_history_policy(rng, 2), control.random_history_policy(rng, 2))
        self.assertLess(r["residual"], 1e-12)

    def test_selection_rule_returns_the_comparator_without_eligible_feedback(self):
        chosen = selection.eq19_select([0.001, -0.002], [True, True], [True, True], [(0,), (1,)], 0.003, 0.002)
        self.assertIsNone(chosen["selected"])
        self.assertEqual(chosen["Q_s"], [])


if __name__ == "__main__":
    unittest.main()
