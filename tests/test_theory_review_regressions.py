"""Regression tests for the uplift review; no base checkout or PyTorch required.

Import fault injection below is deliberately confined to tests. It establishes
that missing/broken/non-finite runtime parity fails; it does not run that runtime.
"""
from __future__ import annotations

import builtins
from contextlib import redirect_stdout
import io
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from cm_theory import admission, inference, operators, selection, spectral, stability, verification
from scripts import verify_theory


class MathematicalBoundaryTests(unittest.TestCase):
    def test_zero_variance_equality_is_admissible(self):
        B, eps, J = 3 / (14 * math.log(80)), 1 / 256, 257
        mean = -2 / 256
        self.assertEqual(admission.admission_upper(mean, 0.0, J, B), -eps)
        self.assertEqual(admission.minimum_admission_threshold(mean, eps, B), J)
        self.assertEqual(admission.minimum_admission_units(mean, eps, B), J)

    def test_positive_variance_still_requires_full_admission_test(self):
        B, eps, J = 3 / (14 * math.log(80)), 1 / 256, 257
        self.assertGreater(admission.admission_upper(-2 / 256, 1e-6, J, B), -eps)

    def test_empty_support_does_not_certify_an_action(self):
        with self.assertRaisesRegex(ValueError, "empty support"):
            stability.score_gap([])
        self.assertEqual(stability.score_gap([0.2]), math.inf)
        self.assertTrue(stability.certified_ball(math.inf, 1.0, 2.0))
        self.assertFalse(stability.certified_ball(2.0, 1.0, 1.0))

    def test_invalid_certificate_numbers_are_rejected(self):
        for args in [(math.nan, 1, 1), (1, -1, 1), (1, 1, -1), (1, math.inf, 1)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                stability.certified_ball(*args)
        with self.assertRaises(ValueError):
            stability.score_gap([0.0, math.nan])

    def test_window_bound_covers_one_sided_band_boundary(self):
        coeff = spectral.chebyshev_window_coefficients()
        # The original uniform grid understated these exact one-sided limits.
        for b, (left, right) in enumerate(spectral.band_edges()):
            inside = np.array([np.nextafter(left, right), np.nextafter(right, left)])
            actual = np.max(np.abs(1.0 - spectral.window_response(coeff[b], inside) ** 2))
            self.assertGreaterEqual(spectral.window_error(coeff[b], b), actual)

    def test_spectral_bounds_cover_off_grid_values_without_eigenvalue_injection(self):
        coeff = spectral.chebyshev_window_coefficients()
        lam = np.linspace(0.0, 2.0, 20001)
        members = spectral.band_membership(lam)
        for b in range(6):
            actual = np.max(np.abs((members == b).astype(float) - spectral.window_response(coeff[b], lam) ** 2))
            self.assertGreaterEqual(spectral.window_error(coeff[b], b, grid=17), actual)
        a = operators.Action(3, "reference", .25, .7, .25, .5, 8)
        b = operators.Action(7, "reference", .25, .5, .4, .08, 8)
        bound = spectral.band_sensitivity(a, b, grid=3)
        actual = np.abs(spectral.action_response(a, lam) - spectral.action_response(b, lam))
        for band in range(6):
            self.assertGreaterEqual(bound[band], actual[members == band].max())

    def test_negative_unbiased_variance_is_not_reported_as_zero_uncertainty(self):
        result = inference.variance_components([[2., -1.], [-2., 1.]])
        self.assertEqual(result["var_mean_unbiased"], -2.0)
        self.assertIsNone(result["se_mean_unbiased"])
        self.assertEqual(result["var_mean"], 2.25)
        self.assertEqual(result["se_mean"], 1.5)
        implied = inference.variance_of_mean(result["sigma_u2"], result["sigma_v2"], result["sigma_e2"], 2, 2)
        self.assertEqual(result["var_mean"], implied)

    def test_selection_scope_and_guarantee_use_the_same_candidate_family(self):
        args = ([-.005, -.020], [True, True], [True, False], [(1,), (0,)], .003, .002)
        default = selection.eq19_select(*args)
        self.assertEqual(default, selection.eq19_select(*args, candidate_scope="feedback_only"))
        self.assertEqual(default["candidate_scope"], "feedback_only")
        self.assertEqual(default["effective_candidates"], [0])
        self.assertEqual(default["selected"], 0)
        self.assertEqual(default["Q_s"], [0])
        self.assertTrue(selection.theorem1_guarantee(args[0], 0, [0., 0.], [True, False], .003, .002)["holds"])

        unrestricted = selection.eq19_select(*args, candidate_scope="all_feasible")
        self.assertEqual(unrestricted["candidate_scope"], "all_feasible")
        self.assertEqual(unrestricted["effective_candidates"], [0, 1])
        self.assertEqual(unrestricted["selected"], 1)
        self.assertEqual(unrestricted["Q_s"], [1])
        self.assertTrue(selection.theorem1_guarantee(args[0], 1, [0., 0.], [True, True], .003, .002)["holds"])

        # This fails because the families differ, not because either declared
        # scope settles the manuscript's unresolved candidate-domain contract.
        mismatched = selection.theorem1_guarantee(args[0], default["selected"], [0., 0.], [True, True], .003, .002)
        self.assertFalse(mismatched["holds"])
        self.assertFalse(mismatched["near_best"])
        self.assertAlmostEqual(mismatched["slack"], -.013)

    def test_unknown_candidate_scope_is_rejected(self):
        for scope in ("", "feedback", None):
            with self.subTest(scope=scope), self.assertRaisesRegex(ValueError, "candidate_scope"):
                selection.eq19_select([-.005], [True], [True], [(0,)], .003, .002, candidate_scope=scope)

    def test_zero_variance_max_t_coordinates_have_zero_radius(self):
        result = selection.max_t_radii(np.zeros((2, 2)), .025, np.random.default_rng(1), draws=10)
        np.testing.assert_array_equal(result, [0., 0.])
        result = selection.max_t_radii(np.diag([0., 1.]), .025, np.random.default_rng(1), draws=100)
        self.assertEqual(result[0], 0.0)
        self.assertGreater(result[1], 0.0)

    def test_nonfinite_and_multidimensional_qualification_inputs_are_rejected(self):
        for raw in ([0., math.nan], [0., math.inf], [[0., 0.], [0., 0.]]):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                admission.admission_upper_from_units(raw)
        with self.assertRaises(ValueError):
            admission.admission_upper(0., math.nan, 20)
        with self.assertRaises(ValueError):
            inference.holm_adjust([.01, math.nan])


class VerificationIntegrityTests(unittest.TestCase):
    def _fault_import(self, name, *args, **kwargs):
        if name in ("cross_morse.statistics", "cm_campaigns.numeric"):
            raise ImportError("injected absent required base runtime")
        return self._original_import(name, *args, **kwargs)

    def test_missing_mandatory_runtime_is_a_failure(self):
        self._original_import = builtins.__import__
        for name in ("theorem1_runtime_parity", "theorem2_runtime_parity"):
            with self.subTest(check=name), mock.patch("builtins.__import__", side_effect=self._fault_import):
                result = verification.run_battery(only=[name])
            self.assertEqual(result["counts"]["FAIL"], 1)
            self.assertEqual(result["counts"]["INFO"], 0)
            self.assertEqual(result["rows"][0]["instances"], 0)
            self.assertIn("not executed", result["rows"][0]["detail"])

    def test_runtime_nan_cannot_disappear_in_max_comparison(self):
        original = builtins.__import__
        def fault(name, *args, **kwargs):
            if name == "cross_morse.statistics":
                return SimpleNamespace(empirical_bernstein=lambda *a, **k: {"upper": math.nan, "clipping_fraction": 0.0})
            return original(name, *args, **kwargs)
        with mock.patch("builtins.__import__", side_effect=fault):
            result = verification.run_battery(only=["theorem2_runtime_parity"])
        self.assertEqual(result["counts"]["FAIL"], 1)
        self.assertIn("non-finite", result["rows"][0]["observed"])

    def test_exported_admission_formula_is_exercised_by_coverage_check(self):
        with mock.patch.object(admission, "admission_upper", side_effect=lambda mean, *a, **k: mean):
            result = verification.run_battery(only=["theorem2_coverage"])
        self.assertGreater(result["counts"]["FAIL"], 0)

    def test_lifecycle_check_requires_successful_and_rejected_admission(self):
        result = verification.run_battery(only=["corollary1_lifecycle"])
        self.assertEqual(result["counts"]["FAIL"], 0, result["rows"])
        self.assertNotIn("0 deployed", result["rows"][0]["detail"])
        with mock.patch.object(admission, "admission_upper", return_value=math.inf):
            result = verification.run_battery(only=["corollary1_lifecycle"])
        self.assertEqual(result["counts"]["FAIL"], 1)

    def test_mixed_valid_and_unknown_selection_is_not_silently_partial(self):
        with self.assertRaises(KeyError):
            verification.run_battery(only=["lemma-2", "typo"])

    def test_failed_battery_does_not_overwrite_verified_documentation(self):
        row = {"check": "injected", "statement_key": "theorem-2", "statement": "Theorem 2", "claim": "required check",
               "status": "FAIL", "observed": "missing runtime", "bound": None, "instances": 0, "detail": "not run"}
        result = {"mode": "quick", "seed": verification.DEFAULT_SEED, "counts": {"PASS": 0, "FAIL": 1, "INFO": 0, "ADVISORY": 0},
                  "rows": [row], "check_generators_selected": 1, "statements_without_checks": []}
        with tempfile.TemporaryDirectory() as directory:
            doc = Path(directory) / "THEORY.md"
            doc.write_text("original documentation")
            with mock.patch.object(verify_theory, "run_battery", return_value=result), \
                    mock.patch.object(verify_theory, "DOC", doc), redirect_stdout(io.StringIO()):
                code = verify_theory.main(["--write-doc"])
            self.assertEqual(code, 1)
            self.assertEqual(doc.read_text(), "original documentation")

    def test_ast_reader_rejects_arguments_the_runtime_constructor_would_reject(self):
        for call in ("Action(0,'a',0,.7,.25,None,8,0,'transport','extra')",
                     "Action(0,'a',0,.7,.25,nu=.5)"):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "fixture.py"
                path.write_text(f"ACTIONS = ({call},)\n")
                with self.assertRaises(ValueError):
                    operators.load_registry(path)


if __name__ == "__main__":
    unittest.main()
