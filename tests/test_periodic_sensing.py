"""Independent work/count and data-boundary checks for periodic spectral sensing."""
import unittest
import numpy as np
import torch

from cross_morse.core import GraphOperator, MeasurementBank, DesignatedSitePlant
from cross_morse.learning import _profile_y, _support


class PeriodicSensingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.g = GraphOperator(torch.tensor([[0, 1, 1, 2, 1, 3], [1, 0, 2, 1, 3, 1]]), 5, "cpu")
        self.h = torch.randn(5, 8)
        self.logits = torch.randn(5, 3)

    def test_sparse_work_follows_actual_acquisitions_while_cheap_features_change(self):
        bank = MeasurementBank(self.g, 8, sensing_period=4, sketch_dim=3)
        bank.prepare(self.h, self.logits)
        self.g.reset_counters()
        values = [bank.measure(self.h * (1 + d), self.logits * (1 + d), d, 8)
                  for d in range(8)]
        # Eight current Dirichlet products + two acquisitions, each two
        # degree-eight recurrences; source preparation is accounted separately.
        self.assertEqual(self.g.sparse_products, 8 + 2 * 2 * 8)
        self.assertEqual(bank.counters["spectral_acquisitions"], 2)
        self.assertEqual(bank.counters["spectral_reuses"], 6)
        torch.testing.assert_close(values[0][9:25], values[3][9:25])
        self.assertNotEqual(values[0][0].item(), values[3][0].item())
        self.assertNotEqual(values[0][8].item(), values[3][8].item())
        self.assertEqual([v[-1].item() for v in values], [0, .25, .5, .75, 0, .25, .5, .75])
        self.assertEqual(values[0].shape, (26,))

    def test_period_one_preserves_original_sensor_law(self):
        original = MeasurementBank(self.g, 8)
        explicit = MeasurementBank(self.g, 8, sensing_period=1)
        for b in (original, explicit):
            b.prepare(self.h, self.logits)
        for d in range(5):
            torch.testing.assert_close(original.measure(self.h * (d + 1), self.logits, d, 5),
                                       explicit.measure(self.h * (d + 1), self.logits, d, 5), rtol=0, atol=0)
        self.assertEqual(original.feature_dim, 25)

    def test_cached_power_uses_current_native_site_and_terminal_law(self):
        model = DesignatedSitePlant(4, 8, 3, encoder_layers=1, dropout=0,
                                    control_depth=8, native_layers=2, backbone="gps", attention_heads=2)
        bank = MeasurementBank(self.g, 8, plant=model, sensing_period=3)
        bank.prepare(self.h, self.logits)
        inserted = bank.measure(self.h, self.logits, 0, 8)
        native = bank.measure(self.h, self.logits, 1, 8)
        # Same cached powers, but A1's gap to the nominal operator changes.
        torch.testing.assert_close(inserted[18] / native[18], torch.tensor(.55 / .45))
        terminal = bank.measure(self.h, self.logits, 8, 8)
        torch.testing.assert_close(terminal[18:25], torch.zeros(7))
        self.assertFalse(bank.metadata()["cached_spectral_values_are_certificates"])

    def test_reset_clears_cached_state_and_invalid_histories_fail(self):
        bank = MeasurementBank(self.g, 8, sensing_period=4)
        bank.prepare(self.h, self.logits)
        bank.measure(self.h, self.logits, 0, 8)
        bank.measure(self.h, self.logits, 2, 8)
        with self.assertRaisesRegex(ValueError, "backwards"):
            bank.measure(self.h, self.logits, 1, 8)
        with self.assertRaisesRegex(ValueError, "fixed constructor"):
            bank.measure(self.h, self.logits, 3, 8, mask=torch.ones(5, dtype=torch.bool))
        bank.prepare(2 * self.h, self.logits)
        value = bank.measure(2 * self.h, self.logits, 0, 8)
        torch.testing.assert_close(value[12:15], torch.ones(3), atol=1e-5, rtol=1e-5)
        self.assertEqual(value[-1].item(), 0)
        for invalid in (True, 0, -1, 1.5):
            with self.assertRaises(ValueError):
                MeasurementBank(self.g, 8, sensing_period=invalid)

    def test_lean_conversion_matches_absence_of_spectral_acquisition(self):
        y = np.ones((3, 26))
        y[:, -1] = [0, .25, .5]
        lean = _profile_y(y, "lean")
        np.testing.assert_array_equal(lean[:, 9:25], 0)
        np.testing.assert_array_equal(lean[:, -1], 0)


class SupportWeightEvidenceTests(unittest.TestCase):
    def test_repeated_depths_do_not_multiply_independent_weight_units(self):
        records = {"y": np.zeros((8, 25)), "depth": np.zeros(8),
                   "action": np.zeros(8), "propensity": np.array([.1] * 6 + [.2, .2]),
                   "trajectory": np.arange(8), "graph_id": np.array([11] * 6 + [22, 22])}
        cell = _support(records, np.arange(8), {"support": {"n_min": 2}})["cells"]["0:0"]
        self.assertEqual(cell["trajectory_count"], 2)
        self.assertAlmostEqual(cell["ess"], 1.8)
        self.assertAlmostEqual(cell["max_normalized_weight"], 2 / 3)
        self.assertAlmostEqual(cell["sum_squared_normalized_weights"], 5 / 9)
        self.assertTrue(cell["support_arithmetic_valid"])
        self.assertFalse(cell["qualified"])
        self.assertLessEqual(cell["ess"], cell["ess_upper_bound_from_max_weight"])


if __name__ == "__main__":
    unittest.main()
