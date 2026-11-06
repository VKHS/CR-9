"""The plotted Eq. (19) band must use the same candidate family as selection."""
from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

try:
    import matplotlib
except ImportError:
    raise unittest.SkipTest("Matplotlib required for selection geometry")

from paper.build.figures import _plt, fig_F3
from paper.build.selection import registered_rule_on_all_configurations


class SelectionFigureTests(unittest.TestCase):
    def fixture(self):
        def candidate(index, route, upper, *, feasible=True, joint=True):
            return {"index": index, "route": route, "route_label": route,
                    "select_lower": upper-.002, "select_upper": upper,
                    "feasible": feasible, "joint_feasible": joint,
                    "complexity": index, "ensemble": 1, "cost": 1., "hash": str(index)}
        return {"dataset": "adversarial software fixture", "scientific_run": False,
                "selection": {"qC_index": 0, "provisional_index": 1,
                              "epsilon_r": .003, "epsilon_eq": .002},
                "admission": {"deployed_index": 1},
                "candidates": [candidate(0, "fixed_sequence", -.08),
                               candidate(1, "radius", -.004),
                               candidate(2, "mpc_rh", -.005),
                               # A failed joint metric/resource gate overrides
                               # the earlier feasibility flag.
                               candidate(3, "one_step", -.20, joint=False),
                               # Older records use their feasibility flag.
                               candidate(4, "radius", -.09, feasible=False, joint=None)]}

    def plot(self, record):
        plt = _plt()
        self.addCleanup(plt.close, "all")
        with patch("paper.build.figures._save", return_value={}):
            fig_F3(plt, [record], Path("unused-test-output"))
        return plt.gcf().axes[0]

    def band(self, ax):
        spans = [artist for artist in ax.patches if artist.get_label() == "near-best band"]
        self.assertEqual(len(spans), 1)
        span = spans[0]
        # Inspect the actual rendered artist in data coordinates; axvspan is a
        # Rectangle or Polygon depending on the installed Matplotlib version.
        points = ax.transData.inverted().transform(span.get_transform().transform(span.get_path().vertices))
        return np.min(points[:, 0]), np.max(points[:, 0])

    def test_open_loop_and_infeasible_rows_cannot_move_the_near_best_band(self):
        record = self.fixture()
        self.assertEqual(registered_rule_on_all_configurations(record), 1)
        np.testing.assert_allclose(self.band(self.plot(record)), [-.005, -.003], rtol=0, atol=1e-12)

    def test_legacy_feasibility_uses_the_same_reference(self):
        record = self.fixture()
        for candidate in record["candidates"]:
            candidate["feasible"] = candidate["index"] in (0, 1, 2)
            candidate.pop("joint_feasible")
        self.assertEqual(registered_rule_on_all_configurations(record), 1)
        np.testing.assert_allclose(self.band(self.plot(record)), [-.005, -.003], rtol=0, atol=1e-12)

    def test_no_eligible_feedback_does_not_draw_an_open_loop_band(self):
        record = self.fixture()
        for candidate in record["candidates"]:
            if candidate["index"] != 0:
                candidate["joint_feasible"] = False
        self.assertEqual(registered_rule_on_all_configurations(record), 0)
        self.assertFalse(self.plot(record).patches)


if __name__ == "__main__":
    unittest.main()
