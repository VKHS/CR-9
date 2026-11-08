"""Prospective selected-strata qualification and pre-selection enforcement.

The small support/score fixtures are regression inputs, not paper measurements.
The lifecycle tests use the real entry and nested-selection implementations.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from cross_morse.qualification import (NOMINAL_ACTION_ID, SELECTED_STRATA_SCHEMA,
                                       selected_strata_support)

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import numpy as np
    from cross_morse.common import atomic_json, read_json
    from cross_morse.pipeline import Pipeline


def support_fixture(*, failed=(), missing=()):
    cells = {f"{s}:{a}": {"stratum": s, "action": a, "qualified": (s, a) not in failed}
             for s in (2, 7) for a in range(10) if (s, a) not in missing}
    return {"registered_strata": [2, 7], "cells": cells,
            "qualified_fraction": sum(c["qualified"] for c in cells.values()) / 20,
            "thresholds": {"p_min": 0.05, "n_min": 64, "max_normalized_weight": 0.2}}


class SelectedStrataSupportTests(unittest.TestCase):
    def test_registry_fraction_can_pass_while_nominal_stratum_fails(self):
        support = support_fixture(failed={(7, 0)})
        self.assertGreaterEqual(support["qualified_fraction"], 0.8)
        result = selected_strata_support(support)
        self.assertFalse(result["passed"])
        self.assertEqual(result["observed_fraction"], 0.5)
        self.assertEqual(result["required_fraction"], 1.0)
        self.assertEqual(result["qualified_strata"], [2])
        self.assertEqual(result["failed_strata"], [7])
        self.assertEqual(result["missing_cells"], [])

    def test_nominal_support_is_required_in_each_stratum_not_every_action(self):
        support = support_fixture(failed={(2, 3), (7, 4)})
        before = copy.deepcopy(support)
        result = selected_strata_support(support)
        self.assertTrue(result["passed"])
        self.assertEqual(result["observed_fraction"], 1.0)
        self.assertEqual(result["registered_strata"], [2, 7])
        self.assertEqual(result["qualified_strata"], [2, 7])
        self.assertEqual(result["nominal_action"], NOMINAL_ACTION_ID)
        self.assertEqual(result["schema"], SELECTED_STRATA_SCHEMA)
        self.assertEqual(result["support_source"], "fit/full/audit.json#/support")
        self.assertNotIn("enforced", result)
        self.assertEqual(support, before)

    def test_empty_registered_strata_never_pass_vacuously(self):
        result = selected_strata_support({"registered_strata": [], "cells": {}})
        self.assertFalse(result["passed"])
        self.assertIsNone(result["observed_fraction"])
        self.assertEqual(result["qualified_strata"], [])

    def test_missing_nominal_cell_fails_and_keeps_registered_denominator(self):
        result = selected_strata_support(support_fixture(missing={(7, 0)}))
        self.assertFalse(result["passed"])
        self.assertEqual(result["observed_fraction"], 0.5)
        self.assertEqual(result["failed_strata"], [7])
        self.assertEqual(result["missing_cells"], ["7:0"])

    def test_registered_strata_require_unique_nonnegative_integer_ids(self):
        for invalid in ([2, 2], [True], ["2"], [2.0], [-1], [[2]], None, {2, 7}):
            with self.subTest(strata=invalid):
                support = support_fixture()
                support["registered_strata"] = invalid
                with self.assertRaisesRegex(ValueError, "registered_strata"):
                    selected_strata_support(support)

    def test_truthy_or_missing_qualified_flags_do_not_pass(self):
        for invalid in ("false", "true", 1, 0, None, [], {}):
            with self.subTest(qualified=invalid):
                support = support_fixture()
                support["cells"]["7:0"]["qualified"] = invalid
                with self.assertRaisesRegex(ValueError, "boolean qualified"):
                    selected_strata_support(support)
        support = support_fixture()
        del support["cells"]["7:0"]["qualified"]
        with self.assertRaisesRegex(ValueError, "boolean qualified"):
            selected_strata_support(support)

    def test_cell_identity_must_agree_with_its_lookup_key(self):
        for field, value in (("stratum", 2), ("stratum", True), ("action", 1), ("action", False)):
            with self.subTest(field=field, value=value):
                support = support_fixture()
                support["cells"]["7:0"][field] = value
                with self.assertRaisesRegex(ValueError, f"inconsistent {field}"):
                    selected_strata_support(support)

    def test_malformed_support_objects_are_rejected(self):
        for invalid in (None, [], {"registered_strata": []}, {"registered_strata": [], "cells": []}):
            with self.subTest(support=invalid), self.assertRaises(ValueError):
                selected_strata_support(invalid)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for lifecycle integration")
class SelectedStrataLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)

    def run_selection(self, support, *, enforced=True, initial_pass=True):
        pipeline = Pipeline.__new__(Pipeline)
        pipeline.out = self.out
        root = Path(__file__).resolve().parents[1]
        pipeline.config = json.loads((root / "configs/arxiv_smoke.json").read_text())
        pipeline.config["selection"].update(outer_folds=2, inner_folds=2, bootstrap_reps=40,
                                            enforce_entry_contract=enforced)
        pipeline.controller = Mock(return_value=SimpleNamespace(support_audit=support))
        pipeline.check_frozen = Mock()
        candidates = [dict(candidate_id=f"candidate_{i}", hash=f"hash_{i}", route=route,
                           profile=profile, ensemble=1, complexity=i, is_open_loop=i == 0)
                      for i, (route, profile) in enumerate((("nominal", "none"), ("one_step", "full")))]
        atomic_json(self.out / "fit/registry.json", {"candidates": candidates})
        atomic_json(self.out / "train/entry_initial.json", {"initial_entry_pass": initial_pass})
        predicted, scored = [], []

        def predict(candidate):
            predicted.append(candidate["candidate_id"])
            return dict(multiply_add_work=100, serial_wall_seconds=0.01, feasible=True)

        def scores(candidate, role):
            self.assertEqual(predicted, [c["candidate_id"] for c in candidates])
            if role == "select":
                # The entry audit and feasibility decision must exist before
                # any selection labels are opened, even for a winning policy.
                entry = read_json(self.out / "select/entry_complete.json")
                self.assertEqual(entry["selected_strata_support"]["schema"], SELECTED_STRATA_SCHEMA)
                if enforced and not entry["passed"] and not candidate["is_open_loop"]:
                    self.assertFalse(candidate["feasible"])
            else:
                self.assertEqual(role, "explore")
                self.assertFalse((self.out / "select/entry_complete.json").exists())
            scored.append(role)
            # Feedback has a clear, paired 0.2-nat improvement on both roles.
            # Support, rather than statistical power, determines eligibility.
            losses = np.linspace(0.4, 0.5, 8) - 0.2 * candidate["complexity"]
            return dict(loss=losses, accuracy=np.full(8, 0.7), counts=np.ones(8),
                        core_ids=np.arange(8), node_ce=float(losses.mean()), node_accuracy=0.7,
                        loss_name="cross_entropy", metric_name="accuracy")

        pipeline.predict_candidate, pipeline.role_scores = predict, scores
        pipeline._stage_select()
        pipeline.controller.assert_called_once_with("full")
        self.assertEqual(scored, ["explore", "explore", "select", "select"])
        return (read_json(self.out / "select/entry_complete.json"),
                read_json(self.out / "select/candidates.json"),
                read_json(self.out / "select/decision.json"))

    def test_failed_a0_support_excludes_better_feedback_before_selection(self):
        entry, candidates, decision = self.run_selection(support_fixture(failed={(7, 0)}))
        self.assertTrue(entry["initial"]["initial_entry_pass"])
        self.assertGreaterEqual(entry["registry_support_fraction"], 0.8)
        self.assertLess(entry["headroom"]["upper"], -0.003)
        self.assertFalse(entry["passed"])
        self.assertEqual(entry["selected_strata_support"]["observed_fraction"], 0.5)
        self.assertTrue(entry["selected_strata_support"]["enforced"])
        self.assertTrue(candidates[0]["feasible"])
        self.assertFalse(candidates[1]["feasible"])
        self.assertEqual(candidates[1]["ineligibility_reason"], "training-only entry contract failed")
        self.assertLess(decision["mean"][1], -0.1)
        self.assertEqual(decision["eligible_indices"], [])
        self.assertEqual(decision["provisional_index"], decision["comparator_index"])

    def test_supported_feedback_remains_eligible(self):
        entry, candidates, decision = self.run_selection(support_fixture())
        self.assertTrue(entry["passed"])
        self.assertTrue(entry["selected_strata_support"]["passed"])
        self.assertTrue(candidates[1]["feasible"])
        self.assertEqual(decision["eligible_indices"], [1])
        self.assertEqual(decision["provisional_index"], 1)

    def test_diagnostic_config_records_failure_without_claiming_enforcement(self):
        entry, candidates, decision = self.run_selection(support_fixture(failed={(7, 0)}), enforced=False)
        self.assertFalse(entry["passed"])
        self.assertFalse(entry["enforced"])
        self.assertFalse(entry["selected_strata_support"]["passed"])
        self.assertFalse(entry["selected_strata_support"]["enforced"])
        self.assertTrue(candidates[1]["feasible"])
        self.assertEqual(decision["provisional_index"], 1)

    def test_support_pass_does_not_override_other_failed_entry_gates(self):
        entry, candidates, decision = self.run_selection(support_fixture(), initial_pass=False)
        self.assertTrue(entry["selected_strata_support"]["passed"])
        self.assertFalse(entry["passed"])
        self.assertFalse(candidates[1]["feasible"])
        self.assertEqual(decision["provisional_index"], decision["comparator_index"])


if __name__ == "__main__":
    unittest.main()
