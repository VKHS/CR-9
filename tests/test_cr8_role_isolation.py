"""Holdout-covariate isolation and executable rejection regressions.

The hostile heldout graphs make all-corpus execution fail; passing requires
restricting execution itself, not only masking labels after the fact.
"""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch, Mock

import numpy as np
import torch

from cross_morse.common import atomic_json, file_hash
from cross_morse.data import GraphData
from cross_morse.pipeline import Pipeline, atomic_npy, selection_fold_plan
from cross_morse.runtime import predict_dataset
from cross_morse.policies import PolicyExecutionError
from cross_morse.statistics import admission, confirmation


def corpus(graph_level):
    graphs, nodes, outputs = 10, 3, 2
    ptr = np.arange(graphs + 1, dtype=np.int64) * nodes
    local = torch.tensor([[0, 1, 2], [1, 2, 0]])
    edges = torch.cat([local + g * nodes for g in range(graphs)], 1)
    graph_roles = {"fit_train": np.array([0, 1]), "fit_val": np.array([2]),
                   "fit": np.array([0, 1, 2]), "explore": np.array([3]),
                   "select": np.array([4, 5]), "confirm": np.array([6]),
                   "compare": np.array([7]), "admit": np.array([8]), "test": np.array([9])}
    splits = {k: v.copy() if graph_level else np.concatenate([
        np.arange(ptr[g], ptr[g+1], dtype=np.int64) for g in v]) for k, v in graph_roles.items()}
    x = torch.repeat_interleave(torch.arange(graphs), nodes).float()[:, None]
    rows = graphs if graph_level else graphs * nodes
    return GraphData(x, edges, torch.zeros(rows, dtype=torch.long), splits,
                     np.arange(graphs) if graph_level else np.repeat(np.arange(graphs), nodes),
                     {"num_classes": outputs, "num_samples": rows,
                      "label_axis": "graph" if graph_level else "node"},
                     graph_ptr=ptr, edge_ptr=np.arange(graphs+1)*local.shape[1],
                     graph_splits=graph_roles).protect_labels()


class RoleIsolationTests(unittest.TestCase):
    def execute(self, data, role, *, output_path=None):
        visited = []
        allowed = set(data.graph_splits[role].tolist())

        def policy(model, x, graph, config, candidate, controller, reference, donors=None):
            graph_id = int(x[0, 0])
            if graph_id not in allowed:
                raise AssertionError("Unopened holdout graph was executed")
            visited.append(graph_id)
            n = 1 if data.metadata["label_axis"] == "graph" else len(x)
            p = np.tile(np.array([[.25, .75]], dtype=np.float32), (n, 1))
            return p, {"multiply_add_work": graph_id + 10,
                       "deployed_multiply_add_work": graph_id + 8,
                       "qualification_multiply_add_work": 2,
                       "actions": [1], "feasible": True,
                       "shuffle_missing_sites": 0, "shuffle_matched_sites": 1,
                       "state_control_valid": True}

        with patch("cross_morse.runtime.graph_operator", return_value=object()), \
             patch("cross_morse.runtime.execute_policy", side_effect=policy):
            predictions, trace = predict_dataset(SimpleNamespace(num_classes=2), data, None, {},
                candidate={}, role=role, output_path=output_path)
        return predictions, trace, visited

    def test_select_cannot_execute_or_reduce_metadata_from_hostile_holdouts(self):
        for graph_level in (False, True):
            with self.subTest(graph_level=graph_level):
                data = corpus(graph_level)
                p, trace, visited = self.execute(data, "select")
                np.testing.assert_array_equal(visited, [4, 5])
                self.assertEqual(trace["graphs"], 2)
                self.assertEqual(trace["multiply_add_work"], 29)
                self.assertEqual(trace["deployed_multiply_add_work"], 25)
                self.assertTrue(trace["feasible"])
                self.assertEqual(trace["execution_role"], "select")
                self.assertTrue(np.isfinite(p[data.splits["select"]]).all())
                self.assertTrue(np.isnan(p[data.splits["admit"]]).all())
                self.assertTrue(np.isnan(p[data.splits["test"]]).all())
                self.assertEqual(data.y.accesses, [])
                # Change unopened covariates so accidental execution would fail.
                data.x[data.graph_ptr[8]:] = -999999
                p2, second, visited2 = self.execute(data, "select")
                np.testing.assert_array_equal(p, p2)
                self.assertEqual(visited, visited2)
                for key in ("multiply_add_work", "feasible", "graph_ids_sha256", "actions_sha256"):
                    self.assertEqual(trace[key], second[key])

    def test_failed_scoped_member_records_partial_work_and_cleans_stream(self):
        data = corpus(True)
        def policy(model, x, *args, **kwargs):
            if int(x[0, 0]) == 5:
                raise PolicyExecutionError("hostile execution fixture", {"multiply_add_work": 20,
                    "deployed_multiply_add_work": 17, "qualification_multiply_add_work": 3})
            return np.array([[.25, .75]], dtype=np.float32), {"multiply_add_work": 10,
                "deployed_multiply_add_work": 8, "qualification_multiply_add_work": 2,
                "actions": [], "feasible": True}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "failed.npy"
            with patch("cross_morse.runtime.graph_operator", return_value=object()), \
                 patch("cross_morse.runtime.execute_policy", side_effect=policy):
                with self.assertRaises(PolicyExecutionError) as caught:
                    predict_dataset(SimpleNamespace(num_classes=2), data, None, {}, candidate={},
                                    role="select", output_path=path)
            trace = caught.exception.trace
            self.assertEqual(trace["execution_role"], "select")
            self.assertEqual(trace["completed_graph_ids"], [4])
            self.assertEqual(trace["failed_graph_id"], 5)
            self.assertEqual(trace["multiply_add_work"], 30)
            self.assertEqual(trace["deployed_multiply_add_work"], 25)
            self.assertTrue(trace["cost_is_partial"])
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_confirmation_donor_counts_include_only_confirmation_graphs(self):
        data = corpus(True)
        _, trace, visited = self.execute(data, "confirm")
        self.assertEqual(visited, [6])
        self.assertEqual(trace["shuffle_matched_sites"], 1)
        self.assertEqual(trace["shuffle_missing_sites"], 0)
        self.assertEqual(trace["prediction_rows"], 1)

    def test_streamed_predictions_leave_unexecuted_rows_explicitly_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "admission.npy"
            data = corpus(False)
            p, trace, _ = self.execute(data, "admit", output_path=path)
            self.assertTrue(path.is_file())
            self.assertTrue(np.isnan(p[data.splits["confirm"]]).all())
            self.assertEqual(trace["prediction_rows"], 3)
            self.assertEqual(trace["corpus_prediction_rows"], 30)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_role_graph_indices_must_match_the_scoring_rows(self):
        data = corpus(True)
        data.splits["select"] = np.array([4, 6])
        with self.assertRaisesRegex(ValueError, "exactly the declared execution role"):
            self.execute(data, "select")

    def test_unknown_execution_role_cannot_fall_back_to_all_graphs(self):
        data = corpus(True)
        with self.assertRaisesRegex(ValueError, "Unknown independent-graph execution role"):
            predict_dataset(SimpleNamespace(num_classes=2), data, None, {}, candidate={}, role="typo")

    def test_pipeline_caches_are_distinct_and_bind_the_execution_role(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = Pipeline.__new__(Pipeline)
            pipeline.out = Path(directory)
            pipeline.config = {"dataset": "Peptides-func"}
            pipeline._active_stage = "select"
            candidate = {"candidate_id": "same", "hash": "fixed-policy"}
            selected = pipeline.probability_path(candidate)
            admitted = pipeline.probability_path(candidate, "admit")
            self.assertNotEqual(selected, admitted)
            atomic_npy(selected, np.array([[.25, .75]], dtype=np.float32))
            atomic_json(selected.with_suffix(".json"), {"candidate": candidate,
                        "execution_role": "admit", "prediction_sha256": file_hash(selected)})
            with self.assertRaisesRegex(RuntimeError, "identity/hash"):
                pipeline.verify_prediction(candidate)

    def test_feedback_confirmation_executes_only_its_reserved_graph_role(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = Pipeline.__new__(Pipeline)
            pipeline.out = Path(directory)
            pipeline.config = {"dataset": "Peptides-func", "seed": 17,
                               "selection": {"bootstrap_reps": 100}}
            pipeline._active_stage = "confirm"
            pipeline.check_frozen = Mock()
            candidates = [dict(candidate_id=str(i), hash=str(i), ensemble=1, route=route,
                               profile=profile, kappa=1)
                          for i, (route, profile) in enumerate((("nominal", "none"),
                              ("one_step", "lean"), ("radius", "lean")))]
            atomic_json(pipeline.out / "select/candidates.json", candidates)
            atomic_json(pipeline.out / "select/decision.json", {
                "comparator_index": 0, "provisional_index": 2, "eligible_indices": [2]})
            predicted = set()
            def predict(candidate, *, role, **kwargs):
                self.assertEqual(role, "confirm")
                predicted.add(candidate["candidate_id"])
                return {"feasible": True, "shuffle_missing_sites": 0,
                        "shuffle_matched_sites": 20, "state_control_valid": True}
            def scores(candidate, role):
                self.assertEqual(role, "confirm")
                self.assertTrue({"0", "1", "2"} <= predicted)
                loss = .5 if candidate.get("state_control") else {"0": .6, "1": .45, "2": .4}[candidate["candidate_id"]]
                return {"loss": np.full(20, loss), "counts": np.ones(20), "node_ce": loss}
            pipeline.predict_candidate = predict
            pipeline.role_scores = scores
            with patch("cross_morse.pipeline.load_records", return_value={}), \
                 patch("cross_morse.pipeline.DonorBank", return_value=object()):
                pipeline._stage_confirm()
            import json
            decision = json.loads((pipeline.out / "confirm/decision.json").read_text())
            self.assertEqual(decision["confirmed_index"], 2)
            self.assertEqual(predicted, {"0", "1", "2", "shuffle_2"})

    def test_favorable_loss_cannot_admit_an_execution_that_failed_support(self):
        args = dict(differences=np.full(5000, -.05), metric_differences=np.full(5000, .01),
                    independent_graphs=True, reps=20)
        self.assertTrue(admission(**args, execution_valid=True)["passed"])
        rejected = admission(**args, execution_valid=False)
        self.assertFalse(rejected["passed"])
        self.assertEqual(rejected["Gadmit"], 0)
        self.assertFalse(rejected["execution_gate"])
        # Historical numerical archives retain the previous output contract.
        self.assertNotIn("execution_gate", admission(**args))

    def test_equal_outer_and_inner_fold_counts_leave_valid_inner_training_sets(self):
        outer, inner = selection_fold_plan(8, 2, 2, 17)
        self.assertEqual(inner.shape, (3, 8))
        for k in range(2):
            self.assertEqual(set(inner[k, outer != k]), {0, 1})
            self.assertTrue((inner[k, outer == k] == -1).all())
        self.assertEqual(set(inner[-1]), {0, 1})
        with self.assertRaisesRegex(ValueError, "Too few outer-training"):
            selection_fold_plan(4, 2, 3, 17)

    def test_peptides_reported_p_value_cannot_retain_feedback(self):
        # One member avoids making assumptions about the missing paper ledger.
        # A Holm-adjusted P value of at least .07 must fail the .05 rule.
        result = confirmation(1, 0, [1], [{"kind": "state", "candidate_index": 1,
            "predecessor_index": 0, "differences": np.full(20, .01), "p_value": .07}], reps=20)
        self.assertEqual(result["confirmed_index"], 0)
        self.assertFalse(result["contrasts"][0]["passes"])
        self.assertEqual(result["downgrades"][0]["to"], 0)


if __name__ == "__main__":
    unittest.main()
