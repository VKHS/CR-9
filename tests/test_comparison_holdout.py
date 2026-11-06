"""The teacher's confirmation data cannot become its arm-comparison holdout."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from cross_morse.split_contract import (
    DATA_ROLE_PROTOCOL, LEAF_ROLES, ROLE_NAMES, require_reserved_comparison,
    validate_reserved_roles,
)
from cm_experiments.holdout import ARM_SCHEMA, comparison_contract, require_arm_contract


def partition():
    return {
        "fit_train": np.arange(54), "fit_val": np.arange(54, 60), "fit": np.arange(60),
        "explore": np.arange(60, 80), "select": np.arange(80, 90),
        "confirm": np.arange(90, 95), "compare": np.arange(95, 100),
        "admit": np.arange(100, 150), "test": np.arange(150, 200),
    }


def metadata():
    return {"schema_version": 3, "data_role_protocol": DATA_ROLE_PROTOCOL,
            "comparison_holdout_reserved_at": "prepare", "num_samples": 200,
            "fingerprint": "1" * 64}


class ComparisonContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "provenance/stages").mkdir(parents=True)
        (self.root / "provenance/stages/00_prepare.json").write_text("{}")
        self.roles = partition()
        self.data = SimpleNamespace(metadata=metadata(), splits=self.roles)
        self.experiment = {"schema": ARM_SCHEMA, "evaluation_role": "D_compare", "max_graphs": 0,
                           "comparison_holdout": comparison_contract(self.data, self.root)}

    def test_reserved_leaves_are_disjoint_and_cover_every_observation(self):
        values = validate_reserved_roles(self.roles, size=200)
        np.testing.assert_array_equal(np.sort(np.concatenate([values[k] for k in LEAF_ROLES])), np.arange(200))

    def test_missing_compare_requires_new_preparation_not_repartition(self):
        old = {k: v for k, v in self.roles.items() if k != "compare"}
        old["confirm"] = np.arange(90, 100)
        with self.assertRaisesRegex(ValueError, "never repartition"):
            validate_reserved_roles(old)

    def test_confirmation_overlap_is_rejected(self):
        wrong = copy.deepcopy(self.roles); wrong["compare"][0] = wrong["confirm"][0]
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_reserved_roles(wrong)

    def test_uncovered_observations_are_rejected(self):
        wrong = copy.deepcopy(self.roles); wrong["compare"] = wrong["compare"][:-1]
        with self.assertRaisesRegex(ValueError, "cover"):
            validate_reserved_roles(wrong)

    def test_preallocation_identity_is_required(self):
        wrong = metadata(); wrong["comparison_holdout_reserved_at"] = "confirm"
        with self.assertRaisesRegex(ValueError, "before fitting"):
            require_reserved_comparison(wrong, self.roles)

    def test_original_holdout_observations_are_accepted(self):
        require_arm_contract(self.experiment, self.roles, results={"evaluation_role": "D_compare"},
                             observations=self.roles["compare"][::-1])

    def test_old_arm_schema_is_never_relabelled_as_independent(self):
        wrong = dict(self.experiment, schema="cm-budget-arm-experiment-2")
        with self.assertRaisesRegex(ValueError, "Rerun preparation"):
            require_arm_contract(wrong, self.roles)

    def test_new_schema_with_confirmation_role_is_rejected(self):
        wrong = dict(self.experiment, evaluation_role="Dconfirm")
        with self.assertRaisesRegex(ValueError, "reserved D_compare"):
            require_arm_contract(wrong, self.roles)

    def test_confirmation_prediction_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside D_compare"):
            require_arm_contract(self.experiment, self.roles, observations=self.roles["confirm"])

    def test_comparison_cannot_be_selected_after_inspecting_it(self):
        wrong = copy.deepcopy(self.experiment)
        wrong["comparison_holdout"]["reserved_at"] = "arm_selection"
        with self.assertRaisesRegex(ValueError, "preallocated"):
            require_arm_contract(wrong, self.roles)

    def test_full_study_cannot_silently_drop_holdout_observations(self):
        with self.assertRaisesRegex(ValueError, "cover the reserved"):
            require_arm_contract(self.experiment, self.roles, observations=self.roles["compare"][:-1])

    def test_results_role_cannot_disagree_with_experiment(self):
        with self.assertRaisesRegex(ValueError, "not evaluated on D_compare"):
            require_arm_contract(self.experiment, self.roles, results={"evaluation_role": "Dconfirm"})


try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "Prepared-data tests require PyTorch")
class PreparedComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def prepare(self, graph_collection=False):
        from cross_morse.data import prepare_data
        cfg = {"dataset": "synthetic", "seed": 17, "data": {
            "synthetic_nodes": 200, "synthetic_features": 8, "synthetic_classes": 4,
            "num_clusters": 8, "partitioner": "balanced", "internal_val_fraction": .1,
        }}
        if graph_collection:
            cfg["benchmark_dataset"] = "PascalVOC-SP"
            cfg["data"].update(synthetic_graphs=80, synthetic_nodes_per_graph=4)
        return prepare_data(self.root / "raw", self.root / "run", cfg)

    def test_actual_prepare_uses_fixed_five_part_training_split(self):
        from cross_morse.data import _make_roles
        official = {"train": np.arange(100), "valid": np.arange(100, 150), "test": np.arange(150, 200)}
        split = _make_roles(official, 200, seed=18, fraction=.1)
        self.assertEqual([len(split[k]) for k in ("fit", "explore", "select", "confirm", "compare")], [60, 20, 10, 5, 5])
        np.testing.assert_array_equal(split["admit"], official["valid"])
        np.testing.assert_array_equal(split["test"], official["test"])
        validate_reserved_roles(split, size=200)

    def test_every_lifecycle_stage_is_denied_comparison_labels(self):
        data = self.prepare().protect_labels()
        for stage in ("prepare", "train", "explore", "fit", "select", "confirm", "admit", "evaluate"):
            with self.subTest(stage=stage), data._label_scope(stage):
                with self.assertRaises(PermissionError):
                    data.labels("compare")
        with data._label_scope("compare"):
            self.assertEqual(len(data.labels("compare")), len(data.splits["compare"]))
            with self.assertRaises(PermissionError):
                data.labels("confirm")

    def test_graph_comparison_never_splits_a_graph(self):
        data = self.prepare(graph_collection=True).protect_labels()
        for graph_id in range(data.num_graphs):
            memberships = [r for r in LEAF_ROLES if graph_id in data.graph_splits[r]]
            self.assertEqual(len(memberships), 1)
        for graph_id in data.graph_splits["compare"]:
            episode = data.graph(int(graph_id))
            self.assertEqual(len(episode.splits["compare"]), episode.num_nodes)
            with data._label_scope("confirm"):
                with self.assertRaises(PermissionError):
                    episode.labels("compare")
            with data._label_scope("compare"):
                self.assertEqual(len(episode.labels("compare")), episode.num_nodes)

    def test_legacy_prepared_data_fails_without_mutating_any_split(self):
        from cross_morse.data import load_data
        self.prepare()
        path = self.root / "run/prepared/metadata.json"
        old = json.loads(path.read_text()); old["schema_version"] = 2
        path.write_text(json.dumps(old))
        split = self.root / "run/prepared/splits.npz"; before = split.read_bytes()
        with self.assertRaisesRegex(ValueError, "never repartition"):
            load_data(self.root / "run")
        self.assertEqual(before, split.read_bytes())

    def test_arm_evaluator_reads_only_preallocated_compare_labels(self):
        from cross_morse.data import prepare_data
        from cm_experiments.runner import evaluate_selected
        from cm_experiments.protocol import budget_contract
        cfg = json.loads((Path(__file__).resolve().parents[1] / "configs/arxiv_smoke.json").read_text())
        data = prepare_data(self.root / "raw", self.root / "parent", cfg).protect_labels()
        manifests = self.root / "parent/provenance/stages"; manifests.mkdir(parents=True)
        (manifests / "00_prepare.json").write_text("{}")
        exp = {"schema": ARM_SCHEMA, "evaluation_role": "D_compare", "max_graphs": 0,
               "ensemble_members": [1], "comparison_holdout": comparison_contract(data, self.root / "parent")}
        contract = budget_contract("ogbn-arxiv")
        output = self.root / "study"; output.mkdir()
        (output / "development_selection.json").write_text("{}")

        class Backend:
            comparator = {"hash": "comparator"}
            teacher = {"hash": "teacher"}

            def execute(self, episode, variant, **kwargs):
                prediction = np.full((episode.num_nodes, data.metadata["num_outputs"]),
                                     1 / data.metadata["num_outputs"], dtype=np.float64)
                return {"prediction": prediction, "traces": [], "status": "EXECUTED",
                        "deployed_work": 1., "executed_work": 1., "wall_seconds": 0.,
                        "embedded_qualification_work": 0., "separate_qualification_work": 0.,
                        "support_qualified": True}

        selected = {a: {"configuration_id": a, "kind": "policy", "origin": "locked_comparator"}
                    for a in ("E3", "E4", "E5", "E6")}
        evaluate_selected(Backend(), data, cfg, exp, contract, selected, output)
        self.assertEqual({x["role"] for x in data.y.accesses}, {"compare"})
        with np.load(output / "predictions.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(np.sort(saved["observation_id"]), np.sort(data.splits["compare"]))
        self.assertEqual(json.loads((output / "results.json").read_text())["evaluation_role"], "D_compare")


if __name__ == "__main__":
    unittest.main()
