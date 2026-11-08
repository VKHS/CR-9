"""Variant lifecycles of cm_experiments/ablation_lifecycle.py and scripts/run_ablation.py: the S17 module
ablations (E7, E8), the E18 policy value over matched confidence/history controls and the separately fitted
factorial cells (S16, S16b).

Planning, statistics, the comparator branch, collection and export run without PyTorch on the bundled
synthetic example runs and on synthetic job outputs; their record sets are written to temporary roots and
traced by scripts.records.verify_set.  Execution (variant lifecycles from a smoke parent, measurement and
export end to end) needs PyTorch and runs in CI.
"""
from __future__ import annotations

import contextlib
import copy
import csv
import importlib.util
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from cm_experiments import ablation_lifecycle as ablation
from cm_experiments import ensemble_grid, factorial, sweeps
from cm_experiments.artifacts import canonical, seal, sha256
from cm_experiments.ensemble_grid import RunnerError, SealedRun

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "runs"
ARXIV, PEPTIDES = EXAMPLES / "arxiv", EXAMPLES / "peptides_func"
TORCH = importlib.util.find_spec("torch") is not None
ENVIRONMENT = {"python": "3.x", "machine": "x86_64", "torch": None}
FIXTURE = dict(allow_fixture=True, allow_source_change=True)     # the examples were sealed by an older runtime


def specs():
    from scripts.records import RECORDS, load_specs
    return load_specs(RECORDS, RECORDS / "specs.json")


def config_of(run):
    return json.loads((Path(run) / "config.json").read_text())


class Workspace(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cm-ablation-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)


# ============================================================================= variants
class VariantTests(unittest.TestCase):
    def test_measurement_schema_and_module_coordinates(self):
        config = config_of(ARXIV)
        names = ablation.measurement_names(config)
        self.assertEqual(len(names), 25)
        self.assertEqual(names[:9], ["entropy", "margin", "source_js", "effective_rank_entropy", "log1p_dirichlet",
                                     "relative_depth", "normalized_mass", "normalized_hops", "log_state_energy"])
        periodic = copy.deepcopy(config)
        periodic["sensor"]["sensing_period"] = 2
        self.assertEqual(ablation.measurement_names(periodic)[-1], "spectral_age_fraction")
        bands = ablation.module_coordinates("spectral_bands", config)
        self.assertEqual(ablation.coordinate_indices(bands, config), [9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
        self.assertEqual(ablation.coordinate_indices(ablation.module_coordinates("source_alignment", config), config),
                         [12, 13, 14])
        self.assertEqual(ablation.module_coordinates("confidence_sensors", config), ["entropy", "margin"])
        self.assertEqual(ablation.module_coordinates("transport_history", config), ["normalized_mass", "normalized_hops"])
        e18 = ablation.module_coordinates(ablation.E18_MODULE, config)
        self.assertEqual(ablation.coordinate_indices(e18, config), list(range(9, 25)))      # exactly the Lean set
        self.assertEqual(ablation.coordinate_indices(ablation.module_coordinates(ablation.E18_MODULE, periodic),
                                                     periodic), list(range(9, 26)))
        with self.assertRaisesRegex(RunnerError, "outside the measurement schema"):
            ablation.coordinate_indices(["band_power_1"], config)

    def test_specifications_name_their_installer_and_are_part_of_the_configuration(self):
        config = config_of(ARXIV)
        ops = ablation.module_operations("transport_history", "no_history", config)
        spec = ablation.variant_spec("transport_history:no_history", ops, study="modules",
                                     module="transport_history", arm="no_history")
        self.assertEqual(spec["installer"]["source_sha256"], sha256(ROOT / "cm_experiments" / "ablation_lifecycle.py"))
        variant = ablation.variant_config(config, spec)
        self.assertEqual(variant[ablation.VARIANT_KEY], spec)
        self.assertEqual({k: v for k, v in variant.items() if k != ablation.VARIANT_KEY}, config)
        self.assertNotEqual(canonical(variant), canonical(config))
        with self.assertRaisesRegex(RunnerError, "itself a variant"):
            ablation.variant_config(variant, spec)
        stale = copy.deepcopy(spec)
        stale["installer"]["source_sha256"] = "0" * 64
        with self.assertRaisesRegex(RunnerError, "another installer source"):
            ablation.validate_spec(stale)
        bad = copy.deepcopy(spec)
        bad["operations"][0]["indices"] = [5, 6]
        with self.assertRaisesRegex(RunnerError, "indices differ"):
            ablation.validate_spec(bad, config)
        depth = ablation.variant_spec("x", [{"op": "delete", "coordinates": ["relative_depth"], "indices": [5]}],
                                      study="modules")
        with self.assertRaisesRegex(RunnerError, "relative depth"):
            ablation.validate_spec(depth)
        twice = ablation.variant_spec("x", [{"op": "remove_reset", "action": 8}, {"op": "remove_reset", "action": 9}],
                                      study="modules")
        with self.assertRaisesRegex(RunnerError, "one operation of each kind"):
            ablation.validate_spec(twice)
        with self.assertRaisesRegex(RunnerError, "unknown observer"):
            ablation.validate_spec(ablation.variant_spec("x", [{"op": "observer", "kind": "smoother"}],
                                                         study="modules"))
        for bad in ({"window": 0, "width": 128}, {"window": 4}, {"window": 4.0, "width": 128}):
            with self.assertRaisesRegex(RunnerError, "window and width"):
                ablation.validate_spec(ablation.variant_spec(
                    "x", [{"op": "observer", "kind": "history_mlp", **bad}], study="modules"))
        with self.assertRaisesRegex(RunnerError, "transport actions"):
            ablation.validate_spec(ablation.variant_spec("x", [{"op": "action_bank", "allowed": [0]}],
                                                         study="factorial"))
        with self.assertRaisesRegex(RunnerError, "'off'"):
            ablation.validate_spec(ablation.variant_spec("x", [{"op": "risk_control", "mode": "on"}],
                                                         study="factorial"))
        with self.assertRaisesRegex(RunnerError, "removes both resets"):
            ablation.validate_spec(ablation.variant_spec(
                "x", [{"op": "remove_reset", "action": 8}, {"op": "risk_control", "mode": "off"}], study="factorial"))
        with self.assertRaisesRegex(RunnerError, "unknown variant operation"):
            ablation.validate_spec(ablation.variant_spec("x", [{"op": "support", "mode": "unrestricted"}],
                                                         study="factorial"))

    def test_arm_operations(self):
        config = config_of(ARXIV)
        self.assertEqual(ablation.module_operations("observer", "ema", config, ema_weight=.3),
                         [{"op": "observer", "kind": "ema", "ema_weight": .3}])
        self.assertEqual(ablation.module_operations("recovery_type", "source_only", config),
                         [{"op": "remove_reset", "action": 9}])
        self.assertEqual(ablation.module_operations("recovery_type", "last_safe_only", config),
                         [{"op": "remove_reset", "action": 8}])
        with self.assertRaisesRegex(RunnerError, "complete arm"):
            ablation.module_operations("observer", "conditioned", config)
        with self.assertRaisesRegex(RunnerError, "donor pool"):
            ablation.module_operations("spectral_bands", "scrambled", config)
        with self.assertRaisesRegex(RunnerError, r"\(0, 1\)"):
            ablation.module_operations("observer", "ema", config, ema_weight=1.0)
        spec = ablation.variant_spec("observer:ema", ablation.module_operations("observer", "ema", config),
                                     study="modules", module="observer", arm="ema")
        self.assertEqual(ablation.observer_ablation(spec), {"variant": "observer:ema", "deleted": [],
                                                            "observer": "ema", "ema_weight": .5})
        history = ablation.module_operations("observer", "history_mlp", config)    # two observer periods, width 128
        self.assertEqual(history, [{"op": "observer", "kind": "history_mlp", "window": 4, "width": 128}])
        self.assertEqual(ablation.module_operations("observer", "history_mlp", config, history_window_length=6,
                                                    history_width=32),
                         [{"op": "observer", "kind": "history_mlp", "window": 6, "width": 32}])
        with self.assertRaisesRegex(RunnerError, "positive integers"):
            ablation.module_operations("observer", "history_mlp", config, history_window_length=0)
        spec = ablation.variant_spec("observer:history_mlp", history, study="modules", module="observer",
                                     arm="history_mlp")
        ablation.validate_spec(spec, config)
        self.assertEqual(ablation.observer_ablation(spec)["history"], {"window": 4, "width": 128})

    def test_factorial_cell_operations(self):
        arxiv, pascal = ("A", "O", "X"), ("S", "O", "X")
        memoryless, no_transport = {"op": "observer", "kind": "memoryless"}, {"op": "action_bank", "allowed": [0, 8, 9]}
        risk_off = {"op": "risk_control", "mode": "off"}
        self.assertEqual(ablation.factorial_operations("111", arxiv), [])
        self.assertEqual(ablation.factorial_operations("000", arxiv), [memoryless, no_transport, risk_off])
        self.assertEqual(ablation.factorial_operations("010", arxiv), [no_transport, risk_off])
        self.assertEqual(ablation.factorial_operations("001", arxiv), [memoryless, no_transport])
        self.assertEqual(ablation.factorial_operations("110", arxiv), [risk_off])
        self.assertEqual(ablation.factorial_operations("011", pascal), [])          # S: the Lean profile, not a fit
        self.assertEqual(ablation.factorial_operations("101", pascal), [memoryless])
        self.assertEqual(ablation.factorial_operations("100", pascal), [memoryless, risk_off])
        for factors, distinct in ((arxiv, 8), (pascal, 4)):         # one lifecycle per distinct operation set
            operations = [ablation.factorial_operations(cell, factors) for cell in factorial.CELLS]
            self.assertEqual(len({canonical(ops) for ops in operations}), distinct)
            for ops in operations:
                ablation.validate_spec(ablation.variant_spec("x", ops, study="factorial"))
        support = {"cells": {"0:0": {"stratum": 0, "action": 0, "qualified": True}}, "qualified_fraction": .1}
        opened = ablation.unrestricted_support(support, config_of(ARXIV))
        self.assertEqual(len(opened["cells"]), 2 * 3 * 10)
        self.assertTrue(all(c["qualified"] for c in opened["cells"].values()))
        self.assertEqual(support["qualified_fraction"], .1)                           # the input is not modified

    def test_donor_index_is_a_deterministic_function_of_seed_and_key(self):
        draws = [ablation.donor_index(7, ("explore", t, t % 2), 10) for t in range(200)]
        self.assertEqual(draws, [ablation.donor_index(7, ("explore", t, t % 2), 10) for t in range(200)])
        self.assertTrue(set(draws) <= set(range(10)))
        self.assertGreater(len(set(draws)), 5)
        self.assertNotEqual(draws, [ablation.donor_index(8, ("explore", t, t % 2), 10) for t in range(200)])


# ============================================================================= history-MLP observer (NumPy parts)
class HistoryObserverTests(unittest.TestCase):
    def test_capacity_follows_the_registered_window_and_width(self):
        from cm_experiments.ensemble_grid import registered_constants
        full = json.loads((ROOT / "configs" / "arxiv_full.json").read_text())
        self.assertEqual(ablation.history_window(full), 2 * full["controller"]["observer_period"])
        self.assertEqual(ablation.history_window(full), 4)
        self.assertIn(f"{ablation.HISTORY_WIDTH}-dimensional",
                      registered_constants()["observer_eq12_table_S9"]["sequence_models"])
        d = full["controller"]["state_dim"]
        self.assertEqual(ablation.history_parameters(4, 128, d), 12048)
        cells = {f"{b}:{a}": [] for b in range(full["controller"]["depth_bins"]) for a in range(10)}
        payload = {"basis": [[0.0] * d] * 25, "conditioned": cells}
        self.assertEqual(ablation.conditioned_parameters(payload), 13120)        # one global + 40 cell transitions
        self.assertEqual(ablation.conditioned_parameters(dict(payload, conditioned={})), 320)

    def test_windows_follow_each_trajectory_and_repeat_its_first_measurement(self):
        rng = np.random.default_rng(3)
        groups, depths = np.repeat([4, 1, 7], [5, 3, 1]), np.r_[np.arange(5), np.arange(3), [0]]
        start = {4: 0, 1: 5, 7: 8}
        latent = rng.normal(size=(9, 2))
        order = rng.permutation(9)
        windows = ablation.history_windows(latent[order], groups[order], depths[order], 3)
        for k, i in enumerate(order):
            expected = [latent[start[groups[i]] + max(depths[i] - lag, 0)] for lag in (2, 1, 0)]   # oldest first
            np.testing.assert_array_equal(windows[k], np.stack(expected))
        with self.assertRaisesRegex(RunnerError, "record_stride 1"):
            ablation.history_windows(latent[:3], np.zeros(3), np.array([0, 1, 3]), 2)

    def test_inputs_forward_pass_and_filter_arithmetic(self):
        rng = np.random.default_rng(5)
        d, w, h = 3, 2, 4
        model = {"window": w, "width": h, "w1": rng.normal(size=(h, w * d + 13)).tolist(),
                 "b1": rng.normal(size=h).tolist(), "w2": rng.normal(size=(d, h)).tolist(),
                 "b2": rng.normal(size=d).tolist(), "scaler": {"mean": [0.1, -0.2, 0.0], "scale": [2.0, 0.5, 1.0]}}
        mlp = ablation.HistoryMLP(model)
        window, u = rng.normal(size=(w, d)), rng.normal(size=3)
        x = np.concatenate([window.reshape(-1), u, np.eye(10)[7]])
        np.testing.assert_array_equal(ablation.history_inputs(window[None], u[None], [7])[0], x)
        self.assertEqual(float(np.abs(ablation.history_inputs(window[None], u[None], [-1])[0, -10:]).sum()), 0.0)
        hidden = np.asarray(model["w1"]) @ x + model["b1"]
        hidden = hidden / (1 + np.exp(-hidden))
        expected = window[-1] + ((np.asarray(model["w2"]) @ hidden + model["b2"]) * model["scaler"]["scale"]
                                 + model["scaler"]["mean"])
        np.testing.assert_allclose(mlp.predict(window, u, 7), expected, rtol=1e-12)
        with self.assertRaisesRegex(ValueError, "declared window"):
            ablation.HistoryMLP(dict(model, window=3))
        a = rng.normal(size=(d, d))
        q = a @ a.T + np.eye(d)
        r = 0.25 * q + 1e-3 * np.eye(d)
        constants = ablation.history_filter(q, r)
        s = q + r
        np.testing.assert_allclose(constants["gain"], q @ np.linalg.inv(s), atol=1e-12)
        np.testing.assert_allclose(constants["posterior"], (np.eye(d) - constants["gain"]) @ q, atol=1e-10)
        prior, observation, previous = rng.normal(size=d), rng.normal(size=d), rng.normal(size=d)
        z, nis, bias, whiteness, error = ablation.history_correct(prior, observation, constants, previous)
        e = observation - prior
        np.testing.assert_allclose(error, e)
        np.testing.assert_allclose(z, prior + constants["gain"] @ e)
        self.assertAlmostEqual(nis, float(e @ np.linalg.solve(s, e)), places=10)
        self.assertAlmostEqual(bias, float(np.mean(e / np.sqrt(np.diag(s)))), places=12)
        self.assertAlmostEqual(whiteness, float(e @ previous / (np.linalg.norm(e) * np.linalg.norm(previous))),
                               places=12)
        self.assertEqual(ablation.history_correct(prior, observation, constants)[3], 0.0)   # no previous innovation


# ============================================================================= planning
class PlanningTests(Workspace):
    def test_modules_plan_writes_variant_lifecycles_a_replicate_and_one_measurement(self):
        path = ablation.plan_modules([ARXIV], self.root / "s17", reps=30, **FIXTURE)
        plan = json.loads(path.read_text())
        lifecycles = [j for j in plan["jobs"] if j["kind"] == "lifecycle"]
        expected = ["complete"] + [f"{m}:{a}" for m, arms in ablation.LIFECYCLE_MODULES.items() for a in arms]
        self.assertEqual([j["variant"]["id"] for j in lifecycles], expected)
        measure, = [j for j in plan["jobs"] if j["kind"] == "measure"]
        self.assertEqual(measure["job"], len(lifecycles))
        self.assertEqual([a["name"] for a in measure["arms"]], expected)
        self.assertEqual(measure["candidate"]["route"], "radius")
        self.assertEqual(measure["candidate"]["profile"], "full")
        self.assertEqual(measure["role"], "compare")
        self.assertEqual(measure["bootstrap"], {"reps": 30, "seed": config_of(ARXIV)["seed"] + 23})
        self.assertEqual(plan["record_set"], "module_ablations")
        self.assertEqual(plan["installer"], ablation.installer_identity())
        for job in lifecycles:
            config = json.loads(Path(job["config"]).read_text())
            self.assertEqual(canonical(config), job["config_sha256"])
            self.assertEqual(config[ablation.VARIANT_KEY], job["variant"])
            self.assertEqual(ablation._resolved(Path(job["config"])), config)
            self.assertEqual(job["through"], "confirm")
            self.assertTrue(job["run"].startswith(str(self.root / "s17" / "runs")))
        self.assertEqual(lifecycles[0]["variant"]["operations"], [])
        history = next(j for j in lifecycles if j["variant"]["id"] == "observer:history_mlp")
        self.assertEqual(history["variant"]["operations"],
                         [{"op": "observer", "kind": "history_mlp", "window": 4, "width": 128}])
        self.assertEqual(plan["settings"]["history_mlp"]["width"], 128)
        self.assertIsNone(plan["settings"]["history_mlp"]["window"])            # rule: two observer periods
        public = ensemble_grid.public_plan(self.root / "s17")
        self.assertNotIn(str(EXAMPLES), public)

    def test_scramble_pool_holds_parent_measurements_at_every_depth(self):
        path = ablation.plan_modules([ARXIV], self.root / "s17", modules=["spectral_bands:scrambled"], reps=30,
                                     scramble_units=5, **FIXTURE)
        plan = json.loads(path.read_text())
        job = next(j for j in plan["jobs"] if j["kind"] == "lifecycle" and j["variant"]["id"].endswith("scrambled"))
        op = job["variant"]["operations"][0]
        pool_file = Path(job["pool_file"])
        self.assertEqual(sha256(pool_file), op["pool"]["sha256"])
        self.assertEqual((op["pool"]["units"], op["pool"]["depths"]), (5, 7))
        with np.load(pool_file) as z:
            block, files = z["block"], list(z["trajectory_files"])
        with np.load(ARXIV / "explore" / files[0]) as z:
            order = np.argsort(z["depth"])
            np.testing.assert_array_equal(block[0, :6], z["y"][order][:, op["indices"]])
            np.testing.assert_array_equal(block[0, 6], z["y_next"][order][-1, op["indices"]])
        self.assertEqual(op["seed"], ablation.DEFAULT_SCRAMBLE_SEED)

    def test_e18_and_parent_complete_arm(self):
        path = ablation.plan_modules([ARXIV], self.root / "e18", study="e18", complete="parent", reps=30, **FIXTURE)
        plan = json.loads(path.read_text())
        self.assertEqual(plan["record_set"], "spectral_policy_value")
        lifecycle, measure = plan["jobs"]
        self.assertEqual(lifecycle["variant"]["id"], "spectral_coordinates:deleted")
        self.assertEqual(lifecycle["variant"]["operations"][0]["indices"], list(range(9, 25)))
        self.assertEqual(measure["arms"][0]["source"], "parent")
        self.assertEqual(measure["arms"][0]["config_sha256"], canonical(config_of(ARXIV)))

    def test_refusals(self):
        with self.assertRaisesRegex(RunnerError, "runtime source"):
            ablation.plan_modules([ARXIV], self.root / "a", allow_fixture=True)
        with self.assertRaisesRegex(RunnerError, "feedback route"):
            ablation.plan_modules([ARXIV], self.root / "b", allow_source_change=True)
        with self.assertRaisesRegex(RunnerError, "outside the repository"):
            ablation.plan_modules([ARXIV], ROOT / "ablation-output", **FIXTURE)
        with self.assertRaisesRegex(RunnerError, "unknown module"):
            ablation.plan_modules([ARXIV], self.root / "c", modules=["bands"], **FIXTURE)
        with self.assertRaisesRegex(RunnerError, "no variant arm"):
            ablation.plan_modules([ARXIV], self.root / "d", modules=["observer:conditioned"], **FIXTURE)
        with self.assertRaisesRegex(RunnerError, "decision roles"):
            ablation.plan_modules([ARXIV], self.root / "e", role="confirm", **FIXTURE)
        self.assertFalse((ROOT / "ablation-output").exists())
        ablation.plan_modules([ARXIV], self.root / "f", modules=["observer:raw"], **FIXTURE)
        with self.assertRaisesRegex(RunnerError, "exists"):
            ablation.plan_modules([ARXIV], self.root / "f", modules=["observer:raw"], **FIXTURE)

    def test_factorial_fit_plans_one_lifecycle_per_cell_variant_and_the_fitted_factorial(self):
        path = ablation.plan_factorial_fit([ARXIV], self.root / "fac", reps=30, **FIXTURE)
        plan = json.loads(path.read_text())
        self.assertEqual(len(plan["jobs"]), 8)
        cells = {c["cell"]: c for c in plan["cells"]}
        self.assertEqual(cells["111"]["operations"], [])
        self.assertEqual(plan["jobs"][cells["111"]["lifecycle_job"]]["variant"]["id"], "complete")
        self.assertEqual(plan["jobs"][cells["010"]["lifecycle_job"]]["variant"]["id"], "factorial-A0-O1-X0")
        fac = json.loads((self.root / "fac" / "factorial" / "plan.json").read_text())
        self.assertEqual(fac["mode"], "fitted")
        self.assertEqual(fac["fitted_plan"]["plan_sha256"], sha256(path))
        for cell in fac["jobs"][0]["cells"]:
            self.assertEqual(cell["fitted"]["operations"], ablation.factorial_operations(cell["cell"], ("A", "O", "X")))
            self.assertEqual(cell["fitted"]["run"], plan["jobs"][cells[cell["cell"]]["lifecycle_job"]]["run"])
            self.assertEqual(cell["definition"]["stopping"] is not None, cell["cell"][1] == "1")   # O = 1 cells stop
            self.assertNotIn("switch_variant", cell)
        self.assertEqual(len({canonical(c["definition"]) for c in fac["jobs"][0]["cells"]}), 8)
        self.assertEqual(fac["stopping_rule"]["threshold_nats"], 1e-7)
        parent = ablation.plan_factorial_fit([ARXIV], self.root / "fac2", complete="parent", through="fit", **FIXTURE)
        plan = json.loads(parent.read_text())
        self.assertEqual(len(plan["jobs"]), 7)
        self.assertEqual({j["through"] for j in plan["jobs"]}, {"fit"})
        self.assertEqual(next(c for c in plan["cells"] if c["cell"] == "111")["source"], "parent")
        with self.assertRaisesRegex(RunnerError, "switch table"):
            factorial.plan_factorial([ARXIV], self.root / "fac3", allow_fixture=True, mode="fitted",
                                     fitted=self.root / "fac", switches={"A": ["action_bank", ["nominal", "registered"]]})
        with self.assertRaisesRegex(RunnerError, "mode is"):
            factorial.plan_factorial([ARXIV], self.root / "fac4", allow_fixture=True, mode="fitted")
        with self.assertRaisesRegex(RunnerError, "one ogbn-arxiv or PascalVOC-SP"):
            ablation.plan_factorial_fit([PEPTIDES], self.root / "fac5", **FIXTURE)


# ============================================================================= statistics
def arm_files(directory, name, *, losses, weights, risk, entropy, false_safe, warnings, rows_sha="r"):
    path = Path(directory) / "arms" / f"{ensemble_grid.slug(name.replace(':', '__'))}.json"
    digest = ablation._save_npz(path.with_suffix(".npz"), unit_id=np.arange(len(losses)), unit_weight=weights,
                                unit_loss=losses, warnings=np.asarray(warnings, dtype=np.int64).reshape(-1, 6),
                                decisions=np.zeros((0, 10)))
    ablation._dump(path, {"risk": {"rmse": risk, "rows_sha256": rows_sha}, "action_entropy_nats": entropy,
                          "false_safe_rate": false_safe, "truncated": False, "work": {"deployed_work": 1.0},
                          "arrays": {"file": path.with_suffix(".npz").name, "sha256": digest}})
    return path


class StatisticsTests(Workspace):
    def test_paired_contrast_of_two_arms(self):
        weights = np.array([3.0, 1.0, 2.0, 4.0])
        complete = np.array([0.80, 0.90, 0.85, 0.70])
        ablated = complete + np.array([0.01, -0.02, 0.03, 0.005])
        warnings_c = [(0, 1, 4, 1, 1, 3), (0, 2, 6, 0, 6, 0), (1, 1, 2, 1, 0, 2)]
        warnings_a = [(0, 1, 4, 1, 3, 1), (0, 2, 6, 0, 2, 4), (1, 1, 2, 1, 2, 0)]
        arm_files(self.root, "complete", losses=complete, weights=weights, risk=0.010, entropy=1.2, false_safe=0.10,
                  warnings=warnings_c)
        arm_files(self.root, "spectral_bands:deleted", losses=ablated, weights=weights, risk=0.013, entropy=1.0,
                  false_safe=0.15, warnings=warnings_a)
        result = ablation.arm_contrast(ablation._arm_arrays(self.root, "complete"),
                                       ablation._arm_arrays(self.root, "spectral_bands:deleted"), reps=200, seed=3)
        values = result["values"]
        self.assertAlmostEqual(values["loss_change_nats"], float(np.dot(weights, ablated - complete) / weights.sum()),
                               places=12)
        self.assertAlmostEqual(values["risk_rmse_change"], 0.003, places=12)
        self.assertAlmostEqual(values["action_entropy_change"], -0.2, places=12)
        self.assertAlmostEqual(values["false_safe_rate_change"], 0.05, places=12)
        self.assertAlmostEqual(values["warning_lead_change_depths"], ((1 - 3) + (0 - 2)) / 2)   # pairs with an onset
        self.assertEqual(result["pairs_with_onset"], 2)
        self.assertEqual(result["onset_mismatches"], 0)
        self.assertLessEqual(result["loss_change"]["lower"], result["loss_change"]["upper"])
        record, arrays = ablation._arm_arrays(self.root, "spectral_bands:deleted")
        without = ablation.arm_contrast(ablation._arm_arrays(self.root, "complete"),
                                        (dict(record, terminal_model_available=False), arrays), reps=50, seed=3)
        self.assertNotIn("warning_lead_change_depths", without["values"])     # no terminal model: not measurable
        arm_files(self.root, "observer:raw", losses=ablated[:3], weights=weights[:3], risk=0.02, entropy=1.0,
                  false_safe=None, warnings=warnings_a, rows_sha="other")
        with self.assertRaisesRegex(RunnerError, "different units"):
            ablation.arm_contrast(ablation._arm_arrays(self.root, "complete"),
                                  ablation._arm_arrays(self.root, "observer:raw"), reps=50, seed=1)

    def test_warning_summary_pooling_and_split_intervals(self):
        self.assertEqual(ablation.warning_summary([0.9, 0.8, 0.7, 0.75, 0.9], [-1, 0.2, 0.1, 0.3]),
                         {"onset": 2, "has_onset": True, "first_warning": 1, "lead": 1})
        self.assertEqual(ablation.warning_summary([0.9, 0.8, 0.7], [-1, np.nan]),
                         {"onset": 2, "has_onset": False, "first_warning": 2, "lead": 0})
        self.assertAlmostEqual(ablation.pool_values("innovation_rmse", [0.03, 0.04]), math.sqrt((0.0009 + 0.0016) / 2))
        self.assertAlmostEqual(ablation.pool_values("loss_change_nats", [0.003, 0.005]), 0.004)
        low, high = ablation.split_interval([0.001, 0.002, 0.003])
        self.assertAlmostEqual((low + high) / 2, 0.002)
        self.assertAlmostEqual(high - 0.002, 4.303 * 0.001 / math.sqrt(3), places=12)
        with self.assertRaises(RunnerError):
            ablation.split_interval([0.1])


# ============================================================================= comparator branch
def gate_directory(root, kind, parent, rows, predictions=None):
    """A sealed input_gate / negative_controls result directory as cm_experiments.predictability writes it."""
    root = Path(root)
    (root / "raw").mkdir(parents=True)
    (root / "plan.json").write_text(json.dumps({"evaluation_role": "test"}))
    (root / "results.json").write_text(json.dumps({"truncated": False}))
    (root / "base_config.json").write_text(json.dumps(config_of(parent)))
    (root / "experiment.json").write_text(json.dumps({"experiment": kind, "parent": {
        "config_sha256": canonical(config_of(parent)), "name": Path(parent).name}}))
    name = "deployment_units.csv" if kind == "input_gate" else "units.csv"
    with (root / "raw" / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if predictions is not None:
        np.savez_compressed(root / "raw" / "predictions.npz", **predictions)
    seal(root)
    return root


def comparator_fixture(root, parent=ARXIV):
    rng = np.random.default_rng(4)
    q_c = rng.dirichlet(np.ones(4), 30).astype(np.float32)
    q_m = rng.dirichlet(np.ones(4), 30).astype(np.float32)
    q_m[0] = np.nan                                             # a numerical failure: q_C serves both arms
    gated = np.where(np.arange(30)[:, None] < 10, q_c, np.nan_to_num(q_m))
    gated[0] = q_c[0]
    y = rng.integers(0, 4, 30)
    clean = [{"unit_id": f"core:{k}", "comparator_loss": repr(0.8), "controller_loss": repr(0.8 + d),
              "deployed_loss": repr(0.8 if k < 1 else 0.8 + d)} for k, d in enumerate((0.03, 0.01, -0.02))]
    controls = [{"condition": "graph irrelevant", "comparator_loss": repr(0.9), "controller_loss": repr(0.95),
                 "deployed_loss": repr(0.9)},
                {"condition": "sensor bias", "comparator_loss": repr(0.9), "controller_loss": "",
                 "deployed_loss": repr(0.9)}]
    gate = gate_directory(root / f"gate-{parent.name}", "input_gate", parent, clean,
                          {"y_true": y, "qC": q_c, "qM": q_m, "qD_gated": gated})
    negative = gate_directory(root / f"controls-{parent.name}", "negative_controls", parent, controls)
    return gate, negative, (y, q_c, q_m, gated)


class ComparatorBranchTests(Workspace):
    def test_branch_on_and_off_from_sealed_gate_runs(self):
        from cross_morse.tasks import metric_score, task_spec
        gate, negative, (y, q_c, q_m, gated) = comparator_fixture(self.root)
        path = ablation.plan_modules([ARXIV], self.root / "s17", modules=["observer:raw"], input_gate=[gate],
                                     negative_controls=[negative], **FIXTURE)
        entry, = json.loads(path.read_text())["posthoc"]
        result = ablation.comparator_branch(entry)
        spec = task_spec(config_of(ARXIV))
        off = np.where(np.isfinite(q_m).all(axis=1)[:, None], q_m, q_c)
        expected = 100 * (metric_score(off, y, spec) - metric_score(gated, y, spec))
        self.assertAlmostEqual(result["values"]["clean_score_change"], expected, places=10)
        # branch off: harms 0.03, 0.01, -0.02, 0.05, 0 (failure served by q_C); branch on: 0, 0.01, -0.02, 0, 0
        self.assertAlmostEqual(result["values"]["worst_harm_change_nats"], 0.05 - 0.01, places=12)
        self.assertEqual(result["worst_harm"]["populations"], ["clean", "graph irrelevant", "sensor bias"])
        self.assertEqual(result["clean_score"]["numerical_failures_served_by_qC"], 1)
        (gate / "raw" / "deployment_units.csv").write_text("changed")
        with self.assertRaisesRegex(ValueError, "Artifact mismatch"):
            ablation.comparator_branch(entry)

    def test_gate_runs_must_belong_to_a_planned_parent(self):
        gate, _, _ = comparator_fixture(self.root, parent=PEPTIDES)
        with self.assertRaisesRegex(RunnerError, "belongs to no planned parent"):
            ablation.plan_modules([ARXIV], self.root / "s17", modules=["observer:raw"], input_gate=[gate], **FIXTURE)
        with self.assertRaisesRegex(RunnerError, "not a sealed experiment"):
            ablation.plan_modules([ARXIV], self.root / "s18", modules=["observer:raw"], input_gate=[self.root],
                                  **FIXTURE)


# ============================================================================= collection and export
def fake_measurement(output, plan, rng, *, units=6):
    """Synthetic measurement outputs in the layout of run_measure_job (exercises collection and export only)."""
    for entry in [j for j in plan["jobs"] if j["kind"] == "measure"]:
        directory = Path(output) / "jobs" / entry["slug"]
        weights = rng.integers(1, 5, units).astype(float)
        base = rng.uniform(0.6, 0.9, units)
        warnings = [(0, s, 4, 1, int(rng.integers(0, 4)), 0) for s in (1, 2)]
        for arm in entry["arms"]:
            shift = 0.0 if arm["name"] == "complete" else rng.uniform(-0.002, 0.006)
            arm_files(directory, arm["name"], losses=base + shift + rng.normal(0, 1e-4, units), weights=weights,
                      risk=0.01 + rng.uniform(0, 0.005), entropy=1.0 + rng.uniform(0, 0.3),
                      false_safe=0.1 + rng.uniform(0, 0.05), warnings=[w[:5] + (4 - w[4],) for w in warnings])
        ablation._dump(directory / "contrasts.json", ablation.job_contrasts(directory, entry))
        ablation._dump(directory / "environment.json", ENVIRONMENT)
        ablation._dump(directory / "measure.json", {"started_utc": "2026-11-07T10:00:00+00:00",
                                                    "finished_utc": "2026-11-07T11:00:00+00:00", "job": entry["job"]})


def module_fixture(root):
    """Two hand-built sealed runs holding the sealed contrasts of the three post-hoc modules (synthetic)."""
    candidates = [
        {"route": "nominal", "profile": "none", "ensemble": 2, "cost": 2.0, "complexity": 0, "hash": "n"},
        {"route": "one_step", "profile": "full", "ensemble": 2, "cost": 4.0, "complexity": 6, "hash": "o"},
        {"route": "mpc_rh", "profile": "full", "ensemble": 2, "horizon": 2, "cost": 6.0, "complexity": 8, "hash": "f"},
        {"route": "mpc_rh", "profile": "lean", "ensemble": 2, "horizon": 2, "cost": 5.0, "complexity": 8, "hash": "l"},
        {"route": "radius", "profile": "full", "ensemble": 2, "cost": 5.5, "complexity": 7, "hash": "r"}]
    stages = {f"stages/{s}.json": {"status": "COMPLETE"} for s in ("select", "confirm", "admit")}
    stages["run_state.json"] = {"status": "COMPLETE", "source_sha256": "0" * 64}

    def write(base, files):
        for relative, content in files.items():
            path = base / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(content))
        return base
    pascal = write(Path(root) / "pascal", {
        **stages, "config.json": {"dataset": "PascalVOC-SP", "scientific_run": False},
        "select/candidates.json": candidates,
        "select/decision.json": {"mean": [0.0, -0.002, -0.006, -0.005, -0.003], "provisional_index": 2},
        "confirm/decision.json": {"provisional_index": 2, "comparator_index": 0, "contrasts": [
            {"kind": "full", "candidate_index": 2, "predecessor_index": 3, "mean": 0.08, "available": True}]},
        "policy_lock.json": {"deployed_index": 2}})
    peptides = write(Path(root) / "peptides", {
        **stages, "config.json": {"dataset": "Peptides-func", "scientific_run": False},
        "select/candidates.json": candidates,
        "select/decision.json": {"mean": [0.0, -0.001, -0.002, -0.002, -0.0035], "provisional_index": 4},
        "confirm/decision.json": {"provisional_index": 4, "comparator_index": 0, "contrasts": [
            {"kind": "radius", "candidate_index": 4, "predecessor_index": 1, "mean": -0.0025}]},
        "policy_lock.json": {"deployed_index": 4}})
    return pascal, peptides


class CollectionAndExportTests(Workspace):
    def plan(self, **kwargs):
        gate, negative, _ = comparator_fixture(self.root)
        options = dict(reps=40, input_gate=[gate], negative_controls=[negative], **FIXTURE)
        options.update(kwargs)
        path = ablation.plan_modules([ARXIV, PEPTIDES], self.root / "s17", **options)
        return json.loads(path.read_text())

    def test_collection_averages_runs_pools_datasets_and_traces_files(self):
        plan = self.plan()
        with self.assertRaisesRegex(RunnerError, "has not completed"):
            ablation.collect_study(self.root / "s17")
        fake_measurement(self.root / "s17", plan, np.random.default_rng(3))
        results = json.loads(ablation.collect_study(self.root / "s17").read_text())
        rows = {(r["module"], r["arm"], r["dataset"]): r for r in results["rows"]}
        arms = [(m, a) for m, module_arms in ablation.LIFECYCLE_MODULES.items() for a in module_arms]
        for module, arm in arms + [("observer", "conditioned")]:
            self.assertIn((module, arm, "pooled"), rows)
            datasets = ("ogbn-arxiv", "Peptides-func")
            for dataset in datasets:      # the observer's term reads one row per arm: pooled rows only
                self.assertEqual((module, arm, dataset) in rows, module != "observer")
            if module == "observer":
                self.assertEqual(set(rows[(module, arm, "pooled")]["detail"]["datasets"]), set(datasets))
        self.assertIn(("comparator_branch", "branch_off", "ogbn-arxiv"), rows)
        self.assertNotIn(("comparator_branch", "branch_off", "pooled"), rows)            # one dataset only
        deleted = rows[("spectral_bands", "deleted", "ogbn-arxiv")]
        self.assertEqual(set(deleted["values"]), {"loss_change_nats", "risk_rmse_change", "action_entropy_change",
                                                  "false_safe_rate_change", "warning_lead_change_depths"})
        contrasts = json.loads((self.root / "s17" / "jobs" / "ogbn-arxiv__arxiv" / "measure" /
                                "contrasts.json").read_text())
        self.assertEqual(deleted["values"]["loss_change_nats"],
                         contrasts["contrasts"]["spectral_bands:deleted"]["values"]["loss_change_nats"])
        pooled = rows[("observer", "raw", "pooled")]["values"]["innovation_rmse"]
        per = [rows[("observer", "raw", "pooled")]["detail"]["datasets"][d]["innovation_rmse"]
               for d in ("ogbn-arxiv", "Peptides-func")]
        self.assertAlmostEqual(pooled, math.sqrt((per[0] ** 2 + per[1] ** 2) / 2), places=12)
        for item in results["deposits"]:
            self.assertEqual(sha256(self.root / "s17" / item["path"]), item["sha256"])
        self.assertTrue(any(p["path"].startswith("posthoc/") for p in results["deposits"]))
        verified = ablation.module_results(self.root / "s17")
        self.assertEqual(len(verified["rows"]), len(results["rows"]))
        (self.root / "s17" / "jobs" / "ogbn-arxiv__arxiv" / "measure" / "contrasts.json").write_text("{}")
        with self.assertRaisesRegex(RunnerError, "changed since collection"):
            ablation.module_results(self.root / "s17")

    def test_ten_module_record_set_through_the_sweep_and_the_s17_display(self):
        from paper.build.displays.testing import build_one, context, fake_records
        from scripts import export_sweep_records
        from scripts.records import read_summary, verify_set
        plan = self.plan()
        fake_measurement(self.root / "s17", plan, np.random.default_rng(5))
        ablation.collect_study(self.root / "s17")
        pascal, peptides = module_fixture(self.root)
        sweep = sweeps.plan_sweep("module_ablations", [pascal, peptides], self.root / "sweep",
                                  ablations=[self.root / "s17"])
        results = json.loads(sweeps.collect(sweep).read_text())
        self.assertEqual(results["unsupported"], [])
        self.assertEqual(results["not_measured"], [])
        self.assertTrue(results["deposits"])
        records = self.root / "records"
        records.mkdir()
        target = export_sweep_records.export(self.root / "sweep" / "results.json", root=records, hardware="CPU",
                                             period="2026-11-07/2026-11-07", replace=False, argv=["test"])
        all_specs = specs()
        spec = all_specs["sets"]["module_ablations"]
        entry = verify_set(target, spec, all_specs)
        self.assertTrue(entry["synthetic"])
        self.assertTrue((target / "raw" / "ablations" / "s17" / "results" / "module_ablations.json").is_file())
        source = json.loads((target / "code" / "SOURCE.json").read_text())
        self.assertIn("cm_experiments/ablation_lifecycle.py", source["experiment_code"])
        rows = read_summary(target, spec, all_specs["summary_columns"])
        self.assertEqual({r["module"] for r in rows}, {m["module"] for m in spec["modules"]})
        observer_rows = {r["arm"] for r in rows if r["module"] == "observer" and r["quantity"] == "innovation_rmse"}
        self.assertEqual(observer_rows, {"raw", "ema", "global", "conditioned", "history_mlp"})   # the extra E8 arm
        out = self.root / "display"
        view = fake_records("module_ablations", rows, spec=spec)
        build_one("supplementary:Table S17", context(records={"module_ablations": view}), out)
        with (out / "tables" / "S17.csv").open(newline="") as handle:
            table = list(csv.DictReader(handle))
        self.assertEqual(len(table), 10)
        self.assertTrue(all(r["observed signature"] for r in table), table)
        observer = next(r for r in table if r["module"] == "observer")
        self.assertEqual(observer["observed signature"].count("/"), 3)     # the generator prints the four registered
        self.assertNotIn("No observations", (out / "tables" / "S17.md").read_text())

    def test_sweep_refuses_changed_ablation_results_and_other_sweeps(self):
        plan = self.plan(input_gate=(), negative_controls=())
        fake_measurement(self.root / "s17", plan, np.random.default_rng(6))
        ablation.collect_study(self.root / "s17")
        with self.assertRaisesRegex(sweeps.SweepError, "only the module_ablations"):
            sweeps.plan_sweep("sensitivity_grids", [ARXIV], self.root / "grids", ablations=[self.root / "s17"])
        self.assertFalse((self.root / "grids").exists())
        pascal, peptides = module_fixture(self.root)
        results = self.root / "s17" / "results" / "module_ablations.json"
        original = results.read_text()
        results.write_text(json.dumps(dict(json.loads(original), truncated=True, synthetic=False)))
        with self.assertRaisesRegex(sweeps.SweepError, "truncated by --max-graphs"):
            sweeps.plan_sweep("module_ablations", [pascal, peptides], self.root / "sweep-truncated",
                              ablations=[self.root / "s17"])
        results.write_text(original)
        path = sweeps.plan_sweep("module_ablations", [pascal, peptides], self.root / "sweep",
                                 ablations=[self.root / "s17"])
        results.write_text(results.read_text().replace('"study"', '"study" ', 1))
        with self.assertRaisesRegex(sweeps.SweepError, "changed since it was attached"):
            sweeps.collect(path)
        with self.assertRaisesRegex(sweeps.SweepError, "not a collected modules study"):
            sweeps.plan_sweep("module_ablations", [pascal], self.root / "sweep2", ablations=[self.root])

    def test_e18_export_against_the_registered_contract(self):
        from scripts.records import verify_set
        path = ablation.plan_modules([ARXIV], self.root / "e18", study="e18", reps=40, **FIXTURE)
        plan = json.loads(path.read_text())
        fake_measurement(self.root / "e18", plan, np.random.default_rng(8))
        registered = specs()
        self.assertEqual(registered["sets"]["spectral_policy_value"], json.loads(json.dumps(ablation.E18_CONTRACT)))
        target = ablation.export_e18(self.root / "e18", root=self.root / "records",
                                     command=["scripts/run_ablation.py", "e18-policy", "export"])
        verify_set(target, registered["sets"]["spectral_policy_value"], registered)
        with self.assertRaisesRegex(RunnerError, "complete measured sets"):
            ablation.export_e18(self.root / "e18", root=ensemble_grid.RECORDS)
        with (target / "summary.csv").open(newline="") as handle:
            lines = list(csv.DictReader(handle))
        self.assertEqual({r["quantity"] for r in lines}, set(ablation.E18_CONTRACT["quantities"]))
        contrasts = json.loads((self.root / "e18" / "jobs" / "ogbn-arxiv__arxiv" / "measure" /
                                "contrasts.json").read_text())["contrasts"]["spectral_coordinates:deleted"]
        value = next(r for r in lines if r["quantity"] == "policy_value_nats")
        self.assertAlmostEqual(float(value["value"]), contrasts["values"]["loss_change_nats"], places=9)
        low = next(r for r in lines if r["quantity"] == "interval_low_nats")
        self.assertAlmostEqual(float(low["value"]), contrasts["loss_change"]["lower"], places=9)
        self.assertTrue((target / "code" / "cm_experiments" / "ablation_lifecycle.py").is_file())


# ============================================================================= fitted factorial orchestration
class FittedFactorialOrchestrationTests(Workspace):
    def test_fitted_cells_execute_inside_their_variant_scope_and_export(self):
        from tests.test_experiments_runners_a import FakeContext, fake_execute
        ablation.plan_factorial_fit([ARXIV], self.root / "fac", reps=30, warmup=1, **FIXTURE)
        output = self.root / "fac" / "factorial"
        scopes, entered = {}, []

        def fake_scopes(plan, entry, run):
            self.assertEqual(plan["mode"], "fitted")
            for cell in entry["cells"]:
                scopes[cell["cell"]] = (Path(cell["fitted"]["run"]), cell["fitted"]["variant"])
            return scopes

        @contextlib.contextmanager
        def fake_scope(found, cell):
            entered.append(cell)
            yield

        def controller(ctx, cell, root):
            return "nominal" if cell["switches"]["action_bank"] == "no_transport" else "registered"

        with patch.object(factorial, "ParentContext", FakeContext), patch.object(factorial, "_execute", fake_execute), \
                patch.object(factorial, "cell_scopes", fake_scopes), patch.object(factorial, "cell_scope", fake_scope), \
                patch.object(factorial, "fitted_cell_controller", controller), \
                patch.object(factorial, "environment_record", return_value=ENVIRONMENT):
            path = factorial.run_factorial(output, trust=True)[0]
        self.assertEqual(set(entered), set(factorial.CELLS))
        result = json.loads(path.read_text())
        self.assertEqual(result["mode"], "fitted")
        cells = {c["cell"]: c for c in result["cells"]}
        self.assertEqual(cells["010"]["fitted"]["variant_id"], "factorial-A0-O1-X0")
        self.assertNotIn("/", cells["010"]["fitted"]["run"])
        for name, cell in cells.items():           # the fake observer stops member 1 at depth 2 in the O = 1 cells
            self.assertEqual(cell["stopping"]["stopped"], 1 if name[1] == "1" else 0)
            self.assertEqual(cell["definition"]["observer"], "conditioned" if name[1] == "1" else "memoryless")
        self.assertEqual(cells["010"]["integration_domain"]["terminal_depths"], [2, 6])
        records = self.root / "records"
        with self.assertRaisesRegex(RunnerError, "no completion record"):
            factorial.export_factorial(output, root=records, allow_partial=True)
        lifecycles = json.loads((self.root / "fac" / "plan.json").read_text())["jobs"]
        for job in lifecycles:            # what run_lifecycle_job writes after the cell's lifecycle is sealed
            ablation._dump(self.root / "fac" / "jobs" / job["slug"] / "lifecycle.json",
                           {"schema": "cm-ablation-lifecycle-1", "job": job["job"], "variant": job["variant"],
                            "config_sha256": job["config_sha256"]})
        # This orchestration test substitutes measurement and fitting, so substitute the verified model package
        # boundary too. Real checkpoint bytes, their seals and hash failures are covered by LifecycleExecutionTests.
        def package(job, parent):
            model = self.root / "fac" / "jobs" / job["slug"] / "fixture_controller.pt"
            model.write_bytes(b"synthetic package fixture: " + job["slug"].encode())
            return {"fit/full/controller.pt": model}, {"synthetic": True, "files": {
                "fit/full/controller.pt": {"sha256": sha256(model), "bytes": model.stat().st_size}}}
        with patch.object(ablation, "controller_artifacts", side_effect=package):
            target = factorial.export_factorial(output, root=records, allow_partial=True)
        from scripts.records import verify_set
        verify_set(target, specs()["sets"]["factorials"], specs())
        provenance = json.loads((target / "provenance.json").read_text())
        self.assertEqual(provenance["mode"], "fitted")
        self.assertIn("separately fitted", provenance["produced_by"])
        self.assertTrue((target / "code" / "cm_experiments" / "ablation_lifecycle.py").is_file())
        self.assertEqual(len(lifecycles), 8)
        for job in lifecycles:
            self.assertTrue((target / "raw" / "fitted" / job["slug"] / "lifecycle.json").is_file())
            package = target / "raw" / "fitted" / job["slug"] / "controller_package"
            self.assertTrue((package / "fit/full/controller.pt").is_file())
            self.assertTrue((package / "CONTROLLERS.json").is_file())
        self.assertNotIn(str(self.root), (target / "code" / "ablation_plan.json").read_text())

    def test_a_changed_ablation_plan_is_refused(self):
        ablation.plan_factorial_fit([ARXIV], self.root / "fac", reps=30, **FIXTURE)
        plan = json.loads((self.root / "fac" / "factorial" / "plan.json").read_text())
        (self.root / "fac" / "plan.json").write_text((self.root / "fac" / "plan.json").read_text() + "\n")
        with self.assertRaisesRegex(RunnerError, "changed after planning"):
            factorial.cell_scopes(plan, plan["jobs"][0], SealedRun(ARXIV))


# ============================================================================= command line
class CommandLineTests(Workspace):
    def call(self, *argv):
        from scripts.run_ablation import main
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(list(map(str, argv)))
        return code, out.getvalue(), err.getvalue()

    def test_plan_run_guard_collect_and_export(self):
        output = self.root / "cli"
        code, out, err = self.call("modules", "plan", "--parent", ARXIV, "--output", output, "--module", "observer:raw",
                                   "--reps", 30, "--allow-fixture", "--allow-source-change")
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)["result"], str(output / "plan.json"))
        code, _, err = self.call("modules", "run", "--output", output, "--device", "cpu")
        self.assertEqual(code, 2)
        self.assertIn("trust", err)
        code, _, err = self.call("modules", "collect", "--output", output)
        self.assertEqual(code, 2)
        self.assertIn("has not completed", err)
        code, _, err = self.call("modules", "export", "--output", output)
        self.assertEqual(code, 2)
        self.assertIn("--sweep", err)
        code, _, err = self.call("e18-policy", "export", "--output", output)
        self.assertEqual(code, 2)
        self.assertIn("not an ablation e18", err)
        code, out, err = self.call("factorial-fit", "plan", "--parent", ARXIV, "--output", self.root / "fac",
                                   "--complete", "parent", "--through", "fit", "--allow-fixture",
                                   "--allow-source-change")
        self.assertEqual(code, 0, err)
        self.assertTrue((self.root / "fac" / "factorial" / "plan.json").is_file())


# ============================================================================= execution (PyTorch; CI)
@unittest.skipUnless(TORCH, "PyTorch is required to execute variant lifecycles")
class LifecycleExecutionTests(unittest.TestCase):
    """A smoke parent, then variant lifecycles, measurements and exports end to end (software checks)."""

    @classmethod
    def setUpClass(cls):
        import torch
        from cross_morse.pipeline import Pipeline
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory(prefix="cm-ablation-runs-")
        cls.root = Path(cls.temporary.name)
        config = json.loads((ROOT / "configs" / "arxiv_smoke.json").read_text())
        config["training"].update(epochs=1, patience=1)
        config["controller"].update(epochs=1, patience=1)
        config["selection"]["bootstrap_reps"] = 20
        config["admission"]["bootstrap_reps"] = 20
        cls.parent = cls.root / "runs" / "arxiv"
        with contextlib.redirect_stdout(io.StringIO()):
            Pipeline(config, cls.root / "data", cls.parent, "cpu").run()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.work = Path(tempfile.mkdtemp(prefix="cm-ablation-", dir=self.root))

    def run_study(self, output):
        with contextlib.redirect_stdout(io.StringIO()):
            return ablation.run_study(output, device="cpu", threads=1, trust=True)

    def test_installer_patches_the_runtime_for_one_block_only(self):
        import cross_morse.exploration as exploration
        import cross_morse.learning as learning
        import cross_morse.policies as policies
        config = config_of(self.parent)
        pool = self.work / ablation.POOL_FILE
        names = ablation.module_coordinates("source_alignment", config)
        record = ablation.scramble_pool(self.parent, names, ablation.coordinate_indices(names, config), config, pool,
                                        seed=1, units=4)
        ops = (ablation.module_operations("transport_history", "no_history", config)
               + [{"op": "observer", "kind": "raw"}, {"op": "remove_reset", "action": 9},
                  {"op": "scramble", "coordinates": names, "indices": ablation.coordinate_indices(names, config),
                   "seed": 1, "pool": record}])
        spec = ablation.variant_spec("all", ops, study="modules")
        before = {(owner, name): getattr(owner, name) for owner, name in (
            (learning, "ConditionedObserver"), (learning, "_fit_observer"), (learning, "_features"),
            (learning, "_support"), (learning, "_fit_one"), (exploration, "sensor_bank"),
            (exploration, "collect_trajectory"), (policies, "sensor_bank"), (policies, "recovery_action"),
            (policies, "beam_plan"), (policies, "_execute_policy"), (learning.Controller, "supported"))}
        with ablation.installed(spec, run_dir=self.work):
            self.assertIs(learning.ConditionedObserver, ablation.observer_class())
            self.assertIs(policies.sensor_bank, ablation._scrambled_sensor_bank)
            self.assertIs(learning._fit_one, before[(learning, "_fit_one")])     # only a history observer needs it
            self.assertEqual(policies.recovery_action([1, 2]), 8)            # A9 removed in feedback decisions
            token = ablation._OPEN_LOOP.set(True)
            self.assertEqual(policies.recovery_action([1, 2], last_safe_available=False), 8)
            ablation._OPEN_LOOP.reset(token)
            with self.assertRaisesRegex(RunnerError, "another ablation variant"):
                ablation.install(spec, run_dir=self.work)
        cell = ablation.variant_spec("cell", ablation.factorial_operations("000", ("A", "O", "X"))[1:]
                                     + ablation.module_operations("observer", "history_mlp", config), study="factorial")
        controller = learning.Controller.load(self.parent / "fit" / "full", "cpu")
        y = np.full(25, 0.5)
        with ablation.installed(cell, run_dir=self.work):
            self.assertIs(learning._support, ablation._support)
            self.assertIs(learning._fit_one, ablation._fit_one)
            self.assertEqual(policies.recovery_action([1, 2]), 0)            # risk control off: nominal fallback
            self.assertEqual(controller.supported(y, 0), [0])                # A0 only: no transport, no reset
            self.assertEqual(controller.supported(np.full(25, np.nan), 0), [])
        self.assertEqual(policies.recovery_action([1, 2]), 9)
        after = {(owner, name): getattr(owner, name) for owner, name in before}
        self.assertEqual(after, before)
        self.assertIsNone(ablation.active_spec())

    def test_history_mlp_observer_lifecycle_and_measurement(self):
        from cross_morse.exploration import load_records
        from cross_morse.learning import Controller, _groups
        output = self.work / "history"
        ablation.plan_modules([self.parent], output, role="test", complete="parent", through="fit", reps=20,
                              allow_fixture=True, modules=["observer:history_mlp"])
        plan = json.loads((output / "plan.json").read_text())
        self.run_study(output)
        job = next(j for j in plan["jobs"] if j["kind"] == "lifecycle")
        run = Path(job["run"])
        config = config_of(self.parent)
        window, d = 2 * config["controller"]["observer_period"], config["controller"]["state_dim"]
        for fold in ("folds/fold_00", "final_fit"):
            self.assertTrue((run / "fit" / "full" / fold / "history_mlp.last.pt").is_file())
        records = load_records(run / "explore")
        np.testing.assert_array_equal(_groups(records), np.unique(np.column_stack(
            (records["seed_index"], records["trajectory"])), axis=0, return_inverse=True)[1])
        with ablation.installed(job["variant"], run_dir=run):
            controller = Controller.load(run / "fit" / "full", "cpu")
            observer = controller.observer
            self.assertEqual(observer.kind, "history_mlp")
            fit = observer.payload["ablation_fit"]["history_fit"]
            self.assertEqual((fit["window"], fit["width"]), (window, 128))
            self.assertEqual(fit["parameters"], ablation.history_parameters(window, 128, d))
            self.assertEqual(ablation.observer_parameters(observer)["transition_parameters"], fit["parameters"])
            rows = np.flatnonzero((records["seed_index"] == records["seed_index"][0])
                                  & (records["trajectory"] == records["trajectory"][0]))
            rows = rows[np.argsort(records["depth"][rows])]
            state = controller.start(records["y"][rows[0]])
            self.assertEqual(state.history_window.shape, (window, d))
            for i in rows[1:]:                      # every update equals the declared predictor-corrector
                depth, previous = int(records["depth"][i]), int(records["previous_action"][i])
                prior = observer.history.predict(state.history_window, observer.normalize(state.y)[5:8], previous)
                state_next = controller.observe(state, records["y"][i], depth, previous)
                if depth % observer.period:
                    np.testing.assert_allclose(state_next.z, prior)
                else:
                    z, nis, *_ = ablation.history_correct(prior, observer.transform(records["y"][i]),
                                                          observer._history, state.innovation)
                    np.testing.assert_allclose(state_next.z, z)
                    self.assertAlmostEqual(state_next.nis, nis)
                state = state_next
        measure = next(j for j in plan["jobs"] if j["kind"] == "measure")
        directory = output / "jobs" / measure["slug"]
        contrasts = json.loads((directory / "contrasts.json").read_text())["contrasts"]
        self.assertTrue(math.isfinite(contrasts["observer:history_mlp"]["values"]["innovation_rmse"]))
        arm = json.loads((directory / "arms" / "observer__history_mlp.json").read_text())
        self.assertEqual(arm["observer_parameters"]["kind"], "history_mlp")
        results = json.loads(ablation.collect_study(output).read_text())
        self.assertIn(("observer", "history_mlp", "ogbn-arxiv"),
                      {(r["module"], r["arm"], r["dataset"]) for r in results["rows"]})

    def test_module_lifecycles_measure_collect_and_export(self):
        from cross_morse.learning import Controller
        output = self.work / "s17"
        ablation.plan_modules([self.parent], output, role="test", reps=20, allow_fixture=True,
                              modules=["spectral_bands:scrambled", "observer:raw", "recovery_type:source_only",
                                       "transport_history"])
        plan = json.loads((output / "plan.json").read_text())
        self.run_study(output)
        jobs = {j["variant"]["id"]: j for j in plan["jobs"] if j["kind"] == "lifecycle"}
        parent = SealedRun(self.parent)
        for job in jobs.values():
            run = ablation.open_lifecycle_run(job, parent)
            for stage in ("prepare", "train"):
                self.assertIn("seeded_from_parent", json.loads((run.root / "stages" / f"{stage}.json").read_text()))
            for checkpoint in sorted((self.parent / "train").glob("*.pt")):
                self.assertEqual(sha256(run.root / "train" / checkpoint.name), sha256(checkpoint))
        replicate = Path(jobs["complete"]["run"])
        for path in sorted((self.parent / "predictions").glob("*.npy")):        # re-execution reproduces the parent
            np.testing.assert_allclose(np.load(replicate / "predictions" / path.name), np.load(path), atol=1e-6)
        scrambled = Path(jobs["spectral_bands:scrambled"]["run"])
        op = jobs["spectral_bands:scrambled"]["variant"]["operations"][0]
        with np.load(scrambled / ablation.POOL_FILE) as z:
            pool = z["block"]
        for name in ("trajectory_000000.npz", "trajectory_000005.npz"):
            with np.load(self.parent / "explore" / name) as a, np.load(scrambled / "explore" / name) as b:
                kept = [i for i in range(a["y"].shape[1]) if i not in op["indices"]]
                np.testing.assert_array_equal(a["y"][:, kept], b["y"][:, kept])
                np.testing.assert_array_equal(a["final_loss"], b["final_loss"])
                order = np.argsort(b["depth"])
                donors = [k for k in range(len(pool)) if np.array_equal(pool[k, :len(order)],
                                                                         b["y"][order][:, op["indices"]])]
                self.assertGreaterEqual(len(donors), 1)          # the block is a parent trajectory's, depth by depth
        history = jobs["transport_history:no_history"]
        with ablation.installed(history["variant"], run_dir=history["run"]):
            controller = Controller.load(Path(history["run"]) / "fit" / "full", "cpu")
            self.assertEqual(controller.observer.ablation["deleted"], ["normalized_hops", "normalized_mass"])
            y = np.zeros(25)
            y[[6, 7]] = [5.0, 7.0]
            normalized = controller.observer.normalize(y)
            self.assertEqual(float(normalized[6]), 0.0)
            self.assertEqual(float(normalized[7]), 0.0)
        measure = next(j for j in plan["jobs"] if j["kind"] == "measure")
        directory = output / "jobs" / measure["slug"]
        contrasts = json.loads((directory / "contrasts.json").read_text())["contrasts"]
        self.assertEqual(set(contrasts), {"spectral_bands:scrambled", "observer:raw", "recovery_type:source_only",
                                          "transport_history:no_history", "observer:conditioned"})
        for name, contrast in contrasts.items():
            for value in contrast["values"].values():
                self.assertTrue(math.isfinite(value), name)
        complete = json.loads((directory / "arms" / "complete.json").read_text())
        self.assertLess(complete["work"]["replay_max_abs_deviation"], 1e-5)
        self.assertEqual(complete["profile"], "full")
        results = json.loads(ablation.collect_study(output).read_text())
        self.assertIn(("observer", "raw", "ogbn-arxiv"), {(r["module"], r["arm"], r["dataset"]) for r in results["rows"]})
        records = self.work / "records"
        target = ablation.export_modules(output, self.work / "sweep", root=records, hardware="CPU",
                                         period="2026-11-07/2026-11-07", argv=["test"])
        from scripts.records import verify_set
        verify_set(target, specs()["sets"]["module_ablations"], specs())

    def test_e18_control_reproduces_the_lean_profile_and_exports(self):
        from scripts import record_mapping
        from scripts.records import verify_set
        output = self.work / "e18"
        ablation.plan_modules([self.parent], output, study="e18", role="test", complete="parent", through="select",
                              reps=20, allow_fixture=True)
        plan = json.loads((output / "plan.json").read_text())
        self.run_study(output)
        run = Path(plan["jobs"][0]["run"])
        candidates = json.loads((self.parent / "select" / "candidates.json").read_text())
        compared = 0
        for c in candidates:
            if c["route"] not in ("one_step", "radius", "mpc_rh", "mpc_ol") or c["profile"] != "full":
                continue
            lean = factorial.profile_counterpart(candidates, c, "lean")
            np.testing.assert_allclose(np.load(run / "predictions" / f"{c['candidate_id']}.npy"),
                                       np.load(self.parent / "predictions" / f"{lean['candidate_id']}.npy"), atol=1e-5)
            compared += 1
        self.assertGreater(compared, 0)
        proposed = specs()
        proposed["sets"]["spectral_policy_value"] = ablation.E18_CONTRACT
        specs_path = self.work / "specs.json"
        specs_path.write_text(json.dumps(proposed))
        with patch.object(ensemble_grid, "SPECS_PATH", specs_path), \
                patch.object(record_mapping, "_specs", return_value=proposed):
            target = ablation.export_e18(output, root=self.work / "records")
            verify_set(target, proposed["sets"]["spectral_policy_value"], proposed)

    def test_factorial_cells_fitted_separately_execute_and_export(self):
        from cross_morse.learning import Controller
        from scripts.records import verify_set
        output = self.work / "fac"
        ablation.plan_factorial_fit([self.parent], output, reps=20, warmup=0, complete="parent", through="fit",
                                    allow_fixture=True)
        plan = json.loads((output / "plan.json").read_text())
        self.run_study(output)
        jobs = {j["variant"]["id"]: j for j in plan["jobs"]}

        def critic(job):
            if job is None:
                return Controller.load(self.parent / "fit" / "full", "cpu").critic.state_dict()
            with ablation.installed(job["variant"], run_dir=job["run"]):
                return Controller.load(Path(job["run"]) / "fit" / "full", "cpu").critic.state_dict()
        parent_critic = critic(None)
        same = critic(jobs["factorial-A1-O1-X0"])           # the support is not part of the fit
        memoryless = critic(jobs["factorial-A1-O0-X1"])      # the observer is
        self.assertTrue(all(np.allclose(parent_critic[k].numpy(), same[k].numpy()) for k in parent_critic))
        self.assertFalse(all(np.allclose(parent_critic[k].numpy(), memoryless[k].numpy()) for k in parent_critic))
        path = next((output / "factorial" / "jobs").rglob("factorial.json"))
        result = json.loads(path.read_text())
        self.assertEqual(result["mode"], "fitted")
        cells = {c["cell"]: c for c in result["cells"]}
        depth = config_of(self.parent)["model"]["depth"]
        executions = np.load(path.parent / "executions.npz")
        for name, cell in cells.items():
            counts = np.asarray(cell["action_counts"])
            if name[0] == "0":                       # A = 0: no transport action A1-A7
                self.assertEqual(int(counts[1:8].sum()), 0)
            if name[2] == "0":                       # X = 0: no reset A8/A9
                self.assertEqual(int(counts[8:].sum()), 0)
            stopped = executions[f"cell{name}_stopped"]
            terminal = executions[f"cell{name}_terminal_depth"]
            self.assertTrue(np.all((terminal >= 1) & (terminal <= depth)))
            self.assertTrue(np.all((terminal < depth) == (stopped == 1)))
            if name[1] == "0":                       # the memoryless observer never stops
                self.assertEqual(int(stopped.sum()), 0)
            self.assertEqual(cell["stopping"]["stopped"], int(stopped.sum()))
            self.assertEqual(int(executions[f"cell{name}_actions"].sum()), int(np.sum(terminal)))
            self.assertEqual(cell["definition"]["admissible_actions"][0], 0)
        for name in ("000", "010"):      # A0 only, no reset: the nominal trajectory, stopped or not
            self.assertEqual(cells[name]["activity"], 0.0)
        # a read-only observer on the nominal trajectory reproduces cell 000 unless it stopped an execution
        self.assertEqual(["000", "010"] in result["coinciding_cells"], cells["010"]["stopping"]["stopped"] == 0)
        target = ablation.export_factorial_fit(output, root=self.work / "records", allow_partial=True)
        verify_set(target, specs()["sets"]["factorials"], specs())
        self.assertTrue((target / "raw" / "ogbn-arxiv" / "executions.npz").is_file())
        for job in plan["jobs"]:
            package = target / "raw" / "fitted" / job["slug"] / "controller_package"
            manifest = json.loads((package / "CONTROLLERS.json").read_text())
            for profile in config_of(self.parent)["sensor"]["profiles"]:
                relative = f"fit/{profile}/controller.pt"
                self.assertEqual(sha256(package / relative), sha256(Path(job["run"]) / relative))
                self.assertEqual(sha256(package / relative), manifest["files"][relative]["sha256"])
            self.assertEqual(manifest["parent_identity"], job["parent_identity"])
        job = plan["jobs"][0]
        import torch
        with patch.object(torch, "load", side_effect=AssertionError("publication must not unpickle checkpoints")):
            ablation.controller_artifacts(job, SealedRun(self.parent))
        model_path = Path(job["run"]) / "fit/full/controller.pt"
        original = model_path.read_bytes()
        model_path.write_bytes(original + b"changed")
        try:
            with self.assertRaisesRegex(RunnerError, "differs from its sealed controller artifact"):
                ablation.controller_artifacts(job, SealedRun(self.parent))
        finally:
            model_path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
