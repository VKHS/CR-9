"""Experiment group B runners: E2, E9, E18, negative controls and the input gate.

The NumPy-only tests cover planning against a sealed parent's declared
identities, the registered constructions and gate arithmetic, summary
recomputation from raw files, tamper refusal and the record-set exporters
(verified with scripts.records.verify_set). The PyTorch tests execute every
runner on smoke lifecycles of the released runtime and export the outputs.
Fixture values below are software fixtures in temporary roots, never records.
"""
from __future__ import annotations

import contextlib
import copy
import csv
import hashlib
import importlib.util
import io
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from cm_experiments import band_audit, input_gate, negative_controls, predictability, search_enumeration
from cm_experiments.artifacts import canonical, seal, sha256, write_json
from cm_experiments.predictability import (PLAN_SCHEMA, RESULTS_SCHEMA, RUN_SCHEMA, ExperimentError, ParentRun,
                                           open_plan)
from scripts.records import load_specs, verify_set

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/runs/arxiv"
SPECS = load_specs(ROOT / "records", ROOT / "records/specs.json")
TORCH = importlib.util.find_spec("torch") is not None


def fixture_parent(root):
    """The bundled synthetic arxiv run's JSON decisions with this checkout's runtime identity."""
    from cross_morse.provenance import current_source_sha256
    target = root / "parent"
    for relative in ("config.json", "select/candidates.json", "confirm/decision.json", "admit/decision.json",
                     "fit/registry.json", *[p.relative_to(EXAMPLE).as_posix() for p in EXAMPLE.glob("stages/*.json")],
                     *[p.relative_to(EXAMPLE).as_posix() for p in EXAMPLE.glob("provenance/stages/*.json")],
                     *[p.relative_to(EXAMPLE).as_posix() for p in EXAMPLE.glob("explore/trajectory_00000[01].*")]):
        (target / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(EXAMPLE / relative, target / relative)
    state = json.loads((EXAMPLE / "run_state.json").read_text())
    state["source_sha256"] = current_source_sha256()
    (target / "run_state.json").write_text(json.dumps(state))
    return target


def identity(dataset="ogbn-arxiv", name="fixture", synthetic=True):
    return {"name": name, "dataset": dataset, "input_dataset": "synthetic", "profile": "smoke",
            "scientific_run": False, "synthetic": synthetic, "config_sha256": "a" * 64, "source_sha256": "b" * 64,
            "sealed_stages": [], "stage_manifest_sha256": {}, "training_seeds": [1, 2], "depth": 6,
            "split_index": None}


def sealed_run(root, experiment, raw, *, plan=None, config=None, parent=None, name="run"):
    """A sealed run directory in the runner's layout, with results recomputed from its raw files."""
    d = root / name
    (d / "raw").mkdir(parents=True)
    parent = parent or identity()
    write_json(d / "plan.json", {"schema": PLAN_SCHEMA, "experiment": experiment,
                                 "record_set": predictability.EXPERIMENTS[experiment]["record_set"],
                                 "parent": dict(parent, path="/abs/parent"), **(plan or {})})
    write_json(d / "experiment.json", {"schema": RUN_SCHEMA, "experiment": experiment, "parent": parent,
                                       "record_set": predictability.EXPERIMENTS[experiment]["record_set"],
                                       "plan_sha256": sha256(d / "plan.json"), "runtime_source_sha256": "c" * 64,
                                       "experiment_code": {}})
    write_json(d / "base_config.json", config or json.loads((ROOT / "configs/arxiv_smoke.json").read_text()))
    for relative, value in raw.items():
        path = d / "raw" / relative
        if isinstance(value, dict) and relative.endswith(".npz"):
            np.savez_compressed(path, **value)
        elif isinstance(value, (dict, list)) and relative.endswith(".json"):
            write_json(path, value)
        else:
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(value[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(value)
    summarize = {"predictability": predictability.summarize_predictability,
                 "search_enumeration": search_enumeration.summarize_search_enumeration,
                 "band_audit": band_audit.summarize_band_audit,
                 "negative_controls": negative_controls.summarize_negative_controls,
                 "input_gate": input_gate.summarize_input_gate}[experiment]
    write_json(d / "results.json", {"schema": RESULTS_SCHEMA, "experiment": experiment,
                                    "record_set": predictability.EXPERIMENTS[experiment]["record_set"],
                                    **summarize([d]), "truncated": False})
    seal(d)
    return d


def summary_values(path):
    with (path / "summary.csv").open(encoding="utf-8", newline="") as handle:
        return {tuple(v for k, v in row.items() if k not in ("quantity", "value", "unit", "source_file", "locator"))
                + (row["quantity"],): float(row["value"]) for row in csv.DictReader(handle)}


def contract(name):
    return SPECS["sets"][name]


# ----------------------------------------------------------------------------- fixtures of each experiment
def e2_raw():
    rng = np.random.default_rng(3)
    n, classes = 40, 4
    y = rng.integers(0, classes, n)
    def probabilities(sharp):
        p = np.full((n, classes), (1 - sharp) / (classes - 1))
        p[np.arange(n), y] = sharp
        return p.astype(np.float32)
    raw = {"predictions.npz": {"observation_id": np.arange(n), "unit_id": np.arange(n) % 5,
                               "input_id": np.zeros(n, dtype=np.int64), "y_true": y,
                               "qC": probabilities(.4), "qM": probabilities(.6), "qpred": probabilities(.5)},
           "sequences.csv": [{"input_id": 0, "member_seed": s, "teacher_length": 6, "predicted_length": 6,
                              "reproduced": int(s == 1), "agreeing_depths": 6 if s == 1 else 3, "a9_to_a8": 0,
                              "support_qualified": 1, "decoder_multiply_adds": 10, "teacher_actions": "0 1 2 0 0 1",
                              "predicted_actions": "0 1 2 0 0 1" if s == 1 else "0 1 2 3 3 3"} for s in (1, 2)],
           "unit_losses.csv": [{"unit_id": 0, "weight": 8, "qC_loss": 1., "qM_loss": .5, "qpred_loss": .7}],
           "predictor.json": {"audit": {}}, "targets.json": {"records": []}}
    expected_gain = -math.log(.4) + math.log(.6)
    return raw, {"phi_seq": .5, "feedback_gain_nats": expected_gain,
                 "phi_gain": (-math.log(.4) + math.log(.5)) / expected_gain, "compared_sequences": 2.,
                 "evaluation_units": 5.}


def e9_raw():
    family = [((0, 0), 1.0, 1.10, 1.05), ((0, 1), 0.9, 1.30, 1.20), ((1, 0), 0.95, 0.80, 0.75)]
    first = search_enumeration.instance_values(family, (0, 1))
    shifted = [(s, h + .1, t, z) for s, h, t, z in family]
    second = search_enumeration.instance_values(shifted, (1, 0))
    rows = []
    for index, (values, beam) in enumerate(((first, "0 1"), (second, "1 0"))):
        rows.append({"input_id": index, "member_seed": 1, "depth": 0, "horizon": 2, "planner": "mpc_rh",
                     "status": "COMPLETE", "reason": "", "family_size": values["family_size"], "beam_sequence": beam,
                     "beam_calls": 7, **{k: repr(values[k]) for k in ("e_H", "e_H_zero_uncertainty", "zeta", "regret",
                                                                       "regret_zero_uncertainty", "j_hat_min",
                                                                       "j_true_min")},
                     "beam_optimal": values["beam_optimal"], "seconds": "0.1"})
    arrays = {"instance": np.repeat([0, 1], 3), "sequence": np.asarray([s for s, *_ in family] * 2),
              "j_hat": np.asarray([h for _, h, _, _ in family] + [h for _, h, _, _ in shifted]),
              "j_true": np.asarray([t for _, _, t, _ in family] * 2),
              "j_true_zero_uncertainty": np.asarray([z for *_, z in family] * 2)}
    return {"instances.csv": rows, "sequences_H2.npz": arrays}, (first, second)


def e18_raw():
    states = []
    for depth in range(4):
        for action in range(1, 8):
            states.append({"trajectory": 0, "member_seed": 1, "graph_id": "", "depth": depth, "baseline": "shared",
                           "action": action, "transport_displacement": repr(1. + depth), "symmetric_displacement": repr(2.),
                           "operator_parity_relative_error": "1e-08", "block_displacement": repr(.5 * action),
                           "retained_statistic": repr(4. * action), "window_bound": repr(10. + depth),
                           "approximation_error_share": "0.25", "degree_factor": "0.5",
                           "update_gain": repr(.5 * action / (1. + depth)), "state_energy": "3.0", "d_min": "2.0",
                           "logged_statistic": repr(4. * action), "same_retention": int(action in (1, 3)),
                           "spectral_displacement": "2.0", "ideal_bound": repr(2. / (.4 + .1 * depth)),
                           "tightness_ratio": repr(.4 + .1 * depth), "omitted_band_share": "0.2",
                           "spectral_parity_relative_error": "1e-09"})
    densities = [{"baseline": "shared", "action": a, "lambda0": lam, "tightness_ratio": repr(.3 + .1 * a)}
                 for a in range(1, 8) for lam in (.1, 1.)]
    constants = {"window_approximation_errors": [.5] * 6, "shared_weight_spectral_norm": 1.7,
                 "activation_lipschitz": 1.0, "native_update_constant_by_depth": {"0": .9, "1": 1.2}}
    return {"states.csv": states, "rankings.csv": [{"trajectory": 0, "depth": d, "baseline": "shared",
                                                    "spearman": repr(.25 * d)} for d in range(4)],
            "model_densities.csv": densities, "replay.csv": [{"trajectory": 0, "depth": 0, "max_abs_deviation": "0.0"}],
            "constants.json": constants, "operators.json": {"graphs": {}}}


def nc_raw():
    rows = []
    for control_index, control in enumerate(negative_controls.CONTROLS):
        for unit in range(4):
            g = int(unit < 1 + control_index % 2)
            rows.append({"dataset": "ogbn-arxiv", "seed": 0, "unit_id": f"core:{unit}", "condition": control,
                         "realization": 0, "input_id": 0, "comparator_loss": "1.0", "deployed_loss": "1.0",
                         "controller_loss": repr(1. + .01 * unit), "fallback": 1 - g, "g_input": g,
                         "gate_score": "1.0", "active": g, "nonnominal_actions": 2, "harm": repr(.01 * unit)})
    return {"units.csv": rows, "constructions.json": {"inputs": []}, "gate.json": {"schema": "fixture"}}


def ig_raw(admitted=True):
    units = []
    for unit in range(5):
        g = int(unit != 0)
        units.append({"dataset": "ogbn-arxiv", "seed": "1+2", "unit_id": f"core:{unit}", "input_id": 0, "g_input": g,
                      "served_controller": g,
                      "gate_score": "1.0", "comparator_work": "10.0", "controller_work": "12.0", "gate_work": "1.0",
                      "actual_work": "13.0", "fallback_predictions_equal": "" if g else 1,
                      "comparator_loss": "1.0", "controller_loss": repr(1. + (.1 if unit == 4 else -.1)),
                      "deployed_loss": "1.0", "harm": repr(.1 if unit == 4 else -.1), "controller_failure": "",
                      "gate_seconds": "0.01", "controller_seconds": "0.11", "comparator_seconds": "0.10",
                      "deployed_seconds": repr(.12 if g else .11), "peak_gpu_bytes": "",
                      "decision_sites": 6 if g else 0, "recovery_sites": 1 if g else 0})
    raw = {"deployment_units.csv": units, "development_units.csv": [{"unit_id": f"core:{u}", "score": "1.0",
                                                                     "g_input": int(u != 9)} for u in range(10)],
           "memory.json": {"host_peak_bytes": 2**31, "device": "cpu"}, "gate.json": {"schema": "fixture"},
           "parity.json": {"scope": "fixture"},
           "timing.csv": [{"input_id": 0, "repeat": 0, "warmup": 1, "branch": "gate", "seconds": "0.01"}]}
    return raw, {"admission": {"passed": admitted}}


# ============================================================================ NumPy-only tests
class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.parent = fixture_parent(self.root)
        # The complete artifact chain needs the checkpoints; planning logic is tested here.
        patcher = mock.patch.object(ParentRun, "verify", return_value=[])
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def test_predictability_plan_declares_teacher_roles_and_parent_identity(self):
        with self.assertRaisesRegex(ExperimentError, "allow-fixture"):
            predictability.plan_predictability(self.parent, self.root / "e2-refused")
        path = predictability.plan_predictability(self.parent, self.root / "e2", allow_fixture=True, epochs=2)
        plan = json.loads(path.read_text())
        self.assertEqual(plan["teacher"]["route"], "mpc_rh")
        self.assertTrue(plan["fixture_teacher_override"])
        self.assertEqual(plan["teacher"]["ensemble"], plan["comparator"]["ensemble"])
        self.assertEqual((plan["fit_role"], plan["evaluation_role"]), ("explore", "compare"))
        self.assertEqual(plan["parent"]["config_sha256"], canonical(json.loads((self.parent / "config.json").read_text())))
        self.assertEqual(plan["parent"]["sealed_stages"][-1], "evaluate")
        self.assertIn("transductive", plan["input_scope"])
        loaded, parent = open_plan(path, "predictability")
        self.assertEqual(loaded, plan)
        with self.assertRaises(ExperimentError):
            open_plan(path, "band_audit")

    def test_a_changed_parent_manifest_rejects_the_plan(self):
        path = predictability.plan_predictability(self.parent, self.root / "e2", allow_fixture=True)
        manifest = self.parent / "provenance/stages/05_confirm.json"
        manifest.write_text(manifest.read_text() + " ")
        with self.assertRaisesRegex(ExperimentError, "changed since planning"):
            open_plan(path, "predictability")

    def test_another_runtime_source_is_refused(self):
        state = json.loads((self.parent / "run_state.json").read_text())
        state["source_sha256"] = "0" * 64
        (self.parent / "run_state.json").write_text(json.dumps(state))
        with self.assertRaisesRegex(ExperimentError, "another runtime source"):
            band_audit.plan_band_audit(self.parent, self.root / "e18", allow_fixture=True)

    def test_outputs_must_be_new_and_outside_the_parent(self):
        with self.assertRaises(ValueError):
            band_audit.plan_band_audit(self.parent, self.parent / "inside", allow_fixture=True)
        (self.root / "exists").mkdir()
        with self.assertRaises(FileExistsError):
            band_audit.plan_band_audit(self.parent, self.root / "exists", allow_fixture=True)

    def test_search_plan_registers_horizons_depths_role_and_prefix(self):
        with self.assertRaisesRegex(ExperimentError, "H <= 4"):
            search_enumeration.plan_search_enumeration(self.parent, self.root / "bad", horizons=(2, 5),
                                                       allow_fixture=True)
        plan = json.loads(search_enumeration.plan_search_enumeration(
            self.parent, self.root / "e9", horizons=(1, 2), allow_fixture=True).read_text())
        self.assertEqual(plan["label_role"], "explore")  # single transductive graph
        self.assertEqual(plan["depths"], [0, 1, 3, 4])   # L = 6: quartiles leaving room for H = 2
        self.assertEqual(plan["prefix_policy"]["route"], "mpc_rh")
        self.assertEqual(plan["prefix_policy"]["ensemble"], 1)
        self.assertEqual(plan["members"], [1])
        ol = json.loads(search_enumeration.plan_search_enumeration(
            self.parent, self.root / "e9-ol", planner="mpc_ol", allow_fixture=True).read_text())
        self.assertEqual((ol["depths"], ol["prefix_policy"], ol["open_loop"]), ([0], None, True))

    def test_band_audit_plan_seals_trajectory_identities(self):
        plan = json.loads(band_audit.plan_band_audit(self.parent, self.root / "e18", max_trajectories=1,
                                                     allow_fixture=True).read_text())
        self.assertEqual(plan["trajectories"], [0])
        self.assertEqual(plan["trajectory_sha256"]["0"], sha256(self.parent / "explore/trajectory_000000.npz"))
        with self.assertRaises(ExperimentError):
            band_audit.plan_band_audit(self.parent, self.root / "e18-bad", trajectories=[7], allow_fixture=True)

    def test_cli_plans_and_public_plans_carry_no_locations(self):
        from scripts.run_experiment_b import main
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = main(["search-enumeration", "plan", "--parent", str(self.parent), "--output", str(self.root / "cli"),
                         "--horizons", "1", "3", "--planner", "mpc_ol", "--allow-fixture"])
        self.assertEqual(code, 0)
        plan = json.loads(Path(json.loads(out.getvalue())["plan"]).read_text())
        self.assertEqual((plan["horizons"], plan["depths"]), ([1, 3], [0]))
        public = predictability._public_plan({"parent": {"path": "/proj/a/run"},
                                              "gate": {"reused": {"path": "/proj/b/raw/gate.json"}}})
        self.assertEqual(public, {"parent": {"path": "run"}, "gate": {"reused": {"path": "gate.json"}}})
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(main(["band-audit", "plan", "--parent", str(self.parent), "--output",
                                   str(self.root / "cli-bad"), "--lambda0", "-1", "--allow-fixture"]), 2)
        self.assertIn("density scales", err.getvalue())

    def test_gate_and_control_plans_read_the_admitted_identities(self):
        gate = json.loads(input_gate.plan_input_gate(self.parent, self.root / "gate", allow_fixture=True).read_text())
        self.assertEqual(gate["admission"]["passed"], False)
        self.assertEqual(gate["confirmed"]["candidate_id"], gate["comparator"]["candidate_id"])
        with self.assertRaisesRegex(ExperimentError, "E17"):
            input_gate.plan_input_gate(self.parent, self.root / "gate-bad", repetitions=1, allow_fixture=True)
        controls = json.loads(negative_controls.plan_negative_controls(
            self.parent, self.root / "nc", allow_fixture=True).read_text())
        self.assertEqual(controls["controls"], list(negative_controls.CONTROLS))
        self.assertEqual(controls["seed"], json.loads((self.parent / "config.json").read_text())["seed"])
        with self.assertRaises(ExperimentError):
            negative_controls.plan_negative_controls(self.parent, self.root / "nc-bad", controls=["noise"],
                                                     allow_fixture=True)


class ConstructionAndArithmeticTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(11)
        n = 60
        a = self.rng.integers(0, n, 400)
        b = self.rng.integers(0, n, 400)
        edges = np.stack([np.r_[a, b], np.r_[b, a]])
        self.pairs, _ = negative_controls.undirected_pairs(edges)
        self.n = n

    def test_undirected_pairs_are_unique_ordered_and_keep_weights(self):
        edges = np.array([[0, 1, 1, 2, 0, 1], [1, 0, 2, 1, 1, 0]])
        pairs, weights = negative_controls.undirected_pairs(edges, np.arange(6.))
        np.testing.assert_array_equal(pairs, [[0, 1], [1, 2]])
        np.testing.assert_array_equal(weights, [0., 2.])

    def test_configuration_model_preserves_degrees_up_to_dropped_stubs(self):
        rewired = negative_controls.configuration_rewire(self.pairs, self.n, self.rng)
        before = np.bincount(self.pairs.ravel(), minlength=self.n)
        after = np.bincount(rewired.ravel(), minlength=self.n)
        self.assertTrue(np.all(after <= before))
        self.assertGreater(after.sum(), .85 * before.sum())
        self.assertTrue(np.all(rewired[:, 0] < rewired[:, 1]))

    def test_anti_alignment_removes_same_class_labelled_pairs(self):
        labels = np.where(np.arange(self.n) % 5 == 0, -1, np.arange(self.n) % 3)
        result, record = negative_controls.anti_align(self.pairs, labels, self.rng)
        same = (labels[result[:, 0]] >= 0) & (labels[result[:, 0]] == labels[result[:, 1]])
        self.assertEqual(int(same.sum()), 0)
        self.assertGreater(record["same_class_pairs_before"], 0)

    def test_inflation_reaches_the_declared_pair_count(self):
        result, record = negative_controls.inflate(self.pairs, self.n, 2., self.rng)
        self.assertEqual(len(result), round(2 * len(self.pairs)))
        self.assertEqual(len({tuple(p) for p in result.tolist()}), len(result))

    def test_constructed_weights_keep_existing_pairs_and_draw_new_ones(self):
        weights = self.rng.uniform(.5, 1.5, len(self.pairs))
        new, new_weights, record = negative_controls.construct("support shift", self.pairs, weights, self.n, self.rng)
        lookup = {tuple(p): w for p, w in zip(self.pairs.tolist(), weights)}
        self.assertTrue(all(lookup[tuple(p)] == w for p, w in zip(new.tolist(), new_weights) if tuple(p) in lookup))
        self.assertTrue(set(new_weights) <= set(weights))
        self.assertEqual(record["pairs"], len(new))

    def test_gate_is_a_robust_envelope_frozen_on_development_units(self):
        rng = np.random.default_rng(5)
        vectors = rng.normal(size=(200, 6))
        vectors[:, 5] = 0.  # a coordinate constant on development units
        gate = input_gate.fit_gate(vectors, [f"v{i}" for i in range(6)], alpha=.05, unit_ids=range(200))
        decisions = [input_gate.gate_decision(gate, v)[1] for v in vectors]
        self.assertAlmostEqual(1 - np.mean(decisions), .05, delta=.01)
        self.assertEqual(input_gate.gate_decision(gate, np.r_[np.zeros(5), 1.])[1], 0)
        self.assertEqual(input_gate.gate_decision(gate, np.zeros(6))[1], 1)
        self.assertEqual(input_gate.gate_decision(gate, np.r_[np.zeros(5), np.nan])[1], 0)
        with self.assertRaises(ExperimentError):
            input_gate.gate_decision(gate, np.zeros(5))

    def test_search_instance_values_follow_proposition_8(self):
        family = [((0,), 1., 2., 2.), ((1,), .5, 2.5, 2.4), ((2,), .7, 1.9, 1.8)]
        values = search_enumeration.instance_values(family, (2,))
        self.assertAlmostEqual(values["zeta"], .2)
        self.assertAlmostEqual(values["e_H"], 2.)
        self.assertAlmostEqual(values["regret"], 0.)
        self.assertLessEqual(values["regret"], 2 * values["e_H"] + values["zeta"])
        self.assertEqual(values["beam_optimal"], 0)
        with self.assertRaises(ExperimentError):
            search_enumeration.instance_values(family, (3,))
        self.assertIsNone(search_enumeration.instance_values(family, None)["zeta"])

    def test_window_errors_cannot_beat_one_half_at_a_band_jump(self):
        coefficients = np.random.default_rng(2).normal(size=(6, 9))
        errors, low, high = band_audit.window_constants(coefficients, cells=6000)
        self.assertTrue(all(e >= .5 - 1e-3 for e in errors))
        self.assertLessEqual(low, high)

    def test_activation_constants_and_rank_correlation(self):
        self.assertEqual(band_audit.activation_lipschitz("relu"), 1.)
        self.assertAlmostEqual(band_audit.activation_lipschitz("silu"), 1.0998, places=3)
        self.assertAlmostEqual(band_audit.activation_lipschitz("gelu"), 1.1289, places=3)
        self.assertAlmostEqual(band_audit.spearman([1, 2, 3], [10, 20, 30]), 1.)
        self.assertAlmostEqual(band_audit.spearman([1, 2, 3], [3, 2, 1]), -1.)
        self.assertIsNone(band_audit.spearman([1, 1, 1], [1, 2, 3]))


class SummaryAndExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def export(self, module, runs, name):
        records = self.root / f"records-{name}"
        result = module.export_records(runs, records_root=records, hardware="CPU fixture", period="2026-11-07",
                                       argv=["fixture"])
        record_set = predictability.EXPERIMENTS[module.__name__.rsplit(".", 1)[1]]["record_set"]
        entry = verify_set(records / record_set, contract(record_set), SPECS)
        self.assertTrue(entry["verified"])
        self.assertTrue(entry["synthetic"])
        return records / record_set, result

    def test_predictability_values_follow_from_the_raw_predictions(self):
        raw, expected = e2_raw()
        run = sealed_run(self.root, "predictability", raw, plan={"input_scope": "fixture"})
        path, result = self.export(predictability, [run], "e2")
        values = summary_values(path)
        for quantity, value in expected.items():
            self.assertAlmostEqual(values[("ogbn-arxiv", quantity)], value, places=6)
        self.assertTrue(result["indexed"])
        self.assertEqual(json.loads((path.parent / "index.json").read_text())["sets"][0]["set"], "predictability")
        self.assertEqual(json.loads((path / "provenance.json").read_text())["record_contract"], "records/specs.json")
        self.assertTrue((path / "raw/ogbn-arxiv-fixture/sequences.csv").is_file())
        self.assertEqual(json.loads((path / "code/ogbn-arxiv-fixture/plan.json").read_text())["parent"]["path"], "parent")

    def test_a_nonpositive_feedback_gain_reports_no_retained_share(self):
        raw, _ = e2_raw()
        raw["predictions.npz"]["qM"] = raw["predictions.npz"]["qC"]
        run = sealed_run(self.root, "predictability", raw, plan={"input_scope": "fixture"})
        summary = predictability.summarize_predictability([run])
        self.assertNotIn("phi_gain", {r["quantity"] for r in summary["rows"]})
        self.assertIn("FEEDBACK_GAIN_NOT_POSITIVE", summary["notes"]["ogbn-arxiv"]["status"])

    def test_search_enumeration_quantiles_and_array_replay(self):
        raw, (first, second) = e9_raw()
        run = sealed_run(self.root, "search_enumeration", raw, plan={"confidence": .95})
        path, _ = self.export(search_enumeration, [run], "e9")
        values = summary_values(path)
        key = ("ogbn-arxiv", "mpc_rh", "2")
        self.assertAlmostEqual(values[key + ("two_e_H_nats",)], 2 * max(first["e_H"], second["e_H"]))
        self.assertAlmostEqual(values[key + ("zeta_H_nats",)], max(first["zeta"], second["zeta"]))
        self.assertEqual(values[key + ("planning_instances",)], 2)
        self.assertEqual(values[key + ("bound_coverage",)], 1.)

    def test_search_rows_that_disagree_with_their_arrays_are_refused(self):
        raw, _ = e9_raw()
        raw["instances.csv"][0]["zeta"] = "0.5"
        with self.assertRaisesRegex(ExperimentError, "objective arrays"):
            sealed_run(self.root, "search_enumeration", raw, plan={"confidence": .95})

    def test_band_audit_medians_and_falsification_flag(self):
        run = sealed_run(self.root, "band_audit", e18_raw())
        path, _ = self.export(band_audit, [run], "e18")
        values = summary_values(path)
        self.assertAlmostEqual(values[("ogbn-arxiv", "shared", "A3", "explore_states", "tightness_ratio")], .55)
        self.assertAlmostEqual(values[("ogbn-arxiv", "shared", "A3", "explore_states", "tightness_ratio_max")], .7)
        self.assertAlmostEqual(values[("ogbn-arxiv", "shared", "A2", "lambda0=0.1", "tightness_ratio")], .5)
        self.assertEqual(values[("ogbn-arxiv", "shared", "A1", "explore_states", "audited_states")], 4)
        self.assertEqual(values[("ogbn-arxiv", "shared", "A1", "explore_states", "logged_statistic_ratio")], 1.)
        self.assertEqual(values[("ogbn-arxiv", "all", "all", "explore_states", "update_lipschitz_max")], 1.2)
        self.assertAlmostEqual(values[("ogbn-arxiv", "all", "all", "explore_states", "ranking_spearman")], .375)
        self.assertFalse(band_audit.summarize_band_audit([run])["notes"]["ogbn-arxiv"]["falsified"])

    def test_negative_controls_cover_the_four_registered_constructions(self):
        from scripts.scientific_evidence_audit import validate_record_coverage
        from cm_campaigns.contract import RECORD_SETS
        run = sealed_run(self.root, "negative_controls", nc_raw())
        path, result = self.export(negative_controls, [run], "nc")
        self.assertTrue(result["indexed"])
        self.assertEqual(json.loads((path.parent / "index.json").read_text())["sets"][0]["set"], "negative_controls")
        values = summary_values(path)
        self.assertEqual(values[("graph irrelevant", "input_fallback")], .75)
        self.assertEqual(values[("misleading graph", "residual_activity")], .5)
        self.assertAlmostEqual(values[("support shift", "worst_pre_fallback_harm")], .03)
        with (path / "summary.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        validate_record_coverage("negative_controls", rows, contract("negative_controls"),
                                 RECORD_SETS["negative_controls"])

    def test_input_gate_contract_values_and_table_s37_rates(self):
        raw, plan = ig_raw()
        run = sealed_run(self.root, "input_gate", raw, plan=plan)
        path, result = self.export(input_gate, [run], "gate")
        values = summary_values(path)
        self.assertAlmostEqual(values[("ogbn-arxiv", "fallback_probability")], .2)
        self.assertAlmostEqual(values[("ogbn-arxiv", "deployed_wall_overhead_percent")], 100 * (.59 / .5 - 1))
        self.assertAlmostEqual(values[("ogbn-arxiv", "cpu_memory_gib")], 2.)
        self.assertNotIn(("ogbn-arxiv", "gpu_memory_gib"), values)
        self.assertEqual(set(result["not_exported"]), {"false_safe_rate", "harm_miss_rate", "development_fallback",
                                                       "expected_work_overhead_percent", "stepwise_support_rejection"})
        outside = {r["quantity"]: r["value"] for r in
                   json.loads((path / "raw/summary.json").read_text())["measurements_outside_contract"]}
        self.assertAlmostEqual(outside["false_safe_rate"], .25)
        self.assertAlmostEqual(outside["harm_miss_rate"], 1.)

    def test_global_return_reports_no_gate_fraction(self):
        raw, plan = ig_raw(admitted=False)
        run = sealed_run(self.root, "input_gate", raw, plan=plan)
        quantities = {r["quantity"] for r in input_gate.summarize_input_gate([run])["rows"]}
        self.assertNotIn("fallback_probability", quantities)
        self.assertNotIn("deployed_wall_overhead_percent", quantities)
        self.assertIn("cpu_memory_gib", quantities)

    def test_a_sealed_summary_that_differs_from_its_raw_files_is_refused(self):
        run = sealed_run(self.root, "negative_controls", nc_raw())
        results = json.loads((run / "results.json").read_text())
        results["rows"][0]["value"] += .1
        (run / "results.json").write_text(json.dumps(results))
        (run / "ARTIFACT_MANIFEST.json").unlink()
        seal(run)
        with self.assertRaisesRegex(ExperimentError, "recomputed"):
            negative_controls.export_records([run], records_root=self.root / "r", hardware="x", period="y")

    def test_an_unsealed_change_is_refused(self):
        run = sealed_run(self.root, "negative_controls", nc_raw())
        with (run / "raw/units.csv").open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            negative_controls.export_records([run], records_root=self.root / "r", hardware="x", period="y")

    def test_official_records_refuse_fixtures_and_truncated_runs(self):
        official = self.root / "official"
        official.mkdir()
        shutil.copyfile(ROOT / "records/specs.json", official / "specs.json")
        fixture = sealed_run(self.root, "negative_controls", nc_raw(), name="synthetic")
        measured = sealed_run(self.root, "predictability", e2_raw()[0], plan={"input_scope": "fixture"},
                              parent=identity(synthetic=False), name="measured")
        truncated = sealed_run(self.root, "predictability", e2_raw()[0], plan={"input_scope": "fixture"},
                               parent=identity(synthetic=False), name="truncated")
        results = json.loads((truncated / "results.json").read_text())
        (truncated / "results.json").write_text(json.dumps(dict(results, truncated=True)))
        (truncated / "ARTIFACT_MANIFEST.json").unlink()
        seal(truncated)
        with mock.patch("scripts.records.RECORDS", official):
            with self.assertRaisesRegex(ExperimentError, "synthetic"):
                negative_controls.export_records([fixture], records_root=official, hardware="x", period="y")
            with self.assertRaisesRegex(ExperimentError, "truncated"):
                predictability.export_records([truncated], records_root=official, hardware="x", period="y")
            self.assertEqual(sorted(p.name for p in official.iterdir()), ["specs.json"])
            result = predictability.export_records([measured], records_root=official, hardware="x", period="y")
        self.assertTrue(result["indexed"])
        self.assertEqual(sorted(p.name for p in official.iterdir()), ["index.json", "predictability", "specs.json"])
        self.assertFalse(json.loads((official / "predictability/provenance.json").read_text())["synthetic"])

    def test_repeated_splits_of_one_dataset_are_pooled(self):
        raw, _ = e2_raw()
        first = sealed_run(self.root, "predictability", raw, plan={"input_scope": "fixture"}, name="split0",
                           parent=identity(name="split_00"))
        raw2 = copy.deepcopy(raw)
        raw2["predictions.npz"]["qpred"] = raw2["predictions.npz"]["qM"]
        second = sealed_run(self.root, "predictability", raw2, plan={"input_scope": "fixture"}, name="split1",
                            parent=identity(name="split_01"))
        path, _ = self.export(predictability, [first, second], "pooled")
        values = summary_values(path)
        self.assertAlmostEqual(values[("ogbn-arxiv", "compared_sequences")], 4)
        gain = -math.log(.4) + math.log(.6)
        retained = ((-math.log(.4) + math.log(.5)) + gain) / 2
        self.assertAlmostEqual(values[("ogbn-arxiv", "phi_gain")], retained / gain, places=6)

    def test_contracts_are_read_from_specs_json_and_match_the_summaries(self):
        required = {"title", "manuscript_tables", "experiments", "keys", "quantities", "table"}
        for name in ("predictability", "search_enumeration", "band_power_audit"):
            spec = contract(name)
            self.assertLessEqual(required, set(spec))
            self.assertEqual(spec["table"], "list")
            self.assertTrue(all(v.rstrip().endswith(")") for v in spec["quantities"].values()), name)
        definitions = {"predictability": predictability.E2_DEFINITIONS,
                       "search_enumeration": search_enumeration.DEFINITIONS,
                       "band_power_audit": band_audit.DEFINITIONS,
                       "negative_controls": negative_controls.DEFINITIONS, "input_gate": input_gate.DEFINITIONS}
        for name, defined in definitions.items():
            self.assertLessEqual(set(contract(name)["quantities"]), set(defined), name)
        with self.assertRaisesRegex(ExperimentError, "not a record set of records/specs.json"):
            predictability.load_contract("no_such_set")

    def test_cli_exports_and_reports_errors(self):
        from scripts.run_experiment_b import main
        run = sealed_run(self.root, "negative_controls", nc_raw())
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = main(["negative-controls", "export", "--results", str(run), "--root", str(self.root / "cli"),
                         "--hardware", "CPU fixture", "--period", "2026-11-07"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.getvalue())["set"], "negative_controls")
        with contextlib.redirect_stderr(io.StringIO()) as err:
            code = main(["negative-controls", "export", "--results", str(run), "--root", str(self.root / "cli"),
                         "--hardware", "CPU fixture", "--period", "2026-11-07"])
        self.assertEqual(code, 2)
        self.assertIn("--replace", err.getvalue())


# ============================================================================ PyTorch execution tests (CI)
def tree_digest(path):
    digest = hashlib.sha256()
    for item in sorted(p for p in Path(path).rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(item.read_bytes())
    return digest.hexdigest()


@unittest.skipUnless(TORCH, "PyTorch is required to execute the frozen smoke plants")
class SmokeExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from cross_morse.common import seed_everything
        from cross_morse.pipeline import Pipeline
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.runs = {}
        for name in ("arxiv", "pascalvoc_sp"):
            config = json.loads((ROOT / f"configs/{name}_smoke.json").read_text())
            seed_everything(config["seed"])
            pipeline = Pipeline(config, cls.root / "cache", cls.root / f"{name}_run", device="cpu")
            with contextlib.redirect_stdout(io.StringIO()):
                pipeline.run("all")
            cls.runs[name] = pipeline.out
        cls.digests = {name: tree_digest(path) for name, path in cls.runs.items()}

    @classmethod
    def tearDownClass(cls):
        import torch
        torch.set_num_threads(cls.threads)
        cls.tmp.cleanup()

    def tearDown(self):
        for name, path in self.runs.items():
            self.assertEqual(tree_digest(path), self.digests[name], f"the {name} parent run was written")

    def execute(self, module, key, plan, name):
        with contextlib.redirect_stdout(io.StringIO()):
            out = getattr(module, f"run_{key}")(plan, device="cpu", trust=True)
        records = self.root / f"records-{name}"
        result = module.export_records([out], records_root=records, hardware="CPU smoke", period="2026-11-07")
        record_set = predictability.EXPERIMENTS[key]["record_set"]
        self.assertTrue(verify_set(records / record_set, contract(record_set), SPECS)["verified"])
        return out, json.loads((out / "results.json").read_text()), result

    def test_untrusted_checkpoints_are_not_loaded(self):
        plan = band_audit.plan_band_audit(self.runs["arxiv"], self.root / "untrusted", allow_fixture=True)
        with self.assertRaisesRegex(ExperimentError, "trust"):
            band_audit.run_band_audit(plan, device="cpu", trust=False)

    def test_predictability_on_independent_graphs(self):
        plan = predictability.plan_predictability(self.runs["pascalvoc_sp"], self.root / "e2", allow_fixture=True,
                                                  epochs=2, patience=2)
        out, results, _ = self.execute(predictability, "predictability", plan, "e2")
        rows = {r["quantity"]: r["value"] for r in results["rows"]}
        sequences = predictability.read_csv(out / "raw/sequences.csv")
        plan_value = json.loads(plan.read_text())
        self.assertEqual(rows["compared_sequences"], len(sequences))
        self.assertEqual(len(sequences) % len(plan_value["ensemble_members"]), 0)
        self.assertTrue(0 <= rows["phi_seq"] <= 1)
        self.assertNotIn("9", " ".join(r["predicted_actions"] for r in sequences).split())
        self.assertTrue(json.loads((out / "raw/predictor.json").read_text())["locked_before_evaluation_label_access"])

    def test_search_enumeration_bounds_hold_per_instance(self):
        plan = search_enumeration.plan_search_enumeration(self.runs["pascalvoc_sp"], self.root / "e9",
                                                          horizons=(1, 2), depths=(0, 2), max_graphs=2,
                                                          allow_fixture=True)
        out, results, _ = self.execute(search_enumeration, "search_enumeration", plan, "e9")
        instances = predictability.read_csv(out / "raw/instances.csv")
        complete = [r for r in instances if r["status"] == "COMPLETE"]
        self.assertTrue(complete)
        for row in complete:
            self.assertGreaterEqual(float(row["zeta"]), 0.)
            self.assertLessEqual(float(row["regret"]), 2 * float(row["e_H"]) + float(row["zeta"]) + 1e-9)
            self.assertEqual(len(row["beam_sequence"].split()), int(row["horizon"]))
        self.assertTrue(any(r["quantity"] == "two_e_H_nats" for r in results["rows"]))

    def test_band_audit_replays_the_logged_statistic_and_never_falsifies(self):
        for name in ("arxiv", "pascalvoc_sp"):
            plan = band_audit.plan_band_audit(self.runs[name], self.root / f"e18-{name}", max_trajectories=2,
                                              allow_fixture=True)
            out, results, _ = self.execute(band_audit, "band_audit", plan, f"e18-{name}")
            states = predictability.read_csv(out / "raw/states.csv")
            replay = predictability.read_csv(out / "raw/replay.csv")
            self.assertLess(max(float(r["max_abs_deviation"]) for r in replay), 1e-4)
            exact = [float(r["tightness_ratio"]) for r in states if r["tightness_ratio"] != ""]
            self.assertTrue(exact)
            self.assertLessEqual(max(exact), 1 + 1e-6)
            self.assertLess(max(float(r["operator_parity_relative_error"]) for r in states), 1e-4)
            ratios = [float(r["retained_statistic"]) / float(r["logged_statistic"]) for r in states
                      if r["logged_statistic"] != "" and float(r["logged_statistic"]) > 1e-6]
            self.assertTrue(np.allclose(ratios, 1., rtol=1e-3))

    def test_input_gate_serves_comparator_bytes_on_fallback(self):
        for name in ("arxiv", "pascalvoc_sp"):
            plan = input_gate.plan_input_gate(self.runs[name], self.root / f"gate-{name}", warmup=1, repetitions=3,
                                              allow_fixture=True)
            with contextlib.redirect_stdout(io.StringIO()):
                out = input_gate.run_input_gate(plan, device="cpu", trust=True)
            units = predictability.read_csv(out / "raw/deployment_units.csv")
            self.assertTrue(units)
            self.assertTrue(all(u["fallback_predictions_equal"] == "1" for u in units if u["served_controller"] == "0"))
            self.assertTrue(all(u["served_controller"] <= u["g_input"] for u in units))
            gate = json.loads((out / "raw/gate.json").read_text())
            self.assertTrue(gate["frozen_before_evaluation_labels"])
            self.assertLess(json.loads((out / "raw/parity.json").read_text())["qC_max_abs_deviation"], 1e-5)
            results = json.loads((out / "results.json").read_text())
            if json.loads(plan.read_text())["admission"]["passed"]:
                records = self.root / f"records-gate-{name}"
                input_gate.export_records([out], records_root=records, hardware="CPU smoke", period="2026-11-07")
                self.assertTrue(verify_set(records / "input_gate", contract("input_gate"), SPECS)["verified"])
            else:
                self.assertIn("GLOBAL_RETURN", results["notes"][json.loads(plan.read_text())["parent"]["dataset"]]["status"])

    def test_negative_controls_execute_every_construction(self):
        plan = negative_controls.plan_negative_controls(self.runs["pascalvoc_sp"], self.root / "nc", max_graphs=4,
                                                        allow_fixture=True)
        out, results, _ = self.execute(negative_controls, "negative_controls", plan, "nc")
        quantities = {(r["keys"]["control"], r["quantity"]) for r in results["rows"]}
        for control in negative_controls.CONTROLS:
            self.assertIn((control, "input_fallback"), quantities)
            self.assertIn((control, "residual_activity"), quantities)
        constructions = json.loads((out / "raw/constructions.json").read_text())["inputs"]
        misleading = [c for c in constructions if c["control"] == "misleading graph"]
        self.assertTrue(all(c["same_class_pairs_after"] == 0 or c["unresolved"] for c in misleading))


if __name__ == "__main__":
    unittest.main()
