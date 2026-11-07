"""Runners of the record-set experiments: E13 attribution, E14 ensemble grid, cross-backbone
substitutions and the equal-budget factorials (cm_experiments/{attribution,ensemble_grid,
backbones,factorial}.py, scripts/run_experiment.py).

Planning, statistics and export run without PyTorch on the bundled synthetic example runs;
their record sets are written to temporary roots and traced by scripts.records.verify_set.
Execution (smoke lifecycles and every runner end to end) needs PyTorch and runs in CI.
"""
from __future__ import annotations

import contextlib
import copy
import csv
import importlib.util
import io
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from cm_experiments import attribution, backbones, ensemble_grid, factorial
from cm_experiments.ensemble_grid import RunnerError, SealedRun, job_directory, utc_now
from scripts.records import RECORDS, build_table, load_specs, read_summary, verify_set

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "runs"
TORCH = importlib.util.find_spec("torch") is not None
ENVIRONMENT = {"python": "3.x", "machine": "x86_64", "torch": None}


def specs():
    return load_specs(RECORDS, RECORDS / "specs.json")


def summary(root, name):
    with (Path(root) / name / "summary.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class Workspace(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cm-runners-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.records = self.root / "records"


# ============================================================================= contracts
class ContractTests(unittest.TestCase):
    def test_keys_match_the_record_contracts_and_registered_constants(self):
        from scripts.record_mapping import FACTORIAL_CELLS, INSERTED_SITE_ARMS
        self.assertEqual(list(attribution.ARMS), INSERTED_SITE_ARMS)
        self.assertEqual(list(factorial.CELLS), FACTORIAL_CELLS)
        self.assertEqual(ensemble_grid.registered_grid(), [1, 2, 4, 8, 12, 16, 20])
        rows = backbones.coverage_rows()
        self.assertEqual(len(rows), 6)
        self.assertEqual({r for r, _ in rows}, set(backbones.REGIME_DATASETS))
        self.assertEqual({m for _, m in rows}, set(backbones.MODEL_BACKBONES))
        sets = specs()["sets"]
        self.assertEqual(sets["inserted_site"]["keys"], ["dataset", "arm"])
        self.assertEqual(sets["ensemble_grid"]["keys"], ["dataset", "ensemble"])
        self.assertEqual(sets["cross_backbone"]["keys"], ["regime", "model"])
        self.assertEqual(set(sets["factorials"]["quantities"]),
                         set(factorial.QUANTITY_UNITS) | {"official_metric"})
        self.assertEqual(set(factorial.DESIGNS), {"ogbn-arxiv", "PascalVOC-SP"})

    def test_registered_full_configurations_give_the_printed_native_site_fractions(self):
        fractions = {}
        for stem in ("pascalvoc_sp_full", "coco_sp_full", "peptides_func_full"):
            config = json.loads((ROOT / "configs" / f"{stem}.json").read_text())
            layout = attribution.designated_sites(config)
            fractions[stem] = layout["native_site_fraction"]
            self.assertEqual(len(layout["native_sites"]) + layout["inserted_sites"], layout["control_depth"])
        self.assertEqual(attribution.designated_sites(json.loads((ROOT / "configs/pascalvoc_sp_full.json").read_text()))
                         ["native_sites"], [1, 22, 43, 64, 85, 106])
        self.assertAlmostEqual(fractions["pascalvoc_sp_full"], 6 / 128)
        self.assertAlmostEqual(fractions["coco_sp_full"], 6 / 128)
        self.assertAlmostEqual(fractions["peptides_func_full"], 8 / 64)
        with self.assertRaises(RunnerError):
            attribution.designated_sites(json.loads((ROOT / "configs/arxiv_full.json").read_text()))


# ============================================================================= E14
class EnsembleGridTests(Workspace):
    def plan(self, **kwargs):
        options = dict(grid=[1, 2, 4], role="test", reps=40)
        options.update(kwargs)
        path = ensemble_grid.plan_ensemble_grid([EXAMPLES / "arxiv"], self.root / "e14", **options)
        return json.loads(path.read_text())

    def test_plan_declares_distinct_checkpoints_in_one_seed_order(self):
        plan = self.plan()
        job = plan["jobs"][0]
        run = SealedRun(EXAMPLES / "arxiv")
        self.assertEqual(job["seed_order"], [1, 2, 3, 4])
        self.assertEqual(job["extra_seeds"], [3, 4])
        self.assertEqual(job["parent_seeds"], [1, 2])
        self.assertEqual(job["comparator"]["candidate_id"], run.comparator()["candidate_id"])
        self.assertEqual(job["identity"]["config_sha256"], run.identity()["config_sha256"])
        self.assertEqual(plan["datasets"][0]["synthetic"], True)
        self.assertEqual(job["bootstrap"], {"reps": 40, "seed": run.config["seed"] + 7})

    def test_plan_refuses_repeated_surplus_or_missing_checkpoints(self):
        for kwargs, message in ((dict(extra_seeds=[2, 3]), "distinct"), (dict(extra_seeds=[3, 4, 5]), "exactly"),
                                (dict(extra_seeds=[3]), "needs 4")):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(RunnerError, message):
                ensemble_grid.plan_ensemble_grid([EXAMPLES / "arxiv"], self.root / "x", grid=[1, 2, 4], **kwargs)
        with self.assertRaisesRegex(RunnerError, "decision roles"):
            ensemble_grid.plan_ensemble_grid([EXAMPLES / "arxiv"], self.root / "x", role="select")
        with self.assertRaisesRegex(RunnerError, "outside the repository"):
            ensemble_grid.plan_ensemble_grid([EXAMPLES / "arxiv"], ROOT / "e14-output", grid=[1, 2])
        self.plan()
        with self.assertRaisesRegex(RunnerError, "exists"):
            self.plan()

    def test_default_grid_is_the_registered_grid(self):
        plan = ensemble_grid.plan_ensemble_grid([EXAMPLES / "arxiv"], self.root / "default")
        plan = json.loads(plan.read_text())
        self.assertEqual(plan["grid"], [1, 2, 4, 8, 12, 16, 20])
        self.assertEqual(plan["jobs"][0]["extra_seeds"], list(range(3, 21)))

    def test_prefix_ensembles_and_paired_contrasts(self):
        from cross_morse.evaluation import score_observations
        config = SealedRun(EXAMPLES / "arxiv").config
        rng = np.random.default_rng(7)
        labels, units = rng.integers(0, 4, 30), np.repeat(np.arange(6), 5)
        members = [rng.dirichlet(np.ones(4), 30).astype(np.float32) for _ in range(4)]
        deployed = rng.dirichlet(np.ones(4), 30).astype(np.float32)
        rows, arrays = ensemble_grid.grid_statistics(iter(members), labels, units, config, [1, 2, 4], [5, 6, 7, 8],
                                                     parent_seeds=[5, 6], comparator_ensemble=2, deployed=deployed,
                                                     reps=50, seed=3)
        self.assertEqual([r["ensemble"] for r in rows], [1, 2, 4])
        self.assertEqual([r["new_checkpoints"] for r in rows], [0, 0, 2])
        self.assertEqual(rows[2]["seeds"], [5, 6, 7, 8])
        for row, size in zip(rows, (1, 2, 4)):
            expected = score_observations(ensemble_grid.ensemble_probabilities(members[:size]), labels, units, config)
            self.assertEqual(row["loss_nats"], expected["node_ce"])
            self.assertEqual((row["official_score"], row["official_score_unit"]), (100 * expected["metric"], "%"))
            np.testing.assert_array_equal(arrays[f"E{size}_unit_loss"], expected["loss"])
        self.assertEqual(rows[1]["minus_comparator_ensemble"]["mean"], 0.0)
        self.assertEqual(rows[1]["minus_comparator_ensemble"]["standard_error"], 0.0)
        deployed_loss = score_observations(deployed, labels, units, config)["loss"]
        counts = arrays["unit_weight"]
        expected = float(np.dot(counts, deployed_loss - arrays["E4_unit_loss"]) / counts.sum())
        self.assertAlmostEqual(rows[2]["deployed_minus_row"]["mean"], expected, places=12)
        with self.assertRaisesRegex(RunnerError, "every ensemble size"):
            ensemble_grid.grid_statistics(iter(members[:2]), labels, units, config, [1, 2, 4], [1, 2],
                                          parent_seeds=[1, 2], comparator_ensemble=2, reps=50, seed=3)

    def test_runtime_ensemble_accumulation(self):
        members = [np.array([[0.2, 0.8]], dtype=np.float32), np.array([[0.6, 0.4]], dtype=np.float32)]
        np.testing.assert_allclose(ensemble_grid.ensemble_probabilities(members), [[0.4, 0.6]], rtol=0, atol=1e-7)
        self.assertEqual(ensemble_grid.ensemble_probabilities(members).dtype, np.float32)

    def write_job(self, plan):
        entry = plan["jobs"][0]
        config = SealedRun(entry["run"]).config
        rng = np.random.default_rng(11)
        labels, units = rng.integers(0, 4, 40), np.repeat(np.arange(8), 5)
        members = [rng.dirichlet(np.ones(4), 40).astype(np.float32) for _ in entry["seed_order"]]
        rows, arrays = ensemble_grid.grid_statistics(iter(members), labels, units, config, plan["grid"],
                                                     entry["seed_order"], parent_seeds=entry["parent_seeds"],
                                                     comparator_ensemble=entry["comparator_ensemble"],
                                                     deployed=members[0], reps=40, seed=1)
        directory = job_directory(self.root / "e14", entry)
        directory.mkdir(parents=True)
        ensemble_grid.write_grid_result(directory, entry, role=plan["role"], rows=rows, arrays=arrays,
                                        checkpoints=[], members=[], reproduction=None, parent_files=[],
                                        started=utc_now(), environment=ENVIRONMENT)
        return rows

    def test_export_traces_every_value_and_refuses_incomplete_or_synthetic_deposits(self):
        plan = self.plan()
        with self.assertRaisesRegex(RunnerError, "has not completed"):
            ensemble_grid.export_ensemble_grid(self.root / "e14", root=self.records, allow_partial=True)
        rows = self.write_job(plan)
        with self.assertRaisesRegex(RunnerError, "coverage"):
            ensemble_grid.export_ensemble_grid(self.root / "e14", root=self.records)
        with self.assertRaisesRegex(RunnerError, "complete measured sets"):
            ensemble_grid.export_ensemble_grid(self.root / "e14", root=RECORDS, allow_partial=True)
        self.assertFalse((RECORDS / "ensemble_grid").exists())
        target = ensemble_grid.export_ensemble_grid(self.root / "e14", root=self.records, allow_partial=True,
                                                    command=["scripts/run_experiment.py", "ensemble-grid", "export"])
        spec = specs()["sets"]["ensemble_grid"]
        entry = verify_set(target, spec, specs())
        self.assertTrue(entry["synthetic"])
        lines = summary(self.records, "ensemble_grid")
        self.assertEqual([(r["dataset"], r["ensemble"], r["quantity"], r["unit"]) for r in lines],
                         [("ogbn-arxiv", str(e), "loss_nats", "nats") for e in (1, 2, 4)])
        for line, row in zip(lines, rows):
            self.assertAlmostEqual(float(line["value"]), row["loss_nats"], places=9)
        header, table, _ = build_table(target, spec, specs())
        self.assertEqual(header[:2], ["dataset", "ensemble sizes"])
        self.assertEqual(table[0][1], "1, 2, 4")
        provenance = json.loads((target / "provenance.json").read_text())
        self.assertFalse(provenance["coverage_complete"])
        self.assertEqual(provenance["grid"], [1, 2, 4])
        dataset = json.loads((target / "raw" / "ogbn-arxiv" / "ensemble_grid.json").read_text())
        self.assertEqual([e["official_score"] for e in dataset["ensembles"]], [r["official_score"] for r in rows])
        self.assertNotIn("/", json.loads((target / "code" / "plan.json").read_text())["jobs"][0]["run"])
        self.assertTrue((target / "code" / "cm_experiments" / "ensemble_grid.py").is_file())
        self.assertTrue((target / "code" / "plan.json").is_file())
        index = json.loads((self.records / "index.json").read_text())
        self.assertEqual([e["set"] for e in index["sets"]], ["ensemble_grid"])
        with self.assertRaisesRegex(Exception, "exists"):
            ensemble_grid.export_ensemble_grid(self.root / "e14", root=self.records, allow_partial=True)
        ensemble_grid.export_ensemble_grid(self.root / "e14", root=self.records, allow_partial=True, replace=True)


# ============================================================================= E13
class FakeController:
    def __init__(self):
        self.counters = {"approx_dense_multiply_adds": 0}

    def supported(self, y, depth):
        return [0, 1, 3, 8]

    def score(self, state, actions, **kwargs):
        return {a: {"score": float(a)} for a in actions}


class AttributionTests(Workspace):
    def plan(self, **kwargs):
        options = dict(role="test", reps=40, allow_fixture=True)
        options.update(kwargs)
        path = attribution.plan_attribution([EXAMPLES / "peptides_func"], self.root / "e13", **options)
        return json.loads(path.read_text())

    def test_plan_builds_four_arms_with_native_site_restriction(self):
        plan = self.plan()
        job = plan["jobs"][0]
        self.assertEqual(job["layout"]["native_sites"], [0, 4])
        self.assertEqual([a["arm"] for a in job["arms"]], list(attribution.ARMS))
        comparator, native = job["arms"][1]["candidate"], job["arms"][0]["candidate"]
        self.assertEqual(native["sequence"], [a if i in (0, 4) else 0 for i, a in enumerate(comparator["sequence"])])
        self.assertEqual(native["site_restriction"], attribution.RESTRICTION)
        self.assertNotEqual(native["hash"], comparator["hash"])
        self.assertEqual(job["arms"][2]["candidate"]["route"], job["arms"][3]["candidate"]["route"])
        self.assertTrue(job["feedback_record"]["fixture_feedback_override"])
        self.assertEqual(job["member_seeds"], [1, 2])

    def test_open_loop_confirmation_is_refused_without_the_fixture_switch(self):
        with self.assertRaisesRegex(RunnerError, "feedback route"):
            attribution.plan_attribution([EXAMPLES / "peptides_func"], self.root / "x", role="test")
        with self.assertRaisesRegex(RunnerError, "E13 covers"):
            attribution.plan_attribution([EXAMPLES / "arxiv"], self.root / "y", role="test", allow_fixture=True)

    def test_native_site_controller_admits_only_the_bypass_at_inserted_sites(self):
        wrapped = attribution.NativeSiteController(FakeController(), [0, 4], 8)
        self.assertEqual(wrapped.supported(None, 0), [0, 1, 3, 8])
        self.assertEqual(wrapped.supported(None, 1), [0])
        self.assertEqual(wrapped.supported(None, 4), [0, 1, 3, 8])
        self.assertEqual(wrapped.score(None, [0, 3])[3]["score"], 3.0)
        self.assertIs(wrapped.counters, wrapped._controller.counters)

    def results(self, plan, rng):
        entry = plan["jobs"][0]
        config = SealedRun(entry["run"]).config
        labels = rng.integers(0, 2, (12, 4)).astype(np.float64)
        probabilities = {arm: rng.uniform(0.05, 0.95, (12, 4)).astype(np.float32) for arm in attribution.ARMS}
        return entry, config, labels, probabilities

    def test_paired_effects_against_the_identity_arm(self):
        from cross_morse.tasks import sample_losses, task_spec
        entry, config, labels, probabilities = self.results(self.plan(), np.random.default_rng(5))
        rows, arrays = attribution.arm_results(probabilities, labels, np.arange(12), config, reps=40, seed=2)
        identity = rows[1]
        self.assertEqual((identity["effect_nats"], identity["standard_error"]), (0.0, 0.0))
        reference = sample_losses(probabilities[attribution.ARMS[1]], labels, task_spec(config))
        for row in (rows[0], rows[2], rows[3]):
            d = sample_losses(probabilities[row["arm"]], labels, task_spec(config)) - reference
            self.assertAlmostEqual(row["effect_nats"], float(d.mean()), places=12)
            self.assertAlmostEqual(row["standard_error"], float(d.std(ddof=1) / math.sqrt(len(d))), places=12)
            self.assertLessEqual(row["interval_low_nats"], row["interval_high_nats"])
        self.assertAlmostEqual(attribution.retained_share(rows), rows[2]["effect_nats"] / rows[3]["effect_nats"])

    def test_export_traces_effects_and_standard_errors(self):
        plan = self.plan()
        entry, config, labels, probabilities = self.results(plan, np.random.default_rng(9))
        rows, arrays = attribution.arm_results(probabilities, labels, np.arange(12), config, reps=40, seed=2)
        ledgers = {arm: {"deployed_work": 100.0 + i, "executed_work": 100.0 + i, "wall_seconds": 1.0, "feasible": True,
                         "native_site_action_counts": [1, 1] + [0] * 8, "inserted_site_action_counts": [6] + [0] * 9,
                         "members": [], "activity": 0.125} for i, arm in enumerate(attribution.ARMS)}
        directory = job_directory(self.root / "e13", entry)
        directory.mkdir(parents=True)
        attribution.write_attribution_result(directory, entry, role="test", rows=rows, arrays=arrays, ledgers=ledgers,
                                             parent_files=[], started=utc_now(), environment=ENVIRONMENT)
        target = attribution.export_attribution(self.root / "e13", root=self.records, allow_partial=True)
        verify_set(target, specs()["sets"]["inserted_site"], specs())
        lines = summary(self.records, "inserted_site")
        self.assertEqual(len(lines), 8)
        self.assertEqual({r["arm"] for r in lines}, set(attribution.ARMS))
        self.assertEqual({r["unit"] for r in lines}, {"nats"})
        identity = [r for r in lines if r["arm"] == attribution.ARMS[1]]
        self.assertEqual({r["value"] for r in identity}, {"0"})
        result = json.loads((target / "raw" / "Peptides-func" / "attribution.json").read_text())
        self.assertEqual(result["arms"][1]["work_ratio"], 1.0)
        self.assertEqual(result["layout"]["native_site_fraction"], 0.25)


# ============================================================================= cross-backbone
class CrossBackboneTests(Workspace):
    ARXIV_ROWS = [("arxiv / homophily", "residual MLP4-512"), ("arxiv / homophily", "frozen GCNII")]

    def test_substitutions_are_configuration_patches_of_the_parent(self):
        pascal = json.loads((ROOT / "configs/pascalvoc_sp_full.json").read_text())
        gine = backbones.apply_patch(pascal, backbones.default_patch("frozen GINE/GRAMA-style"))
        backbones.validate_variant(gine)
        self.assertEqual(attribution.designated_sites(gine)["native_sites"], [16 * i for i in range(8)])
        self.assertEqual({k: gine[k] for k in gine if k != "model"}, {k: pascal[k] for k in pascal if k != "model"})
        arxiv = json.loads((ROOT / "configs/arxiv_full.json").read_text())
        gcnii = backbones.apply_patch(arxiv, backbones.default_patch("frozen GCNII"))
        backbones.validate_variant(gcnii)
        self.assertEqual(gcnii["model"]["substitute_plant"]["source_sha256"],
                         ensemble_grid.sha256(ROOT / "cm_experiments" / "backbones.py"))
        stale = copy.deepcopy(gcnii)
        stale["model"]["substitute_plant"]["source_sha256"] = "0" * 64
        with self.assertRaisesRegex(RunnerError, "substitute plant"):
            backbones.validate_variant(stale)
        with self.assertRaisesRegex(RunnerError, "only model keys"):
            backbones.apply_patch(arxiv, {"/selection/new_key": 1})
        bad = backbones.apply_patch(pascal, {"/model/block_mlp_hidden": 64})
        with self.assertRaisesRegex(RunnerError, "block_mlp"):
            backbones.validate_variant(bad)

    def test_plan_reuses_model_a_and_writes_resolvable_variant_configurations(self):
        path = backbones.plan_cross_backbone([EXAMPLES / "arxiv"], self.root / "s25", rows=self.ARXIV_ROWS, reps=30)
        plan = json.loads(path.read_text())
        self.assertEqual([r["source"] for r in plan["rows"]], ["parent", "lifecycle"])
        job = plan["jobs"][0]
        self.assertTrue(job["substitute_plant"])
        variant = json.loads(Path(job["config"]).read_text())
        self.assertEqual(backbones.resolved(Path(job["config"])), variant)
        self.assertEqual(variant["model"]["backbone"], "gcnii")
        command = backbones.lifecycle_command(job, data_root="/data", device="cuda", threads=8)
        self.assertEqual(command[2:4], ["-m", "cm_experiments.backbones"])
        self.assertEqual(command[-3:], ["--stage", "all", "--resume"])
        with self.assertRaisesRegex(RunnerError, "Roman-empire parent"):
            backbones.plan_cross_backbone([EXAMPLES / "arxiv"], self.root / "x")
        self.assertFalse((self.root / "x" / "plan.json").exists())

    def test_route_stability_replays_the_sealed_selection_before_leaving_folds_out(self):
        for name, route in (("arxiv", "learned_stop"), ("peptides_func", "phase")):
            with self.subTest(run=name):
                result = backbones.route_stability(SealedRun(EXAMPLES / name), reps=30)
                self.assertTrue(result["replay_reproduces_decision"])
                self.assertEqual(result["reported_route"], route)
                self.assertEqual(result["registered_outer_folds"], 4)
                evaluated = [f for f in result["folds"] if f["evaluated"]]
                self.assertEqual(result["outer_folds"], len(evaluated))
                self.assertEqual(result["route_stable_folds"], sum(f["route_matches"] for f in evaluated))
                self.assertTrue(all(f["reason"] for f in result["folds"] if not f["evaluated"]))
        copy_root = self.root / "tampered"
        shutil.copytree(EXAMPLES / "arxiv", copy_root)
        scores = dict(np.load(copy_root / "select/unit_scores.npz"))
        scores["loss"] = scores["loss"] + np.linspace(0, 0.05, scores["loss"].shape[1])
        np.savez(copy_root / "select/unit_scores.npz", **scores)
        with self.assertRaisesRegex(RunnerError, "do not reproduce"):
            backbones.route_stability(SealedRun(copy_root), reps=30)

    def test_export_of_a_sealed_parent_row(self):
        backbones.plan_cross_backbone([EXAMPLES / "arxiv"], self.root / "s25", rows=self.ARXIV_ROWS[:1], reps=30)
        target = backbones.export_cross_backbone(self.root / "s25", root=self.records, allow_partial=True)
        verify_set(target, specs()["sets"]["cross_backbone"], specs())
        lines = {r["quantity"]: r for r in summary(self.records, "cross_backbone")}
        sealed = json.loads((EXAMPLES / "arxiv/final/results_summary.json").read_text())["paired_qD_minus_qC_loss"]
        self.assertEqual(float(lines["effect_nats"]["value"]), sealed["mean"])
        self.assertEqual((lines["route_stable_folds"]["unit"], lines["outer_folds"]["unit"]), ("folds", "folds"))
        row = json.loads((target / "raw" / "ogbn-arxiv_residual_MLP4-512" / "row.json").read_text())
        self.assertEqual(float(lines["outer_folds"]["value"]), row["outer_folds"])
        self.assertEqual(row["runs"][0]["architecture"], "FrozenPlant")


# ============================================================================= factorials
class FactorialTests(Workspace):
    def plan(self):
        path = factorial.plan_factorial([EXAMPLES / "arxiv"], self.root / "fac", role="test", reps=40,
                                        allow_fixture=True)
        return json.loads(path.read_text())

    def test_depth_grid_and_log_depth_integral(self):
        self.assertEqual(factorial.depth_grid(256), [1, 2, 4, 8, 16, 32, 64, 128, 256])
        self.assertEqual(factorial.depth_grid(6), [1, 2, 4, 6])
        values = np.array([[1.0, 2.0], [0.8, 2.0], [0.6, 2.0]])
        np.testing.assert_allclose(factorial.log_depth_integral(values, [1, 2, 4]), [0.8, 2.0])
        uneven = factorial.log_depth_integral(np.array([[1.0], [0.0]]), [2, 6])
        self.assertAlmostEqual(float(uneven[0]), 0.5)

    def test_cells_switch_the_declared_components(self):
        plan = self.plan()
        job = plan["jobs"][0]
        self.assertEqual(job["factors"], ["A", "O", "X"])
        self.assertEqual([c["cell"] for c in job["cells"]], list(factorial.CELLS))
        cell = {c["cell"]: c["switches"] for c in job["cells"]}
        self.assertEqual(cell["000"], {"action_bank": "no_transport", "observer": "memoryless", "profile": "full",
                                       "execution": "unrestricted"})
        self.assertEqual(cell["101"], {"action_bank": "registered", "observer": "memoryless", "profile": "full",
                                       "execution": "risk_controlled"})
        self.assertEqual(cell["111"], {"action_bank": "registered", "observer": "conditioned", "profile": "full",
                                       "execution": "risk_controlled"})
        definition = {c["cell"]: c["definition"] for c in job["cells"]}
        self.assertEqual(definition["000"], {"profile": "full", "observer": "memoryless", "admissible_actions": [0],
                                             "support": "unrestricted", "recovery": "nominal", "stopping": None})
        self.assertEqual(definition["010"]["admissible_actions"], [0])        # the nominal schedule ...
        self.assertEqual(definition["010"]["stopping"]["domain"], "unrestricted")   # ... under the read-only observer
        self.assertEqual(definition["001"]["admissible_actions"], [0, 8, 9])  # X brings the resets with A = 0
        self.assertEqual(definition["011"]["stopping"]["domain"], "registered")
        self.assertEqual(definition["110"]["admissible_actions"], list(range(8)))
        self.assertEqual(definition["111"]["admissible_actions"], list(range(10)))
        switch = {c["cell"]: c["switch_variant"] for c in job["cells"]}
        self.assertIsNone(switch["111"])
        self.assertIsNone(switch["101"])                     # the memoryless observer is a run-time wrapper
        self.assertEqual(switch["010"]["operations"], [{"op": "action_bank", "allowed": [0, 8, 9]},
                                                       {"op": "risk_control", "mode": "off"}])
        self.assertEqual(job["depth_grid"], [1, 2, 4, 6])
        self.assertEqual((plan["mode"], plan["fitted_plan"]), ("switch", None))      # fitted mode: test_ablation_lifecycle
        self.assertIn("declared reconstruction", plan["cell_definition_status"])
        with self.assertRaisesRegex(RunnerError, "mode is 'switch'"):
            factorial.plan_factorial([EXAMPLES / "arxiv"], self.root / "fac-fitted", role="test", mode="fitted",
                                     allow_fixture=True)
        pascal = factorial.cell_switches("010", ("S", "O", "X"), "full")
        self.assertEqual((pascal["profile"], pascal["observer"], pascal["execution"], pascal["action_bank"]),
                         ("lean", "conditioned", "unrestricted", "registered"))     # the action bank held fixed
        with self.assertRaisesRegex(RunnerError, "not executable"):
            factorial.cell_switches("1", ("O",), "full", {"O": ("observer", ("memoryless", "ema"))})

    def test_every_cell_of_both_designs_is_distinct_by_construction(self):
        for factors in (("A", "O", "X"), ("S", "O", "X")):
            cells = [{"cell": c, "definition": factorial.cell_definition(factorial.cell_switches(c, factors, "full"))}
                     for c in factorial.CELLS]
            factorial.check_distinct(cells)
            for j, factor in enumerate(factors):             # every factor changes the definition in every context
                for cell in factorial.CELLS:
                    if cell[j] == "0":
                        on = cell[:j] + "1" + cell[j + 1:]
                        self.assertNotEqual(cells[factorial.CELLS.index(cell)]["definition"],
                                            cells[factorial.CELLS.index(on)]["definition"], (factor, cell))
        table = dict(factorial.FACTOR_SWITCHES, X=("execution", ("risk_controlled", "risk_controlled")))
        collapsed = [{"cell": c, "definition": factorial.cell_definition(factorial.cell_switches(
            c, ("A", "O", "X"), "full", table))} for c in factorial.CELLS]
        with self.assertRaisesRegex(RunnerError, "same executable definition"):
            factorial.check_distinct(collapsed)
        with self.assertRaisesRegex(RunnerError, "two factors switch"):
            factorial.cell_switches("010", ("A", "O", "X"), "full",
                                    dict(factorial.FACTOR_SWITCHES, X=("observer", ("memoryless", "conditioned"))))
        with self.assertRaisesRegex(RunnerError, "same executable definition"):
            factorial.plan_factorial([EXAMPLES / "arxiv"], self.root / "collapsed", role="test", allow_fixture=True,
                                     switches={"A": ["action_bank", ["registered", "registered"]]})

    def test_read_only_observer_stops_at_the_first_eligible_positive_terminal_value(self):
        from types import SimpleNamespace

        class Controller:
            """Records the runtime's calls; terminal value, validity and A0 support by depth."""
            def __init__(self, values, invalid=(), unsupported=()):
                self.values, self.invalid, self.unsupported = values, set(invalid), set(unsupported)
                self.evaluated = []

            def start(self, y, previous_action=-1):
                return SimpleNamespace(z=np.zeros(2), valid=True, depth=0)

            def observe(self, state, y, depth, previous_action):
                return SimpleNamespace(z=np.zeros(2), valid=depth not in self.invalid, depth=depth)

            def supported(self, y, depth):
                return [1, 2] if depth in self.unsupported else [0, 1, 2]

            def terminal_value(self, state):
                self.evaluated.append(state.depth)
                return self.values[state.depth]

        def drive(controller, rule, ys, actions, depth=6):
            """The runtime's call order: start(y0), then observe(state, y_d, d, a_{d-1}) at d = 1..L-1."""
            watcher = factorial.ReadOnlyStop(controller, rule, depth)
            state = watcher.start(ys[0])
            for d in range(1, depth):
                state = watcher.observe(state, ys[d], d, actions[d - 1])
            return watcher

        rule = {"threshold_nats": 1e-7, "first_depth": 1, "domain": "unrestricted"}
        ys = [np.array([0.5 if d in (0, 2) else 1.0, 0.0]) for d in range(6)]     # label-free safe at depths 0, 2
        actions = [0, 3, 0, 5, 0, 0]
        values = {1: -0.2, 2: 5e-8, 3: -0.1, 4: 0.03, 5: 0.5}
        with self.assertRaises(factorial.StopPropagation) as caught:
            drive(Controller(values), rule, ys, actions)
        stop = caught.exception
        self.assertEqual((stop.depth, stop.value), (4, 0.03))      # 5e-8 nats is within the identity tolerance
        self.assertEqual(stop.actions, actions[:4])
        self.assertEqual(stop.anchors, [0, 0, 2, 2])
        controller = Controller(values, invalid={4})                # an invalid observer state never stops
        with self.assertRaises(factorial.StopPropagation) as caught:
            drive(controller, rule, ys, actions)
        self.assertEqual(caught.exception.depth, 5)
        self.assertEqual(controller.evaluated, [1, 2, 3, 5])
        registered = dict(rule, domain="registered")               # risk-controlled: A0 qualified in the stratum
        with self.assertRaises(factorial.StopPropagation) as caught:
            drive(Controller(values, unsupported={4}), registered, ys, actions)
        self.assertEqual(caught.exception.depth, 5)
        watcher = drive(Controller({d: -1.0 for d in range(1, 6)}), rule, ys, actions)   # no stop: L
        self.assertEqual(watcher.summary(), {"stopping_depth": None, "value_at_stop": None, "checked_depths": 5,
                                             "evaluated_depths": 5, "largest_value": -1.0})
        self.assertIsNone(factorial.stop_rule({"cell": "000", "definition": {"stopping": None}}, object()))
        with self.assertRaisesRegex(RunnerError, "terminal-value model"):
            factorial.stop_rule({"cell": "010", "definition": {"stopping": rule}},
                                SimpleNamespace(terminal_available=False))

    def test_joint_bootstrap_reproduces_the_registered_marginal_intervals(self):
        from cross_morse.statistics import paired_bootstrap
        rng = np.random.default_rng(4)
        values = {c: rng.normal(1.0 + 0.01 * i, 0.1, 9) for i, c in enumerate(factorial.CELLS)}
        weights = rng.integers(1, 6, 9).astype(float)
        cells, contrasts, draws = factorial.cell_statistics(values, weights, ("A", "O", "X"), reps=300, seed=8)
        for i, c in enumerate(factorial.CELLS):
            boot = paired_bootstrap(values[c], weights=weights, reps=300, alpha=0.025, seed=8)
            self.assertEqual(cells[c]["interval_low"], boot["lower"])
            self.assertEqual(cells[c]["interval_high"], boot["upper"])
            self.assertEqual(float(np.quantile(draws[:, i], 0.025)), boot["lower"])
        names = [c["contrast"] for c in contrasts]
        self.assertEqual(names, ["A", "O", "X", "A:O", "A:X", "O:X", "A:O:X"])
        point = {c: cells[c]["ld_integrated_loss"] for c in factorial.CELLS}
        main_a = np.mean([point[c] for c in factorial.CELLS if c[0] == "1"]) - np.mean(
            [point[c] for c in factorial.CELLS if c[0] == "0"])
        self.assertAlmostEqual(contrasts[0]["effect_coded_nats"], main_a, places=12)
        self.assertAlmostEqual(contrasts[3]["baseline_anchored_nats"],
                               point["110"] - point["100"] - point["010"] + point["000"], places=12)

    def write_job(self, plan, rng):
        entry = plan["jobs"][0]
        config = SealedRun(entry["run"]).config
        depths = entry["depth_grid"]
        counts = np.full(8, 5.0)
        sums = {c: rng.uniform(0.5, 1.5, (len(depths), 8)) * counts for c in factorial.CELLS}
        terminal = {c: rng.dirichlet(np.ones(4), 40).astype(np.float32) for c in factorial.CELLS}
        ledger = {c: {"wall_seconds": 2.0 + 0.2 * i, "deployed_work": 100.0, "actions": 60, "active": 3 * i,
                      "inputs": 1, "feasible": True, "terminal_depths": [6],
                      "executions": [{"input": 0, "member": s, "terminal_depth": 3 if c[1] == "1" and s == 1 else 6,
                                      "stopped": c[1] == "1" and s == 1, "stop_value": 0.02 if c[1] == "1" else None,
                                      "evaluated_depths": 2, "wall_seconds": 1.0, "actions": 3, "active": 1}
                                     for s in (1, 2)]} for i, c in enumerate(factorial.CELLS)}
        directory = job_directory(self.root / "fac", entry)
        directory.mkdir(parents=True)
        factorial.write_factorial_result(directory, plan, entry, sums=sums, counts=counts, unit_ids=np.arange(8),
                                         ledger=ledger, terminal=terminal, labels=rng.integers(0, 4, 40),
                                         config=config, parent_files=[], started=utc_now(), environment=ENVIRONMENT)
        return sums, depths

    def test_cell_quantities_and_export(self):
        plan = self.plan()
        sums, depths = self.write_job(plan, np.random.default_rng(12))
        result = json.loads((job_directory(self.root / "fac", plan["jobs"][0]) / "factorial.json").read_text())
        cells = {c["cell"]: c for c in result["cells"]}
        self.assertEqual(cells["000"]["relative_time"], 1.0)
        self.assertAlmostEqual(cells["100"]["relative_time"], 2.2 / 2.0)
        self.assertAlmostEqual(cells["010"]["activity"], 6 / 60)
        expected = float(np.mean(factorial.log_depth_integral(sums["111"] / 5.0, depths)))
        self.assertAlmostEqual(cells["111"]["ld_integrated_loss"], expected, places=12)
        self.assertEqual(result["units"]["official_metric"], "%")
        self.assertEqual(cells["010"]["stopping"]["stopped"], 1)
        self.assertEqual(cells["010"]["stopping"]["stopping_depth"]["median"], 3.0)
        self.assertEqual(cells["010"]["stopping"]["terminal_depth"]["max"], 6.0)
        self.assertIsNone(cells["000"]["stopping"]["rule"])
        self.assertEqual(cells["000"]["stopping"]["stopped"], 0)
        self.assertEqual(result["coinciding_cells"], [])
        executions = np.load(job_directory(self.root / "fac", plan["jobs"][0]) / "executions.npz")
        self.assertEqual(executions["cell010_terminal_depth"].tolist(), [3, 6])
        self.assertTrue(np.isnan(executions["cell000_stop_value"]).all())
        target = factorial.export_factorial(self.root / "fac", root=self.records, allow_partial=True)
        verify_set(target, specs()["sets"]["factorials"], specs())
        lines = summary(self.records, "factorials")
        self.assertEqual(len(lines), 48)
        units = {r["quantity"]: r["unit"] for r in lines}
        self.assertEqual(units, {"ld_integrated_loss": "nats", "interval_low": "nats", "interval_high": "nats",
                                 "official_metric": "%", "activity": "fraction", "relative_time": "cell 000 = 1"})
        provenance = json.loads((target / "provenance.json").read_text())
        self.assertEqual(provenance["factor_switches"]["A"], ["action_bank", ["no_transport", "registered"]])
        self.assertEqual(provenance["cell_definitions"]["ogbn-arxiv"]["010"]["stopping"]["domain"], "unrestricted")
        self.assertTrue((target / "raw" / "ogbn-arxiv" / "executions.npz").is_file())
        self.assertTrue((target / "code" / "cm_experiments" / "ablation_lifecycle.py").is_file())   # switch installer


# ============================================================================= job orchestration without the runtime
class FakeData:
    def __init__(self):
        self.scopes = []

    @contextlib.contextmanager
    def _label_scope(self, stage):
        self.scopes.append(stage)
        yield []


class FakeContext:
    """Stands in for ParentContext: the parent's real role IDs, synthetic labels and member predictions,
    and execution traces with the runtime's fields.  It exercises the jobs' orchestration only."""

    def __init__(self, run, device, *, through):
        import hashlib
        self.run, self.config, self.device, self.through = run, run.config, device, through
        self.data, self.reference, self.graph = FakeData(), {}, None
        self.splits = dict(np.load(run.root / "prepared" / "splits.npz"))
        self.multilabel = run.dataset == "Peptides-func"
        self.hash = hashlib.sha256

    def rng(self, *parts):
        return np.random.default_rng(int(self.hash(repr(parts).encode()).hexdigest()[:8], 16))

    def role_ids(self, role):
        return np.asarray(self.splits[role], dtype=np.int64)

    def units(self, role):
        ids = self.role_ids(role)
        return ids if self.multilabel else ids % 3

    def labels(self, role):
        n = len(self.role_ids(role))
        r = self.rng("labels", role)
        return r.integers(0, 2, (n, 4)).astype(float) if self.multilabel else r.integers(0, 4, n)

    def probabilities(self, n, *parts):
        r = self.rng(*parts)
        if self.multilabel:
            return r.uniform(0.05, 0.95, (n, 4)).astype(np.float32)
        return r.dirichlet(np.ones(4), n).astype(np.float32)

    def plant(self, seed, directory=None):
        from types import SimpleNamespace
        layout = attribution.designated_sites(self.config) if self.multilabel else {}
        return SimpleNamespace(seed=int(seed), backbone=layout.get("backbone"), control_depth=self.config["model"]["depth"],
                               native_sites=tuple(layout.get("native_sites", ())))

    def controller_for(self, candidate):
        return FakeController() if candidate["route"] in ensemble_grid.CONTROLLER_ROUTES else None

    def predict(self, model, candidate, controller, role):
        restricted = candidate.get("site_restriction") is not None
        inserted = [4, 0, 0, 0, 0, 0, 0, 0, 0, 0] if restricted else [2, 1, 0, 1, 0, 0, 0, 0, 0, 0]
        totals = {"deployed_multiply_add_work": 100.0 - 20 * restricted, "multiply_add_work": 120.0,
                  "wall_seconds": 0.25, "feasible": True, "native_site_action_counts": [1, 1] + [0] * 8,
                  "inserted_site_action_counts": inserted}
        return self.probabilities(len(self.role_ids(role)), candidate["hash"], model.seed, role), totals

    def episodes(self, role):
        ids = self.role_ids(role)
        yield 0, None, None, ids, np.arange(len(ids))


def fake_execute(model, x, graph, config, candidate, controller, reference, depths, rows, stopping=None):
    """factorial._execute's outputs: the read-only observer of an observer cell stops member 1 at depth 2."""
    seed, depth = int(model.seed), config["model"]["depth"]
    stop = 2 if stopping is not None and seed == 1 else None
    executed = depth if stop is None else stop
    by_depth = [np.random.default_rng(seed * 100 + min(d, executed) + len(rows)).dirichlet(np.ones(4), len(rows))
                .astype(np.float32) for d in depths]
    actions = [0] * executed if controller == "nominal" else ([0, 1] * depth)[:executed]
    return by_depth, {"actions": actions, "wall_seconds": 0.5, "deployed_work": 10.0, "feasible": True,
                      "terminal_depth": executed, "stopped": stop is not None,
                      "stop": None if stopping is None else {"stopping_depth": stop, "value_at_stop": 0.01 if stop else None,
                                                             "checked_depths": executed, "evaluated_depths": executed,
                                                             "largest_value": 0.01}}


class OrchestrationTests(Workspace):
    def test_attribution_job_aggregates_members_and_exports(self):
        output = self.root / "e13"
        attribution.plan_attribution([EXAMPLES / "peptides_func"], output, role="compare", reps=30, allow_fixture=True)
        with patch.object(attribution, "ParentContext", FakeContext), patch.object(attribution, "environment_record",
                                                                                   return_value=ENVIRONMENT):
            with self.assertRaisesRegex(RunnerError, "trust"):
                attribution.run_attribution(output)
            paths = attribution.run_attribution(output, trust=True)
            self.assertEqual(attribution.run_attribution(output, trust=True), paths)   # completed jobs are skipped
        result = json.loads(paths[0].read_text())
        arms = {a["arm"]: a for a in result["arms"]}
        self.assertEqual(arms[attribution.ARMS[1]]["effect_nats"], 0.0)
        self.assertEqual(arms[attribution.ARMS[1]]["work_ratio"], 1.0)
        self.assertAlmostEqual(arms[attribution.ARMS[0]]["work_ratio"], 0.8)
        self.assertEqual(arms[attribution.ARMS[0]]["inserted_site_action_counts"], [8] + [0] * 9)
        self.assertEqual(len(arms[attribution.ARMS[2]]["members"]), 2)
        self.assertAlmostEqual(arms[attribution.ARMS[3]]["activity"], 3 / 6)    # per member: 6 actions, 3 not A0
        self.assertAlmostEqual(arms[attribution.ARMS[0]]["activity"], 1 / 6)
        self.assertEqual(result["role"], "compare")
        member = json.loads((paths[0].parent / "members" / "native_open_loop" / "seed1.json").read_text())
        self.assertEqual(member["arm"], attribution.ARMS[0])
        self.assertEqual(member["candidate"]["site_restriction"], attribution.RESTRICTION)
        target = attribution.export_attribution(output, root=self.records, allow_partial=True)
        verify_set(target, specs()["sets"]["inserted_site"], specs())
        plan = (target / "code" / "plan.json").read_text()
        self.assertNotIn(str(EXAMPLES), plan)
        self.assertIn('"run": "peptides_func"', plan)

    def test_ensemble_grid_job_trains_executes_scores_and_exports(self):
        import sys
        from types import ModuleType
        output = self.root / "e14"
        ensemble_grid.plan_ensemble_grid([EXAMPLES / "arxiv"], output, grid=[1, 2, 3], extra_seeds=[3], reps=30)
        trained = []

        def train_one(data, graph, config, directory, seed, device, baseline=False):
            self.assertEqual(data.scopes[-1], "train")
            Path(directory).mkdir(parents=True, exist_ok=True)
            (Path(directory) / f"plant_seed{seed}_best.pt").write_bytes(b"checkpoint %d" % seed)
            trained.append(seed)
            return {"seed": seed, "epochs_completed": 1, "best_internal_fit_val_loss": 1.2}

        training = ModuleType("cross_morse.training")
        training.train_one = train_one
        with patch.dict(sys.modules, {"cross_morse.training": training}), \
                patch.object(ensemble_grid, "ParentContext", FakeContext), \
                patch.object(ensemble_grid, "environment_record", return_value=ENVIRONMENT):
            path = ensemble_grid.run_ensemble_grid(output, trust=True)[0]
        self.assertEqual(trained, [3])
        grid = json.loads(path.read_text())
        self.assertEqual([c["origin"] for c in grid["checkpoints"]], ["parent", "parent", "extra"])
        self.assertEqual([m["seed"] for m in grid["members"]], [1, 2, 3])
        self.assertEqual(grid["ensembles"][2]["seeds"], [1, 2, 3])
        self.assertIsNotNone(grid["qC_reproduction"])
        self.assertIn("deployed_minus_row", grid["ensembles"][0])
        context = FakeContext(SealedRun(EXAMPLES / "arxiv"), "cpu", through="evaluate")
        candidate = SealedRun(EXAMPLES / "arxiv").comparator()
        members = [context.probabilities(len(context.role_ids("test")), candidate["hash"], s, "test") for s in (1, 2)]
        from cross_morse.evaluation import score_observations
        expected = score_observations(ensemble_grid.ensemble_probabilities(members), context.labels("test"),
                                      context.units("test"), context.config)["node_ce"]
        self.assertEqual(grid["ensembles"][1]["loss_nats"], expected)
        target = ensemble_grid.export_ensemble_grid(output, root=self.records, allow_partial=True)
        verify_set(target, specs()["sets"]["ensemble_grid"], specs())
        self.assertNotIn(str(EXAMPLES), (target / "raw" / "ogbn-arxiv" / "arxiv" / "grid.json").read_text())

    def test_factorial_job_accumulates_depth_losses_and_exports(self):
        output = self.root / "fac"
        factorial.plan_factorial([EXAMPLES / "arxiv"], output, role="test", reps=30, warmup=1, allow_fixture=True)

        def controller(ctx, candidate, switches, cache):
            return "nominal" if switches["action_bank"] == "no_transport" else "registered"

        installed = []

        @contextlib.contextmanager
        def scope(scopes, cell):             # the switch variant the runtime would have installed for the cell
            installed.append((cell, (scopes[cell][1] or {}).get("operations")))
            yield

        with patch.object(factorial, "ParentContext", FakeContext), patch.object(factorial, "_execute", fake_execute), \
                patch.object(factorial, "cell_controller", controller), patch.object(factorial, "cell_scope", scope), \
                patch.object(factorial, "environment_record", return_value=ENVIRONMENT):
            path = factorial.run_factorial(output, trust=True)[0]
        self.assertIn(("000", [{"op": "action_bank", "allowed": [0, 8, 9]}, {"op": "risk_control", "mode": "off"}]),
                      installed)
        self.assertIn(("111", None), installed)
        result = json.loads(path.read_text())
        self.assertEqual(result["mode"], "switch")
        cells = {c["cell"]: c for c in result["cells"]}
        for name, cell in cells.items():
            self.assertIsNone(cell["fitted"])
            self.assertEqual(cell["activity"], 0.0 if name[0] == "0" else 0.5)
            self.assertEqual(cell["relative_time"], 1.0)
            self.assertEqual(cell["inputs"], 1)
            self.assertLessEqual(cell["interval_low"], cell["interval_high"])
            self.assertEqual(cell["stopping"]["stopped"], 1 if name[1] == "1" else 0)   # member 1 stops at depth 2
            self.assertEqual(sum(cell["action_counts"]), 8 if name[1] == "1" else 12)
        self.assertEqual(cells["010"]["integration_domain"]["terminal_depths"], [2, 6])
        executions = np.load(path.parent / "executions.npz")
        self.assertEqual(executions["cell010_stopped"].tolist(), [1, 0])
        self.assertEqual(executions["cell010_member"].tolist(), [1, 2])
        arrays = np.load(path.parent / "unit_dil.npz")
        np.testing.assert_allclose(arrays["cell111_unit_dil"],
                                   factorial.log_depth_integral(arrays["cell111_unit_depth_loss"], [1, 2, 4, 6]))
        self.assertEqual(arrays["joint_bootstrap_cell_means"].shape, (30, 8))
        target = factorial.export_factorial(output, root=self.records, allow_partial=True)
        verify_set(target, specs()["sets"]["factorials"], specs())

    def test_private_infrastructure_identifiers_are_never_deposited(self):
        output = self.root / "e14"
        plan = json.loads(ensemble_grid.plan_ensemble_grid([EXAMPLES / "arxiv"], output, grid=[1, 2], reps=30)
                          .read_text())
        entry = plan["jobs"][0]
        config = SealedRun(entry["run"]).config
        rng = np.random.default_rng(1)
        members = [rng.dirichlet(np.ones(4), 20).astype(np.float32) for _ in range(2)]
        rows, arrays = ensemble_grid.grid_statistics(iter(members), rng.integers(0, 4, 20), np.arange(20) % 4, config,
                                                     [1, 2], [1, 2], parent_seeds=[1, 2], comparator_ensemble=2,
                                                     reps=30, seed=1)
        directory = job_directory(output, entry)
        directory.mkdir(parents=True)
        leaky = dict(ENVIRONMENT, slurm={"SLURM_JOB_PARTITION": "naiss" + "2025-22-123-gpu"})
        ensemble_grid.write_grid_result(directory, entry, role="test", rows=rows, arrays=arrays, checkpoints=[],
                                        members=[], reproduction=None, parent_files=[], started=utc_now(),
                                        environment=leaky)
        with self.assertRaisesRegex(RunnerError, "private infrastructure"):
            ensemble_grid.export_ensemble_grid(output, root=self.records, allow_partial=True)
        self.assertFalse((self.records / "ensemble_grid").exists())


# ============================================================================= command line
class CommandLineTests(Workspace):
    def call(self, *argv):
        from scripts.run_experiment import main
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(list(map(str, argv)))
        return code, out.getvalue(), err.getvalue()

    def test_plan_run_guard_and_export(self):
        output = self.root / "cli"
        code, out, _ = self.call("ensemble-grid", "plan", "--parent", EXAMPLES / "arxiv", "--output", output,
                                 "--grid", 1, 2, "--reps", 30)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["result"], str(output / "plan.json"))
        code, _, err = self.call("ensemble-grid", "run", "--output", output, "--device", "cpu")
        self.assertEqual(code, 2)
        self.assertIn("trust", err)
        code, _, err = self.call("ensemble-grid", "export", "--output", output, "--root", self.records,
                                 "--allow-partial")
        self.assertEqual(code, 2)
        self.assertIn("has not completed", err)
        code, _, err = self.call("cross-backbone", "run", "--output", output)
        self.assertEqual(code, 2)
        self.assertIn("not a cross_backbone plan", err)


# ============================================================================= execution (PyTorch; CI)
@unittest.skipUnless(TORCH, "PyTorch is required to execute the runners")
class GCNIIPlantTests(unittest.TestCase):
    def test_adapter_a0_reproduces_the_native_step_and_identity_is_enforced(self):
        import torch
        import cross_morse.core as core
        import cross_morse.training as training
        from cross_morse.core import GraphOperator
        from cross_morse.tasks import TaskSpec
        model = json.loads((ROOT / "configs/arxiv_smoke.json").read_text())["model"]
        model = backbones.apply_patch({"model": model}, backbones.default_patch("frozen GCNII"))["model"]
        with patch.object(core, "build_plant", core.build_plant), patch.object(training, "build_plant",
                                                                               training.build_plant):
            backbones.install_substitute_plants()
            plant = core.build_plant(12, 4, model, TaskSpec(num_outputs=4)).freeze()
            self.assertIs(training.build_plant, core.build_plant)
            stale = dict(model, substitute_plant=dict(model["substitute_plant"], source_sha256="0" * 64))
            with self.assertRaisesRegex(ValueError, "substitute plant"):
                core.build_plant(12, 4, stale, TaskSpec(num_outputs=4))
        self.assertEqual(type(plant).__name__, "GCNIIPlant")
        self.assertEqual(type(plant).__module__, "cm_experiments.backbones")
        edges = torch.stack([torch.arange(10), (torch.arange(10) + 1) % 10])
        graph = GraphOperator(torch.cat([edges, edges.flip(0)], 1), 10)
        torch.manual_seed(0)
        h0 = plant.encode(torch.randn(10, 12))
        h = h0
        for site in range(model["depth"]):
            self.assertTrue(torch.equal(plant.native_step(h, h0, graph, site=site),
                                        plant.step(h, h0, graph, 0, site=site)))
            h = plant.step(h, h0, graph, 5, site=site)
        self.assertEqual(plant.instrumentation_metadata()["substitute_plant"], backbones.substitute_identity())


@unittest.skipUnless(TORCH, "PyTorch is required to execute the runners")
class RunnerExecutionTests(unittest.TestCase):
    """Smoke lifecycles on synthetic data, then each runner end to end (software checks, not measurements)."""

    @classmethod
    def setUpClass(cls):
        import torch
        from cross_morse.pipeline import Pipeline
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory(prefix="cm-runner-runs-")
        cls.root = Path(cls.temporary.name)
        cls.runs = {}
        for name, stem in (("arxiv", "arxiv_smoke"), ("peptides", "peptides_func_smoke")):
            config = json.loads((ROOT / "configs" / f"{stem}.json").read_text())
            if name == "arxiv":
                config["training"].update(epochs=1, patience=1)
            config["controller"].update(epochs=1, patience=1)
            config["selection"]["bootstrap_reps"] = 20
            config["admission"]["bootstrap_reps"] = 20
            run = cls.root / "runs" / name
            with contextlib.redirect_stdout(io.StringIO()):
                Pipeline(config, cls.root / "data", run, "cpu").run()
            cls.runs[name] = run

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.work = Path(tempfile.mkdtemp(prefix="cm-runner-", dir=self.root))
        self.records = self.work / "records"

    def test_ensemble_grid_trains_extra_checkpoints_and_reproduces_the_sealed_comparator(self):
        output = self.work / "e14"
        ensemble_grid.plan_ensemble_grid([self.runs["arxiv"]], output, grid=[1, 2, 3], extra_seeds=[3], reps=20)
        with contextlib.redirect_stdout(io.StringIO()):
            ensemble_grid.run_ensemble_grid(output, device="cpu", trust=True)
        plan = json.loads((output / "plan.json").read_text())
        directory = job_directory(output, plan["jobs"][0])
        grid = json.loads((directory / "grid.json").read_text())
        self.assertEqual([r["ensemble"] for r in grid["ensembles"]], [1, 2, 3])
        self.assertEqual(grid["ensembles"][2]["new_checkpoints"], 1)
        self.assertTrue((directory / "checkpoints" / "plant_seed3_best.pt").is_file())
        self.assertEqual([c["origin"] for c in grid["checkpoints"]], ["parent", "parent", "extra"])
        self.assertLess(grid["qC_reproduction"]["abs_difference"], 1e-6)
        target = ensemble_grid.export_ensemble_grid(output, root=self.records, allow_partial=True)
        verify_set(target, specs()["sets"]["ensemble_grid"], specs())

    def test_attribution_holds_inserted_sites_at_the_bypass(self):
        output = self.work / "e13"
        try:
            attribution.plan_attribution([self.runs["peptides"]], output, role="test", reps=20, allow_fixture=True)
        except RunnerError as exc:
            if "matched-ensemble" not in str(exc):
                raise
            self.skipTest("this smoke run confirmed a feedback route at another ensemble size than q_C")
        attribution.run_attribution(output, device="cpu", trust=True)
        plan = json.loads((output / "plan.json").read_text())
        result = json.loads((job_directory(output, plan["jobs"][0]) / "attribution.json").read_text())
        arms = {a["arm"]: a for a in result["arms"]}
        for arm in (attribution.ARMS[0], attribution.ARMS[2]):
            self.assertEqual(sum(arms[arm]["inserted_site_action_counts"][1:]), 0)
        self.assertEqual(arms[attribution.ARMS[1]]["effect_nats"], 0.0)
        self.assertEqual(arms[attribution.ARMS[1]]["work_ratio"], 1.0)
        target = attribution.export_attribution(output, root=self.records, allow_partial=True)
        verify_set(target, specs()["sets"]["inserted_site"], specs())

    def test_factorial_cells_execute_and_export(self):
        output = self.work / "fac"
        factorial.plan_factorial([self.runs["arxiv"]], output, role="test", reps=20, warmup=0, allow_fixture=True)
        factorial.run_factorial(output, device="cpu", trust=True)
        plan = json.loads((output / "plan.json").read_text())
        result = json.loads((job_directory(output, plan["jobs"][0]) / "factorial.json").read_text())
        cells = {c["cell"]: c for c in result["cells"]}
        self.assertEqual(list(cells), list(factorial.CELLS))
        self.assertEqual(cells["000"]["relative_time"], 1.0)
        for name in ("000", "010"):      # A0 only, no reset: the nominal trajectory, stopped or not
            self.assertEqual(cells[name]["activity"], 0.0)
        depth = json.loads((self.runs["arxiv"] / "config.json").read_text())["model"]["depth"]
        executions = np.load(job_directory(output, plan["jobs"][0]) / "executions.npz")
        for name, cell in cells.items():
            self.assertTrue(math.isfinite(cell["ld_integrated_loss"]))
            self.assertLessEqual(cell["interval_low"], cell["interval_high"])
            counts = np.asarray(cell["action_counts"])
            if name[0] == "0":
                self.assertEqual(int(counts[1:8].sum()), 0)          # A = 0: no transport action
            if name[2] == "0":
                self.assertEqual(int(counts[8:].sum()), 0)           # X = 0: no reset
            terminal, stopped = executions[f"cell{name}_terminal_depth"], executions[f"cell{name}_stopped"]
            self.assertTrue(np.all((terminal < depth) == (stopped == 1)))
            if name[1] == "0":
                self.assertEqual(int(stopped.sum()), 0)              # no stop without the conditioned observer
        self.assertEqual(["000", "010"] in result["coinciding_cells"], cells["010"]["stopping"]["stopped"] == 0)
        target = factorial.export_factorial(output, root=self.records, allow_partial=True)
        verify_set(target, specs()["sets"]["factorials"], specs())

    def test_cross_backbone_gcnii_lifecycle_and_export(self):
        output = self.work / "s25"
        backbones.plan_cross_backbone([self.runs["arxiv"]], output, rows=CrossBackboneTests.ARXIV_ROWS, reps=20)
        backbones.run_cross_backbone(output, data_root=self.root / "data", device="cpu", threads=1)
        plan = json.loads((output / "plan.json").read_text())
        variant = Path(plan["jobs"][0]["run"])
        inventory = json.loads((variant / "train" / "model_inventory.json").read_text())
        self.assertEqual((inventory["architecture"], inventory["architecture_module"]),
                         ("GCNIIPlant", "cm_experiments.backbones"))
        target = backbones.export_cross_backbone(output, root=self.records, allow_partial=True)
        verify_set(target, specs()["sets"]["cross_backbone"], specs())
        rows = read_summary(target, specs()["sets"]["cross_backbone"], specs()["summary_columns"])
        self.assertEqual({(r["regime"], r["model"]) for r in rows}, set(CrossBackboneTests.ARXIV_ROWS))


if __name__ == "__main__":
    unittest.main()
