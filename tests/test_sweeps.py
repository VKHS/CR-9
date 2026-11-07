"""Sweeps and module ablations of cm_experiments/sweeps.py and their record export.

The recomputations from sealed outputs are checked on the two bundled example
runs (they must reproduce each sealed decision at the registered setting) and on
small hand-built run directories. The lifecycle runner is exercised with a fake
process runner; the tests that execute the released lifecycle need PyTorch.

The buffer sweep (cm_experiments/buffered_admission.py) is checked on hand-computed
walks, on METIS-like partitions of random geometric and stochastic-block graphs
(where the units must fall gradually) and on synthetic sealed runs whose admission
was computed by the runtime's own admission function.
"""
from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import itertools
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from cm_experiments import buffered_admission as buffered
from cm_experiments import sweeps
from scripts import export_sweep_records, records

ROOT = Path(__file__).resolve().parents[1]
ARXIV = ROOT / "examples/runs/arxiv"
PEPTIDES = ROOT / "examples/runs/peptides_func"
HAS_TORCH = importlib.util.find_spec("torch") is not None
BUFFERS = [2, 4, 8, 16, 32]


def write(root: Path, files: dict) -> Path:
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content), encoding="utf-8")
    return root


class Workspace(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


class Variants(unittest.TestCase):
    def test_patch_existing_keys_only(self):
        parent = {"controller": {"beam_width": 16, "horizons": [1, 4]}, "support": {"p_min": 0.05}}
        variant = sweeps.variant_config(parent, {"/controller/beam_width": 4, "/support/p_min": 0.1})
        self.assertEqual(variant["controller"]["beam_width"], 4)
        self.assertEqual(variant["support"]["p_min"], 0.1)
        self.assertEqual(parent["controller"]["beam_width"], 16)        # parent untouched
        with self.assertRaises(sweeps.SweepError):
            sweeps.variant_config(parent, {"/controller/beam": 4})
        with self.assertRaises(sweeps.SweepError):
            sweeps.variant_config(parent, {"/sensor/bands": [0, 1, 2]})
        self.assertEqual(sweeps.config_value(parent, "/controller/horizons"), [1, 4])

    def test_lifecycle_command(self):
        command = sweeps.lifecycle_command(Path("/v/config.json"), Path("/v/run"), "fit", data_root=Path("/d"),
                                           device="cpu", threads=2, python="python")
        self.assertEqual(command[:3], ["python", "-B", str(ROOT / "run_cross_morse.py")])
        self.assertEqual(command[-3:], ["--stage", "fit", "--resume"])
        self.assertEqual(command[command.index("--threads") + 1], "2")
        self.assertEqual(command[command.index("--config") + 1], "/v/config.json")
        self.assertEqual(command[command.index("--stage") + 1], "fit")


class SelectionRule(unittest.TestCase):
    def test_example_runs_reproduce_sealed_selection(self):
        for root in (ARXIV, PEPTIDES):
            run = sweeps.SealedRun(root)
            outcome = sweeps.selection_outcome(run)
            decision = json.loads((root / "select/decision.json").read_text())
            self.assertEqual(outcome["eligible"], decision["Q_s"])
            self.assertEqual(outcome["selected"], decision["provisional_index"])
            self.assertIn("select/decision.json", run.read)

    def test_margins_on_known_limits(self):
        candidates = [
            {"route": "nominal", "is_open_loop": True, "complexity": 0, "ensemble": 1, "cost": 1.0, "hash": "a"},
            {"route": "one_step", "is_open_loop": False, "complexity": 6, "ensemble": 2, "cost": 3.0, "hash": "b"},
            {"route": "radius", "is_open_loop": False, "complexity": 7, "ensemble": 2, "cost": 3.5, "hash": "c"},
            {"route": "mpc_rh", "is_open_loop": False, "complexity": 8, "ensemble": 2, "cost": 5.0, "hash": "d"}]
        decision = {"U": [0.0, -0.0035, -0.0058, -0.0070], "joint_feasible": [True] * 4, "comparator_index": 0}
        # min U = -0.007; eps_eq 0.002 keeps U <= -0.005 (radius, MPC-RH) and radius is simpler; eps_eq 0.001 only MPC-RH
        self.assertEqual(sweeps.eq19(decision, candidates, 0.003, 0.002), {"eligible": [2, 3], "selected": 2,
                                                                            "minimum_upper": -0.007})
        self.assertEqual(sweeps.eq19(decision, candidates, 0.003, 0.001)["eligible"], [3])
        self.assertEqual(sweeps.eq19(decision, candidates, 0.003, 0.004)["selected"], 1)
        self.assertEqual(sweeps.eq19(decision, candidates, 0.008, 0.002)["selected"], 0)   # Q_s empty: q_C
        quantities = sweeps.route_quantities(candidates[1], candidates[2])
        self.assertEqual(quantities, {"selected_complexity": 6.0, "route_changed": 1.0, "route_simpler": 1.0})


class SupportRule(unittest.TestCase):
    def test_known_cells(self):
        def cell(stratum, action, p, ess, w):
            return {"stratum": stratum, "action": action, "min_propensity": p, "ess": ess, "max_normalized_weight": w}
        cells = {"0:0": cell(0, 0, 0.28, 100, 0.05), "0:1": cell(0, 1, 0.09, 70, 0.12),
                 "1:0": cell(1, 0, 0.28, 120, 0.04), "1:2": cell(1, 2, 0.06, 40, 0.18)}
        support = {"cells": cells}
        registered = sweeps.support_summary(support, 0.05, 64, 0.2)
        self.assertEqual(registered["qualified_fraction"], 0.75)
        self.assertEqual(registered["strata_without_transport"], 1.0)    # stratum 1 keeps only A0
        self.assertEqual(sweeps.support_summary(support, 0.05, 32, 0.2)["strata_without_transport"], 0.0)
        self.assertEqual(sweeps.support_summary(support, 0.10, 64, 0.2)["qualified_fraction"], 0.5)

    def test_sealed_masks_reproduced(self):
        for root in (ARXIV, PEPTIDES):
            run = sweeps.SealedRun(root)
            sweeps.check_support_rule(run, run.json("fit/full/audit.json")["support"])

    def test_tampered_mask_detected(self):
        run = sweeps.SealedRun(ARXIV)
        support = json.loads(json.dumps(run.json("fit/full/audit.json")["support"]))
        key = next(iter(support["cells"]))
        support["cells"][key]["qualified"] = not support["cells"][key]["qualified"]
        with self.assertRaises(sweeps.SweepError):
            sweeps.check_support_rule(run, support)


class AdmissionReplay(unittest.TestCase):
    def test_sealed_admission_reproduced(self):
        for root in (ARXIV, PEPTIDES):
            sweeps.check_admission_replay(sweeps.SealedRun(root))

    def test_bernstein_penalty_scales_with_clip(self):
        # zero paired differences: U_a = 14 B log(2/alpha) / (3 (J - 1)), linear in the clip level B
        run = sweeps.SealedRun(PEPTIDES)
        uppers = [sweeps.admission_replay(run, clip)["admission_upper_nats"] for clip in (0.05, 0.1, 0.2)]
        self.assertAlmostEqual(uppers[1], 2 * uppers[0], places=12)
        self.assertAlmostEqual(uppers[2], 4 * uppers[0], places=12)

    def test_repeated_split_upper(self):
        self.assertAlmostEqual(sweeps.repeated_split_upper([-0.006, -0.007, -0.0065]),
                               -0.0065 + 4.303 * 0.0005 / 3 ** 0.5, places=12)


class Planning(Workspace):
    def test_horizon_beam_plan(self):
        with self.assertRaises(sweeps.SweepError) as caught:      # the examples were sealed by an older runtime
            sweeps.plan_sweep("horizon_beam_sweep", [ARXIV], self.tmp / "refused", horizon=2, horizons=[1, 2, 4],
                              beams=[1, 2, 3], profile="full", ensemble=2)
        self.assertIn("source snapshot", str(caught.exception))
        path = sweeps.plan_sweep("horizon_beam_sweep", [ARXIV], self.tmp / "hb", horizon=2, horizons=[1, 2, 4],
                                 beams=[1, 2, 3], profile="full", ensemble=2, allow_source_change=True)
        plan = json.loads(path.read_text())
        self.assertEqual(plan["parent_runtime_source_mismatch"], plan["parents"][0]["runtime_source_sha256"])
        points = {(p["horizon"], p["beam_width"]): p for p in plan["settings"]["points"]}
        self.assertEqual(set(points), {(1, 2), (2, 2), (4, 2), (2, 1), (2, 3)})
        self.assertEqual(points[(2, 2)]["source"], "parent")
        variants = {v["id"]: v for v in plan["variants"]}
        self.assertEqual(set(variants), {"H4_B2", "H2_B1", "H2_B3", "replicate"})
        self.assertEqual(variants["replicate"]["patch"], {})
        self.assertEqual(variants["replicate"]["config_sha256"], variants["replicate"]["parent_config_sha256"])
        self.assertEqual(variants["H4_B2"]["patch"], {"/controller/horizons": [1, 2, 4]})
        self.assertEqual(variants["H2_B1"]["patch"], {"/controller/beam_width": 1})
        config = json.loads((self.tmp / "hb" / variants["H2_B1"]["config"]).read_text())
        parent = json.loads((ARXIV / "config.json").read_text())
        parent["controller"]["beam_width"] = 1
        self.assertEqual(config, parent)
        self.assertEqual(plan["stages"], ["prepare", "train", "explore", "fit", "select"])
        with self.assertRaises(sweeps.SweepError):                       # plans are never overwritten
            sweeps.plan_sweep("horizon_beam_sweep", [ARXIV], self.tmp / "hb", horizon=2, horizons=[1, 2],
                              beams=[2], profile="full", ensemble=2, allow_source_change=True)

    def test_registered_configuration_required(self):
        with self.assertRaises(sweeps.SweepError):                       # beam 2 is not the registered 16
            sweeps.plan_sweep("horizon_beam_sweep", [ARXIV], self.tmp / "a", profile="full", ensemble=2)
        with self.assertRaises(sweeps.SweepError):                       # the deployed rule is not MPC-RH
            sweeps.plan_sweep("horizon_beam_sweep", [ARXIV], self.tmp / "b", horizon=2, horizons=[2], beams=[2])

    def test_buffer_plan(self):
        path = sweeps.plan_sweep("buffer_sensitivity", [ARXIV], self.tmp / "buffer")
        plan = json.loads(path.read_text())
        settings = plan["settings"]
        self.assertEqual(settings["grid"], BUFFERS)
        self.assertEqual(settings["registered_buffer_hops"], 2)
        self.assertEqual(settings["definition"], buffered.DEFINITION_ID)
        job, = settings["jobs"]
        self.assertEqual((job["dataset"], job["output"]), ("ogbn-arxiv", "jobs/ogbn-arxiv_arxiv"))
        self.assertEqual(plan["variants"], [])
        self.assertEqual(sweeps.run_plan(path, data_root=self.tmp, device="cpu", dry_run=True), [])

    def test_buffer_plan_refusals(self):
        refused = {
            "independent graphs": dict(parents=[PEPTIDES]),               # not a buffered-core dataset
            "registered buffer off the grid": dict(parents=[ARXIV], buffers=[4, 8]),
            "one run per dataset": dict(parents=[ARXIV, ARXIV]),
        }
        for name, case in refused.items():
            with self.subTest(name), self.assertRaises(sweeps.SweepError):
                sweeps.plan_sweep("buffer_sensitivity", case["parents"], self.tmp / name.replace(" ", "_"),
                                  buffers=case.get("buffers"))
        with self.assertRaises(sweeps.SweepError):                       # a grid belongs to the buffer sweep only
            sweeps.plan_sweep("sensitivity_grids", [ARXIV], self.tmp / "grids", variants=False, buffers=[2, 4])

    def test_sensitivity_variants(self):
        path = sweeps.plan_sweep("sensitivity_grids", [ARXIV], self.tmp / "grids", allow_source_change=True)
        variants = json.loads(path.read_text())["variants"]
        # registered p_min 0.05 is on its grid; n_min 2 and omega 1.0 of the smoke run are not; one replicate
        self.assertEqual(len(variants), 3 + 4 + 4 + 1)
        self.assertEqual({tuple(v["patch"]) for v in variants},
                         {("/support/p_min",), ("/support/n_min",), ("/support/max_normalized_weight",), ()})


class Running(Workspace):
    def test_runner_executes_stages_in_order_and_resumes(self):
        path = sweeps.plan_sweep("horizon_beam_sweep", [ARXIV], self.tmp / "hb", horizon=2, horizons=[1, 2],
                                 beams=[1, 2], profile="full", ensemble=2, allow_source_change=True)
        calls = []

        def runner(command, **kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(command, 0)
        output = self.tmp / "hb" / "runs" / "H2_B1"
        write(output, {"stages/prepare.json": {"status": "COMPLETE"}, "stages/train.json": {"status": "COMPLETE"}})
        commands = sweeps.run_plan(path, data_root=self.tmp / "data", device="cpu", threads=1, python="py",
                                   runner=runner, only=["H2_B1"])
        self.assertEqual(calls, commands)
        self.assertEqual([c[c.index("--stage") + 1] for c in commands], ["explore", "fit", "select"])
        self.assertTrue((self.tmp / "hb" / "logs" / "H2_B1.log").is_file())

        def failing(command, **kwargs):
            return subprocess.CompletedProcess(command, 3)
        with self.assertRaises(sweeps.SweepError):
            sweeps.run_plan(path, data_root=self.tmp / "data", device="cpu", python="py", runner=failing)

    def test_changed_variant_configuration_refused(self):
        path = sweeps.plan_sweep("horizon_beam_sweep", [ARXIV], self.tmp / "hb", horizon=2, horizons=[1, 2],
                                 beams=[1, 2], profile="full", ensemble=2, allow_source_change=True)
        config = self.tmp / "hb" / "variants" / "H2_B1" / "config.json"
        value = json.loads(config.read_text())
        value["controller"]["top_k"] = 3
        config.write_text(json.dumps(value))
        with self.assertRaises(sweeps.SweepError):
            sweeps.run_plan(path, data_root=self.tmp, device="cpu", dry_run=True)

    def test_collect_needs_the_variant_runs(self):
        path = sweeps.plan_sweep("horizon_beam_sweep", [ARXIV], self.tmp / "hb", horizon=2, horizons=[1, 2],
                                 beams=[1, 2], profile="full", ensemble=2, allow_source_change=True)
        with self.assertRaises(sweeps.SweepError) as caught:
            sweeps.collect(path)
        self.assertIn("has not been run", str(caught.exception))


class Collecting(Workspace):
    def test_horizon_rows_from_parent(self):
        path = sweeps.plan_sweep("horizon_beam_sweep", [ARXIV], self.tmp / "hb", horizon=2, horizons=[1, 2],
                                 beams=[2], profile="full", ensemble=2)
        results = json.loads(sweeps.collect(path).read_text())
        candidates = json.loads((ARXIV / "select/candidates.json").read_text())
        decision = json.loads((ARXIV / "select/decision.json").read_text())
        index = next(j for j, c in enumerate(candidates) if c["route"] == "mpc_rh" and c["profile"] == "full"
                     and c["ensemble"] == 2 and c["horizon"] == 2)
        values = {(r["keys"]["horizon"], r["quantity"]): r["value"] for r in results["rows"]}
        self.assertEqual(values[("2", "selection_effect_nats")], decision["mean"][index])
        self.assertEqual(values[("2", "expected_serial_cost")], candidates[index]["cost"])
        self.assertNotIn(("2", "realized_fraction"), values)              # headroom bound 0: no fraction
        computed = json.loads((self.tmp / "hb" / results["rows"][0]["source_file"]).read_text())
        files = {item["file"] for item in computed["inputs"]}
        self.assertTrue({"select/decision.json", "select/candidates.json", "select/entry_complete.json"} <= files)
        self.assertTrue(results["synthetic"])

    def test_sensitivity_rows_are_exact_at_registered_settings(self):
        path = sweeps.plan_sweep("sensitivity_grids", [ARXIV, PEPTIDES], self.tmp / "grids", variants=False)
        results = json.loads(sweeps.collect(path).read_text())
        rows = {(r["keys"]["parameter"], r["keys"]["setting"], r["keys"]["dataset"], r["quantity"]): r["value"]
                for r in results["rows"]}
        sealed = json.loads((PEPTIDES / "admit/decision.json").read_text())
        self.assertEqual(rows[("clip", "0.1", "Peptides-func", "admission_upper_nats")],
                         sealed["empirical_bernstein"]["upper"])
        self.assertEqual(rows[("epsilon_r", "0.0015", "Peptides-func", "route_changed")], 0.0)   # BCE grid
        self.assertNotIn(("epsilon_r", "0.003", "Peptides-func", "route_changed"), rows)
        self.assertEqual(rows[("p_min", "0.05", "ogbn-arxiv", "route_changed")], 0.0)
        self.assertNotIn(("p_min", "0.1", "ogbn-arxiv", "route_changed"), rows)                # needs a variant run
        support = json.loads((ARXIV / "fit/full/audit.json").read_text())["support"]
        self.assertEqual(rows[("p_min", "0.05", "ogbn-arxiv", "qualified_fraction")], support["qualified_fraction"])

    def stand_in(self, base: Path, variant: dict) -> None:
        """A copy of the sealed parent at the variant's location, carrying the variant configuration."""
        run = base / variant["output"]
        shutil.copytree(ARXIV, run, ignore=shutil.ignore_patterns("member_predictions", "predictions", "explore",
                                                                  "*.pt"))
        shutil.copyfile(base / variant["config"], run / "config.json")

    def test_variant_rows_and_replicate(self):
        path = sweeps.plan_sweep("horizon_beam_sweep", [ARXIV], self.tmp / "hb", horizon=2, horizons=[2],
                                 beams=[1, 2], profile="full", ensemble=2, allow_source_change=True)
        plan = json.loads(path.read_text())
        for variant in plan["variants"]:
            self.stand_in(self.tmp / "hb", variant)
        results = json.loads(sweeps.collect(path).read_text())
        values = {(r["keys"]["beam_width"], r["quantity"]): r["value"] for r in results["rows"]}
        self.assertEqual(values[("1", "selection_effect_nats")], values[("2", "selection_effect_nats")])
        computed = json.loads((self.tmp / "hb" / "computed/horizon_beam_sweep/ogbn-arxiv_H2_B1.json").read_text())
        self.assertEqual(computed["detail"]["source"], "variant")
        self.assertEqual(computed["detail"]["beam_width_executed"], 1)
        replicate, = results["replicates"]
        self.assertEqual(replicate["selection_effect_nats"], replicate["parent_selection_effect_nats"])

    def test_support_variant_routes(self):
        path = sweeps.plan_sweep("sensitivity_grids", [ARXIV], self.tmp / "grids", allow_source_change=True,
                                 grids={"p_min": [0.05, 0.1], "n_min": [2], "max_normalized_weight": [1.0],
                                        "epsilon_r": [0.003], "epsilon_eq": [0.002], "clip": [0.1]})
        plan = json.loads(path.read_text())
        self.assertEqual(sorted(v["id"] for v in plan["variants"]),
                         ["ogbn-arxiv_arxiv_p_min_0.1", "ogbn-arxiv_arxiv_replicate"])
        for variant in plan["variants"]:
            self.stand_in(self.tmp / "grids", variant)
        results = json.loads(sweeps.collect(path).read_text())
        rows = {(r["keys"]["parameter"], r["keys"]["setting"], r["quantity"]): r["value"] for r in results["rows"]}
        self.assertEqual(rows[("p_min", "0.1", "route_changed")], 0.0)
        self.assertEqual(rows[("p_min", "0.1", "eligible_count")], 0.0)
        self.assertEqual(results["replicates"][0]["route_changed"], 0.0)
        computed = json.loads((self.tmp / "grids/computed/sensitivity_grids/ogbn-arxiv_p_min_0.1.json").read_text())
        runs = {Path(item["run"]).name for item in computed["inputs"]}
        self.assertEqual(runs, {"arxiv", "ogbn-arxiv_arxiv_p_min_0.1"})

    def module_fixture(self) -> tuple[Path, Path]:
        candidates = [
            {"route": "nominal", "profile": "none", "ensemble": 2, "cost": 2.0, "complexity": 0, "hash": "n"},
            {"route": "one_step", "profile": "full", "ensemble": 2, "cost": 4.0, "complexity": 6, "hash": "o"},
            {"route": "mpc_rh", "profile": "full", "ensemble": 2, "horizon": 2, "cost": 6.0, "complexity": 8, "hash": "f"},
            {"route": "mpc_rh", "profile": "lean", "ensemble": 2, "horizon": 2, "cost": 5.0, "complexity": 8, "hash": "l"},
            {"route": "radius", "profile": "full", "ensemble": 2, "cost": 5.5, "complexity": 7, "hash": "r"}]
        stages = {f"stages/{s}.json": {"status": "COMPLETE"} for s in ("select", "confirm", "admit")}
        stages["run_state.json"] = {"status": "COMPLETE", "source_sha256": "0" * 64}
        pascal = write(self.tmp / "pascal", {
            **stages, "config.json": {"dataset": "PascalVOC-SP", "scientific_run": True},
            "select/candidates.json": candidates,
            "select/decision.json": {"mean": [0.0, -0.002, -0.006, -0.005, -0.003], "provisional_index": 2},
            "confirm/decision.json": {"provisional_index": 2, "comparator_index": 0, "contrasts": [
                {"kind": "state", "candidate_index": 2, "mean": 0.004},
                {"kind": "full", "candidate_index": 2, "predecessor_index": 3, "mean": 0.08, "available": True}]},
            "policy_lock.json": {"deployed_index": 2}})
        peptides = write(self.tmp / "peptides", {
            **stages, "config.json": {"dataset": "Peptides-func", "scientific_run": True},
            "select/candidates.json": candidates,
            "select/decision.json": {"mean": [0.0, -0.001, -0.002, -0.002, -0.0035], "provisional_index": 4},
            "confirm/decision.json": {"provisional_index": 4, "comparator_index": 0, "contrasts": [
                {"kind": "radius", "candidate_index": 4, "predecessor_index": 1, "mean": -0.0025}]},
            "policy_lock.json": {"deployed_index": 4}})
        return pascal, peptides

    def test_module_ablations(self):
        pascal, peptides = self.module_fixture()
        path = sweeps.plan_sweep("module_ablations", [pascal, peptides], self.tmp / "modules")
        results = json.loads(sweeps.collect(path).read_text())
        values = {(r["keys"]["module"], r["keys"]["dataset"], r["quantity"]): r["value"] for r in results["rows"]}
        self.assertAlmostEqual(values[("sequence_risk", "PascalVOC-SP", "loss_change_nats")], 0.004)    # -0.002 - -0.006
        self.assertAlmostEqual(values[("full_lean_profile", "PascalVOC-SP", "retained_effect_fraction")], 0.92)
        self.assertAlmostEqual(values[("full_lean_profile", "PascalVOC-SP", "cost_fraction")], 0.75)     # (5-2)/(6-2)
        self.assertAlmostEqual(values[("radius_uncertainty", "Peptides-func", "loss_change_nats")], 0.0025)
        self.assertEqual(len(values), 4)
        lifecycle = {"spectral_bands", "source_alignment", "confidence_sensors", "transport_history", "observer",
                     "recovery_type", "comparator_branch"}       # scripts/run_ablation.py modules (--ablations)
        self.assertEqual({m["module"] for m in results["not_measured"]}, lifecycle)
        self.assertLessEqual({m["module"] for m in results["unsupported"]}, lifecycle)
        self.assertFalse(results["synthetic"])

    def test_example_runs_hold_no_module_contrast(self):
        path = sweeps.plan_sweep("module_ablations", [ARXIV, PEPTIDES], self.tmp / "modules")
        self.assertEqual(json.loads(sweeps.collect(path).read_text())["rows"], [])


class Export(Workspace):
    def test_record_set_round_trip(self):
        path = sweeps.plan_sweep("sensitivity_grids", [ARXIV, PEPTIDES], self.tmp / "grids", variants=False)
        results = sweeps.collect(path)
        root = self.tmp / "records"
        root.mkdir()
        target = export_sweep_records.export(results, root=root, hardware="CPU", period="2026-11-07/2026-11-07",
                                             replace=False, argv=["test"])
        specs = export_sweep_records.merged_specs()
        entry = records.verify_set(target, specs["sets"]["sensitivity_grids"], specs)
        rows = json.loads(results.read_text())["rows"]
        self.assertEqual(entry["rows"], len(rows))
        self.assertTrue(entry["synthetic"])
        with (target / "summary.csv").open(newline="") as handle:
            header = next(csv.reader(handle))
        self.assertEqual(header, ["parameter", "setting", "dataset", "quantity", "value", "unit", "source_file",
                                  "locator"])
        index = json.loads((root / "index.json").read_text())
        self.assertEqual([e["set"] for e in index["sets"]], ["sensitivity_grids"])
        with self.assertRaises(export_sweep_records.ExportError):    # existing set needs --replace
            export_sweep_records.export(results, root=root, hardware="CPU", period="p", replace=False, argv=[])
        with self.assertRaises(export_sweep_records.ExportError):    # synthetic sets never enter records/
            export_sweep_records.export(results, root=records.RECORDS, hardware="CPU", period="p", replace=True,
                                        argv=[])

    def test_altered_value_fails_verification(self):
        path = sweeps.plan_sweep("sensitivity_grids", [ARXIV], self.tmp / "grids", variants=False)
        results = sweeps.collect(path)
        data = json.loads(results.read_text())
        data["rows"][0]["value"] += 0.5
        results.write_text(json.dumps(data))
        root = self.tmp / "records"
        root.mkdir()
        with self.assertRaises(export_sweep_records.ExportError) as caught:
            export_sweep_records.export(results, root=root, hardware="CPU", period="p", replace=False, argv=[])
        self.assertIn("differs from its computed file", str(caught.exception))
        self.assertFalse((root / "sensitivity_grids").exists())


# ----------------------------------------------------------------------------- buffer sweep (Table S27)
def geometric_graph(n: int, degree: float, rng, rewire: float = 0.0):
    """Random geometric graph in the unit square; ``rewire`` moves that share of edge ends anywhere."""
    points = rng.random((n, 2))
    pairs = cKDTree(points).query_pairs(np.sqrt(degree / (np.pi * n)), output_type="ndarray")
    src, dst = pairs[:, 0], pairs[:, 1].copy()
    if rewire:
        moved = rng.random(len(src)) < rewire
        dst[moved] = rng.integers(0, n, int(moved.sum()))
    return np.stack([np.r_[src, dst], np.r_[dst, src]]), points


def bisection(points, parts: int) -> np.ndarray:
    """Recursive coordinate bisection: balanced, compact parts (what METIS returns on a geometric graph)."""
    labels = np.zeros(len(points), dtype=np.int64)

    def split(index, count, first):
        if count == 1:
            labels[index] = first
            return
        extent = points[index].max(0) - points[index].min(0)
        order = index[np.argsort(points[index, int(np.argmax(extent))], kind="stable")]
        split(order[:len(order) // 2], count // 2, first)
        split(order[len(order) // 2:], count // 2, first + count // 2)
    split(np.arange(len(points)), parts, 0)
    return labels


def torus_sbm(n: int, parts: int, rng, rows: int = 4):
    """Stochastic-block graph whose blocks sit on a rows x cols torus (dense to grid neighbours, sparse beyond)."""
    cols = parts // rows
    labels = rng.permutation(np.arange(n) % parts)
    members = [np.flatnonzero(labels == b) for b in range(parts)]
    size, src, dst = n / parts, [], []
    for a, b in itertools.combinations_with_replacement(range(parts), 2):
        gap = abs(a // cols - b // cols) + min(abs(a % cols - b % cols), cols - abs(a % cols - b % cols))
        mean = 10.0 * size / 2 if a == b else (3.0 * size / 4 if gap == 1 else 0.5 * size / (parts - 5))
        m = rng.poisson(mean)
        src.append(rng.choice(members[a], m))
        dst.append(rng.choice(members[b], m))
    src, dst = np.concatenate(src), np.concatenate(dst)
    return np.stack([np.r_[src, dst], np.r_[dst, src]]), labels


def hop_coverage(adjacency, clusters, scored, hops: int) -> np.ndarray:
    """cov[a, b]: share of the scored nodes of core b inside the ``hops``-hop context of core a (data._buffers)."""
    cores, n = int(clusters.max()) + 1, adjacency.shape[0]
    coverage = np.zeros((cores, cores))
    for a in range(cores):
        visited = clusters == a
        frontier = np.flatnonzero(visited)
        for _ in range(hops):
            reached = np.zeros(n, dtype=bool)
            reached[adjacency[frontier].indices] = True
            frontier = np.flatnonzero(reached & ~visited)
            visited[frontier] = True
        for b in range(cores):
            coverage[a, b] = visited[scored[clusters[scored] == b]].mean()
    return coverage


def unit_counts(adjacency, clusters, scored, grid=BUFFERS):
    """Resampling units of the module's rule and of the connected-components rule on the same links."""
    core_ids = np.arange(int(clusters.max()) + 1)
    sources = [scored[clusters[scored] == c] for c in core_ids]
    entry = buffered.walk_entry(adjacency, clusters, core_ids, sources, grid)
    strength = {k: buffered.dependence(entry[k]) for k in grid}
    tau = strength[2].max()
    ours = [len(buffered.resampling_blocks(strength[k], tau)["blocks"]) for k in grid]
    chained = [connected_components(sparse.csr_matrix(strength[k] > tau), directed=False)[0] for k in grid]
    return ours, chained, entry


def buffer_run(root: Path, *, dataset: str = "ogbn-arxiv", seed: int = 0, n: int = 3000, effect: float = -0.006,
               reps: int = 200) -> Path:
    """A sealed single-graph run holding exactly the files the buffer sweep reads.

    Geometric graph with 32 compact cores; per-core paired losses with a spatially smooth component and one
    core beyond the clip; admit/decision.json computed by cross_morse.statistics.admission with the admit
    stage's arguments (seed + 6, counts as weights, accuracy differences for the metric gate).
    """
    from cross_morse.statistics import admission
    rng = np.random.default_rng(seed)
    edges, points = geometric_graph(n, 12.0, rng)
    adjacency = buffered.symmetric_adjacency(edges, n)
    edge_index = np.stack([np.repeat(np.arange(n), np.diff(adjacency.indptr)), adjacency.indices.astype(np.int64)])
    clusters = bisection(points, 32)
    admit = np.flatnonzero(rng.random(n) < 0.18)
    counts = np.bincount(clusters[admit], minlength=32)
    assert (counts > 0).all()
    centres = np.stack([points[clusters == c].mean(0) for c in range(32)])
    differences = effect + 0.004 * np.sin(2 * np.pi * centres[:, 0]) + 0.002 * rng.standard_normal(32)
    differences[5] = -0.3                                                  # clipped to -0.1 per core
    comparator = 0.9 + 0.05 * rng.standard_normal(32)
    accuracy = 0.7 + 0.02 * rng.standard_normal(32)
    config = {"dataset": "synthetic", "benchmark_dataset": dataset, "scientific_run": False, "seed": 11 + seed,
              "data": {"buffer_hops": 2, "num_clusters": 32, "partitioner": "balanced"},
              "admission": {"bootstrap_reps": reps, "alpha": 0.025, "epsilon_a": 0.003, "epsilon_metric": 0.001,
                            "clip": 0.1, "rule": "cluster_bootstrap"}}
    ac = config["admission"]
    candidate = comparator + differences
    lifted = accuracy + 0.002 + 0.001 * rng.standard_normal(32)
    decision = admission(candidate - comparator, metric_differences=lifted - accuracy, weights=counts, reps=reps,
                         alpha=ac["alpha"], epsilon_a=ac["epsilon_a"], epsilon_metric=ac["epsilon_metric"],
                         clip=ac["clip"], seed=config["seed"] + 6)
    decision.update(comparator_index=0, confirmed_index=1, deployed_index=1 if decision["passed"] else 0,
                    dataset="synthetic", official_validation_only=True)
    arrays = {"prepared/graph.npz": {"edge_index": edge_index, "x": points.astype(np.float32)},
              "prepared/clusters.npz": {"cluster_ids": clusters}, "prepared/splits.npz": {"admit": admit},
              "admit/paired_units.npz": {"candidate_loss": candidate, "comparator_loss": comparator,
                                         "core_ids": np.arange(32), "counts": counts,
                                         "candidate_accuracy": lifted, "comparator_accuracy": accuracy}}
    for relative, content in arrays.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        np.savez(root / relative, **content)
    metadata = {"num_nodes": n, "preparation_config": {"data": {"buffer_hops": 2}},
                "artifact_sha256": {name: sweeps.sha256_file(root / "prepared" / name)
                                    for name in ("graph.npz", "clusters.npz", "splits.npz")},
                "buffer_audit": {"buffer_hops": 2, "buffer_role": "context_diagnostic_only",
                                 "core_plus_halo_sizes": buffered.context_sizes(adjacency, clusters, 2)}}
    return write(root, {"config.json": config, "run_state.json": {"status": "COMPLETE", "source_sha256": "0" * 64},
                        "stages/admit.json": {"status": "COMPLETE"}, "admit/decision.json": decision,
                        "prepared/metadata.json": metadata})


class BufferUnits(unittest.TestCase):
    def test_independent_sets_against_enumeration(self):
        rng = np.random.default_rng(3)
        for trial in range(60):
            size = int(rng.integers(1, 11))
            upper = np.triu(rng.random((size, size)) < rng.uniform(0.05, 0.8), 1)
            links = upper | upper.T
            independent = [s for r in range(size, 0, -1) for s in itertools.combinations(range(size), r)
                           if not links[np.ix_(s, s)].any()]
            alpha = len(independent[0])
            chosen = buffered.maximum_independent_set(links)
            self.assertEqual(buffered.independence_number(links), alpha)
            self.assertEqual(tuple(chosen), min(s for s in independent if len(s) == alpha))   # lexicographically first

    def test_walk_entry_by_hand(self):
        # path 0-1-2-3-4-5 and an isolated node 6; core 0 = {0, 1, 2} scored at 2, core 1 = {3, 4, 5, 6} scored at 3, 6
        edges = np.array([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5], [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]])
        adjacency = buffered.symmetric_adjacency(edges, 7)
        entry = buffered.walk_entry(adjacency, np.array([0, 0, 0, 1, 1, 1, 1]), [0, 1],
                                    [np.array([2]), np.array([3, 6])], [0, 1, 2, 3])
        self.assertEqual(entry[0][1, 0], 0.0)
        self.assertEqual((entry[1][1, 0], entry[1][0, 1]), (0.5, 0.25))        # 2 -> 3 with 1/2; node 6 never moves
        self.assertEqual((entry[2][1, 0], entry[2][0, 1]), (0.5, 0.25))        # 2 -> 1 -> {0, 2} stays in core 0
        self.assertEqual((entry[3][1, 0], entry[3][0, 1]), (0.625, 0.3125))    # + 2 -> 1 -> 2 -> 3 with 1/8
        self.assertEqual(buffered.context_sizes(adjacency, np.array([0, 0, 0, 1, 1, 1, 1]), 1), [3 + 1, 4 + 1])

    def test_blocks_at_the_registered_buffer_and_above(self):
        strength = buffered.dependence(np.array([[1.0, 0.30, 0.05, 0.0], [0.10, 1.0, 0.35, 0.02],
                                                 [0.04, 0.20, 1.0, 0.25], [0.0, 0.01, 0.10, 1.0]]))
        tau = strength.max()
        self.assertEqual(buffered.resampling_blocks(strength, tau)["blocks"], [[0], [1], [2], [3]])
        grown = strength * 1.2                                             # links 0-1, 1-2; 2-3 stays at 0.30
        found = buffered.resampling_blocks(grown, tau)
        self.assertEqual((found["links"], found["roots"]), (2, [0, 2, 3]))   # alpha = 3: cores 0, 2, 3 independent
        self.assertEqual(found["blocks"], [[0], [1, 2], [3]])              # core 1 joins the root it depends on most

    def test_walk_dependence_refines_k_hop_coverage(self):
        rng = np.random.default_rng(0)
        edges, points = geometric_graph(3000, 12.0, rng)
        adjacency, clusters = buffered.symmetric_adjacency(edges, 3000), bisection(points, 32)
        scored = np.flatnonzero(rng.random(3000) < 0.18)
        _, _, entry = unit_counts(adjacency, clusters, scored)
        for k in (2, 4):
            coverage = hop_coverage(adjacency, clusters, scored, k)
            self.assertTrue((entry[k] <= coverage + 1e-12).all())          # a k-step walk stays in the k-hop context
            self.assertTrue(((entry[k] > 0) == (coverage > 0)).all())

    def test_units_fall_gradually_on_metis_like_partitions(self):
        cases = {"random geometric graph": lambda rng: geometric_graph(3000, 12.0, rng),
                 "small-world geometric graph": lambda rng: geometric_graph(3000, 12.0, rng, rewire=0.02),
                 "torus stochastic-block graph": lambda rng: torus_sbm(3000, 32, rng)}
        for name, make in cases.items():
            with self.subTest(name):
                rng = np.random.default_rng(0)
                edges, extra = make(rng)
                clusters = extra if extra.ndim == 1 else bisection(extra, 32)
                adjacency = buffered.symmetric_adjacency(edges, 3000)
                scored = np.flatnonzero(rng.random(3000) < 0.18)
                ours, chained, _ = unit_counts(adjacency, clusters, scored)
                self.assertEqual(ours[0], 32)
                self.assertTrue(all(b <= a for a, b in zip(ours, ours[1:])), ours)
                self.assertGreaterEqual(ours[1], 16, ours)                 # no collapse at 4 hops
                self.assertTrue(4 <= ours[-1] < 32, ours)                  # the buffer matters, gradually
                self.assertEqual(chained[-1], 1, chained)                  # components of the same links chain into one
                coverage = {k: hop_coverage(adjacency, clusters, scored, k) for k in (2, 4)}
                pairs = ~np.eye(32, dtype=bool)
                links = np.maximum(coverage[4], coverage[4].T) > np.maximum(coverage[2], coverage[2].T)[pairs].max()
                hop_blocks = connected_components(sparse.csr_matrix(links), directed=False)[0]
                if name == "small-world geometric graph":
                    self.assertEqual(hop_blocks, 1)                        # k-hop coverage rule: one block at 4 hops
                if name == "torus stochastic-block graph":
                    self.assertEqual(coverage[2][pairs].max(), 1.0)        # ... or never moves (tau = 1)
                    self.assertEqual(hop_blocks, 32)


class BufferAdmission(Workspace):
    def sweep(self, *parents, buffers=None):
        path = sweeps.plan_sweep("buffer_sensitivity", list(parents), self.tmp / "sweep", buffers=buffers)
        return path

    def test_registered_row_reproduces_and_effect_is_constant(self):
        run = buffer_run(self.tmp / "arxiv_run")
        path = self.sweep(run)
        written = buffered.run_buffer_jobs(path)
        job = json.loads(written[0].read_text())
        sealed = json.loads((run / "admit/decision.json").read_text())
        rows = {row["buffer_hops"]: row for row in job["buffers"]}
        self.assertTrue(job["registered"]["reproduced"])
        self.assertEqual(rows[2]["clipped_effect_nats"], sealed["loss"]["mean"])          # exact
        self.assertEqual(rows[2]["upper_nats"], sealed["loss"]["upper"])
        self.assertEqual(rows[2]["admitted"], float(sealed["mean_gate"] and sealed["upper_gate"]))
        self.assertEqual(rows[2]["resampling_units"], sealed["loss"]["units"])
        self.assertEqual(rows[2]["blocks"], [[c] for c in range(32)])
        self.assertEqual({row["clipped_effect_nats"] for row in rows.values()}, {sealed["loss"]["mean"]})
        units = [rows[k]["resampling_units"] for k in BUFFERS]
        self.assertTrue(all(b <= a for a, b in zip(units, units[1:])) and units[-1] < 32, units)
        clusters = np.load(run / "prepared/clusters.npz")["cluster_ids"]
        weights = np.bincount(clusters[np.load(run / "prepared/splits.npz")["admit"]])
        for row in rows.values():
            self.assertEqual(sorted(c for block in row["blocks"] for c in block), list(range(32)))
            self.assertAlmostEqual(row["half_width_nats"], row["upper_nats"] - row["clipped_effect_nats"], places=15)
        self.assertEqual(job["graph"]["scored_counts"], weights.tolist())
        audit = json.loads((run / "prepared/metadata.json").read_text())["buffer_audit"]
        self.assertEqual(job["graph"]["registered_context_sizes"], audit["core_plus_halo_sizes"])
        # rerunning keeps the completed job; collecting writes one row file per buffer
        self.assertEqual(buffered.run_buffer_jobs(path), written)
        results = json.loads(sweeps.collect(path).read_text())
        values = {(r["keys"]["buffer_hops"], r["quantity"]): r["value"] for r in results["rows"]}
        self.assertEqual(values[("2", "upper_nats")], sealed["loss"]["upper"])
        self.assertEqual(values[("32", "resampling_units")], float(units[-1]))
        deposited = {d["path"] for d in results["deposits"]}
        self.assertEqual(deposited, {f"jobs/ogbn-arxiv_arxiv_run/{name}"
                                     for name in ("buffered_admission.json", "dependence.npz")})

    def test_example_run_reproduces_its_sealed_admission(self):
        path = self.sweep(ARXIV)
        job = json.loads(buffered.run_buffer_jobs(path)[0].read_text())
        self.assertTrue(job["registered"]["reproduced"])
        self.assertTrue(job["graph"]["registered_context_sizes_match_buffer_audit"])   # graph prepared by the runtime
        units = [row["resampling_units"] for row in job["buffers"]]
        self.assertEqual(units[0], 8)
        self.assertTrue(all(b <= a for a, b in zip(units, units[1:])), units)

    def test_unreproduced_registered_row_is_refused(self):
        run = buffer_run(self.tmp / "arxiv_run")
        path = self.sweep(run)
        decision = json.loads((run / "admit/decision.json").read_text())
        decision["loss"]["upper"] += 1e-9
        (run / "admit/decision.json").write_text(json.dumps(decision))
        with self.assertRaises(sweeps.SweepError) as caught:
            buffered.run_buffer_jobs(path)
        self.assertIn("does not reproduce", str(caught.exception))
        self.assertFalse((self.tmp / "sweep/jobs/ogbn-arxiv_arxiv_run/buffered_admission.json").exists())
        with self.assertRaises(sweeps.SweepError):                            # planning refuses it as well
            sweeps.plan_sweep("buffer_sensitivity", [run], self.tmp / "again")

    def test_foreign_graph_and_changed_parent_are_refused(self):
        run = buffer_run(self.tmp / "arxiv_run")
        path = self.sweep(run)
        metadata = json.loads((run / "prepared/metadata.json").read_text())
        original = json.dumps(metadata)
        metadata["buffer_audit"]["core_plus_halo_sizes"][0] += 1
        (run / "prepared/metadata.json").write_text(json.dumps(metadata))
        with self.assertRaises(sweeps.SweepError) as caught:
            buffered.run_buffer_jobs(path)
        self.assertIn("buffer audit", str(caught.exception))
        (run / "prepared/metadata.json").write_text(original)
        buffered.run_buffer_jobs(path)
        units = dict(np.load(run / "admit/paired_units.npz"))
        np.savez(run / "admit/paired_units.npz", **units, note=np.zeros(1))   # same losses, different file
        with self.assertRaises(sweeps.SweepError) as caught:
            sweeps.collect(path)
        self.assertIn("changed after the job ran", str(caught.exception))

    def test_export_and_table_s27(self):
        from paper.build.displays.testing import build_one, context, read_table
        from paper.build.displays.framework import RecordSetView
        arxiv = buffer_run(self.tmp / "arxiv_run", dataset="ogbn-arxiv", seed=0)
        proteins = buffer_run(self.tmp / "proteins_run", dataset="ogbn-proteins", seed=1, effect=-0.005)
        path = self.sweep(arxiv, proteins)
        sweeps.run_plan(path, data_root=self.tmp, device="cpu")
        results = sweeps.collect(path)
        root = self.tmp / "records"
        root.mkdir()
        target = export_sweep_records.export(results, root=root, hardware="CPU", period="2026-11-07/2026-11-07",
                                             replace=False, argv=["test"])
        specs = export_sweep_records.merged_specs()
        spec = specs["sets"]["buffer_sensitivity"]
        entry = records.verify_set(target, spec, specs)
        self.assertEqual(entry["rows"], 2 * 5 * 5)
        self.assertTrue((target / "raw/jobs/ogbn-proteins_proteins_run/dependence.npz").is_file())
        source = json.loads((target / "code/SOURCE.json").read_text())
        self.assertIn("cm_experiments/buffered_admission.py", source["experiment_code"])
        self.assertNotIn("cm_experiments/ablation_lifecycle.py", source["experiment_code"])
        rows = records.read_summary(target, spec, specs["summary_columns"])
        for row in rows:
            row["value"] = float(row["value"])
        view = RecordSetView("buffer_sensitivity", target, spec, rows, synthetic=True)
        build_one("supplementary:Table S27", context(records={"buffer_sensitivity": view}), self.tmp / "out")
        table = read_table(self.tmp / "out", "S27")
        self.assertEqual([r["buffer (hops)"] for r in table], [str(k) for k in BUFFERS] * 2)
        for run, rows_of_run in ((arxiv, table[:5]), (proteins, table[5:])):
            sealed = json.loads((run / "admit/decision.json").read_text())
            first = rows_of_run[0]
            self.assertEqual(first["decision at eps_a"] == "pass", sealed["mean_gate"] and sealed["upper_gate"])
            self.assertEqual(float(first["U_desc"]), round(sealed["loss"]["upper"], 4))
            self.assertEqual({float(r["D_bar"]) for r in rows_of_run}, {round(sealed["loss"]["mean"], 4)})


@unittest.skipUnless(HAS_TORCH, "the released lifecycle needs PyTorch")
class BufferLifecycleSmoke(Workspace):
    """Freshly sealed smoke lifecycles (current runtime source): linear-metric arxiv and ROC-AUC proteins."""

    def test_buffer_sweep_on_sealed_smoke_runs(self):
        import torch
        from cross_morse.pipeline import Pipeline
        torch.set_num_threads(2)
        parents = []
        for stem in ("arxiv_smoke", "proteins_smoke"):
            config = json.loads((ROOT / "configs" / f"{stem}.json").read_text())
            config["training"].update(epochs=1, patience=1)
            config["controller"].update(epochs=1, patience=1)
            config["selection"]["bootstrap_reps"] = 20
            config["admission"]["bootstrap_reps"] = 20
            run = self.tmp / "runs" / stem
            with contextlib.redirect_stdout(io.StringIO()):
                Pipeline(config, self.tmp / "data", run, "cpu").run()
            parents.append(run)
        path = sweeps.plan_sweep("buffer_sensitivity", parents, self.tmp / "buffer")
        sweeps.run_plan(path, data_root=self.tmp / "data", device="cpu")
        for parent in parents:
            found = next((self.tmp / "buffer/jobs").glob(f"*_{parent.name}/buffered_admission.json"))
            job = json.loads(found.read_text())
            sealed = json.loads((parent / "admit/decision.json").read_text())
            self.assertTrue(job["registered"]["reproduced"])
            self.assertEqual(job["buffers"][0]["upper_nats"], sealed["loss"]["upper"])
            units = [row["resampling_units"] for row in job["buffers"]]
            self.assertTrue(all(b <= a for a, b in zip(units, units[1:])), units)
            self.assertEqual({row["clipped_effect_nats"] for row in job["buffers"]}, {sealed["loss"]["mean"]})
        root = self.tmp / "records"
        root.mkdir()
        target = export_sweep_records.export(sweeps.collect(path), root=root, hardware="CPU", period="p",
                                             replace=False, argv=["test"])
        specs = export_sweep_records.merged_specs()
        records.verify_set(target, specs["sets"]["buffer_sensitivity"], specs)


@unittest.skipUnless(HAS_TORCH, "the released lifecycle needs PyTorch")
class LifecycleSmoke(Workspace):
    def test_beam_variant_through_select(self):
        path = sweeps.plan_sweep("horizon_beam_sweep", [ARXIV], self.tmp / "hb", horizon=2, horizons=[2],
                                 beams=[1, 2], profile="full", ensemble=2, replicate=False, allow_source_change=True)
        sweeps.run_plan(path, data_root=self.tmp / "data", device="cpu", threads=1)
        run = self.tmp / "hb" / "runs" / "H2_B1"
        self.assertEqual(json.loads((run / "stages/select.json").read_text())["status"], "COMPLETE")
        self.assertFalse((run / "stages/confirm.json").exists())          # test labels stay closed
        results = json.loads(sweeps.collect(path).read_text())
        beams = {r["keys"]["beam_width"] for r in results["rows"]}
        self.assertEqual(beams, {"1", "2"})


if __name__ == "__main__":
    unittest.main()
