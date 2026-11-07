"""Sweeps and module ablations of cm_experiments/sweeps.py and their record export.

The recomputations from sealed outputs are checked on the two bundled example
runs (they must reproduce each sealed decision at the registered setting) and on
small hand-built run directories. The lifecycle runner is exercised with a fake
process runner; the one test that executes the released lifecycle needs PyTorch.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from cm_experiments import sweeps
from scripts import export_sweep_records, records

ROOT = Path(__file__).resolve().parents[1]
ARXIV = ROOT / "examples/runs/arxiv"
PEPTIDES = ROOT / "examples/runs/peptides_func"
HAS_TORCH = importlib.util.find_spec("torch") is not None


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

    def test_buffer_sweep_refused(self):
        with self.assertRaises(sweeps.SweepError) as caught:
            sweeps.plan_sweep("buffer_sensitivity", [ARXIV], self.tmp / "buffer")
        self.assertIn("context diagnostic", str(caught.exception))
        self.assertFalse((self.tmp / "buffer").exists())

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
