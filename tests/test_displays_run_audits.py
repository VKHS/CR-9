"""Selection, support and audit display generators on synthetic fixtures.

Every run fixture is sealed with the runtime's own statistics (nested selection,
admission bound), so the generators' replays can be checked exactly. Values are
synthetic and exercise the computation and layout, never the paper's results.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cross_morse import statistics
from paper.build.displays import run_audits
from paper.build.displays.framework import (DATASETS, GRAPH_LEVEL, SHORT, DisplayInputError, DisplayValidationError,
                                           RunView, comparison_coverage)
from paper.build.displays.testing import build_one, context, fake_records, fake_run, read_table, write_files

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = json.loads((ROOT / "paper/reported_results.json").read_text(encoding="utf-8"))
SPECS = json.loads((ROOT / "records/specs.json").read_text(encoding="utf-8"))["sets"]
# The measurement-record contracts these generators introduced, registered in records/specs.json.
RECORD_CONTRACTS = {name: SPECS[name] for name in ("action_gap_audit", "tube_audit", "oracle_dispersion",
                                                   "depth_diagnostics")}
EXAMPLE_NODE = ROOT / "examples/runs/arxiv"
EXAMPLE_GRAPH = ROOT / "examples/runs/peptides_func"
# route, complexity, open loop, profile, ensemble, loss offset, normalized cost
REGISTRY_FAMILY = (("nominal", 0, True, "none", 1, 0.10, 1.0), ("best_depth", 1, True, "none", 1, 0.08, 0.5),
                   ("fixed_sequence", 4, True, "none", 1, 0.00, 1.0),
                   ("fixed_sequence", 4, True, "none", 1, 0.012, 1.0),
                   ("mpc_ol", 5, True, "lean", 1, 0.02, 1.2), ("one_step", 6, False, "lean", 1, -0.03, 1.3),
                   ("radius", 7, False, "full", 1, -0.05, 1.4), ("mpc_rh", 8, False, "full", 1, -0.051, 1.6),
                   ("radius", 7, False, "full", 2, -0.0505, 2.8))
UNITS = 8


def fold_plan(n, outer_folds, inner_folds, seed):
    """cross_morse.pipeline.selection_fold_plan (that module imports torch)."""
    order = np.random.default_rng(seed).permutation(n)
    outer = np.empty(n, dtype=np.int64)
    outer[order] = np.arange(n) % outer_folds
    inner = np.full((outer_folds + 1, n), -1, dtype=np.int64)
    for k in range(outer_folds):
        train = order[outer[order] != k]
        inner[k, train] = np.arange(len(train)) % inner_folds
    inner[-1, order] = np.arange(n) % inner_folds
    return outer, inner


def lifecycle_files(dataset, seed, *, entry=None, support=None, stage_seconds=None, admission_differences=None,
                    paired_loss=-0.0045, metric_pp=0.07, violating=(4,)):
    """Every file the run-derived audit displays read, sealed with the runtime statistics."""
    rng = np.random.default_rng(seed)
    candidates = []
    for j, (route, complexity, open_loop, profile, ensemble, _, cost) in enumerate(REGISTRY_FAMILY):
        digest = hashlib.sha256(f"{dataset}-{seed}-{j}".encode()).hexdigest()
        candidates.append({"route": route, "complexity": complexity, "is_open_loop": open_loop, "profile": profile,
                           "ensemble": ensemble, "cost": cost, "hash": digest, "candidate_id": digest[:20],
                           "behavior_id": digest[:16], "feasible": True, "execution_failure": False,
                           "depth": 6, "qualification_cost": 0.0})
    offsets = np.array([row[5] for row in REGISTRY_FAMILY])
    loss = 1.0 + 0.1 * rng.random(UNITS)[:, None] + offsets[None, :] + rng.normal(0.0, 0.002, (UNITS, len(offsets)))
    accuracy = np.full_like(loss, 0.7)
    counts = np.array([5, 7, 6, 8, 5, 6, 7, 6])
    outer, inner = fold_plan(UNITS, 4, 3, seed + 3)
    decision = statistics.nested_selection(loss, accuracy, candidates, outer, inner_fold_ids=inner, reps=40,
                                           alpha=0.025, epsilon_r=0.003, epsilon_eq=0.002, epsilon_metric=0.001,
                                           seed=seed + 4, weights=counts)
    qc, provisional = decision["comparator_index"], decision["provisional_index"]
    differences = (np.asarray(admission_differences, float) if admission_differences is not None
                   else rng.normal(-0.01, 0.004, 12))
    independent = dataset in GRAPH_LEVEL
    admitted = statistics.admission(differences, metric_summary={"valid_interval": True, "lower": 0.0},
                                    weights=np.ones(len(differences)), reps=40, alpha=0.025, epsilon_a=0.003,
                                    epsilon_metric=0.001, clip=0.1, seed=seed + 6, independent_graphs=independent)
    deployed = provisional if admitted["passed"] else qc
    admitted.update(comparator_index=qc, confirmed_index=provisional, deployed_index=deployed,
                    candidate_already_comparator=provisional == qc)
    stage_seconds = stage_seconds or {"prepare": 1.0, "train": 10.0, "explore": 3600.0, "fit": 1800.0,
                                      "select": 7200.0, "confirm": 360.0, "admit": 360.0, "evaluate": 5.0}
    entry = {"delta_plant": -0.018, "upper_plant": -0.012, "pdd": 0.0024, "dil": 0.010, "phi": 0.89,
             "headroom": -0.005, **(entry or {})}
    cells = support or {"0:0": dict(row_count=40, trajectory_count=30, min_propensity=0.28, ess=70.0,
                                    max_normalized_weight=0.05, qualified=True),
                        "0:1": dict(row_count=5, trajectory_count=4, min_propensity=0.09, ess=3.2,
                                    max_normalized_weight=0.4, qualified=False),
                        "0:2": dict(row_count=0, trajectory_count=0, min_propensity=0.0, ess=0.0,
                                    max_normalized_weight=1.0, qualified=False),
                        "1:0": dict(row_count=80, trajectory_count=66, min_propensity=0.27, ess=65.0,
                                    max_normalized_weight=0.06, qualified=True)}
    strata = sorted({int(key.split(":")[0]) for key in cells})
    files = {
        "config.json": {"dataset": dataset, "seed": seed,
                        "selection": {"alpha": 0.025, "epsilon_r": 0.003, "epsilon_eq": 0.002, "epsilon_metric": 0.001,
                                      "max_normalized_work": 18.0, "bootstrap_reps": 40},
                        "admission": {"alpha": 0.025, "epsilon_a": 0.003, "epsilon_metric": 0.001, "clip": 0.1,
                                      "bootstrap_reps": 40},
                        "entry": {"plant_margin": 0.004, "pdd_max": 0.01, "dil_gap_max": 0.02,
                                  "registry_fraction_min": 0.8, "headroom_margin": 0.003},
                        "controller": {"calibration_alpha": 0.05}},
        "environment.json": {"device": "cuda:0", "gpu": "synthetic GPU"},
        "action_registry.json": [{"id": i, "name": "nominal" if i == 0 else f"A{i}"} for i in range(10)],
        "explore/manifest.json": {"trajectories": 120},
        "train/entry_initial.json": {"plant_vs_q0": {"mean": entry["delta_plant"], "upper": entry["upper_plant"]},
                                     "pdd_fraction": entry["pdd"], "dil_gap": entry["dil"], "identity_pass": True,
                                     "depth_grid": [1, 2, 4, 8], "depth_loss": [1.2, 1.0, 0.9, 0.95]},
        "train/manifest.json": {"checkpoints": [{"baseline": True, "seconds_this_attempt": 100.0},
                                                {"baseline": False, "seconds_this_attempt": 7200.0},
                                                {"baseline": False, "seconds_this_attempt": 3600.0}]},
        "select/entry_complete.json": {"registry_support_fraction": entry["phi"], "passed": True,
                                       "headroom": {"upper": entry["headroom"]}},
        "fit/full/audit.json": {"oof_rmse": 0.02, "support": {
            "cells": {key: {"stratum": int(key.split(":")[0]), "action": int(key.split(":")[1]), **value}
                      for key, value in cells.items()},
            "registered_strata": strata,
            "qualified_fraction": sum(c["qualified"] for c in cells.values()) / len(cells),
            "thresholds": {"p_min": 0.05, "n_min": 64.0, "max_normalized_weight": 0.2}}},
        "select/candidates.json": candidates,
        "select/decision.json": decision,
        "select/unit_scores.npz": {"loss": loss, "accuracy": accuracy, "core_ids": np.arange(UNITS),
                                   "counts": counts, "outer_fold": outer, "inner_fold": inner},
        "confirm/decision.json": {"confirmed_index": provisional, "comparator_index": qc,
                                  "provisional_index": provisional},
        "admit/decision.json": admitted,
        "admit/paired_units.npz": {"candidate_loss": 1.0 + differences, "comparator_loss": np.ones(len(differences)),
                                   "core_ids": np.arange(len(differences)), "counts": np.ones(len(differences))},
        "final/results_summary.json": {
            "paired_qD_minus_qC_loss": {"mean": paired_loss, "lower": paired_loss - 0.003,
                                        "upper": paired_loss + 0.003, "method": "paired whole-unit bootstrap"},
            "paired_qD_minus_qC_metric_pp": {"mean": metric_pp}, "models": {"qN": {"loss_nats": 0.87}}},
    }
    for j, candidate in enumerate(candidates):
        files[f"predictions/{candidate['candidate_id']}.json"] = {"feasible": j not in violating,
                                                                  "support_violations": int(j in violating)}
    for stage, seconds in stage_seconds.items():
        files[f"stages/{stage}.json"] = {"stage": stage, "seconds_this_attempt": seconds}
    return files


class Fixture:
    """Temporary run directories for the seven datasets (Roman as a two-split campaign)."""

    def __init__(self, test, **overrides):
        self.temp = tempfile.TemporaryDirectory()
        test.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.overrides = overrides

    def run(self, dataset, name, seed, **kwargs):
        return fake_run(self.base / name, dataset, lifecycle_files(dataset, seed, **{**self.overrides, **kwargs}))

    def runs(self, *, comparator=(), campaign=True, **per_dataset):
        """Synthetic lifecycle runs; datasets in ``comparator`` use the bundled q_C-returning example runs."""
        out = {}
        for index, dataset in enumerate(DATASETS):
            if dataset in comparator:
                out[dataset] = RunView(dataset, EXAMPLE_GRAPH if dataset in GRAPH_LEVEL else EXAMPLE_NODE)
            elif dataset == "Roman-empire" and campaign:
                splits = [self.run(dataset, f"roman/split_{k:02d}", 100 + k, **per_dataset.get(dataset, {}))
                          for k in range(2)]
                out[dataset] = RunView(dataset, self.base / "roman", splits=splits)
            else:
                out[dataset] = self.run(dataset, SHORT[dataset], 10 + index, **per_dataset.get(dataset, {}))
        return out

    def out(self, name):
        return self.base / "out" / name


def example_runs():
    return {d: RunView(d, EXAMPLE_GRAPH if d in GRAPH_LEVEL else EXAMPLE_NODE) for d in DATASETS}


def statuses(rows):
    return {row["status"] for row in rows}


def data_rows(out_dir, stem, name):
    import csv
    with (Path(out_dir) / "data" / stem / f"{name}.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def records_for(name, rows):
    return fake_records(name, rows, spec=RECORD_CONTRACTS[name])


class SelectionDisplayTests(unittest.TestCase):
    def test_s20_replays_algorithm_s4_and_restricts_the_max_t_to_twelve_finalists(self):
        fixture = Fixture(self)
        runs = fixture.runs()
        writer, compared = build_one("supplementary:Table S20", context(runs=runs), fixture.out("s20"),
                                     registry=REGISTRY)
        rows = read_table(fixture.out("s20"), "S20")
        self.assertEqual(len(rows), 7)
        self.assertEqual(len(rows[0]), 8)
        decision = runs["ogbn-arxiv"].json("select/decision.json")
        arxiv = rows[0]
        self.assertEqual(arxiv["|Q|"], str(len(REGISTRY_FAMILY)))
        self.assertEqual(arxiv["|Qs|"], str(len(decision["Q_s"])))
        self.assertEqual(arxiv["rmax"], f"{decision['max_radius']:.4f}")
        self.assertEqual(arxiv["eps_eq + 2 rmax"], f"{0.002 + 2 * decision['max_radius']:.4f}")
        self.assertEqual(arxiv["interpretation"], "conditional near-best selection")
        self.assertEqual(rows[3]["interpretation"], "repeated-split selection")
        self.assertIn(rows[4]["interpretation"], ("graph-level selection", "boundary graph-level selection"))
        # Fewer columns in the studentized maximum can only shrink the radius.
        data = Path(fixture.out("s20") / "data/S20/selection_radii.csv").read_text().splitlines()
        self.assertEqual(len(data), 1 + 6 + 2)                      # Roman keeps both splits
        for line in data[1:]:
            values = line.split(",")
            self.assertLessEqual(float(values[7]), float(values[4]) + 1e-12)
        self.assertNotIn("no_reference", statuses(compared))
        self.assertEqual(len(compared), 7 * 7)

    def test_s20_on_the_bundled_examples_and_a_changed_record_is_refused(self):
        writer, compared = build_one("supplementary:Table S20", context(runs=example_runs()),
                                     Path(tempfile.mkdtemp(dir=self.temporary())), registry=REGISTRY)
        rows = read_table(Path(writer.out_dir), "S20")
        self.assertEqual([r["resamples"] for r in rows], ["80"] * 7)
        self.assertEqual({r["interpretation"] for r in rows}, {"comparator/simple route"})
        fixture = Fixture(self)
        run = fixture.run("ogbn-arxiv", "changed", 3)
        decision = run.json("select/decision.json")
        decision["max_radius"] *= 1.5
        (run.root / "select/decision.json").write_text(json.dumps(decision))
        with self.assertRaises(DisplayInputError):
            run_audits.screened_radius(run)

    def temporary(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return directory.name

    def test_s12_representative_rows_gates_and_campaign_average(self):
        fixture = Fixture(self)
        runs = fixture.runs()
        writer, compared = build_one("supplementary:Table S12", context(runs=runs), fixture.out("s12"),
                                     registry=REGISTRY)
        rows = read_table(fixture.out("s12"), "S12")
        self.assertEqual(len(rows), 7 * 9)
        self.assertEqual(len(rows[0]), 11)
        arxiv = [r for r in rows if r["Dataset"] == "ogbn-arxiv"]
        self.assertEqual([r["route"] for r in arxiv], ["qC", "best-depth", "learned-stop", "phase", "fixed-sequence",
                                                       "MPC-OL", "one-step", "radius", "MPC-RH"])
        self.assertEqual((arxiv[0]["Delta_s"], arxiv[0]["L_s"], arxiv[0]["U_s"]), ("0.00000",) * 3)
        decision = runs["ogbn-arxiv"].json("select/decision.json")
        selected = [r for r in arxiv if r["selected"] == "yes"]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["route"], "radius")
        self.assertEqual(selected[0]["confirmation output"], "radius, E=1")
        self.assertEqual(selected[0]["Delta_s"], f"{decision['mean'][decision['provisional_index']]:.5f}")
        # Unobserved family, support violation of the MPC-OL execution, resource and metric gates.
        self.assertEqual(arxiv[2]["Delta_s"], "")
        self.assertEqual(arxiv[5]["support / metric / resource"], "fail/pass/pass")
        self.assertEqual(arxiv[7]["support / metric / resource"], "pass/pass/pass")
        # Roman: the two split decisions are averaged family by family.
        splits = [m.json("select/decision.json") for m in runs["Roman-empire"].members()]
        roman = [r for r in rows if r["Dataset"] == "Roman-empire"]
        radius = [next(j for j in d["Q_s"] if REGISTRY_FAMILY[j][0] == "radius" and j == d["provisional_index"])
                  for d in splits]
        self.assertEqual(roman[7]["Delta_s"], f"{np.mean([d['mean'][j] for d, j in zip(splits, radius)]):.5f}")
        source = data_rows(fixture.out("s12"), "S12", "representative_rows")
        self.assertEqual(len(source), (6 + 2) * 9)                          # Roman keeps its two splits
        self.assertEqual({r["split"] for r in source if r["dataset"] == "Roman-empire"}, {"0", "1"})
        self.assertNotIn("no_reference", statuses(compared))

    def test_s12b_nested_family_effects_match_an_independent_fold_loop(self):
        fixture = Fixture(self)
        runs = fixture.runs(comparator=("ogbn-products",))
        build_one("supplementary:Table S12b", context(runs=runs), fixture.out("s12b"), registry=REGISTRY)
        rows = read_table(fixture.out("s12b"), "S12b")
        self.assertEqual(len(rows), 6 * 5)                       # products returned q_C: omitted
        self.assertNotIn("ogbn-products", {r["Dataset"] for r in rows})
        view = runs["ogbn-arxiv"]
        data, decision = view.npz("select/unit_scores.npz"), view.json("select/decision.json")
        loss, w = data["loss"], data["counts"].astype(float)
        expected = {}
        for family in ("best_depth", "fixed_sequence", "mpc_ol"):
            members = [j for j, row in enumerate(REGISTRY_FAMILY) if row[0] == family]
            total = 0.0
            for fold in decision["nested_comparator_fold_choices"]:
                held = np.zeros(UNITS, bool)
                held[fold["held_out_original_units"]] = True
                train = w * ~held
                risk = train @ loss / train.sum()
                j = min(members, key=lambda m: risk[m])
                total += float(w[held] @ (loss[held, j] - loss[held, fold["candidate_index"]]))
            expected[family] = total / w.sum()
        arxiv = {r["route"]: r for r in rows if r["Dataset"] == "ogbn-arxiv"}
        self.assertEqual(arxiv["best-depth"]["nested Delta_s"], f"{expected['best_depth']:+.5f}")
        self.assertEqual(arxiv["MPC-OL"]["nested Delta_s"], f"{expected['mpc_ol']:+.5f}")
        self.assertEqual(arxiv["fixed-sequence"]["nested Delta_s"], "+0.00000")
        self.assertEqual(arxiv["fixed-sequence"]["reading"], "identity (qC family)")
        # A single-configuration family carries no re-selection optimism: nested equals outer fold.
        self.assertEqual(arxiv["best-depth"]["outer-fold Delta_s (Table S12)"],
                         f"{decision['mean'][1]:.5f}")
        self.assertEqual(arxiv["best-depth"]["reading"], "no re-selection optimism")
        self.assertEqual(arxiv["learned-stop"]["nested Delta_s"], "")
        folds = [r for r in data_rows(fixture.out("s12b"), "S12b", "nested_family_folds")
                 if r["dataset"] == "ogbn-arxiv"]
        self.assertEqual(len(folds), len(decision["nested_comparator_fold_choices"]) * 4)   # four observed families
        self.assertEqual({r["comparator_family"] for r in folds}, {"fixed_sequence"})

    def test_s12b_refuses_a_nested_trace_that_does_not_replay(self):
        fixture = Fixture(self)
        run = fixture.run("ogbn-arxiv", "trace", 5)
        decision = run.json("select/decision.json")
        decision["nested_comparator_fold_choices"][0]["candidate_index"] = 1
        (run.root / "select/decision.json").write_text(json.dumps(decision))
        with self.assertRaises(DisplayInputError):
            run_audits.nested_family_effects(run_audits.RunRecord(run))

    def test_s12c_readings_on_the_representative_rows_and_the_complete_ledger(self):
        fixture = Fixture(self)
        runs = fixture.runs(comparator=("ogbn-products",))
        build_one("supplementary:Table S12c", context(runs=runs), fixture.out("s12c"), registry=REGISTRY)
        rows = {r["Dataset"]: r for r in read_table(fixture.out("s12c"), "S12c")}
        self.assertEqual(len(rows), 7)
        self.assertEqual(len(rows["arxiv"]), 7)
        decision = runs["ogbn-arxiv"].json("select/decision.json")
        upper = decision["U"]
        eligible = [j for j in (5, 6, 7) if upper[j] < -0.003]       # representatives of the feedback families
        u_min = min(upper[j] for j in eligible)
        near = [j for j in eligible if upper[j] <= u_min + 0.002]
        labels = {5: "one-step", 6: "radius", 7: "MPC-RH"}
        self.assertEqual(rows["arxiv"]["U_min"], f"{u_min:.5f}")
        self.assertEqual(rows["arxiv"]["eps_eq = 0.002 on U_s: near-best -> choice"],
                         f"{', '.join(labels[j] for j in sorted(near))} → {labels[min(near)]}")
        self.assertEqual(rows["arxiv"]["executed selection (complete ledger)"], "radius")
        self.assertEqual(rows["arxiv"]["separated-protocol selection"], "radius")
        self.assertEqual(rows["products"]["eps_eq = 0.002 on U_s: near-best -> choice"], "— → qC")
        self.assertEqual(rows["products"]["U_min"], "")
        self.assertEqual(rows["products"]["executed selection (complete ledger)"], "qC")
        self.assertEqual(len(data_rows(fixture.out("s12c"), "S12c", "selector_readings")), 6 + 2)


class EntryAndSupportTests(unittest.TestCase):
    def test_s4_gates_hand_values_and_campaign_worst_split(self):
        fixture = Fixture(self)
        failing = {"0:0": dict(row_count=40, trajectory_count=30, min_propensity=0.28, ess=70.0,
                               max_normalized_weight=0.05, qualified=True),
                   "1:0": dict(row_count=9, trajectory_count=9, min_propensity=0.28, ess=9.0,
                               max_normalized_weight=0.11, qualified=False)}
        runs = fixture.runs(**{"ogbn-proteins": {"support": failing},
                               "COCO-SP": {"entry": {"headroom": -0.002}}})
        splits = runs["Roman-empire"].members()
        (splits[1].root / "train/entry_initial.json").write_text(json.dumps(
            {"plant_vs_q0": {"mean": -0.024, "upper": -0.015}, "pdd_fraction": 0.0058, "dil_gap": 0.015,
             "identity_pass": True}))
        writer, compared = build_one("supplementary:Table S4", context(runs=runs), fixture.out("s4"),
                                     registry=REGISTRY)
        rows = {r["Dataset"]: r for r in read_table(fixture.out("s4"), "S4")}
        self.assertEqual(len(rows), 7)
        self.assertEqual(len(rows["ogbn-arxiv"]), 9)
        self.assertEqual([rows["ogbn-arxiv"][k] for k in ("Delta_plant", "U(Delta_plant)", "PDD (pp)", "DIL gap",
                                                           "phi_registry", "selected strata", "U(Delta_oracle)",
                                                           "decision")],
                         ["-0.0180", "-0.0120", "0.24", "0.010", "0.89", "1.00", "-0.0050", "pass"])
        self.assertEqual(rows["ogbn-proteins"]["selected strata"], "0.50")       # A0 unsupported in stratum 1
        self.assertEqual(rows["ogbn-proteins"]["decision"], "fail")
        self.assertEqual(rows["COCO-SP"]["decision"], "fail")                    # -0.002 is not below -0.003
        roman = rows["Roman-empire"]
        self.assertEqual((roman["Delta_plant"], roman["U(Delta_plant)"], roman["PDD (pp)"], roman["DIL gap"]),
                         ("-0.0210", "-0.0120", "0.58", "0.015"))
        source = data_rows(fixture.out("s4"), "S4", "entry_gates")
        self.assertEqual(len(source), 6 + 2)
        self.assertEqual(next(r for r in source if r["dataset"] == "COCO-SP")["failed_gates"], "residual headroom")
        self.assertNotIn("no_reference", statuses(compared))
        exact = {row["path"] for row in compared if row["status"] == "exact"}
        self.assertTrue({"ogbn-arxiv/Delta_plant_hat", "ogbn-arxiv/PDD_pp", "ogbn-arxiv/decision"} <= exact)

    def test_s7_extrema_over_observed_cells_and_pooled_campaign_fraction(self):
        fixture = Fixture(self)
        runs = fixture.runs()
        writer, compared = build_one("supplementary:Table S7", context(runs=runs), fixture.out("s7"),
                                     registry=REGISTRY)
        rows = {r["Dataset"]: r for r in read_table(fixture.out("s7"), "S7")}
        self.assertEqual(len(rows), 7)
        self.assertEqual(len(rows["ogbn-arxiv"]), 8)
        self.assertEqual([rows["ogbn-arxiv"][k] for k in ("trajectories", "minimum count", "minimum propensity",
                                                           "ESS (per-cell minimum)",
                                                           "max. normalized weight (per cell)",
                                                           "qualified fraction")],
                         ["120", "4", "0.090", "3", "0.40", "0.50"])
        self.assertEqual(rows["Peptides-func"]["failure/remediation"], "downgrade on failure")
        self.assertEqual(rows["Roman-empire"]["qualified fraction"], "0.50")    # 4 of 8 cells over two splits
        data = (fixture.out("s7") / "data/S7/support_cells.csv").read_text().splitlines()
        self.assertEqual(len(data), 1 + 6 + 2)
        self.assertIn(",1,", data[1])                                            # one empty registered cell
        self.assertNotIn("no_reference", statuses(compared))


class AdmissionTests(unittest.TestCase):
    def test_s13_replays_the_bound_and_reclips_the_units(self):
        fixture = Fixture(self)
        differences = np.array([-0.02, -0.01, -0.015, -0.3, -0.005, -0.012])
        runs = {d: fixture.run(d, SHORT[d], 40 + k, admission_differences=differences)
                for k, d in enumerate(GRAPH_LEVEL)}
        writer, compared = build_one("supplementary:Table S13", context(runs=runs), fixture.out("s13"),
                                     registry=REGISTRY)
        rows = read_table(fixture.out("s13"), "S13")
        self.assertEqual(len(rows), 9)
        self.assertEqual(len(rows[0]), 11)
        self.assertEqual([r["Dataset"] for r in rows[3:5]], ["PascalVOC-SP (B = 0.05)", "PascalVOC-SP (B = 0.20)"])
        log_term = math.log(2 / 0.025)
        for row, clip in ((rows[0], 0.1), (rows[3], 0.05)):
            d = np.clip(differences, -clip, clip)
            s2 = float(np.var(d, ddof=1))
            penalty = math.sqrt(2 * s2 * log_term / 6) + 14 * clip * log_term / (3 * 5)
            self.assertEqual(row["D_bar"], f"{d.mean():.5f}")
            self.assertEqual(row["S2"], f"{s2:.6f}")
            self.assertEqual(row["penalty"], f"{penalty:.6f}")
            self.assertEqual(row["U_a"], f"{d.mean() + penalty:.6f}")
            self.assertEqual(row["G_admit"], "0")                                # penalty exceeds |D_bar| - eps_a
        self.assertEqual(rows[0]["B"], "0.1")
        self.assertEqual(rows[0]["J"], "6")
        source = data_rows(fixture.out("s13"), "S13", "admission_moments")
        self.assertEqual(len(source), 9)
        self.assertEqual(source[0]["clipped_fraction"], repr(1 / 6))
        self.assertNotIn("no_reference", statuses(compared))

    def test_s13_refuses_admission_records_that_do_not_replay(self):
        fixture = Fixture(self)
        run = fixture.run("PascalVOC-SP", "bad", 1)
        decision = run.json("admit/decision.json")
        decision["empirical_bernstein"]["upper"] += 0.01
        (run.root / "admit/decision.json").write_text(json.dumps(decision))
        with self.assertRaises(DisplayInputError):
            run_audits.admission_rows(run, [0.05])

    def test_s48_expected_work_and_metric_units(self):
        fixture = Fixture(self)
        runs = fixture.runs(comparator=("ogbn-products",),
                            **{"Peptides-func": {"admission_differences": [-0.05] * 300, "metric_pp": 0.58},
                               "ogbn-arxiv": {"admission_differences": [-0.05] * 30}})
        gate = fake_records("input_gate", [{"dataset": d, "quantity": "fallback_probability", "value": 0.118}
                                           for d in DATASETS if d != "ogbn-products"])
        writer, compared = build_one("supplementary:Table S48", context(runs=runs, records={"input_gate": gate}),
                                     fixture.out("s48"), registry=REGISTRY)
        rows = {r["Dataset"]: r for r in read_table(fixture.out("s48"), "S48")}
        self.assertEqual(len(rows), 7)
        self.assertEqual(len(rows["ogbn-arxiv"]), 6)
        arxiv = rows["ogbn-arxiv"]
        self.assertEqual(arxiv["evaluated rule"], "radius, E = 1")
        self.assertEqual(arxiv["paired loss over qC"], "-0.00450 CE")
        self.assertEqual(arxiv["official metric change"], "+0.07 pp")
        self.assertEqual(arxiv["expected work"], f"{(1 - 0.118) * 1.4 + 0.118 * 1.0:.2f}")
        self.assertEqual(arxiv["admission"], "descriptive, 30 buffered cores")
        self.assertEqual(rows["Peptides-func"]["official metric change"], "+0.0058 AP")
        self.assertEqual(rows["Peptides-func"]["admission"], "Theorem 2, J = 300; Gadmit = 1")
        self.assertEqual(rows["PascalVOC-SP"]["evaluated rule"], "qC returned")       # J = 12: penalty too large
        self.assertEqual(rows["ogbn-proteins"]["official metric change"], "+0.07 AUC pp")
        self.assertEqual(rows["ogbn-products"]["evaluated rule"], "qC returned")
        self.assertEqual(rows["ogbn-products"]["admission"], "comparator returned")
        self.assertTrue(rows["Roman-empire"]["admission"].startswith("descriptive, 2 official splits"))
        source = data_rows(fixture.out("s48"), "S48", "outcomes")
        self.assertEqual(len(source), 6 + 2)
        self.assertEqual(next(r for r in source if r["dataset"] == "ogbn-arxiv")["fallback_probability"], "0.118")
        self.assertNotIn("no_reference", statuses(compared))

    @staticmethod
    def budget_campaign(root: Path) -> Path:
        """A campaign export whose arm studies carry their Slurm allocations (synthetic values)."""
        header = "Cluster|JobIDRaw|JobID|Submit|Start|End|State|ElapsedRaw|AllocTRES|ExitCode"
        studies = []
        for i, (dataset, seconds) in enumerate((("ogbn-arxiv", 3600), ("ogbn-arxiv", 1800), ("PascalVOC-SP", 7200))):
            job = 500 + i
            write_files(root, {f"study_{i}/environment.json": {"slurm": {"SLURM_JOB_ID": str(job)},
                                                               "device_requested": "cuda"}})
            (root / f"acct_{i}.psv").write_text(
                header + "\n" + f"c|{job}|{job}|2026-01-01T00:00:00|2026-01-01T00:00:01|2026-01-01T02:00:00|"
                f"COMPLETED|{seconds}|cpu=8,gres/gpu=1|0:0\n" + f"c|{job}.0|{job}.0|x|y|z|COMPLETED|10|cpu=8|0:0\n",
                encoding="utf-8")
            studies.append({"dataset": dataset, "study": f"study_{i}", "accounting": [f"acct_{i}.psv"],
                            "job_log": [f"study_{i}/environment.json"]})
        write_files(root, {"study.json": {"studies": studies}})
        return root

    def test_s41_stage_hours_and_campaign_sums(self):
        fixture = Fixture(self)
        rejected = {"admission_differences": [0.01] * 12}
        runs = fixture.runs(**{"ogbn-arxiv": {"admission_differences": [-0.05] * 30},
                               "ogbn-proteins": rejected, "ogbn-products": rejected,
                               "PascalVOC-SP": {"admission_differences": [-0.05] * 300},
                               "Roman-empire": {"admission_differences": [-0.05] * 30}})
        campaign = self.budget_campaign(fixture.base / "budget_campaign")
        ctx = context(runs=runs, campaigns={"budget": campaign})
        writer, compared = build_one("supplementary:Table S41", ctx, fixture.out("s41"), registry=REGISTRY)
        rows = read_table(fixture.out("s41"), "S41")
        self.assertEqual(len(rows), 9)
        self.assertEqual(list(rows[0]), ["component", "unit", "GPU-hours", "record"])
        cells = {r["component"]: r["GPU-hours"] for r in rows}
        self.assertEqual(cells["backbone training (shared by every route)"],
                         "per seed: arxiv 1.5; Roman 1.5; Pascal 1.5 h (Table S5); ensembles of E checkpoints")
        self.assertEqual(cells["exploration logging"], "arxiv 1.00; Roman 2.00; Pascal 1.00")
        self.assertEqual(cells["nested comparator and candidate selection"], "arxiv 2.00; Roman 4.00; Pascal 2.00")
        total = next(v for k, v in cells.items() if k.startswith("total qualification"))
        self.assertEqual(total, "arxiv 3.70; Roman 7.40; Pascal 3.70")
        # Allocated GPUs times elapsed seconds of each arm job; job steps are not added.
        self.assertEqual(cells["independent E3–E6 comparison"], "arxiv 1.50; Pascal 2.00")
        allocations = data_rows(fixture.out("s41"), "S41", "comparison_allocations")
        self.assertEqual(sorted(r["job_id"] for r in allocations), ["500", "501", "502"])
        self.assertIn("share 0.27", next(r["record"] for r in rows if r["component"] == "exploration logging"))
        source = data_rows(fixture.out("s41"), "S41", "stage_hours")
        self.assertEqual(len(source), 6 + 2)
        proteins = next(r for r in source if r["dataset"] == "ogbn-proteins")
        self.assertEqual((proteins["deployed_rule_differs_from_qC"], proteins["total_hours"]), ("False", "3.7"))
        self.assertNotIn("no_reference", statuses(compared))

        # A marker rewritten after an interruption has no timing: that stage and the total are missing, not zero.
        write_files(fixture.base / "roman/split_01", {"stages/admit.json": {
            "stage": "admit", "status": "COMPLETE", "recovered_after_commit": True}})
        build_one("supplementary:Table S41", ctx, fixture.out("s41r"))
        cells = {r["component"]: r["GPU-hours"] for r in read_table(fixture.out("s41r"), "S41")}
        self.assertEqual(cells["admission evaluation"], "arxiv 0.10; Roman —; Pascal 0.10")
        self.assertEqual(next(v for k, v in cells.items() if k.startswith("total qualification")),
                         "arxiv 3.70; Roman —; Pascal 3.70")
        self.assertEqual(cells["exploration logging"], "arxiv 1.00; Roman 2.00; Pascal 1.00")
        source = data_rows(fixture.out("s41r"), "S41", "stage_hours")
        split = next(r for r in source if r["dataset"] == "Roman-empire" and r["split"] == "1")
        self.assertEqual((split["admit_hours"], split["total_hours"]), ("", ""))


def gap_rows(values):
    return [{"dataset": d, "quantity": q, "value": v} for d, row in values.items() for q, v in row.items()]


class RecordContractTests(unittest.TestCase):
    def test_contracts_have_the_registered_spec_fields(self):
        for name, spec in RECORD_CONTRACTS.items():
            self.assertTrue({"title", "manuscript_tables", "experiments", "keys", "quantities", "table",
                             "notes"} <= set(spec), name)

    def test_s21_interpretation_against_the_registered_calibration_alpha(self):
        values = {"ogbn-arxiv": {"median_gap_nats": 0.0062, "p5_gap_nats": 0.0011, "near_zero_fraction": 0.041,
                                 "switch_fraction": 0.036, "recovery_index_stability": 0.98,
                                 "beam_sequence_stability": 0.96, "near_zero_threshold_nats": 0.001},
                  "Peptides-func": {"median_gap_nats": 0.0036, "p5_gap_nats": 0.0005, "near_zero_fraction": 0.094,
                                    "switch_fraction": 0.079, "recovery_index_stability": 0.91,
                                    "beam_sequence_stability": 0.88, "near_zero_threshold_nats": 0.001}}
        out = Path(tempfile.mkdtemp(dir=self.temporary()))
        writer, compared = build_one("supplementary:Table S21", context(
            records={"action_gap_audit": records_for("action_gap_audit", gap_rows(values))}), out, registry=REGISTRY)
        rows = {r["Dataset"]: r for r in read_table(out, "S21")}
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(rows["ogbn-arxiv"]), 8)
        self.assertEqual(rows["ogbn-arxiv"]["interpretation"], "fixed-mode coverage adequate")
        self.assertEqual(rows["Peptides-func"]["interpretation"], "boundary; retain fixed-mode-only wording")
        self.assertEqual(rows["ogbn-arxiv"]["5th percentile gap"], "0.0011")
        source = data_rows(out, "S21", "action_gaps")
        self.assertEqual([(r["dataset"], r["calibration_alpha"]) for r in source],
                         [("ogbn-arxiv", "0.05"), ("Peptides-func", "0.05")])
        self.assertEqual(statuses(compared), {"exact"})

    def test_s22_violation_rate_is_the_complement_of_mode_stable_coverage(self):
        values = {"COCO-SP": {"beta_median": 0.98, "beta_p95": 1.11, "m_eps_op": 0.0022, "reference_term": 0.0009,
                              "tube_coverage": 0.96, "mode_stable_coverage": 0.959, "final_bound_nats": 0.024,
                              "observed_error_median_nats": 0.0057, "observed_error_p95_nats": 0.0082},
                  "Peptides-func": {"beta_median": 0.95, "beta_p95": 1.06, "m_eps_op": 0.0014,
                                    "reference_term": 0.0006, "tube_coverage": 0.96, "mode_stable_coverage": 0.9,
                                    "final_bound_nats": 0.011, "observed_error_median_nats": 0.0031,
                                    "observed_error_p95_nats": 0.0049}}
        out = Path(tempfile.mkdtemp(dir=self.temporary()))
        writer, compared = build_one("supplementary:Table S22", context(
            records={"tube_audit": records_for("tube_audit", gap_rows(values))}), out, registry=REGISTRY)
        rows = {r["Dataset"]: r for r in read_table(out, "S22")}
        self.assertEqual(list(rows), ["COCO", "Peptides"])
        self.assertEqual(len(rows["COCO"]), 9)
        self.assertEqual(rows["Peptides"]["violation rate"], "0.100")
        self.assertEqual(rows["Peptides"]["final bound"], "0.011 BCE")
        self.assertEqual(rows["COCO"]["final bound"], "0.024 CE")
        self.assertEqual(rows["COCO"]["beta median/95%"], "0.98/1.11")
        self.assertEqual([r["violation_rate"] for r in data_rows(out, "S22", "tube")], [repr(1 - 0.959), repr(1 - 0.9)])
        self.assertNotIn("no_reference", statuses(compared))
        self.assertIn("exact", statuses(compared))

    def test_s30_admission_sd_and_registered_retained_share_grid(self):
        fixture = Fixture(self)
        differences = np.array([-0.02, -0.01, -0.015, -0.3, -0.005, -0.012])
        runs = fixture.runs(comparator=("ogbn-products",), **{
            d: {"admission_differences": differences} for d in GRAPH_LEVEL})
        oracle = {d: {"median_oracle_depth": 24, "oracle_depth_q25": 14, "oracle_depth_q75": 40,
                      "fraction_differs_from_qc": 0.38, "inputs": 100} for d in DATASETS}
        out = fixture.out("s30")
        writer, compared = build_one("supplementary:Table S30", context(
            runs=runs, records={"oracle_dispersion": records_for("oracle_dispersion", gap_rows(oracle))}),
            out, registry=REGISTRY)
        rows = {r["Dataset"]: r for r in read_table(out, "S30")}
        self.assertEqual(len(rows), 7)
        self.assertEqual(len(rows["ogbn-arxiv"]), 6)
        self.assertEqual(rows["ogbn-arxiv"]["IQR"], "14–40")
        self.assertEqual(rows["ogbn-arxiv"]["phi_x grid (central)"], "0.20–0.55 (0.35)")   # q_C fixed sequence
        self.assertEqual(rows["ogbn-products"]["phi_x grid (central)"], "")                 # comparator returned
        sd = math.sqrt(float(np.var(np.clip(differences, -0.1, 0.1), ddof=1)))
        self.assertEqual(rows["COCO-SP"]["per-input SD of clipped effect (nats)"], f"{sd:.4f}")
        self.assertEqual(rows["ogbn-arxiv"]["per-input SD of clipped effect (nats)"], "")
        source = {r["dataset"]: r for r in data_rows(out, "S30", "oracle_dispersion")}
        self.assertEqual(len(source), 7)
        self.assertEqual(source["ogbn-arxiv"]["qC_route"], "fixed_sequence")
        self.assertEqual(source["ogbn-arxiv"]["phi_x_grid"], "0.2 0.35 0.55")
        self.assertNotIn("no_reference", statuses(compared))

    def temporary(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return directory.name


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        from tests.test_s14_confirmation import S14CampaignTests
        name = "test_complete_family_reconstructs_effects_and_returns_failed_peptides_to_comparator"
        self.fixture = S14CampaignTests(name)
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.out = self.fixture.base / "displays"

    def test_s14_joint_family_layout_readings_and_optional_margin_tests(self):
        self.fixture.change_result("arxiv/state", p_value_margin=0.0026)
        self.fixture.change_result("products/state", p_value_margin=None)
        ctx = context(campaigns={"confirmation": self.fixture.root})
        writer, compared = build_one("supplementary:Table S14", ctx, self.out, registry=REGISTRY)
        rows = {r["Dataset/claim"]: r for r in read_table(self.out, "S14")}
        self.assertEqual(len(rows), 9)
        self.assertEqual(len(rows["arxiv/state"]), 11)
        self.assertEqual(rows["arxiv/state"]["effect"], "+0.00574 CE")
        self.assertEqual(rows["arxiv/state"]["P vs margin"], "0.0026")
        self.assertEqual(rows["products/state"]["P vs margin"], "not tested")
        self.assertEqual(rows["Roman/RH"]["P vs margin"], "")
        self.assertEqual(rows["COCO/RH"]["raw P"], "< 1e-4")
        self.assertEqual(rows["Peptides/radius"]["Holm P"], "0.070")
        self.assertEqual(rows["arxiv/radius"]["interpretation"],
                         "radius retained on the point estimate; interval does not exclude the margin")
        self.assertEqual(rows["Roman/RH"]["interpretation"], "replanning retained")
        self.assertEqual(rows["products/state"]["interpretation"], "no feedback label; equivalence not established")
        self.assertEqual(rows["Peptides/radius"]["interpretation"],
                         "Holm P = 0.070; registered alpha = 0.05; frozen-rule outcome qC")
        self.assertEqual(rows["Pascal/RH"]["unit/n"], "425 confirmation graphs")
        source = data_rows(self.out, "S14", "joint_family")
        self.assertEqual(len(source), 9)
        self.assertEqual([r["retained"] for r in source].count("True"), 7)
        self.assertEqual(statuses(compared), {"exact", "missing"})
        missing = [row for row in compared if row["status"] == "missing"]
        self.assertEqual(len(missing), 7)
        self.assertTrue(all(row["path"].endswith("P_vs_margin") for row in missing))
        self.assertTrue(writer.missing_content)

    def test_s14_refuses_an_invalid_campaign(self):
        self.fixture.change_result("COCO/RH", p_value_kind="locally_Holm_adjusted")
        ctx = context(campaigns={"confirmation": self.fixture.root})
        with self.assertRaises(DisplayValidationError):
            build_one("supplementary:Table S14", ctx, self.out)

    def test_figure_s9_original_sizes_and_two_way_rescaling(self):
        spec = SPECS["confirmation_sensitivity"]
        rows = []
        for item in REGISTRY["supplementary_table_S14b_separated_data_sensitivity"]:
            for quantity, value in (("units_old", item["J_old"]), ("effect_nats", item["effect"]),
                                    ("ci_low_nats", item["interval_old"][0]), ("ci_high_nats", item["interval_old"][1]),
                                    ("graph_variance_share", item["graph_variance_share"]), ("holm_p", 0.5),
                                    ("label_survival_probability", 0.5)):
                rows.append({"contrast": item["contrast"], "quantity": quantity, "value": value})
        ctx = context(campaigns={"confirmation": self.fixture.root},
                      records={"confirmation_sensitivity": fake_records("confirmation_sensitivity", rows, spec=spec)})
        writer, compared = build_one("supplementary:Figure S9", ctx, self.out, registry=REGISTRY)
        self.assertIn("figures/FigS9.png", writer.files)
        self.assertIn("figures/FigS9.svg", writer.files)
        panel = data_rows(self.out, "FigS9", "panel_a")
        self.assertEqual(len(panel), 9)
        self.assertEqual(panel[0]["annotation"], "unit unchanged (n=32)")
        self.assertEqual(panel[6]["annotation"], "J: 1428 → 425; Holm P 0.0013")
        # Pascal: each side widens by [(1 - g) + g J_old/J_new]^(1/2) about the fixed effect.
        factor = math.sqrt(0.97 + 0.03 * 1428 / 425)
        self.assertAlmostEqual(float(panel[6]["low_new_nats"]), -0.00902 - (-0.00902 + 0.01355) * factor)
        self.assertEqual(statuses(compared), {"exact"})


class FigureTests(unittest.TestCase):
    def constants(self, ctx, **extra):
        ctx.__dict__["constants"] = dict(ctx.constants, **extra)
        return ctx

    def test_figure_s13_measured_panel_from_the_runs(self):
        fixture = Fixture(self)
        runs = fixture.runs(comparator=("ogbn-products",))
        writer, compared = build_one("supplementary:Figure S13", context(runs=runs), fixture.out("f13"),
                                     registry=REGISTRY)
        self.assertTrue({"figures/FigS13.png", "figures/FigS13.svg", "data/FigS13/panel_a.csv",
                         "data/FigS13/panel_b.csv", "data/FigS13/panel_c.csv"} <= set(writer.files))
        panel = (fixture.out("f13") / "data/FigS13/panel_c.csv").read_text().splitlines()
        self.assertEqual(len(panel), 8)
        self.assertIn("ogbn-arxiv,\"radius, E=1\",uncertainty-radius feedback,-0.0045", panel[1])
        self.assertIn("open-loop schedule retained", panel[3])
        self.assertNotIn("no_reference", statuses(compared))

    def test_figure_s14_reads_the_depth_record_and_the_reference_plant(self):
        fixture = Fixture(self)
        runs = fixture.runs()
        rows = [{"dataset": "ogbn-arxiv", "depth": d, "phase": "", "quantity": "source_alignment_change",
                 "value": -0.01 * k} for k, d in enumerate((2, 4, 8))]
        rows += [{"dataset": "Roman-empire", "depth": d, "phase": "", "quantity": "policy_loss_change",
                  "value": -0.002 * k} for k, d in enumerate((2, 4, 8))]
        rows += [{"dataset": "ogbn-arxiv", "depth": "", "phase": "", "quantity": q, "value": v}
                 for q, v in (("first_intervention_depth_q25", 3), ("first_intervention_depth_median", 4),
                              ("first_intervention_depth_q75", 6), ("supported_deterioration_depth", 8))]
        rows += [{"dataset": "ogbn-arxiv", "depth": "", "phase": p, "quantity": "transport_fraction", "value": v}
                 for p, v in (("early", 0.6), ("middle", 0.5), ("late", 0.3))]
        ctx = context(runs=runs, records={"depth_diagnostics": records_for("depth_diagnostics", rows)})
        writer, _ = build_one("supplementary:Figure S14", ctx, fixture.out("f14"))
        panel = (fixture.out("f14") / "data/FigS14/panel_a.csv").read_text().splitlines()
        # Nominal validation loss change from depth 2 (entry record): 0, -0.1, -0.05; then three alignment rows.
        self.assertEqual([line.split(",")[2] for line in panel[1:4]], ["0.0", "-0.09999999999999998",
                                                                      "-0.050000000000000044"])
        markers = (fixture.out("f14") / "data/FigS14/panel_a_markers.csv").read_text().splitlines()
        self.assertEqual(markers[1], "ogbn-arxiv,4,4.0")
        self.assertIn("figures/FigS14.png", writer.files)

    def figure_s17_fixture(self):
        fixture = Fixture(self)
        runs = fixture.runs(comparator=("ogbn-products",), **{
            d: {"admission_differences": [-0.05] * (300 if d in GRAPH_LEVEL else 30)} for d in DATASETS})
        cells = REGISTRY["supplementary_table_S16_arxiv_factorial"]["cells"]
        factorials = fake_records("factorials", [
            {"dataset": "ogbn-arxiv", "cell": cell, "quantity": q, "value": v}
            for cell, value in cells.items()
            for q, v in (("ld_integrated_loss", value["log_depth_CE"]), ("interval_low", value["interval"][0]),
                         ("interval_high", value["interval"][1]))])
        gate = fake_records("input_gate", [{"dataset": d, "quantity": "fallback_probability", "value": 0.1}
                                           for d in DATASETS])
        gaps = {d: {"median_gap_nats": 0.006, "p5_gap_nats": 0.001, "near_zero_fraction": 0.04,
                    "switch_fraction": 0.04, "recovery_index_stability": 0.97, "beam_sequence_stability": 0.95,
                    "near_zero_threshold_nats": 0.001} for d in ("ogbn-arxiv", "COCO-SP")}
        tube = {d: {"beta_median": 0.97, "beta_p95": 1.1, "m_eps_op": 0.002, "reference_term": 0.001,
                    "tube_coverage": 0.96, "mode_stable_coverage": 0.96, "final_bound_nats": 0.02,
                    "observed_error_median_nats": 0.005, "observed_error_p95_nats": 0.008}
                for d in ("ogbn-arxiv", "COCO-SP")}
        ctx = context(runs=runs, records={"factorials": factorials, "input_gate": gate,
                                          "action_gap_audit": records_for("action_gap_audit", gap_rows(gaps)),
                                          "tube_audit": records_for("tube_audit", gap_rows(tube))})
        return fixture, ctx, cells

    def test_figure_s17_panels_and_required_registered_grid(self):
        fixture, ctx, cells = self.figure_s17_fixture()
        self.constants(ctx)
        ctx.constants.pop("regret_bound_table_S28b", None)
        with self.assertRaises(DisplayInputError):
            build_one("supplementary:Figure S17", ctx, fixture.out("f17a"))
        self.constants(ctx, regret_bound_table_S28b={"sigma1_grid": [0.005, 0.007, 0.010]})
        writer, compared = build_one("supplementary:Figure S17", ctx, fixture.out("f17"), registry=REGISTRY)
        names = {Path(f).name for f in writer.files}
        self.assertTrue({"panel_a_curves.csv", "panel_b.csv", "panel_c.csv", "panel_d.csv", "panel_e.csv",
                         "panel_f.csv", "panel_g_curves.csv", "panel_h.csv", "FigS17.png", "FigS17.svg"} <= names)
        slack = {r["dataset"]: float(r["slack"]) for r in data_rows(fixture.out("f17"), "FigS17", "panel_c")}
        for dataset in DATASETS:                                          # Theorem 1: eps_eq + 2 max radius
            decisions = [m.json("select/decision.json") for m in ctx.require_run(dataset).members()]
            self.assertAlmostEqual(slack[dataset], decisions[0]["epsilon_eq"]
                                   + 2 * max(d["max_radius"] for d in decisions))
        contrasts = data_rows(fixture.out("f17"), "FigS17", "panel_d")
        self.assertEqual([row["contrast"] for row in contrasts], ["A", "O", "X", "AO", "AX", "OX", "AOX"])
        expected_width = math.sqrt(sum(((v["interval"][1] - v["interval"][0]) / 2) ** 2
                                       for v in cells.values())) / 4
        for row in contrasts:
            self.assertAlmostEqual(float(row["high_nats"]) - float(row["effect_coded_nats"]), expected_width)
            self.assertAlmostEqual(float(row["effect_coded_nats"]) - float(row["low_nats"]), expected_width)
            self.assertEqual(row["interval_method"], "independence approximation (printed Figure S17d illustration)")
        self.assertEqual(len(data_rows(fixture.out("f17"), "FigS17", "panel_d_cells")), 8)
        self.assertNotIn("data/FigS17/panel_d_paired_joint.csv", writer.files)
        scopes = {key: value for scope in writer.evidence_scope for key, value in scope.items()}
        self.assertFalse(scopes["factorial_joint"]["available"])
        self.assertFalse(scopes["panel_d_interval_method"]["full_figure_comparison"])
        self.assertIn("intervals: independence approximation", (fixture.out("f17") / "figures/FigS17.svg").read_text())
        exact = {row["path"] for row in compared if row["status"] == "exact"
                 and row["group"] == "supplementary_table_S16_arxiv_factorial"}
        self.assertEqual(len(exact), 8)                                   # printed analytical illustration, including its width
        self.assertFalse(any(row["status"] == "missing" and row["path"].endswith("half_width_95") for row in compared))
        # Registering the S28b grid and matching this panel's related S16 values
        # still cannot promote the entire eight-panel figure to full agreement.
        missing, coverage = comparison_coverage("supplementary:Figure S17", writer.comparisons, REGISTRY, {},
                                                scope="measurement")
        self.assertFalse(coverage["complete"])
        self.assertTrue(any(row["group"] == "display:supplementary:Figure S17" for row in missing))
        needed = (fixture.out("f17") / "data/FigS17/panel_g_units_for_smallest_effect.csv").read_text()
        self.assertIn("Hoeffding,0.0045,23670", needed)                  # S28: n for the arxiv effect
        breakeven = (fixture.out("f17") / "data/FigS17/panel_h.csv").read_text().splitlines()
        overhead = (0.9 * 1.4 + 0.1 * 1.0) / 1.0 - 1
        self.assertAlmostEqual(float(breakeven[1].split(",")[6]), 0.035 * 0.87 * overhead / (1 + overhead))
        self.assertNotIn("no_reference", statuses(compared))

    def test_figure_s17_keeps_actual_paired_intervals_separate_from_its_illustration(self):
        from cm_experiments.factorial import cell_statistics
        from paper.build.displays.factorial_data import CELLS
        fixture, ctx, _ = self.figure_s17_fixture()
        self.constants(ctx, regret_bound_table_S28b={"sigma1_grid": [0.005, 0.007, 0.010]})
        # The common per-unit noise cancels in every paired contrast, while the
        # explicitly independent illustration has nonzero error bars.
        noise = np.arange(12) * .002
        units = {cell: .5 + noise - .004 * int(cell[0]) - .002 * int(cell[1]) - .001 * int(cell[2]) for cell in CELLS}
        stats, contrasts, draws = cell_statistics(units, np.ones(12), ("A", "O", "X"), reps=128, seed=7)
        raw_root = fixture.base / "factorial"
        raw_root.mkdir()
        np.savez(raw_root / "unit_dil.npz", unit_id=np.arange(12), unit_weight=np.ones(12),
                 joint_bootstrap_cell_means=draws, **{f"cell{cell}_unit_dil": v for cell, v in units.items()})
        raw = {"schema": "cm-factorial-2", "dataset": "ogbn-arxiv", "factors": ["A", "O", "X"],
               "cells": [{"cell": cell, **stats[cell]} for cell in CELLS], "contrasts": contrasts,
               "bootstrap": {"reps": 128, "seed": 7},
               "unit_dil": {"file": "unit_dil.npz", "sha256": hashlib.sha256((raw_root / "unit_dil.npz").read_bytes()).hexdigest()}}
        (raw_root / "factorial.json").write_text(json.dumps(raw))
        ctx._records["factorials"] = fake_records("factorials", [
            {"dataset": "ogbn-arxiv", "cell": cell, "quantity": q, "value": stats[cell][q], "source_file": "factorial.json"}
            for cell in CELLS for q in ("ld_integrated_loss", "interval_low", "interval_high")], root=raw_root)
        writer, _ = build_one("supplementary:Figure S17", ctx, fixture.out("f17-paired"), registry=REGISTRY)
        displayed = data_rows(fixture.out("f17-paired"), "FigS17", "panel_d")
        paired = data_rows(fixture.out("f17-paired"), "FigS17", "panel_d_paired_joint")
        self.assertEqual(len(paired), 7)
        for illustration, actual in zip(displayed, paired):
            self.assertEqual(illustration["contrast"], actual["contrast"])
            self.assertAlmostEqual(float(illustration["effect_coded_nats"]), float(actual["effect_coded_nats"]))
            self.assertGreater(float(illustration["high_nats"]) - float(illustration["low_nats"]), .001)
            self.assertLess(float(actual["high_nats"]) - float(actual["low_nats"]), 1e-12)
            self.assertIn("paired whole-unit bootstrap", actual["interval_method"])
        joint_scope = next(scope["factorial_joint"] for scope in writer.evidence_scope if "factorial_joint" in scope)
        self.assertTrue(joint_scope["available"])
        self.assertEqual(joint_scope["unit_sha256"], raw["unit_dil"]["sha256"])
        self.assertEqual(joint_scope["companion_source_data"], "data/FigS17/panel_d_paired_joint.csv")
        # An invalid supplied paired source is still an error, never a reason
        # to silently keep only the analytical illustration.
        with (raw_root / "unit_dil.npz").open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(DisplayValidationError, "array hash differs"):
            build_one("supplementary:Figure S17", ctx, fixture.out("f17-invalid"), registry=REGISTRY)

    def test_figure_s17_independence_illustration_requires_valid_eight_cell_inputs(self):
        from paper.build.displays.factorial_data import CELLS
        cells = {cell: {"mean": .5, "low": .49, "high": .51} for cell in CELLS}
        for invalid in ({cell: row for cell, row in cells.items() if cell != "111"},
                        {**cells, "111": {"mean": .5, "low": .51, "high": .49}},
                        {**cells, "111": {"mean": .5, "low": float("nan"), "high": .51}}):
            with self.assertRaises(DisplayValidationError):
                run_audits.independent_factorial_illustration(invalid, ("A", "O", "X"))

    def test_figure_s19_needs_the_registered_scope_table(self):
        fixture = Fixture(self)
        runs = fixture.runs(comparator=("ogbn-products",))
        ctx = context(runs=runs)
        with self.assertRaises(DisplayInputError):
            build_one("supplementary:Figure S19", ctx, fixture.out("f19a"))
        self.constants(ctx, guarantee_scope_figure_S19=[
            {"statement": "Lemma 1 transport validity", "class": "exact", "holds_under": "row-stochastic kernels"},
            {"statement": "Thm. 2 clipped-loss admission", "class": "finite-sample", "holds_under": "i.i.d. graphs"}])
        writer, compared = build_one("supplementary:Figure S19", ctx, fixture.out("f19"), registry=REGISTRY)
        panel = (fixture.out("f19") / "data/FigS19/panel_b.csv").read_text().splitlines()
        self.assertEqual(len(panel), 8)
        self.assertIn("products,8 buffered cores,cluster bootstrap (descriptive),none,0 (qC)", panel[3])
        self.assertIn(">= 95%; policy fixed on D_confirm", panel[5])
        self.assertIn("2 official splits", panel[4])
        self.assertTrue({"figures/FigS19.png", "data/FigS19/panel_a.csv", "data/FigS19/panel_c.csv"}
                        <= set(writer.files))
        self.assertNotIn("no_reference", statuses(compared))


if __name__ == "__main__":
    unittest.main()
