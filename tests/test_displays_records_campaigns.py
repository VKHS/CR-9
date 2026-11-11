"""Display generators built from measurement-record sets and campaign exports.

Fixtures are synthetic: small run directories that hold exactly the files the
generators read, record sets in the summary schema and a stubbed strict replay of
the E3-E6 campaign. Several inputs mirror the inputs behind the printed tables so
that the declared comparisons can be checked against the registry; they are test
inputs, not benchmark observations.
"""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from paper.build.displays import records_campaigns as rc
from paper.build.displays import framework as fw
from paper.build.displays.framework import (DATASETS, GENERATORS, DisplayInputError, DisplayValidationError, RunView,
                                            load_references)
from paper.build.displays.testing import build_one, context, fake_records, fake_run, read_table

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = load_references(ROOT)[0]
SPECS = json.loads((ROOT / "records/specs.json").read_text(encoding="utf-8"))["sets"]
ADMITTED = [d for d in DATASETS if d != "ogbn-products"]
ARMS = ("E3", "E4", "E5", "E6")


def contract(name: str) -> dict:
    return SPECS[name]


def records(name: str, rows: list[dict]):
    spec = SPECS.get(name) or contract(name)
    return fake_records(name, rows, spec=spec)


def seeded(mean: float, sd: float, n: int = 12) -> list[float]:
    """n values with exactly this sample mean and sample SD."""
    base = np.linspace(-1.0, 1.0, n)
    base = (base - base.mean()) / base.std(ddof=1)
    return [float(mean + sd * z) for z in base]


# ---------------------------------------------------------------------------
# Synthetic lifecycle runs
# ---------------------------------------------------------------------------

# dataset: comparator (route, E, cost, horizon), deployed (route, cost) or None (comparator returned),
# paired loss effect (mean, low, high), metric gain in pp (mean, low, high), test scores (qN, qC, qD),
# plant test loss, Lean share, test units, metric
WORLD = {
    "ogbn-arxiv": (("fixed_sequence", 4, 4.0, None), ("radius", 4.7), (-0.0045, -0.0075, -0.0015), (0.07, 0.01, 0.13),
                   (72.93, 73.15, 73.22), 0.87, 0.93, 32, "accuracy"),
    "ogbn-proteins": (("fixed_sequence", 4, 4.0, None), ("radius", 4.8), (-0.0038, -0.0068, -0.0008), (0.11, 0.03, 0.19),
                      (87.02, 87.15, 87.26), 0.25, 0.92, 32, "rocauc"),
    "ogbn-products": (("phase", 2, 2.1, None), None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (87.59, 87.74, 87.74), 0.30, 1.0,
                      64, "accuracy"),
    "Roman-empire": (("mpc_ol", 4, 4.0, 8), ("mpc_rh", 5.4), (-0.0057, -0.0089, -0.0026), (0.26, 0.12, 0.44),
                     (91.76, 92.37, 92.63), 0.25, 0.91, 10, "accuracy"),
    "PascalVOC-SP": (("mpc_ol", 8, 8.0, 16), ("mpc_rh", 10.5), (-0.0084, -0.0128, -0.0041), (0.60, 0.39, 0.83),
                     (46.89, 47.61, 48.21), 1.30, 0.94, 1429, "macro_f1"),
    "COCO-SP": (("mpc_ol", 8, 8.0, 16), ("mpc_rh", 10.8), (-0.0091, -0.0127, -0.0054), (0.65, 0.43, 0.91),
                (31.60, 32.38, 33.03), 2.00, 0.95, 5000, "macro_f1"),
    "Peptides-func": (("fixed_sequence", 4, 4.0, None), ("radius", 4.9), (-0.0030, -0.0047, -0.0013), (0.58, 0.20, 0.96),
                      (0.704, 0.714, 0.720), 0.22, 0.90, 2331, "ap"),
}
PLANT_PARAMETERS = {"ogbn-arxiv": 3116632, "ogbn-proteins": 3517068, "ogbn-products": 3358895,
                    "Roman-empire": 2524946, "PascalVOC-SP": 3140277, "COCO-SP": 3178737, "Peptides-func": 5114130}


def candidate_list(dataset: str, deployed_cost: float):
    """A registry of evaluated configurations at the comparator's ensemble size, with known costs."""
    (qc_route, E, qc_cost, horizon), deployed, *_ = WORLD[dataset]
    c = lambda route, cost, profile="none", **extra: {"route": route, "ensemble": E, "cost": cost, "profile": profile,
                                                       "execution_failure": False, "feasible": True, "depth": 64, **extra}
    items = [c("nominal", 1.0 * E), c("best_depth", 0.5 * E), c("best_depth", 1.0 * E), c("phase", 1.0 * E),
             c("phase", 1.05 * E), c("fixed_sequence", 1.0 * E), c("fixed_sequence", 1.05 * E),
             c("mpc_ol", 1.0 * E, "lean", horizon=8), c("mpc_ol", 1.15 * E, "full", horizon=16),
             c("one_step", 1.12 * E, "full"), c("radius", 1.15 * E, "lean", kappa=1.0),
             c("radius", 1.20 * E, "full", kappa=1.0), c("mpc_rh", 1.30 * E, "lean", horizon=8),
             c("mpc_rh", 1.36 * E, "full", horizon=8),
             {"route": "radius", "ensemble": 2 * E, "cost": 9.9 * E, "profile": "full", "execution_failure": False,
              "depth": 64, "kappa": 1.0}]
    items.append(c(qc_route, qc_cost, "none" if qc_route not in ("mpc_ol",) else "full",
                   **({"horizon": horizon} if horizon else {})))
    qc = len(items) - 1
    if deployed is None:
        return items, qc, qc, {}
    route = deployed[0]
    extra = {"kappa": 2.0} if route == "radius" else {"horizon": 4}
    items.append(c(route, deployed_cost, "full", **extra))
    items.append(c(route, 0.83 * deployed_cost, "lean", **extra))
    qd = len(items) - 2
    return items, qc, qd, {qd: -0.0040, qd + 1: -0.0040 * WORLD[dataset][6]}


def evidence(dataset: str, qc: int, qd: int, deltas: dict, n_candidates: int) -> dict:
    (_, E, *_), _, effect, gain, scores, plant_loss, _, units, metric = WORLD[dataset]
    native = metric == "ap"
    models = {}
    for name, score in zip(("qN", "qC", "qD"), scores):
        models[name] = {"metric": score if native else score / 100.0, "metric_percent": 100.0 * score if native else score,
                        "loss_nats": plant_loss if name == "qN" else plant_loss - 0.01, "test_units": units}
    return {"selection": {"qC_index": qc}, "admission": {"deployed_index": qd},
            "candidates": [{"index": i, "select_delta": deltas.get(i)} for i in range(n_candidates)],
            "task": {"metric": metric, "loss_name": "binary_cross_entropy" if metric in ("rocauc", "ap") else "cross_entropy"},
            "final": {"models": models, "paired_loss": dict(zip(("mean", "lower", "upper"), effect)),
                      "paired_metric_pp": dict(zip(("mean", "lower", "upper"), gain))}}


def make_run(root: Path, dataset: str, *, deployed_cost: float | None = None, splits=()) -> RunView:
    deployed = WORLD[dataset][1]
    cost = deployed_cost if deployed_cost is not None else (deployed[1] if deployed else 0.0)
    items, qc, qd, deltas = candidate_list(dataset, cost)
    files = {"select/candidates.json": items,
             "train/model_inventory.json": {"checkpoints": [
                 {"path": "train/plant_seed1_best.pt", "named_tensor_elements": PLANT_PARAMETERS[dataset]},
                 {"path": "train/q0_seed1_best.pt", "named_tensor_elements": 99}]}}
    return fake_run(root, dataset, files, evidence=evidence(dataset, qc, qd, deltas, len(items)), splits=list(splits))


def make_runs(tmp: Path) -> dict:
    runs = {}
    for dataset in DATASETS:
        root = tmp / "runs" / dataset
        if dataset == "Roman-empire":     # a two-split campaign: costs average over the splits
            splits = [make_run(root / f"split_{i:02d}", dataset, deployed_cost=c) for i, c in enumerate((5.3, 5.5))]
            runs[dataset] = make_run(root, dataset, splits=splits)
        else:
            runs[dataset] = make_run(root, dataset)
    return runs


# ---------------------------------------------------------------------------
# Synthetic record sets
# ---------------------------------------------------------------------------

GATE = {"ogbn-arxiv": (0.118, 13, 15.1, 21.0), "ogbn-proteins": (0.147, 14, 37.1, 44.0),
        "Roman-empire": (0.129, 25, 12.4, 18.0), "PascalVOC-SP": (0.081, 24, 6.1, 12.0),
        "COCO-SP": (0.071, 27, 7.8, 16.0), "Peptides-func": (0.188, 15, 18.6, 25.0)}
# method key: (serial cost, wall time, GPU-h, seed SD, mean score)
BASELINES = {
    "ogbn-arxiv": {"best fixed sequence (qC)": (4.0, 4.0, 10.88, 0.18, 73.15), "radius (deployed)": (4.7, 4.58, 12.8, 0.18, 73.22),
                   "AMP": (4.5, 4.5, 13.6, 0.21, 73.06), "ADMP-GNN": (3.2, 3.2, 13.6, 0.21, 73.02),
                   "best fixed depth": (None, None, 10.88, 0.25, 73.10)},
    "ogbn-proteins": {"best fixed sequence (qC)": (4.0, 4.0, 11.39, 0.33, 87.15), "radius (deployed)": (4.8, 4.66, 13.4, 0.33, 87.26),
                      "AMP": (4.5, 4.5, 14.2, 0.37, 87.05), "GRAMA": (5.6, 5.6, 16.0, 0.37, 87.03)},
    "ogbn-products": {"optimized phase (qC)": (2.1, 2.1, 15.2, 0.20, 87.74), "ADMP-GNN": (1.6, 1.6, 16.2, 0.23, 87.57),
                      "GRAMA": (2.8, 2.8, 18.1, 0.23, 87.49)},
    "Roman-empire": {"MPC-OL (qC)": (4.0, 4.0, 7.82, 0.45, 92.37), "MPC-RH (deployed)": (5.4, 5.16, 9.2, 0.45, 92.63),
                     "Early-Exit GNN": (2.8, 2.8, 9.2, 0.52, 92.24), "ADMP-GNN": (3.2, 3.2, 9.8, 0.52, 92.18)},
    "PascalVOC-SP": {"MPC-OL (qC)": (8.0, 8.0, 9.01, 0.80, 47.61), "MPC-RH (deployed)": (10.5, 10.06, 10.6, 0.80, 48.21),
                     "Graph Mamba": (10.0, 10.0, 12.6, 0.92, 47.41), "GRAMA": (11.2, 11.2, 12.6, 0.92, 47.36)},
    "COCO-SP": {"MPC-OL (qC)": (8.0, 8.0, 10.29, 0.55, 32.38), "MPC-RH (deployed)": (10.8, 10.31, 12.1, 0.55, 33.03),
                "Graph Mamba": (10.0, 10.0, 14.4, 0.63, 32.58), "GRAMA": (11.2, 11.2, 14.4, 0.63, 32.51)},
    "Peptides-func": {"best fixed sequence (qC)": (4.0, 4.0, 9.61, 0.45, 0.714), "radius (deployed)": (4.9, 4.74, 11.3, 0.45, 0.720),
                      "Graph Mamba": (5.0, 5.0, 13.5, 0.52, 0.714), "GRAMA": (5.6, 5.6, 13.5, 0.52, 0.713)},
}


def baselines_set():
    rows = []
    for dataset, methods in BASELINES.items():
        native = dataset == "Peptides-func"
        for method, (serial, wall, gpu, sd, score) in methods.items():
            keys = {"dataset": dataset, "method": method, "seed": ""}
            for quantity, value in (("serial_cost", serial), ("wall_time", wall), ("gpu_hours", gpu)):
                if value is not None:
                    rows.append({**keys, "quantity": quantity, "value": value})
            if method in ("AMP", "ADMP-GNN", "GRAMA", "Graph Mamba", "Early-Exit GNN"):
                rows += [{**keys, "quantity": "hpo_trials", "value": 48}, {**keys, "quantity": "params_millions", "value": 3.3}]
            for seed, value in enumerate(seeded(score, sd / 100.0 if native else sd), start=1):
                rows.append({**keys, "seed": str(seed), "quantity": "score", "value": value})
    return records("baselines", rows)


def gate_set():
    rows = []
    for dataset, (p, overhead, gpu, cpu) in GATE.items():
        for quantity, value in (("fallback_probability", p), ("deployed_wall_overhead_percent", overhead),
                                ("gpu_memory_gib", gpu), ("cpu_memory_gib", cpu)):
            rows.append({"dataset": dataset, "quantity": quantity, "value": value})
    rows += [{"dataset": "ogbn-products", "quantity": "gpu_memory_gib", "value": 64.5},
             {"dataset": "ogbn-products", "quantity": "cpu_memory_gib", "value": 118.0}]
    return records("input_gate", rows)


GRID = (1, 2, 4, 8, 12, 16, 20)
SCORE_PER_NAT = 20.0      # grid scores rise by 20 points (0.2 AP) per nat of comparator loss gain
D_INF = {"ogbn-arxiv": 0.0305, "ogbn-proteins": 0.0088, "Roman-empire": 0.0088, "PascalVOC-SP": 0.0455,
         "COCO-SP": 0.0700, "Peptides-func": 0.0077}


def grid_sets():
    losses, scores = [], []
    for dataset, d_inf in D_INF.items():
        plant = WORLD[dataset][5]
        qc_score = WORLD[dataset][4][1]
        E = WORLD[dataset][0][1]
        per_nat = SCORE_PER_NAT * (0.01 if dataset == "Peptides-func" else 1.0)
        for e in GRID:
            losses.append({"dataset": dataset, "ensemble": str(e), "quantity": "loss_nats",
                           "value": plant - d_inf + d_inf / e, "unit": "nats"})
            scores.append({"dataset": dataset, "ensemble": str(e), "quantity": "official_score",
                           "value": qc_score + per_nat * (d_inf / E - d_inf / e)})
    return records("ensemble_grid", losses), records("ensemble_grid_scores", scores)


# arm values per dataset: effect vs qC, effect vs qM, CI low/high vs qM (mnats), Holm P, work/ceiling; ceiling
ARMS_PAPER = {
    "ogbn-arxiv": (1.15, {"E3": (-1.6, 2.9, 1.0, 4.8, 0.007, 1.00), "E4": (-1.4, 3.1, 1.2, 5.0, 0.005, 1.00),
                          "E5": (-1.3, 3.2, 1.3, 5.1, 0.005, 1.00), "E6": (-4.4, 0.1, -1.8, 2.0, 0.92, 0.92)}),
    "ogbn-proteins": (1.17, {"E3": (-1.3, 2.5, 0.9, 4.1, 0.006, 1.00), "E4": (-1.2, 2.6, 1.0, 4.2, 0.006, 1.00),
                             "E5": (-1.1, 2.7, 1.1, 4.3, 0.005, 1.00), "E6": (-3.7, 0.1, -1.5, 1.7, 0.91, 0.91)}),
    "Roman-empire": (1.30, {"E3": (-1.4, 4.3, 0.7, 7.9, 0.049, 1.00), "E4": (-1.3, 4.4, 0.8, 8.0, 0.049, 1.00),
                            "E5": (-0.5, 5.2, 1.6, 8.8, 0.018, 1.00), "E6": (-5.5, 0.2, -3.4, 3.8, 0.91, 0.85)}),
    "PascalVOC-SP": (1.29, {"E3": (-2.1, 6.3, 5.3, 7.3, 1e-6, 1.00), "E4": (-1.9, 6.5, 5.5, 7.5, 1e-6, 1.00),
                            "E5": (-0.7, 7.7, 6.7, 8.7, 1e-6, 1.00), "E6": (-8.1, 0.3, -0.7, 1.3, 0.55, 0.85)}),
    "COCO-SP": (1.33, {"E3": (-2.3, 6.8, 6.5, 7.1, 1e-6, 1.00), "E4": (-2.0, 7.1, 6.8, 7.4, 1e-6, 1.00),
                       "E5": (-0.7, 8.4, 8.1, 8.7, 1e-6, 1.00), "E6": (-8.7, 0.4, 0.1, 0.7, 0.008, 0.83)}),
    "Peptides-func": (1.18, {"E3": (-1.1, 1.9, 1.2, 2.6, 1e-6, 1.00), "E4": (-1.0, 2.0, 1.3, 2.7, 1e-6, 1.00),
                             "E5": (-0.8, 2.2, 1.5, 2.9, 1e-6, 1.00), "E6": (-2.9, 0.1, -0.6, 0.8, 0.77, 0.92)}),
}
# low and high retained-share configurations of the arxiv arms (effect vs qC, mnats; overhead of E6)
ARXIV_GRID = {"E3": (-0.9, -2.5), "E4": (-0.8, -2.2), "E5": (-1.0, -1.6), "E6": (1.05, 1.11)}


def budget_world():
    summary, rows = [], []
    for dataset, (ceiling, arms) in ARMS_PAPER.items():
        for arm, (vs_c, vs_m, low, high, p, work) in arms.items():
            entry = {"dataset": dataset, "arm": arm, "effect_vs_qC": vs_c / 1000, "effect": vs_m / 1000,
                     "ci_low": low / 1000, "ci_high": high / 1000, "p_holm": p, "work_ratio": work * ceiling,
                     "ceiling": ceiling, "reference_work_ratio": ceiling, "seeds": 12, "units": 425,
                     "inference_scope": "conditional fixed-parent graph-unit comparison"}
            summary.append(entry)
            label = rc.ARM_LABELS[arm] if dataset == "PascalVOC-SP" else arm     # both producer conventions
            keys = {"dataset": dataset, "arm": label, "configuration": "selected"}
            for quantity, field in rc.REPLAY_FIELDS.items():
                rows.append({**keys, "quantity": quantity, "value": entry[field]})
            if dataset == "ogbn-arxiv":
                for name, value in zip(("low share", "high share"), ARXIV_GRID[arm]):
                    other = {**keys, "configuration": name}
                    if arm == "E6":
                        rows.append({**other, "quantity": "work_ratio", "value": value})
                    else:
                        rows.append({**other, "quantity": "effect_vs_qc_nats", "value": value / 1000})
    return {"check": "budget_matched", "passed": True, "summary": summary}, records("budget_matched", rows)


FACTORIAL_CELLS = {
    "ogbn-arxiv": {"000": (0.8703, 0.8660, 0.8746, 72.82, 0.00, 1.00), "100": (0.8666, 0.8622, 0.8710, 72.88, 0.32, 1.08),
                   "010": (0.8685, 0.8640, 0.8730, 72.85, 0.00, 1.05), "001": (0.8657, 0.8614, 0.8700, 72.89, 0.24, 1.10),
                   "110": (0.8578, 0.8533, 0.8623, 72.99, 0.43, 1.14), "101": (0.8547, 0.8502, 0.8592, 73.05, 0.48, 1.18),
                   "011": (0.8535, 0.8490, 0.8580, 73.07, 0.40, 1.16), "111": (0.8445, 0.8401, 0.8489, 73.21, 0.55, 1.25)},
    "PascalVOC-SP": {"000": (1.3230, 1.3168, 1.3292, 47.04, 0.41, 1.08), "001": (1.3183, 1.3121, 1.3245, 47.41, 0.39, 1.11),
                     "010": (1.3174, 1.3112, 1.3236, 47.48, 0.44, 1.12), "011": (1.3124, 1.3062, 1.3186, 47.87, 0.49, 1.17),
                     "100": (1.3170, 1.3108, 1.3232, 47.51, 0.46, 1.16), "101": (1.3121, 1.3059, 1.3183, 47.89, 0.47, 1.19),
                     "110": (1.3110, 1.3048, 1.3172, 47.98, 0.52, 1.21), "111": (1.3061, 1.2999, 1.3123, 48.36, 0.61, 1.29)},
}


def factorial_set():
    rows = []
    units = {"ld_integrated_loss": "nats", "interval_low": "nats", "interval_high": "nats", "official_metric": "%",
             "activity": "fraction", "relative_time": "relative to cell 000"}
    for dataset, cells in FACTORIAL_CELLS.items():
        for cell, values in cells.items():
            for quantity, value in zip(units, values):
                rows.append({"dataset": dataset, "cell": cell, "quantity": quantity, "value": value,
                             "unit": units[quantity]})
    return records("factorials", rows)


def write_campaign(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "study.json").write_text(json.dumps({"schema": "synthetic campaign stub"}), encoding="utf-8")
    return root


def status_counts(rows) -> dict:
    out = {}
    for row in rows:
        out[row["status"]] = out.get(row["status"], 0) + 1
    return out


def lookup(rows, path: str) -> dict:
    return next(row for row in rows if row["path"] == path)


def data_rows(out_dir: Path, stem: str, name: str) -> list[dict]:
    with (out_dir / "data" / stem / f"{name}.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class World(unittest.TestCase):
    """One synthetic world shared by the tests: seven runs (Roman-empire as a two-split campaign) and every set."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.runs = make_runs(cls.tmp)
        cls.replay, budget = budget_world()
        grid, grid_scores = grid_sets()
        cls.records = {
            "baselines": baselines_set(), "input_gate": gate_set(), "ensemble_grid": grid,
            "ensemble_grid_scores": grid_scores, "budget_matched": budget, "factorials": factorial_set(),
            "sensor_parameters": records("sensor_parameters", [
                {"dataset": d, "quantity": q, "value": v} for d, (k, p, f) in {
                    "ogbn-arxiv": (0, 1, 0.21), "ogbn-proteins": (32, 4, 0.061), "Roman-empire": (0, 1, 0.27),
                    "PascalVOC-SP": (64, 2, 0.24), "COCO-SP": (0, 1, 0.25), "Peptides-func": (0, 1, 0.19)}.items()
                for q, v in (("sketch_dim", k), ("sensing_period", p), ("residual_fraction", f))]),
            "nominal_time": records("nominal_time", [
                {"dataset": d, "quantity": "t_nom_seconds", "value": v} for d, v in {
                    "ogbn-arxiv": 9.54, "ogbn-proteins": 9.69, "Roman-empire": 1.16, "PascalVOC-SP": 0.00073,
                    "COCO-SP": 0.00073, "Peptides-func": 0.00056}.items()]),
            "timing_anchors": records("timing_anchors", [
                {"dataset": d, "quantity": q, "value": v} for d, anchors in {
                    "ogbn-arxiv": (14.89, 4.32, 8.64), "ogbn-proteins": (9.22, 7.20, 14.40), "Roman-empire": (1.40, 0.675, 1.35),
                    "PascalVOC-SP": (None, 0.00052, 0.00104), "COCO-SP": (None, 0.00005, 0.00009),
                    "Peptides-func": (None, 0.00040, 0.000794)}.items()
                for q, v in zip(("throughput_anchor_seconds", "training_anchor_low_seconds",
                                 "training_anchor_high_seconds"), anchors) if v is not None]),
        }
        cls.campaign = write_campaign(cls.tmp / "campaign")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def ctx(self, *names, runs=True, campaign=False):
        return context(runs=self.runs if runs else {}, records={n: self.records[n] for n in names},
                       campaigns={"budget": self.campaign} if campaign else {})

    def build(self, display_id, ctx):
        out = Path(tempfile.mkdtemp(dir=self.tmp))
        with mock.patch("cm_campaigns.arm_campaign.check_campaign", return_value=self.replay):
            writer, rows = build_one(display_id, ctx, out, registry=REGISTRY)
        return out, writer, rows

    def assertLayout(self, out, stem, rows, columns):
        table = read_table(out, stem)
        self.assertEqual(len(table), rows)
        self.assertEqual(len(table[0]), columns)
        for suffix in ("csv", "md", "tex"):
            self.assertTrue((out / "tables" / f"{stem}.{suffix}").is_file())
        self.assertTrue(list((out / "data" / stem).glob("*.csv")), f"{stem}: no source data")
        return table

    def assertResolved(self, rows):
        unresolved = [row for row in rows if row["status"] == "no_reference"]
        self.assertEqual(unresolved, [], unresolved[:3])


# ---------------------------------------------------------------------------
# Run-ledger tables
# ---------------------------------------------------------------------------

class LedgerTables(World):

    def test_s23_ledger_and_campaign_mean(self):
        out, writer, rows = self.build("supplementary:Table S23", self.ctx("input_gate", "baselines"))
        table = self.assertLayout(out, "S23", 7, 12)
        arxiv = table[0]
        # C_exp = (1 - 0.118) 4.7 + 0.118 x 4.0 = 4.6174; C_assure = 8.7
        self.assertEqual((arxiv["route/E"], arxiv["expected serial cost"], arxiv["assurance serial cost"]),
                         ("radius/4", "4.617", "8.700"))
        self.assertEqual(arxiv["Lean effect"], "0.930")
        products = table[2]
        self.assertEqual((products["route/E"], products["controller cost"], products["fallback probability"],
                          products["expected serial cost"], products["wall-time overhead"]),
                         ("qC/phase/2", "0.000", "— (global return)", "2.100", "0%"))
        roman = table[3]     # two splits with controller costs 5.3 and 5.5
        self.assertEqual((roman["controller cost"], roman["assurance serial cost"]), ("5.400", "9.400"))
        self.assertResolved(rows)
        self.assertEqual(status_counts(rows).get("differs", 0), 0, [r for r in rows if r["status"] != "exact"])
        self.assertEqual(len(data_rows(out, "S23", "runs")), 8)      # six runs plus two Roman splits

    def test_s6c_sensing_identity(self):
        out, _, rows = self.build("supplementary:Table S6c", self.ctx("sensor_parameters"))
        table = self.assertLayout(out, "S6c", 6, 8)
        proteins = table[1]
        r = 39561252 / (132534 * 512)
        share = 8 * r / (1 + r)
        self.assertEqual(proteins["r"], f"{r:.3f}")
        self.assertEqual(proteins["residual"], f"{1.2 - 1 - share:+.3f}")
        self.assertEqual(proteins["admissible (k/d)/Ps"], f"{0.2 / share:.3f} (reduced sensor)")
        self.assertEqual(proteins["executed sensor (k, Ps, f_res)"], "32, 4, 0.061")
        self.assertResolved(rows)
        self.assertEqual(status_counts(rows).get("differs", 0), 0, [r for r in rows if r["status"] != "exact"])

    def test_s15_layout_routes_and_ranges(self):
        out, _, rows = self.build("supplementary:Table S15", self.ctx("baselines", "input_gate"))
        table = self.assertLayout(out, "S15", 63, 12)
        arxiv = table[:9]
        self.assertEqual([r["method"] for r in arxiv],
                         ["full-depth plant", "best fixed depth", "optimized phase", "best fixed sequence (qC)", "MPC-OL",
                          "radius (deployed)", "MPC-RH", "AMP", "ADMP-GNN"])
        self.assertEqual(arxiv[1]["Cctrl"], "2.0–4.0")         # best-depth configurations cost 2.0 and 4.0 at E = 4
        self.assertEqual(arxiv[4]["Cctrl"], "4.0–4.6")
        self.assertEqual(arxiv[1]["score"], "73.10")            # mean of the deposited per-seed scores
        self.assertEqual((arxiv[5]["Cexp"], arxiv[5]["Cassure"]), ("4.62", "8.7"))
        self.assertEqual(arxiv[6]["Cexp"], "= Cctrl")
        self.assertEqual(arxiv[0]["SD (12 seeds)"], "0.18–0.25")
        products = table[18:27]
        self.assertEqual(products[2]["method"], "optimized phase (qC, returned)")
        self.assertEqual(products[2]["family"], "CROSS-MORSE deployment")
        self.assertEqual(table[54]["score"], "0.704")           # Peptides scores in AP units
        self.assertResolved(rows)
        self.assertEqual(lookup(rows, "ogbn-arxiv/rows/1/C_ctrl")["status"], "exact")
        self.assertEqual(lookup(rows, "ogbn-arxiv/plant_params_M")["status"], "exact")

    def test_s15_declared_ranges_stay_separate_from_observed_costs(self):
        first, _, first_comparisons = self.build("supplementary:Table S15", self.ctx("baselines", "input_gate"))
        declared = data_rows(first, "S15", "declared_reference_ranges")
        self.assertEqual(len(declared), 19)
        amp = next(row for row in declared if row["method_or_route"] == "AMP"
                   and row["quantity"] == "serial_cost_per_member")
        self.assertEqual((amp["low"], amp["high"], amp["basis"]), ("1.05", "1.2", "declared reference range"))
        reference_checks = [row for row in first_comparisons if row["group"] == rc.G_S15_REFERENCE]
        self.assertEqual(len(reference_checks), 19)
        self.assertEqual({row["status"] for row in reference_checks}, {"exact"})

        changed = self.ctx("baselines", "input_gate")
        reference = changed.constants["baseline_reference_ranges_table_S15"]
        reference["external_methods"]["AMP"]["serial_cost_per_member"] = [7.0, 8.0]
        reference["external_methods"]["AMP"]["parameter_multiplier"] = [3.0, 4.0]
        reference["internal_routes"]["best_fixed_depth"]["serial_cost_per_member"] = [0.2, 0.3]
        second, _, second_comparisons = self.build("supplementary:Table S15", changed)
        for relative in ("tables/S15.csv", "data/S15/methods.csv"):
            self.assertEqual((first / relative).read_bytes(), (second / relative).read_bytes())
        self.assertNotEqual(declared, data_rows(second, "S15", "declared_reference_ranges"))
        self.assertEqual([r for r in first_comparisons if r["group"] == rc.G_S15],
                         [r for r in second_comparisons if r["group"] == rc.G_S15])
        differences = [r for r in second_comparisons if r["group"] == rc.G_S15_REFERENCE and r["status"] == "differs"]
        self.assertEqual(len(differences), 3)
        self.assertIn("accounting references, not observed per-rerun costs", (second / "tables/S15.md").read_text())

    def test_s15_rejects_malformed_declared_reference_range(self):
        changed = self.ctx("baselines", "input_gate")
        changed.constants["baseline_reference_ranges_table_S15"]["retraining_multiplier"] = [1.6, 1.1]
        with self.assertRaisesRegex(DisplayInputError, "retraining_multiplier.*ordered range"):
            self.build("supplementary:Table S15", changed)

    def test_s15b_from_baselines(self):
        out, _, rows = self.build("supplementary:Table S15b", self.ctx("baselines"))
        table = self.assertLayout(out, "S15b", 27, 6)
        self.assertEqual([r["method"] for r in table[8:11]], ["optimized phase (qC)", "ADMP-GNN", "GRAMA"])
        self.assertEqual(table[1]["SD (12 seeds)"], "0.18")
        self.assertEqual(table[-1]["SD (12 seeds)"], "0.52")     # Peptides SD in AP percentage points
        self.assertResolved(rows)
        self.assertEqual(status_counts(rows).get("differs", 0), 0, [r for r in rows if r["status"] != "exact"])

    def test_bundled_example_runs(self):
        """The real run layout: both bundled runs returned their comparator."""
        runs = {d: RunView(d, ROOT / "examples/runs" / ("peptides_func" if d in rc.GRAPH_LEVEL else "arxiv"))
                for d in DATASETS}
        gate = gate_set()
        ledger = rc.run_ledger(context(runs=runs, records={"input_gate": gate}), "ogbn-arxiv", gate)
        self.assertTrue(ledger["returned"])
        self.assertEqual(ledger["qc_route"], "learned_stop")
        self.assertAlmostEqual(ledger["C_exp"], ledger["C_qC"])
        out = Path(tempfile.mkdtemp(dir=self.tmp))
        writer, _ = build_one("supplementary:Table S15", context(runs=runs, records={"baselines": baselines_set(),
                                                                                     "input_gate": gate}), out)
        table = read_table(out, "S15")
        # node-level example: seven tabulated routes plus learned stop (its qC) and two baselines; graph-level
        # example: its qC route (phase) is tabulated, so seven routes and two baselines
        self.assertEqual(len(table), 4 * 10 + 3 * 9)
        self.assertIn("learned stop (qC, returned)", [r["method"] for r in table[:10]])


class ResourceTables(World):

    def test_s23b_amortization(self):
        out, _, rows = self.build("supplementary:Table S23b",
                                  self.ctx("input_gate", "baselines", "nominal_time", "timing_anchors"))
        table = self.assertLayout(out, "S23b", 6, 8)
        arxiv = table[0]
        # M* t_nom = 3600 (12.8 - 10.88) / (0.1 x 0.6174) = 111,953 input-s; M* = 111,953 / 9.54
        self.assertEqual(arxiv["M* tnom (10^3 input·s)"], "112")
        self.assertEqual(arxiv["M* (inputs) [range]"], "1.2 × 10^4 [7.5 × 10^3, 2.6 × 10^4]")
        self.assertEqual(arxiv["Wnom (MAC)"], "1.15 × 10^13")
        coco = table[4]
        self.assertEqual(coco["anchor"], "shares PascalVOC-SP (same backbone and graph size)")
        self.assertEqual(coco["tnom [range]"], "0.73 ms [0.52–1.04 ms]")
        self.assertIn("own quotient", coco["training-time anchor"])
        self.assertEqual(table[3]["Wnom (MAC)"], "— (not used; see caption)")
        self.assertResolved(rows)
        exact = {r["path"] for r in rows if r["status"] == "exact"}
        self.assertTrue({"ogbn-arxiv/M_star_inputs", "ogbn-arxiv/W_nom_MAC", "COCO-SP/t_nom_reference_range_ms"} <= exact)

    def test_s26b_law_and_break_even(self):
        out, _, rows = self.build("supplementary:Table S26b", self.ctx("input_gate", "ensemble_grid"))
        table = self.assertLayout(out, "S26b", 6, 8)
        self.assertEqual(table[0]["D_inf [band]"], "0.0305 [0.0261, 0.0348]")
        self.assertEqual(table[0]["break-even at assurance overhead"], "0.0035–0.0047")
        self.assertEqual(table[0]["extra checkpoints for E = 16, 20 (GPU-h)"], "8 (14.4)")
        self.assertEqual(table[5]["break-even at expected overhead (kappa in [0.03, 0.04])"], "0.0003–0.0003")
        self.assertResolved(rows)
        self.assertEqual(status_counts(rows).get("differs", 0), 0, [r for r in rows if r["status"] != "exact"])

    def test_s26_grid_rows_and_verdicts(self):
        out, _, _ = self.build("supplementary:Table S26", self.ctx("input_gate", "ensemble_grid", "ensemble_grid_scores"))
        table = self.assertLayout(out, "S26", 34, 8)
        arxiv = table[:6]
        self.assertEqual([r["configuration"] for r in arxiv],
                         ["qC: fixed sequence", "deployed radius at expected cost", "qC at matched expected cost",
                          "qC at matched assurance cost", "qC at E = 8 (registered grid)", "qC at E = 12 (registered grid)"])
        self.assertEqual(arxiv[2]["break-even threshold (nats)"], "0.0009–0.0012")
        self.assertEqual(arxiv[2]["paired loss effect of deployed rule vs this row (nats)"], "-0.0036 to -0.0033")
        self.assertEqual((arxiv[2]["verdict"], arxiv[3]["verdict"]), ("retained", "break-even"))
        # grid loss gain at E = 12: 0.0305 (1/4 - 1/12) = 0.00508 > 0.0045
        self.assertEqual(arxiv[5]["paired loss effect of deployed rule vs this row (nats)"], "+0.0006")
        self.assertEqual(arxiv[5]["verdict"], "comparator ahead at this size; cost 2.6× the deployed expected cost")
        self.assertEqual(table[18]["configuration"], "qC: MPC-OL H = 16")
        # score at E': grid score at E + (score gain per nat between E = 4 and E = 8) x threshold
        thresholds = [k * 0.87 * (1 / 4 - 1 / 4.6174) for k in (0.03, 0.04)]
        expected = [f"{73.15 + SCORE_PER_NAT * t:.2f}" for t in thresholds]
        self.assertEqual(arxiv[2]["official score"], "–".join(expected))      # both ends print, as in the paper


# ---------------------------------------------------------------------------
# Record-set tables
# ---------------------------------------------------------------------------

class RecordTables(World):

    def test_s16_and_s16b_factorials(self):
        out, _, rows = self.build("supplementary:Table S16", self.ctx("factorials", runs=False))
        table = self.assertLayout(out, "S16", 8, 9)
        self.assertEqual([r["cell"] for r in table], ["000", "100", "010", "001", "110", "101", "011", "111"])
        self.assertEqual(table[0]["95% interval"], "[0.8660, 0.8746]")
        self.assertResolved(rows)
        self.assertEqual(lookup(rows, "effect_coded_main_effects_nats/half_width_95")["status"], "missing")
        self.assertEqual(status_counts(rows).get("differs", 0), 0, [r for r in rows if r["status"] != "exact"])
        effects = {r["contrast"] + "/" + r["kind"]: float(r["value_nats"]) for r in data_rows(out, "S16", "contrasts")}
        self.assertAlmostEqual(effects["A/effect-coded"], -0.0086)
        self.assertAlmostEqual(effects["AOX/baseline-anchored"], 0.0090)
        out, _, rows = self.build("supplementary:Table S16b", self.ctx("factorials", runs=False))
        table = self.assertLayout(out, "S16b", 8, 9)
        self.assertEqual([r["cell"] for r in table][:4], ["000", "001", "010", "011"])
        main = [lookup(rows, f"effect_coded_main_effects_nats/{k}")["status"]
                for k in ("spectral_information", "conditioned_observer", "risk_controlled_execution", "half_width_95")]
        self.assertEqual(main, ["exact"] * 3 + ["missing"])

    def test_s19_negative_controls(self):
        rows = [{"control": c, "quantity": q, "value": v, "unit": u}
                for c, values in {"graph irrelevant": (0.028, 0.963, 0.0018), "misleading graph": (0.107, 0.824, 0.0093),
                                  "sensor bias": (0.168, 0.748, 0.0104), "support shift": (0.064, 0.907, 0.0062)}.items()
                for q, v, u in zip(("residual_activity", "input_fallback", "worst_pre_fallback_harm"), values,
                                   ("fraction", "fraction", "CE"))]
        out, _, compared = self.build("supplementary:Table S19", context(records={"negative_controls": records(
            "negative_controls", rows)}))
        table = self.assertLayout(out, "S19", 4, 6)
        self.assertEqual([r["control"] for r in table], ["graph irrelevant", "misleading graph", "sensor bias", "support shift"])
        self.assertEqual((table[1]["input fallback"], table[1]["worst pre-fallback harm"]), ("0.824", "0.0093 CE"))
        self.assertEqual(table[0]["construction"], "Y ⟂ A | X on independent realizations")
        self.assertEqual(table[0]["expected conclusion"], "comparator equivalence and nonvacuous fallback")
        self.assertEqual(len(compared), 24)
        self.assertEqual({row["status"] for row in compared}, {"exact"})

        # An altered observation and its unit stay visible; the source targets cannot overwrite them.
        altered = [dict(row) for row in rows]
        for row in altered:
            if row["control"] == "misleading graph" and row["quantity"] == "worst_pre_fallback_harm":
                row.update(value=0.02, unit="nats")
        out, _, compared = self.build("supplementary:Table S19", context(records={"negative_controls": records(
            "negative_controls", altered)}))
        self.assertEqual(read_table(out, "S19")[1]["worst pre-fallback harm"], "0.0200 nats")
        raw = next(row for row in data_rows(out, "S19", "controls") if row["control"] == "misleading graph")
        self.assertEqual((raw["worst_pre_fallback_harm"], raw["harm_unit"]), ("0.02", "nats"))
        self.assertEqual(lookup(compared, "misleading graph/worst_pre_fallback_harm")["status"], "differs")
        self.assertEqual(lookup(compared, "misleading graph/harm_unit")["status"], "differs")
        self.assertEqual(lookup(compared, "misleading graph/construction")["status"], "exact")

    def test_s25_cross_backbone(self):
        rows = []
        for (regime, model), values in zip(rc.RECORD_SETS["cross_backbone"]["coverage_rows"],
                                           [(-0.00463, 4, 5), (-0.00371, 3, 5), (-0.00664, 4, 5), (-0.00513, 3, 5),
                                            (-0.00916, 4, 5), (-0.00721, 3, 5)]):
            rows += [{"regime": regime, "model": model, "quantity": q, "value": v}
                     for q, v in zip(("effect_nats", "route_stable_folds", "outer_folds"), values)]
        out, _, _ = self.build("supplementary:Table S25", context(runs=self.runs, records={
            "cross_backbone": records("cross_backbone", rows)}))
        table = self.assertLayout(out, "S25", 3, 5)
        self.assertEqual(table[0]["route stability"], "radius in 4/5 versus 3/5 outer folds")
        self.assertEqual(table[1]["route stability"], "MPC-RH in 4/5 versus 3/5 outer folds")
        self.assertEqual(table[0]["paired effect vs qC"], "-0.00463 versus -0.00371 CE")
        self.assertEqual(table[2]["model B"], "frozen GINE/GRAMA-style; same adapter interface")

    def test_s37_counts(self):
        out, _, rows = self.build("supplementary:Table S37", self.ctx("input_gate"))
        table = self.assertLayout(out, "S37", 10, 5)
        pascal = table[4]
        self.assertEqual((pascal["numerator"], pascal["denominator"], pascal["rate"]), ("116", "1429", "0.081 (main Table 1)"))
        self.assertIn("buffered cores or splits", table[0]["rate (unit)"])
        self.assertResolved(rows)
        self.assertEqual(status_counts(rows).get("differs", 0), 0, [r for r in rows if r["status"] != "exact"])

    def test_s45_inserted_site(self):
        values = {"PascalVOC-SP": ((0.0021, 0.0005), (0.0, 0.0), (-0.0018, 0.00046), (-0.0084, 0.0022)),
                  "COCO-SP": ((0.0022, 0.00015), (0.0, 0.0), (-0.0020, 0.000485), (-0.0091, 0.0019)),
                  "Peptides-func": ((0.0006, 0.00034), (0.0, 0.0), (-0.0011, 0.00028), (-0.0030, 0.0009))}
        rows = [{"dataset": d, "arm": arm, "quantity": q, "value": v} for d, arms in values.items()
                for arm, pair in zip(rc.INSERTED_SITE_ARMS, arms)
                for q, v in zip(("effect_nats", "standard_error"), pair)]
        self.records["inserted_site"] = records("inserted_site", rows)
        out, _, comparisons = self.build("supplementary:Table S45", self.ctx("inserted_site", runs=False))
        table = self.assertLayout(out, "S45", 3, 8)
        self.assertEqual(table[0]["native-site fraction"], "0.047")      # 6 native of 128 controlled sites
        self.assertEqual(table[2]["native-site fraction"], "0.125")
        self.assertEqual(table[0]["arm 2: inserted, open loop (= qC; identity)"], "0 (identity)")
        self.assertEqual(table[0]["arm 1': native, feedback [range]"], "-0.0018 [-0.0027, -0.0009]")
        self.assertResolved(comparisons)


# ---------------------------------------------------------------------------
# Budget-matched campaign
# ---------------------------------------------------------------------------

class BudgetTables(World):

    def test_s46_frontier(self):
        out, _, rows = self.build("supplementary:Table S46", self.ctx("input_gate", "budget_matched", campaign=True))
        table = self.assertLayout(out, "S46", 6, 8)
        arxiv = table[0]
        self.assertEqual(arxiv["E3 predictor"], "-1.6 [-0.9, -2.5]")
        self.assertEqual(arxiv["E6 qB: effect; overhead [range]"], "-4.4; 6% [5, 11]")
        self.assertEqual((arxiv["ceiling"], arxiv["feedback effect"], arxiv["present overhead"]), ("1.15", "-4.5", "15%"))
        self.assertEqual(table[1]["E3 predictor"], "-1.3 [—]")           # only the selected configuration deposited
        self.assertResolved(rows)
        self.assertEqual(lookup(rows, "ogbn-arxiv/E3_range")["status"], "exact")

    def test_s46b_arms(self):
        out, _, rows = self.build("supplementary:Table S46b", self.ctx("budget_matched", runs=False, campaign=True))
        table = self.assertLayout(out, "S46b", 24, 6)
        self.assertEqual(table[0]["effect vs qM [95% interval]"], "+2.9 [+1.0, +4.8]")
        self.assertEqual(table[3]["effect vs qC [grid range]"], "-4.4; overhead 6%")
        self.assertEqual(table[12]["Holm P"], "< 1e-4")
        self.assertEqual(table[12]["arm"], "E3 predictor")
        self.assertEqual(table[3]["work / ceiling"], "0.92")
        self.assertResolved(rows)
        self.assertEqual(lookup(rows, "PascalVOC-SP/arms/E3/holm_P")["status"], "exact")

    def test_replay_disagreement_and_invalid_campaign(self):
        replay = json.loads(json.dumps(self.replay))
        replay["summary"][0]["effect_vs_qC"] += 1e-4
        ctx = self.ctx("budget_matched", runs=False, campaign=True)
        with mock.patch("cm_campaigns.arm_campaign.check_campaign", return_value=replay):
            with self.assertRaisesRegex(DisplayValidationError, "strict campaign replay"):
                build_one("supplementary:Table S46b", ctx, Path(tempfile.mkdtemp(dir=self.tmp)))
        with self.assertRaises(DisplayValidationError):      # the stub study.json is not a campaign export
            build_one("supplementary:Table S46b", self.ctx("budget_matched", runs=False, campaign=True),
                      Path(tempfile.mkdtemp(dir=self.tmp)))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

class Figures(World):

    def assertFigure(self, out, stem, panels):
        for suffix in ("png", "svg"):
            self.assertTrue((out / "figures" / f"{stem}.{suffix}").is_file())
        names = sorted(p.stem for p in (out / "data" / stem).glob("*.csv"))
        for panel in panels:
            self.assertIn(panel, names)
        return names

    def s1_source_context(self):
        """Separate software fixtures from the independently inspected main Table 1 (PDF p10).

        The older WORLD fixture is intentionally unchanged. These inputs test
        source precision and roles; they are not deposited benchmark evidence.
        """
        source_intervals = {
            "ogbn-arxiv": ((.07, .01, .13), (-.00450, -.0075, -.0015)),
            "ogbn-proteins": ((.11, .03, .19), (-.00380, -.0068, -.0008)),
            "ogbn-products": ((0., 0., 0.), (0., 0., 0.)),
            "Roman-empire": ((.26, .10, .42), (-.00572, -.0089, -.0026)),
            "PascalVOC-SP": ((.60, .38, .82), (-.00842, -.0128, -.0041)),
            "COCO-SP": ((.65, .42, .88), (-.00905, -.0127, -.0054)),
            "Peptides-func": ((.58, .09, 1.07), (-.00303, -.0047, -.0013)),
        }
        runs = {}
        for dataset, original in self.runs.items():
            evidence = json.loads(json.dumps(original.evidence))
            gain, loss = source_intervals[dataset]
            evidence["final"]["paired_metric_pp"] = dict(zip(("mean", "lower", "upper"), gain))
            evidence["final"]["paired_loss"] = dict(zip(("mean", "lower", "upper"), loss))
            runs[dataset] = RunView(dataset, original.root, splits=original.splits, evidence=evidence)
        return context(runs=runs, records={"input_gate": self.records["input_gate"]})

    def test_figure_s1(self):
        out, _, rows = self.build("supplementary:Figure S1", self.ctx("input_gate"))
        self.assertFigure(out, "FigS1", ["panel_a_endpoint_effect", "panel_b_loss_effect", "panel_c_mechanism"])
        panel_c = {r["dataset"]: r for r in data_rows(out, "FigS1", "panel_c_mechanism")}
        self.assertEqual(panel_c["ogbn-products"]["mechanism"], "comparator/phase")
        self.assertEqual(float(panel_c["ogbn-products"]["fallback_fraction"]), 1.0)
        peptides = next(r for r in data_rows(out, "FigS1", "panel_a_endpoint_effect") if r["dataset"] == "Peptides-func")
        self.assertAlmostEqual(float(peptides["effect"]), 0.0058)       # AP gain in AP units
        loss = next(r for r in data_rows(out, "FigS1", "panel_b_loss_effect") if r["dataset"] == "Peptides-func")
        self.assertEqual(loss["loss"], "BCE")       # native AP never labels the loss interval
        self.assertResolved(rows)

    def test_figure_s1_full_contract_covers_all_panels_and_identity_roles(self):
        ctx = self.s1_source_context()
        out = Path(tempfile.mkdtemp(dir=self.tmp))
        result = fw.build_displays(ctx, out, only=["supplementary:Figure S1"])["displays"][0]
        self.assertEqual(result["status"], "built")
        self.assertTrue(result["content_complete"])
        self.assertTrue(result["comparison_complete"])
        self.assertEqual(result["comparison_coverage"]["expected_values"], 135)
        self.assertEqual(result["comparison"]["differs"], 0)
        roles = {r["dataset"]: r for r in data_rows(out, "FigS1", "panel_c_roles")}
        self.assertEqual(roles["ogbn-products"]["fallback_role"], "global comparator return")
        self.assertEqual(roles["ogbn-arxiv"]["fallback_role"], "input comparator fallback")
        identity = next(r for r in data_rows(out, "FigS1", "panel_a_endpoint_effect") if r["dataset"] == "ogbn-products")
        self.assertEqual((identity["low"], identity["high"]), ("", ""))
        self.assertEqual(identity["interval_kind"], "comparator identity; no statistical interval")
        observed = next(r for r in data_rows(out, "FigS1", "observed_intervals")
                        if r["dataset"] == "ogbn-products" and r["panel"] == "a")
        self.assertEqual(float(observed["observed_low"]), 0.0)   # original supplied interval is retained

    def test_figure_s1_omitted_interval_comparison_cannot_pass_full_coverage(self):
        _, writer, _ = self.build("supplementary:Figure S1", self.s1_source_context())
        omitted = [c for c in writer.comparisons if c.path != ("panels", "a", "rows", "ogbn-arxiv", "low")]
        registry, metadata, _ = load_references(ROOT)
        missing, coverage = fw.comparison_coverage("supplementary:Figure S1", omitted, registry, metadata, scope="measurement")
        self.assertFalse(coverage["complete"])
        self.assertEqual(coverage["missing_values"], 1)
        self.assertEqual(missing[0]["path"], "panels/a/rows/ogbn-arxiv/low")

    def test_figure_s1_source_perturbation_is_visible_in_independent_targets(self):
        ctx = self.s1_source_context()
        ctx.runs()["Peptides-func"].evidence["final"]["paired_metric_pp"]["mean"] = .68
        changed_gate = records("input_gate", [
            {**row, "value": .2} if row["dataset"] == "ogbn-arxiv" and row["quantity"] == "fallback_probability"
            else dict(row) for row in self.records["input_gate"].rows])
        ctx = context(runs=ctx.runs(), records={"input_gate": changed_gate})
        _, _, comparisons = self.build("supplementary:Figure S1", ctx)
        differs = {row["path"] for row in comparisons if row["group"] == rc.G_FIG_S1 and row["status"] == "differs"}
        self.assertIn("panels/a/rows/Peptides-func/effect", differs)
        self.assertIn("panels/c/rows/ogbn-arxiv/fallback_fraction", differs)
        self.assertIn("panels/c/rows/ogbn-arxiv/marker_cost", differs)
        self.assertEqual(REGISTRY[rc.G_FIG_S1]["panels"]["a"]["rows"]["Peptides-func"]["effect"]["$value"], .0058)

    def test_figure_s1_missing_interval_and_absent_empirical_inputs_stay_incomplete(self):
        ctx = self.s1_source_context()
        ctx.runs()["Peptides-func"].evidence["final"]["paired_loss"]["upper"] = None
        out = Path(tempfile.mkdtemp(dir=self.tmp))
        result = fw.build_displays(ctx, out, only=["supplementary:Figure S1"])["displays"][0]
        self.assertFalse(result["content_complete"])
        self.assertFalse(result["comparison_complete"])
        self.assertEqual(result["comparison"]["missing"], 1)
        result = fw.build_displays(context(), out / "absent", only=["supplementary:Figure S1"])["displays"][0]
        self.assertEqual(result["status"], "awaiting_inputs")

    def test_figure_s2(self):
        rows = [{"dataset": "ogbn-products", "instance": f"sub{i}", "quantity": q, "value": v}
                for i, (e, t, m) in enumerate(((120.0, 100.0, 70.0), (0.7, 0.9, 4.2), (13.0, 12.0, 14.5)))
                for q, v in zip(("processed_edges_millions", "wall_time_seconds", "peak_memory_gib"), (e, t, m))]
        out, _, _ = self.build("supplementary:Figure S2", context(records={"runtime_scaling": records("runtime_scaling",
                                                                                                    rows)}))
        self.assertFigure(out, "FigS2", ["panel_a_wall_time", "panel_b_peak_memory"])
        edges = [float(r["processed_edges_millions"]) for r in data_rows(out, "FigS2", "panel_a_wall_time")]
        self.assertEqual(edges, [0.7, 13.0, 120.0])

    def test_figure_s3(self):
        out, _, rows = self.build("supplementary:Figure S3",
                                  self.ctx("input_gate", "ensemble_grid", "budget_matched", campaign=True))
        self.assertFigure(out, "FigS3", ["panel_a_arxiv", "panel_f_peptides"])
        panel = {r["series"]: r for r in data_rows(out, "FigS3", "panel_a_arxiv")}
        self.assertAlmostEqual(float(panel["grid_E8"]["effect_nats"]), -0.0305 / 8)   # l(8) - l(4)
        self.assertAlmostEqual(float(panel["common_ceiling"]["serial_compute"]), 4.6)
        self.assertAlmostEqual(float(panel["E3"]["range_low"]), -0.0025)
        self.assertAlmostEqual(float(panel["deployed_assurance"]["serial_compute"]), 8.7)
        self.assertResolved(rows)

    def test_figure_s4(self):
        out, writer, rows = self.build("supplementary:Figure S4", self.ctx("baselines", "input_gate"))
        self.assertFigure(out, "FigS4", ["panel_arxiv", "panel_roman", "panel_pascal", "observed_costs"])
        arxiv = {r["method"]: r for r in data_rows(out, "FigS4", "panel_arxiv")}
        self.assertAlmostEqual(float(arxiv["radius (deployed)"]["x_plotted"]), 4.6174)
        self.assertAlmostEqual(float(arxiv["best fixed depth"]["x_plotted"]), 3.0)
        self.assertEqual((float(arxiv["AMP"]["cost_low"]), float(arxiv["AMP"]["cost_high"])), (4.2, 4.8))
        missing = {row["location"] for row in writer.missing_content}
        for dataset, methods in rc.FIGURE_S4_ADDITIONAL_METHODS.items():
            self.assertTrue(all(f"panel/{dataset}/{method}" in missing for method in methods))
        self.assertResolved(rows)

    def test_figure_s4_retains_observed_costs_without_changing_declared_ranges(self):
        modified = records("baselines", [
            {**row, "value": 33.0} if row["quantity"] == "serial_cost" and row["dataset"] == "ogbn-arxiv"
            and row["method"] == "AMP" else dict(row) for row in self.records["baselines"].rows])
        ctx = context(runs=self.runs, records={"baselines": modified, "input_gate": self.records["input_gate"]})
        # Other internal routes also use declared ranges, not this measured configuration span.
        with mock.patch.object(rc, "route_cost_range", return_value=(21.0, 22.0)):
            out, _, rows = self.build("supplementary:Figure S4", ctx)
        arxiv = {r["method"]: r for r in data_rows(out, "FigS4", "panel_arxiv")}
        self.assertEqual((float(arxiv["AMP"]["cost_low"]), float(arxiv["AMP"]["cost_high"])), (4.2, 4.8))
        self.assertEqual((float(arxiv["best fixed depth"]["cost_low"]),
                          float(arxiv["best fixed depth"]["cost_high"])), (2.0, 4.0))
        observed = {r["method"]: r for r in data_rows(out, "FigS4", "observed_costs") if r["dataset"] == "ogbn-arxiv"}
        self.assertEqual(float(observed["AMP"]["observed_cost_low"]), 33.0)
        self.assertEqual(float(observed["best fixed depth"]["observed_cost_high"]), 22.0)
        ranges = [r for r in rows if r["group"] == rc.G_S15_REFERENCE]
        self.assertTrue(ranges)
        self.assertTrue(all(r["status"] == "exact" for r in ranges))

    def test_figure_s4_accepts_additional_named_points_without_changing_s15(self):
        # Deliberately arbitrary software fixtures, not values digitized from the PDF.
        base = self.records["baselines"]
        added = []
        ctx = self.ctx("baselines", "input_gate")
        for dataset in rc.FIGURE_S4_DATASETS:
            _, entries = rc.method_rows(ctx, dataset, base, self.records["input_gate"])
            methods = [r["key"] for r in entries if r["score"] is None] + list(rc.FIGURE_S4_ADDITIONAL_METHODS[dataset])
            for method in methods:
                added.extend({"dataset": dataset, "method": method, "seed": str(seed), "quantity": "score", "value": value}
                             for seed, value in enumerate(seeded(60.0, .1), start=1))
        augmented = records("baselines", base.rows + added)
        ctx = context(runs=self.runs, records={"baselines": augmented, "input_gate": self.records["input_gate"]})
        out, writer, _ = self.build("supplementary:Figure S4", ctx)
        self.assertEqual(writer.missing_content, [])
        for dataset, panel in zip(rc.FIGURE_S4_DATASETS, ("arxiv", "roman", "pascal")):
            plotted = {r["method"]: r for r in data_rows(out, "FigS4", f"panel_{panel}")}
            self.assertEqual(len(plotted), 10)
            for method in rc.FIGURE_S4_ADDITIONAL_METHODS[dataset]:
                self.assertAlmostEqual(float(plotted[method]["official_score"]), 60.0)
            _, table_rows = rc.method_rows(ctx, dataset, augmented, self.records["input_gate"])
            self.assertEqual(len(table_rows), 9)    # S15's fixed scientific scope is unchanged.

    def test_declared_cost_range_rejects_missing_or_reversed_source_input(self):
        ctx = context()
        reference = ctx.constants["baseline_reference_ranges_table_S15"]
        reference["external_methods"]["AMP"]["serial_cost_per_member"] = [1.2, 1.05]
        with self.assertRaisesRegex(DisplayInputError, "finite positive ordered range"):
            rc.s15_figure_cost_range(ctx, 4, method="AMP")
        with self.assertRaisesRegex(DisplayInputError, "finite positive ordered range"):
            rc.s15_figure_cost_range(ctx, 4, method="unknown architecture")

    def test_figure_s16(self):
        diag = [{"diagnostic": "coverage", "level": str(h), "quantity": q, "value": v}
                for h, c in ((1, 0.949), (4, 0.951), (8, 0.950), (16, 0.947), (32, 0.944))
                for q, v in (("coverage", c), ("coverage_ci_low", c - 0.006), ("coverage_ci_high", c + 0.006))]
        diag += [{"diagnostic": "observer", "level": v, "quantity": q, "value": x}
                 for v, rmse, nis in (("raw", 0.0119, 0.84), ("EMA", 0.0093, 0.88), ("global KF", 0.0074, 0.89),
                                      ("conditioned KF", 0.0057, 0.95))
                 for q, x in (("risk_rmse", rmse), ("nis_coverage", nis))]
        diag += [{"diagnostic": "harm", "level": d, "quantity": q, "value": v} for d in DATASETS
                 for q, v in (("false_safe_rate", 0.02), ("false_safe_ci_low", 0.01), ("false_safe_ci_high", 0.03),
                              ("harm_miss_rate", 0.07), ("harm_miss_ci_low", 0.05), ("harm_miss_ci_high", 0.09))]
        diag += [{"diagnostic": "stress", "level": "clean", "quantity": q, "value": v}
                 for q, v in (("input_fallback", 0.11), ("residual_activity", 0.60))]
        controls = [{"control": c, "quantity": q, "value": v}
                    for (c,) in rc.RECORD_SETS["negative_controls"]["coverage_rows"]
                    for q, v in (("residual_activity", 0.05), ("input_fallback", 0.9), ("worst_pre_fallback_harm", 0.001))]
        ctx = context(records={"reliability_diagnostics": records("reliability_diagnostics", diag),
                               "negative_controls": records("negative_controls", controls)})
        out, _, _ = self.build("supplementary:Figure S16", ctx)
        self.assertFigure(out, "FigS16", ["panel_a_coverage", "panel_b_observer", "panel_c_harm", "panel_d_stress"])
        observer = {r["variant"]: float(r["risk_rmse_normalized"]) for r in data_rows(out, "FigS16", "panel_b_observer")}
        self.assertAlmostEqual(observer["conditioned KF"], 0.0057 / 0.0119)
        stress = data_rows(out, "FigS16", "panel_d_stress")
        self.assertEqual([r["condition"] for r in stress][:2], ["clean", "graph irrelevant"])
        self.assertAlmostEqual(float(data_rows(out, "FigS16", "target")[0]["value"]), 0.95)

    def test_figure_s18(self):
        inserted = [{"dataset": d, "arm": arm, "quantity": q, "value": v}
                    for d in ("PascalVOC-SP", "COCO-SP", "Peptides-func")
                    for arm, pair in zip(rc.INSERTED_SITE_ARMS, ((0.002, 0.0005), (0.0, 0.0), (-0.002, 0.0005), (-0.008, 0.002)))
                    for q, v in zip(("effect_nats", "standard_error"), pair)]
        sensitivity = [{"contrast": c, "quantity": q, "value": v}
                       for c, values in {"Pascal/RH": (-0.00902, -0.01355, -0.00449, 1428, 0.03, 0.0013, 0.85),
                                         "COCO/RH": (-0.00962, -0.01372, -0.00552, 5000, 0.015, 0.0005, 0.97),
                                         "Peptides/radius": (-0.00172, -0.00316, -0.00028, 2331, 0.071, 0.07, 0.45)}.items()
                       for q, v in zip(("effect_nats", "ci_low_nats", "ci_high_nats", "units_old", "graph_variance_share",
                                        "holm_p", "label_survival_probability"), values)]
        bound = [{"action": f"A{a}", "spectral_scale": str(s), "quantity": "displacement_ratio", "value": 0.1 * a * s}
                 for a in range(1, 8) for s in (0.05, 0.1, 1.0)]
        ctx = context(records={"inserted_site": records("inserted_site", inserted),
                               "confirmation_sensitivity": records("confirmation_sensitivity", sensitivity),
                               "band_bound_tightness": records("band_bound_tightness", bound)})
        out, _, rows = self.build("supplementary:Figure S18", ctx)
        self.assertFigure(out, "FigS18", ["panel_a_attribution", "panel_b_confirmation_size", "panel_c_bound_tightness"])
        # two-way variance model: half-width x sqrt(0.03 x 1428/425 + 0.97) reproduces the D_confirm interval of S14b
        self.assertEqual(lookup(rows, "0/interval")["status"], "exact")
        self.assertEqual(lookup(rows, "2/interval")["status"], "exact")
        self.assertEqual(len(data_rows(out, "FigS18", "panel_c_bound_tightness")), 21)
        self.assertResolved(rows)


# ---------------------------------------------------------------------------
# Declarations and contracts
# ---------------------------------------------------------------------------

class Declarations(unittest.TestCase):

    MINE = ("S6c", "S15", "S15b", "S16", "S16b", "S19", "S23", "S23b", "S25", "S26", "S26b", "S37", "S45", "S46",
            "S46b", "FigS1", "FigS2", "FigS3", "FigS4", "FigS16", "FigS18")

    def test_generators_and_declared_inputs(self):
        mine = {g.stem: g for g in GENERATORS.values() if g.function.__module__.endswith("records_campaigns")}
        self.assertEqual(sorted(mine), sorted(self.MINE))
        manifest = {row["id"] for row in json.loads((ROOT / "paper/display_manifest.json").read_text())["displays"]}
        for stem, generator in mine.items():
            self.assertIn(generator.display_id, manifest)
            for need in generator.needs:
                kind, _, name = need.partition(":")
                if kind == "records":
                    self.assertIn(name, SPECS, need)
                elif kind == "run":
                    self.assertIn(name, DATASETS)
                else:
                    self.assertIn(kind, ("runs", "campaign"))

    def test_new_contracts(self):
        fields = {"title", "manuscript_tables", "experiments", "keys", "quantities", "table", "notes"}
        for name in ("timing_anchors", "ensemble_grid_scores", "runtime_scaling", "reliability_diagnostics",
                     "confirmation_sensitivity", "band_bound_tightness"):
            spec = contract(name)
            self.assertLessEqual(fields, set(spec), name)
            self.assertEqual(spec["table"], "list")
            self.assertTrue(all(isinstance(v, str) and v for v in spec["quantities"].values()))


if __name__ == "__main__":
    unittest.main()
