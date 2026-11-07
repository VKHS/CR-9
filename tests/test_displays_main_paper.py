"""Main-paper displays (Tables 1-2, Figures 2-6) on synthetic runs, record sets and campaigns.

The bundled example runs are the realistic fixtures for the table layouts. Minimal
synthetic runs with hand-chosen values exercise the computations: every value
below is a software fixture, never a benchmark observation.
"""
from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from cm_campaigns.contract import ADMITTED, EXTERNAL
from cm_theory import admission as adm_theory
from cm_theory import inference
from paper.build.displays.framework import DATASETS, GRAPH_LEVEL, SHORT, DisplayInputError, RunView
from paper.build.displays.testing import build_one, context, fake_records, read_table, write_files

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = json.loads((ROOT / "paper/reported_results.json").read_text(encoding="utf-8"))
FEEDBACK = ("one_step", "radius", "mpc_rh")
OPEN_LOOP_MEANS = {"best_depth": -0.0007, "learned_stop": -0.0010, "phase": -0.0014, "fixed_sequence": -0.0023,
                   "mpc_ol": -0.0032}
HALF_WIDTH = 0.00125
L_PLANT = {"ogbn-arxiv": 0.87, "ogbn-proteins": 0.25, "ogbn-products": 0.40, "Roman-empire": 0.25,
           "PascalVOC-SP": 1.30, "COCO-SP": 2.00, "Peptides-func": 0.22}

# Hand-chosen fixture values, one dict per dataset. effect/gain: (mean, lower, upper) of the paired
# test loss (nats) and official metric (points). admission: (D-bar, upper, units).
PLAN = {
    "ogbn-arxiv": dict(qc="fixed_sequence", route="radius", E=4, C_qC=4.0, C_ctrl=4.7, effect=(-0.0045, -0.0075, -0.0015),
                       gain=(0.07, 0.01, 0.13), metric="accuracy", loss="cross_entropy", eps=0.003,
                       admission=(-0.0046, -0.0008, 32), qual=1.92),
    "ogbn-proteins": dict(qc="fixed_sequence", route="radius", E=4, C_qC=4.0, C_ctrl=4.8, effect=(-0.0038, -0.0068, -0.0008),
                          gain=(0.11, 0.03, 0.19), metric="rocauc", loss="binary_cross_entropy", eps=0.0015,
                          admission=(-0.0039, -0.0007, 32), qual=2.01),
    "ogbn-products": dict(qc="phase", route=None, E=2, C_qC=2.1, metric="accuracy", loss="cross_entropy", eps=0.003,
                          qual=2.28),
    "Roman-empire": dict(qc="mpc_ol", horizon=8, route="mpc_rh", E=4, C_qC=4.0, C_ctrl=5.4, metric="accuracy",
                         loss="cross_entropy", eps=0.003, qual=0.69,
                         splits=[dict(effect=(-0.0050, -0.0080, -0.0020), gain=(0.20, 0.05, 0.35), admission=(-0.0060, -0.0030, 1)),
                                 dict(effect=(-0.00644, -0.0094, -0.0034), gain=(0.32, 0.15, 0.49), admission=(-0.0068, -0.0036, 1))]),
    "PascalVOC-SP": dict(qc="mpc_ol", horizon=16, route="mpc_rh", E=8, C_qC=8.0, C_ctrl=10.5, effect=(-0.00842, -0.0128, -0.0041),
                         gain=(0.60, 0.38, 0.82), metric="macro_f1", loss="cross_entropy", eps=0.003,
                         independent=(-0.00836, -0.005782, 1428), qual=1.59),
    "COCO-SP": dict(qc="mpc_ol", horizon=16, route="mpc_rh", E=8, C_qC=8.0, C_ctrl=10.8, effect=(-0.00905, -0.0127, -0.0054),
                    gain=(0.65, 0.42, 0.88), metric="macro_f1", loss="cross_entropy", eps=0.003,
                    independent=(-0.00901, -0.007922, 5000), qual=1.81),
    "Peptides-func": dict(qc="fixed_sequence", route="radius", E=4, C_qC=4.0, C_ctrl=4.9, effect=(-0.00303, -0.0047, -0.0013),
                          gain=(0.58, 0.09, 1.07), metric="ap", loss="binary_cross_entropy", eps=0.0015,
                          independent=(-0.00311, -0.001544, 2331), qual=1.69),
}
FALLBACK = {"ogbn-arxiv": 0.118, "ogbn-proteins": 0.147, "Roman-empire": 0.129, "PascalVOC-SP": 0.081, "COCO-SP": 0.071,
            "Peptides-func": 0.188}
WALL = {"ogbn-arxiv": 13.0, "ogbn-proteins": 14.0, "Roman-empire": 25.0, "PascalVOC-SP": 24.0, "COCO-SP": 27.0,
        "Peptides-func": 15.0}
T_NOM = {"ogbn-arxiv": 9.54, "ogbn-proteins": 9.69, "Roman-empire": 1.16, "PascalVOC-SP": 7.3e-4, "COCO-SP": 7.3e-4,
         "Peptides-func": 5.6e-4}
D_INF = {"ogbn-arxiv": 0.0305, "ogbn-proteins": 0.0088, "Roman-empire": 0.0088, "PascalVOC-SP": 0.0455, "COCO-SP": 0.07,
         "Peptides-func": 0.0077}


COMPLEXITY = {"nominal": 0, "best_depth": 1, "learned_stop": 2, "phase": 3, "fixed_sequence": 4, "mpc_ol": 5,
              "one_step": 6, "radius": 7, "mpc_rh": 8}


def _candidate(index, route, E, cost, **extra):
    return {"candidate_id": f"c{index:03d}", "hash": f"{index:064x}", "behavior_id": f"{index:016x}", "route": route,
            "ensemble": E, "cost": cost, "complexity": COMPLEXITY[route], "is_open_loop": route not in FEEDBACK, "feasible": True,
            "execution_failure": False, "serial_wall_seconds": 0.01 * cost, "qualification_cost": 0.0,
            "profile": "full" if route in FEEDBACK + ("mpc_ol",) else "none",
            "cost_definition": "deployed sparse/dense multiply-add count normalized by nominal E=1", **extra}


def _interval(mean, lower, upper, units=8):
    return {"mean": mean, "lower": lower, "upper": upper, "se": (upper - lower) / 3.92, "units": units,
            "method": "fixture", "valid_interval": True}


def synthetic_run(root: Path, dataset: str, plan: dict, split: dict | None = None, split_index: int = 0) -> RunView:
    """Minimal sealed-run files read by the evidence summary and the main-paper displays."""
    E, route, returned = plan["E"], plan["route"], plan["route"] is None
    split = split or plan
    feedback_means = ({"one_step": -0.0027, "radius": -0.00463, "mpc_rh": -0.00447} if route != "mpc_rh"
                      else {"one_step": -0.0039, "radius": -0.00539, "mpc_rh": -0.00664})
    sequence = [0] * 8 + [4] * 24 + [6] * 32
    candidates = [_candidate(0, "nominal", E, float(E), depth=64), _candidate(1, "nominal", 1, 1.0, depth=64),
                  _candidate(2, "best_depth", E, 0.8 * E, depth=48), _candidate(3, "learned_stop", E, 0.9 * E),
                  _candidate(4, "phase", E, plan["C_qC"] if plan["qc"] == "phase" else 1.02 * E, sequence=sequence),
                  _candidate(5, "fixed_sequence", E, plan["C_qC"] if plan["qc"] == "fixed_sequence" else 1.03 * E,
                             sequence=[1, 4, 4, 0] * 16),
                  _candidate(6, "mpc_ol", E, plan["C_qC"] if plan["qc"] == "mpc_ol" else 1.1 * E, horizon=plan.get("horizon", 4)),
                  _candidate(7, "one_step", E, 1.12 * E), _candidate(8, "radius", E, plan.get("C_ctrl", 1.18 * E), kappa=1.0),
                  _candidate(9, "mpc_rh", E, plan.get("C_ctrl", 1.33 * E) if route == "mpc_rh" else 1.33 * E,
                             horizon=plan.get("horizon", 4))]
    if route == "mpc_rh":
        candidates[8]["cost"] = 1.18 * E
    means = [0.0, 0.0012, *[OPEN_LOOP_MEANS[r] for r in ("best_depth", "learned_stop", "phase", "fixed_sequence", "mpc_ol")],
             *[feedback_means[r] for r in FEEDBACK]]
    deployed = {"radius": 8, "mpc_rh": 9}.get(route)
    if deployed is not None:
        reference = candidates[deployed]
        for factor, other in ((0.65, 1), (0.8, 2), (1.06, 2 * E), (1.08, 3 * E)):
            extra = {k: reference[k] for k in ("kappa", "horizon") if k in reference}
            candidates.append(_candidate(len(candidates), route, other, reference["cost"] * other / E, **extra))
            means.append(feedback_means[route] * factor)
    if returned:
        means = [abs(m) + 0.0012 for m in means]
    qc = {"phase": 4, "fixed_sequence": 5, "mpc_ol": 6}[plan["qc"]]
    lower = [m - HALF_WIDTH for m in means]
    upper = [m + HALF_WIDTH for m in means]
    eps = plan["eps"]
    eligible = [] if returned else [j for j, c in enumerate(candidates) if c["route"] in FEEDBACK and upper[j] < -eps]
    provisional = qc if returned else deployed
    decision = {"qC_index": qc, "comparator_index": qc, "provisional_index": provisional, "Q_s": eligible,
                "eligible_indices": eligible, "epsilon_r": eps, "epsilon_eq": 0.002, "epsilon_metric": 0.001,
                "alpha": 0.025, "max_radius": 0.0032, "bootstrap_replicates": 80, "candidate_count": len(candidates),
                "mean": means, "lower": lower, "upper": upper,
                "joint_feasible": [c["route"] in FEEDBACK for c in candidates]}
    ledger = [{"candidate_id": c["candidate_id"], "route": c["route"], "profile": c["profile"], "ensemble": c["ensemble"],
               "cost": c["cost"], "feasible": True, "loss_name": plan["loss"], "metric_name": plan["metric"],
               "select_loss": 1.0 + means[j], "select_metric": 0.5, "lower": lower[j], "upper": upper[j],
               "eligible": j in eligible} for j, c in enumerate(candidates)]
    effect = (0.0, 0.0, 0.0) if returned else split["effect"]
    gain = (0.0, 0.0, 0.0) if returned else split["gain"]
    contrasts = []
    if not returned:
        contrasts.append({"kind": "state", "candidate_index": deployed, "predecessor_index": qc, "mean": 0.0057 + 0.0001 * split_index,
                          "lower": 0.0031, "upper": 0.0084, "holm_p_value_zero": 0.0098, "passes": True,
                          "practical_margin": 0.002, "beneficial_direction": 1, "available": True})
        kind, predecessor = ("radius", 7) if route == "radius" else ("mpc", 6)
        contrasts.append({"kind": kind, "candidate_index": deployed, "predecessor_index": predecessor, "mean": -0.0065,
                          "lower": -0.0100, "upper": -0.0029, "holm_p_value_zero": 0.015, "passes": True,
                          "practical_margin": 0.002, "beneficial_direction": -1, "available": True})
    independent = plan.get("independent")
    d_bar, d_upper, units = (0.0, 0.0, 8) if returned else (independent or split["admission"])
    admission = {"Gadmit": 0 if returned else 1, "passed": not returned, "deployed_index": qc if returned else deployed,
                 "comparator_index": qc, "confirmed_index": provisional, "candidate_already_comparator": returned,
                 "independent_graph_guarantee": bool(independent), "epsilon_a": eps, "epsilon_metric": 0.001, "clip": 0.1,
                 "loss": _interval(d_bar, d_bar - 0.004, d_upper, units)}
    if independent:
        admission["empirical_bernstein"] = {"mean": d_bar, "upper": d_upper, "units": units, "penalty": d_upper - d_bar,
                                            "sample_variance": 0.0002}
    base = 70.0 if plan["metric"] != "ap" else 70.0
    scores = {"q0": base - 2.0, "qN": base, "qC": base + 0.22, "qD": base + 0.22 + gain[0]}
    models = {name: {"metric": score / 100.0, "metric_percent": score, "metric_name": plan["metric"],
                     "loss_nats": L_PLANT[dataset] - (0.01 if name != "qN" else 0.0), "loss_name": plan["loss"],
                     "ensemble": E} for name, score in scores.items()}
    hours = plan["qual"]
    stage_seconds = {"explore": 0.40, "fit": 0.15, "select": 0.35, "confirm": 0.05, "admit": 0.05}
    files = {
        "run_state.json": {"task": {"dataset": dataset}, "dataset": "synthetic", "profile": "smoke", "scientific_run": False,
                           "status": "COMPLETE"},
        "config.json": {"selection": {"epsilon_r": eps, "epsilon_eq": 0.002}, "admission": {"epsilon_a": eps},
                        "data": {"split_index": split_index}, "training": {"seeds": [1, 2]}},
        "prepared/metadata.json": {"split_unit": "graph" if dataset in GRAPH_LEVEL else "cluster"},
        "select/candidates.json": candidates,
        "select/decision.json": decision,
        "select/candidate_ledger.csv": ledger,
        "select/entry_complete.json": {"headroom": {"mean": -0.025, "lower": -0.03, "upper": -0.02}, "passed": True},
        "confirm/decision.json": {"contrasts": contrasts, "confirmed_index": provisional, "comparator_index": qc,
                                  "provisional_index": provisional, "downgrades": []},
        "admit/decision.json": admission,
        "final/results_summary.json": {
            "benchmark_dataset": dataset, "metric_definition": plan["metric"], "admitted": not returned, "models": models,
            "task": {"prediction_level": "graph" if dataset in GRAPH_LEVEL else "node"},
            "paired_qD_minus_qC_loss": _interval(*effect), "paired_qD_minus_qC_metric_pp": _interval(*gain),
            "statistical_scope": "fixture"},
        "fit/full/audit.json": {"support": {"cells": {"0:0": {"ess": 91.0, "qualified": True},
                                                      "0:1": {"ess": 120.0, "qualified": True},
                                                      "1:0": {"ess": 40.0, "qualified": False}}}},
        "stages/train.json": {"seconds_this_attempt": 10.0 * 3600, "status": "COMPLETE"},
    }
    for stage, share in stage_seconds.items():
        files[f"stages/{stage}.json"] = {"seconds_this_attempt": share * hours * 3600, "status": "COMPLETE"}
    write_files(root, files)
    return RunView(dataset, root)


def synthetic_runs(base: Path) -> dict:
    runs = {}
    for dataset, plan in PLAN.items():
        if "splits" in plan:
            camp = base / "roman"
            splits = [synthetic_run(camp / f"split_{i:02d}", dataset, plan, split, i) for i, split in enumerate(plan["splits"])]
            runs[dataset] = RunView(dataset, camp, splits=splits)
        else:
            runs[dataset] = synthetic_run(base / dataset, dataset, plan)
    return runs


def example_runs(base: Path) -> dict:
    """The bundled example runs mapped onto the seven datasets; Roman as a two-split campaign."""
    runs = {}
    for dataset in DATASETS:
        source = ROOT / ("examples/runs/peptides_func" if dataset in GRAPH_LEVEL else "examples/runs/arxiv")
        if dataset == "Roman-empire":
            camp = base / "roman_example"
            camp.mkdir(parents=True)
            splits = []
            for i in range(2):
                (camp / f"split_{i:02d}").symlink_to(source, target_is_directory=True)
                splits.append(RunView(dataset, camp / f"split_{i:02d}"))
            runs[dataset] = RunView(dataset, camp, splits=splits)
        else:
            runs[dataset] = RunView(dataset, source)
    return runs


# ---------------------------------------------------------------------------- record sets
def input_gate():
    return fake_records("input_gate", [{"dataset": d, "quantity": q, "value": v} for d in FALLBACK
                                       for q, v in (("fallback_probability", FALLBACK[d]),
                                                    ("deployed_wall_overhead_percent", WALL[d]))])


def ensemble_grid():
    return fake_records("ensemble_grid", [{"dataset": d, "ensemble": str(e), "quantity": "loss_nats",
                                           "value": L_PLANT[d] - 0.05 + D_INF[d] / e, "unit": "nats"}
                                          for d in ADMITTED for e in (1, 2, 4, 8, 12, 16, 20)])


S16_CELLS = {"000": 0.8703, "100": 0.8666, "010": 0.8685, "001": 0.8657, "110": 0.8578, "101": 0.8547, "011": 0.8535,
             "111": 0.8445}


def factorials():
    rows = []
    for dataset, offset in (("ogbn-arxiv", 0.0), ("PascalVOC-SP", 0.45)):
        for cell, loss in S16_CELLS.items():
            loss += offset
            for quantity, value, unit in (("ld_integrated_loss", loss, "nats"), ("interval_low", loss - 0.0043, "nats"),
                                          ("interval_high", loss + 0.0043, "nats"), ("official_metric", 72.9, "percent"),
                                          ("activity", 0.3, "fraction"), ("relative_time", 1.1, "relative to cell 000")):
                rows.append({"dataset": dataset, "cell": cell, "quantity": quantity, "value": value, "unit": unit})
    return fake_records("factorials", rows)


def baselines():
    rows = []
    for dataset in DATASETS:
        ap = PLAN[dataset]["metric"] == "ap"
        for k, method in enumerate(EXTERNAL[dataset]):
            rows.append({"dataset": dataset, "method": method, "seed": "", "quantity": "gpu_hours", "value": 12.0 + 2 * k,
                         "unit": "GPU-h"})
            rows.append({"dataset": dataset, "method": method, "seed": "", "quantity": "serial_cost",
                         "value": PLAN[dataset]["E"] * (1.1 + 0.2 * k), "unit": "nominal executions"})
            for seed in (1, 2, 3):
                score = 70.1 + 0.1 * k + 0.01 * seed
                rows.append({"dataset": dataset, "method": method, "seed": str(seed), "quantity": "score",
                             "value": score / 100.0 if ap else score, "unit": "AP" if ap else "%"})
    return fake_records("baselines", rows)


def nominal_time():
    return fake_records("nominal_time", [{"dataset": d, "quantity": "t_nom_seconds", "value": t, "unit": "s"}
                                         for d, t in T_NOM.items()])


ARM_FIXTURE = {"E3": (-0.0016, 0.0029, 0.0010, 0.0048, 0.007, 1.00), "E4": (-0.0014, 0.0031, 0.0012, 0.0050, 0.005, 1.00),
               "E5": (-0.0013, 0.0032, 0.0013, 0.0051, 0.005, 1.00), "E6": (-0.0044, 0.0001, -0.0018, 0.0020, 0.92, 0.92)}


def budget_matched():
    rows = []
    for dataset in ADMITTED:
        ceiling = round(PLAN[dataset].get("C_ctrl", 4.6) / PLAN[dataset]["C_qC"] - 0.03, 2)
        for arm, (vs_qc, vs_fb, low, high, p, work) in ARM_FIXTURE.items():
            values = {"effect_vs_qc_nats": vs_qc, "effect_vs_feedback_nats": vs_fb, "effect_vs_feedback_ci_low_nats": low,
                      "effect_vs_feedback_ci_high_nats": high, "p_holm": p, "work_ratio": work * ceiling, "ceiling": ceiling}
            rows += [{"dataset": dataset, "arm": arm, "configuration": "selected", "quantity": q, "value": v}
                     for q, v in values.items()]
    return fake_records("budget_matched", rows)


def tube_audit():
    rows = []
    for dataset, (p95, bound) in {"ogbn-arxiv": (0.0065, 0.018), "Roman-empire": (0.0090, 0.025),
                                  "PascalVOC-SP": (0.0078, 0.022), "COCO-SP": (0.0082, 0.024),
                                  "Peptides-func": (0.0049, 0.011)}.items():
        rows += [{"dataset": dataset, "quantity": "observed_error_p95_nats", "value": p95, "unit": "nats"},
                 {"dataset": dataset, "quantity": "observed_error_median_nats", "value": 0.7 * p95, "unit": "nats"},
                 {"dataset": dataset, "quantity": "final_bound_nats", "value": bound, "unit": "nats"}]
    return fake_records("tube_audit", rows, spec=_contract("tube_audit"))


def mechanism_contrasts():
    rows = []
    for dataset in ("ogbn-arxiv", "ogbn-proteins", "ogbn-products", "Peptides-func"):
        for quantity, value in (("effect_nats", -0.0002), ("interval_low_nats", -0.0012), ("interval_high_nats", 0.0007)):
            rows.append({"dataset": dataset, "contrast": "rh_minus_ol", "quantity": quantity, "value": value, "unit": "nats"})
    return fake_records("mechanism_contrasts", rows, spec=_contract("mechanism_contrasts"))


TRACES = {d: (0.30 + 0.05 * i, (0.38, 0.35, 0.05, 0.10, 0.12)) for i, d in enumerate(DATASETS)}
TRACES["ogbn-products"] = (0.08, (0.0, 0.0, 0.0, 0.0, 1.0))


def decision_traces(traces=TRACES):
    rows = []
    for dataset, (entropy, fractions) in traces.items():
        rows.append({"dataset": dataset, "quantity": "same_depth_action_entropy", "value": entropy, "unit": "nats"})
        for (key, _), value in zip((("schedule_nominal", 0), ("state_conditioned", 0), ("exclusive_sequence", 0),
                                    ("recovery", 0), ("input_fallback", 0)), fractions):
            rows.append({"dataset": dataset, "quantity": f"fraction_{key}", "value": value, "unit": "fraction"})
    return fake_records("decision_traces", rows, spec=_contract("decision_traces"))


def _contract(name):
    return json.loads((ROOT / "records/specs.json").read_text(encoding="utf-8"))["sets"][name]


def read_data(writer, name):
    with (writer.out_dir / "data" / writer.stem / f"{name}.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def num(text):
    return float(text) if text not in ("", None) else None


class MainPaperDisplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temporary.name)
        cls.runs = synthetic_runs(cls.base / "runs")
        from tests.test_s14_confirmation import S14CampaignTests
        cls.s14 = S14CampaignTests("test_complete_family_reconstructs_effects_and_returns_failed_peptides_to_comparator")
        cls.s14.setUp()
        cls.records = {"input_gate": input_gate(), "ensemble_grid": ensemble_grid(), "factorials": factorials(),
                       "baselines": baselines(), "nominal_time": nominal_time(), "budget_matched": budget_matched(),
                       "tube_audit": tube_audit(), "mechanism_contrasts": mechanism_contrasts(),
                       "decision_traces": decision_traces()}

    @classmethod
    def tearDownClass(cls):
        cls.s14.tearDown()
        cls.temporary.cleanup()

    def ctx(self, **overrides):
        records = dict(self.records)
        records.update(overrides.pop("records", {}))
        return context(runs=overrides.pop("runs", self.runs), records=records,
                       campaigns={"confirmation": self.s14.root}, **overrides)

    def build(self, display_id, ctx=None, name=None):
        out = self.base / "out" / (name or display_id.replace(":", "_").replace(" ", "_"))
        writer, rows = build_one(display_id, ctx or self.ctx(), out, registry=REGISTRY)
        unresolved = [f"{r['group']}/{r['path']}" for r in rows if r["status"] == "no_reference"]
        self.assertEqual(unresolved, [], "comparison paths must resolve in the registry")
        return writer, rows

    # ------------------------------------------------------------------ Table 1
    def test_table_1_layout_on_example_runs(self):
        runs = example_runs(self.base / "examples_t1")
        writer, rows = self.build("main:Table 1", self.ctx(runs=runs), name="t1_examples")
        table = read_table(writer.out_dir, "T1")
        self.assertEqual(len(table), 7)
        self.assertEqual(len(table[0]), 6)
        self.assertEqual([r["Dataset"] for r in table], [SHORT[d] for d in DATASETS])
        # Both example runs return q_C: the comparator identity row of the printed table.
        self.assertTrue(all(r["deployed policy"] == "q_C returned" for r in table))
        self.assertEqual(table[4]["strongest open-loop comparator q_C"], "phase 4, E = 2")
        ledger = read_table(writer.out_dir, "T1_ledger")
        self.assertEqual((len(ledger), len(ledger[0])), (7, 17))
        self.assertTrue(all(r["admission criterion"] == "not evaluated (Q_s empty)" for r in ledger))

    def test_table_1_cells_from_runs_and_input_gate(self):
        writer, rows = self.build("main:Table 1")
        table = {r["Dataset"]: r for r in read_table(writer.out_dir, "T1")}
        arxiv = table["arxiv"]
        self.assertEqual(arxiv["strongest open-loop comparator q_C"], "fixed sequence, E = 4")
        self.assertEqual(arxiv["deployed policy"], "radius, E = 4")
        self.assertEqual(arxiv["official-metric change; paired loss versus q_C (95% test-stage intervals)"],
                         "+0.07 [0.01, 0.13] pp; -0.00450 [-0.0075, -0.0015] CE")
        self.assertEqual(arxiv["registered reading, representative rows"], "radius")
        self.assertEqual(arxiv["separated-protocol admission; test effect"], "descriptive; -0.0045")
        products = table["products"]
        self.assertEqual(products["strongest open-loop comparator q_C"], "phase 8/32, E = 2")
        self.assertEqual(products["official-metric change; paired loss versus q_C (95% test-stage intervals)"],
                         "0 (comparator identity)")
        self.assertEqual(products["separated-protocol admission; test effect"], "q_C; 0")
        self.assertEqual(table["Pascal"]["strongest open-loop comparator q_C"], "MPC-OL H = 16, E = 8")
        self.assertEqual(table["Pascal"]["separated-protocol admission; test effect"], "G_admit = 1; -0.0084")
        self.assertTrue(table["Pascal"]["official-metric change; paired loss versus q_C (95% test-stage intervals)"]
                        .startswith("+0.60 [0.38, 0.82] F1 pp; -0.00842"))
        self.assertTrue(table["Peptides"]["official-metric change; paired loss versus q_C (95% test-stage intervals)"]
                        .startswith("+0.0058 [0.0009, 0.0107] AP; -0.00303"))
        # Roman: two-split campaign, split mean with the small-sample t interval (t_1 = 12.706).
        roman = table["Roman"]["official-metric change; paired loss versus q_C (95% test-stage intervals)"]
        self.assertTrue(roman.startswith("+0.26 ["))
        self.assertIn("-0.00572", roman)
        half = 12.706 * abs(-0.0050 + 0.00644) / 2                          # t_1 x sd / sqrt(2)
        self.assertIn(f"[{-0.00572 - half:.4f}, {-0.00572 + half:.4f}] CE", roman)
        ledger = {r["Dataset"]: r for r in read_table(writer.out_dir, "T1_ledger")}
        self.assertEqual(ledger["arxiv"]["input-fallback fraction"], "0.118")
        self.assertEqual(ledger["products"]["input-fallback fraction"], "")
        self.assertEqual(ledger["Roman"]["D-bar"], "-0.00640")
        self.assertEqual(ledger["Roman"]["admission criterion"], "descriptive, 2 official splits")
        self.assertEqual(ledger["COCO"]["U_a"], "-0.007922")
        self.assertEqual(ledger["COCO"]["J"], "5000")
        self.assertEqual(ledger["Roman"]["confirmation output"], "state: pass in 2 of 2 splits; mpc: pass in 2 of 2 splits")
        status = {(r["group"], r["path"]): r["status"] for r in rows}
        for path in ("ogbn-arxiv/input_fallback_fraction", "ogbn-arxiv/official_gain_interval_pp",
                     "ogbn-arxiv/loss_effect_interval_nats", "COCO-SP/admission_U_a", "Peptides-func/official_gain_native"):
            self.assertEqual(status[("main_table_1_policy_and_deployment_ledger", path)], "exact", path)
        self.assertEqual(len(read_data(writer, "ledger")), 7)

    # ------------------------------------------------------------------ Table 2
    def test_table_2_layout_on_example_runs(self):
        runs = example_runs(self.base / "examples_t2")
        writer, _ = self.build("main:Table 2", self.ctx(runs=runs), name="t2_examples")
        table = read_table(writer.out_dir, "T2")
        self.assertEqual((len(table), len(table[0])), (7, 10))
        self.assertTrue(all(r["effect over q_C"] == "exactly 0" for r in table))

    def test_table_2_cost_ledger_formulas(self):
        writer, rows = self.build("main:Table 2")
        table = {r["Dataset"]: r for r in read_table(writer.out_dir, "T2")}
        arxiv = table["arxiv"]
        expected = (1 - 0.118) * 4.7 + 0.118 * 4.0                        # Eq. (32)
        self.assertEqual(arxiv["C_expected (overhead)"], f"{expected:.2f} (+15%)")
        self.assertEqual(arxiv["C_assure (overhead)"], "8.7 (+118%)")
        # Table S44 tail model with r = E/(N d) of the registered statistics and f_bar = 0.25.
        r = 1166243 / (169343 * 512)
        v = r * 7 / (1 + r)
        o_fix = 0.175 - 0.25 * v
        tails = [4.0 * (1 + o_fix + (0.25 + s) * v) for s in (0.10, 0.20, 0.35)]
        self.assertEqual(arxiv["C_0.99 [tail scenarios] (overhead)"],
                         f"{tails[1]:.1f} [{tails[0]:.1f}, {tails[2]:.1f}] (+{100 * (tails[1] / 4 - 1):.0f}%)")
        added = expected - 4.0
        self.assertEqual(arxiv["lambda* [range] (mnats; pt per unit)"],
                         f"{4.5 / added:.2f} [{1.5 / added:.1f}, {7.5 / added:.1f}] ({0.07 / added:.2f})")
        self.assertEqual(arxiv["M* t_nom (10^3 input s)"], str(round(3600 * 1.92 / (0.1 * added) / 1000)))
        self.assertEqual(arxiv["effect over q_C"], "+0.07 pp; -0.0045 CE")
        self.assertEqual(table["Peptides"]["effect over q_C"], "+0.0058 AP; -0.00303 BCE")
        self.assertTrue(table["Peptides"]["lambda* [range] (mnats; pt per unit)"].endswith(" AP)"))
        products = table["products"]
        self.assertEqual((products["C_expected (overhead)"], products["C_assure (overhead)"], products["effect over q_C"]),
                         ("2.1 (0%)", "2.1 (0%)", "exactly 0"))
        self.assertEqual(products["C_0.99 [tail scenarios] (overhead)"], "")
        # proteins: 0.25 exceeds the resolvent share the multiplier 1.2 carries; the identity value is used.
        ledger = {r["dataset"]: r for r in read_data(writer, "ledger")}
        from cm_theory.budget import max_resolvent_fraction
        r_proteins = 39561252 / (132534 * 512)
        self.assertAlmostEqual(num(ledger["ogbn-proteins"]["f_bar"]), max_resolvent_fraction(1.2, r_proteins, 0.06, 8))
        self.assertAlmostEqual(num(ledger["ogbn-arxiv"]["f_bar"]), 0.25)
        status = {(r["group"], r["path"]): r["status"] for r in rows}
        self.assertEqual(status[("main_table_2_gain_and_cost_ledger", "ogbn-arxiv/C_099_range")], "exact")
        self.assertEqual(status[("main_table_2_gain_and_cost_ledger", "ogbn-arxiv/M_star_t_nom_thousand_input_seconds")], "exact")

    def test_missing_input_gate_row_is_an_input_error(self):
        gate = fake_records("input_gate", [{"dataset": "ogbn-arxiv", "quantity": "fallback_probability", "value": 0.1}])
        with self.assertRaisesRegex(DisplayInputError, "fallback_probability"):
            build_one("main:Table 2", self.ctx(records={"input_gate": gate}), self.base / "out/missing_gate")

    # ------------------------------------------------------------------ Figure 2
    def test_figure_2_panels(self):
        writer, rows = self.build("main:Figure 2")
        self.assertIn("figures/Fig2.png", writer.files)
        panel_a = read_data(writer, "panel_a")
        ratio = {(r["row"], r["dataset"]): num(r["ratio"]) for r in panel_a}
        self.assertEqual({r["row"] for r in panel_a}, {"prop2", "prop4", "thm1", "prop7", "prop5_expected", "prop5_assure",
                                                       "thm2", "cor2", "s44"})
        self.assertNotIn(("prop2", "ogbn-proteins"), ratio)                       # no tube-audit row for proteins
        self.assertNotIn(("thm2", "ogbn-arxiv"), ratio)                           # Theorem 2 is multigraph only
        self.assertFalse(any(key[1] == "ogbn-products" for key in ratio))           # comparator returned
        self.assertAlmostEqual(ratio[("prop2", "ogbn-arxiv")], 0.0065 / 0.018)
        self.assertAlmostEqual(ratio[("thm1", "ogbn-arxiv")], 0.00463 / (0.002 + 2 * 0.0032))
        self.assertAlmostEqual(ratio[("prop4", "ogbn-arxiv")], 0.00463 / 0.02)
        self.assertAlmostEqual(ratio[("prop7", "ogbn-arxiv")], 0.0045 / (2 * adm_theory.hoeffding_radius(91, 0.1, 10)))
        self.assertAlmostEqual(ratio[("cor2", "ogbn-arxiv")], adm_theory.minimum_admission_units(-0.0046, 0.003) / 32)
        self.assertAlmostEqual(ratio[("thm2", "COCO-SP")], 0.003 / 0.007922)
        expected = (1 - 0.118) * 4.7 + 0.118 * 4.0
        overhead = expected / 4.0 - 1
        self.assertAlmostEqual(ratio[("prop5_expected", "ogbn-arxiv")],
                               0.0305 * overhead / (4 * (1 + overhead)) / 0.0045, places=6)
        self.assertEqual(sum(1 for key in ratio if key[0] == "s44"), 6)
        panel_b = {(r["dataset"], r["stage"]): r for r in read_data(writer, "panel_b")}
        self.assertAlmostEqual(num(panel_b[("ogbn-arxiv", "selection")]["ratio"]), (0.00463 - HALF_WIDTH) / 0.003)
        self.assertAlmostEqual(num(panel_b[("ogbn-arxiv", "test")]["ratio"]), 1.5)
        self.assertAlmostEqual(num(panel_b[("ogbn-arxiv", "test")]["low"]), 0.5)
        self.assertAlmostEqual(num(panel_b[("ogbn-arxiv", "admission")]["ratio"]), 0.0046 / 0.003)
        self.assertAlmostEqual(num(panel_b[("PascalVOC-SP", "admission")]["ratio"]), 0.005782 / 0.003)
        self.assertEqual(panel_b[("ogbn-arxiv", "confirmation")]["statistic"], "arxiv/state")
        self.assertAlmostEqual(num(panel_b[("ogbn-arxiv", "confirmation")]["ratio"]), 0.00574 / 0.002)
        self.assertEqual(panel_b[("Peptides-func", "confirmation")]["statistic"], "Peptides/radius")
        panel_c = read_data(writer, "panel_c")
        self.assertEqual(len(panel_c), 7 * 8)
        arxiv = [r for r in panel_c if r["dataset"] == "ogbn-arxiv"]
        self.assertEqual([r["route"] for r in arxiv if r["deployed"] == "True"], ["radius"])
        self.assertAlmostEqual(num(arxiv[0]["U_min"]), -0.00463 + HALF_WIDTH)
        self.assertTrue(all(r["U_min"] == "" for r in panel_c if r["dataset"] == "ogbn-products"))
        panel_d = read_data(writer, "panel_d")
        self.assertEqual([int(r["rank"]) for r in panel_d], list(range(1, 10)))
        self.assertEqual(panel_d[0]["contrast"], "COCO/RH")
        self.assertEqual(panel_d[-1]["contrast"], "products/state")
        self.assertAlmostEqual(num(panel_d[0]["holm_threshold"]), 0.05 / 9)
        self.assertAlmostEqual(num(panel_d[-1]["holm_threshold"]), 0.05)
        status = {(r["group"], r["path"]): r["status"] for r in rows}
        self.assertEqual(status[("supplementary_table_S14_mechanism_contrasts_submitted_protocol", "7/raw_P")], "exact")

    def test_declared_inputs(self):
        from paper.build.displays.framework import GENERATORS, _load_generators
        _load_generators()
        needs = {display_id: set(GENERATORS[display_id].needs) for display_id in
                 ("main:Table 1", "main:Table 2", "main:Figure 2", "main:Figure 3", "main:Figure 4", "main:Figure 5",
                  "main:Figure 6")}
        self.assertEqual(needs["main:Table 1"], {"runs", "records:input_gate"})
        self.assertEqual(needs["main:Figure 2"], {"runs", "campaign:confirmation", "records:ensemble_grid",
                                                  "records:input_gate", "records:tube_audit"})
        self.assertEqual(needs["main:Figure 6"], {"runs", "records:input_gate", "records:budget_matched"})
        partial = context(runs=self.runs, records={k: v for k, v in self.records.items() if k != "tube_audit"},
                          campaigns={"confirmation": self.s14.root})
        self.assertEqual([n for n in GENERATORS["main:Figure 2"].needs if not partial.available(n)], ["records:tube_audit"])

    # ------------------------------------------------------------------ Figure 3
    def test_figure_3_sources_and_trace_validation(self):
        writer, rows = self.build("main:Figure 3")
        panel_a = {r["dataset"]: r for r in read_data(writer, "panel_a")}
        panel_b = {r["dataset"]: r for r in read_data(writer, "panel_b")}
        self.assertEqual(len(panel_a), 7)
        self.assertEqual(len(panel_b), 7)
        self.assertTrue(panel_a["ogbn-products"]["source"].startswith("joint S14 family (products/state)"))
        self.assertAlmostEqual(num(panel_a["ogbn-arxiv"]["effect"]), 0.00574)
        self.assertEqual(panel_a["PascalVOC-SP"]["source"], "deployed rule's confirm/decision.json")
        roman = panel_a["Roman-empire"]
        self.assertIn("t interval over 2 splits", roman["source"])
        self.assertAlmostEqual(num(roman["effect"]), 0.00575)
        self.assertTrue(panel_b["Roman-empire"]["source"].startswith("joint S14 family (Roman/RH)"))
        self.assertEqual(panel_b["ogbn-arxiv"]["source"], "mechanism_contrasts record set")
        self.assertAlmostEqual(num(panel_b["ogbn-arxiv"]["equivalence_margin"]), 0.002)
        panel_d = read_data(writer, "panel_d")
        self.assertEqual(len(panel_d), 7)
        self.assertAlmostEqual(num(panel_d[2]["input_fallback"]), 1.0)
        self.assertAlmostEqual(num(read_data(writer, "panel_c")[2]["same_depth_action_entropy"]), 0.08)
        broken = dict(TRACES)
        broken["COCO-SP"] = (0.6, (0.5, 0.5, 0.5, 0.0, 0.0))
        with self.assertRaisesRegex(DisplayInputError, "sum to one"):
            build_one("main:Figure 3", self.ctx(records={"decision_traces": decision_traces(broken)}),
                      self.base / "out/broken_traces")
        missing = fake_records("mechanism_contrasts", [], spec=_contract("mechanism_contrasts"))
        with self.assertRaisesRegex(DisplayInputError, "rh_minus_ol of ogbn-arxiv"):
            build_one("main:Figure 3", self.ctx(records={"mechanism_contrasts": missing}), self.base / "out/no_contrasts")

    # ------------------------------------------------------------------ Figure 4
    def test_figure_4_panels(self):
        writer, rows = self.build("main:Figure 4")
        panel_a = read_data(writer, "panel_a")
        self.assertEqual([r["cell"] for r in panel_a], ["000", "100", "010", "001", "110", "101", "011", "111"])
        self.assertAlmostEqual(num(panel_a[-1]["ld_integrated_loss"]), 0.8445)
        panel_b = read_data(writer, "panel_b")
        self.assertEqual([int(r["ensemble"]) for r in panel_b], [1, 2, 4, 8, 12])
        self.assertAlmostEqual(num(panel_b[0]["retained_fraction"]), 0.65 / 1.08)
        self.assertAlmostEqual(num(panel_b[2]["normalized_work"]), 4 / 12)
        self.assertEqual([r["selected"] for r in panel_b].count("True"), 1)
        panel_c = {r["dataset"]: r for r in read_data(writer, "panel_c")}
        self.assertAlmostEqual(num(panel_c["ogbn-arxiv"]["C_expected"]), (1 - 0.118) * 4.7 + 0.118 * 4.0)
        self.assertAlmostEqual(num(panel_c["ogbn-products"]["C_assure"]), 2.1)
        panel_e = {r["dataset"]: r for r in read_data(writer, "panel_e")}
        self.assertAlmostEqual(num(panel_e["ogbn-arxiv"]["qualification_hours"]), 1.92)
        self.assertAlmostEqual(num(panel_e["Roman-empire"]["qualification_hours"]), 2 * 0.69)
        self.assertAlmostEqual(num(panel_e["Roman-empire"]["plant_training_hours"]), 20.0)
        self.assertEqual((num(panel_e["ogbn-arxiv"]["retrained_low_gpu_hours"]),
                          num(panel_e["ogbn-arxiv"]["retrained_high_gpu_hours"])), (12.0, 14.0))
        panel_f = {r["dataset"]: r for r in read_data(writer, "panel_f")}
        self.assertEqual(num(panel_f["COCO-SP"]["wall_time_overhead_pct"]), 27.0)
        self.assertEqual(num(panel_f["ogbn-products"]["wall_time_overhead_pct"]), 0.0)
        status = {(r["group"], r["path"]): r["status"] for r in rows}
        self.assertEqual(status[("supplementary_table_S16_arxiv_factorial", "cells/111/log_depth_CE")], "exact")

    # ------------------------------------------------------------------ Figure 5
    def test_figure_5_economics(self):
        writer, rows = self.build("main:Figure 5")
        panel_a = read_data(writer, "panel_a")
        self.assertEqual(sum(1 for r in panel_a if r["policy"] == "qC"), 7)
        self.assertEqual(sum(1 for r in panel_a if r["policy"] == "deployed"), 6)
        self.assertEqual(sum(1 for r in panel_a if r["policy"] == "retrained"), 14)
        arxiv = {r["policy"] + r["method"]: r for r in panel_a if r["dataset"] == "ogbn-arxiv"}
        expected = (1 - 0.118) * 4.7 + 0.118 * 4.0
        self.assertAlmostEqual(num(arxiv["deployedradius"]["relative_cost"]), expected / 4.0)
        self.assertAlmostEqual(num(arxiv["deployedradius"]["relative_cost_high"]), 8.7 / 4.0)
        self.assertAlmostEqual(num(arxiv["deployedradius"]["score_gain_points"]), 0.22 + 0.07)
        self.assertAlmostEqual(num(arxiv["retrainedAMP"]["score_gain_points"]), 70.12 - 70.0)
        peptides = [r for r in panel_a if r["dataset"] == "Peptides-func" and r["policy"] == "retrained"]
        self.assertAlmostEqual(num(peptides[0]["score_points"]), 70.12)           # native AP x 100
        panel_c = [r for r in read_data(writer, "panel_c") if r["dataset"] == "ogbn-arxiv"]
        self.assertAlmostEqual(num(panel_c[0]["net_benefit_mnats"]), 4.5)
        self.assertEqual({r["cost"] for r in panel_c}, {"expected", "assurance"})
        points = {r["dataset"]: r for r in read_data(writer, "panel_d_operating_points")}
        self.assertAlmostEqual(num(points["ogbn-arxiv"]["M_star_inputs"]), 3600 * 1.92 / (0.1 * (expected - 4.0) * 9.54))
        self.assertEqual(len(read_data(writer, "panel_b")), 6)

    # ------------------------------------------------------------------ Figure 6
    def test_figure_6_budget_matched(self):
        writer, rows = self.build("main:Figure 6")
        frontier = [r for r in read_data(writer, "panel_a") if r["dataset"] == "ogbn-arxiv"]
        self.assertEqual([r["policy"] for r in frontier], ["qC", "feedback", "E3", "E4", "E5", "E6"])
        self.assertAlmostEqual(num(frontier[1]["low_over_g"]), -0.0075 / 0.0045)
        self.assertAlmostEqual(num(frontier[2]["effect_over_g"]), -0.0016 / 0.0045)
        boundary = {r["dataset"]: r for r in read_data(writer, "panel_b")}
        power = {r["dataset"]: r for r in read_data(writer, "panel_c")}
        row = {**boundary["ogbn-arxiv"], **power["ogbn-arxiv"]}
        teacher = -0.0016 - 0.0029
        self.assertAlmostEqual(num(row["phi_measured"]), -0.0016 / teacher)
        self.assertAlmostEqual(num(row["r_B"]), -0.0044 / (-0.0044 - 0.0001))
        self.assertEqual((num(row["phi_grid_low"]), num(row["phi_grid_central"]), num(row["phi_grid_high"])), (0.2, 0.35, 0.55))
        se = (0.0048 - 0.0010) / (2 * 1.959963984540054)
        self.assertAlmostEqual(num(row["paired_se_nats"]), se)
        margin = (1 - 0.35) * 0.0045
        self.assertAlmostEqual(num(row["margin_nats"]), margin)
        self.assertAlmostEqual(num(row["power_unit"]), inference.power_of_contrast(margin, se))
        self.assertAlmostEqual(num(row["power_seed_optimistic"]), inference.power_of_contrast(margin, se / math.sqrt(12)))
        conservative = se * math.sqrt(0.5 + 0.5 / 12)
        self.assertAlmostEqual(num(row["power_seed_conservative"]), inference.power_of_contrast(margin, conservative, 0.05, 4))
        # E3 executes at the ceiling: no spare budget, so the ensemble-spend crossover is the contract value 1.
        self.assertAlmostEqual(num(row["phi_FB_ensemble_low"]), 1.0)
        pascal = {**boundary["PascalVOC-SP"], **power["PascalVOC-SP"]}
        self.assertEqual((num(pascal["phi_grid_low"]), num(pascal["phi_grid_high"])), (0.1, 0.45))
        self.assertEqual(num(pascal["seed_shrinking_share"]), 0.0)
        panel_d = {r["dataset"]: r for r in read_data(writer, "panel_d")}
        self.assertEqual(len(panel_d), 6)
        self.assertAlmostEqual(num(panel_d["ogbn-arxiv"]["assurance_overhead_pct"]), 100 * (8.7 / 4 - 1))

    def test_every_display_writes_one_figure_or_table_per_display(self):
        for display_id, stem in (("main:Figure 2", "Fig2"), ("main:Figure 3", "Fig3"), ("main:Figure 4", "Fig4"),
                                 ("main:Figure 5", "Fig5"), ("main:Figure 6", "Fig6")):
            out = self.base / "out" / f"files_{stem}"
            writer, _ = build_one(display_id, self.ctx(), out, registry=REGISTRY)
            figures = [f for f in writer.files if f.startswith("figures/")]
            self.assertEqual(sorted(figures), [f"figures/{stem}.png", f"figures/{stem}.svg"])
            self.assertTrue(any(f.startswith(f"data/{stem}/panel_") for f in writer.files))


if __name__ == "__main__":
    unittest.main()
