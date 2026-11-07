"""Generators of Supplementary Tables S5c, S17, S18, S27, S29, S43 and Figure S5.

Fixtures are synthetic: small record sets in the contracts of records/specs.d and
records/specs.json, and run directories holding exactly the files a generator
reads. They exercise layout and arithmetic, never the paper's results.
"""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from paper.build.displays import experiments
from paper.build.displays.framework import DATASETS, GRAPH_LEVEL, DisplayInputError, RunView
from paper.build.displays.testing import build_one, context, fake_records, fake_run, read_table

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = json.loads((ROOT / "paper/reported_results.json").read_text(encoding="utf-8"))


def spec(name):
    return json.loads((ROOT / "records/specs.d" / f"{name}.json").read_text(encoding="utf-8"))


def resolved(rows):
    return [row for row in rows if row["status"] != "no_reference"]


class Workspace(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.out = self.tmp / "out"

    def tearDown(self):
        self._tmp.cleanup()


# ---------------------------------------------------------------------------
# Table S5c
# ---------------------------------------------------------------------------

def identity_entry(states, losses, *, backbone="shared_transport", controlled=256, native=None, precision="bf16"):
    tolerance = 0.008 if precision == "bf16" else 1e-5
    native = controlled if native is None else native
    inserted = 0 if native == controlled else controlled - native
    return {"nominal_identity": [{"seed": k + 1, "max_relative_state_error": s, "absolute_fit_val_loss_deviation": l,
                                  "state_tolerance": tolerance, "loss_tolerance": 1e-4 if precision == "bf16" else 1e-6,
                                  "pass": True} for k, (s, l) in enumerate(zip(states, losses))],
            "instrumentation": {"backbone": backbone, "controlled_depth": controlled, "native_site_count": native,
                                "inserted_site_count": inserted}}


MODELS = {"ogbn-arxiv": {"backbone": "shared_transport", "encoder_layers": 4, "hidden_dim": 512},
          "ogbn-proteins": {"backbone": "edge_weighted_transport", "encoder_layers": 4, "hidden_dim": 512},
          "ogbn-products": {"backbone": "shared_transport", "encoder_layers": 3, "hidden_dim": 512},
          "Roman-empire": {"backbone": "heterophily", "encoder_layers": 4, "hidden_dim": 384},
          "PascalVOC-SP": {"backbone": "gps", "encoder_layers": 1, "hidden_dim": 128},
          "COCO-SP": {"backbone": "gps", "encoder_layers": 1, "hidden_dim": 128},
          "Peptides-func": {"backbone": "linear_gine", "encoder_layers": 1, "hidden_dim": 128}}


def s5c_runs(tmp: Path) -> dict:
    runs = {}
    for dataset in DATASETS:
        model = dict(MODELS[dataset], precision="fp32" if dataset == "Roman-empire" else "bf16")
        sites = {"gps": (128, 6), "linear_gine": (64, 8)}.get(model["backbone"], (256, None))
        if dataset == "Roman-empire":
            splits = []
            for k, state in enumerate((3.1e-7, 4.2e-7)):
                entry = identity_entry([state], [2.5e-8 * (k + 1)], backbone="heterophily", precision="fp32")
                splits.append(fake_run(tmp / "roman" / f"split_{k:02d}", dataset,
                                       {"train/entry_initial.json": entry, "config.json": {"model": model}}))
            runs[dataset] = RunView(dataset, tmp / "roman", splits=splits)
            continue
        entry = identity_entry([0.0031, 0.0042], [2.6e-5, 1.1e-5], backbone=model["backbone"],
                               controlled=sites[0], native=sites[1])
        runs[dataset] = fake_run(tmp / dataset, dataset, {"train/entry_initial.json": entry, "config.json": {"model": model}})
    return runs


class TableS5c(Workspace):
    def test_formatting_helpers(self):
        self.assertEqual(experiments.sci(0.0042), "4 × 10^-3")
        self.assertEqual(experiments.sci(0.0025), "3 × 10^-3")          # half up
        self.assertEqual(experiments.sci(1e-4), "10^-4")
        self.assertEqual(experiments.sci(0.0), "0")
        self.assertEqual(experiments.sci(12345.0, 2), "1.2 × 10^4")
        self.assertEqual(experiments.sci(9.96e-3), "10^-2")
        self.assertEqual(experiments.grid_text([0.025, 0.05, 0.1, 32, 0.0015]), "0.025, 0.05, 0.10, 32, 0.0015")

    def test_layout_and_values(self):
        writer, _ = build_one("supplementary:Table S5c", context(runs=s5c_runs(self.tmp)), self.out)
        table = read_table(self.out, "S5c")
        self.assertEqual(len(table), 7)
        self.assertEqual(len(table[0]), 6)
        arxiv, roman, pascal, peptides = table[0], table[3], table[4], table[6]
        self.assertEqual(arxiv["Dataset (backbone)"], "ogbn-arxiv (MLP4-512 transport plant)")
        self.assertEqual(arxiv["max. relative state deviation"], "4 × 10^-3 (limit 8 × 10^-3)")
        self.assertEqual(arxiv["test-loss deviation (nats)"], "3 × 10^-5 (limit 10^-4)")
        self.assertEqual(arxiv["inserted sites"], "none")
        self.assertEqual(roman["precision"], "fp32")
        self.assertEqual(roman["max. relative state deviation"], "4 × 10^-7 (limit 10^-5)")   # max over splits
        self.assertEqual(roman["test-loss deviation (nats)"], "5 × 10^-8 (limit 10^-6)")
        self.assertEqual(pascal["max. relative state deviation"], "4 × 10^-3 at the 6 designated sites (limit 8 × 10^-3)")
        self.assertEqual(pascal["inserted sites"], "122 inserted (20 per block plus 2 boundary sites; 128 controlled "
                                                   "sites with the 6 native sites); zero deviation by construction")
        self.assertEqual(pascal["reporting scope"], "E1 record; inserted-site action mix recorded separately")
        self.assertEqual(peptides["inserted sites"], "56 inserted (7 per native layer; 64 controlled sites with the "
                                                     "8 native sites); zero deviation by construction")
        self.assertEqual(peptides["reporting scope"], "E1 record; nonlinear messages excluded")
        self.assertIn("data/S5c/identity_records.csv", writer.files)
        source = (self.out / "data/S5c/identity_records.csv").read_text().splitlines()
        self.assertEqual(len(source), 1 + 6 * 2 + 2)   # two seeds per run, one per Roman split

    def test_example_runs(self):
        runs = {d: RunView(d, ROOT / ("examples/runs/peptides_func" if d in GRAPH_LEVEL else "examples/runs/arxiv"))
                for d in DATASETS}
        build_one("supplementary:Table S5c", context(runs=runs), self.out)
        table = read_table(self.out, "S5c")
        self.assertEqual(len(table), 7)
        self.assertEqual(table[6]["inserted sites"], "6 inserted (3 per native layer; 8 controlled sites with the 2 "
                                                     "native sites); zero deviation by construction")
        self.assertIn("registered for fp32", (self.out / "tables/S5c.md").read_text())

    def test_missing_run(self):
        runs = s5c_runs(self.tmp)
        runs.pop("COCO-SP")
        with self.assertRaises(DisplayInputError):
            build_one("supplementary:Table S5c", context(runs=runs), self.out)


# ---------------------------------------------------------------------------
# Table S29 and Figure S5
# ---------------------------------------------------------------------------

SWEEP = {(8, 16): (-0.00710, 9.6), (16, 16): (-0.00916, 10.3), (32, 16): (-0.00970, 11.1), (64, 16): (-0.0101, 12.6),
         (16, 4): (-0.0079, 9.1), (16, 8): (-0.0087, 9.7), (16, 32): (-0.0094, 11.4)}
HEADROOM = -0.0262


def sweep_records(points=SWEEP):
    rows = []
    for (h, b), (effect, cost) in points.items():
        keys = {"dataset": "PascalVOC-SP", "horizon": str(h), "beam_width": str(b)}
        rows += [dict(keys, quantity="selection_effect_nats", value=effect),
                 dict(keys, quantity="expected_serial_cost", value=cost),
                 dict(keys, quantity="oracle_headroom_bound_nats", value=HEADROOM),
                 dict(keys, quantity="realized_fraction", value=effect / HEADROOM)]
    return fake_records("horizon_beam_sweep", rows, spec=spec("horizon_beam_sweep"))


class SweepDisplays(Workspace):
    def test_table_s29(self):
        ctx = context(records={"horizon_beam_sweep": sweep_records()})
        writer, rows = build_one("supplementary:Table S29", ctx, self.out)
        table = read_table(self.out, "S29")
        self.assertEqual([r["configuration"] for r in table],
                         ["horizon H = 8 (beam 16)", "horizon H = 16 (beam 16)", "horizon H = 32 (beam 16)",
                          "horizon H = 64 (beam 16)", "beam 4 (H = 16)", "beam 8 (H = 16)", "beam 16 (H = 16)",
                          "beam 32 (H = 16)"])
        self.assertEqual(len(table[0]), 4)
        self.assertEqual(table[0]["selection effect (nats)"], "-0.00710")
        self.assertEqual(table[0]["expected serial cost"], "9.6")
        self.assertEqual(table[0]["realized fraction"], "0.27")        # -0.00710 / -0.0262
        self.assertEqual(table[1]["realized fraction"], table[6]["realized fraction"])
        self.assertEqual(rows, [])
        data = (self.out / "data/S29/sweep.csv").read_text().splitlines()
        self.assertEqual(len(data), 9)

    def test_missing_configuration(self):
        points = dict(SWEEP)
        points.pop((64, 16))
        ctx = context(records={"horizon_beam_sweep": sweep_records(points)})
        with self.assertRaises(DisplayInputError):
            build_one("supplementary:Table S29", ctx, self.out)

    def test_figure_s5(self):
        ctx = context(records={"horizon_beam_sweep": sweep_records()})
        writer, _ = build_one("supplementary:Figure S5", ctx, self.out)
        for name in ("figures/FigS5.png", "figures/FigS5.svg", "data/FigS5/panel_a.csv", "data/FigS5/panel_b.csv"):
            self.assertIn(name, writer.files)
        panel_a = (self.out / "data/FigS5/panel_a.csv").read_text().splitlines()
        panel_b = (self.out / "data/FigS5/panel_b.csv").read_text().splitlines()
        self.assertEqual(len(panel_a), 1 + 4)
        self.assertEqual(len(panel_b), 1 + 4 + 4)


# ---------------------------------------------------------------------------
# Table S27
# ---------------------------------------------------------------------------

class TableS27(Workspace):
    def records(self):
        rows = []
        for dataset, mean in (("ogbn-arxiv", -0.0046), ("ogbn-proteins", -0.0039)):
            for k, hops in enumerate((2, 4, 8, 16, 32)):
                half = 0.0038 + 0.0003 * k
                upper = mean + half
                keys = {"dataset": dataset, "buffer_hops": str(hops)}
                rows += [dict(keys, quantity="clipped_effect_nats", value=mean),
                         dict(keys, quantity="half_width_nats", value=half),
                         dict(keys, quantity="upper_nats", value=upper),
                         dict(keys, quantity="admitted", value=float(upper < 0)),
                         dict(keys, quantity="resampling_units", value=32 - 4 * k)]
        return fake_records("buffer_sensitivity", rows, spec=spec("buffer_sensitivity"))

    def test_layout_decisions_and_comparisons(self):
        writer, rows = build_one("supplementary:Table S27", context(records={"buffer_sensitivity": self.records()}),
                                 self.out, registry=REGISTRY)
        table = read_table(self.out, "S27")
        self.assertEqual(len(table), 10)
        self.assertEqual(len(table[0]), 6)
        self.assertEqual([r["buffer (hops)"] for r in table[:5]], ["2", "4", "8", "16", "32"])
        self.assertEqual(table[0]["U_desc"], "-0.0008")
        self.assertEqual(table[2]["decision at eps_a"], "pass")          # U_desc = -0.0002
        self.assertEqual(table[3]["U_desc"], "+0.0001")
        self.assertEqual(table[3]["decision at eps_a"], "interval includes zero")
        checked = resolved(rows)
        self.assertEqual(len(checked), 4)                                 # D_bar and U_desc of arxiv and proteins
        self.assertTrue(all(r["status"] in ("exact", "differs") for r in checked))


# ---------------------------------------------------------------------------
# Table S18
# ---------------------------------------------------------------------------

def s18_records(constants):
    grids = constants["sensitivity_grids_table_S18"]
    bce = constants["selection_eq19_algorithm_S4"]["datasets_using_BCE_margins"]
    rows = []

    def add(parameter, setting, dataset, **values):
        for quantity, value in values.items():
            rows.append({"parameter": parameter, "setting": f"{setting:g}", "dataset": dataset,
                         "quantity": quantity, "value": value})
    for d, dataset in enumerate(DATASETS):
        for setting in grids["minimum_propensity"]:
            # 0.075 simplifies one dataset of seven, 0.10 two of seven
            simpler = float((setting == 0.075 and d == 0) or (setting == 0.1 and d < 2))
            add("p_min", setting, dataset, qualified_fraction=0.9, strata_without_transport=0.0,
                eligible_count=2.0, selected_complexity=7.0, route_changed=simpler, route_simpler=simpler)
        for setting in grids["minimum_ESS"]:
            change = float(setting == 128 and d == 3)
            add("n_min", setting, dataset, qualified_fraction=0.9, strata_without_transport=1.0 + change,
                eligible_count=2.0, selected_complexity=7.0, route_changed=change, route_simpler=change)
        for setting in grids["maximum_normalized_weight"]:
            add("max_normalized_weight", setting, dataset, qualified_fraction=0.9, strata_without_transport=0.0,
                route_changed=0.0, route_simpler=0.0)
        margins = grids["selection_margin_epsilon_R"]["mean_BCE" if dataset in bce else "CE"]
        for setting in margins:
            add("epsilon_r", setting, dataset, eligible_count=3.0 - float(setting == max(margins) and d == 5),
                selected_complexity=8.0, route_changed=0.0, route_simpler=0.0)
        for setting in grids["equivalence_margin_epsilon_eq"]:
            add("epsilon_eq", setting, dataset, eligible_count=2.0, selected_complexity=8.0,
                route_changed=float(setting == 0.001 and d == 4), route_simpler=0.0)
        for setting in grids["clip_B"]:
            spread = 0.004 if dataset == "Peptides-func" else 0.001
            add("clip", setting, dataset, clipped_fraction=0.0, clipped_mean_nats=-0.004,
                admission_upper_nats=-0.006 + spread * setting / 0.2, raw_maximum_harm_nats=0.01, admitted=1.0)
    return fake_records("sensitivity_grids", rows, spec=spec("sensitivity_grids"))


class TableS18(Workspace):
    def test_layout_and_summaries(self):
        ctx = context()
        ctx = context(records={"sensitivity_grids": s18_records(ctx.constants)})
        writer, rows = build_one("supplementary:Table S18", ctx, self.out)
        table = read_table(self.out, "S18")
        self.assertEqual(len(table), 6)
        self.assertEqual(len(table[0]), 4)
        self.assertEqual([r["parameter"] for r in table],
                         ["minimum propensity", "minimum ESS", "maximum normalized weight", "selection margin εR",
                          "equivalence margin εeq", "clip B"])
        self.assertEqual(table[0]["grid"], "0.025, 0.05, 0.075, 0.10")
        self.assertEqual(table[3]["grid"], "CE: 0.0025, 0.003, 0.005; mean BCE: 0.001, 0.0015, 0.0025")
        self.assertEqual(table[5]["grid"], "0.05, 0.10, 0.20")
        effect = "observed effect on route/support"
        self.assertEqual(table[0][effect], "stricter values simplify 14–29% of selected routes")   # 1/7, 2/7
        self.assertEqual(table[1][effect], "route stable through 96; 128 removes weak strata")
        self.assertIn("ESS ≥ 64 gives ωmax ≤ 0.125", table[2][effect])
        self.assertIn("grid points 0.15–0.25 cannot change any route", table[2][effect])
        self.assertIn("at ESS ≥ 32, ωmax ≤ 0.177", table[2][effect])
        self.assertEqual(table[3][effect], "weak routes disappear under stricter margin")
        self.assertEqual(table[4][effect], "complexity selection changes near the frontier")
        self.assertEqual(table[5][effect], "Peptides most sensitive")
        self.assertEqual(table[5]["required report"], "clipped fractions, tails, Ua, decision")
        self.assertIn("data/S18/records.csv", writer.files)

    def test_requires_every_dataset(self):
        ctx = context()
        records = s18_records(ctx.constants)
        records.rows = [r for r in records.rows if r["dataset"] != "COCO-SP"]
        with self.assertRaises(DisplayInputError):
            build_one("supplementary:Table S18", context(records={"sensitivity_grids": records}), self.out)


# ---------------------------------------------------------------------------
# Table S17
# ---------------------------------------------------------------------------

class TableS17(Workspace):
    def records(self):
        rows = []

        def add(module, arm, dataset, quantity, value):
            rows.append({"module": module, "arm": arm, "dataset": dataset, "quantity": quantity, "value": value})
        add("spectral_bands", "deleted", "pooled", "loss_change_nats", 0.0032)
        add("spectral_bands", "deleted", "pooled", "warning_lead_change_depths", -3.1)
        for arm, value in (("raw", 0.0119), ("ema", 0.0093), ("global", 0.0074), ("conditioned", 0.0057)):
            add("observer", arm, "pooled", "innovation_rmse", value)
        add("radius_uncertainty", "one_step", "ogbn-arxiv", "loss_change_nats", 0.0027)
        add("radius_uncertainty", "one_step", "Peptides-func", "loss_change_nats", 0.0023)
        add("sequence_risk", "sequence_deleted", "PascalVOC-SP", "loss_change_nats", 0.0055)
        add("comparator_branch", "branch_off", "pooled", "clean_score_change", 0.00001)
        add("comparator_branch", "branch_off", "pooled", "worst_harm_change_nats", 0.0118)
        for dataset, retained, cost in (("ogbn-arxiv", 0.93, 0.46), ("ogbn-proteins", 0.95, 0.60),
                                        ("Roman-empire", 0.91, 0.52)):
            add("full_lean_profile", "lean", dataset, "retained_effect_fraction", retained)
            add("full_lean_profile", "lean", dataset, "cost_fraction", cost)
        return fake_records("module_ablations", rows, spec=spec("module_ablations"))

    def test_layout_signatures_and_comparisons(self):
        writer, rows = build_one("supplementary:Table S17", context(records={"module_ablations": self.records()}),
                                 self.out, registry=REGISTRY)
        table = read_table(self.out, "S17")
        self.assertEqual(len(table), 10)
        self.assertEqual(len(table[0]), 5)
        signature = {r["module"]: r["observed signature"] for r in table}
        self.assertEqual(signature["spectral bands"], "CE +0.0032; warning lead -3.1 depths")
        self.assertEqual(signature["observer"], "RMSE 0.0119/0.0093/0.0074/0.0057")
        self.assertEqual(signature["radius uncertainty"], "loss +0.0023–0.0027 without radius")
        self.assertEqual(signature["sequence risk"], "CE +0.0055 on RH regimes")
        self.assertEqual(signature["comparator branch"], "clean score unchanged; worst harm +0.0118 without branch")
        self.assertEqual(signature["Full/Lean profile"], "91–95% effect at 46–60% cost")
        self.assertEqual(signature["source alignment"], "")              # no observations: empty cell
        self.assertEqual(table[0]["removal/downgrade rule"], "remove if no incremental warning or policy value")
        self.assertIn("No observations for: source alignment", (self.out / "tables/S17.md").read_text())
        checked = resolved(rows)
        self.assertEqual(len(checked), 3)                                 # lean_effect of three datasets (S23)


# ---------------------------------------------------------------------------
# Table S43
# ---------------------------------------------------------------------------

ADMITTED = tuple(d for d in DATASETS if d != "ogbn-products")
FEEDBACK = {"ogbn-arxiv": -0.0045, "ogbn-proteins": -0.0038, "Roman-empire": -0.0057, "PascalVOC-SP": -0.0084,
            "COCO-SP": -0.0091, "Peptides-func": -0.0030}
D_INF = {"ogbn-arxiv": 0.0305, "ogbn-proteins": 0.0088, "Roman-empire": 0.0088, "PascalVOC-SP": 0.0455,
         "COCO-SP": 0.0700, "Peptides-func": 0.0077}
CELLS = {"000": 1.3230, "001": 1.3183, "010": 1.3174, "011": 1.3124, "100": 1.3170, "101": 1.3121, "110": 1.3110,
         "111": 1.3061}
QC_ROUTE = {"ogbn-arxiv": "fixed_sequence", "ogbn-proteins": "fixed_sequence", "ogbn-products": "phase",
            "Roman-empire": "mpc_ol", "PascalVOC-SP": "mpc_ol", "COCO-SP": "mpc_ol", "Peptides-func": "fixed_sequence"}
C_QC, C_QD, FALLBACK, GPU_TOTAL, GPU_PLANT = 4.0, 4.6, 0.1, 3.0, 1.2


def evidence_for(dataset):
    returned = dataset == "ogbn-products"
    deployed = "mpc_rh" if QC_ROUTE[dataset] == "mpc_ol" else "radius"
    final = {"paired_loss": {"mean": FEEDBACK.get(dataset, 0.0), "lower": -0.008, "upper": -0.001},
             "paired_metric_pp": {"mean": 0.1, "lower": 0.0, "upper": 0.2},
             "models": {name: {"loss_nats": 0.9, "metric": 0.7, "metric_percent": 70.0, "test_units": 10}
                        for name in ("q0", "qN", "qC", "qD")}}
    evidence = {"selection": {"qC": {"route": QC_ROUTE[dataset], "ensemble": 4}, "qC_index": 0},
                "admission": {"deployed_index": 0 if returned else 1, "Gadmit": int(not returned),
                              "deployed": {"route": QC_ROUTE[dataset] if returned else deployed, "ensemble": 4}},
                "final": final, "task": {"metric": "accuracy", "loss_name": "ce"},
                "cost": {"C_qC": C_QC, "C_qD": C_QC if returned else C_QD}, "entry": {"headroom": {"upper": -0.02}}}
    candidates = [{"route": QC_ROUTE[dataset], "cost": C_QC, "ensemble": 4},
                  {"route": deployed, "cost": C_QD, "ensemble": 4}]
    return evidence, candidates


def s43_inputs(tmp: Path):
    from paper.build.displays.records_campaigns import ROUTE_METHOD
    runs = {}
    for dataset in DATASETS:
        precision = "fp32" if dataset == "Roman-empire" else "bf16"
        entry = identity_entry([0.0021 if dataset != "COCO-SP" else 0.0044], [1e-5], precision=precision)
        evidence, candidates = evidence_for(dataset)
        files = {"train/entry_initial.json": entry, "config.json": {"model": {"precision": precision}},
                 "select/candidates.json": candidates}
        if dataset == "Roman-empire":
            splits = []
            for k, mean in enumerate((-0.006, -0.007, -0.0065)):
                splits.append(fake_run(tmp / "roman" / f"split_{k:02d}", dataset,
                                       {**files, "admit/decision.json": {"loss": {"mean": mean, "upper": mean + 0.002}}},
                                       evidence=evidence))
            runs[dataset] = RunView(dataset, tmp / "roman", splits=splits, evidence=evidence)
            continue
        if dataset in ("ogbn-arxiv", "ogbn-proteins"):
            files["admit/decision.json"] = {"loss": {"mean": -0.0046, "upper": -0.0008}}
        runs[dataset] = fake_run(tmp / dataset, dataset, files, evidence=evidence)
    budget, inserted, grid, gate, nominal, baselines = [], [], [], [], [], []
    for k, dataset in enumerate(ADMITTED):
        arms = {"E3": -0.0016, "E4": -0.0014, "E5": -0.0013, "E6": FEEDBACK[dataset] * 0.97}
        for arm, effect in arms.items():
            keys = {"dataset": dataset, "arm": arm, "configuration": "selected"}
            budget += [dict(keys, quantity="effect_vs_qc_nats", value=effect),
                       dict(keys, quantity="effect_vs_feedback_nats", value=effect - FEEDBACK[dataset]),
                       dict(keys, quantity="work_ratio", value=1.06 if arm == "E6" else 1.1)]
        for e in (1, 2, 4, 8, 12, 16, 20):
            grid.append({"dataset": dataset, "ensemble": str(e), "quantity": "loss_nats", "value": 0.8 + D_INF[dataset] / e})
        gate += [{"dataset": dataset, "quantity": "deployed_wall_overhead_percent", "value": 13.0 + k},
                 {"dataset": dataset, "quantity": "fallback_probability", "value": FALLBACK}]
        nominal.append({"dataset": dataset, "quantity": "t_nom_seconds",
                        "value": 0.00073 if dataset in GRAPH_LEVEL else 9.54})
        deployed = "mpc_rh" if QC_ROUTE[dataset] == "mpc_ol" else "radius"
        baselines += [{"dataset": dataset, "method": f"{ROUTE_METHOD[QC_ROUTE[dataset]]} (qC)", "seed": "",
                       "quantity": "gpu_hours", "value": GPU_PLANT},
                      {"dataset": dataset, "method": f"{ROUTE_METHOD[deployed]} (deployed)", "seed": "",
                       "quantity": "gpu_hours", "value": GPU_TOTAL}]
    for dataset, native in (("PascalVOC-SP", -0.0018), ("COCO-SP", -0.0020), ("Peptides-func", -0.00105)):
        inserted += [{"dataset": dataset, "arm": "native, feedback", "quantity": "effect_nats", "value": native},
                     {"dataset": dataset, "arm": "inserted, feedback (deployed)", "quantity": "effect_nats",
                      "value": FEEDBACK[dataset]}]
    factorials = [{"dataset": "PascalVOC-SP", "cell": cell, "quantity": "ld_integrated_loss", "value": value}
                  for cell, value in CELLS.items()]
    survival = [{"contrast": c, "quantity": "label_survival_probability", "value": v}
                for c, v in (("COCO/RH", 0.97), ("Pascal/RH", 0.85), ("Peptides/radius", 0.45))]
    tightness = [{"action": f"A{a}", "spectral_scale": scale, "quantity": "displacement_ratio", "value": value}
                 for scale, values in (("0.1", (0.35, 0.63, 0.5)), ("0.3", (0.73, 0.731)), ("1.0", (0.82, 0.86)))
                 for a, value in enumerate(values, start=1)]
    oracle = []
    for dataset, depth, differs in (("PascalVOC-SP", 40, 0.71), ("COCO-SP", 36, 0.69), ("Peptides-func", 30, 0.45),
                                    ("ogbn-products", 16, 0.20), ("ogbn-arxiv", 24, 0.38)):
        oracle += [{"dataset": dataset, "quantity": "median_oracle_depth", "value": depth},
                   {"dataset": dataset, "quantity": "fraction_differs_from_qc", "value": differs}]
    records = {"budget_matched": fake_records("budget_matched", budget),
               "inserted_site": fake_records("inserted_site", inserted),
               "ensemble_grid": fake_records("ensemble_grid", grid),
               "input_gate": fake_records("input_gate", gate),
               "nominal_time": fake_records("nominal_time", nominal),
               "baselines": fake_records("baselines", baselines),
               "factorials": fake_records("factorials", factorials),
               "confirmation_sensitivity": fake_records("confirmation_sensitivity", survival),
               "band_bound_tightness": fake_records("band_bound_tightness", tightness),
               "oracle_dispersion": fake_records("oracle_dispersion", oracle)}
    return runs, records


def expected_cost_models(ctx, dataset):
    """Independent restatement of the Table S44 tail model and the Table S6c identity for the fixture."""
    stats = ctx.constants["dataset_statistics_table_S6c_S23b"][dataset]
    width = ctx.constants["backbones_table_S5"][dataset]["hidden"]
    r = stats.get("reported_edges", stats.get("mean_reported_edges_per_graph")) / (
        stats.get("nodes", stats.get("mean_nodes_per_graph")) * width)
    m, order, s = C_QD / C_QC, 8, 0.06
    v = r * (order - 1) / (1 + r)
    f_bar = min(0.25, (m * (1 + r) - (1 + r + s)) / (r * (order - 1)))
    tail = 100 * ((m - 1) - f_bar * v + (f_bar + 0.20) * v)
    return tail, min(1.0, (m - 1) * (1 + r) / (8 * r))


class TableS43(Workspace):
    def test_layout_values_and_comparisons(self):
        runs, records = s43_inputs(self.tmp)
        ctx = context(runs=runs, records=records)
        writer, rows = build_one("supplementary:Table S43", ctx, self.out, registry=REGISTRY)
        table = read_table(self.out, "S43")
        self.assertEqual(len(table), 18)
        self.assertEqual(len(table[0]), 6)
        self.assertEqual([r["experiment"] for r in table],
                         ["E1", "E2", "E3", "E4", "E5", "E6", "E10", "E13", "E14", "E17", "E19", "E17", "E17", "E17",
                          "E18", "admission", "oracle", "factorial"])
        self.assertEqual(table[0]["value"], "4 × 10^-3")                  # COCO bf16 0.0044; Roman fp32 excluded
        self.assertEqual(table[0]["threshold and consequence"], "within 8 × 10^-3")
        self.assertEqual(table[1]["value"], "0.35 [0.20, 0.55] / 0.25 [0.10, 0.45]")
        self.assertTrue(table[2]["value"].startswith("arxiv 2.9; proteins 2.2; Roman 4.1; Pascal 6.8"))
        self.assertTrue(table[3]["value"].startswith("arxiv -1.4; proteins -1.4"))
        self.assertTrue(table[5]["value"].startswith("arxiv 0.970, 6%"))
        self.assertTrue(table[5]["value"].endswith("p1 = —"))
        self.assertEqual(table[6]["value"], "Pascal 0.85; COCO 0.97; Peptides 0.45")
        self.assertEqual(table[7]["value"], "Pascal 0.21; COCO 0.22; Peptides 0.35")
        self.assertEqual(table[8]["value"], "arxiv 0.0305; proteins 0.0088; Roman 0.0088; Pascal 0.0455; "
                                            "COCO 0.0700; Peptides 0.0077")
        tail = {d: expected_cost_models(ctx, d) for d in ADMITTED}
        self.assertEqual(table[9]["value"], "; ".join(f"{experiments.SHORT[d]} {tail[d][0]:.0f}%" for d in ADMITTED))
        self.assertEqual(table[10]["value"], "Table S7")
        self.assertTrue(table[11]["value"].startswith("arxiv 13%; proteins 14%"))
        # M* = 3600 C_qual / (theta (C_exp - C_qC) t_nom), C_exp = (1 - p) C_qD + p C_qC (Table S23b definition)
        added = (1 - FALLBACK) * C_QD + FALLBACK * C_QC - C_QC
        m_star = 3600 * (GPU_TOTAL - GPU_PLANT) / (0.1 * added * 9.54)
        self.assertTrue(table[12]["value"].startswith(f"arxiv 9.5 s, {experiments.sci(m_star, 2)}"))
        self.assertIn("Pascal 0.73 ms", table[12]["value"])
        self.assertEqual(table[13]["quantity"], "sensing implementation on ogbn-proteins and PascalVOC-SP")
        self.assertEqual(table[13]["value"], f"reduced sensor: (k/d)/P_s ≤ {tail['ogbn-proteins'][1]:.2g} and ≤ "
                                             f"{tail['PascalVOC-SP'][1]:.2g}")
        self.assertEqual(table[14]["value"], "0.35–0.63 at λ0 = 0.1; 0.73 at 0.3; 0.82–0.86 at 1.0")
        roman = [-0.006, -0.007, -0.0065]
        mean = sum(roman) / 3
        sd = math.sqrt(sum((x - mean) ** 2 for x in roman) / 2)
        self.assertEqual(table[15]["value"], f"arxiv -0.0046, -0.0008; proteins -0.0046, -0.0008; "
                                             f"Roman {mean:.4f}, {mean + 4.303 * sd / math.sqrt(3):.4f}")
        self.assertEqual(table[16]["value"], "Pascal 40, 0.71; COCO 36, 0.69; Peptides 30, 0.45; products 16, 0.20")
        self.assertEqual(table[17]["value"], "-0.0062, -0.0059, -0.0049")
        self.assertIn("p1 of q_B is not a quantity", (self.out / "tables/S43.md").read_text())
        self.assertEqual(sum(1 for r in table if not r["value"]), 0)
        checked = resolved(rows)
        self.assertEqual(len(checked), len(rows))                          # every declared cell is transcribed
        groups = {r["group"] for r in checked}
        self.assertTrue({"supplementary_table_S42_power_of_budget_matched_test",
                         "supplementary_table_S46_budget_matched_frontier",
                         "supplementary_table_S45_inserted_site_attribution",
                         "supplementary_table_S26b_ensemble_law_and_break_even",
                         "supplementary_table_S23b_nominal_execution_time_and_amortization",
                         "supplementary_table_S30_oracle_dispersion", "supplementary_table_S16b_pascal_factorial",
                         "supplementary_table_S6c_sensing_cost_identity", "main_table_2_gain_and_cost_ledger",
                         "main_table_1_policy_and_deployment_ledger"} <= groups)
        exact = {(r["group"], r["path"]) for r in checked if r["status"] == "exact"}
        self.assertIn(("supplementary_table_S26b_ensemble_law_and_break_even", "ogbn-arxiv/D_inf_hat"), exact)
        self.assertIn(("supplementary_table_S16b_pascal_factorial",
                       "effect_coded_main_effects_nats/spectral_information"), exact)
        self.assertIn(("supplementary_table_S30_oracle_dispersion", "Roman-empire/phi_x_central"), exact)
        self.assertIn(("supplementary_table_S30_oracle_dispersion", "COCO-SP/median_oracle_depth"), exact)
        self.assertIn("data/S43/values.csv", writer.files)

    def test_missing_record_set(self):
        runs, records = s43_inputs(self.tmp)
        records.pop("factorials")
        with self.assertRaises(DisplayInputError):
            build_one("supplementary:Table S43", context(runs=runs, records=records), self.out)


if __name__ == "__main__":
    unittest.main()
