"""Tests of the analytical and arithmetic display generators (paper/build/displays/analytical.py).

The fixture runs and record sets are synthetic: every measured input (costs, effects, admission
moments, headroom, support counts, fallback probabilities, E3 intervals, E6 work, ensemble
losses, action gaps and the S14 contrasts) is set to the value the manuscript prints for that
input, so that the generated cells can be checked against the transcribed registry. Two Note S25
parameters that the registry does not hold (phi_OL, o_dec) are supplied only by the "complete"
fixture to exercise the formulas; the generators never invent them.
"""
from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from statistics import NormalDist

from paper.build.displays import analytical
from paper.build.displays.framework import ROOT, DATASETS, RunView
from paper.build.displays.testing import build_one, context, fake_records, read_table, write_files

REGISTRY = json.loads((ROOT / "paper/reported_results.json").read_text(encoding="utf-8"))
CONSTANTS = json.loads((ROOT / "paper/registered_constants.json").read_text(encoding="utf-8"))
Z = NormalDist().inv_cdf(0.975)
IDS = ["supplementary:Table S6", "supplementary:Table S6b", "supplementary:Table S14b",
       "supplementary:Table S28", "supplementary:Table S28b", "supplementary:Table S33",
       "supplementary:Table S34", "supplementary:Table S42", "supplementary:Table S44",
       "supplementary:Table S47", "supplementary:Figure S6", "supplementary:Figure S7",
       "supplementary:Figure S8", "supplementary:Figure S10", "supplementary:Figure S11",
       "supplementary:Figure S12", "supplementary:Figure S15", "supplementary:Figure S20"]
ADMITTED = analytical.ADMITTED

# ----------------------------------------------------------------------------- synthetic fixture
# Per-dataset inputs equal to the printed input values (Tables 1, S4, S13, S23, S26b, S28, S34, S42).
COSTS = {"ogbn-arxiv": (4.7, 4.0), "ogbn-proteins": (4.8, 4.0), "ogbn-products": (2.1, 2.1),
         "Roman-empire": (5.4, 4.0), "PascalVOC-SP": (10.5, 8.0), "COCO-SP": (10.8, 8.0), "Peptides-func": (4.9, 4.0)}
# Effects carry the unrounded digits that the printed values require (e.g. Table S28's n = 6,761 and Table
# S42's margin 6.32 put |Delta| for Pascal in [0.0084200, 0.0084202)).
EFFECTS = {"ogbn-arxiv": (-0.0045, -0.0075, -0.0015), "ogbn-proteins": (-0.0038, -0.0068, -0.0008),
           "ogbn-products": (0.0, 0.0, 0.0), "Roman-empire": (-0.00572005, -0.0089, -0.0026),
           "PascalVOC-SP": (-0.0084201, -0.01279, -0.00407), "COCO-SP": (-0.00905, -0.01268, -0.00542),
           "Peptides-func": (-0.00303, -0.00474, -0.00132)}
HEADROOM = {"ogbn-arxiv": -0.005, "ogbn-proteins": -0.0045, "ogbn-products": -0.0032, "Roman-empire": -0.021,
            "PascalVOC-SP": -0.026, "COCO-SP": -0.023, "Peptides-func": -0.0035}
QC_ROUTE = {"ogbn-arxiv": "fixed_sequence", "ogbn-proteins": "fixed_sequence", "ogbn-products": "phase",
            "Roman-empire": "mpc_ol", "PascalVOC-SP": "mpc_ol", "COCO-SP": "mpc_ol", "Peptides-func": "fixed_sequence"}
ENSEMBLE = {"ogbn-arxiv": 4, "ogbn-proteins": 4, "ogbn-products": 2, "Roman-empire": 4, "PascalVOC-SP": 8,
            "COCO-SP": 8, "Peptides-func": 4}
PLANT_LOSS = {"ogbn-arxiv": 0.87, "ogbn-proteins": 0.25, "ogbn-products": 0.5, "Roman-empire": 0.25,
              "PascalVOC-SP": 1.30, "COCO-SP": 2.00, "Peptides-func": 0.22}
D_INF = {"ogbn-arxiv": 0.0305, "ogbn-proteins": 0.0088, "Roman-empire": 0.0088, "PascalVOC-SP": 0.0455,
         "COCO-SP": 0.07, "Peptides-func": 0.0077}
ADMISSION = {"PascalVOC-SP": (-0.00836, 0.000214, 1428), "COCO-SP": (-0.00901, 0.000263, 5000),
             "Peptides-func": (-0.00311, 0.000126, 2331)}
SUPPORT = {"ogbn-arxiv": (128, 91), "ogbn-proteins": (144, 96), "ogbn-products": (192, 132), "Roman-empire": (96, 74),
           "PascalVOC-SP": (128, 88), "COCO-SP": (200, 142), "Peptides-func": (150, 104)}
FALLBACK = {"ogbn-arxiv": 0.118, "ogbn-proteins": 0.147, "Roman-empire": 0.129, "PascalVOC-SP": 0.081,
            "COCO-SP": 0.071, "Peptides-func": 0.188}
E3_SE = {"ogbn-arxiv": 0.0009945, "ogbn-proteins": 0.000839, "Roman-empire": 0.001827, "PascalVOC-SP": 0.000503,
         "COCO-SP": 0.000153, "Peptides-func": 0.0003403}
E6_WORK = {"ogbn-arxiv": (1.060, 1.055), "ogbn-proteins": (1.067, 1.062), "Roman-empire": (1.112, 1.108),
           "PascalVOC-SP": (1.100, 1.095), "COCO-SP": (1.105, 1.100), "Peptides-func": (1.094, 1.090)}
GAPS = {"ogbn-arxiv": (0.0062, 0.0011, 0.036), "ogbn-proteins": (0.0057, 0.0010, 0.041),
        "Roman-empire": (0.0049, 0.0008, 0.049), "PascalVOC-SP": (0.0054, 0.0009, 0.043),
        "COCO-SP": (0.0056, 0.0010, 0.041), "Peptides-func": (0.0036, 0.0005, 0.079)}
# Nine S14 contrasts at their original sizes: effect, interval, raw P of the original test, J_old.
CONTRASTS = {"arxiv/state": (0.00574, 0.00312, 0.00836, 0.0014, 32),
             "arxiv/radius": (-0.00286, -0.00481, -0.00091, 0.0030, 32),
             "proteins/state": (0.00485, 0.00264, 0.00706, 0.0016, 32),
             "proteins/radius": (-0.00242, -0.00406, -0.00077, 0.0034, 32),
             "products/state": (0.00036, -0.00064, 0.00136, 0.42, 64),
             "Roman/RH": (-0.00649, -0.01007, -0.00291, 0.0047, 10),
             "Pascal/RH": (-0.00902, -0.01355, -0.00449, 0.0001, 1428),
             "COCO/RH": (-0.00962, -0.01372, -0.00552, 0.00001, 5000),
             "Peptides/radius": (-0.00172, -0.00316, -0.00028, 0.019, 2331)}
SCENARIO = {"phi_OL": 0.20, "decoding_overhead_o_dec": 0.03}   # fixture-only values for the unregistered parameters


def evidence(dataset: str) -> dict:
    effect, lower, upper = EFFECTS[dataset]
    controller, comparator = COSTS[dataset]
    returns = dataset == "ogbn-products"
    record = {"final": {"paired_loss": {"mean": effect, "lower": lower, "upper": upper},
                        "models": {"qN": {"loss_nats": PLANT_LOSS[dataset]}}},
              "cost": {"C_qD": controller, "C_qC": comparator},
              "entry": {"headroom": {"upper": HEADROOM[dataset]}},
              "selection": {"qC_index": 0, "qC": {"route": QC_ROUTE[dataset], "ensemble": ENSEMBLE[dataset]}},
              "admission": {"deployed_index": 0 if returns else 1, "deployed": {"ensemble": ENSEMBLE[dataset]}}}
    if dataset in ADMISSION:
        d_bar, s2, units = ADMISSION[dataset]
        record["admission"]["empirical_bernstein"] = {"mean": d_bar, "sample_variance": s2, "units": units, "clip": 0.1}
    return record


def audit(dataset: str) -> dict:
    count, ess = SUPPORT[dataset]
    cells = {"0:0": {"trajectory_count": count, "ess": float(ess) + 0.4, "qualified": True},
             "0:1": {"trajectory_count": count + 40, "ess": float(ess) + 25.0, "qualified": True},
             "1:0": {"trajectory_count": count + 7, "ess": 31.0, "qualified": False}}
    return {"support": {"cells": cells}}


def records(complete: bool) -> dict:
    gate = fake_records("input_gate", [{"dataset": d, "quantity": "fallback_probability", "value": p}
                                       for d, p in FALLBACK.items()])
    arms = []
    for d in ADMITTED:
        se = E3_SE[d]
        arms += [{"dataset": d, "arm": "E3", "configuration": "selected", "quantity": "effect_vs_feedback_ci_low_nats",
                  "value": 0.002 - Z * se},
                 {"dataset": d, "arm": "E3", "configuration": "selected", "quantity": "effect_vs_feedback_ci_high_nats",
                  "value": 0.002 + Z * se},
                 {"dataset": d, "arm": "E6", "configuration": "selected", "quantity": "work_ratio", "value": E6_WORK[d][0]}]
        if complete:
            arms += [{"dataset": d, "arm": "E6", "configuration": "selected", "quantity": "late_abort_fraction",
                      "value": 0.010},
                     {"dataset": d, "arm": "E6", "configuration": "selected", "quantity": "work_ratio_completed",
                      "value": E6_WORK[d][1]}]
    grid = [{"dataset": d, "ensemble": e, "quantity": "loss_nats", "value": PLANT_LOSS[d] - D_INF[d] + D_INF[d] / e}
            for d in ADMITTED for e in (1, 2, 4, 8, 12, 16, 20)]
    gap_spec = {"keys": ["dataset"], "quantities": {}}
    gaps = [{"dataset": d, "quantity": q, "value": v} for d, values in GAPS.items()
            for q, v in zip(("median_gap_nats", "p5_gap_nats", "switch_fraction"), values)]
    contrast_spec = {"keys": ["contrast"], "quantities": {}}
    contrasts = [{"contrast": key, "quantity": q, "value": v} for key, values in CONTRASTS.items()
                 for q, v in zip(("effect_nats", "interval_low_nats", "interval_high_nats", "p_value_zero",
                                  "unit_count"), values)]
    return {"input_gate": gate, "budget_matched": fake_records("budget_matched", arms),
            "ensemble_grid": fake_records("ensemble_grid", grid),
            analytical.GAP_SET: fake_records(analytical.GAP_SET, gaps, spec=gap_spec),
            analytical.CONTRAST_SET: fake_records(analytical.CONTRAST_SET, contrasts, spec=contrast_spec)}


def build_context(root: Path, *, complete: bool = False, runs=DATASETS):
    views = {}
    for dataset in runs:
        folder = write_files(Path(root) / dataset, {"fit/full/audit.json": audit(dataset)})
        views[dataset] = RunView(dataset, folder, evidence=evidence(dataset))
    ctx = context(runs=views, records=records(complete))
    if complete:
        constants = json.loads(json.dumps(CONSTANTS))
        constants[analytical.BUDGET].update(SCENARIO)
        ctx.__dict__["constants"] = constants
    return ctx


def data_rows(out: Path, stem: str, name: str) -> list[dict]:
    with (Path(out) / "data" / stem / f"{name}.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


LAYOUT = {"S6": (10, 11), "S6b": (8, 8), "S14b": (9, 10), "S28": (7, 6), "S28b": (14, 5), "S33": (18, 12),
          "S34": (6, 9), "S42": (6, 9), "S44": (6, 9), "S47": (6, 9)}
FIGURE_DATA = {"FigS6": ("panel_a_multiplier", "panel_b_admissible_order", "panel_c_proteins_resolvent_fraction"),
               "FigS7": ("panel_a_certificate", "panel_b_gap_density"),
               "FigS8": ("panel_a_clipped_fraction", "panel_b_half_widths"),
               "FigS10": ("panel_a_sensing_share", "panel_b_residual", "panel_c_reduced_sensor"),
               "FigS11": ("panel_a_ledger", "panel_b_ledger"),
               "FigS12": ("power_curves", "central_margins"),
               "FigS15": ("panel_a_response", "panel_b_band_sensitivity", "panel_c_executed_range", "panel_d_headroom"),
               "FigS20": ("panel_a_break_even_points", "panel_a_break_even_curves", "panel_b_reach",
                          "panel_c_half_widths", "panel_d_seed_design")}
# Registry entries whose text differs from the current PDF, and the S47 ensemble-spend crossovers that the
# fixture-only o_dec cannot reproduce for every dataset at once.
EXPECTED_DIFFERENCES = {
    ("supplementary:Table S14b", "2/interpretation"),
    *(("supplementary:Table S14b", f"{i}/interpretation") for i in range(6)),
    ("supplementary:Table S42", "PascalVOC-SP/unit"), ("supplementary:Table S42", "COCO-SP/unit"),
    ("supplementary:Table S42", "Peptides-func/unit"),
    ("supplementary:Table S47", "Roman-empire/phi_B_ensemble"), ("supplementary:Table S47", "Peptides-func/phi_B_ensemble")}
# Supplementary Table S33 as printed (numeric cells after the dataset and scenario columns).
PRINTED_S33 = """
0.20 0.03 0.969 -0.0009 -0.0008 -0.0010 -0.0044 -0.0035 -0.0036 6.0%|0.35 0.08 0.969 -0.0016 -0.0014 -0.0013 -0.0044 -0.0028 -0.0029 6.0%
0.55 0.15 0.969 -0.0025 -0.0022 -0.0016 -0.0044 -0.0019 -0.0020 6.0%|0.20 0.03 0.966 -0.0008 -0.0007 -0.0009 -0.0037 -0.0029 -0.0030 6.7%
0.35 0.08 0.966 -0.0013 -0.0012 -0.0011 -0.0037 -0.0023 -0.0025 6.7%|0.55 0.15 0.966 -0.0021 -0.0019 -0.0013 -0.0037 -0.0016 -0.0017 6.7%
0.10 0.03 0.961 -0.0006 -0.0005 -0.0002 -0.0055 -0.0049 -0.0051 11.2%|0.25 0.08 0.961 -0.0014 -0.0013 -0.0005 -0.0055 -0.0041 -0.0043 11.2%
0.45 0.15 0.961 -0.0026 -0.0023 -0.0009 -0.0055 -0.0029 -0.0031 11.2%|0.10 0.03 0.965 -0.0008 -0.0008 -0.0003 -0.0081 -0.0073 -0.0076 10.0%
0.25 0.08 0.965 -0.0021 -0.0019 -0.0007 -0.0081 -0.0060 -0.0063 10.0%|0.45 0.15 0.965 -0.0038 -0.0034 -0.0013 -0.0081 -0.0043 -0.0046 10.0%
0.10 0.03 0.966 -0.0009 -0.0008 -0.0003 -0.0087 -0.0078 -0.0081 10.5%|0.25 0.08 0.966 -0.0023 -0.0020 -0.0007 -0.0087 -0.0065 -0.0068 10.5%
0.45 0.15 0.966 -0.0041 -0.0037 -0.0014 -0.0087 -0.0047 -0.0050 10.5%|0.20 0.03 0.946 -0.0006 -0.0005 -0.0007 -0.0029 -0.0023 -0.0024 9.4%
0.35 0.08 0.946 -0.0011 -0.0010 -0.0008 -0.0029 -0.0018 -0.0020 9.4%|0.55 0.15 0.946 -0.0017 -0.0015 -0.0011 -0.0029 -0.0012 -0.0014 9.4%
"""


class AnalyticalDisplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.ctx = build_context(root / "inputs", complete=True)
        cls.out = root / "displays"
        cls.results = {display_id: build_one(display_id, cls.ctx, cls.out, registry=REGISTRY) for display_id in IDS}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def table(self, stem):
        return read_table(self.out, stem)

    # -- layout, files and comparisons ----------------------------------------------------------
    def test_complete_printed_layouts(self):
        for stem, (rows, columns) in LAYOUT.items():
            with self.subTest(stem=stem):
                table = self.table(stem)
                self.assertEqual(len(table), rows)
                self.assertEqual(len(table[0]), columns)
                for suffix in ("csv", "md", "tex"):
                    self.assertTrue((self.out / "tables" / f"{stem}.{suffix}").is_file())

    def test_every_figure_panel_has_source_data(self):
        for stem, names in FIGURE_DATA.items():
            with self.subTest(stem=stem):
                for suffix in ("png", "svg"):
                    self.assertTrue((self.out / "figures" / f"{stem}.{suffix}").is_file())
                for name in names:
                    self.assertTrue(data_rows(self.out, stem, name), name)

    def test_comparisons_resolve_and_reproduce_the_printed_values(self):
        for display_id, (writer, rows) in self.results.items():
            with self.subTest(display=display_id):
                self.assertFalse([r for r in rows if r["status"] == "no_reference"])
                differing = {(display_id, r["path"]) for r in rows if r["status"] != "exact"}
                self.assertEqual(differing, {d for d in EXPECTED_DIFFERENCES if d[0] == display_id})
        transcribed = {i for i, (_, rows) in self.results.items() if rows}
        self.assertEqual(transcribed, set(IDS) - {"supplementary:Table S6", "supplementary:Table S6b",
                                                    "supplementary:Table S28", "supplementary:Table S28b",
                                                    "supplementary:Table S33", "supplementary:Figure S11"})

    def test_generators_never_read_the_registry(self):
        source = Path(analytical.__file__).read_text(encoding="utf-8")
        self.assertNotIn("reported_results", source)

    # -- registry tables ---------------------------------------------------------------------------
    def test_s6_registry_and_identity_mass(self):
        rows = {r["ID"]: r for r in self.table("S6")}
        self.assertEqual(rows["A3"]["identity mass"], f"{0.5 ** 9:.4f}")        # (1 - eta)^(T+1) = 0.0020
        self.assertEqual(rows["A7"]["identity mass"], f"{0.92 ** 9:.4f}")       # 0.4722
        self.assertEqual([rows[f"A{i}"]["sparse products"] for i in range(10)], ["1"] * 3 + ["8"] * 5 + ["0"] * 2)
        self.assertEqual(rows["A5"]["mode/range"], "η = 0.20")
        self.assertEqual(rows["A8"]["ρ"], "")
        registered = CONSTANTS["action_registry_table_S6"]["actions"]
        self.assertEqual([rows[f"A{a['id']}"]["explore p_ctrl"] for a in registered],
                         [f"{a['exploration_probability']:.2f}" for a in registered])

    def test_s6b_equals_the_registered_derivations(self):
        derived = CONSTANTS["action_registry_table_S6"]["derived_table_S6b"]
        for row in self.table("S6b"):
            reg = derived[row["ID"]]
            with self.subTest(action=row["ID"]):
                self.assertAlmostEqual(float(row["kernel hops"]), reg["kernel_hops"], places=3)
                self.assertAlmostEqual(float(row["executed range h̄_a (hops/step)"]),
                                       reg["executed_range_hops_per_step"], places=3)
                self.assertAlmostEqual(float(row["executed identity coefficient"]),
                                       reg["executed_identity_coefficient"], places=3)
                self.assertEqual(int(row["max. reach (hops)"]), reg["max_reach_hops"])
                self.assertEqual(row["kernel identity mass"] or None,
                                 None if reg["kernel_identity_mass"] is None else f"{reg['kernel_identity_mass']:.4f}")
        # A7 by hand: identity coefficient (1 - nu) + nu rho [eta + (1 - eta)^9]
        a7 = next(r for r in self.table("S6b") if r["ID"] == "A7")
        self.assertEqual(a7["executed identity coefficient"], f"{0.5 + 0.5 * 0.25 * (0.08 + 0.92 ** 9):.3f}")

    # -- confirmation-size sensitivity -----------------------------------------------------------
    def test_s14b_rescales_only_the_graph_rows(self):
        rows = {r["Dataset/claim"]: r for r in self.table("S14b")}
        self.assertEqual(rows["arxiv/state"]["95% interval (J_new)"], "same")
        self.assertEqual(rows["products/state"]["P vs margin"], "not tested")
        pascal = rows["Pascal/RH"]
        share = (Z * math.sqrt(0.000214 / 1428) / ((0.01279 - 0.00407) / 2)) ** 2    # (hw_graph / hw_test)^2
        se = (0.01355 - 0.00449) / (2 * Z) * math.sqrt((1 - share) + share * 1428 / 425)
        self.assertEqual(pascal["95% interval (J_new)"], f"[{-0.00902 - Z * se:+.5f}, {-0.00902 + Z * se:+.5f}]")
        self.assertEqual(pascal["interpretation"], "replanning retained")
        self.assertEqual((rows["COCO/RH"]["raw P"], rows["COCO/RH"]["Holm P"]), ("< 10^-4", "< 0.001"))
        self.assertTrue(rows["Peptides/radius"]["interpretation"].startswith("interval excludes zero"))

    # -- regret bounds -----------------------------------------------------------------------------
    def test_s28_hoeffding_bound(self):
        rows = {r["Dataset"]: r for r in self.table("S28")}
        arxiv = rows["ogbn-arxiv"]
        self.assertEqual(arxiv["2e at minimum count (nats)"], f"{0.4 * math.sqrt(math.log(400) / 256):.4f}")  # 0.0612
        self.assertEqual(arxiv["n for the dataset's effect"], "23,670")       # 8 B^2 log(2|A|/delta) / 0.0045^2
        self.assertEqual(rows["ogbn-products"]["n for the dataset's effect"], "")
        printed = {"ogbn-proteins": ("144", "96", "0.0577", "0.0707", "33,194"),
                   "Roman-empire": ("96", "74", "0.0707", "0.0805", "14,650"),
                   "PascalVOC-SP": ("128", "88", "0.0612", "0.0738", "6,761"),
                   "COCO-SP": ("200", "142", "0.0490", "0.0581", "5,852"),
                   "Peptides-func": ("150", "104", "0.0565", "0.0679", "52,208")}
        for dataset, values in printed.items():
            self.assertEqual(tuple(list(rows[dataset].values())[1:]), values, dataset)

    def test_s28b_bernstein_grid(self):
        rows = [list(r.values()) for r in self.table("S28b")]
        self.assertEqual([r[0] for r in rows[:9]], ["64", "74", "91", "100", "142", "500", "1,000", "3,018", "10,000"])
        self.assertEqual(rows[0][1:], ["0.0865", "0.1036", "0.1054", "0.1082"])
        self.assertEqual(rows[7][1:], ["0.0126", "0.0027", "0.0030", "0.0034"])
        self.assertEqual(rows[9][0], "target (nats)")
        self.assertEqual(rows[10], ["0.0030 (smallest admitted effect)", "53,257", "2,716", "3,018", "3,529"])
        self.assertEqual(rows[11], ["0.0045 (arxiv effect)", "23,670", "1,725", "1,881", "2,139"])
        self.assertEqual(rows[12], ["0.0084 (Pascal effect)", "6,793", "873", "930", "1,022"])
        # Table 1's COCO effect -0.00905 gives the four-decimal target 0.0091 (S28b prints 0.0090, S26b 0.0091).
        self.assertEqual(rows[13][0], "0.0091 (COCO effect)")
        self.assertEqual(rows[13][1], f"{round(8 * 0.01 * math.log(400) / 0.0091 ** 2):,}")

    # -- retained-share, selective sensing, power and boundaries -------------------------------------
    def test_s33_reproduces_the_printed_model(self):
        printed = [line.split() for chunk in PRINTED_S33.split("\n") if chunk for line in chunk.split("|")]
        self.assertEqual([list(r.values())[2:] for r in self.table("S33")], printed)

    def test_s34_gap_model(self):
        rows = {r["Dataset"]: r for r in self.table("S34")}
        sigma = (math.log(6.2) - math.log(1.1)) / NormalDist().inv_cdf(0.95)
        skip = 1 - NormalDist().cdf((math.log(2.0) - math.log(6.2)) / sigma)
        self.assertEqual(rows["arxiv"]["skip rate"], f"{skip:.2f}")                       # 0.86
        self.assertEqual(rows["arxiv"]["retained fraction"], f"{1 - skip * 0.036:.3f}")   # 0.969
        self.assertEqual(rows["arxiv"]["present overhead"], "15.4%")   # (1 - 0.118)(4.7/4 - 1)
        self.assertEqual(rows["Peptides"]["gap median / p5 (mnats)"], "3.6 / 0.5")

    def test_s42_power(self):
        rows = {r["Dataset"]: r for r in self.table("S42")}
        arxiv = rows["arxiv"]
        self.assertEqual(arxiv["unit"], "32 buffered cores")
        self.assertEqual(rows["Roman"]["unit"], "10 official splits")
        self.assertEqual(arxiv["MDE (mnats): 0.05 / Holm step"],
                         f"{(Z + NormalDist().inv_cdf(0.8)) * 0.9945:.2f} / 3.32")
        self.assertEqual(arxiv["margin FB−E3 [range]"], "2.93 [2.02, 3.60]")              # (1 - phi) x 4.5 mnats
        self.assertEqual(arxiv["power at the unit [range]"], "84% [53, 95]")
        self.assertEqual(arxiv["power, seed-paired: cons.–opt."], "93–100%")
        self.assertEqual(arxiv["power, ensemble spend"], "55%")

    def test_s44_tail_model(self):
        rows = {r["Dataset"]: r for r in self.table("S44")}
        self.assertEqual(rows["proteins"]["f̄"], "0.063 (identity)")
        self.assertEqual(rows["proteins"]["modelled C_0.99 (nominal executions)"], "5.83 / 6.86 / 8.41")
        r = 1166243 / (169343 * 512)
        v = 7 * r / (1 + r)
        self.assertEqual(rows["arxiv"]["v"], f"{v:.3f}")
        self.assertEqual(rows["arxiv"]["modelled C_0.99 (nominal executions)"],
                         " / ".join(f"{4.0 * (1 + 0.175 - 0.25 * v + f * v):.2f}" for f in (0.35, 0.45, 0.60)))

    def test_s47_boundaries(self):
        rows = {r["Dataset"]: r for r in self.table("S47")}
        self.assertEqual(rows["arxiv"]["φ_x^gain grid (central)"], "0.20–0.55 (0.35)")
        self.assertEqual(rows["Roman"]["φ_x^gain grid (central)"], "0.10–0.45 (0.25)")
        self.assertEqual(rows["arxiv"]["φ*_FB, contract"], "1")
        self.assertEqual(rows["arxiv"]["power at the unit [range]; seed-paired range (unit)"],
                         "84% [53, 95]; seed-paired 93–100% (32 buffered cores)")
        self.assertEqual(rows["Pascal"]["MDE (mnats): unit / seed-paired cons.–opt."], "1.4 / 1.7–0.4")
        # phi*_FB = 1 - b(o - o_dec)/g, b(o') = D_inf o' / [E (1 + o')], D_inf = kappa l_plant
        o = (1 - 0.118) * 4.7 / 4.0 + 0.118 - 1
        b = [k * 0.87 * (o - 0.03) / (4 * (1 + o - 0.03)) for k in (0.03, 0.04)]
        self.assertEqual(rows["arxiv"]["φ*_FB, ensemble spend"], f"{1 - b[1] / 0.0045:.2f}–{1 - b[0] / 0.0045:.2f}")

    # -- figures -------------------------------------------------------------------------------------
    def test_figure_s6_admissible_orders(self):
        bars = data_rows(self.out, "FigS6", "panel_b_admissible_order")
        labels = {(r["dataset"], float(r["resolvent_step_share"])): r["label"] for r in bars}
        self.assertEqual((labels["ogbn-arxiv", 0.25], labels["ogbn-arxiv", 0.5]), ("36", "18"))
        self.assertEqual((labels["ogbn-proteins", 0.25], labels["ogbn-proteins", 0.5]), ("3", "2"))
        self.assertEqual(labels["Roman-empire", 0.5], ">64")
        fractions = {int(r["T"]): float(r["f_res"]) for r in data_rows(self.out, "FigS6", "panel_c_proteins_resolvent_fraction")}
        r = 39561252 / (132534 * 512)
        self.assertAlmostEqual(fractions[8], (1.2 * (1 + r) - (1 + r + 0.06)) / (7 * r), places=12)  # 0.063

    def test_figure_s7_lognormal_peak(self):
        curve = [r for r in data_rows(self.out, "FigS7", "panel_b_gap_density") if r["dataset"] == "ogbn-arxiv"]
        sigma = float(curve[0]["sigma_log"])
        self.assertAlmostEqual(max(float(r["density_per_log_gap"]) for r in curve), 1 / (sigma * math.sqrt(2 * math.pi)),
                               places=3)

    def test_figure_s8_clipping_bound(self):
        rows = {r["dataset"]: r for r in data_rows(self.out, "FigS8", "panel_a_clipped_fraction")}
        self.assertAlmostEqual(float(rows["PascalVOC-SP"]["f_max"]), (1427 * 0.000214 / 1428 + 0.00836 ** 2) / 0.01)
        self.assertEqual(f"{100 * float(rows['COCO-SP']['f_max']):.2f}", "3.44")
        ratios = {r["dataset"]: float(r["ratio"]) for r in data_rows(self.out, "FigS8", "panel_b_half_widths")}
        self.assertEqual([f"{ratios[d]:.1f}" for d in ("PascalVOC-SP", "COCO-SP", "Peptides-func")], ["5.7", "8.1", "3.8"])

    def test_figure_s10_sensing_identity(self):
        residual = {r["dataset"]: float(r["residual_share"]) for r in data_rows(self.out, "FigS10", "panel_b_residual")}
        self.assertEqual(f"{100 * residual['ogbn-proteins']:.0f}", "-275")
        sketch = data_rows(self.out, "FigS10", "panel_c_reduced_sensor")
        self.assertEqual([int(r["sketch_width"]) for r in sketch], [4, 8, 16, 32, 64, 128, 256, 512])
        self.assertAlmostEqual(float(sketch[0]["relative_sd_bound_pct"]), 100 * math.sqrt(2 / 4))

    def test_figure_s11_ledgers(self):
        released = [float(r["accumulated_work"]) for r in data_rows(self.out, "FigS11", "panel_a_ledger")]
        held = [float(r["accumulated_work"]) for r in data_rows(self.out, "FigS11", "panel_b_ledger")]
        self.assertEqual(released, [6.0, 8.0, 8.0, 12.0])     # restart after releasing the reserve exceeds W = 10
        self.assertEqual(held, [6.0, 6.0, 10.0])              # Lemma 3: total work at most W

    def test_figure_s12_and_s20_seed_design(self):
        points = {r["dataset"]: r for r in data_rows(self.out, "FigS12", "central_margins")}
        self.assertEqual(f"{float(points['Roman-empire']['power_conservative']):.0f}", "76")
        design = [r for r in data_rows(self.out, "FigS20", "panel_d_seed_design") if r["dataset"] == "ogbn-arxiv"]
        self.assertEqual(len(design), 12)
        self.assertAlmostEqual(float(design[-1]["power_conservative"]), float(points["ogbn-arxiv"]["power_conservative"]))

    def test_figure_s15_band_sensitivity_and_headroom(self):
        registered = CONSTANTS["action_registry_table_S6"]["band_sensitivity_figure_S15b"]
        for row in data_rows(self.out, "FigS15", "panel_b_band_sensitivity"):
            self.assertEqual(f"{float(row['sensitivity_upper_bound']):.2f}",
                             f"{registered[row['action']][int(row['band'])]:.2f}")
        headroom = {r["dataset"]: float(r["unrealized_lower_bound_mnats"])
                    for r in data_rows(self.out, "FigS15", "panel_d_headroom")}
        self.assertAlmostEqual(headroom["ogbn-arxiv"], 5.0 - 4.5)
        self.assertAlmostEqual(headroom["ogbn-products"], 3.2)

    def test_figure_s20_reach_ratio_matches_the_e18_audit(self):
        rows = [r for r in data_rows(self.out, "FigS20", "panel_b_reach") if r["action"] == "A7"]
        largest = rows[-1]
        self.assertGreater(float(largest["upper_bound"]), float(largest["exact_displacement"]))
        self.assertAlmostEqual(float(rows[0]["upper_bound"]), 0.07, delta=0.003)     # band-0 sensitivity as lambda_0 -> 0


class MissingInputTests(unittest.TestCase):
    def test_unregistered_parameters_leave_their_cells_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = build_context(Path(tmp) / "inputs")
            out = Path(tmp) / "displays"
            for display_id in ("supplementary:Table S33", "supplementary:Table S34", "supplementary:Table S42",
                               "supplementary:Table S47"):
                build_one(display_id, ctx, out)
            s33 = read_table(out, "S33")
            self.assertEqual({r["no-observation replanning vs q_C"] for r in s33 if r["Dataset"] == "arxiv"}, {""})
            self.assertTrue(all(r["no-observation replanning vs q_C"] for r in s33 if r["Dataset"] == "Roman"))
            self.assertIn("phi_OL is not registered", (out / "tables/S33.md").read_text(encoding="utf-8"))
            self.assertEqual({r["power, ensemble spend"] for r in read_table(out, "S42")}, {""})
            self.assertEqual({r["φ*_FB, ensemble spend"] for r in read_table(out, "S47")}, {""})
            self.assertEqual({(r["p1"], r["q_B overhead at p1 = 0"]) for r in read_table(out, "S34")}, {("", "")})

    def test_absent_record_sets_await_inputs(self):
        from paper.build.displays.framework import build_displays
        with tempfile.TemporaryDirectory() as tmp:
            ctx = build_context(Path(tmp) / "inputs")
            del ctx._records[analytical.GAP_SET]
            report = build_displays(ctx, Path(tmp) / "displays", registry=REGISTRY,
                                    only={"supplementary:Table S34", "supplementary:Figure S7", "supplementary:Table S44"})
            status = {r["id"]: r["status"] for r in report["displays"]}
            self.assertEqual(status, {"supplementary:Table S34": "awaiting_inputs", "supplementary:Figure S7": "awaiting_inputs",
                                      "supplementary:Table S44": "built"})

    def test_bundled_example_runs(self):
        arxiv = RunView("ogbn-arxiv", ROOT / "examples/runs/arxiv")
        peptides = RunView("Peptides-func", ROOT / "examples/runs/peptides_func")
        ctx = context(runs={"ogbn-arxiv": arxiv, "Peptides-func": peptides})
        cells = json.loads((ROOT / "examples/runs/arxiv/fit/full/audit.json").read_text())["support"]["cells"].values()
        support = analytical.support_counts(ctx, "ogbn-arxiv")
        self.assertEqual(support["minimum_count"], min(c["trajectory_count"] for c in cells))
        self.assertEqual(support["reference_ess"], int(min(c["ess"] for c in cells if c["qualified"]) + 1e-9))
        dep = analytical.deployment(ctx, "ogbn-arxiv")
        self.assertTrue(dep["returns_qC"])
        self.assertEqual(dep["m"], 1.0)
        with self.assertRaises(analytical.DisplayInputError):      # zero-width synthetic intervals
            analytical.graph_variance_share(ctx, "Peptides-func")

    def test_degenerate_example_runs_build_or_await(self):
        # The bundled runs return q_C with zero-width intervals and empty support cells: every display either
        # builds or reports the missing premise; none fails with another error.
        runs = {d: RunView(d, ROOT / ("examples/runs/peptides_func" if d in analytical.GRAPH_LEVEL
                                      else "examples/runs/arxiv")) for d in DATASETS}
        ctx = context(runs=runs, records=records(True))
        built = set()
        with tempfile.TemporaryDirectory() as tmp:
            for display_id in IDS:
                try:
                    build_one(display_id, ctx, Path(tmp))
                    built.add(display_id)
                except analytical.DisplayInputError:
                    pass
            s28 = {r["Dataset"]: r for r in read_table(Path(tmp), "S28")}
        self.assertEqual(built, {"supplementary:Table S6", "supplementary:Table S6b", "supplementary:Table S28",
                                 "supplementary:Figure S7", "supplementary:Figure S11", "supplementary:Figure S15"})
        self.assertEqual((s28["ogbn-arxiv"]["minimum count"], s28["ogbn-arxiv"]["2e at minimum count (nats)"]), ("0", ""))


if __name__ == "__main__":
    unittest.main()
