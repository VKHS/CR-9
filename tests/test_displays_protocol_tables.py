"""Protocol and definition displays render from the repository's configuration and code.

Static displays are built from the real configurations, registered constants and
runtime source. S5 and S5d read small synthetic runs (written here) and the two
bundled example runs; their values exercise the computation, not the paper's results.
"""
from __future__ import annotations

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from paper.build.displays import protocol_tables as P
from paper.build.displays.framework import (DATASETS, GRAPH_LEVEL, RunView, build_displays,
                                           DisplayValidationError, load_references)
from paper.build.displays.testing import build_one, context, fake_run, read_table

ROOT = Path(__file__).resolve().parents[1]
ARXIV = ROOT / "examples" / "runs" / "arxiv"
PEPTIDES = ROOT / "examples" / "runs" / "peptides_func"
REGISTRY = json.loads((ROOT / "paper" / "reported_results.json").read_text(encoding="utf-8"))

# Printed rows x columns of every display (supplementary PDF pages in paper/display_manifest.json).
LAYOUT = {"S1": (7, 8), "S2": (7, 9), "S3": (4, 4), "S5": (7, 9), "S5b": (7, 4), "S5d": (4, 3), "S8": (13, 6),
          "S9": (9, 3), "S10": (9, 7), "S11": (4, 5), "S24": (7, 4), "S31": (34, 4), "S32": (9, 3),
          "S35": (29, 4), "S36": (20, 5), "S38": (12, 3), "S39": (8, 4), "S40": (7, 4)}
STATIC = {"S1": "supplementary:Table S1", "S2": "supplementary:Table S2", "S3": "supplementary:Table S3",
          "S5b": "supplementary:Table S5b", "S8": "supplementary:Table S8", "S9": "supplementary:Table S9",
          "S10": "supplementary:Table S10", "S11": "supplementary:Table S11", "S24": "supplementary:Table S24",
          "S31": "supplementary:Table S31", "S32": "supplementary:Table S32", "S35": "supplementary:Table S35",
          "S36": "supplementary:Table S36", "S38": "supplementary:Table S38", "S39": "supplementary:Table S39",
          "S40": "supplementary:Table S40"}


def table_shape(out: Path, stem: str) -> tuple[int, int]:
    rows = read_table(out, stem)
    return len(rows), len(rows[0])


def read_data(out: Path, stem: str, name: str) -> list[dict]:
    with (out / "data" / stem / f"{name}.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def example_runs() -> dict[str, RunView]:
    """Node-level datasets on the arxiv example, graph-level on Peptides; Roman as a two-split campaign."""
    runs = {}
    for dataset in DATASETS:
        root = PEPTIDES if dataset in GRAPH_LEVEL else ARXIV
        runs[dataset] = (RunView(dataset, root, splits=[RunView(dataset, root), RunView(dataset, root)])
                         if dataset == "Roman-empire" else RunView(dataset, root))
    return runs


def synthetic_run(root: Path, dataset: str, *, parameters: int, seconds: list[float], gpu: str = "GPU-X") -> RunView:
    """A run holding exactly the files S5 reads, with the dataset's full configuration."""
    cfg = json.loads((ROOT / "configs" / f"{P.STEMS[dataset]}_full.json").read_text(encoding="utf-8"))
    cfg["training"]["seeds"] = list(range(1, len(seconds) + 1))
    checkpoints = [{"path": f"train/plant_seed{s}_best.pt", "parameter_count": parameters, "prediction_role": "plant",
                    "sha256": f"{s:064d}"} for s in cfg["training"]["seeds"]]
    checkpoints.append({"path": "train/q0_seed1_best.pt", "parameter_count": 7, "prediction_role": "q0"})
    files = {"config.json": cfg, "environment.json": {"gpu": gpu, "device": "cuda"},
             "train/model_inventory.json": {"checkpoints": checkpoints}}
    for seed, value in zip(cfg["training"]["seeds"], seconds):
        files[f"train/plant_seed{seed}_done.json"] = {"seconds_this_attempt": value, "epochs_completed": 3}
    return fake_run(root, dataset, files)


class StaticLayoutTests(unittest.TestCase):
    """Every static display is complete and writes its source data."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name)
        cls.writers = {stem: build_one(display_id, context(), cls.out)[0] for stem, display_id in STATIC.items()}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_printed_layouts(self):
        for stem in STATIC:
            with self.subTest(stem=stem):
                self.assertEqual(table_shape(self.out, stem), LAYOUT[stem])
                for suffix in ("csv", "md", "tex"):
                    self.assertIn(f"tables/{stem}.{suffix}", self.writers[stem].files)
                self.assertTrue(any(f.startswith(f"data/{stem}/") for f in self.writers[stem].files))

    def test_protocol_text_carries_page_provenance(self):
        manifest = {d["id"]: d for d in json.loads((ROOT / "paper/display_manifest.json").read_text())["displays"]}
        for path in sorted((ROOT / "paper" / "protocol").glob("*.json")):
            with self.subTest(file=path.name):
                record = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(record["source"]["document"], "supplementary")
                self.assertTrue(set(record["source"]["pages"]) <= set(manifest[record["display"]]["pages"]))

    def test_s1_units_and_admission_from_configuration(self):
        rows = {r["Dataset"]: r for r in read_table(self.out, "S1")}
        self.assertEqual(rows["ogbn-arxiv"]["J or repeated units"], "32 METIS cores; 2-hop buffer")
        self.assertEqual(rows["ogbn-products"]["J or repeated units"], "64 METIS cores; 2-hop buffer")
        self.assertEqual(rows["ogbn-proteins"]["primary unit"], "degree-stratified buffered cluster")
        self.assertEqual(rows["Roman-empire"]["primary unit"], "official split")
        self.assertEqual(rows["Roman-empire"]["J or repeated units"], "10 splits")
        # The primary unit already names the degree-stratified partition; the printed cell names only the cores.
        self.assertEqual(rows["ogbn-proteins"]["J or repeated units"], "32 cores; 2-hop buffer")
        self.assertEqual(rows["Roman-empire"]["interval role"], "repeated-split empirical")
        self.assertEqual([rows[d]["interval role"] for d in ("ogbn-arxiv", "ogbn-proteins", "ogbn-products")],
                         ["descriptive"] * 3)
        self.assertEqual([rows[d]["J or repeated units"] for d in GRAPH_LEVEL], ["J = 1428", "J = 5000", "J = 2331"])
        self.assertEqual([rows[d]["formal gate?"] for d in DATASETS], ["no"] * 4 + ["yes"] * 3)
        self.assertIn("B = 0.10, δ_a = 0.025", rows["COCO-SP"]["admission criterion for G_admit"])
        self.assertIn("ε_a = 0.0015", rows["ogbn-proteins"]["admission criterion for G_admit"])
        self.assertIn("clipped-BCE", rows["ogbn-proteins"]["admission criterion for G_admit"])

    def test_s2_allocation_folds_and_seeds(self):
        rows = {r["Stage"]: r for r in read_table(self.out, "S2")}
        self.assertTrue(rows["D_fit"]["frozen allocation"].startswith("60% of official training"))
        self.assertEqual([rows[f"D_{r}"]["frozen allocation"][:3] for r in ("explore", "select")], ["20%", "10%"])
        self.assertTrue(rows["D_compare"]["frozen allocation"].startswith("5%; reserved at preparation"))
        self.assertEqual((rows["D_select"]["outer folds"], rows["D_select"]["inner folds"]), ("5", "4"))
        self.assertEqual(rows["D_explore"]["inner folds"], "5 trajectory folds")
        self.assertEqual(rows["D_fit"]["plant seeds"], "12")
        self.assertEqual(rows["D_admit"]["frozen allocation"], "official validation; disjoint from checkpointing")
        sizes = {r["dataset"]: (int(r["confirm"]), int(r["compare"])) for r in read_data(self.out, "S2",
                                                                                        "multigraph_role_sizes")}
        self.assertEqual(sizes, {"PascalVOC-SP": (425, 425), "COCO-SP": (5664, 5665), "Peptides-func": (544, 544)})

    def test_s3_thresholds_and_operators_from_code(self):
        rows = {r["Gate"]: r for r in read_table(self.out, "S3")}
        self.assertEqual(rows["nominal graph value"]["prespecified criterion"], "U(Δ_plant) < −0.004 nats")
        self.assertEqual(rows["depth qualification"]["prespecified criterion"], "PDD ≤ 1.0 pp; DIL gap ≤ 0.020")
        self.assertEqual(rows["registry support"]["prespecified criterion"], "ϕ_registry ≥ 0.80; selected strata = 1.00")
        self.assertEqual(rows["residual headroom"]["prespecified criterion"], "U(Δ_oracle) < −0.003 nats")
        # One executed response for every gate: feedback candidates become infeasible and q_C is retained.
        self.assertEqual({r["failure action"] for r in rows.values()}, {"retain strongest open loop"})
        strata = [c for c in self.writers["S3"].comparisons if c.path == ("registry_support", "selected_strata")]
        self.assertEqual([c.computed for c in strata], [1.0])
        thresholds = {r["key"]: r for r in read_data(self.out, "S3", "thresholds")}
        self.assertEqual((thresholds["selected_strata_fraction"]["operator"],
                          thresholds["selected_strata_fraction"]["value"]), ("=", "1.0"))

    def test_s5b_site_arithmetic(self):
        rows = read_table(self.out, "S5b")
        self.assertEqual(rows[0]["A0 reproduces native aggregation"], "yes; native (ρ, ν, α) = (0, 0.70, 0.25)")
        self.assertIn("dense updates chunked at 65,536 rows", rows[2]["native call wrapped by the adapter"])
        self.assertTrue(rows[4]["controlled depth index ℓ versus native layers"].startswith(
            "128 controlled transport sites: 21 sub-steps per native block plus 2 boundary sites"))
        sites = {r["dataset"]: r for r in read_data(self.out, "S5b", "sites")}
        self.assertEqual((sites["PascalVOC-SP"]["native_sites"], sites["PascalVOC-SP"]["inserted_sites"]), ("6", "122"))
        self.assertEqual((sites["Peptides-func"]["native_sites"], sites["Peptides-func"]["inserted_sites"]), ("8", "56"))

    def test_s8_dimensions_follow_the_sensor_schema(self):
        rows = {r["measurement"]: r for r in read_table(self.out, "S8")}
        self.assertEqual(rows["reachability statistic"]["dim."], "7")
        self.assertEqual(rows["innovation/NIS"]["dim."], "3")
        self.assertEqual(rows["entropy"]["normalization"], "training-fold median/MAD; clip [−5, 5]")
        self.assertEqual(rows["band power"]["period"], "sensing period P_s")
        self.assertEqual(rows["innovation/NIS"]["period"], "every update")
        cheap = sum(int(r["dim."]) for r in rows.values() if r["period"] == "every depth"
                    and not r["measurement"].startswith("spectral age"))
        spectral = sum(int(r["dim."]) for r in rows.values() if "P_s" in r["period"])
        self.assertEqual((cheap, spectral), (9, 16))

    def test_s9_values_from_configuration_and_model_code(self):
        rows = {r["object"]: r["implemented primary value"] for r in read_table(self.out, "S9")}
        self.assertEqual(rows["action critic"], "MLP 128–128–64, SiLU; dropout 0.10 after the first two hidden "
                                                "layers; mean and log-scale outputs, log-scale clipped to [−7, 3]")
        self.assertEqual(rows["sequence models"], "128-wide input, two 128-wide residual layers, SiLU and a linear "
                                                  "output layer")
        self.assertEqual(rows["filters"], "6 candidates, 3 active, degree 8")
        self.assertEqual(rows["horizons / beam / top-k"], "{1, 4, 8, 16, 32}/16/2")
        self.assertEqual(rows["loss clipping"], "primary B = 0.10; sensitivity B ∈ {0.05, 0.20}")

    def test_s10_and_s11_registries(self):
        rows = read_table(self.out, "S10")
        self.assertEqual([r["feedback?"] for r in rows], ["no"] * 6 + ["yes"] * 3)
        self.assertEqual(rows[5]["route"], "MPC-OL")
        margins = {r["provisional route"]: r["practical margin"] for r in read_table(self.out, "S11")}
        self.assertEqual(margins, {"state-conditioned route": "0.002 nats", "radius": "0.0015 nats",
                                   "MPC-RH": "0.002 nats", "Full profile": "0.02 fraction"})
        fail = {r["provisional route"]: r["fail/equality output"] for r in read_table(self.out, "S11")}
        self.assertEqual(fail["radius"], "one-step if one-step ∈ Q_s, otherwise q_C")

    def test_linkage_tables_name_existing_artifacts(self):
        for stem in ("S31", "S35"):
            with self.subTest(stem=stem):
                rows = read_data(self.out, stem, "linkage")
                self.assertTrue(rows)
                self.assertTrue(all(r["verified"] == "True" for r in rows if r["kind"] != "absent"))
        records = {r["quantity (location)"]: r["repository location"] for r in read_table(self.out, "S31")}
        self.assertEqual(records["main Table 1: U_a (Pascal, COCO, Peptides)"],
                         "run archive: admit/decision.json, admit/paired_units.npz, admit/prediction_execution.json")
        descriptions = {r["quantity (location)"]: r["record"] for r in read_table(self.out, "S31")}
        self.assertEqual(descriptions["main Table 1: U_a (Pascal, COCO, Peptides)"],
                         "runs/<dataset>/admission.json")
        s24 = read_table(self.out, "S24")
        self.assertTrue(s24[0]["reconstruction test"].endswith("(make reexecute)"))

    def test_s36_s38_s39_derived_cells(self):
        e14 = read_table(self.out, "S36")[13]
        self.assertIn("E ∈ {1, 2, 4, 8, 12, 16, 20}", e14["question and controlled comparison"])
        self.assertIn("E = 16 and 20 require", e14["question and controlled comparison"])
        roles = {r["field"]: r["role"] for r in read_table(self.out, "S38")}
        self.assertEqual(roles["dataset"], "key")
        self.assertEqual(roles["data stage"], "not a record column")
        self.assertEqual(roles["observation identifier"], "provenance")
        s39 = {r["stage"]: r for r in read_table(self.out, "S39")}
        self.assertEqual(s39["confirmation (main Eq. (20))"]["multigraph benchmarks (Pascal, COCO, Peptides)"],
                         "D_confirm (425, 5,664 and 544 graphs)")
        self.assertIn("5,665", s39["independent E3–E6 comparison"]["multigraph benchmarks (Pascal, COCO, Peptides)"])


class ComparisonTests(unittest.TestCase):
    """Declared comparisons resolve against printed values and expose code/paper disagreements.

    A restricted injected reference still exercises missing-path behavior. The
    default build uses the full PDF-bound protocol reference sidecar.
    """

    PRINTED = {
        P.S2_GROUP: {"D_select": {"outer_folds": 5, "inner_folds": 4, "controller_seeds": 12},
                     "multigraph_role_sizes": {"PascalVOC-SP": [425, 425], "COCO-SP": [5664, 5665],
                                               "Peptides-func": [544, 544]}},
        P.S3_GROUP: {"nominal_graph_value": {"epsilon_plant_nats": 0.004},
                     "depth_qualification": {"PDD_pp": 1.0, "DIL_gap_nats": 0.020},
                     "registry_support": {"phi_registry_min": 0.80, "selected_strata": 1.00}},
        P.S11_GROUP: {"radius": {"margin": 0.0015, "fail_output": "one-step if one-step ∈ Q_s, otherwise q_C"},
                      "full": {"margin": 0.02, "fail_output": "Lean"}},
    }

    def statuses(self, display_id):
        with tempfile.TemporaryDirectory() as tmp:
            _, rows = build_one(display_id, context(), Path(tmp), registry=self.PRINTED)
        return {r["path"]: r["status"] for r in rows}

    def test_printed_cells_resolve(self):
        s2 = self.statuses("supplementary:Table S2")
        self.assertEqual([s2[f"multigraph_role_sizes/{d}"] for d in GRAPH_LEVEL], ["exact"] * 3)
        self.assertEqual((s2["D_select/outer_folds"], s2["D_select/inner_folds"]), ("exact", "exact"))
        # The runtime fits the observer and consequence models once per profile, not per controller seed.
        self.assertEqual(s2["D_select/controller_seeds"], "missing")
        s3 = self.statuses("supplementary:Table S3")
        self.assertEqual({k: v for k, v in s3.items() if not k.endswith("failure_action")},
                         {"nominal_graph_value/epsilon_plant_nats": "exact", "depth_qualification/PDD_pp": "exact",
                          "depth_qualification/DIL_gap_nats": "exact", "registry_support/phi_registry_min": "exact",
                          "registry_support/selected_strata": "exact",
                          "residual_headroom/epsilon_headroom_nats": "no_reference"})
        s11 = self.statuses("supplementary:Table S11")
        self.assertEqual((s11["radius/margin"], s11["radius/fail_output"], s11["full/margin"]),
                         ("exact", "exact", "exact"))
        self.assertEqual(s11["full/fail_output"], "differs")

    def test_proposed_groups_are_named_consistently(self):
        groups = [getattr(P, name) for name in dir(P) if name.endswith("_GROUP")]
        self.assertEqual(len(groups), len(set(groups)))
        self.assertTrue(all(g.startswith("supplementary_table_S") for g in groups))

    def test_real_pdf_bound_references_cover_all_declared_protocol_comparisons(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = build_displays(context(), Path(tmp), only=STATIC.values())
        by_id = {d["id"]: d for d in report["displays"]}
        self.assertFalse(report["errors"])
        for display in by_id.values():
            self.assertEqual(display["comparison"]["no_reference"], 0, display["id"])
        self.assertEqual(by_id[STATIC["S9"]]["comparison"]["exact"], 18)
        self.assertEqual(by_id[STATIC["S31"]]["comparison"]["exact"], 34)
        self.assertEqual(by_id[STATIC["S35"]]["comparison"]["exact"], 29)
        # The prospective selected-strata gate now executes the printed criterion;
        # unsupported seed replication and different failure actions remain visible.
        self.assertEqual(by_id[STATIC["S3"]]["comparison"]["missing"], 0)
        self.assertGreater(by_id[STATIC["S3"]]["comparison"]["differs"], 0)
        self.assertGreater(by_id[STATIC["S38"]]["comparison"]["differs"], 0)
        self.assertTrue(by_id[STATIC["S2"]]["not_applicable_cells"])

    def test_changed_pdf_identity_invalidates_printed_reference_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "paper").mkdir()
            for name in ("reported_results.json", "sources.json", "display_manifest.json", "printed_display_values.json"):
                shutil.copyfile(ROOT / "paper" / name, root / "paper" / name)
            path = root / "paper/printed_display_values.json"
            reference = json.loads(path.read_text())
            reference["documents"]["supplementary"] = "f" * 64
            path.write_text(json.dumps(reference))
            with self.assertRaisesRegex(DisplayValidationError, "current registered PDFs"):
                load_references(root)
        self.assertIn(P.S5D_GROUP, REGISTRY)


class SourceReadingTests(unittest.TestCase):
    """The literal readers reproduce what the runtime executed in the bundled runs."""

    def test_role_sizes_match_the_prepared_example(self):
        sizes = json.loads((ARXIV / "prepared/metadata.json").read_text())["split_sizes"]
        train = sum(sizes[r] for r in ("fit", "explore", "select", "confirm", "compare"))
        self.assertEqual(P.role_sizes(train), {r: sizes[r] for r in ("fit", "explore", "select", "confirm", "compare")})
        self.assertEqual(P.role_sizes(8498)["confirm"], 425)
        self.assertEqual((P.role_sizes(113286)["confirm"], P.role_sizes(113286)["compare"]), (5664, 5665))

    def test_measurement_schema_matches_the_recorded_sensor_schema(self):
        records = [json.loads(path.read_text())
                   for path in sorted((ARXIV / "member_predictions").glob("*.json"))]
        sensors = [record["sensor_configuration"] for record in records if record.get("sensor_configuration")]
        self.assertTrue(sensors, "the bundled run must include sensing executions")
        ctx = context()
        for sensor in sensors:
            with self.subTest(profile=sensor["profile"]):
                self.assertEqual(P.measurement_schema(ctx, sensor["bands"], sensor["sensing_period"]), sensor["schema"])
        periodic = P.measurement_schema(ctx, [0, 2, 5], 4)
        self.assertEqual((len(periodic), periodic[-1]), (26, "spectral_age_fraction"))

    def test_site_rule_matches_recorded_instrumentation(self):
        cfg = json.loads((PEPTIDES / "config.json").read_text())
        recorded = json.loads((PEPTIDES / "train/plant_seed1_done.json").read_text())["instrumentation"]
        sites = P.designated_sites(context(), cfg)
        self.assertEqual((sites["native"], sites["inserted"]),
                         (recorded["native_site_count"], recorded["inserted_site_count"]))
        self.assertEqual(sites["substeps"], cfg["model"]["depth"] // cfg["model"]["native_layers"])
        gps = P.designated_sites(context(), context().config("PascalVOC-SP"))
        self.assertEqual((gps["substeps"], gps["boundary"], gps["native"], gps["inserted"]), (21, 2, 6, 122))
        self.assertEqual(P.designated_sites(context(), context().config("ogbn-products"))["native"], 64)

    def test_registries_read_from_runtime_source(self):
        ctx = context()
        routes = P.route_registry(ctx)
        self.assertEqual([r for r, _, _ in routes], ["nominal", "best_depth", "learned_stop", "phase",
                                                     "fixed_sequence", "mpc_ol", "one_step", "radius", "mpc_rh"])
        margins = P.confirmation_margins(["state", "radius", "mpc", "full"])
        self.assertEqual({k: v["margin"] for k, v in margins.items()},
                         {"state": 0.002, "radius": 0.0015, "mpc": 0.002, "full": 0.02})
        self.assertEqual(P.confirmation_predecessors(ctx),
                         {"state": "comparator", "radius": "one_step", "mpc": "mpc_ol", "full": "lean"})
        self.assertEqual(P.campaign_seed_count(ctx), 12)
        self.assertEqual(P.results_csv_columns(ctx)[:2], ["model", "ensemble"])
        self.assertEqual(P.training_tolerances(ctx), {"bf16_state_tolerance": "8 × 10^-3",
                                                      "fp32_state_tolerance": "10^-5", "loss_tolerance": "10^-4"})

    def test_printed_paths_the_repository_lacks_are_reported(self):
        ctx = context()
        self.assertTrue(P.check_linkage(ctx, {"records": ["admission"]}, "R01"))
        self.assertTrue(P.check_linkage(ctx, {"run": ["runs/ogbn-arxiv/admission.json"]}, "R02"))
        self.assertFalse(P.check_linkage(ctx, {"run": ["admit/decision.json", "admit/prediction_execution.json"]}, "R02"))

    def test_placeholders(self):
        self.assertEqual(P.fill("T = {order}; {0.005, 0.007}", {"order": 8}), "T = 8; {0.005, 0.007}")
        with self.assertRaises(P.DisplayInputError):
            P.fill("{missing}", {})
        self.assertEqual((P.sci(0.0003), P.sci(1e-05), P.sci(0.05)), ("3e-4", "1e-5", "0.05"))


class MeasurementDisplayTests(unittest.TestCase):
    """S5 and S5d read run inventories, training records and identity audits."""

    def test_s5_counts_and_hours_on_known_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = {}
            for i, dataset in enumerate(DATASETS):
                if dataset == "Roman-empire":
                    splits = [synthetic_run(root / f"roman_{k}", dataset, parameters=2_520_000,
                                            seconds=[1800.0 * (k + 1), 1800.0 * (k + 1)]) for k in range(2)]
                    runs[dataset] = RunView(dataset, root / "roman", splits=splits)
                else:
                    runs[dataset] = synthetic_run(root / P.STEMS[dataset], dataset, parameters=3_116_632 + i,
                                                  seconds=[3600.0, 7200.0])
            writer, rows = build_one("supplementary:Table S5", context(runs=runs), root / "out", registry=REGISTRY)
            table = {r["Dataset"]: r for r in read_table(root / "out", "S5")}
            self.assertEqual(table_shape(root / "out", "S5"), LAYOUT["S5"])
            self.assertEqual(table["ogbn-arxiv"]["params (M)"], "3.12")
            self.assertEqual(table["ogbn-arxiv"]["hardware / time per seed"], "GPU-X; 1.50 h/seed")
            # Roman: mean over two splits x two seeds of 0.5 h and 1.0 h.
            self.assertEqual(table["Roman-empire"]["hardware / time per seed"], "GPU-X; 0.75 h/seed")
            self.assertTrue(table["Roman-empire"]["checkpoint rule"].endswith("2 seeds per split"))
            self.assertEqual(table["ogbn-arxiv"]["optimizer and schedule"],
                             "AdamW 3e-4, wd 1e-4, cosine; 500 epochs, patience 50")
            self.assertEqual(table["Peptides-func"]["optimizer and schedule"],
                             "AdamW 1e-3, wd 1e-5, constant; batch 128; 250 epochs, patience 30")
            self.assertTrue(table["Peptides-func"]["checkpoint rule"].startswith("maximum internal-D_fit validation AP"))
            self.assertIn("sigmoid AP head", table["Peptides-func"]["architecture / head"])
            inventory = read_data(root / "out", "S5", "inventory")
            self.assertEqual(len(inventory), 6 * 2 + 2 * 2)
            self.assertEqual({c.path[1] for c in writer.comparisons},
                             {"depth", "hidden", "parameters_M", "plant_precision", "learning_rate", "weight_decay",
                              "epochs", "patience", "seeds", "hours_per_seed"})

    def test_s5_rejects_inconsistent_checkpoint_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = {d: synthetic_run(root / P.STEMS[d], d, parameters=10, seconds=[1.0]) for d in DATASETS}
            inventory = root / "arxiv" / "train" / "model_inventory.json"
            record = json.loads(inventory.read_text())
            record["checkpoints"].append({"path": "train/plant_seed9_best.pt", "parameter_count": 11})
            inventory.write_text(json.dumps(record))
            with self.assertRaises(P.DisplayInputError):
                build_one("supplementary:Table S5", context(runs=runs), root / "out")

    def test_s5d_on_the_bundled_runs_resolves_every_registry_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            writer, rows = build_one("supplementary:Table S5d", context(runs=example_runs()), out, registry=REGISTRY)
            self.assertEqual(table_shape(out, "S5d"), LAYOUT["S5d"])
            self.assertTrue(rows and all(r["status"] != "no_reference" for r in rows))
            status = {r["path"]: r["status"] for r in rows}
            for path in ("ogbn-arxiv/released_default_parameters", "ogbn-arxiv/inventory_artifact",
                         "ogbn-arxiv/comparison_command", "ogbn-arxiv/released_default_scope", "sensor_fields"):
                self.assertEqual(status[path], "exact", path)
            plants = {r["dataset"]: r for r in read_data(out, "S5d", "plants")}
            self.assertEqual(plants["ogbn-arxiv"]["parameters"], "1396")
            self.assertEqual((plants["Peptides-func"]["native_sites"], plants["Peptides-func"]["inserted_sites"]),
                             ("2", "6"))
            identity = read_data(out, "S5d", "identity")
            self.assertTrue(identity and all(r["pass"] == "True" for r in identity))

    def test_s5d_reconciles_a_full_profile_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = example_runs()
            arxiv = root / "arxiv"
            shutil.copytree(ARXIV, arxiv, ignore=shutil.ignore_patterns("*.pt", "*.npy", "predictions",
                                                                         "member_predictions", "explore"))
            cfg = json.loads((ROOT / "configs/arxiv_full.json").read_text())
            cfg["training"]["seeds"] = [1, 2]
            (arxiv / "config.json").write_text(json.dumps(cfg))
            inventory = json.loads((arxiv / "train/model_inventory.json").read_text())
            for row in inventory["checkpoints"]:
                row["parameter_count"] = 3_116_632
            (arxiv / "train/model_inventory.json").write_text(json.dumps(inventory))
            runs["ogbn-arxiv"] = RunView("ogbn-arxiv", arxiv)
            writer, rows = build_one("supplementary:Table S5d", context(runs=runs), root / "out", registry=REGISTRY)
            status = {r["path"]: r["status"] for r in rows}
            for path in ("ogbn-arxiv/benchmark_parameters_M", "ogbn-arxiv/full_profile_parameters",
                         "ogbn-arxiv/benchmark_architecture"):
                self.assertEqual(status[path], "exact", path)


class FigureAndBuildTests(unittest.TestCase):

    def test_figure_1_panels_and_source_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            writer, _ = build_one("main:Figure 1", context(), out)
            self.assertEqual(sorted(f for f in writer.files if f.startswith("figures/")),
                             ["figures/Fig1.png", "figures/Fig1.svg"])
            nodes = read_data(out, "Fig1", "nodes")
            self.assertEqual({n["panel"] for n in nodes}, {"1. Offline development and qualification",
                                                          "2. Executable deployment logic", "3. Evidence and reporting"})
            families = next(n["content"] for n in nodes if n["node"] == "families")
            self.assertIn("best depth, learned stop, phase, fixed sequence, MPC-OL", families)
            self.assertIn("one-step, radius, MPC-RH", families)
            self.assertTrue(read_data(out, "Fig1", "edges"))

    def test_build_displays_runs_every_assigned_generator(self):
        with tempfile.TemporaryDirectory() as tmp:
            wanted = [*STATIC.values(), "supplementary:Table S5", "supplementary:Table S5d", "main:Figure 1"]
            report = build_displays(context(runs=example_runs()), Path(tmp), registry=REGISTRY, only=wanted)
            status = {r["id"]: r for r in report["displays"]}
            self.assertEqual({i: status[i]["status"] for i in wanted}, {i: "built" for i in wanted})
            self.assertEqual({status[i]["scope"] for i in STATIC.values()}, {"protocol"})
            self.assertEqual(status["main:Figure 1"]["scope"], "illustration")
            self.assertEqual({status[i]["scope"] for i in ("supplementary:Table S5", "supplementary:Table S5d")},
                             {"measurement"})


if __name__ == "__main__":
    unittest.main()
