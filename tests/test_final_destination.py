"""The final destination: design forecast, pre-registered roles and the projected release.

The forecast is checked against what the released runtime actually wrote in the two bundled
runs: their candidate families and their recorded role hashes. NumPy only; no PyTorch.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import final_destination as fd  # noqa: E402
from scripts import paper_artifacts as pa  # noqa: E402

EXAMPLES = ROOT / "examples" / "runs"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class RuntimeParity(unittest.TestCase):
    """The forecast reproduces what the runtime wrote, not what the paper prints."""

    def test_registry_size_reproduces_the_bundled_candidate_families(self):
        for run in sorted(p for p in EXAMPLES.iterdir() if p.is_dir()):
            config = read(run / "config.json")
            decision = read(run / "select" / "decision.json")
            candidates = read(run / "select" / "candidates.json")
            size = fd.registry_size(config)
            self.assertEqual(size["candidates"], decision["candidate_count"], run.name)
            self.assertEqual(size["candidates"], len(candidates), run.name)
            routes = {}
            for c in candidates:
                routes[c["route"]] = routes.get(c["route"], 0) + 1
            per_size = len(size["ensemble_sizes"])
            self.assertEqual({k: v * per_size for k, v in size["families"].items()}, routes, run.name)

    def test_lifted_role_functions_reproduce_the_recorded_role_hashes(self):
        functions = fd.runtime_role_functions(ROOT)
        for run in sorted(p for p in EXAMPLES.iterdir() if p.is_dir()):
            metadata = read(run / "prepared" / "metadata.json")
            config = read(run / "config.json")
            official = dict(np.load(run / "prepared" / "official_splits.npz", allow_pickle=False))
            independent = metadata.get("graph_role_hashes") is not None
            n = sum(len(v) for v in official.values()) if independent else metadata["num_nodes"]
            roles = functions["_make_roles"](official, n, int(config["seed"]), float(config["data"]["internal_val_fraction"]))
            hashes = {k: functions["_array_sha256"](v) for k, v in roles.items()}
            self.assertEqual(hashes, metadata["graph_role_hashes"] if independent else metadata["role_hashes"], run.name)

    def test_role_functions_are_lifted_from_the_runtime_source(self):
        text = (ROOT / "cross_morse" / "data.py").read_text(encoding="utf-8")
        for name in ("_make_roles", "_array_sha256", "_validate_roles", "_indices", "_json_bytes"):
            self.assertIn(f"def {name}(", text)
        with mock.patch.object(Path, "read_text", lambda self, encoding=None: "def other():\n    pass\n"):
            with self.assertRaises(fd.DestinationError):
                fd.runtime_role_functions(ROOT)


class Forecast(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = fd.Inputs(ROOT)
        cls.roles = fd.preregistered_roles(cls.inputs, fd.runtime_role_functions(ROOT))
        cls.rows = fd.forecast(cls.inputs, cls.roles)

    def test_preregistered_roles_match_the_registered_table_s2_counts(self):
        for dataset, value in self.roles.items():
            counts = self.inputs.multi[dataset]
            self.assertEqual(value["sizes"]["confirm"], counts["D_confirm"], dataset)
            self.assertEqual(value["sizes"]["compare"], counts["D_compare"], dataset)
            self.assertEqual(value["sizes"]["admit"], counts["D_admit"], dataset)
            self.assertEqual(value["sizes"]["test"], counts["D_test"], dataset)
            leaves = ("fit_train", "fit_val", "explore", "select", "confirm", "compare", "admit", "test")
            self.assertEqual(sum(value["sizes"][r] for r in leaves), self.inputs.statistics[dataset]["graphs"])

    def test_peptides_row_roles_are_its_graph_roles(self):
        self.assertEqual(self.roles["Peptides-func"]["role_hashes"], self.roles["Peptides-func"]["graph_role_hashes"])
        self.assertIsNone(self.roles["PascalVOC-SP"]["role_hashes"])

    def test_every_forecast_value_is_derived_from_the_released_configuration(self):
        for row in self.rows:
            config = self.inputs.configs[row["dataset"]]
            if row["quantity"].startswith("|Q|"):
                self.assertEqual(row["forecast"], fd.registry_size(config)["candidates"])
            elif row["quantity"] == "exploration trajectories":
                self.assertEqual(row["forecast"], config["exploration"]["trajectories"])
            elif row["quantity"] == "controlled depth L":
                self.assertEqual(row["forecast"], config["model"]["depth"])
            if row["quantity"].startswith("|Q|"):
                # |Q| = |Q_feas| (main Eq. 18) is measured; the registry bounds it from above.
                expected = ("not printed" if row["printed"] is None else
                            "within bound" if row["printed"] <= row["forecast"] else "exceeds bound")
            else:
                expected = ("not printed" if row["printed"] is None else
                            "agrees" if row["forecast"] == row["printed"] else "differs")
            self.assertEqual(row["status"], expected, row)

    def test_every_design_value_agrees_with_the_paper_or_lies_within_its_bound(self):
        conflicts = [r for r in self.rows if r["status"] in ("differs", "exceeds bound")]
        self.assertEqual(conflicts, [])
        self.assertEqual(sum(1 for r in self.rows if r["status"] == "within bound"), 7)


class Projection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = fd.Inputs(ROOT)
        cls.roles = fd.preregistered_roles(cls.inputs, fd.runtime_role_functions(ROOT))

    def test_counts_follow_the_runtime_layout_and_the_release_rules(self):
        for dataset in fd.DATASETS:
            config = self.inputs.configs[dataset]
            run = fd.project_run(self.inputs, dataset, self.roles)
            copies = fd.ROMAN_SPLITS if dataset == "Roman-empire" else 1
            seeds, folds = len(config["training"]["seeds"]), config["controller"]["crossfit_folds"]
            profiles = len(config["sensor"]["profiles"])
            checkpoints = 2 * 2 * seeds + profiles * (1 + 5 + 5 * folds)
            self.assertEqual(run["contents"]["checkpoints"], checkpoints * copies, dataset)
            self.assertEqual(run["contents"]["propensity_logged_trajectories"],
                             2 * config["exploration"]["trajectories"] * copies, dataset)
            self.assertGreater(run["bytes"]["packed_complete"], run["bytes"]["packed_analysis"])
            self.assertEqual(run["parts"]["complete"], max(1, -(-run["bytes"]["packed_complete"] // pa.PART_BYTES)))

    def test_analysis_tier_omits_exactly_the_heavy_files(self):
        families = fd.run_families(self.inputs, "ogbn-arxiv", self.roles)
        heavy = {f["path"] for f in families if pa.is_heavy(f["path"])}
        self.assertIn("train/plant_seed1_best.pt", heavy)
        self.assertIn("predictions/0000000000000000000a.npy", heavy)
        self.assertNotIn("explore/trajectory_000000.npz", heavy)
        self.assertNotIn("prepared/splits.npz", heavy)

    def test_projections_are_marked_and_published_values_replace_them(self):
        doc = fd.build(ROOT)
        self.assertTrue(all(r["projected"] for r in doc["runs"]))
        self.assertTrue(all(r["realized"] is None for r in doc["runs"]))
        manifest = pa.load_manifest(ROOT / "release/paper_artifacts.json")
        published = copy.deepcopy(manifest)
        for entry in published["artifacts"]:
            if entry["id"] == "run:Peptides-func":
                entry.update(status=pa.PUBLISHED, files=1234, bytes=10 ** 9,
                             contents={"checkpoints": 110, "predictions": 1971, "propensity_logged_trajectories": 48000,
                                       "statistical_outputs": 16, "split_and_seed_manifests": 5, "environment": 1,
                                       "configurations": 2},
                             tiers={"analysis": {"bytes": 2 * 10 ** 9, "parts": [{}]},
                                    "complete": {"bytes": 3 * 10 ** 9, "parts": [{}, {}]}})
        with mock.patch.object(pa, "load_manifest", lambda path=pa.MANIFEST: published):
            realized = fd.realized_run(next(a for a in published["artifacts"] if a["id"] == "run:Peptides-func"))
        self.assertEqual(realized["files"], 1234)
        self.assertEqual(realized["parts"]["complete"], 2)
        self.assertFalse(realized["projected"])

    def test_projected_values_never_enter_the_statement_evaluations(self):
        from scripts import statement_check
        before = statement_check.evaluate(ROOT)["items"]
        fd.build(ROOT)
        self.assertEqual(statement_check.evaluate(ROOT)["items"], before)


class Outputs(unittest.TestCase):
    def test_committed_outputs_are_current(self):
        doc = fd.build(ROOT)
        self.assertEqual((ROOT / fd.OUT_JSON).read_text(encoding="utf-8"),
                         json.dumps(doc, indent=1, ensure_ascii=False) + "\n",
                         "release/final_destination.json is not current: make final-destination")
        self.assertEqual((ROOT / fd.OUT_MD).read_text(encoding="utf-8"), fd.render(doc) + "\n",
                         "docs/FINAL_DESTINATION.md is not current: make final-destination")

    def test_projected_values_are_red_and_tables_stay_well_formed(self):
        text = (ROOT / fd.OUT_MD).read_text(encoding="utf-8")
        self.assertIn(r"$\color{red}{\textsf{", text)
        for line in text.splitlines():
            if line.startswith("| Table S20 |"):
                self.assertEqual(line.count(" | "), 5, line)  # |Q| is escaped inside its cell

    def test_runs_report_compares_each_run_with_its_own_configuration(self):
        doc = fd.build(ROOT, EXAMPLES)
        rows = {r["run"]: r for r in doc["runs_report"]}
        self.assertEqual(set(rows), {"arxiv", "peptides_func"})
        for row in rows.values():
            self.assertTrue(row["registry_agrees"])
            self.assertFalse(row["benchmark"])
            self.assertIsNone(row["printed_Q"])
            self.assertTrue(row["role_sizes_follow_the_rule"])
            decision = read(EXAMPLES / row["run"] / "select" / "decision.json")
            self.assertEqual(row["feasible_family"], sum(1 for f in decision["joint_feasible"] if f is True))
            self.assertNotIn("configuration_is_registered", row)   # smoke runs are not compared with the paper


if __name__ == "__main__":
    unittest.main()
