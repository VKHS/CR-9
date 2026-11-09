"""The destination tree: projected publication files, the twelve items on them, the guards and convergence.

The exact values of the projection are recomputed here independently of scripts/destination.py: the
configuration identities from run_cross_morse.py and cross_morse/provenance.py, the runtime fingerprint from
cross_morse's file recipe, the role sizes from the registered floors. NumPy only; no PyTorch.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import destination  # noqa: E402
from scripts import final_destination as fd  # noqa: E402
from scripts import paper_artifacts as pa  # noqa: E402
from scripts import statement_check as sc  # noqa: E402


def fill(value):
    """A realized file in the shape of its projection: every pending value replaced by a concrete one."""
    if isinstance(value, dict):
        return {k: fill(v) for k, v in value.items() if k != destination.MARK}
    if isinstance(value, list):
        return [fill(v) for v in value]
    if destination.is_pending(value):
        return "0" * 64
    return value


class Tree(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc, cls.tree = fd.build_all(ROOT)

    def test_every_json_file_is_marked_as_a_projection_of_its_real_path(self):
        files = [p for p in self.tree if p.endswith(".json")]
        self.assertEqual(len(files), 17)
        for relative in files:
            value = self.tree[relative]
            self.assertTrue(destination.is_projection(value), relative)
            self.assertEqual(value[destination.MARK]["replaces"], relative)
            self.assertIn("exact", value[destination.MARK]["classes"], relative)
        self.assertIn("README.md", self.tree)
        self.assertIn("STATEMENT_COMPLIANCE.md", self.tree)

    def test_committed_tree_is_current(self):
        self.assertEqual(destination.stale(ROOT, self.tree), [],
                         "destination/ is not current: python -B scripts/final_destination.py --write")

    def test_configuration_identities_and_runtime_are_recomputed_exactly(self):
        import run_cross_morse as entry
        from cross_morse.provenance import _json_hash, current_source_sha256
        self.assertEqual(self.doc["destination"]["fingerprint"], current_source_sha256())
        for dataset in fd.DATASETS:
            stem = fd.STEMS[dataset]
            manifest = self.tree[f"manifests/published/{stem}.json"]
            identity = next(a for a in self.tree["release/paper_artifacts.json"]["artifacts"]
                            if a["id"] == f"run:{dataset}")["identity"]
            if dataset == "Roman-empire":
                for index, part in enumerate(manifest["campaign"]["per_split"]):
                    config = entry.resolve_configuration(dataset=dataset, profile="full", split_index=index)
                    self.assertEqual(part["identities"]["config_sha256"], _json_hash(config))
                    self.assertEqual(identity["splits"][f"split_{index:02d}"]["config_sha256"], _json_hash(config))
                    self.assertEqual(part["identities"]["source_sha256"], current_source_sha256())
                self.assertEqual(len({p["identities"]["config_sha256"] for p in manifest["campaign"]["per_split"]}), 10)
            else:
                config = entry.resolve_configuration(dataset=dataset, profile="full")
                self.assertEqual(manifest["identities"]["config_sha256"], _json_hash(config))
                self.assertEqual(identity["config_sha256"], _json_hash(config))
                self.assertEqual(manifest["identities"]["source_sha256"], current_source_sha256())
            self.assertEqual(manifest["seeds"]["bootstrap_seeds"],
                             {"selection": 314159 + 4, "admission": 314159 + 6, "final": 314159 + 7})
            self.assertEqual(len(manifest["seeds"]["training_seeds"]), 12)

    def test_role_sizes_follow_the_registered_floors(self):
        for dataset, (train, valid, test) in destination.OFFICIAL_NODE_SPLITS.items():
            sizes = self.tree[f"manifests/published/{fd.STEMS[dataset]}.json"]["splits"]["sizes"]
            cuts = [int(train * f) for f in (0.6, 0.8, 0.9, 0.95)]
            self.assertEqual([sizes[k] for k in ("fit", "explore", "select", "confirm", "compare")],
                             [cuts[0], cuts[1] - cuts[0], cuts[2] - cuts[1], cuts[3] - cuts[2], train - cuts[3]])
            self.assertEqual((sizes["admit"], sizes["test"]), (valid, test))
            self.assertEqual(sizes["fit_val"], max(1, int(sizes["fit"] * 0.1)))
            shape = destination.provider_shapes(ROOT)["expected"][dataset]
            self.assertEqual(train + valid + test, shape[0], dataset)
        for dataset in fd.INDEPENDENT:
            splits = self.tree[f"manifests/published/{fd.STEMS[dataset]}.json"]["splits"]
            self.assertNotIn("graph_split_sizes", splits)                 # not a field of the executed manifest
            self.assertEqual(splits["graph_role_hashes"], self.doc["preregistered_roles"][dataset]["graph_role_hashes"])
            if dataset in destination.GRAPH_LABELS:
                self.assertEqual(splits["sizes"], self.doc["preregistered_roles"][dataset]["sizes"])
            else:
                self.assertTrue(destination.is_pending(splits["sizes"]))

    def test_compared_paths_exist_in_the_files_publication_writes(self):
        """Every exact or registered value names a field that the real writers emit (bundled runs)."""
        from scripts import run_manifests
        for stem in ("arxiv", "peptides_func"):
            run = ROOT / "examples" / "runs" / stem
            with tempfile.TemporaryDirectory() as tmp:
                real = destination.normalize(run_manifests.run_manifest(run, with_indices=False, out_dir=Path(tmp), stem=stem))
            projected = self.tree[f"manifests/published/{stem}.json"]
            classes = projected[destination.MARK]["classes"]
            body = destination.normalize({k: v for k, v in projected.items() if k != destination.MARK})
            checked = 0
            for path, value in destination.leaves(body):
                if destination.classify(path, classes) in ("exact", "registered"):
                    self.assertIsNot(destination.lookup(real, path), destination.MISSING, (stem, path))
                    checked += 1
            self.assertGreater(checked, 25)
            environment = json.loads((run / "environment.json").read_text(encoding="utf-8"))
            observed = self.tree[f"environment/observed/{stem}.json"]
            for key in observed["members"][0]["environment"]:
                self.assertIn(key, environment, key)

    def test_lrgb_official_split_hashes_are_those_of_the_official_file_order(self):
        import numpy as np
        functions = fd.runtime_role_functions(ROOT)
        counts = fd.Inputs(ROOT).multi
        for dataset in fd.INDEPENDENT:
            n_train, n_valid, n_test = (counts[dataset][k] for k in ("official_train", "D_admit", "D_test"))
            official = {"train": np.arange(n_train), "valid": np.arange(n_train, n_train + n_valid),
                        "test": np.arange(n_train + n_valid, n_train + n_valid + n_test)}
            expected = {k: functions["_array_sha256"](v.astype(np.int64)) for k, v in official.items()}
            splits = self.tree[f"manifests/published/{fd.STEMS[dataset]}.json"]["splits"]
            self.assertEqual(splits["official_split_hashes"], expected, dataset)

    def test_release_lists_every_required_artifact_published(self):
        release = self.tree["release/paper_artifacts.json"]
        required = [a["id"] for a in pa.required_artifacts(ROOT)]
        self.assertEqual([a["id"] for a in release["artifacts"]], required)
        self.assertEqual(len(required), 37)
        self.assertTrue(all(a["status"] == pa.PUBLISHED for a in release["artifacts"]))
        for entry in release["artifacts"]:
            if entry["kind"] == "run":
                self.assertEqual(set(entry["tiers"]), {"analysis", "complete"})
                self.assertGreater(entry["contents"]["checkpoints"], 0)
                self.assertGreater(entry["tiers"]["complete"]["bytes"], entry["tiers"]["analysis"]["bytes"])
            else:
                self.assertEqual(set(entry["tiers"]), {"complete"})

    def test_e12_view_registers_the_fourteen_studies(self):
        from cm_campaigns.contract import EXTERNAL
        view = self.tree["experiments/baselines/INDEX.json"]
        self.assertEqual({(s["dataset"], s["method"]) for s in view["studies"]},
                         {(d, m) for d, methods in EXTERNAL.items() for m in methods})
        for study in view["studies"]:
            registered = study["registered_study"]
            self.assertEqual((registered["hpo_trials"], registered["optimization_seeds"]), (48, 12))
            self.assertEqual(registered["official_splits"],
                             list(range(10)) if study["dataset"] == "Roman-empire" else [0])
            self.assertTrue(destination.is_pending(study["source_commit"]))

    def test_figure_work_covers_exactly_the_figures_without_a_contract(self):
        ledger = self.tree["paper/source_data/INDEX.json"]["displays"]
        to_register = {r["id"] for r in ledger if r["contract"]["now"] == "to register"}
        self.assertEqual(to_register, set(destination.FIGURE_WORK))
        self.assertEqual(len(ledger), 90)
        summary = self.tree["paper/source_data/INDEX.json"]["reconstruction"]
        self.assertEqual(summary["values_exact"], sum(r["registered_targets"] for r in ledger))


class Statement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc, cls.tree = fd.build_all(ROOT)
        cls.now = sc.evaluate(ROOT)

    def test_twelve_of_twelve_on_the_destination_tree(self):
        result = destination.projected_statement(ROOT, self.tree, self.now)
        self.assertEqual((result["fulfilled"], result["total"]), (12, 12))
        self.assertEqual([r["id"] for r in result["items"]], [r["id"] for r in self.now["items"]])

    def test_each_item_depends_on_its_projected_files(self):
        tree = copy.deepcopy(self.tree)
        for entry in tree["release/paper_artifacts.json"]["artifacts"]:
            if entry["id"] == "run:COCO-SP":
                entry["status"] = pa.AWAITING
        result = {r["id"]: r["destination"] for r in destination.projected_statement(ROOT, tree, self.now)["items"]}
        for item in ("checkpoints", "predictions", "trajectories", "statistical_outputs", "one_command",
                     "split_seed_manifests", "source_data"):
            self.assertEqual(result[item], sc.MISSING, item)
        self.assertEqual(result["generators"], sc.FULFILLED)
        tree = copy.deepcopy(self.tree)
        tree["paper/source_data/INDEX.json"]["reconstruction"]["values_differ"] = 1
        result = {r["id"]: r["destination"] for r in destination.projected_statement(ROOT, tree, self.now)["items"]}
        self.assertEqual(result["one_command"], sc.MISSING)
        tree = copy.deepcopy(self.tree)
        tree["experiments/baselines/INDEX.json"]["studies"].pop()
        result = {r["id"]: r["destination"] for r in destination.projected_statement(ROOT, tree, self.now)["items"]}
        self.assertEqual((result["experiment_code"], result["configurations"]), (sc.MISSING, sc.MISSING))

    def test_the_real_statement_is_unchanged_by_the_projection(self):
        self.assertEqual(self.now["counts"][sc.FULFILLED], 2)


class Guards(unittest.TestCase):
    def test_real_loaders_refuse_a_projected_file(self):
        from scripts import statement_evidence
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "paper_artifacts.json"
            shutil.copyfile(ROOT / "destination/release/paper_artifacts.json", target)
            with self.assertRaisesRegex(pa.ArtifactError, "projection"):
                pa.load_manifest(target)
            manifest = Path(tmp) / "arxiv.json"
            shutil.copyfile(ROOT / "destination/manifests/published/arxiv.json", manifest)
            with self.assertRaisesRegex(ValueError, "projection"):
                statement_evidence.read_json(manifest)

    def test_only_the_destination_tools_read_the_tree(self):
        readers = []
        for path in sorted((ROOT / "scripts").glob("*.py")) + sorted((ROOT / "paper" / "build").rglob("*.py")):
            if path.name in ("destination.py", "final_destination.py"):
                continue
            if '"destination/' in path.read_text(encoding="utf-8") or "'destination/" in path.read_text(encoding="utf-8"):
                readers.append(path.relative_to(ROOT).as_posix())
        self.assertEqual(readers, [])


class Convergence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc, cls.tree = fd.build_all(ROOT)

    def realized(self, files: dict) -> tuple[Path, dict]:
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        for relative, value in files.items():
            path = tmp / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value), encoding="utf-8")
        report = destination.converge(tmp, self.tree)
        return tmp, {row["file"]: row for row in report["files"]} | {"summary": report["summary"]}

    def test_nothing_realized_is_pending(self):
        report = destination.converge(ROOT, self.tree)
        self.assertEqual(report["summary"]["pending"], 17)
        self.assertEqual(report["summary"]["differs"], 0)

    def test_a_realized_manifest_matching_the_design_converges(self):
        relative = "manifests/published/peptides_func.json"
        realized = fill(self.tree[relative])
        realized["environment"]["gpu"] = "NVIDIA GH200 480GB"            # registered: the site's device name
        _, rows = self.realized({relative: realized})
        self.assertEqual(rows[relative]["status"], "converged", rows[relative]["findings"])
        self.assertGreater(rows[relative]["counts"].get("exact:agrees", 0), 30)

    def test_an_unregistered_configuration_or_role_differs(self):
        relative = "manifests/published/pascalvoc_sp.json"
        realized = fill(self.tree[relative])
        realized["identities"]["config_sha256"] = "f" * 64
        realized["splits"]["graph_role_hashes"]["test"] = "e" * 64
        _, rows = self.realized({relative: realized})
        self.assertEqual(rows[relative]["status"], "differs")
        self.assertEqual({f["path"] for f in rows[relative]["findings"]},
                         {"identities/config_sha256", "splits/graph_role_hashes/test"})
        self.assertEqual(rows["summary"]["differs"], 1)

    def test_published_run_entries_compare_by_identity_and_tolerance(self):
        projected = self.tree["release/paper_artifacts.json"]
        realized = fill(projected)
        for entry in realized["artifacts"]:
            if entry["kind"] != "run":
                entry["status"] = pa.AWAITING
                continue
            for name, spec in entry["tiers"].items():
                count = spec["parts"]["count"]
                spec["parts"] = [{"asset": f"a{i}", "bytes": 1, "sha256": "0" * 64, "url": "https://x/a"} for i in range(count)]
            if entry["id"] == "run:ogbn-arxiv":
                entry["bytes"] = entry["bytes"] * 3                       # outside the factor of two
        realized["status"] = pa.AWAITING                                  # the record sets are still awaiting
        source = fill(self.tree["paper/source_data/INDEX.json"])
        source["reconstruction"]["artifacts_awaiting_publication"] = ["records:baselines"]
        _, rows = self.realized({"release/paper_artifacts.json": realized, "paper/source_data/INDEX.json": source})
        self.assertEqual(rows["paper/source_data/INDEX.json"]["status"], "pending")
        row = rows["release/paper_artifacts.json"]
        self.assertEqual(row["status"], "converged with notes")
        self.assertEqual([f["path"] for f in row["findings"]], ["artifacts/run:ogbn-arxiv/bytes"])
        self.assertGreater(row["counts"].get("exact:pending", 0), 0)       # unpublished record sets stay pending
        realized["artifacts"][0]["identity"]["source_sha256"] = "1" * 64
        _, rows = self.realized({"release/paper_artifacts.json": realized})
        self.assertEqual(rows["release/paper_artifacts.json"]["status"], "differs")

    def test_command_line_exit_status(self):
        self.assertEqual(fd.main(["--converge"]), 0)


class Forecast(unittest.TestCase):
    def test_table_s20_q_is_a_measured_quantity_within_the_registered_bound(self):
        inputs = fd.Inputs(ROOT)
        rows = fd.forecast(inputs, fd.preregistered_roles(inputs, fd.runtime_role_functions(ROOT)))
        q_rows = [r for r in rows if r["quantity"].startswith("|Q|")]
        self.assertEqual(len(q_rows), 7)
        for row in q_rows:
            self.assertEqual(row["kind"], "bound")
            self.assertEqual(row["forecast"], fd.registry_size(inputs.configs[row["dataset"]])["candidates"])
            self.assertLessEqual(row["printed"], row["forecast"])
            self.assertEqual(row["status"], "within bound")
        self.assertEqual([r for r in rows if r["status"] in ("differs", "exceeds bound")], [])


if __name__ == "__main__":
    unittest.main()
