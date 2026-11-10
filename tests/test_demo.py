"""The CPU demonstration channel: its configurations, planner device, executor and separation from the
paper's evidence (NumPy only; no PyTorch)."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import paper_artifacts as pa  # noqa: E402
from scripts import paper_outputs as po  # noqa: E402
from scripts import reconstruct_paper  # noqa: E402
from scripts import statement_check  # noqa: E402


def module(name: str):
    spec = importlib.util.spec_from_file_location(f"demo_{name}", ROOT / "demo" / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


class Configurations(unittest.TestCase):
    def test_demo_configurations_are_the_pilot_settings_on_an_fp32_plant(self):
        configure = module("configure")
        for dataset, stem in configure.STEMS.items():
            pilot = json.loads((ROOT / "configs" / f"{stem}_pilot.json").read_text(encoding="utf-8"))
            demo = configure.demo_config(dataset, "demo")
            self.assertEqual(demo["model"]["precision"], "fp32")
            changed = copy.deepcopy(demo)
            changed["model"]["precision"] = pilot["model"]["precision"]
            self.assertEqual(changed, pilot, dataset)
            self.assertEqual((demo["profile"], demo["scientific_run"]), ("pilot", False))

    def test_a_reduced_graph_runs_as_a_fixture_routed_to_its_benchmark(self):
        configure = module("configure")
        demo = configure.demo_config("ogbn-products", "demo", graph_npz="/abs/products_demo.npz")
        self.assertEqual((demo["dataset"], demo["benchmark_dataset"], demo["profile"]),
                         ("synthetic", "ogbn-products", "smoke"))
        self.assertEqual(demo["data"]["graph_npz"], "/abs/products_demo.npz")
        self.assertFalse(demo["data"]["download"])
        roman = configure.demo_config("Roman-empire", "demo", split_index=7)
        self.assertEqual((roman["dataset"], roman["data"]["split_index"]), ("Roman-empire", 7))
        with self.assertRaises(ValueError):
            configure.demo_config("ogbn-arxiv", "demo", split_index=1)

    def test_every_reduced_graph_rule_is_declared(self):
        data = module("data")
        self.assertEqual(set(data.RULES), {"ogbn-arxiv", "ogbn-proteins", "ogbn-products", "PascalVOC-SP", "COCO-SP",
                                           "Peptides-func"})
        for rule in data.RULES.values():
            self.assertIn(rule["kind"], ("metis_parts", "metis_parts_per_role", "first_graphs"))


class PlannerDevice(unittest.TestCase):
    def test_cpu_device_rewrites_every_runner_device(self):
        nodes = [{"id": "a", "tasks": [{"commands": [["{python}", "x.py", "--device", "cuda", "--threads", "{threads}"],
                                                     ["bash", "capture.sh", "cuda"]]}],
                  "plan_step": ["{python}", "y.py", "plan", "--device", "cuda"]}]
        rewritten = po.with_device(nodes, "cpu")
        self.assertEqual(rewritten[0]["tasks"][0]["commands"][0][3], "cpu")
        self.assertEqual(rewritten[0]["tasks"][0]["commands"][1], ["bash", "capture.sh", "cuda"])
        self.assertEqual(rewritten[0]["plan_step"][-1], "cpu")
        self.assertNotIn('"--device", "cuda"', json.dumps(rewritten))

    def test_cpu_device_requires_a_fixture_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = po.parser().parse_args(["plan", "--runs", str(ROOT / "examples" / "runs"), "--work",
                                           str(Path(temporary) / "work"), "--device", "cpu"])
            with self.assertRaises(po.PlanError):
                po.command_plan(args, executor=lambda argv: (0, ""))

    def test_held_horizon_sweep_beam_requires_a_fixture_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = po.parser().parse_args(["plan", "--runs", str(ROOT / "examples" / "runs"), "--work",
                                           str(Path(temporary) / "work"), "--horizon-sweep-beam", "16"])
            with self.assertRaises(po.PlanError) as caught:
                po.command_plan(args, executor=lambda argv: (0, ""))
            self.assertIn("--horizon-sweep-beam", str(caught.exception))

    def test_held_horizon_sweep_beam_reaches_the_sweep_planner(self):
        class Recorder:
            opts = {"horizon_sweep": {"profile": "full", "ensemble": 1, "beam": 16}}
            argv = None

            def experiment_dir(self, name):
                return Path("/w") / name

            def parents(self, datasets):
                return ["--parent", "/runs/pascalvoc_sp"]

            def runner_plan(self, argv, out):
                Recorder.argv = argv
                return {"variants": []}

        po.build_sweep(Recorder(), False, "horizon_beam_sweep", ("PascalVOC-SP",), "S29")
        argv = [str(a) for a in Recorder.argv]
        self.assertEqual(argv[argv.index("--beam") + 1], "16")
        self.assertEqual(argv[argv.index("--profile") + 1], "full")


class Executor(unittest.TestCase):
    def test_levels_follow_the_dependencies(self):
        execute = module("execute")
        plan = {"nodes": [{"id": "a", "tasks": [{}], "dependencies": []},
                          {"id": "b", "tasks": [{}], "dependencies": [{"node": "a", "type": "afterok"}]},
                          {"id": "c", "tasks": [{}], "dependencies": [{"node": "b", "type": "afterany"},
                                                                      {"node": "a", "type": "afterok"}]},
                          {"id": "d", "tasks": [], "dependencies": [{"node": "outside", "type": "afterok"}]}]}
        self.assertEqual(execute.levels(plan), {"a": 0, "b": 1, "c": 2, "d": 0})


    def test_every_share_agrees_and_leftovers_spread_under_a_new_salt(self):
        execute = module("execute")
        tasks = [("n%d" % (i // 10), i % 10) for i in range(64)]
        plain = execute.shares(tasks, 16, None)
        self.assertEqual([plain[t] for t in tasks[:17]], list(range(1, 17)) + [1])
        first = execute.shares(tasks, 16, "run-1")
        self.assertEqual(first, execute.shares(list(reversed(tasks)), 16, "run-1"))   # independent of any state
        self.assertEqual(sorted(list(first.values()).count(k) for k in range(1, 17)), [4] * 16)
        left = [t for t in tasks if first[t] == 16]                                    # one share ran out of time
        second = execute.shares(tasks, 16, "run-2")
        self.assertGreater(len({second[t] for t in left}), 1)

    def test_a_skipped_node_is_left_out_of_the_matrix_and_the_shares(self):
        import contextlib
        import io
        execute = module("execute")
        plan = {"schema": po.SCHEMA, "nodes": [
            {"id": "grid", "tasks": [{}, {}, {}], "dependencies": [], "outputs": []},
            {"id": "other", "tasks": [{}, {}], "dependencies": [], "outputs": []},
            {"id": "later", "tasks": [{}], "dependencies": [{"node": "other", "type": "afterok"}], "outputs": []}]}
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            (work / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                execute.main(["--work", str(work), "--matrix", "16", "--skip-node", "grid"])
            self.assertEqual(json.loads(printed.getvalue()), {"l0": [1, 2], "l1": [1]})
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                execute.main(["--work", str(work), "--matrix", "16"])
            self.assertEqual(json.loads(printed.getvalue()), {"l0": [1, 2, 3, 4, 5], "l1": [1]})


class Channels(unittest.TestCase):
    def published(self, relationship: str, scientific: bool) -> dict:
        part = {"asset": "x.complete-0.tar.gz", "bytes": 10, "sha256": "0" * 64,
                "url": "https://github.com/VKHS/CR-9/releases/download/demo-artifacts/x.complete-0.tar.gz"}
        return {"id": "run:ogbn-arxiv", "kind": "run", "status": pa.PUBLISHED, "relationship": relationship,
                "identity": {"x": 1}, "inventory_sha256": "1" * 64, "files": 1, "bytes": 10,
                "contents": pa.content_counts([]), "scientific_only_verification": scientific,
                "tiers": {"complete": {"archive": "x.complete-0.tar.gz", "bytes": 10, "sha256": "0" * 64,
                                       "parts": [part]}}}

    def test_the_paper_manifest_refuses_demonstration_artifacts(self):
        manifest = dict(pa.blank_manifest(), artifacts=[self.published(pa.DEMO_RELATIONSHIP, False)], status=pa.PUBLISHED)
        with self.assertRaises(pa.ArtifactError):
            pa.validate_manifest(manifest)
        with self.assertRaises(pa.ArtifactError):
            pa.validate_manifest(dict(pa.blank_manifest(), release_tag=pa.DEMO_TAG))

    def test_the_demonstration_manifest_holds_only_demonstration_artifacts(self):
        demo = pa.blank_manifest(demonstration=True)
        self.assertEqual((demo["channel"], demo["release_tag"]), (pa.DEMO_CHANNEL, pa.DEMO_TAG))
        pa.validate_manifest(dict(demo, artifacts=[self.published(pa.DEMO_RELATIONSHIP, False)], status=pa.PUBLISHED))
        with self.assertRaises(pa.ArtifactError):
            pa.validate_manifest(dict(demo, artifacts=[self.published("exact_result_producing_run", True)],
                                      status=pa.PUBLISHED))
        with self.assertRaises(pa.ArtifactError):
            pa.validate_manifest(dict(demo, scope=""))

    def test_a_missing_demonstration_manifest_is_a_blank_demonstration(self):
        self.assertEqual(pa.load_manifest(pa.DEMO_MANIFEST)["channel"] if not pa.DEMO_MANIFEST.is_file()
                         else pa.DEMO_CHANNEL, pa.DEMO_CHANNEL)

    def test_reconstruction_refuses_to_mix_the_manifests(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            self.assertEqual(reconstruct_paper.main(["--demo", "--offline", "--manifest", str(pa.MANIFEST),
                                                     "--artifacts", str(out / "a"), "--output-dir", str(out / "p")]), 2)
            demo = out / "demo_artifacts.json"
            demo.write_text(json.dumps(pa.blank_manifest(demonstration=True)), encoding="utf-8")
            self.assertEqual(reconstruct_paper.main(["--offline", "--manifest", str(demo), "--artifacts",
                                                     str(out / "a"), "--output-dir", str(out / "p")]), 2)

    def test_the_statement_reports_the_demonstration_separately(self):
        evaluation = statement_check.evaluate(ROOT)
        before = evaluation["counts"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "release").mkdir()
            self.assertIsNone(statement_check.demonstration(evaluation, root))
            manifest = dict(pa.blank_manifest(demonstration=True), artifacts=[
                self.published(pa.DEMO_RELATIONSHIP, False)])
            (root / "release" / "demo_artifacts.json").write_text(json.dumps(manifest), encoding="utf-8")
            demo = statement_check.demonstration(evaluation, root)
        rows = {row["id"]: row for row in demo["rows"]}
        self.assertTrue(rows["checkpoints"]["demonstrated"])
        self.assertFalse(rows["one_command"]["demonstrated"])
        self.assertEqual(statement_check.evaluate(ROOT)["counts"], before)
        self.assertIn("CPU demonstration (separate from the statement)",
                      statement_check.render(evaluation, None, demo))



class Publication(unittest.TestCase):
    def test_a_second_graph_fills_only_what_the_main_map_awaits(self):
        merge_maps = module("merge_maps")
        main = {"_awaiting": {"records:horizon_beam_sweep": "beam width 4 is not the registered 16",
                              "records:baselines": "E12"}, "_relationships": {"run:ogbn-arxiv": "post_result_revision"},
                "run:ogbn-arxiv": "/runs/arxiv", "records:horizon_beam_sweep": "", "records:baselines": ""}
        extra = {"_awaiting": {"records:baselines": "outside --only"}, "run:ogbn-arxiv": "/other/arxiv",
                 "records:horizon_beam_sweep": "/w2/records/horizon_beam_sweep", "records:baselines": ""}
        self.assertEqual(merge_maps.merge(main, extra), ["records:horizon_beam_sweep"])
        self.assertEqual(main["run:ogbn-arxiv"], "/runs/arxiv")
        self.assertEqual(main["records:horizon_beam_sweep"], "/w2/records/horizon_beam_sweep")
        self.assertEqual(main["_awaiting"], {"records:baselines": "E12"})
        self.assertEqual(main["_relationships"]["records:horizon_beam_sweep"], "post_result_revision")

    def test_the_report_lists_what_no_graph_produced(self):
        report = module("report")
        with tempfile.TemporaryDirectory() as temporary:
            first, second = Path(temporary) / "w", Path(temporary) / "w2"
            artifacts = {"records:a": {"status": "planned"}, "records:b": {"status": "awaiting", "reason": "planner"},
                         "records:c": {"status": "planned"}, "records:d": {"status": "planned"}}
            for work, mapping in ((first, {"records:a": "/w/a", "records:b": "", "records:c": "", "records:d": "/w/d",
                                           "_awaiting": {"records:b": "no run", "records:c": "beam"}}),
                                  (second, {"records:c": "/w2/c", "_awaiting": {}})):
                work.mkdir()
                (work / "plan.json").write_text(json.dumps({"artifacts": artifacts, "nodes": []}), encoding="utf-8")
                (work / "paper-artifacts.map.json").write_text(json.dumps(mapping), encoding="utf-8")
            rows = report.artifact_rows([first, second], {"records:d": "Artifact contains a symlink"})
        self.assertEqual(rows, ["- `records:b`: no run",
                                "- `records:d`: produced, refused by the packer: Artifact contains a symlink"])


if __name__ == "__main__":
    unittest.main()
