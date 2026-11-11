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
            # a node waiting on a skipped node that has not finished cannot run in this call
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                execute.main(["--work", str(work), "--matrix", "16", "--skip-node", "other"])
            self.assertEqual(json.loads(printed.getvalue()), {"l0": [1, 2, 3]})      # "later" holds a share, runs nothing
            for k in range(2):
                marker = work / "state" / "other" / f"task-{k}.json"
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text(json.dumps({"status": "COMPLETE"}), encoding="utf-8")
            self.assertEqual(execute.left_out(work, {n["id"]: n for n in plan["nodes"]}, ["other"]), {"other"})


class BudgetStudiesWithoutScheduler(unittest.TestCase):
    """E3-E6 studies of CPU plans run through scripts/run_budget_study.py instead of the Slurm batch script."""

    def test_cpu_plans_use_the_portable_runner(self):
        class Planner:
            opts = {"device": "cpu", "allow_fixture": True}
        command = po.budget_study_command(Planner())
        self.assertIn("scripts/run_budget_study.py", [str(c) for c in command])
        self.assertIn("--allow-fixture", command)
        Planner.opts = {}
        self.assertEqual(po.budget_study_command(Planner()), ["bash", "hpc/run_budget_arms_arrhenius.sh"])

    def test_the_runner_keeps_the_job_evidence_and_records_no_scheduler(self):
        import os
        import subprocess
        with tempfile.TemporaryDirectory() as temporary:
            source, base = Path(temporary) / "source", Path(temporary) / "parent"
            (source / "scripts").mkdir(parents=True)
            (source / "environment" / "arrhenius").mkdir(parents=True)
            base.mkdir()
            (source / "environment" / "arrhenius" / "capture_environment.sh").write_text(
                'mkdir -p "$2" && echo captured > "$2/observed_environment.json"\n', encoding="utf-8")
            (source / "scripts" / "run_budget_arms.py").write_text(
                "import json, sys\nfrom pathlib import Path\nargs = sys.argv[1:]\n"
                "if '--verify' in args:\n    Path(args[args.index('--report') + 1]).write_text('{}')\n    sys.exit(0)\n"
                "out = Path(args[args.index('--output') + 1]); out.mkdir()\n"
                "(out / 'ARTIFACT_MANIFEST.json').write_text(json.dumps(args))\n", encoding="utf-8")
            study = Path(temporary) / "studies" / "seed-3"
            study.parent.mkdir()
            env = dict(os.environ, CM_SOURCE_ROOT=str(source), CM_BASE_RUN=str(base), CM_ARM_OUTPUT=str(study),
                       CM_TRUST_CHECKPOINTS="1", CM_ARM_SEED="3")
            result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / "run_budget_study.py"), "--device",
                                     "cpu", "--threads", "2", "--allow-fixture"], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            evidence = po.budget_evidence(study)
            self.assertTrue(evidence.name.startswith("seed-3.job-local-"))
            self.assertTrue(po.unscheduled(evidence))
            for name in ("job_id.txt", "command.txt", "stdout.log", "stderr.log", "pip-freeze.txt",
                         "numerical_replay.json", "environment/observed_environment.json"):
                self.assertTrue((evidence / name).is_file(), name)
            argv = json.loads((study / "ARTIFACT_MANIFEST.json").read_text())
            self.assertEqual(argv[argv.index("--seed") + 1], "3")
            self.assertIn("--allow-fixture", argv)
            self.assertEqual(argv[argv.index("--device") + 1], "cpu")
            po.finalize_budget_accounting({"studies": [{"study": str(study)}]})     # nothing to finalize
            self.assertFalse((evidence / "sacct.tsv").exists())
            (evidence / "stdout.log").write_text("arm results\n")          # the fake runner printed nothing
            manifest = Path(temporary) / "campaign.json"
            po.write_budget_manifest({"manifest": str(manifest), "seeds": [3],
                                      "studies": [{"dataset": "ogbn-arxiv", "seed": 3, "split": 0, "parent": str(base),
                                                   "study": str(study)}]},
                                     {"git_commit": "0" * 40, "dirty": False})
            row, = json.loads(manifest.read_text())["studies"]
            self.assertEqual(row["accounting"], [])
            self.assertEqual(row["job_log"], [str(evidence / "stdout.log")])        # the empty stderr is not attached
            again = subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / "run_budget_study.py")], env=env,
                                   capture_output=True, text=True)
            self.assertEqual(again.returncode, 2)                                       # never overwrites a study


class BudgetProvenance(unittest.TestCase):
    def test_a_campaign_without_accounting_takes_its_period_from_the_studies(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign = Path(temporary)
            for i, stamp in enumerate(("2026-11-10T09:55:00+00:00", "2026-11-10T10:40:00+00:00")):
                study = campaign / "studies" / "ogbn-arxiv" / f"seed-{i}" / "split-0"
                study.mkdir(parents=True)
                (study / "experiment.json").write_text(json.dumps({"created_utc": stamp}), encoding="utf-8")
                (study / "environment.json").write_text(json.dumps({"device": "cpu", "python": "3.12"}), encoding="utf-8")
            hardware, period = po.derive_provenance({"kind": "budget", "campaign": str(campaign)})
        self.assertEqual(period, "2026-11-10T09:55:00+00:00/2026-11-10T10:40:00+00:00")
        self.assertTrue(hardware)


class TimingWithoutScheduler(unittest.TestCase):
    """E17 on a CPU plan: no scheduler accounting to capture, and a partial measurement to declare."""

    class Planner:
        def __init__(self, device):
            self.opts = {"device": device} if device else {}
            self.work, self.records = Path("/w"), Path("/w/records")
            self.exports, self.nodes = [], []

        def experiment_dir(self, *parts):
            return self.work.joinpath("experiments", *parts)

        def parents(self, datasets):
            return ["--parent", "/runs/x"]

        def fixture(self, key):
            return []

        def runner_plan(self, argv, out):
            return {"jobs": [{}, {}]}

        def record_set(self, name):
            return str(self.records / name)

        def add_node(self, node_id, **kwargs):
            node = {"id": node_id, "tasks": kwargs.get("tasks") or [], **kwargs}
            self.nodes.append(node)
            return node

        def add_export(self, export_id, **kwargs):
            self.exports.append({"id": export_id, **kwargs})

    def argv_of(self, planner):
        return [[str(a) for a in step["argv"]] for export in planner.exports for step in export["steps"]]

    def test_cpu_plans_export_partial_timings_without_accounting(self):
        cpu, gpu = self.Planner("cpu"), self.Planner(None)
        for planner in (cpu, gpu):
            scaling = po.build_runtime_scaling(planner, True)
            po.build_timing_anchors(planner, True, scaling)
        cpu_steps, gpu_steps = self.argv_of(cpu), self.argv_of(gpu)
        self.assertFalse(any("hpc/capture_arrhenius_accounting.sh" in step for step in cpu_steps))
        self.assertTrue(all("--allow-partial" in step and "--accounting" not in step for step in cpu_steps))
        self.assertTrue(any("hpc/capture_arrhenius_accounting.sh" in step for step in gpu_steps))
        self.assertFalse(any("--allow-partial" in step for step in gpu_steps))


class NarrowRestores(unittest.TestCase):
    def test_a_measurement_share_restores_only_the_lifecycle_runs_its_arms_read(self):
        task_inputs = module("task_inputs")
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "work" / "experiments" / "module_ablations"
            out.mkdir(parents=True)
            runs = {m: [str(out / "runs" / m / v) for v in ("v1", "v2")] for m in ("a", "b")}
            jobs = [{"job": i, "kind": "lifecycle", "run": run} for i, run in enumerate(runs["a"] + runs["b"])]
            jobs += [{"job": 4 + i, "kind": "measure", "arms": [{"source": "parent", "run": "/runs/" + m}] +
                      [{"source": "lifecycle", "run": run} for run in runs[m]]} for i, m in enumerate(("a", "b"))]
            (out / "plan.json").write_text(json.dumps({"jobs": jobs}), encoding="utf-8")
            measure = [{"commands": [], "job_in": {"plan": str(out / "plan.json"), "index": 4 + i, "kind": "measure"}}
                       for i in range(2)]
            plan = {"nodes": [{"id": "lifecycles", "tasks": [{}] * 4, "dependencies": []},
                              {"id": "measure", "tasks": measure,
                               "dependencies": [{"node": "lifecycles", "type": "afterok"}]}]}
            first = task_inputs.unread_runs(plan, level=1, shard=1, shards=2, salt=None)
            second = task_inputs.unread_runs(plan, level=1, shard=2, shards=2, salt=None)
            self.assertEqual(first, sorted(r.lstrip("/") for r in runs["b"]))
            self.assertEqual(second, sorted(r.lstrip("/") for r in runs["a"]))
            self.assertEqual(task_inputs.unread_runs(plan, level=0, shard=1, shards=2, salt=None), [])


class Restore(unittest.TestCase):
    def test_a_study_is_restored_with_its_job_evidence_and_exclusions_hold(self):
        import shutil
        import subprocess
        import zipfile
        if not (shutil.which("unzip") and shutil.which("tar")):
            self.skipTest("unzip and GNU tar are required")
        restore = module("restore")
        sys.modules.setdefault("demo_levels", module("execute"))      # what restore.main registers
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "w"
            files = {"budget/studies/a/seed-0/ARTIFACT_MANIFEST.json": "1",
                     "budget/studies/a/seed-0.job-local-1/process_exit_code.txt": "0",
                     "budget/studies/a/seed-1/ARTIFACT_MANIFEST.json": "2",
                     "budget/studies/a/seed-0/train/plant.pt": "3", "state/n/task-0.json": "{}"}
            for name, text in files.items():
                (work / name).parent.mkdir(parents=True, exist_ok=True)
                (work / name).write_text(text)
            archive = root / "delta.tar.gz"
            subprocess.run(["tar", "-czf", str(archive), "-C", "/", *[str(work / n).lstrip("/") for n in files]],
                           check=True)
            shutil.rmtree(work)
            bundle = root / "artifact.zip"
            with zipfile.ZipFile(bundle, "w") as handle:
                handle.write(archive, "delta.tar.gz")
            plan = {"nodes": [{"id": "studies", "tasks": [{}], "dependencies": [],
                               "outputs": [str(work / "budget/studies/a/seed-0")]}]}
            needed, prefixes = restore.selection(plan, work, level=None, nodes=["studies"], export=None)
            restore.PATTERNS[:] = ["*.pt"]
            try:
                restore.extract(bundle, prefixes)
            finally:
                restore.PATTERNS[:] = []
            got = sorted(str(p.relative_to(work)) for p in work.rglob("*") if p.is_file())
        self.assertEqual(got, ["budget/studies/a/seed-0.job-local-1/process_exit_code.txt",
                               "budget/studies/a/seed-0/ARTIFACT_MANIFEST.json", "state/n/task-0.json"])

    def test_an_include_pattern_extracts_only_the_matching_members(self):
        import shutil
        import subprocess
        import zipfile
        if not (shutil.which("unzip") and shutil.which("tar")):
            self.skipTest("unzip and GNU tar are required")
        restore = module("restore")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "w"
            files = {"exp/plan.json": "{}", "exp/runs/a/select/decision.json": "{}", "exp/runs/a/train/p.pt": "x",
                     "exp/runs/a/prepared/splits.npz": "s", "exp/runs/a/prepared/graph.npz": "g",
                     "state/n/task-0.json": "{}"}
            for name, text in files.items():
                (work / name).parent.mkdir(parents=True, exist_ok=True)
                (work / name).write_text(text)
            archive = root / "delta.tar.gz"
            subprocess.run(["tar", "-czf", str(archive), "-C", "/", *[str(work / n).lstrip("/") for n in files]],
                           check=True)
            shutil.rmtree(work)
            bundle = root / "artifact.zip"
            with zipfile.ZipFile(bundle, "w") as handle:
                handle.write(archive, "delta.tar.gz")
            output = str(work / "exp").lstrip("/")
            restore.WILDCARD_SLASH[0] = True
            try:
                restore.extract(bundle, [str(work / "state").lstrip("/") + "/", f"{output}/*.json",
                                         f"{output}/*/prepared/splits.npz"])
            finally:
                restore.WILDCARD_SLASH[0] = False
            got = sorted(str(p.relative_to(work)) for p in work.rglob("*") if p.is_file())
        self.assertEqual(got, ["exp/plan.json", "exp/runs/a/prepared/splits.npz", "exp/runs/a/select/decision.json",
                               "state/n/task-0.json"])


class PartialExports(unittest.TestCase):
    def test_a_fixture_plan_lets_partial_exporters_write_the_measured_part(self):
        exports = [{"steps": [{"argv": po.py("scripts/run_confirmation_analysis.py", "mechanism", "export", "--output", "o")}]},
                   {"steps": [{"argv": po.py("scripts/run_ablation.py", "modules", "export", "--output", "o")}]},
                   {"steps": [{"argv": po.py("scripts/run_ablation.py", "factorial-fit", "export", "--output", "o")}]},
                   {"steps": [{"argv": po.py("scripts/export_sweep_records.py", "--results", "r")}]},
                   {"steps": [{"argv": po.py("scripts/run_decision_audit.py", "export", "--allow-partial")},
                              {"internal": "accepted"}]}]
        po.declare_partial_exports(exports)
        flags = [[s["argv"].count("--allow-partial") for s in e["steps"] if "argv" in s] for e in exports]
        self.assertEqual(flags, [[1], [0], [1], [0], [1]])


class PartialCoverage(unittest.TestCase):
    def test_only_a_fixture_plan_admits_a_set_that_declares_partial_coverage(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "runtime_scaling"
            path.mkdir()
            (path / "provenance.json").write_text(json.dumps({"coverage_complete": False}), encoding="utf-8")
            with mock.patch("scripts.records.verify_set", return_value={"synthetic": True}):
                self.assertEqual(po.verify_record_set(path, allow_fixture=True), {"synthetic": True})
                with self.assertRaises(po.PlanError):
                    po.verify_record_set(path, allow_fixture=False)


class Channels(unittest.TestCase):
    def published(self, relationship: str, scientific: bool) -> dict:
        part = {"asset": "x.complete-0.tar.gz", "bytes": 10, "sha256": "0" * 64,
                "url": "https://github.com/VKHS/CROSS-MORSE/releases/download/demo-artifacts/x.complete-0.tar.gz"}
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
            # the paper's manifest under another name (the default path itself selects the demonstration's)
            paper = out / "paper_artifacts.json"
            paper.write_bytes(pa.MANIFEST.read_bytes())
            self.assertEqual(reconstruct_paper.main(["--demo", "--offline", "--manifest", str(paper),
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
            # the derived views: the one command runs and builds the displays its artifacts support
            views = root / "demo" / "published" / "paper" / "source_data"
            (views / "reports").mkdir(parents=True)
            (views / "INDEX.json").write_text(json.dumps({"reconstruction": {"displays": 90}}), encoding="utf-8")
            (views / "reports" / "reconstruction.json").write_text(
                json.dumps({"display_generators": {"built": 56, "registered": 90}}), encoding="utf-8")
            derived = {row["id"]: row for row in statement_check.demonstration(evaluation, root)["rows"]}
        rows = {row["id"]: row for row in demo["rows"]}
        self.assertTrue(rows["checkpoints"]["demonstrated"])
        self.assertFalse(rows["one_command"]["demonstrated"])
        self.assertEqual((derived["one_command"]["demonstrated"], derived["one_command"]["evidence"]),
                         ("partly", "56 of 90 displays built"))
        self.assertIs(derived["generators"]["demonstrated"], True)
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
            noted = report.artifact_rows([first, second], {}, {"records:b": "the cause", "records:c": "unused"})
        self.assertEqual(rows, ["- `records:b`: no run",
                                "- `records:d`: produced, refused by the packer: Artifact contains a symlink"])
        # a note states the cause of an absence; an artifact another graph produced stays out of the list
        self.assertEqual(noted, ["- `records:b`: the cause"])

    def test_the_demonstration_views_bind_its_budget_campaign_input(self):
        from cm_campaigns.common import directory_inventory
        with tempfile.TemporaryDirectory() as temporary:
            fetched = Path(temporary).resolve()
            campaign = fetched / "campaigns" / "budget"
            campaign.mkdir(parents=True)
            (campaign / "study.json").write_text("{}")
            verified = {"artifacts": [{"id": "campaign:budget", "kind": "campaign", "path": str(campaign)}]}
            result = {"inputs": [], "supplementary_build": {"demonstration": True, "inputs": [
                {"input_kind": "demonstration_budget_campaign", "source": str(campaign),
                 "files": directory_inventory(campaign)}]}}
            pa._verify_reconstruction_inputs(result, fetched, verified, demonstration=True)
            # the paper's views accept only the strictly replayed campaign
            with self.assertRaisesRegex(pa.ArtifactError, "differ from the fetched campaigns"):
                pa._verify_reconstruction_inputs(result, fetched, verified)

    def test_the_main_graph_table_names_the_further_graph_that_completed_a_node(self):
        report = module("report")
        with tempfile.TemporaryDirectory() as temporary:
            main, further = Path(temporary) / "w", Path(temporary) / "w2"
            node = {"id": "budget.studies.arxiv", "tasks": [{}], "dependencies": []}
            for work, status, options in ((main, "FAILED", {}), (further, "COMPLETE", {"only": ["campaign:budget"]})):
                (work / "state" / node["id"]).mkdir(parents=True)
                (work / "state" / node["id"] / "task-0.json").write_text(json.dumps({"status": status}))
                (work / "plan.json").write_text(json.dumps({"nodes": [node], "options": options, "artifacts": {},
                                                            "declared_choices": {"a": 1, "b": options.get("only")}}))
            elsewhere = report.complete_elsewhere([main, further])
            rows = report.graph_section(main, ["1"], True, main_choices={}, elsewhere=elsewhere)
            text = report.graph_section(further, ["2"], False, main_choices={"a": 1, "b": None})[0]
        self.assertEqual(rows[-1], "| budget.studies.arxiv | 0 | 0/1 | incomplete (FAILED); complete in the graph of "
                                   "`campaign:budget` (below) |")
        self.assertIn('differ from the main graph\'s in: {"b": ["campaign:budget"]}', text)

    def test_the_report_counts_the_displays_built_from_the_release_and_lists_those_waiting(self):
        report = module("report")
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            (workspace / "reconstruction.json").write_text(json.dumps({
                "displays": [{"id": "main:Table 1", "generator": "built"}, {"id": "main:Figure 2", "generator": "built"},
                             {"id": "main:Figure 3", "generator": "awaiting_inputs", "awaiting": ["campaign:confirmation"]}],
                "outputs": [{"id": "display:main:Table 1", "display_ids": ["main:Table 1"], "input_kind": "repository_sources"},
                            {"id": "display:main:Figure 2", "display_ids": ["main:Figure 2"], "input_kind": "synthetic_examples"}],
                "supplementary_build": {"status": "PASS", "outputs": [{}, {}]}}), encoding="utf-8")
            rows = report.display_rows(workspace, {"values_exact": 3, "values_differ": 2, "values_missing": 1})
            fallback = report.display_rows(None, {"complete": 1, "displays": 3})
        self.assertTrue(rows[0].startswith("- Display generators: 2 of 3 built, 1 of them from the release and 1 from"))
        self.assertIn("2 output(s), status PASS", rows[1])
        self.assertIn("3 equal, 2 different, 1 missing", rows[2])
        self.assertEqual(rows[-1], "- `main:Figure 3`: campaign:confirmation")
        self.assertTrue(fallback[0].startswith("- Displays: 1 of 3 complete"))


if __name__ == "__main__":
    unittest.main()
