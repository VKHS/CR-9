"""E17 timing records and the E14 ensemble-grid scores (cm_experiments/timing_records.py,
scripts/run_timing_records.py): records/runtime_scaling, records/timing_anchors and
records/ensemble_grid_scores.

Planning, statistics and export run without PyTorch on the bundled synthetic example runs and on
synthetic job directories written with the runner's own writers (timing rows of the E17 TimingSession
around stand-in executions, or declared rows); every record set goes to a temporary root and is traced
by scripts.records.verify_set. Execution needs PyTorch and runs in CI on fresh smoke lifecycles.
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import math
import re
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cm_experiments import ensemble_grid
from cm_experiments import timing_records as tr
from cm_experiments.ensemble_grid import RunnerError, SealedRun, job_directory, utc_now
from scripts.records import RECORDS, load_specs, read_summary, verify_set

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "runs"
TORCH = importlib.util.find_spec("torch") is not None
ENVIRONMENT = {"python": "3.x", "machine": "x86_64", "torch": None, "gpu": None}


def specs():
    return load_specs(RECORDS, RECORDS / "specs.json")


def summary(target):
    with (Path(target) / "summary.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class Workspace(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cm-timing-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.records = self.root / "records"

    def assert_no_local_paths(self, target):
        """A deposited set names inputs by file name; no local absolute location is published."""
        for path in Path(target).rglob("*"):
            if path.is_file() and path.suffix in (".json", ".csv", ".log", ".psv"):
                for location in (str(self.root), str(EXAMPLES)):
                    self.assertNotIn(location, path.read_text(encoding="utf-8"), path.relative_to(target))


# ----------------------------------------------------------------------------- synthetic executions
def trace(work, *, edges, nodes, peak=None, sparse=3, dense=None):
    """A runtime trace of one member execution (the fields execute_policy reports)."""
    return {"wall_seconds": 1e-4, "multiply_add_work": int(work), "deployed_multiply_add_work": int(work),
            "plant_operations": {"dense_multiply_adds": int(dense if dense is not None else work // 2)},
            "graph_operations": {"sparse_products": sparse, "sparse_multiply_adds": int(work - (dense or work // 2))},
            "peak_cuda_allocated_bytes": peak, "actions": [0] * 4, "terminal_depth": 4, "feasible": True,
            "support_violations": 0}


def execution_records(entry, *, peak=None, members=2, nominal_work=1000, deployed_work=2600):
    edges, nodes = entry["definition"]["edge_entries"] + 3, entry["definition"]["nodes"]

    def nominal():
        return tr.member_record(trace(nominal_work, edges=edges, nodes=nodes, peak=peak), edges, nodes)

    def deployed(call):
        reading = None if peak is None else peak + (1 << 20) * (call % 2)
        return {"members": [tr.member_record(trace(deployed_work // members, edges=edges, nodes=nodes, peak=reading),
                                             edges, nodes) for _ in range(members)], "finite": True}
    return nominal, deployed, edges


def stand_in_executions(entry, *, peak=None, members=2, nominal_work=1000, deployed_work=2600, classes=4):
    """Callables in the shape time_instance drives: a nominal member, then the deployed members one at a time."""
    nominal, deployed, _ = execution_records(entry, peak=peak, members=members, nominal_work=nominal_work,
                                             deployed_work=deployed_work)
    calls = {"deployed": 0}

    def execute_members():
        record = deployed(calls["deployed"])
        calls["deployed"] += 1
        for member in record["members"]:
            sum(range(200))
            yield np.full((entry["definition"]["nodes"], classes), 1 / classes, dtype=np.float32), member
    return nominal, execute_members


def declared_rows(entry, plan, *, nominal_work, nominal_seconds, job_key="unscheduled"):
    """Rows in the E17 layout whose nominal wall time is declared (to check the throughput fit)."""
    rows = []
    for repeat in range(plan["warmup"] + plan["repetitions"]):
        rows.append({"dataset": entry["dataset"], "method": "deployed rule", "seed": entry["member_seeds"][0],
                     "split": entry["split"], "input_id": entry["instance"], "repeat": repeat,
                     "warmup": int(repeat < plan["warmup"]), "seconds": f"{3 * nominal_seconds:.9f}",
                     "nominal_seconds": f"{nominal_seconds + (0.5 if repeat < plan['warmup'] else 0.0):.9f}",
                     "work": 2600, "nominal_work": nominal_work, "job_key": job_key, "log": "session.log"})
    return rows


def complete_scaling_job(output, plan, entry, *, device="cpu", peak=None, rows=None, job_key="unscheduled",
                         nominal_work=1000):
    """A completed job: the runner's E17 session (time_instance) around stand-in executions, or declared rows."""
    stamp = utc_now()
    if rows is None:
        nominal, execute_members = stand_in_executions(entry, peak=peak, nominal_work=nominal_work)
        session = tr.time_instance(entry, plan, job_key=job_key, execute_nominal=nominal,
                                   execute_members=execute_members)
    else:
        nominal, deployed, _ = execution_records(entry, peak=peak, nominal_work=nominal_work)
        session = {"rows": rows, "records": {"nominal": [nominal() for _ in rows],
                                             "deployed": [deployed(i + 1) for i in range(len(rows))]},
                   "counted": {"nominal": nominal(), "deployed": deployed(0)}, "session_started": stamp,
                   "session_finished": stamp}
    return tr.write_scaling_result(output, plan, entry, tr.instance_directory(output, entry), device=device,
                                   job_key=job_key, parent_files=[], started=stamp,
                                   environment=dict(ENVIRONMENT, gpu="test accelerator" if device != "cpu" else None),
                                   **session)


def sacct_export(path, jobs, *, state="COMPLETED", gpus=1):
    """A pipe-delimited sacct export in the columns cm_campaigns.accounting reads (software fixture)."""
    lines = ["Cluster|JobIDRaw|JobID|Submit|Start|End|State|ElapsedRaw|AllocTRES|ExitCode"]
    for number, (job_id, elapsed) in enumerate(jobs, start=7001):
        lines.append(f"testcluster|{number}|{job_id}|2026-10-01T08:00:00|2026-10-01T08:00:{number % 60:02d}|"
                     f"2026-10-01T09:00:00|{state}|{elapsed}|billing=1,cpu=4,gres/gpu={gpus},mem=64G,node=1|0:0")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ============================================================================= contracts
class ContractTests(unittest.TestCase):
    def test_quantities_and_keys_match_the_registered_contracts(self):
        sets = specs()["sets"]
        for name, quantities in tr.QUANTITIES.items():
            with self.subTest(name=name):
                self.assertEqual(set(sets[name]["quantities"]), set(quantities))
        self.assertEqual(sets["runtime_scaling"]["keys"], ["dataset", "instance"])
        self.assertEqual(sets["timing_anchors"]["keys"], ["dataset"])
        self.assertEqual(sets["ensemble_grid_scores"]["keys"], ["dataset", "ensemble"])

    def test_display_generators_read_exactly_these_quantities(self):
        source = (ROOT / "paper" / "build" / "displays" / "records_campaigns.py").read_text(encoding="utf-8")
        read = {"timing_anchors": set(re.findall(r'anchors\.(?:value|require)\("(\w+)"', source)),
                "runtime_scaling": set(re.findall(r'records\.require\("(\w+)", dataset=dataset, instance', source)),
                "ensemble_grid_scores": set(re.findall(r'scores_set\.require\("(\w+)"', source))}
        for name, quantities in read.items():
            with self.subTest(name=name):
                self.assertEqual(quantities, set(tr.QUANTITIES[name]))

    def test_records_root_holds_no_synthetic_or_partial_set_of_this_module(self):
        for name in tr.QUANTITIES:
            self.assertFalse((RECORDS / name).exists(), name)


# ============================================================================= sweep definitions
class SweepTests(unittest.TestCase):
    @staticmethod
    def graph():
        pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (5, 7), (1, 3)]
        edges = np.array(pairs + [(b, a) for a, b in pairs], dtype=np.int64).T
        return edges[:, np.lexsort((edges[1], edges[0]))], 9      # node 8 is isolated

    def test_breadth_first_order_is_seeded_complete_and_level_ordered(self):
        edges, n = self.graph()
        order = tr.bfs_order(edges, n, seed=3, limit=n)
        self.assertEqual(sorted(order.tolist()), list(range(n)))
        np.testing.assert_array_equal(order, tr.bfs_order(edges, n, seed=3, limit=n))
        np.testing.assert_array_equal(tr.bfs_order(edges, n, seed=3, limit=4), order[:4])
        root = int(np.random.default_rng(3).permutation(n)[0])
        self.assertEqual(int(order[0]), root)
        if root in (0, 1, 2, 3, 4):        # the root's component comes first, level by level
            self.assertEqual(set(order[:5].tolist()), {0, 1, 2, 3, 4})
        with self.assertRaises(RunnerError):
            tr.bfs_order(edges, n, seed=3, limit=0)

    def test_subgraphs_are_nested_induced_and_registered_by_size(self):
        edges, n = self.graph()
        instances = tr.subgraph_instances(edges, n, [0.25, 0.75, 0.5], seed=11)
        self.assertEqual([i["instance"] for i in instances], ["bfs-0.75", "bfs-0.5", "bfs-0.25"])
        sets = [set(i["node_ids"].tolist()) for i in instances]
        self.assertTrue(sets[2] <= sets[1] <= sets[0])
        for instance in instances:
            nodes = instance["node_ids"]
            keep, pairs = tr.induced_edges(edges, nodes, n)
            inside = np.isin(edges[0], nodes) & np.isin(edges[1], nodes)
            np.testing.assert_array_equal(keep, inside)
            self.assertEqual(instance["edge_entries"], int(inside.sum()))
            np.testing.assert_array_equal(nodes[pairs], edges[:, keep])
            self.assertEqual(instance["nodes"], len(nodes))
        self.assertEqual([i["nodes"] for i in tr.subgraph_instances(edges, n, [0.5], seed=1)], [5])   # 4.5 -> 5
        with self.assertRaisesRegex(RunnerError, "equal node counts"):
            tr.subgraph_instances(edges, n, [0.5, 0.52], seed=1)

    def test_collection_instances_are_test_graphs_at_edge_count_quantiles(self):
        graph_ptr = np.array([0, 3, 8, 10, 16, 20])
        edge_ptr = np.array([0, 4, 14, 16, 30, 36])           # edge entries 4, 10, 2, 14, 6
        # test graphs ranked by edges: 0 (4), 4 (6), 1 (10), 3 (14); the median rank 1.5 rounds up to 2
        instances = tr.graph_instances(graph_ptr, edge_ptr, [4, 0, 1, 3], [0.0, 0.5, 1.0])
        self.assertEqual([i["graph_id"] for i in instances], [0, 1, 3])
        self.assertEqual([i["edge_entries"] for i in instances], [4, 10, 14])
        self.assertEqual([i["rank"] for i in instances], [0, 2, 3])
        self.assertEqual(instances[1]["nodes"], 5)
        self.assertEqual([i["graph_id"] for i in tr.graph_instances(graph_ptr, edge_ptr, [4, 0, 1], [0.5])], [4])
        self.assertEqual(len(tr.graph_instances(graph_ptr, edge_ptr, [2], [0.0, 1.0])), 1)
        self.assertEqual((tr.half_up(1.5), tr.half_up(2.5), tr.half_up(2.49)), (2, 3, 2))

    def test_npy_hash_equals_the_bytes_numpy_saves(self):
        array = np.random.default_rng(1).random((17, 5)).astype(np.float32)
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        self.assertEqual(tr.npy_sha256(array), hashlib.sha256(buffer.getvalue()).hexdigest())

    def test_throughput_fit_recovers_the_line_and_refuses_flat_or_single_points(self):
        fit = tr.throughput_fit([1e6, 4e6, 9e6], [0.002 + 3e-9 * w for w in (1e6, 4e6, 9e6)])
        self.assertTrue(fit["estimable"])
        self.assertAlmostEqual(fit["slope_seconds_per_multiply_add"], 3e-9, delta=1e-18)
        self.assertAlmostEqual(fit["intercept_seconds"], 0.002, places=12)
        self.assertAlmostEqual(fit["sustained_multiply_adds_per_second"], 1 / 3e-9, delta=1.0)
        self.assertFalse(tr.throughput_fit([5.0], [1.0])["estimable"])
        self.assertFalse(tr.throughput_fit([1.0, 2.0], [2.0, 1.0])["estimable"])


# ============================================================================= runtime_scaling
class ScalingPlanTests(Workspace):
    def plan(self, **kwargs):
        options = dict(fractions=[0.5, 0.25], warmup=1, repetitions=3)
        options.update(kwargs)
        path = tr.plan_runtime_scaling([EXAMPLES / "arxiv", EXAMPLES / "peptides_func"], self.root / "scaling", **options)
        return json.loads(path.read_text())

    def test_plan_registers_full_graphs_subgraphs_and_collection_graphs(self):
        plan = self.plan()
        self.assertEqual([(j["dataset"], j["instance"]) for j in plan["jobs"][:3]],
                         [("ogbn-arxiv", "full"), ("ogbn-arxiv", "bfs-0.5"), ("ogbn-arxiv", "bfs-0.25")])
        run = SealedRun(EXAMPLES / "arxiv")
        full, half = plan["jobs"][0], plan["jobs"][1]
        self.assertEqual(full["definition"]["nodes"], 160)
        self.assertEqual(full["definition"]["edge_entries"], 1204)
        self.assertEqual(full["deployed"]["candidate_id"], run.deployed()["candidate_id"])
        self.assertEqual(full["nominal"]["route"], "nominal")
        self.assertEqual(full["member_seeds"], [1, 2])
        self.assertTrue(full["layout"]["throughput_anchor_applicable"])
        nodes_file = self.root / "scaling" / half["definition"]["node_ids_file"]
        self.assertEqual(ensemble_grid.sha256(nodes_file), half["definition"]["node_ids_sha256"])
        nodes = np.load(nodes_file)
        self.assertEqual(len(nodes), 80)
        self.assertEqual(plan["datasets"][0]["subgraph_seed"], run.config["seed"])
        peptides = [j for j in plan["jobs"] if j["dataset"] == "Peptides-func"]
        self.assertEqual([j["definition"]["kind"] for j in peptides], ["graph"] * len(peptides))
        self.assertFalse(peptides[0]["layout"]["throughput_anchor_applicable"])
        tests = set(np.load(EXAMPLES / "peptides_func" / "prepared" / "graph_splits.npz")["test"].tolist())
        self.assertTrue({j["definition"]["graph_id"] for j in peptides} <= tests)
        public = json.loads(ensemble_grid.public_plan(self.root / "scaling"))
        self.assertNotIn("/", public["jobs"][0]["run"])

    def test_plan_refuses_undeclared_protocols_and_repeated_parents(self):
        for kwargs, message in ((dict(fractions=[1.0]), "strictly between"), (dict(warmup=0), "warm-up"),
                                (dict(repetitions=2), "three measured"), (dict(graph_quantiles=[1.5]), r"\[0, 1\]")):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(RunnerError, message):
                self.plan(**kwargs)
        with self.assertRaisesRegex(RunnerError, "one parent"):
            tr.plan_runtime_scaling([EXAMPLES / "arxiv", EXAMPLES / "arxiv"], self.root / "twice")
        with self.assertRaisesRegex(RunnerError, "outside the repository"):
            tr.plan_runtime_scaling([EXAMPLES / "arxiv"], ROOT / "scaling-output")
        self.plan()
        with self.assertRaisesRegex(RunnerError, "exists"):
            self.plan()


class ScalingSummaryTests(Workspace):
    def test_time_instance_runs_the_e17_session_after_one_counting_execution(self):
        plan = json.loads(tr.plan_runtime_scaling([EXAMPLES / "arxiv"], self.root / "s", fractions=[0.5],
                                                  warmup=2, repetitions=3).read_text())
        entry = plan["jobs"][0]
        log, members = [], 3

        def nominal():
            log.append("nominal")
            return tr.member_record(trace(1000, edges=7, nodes=5), 7, 5)

        def execute_members(value=0.25):
            log.append("deployed")
            for _ in range(members):
                yield np.full((5, 4), value, dtype=np.float32), tr.member_record(trace(900, edges=7, nodes=5), 7, 5)
        session = tr.time_instance(entry, plan, job_key="unscheduled", execute_nominal=nominal,
                                   execute_members=execute_members)
        rows = session["rows"]
        self.assertEqual(len(rows), 5)
        self.assertEqual([r["warmup"] for r in rows], [1, 1, 0, 0, 0])
        self.assertEqual(log[:2], ["nominal", "deployed"])                     # untimed counting execution
        self.assertEqual(log[2:], ["nominal", "deployed", "deployed", "nominal", "nominal", "deployed",
                                   "deployed", "nominal", "nominal", "deployed"])   # E17 alternating order
        self.assertEqual([len(session["records"][k]) for k in ("nominal", "deployed")], [5, 5])
        self.assertEqual(len(session["counted"]["deployed"]["members"]), members)
        self.assertEqual({r["work"] for r in rows}, {2700})
        self.assertEqual({r["nominal_work"] for r in rows}, {1000})
        self.assertEqual({r["input_id"] for r in rows}, {"full"})
        with self.assertRaisesRegex(RunnerError, "nonfinite"):
            tr.time_instance(entry, plan, job_key="unscheduled", execute_nominal=nominal,
                             execute_members=lambda: execute_members(float("nan")))

    def test_wall_time_and_peak_memory_use_the_post_warm_up_repeats_only(self):
        plan = json.loads(tr.plan_runtime_scaling([EXAMPLES / "arxiv"], self.root / "s", fractions=[0.5],
                                                  warmup=2, repetitions=3).read_text())
        entry = plan["jobs"][0]
        rows = declared_rows(entry, plan, nominal_work=1000, nominal_seconds=0.25)
        rows[0]["seconds"] = "9.000000000"                       # a slow warm-up repeat is ignored
        nominal, deployed, edges = execution_records(entry, peak=10 << 30)
        records = {"nominal": [nominal() for _ in rows], "deployed": [deployed(i) for i in range(len(rows))]}
        records["deployed"][0]["members"][0]["peak_cuda_allocated_bytes"] = 99 << 30   # warm-up: ignored
        result = tr.summarize_instance(rows, records, {"nominal": nominal(), "deployed": deployed(0)}, device="cuda:0")
        self.assertAlmostEqual(result["wall_time_seconds"], 0.75, places=12)
        self.assertAlmostEqual(result["nominal_wall_time_seconds"], 0.25, places=12)
        self.assertEqual((result["measured_repeats"], result["warmup_repeats"]), (3, 2))
        self.assertEqual(result["peak_memory_bytes"], (10 << 30) + (1 << 20))
        self.assertAlmostEqual(result["peak_memory_gib"], 10 + 1 / 1024, places=12)
        self.assertEqual(result["processed_edges"], edges)
        self.assertEqual(result["processed_edges_millions"], edges / 1e6)
        cpu = tr.summarize_instance(rows, records, {"nominal": nominal(), "deployed": deployed(0)}, device="cpu")
        self.assertIsNone(cpu["peak_memory_gib"])
        self.assertIn("unavailable", cpu["peak_memory_status"])
        records["deployed"][3]["members"][1]["operator_edges"] = edges + 1
        with self.assertRaisesRegex(RunnerError, "operator changed"):
            tr.summarize_instance(rows, records, {"nominal": nominal(), "deployed": deployed(0)}, device="cpu")
        with self.assertRaisesRegex(RunnerError, "execution record"):
            tr.summarize_instance(rows, {"nominal": [], "deployed": []}, {"nominal": nominal(), "deployed": deployed(0)},
                                  device="cpu")


class ScalingExportTests(Workspace):
    def setUp(self):
        super().setUp()
        self.output = self.root / "scaling"
        self.plan = json.loads(tr.plan_runtime_scaling([EXAMPLES / "arxiv", EXAMPLES / "peptides_func"], self.output,
                                                       fractions=[0.5], graph_quantiles=[0.0, 1.0], warmup=1,
                                                       repetitions=3).read_text())

    def test_cpu_timing_exports_a_traced_partial_set_without_memory_values(self):
        with self.assertRaisesRegex(RunnerError, "has not completed"):
            tr.export_runtime_scaling(self.output, root=self.records, allow_partial=True)
        for entry in self.plan["jobs"]:
            complete_scaling_job(self.output, self.plan, entry)
        with self.assertRaisesRegex(RunnerError, "incomplete"):
            tr.export_runtime_scaling(self.output, root=self.records)
        with self.assertRaisesRegex(RunnerError, "complete measured sets"):
            tr.export_runtime_scaling(self.output, root=RECORDS, allow_partial=True)
        self.assertFalse((RECORDS / "runtime_scaling").exists())
        target = tr.export_runtime_scaling(self.output, root=self.records, allow_partial=True,
                                           command=["scripts/run_timing_records.py", "scaling", "export"])
        entry = verify_set(target, specs()["sets"]["runtime_scaling"], specs())
        self.assertTrue(entry["synthetic"])
        lines = summary(target)
        self.assertEqual(len(lines), 2 * len(self.plan["jobs"]))
        self.assertEqual({r["quantity"] for r in lines}, {"processed_edges_millions", "wall_time_seconds"})
        full = next(r for r in lines if r["instance"] == "full" and r["quantity"] == "processed_edges_millions")
        self.assertAlmostEqual(float(full["value"]), (1204 + 3) / 1e6, places=12)
        provenance = json.loads((target / "provenance.json").read_text())
        self.assertFalse(provenance["measurement_complete"])
        self.assertFalse(provenance["coverage_complete"])
        self.assertTrue(any("peak memory unavailable" in r for r in provenance["incomplete_measurement"]))
        self.assertTrue(any("scheduler accounting" in r for r in provenance["incomplete_measurement"]))
        base = target / "raw" / "ogbn-arxiv" / "arxiv" / "full"
        self.assertEqual(tr.read_timing_rows(base / "timing.csv")[0]["input_id"], "full")
        self.assertTrue((target / "code" / "cm_experiments" / "timing_records.py").is_file())
        self.assertTrue((target / "code" / "scripts" / "run_timing_records.py").is_file())
        self.assertEqual(json.loads((target / "code" / "plan.json").read_text())["record_set"], "runtime_scaling")
        self.assert_no_local_paths(target)

    def test_accounting_binds_each_job_to_a_completed_gpu_allocation(self):
        keys = {}
        for index, entry in enumerate(self.plan["jobs"]):
            keys[entry["instance"], entry["dataset"]] = f"slurm:4242_{index}"
            complete_scaling_job(self.output, self.plan, entry, device="cuda:0", peak=(index + 2) << 30,
                                 job_key=f"slurm:4242_{index}")
        sacct = sacct_export(self.root / "sacct.psv", [(f"4242_{i}", 600 + i) for i in range(len(self.plan["jobs"]))])
        target = tr.export_runtime_scaling(self.output, accounting=[sacct], root=self.records)
        verify_set(target, specs()["sets"]["runtime_scaling"], specs())
        provenance = json.loads((target / "provenance.json").read_text())
        self.assertTrue(provenance["measurement_complete"])
        self.assertTrue(provenance["coverage_complete"])
        binding = json.loads((target / "raw" / "accounting_binding.json").read_text())
        self.assertTrue(all(job["bound"] for job in binding["jobs"]))
        self.assertTrue((target / "raw" / "accounting" / "sacct.psv").is_file())
        peaks = [r for r in summary(target) if r["quantity"] == "peak_memory_gib"]
        self.assertEqual(len(peaks), len(self.plan["jobs"]))
        self.assertAlmostEqual(float(peaks[0]["value"]), 2 + 1 / 1024, places=8)
        self.assert_no_local_paths(target)
        failed = sacct_export(self.root / "failed.psv", [(f"4242_{i}", 600) for i in range(len(self.plan["jobs"]))],
                              state="FAILED")
        with self.assertRaisesRegex(RunnerError, "not a completed GPU allocation"):
            tr.export_runtime_scaling(self.output, accounting=[failed], root=self.root / "other")
        cpu_only = sacct_export(self.root / "cpu.psv", [(f"4242_{i}", 600) for i in range(len(self.plan["jobs"]))],
                                gpus=0)
        with self.assertRaisesRegex(RunnerError, "not a completed GPU allocation"):
            tr.export_runtime_scaling(self.output, accounting=[cpu_only], root=self.root / "other")

    def test_the_figure_s2_generator_reads_the_exported_set(self):
        from paper.build.displays.framework import DisplayInputError
        from paper.build.displays.testing import build_one, context, fake_records
        for index, entry in enumerate(self.plan["jobs"]):
            complete_scaling_job(self.output, self.plan, entry, device="cuda:0", peak=(index + 2) << 30)
        target = tr.export_runtime_scaling(self.output, root=self.records, allow_partial=True)
        rows = summary(target)
        ctx = context(records={"runtime_scaling": fake_records("runtime_scaling", rows, root=target)})
        build_one("supplementary:Figure S2", ctx, self.root / "displays")
        source = next((self.root / "displays").rglob("*panel_a_wall_time*.csv"))
        with source.open(newline="", encoding="utf-8") as handle:
            plotted = list(csv.DictReader(handle))
        self.assertEqual(len(plotted), len(self.plan["jobs"]))
        self.assertEqual({(r["dataset"], r["instance"]) for r in plotted},
                         {(e["dataset"], e["instance"]) for e in self.plan["jobs"]})
        without_memory = [r for r in rows if r["quantity"] != "peak_memory_gib"]
        ctx = context(records={"runtime_scaling": fake_records("runtime_scaling", without_memory, root=target)})
        with self.assertRaises(DisplayInputError):
            build_one("supplementary:Figure S2", ctx, self.root / "displays-cpu")

    def test_altered_raw_files_are_refused(self):
        for entry in self.plan["jobs"]:
            complete_scaling_job(self.output, self.plan, entry)
        timing = tr.instance_directory(self.output, self.plan["jobs"][0]) / "timing.csv"
        timing.write_text(timing.read_text().replace("deployed rule", "deployed rulE", 1))
        with self.assertRaisesRegex(RunnerError, "timing.csv differs"):
            tr.export_runtime_scaling(self.output, root=self.records, allow_partial=True)


# ============================================================================= timing_anchors
class AnchorsTests(Workspace):
    WORK = {"full": 4_000_000, "bfs-0.5": 2_000_000, "bfs-0.25": 1_000_000}
    T0, KAPPA = 0.004, 2.5e-9

    def scaling(self, *, device="cuda:0"):
        output = self.root / "scaling"
        plan = json.loads(tr.plan_runtime_scaling([EXAMPLES / "arxiv", EXAMPLES / "peptides_func"], output,
                                                  fractions=[0.5, 0.25], warmup=1, repetitions=3).read_text())
        for entry in plan["jobs"]:
            work = self.WORK.get(entry["instance"], 1000)
            rows = declared_rows(entry, plan, nominal_work=work, nominal_seconds=self.T0 + self.KAPPA * work)
            complete_scaling_job(output, plan, entry, device=device, peak=1 << 30, rows=rows, nominal_work=work)
        return output

    def plan(self, **kwargs):
        options = dict(scaling_output=self.scaling())
        options.update(kwargs)
        path = tr.plan_timing_anchors([EXAMPLES / "arxiv", EXAMPLES / "peptides_func"], self.root / "anchors", **options)
        return json.loads(path.read_text())

    @staticmethod
    def markers(run):
        run = SealedRun(EXAMPLES / run)
        return [json.loads((run.root / "train" / f"plant_seed{s}_done.json").read_text())
                for s in run.config["training"]["seeds"]]

    def test_training_and_throughput_anchors_follow_their_definitions(self):
        plan = self.plan()
        self.assertEqual([len(j["throughput_instances"]) for j in plan["jobs"]], [3, 0])
        tr.run_timing_anchors(self.root / "anchors")
        arxiv = json.loads((tr.anchor_directory(self.root / "anchors", plan["jobs"][0]) / "anchors.json").read_text())
        quotient = sum(m["seconds_this_attempt"] / m["epochs_completed"] for m in self.markers("arxiv")) / 2
        self.assertAlmostEqual(arxiv["training"]["training_anchor_low_seconds"], quotient / 3.0, places=15)
        self.assertAlmostEqual(arxiv["training"]["training_anchor_high_seconds"], quotient / 1.5, places=15)
        self.assertEqual(arxiv["training"]["time_source"], "seed_marker")
        metadata = json.loads((EXAMPLES / "arxiv" / "prepared" / "metadata.json").read_text())
        w_nom = 6 * (metadata["num_nodes"] * 16 ** 2 + metadata["undirected_edges"] * 16)
        self.assertEqual(arxiv["throughput"]["nominal_work"]["W_nom"], w_nom)
        self.assertAlmostEqual(arxiv["throughput"]["fit"]["slope_seconds_per_multiply_add"], self.KAPPA, delta=1e-17)
        self.assertAlmostEqual(arxiv["throughput"]["fit"]["intercept_seconds"], self.T0, places=9)
        self.assertAlmostEqual(arxiv["throughput"]["throughput_anchor_seconds"], self.KAPPA * w_nom, delta=1e-9)
        peptides = json.loads((tr.anchor_directory(self.root / "anchors", plan["jobs"][1]) / "anchors.json").read_text())
        fit_train = json.loads((EXAMPLES / "peptides_func" / "prepared" / "metadata.json").read_text())[
            "graph_split_sizes"]["fit_train"]
        quotient = sum(m["seconds_this_attempt"] / (m["epochs_completed"] * fit_train)
                       for m in self.markers("peptides_func")) / 2
        self.assertAlmostEqual(peptides["training"]["training_anchor_low_seconds"], quotient / 3.0, places=15)
        self.assertFalse(peptides["throughput"]["applicable"])
        self.assertIsNone(peptides["throughput"]["throughput_anchor_seconds"])
        with self.assertRaisesRegex(RunnerError, "incomplete"):
            tr.export_timing_anchors(self.root / "anchors", root=self.records)
        target = tr.export_timing_anchors(self.root / "anchors", root=self.records, allow_partial=True)
        verify_set(target, specs()["sets"]["timing_anchors"], specs())
        lines = summary(target)
        self.assertEqual(sorted((r["dataset"], r["quantity"]) for r in lines),
                         sorted([("ogbn-arxiv", q) for q in tr.QUANTITIES["timing_anchors"]]
                                + [("Peptides-func", "training_anchor_low_seconds"),
                                   ("Peptides-func", "training_anchor_high_seconds")]))
        provenance = json.loads((target / "provenance.json").read_text())
        self.assertFalse(provenance["measurement_complete"])
        self.assertTrue(any("without anchors" in r for r in provenance["incomplete_measurement"]))
        self.assertTrue((target / "raw" / "ogbn-arxiv" / "train" / "arxiv" / "plant_seed1_done.json").is_file())
        self.assertTrue((target / "raw" / "ogbn-arxiv" / "throughput" / "arxiv" / "full" / "timing.csv").is_file())

    def test_scheduler_accounting_supplies_the_training_time(self):
        sacct = sacct_export(self.root / "train.psv", [(str(9000 + i), 3600 + 60 * i) for i in range(4)])
        with sacct.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="|"))
        job_map = self.root / "job_map.csv"
        with job_map.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["job_key", "dataset", "method", "phase", "seed", "allocation_fraction"])
            for row, (dataset, seed) in zip(rows, [("ogbn-arxiv", 1), ("ogbn-arxiv", 2), ("Peptides-func", 1),
                                                   ("Peptides-func", 2)]):
                key = "|".join(row[k] for k in ("Cluster", "JobIDRaw", "Submit", "Start"))
                writer.writerow([key, dataset, "plant", "backbone_training", seed, 1])
        plan = self.plan(accounting=sacct, job_map=job_map)
        tr.run_timing_anchors(self.root / "anchors")
        arxiv = json.loads((tr.anchor_directory(self.root / "anchors", plan["jobs"][0]) / "anchors.json").read_text())
        epochs = [m["epochs_completed"] for m in self.markers("arxiv")]
        quotient = (3600 / epochs[0] + 3660 / epochs[1]) / 2
        self.assertEqual(arxiv["training"]["time_source"], "scheduler_accounting")
        self.assertAlmostEqual(arxiv["training"]["training_anchor_high_seconds"], quotient / 1.5, places=12)
        target = tr.export_timing_anchors(self.root / "anchors", root=self.records, allow_partial=True)
        verify_set(target, specs()["sets"]["timing_anchors"], specs())
        self.assertTrue((target / "raw" / "accounting" / "train.psv").is_file())
        self.assertTrue((target / "raw" / "accounting" / "job_map.csv").is_file())
        self.assert_no_local_paths(target)

    def test_refusals(self):
        with self.assertRaisesRegex(RunnerError, "forward-equivalent"):
            tr.plan_timing_anchors([EXAMPLES / "arxiv"], self.root / "a", forward_equivalents=(2.0, 2.0))
        with self.assertRaisesRegex(RunnerError, "both"):
            tr.plan_timing_anchors([EXAMPLES / "arxiv"], self.root / "b", accounting=self.root / "x.psv")
        plan = self.plan()
        scaling = Path(plan["scaling"]["output"])
        (scaling / "plan.json").write_text((scaling / "plan.json").read_text() + " ")
        with self.assertRaisesRegex(RunnerError, "changed after"):
            tr.run_timing_anchors(self.root / "anchors")

    def test_missing_throughput_measurement_keeps_the_set_partial(self):
        path = tr.plan_timing_anchors([EXAMPLES / "arxiv"], self.root / "anchors")
        tr.run_timing_anchors(self.root / "anchors")
        result = json.loads((tr.anchor_directory(self.root / "anchors", json.loads(path.read_text())["jobs"][0])
                             / "anchors.json").read_text())
        self.assertIn("not measured", result["throughput"]["status"])
        target = tr.export_timing_anchors(self.root / "anchors", root=self.records, allow_partial=True)
        self.assertNotIn("throughput_anchor_seconds", {r["quantity"] for r in summary(target)})
        self.assertTrue(any("throughput anchor not measured" in r for r in
                            json.loads((target / "provenance.json").read_text())["incomplete_measurement"]))


# ============================================================================= ensemble_grid_scores
class GridScoresTests(Workspace):
    def grid(self, run="arxiv", grid=(1, 2, 4), role="test"):
        output = self.root / f"e14-{run}"
        plan = json.loads(ensemble_grid.plan_ensemble_grid([EXAMPLES / run], output, grid=list(grid), role=role,
                                                           reps=40).read_text())
        return output, plan

    def write_job(self, output, plan, *, seed=5):
        entry = plan["jobs"][0]
        parent = SealedRun(entry["run"])
        with np.load(parent.root / "final" / "test_predictions.npz") as record:
            labels, units = record["labels"], record["cluster_ids"]
        classes = int(parent.state["task"]["num_outputs"])
        rng = np.random.default_rng(seed)
        directory = job_directory(output, entry)
        expected = {"candidate": entry["comparator"], "role": plan["role"]}
        members, files = [], []
        for s in entry["seed_order"]:
            if parent.state["task"]["task_type"] == "multilabel":
                rows = rng.uniform(0.05, 0.95, (len(labels), classes)).astype(np.float32)
            else:
                rows = rng.dirichlet(np.ones(classes), len(labels)).astype(np.float32)
            ensemble_grid.write_member(directory / "members" / f"seed{s}.npy", rows, {**expected, "seed": int(s)})
            members.append(rows)
            files.append({"seed": int(s), "file": f"members/seed{s}.npy",
                          "sha256": ensemble_grid.sha256(directory / "members" / f"seed{s}.npy")})
        rows, arrays = ensemble_grid.grid_statistics(iter(members), labels, units, parent.config, plan["grid"],
                                                     entry["seed_order"], parent_seeds=entry["parent_seeds"],
                                                     comparator_ensemble=entry["comparator_ensemble"], reps=40, seed=1)
        ensemble_grid.write_grid_result(directory, entry, role=plan["role"], rows=rows, arrays=arrays, checkpoints=[],
                                        members=files, reproduction=None, parent_files=[], started=utc_now(),
                                        environment=ENVIRONMENT)
        return rows, members, directory

    def test_scores_are_recomputed_from_the_saved_predictions_with_their_hashes(self):
        output, plan = self.grid()
        with self.assertRaisesRegex(RunnerError, "has not completed"):
            tr.export_ensemble_grid_scores(output, root=self.records, allow_partial=True)
        rows, members, _ = self.write_job(output, plan)
        with self.assertRaisesRegex(RunnerError, "incomplete"):
            tr.export_ensemble_grid_scores(output, root=self.records)
        with self.assertRaisesRegex(RunnerError, "complete measured sets"):
            tr.export_ensemble_grid_scores(output, root=RECORDS, allow_partial=True)
        target = tr.export_ensemble_grid_scores(output, root=self.records, allow_partial=True)
        verify_set(target, specs()["sets"]["ensemble_grid_scores"], specs())
        lines = summary(target)
        self.assertEqual([(r["dataset"], r["ensemble"], r["quantity"], r["unit"]) for r in lines],
                         [("ogbn-arxiv", str(e), "official_score", "%") for e in (1, 2, 4)])
        for line, row in zip(lines, rows):
            self.assertAlmostEqual(float(line["value"]), row["official_score"], places=8)
        scores = json.loads((target / "raw" / "ogbn-arxiv" / "grid_scores.json").read_text())
        for item, size in zip(scores["ensembles"], (1, 2, 4)):
            ensemble = ensemble_grid.ensemble_probabilities(members[:size])
            buffer = io.BytesIO()
            np.save(buffer, ensemble, allow_pickle=False)
            self.assertEqual(item["members"][0]["prediction_sha256"], hashlib.sha256(buffer.getvalue()).hexdigest())
            self.assertEqual(item["members"][0]["seeds"], plan["jobs"][0]["seed_order"][:size])
            self.assertEqual(len(item["members"][0]["member_prediction_sha256"]), size)
        reader = read_summary(target, specs()["sets"]["ensemble_grid_scores"], specs()["summary_columns"])
        self.assertEqual(len(reader), 3)
        provenance = json.loads((target / "provenance.json").read_text())
        self.assertTrue(any("registered E14 grid" in r for r in provenance["incomplete_measurement"]))
        self.assertTrue((target / "raw" / "ogbn-arxiv" / "arxiv" / "grid.json").is_file())
        self.assert_no_local_paths(target)

    def test_graph_level_average_precision_stays_in_native_units(self):
        output, plan = self.grid(run="peptides_func", grid=(1, 2))
        rows, _, _ = self.write_job(output, plan)
        target = tr.export_ensemble_grid_scores(output, root=self.records, allow_partial=True)
        lines = summary(target)
        self.assertEqual({r["unit"] for r in lines}, {"AP"})
        self.assertAlmostEqual(float(lines[1]["value"]), rows[1]["official_score"], places=8)
        self.assertLessEqual(float(lines[1]["value"]), 1.0)

    def test_altered_predictions_and_non_test_grids_are_refused(self):
        output, plan = self.grid()
        _, members, directory = self.write_job(output, plan)
        np.save(directory / "members" / "seed2.npy", members[1][::-1].copy(), allow_pickle=False)
        with self.assertRaisesRegex(RunnerError, "differs from the SHA-256"):
            tr.export_ensemble_grid_scores(output, root=self.records, allow_partial=True)
        compare = self.root / "e14-compare"
        shutil.copytree(output, compare)
        plan_path = compare / "plan.json"
        plan_path.write_text(plan_path.read_text().replace('"role": "test"', '"role": "compare"'))
        with self.assertRaisesRegex(RunnerError, "test-role"):
            tr.export_ensemble_grid_scores(compare, root=self.records, allow_partial=True)


# ============================================================================= command line
class CommandLineTests(Workspace):
    def call(self, *argv):
        from scripts.run_timing_records import main
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main([str(a) for a in argv])
        return code, out.getvalue(), err.getvalue()

    def test_plan_export_and_errors(self):
        output = self.root / "scaling"
        code, out, err = self.call("scaling", "plan", "--parent", EXAMPLES / "arxiv", "--output", output,
                                   "--fractions", "0.5", "--warmup", "1", "--repetitions", "3")
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)["action"], "plan")
        code, _, err = self.call("scaling", "run", "--output", output)
        self.assertEqual(code, 2)
        self.assertIn("trust", err)
        code, _, err = self.call("scaling", "export", "--output", output, "--root", self.records, "--allow-partial")
        self.assertEqual(code, 2)
        self.assertIn("has not completed", err)
        plan = json.loads((output / "plan.json").read_text())
        for entry in plan["jobs"]:
            complete_scaling_job(output, plan, entry)
        code, out, err = self.call("scaling", "export", "--output", output, "--root", self.records, "--allow-partial",
                                   "--hardware", "test host", "--period", "2026-11-01/2026-11-02")
        self.assertEqual(code, 0, err)
        provenance = json.loads((self.records / "runtime_scaling" / "provenance.json").read_text())
        self.assertEqual((provenance["hardware"], provenance["period"]), ("test host", "2026-11-01/2026-11-02"))
        source = json.loads((self.records / "runtime_scaling" / "code" / "SOURCE.json").read_text())
        self.assertEqual(source["command"][:3], ["scripts/run_timing_records.py", "scaling", "export"])
        code, out, err = self.call("anchors", "plan", "--parent", EXAMPLES / "arxiv", "--output", self.root / "anchors",
                                   "--scaling-output", output)
        self.assertEqual(code, 0, err)
        code, out, err = self.call("anchors", "run", "--output", self.root / "anchors")
        self.assertEqual(code, 0, err)
        code, _, err = self.call("grid-scores", "export", "--grid-output", self.root / "missing", "--root", self.records)
        self.assertEqual(code, 2)
        self.assertIn("missing input file", err)


# ============================================================================= execution (PyTorch; CI)
@unittest.skipUnless(TORCH, "PyTorch is required to execute the timing sessions")
class TimingExecutionTests(unittest.TestCase):
    """Smoke lifecycles on synthetic data, then each record set end to end (software checks, not measurements)."""

    @classmethod
    def setUpClass(cls):
        import torch
        from cross_morse.pipeline import Pipeline
        torch.set_num_threads(2)
        cls.temporary = tempfile.TemporaryDirectory(prefix="cm-timing-runs-")
        cls.root = Path(cls.temporary.name)
        cls.runs = {}
        for name, stem in (("arxiv", "arxiv_smoke"), ("peptides", "peptides_func_smoke")):
            config = json.loads((ROOT / "configs" / f"{stem}.json").read_text())
            if name == "arxiv":                 # as tests/test_experiments_runners_a.py builds its smoke runs
                config["training"].update(epochs=1, patience=1)
            config["controller"].update(epochs=1, patience=1)
            config["selection"]["bootstrap_reps"] = 20
            config["admission"]["bootstrap_reps"] = 20
            run = cls.root / "runs" / name
            with contextlib.redirect_stdout(io.StringIO()):
                Pipeline(config, cls.root / "data", run, "cpu").run()
            cls.runs[name] = run

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.work = Path(tempfile.mkdtemp(prefix="cm-timing-", dir=self.root))
        self.records = self.work / "records"

    def test_scaling_sessions_and_anchors_end_to_end(self):
        import torch
        from cross_morse.data import load_data
        from cross_morse.runtime import graph_operator
        output = self.work / "scaling"
        tr.plan_runtime_scaling([self.runs["arxiv"], self.runs["peptides"]], output, fractions=[0.5, 0.25],
                                graph_quantiles=[0.0, 1.0], warmup=1, repetitions=3)
        with self.assertRaisesRegex(RunnerError, "trust"):
            tr.run_runtime_scaling(output, device="cpu")
        with contextlib.redirect_stdout(io.StringIO()):
            done = tr.run_runtime_scaling(output, device="cpu", trust=True)
        plan = json.loads((output / "plan.json").read_text())
        self.assertEqual(len(done), len(plan["jobs"]))
        data = load_data(self.runs["arxiv"])
        full = json.loads(done[0].read_text())
        self.assertEqual(full["instance"], "full")
        self.assertEqual(full["processed_edges"], graph_operator(data, "cpu").num_edges)
        self.assertIsNone(full["peak_memory_gib"])
        self.assertIn("unavailable", full["peak_memory_status"])
        self.assertGreater(full["wall_time_seconds"], 0)
        self.assertGreater(full["nominal_multiply_add_work"], 0)
        subgraph = json.loads(done[1].read_text())
        self.assertLess(subgraph["processed_nodes"], full["processed_nodes"])
        rows = tr.read_timing_rows(done[0].parent / "timing.csv")
        self.assertEqual(len(rows), plan["warmup"] + plan["repetitions"])
        self.assertEqual({r["job_key"] for r in rows}, {tr.timing_job_key()})
        graph = next(json.loads(p.read_text()) for p in done if json.loads(p.read_text())["dataset"] == "Peptides-func")
        gid = graph["definition"]["graph_id"]
        episode = load_data(self.runs["peptides"]).graph(gid)
        self.assertEqual(graph["processed_edges"], graph_operator(episode, "cpu").num_edges)
        self.assertEqual(tr.run_runtime_scaling(output, device="cpu", trust=True), done)     # completed jobs skip
        target = tr.export_runtime_scaling(output, root=self.records, allow_partial=True)
        verify_set(target, specs()["sets"]["runtime_scaling"], specs())
        anchors = self.work / "anchors"
        tr.plan_timing_anchors([self.runs["arxiv"], self.runs["peptides"]], anchors, scaling_output=output)
        tr.run_timing_anchors(anchors)
        target = tr.export_timing_anchors(anchors, root=self.records, allow_partial=True)
        verify_set(target, specs()["sets"]["timing_anchors"], specs())
        result = json.loads((anchors / "jobs" / "ogbn-arxiv" / "arxiv" / "anchors.json").read_text())
        self.assertEqual(len(result["throughput"]["points"]), 3)
        self.assertLess(result["training"]["training_anchor_low_seconds"], result["training"]["training_anchor_high_seconds"])
        self.assertIsNotNone(torch)

    def test_grid_scores_of_an_executed_grid(self):
        output = self.work / "e14"
        ensemble_grid.plan_ensemble_grid([self.runs["arxiv"]], output, grid=[1, 2], reps=20)
        with contextlib.redirect_stdout(io.StringIO()):
            ensemble_grid.run_ensemble_grid(output, device="cpu", trust=True)
        target = tr.export_ensemble_grid_scores(output, root=self.records, allow_partial=True)
        verify_set(target, specs()["sets"]["ensemble_grid_scores"], specs())
        plan = json.loads((output / "plan.json").read_text())
        grid = json.loads((job_directory(output, plan["jobs"][0]) / "grid.json").read_text())
        lines = summary(target)
        for line, row in zip(lines, grid["ensembles"]):
            self.assertTrue(math.isclose(float(line["value"]), row["official_score"], rel_tol=1e-9, abs_tol=1e-9))
        sealed = json.loads((self.runs["arxiv"] / "final" / "results_summary.json").read_text())
        comparator = next(line for line in lines if int(line["ensemble"]) == plan["jobs"][0]["comparator_ensemble"]) \
            if plan["jobs"][0]["comparator_ensemble"] in (1, 2) else None
        if comparator is not None:
            self.assertAlmostEqual(float(comparator["value"]), 100 * sealed["models"]["qC"]["metric"], places=6)


if __name__ == "__main__":
    unittest.main()
