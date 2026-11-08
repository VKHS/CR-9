"""The internal rows of Supplementary Tables S15 and S15b (cm_experiments/internal_baselines.py,
scripts/run_internal_baselines.py) and the coverage contract of the baselines set (cm_campaigns/baselines.py).

Planning, the job orchestration, the collection, the export and the Table S15/S15b generators run without PyTorch
on the bundled synthetic example runs: a stand-in for the runtime supplies the parents' sealed test labels,
synthetic member predictions and execution records with the runtime's fields. Every record set goes to a
temporary root. Execution needs PyTorch and runs in CI on fresh smoke lifecycles. None of these values is a
measurement.
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from cm_campaigns import baselines
from cm_campaigns.contract import BASELINE_SEEDS, DATASETS, EXTERNAL, RECORD_SETS, SLUGS
from cm_experiments import internal_baselines as ib
from cm_experiments import timing_records as tr
from cm_experiments.artifacts import verify
from cm_experiments.ensemble_grid import CONTROLLER_ROUTES, RunnerError, SealedRun
from scripts.records import RECORDS, load_specs, verify_set

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "runs"
TORCH = importlib.util.find_spec("torch") is not None
ENVIRONMENT = {"python": "3.x", "machine": "x86_64", "torch": None, "gpu": None}


def specs():
    return load_specs(RECORDS, RECORDS / "specs.json")


def summary(target) -> list[dict]:
    with (Path(target) / "summary.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# ----------------------------------------------------------------------------- stand-in runtime
class FakeFailure(Exception):
    """A numerical policy exit (cross_morse.policies.PolicyExecutionError carries a reason)."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


class FakeContext:
    """Stands in for ParentContext: the parent's sealed test labels and units (final/test_predictions.npz), synthetic
    member predictions keyed by configuration and checkpoint, execution totals with the runtime's fields."""

    def __init__(self, run, *, failing=(), calls=None, interrupted=()):
        self.run, self.config = run, run.config
        self.splits = dict(np.load(run.root / "prepared" / "splits.npz"))
        with np.load(run.root / "final" / "test_predictions.npz") as record:
            self._labels, self._units = record["labels"], record["cluster_ids"]
        self.multilabel = self._labels.ndim == 2
        self.failing, self.interrupted = set(failing), set(interrupted)
        self.calls = calls if calls is not None else []

    def role_ids(self, role):
        return np.asarray(self.splits[role], dtype=np.int64)

    def labels(self, role):
        assert role == "test"
        return self._labels

    def units(self, role):
        return self._units

    def plant(self, seed):
        return SimpleNamespace(seed=int(seed))

    def controller_for(self, candidate):
        return candidate["route"] if candidate["route"] in CONTROLLER_ROUTES else None

    def predict(self, model, candidate, controller, role):
        assert role == "test"
        if (candidate["route"], model.seed) in self.interrupted:
            raise RuntimeError("allocation ended")
        self.calls.append((candidate["candidate_id"], model.seed))
        if (candidate["route"], model.seed) in self.failing:
            raise FakeFailure("nonfinite controller state")
        return member_predictions(self, candidate, model.seed), {
            "deployed_multiply_add_work": 100, "multiply_add_work": 120, "wall_seconds": 0.25, "feasible": True}


def member_predictions(ctx, candidate, seed):
    rng = np.random.default_rng(int(hashlib.sha256(f"{candidate['hash']}/{seed}".encode()).hexdigest()[:8], 16))
    n = len(ctx.role_ids("test"))
    if ctx.multilabel:
        return rng.uniform(0.05, 0.95, (n, ctx._labels.shape[1])).astype(np.float32)
    return rng.dirichlet(np.ones(int(ctx.run.state["task"]["num_outputs"])), n).astype(np.float32)


def fake_execute(*, nominal_work=1000, member_work=1300, changing=False):
    """execute(model, candidate, controller, instance): a member execution's probabilities and runtime record."""
    calls = {"method": 0}

    def execute(model, candidate, controller, instance):
        work = nominal_work
        if candidate["route"] != "nominal":
            calls["method"] += 1
            work = member_work + (calls["method"] if changing else 0)
        trace = {"wall_seconds": 1e-4, "multiply_add_work": work + 7, "deployed_multiply_add_work": work,
                 "graph_operations": {"sparse_products": 3}, "feasible": True}
        return np.full((5, 4), 0.25, dtype=np.float32), tr.member_record(trace, 7, 5)
    return execute


def complete_jobs(output, *, job_key="unscheduled", device="cpu", failing=(), execute=None, calls=None,
                  interrupted=()):
    """Every planned job through the runner's own orchestration with the stand-in runtime."""
    plan = json.loads((Path(output) / "plan.json").read_text())
    paths = []
    for entry in plan["jobs"]:
        ctx = FakeContext(SealedRun(entry["run"]), failing=failing, calls=calls, interrupted=interrupted)
        paths.append(ib.execute_job(output, plan, entry, ib.job_directory(output, entry), ctx=ctx, device=device,
                                    job_key=job_key, execute=execute or fake_execute(),
                                    environment=dict(ENVIRONMENT, gpu=None if device == "cpu" else "test accelerator"),
                                    failures=(FakeFailure,)))
    return plan, paths


def plan_examples(output, *runs, **kwargs):
    options = {"allow_fixture": True, "warmup": 1, "repetitions": 3}
    options.update(kwargs)
    path = ib.plan_internal_baselines([EXAMPLES / r for r in (runs or ("arxiv", "peptides_func"))], output, **options)
    return json.loads(path.read_text())


def campaign(root: Path, splits=2) -> Path:
    """A campaign directory of split runs (copies of the example arxiv run): the layout of a Roman-empire campaign."""
    root.mkdir(parents=True)
    for index in range(splits):
        shutil.copytree(EXAMPLES / "arxiv", root / f"split_{index:02d}")
    return root


class Workspace(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cm-internal-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.records = self.root / "records"


# ============================================================================= the shared method keys and the audit
def registered_keys(dataset):
    """Comparator and deployed keys of the registered Table S15b rows (paper/reported_results.json)."""
    ledger = json.loads((ROOT / "paper" / "reported_results.json").read_text())[
        "supplementary_table_S15b_per_rerun_resource_ledger"][dataset]
    listed = [row["method"] for row in ledger]
    return (next(m for m in listed if m.endswith(" (qC)")), next((m for m in listed if m.endswith(" (deployed)")), None))


def complete_rows(seeds=BASELINE_SEEDS):
    rows = []
    for dataset in DATASETS:
        for method, role in baselines.dataset_methods(dataset, *registered_keys(dataset)).items():
            keys = {"dataset": dataset, "method": method}
            rows += [dict(keys, seed=str(s), quantity="score") for s in range(1, seeds + 1)]
            rows += [dict(keys, seed="", quantity=q) for q in baselines.REQUIRED[role]]
    return rows


def write_study_files(root: Path, *, skip=()):
    """Every external comparison's metadata plus sealed software fixtures, never measured study outputs."""
    from scripts.export_records import package_study_outputs, study_artifact_inventory
    from tests.test_export_records import write_study
    for dataset in DATASETS:
        for method in EXTERNAL[dataset]:
            if (dataset, method) in skip:
                continue
            commit = hashlib.sha1(f"{dataset}/{method}".encode()).hexdigest()
            plan = {"schema": "cm-upstream-study-1", "dataset": dataset, "method": method,
                    "trials": [{"lr": 10.0 ** -k, "trial": k} for k in range(48)], "seed_ids": list(range(12)),
                    "selection_role": "internal_training_validation", "direction": "maximize",
                    "metric_unit": "native_0_to_1", "source_commit": "a" * 40, "adapter_commit": commit,
                    "adapter_argv": ["{python}", "adapter.py"], "source_directory": "upstream",
                    "adapter_directory": "adapter", **({"official_splits": list(range(10))} if dataset == "Roman-empire" else {})}
            paths = baselines.study_paths(dataset, method, commit)
            for relative, content in ((paths["plan"], json.dumps(plan)), (paths["adapter_archive"], f"tar {commit}")):
                (root / relative).parent.mkdir(parents=True, exist_ok=True)
                (root / relative).write_text(content)
            digest = hashlib.sha256((root / paths["adapter_archive"]).read_bytes()).hexdigest()
            with tempfile.TemporaryDirectory(prefix="cm-study-fixture-") as directory:
                study = Path(directory) / "study"
                write_study(study, study_plan=plan)
                inventory = study_artifact_inventory(study, plan)
                archive = package_study_outputs(root, study, plan, inventory)
                for fit in inventory["final_fits"]:
                    source = study / fit["artifacts"]["predictions"]["path"]
                    destination = (root / Path(paths["plan"]).parent / f"seed-{fit['seed']}" /
                                   f"split-{fit['split']}" / source.name)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, destination)
            (root / paths["upstream"]).write_text(json.dumps({
                "schema": baselines.UPSTREAM_SCHEMA, "dataset": dataset, "method": method, "plan": paths["plan"],
                "source_commit": "a" * 40, "adapter_commit": commit, "source_copied": True, "study_archive": archive,
                "adapter_archive": {"file": paths["adapter_archive"], "sha256": digest, "commit": commit}}))


class ContractTests(unittest.TestCase):
    def test_routes_and_families_follow_the_registry(self):
        from paper.build.displays import records_campaigns
        from scripts._paper_common import ROUTE_NAMES
        self.assertEqual(baselines.ROUTES, tuple(ROUTE_NAMES))
        families = json.loads((ROOT / "paper" / "registered_constants.json").read_text())["policy_families_eq15_table_S10"]
        self.assertEqual([ROUTE_NAMES[r] for r in baselines.OPEN_LOOP_ROUTES], families["open_loop"])
        self.assertEqual([ROUTE_NAMES[r] for r in baselines.FEEDBACK_ROUTES], families["feedback"])
        self.assertIs(records_campaigns.ROUTE_METHOD, baselines.ROUTE_METHOD)      # one definition, displays included
        self.assertIs(records_campaigns.S15_ROUTES, baselines.S15_ROUTES)
        self.assertEqual(set(baselines.S15_ROUTES) - set(baselines.ROUTES), set())

    def test_method_keys_reproduce_the_registered_table_s15_rows(self):
        table = json.loads((ROOT / "paper" / "reported_results.json").read_text())["supplementary_table_S15_method_comparison"]
        for dataset in DATASETS:
            with self.subTest(dataset=dataset):
                printed = [row["method"].replace(" (qC, returned)", " (qC)") for row in table[dataset]["rows"]]
                self.assertEqual(list(baselines.dataset_methods(dataset, *registered_keys(dataset))), printed)
        rows = baselines.internal_methods("fixed_sequence", "radius", False)
        self.assertEqual([r["key"] for r in rows], ["full-depth plant", "best fixed depth", "optimized phase",
                                                    "best fixed sequence (qC)", "MPC-OL", "radius (deployed)", "MPC-RH"])
        returned = baselines.internal_methods("learned_stop", "learned_stop", True)
        self.assertEqual([r["key"] for r in returned][:3], ["full-depth plant", "best fixed depth", "learned stop (qC)"])
        self.assertNotIn(baselines.DEPLOYED, {r["role"] for r in returned})
        self.assertEqual(baselines.parse_key("MPC-RH (deployed)"), ("mpc_rh", baselines.DEPLOYED))
        self.assertEqual(baselines.parse_key("optimized phase"), ("phase", baselines.ROUTE))
        for bad in ("AMP", "radius (qc)", "MPC-OL (deployed) (qC)"):
            with self.assertRaises(ValueError):
                baselines.parse_key(bad)
        with self.assertRaises(ValueError):
            baselines.internal_methods("mpc_ol", "mpc_ol", False)

    def test_display_rows_use_the_shared_keys(self):
        from paper.build.displays import records_campaigns as rc
        from paper.build.displays.framework import GRAPH_LEVEL, RunView
        from paper.build.displays.testing import context, fake_records
        runs = {d: RunView(d, EXAMPLES / ("peptides_func" if d in GRAPH_LEVEL else "arxiv")) for d in DATASETS}
        ctx = context(runs=runs, records={})
        for dataset in ("ogbn-arxiv", "Peptides-func"):
            ledger, rows = rc.method_rows(ctx, dataset, fake_records("baselines", []), None)
            internal = [r["key"] for r in rows if r["role"] != "external"]
            self.assertEqual(internal, [m["key"] for m in baselines.internal_methods(
                ledger["qc_route"], ledger["qd_route"], ledger["returned"])])

    def test_a_complete_set_needs_every_internal_row_and_external_comparison(self):
        from scripts.scientific_evidence_audit import validate_record_coverage
        spec = specs()["sets"]["baselines"]
        rows = complete_rows()
        result = validate_record_coverage("baselines", rows, spec, RECORD_SETS["baselines"])
        self.assertEqual(result["external_comparisons"], 14)
        self.assertEqual(result["internal_rows"], sum(len(baselines.dataset_methods(d, *registered_keys(d))) - 2
                                                      for d in DATASETS))
        cases = {
            "rows missing": [r for r in rows if not (r["dataset"] == "COCO-SP" and r["method"] == "best fixed depth")],
            "no gpu_hours": [r for r in rows if not (r["method"] == "MPC-RH" and r["quantity"] == "gpu_hours")],
            "no serial_cost": [r for r in rows if not (r["method"] == "radius (deployed)" and r["quantity"] == "serial_cost")],
            "scores of 11 seeds": [r for r in rows if not (r["method"] == "GRAMA" and r["seed"] == "3")],
            "unexpected": rows + [{"dataset": "ogbn-arxiv", "method": "radius", "seed": "", "quantity": "gpu_hours"}],
            "datasets missing": [r for r in rows if r["dataset"] != "Roman-empire"],
            "no hpo_trials": [r for r in rows if not (r["method"] == "AMP" and r["quantity"] == "hpo_trials")],
        }
        for message, changed in cases.items():
            with self.subTest(case=message), self.assertRaisesRegex(ValueError, message):
                validate_record_coverage("baselines", changed, spec, RECORD_SETS["baselines"])

    def test_a_complete_set_carries_the_plan_upstream_identity_and_adapter_of_every_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_study_files(root)
            self.assertEqual(baselines.audit(complete_rows(), root)["studies"], 14)
            folder = root / "raw" / SLUGS["COCO-SP"] / "Graph_Mamba"
            upstream = json.loads((folder / "upstream.json").read_text())
            (root / upstream["adapter_archive"]["file"]).write_text("changed")
            with self.assertRaisesRegex(ValueError, "COCO-SP/Graph Mamba: .*differs from the SHA-256"):
                baselines.check_studies(root)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_study_files(root, skip={("Peptides-func", "GRAMA")})
            with self.assertRaisesRegex(ValueError, "Peptides-func/GRAMA"):
                baselines.audit(complete_rows(), root)
            write_study_files(root)
            path = root / "raw" / "arxiv" / "AMP" / "upstream.json"
            value = json.loads(path.read_text())
            path.write_text(json.dumps(dict(value, adapter_commit="c" * 40)))
            with self.assertRaisesRegex(ValueError, "adapter commit"):
                baselines.check_studies(root)


# ============================================================================= planning (torch-free)
class PlanTests(Workspace):
    def test_official_split_aggregation_requires_the_registered_members(self):
        from types import SimpleNamespace
        def member(split, dataset="Roman-empire", synthetic=False):
            return SimpleNamespace(dataset=dataset, synthetic=synthetic, config={"data": {"split_index": split}})
        ib.check_benchmark_members([member(i) for i in range(10)])
        for indices in ([0], list(range(9)), [0, *range(9)], [*range(9), True]):
            with self.subTest(indices=indices), self.assertRaisesRegex(RunnerError, "all ten distinct official splits"):
                ib.check_benchmark_members([member(i) for i in indices])
        with self.assertRaisesRegex(RunnerError, "one benchmark run"):
            ib.check_benchmark_members([member(i, dataset="ogbn-arxiv") for i in (0, 1)])
        with self.assertRaisesRegex(RunnerError, "synthetic and benchmark"):
            ib.check_benchmark_members([member(0), member(1, synthetic=True)])
        ib.check_benchmark_members([member(i, synthetic=True) for i in (0, 1)])

    def test_plan_registers_methods_configurations_inputs_and_stage_times(self):
        plan = plan_examples(self.root / "internal")
        self.assertEqual(plan["record_set"], "baselines")
        self.assertEqual(plan["definitions"], ib.DEFINITIONS)
        arxiv, peptides = plan["datasets"]
        self.assertEqual((arxiv["qc_route"], arxiv["returned"], arxiv["E"]), ("learned_stop", True, 2))
        self.assertEqual([m["key"] for m in arxiv["methods"]],
                         ["full-depth plant", "best fixed depth", "learned stop (qC)", "optimized phase",
                          "best fixed sequence", "MPC-OL", "radius", "MPC-RH"])
        self.assertEqual(peptides["methods"][2]["key"], "optimized phase (qC)")
        job = plan["jobs"][0]
        run = SealedRun(EXAMPLES / "arxiv")
        candidates = run.candidates()
        decision = json.loads((run.root / "select" / "decision.json").read_text())
        methods = {m["key"]: m for m in job["methods"]}
        self.assertEqual(methods["learned stop (qC)"]["registry_index"], decision["qC_index"])
        self.assertTrue(methods["learned stop (qC)"]["timed"])
        self.assertEqual([m["key"] for m in job["methods"] if m["timed"]], ["learned stop (qC)"])
        # every other route: the smallest counts-weighted mean D_select loss at E among the feasible executed
        # candidates; a route without one takes its best executed candidate, recorded as infeasible
        from cross_morse.statistics import _candidate_key
        with np.load(run.root / "select" / "unit_scores.npz") as scores:
            weights, loss = scores["counts"].astype(float), scores["loss"]
        for key, method in methods.items():
            if method["role"] != baselines.ROUTE:
                continue
            executed = [j for j, c in enumerate(candidates) if c["route"] == method["route"] and c["ensemble"] == 2
                        and not c["execution_failure"]]
            ids = [j for j in executed if candidates[j]["feasible"]] or executed
            risk = {j: float(weights @ loss[:, j] / weights.sum()) for j in ids}
            best = min(ids, key=lambda j: (risk[j], _candidate_key(candidates[j], j)))
            with self.subTest(method=key):
                self.assertEqual(method["registry_index"], best)
                self.assertEqual(method["ranking"][0]["registry_index"], best)
                self.assertAlmostEqual(method["selection_risk"], risk[best], places=12)
                self.assertEqual(method["qualification"], method["route"] in baselines.FEEDBACK_ROUTES)
        self.assertFalse(methods["best fixed sequence"]["feasible"])       # no feasible configuration at E
        self.assertIn("no feasible configuration", methods["best fixed sequence"]["rule"])
        self.assertTrue(methods["MPC-RH"]["feasible"])                     # its one feasible configuration
        self.assertTrue(methods["optimized phase"]["feasible"])
        self.assertEqual(job["inputs"], [{"instance": "full", "kind": "full", "nodes": 160, "edge_entries": 1204}])
        self.assertEqual(job["nominal"]["route"], "nominal")
        self.assertEqual(job["nominal"]["ensemble"], 1)
        fit = json.loads((run.root / "stages" / "fit.json").read_text())
        self.assertEqual(job["lifecycle_timing"]["stages"]["fit"]["seconds"], fit["seconds_this_attempt"])
        self.assertEqual(set(job["lifecycle_timing"]["checkpoints"]), {"1", "2"})
        graphs = plan["jobs"][1]["inputs"]
        self.assertEqual([g["quantile"] for g in graphs], [0.0, 0.5, 1.0])
        self.assertEqual({g["kind"] for g in graphs}, {"graph"})
        self.assertNotIn(str(EXAMPLES), json.dumps(json.loads(
            __import__("cm_experiments.ensemble_grid", fromlist=["public_plan"]).public_plan(self.root / "internal"))))

    def test_plan_refusals(self):
        with self.assertRaisesRegex(RunnerError, "not on CUDA"):
            ib.plan_internal_baselines([EXAMPLES / "arxiv"], self.root / "cpu")
        self.assertFalse((self.root / "cpu" / "plan.json").exists())
        for kwargs, message in (({"warmup": 0}, "warm-up"), ({"repetitions": 2}, "three measured"),
                                ({"graph_quantiles": [1.5]}, r"\[0, 1\]")):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(RunnerError, message):
                plan_examples(self.root / "x", **kwargs)
        with self.assertRaisesRegex(RunnerError, "one parent"):
            plan_examples(self.root / "twice", "arxiv", "arxiv")
        with self.assertRaisesRegex(RunnerError, "outside the repository"):
            ib.plan_internal_baselines([EXAMPLES / "arxiv"], ROOT / "internal-output", allow_fixture=True)
        plan_examples(self.root / "internal")
        with self.assertRaisesRegex(RunnerError, "exists"):
            plan_examples(self.root / "internal")

    def test_missing_or_unbound_stage_time_is_refused_not_filled(self):
        run = self.root / "arxiv"
        shutil.copytree(EXAMPLES / "arxiv", run)
        marker = run / "stages" / "confirm.json"
        original = json.loads(marker.read_text())
        marker.write_text(json.dumps({k: v for k, v in original.items() if k != "seconds_this_attempt"}))
        with self.assertRaisesRegex(RunnerError, "records no stage time"):
            ib.plan_internal_baselines([run], self.root / "a", allow_fixture=True)
        marker.write_text(json.dumps(dict(original, manifest_sha256="0" * 64)))
        with self.assertRaisesRegex(RunnerError, "not the completion marker"):
            ib.plan_internal_baselines([run], self.root / "b", allow_fixture=True)

    def test_the_selection_statistic_must_reproduce_the_comparator(self):
        original = ib.selection_risk

        def shifted(candidates, loss, counts, *, where):
            risk = original(candidates, loss, counts, where=where)
            risk[8] = risk[9] - 1.0          # another learned-stop configuration ranks first
            return risk
        with patch.object(ib, "selection_risk", shifted), self.assertRaisesRegex(RunnerError, "ranks candidate 8"):
            plan_examples(self.root / "internal", "arxiv")

    def test_a_campaign_has_one_job_per_split_run(self):
        parent = campaign(self.root / "roman")
        plan = json.loads(ib.plan_internal_baselines([parent], self.root / "internal", allow_fixture=True, warmup=1,
                                                     repetitions=3).read_text())
        self.assertEqual([(j["member"], j["split"]) for j in plan["jobs"]], [("split_00", "split_00"),
                                                                             ("split_01", "split_01")])
        self.assertEqual(plan["datasets"][0]["members"], ["split_00", "split_01"])
        self.assertIn("mean", plan["datasets"][0]["aggregation"])


# ============================================================================= jobs (torch-agnostic orchestration)
class JobTests(Workspace):
    def test_scores_timing_and_gpu_hours_of_one_run(self):
        plan = plan_examples(self.root / "internal")
        _, paths = complete_jobs(self.root / "internal", job_key="slurm:4242_0")
        directory = paths[0].parent
        self.assertEqual(verify(directory)["passed"], True)
        result = json.loads(paths[0].read_text())
        self.assertEqual([m["key"] for m in result["methods"]], [m["key"] for m in plan["jobs"][0]["methods"]])
        # per-seed official scores: each configuration with one checkpoint, scored on the sealed test labels
        from cross_morse.evaluation import score_observations
        ctx = FakeContext(SealedRun(EXAMPLES / "arxiv"))
        run = ctx.run
        for method in result["methods"]:
            candidate = run.candidates()[method["candidate"]["registry_index"]]
            for score in method["scores"]:
                expected = score_observations(member_predictions(ctx, candidate, score["seed"]), ctx._labels,
                                              ctx._units, run.config)
                record = json.loads((directory / score["file"]).read_text())
                self.assertEqual(record["score_reported"], 100.0 * expected["metric"])
                self.assertEqual((record["unit"], record["metric_name"]), ("%", "accuracy"))
                self.assertEqual(record["prediction_sha256"],
                                 tr.npy_sha256(member_predictions(ctx, candidate, score["seed"])))
                prediction = directory / record["prediction_file"]
                np.testing.assert_array_equal(np.load(prediction, allow_pickle=False),
                                              member_predictions(ctx, candidate, score["seed"]))
                self.assertEqual(ib.sha256(prediction), record["prediction_sha256"])
                self.assertEqual(record["checkpoint_sha256"],
                                 ib.sha256(run.root / "train" / f"plant_seed{score['seed']}_best.pt"))
        methods = {m["key"]: m for m in result["methods"]}
        # the evaluate stage sealed q_C and q_N with these configurations: the job rebuilds their ensembles
        self.assertEqual(methods["learned stop (qC)"]["reproduction"]["sealed_model"], "qC")
        self.assertTrue(methods["learned stop (qC)"]["reproduction"]["computed"])
        self.assertEqual(methods["full-depth plant"]["reproduction"]["sealed_model"], "qN")
        self.assertFalse(methods["MPC-OL"]["reproduction"]["computed"])
        # E17 rows: q_C against the nominal execution on the full graph, work in nominal executions
        rows = tr.read_timing_rows(directory / "timing.csv")
        self.assertEqual(len(rows), 4)
        self.assertEqual({r["method"] for r in rows}, {"learned stop (qC)"})
        self.assertEqual({r["input_id"] for r in rows}, {"full"})
        self.assertEqual([int(r["warmup"]) for r in rows], [1, 0, 0, 0])
        self.assertEqual({float(r["work"]) for r in rows}, {2.6})           # two members of 1300 over 1000
        self.assertEqual({r["nominal_work"] for r in rows}, {"1"})
        self.assertEqual({r["job_key"] for r in rows}, {"slurm:4242_0"})
        self.assertEqual({r["seed"] for r in rows}, {"1"})
        session = result["timing"]["sessions"][0]
        self.assertEqual((session["method_work"], session["nominal_work"]), (2600, 1000))
        # GPU-hours: backbone training of the E checkpoints; feedback routes add the qualification stages
        gpu = {m["key"]: m for m in json.loads((directory / "gpu_hours.json").read_text())["methods"]}
        timing = plan["jobs"][0]["lifecycle_timing"]
        backbone = timing["checkpoints"]["1"] + timing["checkpoints"]["2"]
        qualification = math.fsum(timing["stages"][s]["seconds"] for s in ib.QUALIFICATION_STAGES)
        self.assertAlmostEqual(gpu["learned stop (qC)"]["gpu_hours"], backbone / 3600, places=15)
        self.assertAlmostEqual(gpu["radius"]["gpu_hours"], (backbone + qualification) / 3600, places=15)
        self.assertEqual(gpu["MPC-OL"]["qualification_seconds"], {})
        peptides = json.loads(paths[1].read_text())
        self.assertEqual({s["input"] for s in peptides["timing"]["sessions"]},
                         {i["instance"] for i in plan["jobs"][1]["inputs"]})
        self.assertEqual(peptides["methods"][2]["scores"][0]["unit"], "AP")
        for path in directory.rglob("*"):
            if path.is_file():
                self.assertNotIn(str(EXAMPLES).encode(), path.read_bytes(), path.name)

    def test_an_interrupted_job_keeps_its_scores_and_times_again(self):
        plan = plan_examples(self.root / "internal", "arxiv")
        entry = plan["jobs"][0]
        directory = ib.job_directory(self.root / "internal", entry)
        seeds = len(entry["seeds"])

        def broken(model, candidate, controller, instance):
            raise RuntimeError("allocation ended")
        calls = []          # the first attempt ends during the comparator's second checkpoint
        with self.assertRaisesRegex(RuntimeError, "allocation ended"):
            complete_jobs(self.root / "internal", calls=calls, interrupted={("learned_stop", 2)})
        self.assertFalse((directory / ib.MANIFEST).exists())
        self.assertEqual(len(calls), 2 * seeds + 1)
        calls = []          # the second ends in its timing session, after every score
        with self.assertRaisesRegex(RuntimeError, "allocation ended"):
            complete_jobs(self.root / "internal", execute=broken, calls=calls)
        self.assertEqual(len(calls), len(entry["methods"]) * seeds - 2 * seeds - 1)
        (directory / "timing.csv").write_text("stale rows of another allocation")
        later = []
        _, paths = complete_jobs(self.root / "internal", job_key="slurm:77", calls=later)
        self.assertEqual(later, [])                                    # the scores are not executed again
        rows = tr.read_timing_rows(paths[0].parent / "timing.csv")
        self.assertEqual({r["job_key"] for r in rows}, {"slurm:77"})    # timed in the allocation that completes it
        result = json.loads(paths[0].read_text())
        self.assertTrue(result["methods"][0]["reproduction"]["computed"])       # q_N: members of one attempt
        self.assertFalse(result["methods"][2]["reproduction"]["computed"])      # q_C: members of two attempts
        self.assertIn("earlier attempt", result["methods"][2]["reproduction"]["reason"])
        self.assertEqual(complete_jobs(self.root / "internal", calls=later)[1], paths)   # complete: skipped
        self.assertEqual(later, [])

    def test_a_numerical_failure_leaves_the_seed_without_a_score(self):
        plan_examples(self.root / "internal", "arxiv")
        _, paths = complete_jobs(self.root / "internal", failing={("radius", 2)})
        result = json.loads(paths[0].read_text())
        radius = next(m for m in result["methods"] if m["key"] == "radius")
        self.assertEqual(radius["failed_seeds"], [2])
        record = json.loads((paths[0].parent / radius["scores"][1]["file"]).read_text())
        self.assertEqual((record["status"], record["failure_reason"]), ("FAILED", "nonfinite controller state"))
        self.assertNotIn("score_reported", record)
        collected = ib.collect_internal_baselines(self.root / "internal")
        method = next(m for m in collected["datasets"][0]["methods"] if m["key"] == "radius")
        self.assertEqual([s["seed"] for s in method["seeds"]], [1])
        self.assertTrue(any("radius" in r and "seed 2" in r for r in collected["reasons"]))

    def test_resume_refuses_a_saved_score_with_missing_prediction_bytes(self):
        plan_examples(self.root / "internal", "arxiv")
        with self.assertRaisesRegex(RuntimeError, "allocation ended"):
            complete_jobs(self.root / "internal", interrupted={("learned_stop", 2)})
        entry = json.loads((self.root / "internal/plan.json").read_text())["jobs"][0]
        directory = ib.job_directory(self.root / "internal", entry)
        next(directory.glob("scores/*/seed1.npy")).unlink()
        with self.assertRaisesRegex(RunnerError, "prediction bytes are missing or changed"):
            complete_jobs(self.root / "internal")

    def test_collect_refuses_missing_predictions_even_with_a_new_outer_seal(self):
        from cm_experiments.artifacts import seal
        plan_examples(self.root / "internal", "arxiv")
        _, paths = complete_jobs(self.root / "internal")
        directory = paths[0].parent
        next(directory.glob("scores/*/seed1.npy")).unlink()
        (directory / ib.MANIFEST).unlink()
        seal(directory)
        with self.assertRaisesRegex(RunnerError, "per-seed prediction bytes are missing or changed"):
            ib.collect_internal_baselines(self.root / "internal")

    def test_the_session_alternates_and_refuses_work_that_changes(self):
        plan = plan_examples(self.root / "internal", "arxiv")
        entry = plan["jobs"][0]
        method = next(m for m in entry["methods"] if m["timed"])
        log = []

        def nominal():
            log.append("nominal")
            return tr.member_record({"wall_seconds": 0.0, "multiply_add_work": 1000}, 7, 5)

        def members():
            log.append("method")
            for _ in method["member_seeds"]:
                yield np.full((5, 4), 0.25, dtype=np.float32), tr.member_record(
                    {"wall_seconds": 0.0, "multiply_add_work": 1500}, 7, 5)
        session = ib.time_method(entry, plan, method, entry["inputs"][0], job_key="unscheduled",
                                 execute_nominal=nominal, execute_members=members)
        self.assertEqual(log[:2], ["nominal", "method"])                  # the untimed counting execution
        self.assertEqual(log[2:], ["nominal", "method", "method", "nominal", "nominal", "method", "method", "nominal"])
        self.assertEqual(session["work_ratio"], 3.0)
        self.assertEqual({r["method"] for r in session["rows"]}, {method["key"]})
        changing = fake_execute(changing=True)
        with self.assertRaisesRegex(RunnerError, "work differs"):
            ib.time_method(entry, plan, method, entry["inputs"][0], job_key="unscheduled",
                           execute_nominal=lambda: changing(None, {"route": "nominal"}, None, None)[1],
                           execute_members=lambda: (changing(None, {"route": "x"}, None, None) for _ in range(2)))
        with self.assertRaisesRegex(RunnerError, "nonfinite"):
            ib.time_method(entry, plan, method, entry["inputs"][0], job_key="unscheduled", execute_nominal=nominal,
                           execute_members=lambda: iter([(np.full((5, 4), np.nan, dtype=np.float32),
                                                          nominal())]))

    def test_collect_refuses_altered_or_foreign_outputs(self):
        plan_examples(self.root / "internal", "arxiv")
        with self.assertRaisesRegex(RunnerError, "has not completed"):
            ib.collect_internal_baselines(self.root / "internal")
        _, paths = complete_jobs(self.root / "internal")
        collected = ib.collect_internal_baselines(self.root / "internal")
        self.assertEqual(collected["datasets"][0]["dataset"], "ogbn-arxiv")
        self.assertTrue(collected["synthetic"])
        copy = self.root / "copy"
        shutil.copytree(self.root / "internal", copy)
        plan_file = copy / "plan.json"
        plan_file.write_text(plan_file.read_text().replace('"warmup": 1', '"warmup": 1 '))
        with self.assertRaisesRegex(RunnerError, "another plan"):
            ib.collect_internal_baselines(copy)
        score = next(paths[0].parent.glob("scores/radius/seed1.json"))
        score.write_text(score.read_text().replace('"score_reported": ', '"score_reported": 1'))
        with self.assertRaisesRegex(RunnerError, "does not verify"):
            ib.collect_internal_baselines(self.root / "internal")


# ============================================================================= export and the S15/S15b generators
class ExportTests(Workspace):
    def export(self, *extra):
        from scripts.export_records import main
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = main(["baselines", "--internal", str(self.root / "internal"), "--root", str(self.records),
                         "--hardware", "fixture", "--period", "fixture", *map(str, extra)])
        return code, err.getvalue()

    def test_a_campaign_exports_seed_means_and_summed_gpu_hours(self):
        parent = campaign(self.root / "roman")
        ib.plan_internal_baselines([parent], self.root / "internal", allow_fixture=True, warmup=1, repetitions=3)
        _, paths = complete_jobs(self.root / "internal")
        code, err = self.export("--allow-partial")
        self.assertEqual(code, 0, err)
        target = self.records / "baselines"
        verify_set(target, specs()["sets"]["baselines"], specs())
        rows = summary(target)
        mean = next(r for r in rows if r["method"] == "MPC-OL" and r["seed"] == "2")
        self.assertEqual(mean["source_file"], "raw/arxiv/MPC-OL/seed-2/score_mean.json")
        parts = json.loads((target / mean["source_file"]).read_text())
        self.assertEqual([p["run"] for p in parts["splits"]], ["split_00", "split_01"])
        self.assertEqual(len({p["score_reported"] for p in parts["splits"]}), 1)       # identical split copies
        gpu = next(r for r in rows if r["method"] == "radius" and r["quantity"] == "gpu_hours")
        hours = [json.loads(p.with_name("gpu_hours.json").read_text())["methods"][6]["gpu_hours"] for p in paths]
        self.assertAlmostEqual(float(gpu["value"]), sum(hours), places=12)
        self.assertEqual(gpu["source_file"], "raw/arxiv/radius/gpu_hours.json")
        timing = json.loads((target / "raw/arxiv/learned_stop_qC/timing_ratios.json").read_text())
        self.assertEqual(timing["timing_files"], ["raw/internal/arxiv/split_00/timing.csv",
                                                  "raw/internal/arxiv/split_01/timing.csv"])
        self.assertEqual(timing["observations"], 6)
        self.assertTrue((target / "code" / "cm_experiments" / "internal_baselines.py").is_file())
        self.assertNotIn(str(self.root), (target / "code" / "internal" / "internal" / "plan.json").read_text())

    def test_tables_s15_and_s15b_read_the_exported_rows(self):
        from paper.build.displays.framework import GRAPH_LEVEL, RunView
        from paper.build.displays.testing import build_one, context, fake_records, read_table
        plan_examples(self.root / "internal")
        complete_jobs(self.root / "internal")
        code, err = self.export("--allow-partial")
        self.assertEqual(code, 0, err)
        rows = summary(self.records / "baselines")
        runs = {d: RunView(d, EXAMPLES / ("peptides_func" if d in GRAPH_LEVEL else "arxiv")) for d in DATASETS}
        ctx = context(runs=runs, records={"baselines": fake_records("baselines", rows)})
        build_one("supplementary:Table S15b", ctx, self.root / "displays")
        ledger = {r["method"]: r for r in read_table(self.root / "displays", "S15b") if r["Dataset"] == "ogbn-arxiv"}
        values = {(r["method"], r["quantity"]): float(r["value"]) for r in rows if r["dataset"] == "ogbn-arxiv" and not r["seed"]}
        self.assertEqual(ledger["learned stop (qC)"]["serial cost Cctrl"], f"{values[('learned stop (qC)', 'serial_cost')]:.1f}")
        self.assertEqual(ledger["learned stop (qC)"]["GPU-h"], f"{values[('learned stop (qC)', 'gpu_hours')]:.2f}")
        self.assertNotIn(ledger["learned stop (qC)"]["SD (12 seeds)"], ("", "—"))
        build_one("supplementary:Table S15", context(runs=runs, records={"baselines": fake_records("baselines", rows),
                                                                         "input_gate": fake_records("input_gate", [])}),
                  self.root / "s15")
        table = [r for r in read_table(self.root / "s15", "S15") if r["Dataset"] == "ogbn-arxiv"]
        phase = next(r for r in table if r["method"] == "optimized phase")
        scores = [float(r["value"]) for r in rows if r["method"] == "optimized phase" and r["dataset"] == "ogbn-arxiv"
                  and r["quantity"] == "score"]
        self.assertEqual(phase["score"], f"{sum(scores) / len(scores):.2f}")
        self.assertNotIn(phase["GPU-h"], ("", "—"))


# ============================================================================= command line
class CommandLineTests(Workspace):
    def call(self, *argv):
        from scripts.run_internal_baselines import main
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main([str(a) for a in argv])
        return code, out.getvalue(), err.getvalue()

    def test_plan_run_guard_and_export(self):
        output = self.root / "internal"
        code, _, err = self.call("plan", "--parent", EXAMPLES / "arxiv", "--output", output)
        self.assertEqual(code, 2)
        self.assertIn("not on CUDA", err)
        code, out, err = self.call("plan", "--parent", EXAMPLES / "arxiv", "--output", output, "--allow-fixture",
                                   "--warmup", "1", "--repetitions", "3")
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)["result"], str(output / "plan.json"))
        code, _, err = self.call("run", "--output", output, "--device", "cpu")
        self.assertEqual(code, 2)
        self.assertIn("trust", err)
        code, _, err = self.call("export", "--output", output, "--root", self.records, "--hardware", "h", "--period", "p",
                                 "--allow-partial")
        self.assertEqual(code, 2)
        self.assertIn("has not completed", err)
        complete_jobs(output)
        code, out, err = self.call("export", "--output", output, "--root", self.records, "--hardware", "h", "--period",
                                   "p", "--allow-partial")
        self.assertEqual(code, 0, err)
        source = json.loads((self.records / "baselines" / "code" / "SOURCE.json").read_text())
        self.assertEqual(source["command"][:2], ["scripts/run_internal_baselines.py", "export"])


# ============================================================================= execution (PyTorch; CI)
@unittest.skipUnless(TORCH, "PyTorch is required to execute the configurations")
class InternalExecutionTests(unittest.TestCase):
    """Smoke lifecycles on synthetic data, then the runner end to end (software checks, not measurements)."""

    @classmethod
    def setUpClass(cls):
        import torch
        from cross_morse.pipeline import Pipeline
        torch.set_num_threads(2)
        cls.temporary = tempfile.TemporaryDirectory(prefix="cm-internal-runs-")
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

    def test_scores_timing_gpu_hours_and_export_end_to_end(self):
        work = Path(tempfile.mkdtemp(prefix="cm-internal-", dir=self.root))
        output = work / "internal"
        with self.assertRaisesRegex(RunnerError, "not on CUDA"):
            ib.plan_internal_baselines([self.runs["arxiv"]], output)
        plan = json.loads(ib.plan_internal_baselines([self.runs["arxiv"], self.runs["peptides"]], output,
                                                     allow_fixture=True, warmup=1, repetitions=3).read_text())
        with self.assertRaisesRegex(RunnerError, "trust"):
            ib.run_internal_baselines(output, device="cpu")
        with contextlib.redirect_stdout(io.StringIO()):
            done = ib.run_internal_baselines(output, device="cpu", trust=True)
        self.assertEqual(len(done), len(plan["jobs"]))
        self.assertEqual(ib.run_internal_baselines(output, device="cpu", trust=True), done)    # completed: skipped
        for path, entry in zip(done, plan["jobs"]):
            result = json.loads(path.read_text())
            self.assertTrue(verify(path.parent)["passed"])
            self.assertEqual([m["key"] for m in result["methods"]], [m["key"] for m in entry["methods"]])
            for method in result["methods"]:
                self.assertEqual([s["seed"] for s in method["scores"]], entry["seeds"])
                for score in method["scores"]:
                    if score["status"] == "COMPLETE":
                        self.assertTrue(math.isfinite(score["score_reported"]))
                if method["reproduction"].get("computed"):
                    # the one-member executions rebuild the sealed ensemble of the same configuration
                    self.assertLess(method["reproduction"]["abs_difference"], 1e-6)
            rows = tr.read_timing_rows(path.parent / "timing.csv")
            timed = [m["key"] for m in entry["methods"] if m["timed"]]
            self.assertEqual(len(rows), len(timed) * len(entry["inputs"]) * 4)
            self.assertEqual({r["job_key"] for r in rows}, {tr.timing_job_key()})
            for session in result["timing"]["sessions"]:
                self.assertGreater(session["work_ratio"], 0)
                if entry["inputs"][0]["kind"] == "full":    # the full graph: the ledger cost of select/candidates.json
                    self.assertAlmostEqual(session["work_ratio"], session["ledger_cost"], places=9)
        collected = ib.collect_internal_baselines(output)
        self.assertTrue(any("not GPU-hours" in r for r in collected["reasons"]))
        from scripts.export_records import main
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(["baselines", "--internal", str(output), "--root", str(work / "records"), "--hardware", "cpu",
                         "--period", "test", "--allow-partial"])
        self.assertEqual(code, 0)
        verify_set(work / "records" / "baselines", specs()["sets"]["baselines"], specs())


if __name__ == "__main__":
    unittest.main()
