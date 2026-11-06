"""scripts/export_records.py: experiment outputs -> traceable record sets.

The campaign, upstream-study, timing and scheduler inputs used here are test fixtures generated in a
temporary directory; they exercise the exporter's arithmetic and its refusals, not any experiment.
The from-runs tests read the bundled synthetic example runs.
"""
from __future__ import annotations

import csv
import copy
import hashlib
import importlib.util
import json
import math
import shutil
import statistics
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from scripts import export_records, records  # noqa: E402

RUNS = ["arxiv", "peptides_func"]
HAS_SCIPY = importlib.util.find_spec("scipy") is not None


def run(argv: list[str]) -> int:
    return export_records.main(argv)


def read_summary(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class FromRunsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "records"
        self.root.mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def export(self, name: str) -> Path:
        argv = ["from-runs", "--set", name, "--root", str(self.root)]
        for stem in RUNS:
            argv += ["--run", str(ROOT / "examples" / "runs" / stem)]
        self.assertEqual(run(argv), 0)
        return self.root / name

    def role_fixture(self) -> Path:
        """Small sealed metadata fixture for the current layout, not an experiment."""
        source = ROOT / "examples/runs/peptides_func"
        destination = self.tmp / "role_fixture"
        destination.mkdir()
        for name in ("config.json", "environment.json", "run_state.json"):
            shutil.copyfile(source / name, destination / name)
        summary = json.loads((source / "final/results_summary.json").read_text())
        summary["prediction_storage_scope"] = "test_role"
        summary["test_role_execution"] = {}
        for path in (source / "predictions").glob("*.json"):
            record = json.loads(path.read_text())
            identity = record["candidate"]
            if (identity["candidate_id"] != summary["deployed"]["candidate_id"]
                    and not (identity["route"] == "nominal"
                             and identity["ensemble"] == summary["comparator"]["ensemble"])):
                continue
            record.update(execution_role="test", role_scoped_execution=True)
            for i, member in enumerate(record["members"]):
                member.update(execution_role="test", graphs=summary["models"]["qN"]["test_units"],
                              wall_seconds=(i + 1) * .01 * summary["models"]["qN"]["test_units"])
            relative = f"predictions/test/{path.name}"
            export_records.dump_json(destination / relative, record)
            summary["test_role_execution"][identity["candidate_id"]] = record
            if identity["route"] == "nominal":
                earlier = copy.deepcopy(record)
                earlier["execution_role"] = "confirm"
                for member in earlier["members"]:
                    member.update(execution_role="confirm", wall_seconds=999.)
                export_records.dump_json(destination / "predictions/confirm" / path.name, earlier)
                export_records.dump_json(destination / "predictions" / path.name, earlier)
        export_records.dump_json(destination / "final/results_summary.json", summary)
        self.seal_fixture(destination)
        return destination

    def seal_fixture(self, run: Path) -> None:
        """Seal deliberately constructed test metadata; no benchmark files are made."""
        from scripts.archive_run import STAGES
        config = json.loads((run / "config.json").read_text())
        config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                               allow_nan=False).encode()).hexdigest()
        previous = None
        for i, stage in enumerate(STAGES):
            # Current evaluate seals the final summary. Test prediction caches
            # are optional derived copies, not additional stage outputs.
            paths = ([run / "config.json", run / "environment.json"] if stage == "prepare" else
                     [run / "final/results_summary.json"] if stage == "evaluate" else [])
            value = {"stage": stage, "source_sha256": "a" * 64, "config_sha256": config_hash,
                     "unit_sha256": "b" * 64, "dataset": "synthetic", "parent_manifest": previous,
                     "created_utc": "2026-01-01T00:00:00Z", "inputs": [],
                     "outputs": [{"path": path.relative_to(run).as_posix(), "bytes": path.stat().st_size,
                                  "sha256": export_records.sha256_file(path)} for path in paths]}
            path = run / "provenance/stages" / f"{i:02d}_{stage}.json"
            export_records.dump_json(path, value)
            previous = {"path": path.relative_to(run).as_posix(), "bytes": path.stat().st_size,
                        "sha256": export_records.sha256_file(path)}

    def export_fixture(self, fixture: Path, name: str) -> int:
        return run(["from-runs", "--set", name, "--root", str(self.root), "--replace", "--run", str(fixture)])

    def test_current_test_role_exports_ignore_earlier_and_flat_records(self):
        fixture = self.role_fixture()
        for name in ("sensor_parameters", "nominal_time"):
            with self.subTest(name=name):
                self.assertEqual(self.export_fixture(fixture, name), 0)
                source = json.loads((self.root / name / "code/SOURCE.json").read_text())
                executions = [r for r in source["inputs"] if r.get("json_pointer")]
                self.assertEqual(len(executions), 1)
                self.assertEqual(executions[0]["file"], "role_fixture/final/results_summary.json")
                self.assertTrue(executions[0]["json_pointer"].startswith("/test_role_execution/"))
        timing = json.loads((self.root / "nominal_time/raw/role_fixture/t_nom.json").read_text())
        self.assertAlmostEqual(timing["t_nom_seconds"], .015)
        self.assertTrue(json.loads((self.root / "nominal_time/provenance.json").read_text())["synthetic"])

    def test_current_exports_do_not_require_optional_test_cache_files(self):
        fixture = self.role_fixture()
        shutil.rmtree(fixture / "predictions")
        self.assertEqual(self.export_fixture(fixture, "sensor_parameters"), 0)
        self.assertEqual(self.export_fixture(fixture, "nominal_time"), 0)

    def test_missing_sealed_test_record_never_falls_back_to_an_unsealed_cache(self):
        fixture = self.role_fixture()
        summary = json.loads((fixture / "final/results_summary.json").read_text())
        candidate_id = summary["deployed"]["candidate_id"]
        del summary["test_role_execution"][candidate_id]
        export_records.dump_json(fixture / "final/results_summary.json", summary)
        self.seal_fixture(fixture)
        self.assertEqual(self.export_fixture(fixture, "sensor_parameters"), 2)
        summary["test_role_execution"] = {}
        export_records.dump_json(fixture / "final/results_summary.json", summary)
        self.seal_fixture(fixture)
        self.assertEqual(self.export_fixture(fixture, "nominal_time"), 2)

    def test_wrong_execution_role_is_rejected_even_with_consistent_summary(self):
        fixture = self.role_fixture()
        summary = json.loads((fixture / "final/results_summary.json").read_text())
        candidate_id = summary["deployed"]["candidate_id"]
        path = fixture / "predictions/test" / f"{candidate_id}.json"
        record = json.loads(path.read_text())
        record["members"][0]["execution_role"] = "confirm"
        summary["test_role_execution"][candidate_id] = record
        export_records.dump_json(path, record)
        export_records.dump_json(fixture / "final/results_summary.json", summary)
        self.seal_fixture(fixture)
        self.assertEqual(self.export_fixture(fixture, "sensor_parameters"), 2)

    def test_nominal_wrong_ensemble_cannot_replace_missing_comparator_ensemble(self):
        fixture = self.role_fixture()
        summary = json.loads((fixture / "final/results_summary.json").read_text())
        for record in summary["test_role_execution"].values():
            if record["candidate"]["route"] == "nominal":
                record["candidate"]["ensemble"] = 1
                record["members"] = record["members"][:1]
        export_records.dump_json(fixture / "final/results_summary.json", summary)
        self.seal_fixture(fixture)
        self.assertEqual(self.export_fixture(fixture, "nominal_time"), 2)

    def test_changed_sealed_final_execution_record_is_rejected(self):
        fixture = self.role_fixture()
        path = fixture / "final/results_summary.json"
        summary = json.loads(path.read_text())
        record = next(iter(summary["test_role_execution"].values()))
        record["members"][0]["wall_seconds"] *= 10
        export_records.dump_json(path, summary)
        self.assertEqual(self.export_fixture(fixture, "nominal_time"), 2)

    def test_test_record_must_match_final_prediction_identity_and_graph_count(self):
        fixture = self.role_fixture()
        summary = json.loads((fixture / "final/results_summary.json").read_text())
        candidate_id = summary["deployed"]["candidate_id"]
        path = fixture / "predictions/test" / f"{candidate_id}.json"
        original = json.loads(path.read_text())
        for corruption in ("prediction", "graphs"):
            with self.subTest(corruption=corruption):
                record = copy.deepcopy(original)
                if corruption == "prediction":
                    record["prediction_sha256"] = "c" * 64
                else:
                    record["members"][0]["graphs"] += 1
                summary["test_role_execution"][candidate_id] = record
                export_records.dump_json(path, record)
                export_records.dump_json(fixture / "final/results_summary.json", summary)
                self.seal_fixture(fixture)
                self.assertEqual(self.export_fixture(fixture, "sensor_parameters"), 2)

    def test_sensor_parameters_read_from_sealed_runs(self):
        target = self.export("sensor_parameters")
        rows = read_summary(target / "summary.csv")
        self.assertEqual(len(rows), 4 * len(RUNS))
        for stem in RUNS:
            config = json.loads((ROOT / "examples/runs" / stem / "config.json").read_text())
            mine = {r["quantity"]: r for r in rows if r["source_file"].startswith(f"raw/{stem}/")}
            self.assertEqual(float(mine["sketch_dim"]["value"]), config["sensor"]["sketch_dim"])
            self.assertEqual(float(mine["observer_period"]["value"]), config["controller"]["observer_period"])
            self.assertEqual(float(mine["sensing_period"]["value"]), config["sensor"]["sensing_period"])
            f_res = float(mine["residual_fraction"]["value"])
            self.assertTrue(0.0 <= f_res <= 1.0)
        provenance = json.loads((target / "provenance.json").read_text())
        self.assertIs(provenance["synthetic"], True)
        self.assertEqual(records.cmd_check(self.root, export_records.SPECS_PATH), 0)

    def test_nominal_time_is_positive_and_traced(self):
        target = self.export("nominal_time")
        rows = read_summary(target / "summary.csv")
        self.assertEqual(len(rows), len(RUNS))
        self.assertTrue(all(float(r["value"]) > 0 for r in rows))
        self.assertTrue(all(r["dataset"].endswith("(synthetic)") for r in rows))
        self.assertEqual(records.cmd_check(self.root, export_records.SPECS_PATH), 0)

    def test_tables_carry_the_synthetic_banner(self):
        self.export("sensor_parameters")
        out = self.tmp / "tables"
        self.assertEqual(records.cmd_tables(self.root, out, export_records.SPECS_PATH), 0)
        text = (out / "sensor_parameters.md").read_text()
        self.assertIn("SYNTHETIC EXAMPLE", text)
        self.assertIn("not benchmark measurements", text)

    def test_synthetic_runs_never_enter_the_official_records_root(self):
        argv = ["from-runs", "--set", "nominal_time", "--root", str(records.RECORDS), "--run",
                str(ROOT / "examples/runs/arxiv")]
        before = sorted(p.name for p in records.RECORDS.iterdir())
        self.assertEqual(run(argv), 2)
        self.assertEqual(sorted(p.name for p in records.RECORDS.iterdir()), before)

    def test_a_changed_raw_value_fails_the_check(self):
        target = self.export("nominal_time")
        t_nom = target / "raw" / "arxiv" / "t_nom.json"
        data = json.loads(t_nom.read_text())
        data["t_nom_seconds"] *= 2
        t_nom.write_text(json.dumps(data))
        self.assertEqual(records.cmd_check(self.root, export_records.SPECS_PATH), 1)

    def test_existing_set_needs_replace(self):
        self.export("nominal_time")
        argv = ["from-runs", "--set", "nominal_time", "--root", str(self.root), "--run", str(ROOT / "examples/runs/arxiv")]
        self.assertEqual(run(argv), 2)
        self.assertEqual(run(argv + ["--replace"]), 0)


ADMITTED = ["ogbn-arxiv", "ogbn-proteins", "Roman-empire", "PascalVOC-SP", "COCO-SP", "Peptides-func"]
ARMS = ["E3", "E4", "E5", "E6"]


def write_campaign(directory: Path, *, passed=True, complete=True, contrasts=None) -> Path:
    """A campaign export in the layout of cm_experiments.campaign.export_campaign (fixture values)."""
    directory.mkdir(parents=True)
    rng = np.random.default_rng(7)
    summary = []
    for d in ADMITTED:
        for a in ARMS:
            effect = float(rng.normal(0.002, 0.001))
            summary.append({"dataset": d, "arm": a, "configuration": "development-selected per controller replicate",
                            "effect": effect, "ci_low": effect - 0.001, "ci_high": effect + 0.001,
                            "p_raw": 0.01, "p_holm": 0.04, "effect_vs_qC": float(rng.normal(-0.003, 0.001)),
                            "work_ratio": 1.1, "reference_work_ratio": 1.2, "ceiling": 1.25})
    if contrasts is not None:
        summary = summary[:contrasts]
    study = {"schema": "cm-budget-matched-3", "record_id": "budget_matched", "seed_ids": list(range(12))}
    (directory / "study.json").write_text(json.dumps(study))
    analysis = {"check": "budget_matched", "passed": passed, "summary": summary}
    (directory / ("analysis.json" if complete else "partial_analysis.json")).write_text(json.dumps(analysis))
    (directory / "coverage.json").write_text(json.dumps({"complete": complete}))
    (directory / "seed_manifest.json").write_text(json.dumps({"seed_ids": list(range(12))}))
    (directory / "paired_observations.csv").write_text("dataset,seed,split,unit_id,arm,qc_loss,qm_loss,arm_loss\n"
                                                       "ogbn-arxiv,0,0,0,E3,1.0,0.99,0.995\n")
    return directory


class BudgetMatchedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "records"
        self.root.mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def argv(self, campaign: Path) -> list[str]:
        return ["budget-matched", "--campaign", str(campaign), "--root", str(self.root),
                "--hardware", "fixture", "--period", "fixture"]

    def test_every_value_is_read_from_the_campaign_analysis(self):
        campaign = write_campaign(self.tmp / "campaign")
        self.assertEqual(run(self.argv(campaign)), 0)
        target = self.root / "budget_matched"
        rows = read_summary(target / "summary.csv")
        self.assertEqual(len(rows), 24 * 7)
        analysis = json.loads((campaign / "analysis.json").read_text())
        effects = {(r["dataset"], r["arm"]): float(r["value"]) for r in rows if r["quantity"] == "effect_vs_qc_nats"}
        for entry in analysis["summary"]:
            self.assertTrue(math.isclose(effects[entry["dataset"], entry["arm"]], entry["effect_vs_qC"], rel_tol=1e-9))
        self.assertEqual(records.cmd_check(self.root, export_records.SPECS_PATH), 0)
        provenance = json.loads((target / "provenance.json").read_text())
        self.assertIs(provenance["synthetic"], False)
        self.assertIs(provenance["complete_campaign"], True)
        # the per-unit observations stay in the campaign archive and are named by hash
        self.assertFalse((target / "raw" / "paired_observations.csv").exists())
        source = json.loads((target / "code" / "SOURCE.json").read_text())
        self.assertIn("paired_observations.csv", {i["file"] for i in source["inputs"]})

    def test_partial_or_failed_campaigns_are_refused(self):
        self.assertEqual(run(self.argv(write_campaign(self.tmp / "partial", complete=False))), 2)
        self.assertEqual(run(self.argv(write_campaign(self.tmp / "failed", passed=False))), 2)
        self.assertEqual(run(self.argv(write_campaign(self.tmp / "short", contrasts=20))), 2)
        self.assertFalse((self.root / "budget_matched").exists())


def plan(dataset="ogbn-arxiv", method="AMP") -> dict:
    return {"schema": "cm-upstream-study-1", "dataset": dataset, "method": method,
            "trials": [{"lr": 10 ** (-3 - i / 48), "layers": 2 + i % 3, "trial": i} for i in range(48)],
            "seed_ids": list(range(1, 13)), "search_seed": 0, "selection_role": "internal_training_validation",
            "direction": "maximize", "metric_unit": "native_0_to_1", "source_commit": "a" * 40, "adapter_commit": "b" * 40,
            "adapter_argv": ["{python}", "adapter.py"], "source_directory": "upstream", "adapter_directory": "adapter"}


SACCT_HEADER = "Cluster|JobIDRaw|JobID|Submit|Start|End|State|ElapsedRaw|AllocTRES|ExitCode"


def sacct_row(job: int) -> str:
    return (f"arrhenius|{job}|{job}|2026-11-02T08:00:00|2026-11-02T08:00:05|2026-11-02T09:00:05|COMPLETED|3600|"
            f"billing=16,cpu=16,gres/gpu=1,mem=64G,node=1|0:0")


def job_key(job: int) -> str:
    return f"arrhenius|{job}|2026-11-02T08:00:00|2026-11-02T08:00:05"


def write_study(root: Path, *, drop_seed=None, classes=4) -> tuple[Path, list[int], dict[int, float]]:
    """An upstream study in the layout of cm_experiments.upstream.execute_attempt (fixture values)."""
    p = plan()
    rng = np.random.default_rng(11)
    jobs, scores = [], {}
    for i in range(48):
        leaf = root / "hpo" / f"trial-{i:03d}"
        leaf.mkdir(parents=True)
        (leaf / "plan.json").write_text(json.dumps(p))
        (leaf / "attempt.json").write_text(json.dumps({"phase": "hpo", "index": i, "seed": 0, "split": 0,
                                                       "status": "PRUNED" if i % 7 == 0 else "COMPLETE",
                                                       "development_score": 0.7, "elapsed_seconds": 10.0,
                                                       "process_exit_code": 0, "reason": None}))
        (leaf / "invocation.json").write_text(json.dumps({"slurm": {"SLURM_JOB_ID": str(100000 + i)}}))
        jobs.append(100000 + i)
    n = 400
    y = rng.integers(0, classes, size=n)
    for k, seed in enumerate(p["seed_ids"]):
        if seed == drop_seed:
            continue
        leaf = root / "final" / f"seed-{seed}" / "split-0"
        (leaf / "adapter_output").mkdir(parents=True)
        (leaf / "plan.json").write_text(json.dumps(p))
        (leaf / "attempt.json").write_text(json.dumps({"phase": "final", "index": k, "seed": seed, "split": 0,
                                                       "status": "COMPLETE", "elapsed_seconds": 100.0}))
        (leaf / "invocation.json").write_text(json.dumps({"slurm": {"SLURM_JOB_ID": str(200000 + k)}}))
        jobs.append(200000 + k)
        logits = rng.normal(size=(n, classes)) + 1.5 * np.eye(classes)[y] * rng.uniform(0.5, 1.5)
        prob = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        np.savez(leaf / "adapter_output" / "predictions.npz", y_true=y, y_pred=prob,
                 observation_id=np.arange(n), unit_id=np.arange(n) % 32)
        (leaf / "adapter_output" / "receipt.json").write_text(json.dumps({"status": "COMPLETE", "predictions": "predictions.npz",
                                                                           "parameter_count": 3_255_000}))
        scores[seed] = float((prob.argmax(1) == y).mean() * 100)
    return root, jobs, scores


def write_timing(path: Path, job: int, *, method="AMP", ratio=1.125, key=None) -> None:
    from cm_campaigns.timing import COLUMNS
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS), lineterminator="\n")
        writer.writeheader()
        for r in range(8):
            nominal = 0.010 + 0.0001 * r
            writer.writerow({"dataset": "ogbn-arxiv", "method": method, "seed": 1, "split": "official",
                             "input_id": "official-test", "repeat": r, "warmup": int(r < 3),
                             "seconds": f"{nominal * (ratio + 0.01 * r):.9f}", "nominal_seconds": f"{nominal:.9f}",
                             "work": 4.5e9, "nominal_work": 1.0e9, "job_key": key or job_key(job), "log": "timing.log"})


@unittest.skipUnless(HAS_SCIPY, "the official-metric scorer needs SciPy (rankdata)")
class BaselinesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "records"
        self.root.mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def inputs(self, **kwargs):
        study, jobs, scores = write_study(self.tmp / "study", **kwargs)
        timing_job = 300000
        sacct = self.tmp / "sacct.psv"
        sacct.write_text("\n".join([SACCT_HEADER] + [sacct_row(j) for j in jobs + [timing_job]]) + "\n")
        timing = self.tmp / "timing.csv"
        write_timing(timing, timing_job)
        return study, sacct, timing, jobs, scores

    def argv(self, study, sacct, timing):
        return ["baselines", "--study", str(study), "--accounting", str(sacct), "--timing", str(timing),
                "--root", str(self.root), "--hardware", "fixture", "--period", "fixture"]

    def test_scores_hours_and_ratios_are_recomputed(self):
        study, sacct, timing, jobs, scores = self.inputs()
        self.assertEqual(run(self.argv(study, sacct, timing)), 0)
        rows = read_summary(self.root / "baselines" / "summary.csv")
        value = {(r["quantity"], r["seed"]): float(r["value"]) for r in rows}
        self.assertEqual(value[("hpo_trials", "")], 48)
        self.assertAlmostEqual(value[("gpu_hours", "")], len(jobs), places=9)   # one GPU-hour per allocation
        self.assertAlmostEqual(value[("params_millions", "")], 3.255)
        for seed, score in scores.items():
            self.assertAlmostEqual(value[("score", str(seed))], score, places=6)
        self.assertGreater(value[("serial_cost", "")], 4.0)
        self.assertGreater(value[("wall_time", "")], 1.0)
        self.assertEqual(records.cmd_check(self.root, export_records.SPECS_PATH), 0)
        out = self.tmp / "tables"
        self.assertEqual(records.cmd_tables(self.root, out, export_records.SPECS_PATH), 0)
        table = (out / "baselines.csv").read_text().splitlines()
        header, line = next(csv.reader([table[1]])), next(csv.reader([table[2]]))
        sd = float(line[header.index("score SD")])
        self.assertAlmostEqual(sd, statistics.stdev(scores.values()), places=5)
        # predictions stay in the study archive, named by hash and path
        self.assertEqual(list((self.root / "baselines").rglob("*.npz")), [])
        score = json.loads(next((self.root / "baselines").rglob("score.json")).read_text())
        self.assertEqual(len(score["predictions_sha256"]), 64)
        self.assertTrue(score["predictions_in_study_archive"].startswith("final/seed-"))

    def test_copied_predictions_are_part_of_the_traced_set(self):
        study, sacct, timing, _, _ = self.inputs()
        self.assertEqual(run(self.argv(study, sacct, timing) + ["--copy-predictions"]), 0)
        self.assertEqual(len(list((self.root / "baselines").rglob("*.npz"))), 12)
        self.assertEqual(records.cmd_check(self.root, export_records.SPECS_PATH), 0)

    def test_a_missing_registered_seed_stops_the_export(self):
        study, sacct, timing, _, _ = self.inputs(drop_seed=7)
        self.assertEqual(run(self.argv(study, sacct, timing)), 2)
        self.assertFalse((self.root / "baselines" / "summary.csv").exists())

    def test_timing_outside_the_accounting_stops_the_export(self):
        study, sacct, timing, _, _ = self.inputs()
        write_timing(timing, 999999)
        self.assertEqual(run(self.argv(study, sacct, timing)), 2)

    def test_allocations_absent_from_accounting_stop_the_export(self):
        study, sacct, timing, jobs, _ = self.inputs()
        lines = sacct.read_text().splitlines()
        sacct.write_text("\n".join(lines[:-3] + lines[-1:]) + "\n")
        self.assertEqual(run(self.argv(study, sacct, timing)), 2)


if __name__ == "__main__":
    unittest.main()
