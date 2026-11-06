"""State/byte binding and refusal tests for the author registration workflow.

The bundled synthetic run is used only as a sealed byte fixture. Production
orchestration below explicitly mocks scientific eligibility and all Zenodo I/O;
it does not turn that fixture into an official experiment or contact a service.
"""
from __future__ import annotations

import argparse
import contextlib
import gzip
import io
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from scripts import evidence_pipeline as pipeline
from scripts import evidence_preflight as preflight
from scripts import emit_campaign_binding as campaign
from scripts.archive_run import sha256_file, verify_stage_chain

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/runs/arxiv"
DATASET = "ogbn-arxiv"


def record_worker(plan, dataset, start):
    start.wait(10)
    for index in range(8):
        pipeline.record(plan, dataset, f"check_{index}", {"status": "RECORDED"})


def pack_worker(plan, start, output):
    start.wait(10)
    try:
        args = argparse.Namespace(allow_blocked=True, dataset=[DATASET])
        output.put(pipeline.step_pack(plan, [DATASET], args)[0]["status"])
    except Exception as exc:
        output.put(type(exc).__name__ + ": " + str(exc))


class PipelineSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cm-registration-safety-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.plan_path = self.root / "plan.json"
        plan = pipeline.template()
        plan.update(deposits_dir=str(self.root / "deposits"), license="cc-by-4.0",
                    relationship_to_manuscript="exact_result_producing_run",
                    datasets={DATASET: {"run_root": str(EXAMPLE)}})
        self.plan_path.write_text(json.dumps(plan), encoding="utf-8")
        self.plan = pipeline.load_plan(self.plan_path)

    def args(self, **values):
        return argparse.Namespace(**(dict(full_audit="never", allow_blocked=False, allow_integrity_only=False,
                                          allow_pending_audit=False, sandbox=False, confirm=False, dataset=[DATASET],
                                          set=None, replace=False, force=False) | values))

    def examine(self):
        with contextlib.redirect_stderr(io.StringIO()):
            pipeline.step_preflight(self.plan, [DATASET], self.args())
        return pipeline.current(self.plan, DATASET)

    def pack(self):
        self.examine()
        with contextlib.redirect_stderr(io.StringIO()):
            pipeline.step_pack(self.plan, [DATASET], self.args(allow_blocked=True))
        return pipeline.current(self.plan, DATASET)

    def test_resealed_results_cannot_reuse_protocol_only_preflight(self):
        source = self.root / "run"
        shutil.copytree(EXAMPLE, source)
        self.plan["datasets"][DATASET]["run_root"] = str(source)
        before = self.examine()["preflight"]
        result = source / "final/results_summary.json"
        result.write_text(result.read_text(encoding="utf-8") + " ", encoding="utf-8")
        last = source / "provenance/stages/07_evaluate.json"
        stage = json.loads(last.read_text(encoding="utf-8"))
        for row in stage["outputs"]:
            if row["path"] == "final/results_summary.json":
                row.update(sha256=sha256_file(result), bytes=result.stat().st_size)
        last.write_text(json.dumps(stage), encoding="utf-8")
        chain = verify_stage_chain(source)
        self.assertEqual(before["identities"], {key: chain[key] for key in before["identities"]})
        self.assertNotEqual(before["sealed_content_sha256"], preflight.stage_content_sha256(source))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaisesRegex(pipeline.PipelineError, "sealed results changed"):
            pipeline.step_pack(self.plan, [DATASET], self.args(allow_blocked=True))
        self.assertFalse(pipeline.archive_path(self.plan, DATASET).exists())
        self.assertFalse(any(pipeline.deposits_dir(self.plan).glob("*.partial")))

    def test_recompressed_archive_cannot_reuse_a_saved_byte_identity(self):
        state = self.pack()
        archive = Path(state["archive"]["path"])
        previous = archive.read_bytes()
        archive.write_bytes(gzip.compress(gzip.decompress(previous), compresslevel=1, mtime=1))
        self.assertNotEqual(sha256_file(archive), state["archive"]["sha256"])
        with self.assertRaisesRegex(pipeline.PipelineError, "archive bytes changed"):
            pipeline._gate(DATASET, state["preflight"], state["archive"], self.args(), publishing=False)

    def test_reference_or_validator_changes_require_new_preflight(self):
        state = self.pack()
        changed = dict(state["preflight"]["validation_context"], validation_source_sha256="0" * 64)
        with mock.patch.object(preflight, "validation_context", return_value=changed):
            with self.assertRaisesRegex(pipeline.PipelineError, "validation code or manuscript"):
                pipeline._gate(DATASET, state["preflight"], state["archive"], self.args(), publishing=False)
        changed_plan = dict(self.plan, relationship_to_manuscript="post_result_revision")
        with self.assertRaisesRegex(pipeline.PipelineError, "manuscript declaration changed"):
            pipeline._preflight_context(state["preflight"], changed_plan, DATASET)

    def test_context_hashes_actual_validator_and_manuscript_bytes(self):
        thin = self.root / "checker"
        (thin / "scripts").mkdir(parents=True)
        (thin / "paper/build").mkdir(parents=True)
        script = thin / "scripts/validator.py"
        script.write_text("version = 1\n")
        for name in ("registered_constants.json", "reported_results.json", "sources.json"):
            (thin / "paper" / name).write_text("{}\n")
        with mock.patch.object(preflight, "ROOT", thin):
            first = preflight.validation_context()
            script.write_text("version = 2\n")
            second = preflight.validation_context()
            self.assertNotEqual(first["validation_source_sha256"], second["validation_source_sha256"])
            (thin / "paper/reported_results.json").write_text('{"updated": true}\n')
            third = preflight.validation_context()
            self.assertNotEqual(second["manuscript_inputs"], third["manuscript_inputs"])

    def test_failed_and_pending_checks_never_allow_production_override(self):
        state = self.pack()
        # Only the archive's scientific eligibility is mocked to isolate the verdict rule.
        official_scope = {"profile": "full", "scientific_run": True}
        with mock.patch.object(pipeline, "_verified_archive", return_value=official_scope):
            for verdict in ("BLOCKED", "PENDING_FULL_AUDIT"):
                with self.subTest(verdict=verdict), self.assertRaisesRegex(pipeline.VerdictRefusal, "cannot be overridden"):
                    pipeline._gate(DATASET, dict(state["preflight"], verdict=verdict), state["archive"],
                                   self.args(allow_blocked=True, allow_pending_audit=True, allow_integrity_only=True),
                                   publishing=True)

    def test_receipt_must_bind_exact_archive_and_service(self):
        state = self.pack()
        archive = Path(state["archive"]["path"])
        identity = state["archive"]
        receipt_path = archive.with_name(archive.name + ".deposit.json")
        receipt = {"schema": "cross-morse-zenodo-receipt-1", "service": "https://zenodo.org/api",
                   "filename": archive.name, "bytes": identity["bytes"], "sha256": identity["sha256"],
                   "status": "published", "published": True, "doi": "10.5281/zenodo.7", "reserved_doi": "10.5281/zenodo.7"}
        for change, message in (({"sha256": "0" * 64}, "archive bytes"),
                                ({"service": "https://sandbox.zenodo.org/api"}, "different service"),
                                ({"status": "draft"}, "publication state"),
                                ({"doi": "10.5072/zenodo.7"}, "published DOI")):
            with self.subTest(change=change):
                receipt_path.write_text(json.dumps(receipt | change))
                with self.assertRaisesRegex(pipeline.PipelineError, message):
                    pipeline._bound_receipt(archive, identity, False)

    def test_negative_deposit_returncode_cannot_reuse_a_previous_receipt(self):
        state = self.pack()
        archive = Path(state["archive"]["path"])
        archive.with_name(archive.name + ".sandbox.deposit.json").write_text(json.dumps({"status": "published"}))
        with mock.patch.dict(os.environ, {"ZENODO_SANDBOX_TOKEN": "test-only"}), \
                mock.patch.object(pipeline, "tool", return_value=subprocess.CompletedProcess([], -9, "", "interrupted")):
            with self.assertRaisesRegex(pipeline.PipelineError, "deposit failed"):
                pipeline.step_deposit(self.plan, [DATASET], self.args(sandbox=True), publish=False)
        self.assertNotIn("sandbox_deposit", pipeline.current(self.plan, DATASET))

    def test_preflight_exit_code_distinguishes_completed_checks(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = pipeline.main(["preflight", "--plan", str(self.plan_path), "--full-audit", "never"])
        self.assertEqual(code, 1)
        self.assertEqual(pipeline.current(self.plan, DATASET)["preflight"]["verdict"], "BLOCKED")
        with mock.patch.object(pipeline, "step_preflight", return_value=[{"dataset": DATASET, "status": "INTEGRITY_ONLY"}]), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pipeline.main(["preflight", "--plan", str(self.plan_path)]), 0)

    def test_missing_checks_and_empty_reports_cannot_be_ready(self):
        self.assertEqual(preflight.verdict({"runs": [], "manuscript": {"binding_status": "MATCH"}})[0], "BLOCKED")
        checks = [{"check": name, "status": "PASS", "detail": "fixture"} for name in preflight.INTEGRITY]
        report = {"runs": [{"path": "/fixture", "checks": checks}],
                  "manuscript": {"binding_status": "MATCH", "relationship": "exact_result_producing_run"}}
        self.assertEqual(preflight.verdict(report)[0], "PENDING_FULL_AUDIT")
        report["runs"][0]["checks"] = checks[:-1] + [{"check": "full_audit", "status": "PASS", "detail": "fixture"}]
        self.assertEqual(preflight.verdict(report)[0], "BLOCKED")

    def test_reports_and_deposits_cannot_mutate_source_run(self):
        report = {"root": str(self.root / "source")}
        with self.assertRaisesRegex(preflight.PreflightError, "inside an examined run"):
            preflight.write_outputs([report], self.root / "source/reports")
        unsafe = json.loads(self.plan_path.read_text())
        unsafe["datasets"][DATASET]["run_root"] = str(self.root / "source")
        unsafe["deposits_dir"] = str(self.root / "source/deposits")
        self.plan_path.write_text(json.dumps(unsafe))
        with self.assertRaisesRegex(pipeline.PipelineError, "inside a source run"):
            pipeline.load_plan(self.plan_path)

    @unittest.skipUnless(os.name == "posix", "POSIX locks and fork are required")
    def test_parallel_state_updates_keep_every_dataset_entry(self):
        context = multiprocessing.get_context("fork")
        start = context.Event()
        datasets = list(pipeline.DATASETS[:3])
        workers = [context.Process(target=record_worker, args=(self.plan, dataset, start)) for dataset in datasets]
        for worker in workers:
            worker.start()
        start.set()
        for worker in workers:
            worker.join(20)
            if worker.is_alive():
                worker.terminate()
                worker.join()
            self.assertEqual(worker.exitcode, 0)
        state = pipeline.load_state(self.plan)
        self.assertEqual(len(state["history"]), 24)
        self.assertEqual(set(state["datasets"]), set(datasets))
        self.assertTrue(all(len(state["datasets"][dataset]) == 8 for dataset in datasets))

    @unittest.skipUnless(os.name == "posix", "POSIX locks and fork are required")
    def test_parallel_pack_attempts_do_not_replace_each_others_archive(self):
        self.examine()
        context = multiprocessing.get_context("fork")
        start, output = context.Event(), context.Queue()
        workers = [context.Process(target=pack_worker, args=(self.plan, start, output)) for _ in range(2)]
        for worker in workers:
            worker.start()
        start.set()
        for worker in workers:
            worker.join(30)
            if worker.is_alive():
                worker.terminate()
                worker.join()
            self.assertEqual(worker.exitcode, 0)
        self.assertEqual(sorted(output.get(timeout=5) for _ in workers), ["PACKED", "VERIFIED_EXISTING"])
        state = pipeline.current(self.plan, DATASET)
        pipeline._verified_archive(DATASET, state["archive"])

    def test_production_orchestration_with_mocked_eligibility_and_transport(self):
        """Exercise draft → publish → ingest using real fixture bytes, never real approval."""
        from scripts import ingest_run
        state = self.pack()
        actual_archive = dict(state["archive"])
        # The following two changes are a test double for a successful official-data
        # preflight. The actual fixture remains synthetic; byte checks remain real.
        pipeline.record(self.plan, DATASET, "preflight", state["preflight"] | {"verdict": "READY"})
        pipeline.record(self.plan, DATASET, "archive", actual_archive | {"profile": "full", "scientific_run": True})
        checkout = self.root / "checkout"
        (checkout / "runs").mkdir(parents=True)
        index = checkout / "runs/index.json"
        index.write_text('{"schema":"cross-morse-runs-index-1","runs":[]}\n')
        real_verify = pipeline._verified_archive
        calls = []

        def mocked_eligibility(dataset, archive):
            manifest = real_verify(dataset, archive | {"profile": actual_archive["profile"],
                                                       "scientific_run": actual_archive["scientific_run"]})
            return manifest | {"profile": "full", "scientific_run": True}

        def fake_tool(argv, **kwargs):
            calls.append(argv)
            if argv[0] == "scripts/deposit_zenodo.py":
                archive = Path(argv[2])
                published = "--publish" in argv
                receipt = {"schema": "cross-morse-zenodo-receipt-1", "service": "https://zenodo.org/api",
                           "filename": archive.name, "sha256": actual_archive["sha256"], "bytes": actual_archive["bytes"],
                           "status": "published" if published else "draft", "published": published,
                           "deposition_id": 7, "doi": "10.5281/zenodo.7" if published else None,
                           "reserved_doi": "10.5281/zenodo.7"}
                archive.with_name(archive.name + ".deposit.json").write_text(json.dumps(receipt))
            elif argv[0] == "scripts/ingest_run.py":
                ingest_run.ingest(Path(argv[1]), doi=argv[3], runs_root=checkout / "runs", index_path=index, replace=False)
            return subprocess.CompletedProcess(argv, 0, "", "")

        with mock.patch.object(pipeline, "ROOT", checkout), \
                mock.patch.object(pipeline, "_verified_archive", side_effect=mocked_eligibility), \
                mock.patch.object(pipeline, "tool", side_effect=fake_tool), \
                mock.patch.object(ingest_run, "load_index", side_effect=lambda path=index: json.loads(Path(path).read_text())), \
                mock.patch.dict(os.environ, {"ZENODO_TOKEN": "test-only"}):
            self.assertEqual(pipeline.step_deposit(self.plan, [DATASET], self.args(), publish=False)[0]["status"], "draft")
            self.assertEqual(pipeline.step_deposit(self.plan, [DATASET], self.args(confirm=True), publish=True)[0]["status"], "published")
            self.assertEqual(pipeline.step_register(self.plan, [DATASET], self.args())[0]["status"], "REGISTERED")
            self.assertEqual(pipeline.step_register(self.plan, [DATASET], self.args())[0]["status"], "ALREADY_REGISTERED")
        self.assertEqual(sum(command[0] == "scripts/ingest_run.py" for command in calls), 1)
        self.assertEqual(sum(command[0] == "scripts/run_manifests.py" for command in calls), 2)
        entry = json.loads(index.read_text())["runs"][0]
        self.assertEqual(entry["archive_sha256"], actual_archive["sha256"])
        self.assertTrue((checkout / "runs/arxiv/final/test_predictions.npz").is_file())
        # No official index or manuscript evidence is touched by this isolated routing test.

    def test_pipeline_stages_traced_records_and_preserves_them_on_failed_replacement(self):
        from cm_campaigns.contract import RECORD_SETS
        from scripts.record_mapping import MAPPING_SCHEMA
        checkout = self.root / "record-checkout"
        records = checkout / "records"
        records.mkdir(parents=True)
        shutil.copyfile(ROOT / "records/specs.json", records / "specs.json")
        raw = self.root / "source-records"
        raw.mkdir()
        mapping = {"schema": MAPPING_SCHEMA, "set": "nominal_time", "raw_files": [], "rows": [],
                   "provenance": {"experiment": "isolated exporter test", "produced_by": "unit test fixture",
                                  "hardware": "test host", "period": "test period", "protocol": "test protocol"}}
        for index, dataset in enumerate(RECORD_SETS["nominal_time"]["datasets"]):
            source = raw / f"{index}.json"
            source.write_text(json.dumps({"seconds": 0.5 + index / 8}))
            mapping["raw_files"].append({"from": str(source), "to": f"raw/{index}.json"})
            mapping["rows"].append({"keys": {"dataset": dataset}, "quantity": "t_nom_seconds", "unit": "s",
                                    "source_file": f"raw/{index}.json", "locator": "json:/seconds"})
        mapping_file = self.root / "record-mapping.json"
        mapping_file.write_text(json.dumps(mapping))
        plan = self.plan | {"record_sets": {"nominal_time": {"mapping": str(mapping_file)}}}
        regenerated = []

        def local_tool(argv, **kwargs):
            if argv[0] in ("scripts/export_records.py", "scripts/records.py"):
                command = [sys.executable, "-B", str(ROOT / argv[0]), *argv[1:]]
                if argv[0] == "scripts/records.py":
                    command += ["--root", str(records)]
                result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
                if result.returncode:
                    raise pipeline.PipelineError(result.stderr)
                return result
            regenerated.append(argv[0])
            return subprocess.CompletedProcess(argv, 0, "", "")

        with mock.patch.object(pipeline, "ROOT", checkout), mock.patch.object(pipeline, "tool", side_effect=local_tool):
            self.assertEqual(pipeline.step_records(plan, self.args(set=["nominal_time"]))[0]["status"], "EXPORTED")
            old_index = (records / "index.json").read_bytes()
            old_manifest = (records / "nominal_time/MANIFEST.json").read_bytes()
            old_summary = (records / "nominal_time/summary.csv").read_bytes()
            self.assertEqual(pipeline.step_records(plan, self.args(set=["nominal_time"]))[0]["status"], "VERIFIED_EXISTING")
            # A replacement lacking an actual input must not delete the existing evidence.
            (raw / "0.json").unlink()
            with self.assertRaises(pipeline.PipelineError):
                pipeline.step_records(plan, self.args(set=["nominal_time"], replace=True))
            self.assertEqual((records / "index.json").read_bytes(), old_index)
            self.assertEqual((records / "nominal_time/MANIFEST.json").read_bytes(), old_manifest)
            self.assertEqual((records / "nominal_time/summary.csv").read_bytes(), old_summary)
        self.assertEqual(regenerated, ["scripts/availability.py", "scripts/claims_ledger.py", "scripts/manuscript_availability.py"] * 2)
        self.assertFalse(list(checkout.parent.glob(".cm-record-export-*")))

    def test_record_step_cannot_report_a_different_set_or_root_as_exported(self):
        with self.assertRaisesRegex(pipeline.PipelineError, "different or unsupported record set"):
            pipeline._record_command("factorials", {"exporter": ["from-runs", "--set", "nominal_time"]})
        with self.assertRaisesRegex(pipeline.PipelineError, "override the managed output"):
            pipeline._record_command("nominal_time", {"exporter": ["from-runs", "--set", "nominal_time", "--root=/elsewhere"]})
        wrong = self.root / "wrong.mapping.json"
        wrong.write_text('{"set":"nominal_time"}')
        with self.assertRaisesRegex(pipeline.PipelineError, "mapping names a different"):
            pipeline._record_command("factorials", {"mapping": str(wrong)})


class CampaignToleranceTests(unittest.TestCase):
    def test_campaign_emitter_cannot_inflate_the_printed_tolerance(self):
        identities = [{"split_index": i, "source_sha256": "a" * 64, "config_sha256": "b" * 64,
                       "unit_sha256": f"{i:064x}", "dataset": "Roman-empire"} for i in range(10)]
        for scale in (True, float("nan"), 0, -1, 1.0001, 100):
            with self.subTest(scale=scale), self.assertRaises(campaign.CampaignBindingError):
                campaign.build_binding(identities, relationship="exact_result_producing_run", tolerance_scale=scale)
        with self.assertRaisesRegex(campaign.CampaignBindingError, "ten complete split identities"):
            campaign.build_binding(identities[:-1], relationship="exact_result_producing_run")


if __name__ == "__main__":
    unittest.main()
