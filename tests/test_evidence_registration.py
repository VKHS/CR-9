"""Evidence registration: preflight, plan pipeline, campaign binding, record mappings, sources and readiness.

No network, no PyTorch and no writes into the checkout; Zenodo and make are replaced by recorders.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts import emit_campaign_binding as campaign  # noqa: E402
from scripts import evidence_pipeline as pipeline  # noqa: E402
from scripts import evidence_preflight as preflight  # noqa: E402
from scripts import readiness  # noqa: E402
from scripts import record_mapping  # noqa: E402
from scripts import register_manuscript_sources as sources  # noqa: E402

EXAMPLE = PROJECT / "examples" / "runs" / "arxiv"


def temporary(test: unittest.TestCase) -> Path:
    path = Path(tempfile.mkdtemp(prefix="cm-evidence-test-"))
    test.addCleanup(shutil.rmtree, path, True)
    return path


def check(report: dict, name: str) -> dict:
    return next(c for run in report["runs"] for c in run["checks"] if c["check"] == name)


class PreflightTests(unittest.TestCase):
    def test_synthetic_example_is_blocked_by_scope_before_any_manuscript_claim(self):
        report = preflight.preflight_dataset("ogbn-arxiv", EXAMPLE, "run", full_audit="never")
        self.assertEqual(report["verdict"], "BLOCKED")
        self.assertEqual(check(report, "stage_chain")["status"], "PASS")
        self.assertEqual(check(report, "sealed_inventory")["status"], "PASS")
        self.assertEqual(check(report, "scope")["status"], "FAIL")
        self.assertIn("synthetic", check(report, "scope")["detail"])
        self.assertEqual(check(report, "full_audit")["status"], "NOT_RUN")
        self.assertIsNone(report["manuscript"])
        self.assertTrue(any("scope FAIL" in blocker for blocker in report["blockers"]))
        self.assertEqual(set(report["identities"]), {"source_sha256", "config_sha256", "unit_sha256"})
        json.dumps(report)

    def test_altered_sealed_output_fails_the_chain_and_skips_later_checks(self):
        run = temporary(self) / "arxiv"
        shutil.copytree(EXAMPLE, run)
        summary = run / "final" / "results_summary.json"
        summary.write_text(summary.read_text(encoding="utf-8") + " ", encoding="utf-8")
        report = preflight.preflight_dataset("ogbn-arxiv", run, "run", full_audit="never")
        self.assertEqual(report["verdict"], "BLOCKED")
        self.assertEqual(check(report, "stage_chain")["status"], "FAIL")
        for name in ("sealed_inventory", "scope", "data_roles", "numeric_replay", "full_audit"):
            self.assertEqual(check(report, name)["status"], "SKIPPED", name)

    def test_campaign_rules_and_missing_splits(self):
        root = temporary(self)
        with self.assertRaises(preflight.PreflightError):
            preflight.preflight_dataset("ogbn-arxiv", root, "campaign")
        with self.assertRaises(preflight.PreflightError):
            preflight.preflight_dataset("Roman-empire", root, "run")
        report = preflight.preflight_dataset("Roman-empire", root, "campaign", full_audit="never")
        self.assertEqual(report["verdict"], "BLOCKED")
        self.assertIn("split_00", report["structure_error"])

    def test_verdict_ladder(self):
        def report(full="PASS", integrity="PASS", source="PASS", binding="MATCH", relationship="exact_result_producing_run", missing=()):
            checks = [{"check": name, "status": source if name == "runtime_source" else integrity, "detail": "x"}
                      for name in preflight.INTEGRITY]
            checks += [{"check": "full_audit", "status": full, "detail": "y"}]
            return {"runs": [{"path": "/r/run", "checks": checks}],
                    "manuscript": {"binding_status": binding, "binding_coverage_missing": list(missing),
                                   "binding_results": [], "relationship": relationship}}
        self.assertEqual(preflight.verdict(report())[0], "READY")
        self.assertEqual(preflight.verdict(report(source="WARN"))[0], "BLOCKED")
        self.assertEqual(preflight.verdict(report(binding="MISMATCH"))[0], "INTEGRITY_ONLY")
        self.assertEqual(preflight.verdict(report(binding="INCOMPLETE", missing=["evaluated_arm:/x"]))[0], "INTEGRITY_ONLY")
        self.assertEqual(preflight.verdict(report(full="NOT_RUN"))[0], "PENDING_FULL_AUDIT")
        self.assertEqual(preflight.verdict(report(full="FAIL"))[0], "BLOCKED")
        self.assertEqual(preflight.verdict(report(integrity="FAIL"))[0], "BLOCKED")
        verdict, reasons, _ = preflight.verdict(report(relationship=None))
        self.assertEqual(verdict, "INTEGRITY_ONLY")
        self.assertTrue(any("declare relationship" in reason for reason in reasons))

    def test_reports_stay_outside_the_checkout_and_merge(self):
        with self.assertRaises(preflight.PreflightError):
            preflight.write_outputs([], PROJECT / "preflight-reports")
        out = temporary(self)
        base = preflight.preflight_dataset("ogbn-arxiv", EXAMPLE, "run", full_audit="never")
        other = dict(base, dataset="Peptides-func")
        preflight.write_outputs([base], out, merge=True)
        preflight.write_outputs([other], out, merge=True)
        summary = json.loads((out / "summary.json").read_text())
        self.assertEqual(set(summary["datasets"]), {"ogbn-arxiv", "Peptides-func"})
        self.assertIn("Peptides-func", (out / "PREFLIGHT.md").read_text())


class CampaignBindingTests(unittest.TestCase):
    identities = [{"split_index": i, "source_sha256": "a" * 64, "config_sha256": "b" * 64,
                   "unit_sha256": f"{i:064x}", "dataset": "Roman-empire"} for i in range(10)]

    def test_binding_covers_every_pointer_the_campaign_audit_reads(self):
        from scripts.scientific_evidence_audit import BINDING_POINTERS, binding_coverage
        binding = campaign.build_binding(self.identities, relationship="exact_result_producing_run")
        self.assertEqual(binding_coverage(binding, campaign=True, dataset="Roman-empire", manuscript_root=PROJECT), [])
        self.assertEqual({r["json_pointer"] for r in binding["results"]}, BINDING_POINTERS)
        self.assertTrue(all(r["artifact"] == campaign.ARTIFACT for r in binding["results"]))
        self.assertEqual(binding["aggregation"], "paired_official_split_t95")
        self.assertEqual(binding["run_identities"], {"dataset": "Roman-empire", "splits": self.identities})
        registry = json.loads((PROJECT / "paper" / "reported_results.json").read_text())
        table1 = registry["main_table_1_policy_and_deployment_ledger"]["Roman-empire"]
        mean = next(r for r in binding["results"] if r["json_pointer"] == "/paired_qD_minus_qC_loss/mean")
        self.assertEqual(mean["reported_value"], table1["loss_effect_nats"])
        self.assertGreater(mean["absolute_tolerance"], 0)

    def test_declarations_and_layout_are_required(self):
        with self.assertRaises(campaign.CampaignBindingError):
            campaign.build_binding(self.identities, relationship="assumed")
        with self.assertRaises(campaign.CampaignBindingError):
            campaign.build_binding(self.identities, relationship="post_result_revision", identifier=" ")
        with self.assertRaises(campaign.CampaignBindingError):
            campaign.split_paths(temporary(self))
        with self.assertRaises(campaign.CampaignBindingError):
            campaign.output_path(PROJECT / "paper" / "bindings" / "new.json", temporary(self))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.root = temporary(self)
        self.plan_path = self.root / "plan.json"
        plan = pipeline.template()
        plan.update(deposits_dir=str(self.root / "deposits"), license="cc-by-4.0",
                    relationship_to_manuscript="exact_result_producing_run",
                    datasets={"ogbn-arxiv": {"run_root": str(EXAMPLE)}})
        self.plan_path.write_text(json.dumps(plan))
        self.plan = pipeline.load_plan(self.plan_path)

    def args(self, **values):
        defaults = dict(full_audit="never", allow_blocked=False, allow_integrity_only=False, allow_pending_audit=False,
                        sandbox=False, confirm=False, force=False, set=None, replace=False)
        return argparse.Namespace(**{**defaults, **values})

    def test_template_and_plan_validation(self):
        template = pipeline.template()
        self.assertEqual(set(template["datasets"]), set(pipeline.DATASETS))
        self.assertEqual(set(template["record_sets"]), set(pipeline.RECORD_SETS))
        for change, message in (({"schema": "other"}, "schema"), ({"deposits_dir": "relative/dir"}, "absolute"),
                                ({"deposits_dir": str(PROJECT / "deposits")}, "outside"),
                                ({"relationship_to_manuscript": "assumed"}, "relationship"),
                                ({"datasets": {"Cora": {"run_root": "/x"}}}, "dataset")):
            with self.subTest(change=change):
                bad = self.root / "bad.json"
                bad.write_text(json.dumps({**json.loads(self.plan_path.read_text()), **change}))
                with self.assertRaisesRegex(pipeline.PipelineError, message):
                    pipeline.load_plan(bad)

    def test_datasets_that_are_not_ready_are_skipped_unless_named(self):
        self.assertEqual(pipeline.step_pack(self.plan, ["ogbn-arxiv"], self.args()),
                         [{"dataset": "ogbn-arxiv", "status": "SKIPPED", "detail": "no preflight yet"}])
        with self.assertRaisesRegex(pipeline.PipelineError, "run the preflight first"):
            pipeline.step_pack(self.plan, ["ogbn-arxiv"], self.args(dataset=["ogbn-arxiv"]))
        with mock.patch.dict(os.environ, {"ZENODO_TOKEN": "token"}):
            rows = pipeline.step_deposit(self.plan, ["ogbn-arxiv"], self.args(), publish=False)
            self.assertEqual(rows[0]["status"], "SKIPPED")
            self.assertIn("no archive", rows[0]["detail"])
            with self.assertRaisesRegex(pipeline.PipelineError, "pack the archive first"):
                pipeline.step_deposit(self.plan, ["ogbn-arxiv"], self.args(dataset=["ogbn-arxiv"]), publish=False)

    def test_datasets_without_a_run_root_wait_unless_named(self):
        plan = dict(self.plan, datasets={"ogbn-arxiv": {"run_root": str(EXAMPLE)}, "ogbn-proteins": {"run_root": None}})
        with contextlib.redirect_stderr(io.StringIO()) as note:
            self.assertEqual(pipeline.selected(plan, None, need_root=True), ["ogbn-arxiv"])
        self.assertIn("skip ogbn-proteins", note.getvalue())
        self.assertEqual(pipeline.selected(plan, None, need_root=False), ["ogbn-arxiv", "ogbn-proteins"])
        with self.assertRaisesRegex(pipeline.PipelineError, "existing absolute directory"):
            pipeline.selected(plan, ["ogbn-proteins"], need_root=True)
        with self.assertRaisesRegex(pipeline.PipelineError, "existing absolute directory"):
            pipeline.selected(dict(plan, datasets={"ogbn-arxiv": {"run_root": "relative/run"}}), None, need_root=True)
        with self.assertRaisesRegex(pipeline.PipelineError, "not in the plan"):
            pipeline.selected(plan, ["COCO-SP"], need_root=False)

    def test_preflight_pack_and_gates(self):
        rows = pipeline.step_preflight(self.plan, ["ogbn-arxiv"], self.args())
        self.assertEqual(rows, [{"dataset": "ogbn-arxiv", "status": "BLOCKED"}])
        self.assertTrue((self.root / "deposits" / "preflight" / "PREFLIGHT.md").is_file())
        self.assertEqual(pipeline.step_pack(self.plan, ["ogbn-arxiv"], self.args())[0]["status"], "SKIPPED")
        self.assertEqual(pipeline.step_pack(self.plan, ["ogbn-arxiv"], self.args(allow_blocked=True))[0]["status"], "PACKED")
        self.assertEqual(pipeline.step_pack(self.plan, ["ogbn-arxiv"], self.args(allow_blocked=True))[0]["status"],
                         "VERIFIED_EXISTING")
        entry = pipeline.current(self.plan, "ogbn-arxiv")
        self.assertEqual(entry["archive"]["identities"], entry["preflight"]["identities"])
        with self.assertRaisesRegex(pipeline.PipelineError, "only full-profile scientific"):
            pipeline._gate("ogbn-arxiv", entry["preflight"], entry["archive"], self.args(), publishing=True)
        with self.assertRaisesRegex(pipeline.PipelineError, "only full-profile scientific"):
            pipeline._gate("ogbn-arxiv", entry["preflight"], entry["archive"], self.args(allow_blocked=True), publishing=True)
        changed = dict(entry["archive"], identities=dict(entry["archive"]["identities"], unit_sha256="0" * 64))
        with self.assertRaisesRegex(pipeline.PipelineError, "different sealed identities"):
            pipeline._gate("ogbn-arxiv", entry["preflight"], changed, self.args(), publishing=False)
        history = json.loads((self.root / "deposits" / "evidence-state.json").read_text())["history"]
        self.assertEqual([h["step"] for h in history], ["preflight", "archive", "archive"])

    def test_sandbox_transport_and_production_refusal_with_recorded_services(self):
        pipeline.step_preflight(self.plan, ["ogbn-arxiv"], self.args())
        pipeline.step_pack(self.plan, ["ogbn-arxiv"], self.args(allow_blocked=True))
        archived = pipeline.current(self.plan, "ogbn-arxiv")["archive"]
        archive = Path(archived["path"])
        calls = []

        def fake_tool(argv, check=True, env=None):
            calls.append(argv)
            sandbox = "--sandbox" in argv
            receipt = archive.with_name(archive.name + (".sandbox.deposit.json" if sandbox else ".deposit.json"))
            published = "--publish" in argv
            doi = f"{'10.5072' if sandbox else '10.5281'}/zenodo.7"
            receipt.write_text(json.dumps({"schema": "cross-morse-zenodo-receipt-1",
                                           "service": "https://sandbox.zenodo.org/api" if sandbox else "https://zenodo.org/api",
                                           "filename": archive.name, "sha256": archived["sha256"], "bytes": archived["bytes"],
                                           "status": "published" if published else "draft", "deposition_id": 7,
                                           "reserved_doi": doi, "doi": doi if published else None,
                                           "published": published}))
            return subprocess.CompletedProcess(argv, 0, "", "")

        with mock.patch.dict(os.environ, {"ZENODO_TOKEN": "token", "ZENODO_SANDBOX_TOKEN": "sandbox-token"}), \
                mock.patch.object(pipeline, "tool", fake_tool):
            with self.assertRaisesRegex(pipeline.PipelineError, "--confirm"):
                pipeline.step_deposit(self.plan, ["ogbn-arxiv"], self.args(allow_blocked=True), publish=True)
            draft = pipeline.step_deposit(self.plan, ["ogbn-arxiv"], self.args(), publish=False)
            self.assertEqual(draft[0]["status"], "draft")
            with self.assertRaisesRegex(pipeline.PipelineError, "only full-profile scientific"):
                pipeline.step_deposit(self.plan, ["ogbn-arxiv"], self.args(confirm=True, dataset=["ogbn-arxiv"]), publish=True)
            skipped = pipeline.step_deposit(self.plan, ["ogbn-arxiv"], self.args(confirm=True), publish=True)
            self.assertEqual(skipped[0]["status"], "SKIPPED")
            self.assertIn("only full-profile scientific", skipped[0]["detail"])
            self.assertFalse(any("--publish" in call for call in calls))
            with self.assertRaisesRegex(pipeline.PipelineError, "only full-profile scientific"):
                pipeline.step_deposit(self.plan, ["ogbn-arxiv"], self.args(confirm=True, allow_blocked=True,
                                      dataset=["ogbn-arxiv"]), publish=True)
            done = pipeline.step_deposit(self.plan, ["ogbn-arxiv"], self.args(confirm=True, sandbox=True), publish=True)
        self.assertEqual(done[0]["detail"], "10.5072/zenodo.7")
        self.assertIn("--license", calls[0])
        self.assertIn("cc-by-4.0", calls[0])
        self.assertIn("--publish", calls[-1])
        self.assertTrue(pipeline.current(self.plan, "ogbn-arxiv")["sandbox_deposit"]["published"])
        self.assertFalse(pipeline.current(self.plan, "ogbn-arxiv")["deposit"]["published"])
        with mock.patch.object(pipeline, "tool") as tools:
            self.assertEqual(pipeline.step_register(self.plan, ["ogbn-arxiv"], self.args())[0]["status"], "SKIPPED")
            with self.assertRaisesRegex(pipeline.PipelineError, "no published production"):
                pipeline.step_register(self.plan, ["ogbn-arxiv"], self.args(dataset=["ogbn-arxiv"]))
            tools.assert_not_called()

    def test_records_step_rejects_unverified_existing_sets_and_preserves_failed_replacements(self):
        root = temporary(self)
        (root / "records" / "factorials").mkdir(parents=True)
        previous = root / "records/factorials/keep.txt"
        previous.write_text("existing measurement bytes")
        shutil.copyfile(PROJECT / "records/specs.json", root / "records/specs.json")
        mapping = root / "factorials.mapping.json"
        mapping.write_text(json.dumps({"schema": "cross-morse-record-mapping-1", "set": "factorials"}))
        plan = dict(self.plan, record_sets={"factorials": {"exporter": None, "mapping": str(mapping)}})
        with mock.patch.object(pipeline, "ROOT", root), mock.patch.object(pipeline, "tool") as tool:
            with self.assertRaises(ValueError):
                pipeline.step_records(plan, self.args())
            tool.assert_not_called()
            tool.side_effect = pipeline.PipelineError("exporter failed")
            with self.assertRaisesRegex(pipeline.PipelineError, "exporter failed"):
                pipeline.step_records(plan, self.args(replace=True, set=["factorials"]))
            self.assertEqual(previous.read_text(), "existing measurement bytes")
            with self.assertRaisesRegex(pipeline.PipelineError, "not both"):
                pipeline.step_records(dict(plan, record_sets={"inserted_site": {"exporter": ["x"], "mapping": "/m"}}), self.args())

    def test_status_lists_every_dataset_and_record_set(self):
        text = pipeline.status_markdown(pipeline.status_rows(self.plan, pipeline.load_state(self.plan)))
        for name in (*pipeline.DATASETS, *pipeline.RECORD_SETS):
            self.assertIn(name, text)


class RecordMappingTests(unittest.TestCase):
    def setUp(self):
        self.root = temporary(self)
        self.records = self.root / "records"
        self.records.mkdir()
        shutil.copyfile(PROJECT / "records" / "specs.json", self.records / "specs.json")
        self.raw = self.root / "outputs"
        self.raw.mkdir()

    def mapping(self, **change):
        datasets = ["ogbn-arxiv", "ogbn-proteins", "Roman-empire", "PascalVOC-SP", "COCO-SP", "Peptides-func"]
        for i, dataset in enumerate(datasets):
            (self.raw / f"{i}.json").write_text(json.dumps({"timing": {"t_nom": 0.5 + i / 8, "a": 3.0 + i, "b": 1.5}}))
        rows = [{"keys": {"dataset": d}, "quantity": "t_nom_seconds", "unit": "s", "source_file": f"raw/{i}.json",
                 "locator": "json:/timing/t_nom"} for i, d in enumerate(datasets)]
        rows[0] = {"keys": {"dataset": datasets[0]}, "quantity": "t_nom_seconds", "unit": "s",
                   "derive": {"op": "ratio", "inputs": [{"source_file": "raw/0.json", "locator": "json:/timing/a"},
                                                        {"source_file": "raw/0.json", "locator": "json:/timing/b"}]}}
        value = {"schema": record_mapping.MAPPING_SCHEMA, "set": "nominal_time",
                 "provenance": {"experiment": "E17 test", "produced_by": "unit test", "hardware": "test host",
                                "period": "2026-11-03/2026-11-03", "protocol": "test protocol"},
                 "raw_files": [{"from": str(self.raw / "{i}.json"), "to": "raw/{i}.json",
                                "for_each": {"i": [str(i) for i in range(6)]}}],
                 "rows": rows}
        value.update(change)
        path = self.root / "mapping.json"
        path.write_text(json.dumps(value))
        return path

    def test_templates_cover_the_audit_contract_for_every_set(self):
        from cm_campaigns.contract import RECORD_SETS
        from scripts.scientific_evidence_audit import validate_record_coverage
        specs = json.loads((PROJECT / "records" / "specs.json").read_text())
        for name in RECORD_SETS:
            with self.subTest(set=name):
                template = record_mapping.template(name)
                rows = [{**{k: str(v) for k, v in row["keys"].items()}, "quantity": row["quantity"]}
                        for entry in template["rows"] for row in record_mapping._expand(entry)]
                validate_record_coverage(name, rows, specs["sets"][name], RECORD_SETS[name])
                self.assertTrue(all(row["source_file"] is None and row["locator"] is None for row in template["rows"]))

    def test_build_traces_every_value_and_records_derivations(self):
        from scripts.records import cmd_check
        target = record_mapping.build(self.mapping(), records_root=self.records)
        self.assertEqual(cmd_check(self.records, self.records / "specs.json"), 0)
        rows = (target / "summary.csv").read_text().splitlines()
        self.assertEqual(rows[0], "dataset,quantity,value,unit,source_file,locator")
        self.assertIn("ogbn-arxiv,t_nom_seconds,2,s,raw/derived/row_00000.json,json:/value", rows)
        self.assertIn("ogbn-proteins,t_nom_seconds,0.625,s,raw/1.json,json:/timing/t_nom", rows)
        derived = json.loads((target / "raw" / "derived" / "row_00000.json").read_text())
        self.assertEqual(derived["op"], "ratio")
        self.assertEqual(len(derived["inputs"]), 2)
        self.assertTrue(all(len(item["sha256"]) == 64 for item in derived["inputs"]))
        public = json.loads((target / "code" / "MAPPING.json").read_text())
        self.assertNotIn(str(self.raw), json.dumps(public))
        provenance = json.loads((target / "provenance.json").read_text())
        self.assertIs(provenance["synthetic"], False)
        with self.assertRaisesRegex(record_mapping.MappingError, "exists"):
            record_mapping.build(self.mapping(), records_root=self.records)
        record_mapping.build(self.mapping(), records_root=self.records, replace=True)

    def test_refusals(self):
        base = json.loads(self.mapping().read_text())
        cases = {
            "locator missing": dict(rows=[dict(base["rows"][1], locator=None)] + base["rows"][2:]),
            "duplicate row": dict(rows=base["rows"] + [base["rows"][1]]),
            "no for_each values": dict(raw_files=[{"from": str(self.raw / "{i}.json"), "to": "raw/{i}.json"}]),
            "coverage required": dict(rows=base["rows"][:3]),
            "provenance lacks": dict(provenance={"experiment": "x"}),
            "measured sets only": dict(provenance=dict(base["provenance"], synthetic=True)),
            "quantity is not defined": dict(rows=[dict(base["rows"][1], quantity="seconds")] + base["rows"][2:]),
        }
        for message, change in cases.items():
            with self.subTest(case=message):
                with self.assertRaisesRegex(record_mapping.MappingError, message):
                    record_mapping.build(self.mapping(**change), records_root=self.records)
        record_mapping.build(self.mapping(rows=base["rows"][:3]), records_root=self.records, allow_partial=True)
        complete = self.mapping()
        # A fictional allocation id, split so the release audit does not flag this source.
        (self.raw / "5.json").write_text(json.dumps({"timing": {"t_nom": 1.0}, "account": "nai" + "ss2099-1-1"}))
        with self.assertRaisesRegex(record_mapping.MappingError, "private infrastructure"):
            record_mapping.build(complete, records_root=self.records, replace=True)
        record_mapping.build(complete, records_root=self.records, replace=True, allow_private=True)

    def test_export_records_cli_writes_templates_outside_the_checkout_only(self):
        output = self.root / "factorials.mapping.json"
        result = subprocess.run([sys.executable, "-B", "scripts/export_records.py", "mapping-template", "--set", "factorials",
                                 "--output", str(output)], cwd=PROJECT, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(output.read_text())["set"], "factorials")
        refused = subprocess.run([sys.executable, "-B", "scripts/export_records.py", "mapping-template", "--set", "factorials",
                                  "--output", str(PROJECT / "factorials.mapping.json")], cwd=PROJECT, capture_output=True,
                                 text=True, timeout=120)
        self.assertEqual(refused.returncode, 2)
        self.assertFalse((PROJECT / "factorials.mapping.json").exists())


class ManuscriptSourceTests(unittest.TestCase):
    def test_numbers_labels_and_presence(self):
        found = sources.numbers("effect \u22120.0045, 1,138,216 parameters and 73.15 points; seeds 1, 2, 4")
        for value in (-0.0045, 1138216.0, 73.15, 4.0):
            self.assertIn(value, found)
        self.assertTrue(sources.present(-0.0045, found))
        self.assertFalse(sources.present(0.0046, found))
        self.assertEqual(sources.label_tokens("Table S16 and Figure S17d"), ["S16", "S17"])
        self.assertEqual(sources.label_tokens("Tables S46 and S46b"), ["S46", "S46b"])
        self.assertEqual(sources.label_tokens("Table 1"), ["Table 1"])

    def test_regressions_flag_changed_groups_only(self):
        before = {"groups": {"a": {"coverage": 0.98, "label_found": True, "label": "Table S1", "pages": [1]},
                             "b": {"coverage": 0.90, "label_found": True, "label": "Table S2", "pages": [2]}}}
        after = {"groups": {"a": {"coverage": 0.70, "label_found": True, "label": "Table S1", "pages": [1]},
                            "b": {"coverage": 0.895, "label_found": False, "label": "Table S2", "pages": [2]}}}
        flags = sources.regressions(after, before)
        self.assertEqual(len(flags), 2)
        self.assertIn("a: coverage", flags[0])
        self.assertIn("no longer on pages", flags[1])
        self.assertEqual(sources.regressions(after, None), [])

    def test_register_requires_a_current_completed_author_review(self):
        root = temporary(self)
        for name in ("sources.json", "reported_results.json", "registered_constants.json"):
            shutil.copyfile(PROJECT / "paper" / name, root / name)
        original = json.loads((root / "sources.json").read_text())
        pdfs = {role: root / f"{role}.pdf" for role in original["documents"]}
        for role, path in pdfs.items():
            path.write_bytes(b"%PDF-test-" + role.encode())

        class FakeExtractor:
            kind = "fixture"
            def __init__(self, path):
                self.path = path
            def pages(self):
                return original["documents"][self.path.stem]["pages"]
            def text(self, page):
                return "Table 1 S5 1,138,216 3.1 0.05"

        with mock.patch.object(sources, "SOURCES", root / "sources.json"), \
                mock.patch.object(sources, "REPORTED", root / "reported_results.json"), \
                mock.patch.object(sources, "CONSTANTS", root / "registered_constants.json"), \
                mock.patch.object(sources, "CHECK", root / "transcription_check.json"), \
                mock.patch.object(sources, "Extractor", FakeExtractor):
            result = sources.report(pdfs["main"], pdfs["supplementary"])
            notes = sources.review_template(result)
            with self.assertRaisesRegex(sources.SourceError, "reviewed_by"):
                sources.register(result, pdf_paths=pdfs, review_notes=notes)
            notes["reviewed_by"] = "test fixture author"
            for group in notes["groups"].values():
                group.update(review_status="reviewed", note="Fixture contextual check including units and derived values.")
            sources.register(result, pdf_paths=pdfs, review_notes=notes)
            written = json.loads((root / "sources.json").read_text())
            self.assertEqual(written["documents"]["main"]["sha256"], sources.identity(pdfs["main"])["sha256"])
            self.assertEqual(written["table_groups"], original["table_groups"])
            stored = json.loads((root / "transcription_check.json").read_text())
            self.assertEqual(stored["author_review"]["groups"], notes["groups"])
            self.assertIs(stored["spot_check"]["verifies_transcription"], False)
            with self.assertRaisesRegex(sources.SourceError, "not a review record"):
                sources.register(result, pdf_paths=pdfs, review_notes=notes, accept_changes=True)

    def test_cli_refuses_pdfs_in_the_checkout_and_non_pdfs(self):
        inside = subprocess.run([sys.executable, "-B", "scripts/register_manuscript_sources.py", "--manuscript-pdf",
                                 str(PROJECT / "README.md"), "--supplementary-pdf", str(PROJECT / "README.md")],
                                cwd=PROJECT, capture_output=True, text=True, timeout=120)
        self.assertEqual(inside.returncode, 2)
        self.assertIn("outside the repository", inside.stderr)
        text = temporary(self) / "not.pdf"
        text.write_text("plain text")
        outside = subprocess.run([sys.executable, "-B", "scripts/register_manuscript_sources.py", "--manuscript-pdf",
                                  str(text), "--supplementary-pdf", str(text)], cwd=PROJECT, capture_output=True, text=True,
                                 timeout=120)
        self.assertEqual(outside.returncode, 2)
        self.assertIn("not a PDF", outside.stderr)


class ReadinessTests(unittest.TestCase):
    def test_checklist_names_every_item_and_fails_without_evidence(self):
        out = temporary(self) / "readiness.json"
        review = out.parent / "review.json"
        review.write_text(json.dumps({"verdict": "EVIDENCE COMPLETE"}))
        with contextlib.redirect_stdout(io.StringIO()):
            code = readiness.main(["--output", str(out), "--review-report", str(review)])
        items = {i["item"]: i for i in json.loads(out.read_text())["items"]}
        self.assertEqual(set(items), {"evidence", "bindings", "records", "campaign", "confirmation_family",
                                      "pdfs", "transcription", "baselines", "decisions", "generated", "runtime", "review"})
        self.assertEqual(code, 1)
        self.assertFalse(readiness.is_ready(list(items.values())))
        self.assertEqual(items["confirmation_family"]["status"], "TODO")
        self.assertIn("--confirmation-campaign", items["confirmation_family"]["detail"])
        self.assertEqual(items["generated"]["status"], "PASS")
        self.assertEqual(items["runtime"]["status"], "PASS")
        self.assertEqual(items["review"]["status"], "FAIL")
        self.assertIn(items["evidence"]["status"], {"PASS", "TODO"})
        self.assertTrue(all(i["next"] for i in items.values() if i["status"] != "PASS"))
        self.assertTrue(out.with_suffix(".md").is_file())


class DepositedAbsentTests(unittest.TestCase):
    """Offline descriptors of synthetic fixtures never assert public DOI availability."""

    def setUp(self):
        from scripts.archive_run import pack
        from scripts.ingest_run import ingest
        self.root = temporary(self)
        self.runs = self.root / "runs"
        self.runs.mkdir()
        self.index = self.runs / "index.json"
        self.out = self.root / "manifests"
        archive = self.root / "arxiv.tar.gz"
        pack(EXAMPLE, archive, campaign=False)
        self.held = ingest(archive, doi="10.5281/zenodo.7", runs_root=self.runs, index_path=self.index, replace=False)
        self.source = self.held["run_source_sha256"][0]
        code, text = self.manifests("--with-indices")
        self.assertEqual(code, 0, text)
        self.committed = self.out / "official" / "arxiv.json"
        shutil.rmtree(self.runs / "arxiv")

    def write(self, entries):
        self.index.write_text(json.dumps({"schema": "cross-morse-runs-index-1", "runs": entries}))

    def manifests(self, *extra):
        from scripts.run_manifests import main
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(["runs", "--index", str(self.index), "--label", "official", "--output-root", str(self.out), *extra])
        return code, stdout.getvalue() + stderr.getvalue()

    def chains(self, **flags):
        from scripts.ingest_run import check as check_chains
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return check_chains(self.runs, self.index, **flags)

    def test_held_runs_need_a_production_doi_and_a_matching_committed_manifest(self):
        code, text = self.manifests("--check", "--allow-empty", "--allow-deposited-absent")
        self.assertEqual(code, 0, text)
        self.assertIn("registration metadata agrees", text)
        self.assertIn("NOT VERIFIED", text)
        code, text = self.manifests("--check", "--allow-empty")
        self.assertEqual(code, 2)
        self.assertIn("no completed run or campaign", text)
        self.assertEqual(self.chains(allow_deposited_absent=True), 0)
        self.assertEqual(self.chains(), 1)
        self.write([dict(self.held, deposit_doi="10.5072/zenodo.7")])
        self.assertEqual(self.manifests("--check", "--allow-deposited-absent")[0], 2)
        self.assertEqual(self.chains(allow_deposited_absent=True), 1)
        self.write([dict(self.held, run_source_sha256=["0" * 64])])
        code, text = self.manifests("--check", "--allow-deposited-absent")
        self.assertEqual(code, 1)
        self.assertIn("run source identities", text)
        self.write([dict(self.held, dataset="ogbn-proteins", stem="proteins", path="proteins")])
        self.assertIn("manifest is missing", self.manifests("--check", "--allow-deposited-absent")[1])
        self.write([self.held])
        (self.runs / "arxiv").mkdir()
        self.assertEqual(self.manifests("--check", "--allow-deposited-absent")[0], 2)
        self.assertEqual(self.chains(allow_deposited_absent=True), 1)

    def test_present_runs_are_still_extracted_and_checked_beside_held_ones(self):
        from scripts.archive_run import pack
        from scripts.ingest_run import ingest
        archive = self.root / "peptides.tar.gz"
        pack(PROJECT / "examples/runs/peptides_func", archive, campaign=False)
        ingest(archive, doi="10.5281/zenodo.8", runs_root=self.runs, index_path=self.index, replace=False)
        self.assertEqual(self.manifests("--with-indices", "--allow-deposited-absent")[0], 0)
        self.assertTrue((self.out / "official" / "peptides_func.json").is_file())
        self.assertTrue((self.out / "official" / "peptides_func_splits.npz").is_file())
        self.assertEqual(self.manifests("--check", "--allow-deposited-absent")[0], 0)
        self.assertEqual(self.chains(allow_deposited_absent=True), 0)
        summary = self.runs / "peptides_func" / "final" / "results_summary.json"
        summary.write_text(summary.read_text() + " ")
        self.assertEqual(self.chains(allow_deposited_absent=True), 1)

    def test_flag_is_limited_to_the_official_index(self):
        from scripts.run_manifests import main
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["runs", "--runs-root", str(self.runs), "--label", "examples", "--allow-deposited-absent"])


@unittest.skipUnless(shutil.which("make"), "GNU Make is required")
class EvidenceMakeRoutingTests(unittest.TestCase):
    def setUp(self):
        self.root = temporary(self)
        self.project = self.root / "checkout with spaces"
        self.project.mkdir()
        shutil.copyfile(PROJECT / "Makefile", self.project / "Makefile")
        self.capture = self.root / "calls.jsonl"
        self.interpreter = self.root / "recording python"
        self.interpreter.write_text(f"#!{sys.executable}\nimport json, os, sys\n"
                                    "with open(os.environ['CM_MAKE_CAPTURE'], 'a') as stream:\n"
                                    "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n")
        self.interpreter.chmod(0o755)

    def invoke(self, target, *, succeeds=True, **variables):
        self.capture.unlink(missing_ok=True)
        result = subprocess.run(["make", "--no-print-directory", target, f"PYTHON={self.interpreter}",
                                 *(f"{k}={v}" for k, v in variables.items())], cwd=self.project, text=True,
                                capture_output=True, timeout=30, env=dict(os.environ, CM_MAKE_CAPTURE=str(self.capture)))
        self.assertEqual(result.returncode == 0, succeeds, result.stdout + result.stderr)
        return [json.loads(line) for line in self.capture.read_text().splitlines()] if self.capture.exists() else []

    def test_plan_is_required_and_steps_route_to_the_pipeline(self):
        self.invoke("evidence-status", succeeds=False)
        calls = self.invoke("evidence-preflight", EVIDENCE_PLAN="/p/plan.json", EVIDENCE_DATASET="COCO-SP")
        self.assertEqual(calls, [["-B", "scripts/evidence_pipeline.py", "preflight", "--plan", "/p/plan.json",
                                  "--dataset", "COCO-SP", "--full-audit", "auto"]])
        for target, step in (("evidence-pack", "pack"), ("evidence-deposit", "deposit"), ("evidence-register", "register"),
                             ("evidence-bind", "bind"), ("evidence-records", "records"), ("evidence-status", "status")):
            with self.subTest(target=target):
                self.assertEqual(self.invoke(target, EVIDENCE_PLAN="/p/plan.json")[0][1:3], ["scripts/evidence_pipeline.py", step])

    def test_publication_needs_an_explicit_one_and_flags_pass_only_when_one(self):
        self.invoke("evidence-publish", succeeds=False, EVIDENCE_PLAN="/p/plan.json")
        self.invoke("evidence-publish", succeeds=False, EVIDENCE_PLAN="/p/plan.json", PUBLISH="yes")
        calls = self.invoke("evidence-publish", EVIDENCE_PLAN="/p/plan.json", PUBLISH=1, ALLOW_INTEGRITY_ONLY=1,
                            ALLOW_BLOCKED=0)
        self.assertIn("--confirm", calls[0])
        self.assertIn("--allow-integrity-only", calls[0])
        self.assertNotIn("--allow-blocked", calls[0])
        self.assertNotIn("--sandbox", calls[0])
        self.assertEqual(self.invoke("evidence-publish", succeeds=False, EVIDENCE_PLAN="/p/plan.json", PUBLISH=1,
                                     ALLOW_BLOCKED=1), [])
        calls = self.invoke("evidence-publish", EVIDENCE_PLAN="/p/plan.json", PUBLISH=1, SANDBOX=1)
        self.assertIn("--sandbox", calls[0])
        self.assertNotIn("--allow-blocked", calls[0])

    def test_readiness_mapping_and_sources_routing(self):
        self.assertEqual(self.invoke("readiness"), [["-B", "scripts/readiness.py", "--index", "runs/index.json", "--records", "records",
                                                   "--manifests-root", "manifests"]])
        calls = self.invoke("readiness", MANUSCRIPT_PDF="/m.pdf", REVIEW_REPORT="/r/review.json")
        self.assertEqual(calls[0], ["-B", "scripts/readiness.py", "--index", "runs/index.json", "--records", "records",
                                    "--manifests-root", "manifests", "--manuscript-pdf", "/m.pdf", "--review-report", "/r/review.json"])
        self.invoke("record-from-mapping", succeeds=False)
        calls = self.invoke("record-from-mapping", MAPPING="/m/set.mapping.json", REPLACE=1)
        self.assertEqual(calls[0], ["-B", "scripts/export_records.py", "from-mapping", "--mapping", "/m/set.mapping.json", "--replace"])
        self.assertEqual([c[1] for c in calls[1:]], ["scripts/availability.py", "scripts/claims_ledger.py",
                                                     "scripts/manuscript_availability.py"])
        self.invoke("manuscript-sources", succeeds=False, MANUSCRIPT_PDF="/m.pdf")
        self.invoke("register-manuscript-sources", succeeds=False, MANUSCRIPT_PDF="/m.pdf", SUPPLEMENTARY_PDF="/s.pdf")
        calls = self.invoke("register-manuscript-sources", MANUSCRIPT_PDF="/m.pdf", SUPPLEMENTARY_PDF="/s.pdf",
                            REVIEW_NOTES="/p/review notes.json")
        self.assertEqual(calls[0][-3:], ["--register", "--review-notes", "/p/review notes.json"])
        calls = self.invoke("manuscript-sources", MANUSCRIPT_PDF="/m.pdf", SUPPLEMENTARY_PDF="/s.pdf",
                            REVIEW_TEMPLATE="/p/review template.json")
        self.assertEqual(calls[0][-2:], ["--review-template", "/p/review template.json"])
        self.invoke("campaign-binding", succeeds=False, RUN_ROOT="/c")
        calls = self.invoke("campaign-binding", RUN_ROOT="/c", RELATIONSHIP="exact_result_producing_run", OUTPUT="/o.json")
        self.assertEqual(calls[0][1:3], ["scripts/emit_campaign_binding.py", "/c"])


class HpcScriptTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("bash"), "bash is required")
    def test_pack_script_requires_its_environment(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("CM_")}
        result = subprocess.run(["bash", str(PROJECT / "hpc" / "pack_evidence_arrhenius.sh")], cwd=PROJECT, env=env,
                                capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CM_SOURCE_ROOT", result.stderr)


if __name__ == "__main__":
    unittest.main()
