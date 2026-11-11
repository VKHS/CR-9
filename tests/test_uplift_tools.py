"""Reviewer verdicts, registration prose and protocol diagnostics; no network or PyTorch."""
from __future__ import annotations

import csv
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

from scripts import check_decision_consistency as decisions  # noqa: E402
from scripts import check_measurement_degeneracy as degeneracy  # noqa: E402
from scripts import claims_ledger, manuscript_availability, review  # noqa: E402
from scripts.availability import BUILT_DISPLAYS, DATASETS  # noqa: E402


class ClaimsLedgerTests(unittest.TestCase):
    def test_run_derived_set_matches_the_availability_display_list(self):
        # Table-column support agrees; figure summaries have separately bounded scope.
        self.assertIn("Tables 1–2", BUILT_DISPLAYS)
        self.assertIn("S13–S15", BUILT_DISPLAYS)
        for token in claims_ledger.RUN_DERIVED - {"1", "2", "S14"}:
            self.assertIn(token, BUILT_DISPLAYS, token)
        self.assertIn("partial internal-method map", BUILT_DISPLAYS)
        self.assertIn("entry-loss depth diagnostic", BUILT_DISPLAYS)
        self.assertNotIn("S14b", BUILT_DISPLAYS)
        self.assertEqual(claims_ledger.table_tokens("Tables S46 and S46b"), {"S46", "S46b"})
        self.assertEqual(claims_ledger.table_tokens("Table 1"), {"1"})
        self.assertEqual(claims_ledger.table_tokens("Table S16 and Figure S17d"), {"S16"})

    def test_ledger_covers_every_registry_group_and_committed_files_are_current(self):
        ledger = claims_ledger.build(PROJECT)
        sources = json.loads((PROJECT / "paper" / "sources.json").read_text())
        self.assertEqual(len(ledger["rows"]), len(sources["table_groups"]))
        self.assertEqual(claims_ledger.check(ledger, ledger_path=claims_ledger.LEDGER, badge_path=claims_ledger.BADGE), [])
        badge = claims_ledger.badge(ledger)
        self.assertEqual(badge["schemaVersion"], 1)
        # This release publishes the measured outputs as assets of the repository's release
        # (release_scope.json result_distribution); a registration release reports archive coverage.
        self.assertEqual(ledger["result_distribution"], "repository_release")
        self.assertIn("archives", claims_ledger.badge(dict(ledger, result_distribution="registered_archives"))["message"])
        text = claims_ledger.render(ledger)
        self.assertIn("| Table S15b (SI) |", text)
        self.assertIn("make paper", text)

    def test_status_reflects_registered_archives_and_record_sets(self):
        with tempfile.TemporaryDirectory(prefix="cm-claims-") as tmp:
            root = Path(tmp)
            for rel in ("paper/sources.json", "records/specs.json", "release/release_scope.json", "records/index.json",
                        "examples/records/index.json", "cm_experiments/__init__.py", "cm_campaigns/__init__.py",
                        "scripts/run_upstream_baseline.py", "scripts/export_records.py"):
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(PROJECT / rel, root / rel)
            # A registration release: archives registered in runs/index.json carry the results.
            scope_path = root / "release/release_scope.json"
            scope = json.loads(scope_path.read_text())
            scope["result_distribution"] = "registered_archives"
            scope_path.write_text(json.dumps(scope))
            (root / "cross_morse").mkdir()
            (root / "runs").mkdir()
            entry = {"stem": "arxiv", "dataset": "ogbn-arxiv", "path": "arxiv", "kind": "run", "profile": "full",
                     "scientific_run": True, "archive_name": "a.tar.gz", "archive_sha256": "0" * 64, "archive_bytes": 1,
                     "deposit_doi": "10.5281/zenodo.1", "verified": True}
            (root / "runs" / "index.json").write_text(json.dumps({"schema": "cross-morse-runs-index-1", "runs": [entry]}))
            ledger = claims_ledger.build(root)
            self.assertEqual(ledger["official_archives"], 1)
            self.assertEqual(ledger["dois"]["ogbn-arxiv"], "10.5281/zenodo.1")
            table1 = next(r for r in ledger["rows"] if r["label"].startswith("Table 1"))
            self.assertIn("Registered archives: ogbn-arxiv; DOI references: 1", table1["status"][0])
            self.assertFalse(ledger["registered_coverage_complete"])
            self.assertFalse(ledger["scientific_evidence_verified"])
            self.assertNotIn("ingested and verified", claims_ledger.render(ledger))


def degenerate_rows():
    plants = {"A": 4.0, "B": 4.0, "C": 2.0, "D": 8.0}
    rows = []
    for d, serial in plants.items():
        rows.append({"dataset": d, "method": "best fixed sequence (qC)", "serial_cost": serial, "wall_time": serial,
                     "gpu_hours": 10.0, "sd": 0.20, "external": False, "plant": True})
        rows.append({"dataset": d, "method": "X", "serial_cost": serial * 1.4, "wall_time": serial * 1.4,
                     "gpu_hours": 14.0, "sd": 0.23, "external": True, "plant": False})
        rows.append({"dataset": d, "method": "Y", "serial_cost": serial * 1.25, "wall_time": serial * 1.25,
                     "gpu_hours": 12.5, "sd": 0.23, "external": True, "plant": False})
    return rows


class DegeneracyGuardTests(unittest.TestCase):
    def test_constant_multipliers_shared_sds_and_wall_equal_serial_are_flagged(self):
        report = degeneracy.analyse(degenerate_rows())
        self.assertEqual(report["status"], "REVIEW_REQUIRED")
        self.assertFalse(report["measurement_origin_verified"])
        tests = {f["test"] for f in report["findings"]}
        self.assertEqual(tests, {"T1", "T2", "T3a", "T3b", "T4"})

    def test_nonrepeated_values_do_not_establish_measurement_origin(self):
        rows = degenerate_rows()
        jitter = [1.07, 0.93, 1.11, 0.96, 1.04, 0.91, 1.09, 0.97]
        k = 0
        for r in rows:
            if r["external"]:
                r["serial_cost"] = round(r["serial_cost"] * jitter[k % 8], 3)
                r["wall_time"] = round(r["serial_cost"] * 0.97, 3)
                r["gpu_hours"] = round(r["gpu_hours"] * jitter[(k + 3) % 8], 2)
                r["sd"] = round(0.20 + 0.03 * jitter[(k + 5) % 8] * (1 + k % 3), 3)
                k += 1
        report = degeneracy.analyse(rows)
        self.assertEqual(report["status"], "NO_PATTERNS_DETECTED")
        self.assertFalse(report["measurement_origin_verified"])

    def test_record_set_rows_are_pivoted_from_summary_csv(self):
        with tempfile.TemporaryDirectory(prefix="cm-degen-") as tmp:
            root = Path(tmp)
            (root / "baselines").mkdir()
            with (root / "baselines" / "summary.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["dataset", "method", "seed", "quantity", "value", "unit", "source_file", "locator"])
                for d in ("ogbn-arxiv", "ogbn-proteins"):
                    writer.writerow([d, "best fixed sequence (qC)", "", "serial_cost", "4.0", "nominal", "raw/t.log", "regex:x"])
                    writer.writerow([d, "AMP", "", "serial_cost", "4.5", "nominal", "raw/t.log", "regex:x"])
                    writer.writerow([d, "AMP", "", "wall_time", "4.5", "nominal", "raw/t.log", "regex:x"])
                    for seed, score in ((1, 73.0), (2, 73.2), (3, 73.1)):
                        writer.writerow([d, "AMP", seed, "score", str(score), "%", "raw/m.json", "json:/s"])
            rows = degeneracy.rows_from_records(root)
            amp = [r for r in rows if r["method"] == "AMP"]
            self.assertEqual(len(amp), 2)
            self.assertTrue(all(r["external"] for r in amp))
            self.assertAlmostEqual(amp[0]["sd"], 0.1, places=6)
            self.assertIsNone(degeneracy.rows_from_records(root / "absent"))

    def test_registry_mode_reads_table_s15b(self):
        report = degeneracy.analyse(degeneracy.rows_from_registry(PROJECT))
        self.assertIn(report["status"], {"NO_PATTERNS_DETECTED", "REVIEW_REQUIRED"})
        self.assertEqual(report["external_rows"], 14)


def registry_fixture(holm_peptides):
    table1 = {
        "ogbn-arxiv": {"provisional": "radius/4", "deployed_rule": "radius", "lifecycle_resolution": "radius"},
        "ogbn-products": {"provisional": "same phase", "deployed_rule": "qC returned", "lifecycle_resolution": "return_qC"},
        "PascalVOC-SP": {"provisional": "MPC-RH/8", "deployed_rule": "MPC-RH", "lifecycle_resolution": "MPC-RH"},
        "Peptides-func": {"provisional": "radius/4", "deployed_rule": "radius", "lifecycle_resolution": "return_qC",
                          "registered_confirmation_pass": False},
    }
    s14 = [
        {"dataset": "ogbn-arxiv", "kind": "radius minus one-step", "effect": -0.00286, "margin": 0.0015, "holm_P": 0.015},
        {"dataset": "PascalVOC-SP", "kind": "MPC-RH minus MPC-OL", "effect": -0.00902, "margin": 0.002, "holm_P": "< 0.001"},
        {"dataset": "Peptides-func", "kind": "radius minus one-step", "effect": -0.00172, "margin": 0.0015, "holm_P": holm_peptides},
    ]
    return {"main_table_1_policy_and_deployment_ledger": table1,
            "supplementary_table_S14_mechanism_contrasts_submitted_protocol": s14}


class DecisionConsistencyTests(unittest.TestCase):
    def test_rule_outcome_is_compared_with_the_deployed_policy(self):
        report = decisions.evaluate(registry_fixture(0.07))
        by = {r["dataset"]: r for r in report["rows"]}
        self.assertEqual(by["ogbn-arxiv"]["status"], "CONSISTENT")
        self.assertEqual(by["ogbn-products"]["status"], "CONSISTENT")
        self.assertEqual(by["PascalVOC-SP"]["status"], "CONSISTENT")
        self.assertEqual(by["Peptides-func"]["status"], "DISCREPANCY")
        self.assertEqual(report["discrepancies"], ["Peptides-func"])
        corrected = registry_fixture(0.04)
        corrected["main_table_1_policy_and_deployment_ledger"]["Peptides-func"].update(
            registered_confirmation_pass=True, lifecycle_resolution="radius")
        self.assertEqual(decisions.evaluate(corrected)["status"], "CONSISTENT")
        self.assertEqual(by["Peptides-func"]["registry_status"], "CONSISTENT")
        self.assertEqual(by["Peptides-func"]["manuscript_status"], "DISCREPANCY")

    def test_p_value_bounds_parse(self):
        self.assertEqual(decisions.parse_p("< 0.001"), 0.001)
        self.assertEqual(decisions.parse_p(0.07), 0.07)
        self.assertEqual(decisions.provisional_route("MPC-RH/8"), "MPC-RH")
        self.assertEqual(decisions.provisional_route("same phase"), "qC")

    def test_confirmation_boundary_matches_runtime_and_bounded_printing(self):
        contrast = {"kind": "radius minus one-step", "effect": -.003,
                    "margin": .0015, "holm_P": .05}
        def result(value):
            return decisions.contrast_result({**contrast, "holm_P": value},
                margin=.0015, alpha=.05, beneficial_sign=-1)
        # Numeric equality is rejected by the frozen runtime. A strict printed
        # bound at alpha establishes significance without inventing an exact P.
        self.assertFalse(result(.05)["confirmation_pass"])
        self.assertTrue(result("< 0.05")["confirmation_pass"])
        self.assertTrue(result(.049)["confirmation_pass"])
        self.assertIsNone(result("< 0.10")["confirmation_pass"])

    def test_committed_registry_evaluates(self):
        registry = json.loads((PROJECT / "paper" / "reported_results.json").read_text())
        report = decisions.evaluate(registry)
        self.assertEqual(len(report["rows"]), 7)
        self.assertIn(report["status"], {"CONSISTENT", "DISCREPANCY"})


def write_complete_reports(root: Path):
    audit = {"publication_ready": True, "status": "PASS", "blockers": [],
             "confirmation_family": {"status": "PASS"},
             "official_archives": {"validated": 7, "required": 7},
             "auxiliary_records": {"validated": 10, "required": 10},
             "manuscript_bindings": {"validated": 7, "required": 7},
             "manuscript_documents": {"pdf_bytes_verified": {"main": True, "supplementary": True}}}
    (root / "scientific-evidence-audit.json").write_text(json.dumps(audit))
    (root / "fetch-deposits.json").write_text(json.dumps({
        "public_verification": {"complete": True}, "failed": 0, "missing_doi": 0, "missing_datasets": [],
        "entries": [{"dataset": dataset, "status": "PRESENT", "detail": "verified"} for dataset in DATASETS]}))
    comparison = root / "compare" / "comparison-current"
    comparison.mkdir(parents=True, exist_ok=True)
    (comparison / "coverage.json").write_text(json.dumps({
        "status": "PASS", "compared_datasets": 7, "required_datasets": 7, "model_inventory_verified_datasets": 7}))
    (root / "decision-consistency.json").write_text(json.dumps({
        "status": "CONSISTENT", "rows": [{"dataset": dataset, "status": "CONSISTENT"} for dataset in DATASETS]}))


class ReviewAndStatementTests(unittest.TestCase):
    def test_dry_run_lists_offline_and_evidence_plans(self):
        with tempfile.TemporaryDirectory(prefix="cm-review-") as tmp:
            offline = review.plan(Path(tmp), offline=True, index=PROJECT / "runs" / "index.json", records=PROJECT / "records")
            evidence = review.plan(Path(tmp), offline=False, index=PROJECT / "runs" / "index.json", records=PROJECT / "records")
            self.assertTrue(any("reproduce_examples.py" in " ".join(s.argv) for s in offline))
            self.assertTrue(any("fetch_deposits.py" in " ".join(s.argv) for s in evidence))
            self.assertTrue(all((PROJECT / s.argv[0]).is_file() for s in offline + evidence))
            self.assertTrue(all(s.evidence for s in evidence if s.group in {"evidence", "displays", "comparison", "audit"}))

    def test_report_renders_verdicts_from_step_outcomes(self):
        with tempfile.TemporaryDirectory(prefix="cm-review-") as tmp:
            root = Path(tmp)
            steps = review.plan(root, offline=False, index=PROJECT / "runs" / "index.json", records=PROJECT / "records")
            for s in steps:
                s.status, s.code = ("PENDING" if s.evidence else "PASS"), (1 if s.evidence else 0)
            text = review.render(steps, root, offline=False, index=PROJECT / "runs" / "index.json", records=PROJECT / "records")
            self.assertIn("EVIDENCE INCOMPLETE", text)
            self.assertIn("| R1.3 |", text)
            steps[1].status, steps[1].code = "FAIL", 1
            self.assertIn("SOFTWARE FAILURE", review.render(steps, root, offline=False, index=PROJECT / "runs" / "index.json", records=PROJECT / "records"))
            write_complete_reports(root)
            for s in steps:
                s.status, s.code = "PASS", 0
            self.assertIn("EVIDENCE COMPLETE", review.render(steps, root, offline=False, index=PROJECT / "runs" / "index.json", records=PROJECT / "records"))
            self.assertIn("SOFTWARE-ONLY", review.render(steps, root, offline=True, index=PROJECT / "runs" / "index.json", records=PROJECT / "records"))

    def test_manuscript_paragraph_names_only_registered_deposits(self):
        scope = {"repository_url": "https://github.com/VKHS/CROSS-MORSE", "code_archive_doi": None}
        empty = {"scope": scope, "official": {}, "records": {"verified": [], "defined": ["baselines"]}}
        text = manuscript_availability.render(empty)
        self.assertNotIn("deposited at", text)
        self.assertIn("original benchmark archives can be supplied through the separate registration workflow", text)
        self.assertNotIn("0 of 7", text)
        full = {"scope": {**scope, "code_archive_doi": "10.5281/zenodo.9"},
                "official": {"ogbn-arxiv": {"deposit_doi": "10.5281/zenodo.1"}, "COCO-SP": {"deposit_doi": None}},
                "records": {"verified": ["baselines"], "defined": ["baselines"]}}
        text = manuscript_availability.render(full)
        self.assertIn("ogbn-arxiv: https://doi.org/10.5281/zenodo.1", text)
        self.assertIn("No deposit DOI is registered for the indexed archives of COCO-SP", text)
        self.assertIn("registered code-archive DOI reference is https://doi.org/10.5281/zenodo.9", text)
        self.assertIn("Registered record sets are baselines", text)
        self.assertNotIn("archived at", text)
        self.assertEqual(manuscript_availability.render(json.loads(json.dumps(empty))), manuscript_availability.render(empty))

    def test_committed_statement_is_current(self):
        expected = manuscript_availability.render(claims_ledger.registration_state(PROJECT))
        self.assertEqual((PROJECT / "paper" / "AVAILABILITY_STATEMENT.txt").read_text(encoding="utf-8"), expected)


class EvidenceRegressionTests(unittest.TestCase):
    def test_offline_failure_cannot_be_labelled_software_only(self):
        step = review.Step("broken", "software", [])
        step.status, step.code = "FAIL", 1
        self.assertEqual(review.verdict([step], {}, offline=True)[0], "SOFTWARE FAILURE")

    def test_positive_audit_does_not_hide_fetch_or_display_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_reports(root)
            step = review.Step("download", "evidence", [], evidence=True)
            step.status, step.code = "UNVERIFIED", 1
            self.assertEqual(review.verdict([step], review.reports(root), offline=False)[0], "EVIDENCE INCOMPLETE")
            step.status, step.code = "PASS", 0
            data = review.reports(root)
            data["fetch"]["public_verification"]["complete"] = False
            self.assertEqual(review.verdict([step], data, offline=False)[0], "EVIDENCE INCOMPLETE")
            data = review.reports(root)
            data["decisions"]["status"] = "DISCREPANCY"
            self.assertEqual(review.verdict([step], data, offline=False)[0], "EVIDENCE INCOMPLETE")
            data = review.reports(root)
            data["audit"]["manuscript_documents"]["pdf_bytes_verified"]["main"] = False
            self.assertEqual(review.verdict([step], data, offline=False)[0], "EVIDENCE INCOMPLETE")

    def test_review_does_not_reuse_a_previous_success_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_reports(root)
            step = review.Step("mocked software", "software", [])
            def completed(*_args, **_kwargs):
                step.status, step.code = "PASS", 0
            with mock.patch.object(review, "plan", return_value=[step]), mock.patch.object(step, "run", side_effect=completed):
                self.assertEqual(review.main(["--report-root", str(root), "--require-evidence"]), 1)
            summary = json.loads((root / "review.json").read_text())
            self.assertEqual(summary["verdict"], "EVIDENCE INCOMPLETE")
            self.assertNotEqual(Path(summary["artifact_directory"]), root)
            self.assertEqual(json.loads((root / "scientific-evidence-audit.json").read_text())["publication_ready"], True)

    def test_custom_evidence_paths_reach_all_relevant_steps(self):
        index, records = Path("/external/runs/index.json"), Path("/external/records")
        steps = review.plan(Path("/report"), offline=False, index=index, records=records,
                            budget_campaign=Path("/budget"), manuscript_pdf=Path("/main.pdf"), supplementary_pdf=Path("/si.pdf"))
        audit = next(step for step in steps if step.group == "audit")
        for option, path in (("--index", str(index)), ("--records", str(records)), ("--budget-campaign", "/budget"),
                             ("--manuscript-pdf", "/main.pdf"), ("--supplementary-pdf", "/si.pdf")):
            self.assertEqual(audit.argv[audit.argv.index(option) + 1], path)
        chain = next(step for step in steps if "ingest_run.py" in step.argv[0])
        self.assertEqual(chain.argv[chain.argv.index("--runs-root") + 1], str(index.parent))
        for script in ("scripts/claims_ledger.py", "scripts/manuscript_availability.py"):
            selected = next(step for step in steps if step.argv[0] == script and "--output" in step.argv)
            self.assertEqual(selected.argv[selected.argv.index("--index") + 1], str(index))
            self.assertEqual(selected.argv[selected.argv.index("--records") + 1], str(records))

    def test_full_registration_does_not_establish_reconstruction(self):
        specs = json.loads((PROJECT / "records/specs.json").read_text())["sets"]
        state = {"official": {dataset: {"deposit_doi": "10.5281/zenodo.1234"} for dataset in DATASETS},
                 "records": {"verified": list(specs), "defined": list(specs)}, "source_sha256": "a" * 64}
        with mock.patch.object(claims_ledger, "registration_state", return_value=state):
            ledger = claims_ledger.build(PROJECT)
        self.assertTrue(ledger["registered_coverage_complete"])
        self.assertFalse(ledger["scientific_evidence_verified"])
        self.assertNotEqual(claims_ledger.badge(ledger)["color"], "brightgreen")
        self.assertIn("no dedicated display builder", claims_ledger.render(ledger))

    def test_duplicate_registration_and_invalid_doi_are_rejected(self):
        for value in ("10.5281/zenodo.1234 extra", "https://doi.org/10.5281/zenodo.1234", 1234):
            with self.assertRaises(ValueError):
                claims_ledger.registered_doi(value)
        with tempfile.TemporaryDirectory() as tmp:
            index = Path(tmp) / "index.json"
            entry = {"dataset": DATASETS[0], "verified": True, "scientific_run": True, "profile": "full"}
            index.write_text(json.dumps({"schema": "cross-morse-runs-index-1", "runs": [entry, entry]}))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                claims_ledger.registration_state(PROJECT, index_path=index)

    def test_pattern_findings_are_advisory_and_absence_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "patterns.json"
            self.assertEqual(degeneracy.main(["--records", tmp, "--output", str(output)]), 0)
            self.assertEqual(json.loads(output.read_text())["status"], "NOT_ASSESSED")
            self.assertEqual(degeneracy.main(["--records", tmp, "--require"]), 1)
            with mock.patch.object(degeneracy, "rows_from_registry", return_value=degenerate_rows()):
                self.assertEqual(degeneracy.main(["--registry"]), 0)
                self.assertEqual(degeneracy.main(["--registry", "--strict"]), 1)

    def test_missing_confirmation_or_unproven_fallback_is_undetermined(self):
        registry = registry_fixture(0.07)
        registry["supplementary_table_S14_mechanism_contrasts_submitted_protocol"].pop()
        report = decisions.evaluate(registry)
        self.assertEqual(report["status"], "UNDETERMINED")
        registry = registry_fixture(0.07)
        registry["main_table_1_policy_and_deployment_ledger"]["Peptides-func"]["deployed_rule"] = "one-step"
        report = decisions.evaluate(registry)
        self.assertEqual(report["status"], "UNDETERMINED")
        registry["main_table_1_policy_and_deployment_ledger"]["Peptides-func"]["deployed_rule"] = "MPC-RH"
        self.assertEqual(decisions.evaluate(registry)["status"], "DISCREPANCY")

    def test_state_prerequisite_and_canonical_flag_are_checked(self):
        registry = registry_fixture(0.04)
        # Radius significance cannot override a failed state-dependence contrast.
        registry["supplementary_table_S14_mechanism_contrasts_submitted_protocol"].append(
            {"dataset": "ogbn-arxiv", "kind": "conditional destruction minus intact",
             "effect": 0.004, "margin": 0.002, "holm_P": 0.20})
        report = decisions.evaluate(registry)
        self.assertIn("ogbn-arxiv", report["discrepancies"])
        self.assertIn("Peptides-func", report["discrepancies"])
        peptide = next(row for row in report["rows"] if row["dataset"] == "Peptides-func")
        self.assertIn("Registry confirmation flag disagrees", " ".join(peptide["issues"]))

    def test_confirmation_p_values_are_finite_and_bounds_not_fabricated(self):
        for value in (float("nan"), float("inf"), -0.1, 1.1, True):
            with self.assertRaises(ValueError):
                decisions.parse_p(value)
        registry = registry_fixture("< 0.10")
        self.assertEqual(decisions.evaluate(registry)["status"], "UNDETERMINED")
        self.assertIsNone(decisions.provisional_route("unrecognized policy"))


@unittest.skipUnless(shutil.which("make"), "GNU Make is not available")
class MakeRoutingTests(unittest.TestCase):
    """The new targets route to the new scripts; a recording interpreter stands in for Python."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="cm-make-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.project = self.tmp / "src"
        self.project.mkdir()
        shutil.copyfile(PROJECT / "Makefile", self.project / "Makefile")
        self.capture = self.tmp / "calls.jsonl"
        self.interpreter = self.tmp / "recording-python"
        self.interpreter.write_text(f"#!{sys.executable}\nimport json, os, sys\n"
                                    "with open(os.environ['CM_MAKE_CAPTURE'], 'a') as stream:\n"
                                    "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n")
        self.interpreter.chmod(0o755)

    def invoke(self, target, **variables):
        command = ["make", "--no-print-directory", target, f"PYTHON={self.interpreter}"]
        command += [f"{key}={value}" for key, value in variables.items()]
        result = subprocess.run(command, cwd=self.project, capture_output=True, text=True,
                                env=dict(os.environ, CM_MAKE_CAPTURE=str(self.capture)), timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return [json.loads(line) for line in self.capture.read_text().splitlines()] if self.capture.exists() else []

    def test_review_offline_and_evidence_routing(self):
        calls = self.invoke("review", REVIEW_OFFLINE="1", REVIEW_ROOT="/r/review")
        self.assertEqual(calls, [["-B", "scripts/review.py", "--report-root", "/r/review", "--index", "runs/index.json",
                                  "--records", "records", "--deposits-dir", "../cm-deposits", "--offline"]])
        self.capture.unlink()
        calls = self.invoke("review", REQUIRE_EVIDENCE="1", REVIEW_ROOT="/r/review")
        self.assertEqual(calls[0][-1], "--require-evidence")
        self.assertNotIn("--offline", calls[0])

    def test_deposit_and_fetch_routing(self):
        calls = self.invoke("deposit", ARCHIVE="/d/a.tar.gz", PUBLISH="1", SANDBOX="1", CODE_DOI="10.5281/zenodo.9")
        self.assertEqual(calls, [["-B", "scripts/deposit_zenodo.py", "archive", "/d/a.tar.gz", "--publish", "--sandbox",
                                  "--code-doi", "10.5281/zenodo.9"]])
        self.capture.unlink()
        calls = self.invoke("fetch-deposits", REPORT_ROOT="/r")
        self.assertEqual(calls[0][:3], ["-B", "scripts/fetch_deposits.py", "--index"])
        self.assertEqual([c[1] for c in calls], ["scripts/fetch_deposits.py"])

    def test_guards_and_ledger_routing(self):
        for target, script in (("degeneracy-check", "scripts/check_measurement_degeneracy.py"),
                               ("decision-check", "scripts/check_decision_consistency.py"),
                               ("claims-check", "scripts/claims_ledger.py"),
                               ("availability-statement-check", "scripts/manuscript_availability.py")):
            if self.capture.exists():
                self.capture.unlink()
            calls = self.invoke(target, REPORT_ROOT="/r")
            self.assertEqual(calls[0][1], script, target)

    def test_deposit_requires_an_archive(self):
        result = subprocess.run(["make", "--no-print-directory", "deposit", f"PYTHON={self.interpreter}"],
                                cwd=self.project, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 2)
        self.assertIn("usage", result.stderr)


if __name__ == "__main__":
    unittest.main()
