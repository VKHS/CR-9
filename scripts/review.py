#!/usr/bin/env python3
"""Run reviewer checks and write a bounded, fresh evidence report.

--offline replays the bundled synthetic examples without contacting a deposit
service. Evidence mode checks the selected run index and records, public deposit
bytes, supported displays, manuscript comparisons, the S14 confirmation family
and the scientific audit.
--require-evidence exits 1 unless all those checks pass, the printed decisions are
consistent, and both declared manuscript PDFs have been byte-verified.

Every invocation has a new artifact directory. Older reports cannot supply a
successful audit to a later failed or offline invocation. The stable output files
are REVIEW_REPORT.md and review.json under --report-root.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts._paper_common import runtime_source_fingerprint  # noqa: E402
from scripts.availability import DATASETS  # noqa: E402

CONCERNS = [
    ("R1.1", "expected cost and matched-budget comparison",
     ["paper/tables/T2.*", "paper/records/budget_matched.*", "paper/records/ensemble_grid.*", "paper/records/nominal_time.*"]),
    ("R1.2", "admission criteria and their dataset scope",
     ["paper/tables/T1.*", "paper/tables/S13.*", "scientific-evidence-audit.json"]),
    ("R1.3", "baseline timing, dispersion and accounting provenance",
     ["paper/records/baselines.*", "degeneracy-check.json", "selected records root/baselines/raw/"]),
    ("R1.4 / R2.2", "availability supported by accessible artifacts",
     ["fetch-deposits.json", "scientific-evidence-audit.json", "CLAIMS.md", "AVAILABILITY_STATEMENT.txt"]),
    ("R2.1", "feedback relative to optimized open-loop schedules",
     ["paper/records/budget_matched.*", "scientific-evidence-audit.json (budget invariants)", "checkout docs/METHOD.md"]),
    ("R2.3", "supported models and limits of the guarantees",
     ["compare/ (S5 inventories)", "scientific-evidence-audit.json (run invariants)", "checkout docs/METHOD.md"]),
]


def git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
        return result.stdout.strip() or None if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _file_identity(path: Path) -> dict | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"review input is not a regular file: {path}")
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
            size += len(block)
    return {"sha256": digest.hexdigest(), "bytes": size}


def _directory_identity(path: Path | None) -> dict | None:
    """Bind every campaign input, including raw observations and nested code.

    Hash the relative path as well as the bytes so renaming, adding or removing
    a dependency changes the identity. Campaign reports belong outside this
    directory. A summary flag or a few top-level JSON files are insufficient.
    """
    if path is None:
        return None
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"review campaign cannot be a symlink: {path}")
    if not path.exists():
        return None
    if not path.is_dir():
        raise ValueError(f"review campaign is not a directory: {path}")
    files = {}
    for member in sorted(path.rglob("*")):
        if member.is_symlink():
            raise ValueError(f"review campaign contains a symlink: {member}")
        if member.is_file():
            files[member.relative_to(path).as_posix()] = _file_identity(member)
        elif not member.is_dir():
            raise ValueError(f"review campaign input is not a regular file: {member}")
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "files": len(files)}


def input_identity(*, index: Path, records: Path, budget_campaign: Path | None = None,
                   confirmation_campaign: Path | None = None,
                   manuscript_pdf: Path | None = None, supplementary_pdf: Path | None = None,
                   root: Path = None) -> dict:
    """Bind saved reports to their selected inputs and the code that checked them.

    This snapshots registrations, definitions and every supplied campaign file.
    Public archive byte checks and scientific invariants remain in the invocation
    artifacts. Documentation-only commits do not invalidate a saved audit.
    """
    root = Path(root or ROOT).resolve()
    index, records = Path(index).resolve(), Path(records).resolve()
    files = {}
    for directory in ("scripts", "cm_campaigns", "cm_experiments", "cm_records", "configs", "manifests/registered", "paper/build", "environment"):
        for path in sorted((root / directory).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".json", ".yaml", ".yml", ".toml", ".lock", ".sh", ".def"}:
                files[path.relative_to(root).as_posix()] = _file_identity(path)
    for path in sorted(root.glob("requirements*.txt")):
        files[path.name] = _file_identity(path)
    for name in ("Makefile", "pyproject.toml", "Dockerfile"):
        files[name] = _file_identity(root / name)
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    registries = {name: _file_identity(root / name) for name in (
        "paper/sources.json", "paper/reported_results.json", "paper/registered_constants.json",
        "paper/transcription_check.json", "records/specs.json", "docs/VALIDATION.json")}
    bindings = {path.name: _file_identity(path) for path in sorted((root / "paper/bindings").glob("*.json"))}
    manifests = {path.name: _file_identity(path) for path in sorted((records).glob("*/MANIFEST.json"))}
    campaign = Path(budget_campaign).resolve() if budget_campaign else None
    confirmation = Path(confirmation_campaign).resolve() if confirmation_campaign else None
    return {"schema": "cross-morse-review-inputs-1", "runtime_source_sha256": runtime_source_fingerprint(root),
            "tooling_sha256": hashlib.sha256(encoded).hexdigest(), "tooling_files": len(files),
            "registries": registries, "bindings": bindings,
            "selection": {"repository_root": str(root), "run_index": str(index), "records_root": str(records),
                          "budget_campaign": str(campaign) if campaign else None,
                          "confirmation_campaign": str(confirmation) if confirmation else None},
            "run_index": _file_identity(index), "records_index": _file_identity(records / "index.json"),
            "record_manifests": manifests,
            "campaign": _directory_identity(campaign),
            "confirmation_campaign": _directory_identity(confirmation),
            "pdfs": {"main": _file_identity(Path(manuscript_pdf).resolve()) if manuscript_pdf else None,
                     "supplementary": _file_identity(Path(supplementary_pdf).resolve()) if supplementary_pdf else None}}


def artifact_identities(directory: Path) -> dict:
    directory = Path(directory).resolve()
    paths = {name: directory / name for name in ("scientific-evidence-audit.json", "fetch-deposits.json",
                                                "degeneracy-check.json", "decision-consistency.json")}
    comparisons = list((directory / "compare").glob("comparison-*/coverage.json"))
    if len(comparisons) > 1:
        raise ValueError("ambiguous saved comparison reports")
    if comparisons:
        paths[comparisons[0].relative_to(directory).as_posix()] = comparisons[0]
    if any(not path.resolve().is_relative_to(directory) for path in paths.values()):
        raise ValueError("review artifact escapes its invocation directory")
    return {name: _file_identity(path) for name, path in paths.items()}


def validate_saved_review(path: Path, *, index: Path, records: Path, budget_campaign: Path | None = None,
                          confirmation_campaign: Path | None = None,
                          manuscript_pdf: Path | None = None, supplementary_pdf: Path | None = None) -> dict:
    """Re-evaluate a saved verdict and reject stale, partial or different-scope reports."""
    saved = read_json(Path(path))
    if not saved or saved.get("schema") != "cross-morse-review-1" or saved.get("offline") is not False:
        raise ValueError("a completed evidence-mode review.json is required")
    if saved.get("verdict") != "EVIDENCE COMPLETE" or saved.get("inputs_unchanged") is not True:
        raise ValueError("saved review is incomplete or its checked inputs changed during execution")
    options = {"index": Path(index).resolve(), "records": Path(records).resolve(),
               "budget_campaign": Path(budget_campaign).resolve() if budget_campaign else None,
               "confirmation_campaign": Path(confirmation_campaign).resolve() if confirmation_campaign else None,
               "manuscript_pdf": Path(manuscript_pdf).resolve() if manuscript_pdf else None,
               "supplementary_pdf": Path(supplementary_pdf).resolve() if supplementary_pdf else None}
    if saved.get("input_identity") != input_identity(**options):
        raise ValueError("saved review is stale or checked other selected inputs/tooling; rerun make review")
    if not isinstance(saved.get("artifact_directory"), str):
        raise ValueError("saved review has no invocation artifact directory")
    invocation = Path(saved["artifact_directory"]).resolve()
    if not invocation.is_dir() or saved.get("artifact_sha256") != artifact_identities(invocation):
        raise ValueError("saved review artifacts are missing or changed")
    expected = plan(invocation, offline=False, **options)
    entries = saved.get("steps")
    if not isinstance(entries, list) or len(entries) != len(expected):
        raise ValueError("saved review is missing required step records")
    for step, entry in zip(expected, entries):
        if (not isinstance(entry, dict) or entry.get("name") != step.name or entry.get("group") != step.group
                or entry.get("status") not in {"PASS", "WARN"} or entry.get("exit") != 0
                or (entry.get("status") == "WARN" and step.group != "guards")):
            raise ValueError("saved review contains incomplete, reordered or failed steps")
        step.status, step.code = entry["status"], entry["exit"]
    category, reasons = verdict(expected, reports(invocation), offline=False)
    if category != "EVIDENCE COMPLETE":
        raise ValueError("saved artifacts do not support EVIDENCE COMPLETE: " + "; ".join(reasons))
    audit = reports(invocation)["audit"]
    expected_inputs = {"repository_root": str(ROOT.resolve()), "run_index": str(options["index"]),
                       "records_root": str(options["records"]),
                       "confirmation_campaign": str(options["confirmation_campaign"]) if options["confirmation_campaign"] else None}
    if audit.get("inputs") != expected_inputs:
        raise ValueError("saved scientific audit used different selected inputs")
    return saved


class Step:
    def __init__(self, name: str, group: str, argv: list[str], *, evidence: bool = False, optional: bool = False):
        self.name, self.group, self.argv, self.evidence, self.optional = name, group, argv, evidence, optional
        self.status, self.seconds, self.tail, self.code = "NOT RUN", 0.0, "", None

    def run(self, python: str, cwd: Path) -> None:
        start = time.monotonic()
        try:
            result = subprocess.run([python, "-B", *self.argv], cwd=cwd, capture_output=True, text=True, timeout=7200)
            self.code = result.returncode
            output = (result.stdout + ("\n" + result.stderr if result.stderr else "")).strip()
        except subprocess.TimeoutExpired:
            self.code, output = 124, "timed out"
        except OSError as exc:
            self.code, output = 127, str(exc)
        self.seconds = time.monotonic() - start
        self.tail = "\n".join(output.splitlines()[-25:])[-16000:]
        if self.code == 0:
            self.status = "PASS"
        elif self.code in {124, 127} or self.code < 0:
            self.status = "FAIL"
        elif self.evidence:
            self.status = "UNVERIFIED"
        elif self.optional and self.code == 1:
            self.status = "WARN"
        else:
            self.status = "FAIL"


def plan(report_root: Path, *, offline: bool, index: Path, records: Path,
         budget_campaign: Path | None = None, confirmation_campaign: Path | None = None,
         manuscript_pdf: Path | None = None,
         supplementary_pdf: Path | None = None, deposits_dir: Path | None = None) -> list[Step]:
    paper = report_root / "paper"
    deposits_dir = deposits_dir or report_root.parent / "deposits"
    specs = str(ROOT / "records" / "specs.json")
    selected = ["--index", str(index), "--records", str(records)]
    availability_args = ["scripts/availability.py", "--check"]
    if not offline:
        for option, path in (("--budget-campaign", budget_campaign),
                             ("--confirmation-campaign", confirmation_campaign)):
            if path is not None:
                availability_args += [option, str(Path(path).resolve())]
    steps = [
        Step("runtime source identity", "software", ["scripts/source_fingerprint.py", "--validation-record", "docs/VALIDATION.json"]),
        Step("release audit", "software", ["scripts/release_audit.py", "--root", ".", "--strict"]),
        Step("registered constants and reference formulas", "software", ["scripts/check_paper_constants.py"]),
        Step("manuscript source metadata", "software", ["scripts/check_manuscript_sources.py"]),
        Step("Table S5 inventory documentation", "software", ["scripts/model_inventory.py", "--check-docs"]),
        Step("registered configuration manifests", "software", ["scripts/run_manifests.py", "registered", "--check"]),
        Step("repository availability claims are current", "software", availability_args),
        Step("committed claim map is current", "software", ["scripts/claims_ledger.py", "--check"]),
        Step("committed availability paragraph is current", "software", ["scripts/manuscript_availability.py", "--check"]),
    ]
    if offline:
        steps += [
            Step("bundled examples: artifact inventory", "examples", ["scripts/artifact_inventory.py", "--check-static"]),
            Step("bundled examples: numeric replay", "examples",
                 ["scripts/reproduce_examples.py", "--runs-root", "examples/runs", "--output-dir", str(report_root / "examples")]),
            Step("bundled examples: tables and figures", "examples",
                 ["scripts/build_paper.py", "all", "--runs-root", "examples/runs", "--label", "examples", "--output-dir", str(paper)]),
            Step("bundled examples: record tables", "examples",
                 ["scripts/records.py", "tables", "--root", "examples/records", "--specs", specs, "--output-dir", str(paper / "records")]),
        ]
    else:
        audit_args = ["scripts/scientific_evidence_audit.py", *selected, "--output", str(report_root / "scientific-evidence-audit.json")]
        for option, path in (("--budget-campaign", budget_campaign), ("--confirmation-campaign", confirmation_campaign),
                             ("--manuscript-pdf", manuscript_pdf), ("--supplementary-pdf", supplementary_pdf)):
            if path is not None:
                audit_args += [option, str(path)]
        steps += [
            Step("fetch and verify registered public deposits", "evidence",
                 ["scripts/fetch_deposits.py", "--index", str(index), "--runs-root", str(index.parent),
                  "--deposits-dir", str(deposits_dir), "--output", str(report_root / "fetch-deposits.json")], evidence=True),
            Step("sealed stage chains of materialized archives", "evidence",
                 ["scripts/ingest_run.py", "--check", "--index", str(index), "--runs-root", str(index.parent)], evidence=True),
            Step("record values traced to raw files", "evidence",
                 ["scripts/records.py", "check", "--root", str(records), "--specs", specs], evidence=True),
            Step("supported run-derived tables and figures", "displays",
                 ["scripts/build_paper.py", "all", "--index", str(index), "--label", "official", "--output-dir", str(paper)], evidence=True),
            Step("registered record-table builders", "displays",
                 ["scripts/records.py", "tables", "--root", str(records), "--specs", specs, "--output-dir", str(paper / "records")], evidence=True),
            Step("comparison with the manuscript registry", "comparison",
                 ["scripts/compare_registered_runs.py", "--index", str(index), "--output-dir", str(report_root / "compare")], evidence=True),
            Step("scientific evidence audit", "audit", audit_args, evidence=True),
        ]
    steps += [
        Step("baseline pattern diagnostic", "guards",
             ["scripts/check_measurement_degeneracy.py", "--records", str(records), "--output", str(report_root / "degeneracy-check.json")]),
        Step("printed decision-rule consistency", "guards",
             ["scripts/check_decision_consistency.py", "--output", str(report_root / "decision-consistency.json"), "--warn-only"]),
        Step("claim map for selected evidence", "software",
             ["scripts/claims_ledger.py", *selected, "--output", str(report_root), "--json", str(report_root / "claims.json")]),
        Step("availability paragraph for selected evidence", "software",
             ["scripts/manuscript_availability.py", *selected, "--output", str(report_root / "AVAILABILITY_STATEMENT.txt")]),
    ]
    return steps


def read_json(path: Path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def reports(report_root: Path) -> dict:
    comparisons = list((report_root / "compare").glob("comparison-*/coverage.json"))
    # More than one report is ambiguous. Main always supplies a fresh invocation directory.
    return {"audit": read_json(report_root / "scientific-evidence-audit.json"),
            "compare": read_json(comparisons[0]) if len(comparisons) == 1 else None,
            "fetch": read_json(report_root / "fetch-deposits.json"),
            "patterns": read_json(report_root / "degeneracy-check.json"),
            "decisions": read_json(report_root / "decision-consistency.json")}


def verdict(steps: list[Step], data: dict, *, offline: bool) -> tuple[str, list[str]]:
    failures = [step.name for step in steps if step.status == "FAIL" or step.status == "NOT RUN"]
    if failures:
        return "SOFTWARE FAILURE", ["Failed or unexecuted software checks: " + ", ".join(failures)]
    if offline:
        if any(step.status not in {"PASS", "WARN"} for step in steps):
            return "SOFTWARE FAILURE", ["The bundled-example workflow did not complete."]
        return "SOFTWARE-ONLY", ["Bundled synthetic examples replayed; public deposits and official benchmark evidence were not verified."]
    reasons = [f"{step.name}: {step.status}" for step in steps if step.status not in {"PASS", "WARN"}]
    audit = data.get("audit") or {}
    if audit.get("publication_ready") is not True or audit.get("status") != "PASS" or audit.get("blockers") != []:
        reasons.append("The scientific evidence audit has not passed.")
    if (audit.get("confirmation_family") or {}).get("status") != "PASS":
        reasons.append("The original nine-contrast S14 confirmation family has not passed its joint audit.")
    for key, expected in (("official_archives", 7), ("auxiliary_records", 10), ("manuscript_bindings", 7)):
        coverage = audit.get(key) or {}
        if coverage.get("validated") != expected or coverage.get("required") != expected:
            reasons.append(f"Audit coverage is incomplete: {key}.")
    pdfs = (audit.get("manuscript_documents") or {}).get("pdf_bytes_verified") or {}
    if any(pdfs.get(role) is not True for role in ("main", "supplementary")):
        reasons.append("Both declared manuscript PDFs must be byte-verified (--manuscript-pdf and --supplementary-pdf).")
    fetch = data.get("fetch") or {}
    if ((fetch.get("public_verification") or {}).get("complete") is not True
            or fetch.get("failed", 0) or fetch.get("missing_doi", 0) or fetch.get("missing_datasets", [])
            or {row.get("dataset") for row in fetch.get("entries", [])} != set(DATASETS)):
        reasons.append("Public DOI records and pinned archive bytes have not been verified for all seven datasets.")
    comparison = data.get("compare") or {}
    if (comparison.get("status") != "PASS" or comparison.get("compared_datasets") != 7
            or comparison.get("model_inventory_verified_datasets") != 7):
        reasons.append("The official-run comparison and sealed Table S5 inventories have not all passed.")
    decisions = data.get("decisions") or {}
    if (decisions.get("status") != "CONSISTENT"
            or {row.get("dataset") for row in decisions.get("rows", [])} != set(DATASETS)):
        reasons.append("Printed decisions and the registered confirmation rule are inconsistent or undetermined.")
    if reasons:
        return "EVIDENCE INCOMPLETE", reasons
    return "EVIDENCE COMPLETE", ["All checks in the registered audit scope passed: public run deposits, supported display builds, "
                                  "core result bindings, required auxiliary records, the joint S14 confirmation family "
                                  "and manuscript byte identities. "
                                  "This does not certify every paper claim or rerun the benchmark training."]


def render(steps: list[Step], report_root: Path, *, offline: bool, index: Path, records: Path) -> str:
    data = reports(report_root)
    audit, comparison, fetch = data["audit"], data["compare"], data["fetch"]
    patterns, decisions = data["patterns"], data["decisions"]
    category, reasons = verdict(steps, data, offline=offline)
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    lines = ["# CROSS-MORSE review report", "", f"- Generated: {now} ({'offline' if offline else 'evidence'} mode)",
             f"- Git commit: `{git_commit() or 'not a git checkout'}`",
             f"- Runtime source identity: `{runtime_source_fingerprint(ROOT)}`",
             f"- Python {platform.python_version()} on {platform.platform()}",
             f"- Run index: `{index}`; records root: `{records}`",
             f"- This invocation's artifacts: `{report_root}`", "", "## Verdict", "", f"**{category}**", ""]
    lines += [f"- {reason}" for reason in reasons]
    lines += ["", "## Steps", "", "| Group | Step | Status | Seconds |", "|---|---|---|---|"]
    for step in steps:
        lines.append(f"| {step.group} | {step.name} | **{step.status}** | {step.seconds:.1f} |")
    lines += ["", "PASS means the command completed successfully; it is not a general scientific verdict. "
              "UNVERIFIED means an evidence step did not pass because its inputs are missing, invalid or inconsistent; "
              "the step output and audit explain which. WARN records an advisory finding. FAIL records a failed "
              "software check or process execution.", ""]
    if fetch:
        lines += ["## Registered deposits", "", "| Dataset | Registered DOI | Status | Detail |", "|---|---|---|---|"]
        for row in fetch.get("entries", []):
            cells = [row.get("dataset", "—"), row.get("doi") or "none", row.get("status", "—"), row.get("detail", "")]
            lines.append("| " + " | ".join(str(cell).replace("|", r"\|").replace("\n", " ") for cell in cells) + " |")
        if not fetch.get("entries"):
            lines.append("| — | — | NONE REGISTERED | No archives are registered in the selected index |")
        lines += ["", "Public verification complete: **" + str((fetch.get("public_verification") or {}).get("complete", False)) + "**.", ""]
    if comparison:
        lines += ["## Manuscript comparison", "",
                  f"Status: **{comparison.get('status')}**; datasets compared: {comparison.get('compared_datasets')}/{comparison.get('required_datasets')}; "
                  f"sealed model inventories satisfying Table S5: {comparison.get('model_inventory_verified_datasets')}.", ""]
        lines += [f"- {issue}" for issue in comparison.get("issues", [])]
        lines.append("")
    if audit:
        lines += ["## Scientific evidence audit", "", f"Status: **{audit.get('status')}**; publication_ready: **{audit.get('publication_ready')}**.", "",
                  f"Scope: {audit.get('scope', 'See scientific-evidence-audit.json')}", ""]
        for label, key in (("Official archives", "official_archives"), ("Auxiliary record sets", "auxiliary_records"), ("Manuscript bindings", "manuscript_bindings")):
            coverage = audit.get(key) or {}
            lines.append(f"- {label}: {coverage.get('validated', 0)}/{coverage.get('required', '?')} validated.")
        pdfs = (audit.get("manuscript_documents") or {}).get("pdf_bytes_verified") or {}
        lines.append(f"- Manuscript PDF bytes verified: main={pdfs.get('main', False)}, supplementary={pdfs.get('supplementary', False)}.")
        lines.append(f"- Original S14 confirmation family: {(audit.get('confirmation_family') or {}).get('status', 'UNAVAILABLE')}.")
        lines += ["", *[f"- Limit: {limit}" for limit in audit.get("limits", [])], ""]
        grouped = {}
        for blocker in audit.get("blockers", []):
            grouped.setdefault(blocker.get("code", "?"), []).append(blocker)
        if grouped:
            lines += ["| Audit blocker | Count | Locations |", "|---|---|---|"]
            for code, items in sorted(grouped.items()):
                locations = ", ".join(str(item.get("location")) for item in items[:8])
                lines.append(f"| `{code}` | {len(items)} | {locations}" + (" …" if len(items) > 8 else "") + " |")
            lines.append("")
    lines += ["## Diagnostics", ""]
    if patterns:
        lines.append(f"Baseline pattern inspection: **{patterns.get('status')}** ({patterns.get('external_rows', 0)} external rows).")
        lines += ["", "Patterns request examination of the original timing and accounting records. "
                  "They neither prove invalid measurements nor validate their origin.", ""]
        lines += [f"- [{finding['test']}] {finding.get('method', '')} {finding['detail']}" for finding in patterns.get("findings", [])]
    if decisions:
        lines += ["", f"Printed decision-rule consistency: **{decisions.get('status')}**.", ""]
        for row in decisions.get("rows", []):
            if row.get("status") != "CONSISTENT":
                lines.append(f"- {row['dataset']}: Table 1 `{row['deployed']}` ({row.get('manuscript_status')}); "
                             f"canonical registry `{row.get('registry_resolution')}` ({row.get('registry_status')}); "
                             f"confirmation permits `{row['rule_outcome']}`. {row.get('reason', '')}")
    lines += ["", "## Reviewer concerns and relevant artifacts", "",
              "Paths below are relative to this invocation's artifact directory unless marked as checkout/selected records. "
              "Missing evidence leaves the corresponding outputs unavailable; the paths are not a claim that every item was produced.", "",
              "| Concern | Subject | Artifacts when available |", "|---|---|---|"]
    for code, subject, artifacts in CONCERNS:
        lines.append(f"| {code} | {subject} | " + "; ".join(f"`{artifact}`" for artifact in artifacts) + " |")
    lines += ["", "## Step output (last lines)", ""]
    for step in steps:
        lines += [f"### {step.name} — {step.status} (exit {step.code})", "", "````text", step.tail or "(no output)", "````", ""]
    claims_path = report_root / "CLAIMS.md"
    if claims_path.is_file():
        lines += ["---", "", claims_path.read_text(encoding="utf-8")]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--index", type=Path, default=ROOT / "runs" / "index.json")
    parser.add_argument("--records", type=Path, default=ROOT / "records")
    parser.add_argument("--deposits-dir", type=Path, help="reusable byte-verified archive cache (default: report-root/deposits)")
    parser.add_argument("--budget-campaign", type=Path)
    parser.add_argument("--confirmation-campaign", type=Path, help="original nine-contrast S14 campaign and its bound inputs")
    parser.add_argument("--manuscript-pdf", type=Path)
    parser.add_argument("--supplementary-pdf", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--require-evidence", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    report_root = args.report_root.resolve()
    index, records = args.index.resolve(), args.records.resolve()
    if args.offline and any((args.budget_campaign, args.confirmation_campaign, args.manuscript_pdf, args.supplementary_pdf)):
        parser.error("official campaign/PDF arguments apply only to evidence mode")
    if report_root == ROOT:
        parser.error("report-root must be a dedicated output directory, not the repository root")
    if any(report_root.is_relative_to(campaign.resolve()) for campaign in
           (args.budget_campaign, args.confirmation_campaign) if campaign):
        parser.error("report-root must be outside campaign input directories")
    options = {"offline": args.offline, "index": index, "records": records,
               "budget_campaign": args.budget_campaign.resolve() if args.budget_campaign else None,
               "confirmation_campaign": args.confirmation_campaign.resolve() if args.confirmation_campaign else None,
               "manuscript_pdf": args.manuscript_pdf.resolve() if args.manuscript_pdf else None,
               "supplementary_pdf": args.supplementary_pdf.resolve() if args.supplementary_pdf else None,
               "deposits_dir": args.deposits_dir.resolve() if args.deposits_dir else report_root / "deposits"}
    if args.dry_run:
        for step in plan(report_root / "<fresh-invocation>", **options):
            print(f"[{step.group}] {step.name}: {' '.join(step.argv)}")
        return 0
    report_root.mkdir(parents=True, exist_ok=True)
    invocation = Path(tempfile.mkdtemp(prefix="review-", dir=report_root))
    # An interrupted new invocation must not leave an older success at the
    # stable report paths looking like its own result.
    (report_root / "REVIEW_REPORT.md").write_text(
        "# CROSS-MORSE review report\n\nReview in progress. No verdict is available for this invocation.\n\n"
        f"Artifacts: `{invocation}`.\n", encoding="utf-8")
    (report_root / "review.json").write_text(json.dumps(
        {"schema": "cross-morse-review-1", "verdict": "IN PROGRESS", "artifact_directory": str(invocation)}, indent=1) + "\n",
        encoding="utf-8")
    identity_options = {key: options[key] for key in ("index", "records", "budget_campaign", "confirmation_campaign",
                                                    "manuscript_pdf", "supplementary_pdf")}
    before, identity_error = None, None
    try:
        before = input_identity(**identity_options)
    except (OSError, ValueError) as exc:
        identity_error = str(exc)
    steps = plan(invocation, **options)
    for step in steps:
        step.run(args.python, ROOT)
        if step.group == "guards" and step.status == "PASS":
            path = invocation / ("decision-consistency.json" if "decision" in step.name else "degeneracy-check.json")
            diagnostic = read_json(path)
            if diagnostic is None:
                step.status = "FAIL"
                step.tail += "\nRequired diagnostic JSON was not produced."
            elif diagnostic.get("status") in {"DISCREPANCY", "UNDETERMINED", "REVIEW_REQUIRED"}:
                step.status = "WARN"
        print(f"{step.status:10s} {step.name} ({step.seconds:.1f}s)")
    after, artifacts = None, None
    try:
        after = input_identity(**identity_options)
        artifacts = artifact_identities(invocation)
        if before != after:
            identity_error = "registered inputs or verification tooling changed during review; rerun on stable inputs"
    except (OSError, ValueError) as exc:
        identity_error = str(exc)
    if identity_error:
        failed = Step("review input and artifact identities", "software", [])
        failed.status, failed.code, failed.tail = "FAIL", 2, identity_error
        steps.append(failed)
    category, reasons = verdict(steps, reports(invocation), offline=args.offline)
    report = render(steps, invocation, offline=args.offline, index=index, records=records)
    (report_root / "REVIEW_REPORT.md").write_text(report, encoding="utf-8")
    summary = {"schema": "cross-morse-review-1", "offline": args.offline, "verdict": category, "reasons": reasons,
               "generated_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
               "input_identity": after, "inputs_unchanged": identity_error is None and before == after,
               "artifact_sha256": artifacts,
               "artifact_directory": str(invocation), "run_index": str(index), "records_root": str(records),
               "deposits_directory": str(options["deposits_dir"]),
               "steps": [{"name": step.name, "group": step.group, "status": step.status,
                          "seconds": round(step.seconds, 2), "exit": step.code} for step in steps]}
    (report_root / "review.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(f"\n{category}; wrote {report_root / 'REVIEW_REPORT.md'}")
    return 1 if category == "SOFTWARE FAILURE" or args.require_evidence and category != "EVIDENCE COMPLETE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
