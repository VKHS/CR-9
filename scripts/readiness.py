#!/usr/bin/env python3
"""Read-only author readiness checklist for selected evidence and manuscript files.

PASS has the scope named in each row. TODO means required evidence or a dependency
is unavailable; FAIL means a supplied check did not pass. WARN is advisory (for
example repeated baseline patterns), and does not establish invalid measurements.
Exit 0 requires all blocking items to pass and a current, complete evidence-mode
review report. Source review identifies its actual reviewer and scope; it is not
a numerical certificate.

Supply --index and --records when they differ from the checkout defaults. A saved
review of other roots or earlier registry/tooling bytes cannot satisfy this check.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

DATASETS = ("ogbn-arxiv", "ogbn-proteins", "ogbn-products", "Roman-empire", "PascalVOC-SP", "COCO-SP", "Peptides-func")
STEMS = {"ogbn-arxiv": "arxiv", "ogbn-proteins": "proteins", "ogbn-products": "products", "Roman-empire": "roman_empire",
         "PascalVOC-SP": "pascalvoc_sp", "COCO-SP": "coco_sp", "Peptides-func": "peptides_func"}
VERSION_DOI = re.compile(r"10\.5281/zenodo\.[1-9][0-9]*")


def item(name: str, status: str, detail: str, action: str = "") -> dict:
    return {"item": name, "status": status, "blocking": status != "WARN", "detail": detail, "next": action}


def _quiet(argv: list[str]) -> int:
    return subprocess.run([sys.executable, "-B", *argv], cwd=ROOT, capture_output=True, text=True, timeout=1200).returncode


def _selected(args) -> tuple[Path, Path]:
    return (Path(getattr(args, "index", None) or ROOT / "runs/index.json").resolve(),
            Path(getattr(args, "records", None) or ROOT / "records").resolve())


def _bindings(runs: dict, index: Path) -> dict:
    from scripts.check_result_binding import check_binding
    from scripts.ingest_run import registered_path
    from scripts.scientific_evidence_audit import binding_coverage
    from scripts.emit_campaign_binding import check as check_campaign_binding
    matched, pending, errors = [], [], []
    for dataset in DATASETS:
        path = ROOT / "paper/bindings" / f"{STEMS[dataset]}_full.json"
        try:
            binding = json.loads(path.read_text(encoding="utf-8"))
            if binding.get("status") != "author_completed" or dataset not in runs:
                pending.append(dataset)
                continue
            run = registered_path(runs[dataset], index.parent)
            if not run.is_dir():
                pending.append(dataset)
                continue
            campaign = dataset == "Roman-empire"
            missing = binding_coverage(binding, campaign=campaign, dataset=dataset, manuscript_root=ROOT)
            if missing:
                raise ValueError("missing or invalid core targets: " + ", ".join(missing))
            if campaign:
                paths = sorted(run.glob("split_*"))
                if len(paths) != 10:
                    raise ValueError("Roman-empire requires ten split runs")
                result = check_campaign_binding(paths, binding)
            else:
                result = check_binding(run, binding)
            if result.get("status") != "MATCH":
                raise ValueError("bound values do not match the sealed outputs")
            matched.append(dataset)
        except FileNotFoundError:
            pending.append(dataset)
        except (OSError, ValueError, KeyError, TypeError, IndexError, RuntimeError) as exc:
            errors.append(f"{dataset}: {exc}")
    return {"matched": matched, "pending": pending, "errors": errors}


def evidence_items(args) -> list[dict]:
    from cm_campaigns.contract import RECORD_SETS
    from scripts.ingest_run import check_absent_manifest, check_present, load_index, registered_path, registration_descriptor
    from scripts.records import load_specs, read_summary, verify_set
    from scripts.scientific_evidence_audit import validate_record_coverage
    index_path, records_root = _selected(args)
    manifests_root = Path(getattr(args, "manifests_root", None) or ROOT / "manifests").resolve()
    index = load_index(index_path)
    runs, errors = {}, []
    for entry in index.get("runs", []):
        if not isinstance(entry, dict) or entry.get("dataset") not in DATASETS:
            errors.append("invalid or unknown dataset registration")
            continue
        dataset = entry["dataset"]
        if dataset in runs:
            errors.append(f"duplicate archive registration: {dataset}")
        runs[dataset] = entry
    registered, materialized = [], []
    for dataset, entry in runs.items():
        try:
            registration_descriptor(entry)
            check_absent_manifest(entry, manifests_root / "official")
            if (entry.get("verified") is not True or entry.get("profile") != "full"
                    or entry.get("scientific_run") is not True or not VERSION_DOI.fullmatch(str(entry.get("deposit_doi") or ""))):
                raise ValueError("official full-profile production DOI registration is incomplete")
            registered.append(dataset)
            payload = registered_path(entry, index_path.parent)
            if payload.exists():
                check_present(entry, payload)
                materialized.append(dataset)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"{dataset}: {exc}")
    status = "FAIL" if errors else ("PASS" if len(registered) == 7 and len(materialized) == 7 else "TODO")
    items = [item("evidence", status, f"{len(registered)}/7 checked archive registrations, {len(materialized)}/7 current local payload inventories verified; "
                  "public DOI access requires the evidence review"
                  + ("; " + "; ".join(errors[:3]) if errors else ""),
                  "" if status == "PASS" else "evidence_pipeline.py preflight/pack/deposit/publish/register "
                                            "(docs/EVIDENCE_REGISTRATION.md)")]
    try:
        bound = _bindings(runs, index_path)
        status = "FAIL" if bound["errors"] else ("PASS" if len(bound["matched"]) == 7 else "TODO")
        detail = f"{len(bound['matched'])}/7 core bindings match available sealed outputs"
        if bound["errors"]:
            detail += "; " + "; ".join(bound["errors"][:3])
    except ImportError as exc:
        status, detail = "TODO", f"binding replay dependency unavailable: {exc}"
    items.append(item("bindings", status, detail, "" if status == "PASS" else
                      "materialize registered runs and run evidence_pipeline.py bind --plan PLAN"))
    verified, invalid = [], []
    specs = load_specs(ROOT / "records", ROOT / "records/specs.json")
    traced = _quiet(["scripts/records.py", "check", "--root", str(records_root), "--specs", str(ROOT / "records/specs.json")]) == 0
    for name, contract in RECORD_SETS.items():
        path = records_root / name
        if not path.is_dir():
            continue
        try:
            result = verify_set(path, specs["sets"][name], specs)
            if result.get("synthetic") is not False:
                raise ValueError("synthetic records are not official evidence")
            validate_record_coverage(name, read_summary(path, specs["sets"][name], specs["summary_columns"]),
                                     specs["sets"][name], contract)
            verified.append(name)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            invalid.append(f"{name}: {exc}")
    status = "FAIL" if invalid or not traced else ("PASS" if len(verified) == len(RECORD_SETS) else "TODO")
    items.append(item("records", status, f"{len(verified)}/{len(RECORD_SETS)} sets pass coverage, manifests and raw-value tracing"
                      + ("; " + "; ".join(invalid[:3]) if invalid else "") + ("; records index check failed" if not traced else ""),
                      "" if status == "PASS" else "export_records.py (exporters or from-mapping), then make records-index"))
    if args.budget_campaign is None:
        items.append(item("campaign", "TODO", "no --budget-campaign supplied", "pass the complete extracted E3-E6 campaign"))
    else:
        try:
            from cm_campaigns.arm_campaign import check_campaign
            campaign = Path(args.budget_campaign).resolve()
            checked = check_campaign(campaign, json.loads((campaign / "study.json").read_text(encoding="utf-8")), strict=True)
            if checked.get("passed") is not True:
                raise ValueError("campaign validator did not report passed=true")
            items.append(item("campaign", "PASS", "E3-E6 campaign passed its strict source, role, selection and accounting checks"))
        except ImportError as exc:
            items.append(item("campaign", "TODO", f"campaign replay dependency unavailable: {exc}",
                              "use the full experiment environment and rerun readiness"))
        except (OSError, ValueError, KeyError, TypeError, IndexError, RuntimeError) as exc:
            items.append(item("campaign", "FAIL", f"campaign check failed: {exc}",
                              "inspect the original records; export with scripts/arm_campaign.py export --complete"))
    confirmation = getattr(args, "confirmation_campaign", None)
    if confirmation is None:
        items.append(item("confirmation_family", "TODO", "no --confirmation-campaign supplied",
                          "pass the original nine-contrast S14 family with its prespecified protocol and raw observations"))
    else:
        try:
            from cm_campaigns.confirmation import check_campaign as check_confirmation_campaign
            checked = check_confirmation_campaign(Path(confirmation).resolve(), manuscript_root=ROOT)
            if checked.get("status") != "PASS":
                raise ValueError("S14 confirmation-family validator did not report status=PASS")
            items.append(item("confirmation_family", "PASS",
                              "S14 joint Holm family and frozen decisions replayed from the supplied original inputs"))
        except ImportError as exc:
            items.append(item("confirmation_family", "TODO", f"S14 replay dependency unavailable: {exc}",
                              "use the full experiment environment and rerun readiness"))
        except (OSError, ValueError, KeyError, TypeError, IndexError, RuntimeError) as exc:
            items.append(item("confirmation_family", "FAIL", f"S14 family check failed: {exc}",
                              "inspect original unrounded tests, paired observations, protocol and pre-admission locks"))
    return items


def manuscript_items(args) -> list[dict]:
    items = []
    if args.manuscript_pdf and args.supplementary_pdf:
        from scripts.check_manuscript_sources import inspect_sources
        checked = inspect_sources(ROOT, {"main": args.manuscript_pdf, "supplementary": args.supplementary_pdf})
        verified = checked.get("pdf_bytes_verified", {})
        if checked["status"] == "PASS" and all(verified.get(r) is True for r in ("main", "supplementary")):
            items.append(item("pdfs", "PASS", "the supplied PDFs are byte-identical to paper/sources.json"))
        else:
            items.append(item("pdfs", "FAIL", "; ".join(map(str, checked.get("issues", []))) or "PDF identity differs",
                              "inspect register_manuscript_sources.py output; complete a bound contextual source review before --register"))
    else:
        items.append(item("pdfs", "TODO", "both final PDFs are required", "rerun with --manuscript-pdf and --supplementary-pdf"))
    from scripts.register_manuscript_sources import check_review_record
    try:
        reviewed = check_review_record(ROOT)
        items.append(item("transcription", "PASS", f"byte-bound contextual source review covers {len(reviewed['groups'])} groups; "
                          "this records contextual review, not automatic numerical verification"))
    except FileNotFoundError:
        items.append(item("transcription", "TODO", "no contextual source review recorded",
                          "make manuscript-sources REVIEW_TEMPLATE=...; complete it; register with REVIEW_NOTES=..."))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        items.append(item("transcription", "FAIL", str(exc), "review the current PDFs/registry and register fresh bound review notes"))
    from scripts.check_measurement_degeneracy import analyse, rows_from_records, rows_from_registry
    _, records_root = _selected(args)
    baseline_rows = rows_from_records(records_root)
    scope = "deposited baseline summary" if baseline_rows is not None else "registry only; baseline records unavailable"
    patterns = analyse(baseline_rows if baseline_rows is not None else rows_from_registry(ROOT))
    findings = patterns.get("findings", [])
    if findings:
        items.append(item("baselines", "WARN", f"{scope}: {len(findings)} repeated-value patterns need contextual inspection; "
                          "patterns neither prove invalid measurements nor verify their origin",
                          "inspect the original E17 timings/accounting and label measured or declared quantities accurately"))
    elif baseline_rows is None or not patterns.get("external_rows"):
        items.append(item("baselines", "TODO", "no assessable external-baseline records", "export the baseline record set"))
    else:
        items.append(item("baselines", "PASS", "no registered repetition signatures in the supplied baseline summary; "
                          "record traceability is checked separately"))
    from scripts.check_decision_consistency import evaluate
    decisions = evaluate(json.loads((ROOT / "paper/reported_results.json").read_text(encoding="utf-8")))
    bad = [r for r in decisions["rows"] if r["status"] != "CONSISTENT"]
    complete = {r["dataset"] for r in decisions["rows"]} == set(DATASETS)
    items.append(item("decisions", "PASS" if not bad and complete else "FAIL",
                      "Table 1 policies follow the registered confirmation rule" if not bad and complete else
                      "; ".join(f"{r['dataset']}: Table 1 {r['deployed']!r}, rule permits {r['rule_outcome']!r}" for r in bad)
                      or "decision coverage is incomplete",
                      "" if not bad and complete else "reconcile manuscript policy labels with the actual frozen confirmation records"))
    return items


def repository_items(args) -> list[dict]:
    items = []
    availability_args = ["scripts/availability.py", "--check"]
    for option, path in (("--budget-campaign", getattr(args, "budget_campaign", None)),
                         ("--confirmation-campaign", getattr(args, "confirmation_campaign", None))):
        if path is not None:
            availability_args += [option, str(Path(path).resolve())]
    stale = [name for name, argv in (("AVAILABILITY.md", availability_args),
                                     ("paper/CLAIMS.md", ["scripts/claims_ledger.py", "--check"]),
                                     ("paper/AVAILABILITY_STATEMENT.txt", ["scripts/manuscript_availability.py", "--check"]))
             if _quiet(argv) != 0]
    items.append(item("generated", "PASS" if not stale else "FAIL",
                      "committed availability texts match the checkout registrations" if not stale else "stale: " + ", ".join(stale),
                      "" if not stale else "make availability claims availability-statement"))
    from scripts._paper_common import runtime_source_fingerprint
    record = json.loads((ROOT / "docs/VALIDATION.json").read_text(encoding="utf-8"))
    current = runtime_source_fingerprint(ROOT)
    same = record.get("source_sha256") == current
    items.append(item("runtime", "PASS" if same else "FAIL",
                      f"validation record {record.get('date')} identifies runtime {str(record.get('source_sha256'))[:12]}; "
                      f"checkout {current[:12]}; no new runtime test is implied",
                      "" if same else "rerun the required runtime checks and record their actual results in docs/VALIDATION.json"))
    if args.review_report:
        from scripts.review import validate_saved_review
        index, records = _selected(args)
        try:
            validate_saved_review(args.review_report, index=index, records=records, budget_campaign=args.budget_campaign,
                                  confirmation_campaign=getattr(args, "confirmation_campaign", None),
                                  manuscript_pdf=args.manuscript_pdf, supplementary_pdf=args.supplementary_pdf)
            items.append(item("review", "PASS", "saved EVIDENCE COMPLETE verdict recomputes from unchanged artifacts for current selected inputs and tooling"))
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            items.append(item("review", "FAIL", str(exc), "rerun make review REQUIRE_EVIDENCE=1 with the same selected evidence roots, campaign and PDFs"))
    else:
        items.append(item("review", "TODO", "no evidence-mode review.json supplied", "run make review, then pass --review-report REVIEW_ROOT/review.json"))
    return items


def is_ready(items: list[dict]) -> bool:
    return all(row["status"] == "PASS" or (row["status"] == "WARN" and row.get("blocking") is False) for row in items)


def markdown(items: list[dict]) -> str:
    done = sum(i["status"] == "PASS" for i in items)
    warnings = sum(i["status"] == "WARN" for i in items)
    lines = ["# Author evidence readiness", "", f"**{done}/{len(items)}** items pass; {warnings} advisory warning(s).",
             "Readiness applies only to the explicitly selected evidence roots and files. A PASS is limited to the scope in its row.", "",
             "| Item | Status | Detail | Next |", "|---|---|---|---|"]
    for entry in items:
        lines.append("| " + " | ".join(str(entry[k]).replace("|", "\\|").replace("\n", " ")
                                      for k in ("item", "status", "detail", "next")) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, default=ROOT / "runs/index.json")
    parser.add_argument("--records", type=Path, default=ROOT / "records")
    parser.add_argument("--manifests-root", type=Path, default=ROOT / "manifests", help="manifest root containing official/")
    parser.add_argument("--manuscript-pdf", type=Path)
    parser.add_argument("--supplementary-pdf", type=Path)
    parser.add_argument("--budget-campaign", type=Path)
    parser.add_argument("--confirmation-campaign", type=Path, help="original nine-contrast S14 campaign and its bound inputs")
    parser.add_argument("--review-report", type=Path, help="review.json written by make review")
    parser.add_argument("--output", type=Path, help="JSON outside the checkout (a .md is written beside it)")
    args = parser.parse_args(argv)
    try:
        if args.output:
            output = args.output.resolve()
            inputs = [p.resolve() for p in (args.index, args.manuscript_pdf, args.supplementary_pdf, args.review_report) if p]
            if output.is_relative_to(ROOT) or output.suffix != ".json" or output in inputs or output.with_suffix(".md") in inputs:
                raise ValueError("write readiness to a .json file outside the checkout, separate from its inputs")
            if any(output.is_relative_to(campaign.resolve()) for campaign in
                   (args.budget_campaign, args.confirmation_campaign) if campaign):
                raise ValueError("write readiness reports outside campaign input directories")
        items = evidence_items(args) + manuscript_items(args) + repository_items(args)
        text = markdown(items)
        if args.output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({"schema": "cross-morse-readiness-2", "ready": is_ready(items),
                                          "run_index": str(args.index.resolve()), "records_root": str(args.records.resolve()),
                                          "items": items}, indent=1) + "\n", encoding="utf-8")
            output.with_suffix(".md").write_text(text, encoding="utf-8")
        print(text)
        return 0 if is_ready(items) else 1
    except (OSError, ValueError, KeyError, TypeError, ImportError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
