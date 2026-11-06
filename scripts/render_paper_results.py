#!/usr/bin/env python3
"""Validate manuscript reference metadata and optionally export a compact report.

The canonical reference values are in paper/reported_results.json. The source
index records PDF hashes, byte sizes, reviewed page counts and table locations.
PDFs need not be distributed: an explicit null bundled_path declares metadata-only
validation. A declared bundled file, or an explicitly supplied local copy, must
match its recorded bytes. This does not verify an experiment.

Without --output-dir this command only validates. To create optional reports:
    python -B scripts/render_paper_results.py --output-dir ../cm-reports/registry
    python -B scripts/render_paper_results.py --check --output-dir ../cm-reports/registry
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._paper_common import DATASETS, ROOT, load_registered, resolved_policy_view

KINDS = {"author_reported_measurements", "derived_arithmetic", "analytical_sensitivity", "mixed"}


def read_json(path: Path) -> dict:
    def invalid(value):
        raise ValueError(f"Non-finite JSON number: {value}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid)


def _verify_pdf_bytes(path: Path, source: dict, label: str) -> str:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        if not handle.read(5).startswith(b"%PDF-"):
            raise ValueError(f"Not a PDF: {label}")
        handle.seek(0)
        for block in iter(lambda: handle.read(2**20), b""):
            size += len(block)
            digest.update(block)
    if digest.hexdigest() != source["sha256"] or size != source["byte_size"]:
        raise ValueError(f"PDF content identity changed: {label}")
    return digest.hexdigest()


def validate_sources(root: Path, reported: dict, index: dict, local_pdfs: dict | None = None) -> dict:
    """Check recorded identities and provenance, and verify any supplied PDF bytes.

    ``bundled_path`` must explicitly be null or a safe repository-relative path.
    A null path records a reference without distributing its PDF. Supplying an
    external copy adds byte verification; it never excuses a missing or changed
    file that the index declares as bundled.
    """
    root = root.resolve()
    local_pdfs = local_pdfs or {}
    if index.get("schema_version") != "paper-sources-1":
        raise ValueError("Unsupported paper source index")
    documents = index.get("documents", {})
    if not isinstance(documents, dict) or set(documents) != {"main", "supplementary"}:
        raise ValueError("Both manuscript PDF sources must be identified")
    if set(local_pdfs) - set(documents):
        raise ValueError("Unknown local PDF role")
    identities, bytes_verified, verification_sources = {}, {}, {}
    for name, source in documents.items():
        if not isinstance(source, dict):
            raise ValueError(f"Invalid source record for {name}")
        digest = source.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Invalid SHA-256 for {name}")
        if type(source.get("pages")) is not int or source["pages"] < 1:
            raise ValueError(f"Invalid page count for {name}")
        if type(source.get("byte_size")) is not int or source["byte_size"] < 1:
            raise ValueError(f"Invalid byte size for {name}")
        if "bundled_path" not in source:
            raise ValueError(f"Missing PDF distribution declaration for {name}: use a relative path or null")
        relative = source["bundled_path"]
        checked_sources = []
        if relative is not None:
            if (not isinstance(relative, str) or not relative or "\\" in relative
                    or Path(relative).is_absolute() or ".." in Path(relative).parts):
                raise ValueError(f"Invalid relative PDF path: {relative}")
            path = root / relative
            if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
                raise ValueError(f"Missing PDF or path outside repository: {relative}")
            _verify_pdf_bytes(path, source, relative)
            checked_sources.append("bundled_pdf")
        if name in local_pdfs:
            path = Path(local_pdfs[name])
            _verify_pdf_bytes(path, source, str(path))
            checked_sources.append("supplied_local_pdf")
        identities[name] = digest
        bytes_verified[name] = bool(checked_sources)
        verification_sources[name] = checked_sources or ["recorded_metadata"]
    table_keys = {key for key in reported if not key.startswith("_")}
    groups = index.get("table_groups", {})
    if not isinstance(groups, dict):
        raise ValueError("Table provenance must be an object")
    if set(groups) != table_keys:
        raise ValueError(f"Table provenance coverage differs: {sorted(set(groups) ^ table_keys)}")
    for key, group in groups.items():
        if (not isinstance(group, dict) or group.get("source") not in documents
                or group.get("evidence_kind") not in KINDS):
            raise ValueError(f"Invalid source/evidence kind: {key}")
        pages = group.get("pages", [])
        if (not isinstance(pages, list) or not pages
                or any(type(p) is not int or p < 1 or p > documents[group["source"]]["pages"] for p in pages)):
            raise ValueError(f"Invalid source page: {key}")
        if not group.get("label") or not group.get("note"):
            raise ValueError(f"Missing provenance interpretation: {key}")
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Non-finite number in reported registry")
        if isinstance(value, dict):
            for v in value.values():
                finite(v)
        elif isinstance(value, list):
            for v in value:
                finite(v)
    finite(reported)
    count = sum(bytes_verified.values())
    mode = "pdf_bytes_verified" if count == len(documents) else "mixed" if count else "reference_metadata_only"
    return {"pdf_sha256": identities, "pdf_bytes_verified": bytes_verified,
            "verification_sources": verification_sources, "verification_mode": mode,
            "table_groups_checked": len(groups),
            "scope": "Recorded PDF identities, reviewed page counts and table provenance; "
                     "PDF bytes checked only where pdf_bytes_verified is true. No experiment verification."}


def table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |",
                       "|" + "|".join("---" for _ in header) + "|"] +
                      ["| " + " | ".join(("—" if c is None else str(c)).replace("|", "\\|") for c in row) + " |" for row in rows]) + "\n"


def score_rows(reported):
    comparison = reported["supplementary_table_S15_method_comparison"]
    for dataset in DATASETS:
        block = comparison[dataset]
        native = next(r for r in block["rows"] if r["method"] == "full-depth plant")
        comparator = next(r for r in block["rows"] if "(qC" in r["method"])
        deployed = next(r for r in block["rows"] if r["family"] == "CROSS-MORSE deployment")
        yield dataset, block, native, comparator, deployed


def signed(value, digits=5):
    return f"{value:+.{digits}f}".replace("-", "−")


def interval(value, digits=5):
    return "[" + ", ".join(signed(x, digits) for x in value) + "]"


def render_overview(reported: dict, index: dict, registered: dict | None = None) -> str:
    """Create an optional compact view; the JSON remains the reference record."""
    registered = load_registered() if registered is None else registered
    rows, arms = [], []
    for dataset in DATASETS:
        row = resolved_policy_view(reported, dataset, registered)
        loss = ("0 (identity)" if row["comparator_identity"] else "unavailable" if row["loss_effect_nats"] is None
                else signed(row["loss_effect_nats"]) + " " + interval(row["loss_effect_interval_nats"]))
        gain = ("0 (identity)" if row["comparator_identity"] else "unavailable" if row["official_gain_pp"] is None
                else signed(row["official_gain_pp"], 2) + " " + interval(row["official_gain_interval_pp"], 2))
        rows.append([dataset, row["deployed_rule"], row["loss_name"], loss, gain, row["result_origin"]])
        arm = row["reported_arm"]
        if arm["deployed_rule"] != row["deployed_rule"]:
            arms.append([dataset, arm["deployed_rule"], signed(arm["loss_effect_nats"]) + " " + interval(arm["loss_effect_interval_nats"]),
                         signed(arm["official_gain_pp"], 2) + " " + interval(arm["official_gain_interval_pp"], 2), arm["official_score"]])
    lines = ["# Policy outcomes and reported arm results", "",
             "Generated from the reported values and registered frozen confirmation rule. Final-policy effects "
             "use comparator identity when that rule returns qC; these zeros are algebraic identities, not new measurements.", "",
             "Loss differences are final policy minus comparator, in nats. Metric gains use percentage points; "
             "brackets are reported paired 95% intervals. Absolute scores inherited from qC remain reported comparator references.", "",
             table(["Dataset", "Final rule", "Loss", "Loss difference [95% CI]", "Metric gain [95% CI], pp", "Origin"], rows)]
    if arms:
        lines += ["## Evaluated arms", "",
                  "These reported arm observations are separate from the final-policy outcomes above.", "",
                  table(["Dataset", "Evaluated arm", "Loss difference [95% CI], nats", "Metric gain [95% CI], pp", "Official score (S15 units)"], arms)]
    lines += ["The companion `paper_summary.csv` retains separately named evaluated-arm effects and costs. "
             "Blank numeric entries mean unavailable or not applicable. No benchmark is executed by this export.", "",
             "## Source identities", "",
             "`paper/sources.json` retains the document identities and table/page mappings. This table records "
             "those metadata; PDF byte verification requires a declared bundled file or an explicitly supplied local copy.", ""]
    sources = [[record.get("document", name), record["pages"], record["byte_size"], record["sha256"],
                record["bundled_path"] or "not distributed"] for name, record in index["documents"].items()]
    lines.append(table(["Document", "Reviewed pages", "Bytes", "SHA-256", "Bundled path"], sources))
    return "\n".join(lines) + "\n"


def render_csv(reported: dict, index: dict, registered: dict | None = None) -> str:
    registered = load_registered() if registered is None else registered
    out = io.StringIO(newline="")
    columns = ["dataset", "score_unit", "q0_score", "qN_score", "qC_score", "qD_score", "loss_name",
               "paired_qD_minus_qC_loss_nats", "loss_CI95_lower", "loss_CI95_upper", "metric_gain_pp",
               "metric_CI95_lower_pp", "metric_CI95_upper_pp", "qC_identity", "qD_route", "ensemble_E",
               "expected_cost", "expected_overhead_percent", "assurance_cost", "modelled_C099",
               "admission_scope", "registered_confirmation_pass", "lifecycle_resolution", "selection_status", "evidence_kind", "qD_origin",
               "evaluated_arm_route", "evaluated_arm_score", "evaluated_arm_loss_nats", "evaluated_arm_loss_CI95_lower", "evaluated_arm_loss_CI95_upper",
               "evaluated_arm_metric_gain_pp", "evaluated_arm_metric_CI95_lower_pp", "evaluated_arm_metric_CI95_upper_pp",
               "evaluated_arm_expected_cost", "evaluated_arm_assurance_cost", "evaluated_arm_G_admit",
               "effect_source", "score_source", "main_pdf_sha256", "supplementary_pdf_sha256"]
    writer = csv.DictWriter(out, columns, lineterminator="\n")
    writer.writeheader()
    for d, block, native, comparator, deployed in score_rows(reported):
        r = resolved_policy_view(reported, d, registered)
        c, arm = r["costs"], r["reported_arm"]
        loss_ci = r["loss_effect_interval_nats"] or [None, None]
        metric_ci = r["official_gain_interval_pp"] or [None, None]
        arm_loss_ci = arm["loss_effect_interval_nats"] or [None, None]
        arm_metric_ci = arm["official_gain_interval_pp"] or [None, None]
        writer.writerow(dict(dataset=d, score_unit=block["score_unit"], q0_score=None,
            qN_score=native["official_score"], qC_score=comparator["official_score"], qD_score=r["scores"]["qD"]["official_score"],
            loss_name=r["loss_name"], paired_qD_minus_qC_loss_nats=r["loss_effect_nats"],
            loss_CI95_lower=loss_ci[0], loss_CI95_upper=loss_ci[1], metric_gain_pp=r["official_gain_pp"],
            metric_CI95_lower_pp=metric_ci[0], metric_CI95_upper_pp=metric_ci[1], qC_identity=r["qC_identity"],
            qD_route=r["deployed_rule"], ensemble_E=r["deployed_E"], expected_cost=c["C_expected"],
            expected_overhead_percent=c["expected_overhead_pct"], assurance_cost=c["C_assure"], modelled_C099=c["C_099"],
            admission_scope=r["scope"], registered_confirmation_pass=r.get("registered_confirmation_pass"),
            lifecycle_resolution=r.get("lifecycle_resolution", ""), selection_status=r.get("selection_status", ""),
            evidence_kind="manuscript_reference; cost_model_separately_classified",
            qD_origin=r["result_origin"], evaluated_arm_route=arm["deployed_rule"], evaluated_arm_score=arm["official_score"],
            evaluated_arm_loss_nats=arm["loss_effect_nats"], evaluated_arm_loss_CI95_lower=arm_loss_ci[0], evaluated_arm_loss_CI95_upper=arm_loss_ci[1],
            evaluated_arm_metric_gain_pp=arm["official_gain_pp"], evaluated_arm_metric_CI95_lower_pp=arm_metric_ci[0], evaluated_arm_metric_CI95_upper_pp=arm_metric_ci[1],
            evaluated_arm_expected_cost=arm["costs"]["C_expected"], evaluated_arm_assurance_cost=arm["costs"]["C_assure"], evaluated_arm_G_admit=arm["G_admit"],
            effect_source=r["decision_basis"]["source"] if r["decision_basis"] else "main Tables 1-2; comparator identity where returned",
            score_source="supplementary Table S15 pp11-12; qD inherits qC when the comparator is returned",
            main_pdf_sha256=index["documents"]["main"]["sha256"], supplementary_pdf_sha256=index["documents"]["supplementary"]["sha256"]))
    return out.getvalue()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate and, with --output-dir, check existing optional reports")
    parser.add_argument("--validate-only", action="store_true", help="validate metadata and any supplied PDFs without rendering")
    parser.add_argument("--output-dir", type=Path, help="explicit destination for optional REGISTRY_SUMMARY.md and paper_summary.csv")
    parser.add_argument("--manuscript-pdf", type=Path, help="local main manuscript copy to verify byte for byte")
    parser.add_argument("--supplementary-pdf", type=Path, help="local supplement copy to verify byte for byte")
    args = parser.parse_args(argv)
    if args.validate_only and args.output_dir:
        parser.error("--validate-only cannot be combined with --output-dir")
    try:
        reported = read_json(ROOT / "paper/reported_results.json")
        index = read_json(ROOT / "paper/sources.json")
        local = {name: path for name, path in (("main", args.manuscript_pdf), ("supplementary", args.supplementary_pdf)) if path}
        result = validate_sources(ROOT, reported, index, local)
        if args.output_dir:
            views = {args.output_dir / "REGISTRY_SUMMARY.md": render_overview(reported, index),
                     args.output_dir / "paper_summary.csv": render_csv(reported, index)}
            for path, content in views.items():
                if args.check:
                    if not path.is_file() or path.read_text(encoding="utf-8") != content:
                        raise ValueError(f"Stale generated view: {path}")
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content, encoding="utf-8")
        checked = sum(result["pdf_bytes_verified"].values())
        identity = (f"{checked} of {len(index['documents'])} manuscript PDF(s) verified byte for byte" if checked
                    else "reference metadata only; no PDF bytes verified")
        action = ("optional reports checked" if args.check else "optional reports written") if args.output_dir else "no reports written"
        print(f"{identity}; {result['table_groups_checked']} table groups; {action}. No benchmark execution performed.")
        return 0
    except (KeyError, TypeError, ValueError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
