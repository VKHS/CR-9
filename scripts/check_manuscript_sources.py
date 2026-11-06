#!/usr/bin/env python3
"""Check manuscript reference identities and page provenance.

An explicit null bundled_path declares a PDF that is not distributed. Its hash,
size and reviewed page count are checked as metadata. Declared bundled PDFs and
any supplied local copies must match their recorded bytes. Neither mode verifies
experimental measurements.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))
from scripts.render_paper_results import read_json, validate_sources


def inspect_sources(root: Path = ROOT, local_pdfs: dict | None = None) -> dict:
    root = Path(root).resolve()
    report = {
        "schema": "cross-morse-manuscript-source-check-1",
        "status": "FAIL",
        "scope": "Recorded PDF identities, reviewed page counts and table provenance; "
                 "PDF bytes checked only where pdf_bytes_verified is true. No experiment verification.",
        "documents": {},
        "issues": [],
    }
    try:
        source = read_json(root / "paper/sources.json")
        reported = read_json(root / "paper/reported_results.json")
        checked = validate_sources(root, reported, source, local_pdfs)
        report.update(checked)
        for role, record in source["documents"].items():
            report["documents"][role] = {
                "path": record["bundled_path"], "sha256": record["sha256"],
                "byte_size": record["byte_size"], "pages": record["pages"],
                "pdf_bytes_verified": checked["pdf_bytes_verified"][role],
                "verification_sources": checked["verification_sources"][role],
            }
        for name, registry in (("reported_results.json", reported),
                               ("registered_constants.json", read_json(root / "paper/registered_constants.json"))):
            provenance = registry.get("_provenance", {})
            for key, role in (("main_text_pages", "main"), ("supplementary_pages", "supplementary")):
                if key in provenance and provenance[key] != source["documents"][role]["pages"]:
                    report["issues"].append(f"{name}: {key} differs from the recorded source page count")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report["issues"].append(str(exc))
    report["status"] = "FAIL" if report["issues"] else "PASS"
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manuscript-pdf", type=Path, help="local main manuscript copy to verify byte for byte")
    parser.add_argument("--supplementary-pdf", type=Path, help="local supplement copy to verify byte for byte")
    args = parser.parse_args(argv)
    local = {name: path for name, path in (("main", args.manuscript_pdf), ("supplementary", args.supplementary_pdf)) if path}
    report = inspect_sources(args.root, local)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
