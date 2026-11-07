#!/usr/bin/env python3
"""One command: reconstruct every table and figure of the paper from its published outputs.

    make paper                                  # analysis tier (no checkpoints, no prediction arrays)
    make paper PAPER_TIER=complete              # everything, including checkpoints and predictions
    python -B scripts/reconstruct_paper.py --artifacts ../cm-artifacts --output-dir ../cm-paper

Steps:
1. ``release/paper_artifacts.json`` lists the measured outputs published with the
   GitHub release ``paper-artifacts`` of this repository (runs, record sets,
   campaigns, observed environment), each with the SHA-256 of every part, archive
   and file.
2. Every published artifact is downloaded and verified byte for byte, then unpacked
   (``scripts/paper_artifacts.py fetch``).
3. ``scripts/reconstruct.py`` verifies every run's sealed eight-stage chain and every
   record set's trace to its raw files, and the display generators
   (``paper/build/displays``) write all numbered tables and figures with their source
   data.
4. Every computed value that the paper prints is compared with the printed value at
   the printed precision.

Exit status 0 means every display was generated in full from the published outputs
and every compared value agrees; 1 means outputs are still awaiting publication or a
value differs (the report names which); 2 means an input failed verification or a
generator failed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import paper_artifacts as artifacts  # noqa: E402
from scripts.reconstruct import ReconstructionError, reconstruct  # noqa: E402


def inputs_from(directory: Path) -> dict:
    """Locate fetched runs, record sets and campaigns under an artifact directory."""
    directory = Path(directory)
    runs = directory / "runs"
    records = directory / "records" / "index.json"
    campaigns = directory / "campaigns"
    return {
        "runs_root": runs if runs.is_dir() and any(p.is_dir() and not p.name.startswith(".")
                                                   for p in runs.iterdir()) else None,
        "records_index": records if records.is_file() else None,
        "campaigns": ({p.name: p for p in sorted(campaigns.iterdir()) if p.is_dir() and not p.name.startswith(".")}
                      if campaigns.is_dir() else {}),
    }


def summarize(result: dict, manifest: dict) -> dict:
    generators = result.get("display_generators") or {}
    comparison = generators.get("comparison_with_printed_values") or {}
    awaiting = [a["id"] for a in manifest["artifacts"] if a.get("status") != artifacts.PUBLISHED]
    full = [d["id"] for d in result["displays"] if d["full_numerical_display"]]
    return {"displays": len(result["displays"]), "complete": len(full),
            "incomplete": [d["id"] for d in result["displays"] if not d["full_numerical_display"]],
            "values_exact": comparison.get("exact", 0), "values_differ": comparison.get("differs", 0),
            "values_not_comparable": comparison.get("not_comparable", 0),
            "artifacts_awaiting_publication": awaiting,
            "generator_errors": generators.get("errors") or {}, "module_errors": generators.get("module_errors") or {}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", type=Path, default=ROOT.parent / "cm-artifacts",
                        help="where published outputs are downloaded and unpacked")
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "cm-paper",
                        help="the reconstruction workspace (tables, figures, source data, comparisons)")
    parser.add_argument("--tier", choices=["analysis", "complete"], default="analysis")
    parser.add_argument("--manifest", type=Path, default=artifacts.MANIFEST)
    parser.add_argument("--offline", action="store_true", help="use already downloaded artifacts only")
    args = parser.parse_args(argv)
    try:
        manifest = artifacts.load_manifest(args.manifest)
        published = [a for a in manifest["artifacts"] if a.get("status") == artifacts.PUBLISHED]
        if published and not args.offline:
            report = artifacts.fetch(args.artifacts, manifest_path=args.manifest, tier=args.tier)
            for row in report["published"]:
                print(f"verified {row['id']} ({row['tier']})")
        found = inputs_from(args.artifacts)
        campaigns = dict(found["campaigns"])
        result = reconstruct(args.output_dir, runs_root=found["runs_root"], records_index=found["records_index"],
                             confirmation_campaign=campaigns.pop("confirmation", None),
                             budget_campaign=campaigns.pop("budget", None), campaigns=campaigns,
                             label=f"paper-artifacts ({args.tier} tier)")
    except (artifacts.ArtifactError, ReconstructionError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    status = summarize(result, manifest)
    status_path = args.output_dir.parent / f"{args.output_dir.name}.status.json"
    status_path.write_text(json.dumps(status, indent=1) + "\n", encoding="utf-8")
    print(f"Displays complete: {status['complete']} of {status['displays']}")
    print(f"Printed values reproduced: {status['values_exact']} exact, {status['values_differ']} differ, "
          f"{status['values_not_comparable']} not comparable")
    if status["artifacts_awaiting_publication"]:
        print("Awaiting publication: " + ", ".join(status["artifacts_awaiting_publication"]))
    for name, error in {**status["generator_errors"], **status["module_errors"]}.items():
        print(f"Generator failure: {name}: {error}", file=sys.stderr)
    print(f"Report: {args.output_dir / 'RECONSTRUCTION_REPORT.md'}; status: {status_path}")
    if status["generator_errors"] or status["module_errors"]:
        return 2
    if status["artifacts_awaiting_publication"]:
        print(f"{len(status['artifacts_awaiting_publication'])} artifact(s) await publication of the measured "
              "outputs (hpc/publish_paper_artifacts_arrhenius.sh); displays that need them are not yet complete.",
              file=sys.stderr)
        return 1
    if status["incomplete"] or status["values_differ"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
