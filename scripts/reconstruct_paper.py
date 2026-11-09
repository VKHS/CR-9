#!/usr/bin/env python3
"""One command: reconstruct every table and figure of the paper from its published outputs.

    make paper                                  # omits heavy lifecycle checkpoint/prediction files
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

``make demo-paper`` runs the same steps on the CPU demonstration (``--demo``): its own
manifest ``release/demo_artifacts.json`` and release ``demo-artifacts``, a workspace
labelled as the demonstration and a report that opens with its scope. Its values are not
the paper's measurements; their comparison with the printed values is reported, and the
exit status counts only verification and generation (0: every published demonstration
artifact verified and every generator ran).

Exit status 0 means every display was generated in full from the published outputs
and every required value has a comparable printed reference and agrees; 1 means
outputs or comparison definitions are missing, a value differs, or content is
incomplete (the report names which); 2 means an input failed verification or a
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


def display_summary(result: dict) -> dict:
    """Summarize reconstruction itself, independently of how its inputs arrived."""
    generators = result.get("display_generators") or {}
    comparison = generators.get("comparison_with_printed_values") or {}
    full = [d["id"] for d in result["displays"] if d["full_numerical_display"]]
    return {"displays": len(result["displays"]), "complete": len(full),
            "incomplete": [d["id"] for d in result["displays"] if not d["full_numerical_display"]],
            "values_exact": comparison.get("exact", 0), "values_differ": comparison.get("differs", 0),
            "values_not_comparable": comparison.get("not_comparable", 0),
            "values_no_reference": comparison.get("no_reference", 0),
            "values_missing": comparison.get("missing", 0),
            "comparison_coverage_complete": generators.get("comparison_coverage_complete") is True,
            "display_content_incomplete": [d["id"] for d in result["displays"]
                                           if d.get("content_complete") is not True],
            "generator_errors": generators.get("errors") or {}, "module_errors": generators.get("module_errors") or {}}


def summarize(result: dict, manifest: dict) -> dict:
    return {**display_summary(result),
            "artifacts_awaiting_publication": [a["id"] for a in manifest["artifacts"]
                if a.get("required", True) and a.get("status") != artifacts.PUBLISHED]}


def materialization_status(verification: dict, manifest: dict, tier: str) -> dict:
    """A published descriptor or an analysis cache does not supply complete bytes.

    Canonical artifact requirements survive missing manifest rows and edited
    ``required`` flags. Optional extra artifacts do not block reconstruction.
    A complete cache also satisfies an analysis request, never the reverse.
    """
    required = {row["id"] for row in artifacts.required_artifacts()}
    required.update(row["id"] for row in manifest["artifacts"] if row.get("required", True))
    entries = {row["id"]: row for row in manifest["artifacts"]}
    published = {key for key, row in entries.items() if row.get("status") == artifacts.PUBLISHED}
    present = {row["id"]: row for row in verification["artifacts"]}
    missing = (required & published) & (set(verification["awaiting"]) | (required - set(present)))
    incomplete_tiers = [{"id": key, "available_tier": present[key]["tier"], "requested_tier": tier}
                        for key in sorted(required & published & set(present))
                        if tier == "complete" and present[key]["tier"] != "complete"]
    return {"requested_tier": tier, "required_artifacts": sorted(required),
            "not_registered": sorted(required - set(entries)),
            "not_published": sorted(required - published),
            "missing_local": sorted(missing), "incomplete_tiers": incomplete_tiers,
            "verified_tiers": {key: row["tier"] for key, row in sorted(present.items())}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", type=Path, default=ROOT.parent / "cm-artifacts",
                        help="where published outputs are downloaded and unpacked")
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "cm-paper",
                        help="the reconstruction workspace (tables, figures, source data, comparisons)")
    parser.add_argument("--tier", choices=["analysis", "complete"], default="analysis")
    parser.add_argument("--manifest", type=Path, default=artifacts.MANIFEST)
    parser.add_argument("--offline", action="store_true", help="use already downloaded artifacts only")
    parser.add_argument("--demo", action="store_true",
                        help="the CPU demonstration (release/demo_artifacts.json); not the paper's measurements")
    args = parser.parse_args(argv)
    if args.demo and args.manifest == artifacts.MANIFEST:
        args.manifest = artifacts.DEMO_MANIFEST
    label = (f"demo-artifacts: CPU demonstration at reduced scale, not the paper's measurements "
             f"(requested {args.tier} tier)" if args.demo else f"paper-artifacts (requested {args.tier} tier)")
    try:
        manifest = artifacts.load_manifest(args.manifest)
        if (manifest.get("channel") == artifacts.DEMO_CHANNEL) != bool(args.demo):
            raise artifacts.ArtifactError("The demonstration manifest is reconstructed with --demo (make demo-paper), "
                                          "the paper's manifest without it (make paper)")
        published = [a for a in manifest["artifacts"] if a.get("status") == artifacts.PUBLISHED]
        if published and not args.offline:
            report = artifacts.fetch(args.artifacts, manifest_path=args.manifest, tier=args.tier)
            for row in report["published"]:
                print(f"verified {row['id']} ({row['tier']})")
        # Offline reuse must satisfy the same current release inventory and
        # artifact identity as a fresh fetch; a prior download stamp is not proof.
        verified = artifacts.verify_fetched(args.artifacts, manifest_path=args.manifest, require_all=False)
        materialization = materialization_status(verified, manifest, args.tier)
        found = inputs_from(args.artifacts)
        campaigns = dict(found["campaigns"])
        result = reconstruct(args.output_dir, runs_root=found["runs_root"], records_index=found["records_index"],
                             confirmation_campaign=campaigns.pop("confirmation", None),
                             budget_campaign=campaigns.pop("budget", None), campaigns=campaigns,
                             label=label)
    except (artifacts.ArtifactError, ReconstructionError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    status = summarize(result, manifest)
    status["artifacts_awaiting_publication"] = materialization["not_published"]
    status["artifact_materialization"] = materialization
    if args.demo:
        status["channel"] = artifacts.DEMO_CHANNEL
        status["scope"] = manifest.get("scope", artifacts.DEMO_SCOPE)
        report = args.output_dir / "RECONSTRUCTION_REPORT.md"
        if report.is_file():
            report.write_text(f"> **Demonstration.** {status['scope']}\n\n" + report.read_text(encoding="utf-8"),
                              encoding="utf-8")
    status_path = args.output_dir.parent / f"{args.output_dir.name}.status.json"
    status_path.write_text(json.dumps(status, indent=1) + "\n", encoding="utf-8")
    print(f"Displays complete: {status['complete']} of {status['displays']}")
    print(f"Printed values reproduced: {status['values_exact']} exact, {status['values_differ']} differ, "
          f"{status['values_not_comparable']} not comparable, {status['values_no_reference']} without a reference, "
          f"{status['values_missing']} missing")
    if status["artifacts_awaiting_publication"]:
        print("Awaiting publication: " + ", ".join(status["artifacts_awaiting_publication"]))
    if materialization["not_registered"]:
        print("Required artifacts absent from the manifest: " + ", ".join(materialization["not_registered"]))
    if materialization["missing_local"]:
        print("Published artifacts not available locally: " + ", ".join(materialization["missing_local"]))
    if materialization["incomplete_tiers"]:
        print("Complete-tier files still needed: " + ", ".join(row["id"] for row in materialization["incomplete_tiers"]))
    for name, error in {**status["generator_errors"], **status["module_errors"]}.items():
        print(f"Generator failure: {name}: {error}", file=sys.stderr)
    print(f"Report: {args.output_dir / 'RECONSTRUCTION_REPORT.md'}; status: {status_path}")
    if status["generator_errors"] or status["module_errors"]:
        return 2
    if args.demo:
        print("Demonstration: values from reduced-scale CPU runs; differences from the printed values are expected.")
        return 1 if materialization["missing_local"] or materialization["incomplete_tiers"] else 0
    if materialization["missing_local"] or materialization["incomplete_tiers"]:
        print("Fetch the requested artifact tier before treating this offline reconstruction as complete.",
              file=sys.stderr)
        return 1
    if status["artifacts_awaiting_publication"]:
        print(f"{len(status['artifacts_awaiting_publication'])} artifact(s) await publication of the measured "
              "outputs (hpc/produce_paper_outputs_arrhenius.sh, then hpc/publish_paper_artifacts_arrhenius.sh all); "
              "displays that need them are not yet complete.",
              file=sys.stderr)
        return 1
    if (status["incomplete"] or status["values_differ"] or status["values_not_comparable"]
            or status["values_no_reference"] or status["values_missing"]
            or not status["comparison_coverage_complete"] or status["display_content_incomplete"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
