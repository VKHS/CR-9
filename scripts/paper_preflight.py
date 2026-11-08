#!/usr/bin/env python3
"""Reconstruct and assess original experiment folders before publishing archives.

    python -B scripts/paper_preflight.py --map /data/paper-artifacts.map.json \
        --work /data/new-preflight [--copy-mode hardlink|copy]

The input map is release/templates/paper-artifacts.map.json with absolute paths
and explicit _relationships. The workspace must be new and separate from every
input and the checkout. Hardlinks share verified bytes with the original files;
use copy for independent bytes or different filesystems. No source is chmodded.

This command neither uploads artifacts nor changes the publication manifest.
Exit 0 means locally ready for publication, not publicly available; 1 means
inputs or reconstruction are incomplete; 2 means invalid inputs or a failure.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import paper_artifacts as artifacts, records  # noqa: E402
from scripts._source_snapshots import resolve_runtime_source, runtime_source_fingerprint  # noqa: E402
from scripts.reconstruct import reconstruct, source_inventory, verify_workspace  # noqa: E402
from scripts.reconstruct_paper import display_summary, inputs_from  # noqa: E402
from scripts.statement_evidence import reconstruction_problems  # noqa: E402

SCHEMA = "cross-morse-publication-preflight-1"
ERRORS = (OSError, ValueError, KeyError, TypeError)


class PreflightError(ValueError):
    """The input map or workspace is unsafe or malformed."""


def _json(path: Path) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise PreflightError(f"Duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    def reject(value):
        raise PreflightError(f"Non-finite JSON value in {path}: {value}")

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=reject)


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _path(value, label, *, absolute=False) -> Path:
    if not isinstance(value, (str, Path)) or not str(value) or any(ord(c) < 32 for c in str(value)):
        raise PreflightError(f"{label}: a nonempty path without control characters is required")
    path = Path(value).expanduser()
    if absolute and not path.is_absolute():
        raise PreflightError(f"{label}: use an absolute directory path")
    path = path.absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise PreflightError(f"{label}: paths must not traverse symlinks: {path}")
    return path.resolve()


def _separate(work: Path, protected: list[Path]) -> None:
    if work.exists():
        raise PreflightError(f"Workspace already exists; choose a new directory: {work}")
    for path in protected:
        if work.is_relative_to(path) or path.is_relative_to(work):
            raise PreflightError(f"Workspace overlaps the checkout or an input: {path}")


def _inventory(root: Path) -> list[dict]:
    """Inventory every ordinary file, including any existing archive metadata."""
    _path(root, "Input directory")
    if not root.is_dir():
        raise PreflightError(f"Input directory is absent: {root}")
    rows = []
    for folder, dirs, names in os.walk(root):
        dirs.sort()
        for name in dirs:
            if (Path(folder) / name).is_symlink():
                raise PreflightError(f"Symlink inside input: {Path(folder) / name}")
        for name in sorted(names):
            path = Path(folder) / name
            if path.is_symlink() or not path.is_file():
                raise PreflightError(f"Input must contain only ordinary files: {path}")
            rows.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                         "sha256": artifacts.sha256_file(path)})
    return sorted(rows, key=lambda row: row["path"])


def _materialize(source: Path, target: Path, files: list[dict], mode: str) -> None:
    target.mkdir(parents=True, exist_ok=False)
    for row in files:
        origin, output = source / row["path"], target / row["path"]
        _path(origin, "Input file")
        if not origin.is_file():
            raise PreflightError(f"Input file disappeared during materialization: {origin}")
        output.parent.mkdir(parents=True, exist_ok=True)
        # No chmod or copystat on a hardlinked file: it is the original inode.
        if mode == "hardlink":
            try:
                os.link(origin, output, follow_symlinks=False)
            except OSError as exc:
                if exc.errno == errno.EXDEV:
                    raise PreflightError("Inputs and workspace are on different filesystems; rerun in a new "
                                         "workspace with --copy-mode copy") from exc
                raise
        else:
            shutil.copy2(origin, output, follow_symlinks=False)


def _runtime_sources(identifier: str, source: Path, identity: dict) -> list[dict]:
    """Keep exact old source, including snapshots beside an original run folder."""
    if not identifier.startswith("run:"):
        return []
    runs = ([(source / name, value) for name, value in identity["splits"].items()]
            if identity.get("campaign") else [(source, identity)])
    dependencies = []
    for child, value in runs:
        expected = value["source_sha256"]
        snapshot = resolve_runtime_source(expected, ROOT, run_root=child)
        checkout_source = snapshot == ROOT.resolve() or snapshot == (
            ROOT / "examples/source_snapshots" / expected).resolve()
        extra = not snapshot.is_relative_to(source) and not checkout_source
        relative = ((child.relative_to(source) / "source").as_posix() if extra else None)
        if relative and (source / relative).exists():
            raise PreflightError(f"Conflicting frozen source folder: {source / relative}")
        dependencies.append({"source": str(snapshot), "source_sha256": expected,
                             "additional_materialization": relative,
                             "files": _inventory(snapshot) if extra else []})
    return dependencies


def _expected_files(files: list[dict], runtime_sources: list[dict]) -> list[dict]:
    result = list(files)
    for dependency in runtime_sources:
        prefix = dependency["additional_materialization"]
        if prefix:
            result.extend({**row, "path": f"{prefix}/{row['path']}"} for row in dependency["files"])
    return sorted(result, key=lambda row: row["path"])


def _missing_input(exc: Exception) -> bool:
    text = str(exc)
    return isinstance(exc, FileNotFoundError) or any(value in text for value in (
        " is absent", " is missing", " missing (", "neither present nor a declared omission",
        "exact runtime source snapshot", "the runtime source", "not complete"))


def _report_markdown(report: dict) -> str:
    def escape(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["# Local publication preflight", "", f"**{report['verdict']}**", "",
             "This assessment reconstructs local inputs. Public availability has not been verified, "
             "and this report does not fulfill the code availability statement.", "",
             "READY_FOR_PUBLICATION means the local input contracts and registered reconstruction checks passed. "
             "Publication and release verification remain separate steps. Historical relationships are explicit "
             "author declarations; this check does not authenticate experiment history.", "",
             f"Materialization: **{report['copy_mode']}**. " + (
                 "The input workspace shares original file bytes through hardlinks; subsequent edits to either "
                 "path affect both. It is not an independent immutable copy."
                 if report["copy_mode"] == "hardlink" else
                 "The input workspace contains separate copies, verified against the original bytes."), "",
             "Original identities, source inventories and materialized bytes are checked again after reconstruction.", "",
             "## Inputs", "", "| Artifact | Local verification | Declared relationship |",
             "| --- | --- | --- |"]
    for row in report["artifacts"]:
        lines.append(f"| {escape(row['id'])} | {escape(row['status'])} | "
                     f"{escape(row.get('relationship') or 'declaration needed')} |")
    lines.extend(["", "## Findings", ""])
    if report["issues"]:
        lines.extend(f"- **{escape(row['scope'])}**: {escape(row['message'])}" for row in report["issues"])
    else:
        lines.append("All local checks passed. The report makes no public availability claim.")
    lines.extend(["", "## Reconstruction", ""])
    if report["reconstruction"].get("summary"):
        summary = report["reconstruction"]["summary"]
        lines.extend([f"Displays generated in full: {summary['complete']}.", "",
                      f"Compared values: {summary['values_exact']} exact; {summary['values_differ']} differ; "
                      f"{summary['values_missing']} missing.", "",
                      "See [the reconstruction report](reconstruction/RECONSTRUCTION_REPORT.md) "
                      "for display-level inputs, outputs and comparisons."])
    else:
        lines.append("Reconstruction did not finish; the findings above identify the failure.")
    lines.extend(["", "[Machine-readable report](report.json) includes input paths, identity hashes, "
                  "verification results and links to the complete inventories.", ""])
    return "\n".join(lines)


def assess(map_path: Path, work: Path, *, copy_mode="hardlink", _software_fixture=False) -> dict:
    """Read-only assessment of supplied evidence; fixture scope is not a CLI option."""
    if copy_mode not in {"hardlink", "copy"}:
        raise PreflightError("copy_mode must be hardlink or copy")
    map_path, work = _path(map_path, "Input map"), _path(work, "Workspace")
    map_sha = artifacts.sha256_file(map_path)
    mapping = _json(map_path)
    if not isinstance(mapping, dict) or not isinstance(mapping.get("_relationships", {}), dict):
        raise PreflightError("Input map must be an object with an object-valued _relationships")
    required = artifacts.required_artifacts()
    ids = {row["id"] for row in required}
    relationships = mapping.get("_relationships", {})
    unknown = {key for key in mapping if not key.startswith("_")} - ids
    if unknown or set(relationships) - ids:
        raise PreflightError(f"Unknown artifact IDs in input map: {sorted(unknown | (set(relationships) - ids))}")
    paths, protected = {}, [ROOT, map_path]
    for identifier in sorted(ids):
        value = mapping.get(identifier, "")
        if not isinstance(value, str):
            raise PreflightError(f"{identifier}: map entry must be an absolute path or an empty string")
        relationship = relationships.get(identifier, "")
        if not isinstance(relationship, str) or (relationship and relationship not in artifacts.RELATIONSHIPS):
            raise PreflightError(f"{identifier}: invalid historical relationship {relationship!r}")
        if value:
            paths[identifier] = _path(value, identifier, absolute=True)
            protected.append(paths[identifier])
            if identifier.startswith("run:"):
                protected.extend((paths[identifier] / "source", paths[identifier].parent / "source"))
    _separate(work, protected)
    before = source_inventory(ROOT)
    manifest_sha = artifacts.sha256_file(artifacts.MANIFEST)
    if artifacts.sha256_file(map_path) != map_sha:
        raise PreflightError("Input map changed while reading it")
    work.mkdir(parents=True, exist_ok=False)
    report = {"schema": SCHEMA, "scope": "local publication preflight",
              "software_fixture": bool(_software_fixture), "public_availability_verified": False,
              "availability_statement_fulfilled": False, "historical_identity_authenticated": False,
              "input_map": {"path": str(map_path), "sha256": map_sha}, "workspace": str(work),
              "copy_mode": copy_mode, "artifacts": [], "issues": [], "reconstruction": {},
              "builder_sources_before": before, "publication_manifest_sha256_before": manifest_sha}
    checks = []

    def issue(scope, message, severity="invalid"):
        report["issues"].append({"scope": scope, "severity": severity, "message": str(message)})

    for contract in required:
        identifier = contract["id"]
        source = paths.get(identifier)
        row = {"id": identifier, "kind": artifacts._kind(identifier), "status": "MISSING",
               "relationship": relationships.get(identifier) or None,
               "source": str(source) if source is not None else None}
        report["artifacts"].append(row)
        if not row["relationship"]:
            issue(identifier, "Declare the original-run or post-result relationship in _relationships.", "incomplete")
        if source is None or not source.exists():
            issue(identifier, "Supply the existing original artifact directory in the input map.", "incomplete")
            continue
        if not source.is_dir():
            row["status"] = "INVALID"
            issue(identifier, "Mapped input must be a directory")
            continue
        target = artifacts._materialized_path(work / "inputs", row)
        try:
            identity = artifacts._verify_source(identifier, source, scientific_only=not _software_fixture)
            if row["relationship"] == "exact_result_producing_run" and identity.get("post_hoc") is True:
                raise PreflightError("A post hoc campaign cannot be declared the original result-producing campaign")
            files = _inventory(source)
            dependencies = _runtime_sources(identifier, source, identity)
            _materialize(source, target, files, copy_mode)
            for dependency in dependencies:
                if dependency["additional_materialization"]:
                    _materialize(Path(dependency["source"]), target / dependency["additional_materialization"],
                                 dependency["files"], copy_mode)
            expected = _expected_files(files, dependencies)
            inventory = {"schema": SCHEMA + "-inventory", "artifact": identifier, "identity": identity,
                         "source": str(source), "relationship": row["relationship"], "files": files,
                         "runtime_sources": dependencies, "materialized_inventory_sha256": _digest(expected)}
            inventory_file = Path("inventories") / f"{artifacts.asset_stem(identifier)}.json"
            artifacts.write_json(work / inventory_file, inventory)
            row.update(status="VERIFIED", identity=identity, inventory_file=inventory_file.as_posix(),
                       inventory_sha256=_digest(files), files=len(files), bytes=sum(f["bytes"] for f in files),
                       materialized_path=str(target), runtime_sources=[{k: v for k, v in d.items() if k != "files"}
                                                                     for d in dependencies])
            checks.append((row, source, target, identity, files, dependencies, expected))
        except ERRORS as exc:
            missing = _missing_input(exc) and not target.exists()
            row.update(status="INCOMPLETE" if missing else "INVALID", error=str(exc))
            issue(identifier, exc, "incomplete" if missing else "invalid")
            if target.exists():
                rejected = work / "rejected" / artifacts.asset_stem(identifier)
                rejected.parent.mkdir(exist_ok=True)
                target.rename(rejected)

    try:
        records_root = work / "inputs/records"
        if records_root.is_dir() and any(records_root.iterdir()):
            specs = records.load_specs(records_root, ROOT / "records/specs.json")
            entries = [records.verify_set(path, specs["sets"][path.name], specs)
                       for path in records.deposited_sets(records_root, specs)]
            records.write_index(records_root, specs, entries)
        found = inputs_from(work / "inputs")
        campaigns = dict(found["campaigns"])
        result = reconstruct(work / "reconstruction", runs_root=found["runs_root"],
                             records_index=found["records_index"],
                             confirmation_campaign=campaigns.pop("confirmation", None),
                             budget_campaign=campaigns.pop("budget", None), campaigns=campaigns,
                             label="local publication preflight" + (" (software fixture)" if _software_fixture else ""))
        summary = display_summary(result)
        problems = reconstruction_problems(summary, len(_json(ROOT / "paper/display_manifest.json")["displays"]))
        report["reconstruction"] = {"path": str(work / "reconstruction"), "summary": summary, "problems": problems}
        for problem in problems:
            issue("reconstruction", problem, "invalid" if problem in {"generator errors", "module errors"}
                  else "incomplete")
    except ERRORS as exc:
        report["reconstruction"]["error"] = str(exc)
        issue("reconstruction", exc)

    for row, source, target, identity, files, dependencies, expected in checks:
        try:
            if _inventory(source) != files or artifacts._verify_source(
                    row["id"], source, scientific_only=not _software_fixture) != identity:
                raise PreflightError("Original bytes or sealed identity changed during reconstruction")
            for dependency in dependencies:
                snapshot = _path(dependency["source"], "Frozen runtime source")
                if snapshot != ROOT and any(p.is_symlink() for p in snapshot.rglob("*")):
                    raise PreflightError("Original frozen runtime source acquired a symlink")
                if runtime_source_fingerprint(snapshot) != dependency["source_sha256"] or (
                        dependency["additional_materialization"] and _inventory(snapshot) != dependency["files"]):
                    raise PreflightError("Original frozen runtime source changed during reconstruction")
            row["source_unchanged_after_reconstruction"] = True
            if _inventory(target) != expected:
                raise PreflightError("Materialized input bytes changed during reconstruction")
            artifacts._verify_source(row["id"], target, scientific_only=not _software_fixture)
            row["materialization_unchanged_after_reconstruction"] = True
        except ERRORS as exc:
            row["status"] = "INVALID"
            issue(row["id"], exc)
    try:
        if report["reconstruction"].get("summary") is not None:
            verified = verify_workspace(work / "reconstruction")
            if display_summary(verified) != report["reconstruction"]["summary"]:
                raise PreflightError("Reconstruction summary changed during source re-verification")
            report["reconstruction"]["workspace_verified_after_inputs"] = True
        report["builder_sources_after"] = source_inventory(ROOT)
        report["publication_manifest_sha256_after"] = artifacts.sha256_file(artifacts.MANIFEST)
        if report["builder_sources_after"] != before:
            issue("builder", "Builder or registered paper sources changed during reconstruction")
        if report["publication_manifest_sha256_after"] != manifest_sha:
            issue("publication manifest", "Publication manifest changed during reconstruction")
        if artifacts.sha256_file(map_path) != map_sha:
            issue("input map", "Input map changed during reconstruction")
    except ERRORS as exc:
        issue("source re-verification", exc)
    invalid = any(row["severity"] == "invalid" for row in report["issues"])
    report["exit_code"] = 2 if invalid else 1 if report["issues"] else 0
    report["verdict"] = ("INVALID" if invalid else "INCOMPLETE" if report["issues"] else
                         "SOFTWARE_FIXTURE_COMPLETE" if _software_fixture else "READY_FOR_PUBLICATION")
    artifacts.write_json(work / "report.json", report)
    (work / "REPORT.md").write_text(_report_markdown(report), encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map", required=True, type=Path, help="artifact input map with explicit _relationships")
    parser.add_argument("--work", required=True, type=Path, help="new directory outside the checkout and inputs")
    parser.add_argument("--copy-mode", choices=["hardlink", "copy"], default="hardlink")
    args = parser.parse_args(argv)
    try:
        report = assess(args.map, args.work, copy_mode=args.copy_mode)
    except ERRORS as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Local publication preflight: {report['verdict']}")
    print(f"Report: {args.work / 'REPORT.md'}; status: {args.work / 'report.json'}")
    print("This command performs no upload. Public availability has not been verified.")
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
