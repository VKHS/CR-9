"""Read-only discovery for evidence_pipeline.py; no registration or author declarations.

The existing complete-run inventory checks the sealed bytes, producing source,
split membership and numeric archive structure. Discovery deliberately stops
short of preflight's statistical replay, support reconstruction and manuscript
binding. Multiple eligible paths are never ranked by age, name or outcome.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

from scripts._paper_common import DATASETS

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "cross-morse-evidence-discovery-1"
SCOPE = ("Candidate discovery and complete-run byte inventory only. No statistical replay, "
         "checkpoint execution, support reconstruction, manuscript agreement, author attestation "
         "or public-deposit verification is established.")


class DiscoveryError(ValueError):
    """The requested scan or its private output paths are unsafe or invalid."""


def _real_directory(path: Path, label: str) -> Path:
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise DiscoveryError(f"{label} must use a real directory path, not a symlink: {path}")
    path = path.resolve()
    if not path.is_dir():
        raise DiscoveryError(f"{label} is not a directory: {path}")
    return path


def _object(path: Path, relative: str) -> dict:
    from scripts.artifact_inventory import read_json, safe_file
    value = read_json(safe_file(path, relative))
    if not isinstance(value, dict):
        raise DiscoveryError(f"{relative} must be a JSON object")
    return value


def _content_identity(path: Path) -> str:
    """Use the preflight's stage-content identity without loading checkpoints."""
    from scripts.artifact_inventory import safe_file
    from scripts.archive_run import STAGES, sha256_file
    from scripts.evidence_preflight import canonical_sha256
    rows = []
    for index, stage in enumerate(STAGES):
        relative = f"provenance/stages/{index:02d}_{stage}.json"
        source = safe_file(path, relative)
        rows.append({"path": relative, "bytes": source.stat().st_size, "sha256": sha256_file(source)})
    return canonical_sha256(rows)


def _run(path: Path, *, campaign_member: bool = False) -> dict:
    """Inspect one leaf run; exceptions become explicit rejections at the caller."""
    from scripts.artifact_inventory import build_inventory
    config = _object(path, "config.json")
    dataset = config.get("dataset")
    reasons = []
    if dataset not in DATASETS:
        reasons.append(f"dataset {dataset!r} is not an official benchmark run; synthetic runs are excluded")
    if config.get("profile") != "full":
        reasons.append(f"profile {config.get('profile')!r} is not full (smoke and pilot runs are excluded)")
    if config.get("scientific_run") is not True:
        reasons.append("config.json does not declare scientific_run=true")
    if dataset == "Roman-empire" and not campaign_member:
        reasons.append("Roman-empire requires the complete parent campaign with split_00 through split_09")
    if reasons:
        raise DiscoveryError("; ".join(reasons))

    state = _object(path, "run_state.json")
    metadata = _object(path, "prepared/metadata.json")
    summary = _object(path, "final/results_summary.json")
    if state.get("status") != "COMPLETE" or state.get("last_complete_stage") != "evaluate":
        reasons.append("run_state.json does not mark the final evaluate stage COMPLETE")
    for name, value in (("run_state.json", state), ("final/results_summary.json", summary)):
        if (value.get("dataset") != dataset or value.get("profile") != "full"
                or value.get("scientific_run") is not True):
            reasons.append(f"{name} disagrees with the official full-profile scope")
    if (metadata.get("dataset") != dataset or metadata.get("is_benchmark_data") is not True
            or metadata.get("scientific_run") is not True):
        reasons.append("prepared/metadata.json does not identify official benchmark observations for this dataset")
    if any(value.get("benchmark_dataset", dataset) != dataset for value in (config, metadata, summary)):
        reasons.append("benchmark routing labels disagree with the dataset")
    if reasons:
        raise DiscoveryError("; ".join(reasons))

    before = _content_identity(path)
    inventory = build_inventory(path)
    after = _content_identity(path)
    if before != after:
        raise DiscoveryError("sealed stage content changed during discovery; inspect an immutable completed run")
    if (inventory.get("status") != "PASS" or inventory.get("dataset") != dataset
            or inventory.get("profile") != "full" or inventory.get("scientific_run") is not True):
        raise DiscoveryError("the complete-run inventory did not verify the declared official scope")
    return {"path": str(path), "kind": "run", "dataset": dataset, "status": "CANDIDATE",
            "reasons": [], "identities": inventory["identities"],
            "sealed_content_sha256": after, "source_resolution": inventory["source_resolution"],
            "verification_scope": inventory["verification_scope"], "seeds": inventory["seeds"],
            "sealed_artifacts": inventory["sealed_artifact_count"], "bytes": inventory["all_file_bytes"],
            "next": "Review this path and run evidence_pipeline.py preflight --full-audit always; no readiness verdict is implied."}


def inspect_candidate(path: Path, kind: str) -> dict:
    path = Path(path).absolute()
    try:
        path = _real_directory(path, "candidate root")
        if kind == "run":
            return _run(path)
        if kind != "campaign":
            raise DiscoveryError(f"unknown candidate kind: {kind}")
        from scripts.emit_campaign_binding import split_identities, split_paths
        paths = split_paths(path)
        runs = [_run(split, campaign_member=True) for split in paths]
        if any(row["dataset"] != "Roman-empire" for row in runs):
            raise DiscoveryError("a split_* campaign must contain only Roman-empire official runs")
        identities = split_identities(paths)
        if any(row["sealed_content_sha256"] != _content_identity(split) for split, row in zip(paths, runs)):
            raise DiscoveryError("a split's sealed content changed during campaign discovery")
        return {"path": str(path), "kind": "campaign", "dataset": "Roman-empire", "status": "CANDIDATE",
                "reasons": [], "identities": identities, "runs": runs,
                "sealed_content_sha256": [{"split": split.name, "sha256": row["sealed_content_sha256"]}
                                          for split, row in zip(paths, runs)],
                "bytes": sum(row["bytes"] for row in runs),
                "next": "Review the complete ten-split campaign and run evidence_pipeline.py preflight --full-audit always."}
    except (OSError, ValueError, KeyError, TypeError, IndexError, RuntimeError, ImportError, zipfile.BadZipFile) as exc:
        dataset = "Roman-empire" if kind == "campaign" else None
        if kind == "run" and not any(p.is_symlink() for p in (path, *path.parents)):
            try:
                config = _object(path, "config.json")
                label = config.get("benchmark_dataset", config.get("dataset"))
                dataset = label if label in DATASETS else None
            except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError):
                pass
        return {"path": str(path), "kind": kind, "dataset": dataset, "status": "REJECTED",
                "reasons": [f"{type(exc).__name__}: {exc}"],
                "next": "Use the original complete full-profile output and its verified producing source; do not edit seals or relabel an example."}


def discover(root: Path, *, max_depth: int = 6, max_directories: int = 10000) -> dict:
    """Scan only this root, in stable path order, stopping at run/campaign leaves.

    A traversal error or bound prevents any uniqueness claim: there may be an
    unvisited second candidate. Symlinks are reported and never followed.
    """
    root = _real_directory(root, "discovery root")
    if type(max_depth) is not int or max_depth < 0 or type(max_directories) is not int or max_directories < 1:
        raise DiscoveryError("max_depth must be nonnegative and max_directories must be positive integers")
    candidates, issues, excluded = [], [], []
    visited = 0

    def on_error(error):
        issues.append({"path": str(error.filename or root), "reason": f"directory could not be read: {error}"})

    for folder, directories, filenames in os.walk(root, topdown=True, followlinks=False, onerror=on_error):
        path = Path(folder)
        directories.sort()
        if visited >= max_directories:
            issues.append({"path": str(path), "reason": f"max_directories={max_directories} reached before this directory"})
            break
        visited += 1
        split_names = [name for name in directories if name.startswith("split_")]
        run_marker = ("run_state.json" in filenames or (path / "provenance/stages").exists()
                      or (path / "final/results_summary.json").exists())
        if split_names and run_marker:
            candidates.append({"path": str(path), "kind": "ambiguous_layout", "dataset": None,
                               "status": "REJECTED", "reasons": ["directory has both a run and split_* campaign layout"],
                               "next": "Separate the original run and campaign roots; do not choose a layout implicitly."})
            directories[:] = []
            continue
        if split_names or run_marker or "config.json" in filenames:
            candidates.append(inspect_candidate(path, "campaign" if split_names else "run"))
            if split_names or run_marker:
                directories[:] = []
                continue
            # A bare configuration can be an interrupted run or a wrapper for
            # real runs. Record its rejection without hiding nested candidates.
        kept = []
        for name in directories:
            child = path / name
            if child.is_symlink():
                excluded.append({"path": str(child), "reason": "symlink directory excluded; links are never followed"})
            else:
                kept.append(name)
        directories[:] = kept
        depth = len(path.relative_to(root).parts)
        if depth >= max_depth and directories:
            issues.append({"path": str(path), "reason": f"max_depth={max_depth} reached with unvisited directories",
                           "unvisited": list(directories)})
            directories[:] = []

    candidates.sort(key=lambda row: (row["path"], row["kind"]))
    complete = not issues
    datasets = []
    for dataset in DATASETS:
        eligible = [row["path"] for row in candidates if row["dataset"] == dataset and row["status"] == "CANDIDATE"]
        rejected = [row["path"] for row in candidates if row["dataset"] == dataset and row["status"] == "REJECTED"]
        status = ("UNRESOLVED_SCAN" if not complete else "AMBIGUOUS" if len(eligible) > 1
                  else "UNIQUE_CANDIDATE" if eligible else "MISSING")
        datasets.append({"dataset": dataset, "status": status, "candidates": eligible, "rejected": rejected,
                         "proposed_run_root": eligible[0] if status == "UNIQUE_CANDIDATE" else None})
    count = sum(row["status"] == "UNIQUE_CANDIDATE" for row in datasets)
    ambiguous = sum(row["status"] == "AMBIGUOUS" for row in datasets)
    status = ("INCOMPLETE_SCAN" if not complete else "REVIEW_REQUIRED" if ambiguous
              else "CANDIDATES_FOUND" if count else "NO_CANDIDATES")
    return {"schema": SCHEMA, "scope": SCOPE, "root": str(root), "status": status,
            "scan_complete": complete, "limits": {"max_depth": max_depth, "max_directories": max_directories},
            "directories_visited": visited, "issues": sorted(issues, key=lambda row: (row["path"], row["reason"])),
            "excluded": sorted(excluded, key=lambda row: row["path"]), "candidates": candidates, "datasets": datasets,
            "unique_datasets": count, "ambiguous_datasets": ambiguous,
            "next": "Review paths and unresolved datasets, then use the existing plan-driven preflight. Discovery is not registration."}


def proposed_plan(report: dict, deposits: Path) -> dict:
    from scripts.evidence_pipeline import template
    deposits = Path(deposits).absolute()
    if any(p.is_symlink() for p in (deposits, *deposits.parents)):
        raise DiscoveryError("deposits_dir must not use symlink paths")
    deposits = deposits.resolve()
    if deposits.is_relative_to(ROOT) or deposits.is_relative_to(Path(report["root"])):
        raise DiscoveryError("deposits_dir must be outside the checkout and the scanned source root")
    if deposits.exists() and not deposits.is_dir():
        raise DiscoveryError("deposits_dir is not a directory")
    plan = template()
    plan["deposits_dir"] = str(deposits)
    found = {row["path"]: row for row in report["candidates"] if row["status"] == "CANDIDATE"}
    for row in report["datasets"]:
        path = row["proposed_run_root"]
        if path is not None:
            candidate = found[path]
            plan["datasets"][row["dataset"]].update(run_root=path, _discovery={
                "identities": candidate["identities"], "sealed_content_sha256": candidate["sealed_content_sha256"],
                "scope": "Proposed unique candidate only; preflight must recheck the original artifacts."})
    plan["_discovery"] = {"root": report["root"], "scan_complete": report["scan_complete"],
                          "status": report["status"], "scope": SCOPE,
                          "report_canonical_sha256": hashlib.sha256(json.dumps(report, sort_keys=True, separators=(",", ":"),
                                                                     allow_nan=False).encode()).hexdigest()}
    return plan


def markdown(report: dict) -> str:
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    lines = ["# Evidence discovery", "", f"**{report['status']}** — {report['unique_datasets']}/7 unique dataset candidates; "
             f"{report['ambiguous_datasets']} ambiguous dataset(s).", "", report["scope"], "",
             f"Root: `{cell(report['root'])}`. Only real directories are scanned; run/campaign payloads are leaves.", "",
             "| Dataset | Status | Proposed path |", "|---|---|---|"]
    for row in report["datasets"]:
        lines.append(f"| {cell(row['dataset'])} | {row['status']} | {cell(row['proposed_run_root'] or 'unset')} |")
    lines += ["", "## Examined candidates", "", "| Path | Dataset | Status | Reason / next action |", "|---|---|---|---|"]
    for row in report["candidates"]:
        detail = "; ".join(row["reasons"]) if row["reasons"] else row["next"]
        lines.append(f"| {cell(row['path'])} | {cell(row['dataset'] or 'unknown')} | {row['status']} | {cell(detail)} |")
    if report["issues"] or report["excluded"]:
        lines += ["", "## Scan limits and exclusions", ""]
        for row in report["issues"] + report["excluded"]:
            lines.append(f"- `{cell(row['path'])}`: {cell(row['reason'])}")
    lines += ["", "Duplicate candidates remain unset; no loss, timestamp or filename is used to select a winner. "
              "A traversal error or limit leaves every proposed path unset. Missing datasets and all ten measurement "
              "sets still require original evidence. Review the draft plan's licence and manuscript relationship yourself; "
              "discovery fills neither those declarations nor DOI, record-export or campaign fields.", ""]
    return "\n".join(lines)


def write_outputs(report: dict, output: Path, *, plan_output: Path | None = None,
                  deposits: Path | None = None) -> None:
    """Create private reports/plan exclusively, outside both checkout and sources."""
    if (plan_output is None) != (deposits is None):
        raise DiscoveryError("--plan-output and --deposits-dir must be supplied together")
    if Path(output).suffix != ".json" or plan_output is not None and Path(plan_output).suffix != ".json":
        raise DiscoveryError("discovery report and plan outputs must end in .json")
    source = Path(report["root"])
    paths = [Path(output), Path(output).with_suffix(".md")]
    if plan_output is not None:
        paths.append(Path(plan_output))
    targets = []
    for path in paths:
        path = path.absolute()
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise DiscoveryError("private output paths must not be symlinks")
        path = path.resolve()
        if path.is_relative_to(ROOT) or path.is_relative_to(source):
            raise DiscoveryError("write discovery outputs outside the checkout and scanned source root")
        if path.exists():
            raise DiscoveryError(f"refusing to overwrite {path}; choose new output paths")
        targets.append(path)
    if len(set(targets)) != len(targets):
        raise DiscoveryError("report, Markdown and plan paths must be distinct")
    texts = [json.dumps(report, indent=1, sort_keys=True, allow_nan=False) + "\n", markdown(report)]
    if plan_output is not None:
        texts.append(json.dumps(proposed_plan(report, deposits), indent=1, sort_keys=True, allow_nan=False) + "\n")
    created = []
    try:
        for index, (path, text) in enumerate(zip(targets, texts)):
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".discovery-", suffix=".json",
                                             encoding="utf-8", delete=False) as stream:
                temporary = Path(stream.name)
                try:
                    os.fchmod(stream.fileno(), 0o600)
                    stream.write(text)
                    stream.flush()
                    os.fsync(stream.fileno())
                    if index == 2:
                        from scripts.evidence_pipeline import load_plan
                        load_plan(temporary)
                    os.link(temporary, path)  # publish completed bytes without replacing another writer's file
                    created.append(path)
                finally:
                    temporary.unlink(missing_ok=True)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
