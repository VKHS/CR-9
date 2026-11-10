#!/usr/bin/env python3
"""Publish and retrieve the paper's measured outputs as assets of this repository.

The registered availability statement places the checkpoints, predictions,
propensity-logged trajectories, statistical outputs, split and seed manifests and
the observed environment of the reported experiments in the CROSS-MORSE GitHub
repository. They are too large for Git, so they are attached to the GitHub release
``paper-artifacts`` of the repository and listed, with SHA-256 identities, in the
committed manifest ``release/paper_artifacts.json``.

    plan     list the artifacts every table and figure needs (from the display
             generators) and refresh the manifest; ``--check`` keeps it current
    pack     on the cluster: verify and archive completed runs, record sets and
             campaigns; split archives into parts below GitHub's 2 GiB asset limit
    publish  upload the packed parts to the release and record them in the manifest
    fetch    download, verify and unpack the published artifacts (``analysis`` tier:
             everything a display reads; ``complete`` tier: also checkpoints and
             prediction arrays)
    status   print what is published and what is still awaiting the measured outputs

A manifest entry is either ``awaiting_measured_outputs`` (nothing published yet) or
``published`` (asset names, byte counts and SHA-256 of every part, of the whole
archive and of every file inside it). Nothing here creates or alters a measurement.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "cross-morse-paper-artifacts-1"
MANIFEST = ROOT / "release" / "paper_artifacts.json"
REPOSITORY = "VKHS/CR-9"
TAG = "paper-artifacts"
PART_BYTES = 1900 * 1024 * 1024          # GitHub limit: 2 GiB per release asset
INVENTORY_NAME = "ARTIFACT_INVENTORY.json"
SUBSET_NAME = "ANALYSIS_SUBSET.json"
RUN_ARCHIVE_MANIFEST = "ARCHIVE_MANIFEST.json"
AWAITING = "awaiting_measured_outputs"
PUBLISHED = "published"
RELATIONSHIPS = ("exact_result_producing_run", "post_result_revision", "refactor_with_prediction_equivalence")
# The CPU demonstration (demo/README.md) is a separate channel: its own manifest and release, one relationship.
DEMO_MANIFEST = ROOT / "release" / "demo_artifacts.json"
DEMO_TAG = "demo-artifacts"
DEMO_CHANNEL = "demonstration"
DEMO_RELATIONSHIP = "reduced_scale_demonstration"
DEMO_SCOPE = ("CPU demonstration of the released code at reduced scale (demo/README.md). Every artifact was produced by "
              "the repository's own lifecycle and runners on public benchmark data on hosted CPU runners. The values "
              "are not the paper's measurements and are not expected to equal the printed values.")
DEMO_AWAITING = "not produced by the CPU demonstration (demo/README.md names the reason)"
HASH = re.compile(r"[0-9a-f]{64}\Z")
DATASETS = ("ogbn-arxiv", "ogbn-proteins", "ogbn-products", "Roman-empire",
            "PascalVOC-SP", "COCO-SP", "Peptides-func")
STEMS = {"ogbn-arxiv": "arxiv", "ogbn-products": "products", "ogbn-proteins": "proteins",
         "Roman-empire": "roman_empire", "PascalVOC-SP": "pascalvoc_sp", "COCO-SP": "coco_sp",
         "Peptides-func": "peptides_func"}
STATEMENT_ITEMS = {
    "run": ["configurations", "split and seed manifests", "environment specification", "checkpoints",
            "predictions", "propensity-logged trajectories", "statistical outputs", "source data"],
    "records": ["statistical outputs", "source data"],
    "campaign": ["statistical outputs", "predictions", "source data"],
    "environment": ["environment specification"],
}


class ArtifactError(ValueError):
    """An artifact, its manifest or a transfer does not satisfy the declared identities."""


def require(condition, message: str) -> None:
    if not condition:
        raise ArtifactError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, f"Duplicate JSON key in {path}: {key}")
            value[key] = item
        return value
    def reject(value):
        raise ArtifactError(f"Nonfinite JSON in {path}: {value}")
    value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=reject)
    # A projected file of destination/ (scripts/destination.py) is never a publication record.
    require(not (isinstance(value, dict) and "projection" in value),
            f"{path} is a projection from destination/, not a published file")
    return value


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=False, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# What the paper needs
# ---------------------------------------------------------------------------

def _kind(artifact_id: str) -> str:
    require(isinstance(artifact_id, str) and artifact_id.count(":") == 1, f"Invalid artifact ID: {artifact_id!r}")
    kind = artifact_id.split(":", 1)[0]
    require(kind in STATEMENT_ITEMS, f"Unknown artifact kind: {artifact_id}")
    name = artifact_id.split(":", 1)[1]
    require(bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name)), f"Invalid artifact name: {artifact_id}")
    require(kind != "run" or name in DATASETS, f"Unknown benchmark artifact: {artifact_id}")
    return kind


def relative_path(value: str, *, basename: bool = False) -> str:
    require(isinstance(value, str) and value and "\\" not in value and not any(ord(c) < 32 for c in value),
            f"Invalid artifact path: {value!r}")
    path = PurePosixPath(value)
    require(not path.is_absolute() and all(part not in {"", ".", ".."} for part in value.split("/"))
            and (not basename or len(path.parts) == 1), f"Invalid artifact path: {value!r}")
    return value


def contained(root: Path, relative: str, *, exists: bool = True) -> Path:
    path = Path(root) / relative_path(relative)
    for part in (path, *path.parents):
        if part == root:
            break
        require(not part.is_symlink(), f"Symlinks are not artifact inputs: {path}")
    require(path.resolve().is_relative_to(Path(root).resolve()), f"Artifact path escapes its directory: {relative}")
    require(not exists or path.is_file(), f"Artifact file is absent: {path}")
    return path


def _hash(value, label):
    require(isinstance(value, str) and HASH.fullmatch(value), f"Invalid SHA-256 for {label}")


def _files(rows: list, label: str) -> dict:
    require(isinstance(rows, list), f"Invalid file inventory: {label}")
    result = {}
    for row in rows:
        require(isinstance(row, dict), f"Invalid file descriptor: {label}")
        name = relative_path(row.get("path"))
        require(name not in result and type(row.get("bytes")) is int and row["bytes"] >= 0,
                f"Duplicate or invalid inventory file: {label}/{name}")
        _hash(row.get("sha256"), f"{label}/{name}")
        result[name] = row
    return result


def asset_stem(artifact_id: str) -> str:
    kind, _, name = artifact_id.partition(":")
    if kind == "run":
        return f"run-{STEMS[name]}"
    return f"{kind}-{name.replace('_', '-')}"


def required_artifacts(root: Path = ROOT) -> list[dict]:
    """Every artifact a display generator declares, plus the observed GPU environment."""
    from paper.build.displays.framework import GENERATORS, _load_generators
    _load_generators()
    needs = set()
    for generator in GENERATORS.values():
        needs.update(generator.needs)
    ids = {f"run:{d}" for d in DATASETS}            # the lifecycle runs carry checkpoints and trajectories
    for need in needs:
        kind, _, name = need.partition(":")
        if kind == "run":
            ids.add(f"run:{name}")
        elif kind in ("records", "campaign"):
            ids.add(f"{kind}:{name}")
    ids.add("environment:gpu")
    order = {"run": 0, "records": 1, "campaign": 2, "environment": 3}
    rows = []
    for artifact_id in sorted(ids, key=lambda a: (order[_kind(a)], DATASETS.index(a[4:]) if a.startswith("run:")
                                                 else 0, a)):
        kind, _, name = artifact_id.partition(":")
        displays = sorted({g.display_id for g in GENERATORS.values()
                           if artifact_id in g.needs or (kind == "run" and "runs" in g.needs)
                           or (kind == "records" and f"records:{name}" in g.needs)
                           or (kind == "campaign" and f"campaign:{name}" in g.needs)})
        rows.append({"id": artifact_id, "kind": kind, "name": name, "asset_stem": asset_stem(artifact_id),
                     "statement_items": STATEMENT_ITEMS[kind], "displays": displays})
    return rows


def blank_manifest(demonstration: bool = False) -> dict:
    tag = DEMO_TAG if demonstration else TAG
    return {"schema": SCHEMA, "repository": REPOSITORY, "release_tag": tag,
            "release_url": f"https://github.com/{REPOSITORY}/releases/tag/{tag}",
            "part_bytes": PART_BYTES,
            "statement": "release/availability_statement.json",
            **({"channel": DEMO_CHANNEL, "scope": DEMO_SCOPE} if demonstration else {}),
            "status": AWAITING, "artifacts": []}


def load_manifest(path: Path = MANIFEST) -> dict:
    if not Path(path).is_file():
        return blank_manifest(Path(path).resolve() == DEMO_MANIFEST.resolve())
    manifest = read_json(path)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict) -> dict:
    """Validate publication descriptors; this is not a check of remote or extracted bytes."""
    require(isinstance(manifest, dict) and manifest.get("schema") == SCHEMA, "Unexpected artifact manifest schema")
    require(manifest.get("status") in {AWAITING, PUBLISHED} and isinstance(manifest.get("artifacts"), list),
            "Invalid artifact manifest state")
    seen = set()
    for entry in manifest["artifacts"]:
        require(isinstance(entry, dict), "Invalid artifact entry")
        identifier = entry.get("id")
        kind = _kind(identifier)
        require(identifier not in seen, f"Duplicate artifact: {identifier}")
        seen.add(identifier)
        require(entry.get("kind") == kind and entry.get("status") in {AWAITING, PUBLISHED},
                f"Invalid artifact kind or state: {identifier}")
        if "asset_stem" in entry:
            require(entry["asset_stem"] == asset_stem(identifier), f"Invalid asset stem: {identifier}")
        if entry["status"] != PUBLISHED:
            continue
        if manifest.get("channel") == DEMO_CHANNEL:
            require(entry.get("relationship") == DEMO_RELATIONSHIP and entry.get("scientific_only_verification") is False,
                    f"The demonstration channel publishes demonstration artifacts only: {identifier}")
        else:
            require(entry.get("relationship") in RELATIONSHIPS, f"Declare the manuscript relationship: {identifier}")
        require(isinstance(entry.get("identity"), dict) and bool(entry["identity"]), f"Missing artifact identity: {identifier}")
        _hash(entry.get("inventory_sha256"), identifier)
        for field in ("files", "bytes"):
            require(type(entry.get(field)) is int and entry[field] > 0, f"Invalid {field}: {identifier}")
        counts = entry.get("contents")
        require(isinstance(counts, dict) and set(counts) == set(content_counts([]))
                and all(type(v) is int and v >= 0 for v in counts.values()), f"Invalid content counts: {identifier}")
        if "scientific_only_verification" in entry:
            require(type(entry["scientific_only_verification"]) is bool, f"Invalid verification scope: {identifier}")
        tiers = entry.get("tiers")
        require(isinstance(tiers, dict) and "complete" in tiers and set(tiers) <= {"complete", "analysis"},
                f"Missing complete tier: {identifier}")
        for tier, spec in tiers.items():
            require(isinstance(spec, dict), f"Invalid tier descriptor: {identifier}/{tier}")
            relative_path(spec.get("archive"), basename=True)
            _hash(spec.get("sha256"), f"{identifier}/{tier}")
            require(type(spec.get("bytes")) is int and spec["bytes"] > 0, f"Invalid archive size: {identifier}/{tier}")
            parts = spec.get("parts")
            require(isinstance(parts, list) and bool(parts), f"Missing archive parts: {identifier}/{tier}")
            names = set()
            for part in parts:
                require(isinstance(part, dict), f"Invalid part: {identifier}/{tier}")
                name = relative_path(part.get("asset"), basename=True)
                require(name not in names, f"Duplicate archive part: {name}")
                names.add(name)
                require(type(part.get("bytes")) is int and 0 < part["bytes"] < 2 * 1024**3,
                        f"Invalid release-asset size: {name}")
                _hash(part.get("sha256"), name)
                url = urllib.parse.urlparse(part.get("url", ""))
                require(url.scheme in {"https", "http"} and bool(url.netloc) and not url.username and not url.password,
                        f"Invalid asset URL: {name}")
            require(sum(p["bytes"] for p in parts) == spec["bytes"], f"Part sizes do not equal archive size: {identifier}/{tier}")
    require(manifest.get("channel") in (None, DEMO_CHANNEL), "Unknown artifact channel")
    if manifest.get("channel") == DEMO_CHANNEL:
        require(manifest.get("release_tag") == DEMO_TAG and bool(manifest.get("scope")),
                "The demonstration manifest names its own release and scope")
    else:
        require(manifest.get("release_tag", TAG) != DEMO_TAG, "The paper's manifest cannot name the demonstration release")
    all_published = bool(manifest["artifacts"]) and all(e["status"] == PUBLISHED for e in manifest["artifacts"])
    require(manifest["status"] == (PUBLISHED if all_published else AWAITING), "Manifest state differs from its artifact descriptors")
    return manifest


def planned_manifest(existing: dict, root: Path = ROOT) -> dict:
    """Refresh the required list, keeping every published entry and its identities."""
    manifest = (blank_manifest(existing.get("channel") == DEMO_CHANNEL)
                | {k: v for k, v in existing.items() if k not in {"artifacts", "status"}})
    current = {entry["id"]: entry for entry in existing.get("artifacts", [])}
    artifacts = []
    for row in required_artifacts(root):
        entry = current.get(row["id"])
        if entry and entry.get("status") == PUBLISHED:
            artifacts.append({**entry, **{k: row[k] for k in ("kind", "name", "asset_stem", "statement_items",
                                                              "displays")}})
        else:
            artifacts.append({**row, "status": AWAITING,
                              "awaiting": "the measured outputs of the reported experiment; publish with "
                                          "hpc/publish_paper_artifacts_arrhenius.sh"})
    extra = [e for i, e in current.items() if e.get("status") == PUBLISHED
             and i not in {a["id"] for a in artifacts}]
    artifacts.extend(dict(e, required=False) for e in extra)
    manifest["artifacts"] = artifacts
    manifest["status"] = PUBLISHED if artifacts and all(a["status"] == PUBLISHED for a in artifacts) else AWAITING
    return manifest


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------

HEAVY_SUFFIXES = (".pt",)


def is_heavy(relative: str) -> bool:
    """Files left out of the analysis tier: checkpoints and full prediction arrays."""
    parts = PurePosixPath(relative).parts
    if parts and parts[0].startswith("split_"):
        relative = PurePosixPath(*parts[1:]).as_posix()
    name = PurePosixPath(relative).name
    return (relative.endswith(HEAVY_SUFFIXES)
            or relative.startswith(("member_predictions/", "predictions/"))
            or relative == "final/test_predictions.npz"
            or (relative.startswith("final/") and name.endswith("_probabilities.npy"))
            or name == "oof_predictions.npz")


def content_counts(files: list[dict]) -> dict:
    """Count the statement's categories inside one archive inventory."""
    counts = {"checkpoints": 0, "predictions": 0, "propensity_logged_trajectories": 0,
              "statistical_outputs": 0, "split_and_seed_manifests": 0, "environment": 0, "configurations": 0}
    for entry in files:
        rel = entry["path"]
        parts = PurePosixPath(rel).parts
        if parts and parts[0].startswith("split_"):
            rel = PurePosixPath(*parts[1:]).as_posix()
        if rel.endswith(".pt"):
            counts["checkpoints"] += 1
        elif (rel.startswith(("member_predictions/", "predictions/")) or "probabilities" in rel
              or rel.endswith(("test_predictions.npz", "oof_predictions.npz"))):
            counts["predictions"] += 1
        elif rel.startswith("explore/trajectory_"):
            counts["propensity_logged_trajectories"] += 1
        elif rel.startswith(("select/", "confirm/", "admit/")) or rel in (
                "policy_lock.json", "final/results_summary.json", "final/paired_test_units.npz",
                "final/results.csv", "summary.csv") or rel.startswith(("raw/", "analysis")):
            counts["statistical_outputs"] += 1
        elif rel.startswith("prepared/") and rel.endswith(("splits.npz", "clusters.npz", "metadata.json")):
            counts["split_and_seed_manifests"] += 1
        elif rel == "environment.json" or rel.startswith("launcher/"):
            counts["environment"] += 1
        elif rel in ("config.json", "action_registry.json") or rel.startswith("source/configs/"):
            counts["configurations"] += 1
    return counts


def _inventory(root: Path) -> list[dict]:
    entries = []
    for folder, dirs, names in os.walk(root):
        dirs.sort()
        for name in dirs:
            require(not (Path(folder) / name).is_symlink(), f"Symlink directories are not archived: {Path(folder) / name}")
        for name in sorted(names):
            path = Path(folder) / name
            require(not path.is_symlink() and path.is_file(), f"Only regular files are archived: {path}")
            relative = path.relative_to(root).as_posix()
            if relative in (INVENTORY_NAME, RUN_ARCHIVE_MANIFEST, SUBSET_NAME):
                continue
            entries.append({"path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return entries


def _tar(output: Path, top: str, root: Path, files: list[str], extra: dict[str, dict],
         *, source_files: dict[str, Path] | None = None) -> None:
    """Deterministic gzip tar: identical inputs give identical bytes, so re-publishing uploads nothing."""
    import gzip
    with output.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0,
                                                   compresslevel=6) as compressed, \
            tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name, value in extra.items():
            data = json.dumps(value, indent=1, sort_keys=True).encode("utf-8")
            info = tarfile.TarInfo(f"{top}/{name}")
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            tar.addfile(info, io.BytesIO(data))
        for relative in files:
            source = (source_files or {}).get(relative, root / relative)
            info = tar.gettarinfo(str(source), arcname=f"{top}/{relative}")
            info.mtime, info.uid, info.gid, info.uname, info.gname, info.mode = 0, 0, 0, "", "", 0o644
            info.pax_headers = {}
            with source.open("rb") as handle:
                tar.addfile(info, handle)


def _verify_source(artifact_id: str, source: Path, *, scientific_only: bool = True) -> dict:
    """Validate a supplied directory with the repository's own checkers before packing."""
    _kind(artifact_id)
    require(source.is_dir() and not source.is_symlink(), f"Artifact directory is absent or linked: {source}")
    require(not any(p.is_symlink() for p in source.rglob("*")), f"Artifact contains a symlink: {source}")
    kind, _, name = artifact_id.partition(":")
    if kind == "run":
        splits = sorted(p for p in source.glob("split_*") if (p / "run_state.json").is_file())
        if splits:
            require(name == "Roman-empire", f"Only Roman-empire is published as a split campaign: {source}")
            require([p.name for p in splits] == [f"split_{i:02d}" for i in range(10)],
                    f"Roman-empire campaign needs split_00 through split_09, found {[p.name for p in splits]}")
            identities = {}
            for index, path in enumerate(splits):
                identities[path.name] = _verify_run(path, name, scientific_only=scientific_only)
                split = read_json(path / "config.json").get("data", {}).get("split_index")
                require(type(split) is int and split == index, f"{path}: sealed split ID differs from directory")
            return {"campaign": True, "splits": identities}
        require(name != "Roman-empire", "Roman-empire requires the ten-split campaign")
        return {"campaign": False, **_verify_run(source, name, scientific_only=scientific_only)}
    if kind == "records":
        from scripts import records as measurement_records
        specs = measurement_records.load_specs(source.parent, ROOT / "records/specs.json")
        require(name in specs["sets"], f"{artifact_id}: no record contract named {name}")
        require(source.name == name, f"{artifact_id}: a record-set directory must be named {name}")
        entry = measurement_records.verify_set(source, specs["sets"][name], specs)
        require(not entry.get("synthetic") or not scientific_only,
                f"{artifact_id}: synthetic example records are not paper artifacts")
        identity = {"manifest_sha256": entry["manifest_sha256"], "rows": entry.get("rows")}
        if scientific_only:
            provenance = read_json(source / "provenance.json")
            require(provenance.get("coverage_complete") is not False, f"{artifact_id}: partial record coverage")
            from cm_campaigns.contract import RECORD_SETS
            if name in RECORD_SETS:
                from scripts.scientific_evidence_audit import validate_record_coverage
                rows = measurement_records.read_summary(source, specs["sets"][name], specs["summary_columns"])
                identity["coverage"] = validate_record_coverage(name, rows, specs["sets"][name], RECORD_SETS[name], path=source)
        return identity
    if kind == "campaign":
        if name == "confirmation":
            from cm_campaigns.confirmation import check_campaign
            checked = check_campaign(source, manuscript_root=ROOT)
            require(checked.get("status") in {"PASS", "REFERENCE_MISMATCH"}, "Confirmation campaign failed its evidence contract")
            return {"input_identity": checked["input_identity"], "verification_status": checked["status"],
                    "post_hoc": checked.get("post_hoc", False),
                    "production": read_json(source / "campaign.json").get("production"),
                    "admission_findings": checked.get("admission_findings", []),
                    "historical_identity_authenticated": False}
        if name == "budget":
            from cm_campaigns.arm_campaign import check_campaign
            checked = check_campaign(source, read_json(contained(source, "study.json")), strict=scientific_only)
            require(checked.get("passed") is True, "Budget campaign failed its evidence contract")
            return {"verification_status": "PASS", "coverage": checked["coverage"], "scope": checked["scope"],
                    "historical_table_reproduction": checked.get("historical_table_reproduction", False)}
        raise ArtifactError(f"No evidence validator for campaign: {name}")
    if not scientific_only:
        require(any(source.iterdir()), f"{artifact_id}: empty environment capture {source}")
        return {"verification_scope": "software fixture"}
    for relative in ("observed_environment.json", "requirements-gpu.lock", "nvidia-smi.txt", "modules.txt"):
        require(contained(source, relative).stat().st_size > 0, f"GPU environment capture is empty: {relative}")
    observed = read_json(source / "observed_environment.json")
    require(observed.get("cuda_available") is True and type(observed.get("gpu_count")) is int
            and observed["gpu_count"] > 0 and bool(observed.get("gpu")) and bool(observed.get("torch"))
            and bool(observed.get("python")) and isinstance(observed.get("packages"), dict) and bool(observed["packages"]),
            "Environment capture must identify its actual Python, packages and available GPU")
    return {"observed_environment_sha256": sha256_file(source / "observed_environment.json"),
            "gpu": observed["gpu"], "python": observed["python"], "torch": observed["torch"]}


def _verify_run(path: Path, dataset: str, *, scientific_only: bool) -> dict:
    identities, _ = verify_analysis_run(path, {})
    state, config = read_json(path / "run_state.json"), read_json(path / "config.json")
    final = read_json(path / "final/results_summary.json")
    require((state.get("task") or {}).get("dataset") == dataset == final.get("benchmark_dataset")
            == config.get("benchmark_dataset", config.get("dataset")), f"{path}: sealed dataset scope differs")
    require(state.get("scientific_run") is config.get("scientific_run") is final.get("scientific_run")
            and type(config.get("scientific_run")) is bool and state.get("profile") == config.get("profile"),
            f"{path}: bookkeeping scientific/profile scope differs from sealed outputs")
    require(not scientific_only or (config["scientific_run"] is True and config.get("profile") == "full"),
            f"{path}: not a full scientific run")
    return {**identities, "profile": config.get("profile"), "scientific_run": config["scientific_run"]}


def split_parts(path: Path, part_bytes: int = PART_BYTES) -> list[dict]:
    """Split ``path`` into ``<name>.partNNN`` files when it exceeds ``part_bytes``."""
    require(type(part_bytes) is int and 0 < part_bytes < 2 * 1024**3, "Part size must be positive and below 2 GiB")
    size = path.stat().st_size
    if size <= part_bytes:
        return [{"asset": path.name, "bytes": size, "sha256": sha256_file(path)}]
    parts = []
    with path.open("rb") as handle:
        index = 0
        while True:
            target = path.with_name(f"{path.name}.part{index:03d}")
            digest, written = hashlib.sha256(), 0
            with target.open("wb") as out:
                while written < part_bytes:
                    block = handle.read(min(1 << 20, part_bytes - written))
                    if not block:
                        break
                    out.write(block)
                    digest.update(block)
                    written += len(block)
            if written == 0:
                target.unlink()
                break
            parts.append({"asset": target.name, "bytes": written, "sha256": digest.hexdigest()})
            index += 1
    path.unlink()
    return parts


def _runtime_file_inventory(snapshot: Path, expected: str) -> list[dict]:
    """Inventory only the Python files covered by the recorded runtime identity."""
    from scripts._source_snapshots import runtime_files
    require(not any(p.is_symlink() for p in (snapshot, *snapshot.parents))
            and not any(p.is_symlink() for p in snapshot.rglob("*")),
            f"Frozen runtime source must not traverse or contain a symlink: {snapshot}")
    files = []
    for path in runtime_files(snapshot):
        relative = path.relative_to(snapshot).as_posix()
        path = contained(snapshot, relative)
        files.append({"path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    hashes = {row["path"]: row["sha256"] for row in files}
    actual = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    require("cross_morse/pipeline.py" in hashes and actual == expected,
            f"Frozen runtime source changed or does not match {expected}: {snapshot}")
    return files


def _packing_runtime_sources(artifact_id: str, source: Path, identity: dict, out_dir: Path) -> list[dict]:
    """Carry external run/campaign snapshots into portable run archives.

    Current and bundled runtime sources travel with the checkout. A snapshot
    already inside the artifact is inventoried normally. For other resolved
    snapshots, add only the exact fingerprinted runtime files under the run's
    own source/ directory; never copy a checkpoint tree or rewrite a run.
    """
    from scripts._source_snapshots import resolve_runtime_source
    if _kind(artifact_id) != "run":
        return []
    runs = ([(source / name, value) for name, value in sorted(identity["splits"].items())]
            if identity.get("campaign") else [(source, identity)])
    dependencies = []
    for run, value in runs:
        expected = value["source_sha256"]
        try:
            snapshot = resolve_runtime_source(expected, ROOT, run_root=run)
        except ValueError as exc:
            raise ArtifactError(f"{artifact_id}: {exc}") from exc
        if (snapshot.is_relative_to(source) or snapshot == ROOT.resolve()
                or snapshot == (ROOT / "examples/source_snapshots" / expected).resolve()):
            continue
        require(not out_dir.resolve().is_relative_to(snapshot) and not snapshot.is_relative_to(out_dir.resolve()),
                "Packing output must be separate from the external runtime source")
        prefix = (run.relative_to(source) / "source").as_posix()
        require(not (source / prefix).exists() and not (source / prefix).is_symlink(),
                f"Conflicting frozen source folder: {source / prefix}; preserve or reconcile its original bytes first")
        dependencies.append({"snapshot": snapshot, "prefix": prefix, "source_sha256": expected,
                             "files": _runtime_file_inventory(snapshot, expected)})
    return dependencies


def pack_one(artifact_id: str, source: Path, out_dir: Path, *, relationship: str,
             part_bytes: int = PART_BYTES, scientific_only: bool = True) -> dict:
    source = Path(source).absolute()
    require(not any(p.is_symlink() for p in (source, *source.parents)), "Packing input traverses a symlink")
    source = source.resolve()
    require(source.is_dir(), f"{artifact_id}: {source} is not a directory")
    require(relationship in RELATIONSHIPS or (relationship == DEMO_RELATIONSHIP and not scientific_only),
            f"Explicit original/reproduction relationship is required: {artifact_id}")
    require(type(part_bytes) is int and 0 < part_bytes < 2 * 1024**3, "Part size must be positive and below 2 GiB")
    out_dir = Path(out_dir).absolute()
    require(not any(p.is_symlink() for p in (out_dir, *out_dir.parents)), "Packing output traverses a symlink")
    require(not out_dir.resolve().is_relative_to(source) and not source.is_relative_to(out_dir.resolve()),
            "Packing output must be separate from the artifact input")
    kind = _kind(artifact_id)
    identity = _verify_source(artifact_id, source, scientific_only=scientific_only)
    require(not (relationship == "exact_result_producing_run" and identity.get("post_hoc") is True),
            "A post hoc confirmation campaign cannot be declared the original result-producing campaign")
    stem = asset_stem(artifact_id)
    original_files = _inventory(source)
    dependencies = _packing_runtime_sources(artifact_id, source, identity, out_dir)
    additions = [{**row, "path": f"{dependency['prefix']}/{row['path']}"}
                 for dependency in dependencies for row in dependency["files"]]
    files = sorted(original_files + additions, key=lambda row: row["path"]) if additions else original_files
    _files(files, artifact_id)
    require(bool(files), f"Artifact has no files: {artifact_id}")
    inventory = {"schema": "cross-morse-artifact-inventory-1", "artifact": artifact_id, "kind": kind,
                 "source_directory_name": source.name, "identity": identity, "files": files}
    inventory["inventory_sha256"] = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    if dependencies:
        inventory["runtime_source_additions"] = [
            {"path": dependency["prefix"], "source_sha256": dependency["source_sha256"],
             "scope": "Python files covered by the recorded runtime fingerprint"}
            for dependency in dependencies]
    out_dir.mkdir(parents=True, exist_ok=True)
    # Build both tiers before changing the selected entry. Content-addressed names
    # preserve earlier release assets and make interrupted or repeated packs safe.
    with tempfile.TemporaryDirectory(prefix=f".pack-{stem}-", dir=out_dir) as temporary:
        stage = Path(temporary)
        source_files = {}
        for dependency in dependencies:
            for row in dependency["files"]:
                relative = f"{dependency['prefix']}/{row['path']}"
                staged = stage / "runtime" / relative
                staged.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(contained(dependency["snapshot"], row["path"]), staged)
                require(staged.stat().st_size == row["bytes"] and sha256_file(staged) == row["sha256"],
                        f"Frozen runtime source changed while staging: {relative}")
                source_files[relative] = staged
        tiers = {}
        for tier in (("complete", "analysis") if kind == "run" else ("complete",)):
            selected = files if tier == "complete" else [f for f in files if not is_heavy(f["path"])]
            extra = {INVENTORY_NAME: inventory}
            if tier == "analysis":
                extra[SUBSET_NAME] = {"schema": "cross-morse-analysis-subset-1", "artifact": artifact_id,
                    "complete_inventory_sha256": inventory["inventory_sha256"],
                    "rule": "Checkpoints and full prediction arrays are retained in the complete tier.",
                    "omitted": [f for f in files if is_heavy(f["path"])]}
            archive = stage / "archive.tar.gz"
            _tar(archive, stem, source, [f["path"] for f in selected], extra, source_files=source_files)
            digest, size = sha256_file(archive), archive.stat().st_size
            named = stage / f"{stem}.{tier}-{digest}.tar.gz"
            archive.rename(named)
            tiers[tier] = {"archive": named.name, "bytes": size, "sha256": digest,
                           "parts": split_parts(named, part_bytes)}
        require(_inventory(source) == original_files, f"{artifact_id}: input files changed while packing")
        for dependency in dependencies:
            require(_runtime_file_inventory(dependency["snapshot"], dependency["source_sha256"]) == dependency["files"],
                    f"{artifact_id}: runtime source files changed while packing")
        for tier in tiers.values():
            for part in tier["parts"]:
                target = contained(out_dir, part["asset"], exists=False)
                if target.exists():
                    require(target.stat().st_size == part["bytes"] and sha256_file(target) == part["sha256"],
                            f"Existing content-addressed part differs: {target}")
        for tier in tiers.values():
            for part in tier["parts"]:
                target = out_dir / part["asset"]
                if not target.exists():
                    os.replace(stage / part["asset"], target)
        entry = {"id": artifact_id, "kind": kind, "relationship": relationship, "identity": identity,
                 "inventory_sha256": inventory["inventory_sha256"], "files": len(files),
                 "bytes": sum(f["bytes"] for f in files), "contents": content_counts(files),
                 "scientific_only_verification": bool(scientific_only), "tiers": tiers, "packed_utc": utc_now()}
        write_json(out_dir / f"{stem}.entry.json", entry)
    return entry


# ---------------------------------------------------------------------------
# GitHub release transport
# ---------------------------------------------------------------------------

def github_token() -> str | None:
    for name in ("GITHUB_TOKEN", "GH_TOKEN"):
        if os.environ.get(name):
            return os.environ[name]
    gh = shutil.which("gh")
    if gh:
        import subprocess
        result = subprocess.run([gh, "auth", "token"], capture_output=True, text=True, check=False)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return None


class GitHub:
    """The four release calls this tool needs, over the REST API."""

    def __init__(self, repository: str = REPOSITORY, token: str | None = None,
                 api: str = "https://api.github.com"):
        self.repository = repository
        self.token = token
        self.api = api.rstrip("/")

    def _request(self, method: str, url: str, *, body=None, headers=None, ok=(200,)):
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
                   "User-Agent": "cross-morse-paper-artifacts", **(headers or {})}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        data = body
        if isinstance(body, (dict, list)):
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                payload = response.read()
                status = response.status
        except urllib.error.HTTPError as error:
            payload, status = error.read(), error.code
        if status not in ok:
            raise ArtifactError(f"{method} {url} returned HTTP {status}: {payload[:300]!r}")
        return status, (json.loads(payload) if payload and payload[:1] in (b"{", b"[") else payload)

    def release(self, tag: str) -> dict | None:
        status, value = self._request("GET", f"{self.api}/repos/{self.repository}/releases/tags/{tag}",
                                      ok=(200, 404))
        return value if status == 200 else None

    def create_release(self, tag: str, *, target: str, body: str) -> dict:
        _, value = self._request("POST", f"{self.api}/repos/{self.repository}/releases", ok=(201,), body={
            "tag_name": tag, "target_commitish": target, "name": "Paper artifacts", "body": body,
            "draft": False, "prerelease": False, "make_latest": "false"})
        return value

    def assets(self, release: dict) -> dict[str, dict]:
        found, page = {}, 1
        while True:
            _, value = self._request("GET", f"{self.api}/repos/{self.repository}/releases/{release['id']}/assets"
                                            f"?per_page=100&page={page}")
            found.update({asset["name"]: asset for asset in value})
            if len(value) < 100:
                return found
            page += 1

    def delete_asset(self, asset: dict) -> None:
        self._request("DELETE", f"{self.api}/repos/{self.repository}/releases/assets/{asset['id']}", ok=(204,))

    def upload(self, release: dict, path: Path, name: str) -> dict:
        base = release["upload_url"].split("{", 1)[0]
        size = path.stat().st_size
        with path.open("rb") as handle:
            _, value = self._request("POST", f"{base}?{urllib.parse.urlencode({'name': name})}", body=handle,
                                     headers={"Content-Type": "application/octet-stream",
                                              "Content-Length": str(size)}, ok=(201,))
        return value


def download(url: str, target: Path, *, token: str | None = None) -> None:
    headers = {"User-Agent": "cross-morse-paper-artifacts", "Accept": "application/octet-stream"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    temporary = target.with_name(target.name + ".download")
    require(not target.is_symlink() and not temporary.is_symlink(), "Download target is linked")
    with urllib.request.urlopen(request, timeout=600) as response, temporary.open("wb") as out:
        shutil.copyfileobj(response, out, 1 << 20)
    os.replace(temporary, target)


def publish(staged: Path, manifest_path: Path = MANIFEST, *, client: GitHub, tag: str = TAG,
            target: str = "main", download_base: str | None = None, demonstration: bool = False) -> dict:
    """Publish immutable asset versions after preflight; switch the local manifest last.

    ``demonstration`` publishes the CPU demonstration to its own release (DEMO_TAG) and manifest
    (DEMO_MANIFEST); its entries carry DEMO_RELATIONSHIP only and never enter the paper's manifest."""
    if demonstration:
        require(tag == DEMO_TAG and Path(manifest_path).resolve() != MANIFEST.resolve(),
                "The demonstration publishes to its own release and manifest")
    else:
        require(tag != DEMO_TAG, "The paper's outputs are not published to the demonstration release")
    staged = Path(staged).resolve()
    entries = [read_json(p) for p in sorted(staged.glob("*.entry.json"))]
    require(entries, f"No packed artifacts (*.entry.json) under {staged}")
    base = (download_base or f"https://github.com/{client.repository}/releases/download/{tag}").rstrip("/")
    # Validate the entire transfer before creating a release or uploading anything.
    # Duplicate IDs, unsafe local names and a damaged later part cannot leave a
    # partially selected release manifest or delete a previous published version.
    for entry in entries:
        for spec in entry.get("tiers", {}).values():
            for part in spec.get("parts", []):
                path = contained(staged, relative_path(part.get("asset"), basename=True))
                require(path.stat().st_size == part.get("bytes") and sha256_file(path) == part.get("sha256"),
                        f"Staged part changed or missing: {path}")
                part["url"] = f"{base}/{urllib.parse.quote(part['asset'])}"
    channel = {"channel": DEMO_CHANNEL, "release_tag": DEMO_TAG, "scope": DEMO_SCOPE} if demonstration else {}
    validate_manifest({"schema": SCHEMA, "status": PUBLISHED, **channel,
                       "artifacts": [dict(entry, status=PUBLISHED) for entry in entries]})
    old_manifest = planned_manifest(load_manifest(manifest_path))
    body = ("CPU demonstration of the CROSS-MORSE code at reduced scale; not the paper's measurements. "
            "release/demo_artifacts.json records each artifact's identities and asset hashes; `make demo-paper` "
            "verifies the downloaded bytes and rebuilds every table and figure from them (demo/README.md)."
            if demonstration else
            "Archived experimental outputs for the CROSS-MORSE paper and its reproductions. "
            "release/paper_artifacts.json records each artifact's source identities, declared "
            "relationship to the paper and immutable asset hashes. `make paper` verifies the "
            "downloaded bytes and reports display coverage and numerical comparisons.")
    release = client.release(tag) or client.create_release(tag, target=target, body=body)
    present = client.assets(release)
    for entry in entries:
        for tier in entry["tiers"].values():
            for part in tier["parts"]:
                path = staged / part["asset"]
                asset = present.get(part["asset"])
                digest = ((asset or {}).get("digest") or "").removeprefix("sha256:")
                if asset is None:
                    asset = client.upload(release, path, part["asset"])
                    present[part["asset"]] = asset
                    digest = (asset.get("digest") or "").removeprefix("sha256:")
                require(asset.get("size") == part["bytes"] and asset.get("state", "uploaded") == "uploaded",
                        f"Published asset size/state differs: {part['asset']}; existing assets are never replaced")
                if not digest:
                    with tempfile.TemporaryDirectory(prefix="cm-upload-check-") as temporary:
                        checked = Path(temporary) / "asset"
                        download(part["url"], checked, token=client.token)
                        digest = sha256_file(checked)
                require(digest == part["sha256"],
                        f"Published asset digest differs: {part['asset']}; preserve the old asset and repack into a new version")
    manifest = old_manifest
    by_id = {entry["id"]: entry for entry in entries}
    updated = []
    for current in manifest["artifacts"]:
        entry = by_id.pop(current["id"], None)
        if entry is None:
            updated.append(current)
            continue
        updated.append({**{k: current[k] for k in ("id", "kind", "name", "asset_stem", "statement_items", "displays")
                           if k in current}, "status": PUBLISHED, **entry, "published_utc": utc_now()})
    for entry in by_id.values():
        updated.append({**entry, "status": PUBLISHED, "required": False, "published_utc": utc_now()})
    if demonstration:
        updated = [dict(a, awaiting=DEMO_AWAITING) if a["status"] != PUBLISHED else a for a in updated]
        manifest.update(channel)
    manifest.update(artifacts=updated, repository=client.repository, release_tag=tag,
                    release_url=release.get("html_url", f"https://github.com/{client.repository}/releases/tag/{tag}"),
                    status=PUBLISHED if all(a["status"] == PUBLISHED for a in updated) else AWAITING)
    validate_manifest(manifest)
    write_json(manifest_path, manifest)
    return manifest


# ---------------------------------------------------------------------------
# Retrieval and verification
# ---------------------------------------------------------------------------

def _safe_extract(archive: Path, destination: Path) -> str:
    """Extract a single-top-level archive with regular files only; return the top directory."""
    with tarfile.open(archive, "r:*") as tar:
        members = tar.getmembers()
        tops, names = set(), set()
        for member in members:
            relative_path(member.name.rstrip("/") if member.isdir() else member.name)
            name = PurePosixPath(member.name)
            require(name.as_posix() not in names, f"Duplicate archive member: {member.name}")
            names.add(name.as_posix())
            require(member.isfile() or member.isdir(), f"Only regular files are allowed: {member.name}")
            tops.add(name.parts[0])
        require(len(tops) == 1, f"An artifact archive has one top-level directory, found {sorted(tops)}")
        top = tops.pop()
        target = destination / top
        require(not target.exists(), f"Refusing to overwrite {target}")
        tar.extractall(destination, members=members, filter="data")
    return top


def verify_unpacked(directory: Path, *, inventory: dict | None = None) -> dict:
    """Check every extracted file against the packed inventory; analysis tiers list what they omit."""
    directory = Path(directory)
    require(directory.is_dir() and not directory.is_symlink()
            and not any(p.is_symlink() for p in directory.rglob("*")), f"Artifact contains a symlink: {directory}")
    external = inventory is not None
    inventory = inventory if external else read_json(contained(directory, INVENTORY_NAME))
    require(isinstance(inventory, dict) and inventory.get("schema") == "cross-morse-artifact-inventory-1",
            f"Invalid artifact inventory: {directory}")
    files = _files(inventory.get("files"), str(directory))
    require(hashlib.sha256(json.dumps(inventory["files"], sort_keys=True).encode()).hexdigest()
            == inventory["inventory_sha256"], f"Inventory digest changed in {directory}")
    present = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if not external:
        present -= {INVENTORY_NAME, SUBSET_NAME}
    omitted = set()
    subset_path = directory / SUBSET_NAME
    if subset_path.is_file():
        require(not external and inventory.get("kind") == "run", f"Unexpected analysis-subset declaration: {directory}")
        subset = read_json(subset_path)
        require(subset.get("schema") == "cross-morse-analysis-subset-1"
                and subset.get("artifact") == inventory.get("artifact"), f"Invalid analysis subset: {directory}")
        require(subset["complete_inventory_sha256"] == inventory["inventory_sha256"],
                f"Analysis subset belongs to another inventory: {directory}")
        omitted = set(_files(subset["omitted"], str(subset_path)))
        require(all(is_heavy(p) and files.get(p) == f for p, f in ((f["path"], f) for f in subset["omitted"])),
                f"Analysis subset omits a file that the analysis tier must contain: {directory}")
    require(present | omitted == set(files) and not (present & omitted),
            f"Unpacked files differ from the inventory in {directory}")
    for relative in sorted(present):
        path = contained(directory, relative)
        expected = files[relative]
        require(path.stat().st_size == expected["bytes"] and sha256_file(path) == expected["sha256"],
                f"File differs from its inventory: {directory / relative}")
    return {"files": len(present), "omitted": len(omitted), "inventory_sha256": inventory["inventory_sha256"]}


def artifact_identity(entry: dict, tier: str) -> dict:
    """Canonical manifest-to-materialization link used by derived views and offline gates."""
    return {"id": entry["id"], "kind": entry["kind"], "relationship": entry["relationship"], "tier": tier,
            "archive_sha256": entry["tiers"][tier]["sha256"], "inventory_sha256": entry["inventory_sha256"],
            "identity": entry["identity"], "scientific_only_verification": entry.get("scientific_only_verification")}


def _materialized_path(destination: Path, entry: dict) -> Path:
    layout = {"run": "runs", "records": "records", "campaign": "campaigns", "environment": "environment"}
    kind, name = entry["id"].split(":", 1)
    return destination / layout[kind] / (STEMS[name] if kind == "run" else name)


def _verify_materialized(destination: Path, entry: dict, stamp: dict, *, match_current=True) -> dict:
    target = _materialized_path(destination, entry)
    require(stamp.get("id") == entry["id"] and stamp.get("tier") in {"complete", "analysis"}
            and Path(stamp.get("path", "")).absolute() == target.absolute(),
            f"Fetched stamp belongs to another artifact or path: {entry['id']}")
    require(not any(p.is_symlink() for p in (target, *target.parents)), f"Fetched artifact path is linked: {target}")
    if match_current:
        tier = stamp["tier"]
        require(tier in entry["tiers"] and stamp.get("sha256") == entry["tiers"][tier]["sha256"]
                and stamp.get("inventory_sha256") == entry["inventory_sha256"],
                f"Fetched artifact differs from the current manifest: {entry['id']}")
    inventory = (read_json(contained(target, INVENTORY_NAME)) if entry["kind"] == "run" else stamp.get("inventory"))
    require(isinstance(inventory, dict) and inventory.get("artifact") == entry["id"]
            and inventory.get("kind") == entry["kind"]
            and inventory.get("inventory_sha256") == stamp.get("inventory_sha256"),
            f"Fetched inventory belongs to another artifact: {entry['id']}")
    if match_current:
        require(inventory.get("identity") == entry["identity"]
                and len(inventory.get("files", [])) == entry["files"]
                and sum(f["bytes"] for f in inventory["files"]) == entry["bytes"]
                and content_counts(inventory["files"]) == entry["contents"],
                f"Fetched content descriptors differ from the current manifest: {entry['id']}")
    checked = verify_unpacked(target, inventory=inventory if entry["kind"] != "run" else None)
    require(not (stamp["tier"] == "complete" and checked["omitted"]), f"Complete tier omits files: {entry['id']}")
    return checked


def verify_fetched(directory: Path, *, manifest_path: Path = MANIFEST, require_all: bool = False) -> dict:
    """Recheck current published bytes and canonical paths, including cached/offline downloads."""
    directory = Path(directory).resolve()
    manifest = load_manifest(manifest_path)
    rows, awaiting, expected_paths = [], [], set()
    for entry in manifest["artifacts"]:
        if entry["status"] != PUBLISHED:
            awaiting.append(entry["id"])
            continue
        path = _materialized_path(directory, entry)
        expected_paths.add(path)
        stamp_path = directory / ".artifacts" / f"{asset_stem(entry['id'])}.json"
        if not stamp_path.is_file():
            require(not path.exists(), f"Unverified artifact directory has no managed download stamp: {path}")
            awaiting.append(entry["id"])
            continue
        stamp = read_json(contained(directory, stamp_path.relative_to(directory).as_posix()))
        checked = _verify_materialized(directory, entry, stamp)
        rows.append({**artifact_identity(entry, stamp["tier"]), "path": str(path), "verification": checked})
    for folder in ("runs", "records", "campaigns", "environment"):
        base = directory / folder
        if base.is_dir():
            actual = {p for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")}
            require(actual <= expected_paths, f"Fetched directory contains unregistered artifact directories: {sorted(str(p) for p in actual - expected_paths)}")
    require(not require_all or not awaiting, "Published artifacts are not fully materialized: " + ", ".join(awaiting))
    return {"artifact_manifest_sha256": sha256_file(manifest_path), "artifacts": rows, "awaiting": awaiting}


def fetch(destination: Path, *, manifest_path: Path = MANIFEST, tier: str = "analysis",
          only: list[str] | None = None, token: str | None = None, cache: Path | None = None) -> dict:
    """Download, verify and unpack every published artifact into ``destination``.

    Layout: ``runs/<stem>/`` (Roman-empire: the campaign with ``split_*``),
    ``records/<set>/`` with ``records/index.json``, ``campaigns/<name>/`` and
    ``environment/<name>/``; ``.artifacts/`` keeps each artifact's inventory and the
    identity of the archive it came from. Directories already verified are reused.
    """
    require(tier in ("analysis", "complete"), "tier is analysis or complete")
    manifest = load_manifest(manifest_path)
    destination = Path(destination).resolve()
    cache = Path(cache) if cache else destination / ".downloads"
    require(not any(p.is_symlink() for p in (cache, *cache.parents)), "Download cache traverses a symlink")
    cache.mkdir(parents=True, exist_ok=True)
    stamps = destination / ".artifacts"
    require(not stamps.is_symlink(), "Artifact stamp directory is linked")
    stamps.mkdir(parents=True, exist_ok=True)
    layout = {"run": "runs", "records": "records", "campaign": "campaigns", "environment": "environment"}
    report = {"schema": "cross-morse-fetched-artifacts-1", "tier": tier,
              "manifest_sha256": sha256_file(manifest_path), "published": [], "awaiting": [],
              "destination": str(destination)}
    for entry in manifest["artifacts"]:
        if only and entry["id"] not in only:
            continue
        if entry.get("status") != PUBLISHED:
            report["awaiting"].append(entry["id"])
            continue
        kind, name = entry["kind"], entry["id"].split(":", 1)[1]
        folder = destination / layout[kind]
        folder.mkdir(parents=True, exist_ok=True)
        use = tier if tier in entry["tiers"] else "complete"
        spec = entry["tiers"][use]
        stamp = stamps / f"{entry['asset_stem'] if 'asset_stem' in entry else asset_stem(entry['id'])}.json"
        previous = None
        require(not stamp.is_symlink(), f"Artifact stamp is linked: {stamp}")
        if stamp.is_file():
            previous = read_json(stamp)
            _verify_materialized(destination, entry, previous, match_current=False)
            if previous.get("sha256") == spec["sha256"] and previous.get("tier") == use:
                checked = _verify_materialized(destination, entry, previous)
                report["published"].append({"id": entry["id"], "path": previous["path"], "tier": use, "reused": True, **checked})
                continue
        archive = contained(cache, spec["archive"], exists=False)
        if not (archive.is_file() and archive.stat().st_size == spec["bytes"] and sha256_file(archive) == spec["sha256"]):
            pieces = []
            for part in spec["parts"]:
                piece = contained(cache, part["asset"], exists=False)
                if not (piece.is_file() and piece.stat().st_size == part["bytes"]
                        and sha256_file(piece) == part["sha256"]):
                    download(part["url"], piece, token=token)
                require(piece.stat().st_size == part["bytes"] and sha256_file(piece) == part["sha256"],
                        f"Downloaded part differs from the manifest: {part['asset']}")
                pieces.append(piece)
            if pieces != [archive]:
                temporary_archive = contained(cache, archive.name + ".assembling", exists=False)
                with temporary_archive.open("wb") as out:
                    for piece in pieces:
                        with piece.open("rb") as handle:
                            shutil.copyfileobj(handle, out, 1 << 20)
                require(temporary_archive.stat().st_size == spec["bytes"] and sha256_file(temporary_archive) == spec["sha256"],
                        f"Reassembled archive differs: {spec['archive']}")
                os.replace(temporary_archive, archive)
            require(archive.stat().st_size == spec["bytes"] and sha256_file(archive) == spec["sha256"],
                    f"Reassembled archive differs: {spec['archive']}")
        with tempfile.TemporaryDirectory(dir=folder, prefix=".unpack-") as temporary:
            top = _safe_extract(archive, Path(temporary))
            unpacked = Path(temporary) / top
            check = verify_unpacked(unpacked)
            require(check["inventory_sha256"] == entry["inventory_sha256"],
                    f"{entry['id']}: unpacked inventory differs from the manifest")
            inventory = read_json(unpacked / INVENTORY_NAME)
            require(inventory.get("artifact") == entry["id"] and inventory.get("kind") == kind
                    and inventory.get("identity") == entry["identity"] and len(inventory["files"]) == entry["files"]
                    and sum(f["bytes"] for f in inventory["files"]) == entry["bytes"]
                    and content_counts(inventory["files"]) == entry["contents"],
                    f"{entry['id']}: unpacked artifact descriptors differ from the manifest")
            require(not (use == "complete" and check["omitted"]), f"{entry['id']}: complete tier omits files")
            target = _materialized_path(destination, entry)
            require(not target.exists() or previous is not None, f"{target} is not a managed download; refusing replacement")
            if kind != "run":
                # Record sets and campaigns are verified against their own manifests; keep them byte-identical.
                (unpacked / INVENTORY_NAME).unlink()
            backup = Path(temporary) / "previous"
            if target.exists():
                os.replace(target, backup)
            try:
                os.replace(unpacked, target)
                write_json(stamp, {"id": entry["id"], "tier": use, "sha256": spec["sha256"], "path": str(target),
                                   "inventory_sha256": entry["inventory_sha256"], "verified": check,
                                   "inventory": inventory if kind != "run" else INVENTORY_NAME})
            except BaseException:
                if target.exists():
                    os.replace(target, Path(temporary) / "failed")
                if backup.exists():
                    os.replace(backup, target)
                raise
        report["published"].append({"id": entry["id"], "path": str(target), "tier": use, "reused": False, **check})
    records = destination / "records"
    if records.is_dir() and any(p.is_dir() and not p.name.startswith(".") for p in records.iterdir()):
        from scripts import records as measurement_records
        specs = measurement_records.load_specs(records, ROOT / "records/specs.json")
        entries = [measurement_records.verify_set(p, specs["sets"][p.name], specs)
                   for p in measurement_records.deposited_sets(records, specs)]
        measurement_records.write_index(records, specs, entries)
    write_json(destination / "FETCHED.json", report)
    return report


def analysis_omissions(path: Path) -> dict[str, dict] | None:
    """Files an analysis-tier run or campaign omits, keyed by path; ``None`` for a complete tier."""
    subset = Path(path) / SUBSET_NAME
    if not subset.is_file():
        return None
    value = read_json(subset)
    require(value.get("schema") == "cross-morse-analysis-subset-1", f"Unexpected analysis subset in {path}")
    return {f["path"]: {"path": f["path"], "sha256": f["sha256"], "bytes": f["bytes"]} for f in value["omitted"]}


def verify_analysis_run(run_root: Path, omitted: dict[str, dict]) -> tuple[dict, dict]:
    from cross_morse.provenance import ProvenanceError
    try:
        return _verify_analysis_run(run_root, omitted)
    except ProvenanceError as exc:
        raise ArtifactError(str(exc)) from exc


def _verify_analysis_run(run_root: Path, omitted: dict[str, dict]) -> tuple[dict, dict]:
    """Verify the sealed stage chain of a run whose checkpoints and prediction arrays are omitted.

    Every present sealed file must match its stage manifest byte for byte; every absent
    sealed file must be a checkpoint or prediction array the analysis subset declares with
    the identical SHA-256 and size. Returns the run identities and the source resolution.
    """
    from cross_morse import provenance as pv
    from scripts._source_snapshots import resolve_runtime_source
    root = Path(run_root).absolute()
    require(not any(p.is_symlink() for p in (root, *root.parents)), "Run verification must not traverse a symlink")
    root = root.resolve()
    require(not any(p.is_symlink() for p in root.rglob("*")), f"Run contains a symlink: {root}")
    config = read_json(contained(root, "config.json"))
    expected = {pv.stage_manifest_path(root, stage).name for stage in pv.STAGES}
    require({p.name for p in (root / "provenance/stages").glob("*.json")} == expected,
            f"{root}: exactly the eight lifecycle stage manifests are required")
    cache, sealed, identities, parent = {}, {}, None, None
    for index, stage in enumerate(pv.STAGES):
        manifest_path = pv.stage_manifest_path(root, stage)
        require(manifest_path.is_file(), f"{root}: stage manifest {stage} is missing")
        value = read_json(manifest_path)
        pv._validate_manifest_shape(value)
        require(value["stage"] == stage, f"{root}: stage manifest name and content differ ({stage})")
        ids = {k: value[k] for k in ("source_sha256", "config_sha256", "unit_sha256")}
        if identities is None:
            for key, digest in ids.items():
                _hash(digest, key)
            require(ids["config_sha256"] == pv._json_hash(config), f"{root}: configuration content changed")
            require(ids["unit_sha256"] == sha256_file(root / pv.UNIT_MANIFEST_PATH), f"{root}: split identity changed")
            identities = ids
        require(ids == identities and value["dataset"] == config["dataset"], f"{root}: identities differ across stages")
        require(value["parent_manifest"] == parent, f"{root}: parent manifest link broken at {stage}")
        for field in ("inputs", "outputs"):
            seen = set()
            for declared in value[field]:
                relative = declared.get("path") if isinstance(declared, dict) else None
                require(isinstance(relative, str), f"{root}: malformed {stage}.{field} descriptor")
                contained(root, relative, exists=False)
                require(relative not in seen, f"{root}: duplicate {stage}.{field} descriptor: {relative}")
                seen.add(relative)
                if (root / relative).is_file():
                    pv._verify_artifact(root, declared, cache)
                else:
                    require(is_heavy(relative) and omitted.get(relative) == declared,
                            f"{root}: sealed file {relative} is neither present nor a declared omission")
                if field == "outputs":
                    require(relative not in sealed, f"{root}: {relative} sealed twice")
                    sealed[relative] = declared
                else:
                    require(index == 0 or sealed.get(relative) == declared,
                            f"{root}: {stage} input differs from its sealed output: {relative}")
        parent = pv._cached_artifact(root, manifest_path, cache)
    try:
        runtime = resolve_runtime_source(identities["source_sha256"], ROOT, run_root=root)
    except ValueError as exc:
        raise ArtifactError(f"{root}: {exc}") from exc
    resolution = ("current or bundled runtime source" if runtime == ROOT.resolve()
                  or runtime == (ROOT / "examples/source_snapshots" / identities["source_sha256"]).resolve()
                  else "the run's own frozen source snapshot")
    state = read_json(root / "run_state.json")
    require(state.get("status") == "COMPLETE" and state.get("last_complete_stage") == "evaluate",
            f"{root}: run bookkeeping is not complete")
    require(all(state.get(key) == identities[key] for key in ("source_sha256", "config_sha256")),
            f"{root}: run bookkeeping identities differ from the sealed chain")
    evaluation = read_json(pv.stage_manifest_path(root, "evaluate"))
    evaluator = evaluation.get("evaluator")
    require(isinstance(evaluator, dict) and evaluator.get("sha256") == sha256_file(Path(runtime) / "cross_morse/pipeline.py")
            and sealed.get(evaluator.get("path")) == evaluator
            and evaluator in evaluation["inputs"], f"{root}: frozen evaluator identity differs")
    return identities, {"sealed_files": len(sealed), "omitted": sum(1 for p in sealed if not (root / p).is_file()),
                        "source_resolution": resolution}



def map_template(path: Path) -> dict:
    """A map of every required artifact to the directory that holds it on the cluster (paths left empty)."""
    rows = required_artifacts()
    hints = {"run": "completed full-profile run directory (Roman-empire: the campaign holding split_00 ... split_09)",
             "records": "verified measurement-record set directory named after the set (summary.csv, raw/, "
                        "provenance.json, MANIFEST.json)",
             "campaign": "complete campaign export directory",
             "environment": "directory with the captured GPU environment of the runs (setup lock, pip freeze, "
                            "module list, nvidia-smi)"}
    template = {"_instructions": "Fill every path with an absolute directory on the cluster and explicitly declare each "
                                 "historical relationship in _relationships. For newly executed experiments use "
                                 "post_result_revision. Then run: bash hpc/publish_paper_artifacts_arrhenius.sh all <this file>",
                "_kinds": hints, "_relationships": {row["id"]: "" for row in rows}}
    template.update({row["id"]: "" for row in rows})
    write_json(path, template)
    return template


BASELINE_STUDIES = Path("experiments") / "baselines"
BASELINE_SCHEMA = "cross-morse-baseline-studies-1"


def _inside(root: Path, relative: str) -> Path:
    try:
        return contained(root, relative)
    except ArtifactError as exc:
        raise ArtifactError(f"{relative} is not a file inside {root}: {exc}") from exc


def builder_source_inventory(checkout: Path = ROOT) -> dict:
    """Reconstruction implementation and registered inputs; never generated report bytes."""
    from scripts.reconstruct import source_inventory
    checkout = Path(checkout).resolve()
    files = source_inventory(checkout)
    paths = [checkout / name for name in ("paper/registered_constants.json", "paper/display_manifest.json",
             "paper/printed_display_values.json", "paper/sources.json", "paper/reported_results.json")]
    paths.extend((checkout / "configs").rglob("*.json"))
    for path in paths:
        if path.is_file():
            files[path.relative_to(checkout).as_posix()] = sha256_file(path)
    return dict(sorted(files.items()))


def view_files(directory: Path) -> dict:
    require(not any(p.is_symlink() for p in directory.rglob("*")), f"View contains a symlink: {directory}")
    return {p.relative_to(directory).as_posix(): sha256_file(p) for p in sorted(directory.rglob("*"))
            if p.is_file() and p != directory / "INDEX.json"}


def _install_views(stage: Path, checkout: Path, relative_names: list[str]) -> None:
    """Replace verified generated views as a transaction; restore every earlier view on failure."""
    backups = stage / "previous"
    installed, moved = [], []
    for name in relative_names:
        target = checkout / name
        require(not any(p.is_symlink() for p in (target, *target.parents)), f"View target follows a symlink: {target}")
        if target.exists():
            require(target.is_dir() and (target / "INDEX.json").is_file(),
                    f"Refusing to replace an unmanaged directory: {target}; retain it elsewhere before deriving")
    try:
        for name in relative_names:
            target, new = checkout / name, stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                backup = backups / name
                backup.parent.mkdir(parents=True, exist_ok=True)
                os.replace(target, backup)
                moved.append(name)
            os.replace(new, target)
            installed.append(name)
    except BaseException:
        for name in reversed(installed):
            shutil.rmtree(checkout / name)
        for name in reversed(moved):
            os.replace(backups / name, checkout / name)
        raise


def _baseline_view(records: Path, target: Path, provenance: dict | None = None) -> list[str] | None:
    source = Path(records) / "baselines"
    if not source.is_dir():
        return None
    studies = []
    for upstream_path in sorted(source.glob("raw/*/*/upstream.json")):
        require(not upstream_path.is_symlink(), f"Baseline metadata is a symlink: {upstream_path}")
        upstream = read_json(upstream_path)
        archive = upstream.get("adapter_archive") or {}
        plan = _inside(source, upstream["plan"])
        adapter = _inside(source, archive["file"])
        require(sha256_file(adapter) == archive["sha256"], f"{archive['file']} differs from its recorded SHA-256")
        folder = target / upstream_path.parent.relative_to(source / "raw")
        folder.mkdir(parents=True, exist_ok=True)
        files = {"plan": folder / "plan.json", "upstream": folder / "upstream.json", "adapter_archive": folder / adapter.name}
        for source_file, key in ((plan, "plan"), (upstream_path, "upstream"), (adapter, "adapter_archive")):
            shutil.copyfile(source_file, files[key])
        extra = {}
        if upstream.get("study_archive"):
            descriptor = upstream["study_archive"]
            manifest_file = _inside(source, descriptor["manifest_file"])
            require(sha256_file(manifest_file) == descriptor["manifest_sha256"], "E12 study manifest differs from its sealed descriptor")
            files["study_manifest"] = folder / "study_files.json"
            shutil.copyfile(manifest_file, files["study_manifest"])
            extra["study_archive"] = {**descriptor, "artifact": "records:baselines",
                                      "transport": "release artifact; full study bytes are not copied into Git"}
        studies.append({"dataset": upstream["dataset"], "method": upstream["method"], **extra,
                        "adapter_commit": archive.get("commit"), "source_commit": upstream.get("source_commit"),
                        "upstream_repository": upstream.get("upstream_repository") or upstream.get("source_repository"),
                        **{key: {"file": path.relative_to(target).as_posix(), "sha256": sha256_file(path)}
                           for key, path in files.items()}})
    require(studies, "Published baselines record set contains no study plans and adapter archives")
    write_json(target / "INDEX.json", {"schema": BASELINE_SCHEMA, "source": "records:baselines",
               "description": "E12 study plans and adapter code copied from the published baselines record set; "
                              "the full sealed studies, checkpoints and predictions remain in that release asset",
               **(provenance or {}), "studies": studies, "files": view_files(target)})
    return [f"{study['dataset']}/{study['method']}" for study in studies]


def derive_baseline_studies(records: Path, checkout: Path, *, provenance: dict | None = None) -> list[str] | None:
    """Atomically copy the small E12 plans and adapters, without moving large study bytes into Git."""
    checkout = Path(checkout).resolve()
    if not (Path(records) / "baselines").is_dir():
        return None
    with tempfile.TemporaryDirectory(prefix=".cm-baseline-view-", dir=checkout.parent) as temporary:
        stage = Path(temporary)
        studies = _baseline_view(records, stage / BASELINE_STUDIES, provenance)
        _install_views(stage, checkout, [BASELINE_STUDIES.as_posix()])
    return studies


def _verify_reconstruction_inputs(result: dict, fetched: Path, verified: dict) -> None:
    """Bind the complete measured input set, not only the inputs a stale report happened to name."""
    inputs = result.get("inputs")
    require(isinstance(inputs, list) and all(isinstance(item, dict) for item in inputs),
            "Reconstruction inputs must be a list of input descriptors")
    expected_runs, expected_records, expected_campaigns = {}, {}, {}
    keys = ("source_sha256", "config_sha256", "unit_sha256")
    for item in verified["artifacts"]:
        path = Path(item["path"]).resolve()
        if item["kind"] == "run":
            identity = item["identity"]
            parts = identity.get("splits") if identity.get("campaign") else {"": identity}
            for name, identifiers in parts.items():
                expected_runs[str(path / name)] = {key: identifiers[key] for key in keys}
        elif item["kind"] == "records":
            expected_records[item["id"].split(":", 1)[1]] = item["identity"]["manifest_sha256"]
        elif item["kind"] == "campaign":
            expected_campaigns[str(path)] = item["id"].split(":", 1)[1]
    seen_runs, seen_records = {}, []
    for item in inputs:
        path = Path(item.get("path", "")).resolve()
        require(path.is_relative_to(fetched), f"Reconstruction input is outside the fetched release: {path}")
        if item.get("sha256"):
            require(path.is_file() and sha256_file(path) == item["sha256"], f"Reconstruction input changed: {path}")
        if item.get("kind") == "sealed_run":
            require(str(path) not in seen_runs, f"Reconstruction repeats a sealed run: {path}")
            seen_runs[str(path)] = item.get("identities")
            stages = {p.name: sha256_file(p) for p in sorted((path / "provenance/stages").glob("*.json"))}
            require(stages == item.get("stage_manifests"), f"Reconstruction run provenance changed: {path}")
        elif item.get("kind") == "measurement_records":
            require(path == fetched / "records/index.json", "Reconstruction uses another measurement-record index")
            seen_records.append(item)
    require(seen_runs == expected_runs, "Reconstruction sealed-run inputs differ from the complete fetched run set")
    require(len(seen_records) == (1 if expected_records else 0),
            "Reconstruction record inputs differ from the fetched record sets")
    if seen_records:
        entries = seen_records[0].get("sets")
        require(isinstance(entries, list) and len(entries) == len(expected_records)
                and {row.get("set"): row.get("manifest_sha256") for row in entries} == expected_records,
                "Reconstruction record identities differ from the fetched record sets")
    # Campaign input inventories are carried by the supplementary builder, even
    # when individual numbered display generators also consume the same campaign.
    from cm_campaigns.common import directory_inventory
    kinds = {"confirmation": "original_confirmation_campaign", "budget": "strict_budget_campaign"}
    observed = {}
    for item in (result.get("supplementary_build") or {}).get("inputs", []):
        if item.get("input_kind") not in kinds.values():
            continue
        path = str(Path(item.get("source", "")).resolve())
        require(path in expected_campaigns and path not in observed,
                "Reconstruction contains an unexpected or repeated campaign input")
        require(item.get("input_kind") == kinds[expected_campaigns[path]]
                and item.get("files") == directory_inventory(Path(path)),
                f"Reconstruction campaign files changed: {path}")
        observed[path] = expected_campaigns[path]
    require(observed == expected_campaigns, "Reconstruction campaign inputs differ from the fetched campaigns")


def derive(fetched: Path, *, workspace: Path | None = None, checkout: Path = ROOT,
           manifest_path: Path = MANIFEST, install_root: Path | None = None) -> dict:
    """Verify the release and workspace, stage every small view, then install them together.

    ``install_root`` (default: the checkout) is where the views are installed; the CPU demonstration's
    views go under demo/published/ so that they never replace the paper's views."""
    from scripts.reconstruct import source_inventory, verify_workspace
    from scripts.reconstruct_paper import summarize
    from scripts import run_manifests
    fetched, checkout = Path(fetched).resolve(), Path(checkout).resolve()
    install_root = checkout if install_root is None else Path(install_root).resolve()
    manifest = load_manifest(manifest_path)
    require((manifest.get("channel") == DEMO_CHANNEL) == (install_root != checkout),
            "Demonstration views are installed under demo/published/ and the paper's views in the checkout")
    verified = verify_fetched(fetched, manifest_path=manifest_path)
    links = [artifact_identity(entry, row["tier"]) for row in verified["artifacts"]
             for entry in manifest["artifacts"] if entry["id"] == row["id"]]
    require({row["id"] for row in links} == {entry["id"] for entry in manifest["artifacts"] if entry["status"] == PUBLISHED},
            "Fetch and verify every published artifact before deriving the committed views")
    provenance = {"artifact_manifest_sha256": sha256_file(manifest_path), "artifacts": links}
    builders = builder_source_inventory(checkout)
    result = None
    if workspace is not None:
        workspace = Path(workspace).resolve()
        result = verify_workspace(workspace)
        require(result.get("software_sources") == source_inventory(checkout),
                "Reconstruction tools or registered inputs changed; rebuild the workspace before deriving")
        _verify_reconstruction_inputs(result, fetched, verified)
    written, names = {}, []
    install_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cm-published-views-", dir=install_root.parent) as temporary:
        stage = Path(temporary)
        studies = _baseline_view(fetched / "records", stage / BASELINE_STUDIES, provenance)
        if studies:
            names.append(BASELINE_STUDIES.as_posix())
            written["baseline_studies"] = studies
        runs = fetched / "runs"
        found = [(STEMS[entry["id"].split(":", 1)[1]], Path(row["path"]))
                 for row in verified["artifacts"] for entry in manifest["artifacts"]
                 if entry["id"] == row["id"] and entry["kind"] == "run"]
        if found:
            target = stage / "manifests/published"
            run_manifests.write_runs(found, target, with_indices=True)
            write_json(target / "INDEX.json", {"schema": "cross-morse-published-run-manifests-1",
                       **provenance, "files": view_files(target)})
            names.append("manifests/published")
            written["split_and_seed_manifests"] = sorted(p.name for p in target.iterdir())
            observed = stage / "environment/observed"
            for stem, run in found:
                members = sorted(run.glob("split_*")) or [run]
                rows = []
                for member in members:
                    environment = contained(member, "environment.json")
                    row = {"run": member.relative_to(runs).as_posix(), "environment": read_json(environment)}
                    launcher = member / "launcher"
                    if launcher.is_dir():
                        row["launcher_records"] = {p.name: p.read_text(encoding="utf-8", errors="replace")
                                                   for p in sorted(launcher.glob("*.txt")) if p.stat().st_size < 2 << 20}
                    rows.append(row)
                write_json(observed / f"{stem}.json", {"schema": "cross-morse-observed-environment-1",
                           "dataset_run": stem, "members": rows})
            write_json(observed / "INDEX.json", {"schema": "cross-morse-observed-environments-1",
                       **provenance, "files": view_files(observed)})
            names.append("environment/observed")
            written["observed_environments"] = sorted(p.name for p in observed.glob("*.json"))
        if result is not None:
            displays = workspace / "displays"
            require(displays.is_dir(), f"No display outputs in {workspace}; reconstruct first")
            target = stage / "paper/source_data"
            for folder in ("tables", "data", "compare"):
                for path in sorted((displays / folder).rglob("*")) if (displays / folder).is_dir() else []:
                    if path.is_file() and path.suffix in (".csv", ".json"):
                        destination = target / path.relative_to(displays)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(path, destination)
            reports = {}
            for key, filename in (("reconstruction", "reconstruction.json"), ("workspace_files", "FILES.json")):
                destination = target / "reports" / filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(workspace / filename, destination)
                reports[key] = {"path": destination.relative_to(target).as_posix(), "sha256": sha256_file(destination)}
            status = summarize(result, manifest)
            write_json(target / "reports/status.json", status)
            reports["status"] = {"path": "reports/status.json", "sha256": sha256_file(target / "reports/status.json")}
            index = {"schema": "cross-morse-source-data-1", "label": result.get("label"), **provenance,
                     "inputs_sha256": hashlib.sha256(json.dumps(result.get("inputs"), sort_keys=True).encode()).hexdigest(),
                     "reconstruction": status, "reports": reports, "builder_sources": builders,
                     "software_sources": result.get("software_sources"), "files": view_files(target)}
            write_json(target / "INDEX.json", index)
            names.append("paper/source_data")
            written["source_data_files"] = len(index["files"])
        require(sha256_file(manifest_path) == provenance["artifact_manifest_sha256"], "Artifact manifest changed during derive")
        require(builder_source_inventory(checkout) == builders, "Builder sources changed during derive")
        _install_views(stage, install_root, names)
    return written


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def status_lines(manifest: dict) -> list[str]:
    lines = [f"Release: {manifest['release_url']}"]
    for entry in manifest["artifacts"]:
        if entry["status"] == PUBLISHED:
            size = entry.get("bytes", 0) / 2**30
            lines.append(f"  published  {entry['id']:<32} {entry.get('files', 0):>7} files  {size:8.2f} GiB")
        else:
            lines.append(f"  AWAITING   {entry['id']:<32} measured outputs not yet published")
    lines.append("Artifact identities and per-file inventories are recorded in the release manifest.")
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--check", action="store_true")
    p = sub.add_parser("pack")
    p.add_argument("--map", type=Path, required=True,
                   help="JSON object {artifact id: absolute directory}, e.g. {\"run:ogbn-arxiv\": \"/proj/.../run\"}")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--relationship", choices=RELATIONSHIPS,
                   help="explicit relationship for every mapped artifact; otherwise use map._relationships")
    p.add_argument("--require-complete", action="store_true", help="refuse a map missing any required artifact")
    p.add_argument("--part-bytes", type=int, default=PART_BYTES)
    p.add_argument("--demo", action="store_true",
                   help="pack the CPU demonstration (a software-fixture map; relationship reduced_scale_demonstration)")
    p = sub.add_parser("publish")
    p.add_argument("--staged", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--repository", default=REPOSITORY)
    p.add_argument("--tag", default=None)
    p.add_argument("--demo", action="store_true", help=f"publish to {DEMO_TAG} and release/demo_artifacts.json")
    p.add_argument("--api-url", default="https://api.github.com")
    p.add_argument("--download-base", default=None)
    p.add_argument("--target", default="main",
                   help="the commit or branch the release tag is created on, when the release does not exist yet")
    p = sub.add_parser("fetch")
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--tier", choices=["analysis", "complete"], default="analysis")
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--only", nargs="*")
    p.add_argument("--demo", action="store_true", help="the CPU demonstration's manifest")
    p = sub.add_parser("status")
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--demo", action="store_true", help="the CPU demonstration's manifest")
    p = sub.add_parser("map-template")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("derive")
    p.add_argument("--fetched", type=Path, required=True, help="a directory filled by `fetch`")
    p.add_argument("--workspace", type=Path, help="the reconstruction workspace written by scripts/reconstruct_paper.py")
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--demo", action="store_true", help="the CPU demonstration: views under demo/published/")
    p = sub.add_parser("verify")
    p.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    if args.command in ("publish", "fetch", "status", "derive"):
        if args.manifest is None:
            args.manifest = DEMO_MANIFEST if args.demo else MANIFEST
    if args.command == "publish" and args.tag is None:
        args.tag = DEMO_TAG if args.demo else TAG
    try:
        if args.command == "plan":
            current = load_manifest(args.manifest)
            planned = planned_manifest(current)
            if args.check:
                if planned != current:
                    print(f"{args.manifest} is not current; run: python -B scripts/paper_artifacts.py plan",
                          file=sys.stderr)
                    return 1
                print(f"{args.manifest} lists the {len(planned['artifacts'])} artifacts the displays require")
                return 0
            write_json(args.manifest, planned)
            print("\n".join(status_lines(planned)))
            return 0
        if args.command == "pack":
            document = read_json(args.map)
            require(isinstance(document, dict), "Artifact map must be an object")
            require(args.demo or not document.get("_fixture"), "A software-fixture production map cannot be published")
            mapping = {k: v for k, v in document.items() if not k.startswith("_") and v}
            require(all(isinstance(v, str) and Path(v).is_absolute() for v in mapping.values()),
                    "Every mapped artifact must name an absolute directory")
            required = {row["id"] for row in required_artifacts()}
            missing = sorted(required - set(mapping))
            require(not args.require_complete or not missing, "Publication map is incomplete: " + ", ".join(missing))
            if args.demo:
                require(args.relationship in (None, DEMO_RELATIONSHIP), "The demonstration has one relationship")
                relationships = {key: DEMO_RELATIONSHIP for key in mapping}
            else:
                relationships = {key: args.relationship or document.get("_relationships", {}).get(key)
                                 for key in mapping}
            require(all(value in RELATIONSHIPS or (args.demo and value == DEMO_RELATIONSHIP)
                        for value in relationships.values()),
                    "Declare every historical relationship with --relationship or map._relationships; "
                    "newly executed experiments use post_result_revision")
            unknown = sorted(set(mapping) - required)
            require(not unknown, f"Not required by any display: {unknown}")
            for artifact_id, directory in mapping.items():
                entry = pack_one(artifact_id, Path(directory), args.output_dir, relationship=relationships[artifact_id],
                                 part_bytes=args.part_bytes, scientific_only=not args.demo)
                parts = sum(len(t["parts"]) for t in entry["tiers"].values())
                print(f"packed {artifact_id}: {entry['files']} files, {parts} asset part(s)")
            missing = sorted(required - set(mapping))
            if missing:
                print("not in this map (still awaiting): " + ", ".join(missing))
            return 0
        if args.command == "publish":
            token = github_token()
            require(token, "Set GITHUB_TOKEN (contents: write on the repository) or log in with `gh auth login`")
            manifest = publish(args.staged, args.manifest, client=GitHub(args.repository, token, args.api_url),
                               tag=args.tag, target=args.target, download_base=args.download_base,
                               demonstration=args.demo)
            print("\n".join(status_lines(manifest)))
            print(f"Commit {args.manifest} so that `make paper` can retrieve the published artifacts.")
            return 0
        if args.command == "fetch":
            report = fetch(args.dest, manifest_path=args.manifest, tier=args.tier, only=args.only,
                           token=os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN"))
            for row in report["published"]:
                print(f"verified {row['id']} ({row['tier']}): {row['path']}")
            if report["awaiting"]:
                print("awaiting publication: " + ", ".join(report["awaiting"]), file=sys.stderr)
                return 1
            return 0
        if args.command == "status":
            manifest = load_manifest(args.manifest)
            print("\n".join(status_lines(manifest)))
            return 0 if manifest["status"] == PUBLISHED else 1
        if args.command == "map-template":
            template = map_template(args.output)
            print(f"wrote {args.output}: {sum(not k.startswith("_") for k in template)} artifacts to locate")
            return 0
        if args.command == "derive":
            print(json.dumps(derive(args.fetched, workspace=args.workspace, manifest_path=args.manifest,
                                    install_root=ROOT / "demo" / "published" if args.demo else None), indent=1))
            return 0
        if args.command == "verify":
            print(json.dumps(verify_unpacked(args.directory), indent=1))
            return 0
    except (ArtifactError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
