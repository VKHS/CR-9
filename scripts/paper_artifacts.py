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
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=False, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# What the paper needs
# ---------------------------------------------------------------------------

def _kind(artifact_id: str) -> str:
    kind = artifact_id.split(":", 1)[0]
    require(kind in STATEMENT_ITEMS, f"Unknown artifact kind: {artifact_id}")
    return kind


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


def blank_manifest() -> dict:
    return {"schema": SCHEMA, "repository": REPOSITORY, "release_tag": TAG,
            "release_url": f"https://github.com/{REPOSITORY}/releases/tag/{TAG}",
            "part_bytes": PART_BYTES,
            "statement": "release/availability_statement.json",
            "status": AWAITING, "artifacts": []}


def load_manifest(path: Path = MANIFEST) -> dict:
    if not Path(path).is_file():
        return blank_manifest()
    manifest = read_json(path)
    require(manifest.get("schema") == SCHEMA, f"Unexpected artifact manifest schema in {path}")
    return manifest


def planned_manifest(existing: dict, root: Path = ROOT) -> dict:
    """Refresh the required list, keeping every published entry and its identities."""
    manifest = blank_manifest() | {k: v for k, v in existing.items() if k not in {"artifacts", "status"}}
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
    manifest["status"] = PUBLISHED if all(a["status"] == PUBLISHED for a in artifacts) else AWAITING
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
        for name in sorted(names):
            path = Path(folder) / name
            require(not path.is_symlink(), f"Symlinks are not archived: {path}")
            relative = path.relative_to(root).as_posix()
            if relative in (INVENTORY_NAME, RUN_ARCHIVE_MANIFEST, SUBSET_NAME):
                continue
            entries.append({"path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return entries


def _tar(output: Path, top: str, root: Path, files: list[str], extra: dict[str, dict]) -> None:
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
            source = root / relative
            info = tar.gettarinfo(str(source), arcname=f"{top}/{relative}")
            info.mtime, info.uid, info.gid, info.uname, info.gname, info.mode = 0, 0, 0, "", "", 0o644
            info.pax_headers = {}
            with source.open("rb") as handle:
                tar.addfile(info, handle)


def _verify_source(artifact_id: str, source: Path, *, scientific_only: bool = True) -> dict:
    """Validate a supplied directory with the repository's own checkers before packing."""
    kind, _, name = artifact_id.partition(":")
    if kind == "run":
        from scripts import archive_run
        splits = sorted(p for p in source.glob("split_*") if (p / "run_state.json").is_file())
        if splits:
            require(name == "Roman-empire", f"Only Roman-empire is published as a split campaign: {source}")
            require(len(splits) == 10, f"Roman-empire campaign needs ten split runs, found {len(splits)}")
            chains = {p.name: archive_run.verify_stage_chain(p) for p in splits}
            datasets = {(read_json(p / "run_state.json").get("task") or {}).get("dataset") for p in splits}
            require(datasets == {name}, f"{artifact_id}: split runs belong to {sorted(map(str, datasets))}")
            return {"campaign": True, "splits": {k: {"source_sha256": v["source_sha256"],
                                                     "config_sha256": v["config_sha256"],
                                                     "unit_sha256": v["unit_sha256"]} for k, v in chains.items()}}
        chain = archive_run.verify_stage_chain(source)
        state = read_json(source / "run_state.json")
        require((state.get("task") or {}).get("dataset") == name,
                f"{artifact_id}: the run at {source} is a {(state.get('task') or {}).get('dataset')} run")
        require(state.get("scientific_run") is True or not scientific_only,
                f"{artifact_id}: {source} is not a full scientific run")
        return {"campaign": False, "source_sha256": chain["source_sha256"], "config_sha256": chain["config_sha256"],
                "unit_sha256": chain["unit_sha256"], "profile": state.get("profile")}
    if kind == "records":
        from scripts import records as measurement_records
        specs = measurement_records.load_specs(source.parent, ROOT / "records/specs.json")
        require(name in specs["sets"], f"{artifact_id}: no record contract named {name}")
        require(source.name == name, f"{artifact_id}: a record-set directory must be named {name}")
        entry = measurement_records.verify_set(source, specs["sets"][name], specs)
        require(not entry.get("synthetic") or not scientific_only,
                f"{artifact_id}: synthetic example records are not paper artifacts")
        return {"manifest_sha256": entry["manifest_sha256"], "rows": entry.get("rows")}
    if kind == "campaign":
        require(any(source.iterdir()), f"{artifact_id}: empty campaign directory {source}")
        return {"files": sum(1 for p in source.rglob("*") if p.is_file())}
    require(any(source.iterdir()), f"{artifact_id}: empty environment capture {source}")
    return {"files": sum(1 for p in source.rglob("*") if p.is_file())}


def split_parts(path: Path, part_bytes: int = PART_BYTES) -> list[dict]:
    """Split ``path`` into ``<name>.partNNN`` files when it exceeds ``part_bytes``."""
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


def pack_one(artifact_id: str, source: Path, out_dir: Path, *, relationship: str,
             part_bytes: int = PART_BYTES, scientific_only: bool = True) -> dict:
    source = Path(source).resolve()
    require(source.is_dir(), f"{artifact_id}: {source} is not a directory")
    kind = _kind(artifact_id)
    identity = _verify_source(artifact_id, source, scientific_only=scientific_only)
    stem = asset_stem(artifact_id)
    files = _inventory(source)
    inventory = {"schema": "cross-morse-artifact-inventory-1", "artifact": artifact_id, "kind": kind,
                 "source_directory_name": source.name, "identity": identity, "files": files}
    inventory["inventory_sha256"] = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    out_dir.mkdir(parents=True, exist_ok=True)
    complete = out_dir / f"{stem}.tar.gz"
    require(not complete.exists(), f"Refusing to overwrite {complete}")
    _tar(complete, stem, source, [f["path"] for f in files], {INVENTORY_NAME: inventory})
    tiers = {"complete": {"archive": complete.name, "bytes": complete.stat().st_size,
                          "sha256": sha256_file(complete)}}
    if kind == "run":
        light = [f for f in files if not is_heavy(f["path"])]
        omitted = [f for f in files if is_heavy(f["path"])]
        analysis = out_dir / f"{stem}.analysis.tar.gz"
        subset = {"schema": "cross-morse-analysis-subset-1", "artifact": artifact_id,
                  "complete_inventory_sha256": inventory["inventory_sha256"],
                  "rule": "checkpoints (*.pt), member/candidate prediction arrays, final probability arrays, "
                          "final/test_predictions.npz and out-of-fold prediction arrays are in the complete tier",
                  "omitted": omitted}
        _tar(analysis, stem, source, [f["path"] for f in light], {INVENTORY_NAME: inventory, SUBSET_NAME: subset})
        tiers["analysis"] = {"archive": analysis.name, "bytes": analysis.stat().st_size,
                             "sha256": sha256_file(analysis)}
    for tier in tiers.values():
        tier["parts"] = split_parts(out_dir / tier["archive"], part_bytes)
    entry = {"id": artifact_id, "kind": kind, "relationship": relationship, "identity": identity,
             "inventory_sha256": inventory["inventory_sha256"], "files": len(files),
             "bytes": sum(f["bytes"] for f in files), "contents": content_counts(files),
             "tiers": tiers, "packed_utc": utc_now()}
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
    with urllib.request.urlopen(request, timeout=600) as response, temporary.open("wb") as out:
        shutil.copyfileobj(response, out, 1 << 20)
    os.replace(temporary, target)


def publish(staged: Path, manifest_path: Path = MANIFEST, *, client: GitHub, tag: str = TAG,
            target: str = "main", download_base: str | None = None) -> dict:
    """Upload every staged part not yet present with identical size; record them in the manifest."""
    staged = Path(staged)
    entries = [read_json(p) for p in sorted(staged.glob("*.entry.json"))]
    require(entries, f"No packed artifacts (*.entry.json) under {staged}")
    release = client.release(tag) or client.create_release(
        tag, target=target, body="Measured outputs of the experiments reported in the CROSS-MORSE paper. "
                                 "release/paper_artifacts.json in the repository lists every asset with its "
                                 "SHA-256; `make paper` downloads, verifies and rebuilds every table and figure.")
    present = client.assets(release)
    base = (download_base or f"https://github.com/{client.repository}/releases/download/{tag}").rstrip("/")
    for entry in entries:
        for tier in entry["tiers"].values():
            for part in tier["parts"]:
                path = staged / part["asset"]
                require(path.is_file() and path.stat().st_size == part["bytes"] and sha256_file(path) == part["sha256"],
                        f"Staged part changed or missing: {path}")
                asset = present.get(part["asset"])
                digest = ((asset or {}).get("digest") or "").removeprefix("sha256:")
                if not (asset is not None and asset.get("size") == part["bytes"]
                        and asset.get("state", "uploaded") == "uploaded" and digest == part["sha256"]):
                    # Absent, incomplete, different, or of unknown digest: replace it with the staged bytes.
                    if asset is not None:
                        client.delete_asset(asset)
                    asset = client.upload(release, path, part["asset"])
                    require(asset.get("size") == part["bytes"], f"Upload size mismatch: {part['asset']}")
                    digest = (asset.get("digest") or "").removeprefix("sha256:")
                    require(not digest or digest == part["sha256"], f"Upload digest mismatch: {part['asset']}")
                part["url"] = f"{base}/{urllib.parse.quote(part['asset'])}"
    manifest = planned_manifest(load_manifest(manifest_path))
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
    manifest.update(artifacts=updated, release_url=release.get("html_url", manifest["release_url"]),
                    status=PUBLISHED if all(a["status"] == PUBLISHED for a in updated) else AWAITING)
    write_json(manifest_path, manifest)
    return manifest


# ---------------------------------------------------------------------------
# Retrieval and verification
# ---------------------------------------------------------------------------

def _safe_extract(archive: Path, destination: Path) -> str:
    """Extract a single-top-level archive with regular files only; return the top directory."""
    with tarfile.open(archive, "r:*") as tar:
        members = tar.getmembers()
        tops = set()
        for member in members:
            name = PurePosixPath(member.name)
            require(not name.is_absolute() and ".." not in name.parts, f"Unsafe archive path: {member.name}")
            require(member.isfile() or member.isdir(), f"Only regular files are allowed: {member.name}")
            tops.add(name.parts[0])
        require(len(tops) == 1, f"An artifact archive has one top-level directory, found {sorted(tops)}")
        top = tops.pop()
        target = destination / top
        require(not target.exists(), f"Refusing to overwrite {target}")
        tar.extractall(destination, members=members, filter="data")
    return top


def verify_unpacked(directory: Path) -> dict:
    """Check every extracted file against the packed inventory; analysis tiers list what they omit."""
    directory = Path(directory)
    inventory = read_json(directory / INVENTORY_NAME)
    files = {f["path"]: f for f in inventory["files"]}
    require(hashlib.sha256(json.dumps(inventory["files"], sort_keys=True).encode()).hexdigest()
            == inventory["inventory_sha256"], f"Inventory digest changed in {directory}")
    present = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    present -= {INVENTORY_NAME, SUBSET_NAME}
    omitted = set()
    subset_path = directory / SUBSET_NAME
    if subset_path.is_file():
        subset = read_json(subset_path)
        require(subset["complete_inventory_sha256"] == inventory["inventory_sha256"],
                f"Analysis subset belongs to another inventory: {directory}")
        omitted = {f["path"] for f in subset["omitted"]}
        require(all(is_heavy(p) and files.get(p) == f for p, f in ((f["path"], f) for f in subset["omitted"])),
                f"Analysis subset omits a file that the analysis tier must contain: {directory}")
    require(present | omitted == set(files) and not (present & omitted),
            f"Unpacked files differ from the inventory in {directory}")
    for relative in sorted(present):
        path = directory / relative
        expected = files[relative]
        require(path.stat().st_size == expected["bytes"] and sha256_file(path) == expected["sha256"],
                f"File differs from its inventory: {directory / relative}")
    return {"files": len(present), "omitted": len(omitted), "inventory_sha256": inventory["inventory_sha256"]}


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
    cache.mkdir(parents=True, exist_ok=True)
    stamps = destination / ".artifacts"
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
        if stamp.is_file():
            previous = read_json(stamp)
            if previous.get("sha256") == spec["sha256"] and Path(previous["path"]).is_dir():
                report["published"].append({"id": entry["id"], "path": previous["path"], "tier": use, "reused": True})
                continue
            raise ArtifactError(f"{previous['path']} holds another version of {entry['id']}; remove it and "
                                f"{stamp} first")
        archive = cache / spec["archive"]
        if not (archive.is_file() and sha256_file(archive) == spec["sha256"]):
            pieces = []
            for part in spec["parts"]:
                piece = cache / part["asset"]
                if not (piece.is_file() and piece.stat().st_size == part["bytes"]
                        and sha256_file(piece) == part["sha256"]):
                    download(part["url"], piece, token=token)
                require(piece.stat().st_size == part["bytes"] and sha256_file(piece) == part["sha256"],
                        f"Downloaded part differs from the manifest: {part['asset']}")
                pieces.append(piece)
            if pieces != [archive]:
                with archive.open("wb") as out:
                    for piece in pieces:
                        with piece.open("rb") as handle:
                            shutil.copyfileobj(handle, out, 1 << 20)
            require(sha256_file(archive) == spec["sha256"], f"Reassembled archive differs: {spec['archive']}")
        with tempfile.TemporaryDirectory(dir=folder, prefix=".unpack-") as temporary:
            top = _safe_extract(archive, Path(temporary))
            unpacked = Path(temporary) / top
            check = verify_unpacked(unpacked)
            require(check["inventory_sha256"] == entry["inventory_sha256"],
                    f"{entry['id']}: unpacked inventory differs from the manifest")
            inventory = read_json(unpacked / INVENTORY_NAME)
            local = STEMS[name] if kind == "run" else inventory["source_directory_name"]
            target = folder / local
            require(not target.exists(), f"{target} exists; remove it to fetch {entry['id']}")
            if kind != "run":
                # Record sets and campaigns are verified against their own manifests; keep them byte-identical.
                (unpacked / INVENTORY_NAME).unlink()
            os.replace(unpacked, target)
        write_json(stamp, {"id": entry["id"], "tier": use, "sha256": spec["sha256"], "path": str(target),
                           "inventory_sha256": entry["inventory_sha256"], "verified": check,
                           "inventory": inventory if kind != "run" else INVENTORY_NAME})
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
    """Verify the sealed stage chain of a run whose checkpoints and prediction arrays are omitted.

    Every present sealed file must match its stage manifest byte for byte; every absent
    sealed file must be a checkpoint or prediction array the analysis subset declares with
    the identical SHA-256 and size. Returns the run identities and the source resolution.
    """
    from cross_morse import provenance as pv
    from scripts._source_snapshots import resolve_runtime_source, runtime_source_fingerprint
    root = Path(run_root).resolve()
    config = read_json(root / "config.json")
    cache, sealed, identities, parent = {}, {}, None, None
    for index, stage in enumerate(pv.STAGES):
        manifest_path = pv.stage_manifest_path(root, stage)
        require(manifest_path.is_file(), f"{root}: stage manifest {stage} is missing")
        value = read_json(manifest_path)
        pv._validate_manifest_shape(value)
        require(value["stage"] == stage, f"{root}: stage manifest name and content differ ({stage})")
        ids = {k: value[k] for k in ("source_sha256", "config_sha256", "unit_sha256")}
        if identities is None:
            require(ids["config_sha256"] == pv._json_hash(config), f"{root}: configuration content changed")
            require(ids["unit_sha256"] == sha256_file(root / pv.UNIT_MANIFEST_PATH), f"{root}: split identity changed")
            identities = ids
        require(ids == identities and value["dataset"] == config["dataset"], f"{root}: identities differ across stages")
        require(value["parent_manifest"] == parent, f"{root}: parent manifest link broken at {stage}")
        for field in ("inputs", "outputs"):
            for declared in value[field]:
                relative = declared.get("path") if isinstance(declared, dict) else None
                require(isinstance(relative, str), f"{root}: malformed {stage}.{field} descriptor")
                if (root / relative).is_file():
                    pv._verify_artifact(root, declared, cache)
                else:
                    require(is_heavy(relative) and omitted.get(relative) == declared,
                            f"{root}: sealed file {relative} is neither present nor a declared omission")
                if field == "outputs":
                    require(relative not in sealed, f"{root}: {relative} sealed twice")
                    sealed[relative] = declared
                else:
                    require(index == 0 or relative not in sealed or sealed[relative] == declared,
                            f"{root}: {stage} input differs from its sealed output: {relative}")
        parent = pv._cached_artifact(root, manifest_path, cache)
    try:
        resolve_runtime_source(identities["source_sha256"], ROOT)
        resolution = "current or bundled runtime source"
    except ValueError:
        own = root / "source"
        require(own.is_dir() and runtime_source_fingerprint(own) == identities["source_sha256"],
                f"{root}: the runtime source {identities['source_sha256'][:12]} of this run is not available")
        resolution = "the run's own frozen source snapshot"
    state = read_json(root / "run_state.json")
    require(state.get("status") == "COMPLETE" and state.get("last_complete_stage") == "evaluate",
            f"{root}: run bookkeeping is not complete")
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
    template = {"_instructions": "Fill every path with an absolute directory on the cluster, delete entries that do not "
                                 "apply, then run: bash hpc/publish_paper_artifacts_arrhenius.sh all <this file>",
                "_kinds": hints}
    template.update({row["id"]: "" for row in rows})
    write_json(path, template)
    return template


BASELINE_STUDIES = Path("experiments") / "baselines"
BASELINE_SCHEMA = "cross-morse-baseline-studies-1"


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()) and path.is_file(), f"{relative} is not a file inside {root}")
    return path


def derive_baseline_studies(records: Path, checkout: Path) -> list[str] | None:
    """The E12 study plans and adapter code of the published baselines set, as files of the checkout.

    The baselines record set carries, for every external comparison, the validated study plan (48 HPO
    configurations, 12 optimization seeds, pinned upstream source and adapter commits), ``upstream.json``
    (upstream repository and commits) and the adapter checkout archived at its pinned commit. They are copied
    byte for byte into ``experiments/baselines/<dataset>/<method>/`` with ``INDEX.json``.
    """
    source = Path(records) / "baselines"
    if not source.is_dir():
        return None
    target = Path(checkout) / BASELINE_STUDIES
    if target.exists():
        shutil.rmtree(target)
    studies = []
    for upstream_path in sorted(source.glob("raw/*/*/upstream.json")):
        upstream = read_json(upstream_path)
        archive = upstream.get("adapter_archive") or {}
        plan = _inside(source, upstream["plan"])
        adapter = _inside(source, archive["file"])
        require(sha256_file(adapter) == archive["sha256"], f"{archive['file']} differs from its recorded SHA-256")
        folder = target / upstream_path.parent.relative_to(source / "raw")
        folder.mkdir(parents=True, exist_ok=True)
        files = {"plan": folder / "plan.json", "upstream": folder / "upstream.json", "adapter_archive": folder / adapter.name}
        shutil.copyfile(plan, files["plan"])
        shutil.copyfile(upstream_path, files["upstream"])
        shutil.copyfile(adapter, files["adapter_archive"])
        studies.append({"dataset": upstream["dataset"], "method": upstream["method"],
                        "adapter_commit": archive.get("commit"), "source_commit": upstream.get("source_commit"),
                        "upstream_repository": upstream.get("upstream_repository") or upstream.get("source_repository"),
                        **{key: {"file": path.relative_to(target).as_posix(), "sha256": sha256_file(path)}
                           for key, path in files.items()}})
    if not studies:
        return None
    write_json(target / "INDEX.json", {"schema": BASELINE_SCHEMA, "source": "records:baselines",
                                       "description": "E12 study plans and adapter code of the external comparisons, "
                                                      "copied from the published baselines record set",
                                       "studies": studies})
    return [f"{s['dataset']}/{s['method']}" for s in studies]


def derive(fetched: Path, *, workspace: Path | None = None, checkout: Path = ROOT) -> dict:
    """Commit-sized views of the published outputs: split/seed manifests, observed environments, source data,
    and the E12 study plans and adapter code."""
    import subprocess
    fetched, checkout = Path(fetched).resolve(), Path(checkout).resolve()
    written = {}
    studies = derive_baseline_studies(fetched / "records", checkout)
    if studies:
        written["baseline_studies"] = studies
    runs = fetched / "runs"
    if runs.is_dir() and any(p.is_dir() for p in runs.iterdir()):
        target = checkout / "manifests" / "published"
        if target.exists():
            shutil.rmtree(target)
        subprocess.run([sys.executable, "-B", str(checkout / "scripts/run_manifests.py"), "runs",
                        "--runs-root", str(runs), "--label", "published", "--output-root",
                        str(checkout / "manifests"), "--with-indices"], check=True, cwd=checkout)
        written["split_and_seed_manifests"] = sorted(p.name for p in target.iterdir())
        observed = checkout / "environment" / "observed"
        observed.mkdir(parents=True, exist_ok=True)
        for run in sorted(p for p in runs.iterdir() if p.is_dir() and not p.name.startswith(".")):
            members = sorted(run.glob("split_*")) or [run]
            rows = []
            for member in members:
                environment = member / "environment.json"
                row = {"run": member.relative_to(runs).as_posix(),
                       "environment": read_json(environment) if environment.is_file() else None}
                launcher = member / "launcher"
                if launcher.is_dir():
                    row["launcher_records"] = {p.name: p.read_text(encoding="utf-8", errors="replace")
                                               for p in sorted(launcher.glob("*.txt")) if p.stat().st_size < 2 << 20}
                rows.append(row)
            write_json(observed / f"{run.name}.json", {"schema": "cross-morse-observed-environment-1",
                                                       "dataset_run": run.name, "members": rows})
        written["observed_environments"] = sorted(p.name for p in observed.glob("*.json"))
    if workspace is not None:
        displays = Path(workspace) / "displays"
        require(displays.is_dir(), f"No display outputs in {workspace}; run scripts/reconstruct_paper.py first")
        target = checkout / "paper" / "source_data"
        if target.exists():
            shutil.rmtree(target)
        index = {}
        for folder in ("tables", "data", "compare"):
            for path in sorted((displays / folder).rglob("*.csv")) if (displays / folder).is_dir() else []:
                relative = path.relative_to(displays).as_posix()
                destination = target / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
                index[relative] = sha256_file(destination)
        reconstruction = read_json(Path(workspace) / "reconstruction.json")
        generators = reconstruction.get("display_generators") or {}
        comparison = generators.get("comparison_with_printed_values") or {}
        write_json(target / "INDEX.json", {
            "schema": "cross-morse-source-data-1", "label": reconstruction.get("label"),
            "inputs_sha256": hashlib.sha256(json.dumps(reconstruction.get("inputs"), sort_keys=True).encode()).hexdigest(),
            "reconstruction": {"displays": len(reconstruction["displays"]),
                               "complete": sum(1 for d in reconstruction["displays"] if d["full_numerical_display"]),
                               "values_exact": comparison.get("exact", 0), "values_differ": comparison.get("differs", 0),
                               "generator_errors": sorted((generators.get("errors") or {}))},
            "files": index})
        written["source_data_files"] = len(index)
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
    published = sum(1 for e in manifest["artifacts"] if e["status"] == PUBLISHED)
    lines.append(f"{published} of {len(manifest['artifacts'])} required artifacts published")
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
    p.add_argument("--relationship", default="exact_result_producing_run",
                   choices=["exact_result_producing_run", "post_result_revision", "refactor_with_prediction_equivalence"])
    p.add_argument("--part-bytes", type=int, default=PART_BYTES)
    p = sub.add_parser("publish")
    p.add_argument("--staged", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--repository", default=REPOSITORY)
    p.add_argument("--tag", default=TAG)
    p.add_argument("--api-url", default="https://api.github.com")
    p.add_argument("--download-base", default=None)
    p = sub.add_parser("fetch")
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--tier", choices=["analysis", "complete"], default="analysis")
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--only", nargs="*")
    p = sub.add_parser("status")
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p = sub.add_parser("map-template")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("derive")
    p.add_argument("--fetched", type=Path, required=True, help="a directory filled by `fetch`")
    p.add_argument("--workspace", type=Path, help="the reconstruction workspace written by scripts/reconstruct_paper.py")
    p = sub.add_parser("verify")
    p.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
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
            mapping = {k: v for k, v in read_json(args.map).items() if not k.startswith("_") and v}
            required = {row["id"] for row in required_artifacts()}
            unknown = sorted(set(mapping) - required)
            require(not unknown, f"Not required by any display: {unknown}")
            for artifact_id, directory in mapping.items():
                entry = pack_one(artifact_id, Path(directory), args.output_dir, relationship=args.relationship,
                                 part_bytes=args.part_bytes)
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
                               tag=args.tag, download_base=args.download_base)
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
            print(f"wrote {args.output}: {len(template) - 2} artifacts to locate")
            return 0
        if args.command == "derive":
            print(json.dumps(derive(args.fetched, workspace=args.workspace), indent=1))
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
