#!/usr/bin/env python3
"""Fetch pinned run archives by published Zenodo version DOI, then verify locally.

The input index is read only. Public record identity, asset size/checksum, the
registered SHA-256, archive membership and sealed stage chains must agree.
Existing verified downloads are reused; conflicting files are preserved and
reported. --local supplies a cache, not an exemption from DOI verification.
--status makes no network requests and cannot establish public availability.
--force repairs an existing run directory only after staging a verified copy.
No archive code is executed or unpickled. Standard library only.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cm_campaigns import archives as zenodo  # noqa: E402
from cm_campaigns.common import directory_inventory, loads, valid_hash, write_json  # noqa: E402
from scripts.archive_run import MANIFEST_NAME, verify_stage_chain  # noqa: E402
from scripts.deposit_zenodo import DATASETS, file_identity, verified_run_manifest  # noqa: E402
from scripts.ingest_run import INDEX, STEMS, ingest, load_index  # noqa: E402


class FetchError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise FetchError(message)


def _safe_path(path):
    require(not path.is_symlink() and not any(p.is_symlink() for p in path.parents), f"Path traverses a symlink: {path}")


def _entries(index):
    entries = index.get("runs")
    require(isinstance(entries, list), "Index runs must be a list")
    seen, filenames = set(), set()
    for entry in entries:
        require(isinstance(entry, dict), "Invalid index entry")
        dataset = entry.get("dataset")
        require(dataset in STEMS and dataset not in seen, "Unknown or duplicate index dataset")
        seen.add(dataset)
        require(entry.get("path") == entry.get("stem") == STEMS[dataset], "Index path/stem must be the canonical dataset directory")
        name = entry.get("archive_name")
        require(isinstance(name, str) and name not in ("", ".", "..") and "/" not in name and "\\" not in name
                and not any(ord(c) < 32 for c in name) and name not in filenames, "Unsafe or ambiguous archive filename")
        filenames.add(name)
        require(valid_hash(entry.get("archive_sha256")) and type(entry.get("archive_bytes")) is int
                and entry["archive_bytes"] > 0, "Index needs a SHA-256 and positive archive byte count")
        require(entry.get("kind") in ("run", "campaign") and (entry["kind"] != "campaign" or dataset == "Roman-empire"), "Invalid archive kind")
        require(type(entry.get("scientific_run")) is bool and entry.get("profile") in ("smoke", "full"), "Invalid run profile/scope")
        sources = entry.get("run_source_sha256")
        require(isinstance(sources, list) and sources and len(sources) == len(set(sources))
                and all(valid_hash(v) for v in sources), "Index needs the run source identity/identities")
        if entry.get("deposit_doi"):
            zenodo.zenodo_id(entry["deposit_doi"])
    return entries


def _matches_file(path, entry):
    identity = file_identity(path)
    require(identity["bytes"] == entry["archive_bytes"] and identity["sha256"] == entry["archive_sha256"],
            f"Archive differs from repository pin: {path}")
    return identity


def _matches_manifest(manifest, entry):
    for key in ("dataset", "kind", "profile", "scientific_run", "archive_sha256", "archive_bytes"):
        require(manifest.get(key) == entry.get(key), f"Archive/index {key} mismatch")
    sources = manifest["source_sha256"] if manifest["kind"] == "campaign" else [manifest["source_sha256"]]
    require(sorted(sources) == sorted(entry["run_source_sha256"]), "Archive/index run source identity mismatch")
    if "files" in entry:
        require(len(manifest["files"]) == entry["files"], "Archive/index file count mismatch")


def present_and_verified(runs_root: Path, entry: dict, manifest: dict | None = None) -> bool:
    """Local integrity only; callers separately establish the published DOI binding."""
    path = runs_root / entry["path"]
    if not path.is_dir() or path.is_symlink():
        return False
    try:
        _safe_path(path)
        if manifest is None:
            # Do not claim a tree is bound to a pinned archive without that archive.
            return False
        actual = directory_inventory(path, exclude=(MANIFEST_NAME,))
        require(actual == sorted(manifest["files"], key=lambda row: row["path"]), "Unpacked tree differs from archived inventory")
        raw_manifest = loads((path / MANIFEST_NAME).read_text(encoding="utf-8"))
        expected = {k: v for k, v in manifest.items() if k not in ("archive_sha256", "archive_bytes", "top_level_directory")}
        require(raw_manifest == expected, "Unpacked archive manifest changed")
        roots = [path]
        if entry["kind"] == "campaign":
            require({p.name for p in path.glob("split_*")} == {f"split_{i:02d}" for i in range(10)}, "Incomplete Roman campaign")
            roots = [path / f"split_{i:02d}" for i in range(10)]
        sources = sorted({verify_stage_chain(p)["source_sha256"] for p in roots})
        require(sources == sorted(entry["run_source_sha256"]), "Unpacked source identity mismatch")
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return True


def local_candidate(entry: dict, directories: list[Path]) -> Path | None:
    for directory in directories:
        candidate = directory / entry["archive_name"]
        _safe_path(candidate)
        if candidate.exists():
            _matches_file(candidate, entry)
            return candidate
    return None


def fetch_entry(entry: dict, *, deposits_dir: Path, local: Path | None = None, tools=zenodo) -> Path:
    doi = entry.get("deposit_doi")
    require(doi, f"{entry['dataset']}: no deposit DOI is registered")
    metadata = tools.get_metadata(doi)
    rid = zenodo.zenodo_id(doi)
    require(str(metadata.get("id")) == rid and metadata.get("doi") == f"10.5281/zenodo.{rid}"
            and metadata.get("is_published", True) is True and metadata.get("metadata", {}).get("publication_date"), "DOI is not a published version record")
    asset = zenodo.asset_metadata(doi, entry["archive_name"], metadata)
    require(asset["bytes"] == entry["archive_bytes"], "Published asset size differs from the repository pin")
    binding = dict(asset, sha256=entry["archive_sha256"])

    def verify(path):
        _matches_file(path, entry)
        bound = zenodo.bind_local(path, doi, entry["archive_name"], metadata=metadata)
        require(bound["sha256"] == entry["archive_sha256"], "Published asset checksum differs from pinned local bytes")

    target = local or deposits_dir / entry["archive_sha256"] / entry["archive_name"]
    _safe_path(target)
    if target.exists():
        verify(target)  # Preserve a corrupt/conflicting cache file; never unlink it.
        return target
    require(local is None, "Selected local archive disappeared")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cm-fetch-", dir=target.parent) as temporary:
        pending = Path(temporary) / entry["archive_name"]
        tools.download(binding, pending)
        verify(pending)
        os.link(pending, target)  # Exclusive: never overwrite a file that appeared.
    return target


def _materialize(archive, manifest, entry, runs_root, force):
    """Use the existing importer in isolation; never rewrite the input index."""
    target = runs_root / entry["path"]
    _safe_path(target)
    require(not target.exists() or force, f"Existing run differs from deposit: {target}; inspect it or use --force to repair")
    runs_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cm-ingest-", dir=runs_root) as temporary:
        stage = Path(temporary)
        staged_root = stage / "runs"
        staged_root.mkdir()
        registered = ingest(archive, doi=entry.get("deposit_doi"), runs_root=staged_root,
                            index_path=stage / "index.json", replace=False)
        require(all(registered.get(k) == entry.get(k) for k in ("dataset", "kind", "profile", "scientific_run", "archive_sha256", "archive_bytes")),
                "Imported archive differs from index")
        require(present_and_verified(staged_root, entry, manifest), "Staged run failed archive and chain verification")
        backup = stage / "previous"
        if target.exists():
            require(target.is_dir(), "Existing run target is not a directory")
            target.rename(backup)
        try:
            require(not target.exists(), "Run destination appeared during ingestion")
            (staged_root / entry["path"]).rename(target)
        except BaseException:
            if backup.exists() and not target.exists():
                backup.rename(target)
            raise


def run(*, index_path: Path, runs_root: Path, deposits_dir: Path, datasets: set[str] | None,
        local: list[Path], status_only: bool, force: bool, tools=zenodo) -> dict:
    require(datasets is None or bool(datasets) and datasets <= set(DATASETS), "Unknown or empty dataset selection")
    _safe_path(index_path)
    require(index_path.is_file(), "Run index does not exist")
    _safe_path(runs_root)
    _safe_path(deposits_dir)
    entries = _entries(load_index(index_path))
    required = set(DATASETS) if datasets is None else datasets
    selected = [e for e in entries if e["dataset"] in required]
    missing = sorted(required - {e["dataset"] for e in selected})
    report = {"schema": "cross-morse-fetch-report-1", "entries": [], "fetched": 0, "present": 0,
              "missing_doi": 0, "failed": 0, "missing_datasets": missing,
              "public_verification": {"complete": False, "verified_datasets": [],
                                      "scope": "published archive bytes and local stage integrity; scientific agreement requires separate audit"}}
    for entry in selected:
        row = {"dataset": entry["dataset"], "doi": entry.get("deposit_doi"), "archive": entry["archive_name"],
               "sha256": entry["archive_sha256"], "status": None, "detail": "", "public_verified": False}
        try:
            if not entry.get("deposit_doi"):
                row.update(status="NO_DOI", detail="No published version DOI registered; local presence cannot establish public availability")
                report["missing_doi"] += 1
            elif status_only:
                row.update(status="REGISTERED", detail="Index only; DOI and bytes have not been checked")
            else:
                candidate = local_candidate(entry, local)
                archive = fetch_entry(entry, deposits_dir=deposits_dir, local=candidate, tools=tools)
                manifest = verified_run_manifest(archive)
                _matches_manifest(manifest, entry)
                if not force and present_and_verified(runs_root, entry, manifest):
                    row.update(status="PRESENT", detail="Published bytes, exact local inventory and stage chains verified")
                    report["present"] += 1
                else:
                    _materialize(archive, manifest, entry, runs_root, force)
                    row.update(status="INGESTED", detail="Verified published archive materialized; input index unchanged")
                    report["fetched"] += 1
                row["public_verified"] = True
                report["public_verification"]["verified_datasets"].append(entry["dataset"])
        except (OSError, ValueError, KeyError, TypeError, tarfile.TarError) as exc:
            row.update(status="FAILED", detail=str(exc))
            report["failed"] += 1
        report["entries"].append(row)
    report["public_verification"]["complete"] = (not status_only and not missing and not report["failed"]
        and not report["missing_doi"] and set(report["public_verification"]["verified_datasets"]) == required)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--deposits-dir", type=Path, default=ROOT.parent / "cm-deposits")
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--local", action="append", type=Path, default=[])
    parser.add_argument("--status", action="store_true", help="Inspect registration only; no network or writes except an optional report")
    parser.add_argument("--force", action="store_true", help="Repair an existing run only after a replacement verifies")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.output:
            _safe_path(args.output)
            require(not args.output.resolve().is_relative_to(ROOT.resolve())
                    and not args.output.resolve().is_relative_to(args.runs_root.resolve())
                    and args.output.resolve() != args.index.resolve(), "Keep reports outside the checkout, run data and input index")
        report = run(index_path=args.index, runs_root=args.runs_root, deposits_dir=args.deposits_dir,
                     datasets=set(args.dataset) if args.dataset else None, local=args.local,
                     status_only=args.status, force=args.force)
        for row in report["entries"]:
            print(f"{row['status']:10s} {row['dataset']:14s} doi={row['doi']} {row['detail']}")
        if report["missing_datasets"]:
            print("Unregistered datasets: " + ", ".join(report["missing_datasets"]))
        if args.output:
            write_json(args.output, report, replace=True)
        return 2 if report["failed"] else (0 if args.status or report["public_verification"]["complete"] else 1)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
