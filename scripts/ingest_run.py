#!/usr/bin/env python3
"""Verify a deposited run archive and register it under ``runs/`` for the paper build.

    python -B scripts/ingest_run.py ../deposits/arxiv_full.tar.gz --doi 10.5281/zenodo.XXXXXXX
    python -B scripts/ingest_run.py --check            # re-verify every registered run
    python -B scripts/ingest_run.py --check --allow-deposited-absent   # metadata-only check for absent payloads
    python -B scripts/ingest_run.py --list

The archive is verified (``scripts/archive_run.py --verify``), extracted into
``runs/<stem>/`` (``runs/roman_empire/`` for a campaign) and recorded in ``runs/index.json``
with: dataset, profile, whether the run is an official-data scientific run, the archive
SHA-256 and byte size, the deposit DOI, each run's source/configuration/unit identities and the
source identity of this checkout. ``make tables`` and ``make figures`` read ``runs/index.json``;
``scripts/availability.py`` derives the availability statement from it.  Only archives whose
stage chain verifies are registered; an index entry is never written by hand. Offline
descriptor checks defer public DOI and payload verification to the fetch/review tools.

Standard library only.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import fcntl
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.archive_run import ArchiveError, MANIFEST_NAME, sha256_file, verify_stage_chain  # noqa: E402
from scripts._paper_common import runtime_source_fingerprint  # noqa: E402
from scripts.deposit_zenodo import verified_run_manifest  # noqa: E402
from cm_campaigns.archives import directory_inventory  # noqa: E402

STEMS = {"ogbn-arxiv": "arxiv", "ogbn-products": "products", "ogbn-proteins": "proteins", "Roman-empire": "roman_empire",
         "PascalVOC-SP": "pascalvoc_sp", "COCO-SP": "coco_sp", "Peptides-func": "peptides_func"}
INDEX = ROOT / "runs" / "index.json"
SCHEMA = "cross-morse-runs-index-1"
DEPOSIT_DOI = re.compile(r"10\.5281/zenodo\.[1-9][0-9]*")
HASH = re.compile(r"[0-9a-f]{64}")
IDENTITY_KEYS = ("source_sha256", "config_sha256", "unit_sha256")
ROMAN_SPLITS = tuple(f"split_{i:02d}" for i in range(10))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ArchiveError(message)


def registered_path(entry: dict, runs_root: Path) -> Path:
    """Canonical registration paths cannot escape the index or hide dangling links."""
    require(isinstance(entry, dict), "registered run entry must be an object")
    dataset = entry.get("dataset")
    require(dataset in STEMS, f"unknown registered dataset: {dataset!r}")
    require(entry.get("stem") == entry.get("path") == STEMS[dataset],
            "registered path/stem must be the canonical dataset directory")
    path = runs_root / entry["path"]
    require(not path.is_symlink() and not any(p.is_symlink() for p in path.parents),
            f"registered path traverses a symlink: {path}")
    return path


def _identities(value: dict) -> dict:
    require(isinstance(value, dict) and all(isinstance(value.get(k), str) and HASH.fullmatch(value[k])
                                          for k in IDENTITY_KEYS), "invalid source/configuration/unit identities")
    return {key: value[key] for key in IDENTITY_KEYS}


def registration_descriptor(entry: dict) -> dict:
    """Describe a registration, not a fresh verification of a public deposit."""
    require(isinstance(entry, dict) and entry.get("dataset") in STEMS
            and entry.get("stem") == entry.get("path") == STEMS[entry["dataset"]],
            "registration needs a canonical dataset path/stem")
    require(entry.get("verified") is True, "registration is not marked verified")
    require(entry.get("kind") in ("run", "campaign")
            and (entry["kind"] != "campaign" or entry["dataset"] == "Roman-empire"), "invalid registered archive kind")
    require(entry.get("profile") in ("smoke", "pilot", "full") and type(entry.get("scientific_run")) is bool,
            "invalid registered profile/scientific-run flag")
    require(not (entry["dataset"] == "Roman-empire" and entry["profile"] == "full" and entry["scientific_run"] is True)
            or entry["kind"] == "campaign", "full scientific Roman registration requires a campaign with all ten splits")
    name = entry.get("archive_name")
    require(isinstance(name, str) and name not in ("", ".", "..") and "/" not in name and "\\" not in name
            and not any(ord(c) < 32 for c in name), "invalid registered archive filename")
    require(isinstance(entry.get("archive_sha256"), str) and HASH.fullmatch(entry["archive_sha256"])
            and type(entry.get("archive_bytes")) is int and entry["archive_bytes"] > 0,
            "registration needs an archive SHA-256 and positive byte count")
    require(type(entry.get("files")) is int and entry["files"] > 0, "registration needs a positive archive file count")
    identities = entry.get("run_identities")
    if entry["kind"] == "campaign":
        require(isinstance(identities, dict) and set(identities) == set(ROMAN_SPLITS),
                "registered Roman campaign needs all ten split identities")
        identities = {name: _identities(identities[name]) for name in ROMAN_SPLITS}
        sources = sorted({value["source_sha256"] for value in identities.values()})
    else:
        identities = _identities(identities)
        sources = [identities["source_sha256"]]
    require(entry.get("run_source_sha256") == sources, "registered run source identities differ from the sealed identities")
    require(entry.get("deposit_doi") is None or isinstance(entry["deposit_doi"], str), "invalid deposit DOI value")
    return {"schema": "cross-morse-run-registration-1",
            **{k: entry[k] for k in ("stem", "dataset", "path", "kind", "profile", "scientific_run", "archive_name",
                                     "archive_sha256", "archive_bytes", "files", "deposit_doi", "run_source_sha256")},
            "run_identities": identities}


def deposited_absent(entry: dict, runs_root: Path) -> bool:
    path = registered_path(entry, runs_root)
    return (entry.get("verified") is True and isinstance(entry.get("deposit_doi"), str)
            and DEPOSIT_DOI.fullmatch(entry["deposit_doi"]) is not None and not path.exists())


def check_registration_manifest(entry: dict, manifest: dict, out_dir: Path) -> None:
    """Check committed registration metadata and copied split bytes; no network assertion."""
    descriptor = registration_descriptor(entry)
    require(isinstance(manifest, dict) and manifest.get("schema") == "cross-morse-run-manifest-1",
            "unexpected committed run-manifest schema")
    require(manifest.get("registration") == descriptor,
            "committed manifest registration differs from the index (archive bytes, DOI or run identities)")
    for key in ("dataset", "profile", "scientific_run"):
        require(manifest.get(key) == entry[key], f"committed manifest {key} differs from the index")
    require(all(isinstance(manifest.get(key), dict) for key in
                ("seeds", "splits", "source_data", "policy_lock", "environment")), "incomplete committed run manifest")
    parts = [(entry["stem"], manifest)]
    if entry["kind"] == "campaign":
        campaign = manifest.get("campaign")
        require(isinstance(campaign, dict) and campaign.get("splits") == list(range(10)),
                "committed Roman campaign requires splits 0 through 9")
        rows = campaign.get("per_split")
        require(isinstance(rows, list) and [r.get("run_name") for r in rows if isinstance(r, dict)] == list(ROMAN_SPLITS),
                "committed Roman campaign requires all ten ordered split records")
        parts = []
        for i, row in enumerate(rows):
            name = ROMAN_SPLITS[i]
            require(row.get("split_index") == i and _identities(row.get("identities")) == entry["run_identities"][name],
                    f"committed Roman {name} identities differ from the index")
            require(row.get("unit_sha256") == row["identities"]["unit_sha256"], f"committed {name} unit identity mismatch")
            parts.append((f"{entry['stem']}_{name}", row))
        require(_identities(manifest.get("identities")) == entry["run_identities"][ROMAN_SPLITS[0]],
                "committed campaign first-split identities differ from the index")
    else:
        require("campaign" not in manifest, "committed run unexpectedly has campaign metadata")
        require(_identities(manifest.get("identities")) == entry["run_identities"], "committed run identities differ from the index")
    for stem, part in parts:
        arrays = part.get("arrays") if entry["kind"] == "campaign" else part["splits"].get("arrays")
        require(isinstance(arrays, dict) and "splits.npz" in arrays, f"{stem}: split array identity is missing")
        expected_files = {}
        for name, identity in arrays.items():
            require(name in ("splits.npz", "official_splits.npz", "clusters.npz", "graph_splits.npz")
                    and isinstance(identity, dict) and isinstance(identity.get("sha256"), str)
                    and HASH.fullmatch(identity["sha256"]) and type(identity.get("bytes")) is int and identity["bytes"] > 0,
                    f"{stem}: invalid split-array descriptor")
            expected_files[f"{stem}_{name}"] = identity
        require(arrays["splits.npz"]["sha256"] == part["identities"]["unit_sha256"], f"{stem}: split identity mismatch")
        copied = part.get("index_arrays_copied", [])
        require(isinstance(copied, list) and len(copied) == len(set(copied))
                and all(name in expected_files for name in copied), f"{stem}: invalid copied split-array list")
        for name in copied:
            path = out_dir / name
            require(path.is_file() and not path.is_symlink(), f"missing committed split array: {path}")
            identity = expected_files[name]
            require(path.stat().st_size == identity["bytes"] and sha256_file(path) == identity["sha256"],
                    f"committed split array differs from its recorded bytes: {path}")


def check_absent_manifest(entry: dict, manifests_dir: Path) -> None:
    require(DEPOSIT_DOI.fullmatch(str(entry.get("deposit_doi") or "")), "absent payload needs a production Zenodo DOI registration")
    path = manifests_dir / f"{entry['stem']}.json"
    require(path.is_file() and not path.is_symlink() and not any(p.is_symlink() for p in path.parents),
            f"committed registration manifest is missing or symlinked: {path}")
    check_registration_manifest(entry, json.loads(path.read_text(encoding="utf-8")), manifests_dir)


def load_index(path: Path = INDEX) -> dict:
    if path.is_file():
        index = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(index, dict) or index.get("schema") != SCHEMA or not isinstance(index.get("runs"), list):
            raise ArchiveError(f"unexpected index schema in {path}")
        return index
    return {"schema": SCHEMA, "runs": []}


def save_index(index: dict, path: Path = INDEX) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    index["runs"].sort(key=lambda e: e["stem"])
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        pending = Path(handle.name)
        try:
            handle.write(json.dumps(index, indent=1, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            pending.unlink(missing_ok=True)
            raise
    try:
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


@contextmanager
def _index_lock(index_path: Path):
    index_path.parent.mkdir(parents=True, exist_ok=True)
    lock = index_path.with_name(f".{index_path.name}.lock")
    require(not lock.is_symlink(), f"index lock is a symlink: {lock}")
    with lock.open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _safe_extract(tar: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in tar.getmembers():
        target = (destination / member.name).resolve()
        if not target.is_relative_to(destination):
            raise ArchiveError(f"archive member escapes the destination: {member.name}")
        if not (member.isfile() or member.isdir()):
            raise ArchiveError(f"unsupported archive member type: {member.name}")
    tar.extractall(destination, filter="data")


def run_identities(path: Path, kind: str) -> dict:
    if kind == "campaign":
        require(path.is_dir() and {p.name for p in path.glob("split_*")} == set(ROMAN_SPLITS),
                f"registered Roman campaign needs all ten split directories: {path}")
        for name in ROMAN_SPLITS:
            require((path / name).is_dir() and not (path / name).is_symlink(), f"invalid campaign split: {name}")
            config = json.loads((path / name / "config.json").read_text(encoding="utf-8"))
            require((config.get("data") or {}).get("split_index") == int(name[6:]), f"campaign split/configuration mismatch: {name}")
        return {name: _identities(verify_stage_chain(path / name)) for name in ROMAN_SPLITS}
    require(kind == "run" and path.is_dir() and not path.is_symlink(), f"registered run directory is missing: {path}")
    return _identities(verify_stage_chain(path))


def check_present(entry: dict, path: Path) -> dict:
    """Recheck local sealed payload bytes, including complete campaign membership."""
    identities = run_identities(path, entry["kind"])
    sources = sorted({part["source_sha256"] for part in identities.values()}) if entry["kind"] == "campaign" else [identities["source_sha256"]]
    require(sources == entry.get("run_source_sha256"), "local run source identities differ from the registration")
    if "run_identities" in entry:
        require(identities == entry["run_identities"], "local source/configuration/unit identities differ from the registration")
    manifest = json.loads((path / MANIFEST_NAME).read_text(encoding="utf-8"))
    require(manifest.get("schema") == "cross-morse-run-archive-1", "missing or invalid extracted archive manifest")
    for key in ("dataset", "kind", "profile", "scientific_run"):
        require(manifest.get(key) == entry.get(key), f"extracted archive/index {key} mismatch")
    files = manifest.get("files")
    require(isinstance(files, list) and len(files) == entry.get("files"), "extracted archive/index file count mismatch")
    require(directory_inventory(path, exclude=(MANIFEST_NAME,)) == sorted(files, key=lambda row: row["path"]),
            "local run bytes differ from the archived inventory")
    return identities


def ingest(archive: Path, *, doi: str | None, runs_root: Path, index_path: Path, replace: bool) -> dict:
    # The shared deposit verifier checks safe membership, every byte and all stage links;
    # it performs no network requests and never loads checkpoints.
    manifest = verified_run_manifest(archive, allowed_profiles=("smoke", "pilot", "full"))
    dataset = manifest.get("dataset")
    if dataset not in STEMS:
        raise ArchiveError(f"archive dataset {dataset!r} is not one of the seven benchmarks")
    stem = STEMS[dataset]
    target = runs_root / stem
    require(not target.is_symlink() and not any(p.is_symlink() for p in target.parents), f"run destination is symlinked: {target}")
    require(not index_path.is_symlink(), f"index destination is symlinked: {index_path}")
    if target.exists() and not replace:
        raise ArchiveError(f"{target} already exists; pass --replace to re-ingest")
    runs_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".ingest_{stem}_", dir=runs_root) as temporary:
        staging = Path(temporary)
        with tarfile.open(archive, "r:*") as tar:
            _safe_extract(tar, staging / "payload")
        extracted = staging / "payload" / manifest["top_level_directory"]
        expected = {k: v for k, v in manifest.items() if k not in ("archive_sha256", "archive_bytes", "top_level_directory")}
        require(json.loads((extracted / MANIFEST_NAME).read_text(encoding="utf-8")) == expected,
                "archive manifest changed after verification")
        require(archive.stat().st_size == manifest["archive_bytes"] and sha256_file(archive) == manifest["archive_sha256"],
                "archive bytes changed after verification")
        identities = run_identities(extracted, manifest["kind"])
        run_sources = (sorted({c["source_sha256"] for c in identities.values()}) if manifest["kind"] == "campaign"
                       else [identities["source_sha256"]])
        entry = {"stem": stem, "dataset": dataset, "path": stem, "kind": manifest["kind"],
                 "profile": manifest.get("profile"), "scientific_run": manifest["scientific_run"],
                 "archive_name": archive.name, "archive_sha256": manifest["archive_sha256"],
                 "archive_bytes": manifest["archive_bytes"], "deposit_doi": doi,
                 "run_source_sha256": run_sources, "run_identities": identities,
                 "checkout_source_sha256": runtime_source_fingerprint(ROOT),
                 "files": len(manifest["files"]), "verified": True,
                 "ingested_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()}
        registration_descriptor(entry)
        check_present(entry, extracted)
        with _index_lock(index_path):
            index = load_index(index_path)
            require(not target.exists() or replace, f"{target} already exists; pass --replace to re-ingest")
            require(not target.exists() or (target.is_dir() and not target.is_symlink()), f"unsafe existing target: {target}")
            backup = staging / "previous"
            if target.exists():
                target.rename(backup)
            try:
                extracted.rename(target)
                index["runs"] = [e for e in index["runs"] if e["stem"] != stem] + [entry]
                save_index(index, index_path)
            except BaseException:
                if target.exists():
                    shutil.rmtree(target)
                if backup.exists():
                    backup.rename(target)
                raise
    return entry


def check(runs_root: Path, index_path: Path, *, allow_deposited_absent: bool = False,
          manifests_dir: Path | None = None) -> int:
    """Check local bytes, or explicitly defer absent payloads after matching committed metadata."""
    index = load_index(index_path)
    manifests_dir = manifests_dir or index_path.parent.parent / "manifests" / "official"
    problems, held = [], []
    seen = set()
    for entry in index["runs"]:
        try:
            path = registered_path(entry, runs_root)
            require(entry.get("verified") is True, "registered entry is not verified")
            require(entry["stem"] not in seen, f"duplicate registered dataset: {entry['dataset']}")
            seen.add(entry["stem"])
            if allow_deposited_absent and deposited_absent(entry, runs_root):
                check_absent_manifest(entry, manifests_dir)
                held.append(f"{entry['stem']} ({entry['deposit_doi']})")
                continue
            check_present(entry, path)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            problems.append(f"{entry.get('stem', '?') if isinstance(entry, dict) else '?'}: {exc}")
    for p in problems:
        print("CHECK FAILED:", p, file=sys.stderr)
    if problems:
        return 1
    print(f"{len(index['runs']) - len(held)} registered run(s) verify against their local archive inventory and sealed stage chains"
          + (f"; {len(held)} absent payload(s), committed registration metadata checked: {', '.join(held)}. "
             "Deposit availability and payload bytes are NOT VERIFIED here; make fetch-deposits retrieves and verifies them."
             if held else ""))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("archive", nargs="?", type=Path)
    parser.add_argument("--doi", help="deposit DOI of the archive (recorded in the index)")
    parser.add_argument("--runs-root", type=Path, help="payload directory (default: parent of --index)")
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--allow-deposited-absent", action="store_true",
                        help="with --check: check committed metadata for absent payloads; public availability remains unverified")
    parser.add_argument("--manifests-dir", type=Path, help="committed official manifests (default: sibling manifests/official of index parent)")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    if (args.allow_deposited_absent or args.manifests_dir) and not args.check:
        parser.error("--allow-deposited-absent and --manifests-dir require --check")
    args.runs_root = args.runs_root or args.index.parent
    try:
        if args.list:
            for entry in load_index(args.index)["runs"]:
                print(f"{entry['stem']:14s} {entry['dataset']:14s} profile={entry['profile']} scientific={entry['scientific_run']} "
                      f"doi={entry.get('deposit_doi')} archive={entry['archive_sha256'][:12]}")
            return 0
        if args.check:
            return check(args.runs_root, args.index, allow_deposited_absent=args.allow_deposited_absent,
                         manifests_dir=args.manifests_dir)
        if not args.archive:
            parser.error("an archive path is required")
        entry = ingest(args.archive, doi=args.doi, runs_root=args.runs_root, index_path=args.index, replace=args.replace)
        print(json.dumps(entry, indent=1))
        return 0
    except (ArchiveError, OSError, ValueError, TypeError, KeyError, tarfile.TarError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
