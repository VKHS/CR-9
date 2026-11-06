#!/usr/bin/env python3
"""Inspect and deposit verified archives or a clean Git snapshot on Zenodo.

``plan ARCHIVE`` is offline. ``archive ARCHIVE --license LICENSE`` creates or
resumes a draft; repeating it with ``--publish`` publishes that same draft.
Production uses ZENODO_TOKEN; --sandbox uses ZENODO_SANDBOX_TOKEN. A reserved
DOI is not public availability. Receipts bind the service, metadata and bytes.
API reference: https://developers.zenodo.org/ (legacy deposition API).
No archive code is executed or unpickled. Standard library only.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit
from urllib.request import Request, HTTPRedirectHandler, build_opener
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cm_campaigns.archives import verify_zip, MAX_JSON  # noqa: E402
from cm_campaigns.common import loads, relative, valid_hash, write_json  # noqa: E402
from scripts.archive_run import STAGES, MANIFEST_NAME, verify_archive  # noqa: E402

PRODUCTION = "https://zenodo.org/api"
SANDBOX = "https://sandbox.zenodo.org/api"
DATASETS = ("ogbn-arxiv", "ogbn-proteins", "ogbn-products", "Roman-empire", "PascalVOC-SP", "COCO-SP", "Peptides-func")
CHUNK = 4 * 1024 * 1024
RECEIPT_SCHEMA = "cross-morse-zenodo-receipt-1"


class DepositError(ValueError):
    """Local evidence or a service response cannot be used safely."""


def require(condition, message):
    if not condition:
        raise DepositError(message)


def file_identity(path: Path) -> dict:
    require(path.is_file() and not path.is_symlink(), f"Not a regular local file: {path}")
    hashes = {name: hashlib.new(name) for name in ("sha256", "md5")}
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            size += len(chunk)
            for digest in hashes.values():
                digest.update(chunk)
    return {"filename": path.name, "bytes": size, **{k: v.hexdigest() for k, v in hashes.items()}}


def sha256_file(path: Path) -> str:
    return file_identity(path)["sha256"]


def git_commit(root: Path = ROOT) -> str | None:
    out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
    return out.stdout.strip() if out.returncode == 0 else None


def verified_run_manifest(path: Path, *, allowed_profiles=("smoke", "full")) -> dict:
    """Check safe TAR membership, byte inventory and eight-stage links without
    extracting checkpoints. Integrity does not establish manuscript agreement.
    """
    with tarfile.open(path, "r:*") as tar:
        members, total = {}, 0
        for member in tar:
            name = member.name.rstrip("/") if member.isdir() else member.name
            relative(name)
            require(name not in members, f"Duplicate archive path: {name}")
            require(member.isfile() or member.isdir(), f"Unsupported archive member: {name}")
            total += member.size
            require(len(members) < 2000000 and total <= 200 * 1024 ** 3, "Archive exceeds inspection bounds")
            members[name] = member
        tops = {name.split("/", 1)[0] for name in members}
        require(len(tops) == 1, "Archive needs one top-level directory")
        top = tops.pop()

        def read(rel):
            member = members.get(f"{top}/{rel}")
            require(member is not None and member.isfile() and member.size <= MAX_JSON, f"Missing/oversized JSON: {rel}")
            return loads(tar.extractfile(member).read(MAX_JSON + 1).decode("utf-8"))

        manifest = read(MANIFEST_NAME)
        rows = manifest.get("files", [])
        require(isinstance(rows, list) and rows, "Missing archive byte inventory")
        inventory = {}
        for row in rows:
            require(isinstance(row, dict) and set(row) == {"path", "bytes", "sha256"}, "Invalid archive inventory row")
            relative(row["path"])
            require(row["path"] not in inventory and row["path"] != MANIFEST_NAME, "Duplicate inventory path")
            require(valid_hash(row["sha256"]) and type(row["bytes"]) is int and row["bytes"] >= 0, "Invalid archive identity")
            inventory[row["path"]] = row
        require(manifest.get("dataset") in DATASETS, "Unknown archive dataset")
        require(manifest.get("profile") in allowed_profiles and type(manifest.get("scientific_run")) is bool,
                "Invalid archive profile or scientific-run flag")
        kind = manifest.get("kind")
        require(kind in ("run", "campaign"), "Unsupported run archive kind")
        prefixes = [""]
        if kind == "campaign":
            require(manifest["dataset"] == "Roman-empire", "Only Roman-empire uses campaign archives")
            require(set(manifest.get("splits", {})) == {f"split_{i:02d}" for i in range(10)}, "Roman campaign needs all ten splits")
            prefixes = [f"split_{i:02d}/" for i in range(10)]
        sources = set()
        for prefix in prefixes:
            declared = manifest if not prefix else manifest["splits"][prefix[:-1]]
            state = read(prefix + "run_state.json")
            require(state.get("status") == "COMPLETE" and state.get("last_complete_stage") == "evaluate", "Run is incomplete")
            require(state.get("task", {}).get("dataset") == manifest["dataset"], "Run/descriptor dataset mismatch")
            for field in ("profile", "scientific_run"):
                require(state.get(field) == declared.get(field) == manifest.get(field), f"Run/descriptor {field} mismatch")
            sealed, identity, parent = {}, None, None
            for i, stage in enumerate(STAGES):
                rel = f"provenance/stages/{i:02d}_{stage}.json"
                value = read(prefix + rel)
                current = tuple(value.get(k) for k in ("source_sha256", "config_sha256", "unit_sha256"))
                require(all(valid_hash(v) for v in current), f"Invalid stage identities: {stage}")
                require(identity is None or identity == current, f"Stage identity changes: {stage}")
                require(value.get("stage") == stage and value.get("stage_index") == i, f"Incorrect stage name/index: {stage}")
                identity = current
                require(value.get("parent_manifest") == parent, f"Broken parent link: {stage}")
                for row in value.get("inputs", []):
                    require(i == 0 or sealed.get(row["path"]) == row, f"Unsealed/changed stage input: {row['path']}")
                for row in value.get("outputs", []):
                    expected = dict(row, path=prefix + row["path"])
                    require(inventory.get(expected["path"]) == expected, f"Stage output absent/different in inventory: {row['path']}")
                    require(row["path"] not in sealed or sealed[row["path"]] == row, f"Stage replaced an earlier output: {row['path']}")
                    sealed[row["path"]] = row
                parent = dict(inventory.get(prefix + rel, {}), path=rel)
                require(set(parent) == {"path", "bytes", "sha256"}, f"Stage manifest is not sealed: {rel}")
            config = read(prefix + "config.json")
            config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
            require(config_hash == identity[1], "Configuration identity differs from stage chain")
            require(config.get("profile") == state["profile"] and config.get("scientific_run") == state["scientific_run"],
                    "Configuration/run scope mismatch")
            require(inventory.get(prefix + "prepared/splits.npz", {}).get("sha256") == identity[2], "Unit identity differs from stage chain")
            require(state.get("source_sha256") == identity[0] and state.get("config_sha256") == identity[1], "Run state identity differs from stage chain")
            chain = declared.get("chain", {})
            require(tuple(chain.get(k) for k in ("source_sha256", "config_sha256", "unit_sha256")) == identity, "Descriptor chain identity mismatch")
            if prefix:
                require(config.get("data", {}).get("split_index") == int(prefix[6:8]), "Roman split index/configuration mismatch")
            sources.add(identity[0])
        actual_sources = sorted(sources) if kind == "campaign" else next(iter(sources))
        require(manifest.get("source_sha256") == actual_sources, "Archive source identity mismatch")
    return verify_archive(path)  # Stream/hash every file using the existing verifier.


def describe_archive(path: Path) -> dict:
    require(path.is_file() and not path.is_symlink(), f"Archive does not exist or is a symlink: {path}")
    if path.name.endswith((".tar.gz", ".tgz", ".tar")):
        manifest = verified_run_manifest(path)
        return {"kind": manifest["kind"], "format": "run-archive", "dataset": manifest["dataset"],
                "profile": manifest["profile"], "scientific_run": manifest["scientific_run"],
                "files": len(manifest["files"]), "run_source_sha256": manifest["source_sha256"]}
    if path.suffix == ".zip":
        descriptor = verify_zip(path)
        return {"kind": descriptor["kind"], "format": "evidence-zip", "record_id": descriptor["record_id"],
                "datasets": descriptor["datasets"], "files": len(descriptor["files"]), "source": descriptor["source"]}
    raise DepositError("Deposit candidates are run archives (.tar.gz) or evidence ZIPs (.zip)")


def build_metadata(kind: str, *, archive: Path | None, info: dict, root: Path = ROOT,
                   code_doi: str | None = None, commit: str | None = None, license_id: str | None = None) -> dict:
    base = loads((root / ".zenodo.json").read_text(encoding="utf-8"))
    related = [{k: v for k, v in item.items() if k != "scheme"} for item in base.get("related_identifiers", [])]
    if code_doi:
        require(re.fullmatch(r"10\.(?:5281|5072)/zenodo\.[1-9][0-9]*", code_doi) is not None, "Use a Zenodo version DOI for --code-doi")
        related.append({"identifier": code_doi, "relation": "isSupplementTo", "resource_type": "software"})
    common = {"creators": base["creators"], "keywords": base.get("keywords", []), "access_right": "open",
              "publication_date": dt.date.today().isoformat(), "related_identifiers": related, "prereserve_doi": True}
    if kind == "code":
        license_id = base.get("license")
        if isinstance(license_id, dict):
            license_id = license_id.get("id")
        title = base.get("title", "CROSS-MORSE reference implementation")
        description = f"Git source snapshot of CROSS-MORSE at commit {commit}. Official datasets and benchmark evidence are not supplied by a code snapshot."
    elif info["format"] == "run-archive":
        scope = "declared scientific run" if info["scientific_run"] else "software validation, not benchmark evidence"
        label = "ten-split campaign" if info["kind"] == "campaign" else "eight-stage run"
        title = f"CROSS-MORSE {label}: {info['dataset']} ({info['profile']} profile)"
        description = (f"Verified {label} archive; {scope}. Dataset {info['dataset']}; profile {info['profile']}; "
                       f"{info['files']} files; runtime source identity {info['run_source_sha256']}. "
                       "Byte and stage-chain integrity do not establish agreement with manuscript results.")
    else:
        title = f"CROSS-MORSE evidence archive: {info['record_id']}"
        description = (f"Sealed {info['kind']} archive {info['record_id']} covering {', '.join(info['datasets'])}. "
                       f"{info['files']} files. Descriptor and file identities verified; scientific claims require the repository evidence audit.")
    if archive is not None:
        identity = file_identity(archive)
        description += f" File {archive.name}; SHA-256 {identity['sha256']}; {identity['bytes']} bytes."
        if kind != "code":
            common["version"] = "sha256-" + identity["sha256"]
    if kind == "code" and commit:
        common["version"] = commit
    if license_id:
        require(isinstance(license_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+_-]*", license_id), "Supply a Zenodo license identifier")
        common["license"] = license_id.lower()
    return {**common, "upload_type": "software" if kind == "code" else "dataset", "title": title, "description": description}


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise DepositError("Authenticated Zenodo requests do not follow redirects")


class Transport:
    """Same-origin authenticated API requests; injectable opener for tests."""
    def __init__(self, base: str, token: str, *, opener=None, timeout: int = 300):
        self.base = base.rstrip("/")
        require(self.base in (PRODUCTION, SANDBOX), "Unsupported Zenodo service")
        require(isinstance(token, str) and token and not any(c.isspace() for c in token), "Missing or invalid Zenodo token")
        self.token, self.timeout = token, timeout
        self.opener = opener or build_opener(_NoRedirect()).open

    def _url(self, url):
        parsed = urlsplit(url)
        require(parsed.scheme == "https" and parsed.netloc == urlsplit(self.base).netloc
                and parsed.path.startswith("/api/") and not parsed.query and not parsed.fragment,
                "Refusing authenticated URL outside the selected Zenodo API")
        return url

    def _call(self, method, url, *, body=None, stream=None, length=None, authenticated=True):
        self._url(url)
        headers = {"Accept": "application/json", "User-Agent": "CROSS-MORSE-deposit/1"}
        if authenticated:
            headers["Authorization"] = f"Bearer {self.token}"
        if body is not None:
            headers["Content-Type"] = "application/json"
        if stream is not None:
            headers.update({"Content-Type": "application/octet-stream", "Content-Length": str(length)})
        request = Request(url, data=stream if stream is not None else body, headers=headers, method=method)
        try:
            with self.opener(request, timeout=self.timeout) as response:
                self._url(response.geturl())
                require(response.status in (200, 201, 202), f"Zenodo returned HTTP {response.status}")
                payload = response.read(MAX_JSON + 1)
        except HTTPError as exc:
            raise DepositError(f"Zenodo {method} failed with HTTP {exc.code}") from None
        require(len(payload) <= MAX_JSON, "Zenodo JSON response exceeds limit")
        result = loads(payload.decode("utf-8")) if payload else {}
        require(isinstance(result, dict), "Zenodo response must be an object")
        return result

    def json(self, method, path, obj=None):
        url = path if "://" in path else f"{self.base}/{path.lstrip('/')}"
        return self._call(method, url, body=json.dumps(obj).encode("utf-8") if obj is not None else None)

    def public_json(self, path):
        return self._call("GET", f"{self.base}/{path.lstrip('/')}", authenticated=False)

    def upload(self, bucket, path):
        self._url(bucket)
        require(re.fullmatch(r"/api/files/[A-Za-z0-9-]+/?", urlsplit(bucket).path), "Invalid bucket path")
        with path.open("rb") as stream:
            return self._call("PUT", f"{bucket.rstrip('/')}/{quote(path.name, safe='')}", stream=stream, length=path.stat().st_size)


def receipt_path(path: Path, service: str) -> Path:
    return path.with_name(path.name + (".sandbox.deposit.json" if service == SANDBOX else ".deposit.json"))


def _save(path, record):
    require(not path.is_symlink() and not any(p.is_symlink() for p in path.parents), "Receipt path must not traverse symlinks")
    write_json(path, record, replace=True)


def _metadata_hash(metadata):
    return hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_record(path: Path, record: dict) -> Path:
    out = receipt_path(path, record["service"])
    _save(out, record)
    return out


@contextmanager
def _receipt_lock(path):
    require(not path.is_symlink() and not any(p.is_symlink() for p in path.parents), "Receipt path must not traverse symlinks")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + ".lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise DepositError(f"Receipt is locked: {lock}; check for another process before removing a stale lock") from None
    try:
        os.close(fd)
        yield
    finally:
        lock.unlink(missing_ok=True)


def _asset_matches(asset, identity):
    names = [asset[key] for key in ("key", "filename", "name") if key in asset]
    require(names and all(name == identity["filename"] for name in names), "Zenodo asset filename mismatch")
    sizes = []
    for key in ("size", "filesize"):
        if key not in asset:
            continue
        value = asset[key]
        # The legacy API quickstart serializes filesize as "27"; the modern
        # bucket API and legacy schema specify integer bytes. Accept both exact
        # representations only for that legacy field, never floats or booleans.
        if key == "filesize" and isinstance(value, str) and re.fullmatch(r"0|[1-9][0-9]*", value):
            value = int(value)
        require(type(value) is int and value == identity["bytes"], "Zenodo asset byte count mismatch")
        sizes.append(value)
    require(sizes, "Zenodo asset byte count missing")
    checksum = asset.get("checksum", "")
    if isinstance(checksum, str) and re.fullmatch(r"[a-f0-9]{32}", checksum):
        checksum = "md5:" + checksum
    require(checksum in ("md5:" + identity["md5"], "sha256:" + identity["sha256"]), "Zenodo asset checksum mismatch or missing")


def _files(response):
    files = response.get("files", [])
    if isinstance(files, dict):
        files = list(files.get("entries", {}).values())
    require(isinstance(files, list), "Invalid Zenodo files response")
    return files


def _related_identifiers(items):
    """Zenodo normalizes DOI resolver URLs; retain every other identity field."""
    normalized = []
    for item in items:
        row = {key: value for key, value in item.items() if key != "scheme"}
        identifier = row.get("identifier")
        if isinstance(identifier, str):
            candidate = identifier
            parsed = urlsplit(identifier)
            if (parsed.scheme in ("http", "https") and parsed.netloc.lower() in ("doi.org", "dx.doi.org")
                    and not parsed.query and not parsed.fragment):
                candidate = parsed.path.removeprefix("/")
            if re.fullmatch(r"10\.[0-9]{4,9}/[^\s?#]+", candidate):
                row["identifier"] = candidate.lower()
        normalized.append(row)
    return normalized


def _metadata_matches(actual, expected):
    require(isinstance(actual, dict), "Missing deposited metadata")
    for key, value in expected.items():
        if key == "prereserve_doi":
            continue
        observed = actual.get(key)
        if key == "license" and isinstance(observed, dict):
            observed = observed.get("id")
        if key == "related_identifiers" and isinstance(observed, list):
            observed = _related_identifiers(observed)
            value = _related_identifiers(value)
        require(observed == value, f"Zenodo metadata differs from receipt: {key}")


def _published_record(transport, deposited, record, identity):
    doi = deposited.get("doi") or deposited.get("metadata", {}).get("prereserve_doi", {}).get("doi")
    prefix = "10.5072" if transport.base == SANDBOX else "10.5281"
    require(isinstance(doi, str) and re.fullmatch(re.escape(prefix) + r"/zenodo\.[1-9][0-9]*", doi), "Published DOI/service mismatch")
    rid = doi.rsplit(".", 1)[1]
    public = transport.public_json(f"records/{rid}")
    require(str(public.get("id")) == rid and public.get("doi") == doi and public.get("is_published", True) is True
            and public.get("metadata", {}).get("publication_date"), "Published record identity/state mismatch")
    files = _files(public)
    require(len(files) == 1, "Published record has unexpected/missing files")
    _asset_matches(files[0], identity)
    record.update(status="published", published=True, doi=doi, reserved_doi=doi,
                  record_url=f"{transport.base.removesuffix('/api')}/records/{rid}")


def deposit_file(transport: Transport, path: Path, metadata: dict, *, publish: bool,
                 receipt: Path | None = None, deposition_id: int | None = None) -> dict:
    """Resume a bound deposition. An uncertain creation is never repeated."""
    identity = file_identity(path)
    require(identity["bytes"] > 0, "Cannot deposit an empty file")
    require(metadata.get("license"), "Archive license is required; select --license for material you may redistribute")
    out = receipt or receipt_path(path, transport.base)
    require(out.absolute() != path.absolute(), "Receipt cannot overwrite the archive")
    with _receipt_lock(out):
        if out.exists():
            record = loads(out.read_text(encoding="utf-8"))
            require(record.get("schema") == RECEIPT_SCHEMA, "Unrecognized receipt; do not overwrite it")
            require(record.get("service") == transport.base, "Receipt belongs to a different Zenodo service")
            require(all(record.get(k) == v for k, v in identity.items()), "Local file differs from the receipt")
            require(record.get("metadata_sha256") == _metadata_hash(record["metadata"]), "Receipt metadata hash mismatch")
            expected = dict(metadata, publication_date=record["metadata"]["publication_date"])
            require(expected == record["metadata"], "Metadata differs from the saved draft")
        else:
            record = {"schema": RECEIPT_SCHEMA, **identity, "service": transport.base, "metadata": metadata,
                      "metadata_sha256": _metadata_hash(metadata),
                      "status": "creation_pending", "deposition_id": None, "reserved_doi": None, "doi": None,
                      "published": False, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
            _save(out, record)  # Intent precedes the only non-idempotent request.
            if deposition_id is None:
                created = transport.json("POST", "deposit/depositions", {})
                require(type(created.get("id")) is int and created["id"] > 0, "Create did not return an ID; inspect account before retrying")
                record.update(deposition_id=created["id"], status="draft")
                _save(out, record)
        if deposition_id is not None:
            require(type(deposition_id) is int and deposition_id > 0, "Invalid recovery deposition ID")
            require(record["deposition_id"] in (None, deposition_id), "Recovery ID conflicts with the receipt")
            record["deposition_id"] = deposition_id
            _save(out, record)
        require(record["deposition_id"] is not None, "Previous create outcome is unknown; inspect Zenodo and resume with --deposition-id, never create another draft blindly")
        endpoint = f"deposit/depositions/{record['deposition_id']}"
        current = transport.json("GET", endpoint)
        require(current.get("id") == record["deposition_id"] and type(current.get("submitted")) is bool, "Deposition identity/state mismatch")
        if current["submitted"]:
            _metadata_matches(current.get("metadata"), record["metadata"])
            _published_record(transport, current, record, identity)
            _save(out, record)
            return record
        require(not record["published"], "Previously published deposition now reports a draft")
        if record["status"] == "publishing":
            # HTTP 202 or a lost publish response is not proof of completion.
            # Query the same draft on rerun; do not replay a non-idempotent action.
            return record
        if current.get("metadata", {}).get("title"):
            _metadata_matches(current["metadata"], record["metadata"])
        files = _files(current)
        require(len(files) <= 1, "Draft has unexpected extra files")
        if files:
            _asset_matches(files[0], identity)
        else:
            _asset_matches(transport.upload(current.get("links", {}).get("bucket", ""), path), identity)
        require(file_identity(path) == identity, "Local file changed during upload")
        current = transport.json("PUT", endpoint, {"metadata": record["metadata"]})
        require(current.get("id") == record["deposition_id"] and current.get("submitted") is False, "Metadata response identity/state mismatch")
        _metadata_matches(current.get("metadata"), record["metadata"])
        reserved = current.get("metadata", {}).get("prereserve_doi", {}).get("doi")
        prefix = "10.5072" if transport.base == SANDBOX else "10.5281"
        require(isinstance(reserved, str) and re.fullmatch(re.escape(prefix) + r"/zenodo\.[1-9][0-9]*", reserved), "Reserved DOI missing or belongs to another service")
        record.update(reserved_doi=reserved, status="draft")
        _save(out, record)
        if publish:
            record["status"] = "publishing"
            _save(out, record)
            transport.json("POST", endpoint + "/actions/publish")
            current = transport.json("GET", endpoint)
            require(current.get("id") == record["deposition_id"], "Publication response ID mismatch")
            if current.get("submitted") is True:
                _published_record(transport, current, record, identity)
            _save(out, record)
        return record


def code_snapshot(root: Path, destination: Path) -> tuple[Path, str]:
    commit = git_commit(root)
    require(commit and re.fullmatch(r"[a-f0-9]{40,64}", commit), "Code deposit requires a Git checkout")
    status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True, timeout=30)
    require(status.returncode == 0 and not status.stdout.strip(), "Code deposit requires a clean committed checkout")
    require(not destination.resolve().is_relative_to(root.resolve()), "Store code deposits outside the checkout")
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"cross-morse-{commit[:12]}.zip"
    fd, name = tempfile.mkstemp(prefix=".cm-code-", dir=destination)
    temp = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            subprocess.run(["git", "-C", str(root), "archive", "--format=zip", f"--prefix=cross-morse-{commit[:12]}/", commit], stdout=stream, check=True, timeout=600)
        if target.exists():
            require(not target.is_symlink() and file_identity(target) == dict(file_identity(temp), filename=target.name), "Existing code snapshot has different bytes")
        else:
            os.link(temp, target)
    finally:
        temp.unlink(missing_ok=True)
    return target, commit


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "archive", "code"):
        item = sub.add_parser(command)
        if command != "code":
            item.add_argument("archive", type=Path)
            item.add_argument("--license", dest="license_id", help="Zenodo license ID covering the material; no data license is assumed")
            item.add_argument("--code-doi")
        else:
            item.add_argument("--output-dir", type=Path, default=ROOT.parent / "cm-deposits")
        if command != "plan":
            item.add_argument("--publish", action="store_true", help="publish the saved draft; irreversible public disclosure")
            item.add_argument("--sandbox", action="store_true")
            item.add_argument("--deposition-id", type=int, help="recover an uncertain create after inspecting the selected Zenodo account")
    args = parser.parse_args(argv)
    try:
        commit = git_commit(args.root)
        if args.command == "code":
            archive, commit = code_snapshot(args.root, args.output_dir)
            info = {}
        else:
            archive, info = args.archive, describe_archive(args.archive)
        metadata = build_metadata(args.command, archive=archive, info=info, root=args.root, commit=commit,
                                  code_doi=getattr(args, "code_doi", None), license_id=getattr(args, "license_id", None))
        if args.command == "plan":
            print(json.dumps({"archive": str(archive), **file_identity(archive), "descriptor": info, "metadata": metadata,
                              "required_before_deposit": [] if metadata.get("license") else ["--license"]}, indent=1))
            return 0
        token_name = "ZENODO_SANDBOX_TOKEN" if args.sandbox else "ZENODO_TOKEN"
        token = os.environ.get(token_name, "")
        require(token, f"Set {token_name} with deposit:write and deposit:actions scopes")
        transport = Transport(SANDBOX if args.sandbox else PRODUCTION, token)
        record = deposit_file(transport, archive, metadata, publish=args.publish, deposition_id=args.deposition_id)
        print(json.dumps(record, indent=1, sort_keys=True))
        print(f"Receipt: {receipt_path(archive, transport.base)}", file=sys.stderr)
        if record["published"]:
            print(f"Verified {'sandbox test' if args.sandbox else 'public'} DOI: {record['doi']}", file=sys.stderr)
        else:
            print(f"{record['status']}: reserved DOI {record['reserved_doi']} is not verified public availability. Rerun the same command with --publish to resume.", file=sys.stderr)
        return 1 if record["status"] == "publishing" else 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError, tarfile.TarError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
