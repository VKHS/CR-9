#!/usr/bin/env python3
"""Identify manuscript PDFs and record a bounded contextual source review.

The default is read-only. A numeric token search reports occurrences on each
registered table group's pages. It cannot establish table-cell correspondence,
units, signs in context, rounding, the correctness of a derivation or a result's
experimental origin. Its percentages are search diagnostics, not percentages of
correctly transcribed results. Missing tokens need contextual review, not an
automatic change to the paper or the registry.

Use --review-template FILE to write a private review skeleton bound to the exact
PDFs, registry files and page map. Complete each group's review_status and note,
then use --register --review-notes FILE. Registration identifies the PDFs and
records that contextual review and its named reviewer; it neither copies PDFs
nor verifies measurements. The legacy ``author_review`` JSON key is retained
for compatibility and does not identify who performed the review.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

SOURCES = ROOT / "paper" / "sources.json"
REPORTED = ROOT / "paper" / "reported_results.json"
CONSTANTS = ROOT / "paper" / "registered_constants.json"
CHECK = ROOT / "paper" / "transcription_check.json"
CHECK_SCHEMA = "cross-morse-transcription-check-2"
REVIEW_SCHEMA = "cross-morse-manuscript-review-1"
NUMBER = re.compile(r"(?<![\w.,])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w]|,\d)")
SKIP_KEYS = {"note", "notes", "interpretation", "scope", "source", "label", "unit", "units", "method", "route"}
REGRESSION = 0.02
FLOOR = 0.90
SEARCH_SCOPE = ("Page-wide numeric token occurrence only. A match can belong to another table cell or quantity; "
                "a miss can reflect a unit conversion, rounding, scientific notation or a documented derivation. "
                "Neither the occurrence rate nor the contextual source review verifies experimental measurements.")


class SourceError(ValueError):
    """The PDFs or the contextual source review cannot be registered safely."""


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_digest(value) -> str:
    return _digest(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def identity(path: Path) -> dict:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise SourceError(f"not a regular file: {path}")
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        if not stream.read(5).startswith(b"%PDF-"):
            raise SourceError(f"not a PDF: {path}")
        stream.seek(0)
        for block in iter(lambda: stream.read(1 << 20), b""):
            size += len(block)
            digest.update(block)
    return {"sha256": digest.hexdigest(), "byte_size": size, "name": path.name}


class Extractor:
    """Poppler when available, otherwise pypdf; extraction failures are explicit."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.kind = "poppler" if shutil.which("pdftotext") and shutil.which("pdfinfo") else None
        self._reader = None
        if self.kind is None:
            try:
                import pypdf  # noqa: F401
                self.kind = "pypdf"
            except ImportError:
                pass

    def pages(self) -> int | None:
        if self.kind == "poppler":
            out = subprocess.run(["pdfinfo", str(self.path)], capture_output=True, text=True, timeout=120, check=True)
            match = re.search(r"^Pages:\s+(\d+)", out.stdout, re.M)
            if not match:
                raise SourceError(f"pdfinfo returned no page count for {self.path.name}")
            return int(match.group(1))
        if self.kind == "pypdf":
            return len(self._pypdf().pages)
        return None

    def text(self, page: int) -> str:
        if self.kind == "poppler":
            out = subprocess.run(["pdftotext", "-f", str(page), "-l", str(page), "-layout", str(self.path), "-"],
                                 capture_output=True, text=True, timeout=120, check=True)
            return out.stdout
        if self.kind == "pypdf":
            return self._pypdf().pages[page - 1].extract_text() or ""
        raise SourceError("no PDF text extractor (install poppler-utils or pypdf)")

    def _pypdf(self):
        if self._reader is None:
            import pypdf
            self._reader = pypdf.PdfReader(str(self.path))
        return self._reader


def numbers(text: str) -> set[float]:
    text = text.replace("−", "-").replace("–", "-").replace(" ", " ")
    found = set()
    for match in NUMBER.finditer(text):
        value = float(match.group(0).replace(",", ""))
        if math.isfinite(value):
            found.add(value)
    return found


def leaves(value, path: str = ""):
    if isinstance(value, dict):
        for key, item in value.items():
            if not key.startswith("_") and key not in SKIP_KEYS:
                yield from leaves(item, f"{path}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from leaves(item, f"{path}/{index}")
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        yield path, float(value)


def present(value: float, printed: set[float]) -> bool:
    # Opposite signs are different observations. Contextual sign conventions or
    # unit conversions belong in the author's per-group review, not this search.
    tolerance = 1e-9 * max(1.0, abs(value))
    return any(abs(t - value) <= tolerance for t in printed)


def label_tokens(label: str) -> list[str]:
    tokens, kind = [], "Table"
    for match in re.finditer(r"\b(Tables?|Figures?)\b|\bS(\d+)([a-z]?)\b|\b([12])\b", label):
        if match.group(1):
            kind = "Figure" if match.group(1).startswith("Figure") else "Table"
        elif match.group(2):
            tokens.append("S" + match.group(2) + ("" if kind == "Figure" else match.group(3)))
        elif kind == "Table" and re.search(r"\bTables? " + match.group(4) + r"\b", label):
            tokens.append(f"Table {match.group(4)}")
    return tokens


def spot_check(extractors: dict[str, Extractor], sources: dict, reported: dict) -> dict:
    cache: dict[tuple[str, int], str] = {}
    groups = {}
    for key, group in sources["table_groups"].items():
        if key not in reported:
            raise SourceError(f"page map group {key!r} is absent from reported_results.json")
        extractor = extractors[group["source"]]
        text = ""
        for page in group["pages"]:
            cache_key = (group["source"], page)
            if cache_key not in cache:
                cache[cache_key] = extractor.text(page)
            text += "\n" + cache[cache_key]
        printed = numbers(text)
        values = list(leaves(reported[key]))
        missing = [(p, v) for p, v in values if not present(v, printed)]
        tokens = label_tokens(group.get("label", ""))
        groups[key] = {"label": group.get("label"), "pages": group["pages"], "values": len(values),
                       "found": len(values) - len(missing),
                       "coverage": round((len(values) - len(missing)) / len(values), 4) if values else None,
                       "label_found": all(re.search(r"(?<![\w])" + re.escape(t) + r"(?![\w])", text) for t in tokens) if tokens else None,
                       "missing_examples": [{"pointer": p, "value": v} for p, v in missing[:8]]}
    total, found = sum(g["values"] for g in groups.values()), sum(g["found"] for g in groups.values())
    return {"method": "signed_page_token_occurrence", "verifies_transcription": False, "scope": SEARCH_SCOPE,
            "groups": groups, "values": total, "found": found, "coverage": round(found / total, 4) if total else None}


def low_groups(spot: dict) -> list[str]:
    """Advisory search flags only; a low occurrence rate is not a failed transcription."""
    flags = []
    for key, group in spot["groups"].items():
        if group.get("coverage") is not None and group["coverage"] < FLOOR:
            flags.append(f"{key}: numeric occurrence rate {group['coverage']:.1%}; inspect units, rounding and context")
        if group.get("label_found") is False:
            flags.append(f"{key}: label {group['label']!r} not found on pages {group['pages']}; inspect the page map")
    return flags


def regressions(current: dict, previous: dict | None) -> list[str]:
    if not previous:
        return []
    flags = []
    for key, now in current["groups"].items():
        before = (previous.get("groups") or {}).get(key)
        if before is None:
            continue
        if before.get("coverage") is not None and now.get("coverage") is not None and now["coverage"] < before["coverage"] - REGRESSION:
            flags.append(f"{key}: coverage (numeric occurrence) {before['coverage']:.1%} -> {now['coverage']:.1%}; inspect context")
        if before.get("label_found") is True and now.get("label_found") is False:
            flags.append(f"{key}: label {now['label']!r} no longer on pages {now['pages']}; inspect the page map")
    return flags


def _page_map_digest(sources: dict) -> str:
    return _json_digest({"table_groups": sources["table_groups"],
                         "pages": {role: d["pages"] for role, d in sources["documents"].items()}})


def registry_identity(sources: dict, *, reported: Path = None, constants: Path = None) -> dict:
    return {"registry_sha256": {"reported_results": _digest((reported or REPORTED).read_bytes()),
                                "registered_constants": _digest((constants or CONSTANTS).read_bytes())},
            "page_map_sha256": _page_map_digest(sources)}


def review_identity(result: dict) -> dict:
    return {"pdf_sha256": {role: d["sha256"] for role, d in result["documents"].items()},
            "registry_sha256": result["registry_sha256"], "page_map_sha256": result["page_map_sha256"]}


def review_template(result: dict) -> dict:
    groups = result["table_groups"]
    return {"schema": REVIEW_SCHEMA, **review_identity(result), "reviewed_by": None,
            "_instructions": ("Review every declared group against the exact PDFs, including cell/column meaning, "
                              "units, signs and disclosed derived values. Set review_status to reviewed and explain "
                              "the check or any equivalent units/rounding/derivations in note. Unresolved groups "
                              "must stay pending. Name the actual reviewer in reviewed_by, including AI assistance when used. "
                              "This is a contextual source review, not independent measurement verification."),
            "groups": {key: {"review_status": "pending", "note": ""} for key in groups}}


def validate_review(notes: dict, expected: dict, groups) -> dict:
    if not isinstance(notes, dict) or notes.get("schema") != REVIEW_SCHEMA:
        raise SourceError("a completed --review-notes file from --review-template is required")
    for key in ("pdf_sha256", "registry_sha256", "page_map_sha256"):
        if notes.get(key) != expected[key]:
            raise SourceError(f"review notes are stale: {key} changed; review the current inputs")
    if not isinstance(notes.get("reviewed_by"), str) or not notes["reviewed_by"].strip():
        raise SourceError("review notes need reviewed_by")
    reviewed = notes.get("groups")
    if not isinstance(reviewed, dict) or set(reviewed) != set(groups):
        raise SourceError("review notes must cover exactly the current page-map groups")
    for key, entry in reviewed.items():
        if (not isinstance(entry, dict) or entry.get("review_status") != "reviewed"
                or not isinstance(entry.get("note"), str) or not entry["note"].strip()):
            raise SourceError(f"{key}: a completed contextual review and note are required")
    return {"schema": REVIEW_SCHEMA, **expected, "reviewed_by": notes["reviewed_by"].strip(),
            "scope": "Contextual source review of registry interpretation and page provenance by the named reviewer; "
                     "not verification of measurements or an attestation of personal review by an author.",
            "groups": {key: {"review_status": "reviewed", "note": reviewed[key]["note"].strip()} for key in groups}}


def report(main_pdf: Path, supplementary_pdf: Path, *, spot: bool = True) -> dict:
    source_bytes, reported_bytes, constants_bytes = SOURCES.read_bytes(), REPORTED.read_bytes(), CONSTANTS.read_bytes()
    sources, reported = json.loads(source_bytes), json.loads(reported_bytes)
    pdfs = {"main": Path(main_pdf), "supplementary": Path(supplementary_pdf)}
    extractors = {role: Extractor(path) for role, path in pdfs.items()}
    documents = {}
    for role, path in pdfs.items():
        recorded, ident = sources["documents"][role], identity(path)
        pages = extractors[role].pages()
        documents[role] = {**ident, "pages": pages, "recorded_sha256": recorded["sha256"],
                           "recorded_pages": recorded["pages"], "registered": ident["sha256"] == recorded["sha256"]
                           and ident["byte_size"] == recorded["byte_size"],
                           "pages_match": None if pages is None else pages == recorded["pages"]}
    for key, group in sources["table_groups"].items():
        pages = documents[group["source"]]["pages"]
        if not group.get("pages") or any(type(p) is not int or p < 1 or (pages is not None and p > pages) for p in group["pages"]):
            raise SourceError(f"{key}: page mapping lies outside the supplied PDF")
    result = {"schema": CHECK_SCHEMA, "scope": SEARCH_SCOPE, "documents": documents,
              "sources_sha256": _digest(source_bytes),
              "registry_sha256": {"reported_results": _digest(reported_bytes), "registered_constants": _digest(constants_bytes)},
              "page_map_sha256": _page_map_digest(sources), "table_groups": list(sources["table_groups"]),
              "extractor": extractors["main"].kind or extractors["supplementary"].kind}
    if spot and all(e.kind for e in extractors.values()):
        result["spot_check"] = spot_check(extractors, sources, reported)
        previous = json.loads(CHECK.read_text(encoding="utf-8")) if CHECK.is_file() else None
        # Old searches ignored signs and counted fragments of comma-grouped numbers.
        comparable = previous and previous.get("schema") == CHECK_SCHEMA
        result["regressions"] = regressions(result["spot_check"], previous.get("spot_check") if comparable else None)
        result["below_floor"] = low_groups(result["spot_check"])
    if any(identity(path)["sha256"] != documents[role]["sha256"] for role, path in pdfs.items()):
        raise SourceError("a PDF changed during inspection; rerun with stable inputs")
    if (SOURCES.read_bytes(), REPORTED.read_bytes(), CONSTANTS.read_bytes()) != (source_bytes, reported_bytes, constants_bytes):
        raise SourceError("registry inputs changed during inspection; rerun with stable inputs")
    return result


@contextmanager
def _registration_lock():
    import fcntl
    descriptor = os.open(SOURCES.parent, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _atomic_write(path: Path, data: bytes) -> None:
    if path.is_symlink():
        raise SourceError(f"refusing to replace a symlink: {path}")
    fd, name = tempfile.mkstemp(prefix=".source-register-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def register(result: dict, *, pdf_paths: dict[str, Path], review_notes: dict,
             accept_changes: bool = False, skip_spot_check: bool = False) -> None:
    if accept_changes:
        raise SourceError("--accept-changes is not a review record; complete --review-notes instead")
    with _registration_lock():
        fresh = report(pdf_paths["main"], pdf_paths["supplementary"], spot=not skip_spot_check)
        if review_identity(fresh) != review_identity(result) or fresh["sources_sha256"] != result["sources_sha256"]:
            raise SourceError("inputs changed since the report; inspect and review the current inputs")
        sources = json.loads(SOURCES.read_text(encoding="utf-8"))
        for role, document in fresh["documents"].items():
            if document["pages"] is None:
                raise SourceError(f"{role}: page count unavailable; install poppler-utils or pypdf")
            if document["pages"] != sources["documents"][role]["pages"]:
                raise SourceError(f"{role}: page count changed; update and review table page mappings before registration")
        reviewed = validate_review(review_notes, review_identity(fresh), fresh["table_groups"])
        stamp = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
        for role, document in fresh["documents"].items():
            sources["documents"][role].update(
                sha256=document["sha256"], byte_size=document["byte_size"], original_attachment_name=document["name"],
                identity_provenance=(f"Source identity registered on {stamp} with a byte-bound contextual source review in "
                                     "paper/transcription_check.json. Numeric token searches are advisory; "
                                     "PDF bytes are verified only when a matching local copy is supplied."))
        record = {"schema": CHECK_SCHEMA, "registered_utc": stamp, **review_identity(fresh),
                  "scope": SEARCH_SCOPE, "extractor": fresh.get("extractor"), "spot_check": fresh.get("spot_check"),
                  "author_review": reviewed}
        # Publish the check first: an interrupted pair cannot make an old source
        # identity appear reviewed. Restore the previous check if the second write fails.
        if CHECK.is_symlink():
            raise SourceError("transcription_check.json must not be a symlink")
        old_check = CHECK.read_bytes() if CHECK.exists() else None
        _atomic_write(CHECK, (json.dumps(record, indent=1, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
        try:
            _atomic_write(SOURCES, (json.dumps(sources, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
        except BaseException:
            if old_check is None:
                CHECK.unlink(missing_ok=True)
            else:
                _atomic_write(CHECK, old_check)
            raise


def check_review_record(root: Path = ROOT) -> dict:
    """Check source-review identity and coverage, not the truth of its attestations."""
    root = Path(root)
    stored = json.loads((root / "paper/transcription_check.json").read_text(encoding="utf-8"))
    if stored.get("schema") != CHECK_SCHEMA:
        raise SourceError("the saved record predates the contextual, byte-bound source review")
    sources = json.loads((root / "paper/sources.json").read_text(encoding="utf-8"))
    expected = {"pdf_sha256": {role: d["sha256"] for role, d in sources["documents"].items()},
                **registry_identity(sources, reported=root / "paper/reported_results.json", constants=root / "paper/registered_constants.json")}
    if any(stored.get(k) != value for k, value in expected.items()):
        raise SourceError("the saved contextual source review is stale for the PDF identities, registry or page map")
    return validate_review(stored.get("author_review"), expected, sources["table_groups"])


def markdown(result: dict) -> str:
    lines = ["# Manuscript source identity and numeric search", "", SEARCH_SCOPE, ""]
    for role, d in result["documents"].items():
        lines.append(f"- {role}: `{d['name']}` sha256 `{d['sha256'][:16]}…`, {d['byte_size']} bytes, {d['pages']} pages; "
                     f"{'registered identity' if d['registered'] else 'different from the registered identity'}"
                     + ("" if d["pages_match"] in (None, True) else f"; recorded page count is {d['recorded_pages']}"))
    spot = result.get("spot_check")
    if spot:
        rate = "n/a" if spot["coverage"] is None else f"{spot['coverage']:.1%}"
        lines += ["", f"Numeric search ({result.get('extractor')}): **{spot['found']}/{spot['values']}** values have a matching "
                  f"signed numeric token somewhere on the declared pages ({rate}). This is not a transcription accuracy rate.", "",
                  "| Group | Label | Pages | Numeric occurrence | Label found |", "|---|---|---|---|---|"]
        for key, g in spot["groups"].items():
            cov = "—" if g["coverage"] is None else f"{g['coverage']:.1%}"
            lines.append(f"| `{key}` | {g['label']} | {', '.join(map(str, g['pages']))} | {cov} | {g['label_found']} |")
        if result.get("regressions"):
            lines += ["", "Search changes since the previous comparable search:", ""] + [f"- {r}" for r in result["regressions"]]
        partial = [(key, g) for key, g in spot["groups"].items() if g.get("missing_examples")]
        if partial:
            lines += ["", "Values without matching tokens (examples; inspect units, rounding, derived quantities and context):", ""]
            for key, g in partial:
                lines.append(f"- `{key}`: " + "; ".join(f"`{m['pointer']}` = {m['value']}" for m in g["missing_examples"]))
    else:
        lines += ["", "Numeric search not run. This supplies no automatic check of numerical correspondence."]
    return "\n".join(lines) + "\n"


def _output_path(path: Path, inputs: list[Path], *, new: bool = False) -> Path:
    output = path.resolve()
    if output.is_relative_to(ROOT) or output.suffix != ".json":
        raise SourceError("write reports and review notes to a .json file outside the checkout")
    if output in inputs or output.with_suffix(".md") in inputs:
        raise SourceError("output would overwrite an input file")
    if new and output.exists():
        raise SourceError("review-template output already exists; preserve completed review notes")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manuscript-pdf", type=Path, required=True)
    parser.add_argument("--supplementary-pdf", type=Path, required=True)
    parser.add_argument("--register", action="store_true", help="register source identities and their completed contextual review")
    parser.add_argument("--review-template", type=Path, help="new private .json skeleton to complete before registration")
    parser.add_argument("--review-notes", type=Path, help="completed per-group contextual review, bound to these inputs")
    parser.add_argument("--accept-changes", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-spot-check", action="store_true", help="omit the advisory numeric search; page counts and contextual review remain required")
    parser.add_argument("--output", type=Path, help="JSON report outside the checkout (a .md is written beside it)")
    args = parser.parse_args(argv)
    try:
        if args.accept_changes:
            raise SourceError("--accept-changes is not a contextual review; use --review-template and --review-notes")
        if args.register and args.review_template:
            raise SourceError("write and complete the review template before using --register")
        paths = {"main": args.manuscript_pdf.resolve(), "supplementary": args.supplementary_pdf.resolve()}
        for pdf in paths.values():
            if pdf.is_relative_to(ROOT):
                raise SourceError("keep manuscript PDFs outside the repository; they are identified, never distributed")
        result = report(paths["main"], paths["supplementary"], spot=not args.skip_spot_check)
        inputs = [*paths.values(), *([args.review_notes.resolve()] if args.review_notes else [])]
        if args.output:
            output = _output_path(args.output, inputs)
            output.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n", encoding="utf-8")
            output.with_suffix(".md").write_text(markdown(result), encoding="utf-8")
        if args.review_template:
            output = _output_path(args.review_template, inputs + ([args.output.resolve()] if args.output else []), new=True)
            output.write_text(json.dumps(review_template(result), indent=1) + "\n", encoding="utf-8")
            print(f"wrote private review template: {output}")
        print(markdown(result))
        if args.register:
            if not args.review_notes:
                raise SourceError("--register requires completed --review-notes; numeric matches alone are insufficient")
            notes = json.loads(args.review_notes.read_text(encoding="utf-8"))
            register(result, pdf_paths=paths, review_notes=notes, skip_spot_check=args.skip_spot_check)
            print("registered source identities and contextual review; run make registry-check and make manifest, then commit")
            return 0
        return 0 if all(d["registered"] and d["pages_match"] is True for d in result["documents"].values()) else 1
    except (SourceError, OSError, ValueError, KeyError, TypeError, IndexError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
