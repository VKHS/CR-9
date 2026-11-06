"""Locator-driven measurement record sets: every summary value is read from a deposited raw file.

Used by ``scripts/export_records.py mapping-template`` and ``scripts/export_records.py from-mapping``
for the record sets whose original outputs are not in an exporter's input format (factorials, negative
controls, input gate, cross-backbone, ensemble grid, inserted-site attribution, and baselines or E3-E6
outputs produced outside the released drivers).

A mapping file (schema ``cross-morse-record-mapping-1``) names

  raw_files   original output files (or directories) to copy unchanged into the set's raw/ or code/
  rows        one entry per summary row: the set's keys, the quantity, its unit, the deposited
              source_file and a locator (json:, csv: or regex:, as in scripts/records.py)
  provenance  experiment, produced_by, hardware, period and protocol (records/specs.json)

``for_each`` on a row or a raw_files entry expands placeholders such as ``{seed}`` or ``{cell}``
(cartesian product of the listed values).  A row may instead ``derive`` its value from located
inputs with op sum, mean, ratio (first/second) or difference (first minus second): the result and
the SHA-256 of every input are written to raw/derived/ and the summary row points at that file.

Summary values are read from the supplied files, with declared arithmetic recorded separately.
``scripts/records.py check`` re-reads values and recomputes mapped derivations from hashed inputs.
This establishes file-to-summary traceability; it does not establish that a supplied file is an
original measurement or that a declared unit is correct. The set is staged, checked for coverage
and private infrastructure identifiers, and installed with its index under a replacement lock.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MAPPING_SCHEMA = "cross-morse-record-mapping-1"
PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
OPS = ("sum", "mean", "ratio", "difference")
SEEDS = [str(seed) for seed in range(1, 13)]
FACTORIAL_CELLS = ["000", "100", "010", "001", "110", "101", "011", "111"]
INSERTED_SITE_ARMS = ["native, open loop", "inserted, open loop (q_C identity)", "native, feedback",
                      "inserted, feedback (deployed)"]
BUDGET_ARMS = ["E3 predictor", "E4 distilled", "E5 no-observation replanning", "E6 budget-aware q_B"]


class MappingError(ValueError):
    """A mapping cannot produce a complete, traceable record set."""


def _specs():
    from scripts.records import RECORDS, load_specs
    return load_specs(RECORDS, RECORDS / "specs.json")


def _contract(name: str) -> dict:
    from cm_campaigns.contract import RECORD_SETS
    if name not in RECORD_SETS:
        raise MappingError(f"{name} is not a registered record set")
    return RECORD_SETS[name]


def _unit(description: str):
    match = re.search(r"\(([^()]{1,24})\)\s*$", description or "")
    return match.group(1) if match else None


def _registry() -> dict:
    return json.loads((ROOT / "paper" / "reported_results.json").read_text(encoding="utf-8"))


def _row(keys: dict, quantity: str, spec: dict, **extra) -> dict:
    return {"keys": keys, "quantity": quantity, "unit": _unit(spec["quantities"][quantity]),
            "source_file": None, "locator": None, **extra}


def template(name: str) -> dict:
    """A complete row skeleton covering exactly what the audit requires for the set."""
    specs = _specs()
    if name not in specs["sets"]:
        raise MappingError(f"{name} is not a set of records/specs.json")
    spec, contract = specs["sets"][name], _contract(name)
    per_seed = set(spec.get("per_seed_quantities", []))
    rows = []
    if name == "baselines":
        ledger = _registry()["supplementary_table_S15b_per_rerun_resource_ledger"]
        for dataset in contract["datasets"]:
            for method in [row["method"] for row in ledger[dataset]]:
                for quantity in spec["quantities"]:
                    if quantity in per_seed:
                        rows.append(_row({"dataset": dataset, "method": method, "seed": "{seed}"}, quantity, spec,
                                         for_each={"seed": SEEDS}))
                    else:
                        rows.append(_row({"dataset": dataset, "method": method, "seed": ""}, quantity, spec))
    elif contract.get("coverage_keys"):
        for values in contract["coverage_rows"]:
            for quantity in spec["quantities"]:
                rows.append(_row(dict(zip(contract["coverage_keys"], values)), quantity, spec))
    else:
        extra_keys = [k for k in spec["keys"] if k != "dataset"]
        loops = {"cell": FACTORIAL_CELLS, "arm": INSERTED_SITE_ARMS if name == "inserted_site" else BUDGET_ARMS,
                 "ensemble": [str(e) for e in (1, 2, 4, 8, 12, 16, 20)], "configuration": ["selected"]}
        for dataset in contract["datasets"]:
            for quantity in spec["quantities"]:
                keys = {"dataset": dataset, **{k: "{" + k + "}" for k in extra_keys}}
                for_each = {k: loops[k] for k in extra_keys}
                rows.append(_row(keys, quantity, spec, **({"for_each": for_each} if for_each else {})))
    return {"schema": MAPPING_SCHEMA, "set": name,
            "_comment": ("Fill unit, source_file and locator of every row, list every original file under raw_files, "
                         "and complete provenance. Relative from paths resolve beside this mapping file. {placeholders} expand over for_each. A row may use "
                         "derive={op: sum|mean|ratio|difference, inputs: [{source_file, locator}, ...]} instead of "
                         "source_file/locator. Then: python -B scripts/export_records.py from-mapping --mapping THIS_FILE"),
            "_quantities": spec["quantities"], "_keys": spec["keys"], "_manuscript_tables": spec["manuscript_tables"],
            "provenance": {field: None for field in specs["provenance_fields"]},
            "raw_files": [{"from": None, "to": "raw/REPLACE/original_output.json"}],
            "rows": rows}


def _expand(entry: dict) -> list[dict]:
    if not isinstance(entry, dict):
        raise MappingError("mapping rows and raw_files entries must be objects")
    loops = entry.get("for_each") or {}
    if not isinstance(loops, dict) or any(not isinstance(v, list) or not v for v in loops.values()):
        raise MappingError("for_each must map placeholder names to nonempty lists")
    names = list(loops)
    expanded = []
    for values in itertools.product(*[loops[n] for n in names]) if names else [()]:
        binding = {n: str(v) for n, v in zip(names, values)}

        def fill(value):
            if isinstance(value, str):
                missing = [m for m in PLACEHOLDER.findall(value) if m not in binding]
                if missing:
                    raise MappingError(f"placeholder(s) {missing} have no for_each values: {value!r}")
                return PLACEHOLDER.sub(lambda m: binding[m.group(1)], value)
            if isinstance(value, dict):
                return {k: fill(v) for k, v in value.items() if k != "for_each"}
            if isinstance(value, list):
                return [fill(v) for v in value]
            return value
        expanded.append(fill(copy.deepcopy(entry)))
    return expanded


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(text: str) -> Path:
    if not isinstance(text, str) or not text or "\\" in text:
        raise MappingError(f"set paths must be nonempty POSIX strings under raw/ or code/: {text!r}")
    rel = PurePosixPath(text)
    if (rel.is_absolute() or ".." in rel.parts or not rel.parts or rel.parts[0] not in ("raw", "code")
            or rel.as_posix() != text):
        raise MappingError(f"set paths must lie under raw/ or code/: {text!r}")
    return Path(rel)


def _reserved(path: Path) -> bool:
    return (path.parts[:2] == ("raw", "derived") or path.as_posix() in {"code/MAPPING.json", "code/SOURCE.json"}
            or path.name == "MANIFEST.json")


def _copy(target: Path, entry: dict, inputs: list[dict], *, base: Path) -> None:
    source, destination = entry.get("from"), entry.get("to")
    if not isinstance(source, str) or not source:
        raise MappingError("every raw_files entry needs from: the original file or directory")
    source_path = Path(source).expanduser()
    if not source_path.is_absolute():
        source_path = base / source_path
    if source_path.is_symlink() or not (source_path.is_file() or source_path.is_dir()):
        raise MappingError(f"original output missing, not regular, or a symlink: {source}")
    rel = _relative(destination)
    if source_path.is_file():
        items = [(source_path, rel)]
    else:
        discovered = sorted(source_path.rglob("*"))
        if any(p.is_symlink() or not (p.is_file() or p.is_dir()) for p in discovered):
            raise MappingError(f"symlinks and special files are not deposited: {source}")
        items = [(p, rel / p.relative_to(source_path)) for p in discovered if p.is_file()]
    for item, item_rel in items:
        if _reserved(item_rel):
            raise MappingError(f"destination {item_rel.as_posix()} is reserved for generated metadata; choose another path")
        out = target / item_rel
        before = _sha256(item)
        if out.exists():
            if not out.is_file() or _sha256(out) != before:
                raise MappingError(f"two different originals map to {item_rel.as_posix()}")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(item, out)
        copied = _sha256(out)
        if copied != before or _sha256(item) != before:
            raise MappingError(f"original changed during copying: {item.name}")
        inputs.append({"path": item_rel.as_posix(), "original_name": item.name, "sha256": copied,
                       "bytes": out.stat().st_size})


def _located(target: Path, source_file: str, locator: str, where: str) -> tuple[str, float]:
    from scripts.records import locate, parse_number
    rel = _relative(source_file)
    path = target / rel
    if not path.is_file():
        raise MappingError(f"{where}: {source_file} is not among the copied files")
    if not isinstance(locator, str) or not locator:
        raise MappingError(f"{where}: locator missing")
    text = str(locate(path, locator, where)).strip()
    return text, parse_number(text, where)


def _calculated(op: str, values: list[float], where: str) -> float:
    if op not in OPS or not values:
        raise MappingError(f"{where}: derive needs op in {OPS} and a nonempty inputs list")
    if op in ("ratio", "difference") and len(values) != 2:
        raise MappingError(f"{where}: {op} needs exactly two inputs")
    if op == "sum":
        value = math.fsum(values)
    elif op == "mean":
        value = math.fsum(values) / len(values)
    elif op == "ratio":
        if values[1] == 0:
            raise MappingError(f"{where}: ratio denominator is zero")
        value = values[0] / values[1]
    else:
        value = values[0] - values[1]
    if not math.isfinite(value):
        raise MappingError(f"{where}: derived value is not finite")
    return value


def _derive(target: Path, spec: dict, where: str, index: int) -> tuple[str, str, str]:
    if not isinstance(spec, dict) or not isinstance(spec.get("inputs"), list) or not spec["inputs"]:
        raise MappingError(f"{where}: derive needs an operation and nonempty inputs list")
    op, inputs = spec.get("op"), spec["inputs"]
    located, values = [], []
    for item in inputs:
        if not isinstance(item, dict) or set(item) != {"source_file", "locator"}:
            raise MappingError(f"{where}: derived inputs must name only source_file and locator")
        rel = _relative(item["source_file"])
        if _reserved(rel):
            raise MappingError(f"{where}: derived inputs must refer to copied originals, not generated metadata")
        text, value = _located(target, item["source_file"], item["locator"], where)
        located.append({"source_file": item["source_file"], "locator": item["locator"], "raw_text": text,
                        "sha256": _sha256(target / rel)})
        values.append(value)
    from scripts.export_records import fmt
    printed = fmt(_calculated(op, values, where))
    rel = f"raw/derived/row_{index:05d}.json"
    out = target / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise MappingError(f"{where}: derived output already exists: {rel}")
    out.write_text(json.dumps({"schema": "cross-morse-derived-value-1", "op": op, "inputs": located,
                               "value": float(printed), "definition": spec.get("definition", "")},
                              indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return printed, rel, "json:/value"


def verify_derivations(target: Path, rows: list[dict]) -> None:
    """Read-only replay for mapped values; invoked by records.verify_set, including reindexing."""
    from scripts.export_records import fmt
    mapping_file = target / "code/MAPPING.json"
    mapping = json.loads(mapping_file.read_text(encoding="utf-8"))
    provenance = json.loads((target / "provenance.json").read_text(encoding="utf-8"))
    if (mapping.get("schema") != MAPPING_SCHEMA or mapping.get("set") != target.name
            or provenance.get("public_mapping_sha256") != _sha256(mapping_file)):
        raise MappingError(f"{target.name}: public mapping identity differs from its provenance")
    declarations = [row for entry in mapping.get("rows", []) for row in _expand(entry)]
    if len(declarations) != len(rows):
        raise MappingError(f"{target.name}: summary and mapping row coverage differ")
    declarations_by_key = {}
    for index, declaration in enumerate(declarations):
        keys = declaration.get("keys") or {}
        identity = tuple(sorted((key, str(value)) for key, value in keys.items())) + (("quantity", declaration.get("quantity")),)
        if identity in declarations_by_key:
            raise MappingError(f"{target.name}: duplicated mapping declaration")
        declarations_by_key[identity] = (index, declaration)
    for row in rows:
        identity = tuple(sorted((key, str(value)) for key, value in row.items()
                                if key not in {"quantity", "value", "unit", "source_file", "locator"})) + (("quantity", row["quantity"]),)
        if identity not in declarations_by_key:
            raise MappingError(f"{target.name}: summary row has no mapping declaration")
        index, declaration = declarations_by_key[identity]
        derived = declaration.get("derive")
        expected_file = f"raw/derived/row_{index:05d}.json" if derived else declaration.get("source_file")
        expected_locator = "json:/value" if derived else declaration.get("locator")
        if (row["source_file"] != expected_file or row["locator"] != expected_locator
                or row["unit"] != str(declaration.get("unit", "")).strip()):
            raise MappingError(f"{target.name}: summary source, locator or unit differs from its mapping")
        rel = _relative(row["source_file"])
        if rel.parts[:2] != ("raw", "derived"):
            continue
        where = f"{target.name}/{rel.as_posix()}"
        if row["locator"] != "json:/value":
            raise MappingError(f"{where}: mapped derived rows must locate json:/value")
        spec = json.loads((target / rel).read_text(encoding="utf-8"))
        if (spec.get("schema") != "cross-morse-derived-value-1"
                or not isinstance(spec.get("inputs"), list) or not spec["inputs"]
                or any(not isinstance(entry, dict) for entry in spec["inputs"])):
            raise MappingError(f"{where}: invalid derived-value record")
        if (not isinstance(derived, dict) or spec.get("op") != derived.get("op")
                or [{"source_file": entry.get("source_file"), "locator": entry.get("locator")} for entry in spec["inputs"]]
                != derived.get("inputs")):
            raise MappingError(f"{where}: derivation differs from its mapping declaration")
        values = []
        for entry in spec["inputs"]:
            if not isinstance(entry, dict):
                raise MappingError(f"{where}: invalid derivation input")
            source = _relative(entry.get("source_file"))
            if _reserved(source):
                raise MappingError(f"{where}: derivation input is not a copied original")
            text, value = _located(target, entry.get("source_file"), entry.get("locator"), where)
            if _sha256(target / source) != entry.get("sha256") or text != entry.get("raw_text"):
                raise MappingError(f"{where}: derivation input bytes or located text changed")
            values.append(value)
        expected = float(fmt(_calculated(spec.get("op"), values, where)))
        actual = spec.get("value")
        if isinstance(actual, bool) or not isinstance(actual, (int, float)) or actual != expected:
            raise MappingError(f"{where}: derived value does not recompute from the copied inputs")


@contextmanager
def _install_lock(root: Path):
    import fcntl
    descriptor = os.open(root, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def install_verified_set(staging_root: Path, target: Path, records_root: Path, specs: dict, *, replace: bool) -> Path:
    from scripts.records import deposited_sets, verify_set, write_index
    final = records_root / target.name
    with _install_lock(records_root):
        if final.is_symlink() or (records_root / "index.json").is_symlink():
            raise MappingError("record destination and index must not be symlinks")
        if final.exists() and not replace:
            raise MappingError(f"{final} exists; pass --replace to rebuild it")
        # Existing sets retain their manifests. Importing one new set must not
        # silently reindex or re-bless altered, unrelated measurements.
        entries = [verify_set(p, specs["sets"][p.name], specs) for p in deposited_sets(records_root, specs)
                   if p.name != target.name]
        entries.append(verify_set(target, specs["sets"][target.name], specs))
        write_index(staging_root, specs, entries)
        backup = staging_root / "previous-set"
        moved_old, installed = False, False
        try:
            if final.exists():
                os.replace(final, backup)
                moved_old = True
            os.replace(target, final)
            installed = True
            # This is the commit point; failures before it restore the old set
            # and leave the old index bytes unchanged.
            os.replace(staging_root / "index.json", records_root / "index.json")
        except BaseException:
            if installed:
                os.replace(final, target)
            if moved_old:
                os.replace(backup, final)
            raise
    return final


def _private_scan(target: Path, extra: dict | None = None) -> list[str]:
    """Same identifiers the public release audit refuses, applied to every deposited text file."""
    from scripts.release_audit import PRIVATE_PATTERNS
    texts = [(path.relative_to(target).as_posix(), path) for path in sorted(target.rglob("*")) if path.is_file()]
    findings = []
    for label, item in texts + [(name, value) for name, value in (extra or {}).items()]:
        if isinstance(item, Path):
            try:
                text = item.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
        else:
            text = item
        for pattern in PRIVATE_PATTERNS:
            match = pattern.search(text)
            if match:
                findings.append(f"{label}: matches a private infrastructure pattern near "
                                f"line {text.count(chr(10), 0, match.start()) + 1}")
    return findings


def _public_mapping(mapping: dict) -> dict:
    """The mapping as deposited: original locations reduced to file names."""
    public = {k: v for k, v in mapping.items() if not k.startswith("_")}
    public["raw_files"] = [{k: (Path(v).name if k == "from" and isinstance(v, str) else v) for k, v in entry.items()}
                           for entry in mapping.get("raw_files") or []]
    return public


def build(mapping_path: Path, *, records_root: Path, replace: bool = False, allow_partial: bool = False,
          allow_private: bool = False, argv: list[str] | None = None) -> Path:
    from scripts.export_records import finish_set
    from scripts.records import load_specs
    from scripts.scientific_evidence_audit import validate_record_coverage
    mapping_path = Path(mapping_path).resolve()
    mapping_bytes = mapping_path.read_bytes()
    mapping = json.loads(mapping_bytes)
    if not isinstance(mapping, dict):
        raise MappingError("mapping must be an object")
    if mapping.get("schema") != MAPPING_SCHEMA:
        raise MappingError(f"mapping schema must be {MAPPING_SCHEMA}")
    specs = _specs()
    name = mapping.get("set")
    if name not in specs["sets"]:
        raise MappingError(f"unknown record set {name!r}")
    spec, contract = specs["sets"][name], _contract(name)
    provenance = dict(mapping.get("provenance") or {})
    empty = [f for f in specs["provenance_fields"] if not str(provenance.get(f) or "").strip()]
    if empty:
        raise MappingError(f"provenance lacks {empty}")
    if provenance.get("synthetic", False) is not False:
        raise MappingError("records/ holds measured sets only; synthetic examples belong under examples/records/")
    records_root = Path(records_root).resolve()
    final = records_root / name
    if final.exists() and not replace:
        raise MappingError(f"{final} exists; pass --replace to rebuild it from the mapping")
    if not records_root.is_dir() or not (records_root / "specs.json").is_file():
        raise MappingError("records root needs the registered specs.json before importing a mapping")
    if load_specs(records_root, records_root / "specs.json") != specs:
        raise MappingError("the target record specifications differ from the registered contract")
    if records_root == (ROOT / "records").resolve() and (allow_private or allow_partial):
        raise MappingError("partial or private mappings are for external staging roots only, never public records/")
    staging_parent = ROOT.parent if records_root.is_relative_to(ROOT) else records_root.parent
    if staging_parent.stat().st_dev != records_root.stat().st_dev:
        raise MappingError("staging and records must be on the same filesystem for atomic installation")
    staging_root = Path(tempfile.mkdtemp(prefix=f"cm-record-{name}-", dir=staging_parent))
    try:
        target = staging_root / name
        (target / "raw").mkdir(parents=True)
        (target / "code").mkdir()
        inputs: list[dict] = []
        for entry in mapping.get("raw_files") or []:
            for expanded in _expand(entry):
                _copy(target, expanded, inputs, base=mapping_path.parent)
        if not inputs:
            raise MappingError("raw_files lists no original output")
        rows, seen = [], set()
        per_seed = set(spec.get("per_seed_quantities", []))
        expanded_rows = [row for entry in mapping.get("rows") or [] for row in _expand(entry)]
        if not expanded_rows:
            raise MappingError("the mapping has no rows")
        for index, row in enumerate(expanded_rows):
            keys = row.get("keys") or {}
            where = f"row {index} {json.dumps(keys)} {row.get('quantity')}"
            if list(keys) != spec["keys"] and set(keys) != set(spec["keys"]):
                raise MappingError(f"{where}: keys must be exactly {spec['keys']}")
            if "value" in row:
                raise MappingError(f"{where}: inline values are not accepted; use an original-file locator")
            if row.get("derive") and (row.get("source_file") or row.get("locator")):
                raise MappingError(f"{where}: choose either derive or source_file/locator")
            quantity = row.get("quantity")
            if quantity not in spec["quantities"]:
                raise MappingError(f"{where}: quantity is not defined for {name}")
            if not isinstance(row.get("unit"), str) or not row["unit"].strip():
                raise MappingError(f"{where}: unit missing")
            if "seed" in spec["keys"] and (quantity in per_seed) != bool(str(keys.get("seed", "")).strip()):
                raise MappingError(f"{where}: seed must be set exactly for per-seed quantities {sorted(per_seed)}")
            identity = tuple(str(keys[k]) for k in spec["keys"]) + (quantity,)
            if identity in seen:
                raise MappingError(f"{where}: duplicate row")
            seen.add(identity)
            if row.get("derive"):
                value, source_file, locator = _derive(target, row["derive"], where, index)
            else:
                value, _ = _located(target, row.get("source_file"), row.get("locator"), where)
                source_file, locator = row["source_file"], row["locator"]
            rows.append({**{k: str(keys[k]) for k in spec["keys"]}, "quantity": quantity, "value": value,
                         "unit": str(row["unit"]).strip(), "source_file": source_file, "locator": locator})
        coverage_complete = True
        try:
            validate_record_coverage(name, rows, spec, contract)
        except ValueError as exc:
            coverage_complete = False
            if not allow_partial:
                raise MappingError(f"coverage required by the scientific audit is incomplete: {exc}") from None
        (target / "code" / "MAPPING.json").write_text(json.dumps(_public_mapping(mapping), indent=1, sort_keys=True,
                                                                 default=str) + "\n", encoding="utf-8")
        private = _private_scan(target, {"provenance": json.dumps(provenance)})
        if private and not allow_private:
            raise MappingError("redact private infrastructure identifiers before depositing (or pass "
                               "--allow-private-patterns for a non-public set):\n  " + "\n  ".join(private[:10]))
        provenance.setdefault("synthetic", False)
        provenance["built_by"] = "scripts/export_records.py from-mapping (scripts/record_mapping.py)"
        if mapping_path.read_bytes() != mapping_bytes:
            raise MappingError("mapping changed during export; repeat with stable inputs")
        provenance["mapping_sha256"] = hashlib.sha256(mapping_bytes).hexdigest()
        provenance["public_mapping_sha256"] = _sha256(target / "code/MAPPING.json")
        provenance["coverage_complete"] = coverage_complete
        provenance["traceability_scope"] = "Located file values and recomputed declared derivations; original measurement provenance and units remain author declarations."
        finish_set(staging_root, target, keys=list(spec["keys"]), rows=rows, provenance=provenance,
                   argv=argv or ["scripts/export_records.py", "from-mapping", "--mapping", mapping_path.name],
                   inputs=inputs)
        # Include generated metadata in the privacy scan too (including the
        # command/provenance fields), before any public destination is touched.
        private = _private_scan(target)
        if private and not allow_private:
            raise MappingError("generated metadata contains private infrastructure identifiers: " + "; ".join(private[:10]))
        return install_verified_set(staging_root, target, records_root, specs, replace=replace)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
