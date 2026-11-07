#!/usr/bin/env python3
"""Generate the manuscript registry's claim-to-evidence map and registration badge.

This deterministic map describes indexes and available builders. Index flags and
DOI strings are registrations, not fresh integrity checks, proof of publication,
or proof that every manuscript value is supported. `make review` performs those
separate checks and states the bounded scope of its scientific evidence audit.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.availability import DATASETS  # noqa: E402
from scripts._paper_common import runtime_source_fingerprint  # noqa: E402

LEDGER = ROOT / "paper" / "CLAIMS.md"
BADGE = ROOT / "docs" / "badges" / "evidence.json"
RUN_DERIVED = {"1", "2", "S4", "S12", "S12c", "S13", "S14", "S15", "S23", "S41"}
KIND_TEXT = {"author_reported_measurements": "measurement", "mixed": "measurement + derived",
             "derived_arithmetic": "derived arithmetic", "analytical_sensitivity": "analytical sensitivity"}
DOI = re.compile(r"10\.\d{4,9}/[A-Za-z0-9._;()/:-]+")


def registered_doi(value) -> str | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str) or not DOI.fullmatch(value):
        raise ValueError(f"Invalid registered DOI: {value!r}")
    return value


def registration_state(root: Path = ROOT, *, index_path: Path | None = None, records_root: Path | None = None) -> dict:
    """Read registrations without upgrading index flags to fresh verification."""
    root = Path(root)
    index_path = index_path or root / "runs" / "index.json"
    records_root = records_root or root / "records"
    scope = json.loads((root / "release" / "release_scope.json").read_text(encoding="utf-8"))
    registered_doi(scope.get("code_archive_doi"))
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {"schema": "cross-morse-runs-index-1", "runs": []}
    if index.get("schema") != "cross-morse-runs-index-1" or not isinstance(index.get("runs"), list):
        raise ValueError("Invalid runs index schema")
    official, seen = {}, set()
    for entry in index["runs"]:
        if not isinstance(entry, dict) or entry.get("dataset") not in DATASETS or entry["dataset"] in seen:
            raise ValueError("Run index has an unknown or duplicate dataset entry")
        seen.add(entry["dataset"])
        registered_doi(entry.get("deposit_doi"))
        if entry.get("verified") is True and entry.get("scientific_run") is True and entry.get("profile") == "full":
            official[entry["dataset"]] = entry
    specs = json.loads((root / "records" / "specs.json").read_text(encoding="utf-8"))
    if specs.get("schema") != "cross-morse-records-specs-1":
        raise ValueError("Invalid record specifications")
    records_index_path = records_root / "index.json"
    indexed = json.loads(records_index_path.read_text(encoding="utf-8")) if records_index_path.is_file() else {"schema": "cross-morse-records-index-1", "sets": []}
    if indexed.get("schema") != "cross-morse-records-index-1" or not isinstance(indexed.get("sets"), list):
        raise ValueError("Invalid records index schema")
    verified, seen = [], set()
    for entry in indexed["sets"]:
        if not isinstance(entry, dict) or entry.get("set") not in specs["sets"] or entry["set"] in seen:
            raise ValueError("Record index has an unknown or duplicate set")
        seen.add(entry["set"])
        if entry.get("verified") is True and entry.get("synthetic") is False:
            verified.append(entry["set"])
    return {"scope": scope, "official": official, "records": {"verified": verified, "defined": list(specs["sets"])},
            "source_sha256": runtime_source_fingerprint(root)}


def table_tokens(label: str) -> set[str]:
    """Extract tables, excluding figure labels in combined registry headings."""
    tables = re.split(r"\b(?:and\s+)?Figures?\b", label, maxsplit=1)[0]
    if not re.search(r"\bTables?\b", tables):
        return set()
    return set(re.findall(r"\bS\d+[a-z]?\b|(?<=Table )\b[12]\b|(?<=Tables )\b[12]\b", tables))


def record_sets_by_table(specs: dict) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for name, spec in specs["sets"].items():
        for label in spec.get("manuscript_tables", []):
            for token in table_tokens(label):
                mapping.setdefault(token, []).append(name)
    return mapping


def build(root: Path = ROOT, *, index_path: Path | None = None, records_root: Path | None = None) -> dict:
    sources = json.loads((root / "paper" / "sources.json").read_text(encoding="utf-8"))
    specs = json.loads((root / "records" / "specs.json").read_text(encoding="utf-8"))
    st = registration_state(root, index_path=index_path, records_root=records_root)
    official, records = st["official"], st["records"]
    registered_sets = set(records["verified"])
    by_table = record_sets_by_table(specs)
    dois = {dataset: registered_doi(entry.get("deposit_doi")) for dataset, entry in official.items()}
    n_official = len(official)
    regenerate = (st.get("scope") or {}).get("result_distribution") == "regenerate" and not official
    if (st.get("scope") or {}).get("result_distribution") == "repository_release":
        return _repository_release_ledger(root, sources, st)
    rows = []
    for key, group in sources["table_groups"].items():
        label = group.get("label", key)
        tokens = table_tokens(label)
        kind = group.get("evidence_kind", "mixed")
        sets = sorted({s for token in tokens for s in by_table.get(token, [])})
        run_derived = bool(tokens & RUN_DERIVED)
        experiments = sorted({e for s in sets for e in specs["sets"][s].get("experiments", [])})
        producers, required, status = [], [], []
        if run_derived:
            if regenerate:
                producers.append("run-derived columns: `scripts/build_paper.py` on the runs of a reproduction suite")
                required.append("completed full-profile runs of the reported datasets")
                status.append("`run_cross_morse.sh reproduce` generates run artifacts; its report rebuilds the supported columns")
            else:
                producers.append("run-derived columns: eight-stage archive (`make paper`)")
                required.append("official run archives for the reported datasets")
                status.append("Registered archives: " + (", ".join(sorted(official)) or "see the run index")
                              + f"; DOI references: {sum(bool(v) for v in dois.values())}")
        for name in sets:
            experiments_text = ", ".join(specs["sets"][name].get("experiments", [])) or "declared experiment records"
            producers.append(f"record set `{name}` ({experiments_text}; `make records-tables`)")
            required.append(f"`records/{name}/` with raw sources")
            if regenerate:
                status.append(f"`{name}`: regenerated by its experiment driver (docs/EXPERIMENTS.md); `make records-check` traces it")
            else:
                status.append(f"`{name}` {'registered' if name in registered_sets else 'not registered'}; trace check required")
        if not run_derived and not sets:
            if tokens == {"S5d"}:
                producers.append("`scripts/model_inventory.py`; sealed `train/model_inventory.json` for benchmark identity")
                required.append("archived checkpoints and inventories for the original benchmark models")
                status.append("released configurations are countable; historical identity requires run evidence")
            elif kind in {"derived_arithmetic", "analytical_sensitivity"}:
                producers.append("registered formulas and inputs (`make registry-check` covers enumerated identities)")
                required.append("the source observations for any measured inputs")
                status.append("arithmetic validation does not establish the measured inputs")
            else:
                producers.append("source observations and the comparison target in `paper/reported_results.json`")
                required.append("original observations identified by the manuscript provenance")
                status.append("no dedicated display builder is declared for this registry group")
        rows.append({"key": key, "label": label, "pages": group.get("pages", []), "source": group.get("source"),
                     "evidence_kind": kind, "experiments": experiments, "producers": producers,
                     "required": required, "status": status})
    rows.sort(key=lambda row: (0 if row["source"] == "main" else 1, _label_sort(row["label"])))
    counts = {row["label"]: sum(other["label"] == row["label"] for other in rows) for row in rows}
    for row in rows:
        if counts[row["label"]] > 1:
            row["label"] = f"{row['label']} (`{row['key']}`)"
    return {"schema": "cross-morse-claims-ledger-1", "official_archives": n_official, "official_total": len(DATASETS),
            "dois": dois, "record_sets_registered": sorted(registered_sets), "record_sets_defined": sorted(records["defined"]),
            "source_sha256": st["source_sha256"], "result_distribution": "regenerate" if regenerate else "registered_archives",
            "registered_coverage_complete": set(official) == set(DATASETS) and registered_sets == set(records["defined"]),
            "scientific_evidence_verified": False, "rows": rows}


def _repository_release_ledger(root: Path, sources: dict, st: dict) -> dict:
    """Each registry group: the generators of its displays, the artifacts they read and their publication state."""
    from paper.build.displays.framework import GENERATORS, _load_generators
    from scripts import paper_artifacts
    _load_generators()
    manifest = paper_artifacts.load_manifest(root / "release" / "paper_artifacts.json")
    state = {a["id"]: a.get("status") for a in manifest["artifacts"]}
    displays = json.loads((root / "paper" / "display_manifest.json").read_text(encoding="utf-8"))["displays"]
    rows = []
    for key, group in sources["table_groups"].items():
        label = group.get("label", key)
        users = [d for d in displays if key in d.get("registry_groups", [])]
        producers, required, status = [], [], []
        for display in users:
            generator = GENERATORS.get(display["id"])
            if generator is None:
                producers.append(f"{display['label']}: no generator")
                continue
            module = generator.function.__module__.rsplit(".", 1)[-1]
            producers.append(f"{display['label']}: `paper/build/displays/{module}.py` ({generator.stem})")
            for need in generator.needs:
                if need == "static":
                    continue
                kind, _, name = need.partition(":")
                ids = ([f"run:{d}" for d in DATASETS] if kind == "runs" else [need])
                for artifact_id in ids:
                    if artifact_id not in required:
                        required.append(artifact_id)
        awaiting = [a for a in required if state.get(a) != "published"]
        if not producers:
            status.append("no numbered display reads this group")
        elif not required:
            status.append("generated from the repository's configuration and analytical code")
        elif awaiting:
            status.append("awaiting publication: " + ", ".join(f"`{a}`" for a in awaiting))
        else:
            status.append("inputs published; `make paper` rebuilds and compares every printed value")
        rows.append({"key": key, "label": label, "pages": group.get("pages", []), "source": group.get("source"),
                     "evidence_kind": group.get("evidence_kind", "mixed"), "experiments": [],
                     "producers": producers, "required": [f"`{r}`" for r in required] or ["repository sources"],
                     "status": status})
    rows.sort(key=lambda row: (0 if row["source"] == "main" else 1, _label_sort(row["label"])))
    counts = {row["label"]: sum(other["label"] == row["label"] for other in rows) for row in rows}
    for row in rows:
        if counts[row["label"]] > 1:
            row["label"] = f"{row['label']} (`{row['key']}`)"
    from scripts import statement_check
    compliance = statement_check.evaluate(root)
    return {"schema": "cross-morse-claims-ledger-1", "official_archives": 0, "official_total": len(DATASETS),
            "dois": {}, "record_sets_registered": [], "record_sets_defined": sorted(st["records"]["defined"]),
            "source_sha256": st["source_sha256"], "result_distribution": "repository_release",
            "release_url": manifest["release_url"],
            "artifacts_published": sum(1 for a in manifest["artifacts"] if a.get("status") == "published"),
            "artifacts_required": len(manifest["artifacts"]),
            "statement_fulfilled": compliance["counts"]["fulfilled"], "statement_items": len(compliance["items"]),
            "registered_coverage_complete": manifest["status"] == "published",
            "scientific_evidence_verified": False, "rows": rows}


def _label_sort(label: str):
    match = re.search(r"(S?)(\d+)([a-z]?)", label)
    return (1 if match.group(1) else 0, int(match.group(2)), match.group(3)) if match else (9, 0, "")


def _render_rows(ledger: dict) -> list[str]:
    lines = ["| Manuscript item | Pages | Evidence kind | Builder or source | Required evidence | Status |",
             "|---|---|---|---|---|---|"]
    for row in ledger["rows"]:
        pages = ", ".join(str(p) for p in row["pages"]) or "—"
        doc = "main" if row["source"] == "main" else "SI"
        cells = [f"{row['label']} ({doc})", pages, KIND_TEXT.get(row["evidence_kind"], row["evidence_kind"]),
                 "; ".join(row["producers"]), "; ".join(row["required"]), "; ".join(row["status"])]
        lines.append("| " + " | ".join(cell.replace("|", r"\|") for cell in cells) + " |")
    return lines


def render(ledger: dict) -> str:
    n, total = ledger["official_archives"], ledger["official_total"]
    registered, defined = ledger["record_sets_registered"], ledger["record_sets_defined"]
    if ledger.get("result_distribution") == "repository_release":
        lines = ["# Claim-to-evidence map", "",
                 "Generated from `paper/sources.json`, `paper/display_manifest.json`, the display generators and "
                 "`release/paper_artifacts.json`. `make claims-check` checks freshness.", "",
                 "## Outputs in this release", "",
                 f"- The measured outputs of the reported experiments are assets of the [release]({ledger['release_url']}): "
                 f"{ledger['artifacts_published']} of {ledger['artifacts_required']} required artifacts published.",
                 "- `make paper` downloads and verifies them, rebuilds every numbered display with "
                 "`paper/build/displays/` and compares every computed value with the printed one.",
                 f"- Availability statement: {ledger['statement_fulfilled']} of {ledger['statement_items']} items "
                 "fulfilled ([compliance](../STATEMENT_COMPLIANCE.md)).",
                 f"- Runtime source identity: `{ledger['source_sha256']}`.", "",
                 "## Manuscript groups, generators and inputs", ""]
        return "\n".join(lines + _render_rows(ledger)) + "\n"
    if ledger.get("result_distribution") == "regenerate":
        lines = ["# Claim-to-evidence map", "",
                 "Generated from `paper/sources.json`, `records/specs.json` and the run/record indexes. "
                 "`make claims-check` checks freshness. This map covers the registered manuscript groups, "
                 "including those without a dedicated display builder.", "", "## Results in this release", "",
                 "- This is a code release: run outputs are regenerated with `bash run_cross_morse.sh reproduce` "
                 "([guide](../docs/HPC.md)); its report rebuilds the run-derived displays below from the regenerated runs "
                 "and places every compared quantity beside the printed value.",
                 "- Record sets of the experiments outside the lifecycle are regenerated by their drivers "
                 "([Experiments](../docs/EXPERIMENTS.md)) and traced to their raw files by `make records-check`.",
                 f"- Runtime source identity: `{ledger['source_sha256']}`.", "",
                 "## Manuscript groups and builders", "",
                 "Run builders reconstruct the named run-derived columns; record builders emit files named after the "
                 "record set (for example `baselines.tex`), with the corresponding manuscript table labels in their titles. "
                 "Analytical calculations use registered formulas and inputs. Registry plots remain transcriptions or "
                 "analytical scenarios rather than replays of benchmark observations.", "",
                 "| Manuscript item | Pages | Evidence kind | Builder or source | Required evidence | Status |",
                 "|---|---|---|---|---|---|"]
        for row in ledger["rows"]:
            pages = ", ".join(str(p) for p in row["pages"]) or "—"
            doc = "main" if row["source"] == "main" else "SI"
            cells = [f"{row['label']} ({doc})", pages, KIND_TEXT.get(row["evidence_kind"], row["evidence_kind"]),
                     "; ".join(row["producers"]), "; ".join(row["required"]), "; ".join(row["status"])]
            lines.append("| " + " | ".join(cell.replace("|", r"\|") for cell in cells) + " |")
        return "\n".join(lines) + "\n"
    lines = ["# Claim-to-evidence map", "",
             "Generated from `paper/sources.json`, `records/specs.json` and the run/record indexes. "
             "`make claims-check` checks freshness. This map covers the registered manuscript groups, "
             "including those without a dedicated display builder.", "", "## Registered coverage", "",
             "- Registered full-profile datasets: " + (", ".join(sorted(ledger["dois"])) or "see the run index and archive workflow") + ".",
             "- Registered measurement sets: " + (", ".join(f"`{name}`" for name in sorted(registered)) or "see the record index and experiment drivers") + ".",
             f"- Runtime source identity: `{ledger['source_sha256']}`.", "",
             "Registration flags and DOI references do not establish fresh byte integrity, public deposit availability, "
             "or agreement with the manuscript. `make review` performs those checks; its scientific audit covers "
             "enumerated core results and declared auxiliary records, not every statement in the paper.", "",
             "| Dataset | Index registration | Registered DOI reference |", "|---|---|---|"]
    for dataset in DATASETS:
        doi = ledger["dois"].get(dataset)
        label = "registered" if dataset in ledger["dois"] else "not registered"
        lines.append(f"| {dataset} | {label} | " + (f"[{doi}](https://doi.org/{doi})" if doi else "—") + " |")
    lines += ["", "| Record set | Index registration |", "|---|---|"]
    for name in defined:
        lines.append(f"| `{name}` | {'registered' if name in registered else 'not registered'} |")
    lines += ["", "## Manuscript groups and builders", "",
              "Run builders reconstruct the named run-derived columns; record builders emit files named after the "
              "record set (for example `baselines.tex`), with the corresponding manuscript table labels in their titles. "
              "Analytical calculations use registered formulas and inputs. Registry plots remain transcriptions or "
              "analytical scenarios rather than replays of benchmark observations.", "",
              "| Manuscript item | Pages | Evidence kind | Builder or source | Required evidence | Status |",
              "|---|---|---|---|---|---|"]
    for row in ledger["rows"]:
        pages = ", ".join(str(p) for p in row["pages"]) or "—"
        doc = "main" if row["source"] == "main" else "SI"
        cells = [f"{row['label']} ({doc})", pages, KIND_TEXT.get(row["evidence_kind"], row["evidence_kind"]),
                 "; ".join(row["producers"]), "; ".join(row["required"]), "; ".join(row["status"])]
        lines.append("| " + " | ".join(cell.replace("|", r"\|") for cell in cells) + " |")
    return "\n".join(lines) + "\n"


def badge(ledger: dict) -> dict:
    if ledger.get("result_distribution") == "repository_release":
        done = ledger["statement_fulfilled"] == ledger["statement_items"]
        return {"schemaVersion": 1, "label": "availability statement",
                "message": f"{ledger['statement_fulfilled']}/{ledger['statement_items']} fulfilled · outputs "
                           f"{ledger['artifacts_published']}/{ledger['artifacts_required']} published",
                "color": "brightgreen" if done else "blue"}
    if ledger.get("result_distribution") == "regenerate":
        return {"schemaVersion": 1, "label": "reproduction", "message": "run_cross_morse.sh reproduce", "color": "blue"}
    return {"schemaVersion": 1, "label": "artifact index", "message": "archives and records", "color": "blue"}


def write(ledger: dict, *, ledger_path: Path, badge_path: Path) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    badge_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(render(ledger), encoding="utf-8")
    badge_path.write_text(json.dumps(badge(ledger), indent=1) + "\n", encoding="utf-8")


def check(ledger: dict, *, ledger_path: Path, badge_path: Path) -> list[str]:
    problems = []
    for path, expected in ((ledger_path, render(ledger)), (badge_path, json.dumps(badge(ledger), indent=1) + "\n")):
        if not path.is_file() or path.read_text(encoding="utf-8") != expected:
            problems.append(f"{path} is stale; run `make claims`")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--output", type=Path, help="directory for CLAIMS.md and evidence.json instead of the tree")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    try:
        ledger = build(args.root, index_path=args.index, records_root=args.records)
        ledger_path = args.output / "CLAIMS.md" if args.output else args.root / "paper" / "CLAIMS.md"
        badge_path = args.output / "evidence.json" if args.output else args.root / "docs" / "badges" / "evidence.json"
        if args.check:
            problems = check(ledger, ledger_path=ledger_path, badge_path=badge_path)
            for problem in problems:
                print("CHECK FAILED:", problem, file=sys.stderr)
            if problems:
                return 1
            print(f"claim map matches the indexes ({ledger['official_archives']}/{ledger['official_total']} archives, "
                  f"{len(ledger['record_sets_registered'])}/{len(ledger['record_sets_defined'])} record sets registered)")
            return 0
        write(ledger, ledger_path=ledger_path, badge_path=badge_path)
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(ledger, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {ledger_path} and {badge_path} ({len(ledger['rows'])} manuscript groups)")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
