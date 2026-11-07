#!/usr/bin/env python3
"""Render source-bound paper results and a complete table/figure support map.

This reads the manuscript registry and artifact registration indexes. It does
not execute a benchmark, infer missing observations, or certify a measurement.
The reviewed display inventory is immutable until explicitly updated for new
PDFs; rendering never silently rebinds it to a different manuscript.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from scripts._paper_common import DATASETS, resolved_policy_view
from scripts.check_reviewer_faq import make_targets
from scripts.claims_ledger import registration_state
from scripts.render_paper_results import read_json, render_csv, validate_sources

SCHEMA = "cross-morse-display-manifest-1"
OUTPUTS = ("paper/PAPER_SUPPORT.md", "paper/RESULTS.md", "paper/RESULTS.csv")
KINDS = {
    "protocol": "Protocol specification",
    "reported_measurements": "Reported measurements",
    "mixed": "Measurements and/or derived quantities",
    "analytical": "Mathematical or analytical calculation",
    "explanatory": "Method illustration",
}
COVERAGE = {
    "run_columns": "Run-derived columns",
    "record_outputs": "Record-set tables",
    "protocol_check": "Protocol checks",
    "arithmetic_check": "Theory/arithmetic checks",
    "inventory_check": "Model inventory checks",
    "partial_panels": "Related plots / partial panels",
    "method_illustration": "Method documentation",
    "original_analysis_required": "Original analysis required",
    "full_display_generator": "Full-display generator",
}
SUPPLEMENT_SUFFIXES = ("5b", "5c", "5d", "6b", "6c", "12b", "12c", "14b",
                       "15b", "16b", "23b", "26b", "28b", "46b")


def expected_displays() -> set[str]:
    """The complete numbering reviewed in the registered November 2026 PDFs."""
    return (
        {f"main:Table {i}" for i in range(1, 3)}
        | {f"main:Figure {i}" for i in range(1, 7)}
        | {f"supplementary:Table S{i}" for i in range(1, 49)}
        | {f"supplementary:Table S{i}" for i in SUPPLEMENT_SUFFIXES}
        | {f"supplementary:Figure S{i}" for i in range(1, 21)}
    )


def _path(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Invalid repository path: {value!r}")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"Invalid repository path: {value!r}")
    result = root / path
    if result.is_symlink() or not result.is_file() or not result.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Referenced repository file is absent or unsafe: {value}")
    return result


def validate_manifest(root: Path, manifest: dict) -> dict:
    root = Path(root)
    sources = read_json(root / "paper/sources.json")
    reported = read_json(root / "paper/reported_results.json")
    validate_sources(root, reported, sources)
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unsupported display-manifest schema")
    documents = {role: {key: record[key] for key in ("sha256", "byte_size", "pages")}
                 for role, record in sources["documents"].items()}
    if manifest.get("documents") != documents:
        raise ValueError("Display inventory does not identify the current PDFs; review the new sources explicitly")
    for filename in ("registered_constants.json", "reported_results.json"):
        expected = hashlib.sha256((root / "paper" / filename).read_bytes()).hexdigest()
        if manifest.get("registry_sha256", {}).get(filename) != expected:
            raise ValueError(f"Display inventory requires review against changed {filename}")
    displays = manifest.get("displays")
    if not isinstance(displays, list) or any(not isinstance(row, dict) for row in displays):
        raise ValueError("Display inventory must contain object rows")
    ids = [row.get("id") for row in displays]
    if any(not isinstance(value, str) for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Display IDs must be unique strings")
    expected = expected_displays()
    if set(ids) != expected:
        raise ValueError(f"Incomplete display inventory: missing={sorted(expected - set(ids))}; unexpected={sorted(set(ids) - expected)}")
    targets = make_targets((root / "Makefile").read_text(encoding="utf-8"))
    record_sets = read_json(root / "records/specs.json")["sets"]
    groups = sources["table_groups"]
    for row in displays:
        role, kind, label = row.get("source"), row.get("kind"), row.get("label")
        if role not in documents or kind not in {"table", "figure"} or row["id"] != f"{role}:{label}":
            raise ValueError(f"Invalid display identity: {row['id']}")
        if not isinstance(label, str) or not label.startswith(kind.capitalize() + " "):
            raise ValueError(f"Display kind disagrees with label: {row['id']}")
        pages = row.get("pages")
        if (not isinstance(pages, list) or not pages
                or any(type(page) is not int or not 1 <= page <= documents[role]["pages"] for page in pages)
                or pages != sorted(set(pages))):
            raise ValueError(f"Invalid PDF pages: {row['id']}")
        if row.get("evidence_kind") not in KINDS or row.get("coverage") not in COVERAGE:
            raise ValueError(f"Unknown evidence or reconstruction scope: {row['id']}")
        for field in ("title", "required_evidence"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Missing {field}: {row['id']}")
        for field in ("implementation", "commands", "registry_groups", "record_sets", "notes"):
            values = row.get(field)
            if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f"Invalid {field}: {row['id']}")
        if not row["implementation"] or not row["commands"]:
            raise ValueError(f"Missing implementation or checking path: {row['id']}")
        for filename in row["implementation"]:
            _path(root, filename)
        for command in row["commands"]:
            match = re.fullmatch(r"make ([A-Za-z0-9_-]+)", command)
            if not match or match.group(1) not in targets:
                raise ValueError(f"Unknown checking command for {row['id']}: {command}")
        for group in row["registry_groups"]:
            if group not in groups:
                raise ValueError(f"Unknown numerical registry group for {row['id']}: {group}")
        for name in row["record_sets"]:
            if name not in record_sets:
                raise ValueError(f"Unknown record set for {row['id']}: {name}")
        for related in row.get("related_tables", []):
            if related not in expected or ":Table " not in related:
                raise ValueError(f"Unknown related table for {row['id']}: {related}")
    return sources


def _escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers: list[str], rows: list[list[object]]) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "|" + "---|" * len(headers),
                      *("| " + " | ".join(_escape(cell) for cell in row) + " |" for row in rows)])


def _source_table(sources: dict) -> str:
    return _table(["Document", "Pages", "Bytes", "SHA-256"], [
        [role, row["pages"], f"{row['byte_size']:,}", f"`{row['sha256']}`"]
        for role, row in sources["documents"].items()
    ])


def _code_release(state: dict) -> bool:
    return (state.get("scope") or {}).get("result_distribution") == "regenerate" and not state.get("official")


def _repository_release(state: dict) -> bool:
    return (state.get("scope") or {}).get("result_distribution") == "repository_release"


def _regeneration_note(state: dict) -> str | None:
    if _repository_release(state):
        return ("**Every numbered display has a generator** in `paper/build/displays/`: measured displays are rebuilt "
                "from the outputs published with the repository's release (`release/paper_artifacts.json`), protocol "
                "and analytical displays from the repository's configuration, registered constants and analytical "
                "code. `make paper` runs them all, writes each display's source data and compares every computed value "
                "with the printed one ([reconstruction guide](../docs/RECONSTRUCTION.md), "
                "[statement compliance](../STATEMENT_COMPLIANCE.md)).")
    if not _code_release(state):
        return None
    return ("**Results in this release:** regenerated, not distributed. `bash run_cross_morse.sh reproduce` executes "
            "the eight-stage lifecycle of every dataset on a Slurm GPU cluster; its report rebuilds the run-derived "
            "displays with `scripts/build_paper.py` and places every compared quantity beside the printed value "
            "([guide](../docs/HPC.md)). The record sets of the experiments outside the lifecycle are regenerated by "
            "their drivers ([Experiments](../docs/EXPERIMENTS.md)).")


def render_support(manifest: dict, sources: dict, state: dict) -> str:
    displays = manifest["displays"]
    counts = Counter(row["coverage"] for row in displays)
    lines = ["# Complete paper support inventory", "",
             "[Reported results](RESULTS.md) · [Manuscript review](MANUSCRIPT_REVIEW.md) · "
             "[Numerical claims ledger](CLAIMS.md) · [Artifact availability](../AVAILABILITY.md)", "",
             "This source-bound inventory covers **90 numbered displays**: two main tables, six main figures, "
             "62 supplementary tables and 20 supplementary figures. The authors identify the supplied PDFs as "
             "the completed measured study. The inventory names the available implementation and reconstruction "
             "path for every display, including those outside the 34 curated numerical registry groups.", "",
             (_regeneration_note(state) or
              "Registered experiment artifacts are identified in `runs/index.json` and `records/index.json`. "
              "The artifact catalog distinguishes supplied files from generated products and original experiments "
              "from subsequent reproductions. "
              "Fresh verification of the registered files is performed by `make review` and `make scientific-audit`."), "",
             "## What the checking paths cover", "",
             ("The first implementation of every display is its full-display generator in `paper/build/displays/`; "
              "`make paper` runs it on the published outputs and writes the complete table or figure with its source "
              "data. The other commands are related checks and builders: run builders reconstruct their documented "
              "columns, record builders emit set-level summaries. Mathematical and protocol checks do not establish "
              "the empirical origin of an input." if _repository_release(state) else
              "A command in the tables below is a relevant check or builder. It does not imply that the command "
              "recreates the entire printed display. Run builders reconstruct their documented columns; "
              "record builders emit set-level summaries; figure builders provide related summaries and selected panel data. The original "
              "analysis and artwork remain required where no full-display builder is declared. Mathematical "
              "and protocol checks do not establish the empirical origin of an input."), "",
             _table(["Checking or reconstruction scope", "Displays"],
                    [[label, counts[key]] for key, label in COVERAGE.items()]), "",
             "[display_manifest.json](display_manifest.json) retains each display's complete implementation "
             "list, related numerical groups, record sets, required evidence and qualifications. "
             "Its reviewed PDF and registry identities are checked before either report is rendered.", ""]
    for title, role, kind in (("Main paper", "main", None),
                              ("Supplementary tables", "supplementary", "table"),
                              ("Supplementary figures", "supplementary", "figure")):
        selected = [row for row in displays if row["source"] == role and (kind is None or row["kind"] == kind)]
        selected.sort(key=lambda row: (0 if row["kind"] == "table" else 1,
                                       int(re.search(r"\d+", row["label"]).group()), row["label"]))
        rows = []
        for row in selected:
            paths = "; ".join(f"[{Path(path).name}](../{path})" for path in row["implementation"])
            commands = "; ".join(f"`{command}`" for command in row["commands"])
            rows.append([f"**{row['label']}** — {row['title']}", ", ".join(map(str, row["pages"])),
                         KINDS[row["evidence_kind"]], COVERAGE[row["coverage"]], paths, commands])
        lines += [f"## {title}", "", _table(["Display", "PDF pages", "Content", "Scope", "Implementation / analysis", "Command"], rows), ""]
    lines += ["## Decisions and analysis qualifications", ""]
    for row in displays:
        if row["notes"]:
            lines.append(f"- **{row['label']} ({'main' if row['source'] == 'main' else 'SI'}):** " + " ".join(row["notes"]))
    lines += ["", "The [E1–E20 support contract](../docs/PAPER_SUPPORT.md) supplies the remaining "
              "experiment-level requirements, including source identity, pairing, supported actions, separated "
              "roles, exact evaluator, timing denominator and the distinction between the formal and executed algorithms.", "",
              "## Reviewed source identities", "", _source_table(sources), "",
              "The PDFs are identified as external source documents. Their bytes are verified when matching "
              "copies are supplied. The contextual review is bound in [transcription_check.json](transcription_check.json).", "",
              "Regenerate with `make paper-report`; check freshness with `make paper-report-check`. "
              "Changing a PDF or numerical registry requires explicit review and update of the inventory before regeneration.", ""]
    return "\n".join(lines)


def _number(value, digits: int = 5, *, signed: bool = False) -> str:
    if value is None:
        return "not applicable"
    return format(value, f"{'+' if signed else ''}.{digits}f")


def _interval(value, digits: int = 5) -> str:
    return "not applicable" if value is None else "[" + ", ".join(_number(x, digits) for x in value) + "]"


def render_results(reported: dict, registered: dict, sources: dict, state: dict) -> str:
    ledger = reported["main_table_1_policy_and_deployment_ledger"]
    costs = reported["main_table_2_gain_and_cost_ledger"]
    rows, cost_rows, resolved_rows = [], [], []
    for dataset in DATASETS:
        row, cost = ledger[dataset], costs[dataset]
        evaluated = " (evaluated arm)" if row.get("evaluation_status") == "evaluated_arm" else ""
        rows.append([dataset, f"{row['deployed_rule']}, E={row['deployed_E']}{evaluated}", row["loss_name"],
                     _number(row["loss_effect_nats"], signed=True), _interval(row["loss_effect_interval_nats"]),
                     f"{_number(row['official_gain_pp'], 2, signed=True)} {row['official_metric']}",
                     _interval(row["official_gain_interval_pp"], 2)])
        cost_label = f"{dataset} (evaluated {row['deployed_rule']} arm)" if row.get("evaluation_status") == "evaluated_arm" else dataset
        cost_rows.append([cost_label, f"{cost['C_qC']:g}", f"{cost['C_expected']:g}",
                          f"{cost['expected_overhead_pct']}%", f"{cost['C_assure']:g}",
                          "not applicable" if cost["C_099"] is None else f"{cost['C_099']:g}"])
        view = resolved_policy_view(reported, dataset, registered)
        regenerated = _code_release(state) or _repository_release(state)
        origin = ("Comparator identity; no new measurement" if view["comparator_identity"]
                  else ("Reported arm; a regenerated run records the full selector ledger" if regenerated
                        else "Reported arm; full selector ledger required") if row.get("selection_status")
                  else ("Reported policy outcome; a regenerated run records its own decision trail" if regenerated
                        else "Reported policy outcome; original run required for replay"))
        resolved_rows.append([dataset, view["deployed_rule"], _number(view["loss_effect_nats"], signed=True), origin])
    lines = ["# Results reported in the paper", "",
             "[Complete display support](PAPER_SUPPORT.md) · [Contextual manuscript review](MANUSCRIPT_REVIEW.md) · "
             "[Machine-readable results](RESULTS.csv) · [Artifact availability](../AVAILABILITY.md)", "",
             "The authors identify the supplied main paper and supplementary information as a completed measured "
             "study. The values below preserve their reported numerical values. They are generated from "
             "[reported_results.json](reported_results.json), with the original document identities recorded below. "
             "No benchmark observations are created by this report.", "",
             "## Main Table 1: reported evaluated outcomes", "",
             "Loss is the paired CE or mean multilabel BCE difference against the named dataset's open-loop "
             "comparator; negative values favor the evaluated rule. Metric changes and their intervals are "
             "percentage points. Peptides +0.58 AP percentage points is +0.0058 in native AP units. "
             "Brackets reproduce reported paired 95% intervals.", "",
             _table(["Dataset", "Reported rule", "Loss", "Loss difference, nats", "95% interval, nats", "Metric gain, pp", "95% interval, pp"], rows), "",
             "**Peptides:** the radius values are the reported evaluated-arm measurements. The printed Holm "
             "value 0.070 does not pass the registered 0.05 threshold, so the registered confirmation rule "
             "returns q_C; a different final policy requires its own corrected confirmation record. The "
             "evaluated-arm values remain visible here and separately named in the CSV. A regenerated run "
             "records its own confirmation decision in `confirm/decision.json`.", "",
             "## Main Table 2: reported work and analytical cost quantities", "",
             "The unit is one nominal full-depth execution. Expected cost is branch-weighted work; the printed "
             "overhead percentages retain the paper's rounding. Assurance cost and the modelled 0.99 quantile "
             "have different meanings. The latter is a cost-model calculation, not a measured maximum or "
             "an empirical timing percentile.", "",
             _table(["Dataset", "Comparator cost", "Expected cost", "Printed overhead", "Assurance cost", "Modelled C_0.99"], cost_rows), "",
             "## Registered confirmation interpretation", "",
             "This separate view applies the frozen confirmation interpretation already recorded in the "
             "repository. When it returns qC, zero paired effect follows from comparator identity; it is not "
             "a replacement measurement. " + ("A regenerated run records the complete decision trail of its own "
             "deployment: selection ledger, confirmation and admission records and the policy lock." if _code_release(state)
             else "Each published run carries the complete decision trail of its deployment: selection ledger, "
             "confirmation and admission records and the policy lock." if _repository_release(state)
             else "The full original decision record remains required for independent verification of each historical "
             "deployment."), "",
             _table(["Dataset", "Registered final rule", "Loss difference, nats", "Interpretation"], resolved_rows), "",
             "On Roman, Pascal and COCO the complete eligible candidate ledger decides between the reported "
             "MPC-RH choices and the radius preference of the representative rows in S12/S12c; a regenerated "
             "run records that ledger in `select/candidates.json` and `select/decision.json`. S14 reports one "
             "joint nine-test family; local per-run tests have a different scope. See [the source review](MANUSCRIPT_REVIEW.md) "
             "and [implementation notes](IMPLEMENTATION_NOTES.md).", ""] + (
             ["## Reproducing these results", "",
              "`make paper` downloads the measured outputs published with the repository's release, verifies every "
              "byte, rebuilds Tables 1–2 and every other numbered display with `paper/build/displays/` and places each "
              "value above beside the rebuilt one (`displays/compare/`). `bash run_cross_morse.sh reproduce` "
              "regenerates the runs independently on another Slurm GPU cluster ([HPC guide](../docs/HPC.md)). See the "
              "[reconstruction guide](../docs/RECONSTRUCTION.md) and [publication guide](../docs/PUBLICATION.md).", ""]
             if _repository_release(state) else
             ["## Regenerating these results", "",
              "This release includes synthetic example outputs. `bash run_cross_morse.sh reproduce` executes the "
              "registered dataset lifecycles on a Slurm GPU cluster and writes `REPRODUCTION_REPORT.md`, which rebuilds supported columns of Tables 1–2 from the regenerated "
              "runs and places each value of the tables above beside the regenerated one; `make compare RUN_ROOT=...` "
              "compares a single run. The [HPC guide](../docs/HPC.md) covers the site settings, the compute and the "
              "outputs. `make reconstruct` combines registered runs and auxiliary records in a hashed workspace; "
              "its report states the coverage of each numbered display. See the "
              "[reconstruction guide](../docs/RECONSTRUCTION.md) and [artifact catalog](../docs/ARTIFACT_CATALOG.md).", ""] if _code_release(state) else
             ["## Reproduction inputs", "",
              "The run and measurement-record indexes identify registered experiment artifacts. The source registry "
              "records what the paper reports; `make reconstruct` builds supported outputs from original "
              "observations after registration. `make scientific-audit` checks their evidence and decision records. "
              "The [registration guide](../docs/EVIDENCE_REGISTRATION.md) covers the already completed runs.", ""]) + [
             "## Source identities", "", _source_table(sources), "",
             "Regenerate with `make paper-report`; check with `make paper-report-check`. The companion CSV "
             "contains both the registered final-rule view and separately named reported-arm effects and costs.", ""]
    return "\n".join(lines)


def build(root: Path = ROOT) -> dict[str, str]:
    root = Path(root)
    manifest = read_json(root / "paper/display_manifest.json")
    sources = validate_manifest(root, manifest)
    reported = read_json(root / "paper/reported_results.json")
    registered = read_json(root / "paper/registered_constants.json")
    state = registration_state(root)
    return {
        "paper/PAPER_SUPPORT.md": render_support(manifest, sources, state),
        "paper/RESULTS.md": render_results(reported, registered, sources, state),
        "paper/RESULTS.csv": render_csv(reported, sources, registered),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        outputs = build(args.root)
        stale = []
        for relative, content in outputs.items():
            path = args.root / relative
            if args.check:
                if not path.is_file() or path.read_text(encoding="utf-8") != content:
                    stale.append(relative)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
        if stale:
            raise ValueError("Stale paper reports; run make paper-report: " + ", ".join(stale))
        print(json.dumps({"status": "PASS", "displays": len(expected_displays()),
                          "mode": "check" if args.check else "write",
                          "scope": "Current manuscript identities, reviewed display inventory and reported-value views; no empirical replay"}))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
