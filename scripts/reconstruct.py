#!/usr/bin/env python3
"""Build a traceable paper workspace from sealed runs and measurement records.

The display manifest is the complete inventory, not a claim that every display
has an exact generator. Each generated output records its actual relationship
to that inventory. Protocol checks and reported comparison values never become
measured results merely because this command ran successfully.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from scripts import build_paper, records as measurement_records
from scripts.artifact_inventory import verify_run_bytes
from scripts.ingest_run import check_present, registered_path, registration_descriptor
from scripts.paper_support_report import validate_manifest
from scripts.paper_artifacts import analysis_omissions, verify_analysis_run

SCHEMA = "cross-morse-reconstruction-1"
FILES_SCHEMA = "cross-morse-reconstruction-files-1"
TABLE_RELATIONS = {
    "T1": ["main:Table 1"], "T2": ["main:Table 2"],
    **{name: [f"supplementary:Table {name}"]
       for name in ("S4", "S12", "S12c", "S13", "S14", "S15", "S23", "S41")},
    "M1": [],
}
FIGURE_RELATIONS = {
    "F1": ["main:Figure 4"], "F2": ["main:Figure 5"],
    "F3": ["main:Figure 2"], "F4": [],
    "F5": ["supplementary:Table S41"], "F6": ["supplementary:Figure S9"],
}
COVERAGE = {"columns", "partial_panels", "full_numerical_display", "supporting_output"}
CAMPAIGN_ARGUMENTS = {"confirmation", "budget"}


class ReconstructionError(ValueError):
    """A requested input or output does not satisfy its declared contract."""


def read_json(path: Path):
    def reject(value):
        raise ReconstructionError(f"Non-finite JSON value in {path}: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_file(directory: Path, name: str) -> Path:
    """Only regular, contained files may enter an exported output manifest."""
    if (not isinstance(name, str) or not name or "\\" in name
            or PurePosixPath(name).is_absolute()
            or any(part in {"", ".", ".."} for part in name.split("/"))):
        raise ReconstructionError(f"Invalid artifact path: {name!r}")
    path = directory
    for part in PurePosixPath(name).parts:
        path = path / part
        if path.is_symlink():
            raise ReconstructionError(f"Symlink is not a reconstruction artifact: {name}")
    if not path.is_file() or not path.resolve().is_relative_to(directory.resolve()):
        raise ReconstructionError(f"Missing or escaping reconstruction artifact: {name}")
    return path


def source_inventory(root: Path) -> dict:
    paths = [root / name for name in ("paper/display_manifest.json", "paper/reported_results.json",
             "paper/printed_display_values.json", "paper/sources.json", "paper/registered_constants.json", "records/specs.json")]
    for directory in ("scripts", "paper/build", "cross_morse", "cm_campaigns", "cm_experiments", "cm_theory"):
        paths.extend((root / directory).rglob("*.py"))
    paths.extend((root / "paper/protocol").glob("*.json"))
    paths.extend((root / "configs").rglob("*.json"))
    paths.extend(root.glob("run_cross_morse*.py"))
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(set(paths)) if p.is_file()}


def _run_inputs(runs_root: Path | None, index: Path | None) -> tuple[list, list, list]:
    found, inputs, unavailable = [], [], []
    if runs_root is not None:
        path = runs_root.resolve()
        if not path.is_dir():
            raise ReconstructionError(f"Run directory does not exist: {path}")
        if build_paper.is_run(path) or build_paper.is_campaign(path):
            found = [(path.name, path)]
        else:
            found = build_paper.discover(path)
    elif index is not None:
        data = read_json(index)
        if data.get("schema") != "cross-morse-runs-index-1" or not isinstance(data.get("runs"), list):
            raise ReconstructionError(f"Invalid run index: {index}")
        inputs.append({"kind": "run_index", "path": str(index.resolve()), "sha256": sha256(index)})
        seen = set()
        for row in data["runs"]:
            if not isinstance(row, dict) or not isinstance(row.get("path"), str):
                raise ReconstructionError("Every run-index entry must name its run path")
            descriptor = registration_descriptor(row)
            path = registered_path(row, index.resolve().parent)
            if str(path) in seen:
                raise ReconstructionError(f"Duplicate run-index path: {path}")
            seen.add(str(path))
            if not path.exists():
                unavailable.append({"path": str(path), "reason": "Registered run bytes must be materialized first"})
            else:
                check_present(row, path)
                inputs.append({"kind": "run_registration", "path": str(path), "registration": descriptor,
                               "verification": "local archived inventory and sealed identities match registration"})
                found.append((row.get("stem") or path.name, path))
    for _, path in found:
        children = ([path] if build_paper.is_run(path) else
                    sorted(p for p in path.glob("split_*") if build_paper.is_run(p)))
        datasets, profiles, split_ids = set(), set(), []
        omissions = analysis_omissions(path)
        for child in children:
            if omissions is None:
                _, config, identities, _, _, _, _, _ = verify_run_bytes(child)
                verification = "all sealed bytes and eight-stage chain; no checkpoint deserialization"
            else:
                prefix = "" if child == path else child.relative_to(path).as_posix() + "/"
                local = {rel[len(prefix):]: dict(d, path=rel[len(prefix):])
                         for rel, d in omissions.items() if rel.startswith(prefix)}
                identities, info = verify_analysis_run(child, local)
                config = read_json(child / "config.json")
                verification = (f"analysis tier: eight-stage chain and every present sealed byte; "
                                f"{info['omitted']} checkpoint/prediction files omitted with their sealed identities "
                                f"({info['source_resolution']})")
            state = read_json(child / "run_state.json")
            if (state.get("profile") != config.get("profile")
                    or state.get("scientific_run") != config.get("scientific_run")):
                raise ReconstructionError("Run bookkeeping data scope differs from the sealed configuration")
            final = read_json(child / "final/results_summary.json")
            if (state.get("task", {}).get("dataset") != final.get("benchmark_dataset")
                    or state.get("scientific_run") != final.get("scientific_run")):
                raise ReconstructionError("Run bookkeeping dataset scope differs from the sealed final summary")
            datasets.add(final["benchmark_dataset"])
            profiles.add(state.get("profile"))
            split_ids.append(config.get("data", {}).get("split_index"))
            inputs.append({"kind": "sealed_run", "path": str(child), "identities": identities,
                           "stage_manifests": {
                               p.name: sha256(p) for p in sorted((child / "provenance/stages").glob("*.json"))},
                           "verification": verification,
                           "profile": config.get("profile"), "scientific_run": config.get("scientific_run"),
                           "historical_relationship": "not inferred from run completion"})
        if len(datasets) != 1 or len(profiles) != 1:
            raise ReconstructionError(f"A campaign must use one dataset and profile: {path}")
        if not build_paper.is_run(path) and datasets != {"Roman-empire"}:
            raise ReconstructionError("Only Roman-empire has a registered split-campaign aggregation; supply other datasets as individual runs")
        if not build_paper.is_run(path) and datasets == {"Roman-empire"}:
            if (len(split_ids) != 10 or any(type(index) is not int for index in split_ids)
                    or set(split_ids) != set(range(10))):
                raise ReconstructionError("Roman-empire campaign requires each of the ten official split IDs exactly once")
    return found, inputs, unavailable


def _record_inputs(index: Path | None, specs_path: Path) -> tuple[list, dict]:
    if index is None:
        return [], {}
    index = index.resolve()
    data = read_json(index)
    if data.get("schema") != "cross-morse-records-index-1" or not isinstance(data.get("sets"), list):
        raise ReconstructionError(f"Invalid measurement-record index: {index}")
    specs = measurement_records.load_specs(index.parent, specs_path)
    for path in index.parent.rglob("*"):
        if path.is_symlink():
            raise ReconstructionError(f"Measurement-record input contains a symlink: {path}")
    entries = [measurement_records.verify_set(p, specs["sets"][p.name], specs)
               for p in measurement_records.deposited_sets(index.parent, specs)]
    indexed = data["sets"]
    if (any(not isinstance(row, dict) or row.get("verified") is not True for row in indexed)
            or len(indexed) != len(entries)
            or {row.get("set"): row.get("manifest_sha256") for row in indexed}
            != {row["set"]: row["manifest_sha256"] for row in entries}):
        raise ReconstructionError("Measurement-record index differs from the verified deposited sets")
    return entries, {"kind": "measurement_records", "path": str(index), "sha256": sha256(index),
                     "sets": entries, "verification": "manifest hashes and summary-to-raw-value traceability"}


def _benchmark_dataset(path: Path) -> str:
    """The benchmark dataset of one discovered run, or of a split campaign's first split (sealed final summary)."""
    child = path if build_paper.is_run(path) else next(p for p in sorted(path.glob("split_*")) if build_paper.is_run(p))
    return read_json(child / "final/results_summary.json")["benchmark_dataset"]


def display_context(root: Path, found: list, collected: list, record_entries: list, records_index,
                    campaigns: dict, synthetic: bool):
    """Inputs of the display generators: verified runs, verified record sets and campaign exports.

    ``collected`` is in the paper's dataset order (build_paper.collect), ``found`` in discovery order; each run
    is paired with its own dataset, whatever its directory is called."""
    from paper.build.displays.framework import DisplayContext, RecordSetView, RunView
    runs = {}
    datasets = {record["dataset"] for record in collected}
    for _, path in found:
        dataset = _benchmark_dataset(path)
        if dataset not in datasets or dataset in runs:
            raise ReconstructionError(f"Run {path} does not pair with exactly one collected dataset record")
        splits = [RunView(dataset, p) for p in sorted(path.glob("split_*")) if build_paper.is_run(p)]
        runs[dataset] = RunView(dataset, path, splits=splits if not build_paper.is_run(path) else ())
    records = {}
    if record_entries:
        base = Path(records_index).resolve().parent
        specs = measurement_records.load_specs(base, root / "records/specs.json")
        for entry in record_entries:
            spec = specs["sets"][entry["set"]]
            rows = measurement_records.read_summary(base / entry["set"], spec, specs["summary_columns"])
            for row in rows:
                row["value"] = float(row["value"])
            records[entry["set"]] = RecordSetView(entry["set"], base / entry["set"], spec, rows,
                                                  synthetic=bool(entry.get("synthetic")))
    return DisplayContext(root=root, runs=runs, records=records,
                          campaigns={k: Path(v).resolve() for k, v in campaigns.items() if v is not None},
                          synthetic=synthetic)


def add_display_outputs(outputs: list, stage: Path, result: dict, ids: set, synthetic: bool) -> None:
    for row in result["displays"]:
        if row["status"] != "built":
            continue
        artifact_inputs = [need for need in row["needs"] if need != "static"]
        kind = ("repository_sources" if not artifact_inputs else
                "synthetic_examples" if synthetic else "verified_experiment_artifacts")
        complete = row.get("content_complete") is True and row.get("comparison_complete") is True
        coverage = ("full_numerical_display" if complete else
                    "columns" if ":Table " in row["id"] else "partial_panels")
        add_output(outputs, stage, {"id": "display:" + row["id"],
                   "files": ["displays/" + name for name in row["files"]], "display_ids": [row["id"]],
                   "coverage": coverage, "input_kind": kind, "scope": row["scope"],
                   "comparison": row["comparison"],
                   "content_complete": row.get("content_complete", False),
                   "comparison_complete": row.get("comparison_complete", False),
                   "comparison_coverage": row.get("comparison_coverage", {}),
                   "missing_content": row.get("missing_content", []),
                   "evidence_scope": row.get("evidence_scope", []),
                   "note": "Display rendered from declared inputs. Complete coverage requires every required "
                           "cell and a reviewed comparison target; see displays/compare/<stem>.csv and the "
                           "content/comparison metadata for missing values and scope."}, ids)


def add_output(outputs: list, directory: Path, entry: dict, display_ids: set) -> None:
    if (not isinstance(entry.get("id"), str) or not entry["id"]
            or entry["id"] in {r["id"] for r in outputs}):
        raise ReconstructionError(f"Invalid or duplicate generated output ID: {entry.get('id')}")
    if entry.get("coverage") not in COVERAGE:
        raise ReconstructionError(f"Unknown output coverage: {entry.get('coverage')}")
    if not isinstance(entry.get("display_ids"), list) or not set(entry["display_ids"]) <= display_ids:
        raise ReconstructionError(f"Output references an unknown manuscript display: {entry['id']}")
    if not isinstance(entry.get("files"), list) or not entry["files"]:
        raise ReconstructionError(f"Output has no generated files: {entry['id']}")
    entry = dict(entry)
    entry["file_sha256"] = {name: sha256(relative_file(directory, name)) for name in entry["files"]}
    outputs.append(entry)


def display_status(manifest: dict, outputs: list) -> list[dict]:
    rows = []
    for display in manifest["displays"]:
        related = [entry for entry in outputs if display["id"] in entry["display_ids"]]
        full = any(entry["coverage"] == "full_numerical_display"
                   and entry.get("input_kind") != "synthetic_examples" for entry in related)
        rows.append({"id": display["id"], "title": display["title"], "pages": display["pages"],
                     "state": "generated" if related else "inputs_or_generator_required",
                     "full_numerical_display": full,
                     "generated_outputs": [entry["id"] for entry in related],
                     "coverage": sorted({entry["coverage"] for entry in related}),
                     "available_support": display["coverage"],
                     "required_evidence": display["required_evidence"],
                     "support_commands": display["commands"]})
    return rows


def render_report(result: dict) -> str:
    lines = ["# Paper reconstruction report", "",
             f"Build: **{result['build_status']}** · Label: **{result['label']}**", "",
             "This workspace records generated numerical outputs, their input identities and their relationship "
             "to every numbered display in the registered paper and supplement. A successful build means the "
             "supported outputs were generated. The display inventory below states the coverage of each output.", "",
             "The source registries supply comparison targets and display identities. They are not substituted "
             "for missing experimental observations. Original experiment identity and agreement with the paper "
             "require the separate result bindings and scientific evidence audit.", ""]
    if result["synthetic_examples"]:
        lines += ["**Synthetic software examples are included in this build. Their values demonstrate the "
                  "workflow and are not the paper's benchmark observations.**", ""]
    lines += ["## Generated outputs", "", "| Output | Coverage | Displays | Files |",
              "|---|---|---|---|"]
    for entry in result["outputs"]:
        links = " · ".join(f"[{Path(name).name}]({name})" for name in entry["files"])
        lines.append(f"| {entry['id']} | {entry['coverage'].replace('_', ' ')} | "
                     f"{', '.join(entry['display_ids']) or 'Supporting artifact'} | {links} |")
    if not result["outputs"]:
        lines += ["| Input materialization | Input-dependent build | See the display inventory | "
                  "Provide sealed run directories or registered measurement records |"]
    generators = result.get("display_generators")
    if generators:
        counts = generators["comparison_with_printed_values"]
        lines += ["", "## Display generators and comparison with the paper", "",
                  f"{generators['built']} of {len(result['displays'])} numbered displays were rendered; "
                  f"{sum(d['full_numerical_display'] for d in result['displays'])} have complete numerical coverage; "
                  f"{generators['awaiting_inputs']} await their experiment inputs; "
                  f"{generators['without_generator']} have no generator; "
                  f"{len(generators.get('errors') or {})} generator errors.", "",
                  f"Printed values compared: **{counts['exact']} exact** at the printed precision, "
                  f"**{counts['differs']} differ**, {counts['not_comparable']} not comparable, "
                  f"{counts['no_reference']} without a transcribed reference, {counts.get('missing', 0)} missing. "
                  "Each display's `displays/compare/<stem>.csv` lists every comparison.", "",
                  "| Display | Generator | Exact | Differs | Awaiting |", "|---|---|---|---|---|"]
        for display in result["displays"]:
            comp = display.get("comparison") or {}
            awaiting = "; ".join(display.get("awaiting") or []) or ""
            lines.append(f"| {display['id']} | {display.get('generator', 'no_generator').replace('_', ' ')} | "
                         f"{comp.get('exact', '')} | {comp.get('differs', '')} | {awaiting} |")
    lines += ["", "## Display inventory", "",
              "A protocol or arithmetic check remains a support command. It does not count as a regenerated "
              "table or figure. Column and partial-panel builders retain those scopes even when all their "
              "own output files have been written.", "", "| Display | Generated support | Coverage |",
              "|---|---|---|"]
    for display in result["displays"]:
        support = ", ".join(display["generated_outputs"]) or "Input or dedicated generator required"
        coverage = ", ".join(display["coverage"]).replace("_", " ") or display["available_support"].replace("_", " ")
        lines.append(f"| {display['id']} | {support} | {coverage} |")
    if result["unavailable_inputs"]:
        lines += ["", "## Input materialization", ""]
        for row in result["unavailable_inputs"]:
            lines.append(f"- `{row['path']}`: {row['reason']}.")
    lines += ["", "## Reuse and verification", "",
              "`reconstruction.json` holds input and software identities, every output relationship, "
              "and the complete display status. `FILES.json` hashes the exported workspace. "
              "Run `python -B scripts/reconstruct.py --verify /path/to/workspace` from the repository "
              "to detect missing, extra or changed output files. `--require-all-displays` returns a nonzero "
              "status unless every display has complete numerical output from non-synthetic inputs.", ""]
    return "\n".join(lines)


def write_file_manifest(directory: Path) -> None:
    names = sorted(p.relative_to(directory).as_posix() for p in directory.rglob("*")
                   if p.is_file() and p != directory / "FILES.json")
    files = {name: {"sha256": sha256(relative_file(directory, name)),
                    "bytes": relative_file(directory, name).stat().st_size} for name in names}
    write_json(directory / "FILES.json", {"schema": FILES_SCHEMA, "files": files})


def verify_workspace(directory: Path) -> dict:
    directory = Path(directory).resolve()
    if any(path.is_symlink() for path in directory.rglob("*")):
        raise ReconstructionError("A reconstruction workspace may not contain symlinks")
    manifest = read_json(relative_file(directory, "FILES.json"))
    if manifest.get("schema") != FILES_SCHEMA or not isinstance(manifest.get("files"), dict):
        raise ReconstructionError("Invalid reconstruction file manifest")
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if actual != set(manifest["files"]) | {"FILES.json"}:
        raise ReconstructionError("Reconstruction file set differs from FILES.json")
    for name, expected in manifest["files"].items():
        path = relative_file(directory, name)
        if not isinstance(expected, dict) or expected != {"sha256": sha256(path), "bytes": path.stat().st_size}:
            raise ReconstructionError(f"Reconstruction artifact changed: {name}")
    result = read_json(relative_file(directory, "reconstruction.json"))
    if result.get("schema") != SCHEMA:
        raise ReconstructionError("Invalid reconstruction report schema")
    return result


def reconstruct(output_dir: Path, *, runs_root=None, index=None, records_index=None,
                confirmation_campaign=None, budget_campaign=None, campaigns=None,
                label="reconstruction", root=ROOT, demonstration=False, scope_note: str | None = None) -> dict:
    """Build in a new staging directory, then atomically publish a managed workspace.

    ``demonstration`` (scripts/reconstruct_paper.py --demo) builds the supplementary displays of the CPU
    demonstration from its fixture-derived record sets and campaign, labelled as such; a supplementary input that
    still cannot be built is then reported among the generator errors instead of stopping the reconstruction.
    ``scope_note`` opens the report (the demonstration's scope); it is written before the file manifest, so the
    workspace verifies as built."""
    if scope_note is not None and (not isinstance(scope_note, str) or not scope_note.strip()):
        raise ReconstructionError("A scope note must be nonempty text")
    root, output_dir = Path(root).resolve(), Path(output_dir).absolute()
    if not isinstance(label, str) or not label.strip() or "\n" in label or "|" in label:
        raise ReconstructionError("Build label must be nonempty text on one line, without a table separator")
    if runs_root is not None and index is not None:
        raise ReconstructionError("Choose a run directory or a registered run index")
    # Never replace source or input directories, even if a forged marker exists.
    campaigns = {**(campaigns or {}), **{name: path for name, path in
                                         (("confirmation", confirmation_campaign), ("budget", budget_campaign))
                                         if path is not None}}
    protected = [root] + [Path(p).resolve() for p in
                         (runs_root, index, records_index, *campaigns.values()) if p is not None]
    if any(p == output_dir.resolve() or p.is_relative_to(output_dir.resolve()) for p in protected):
        raise ReconstructionError("The output directory would replace source or input files")
    if runs_root is not None and output_dir.resolve().is_relative_to(Path(runs_root).resolve()):
        raise ReconstructionError("Keep the reconstruction workspace outside the input run directory")
    if any(path.is_symlink() for path in (output_dir, *output_dir.parents)):
        raise ReconstructionError("Output directory must not traverse a symlink")
    input_directories = [Path(p).resolve() for p in campaigns.values() if p is not None]
    if records_index is not None:
        input_directories.append(Path(records_index).resolve().parent)
    if any(output_dir.resolve().is_relative_to(path) for path in input_directories):
        raise ReconstructionError("Keep the reconstruction workspace outside the input evidence directories")
    if output_dir.exists() and (not output_dir.is_dir() or not (output_dir / "reconstruction.json").is_file()):
        raise ReconstructionError("Existing output directory is not a managed reconstruction workspace")
    if output_dir.exists():
        verify_workspace(output_dir)
    manifest = read_json(root / "paper/display_manifest.json")
    validate_manifest(root, manifest)
    ids = {row["id"] for row in manifest["displays"]}
    found, inputs, unavailable = _run_inputs(Path(runs_root) if runs_root else None, Path(index) if index else None)
    if any(output_dir.resolve().is_relative_to(path.resolve()) for _, path in found):
        raise ReconstructionError("Keep the reconstruction workspace outside the registered run directory")
    record_entries, record_input = _record_inputs(Path(records_index) if records_index else None,
                                                 root / "records/specs.json")
    if record_input:
        inputs.append(record_input)
    collected = build_paper.collect(found) if found else []
    datasets = [record["dataset"] for record in collected]
    if len(set(datasets)) != len(datasets):
        raise ReconstructionError("Multiple runs for one dataset would overwrite its evidence; use one declared campaign")
    synthetic = any(not record.get("scientific_run") for record in collected) or any(
        entry.get("synthetic") for entry in record_entries)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cm-reconstruction-", dir=output_dir.parent) as temporary:
        stage = Path(temporary) / "workspace"
        stage.mkdir()
        outputs = []
        if collected:
            summary = build_paper.write_all(collected, stage / "runs", label, figures=True)
            input_kind = "synthetic_examples" if summary["synthetic_examples"] else "sealed_run_artifacts"
            for family, relations, coverage in (("tables", TABLE_RELATIONS, "columns"),
                                                 ("figures", FIGURE_RELATIONS, "partial_panels")):
                for name, paths in summary[family].items():
                    add_output(outputs, stage, {"id": f"run:{name}",
                               "files": [f"runs/{family}/{path}" for path in paths.values()],
                               "display_ids": relations[name], "coverage": coverage if relations[name] else "supporting_output",
                               "input_kind": input_kind, "datasets": datasets,
                               "note": "Run-derived columns or related plots; unmeasured fields remain empty."}, ids)
        if record_entries:
            with contextlib.redirect_stdout(io.StringIO()):
                measurement_records.cmd_tables(Path(records_index).resolve().parent, stage / "records",
                                                root / "records/specs.json")
            for entry in record_entries:
                displays = [row["id"] for row in manifest["displays"] if row["kind"] == "table"
                            and entry["set"] in row["record_sets"]]
                add_output(outputs, stage, {"id": "record:" + entry["set"],
                           "files": [f"records/{entry['set']}.{suffix}" for suffix in ("md", "csv", "tex")],
                           "display_ids": displays, "coverage": "columns",
                           "input_kind": "synthetic_examples" if entry["synthetic"] else "traced_measurement_records",
                           "note": "Set-level table; exact manuscript field and layout coverage is not inferred."}, ids)
        supplementary, supplementary_errors = None, {}
        if confirmation_campaign or budget_campaign or any(entry["set"] in {"factorials", "ensemble_grid"}
                                                           and (demonstration or not entry["synthetic"])
                                                           for entry in record_entries):
            from paper.build.supplement import build_supplement
            supplementary = build_supplement(stage / "supplement", records_index=records_index,
                                             confirmation_campaign=confirmation_campaign,
                                             budget_campaign=budget_campaign, manuscript_root=root,
                                             demonstration=demonstration)
            if supplementary.get("errors") and not demonstration:
                raise ReconstructionError("Supplementary build failed: " + str(supplementary["errors"]))
            supplementary_errors = {f"supplement:{row['input']}": row["error"] for row in supplementary.get("errors", [])}
            for entry in supplementary.get("outputs", []):
                entry = {**entry, "files": ["supplement/" + name for name in entry["files"]]}
                add_output(outputs, stage, entry, ids)
        from paper.build.displays.framework import build_displays
        context = display_context(root, found, collected, record_entries, records_index, campaigns, synthetic)
        generated = build_displays(context, stage / "displays")
        add_display_outputs(outputs, stage, generated, ids, synthetic)
        displays = display_status(manifest, outputs)
        by_id = {row["id"]: row for row in generated["displays"]}
        for row in displays:
            built = by_id.get(row["id"], {})
            row["generator"] = built.get("status", "no_generator")
            row["comparison"] = built.get("comparison")
            row["awaiting"] = built.get("missing", [])
            row["content_complete"] = built.get("content_complete", False)
            row["missing_content"] = built.get("missing_content", [])
            row["comparison_complete"] = built.get("comparison_complete", False)
            row["comparison_coverage"] = built.get("comparison_coverage", {})
            row["evidence_scope"] = built.get("evidence_scope", [])
        comparison = {state: sum((row.get("comparison") or {}).get(state, 0) for row in generated["displays"])
                      for state in ("exact", "differs", "not_comparable", "no_reference", "missing")}
        # Displays generated from the repository's own sources (protocol, analytical) need no
        # experiment input; they never turn a workspace without inputs into a reconstruction.
        experimental = [o for o in outputs if o.get("input_kind") != "repository_sources"]
        result = {"schema": SCHEMA, "label": label,
                  "build_status": "SUPPORTED_OUTPUTS_BUILT" if experimental else "INPUTS_REQUIRED",
                  "all_displays_numerically_reconstructed": all(row["full_numerical_display"] for row in displays),
                  "historical_experiment_identity_verified": False,
                  "synthetic_examples": synthetic, "inputs": inputs, "unavailable_inputs": unavailable,
                  "documents": manifest["documents"], "software_sources": source_inventory(root),
                  "environment": {"python": platform.python_version(), "platform": platform.system(),
                                  **{name: importlib.metadata.version(name) for name in ("numpy", "matplotlib")}},
                  "outputs": outputs, "displays": displays, "supplementary_build": supplementary,
                  "display_generators": {"registered": generated["generators"], "built": generated["built"],
                                         "awaiting_inputs": sum(1 for r in generated["displays"]
                                                                if r["status"] == "awaiting_inputs"),
                                         "without_generator": sum(1 for r in generated["displays"]
                                                                  if r["status"] == "no_generator"),
                                         "errors": {**{r["id"]: r["error"] for r in generated["displays"]
                                                       if r["status"] == "error"}, **supplementary_errors},
                                         "module_errors": generated.get("module_errors", {}),
                                         "comparison_sources": generated.get("comparison_sources", []),
                                         "comparison_coverage_complete": all(d["comparison_complete"] for d in displays),
                                         "display_content_incomplete": [d["id"] for d in displays if not d["content_complete"]],
                                         "comparison_with_printed_values": comparison}}
        write_json(stage / "reconstruction.json", result)
        report = render_report(result)
        if scope_note is not None:
            report = f"> **Demonstration.** {scope_note.strip()}\n\n" + report
        (stage / "RECONSTRUCTION_REPORT.md").write_text(report, encoding="utf-8")
        write_file_manifest(stage)
        verify_workspace(stage)
        backup = Path(temporary) / "previous"
        if output_dir.exists():
            os.replace(output_dir, backup)
        try:
            os.replace(stage, output_dir)
        except BaseException:
            if backup.exists():
                os.replace(backup, output_dir)
            raise
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--runs-root", type=Path)
    source.add_argument("--index", type=Path)
    parser.add_argument("--records-index", type=Path)
    parser.add_argument("--confirmation-campaign", type=Path)
    parser.add_argument("--budget-campaign", type=Path)
    parser.add_argument("--campaign", action="append", default=[], metavar="NAME=PATH",
                        help="an additional campaign export a display declares (repeatable)")
    parser.add_argument("--label", default="reconstruction")
    parser.add_argument("--require-all-displays", action="store_true")
    parser.add_argument("--verify", type=Path, help="verify an existing workspace without rebuilding it")
    args = parser.parse_args(argv)
    try:
        if args.verify:
            if any((args.output_dir, args.runs_root, args.index, args.records_index,
                    args.confirmation_campaign, args.budget_campaign)):
                raise ReconstructionError("Verification cannot be combined with build inputs")
            result = verify_workspace(args.verify)
            print(f"Reconstruction workspace bytes verified: {args.verify}")
        else:
            if args.output_dir is None:
                parser.error("--output-dir is required for a build")
            extra = {}
            for item in args.campaign:
                name, _, path = item.partition("=")
                if not name or not path:
                    raise ReconstructionError(f"--campaign expects NAME=PATH, got {item!r}")
                extra[name] = Path(path)
            result = reconstruct(args.output_dir, runs_root=args.runs_root, index=args.index,
                                 records_index=args.records_index, confirmation_campaign=args.confirmation_campaign,
                                 budget_campaign=args.budget_campaign, campaigns=extra, label=args.label)
            print(f"{result['build_status']}: {args.output_dir / 'RECONSTRUCTION_REPORT.md'}")
        if args.require_all_displays and not result["all_displays_numerically_reconstructed"]:
            print("The workspace does not reconstruct every numerical display; see the per-display coverage.", file=sys.stderr)
            return 1
        return 0 if args.verify or (result["outputs"] and not result["unavailable_inputs"]) else 1
    except (OSError, ValueError, ImportError, TypeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
