#!/usr/bin/env python3
"""Build and verify the typed inventory behind the availability statement.

The catalog distinguishes shipped files, reported numerical references,
synthetic examples, documented products of a new execution, provider datasets
and registered experimental archives. It never runs a producer, downloads data,
fills a result cell or infers historical origin from a matching measurement.

    python -B scripts/artifact_catalog.py
    python -B scripts/artifact_catalog.py --check
    python -B scripts/artifact_catalog.py --output-dir /abs/catalog-review

Registered local payloads are rechecked with the existing archive/record
validators. A deposit reference with an absent payload requires the committed
registration manifest; it remains a reference until public retrieval is checked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import sys
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

SPEC = "release/artifact_catalog_spec.json"
CATALOG = "release/artifact_catalog.json"
GUIDE = "docs/ARTIFACT_CATALOG.md"
SCHEMA = "cross-morse-artifact-catalog-1"
SPEC_SCHEMA = "cross-morse-artifact-catalog-spec-1"
CLAIMS = {
    "experiment_code", "configurations", "source_data", "split_seed_manifests",
    "environments", "checkpoints", "predictions", "propensity_logs",
    "statistical_analysis", "table_figure_generators",
}
ASSET_CLASSES = {
    "implementation", "configuration", "design_manifest", "environment_specification",
    "reported_reference", "synthetic_output", "analysis_builder", "documentation",
}
CLASS_LABELS = {
    "implementation": "Implementation",
    "configuration": "Configuration",
    "design_manifest": "Registered design",
    "environment_specification": "Environment specification",
    "reported_reference": "Reported numerical reference",
    "synthetic_output": "Synthetic example output",
    "analysis_builder": "Analysis or display builder",
    "documentation": "Contract documentation",
}
IDENTITY_KEYS = ("source_sha256", "config_sha256", "unit_sha256")
RELATIONSHIPS = {
    "exact_result_producing_run": "declared_original_benchmark",
    "refactor_with_prediction_equivalence": "declared_prediction_equivalent_revision",
    "post_result_revision": "declared_new_reproduction",
}
HASH = re.compile(r"^[0-9a-f]{64}$")
ROLES = ("fit", "explore", "select", "confirm", "compare", "admit", "test")


class CatalogError(ValueError):
    """A catalog declaration does not match the available files or contracts."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CatalogError(message)


def read_json(path: Path):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, f"Duplicate JSON key in {path}: {key}")
            result[key] = value
        return result
    def reject(value):
        raise CatalogError(f"Nonfinite JSON value in {path}: {value}")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=reject)


def relative(value: object, *, pattern: bool = False) -> str:
    require(isinstance(value, str) and bool(value) and "\\" not in value
            and not any(ord(c) < 32 for c in value), f"Invalid relative path: {value!r}")
    path = PurePosixPath(value)
    require(not path.is_absolute() and ".." not in path.parts and path.as_posix() == value
            and value != ".", f"Invalid relative path: {value!r}")
    require(pattern or not any(c in value for c in "*?[]{}"), f"Unexpected path pattern: {value}")
    return value


def safe_path(root: Path, value: object, *, kind: str = "file", exists: bool = True) -> Path:
    value = relative(value)
    path = root / value
    for parent in (path, *path.parents):
        if parent == root:
            break
        require(not parent.is_symlink(), f"Catalog path traverses a symlink: {value}")
    require(path.resolve().is_relative_to(root.resolve()), f"Catalog path escapes the repository: {value}")
    if exists:
        require(path.is_file() if kind == "file" else path.is_dir(), f"Catalog {kind} is absent: {value}")
    return path


def fingerprint(path: Path, root: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return {"path": path.relative_to(root).as_posix(), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def expand(root: Path, pattern: str) -> list[Path]:
    relative(pattern, pattern=True)
    paths = sorted(root.glob(pattern))
    require(bool(paths), f"Bundled file pattern matches nothing: {pattern}")
    for path in paths:
        safe_path(root, path.relative_to(root).as_posix())
    return paths


def string_list(value: object, where: str, *, nonempty: bool = False) -> list[str]:
    require(isinstance(value, list) and all(isinstance(x, str) and x.strip() for x in value)
            and len(value) == len(set(value)) and (not nonempty or bool(value)), f"Invalid {where}")
    return value


def validate_spec(root: Path, spec: dict) -> None:
    require(isinstance(spec, dict) and spec.get("schema") == SPEC_SCHEMA, "Unsupported artifact catalog specification")
    entries, commands = spec.get("entries"), spec.get("commands")
    require(isinstance(entries, list) and all(isinstance(row, dict) for row in entries), "Catalog entries must be objects")
    ids = [row.get("id") for row in entries]
    require(all(isinstance(value, str) for value in ids) and len(ids) == len(set(ids)) and set(ids) == CLAIMS,
            "The catalog must cover every availability-statement item exactly once")
    require(isinstance(commands, dict) and bool(commands), "Catalog commands are missing")
    for key, command in commands.items():
        require(isinstance(command, dict) and command.get("mode") in {"check", "execute", "build", "fetch", "register"},
                f"Invalid command mode: {key}")
        argv = command.get("argv")
        require(isinstance(argv, list) and bool(argv) and all(isinstance(arg, str) and arg
                and not any(ord(c) < 32 for c in arg) for arg in argv), f"Invalid command argv: {key}")
        require(isinstance(command.get("scope"), str) and command["scope"].strip(), f"Missing command scope: {key}")
        for path in string_list(command.get("entrypoints"), f"{key} entrypoints", nonempty=True):
            safe_path(root, path)
        if argv[0] == "make":
            from scripts.check_reviewer_faq import make_targets
            targets = make_targets(safe_path(root, "Makefile").read_text(encoding="utf-8"))
            requested = [arg for arg in argv[1:] if "=" not in arg and not arg.startswith("-")]
            require(bool(requested) and all(target in targets for target in requested), f"Unknown Make target in {key}")
        elif argv[0] in {"python", "bash"}:
            entrypoint = next((arg for arg in argv[1:] if not arg.startswith("-")), None)
            require(entrypoint in command["entrypoints"], f"Command entry point is not declared: {key}")
        else:
            raise CatalogError(f"Unsupported producer command: {key}")
        string_list(command.get("requires"), f"{key} requirements")
    for row in entries:
        key = row["id"]
        require(all(isinstance(row.get(field), str) and row[field].strip() for field in ("title", "scope")),
                f"Missing title/scope for {key}")
        bundled = row.get("bundled")
        require(isinstance(bundled, list) and bool(bundled), f"Missing bundled support for {key}")
        for asset in bundled:
            require(isinstance(asset, dict) and asset.get("artifact_class") in ASSET_CLASSES, f"Invalid asset class: {key}")
            for pattern in string_list(asset.get("patterns"), f"{key} bundled patterns", nonempty=True):
                relative(pattern, pattern=True)
                if pattern.startswith(("examples/runs/", "examples/records/")):
                    require(asset["artifact_class"] == "synthetic_output", "Example artifacts must remain synthetic")
                if pattern in {"paper/reported_results.json", "paper/registered_constants.json"}:
                    require(asset["artifact_class"] == "reported_reference", "The manuscript registry is a reported reference")
                if asset["artifact_class"] == "synthetic_output":
                    require(pattern.startswith(("examples/runs/", "examples/records/")), "Synthetic outputs must identify bundled examples")
        generated = row.get("generated")
        require(isinstance(generated, list), f"Invalid generated contracts: {key}")
        for output in generated:
            require(isinstance(output, dict) and output.get("root") in {"run", "suite", "record_set", "report", "checkout"},
                    f"Invalid generated output root: {key}")
            for pattern in string_list(output.get("patterns"), f"{key} generated patterns", nonempty=True):
                relative(pattern, pattern=True)
            for producer in string_list(output.get("producers"), f"{key} producers", nonempty=True):
                require(producer in commands, f"Unknown producer: {producer}")
            require(isinstance(output.get("scope"), str) and output["scope"].strip(), f"Missing generated scope: {key}")
            string_list(output.get("identity_fields"), f"{key} identity fields", nonempty=True)
        for command in string_list(row.get("commands"), f"{key} commands", nonempty=True):
            require(command in commands, f"Unknown catalog command: {command}")
        for path in string_list(row.get("contracts"), f"{key} contract paths", nonempty=True):
            safe_path(root, path)
        require(type(row.get("provider_data")) is bool, f"Invalid provider-data flag: {key}")
        for index in string_list(row.get("artifact_indexes"), f"{key} artifact indexes"):
            require(index in {"runs/index.json", "records/index.json"}, f"Unknown artifact index: {index}")
            safe_path(root, index)


def example_runs(root: Path, scope: dict) -> list[dict]:
    from scripts.archive_run import verify_stage_chain
    examples = []
    paths = string_list(scope.get("bundled_example_run_roots", []), "bundled example roots")
    for value in paths:
        require(value.startswith("examples/runs/"), "Bundled examples must stay under examples/runs/")
        path = safe_path(root, value, kind="directory")
        for child in path.rglob("*"):
            require(not child.is_symlink(), f"Synthetic example contains a symlink: {child.relative_to(root)}")
        state = read_json(safe_path(root, f"{value}/run_state.json"))
        config = read_json(safe_path(root, f"{value}/config.json"))
        require(state.get("scientific_run") is False and config.get("scientific_run") is False
                and state.get("last_complete_stage") == "evaluate" and state.get("status") == "COMPLETE",
                f"Bundled example is not a complete declared synthetic run: {value}")
        identities = verify_stage_chain(path)
        require(all(HASH.fullmatch(str(identities.get(k, ""))) for k in IDENTITY_KEYS), f"Invalid example identity: {value}")
        require(all(state.get(k) == identities[k] for k in ("source_sha256", "config_sha256")),
                f"Example run-state identity differs from its stage chain: {value}")
        examples.append({"path": value, "artifact_class": "synthetic_example", "dataset_adapter": config.get("benchmark_dataset"),
                         "profile": config.get("profile"), "identities": {k: identities[k] for k in IDENTITY_KEYS},
                         "sealed_stage_count": len(identities["stages"]), "sealed_artifacts": identities["sealed_artifacts"],
                         "verification": "sealed output hashes and stage links checked; no numerical or checkpoint execution"})
    return examples


def binding_relationship(root: Path, entry: dict) -> dict:
    """Describe an identity-bound author declaration, never authenticate its history."""
    result = {"artifact_class": "benchmark_relationship_unspecified", "declared_relationship": None,
              "binding": None, "identity_checked": False, "historical_origin_authenticated": False}
    if entry["scientific_run"] is False:
        result["artifact_class"] = "synthetic_example"
        return result
    value = f"paper/bindings/{entry['stem']}_full.json"
    path = safe_path(root, value, exists=False)
    if not path.is_file():
        return result
    binding = read_json(path)
    if binding.get("status") != "author_completed":
        return result
    require(binding.get("schema_version") == "result-binding-1", f"Unsupported result binding schema: {value}")
    relationship = binding.get("relationship_to_manuscript")
    require(relationship in RELATIONSHIPS, f"Unknown manuscript relationship in {value}")
    identities = binding.get("run_identities")
    require(isinstance(identities, dict) and identities.get("dataset") == entry["dataset"], f"Binding dataset mismatch: {value}")
    if entry["kind"] == "campaign":
        rows = identities.get("splits")
        require(isinstance(rows, list) and len(rows) == 10 and all(isinstance(row, dict) for row in rows),
                f"Campaign binding needs ten split identities: {value}")
        require(all(type(row.get("split_index")) is int for row in rows)
                and [row["split_index"] for row in rows] == list(range(10)), f"Campaign binding split order mismatch: {value}")
        for i, row in enumerate(rows):
            require(row.get("dataset") == entry["dataset"] and all(row.get(k) == entry["run_identities"][f"split_{i:02d}"][k]
                    for k in IDENTITY_KEYS), f"Campaign binding identity mismatch: {value}, split {i}")
    else:
        require(all(identities.get(k) == entry["run_identities"][k] for k in IDENTITY_KEYS), f"Binding identity mismatch: {value}")
    result.update(artifact_class=RELATIONSHIPS[relationship], declared_relationship=relationship,
                  binding=fingerprint(path, root), identity_checked=True)
    return result


def registered_runs(root: Path) -> list[dict]:
    from scripts.ingest_run import check_absent_manifest, check_present, registered_path, registration_descriptor
    index = read_json(safe_path(root, "runs/index.json"))
    require(index.get("schema") == "cross-morse-runs-index-1" and isinstance(index.get("runs"), list), "Invalid run index")
    runs, seen = [], set()
    for entry in index["runs"]:
        descriptor = registration_descriptor(entry)
        require(entry["dataset"] not in seen, "Duplicate run-index dataset")
        seen.add(entry["dataset"])
        path = registered_path(entry, root / "runs")
        if path.exists():
            check_present(entry, path)
            materialization = "local_payload_verified"
        else:
            check_absent_manifest(entry, root / "manifests/official")
            materialization = "deposit_reference_metadata_verified"
        runs.append({"dataset": entry["dataset"], "index": "runs/index.json", "registration": descriptor,
                     "materialization": materialization, "public_download_verified": False,
                     "manuscript_relationship": binding_relationship(root, entry)})
    return runs


def registered_records(root: Path) -> list[dict]:
    from scripts.records import load_specs, verify_set
    specs = load_specs(root / "records")
    index = read_json(safe_path(root, "records/index.json"))
    require(index.get("schema") == "cross-morse-records-index-1" and isinstance(index.get("sets"), list), "Invalid record index")
    records, seen = [], set()
    for entry in index["sets"]:
        require(isinstance(entry, dict) and entry.get("set") in specs["sets"] and entry["set"] not in seen,
                "Unknown or duplicate record set")
        name = entry["set"]
        seen.add(name)
        path = safe_path(root, f"records/{name}", kind="directory")
        for child in path.rglob("*"):
            require(not child.is_symlink(), f"Record set contains a symlink: {child.relative_to(root)}")
        checked = verify_set(path, specs["sets"][name], specs)
        require(checked == entry and checked.get("synthetic") is False, f"Record index disagrees with its verified measurement set: {name}")
        records.append({"artifact_class": "measurement_records_origin_unspecified", "index": "records/index.json",
                        "registration": checked, "materialization": "local_payload_and_value_trace_verified",
                        "historical_origin_authenticated": False})
    return records


def build(root: Path = ROOT, *, spec: dict | None = None) -> dict:
    from scripts._paper_common import DATASETS, runtime_source_fingerprint
    root = Path(root).resolve()
    spec = spec if spec is not None else read_json(safe_path(root, SPEC))
    validate_spec(root, spec)
    scope = read_json(safe_path(root, "release/release_scope.json"))
    cache = {}

    def identity(path):
        key = path.relative_to(root).as_posix()
        if key not in cache:
            cache[key] = fingerprint(path, root)
        return cache[key]

    entries = []
    for row in spec["entries"]:
        bundled = []
        for asset in row["bundled"]:
            matched = sorted({path for pattern in asset["patterns"] for path in expand(root, pattern)})
            bundled.append({**asset, "files": [identity(path) for path in matched]})
        generated = [{**output, "artifact_class": "new_execution_contract", "materialized_by_catalog": False}
                     for output in row["generated"]]
        entries.append({**row, "bundled": bundled, "generated": generated,
                        "contract_identities": [identity(safe_path(root, path)) for path in row["contracts"]]})
    sources = read_json(safe_path(root, "data/sources.json"))
    require(sources.get("schema_version") == "cross-morse-data-sources-1", "Invalid dataset source registry")
    providers = sources.get("datasets")
    require(isinstance(providers, list) and all(isinstance(row, dict) for row in providers)
            and {row.get("dataset") for row in providers} == set(DATASETS) and len(providers) == len(DATASETS),
            "Dataset source registry must identify the seven unique benchmark datasets")
    for row in providers:
        require(row.get("provider_raw_data_bundled") is False, "Provider data must not be relabeled as bundled benchmark results")
        for field in ("source_documentation", "download_url"):
            if row.get(field) is not None:
                parsed = urlparse(row[field])
                require(parsed.scheme == "https" and parsed.netloc and not parsed.username and not parsed.password,
                        f"Invalid provider URL: {row.get('dataset')}")
    loader_hashes = sources.get("loader_file_sha256")
    require(isinstance(loader_hashes, dict) and bool(loader_hashes), "Dataset loader identities are missing")
    for path, digest in loader_hashes.items():
        require(identity(safe_path(root, path))["sha256"] == digest, f"Dataset loader source identity changed: {path}")
    inputs = [SPEC, "scripts/artifact_catalog.py", "release/release_scope.json", "runs/index.json", "records/index.json", "records/specs.json",
              "data/sources.json", "paper/sources.json", "paper/display_manifest.json"]
    runs, records = registered_runs(root), registered_records(root)
    if scope.get("original_result_archives_included") is True:
        require(len(runs) == 7 and all(row["materialization"] == "local_payload_verified"
                and row["manuscript_relationship"]["artifact_class"] == "declared_original_benchmark" for row in runs),
                "The declared original-archive release needs seven identity-bound original-run declarations and local payloads")
    return {
        "schema": SCHEMA,
        "scope": "File availability, generating contracts, registered identities and local byte/trace checks; no execution, public-download check or scientific certification",
        "runtime_source_sha256": runtime_source_fingerprint(root),
        "inputs": [identity(safe_path(root, path)) for path in inputs],
        "commands": {key: {**value, "entrypoint_identities": [identity(safe_path(root, path)) for path in value["entrypoints"]]}
                     for key, value in spec["commands"].items()},
        "entries": entries,
        "provider_datasets": [{**row, "artifact_class": "external_provider_dataset", "network_checked": False} for row in providers],
        "execution_contract": {"identity_fields": list(IDENTITY_KEYS), "roles": list(ROLES),
                               "rule": sources.get("role_split_rule"), "source_checksums": sources.get("checksum_policy")},
        "synthetic_examples": example_runs(root, scope),
        "registered_runs": runs,
        "registered_measurement_sets": records,
        "all_paper_displays_reproduced": False,
        "scientific_evidence_verified": False,
    }


def _cell(value: str) -> str:
    return value.replace("|", r"\|").replace("\n", " ")


def render(catalog: dict) -> str:
    lines = ["# Artifact catalog", "",
             "[Availability](../AVAILABILITY.md) · [Reconstruction guide](RECONSTRUCTION.md) · "
             "[Display support](../paper/PAPER_SUPPORT.md) · [Machine-readable catalog](../release/artifact_catalog.json)", "",
             "This catalog maps each item in the availability statement to shipped files, dataset providers, "
             "execution products and producing commands. Bundled files have SHA-256 and byte identities in the JSON "
             "catalog. A generated-file contract describes what an execution produces; it does not represent a file "
             "already available for download. Paths use the indicated root: the checkout, one run, the lifecycle suite, "
             "one imported record set, or the relevant command's report/output directory.", "",
             "## Artifact types", "",
             "- **Implementation and registered design:** executable code, configurations, prescribed seeds, split rules and environments.",
             "- **Reported numerical references:** transcribed paper values and protocol constants with source provenance.",
             "- **Synthetic outputs:** complete software examples with their own historical source identities.",
             "- **New execution products:** checkpoints, predictions, logs and analyses created by the declared producer.",
             "- **Registered experiment artifacts:** original or subsequent runs distinguished by an explicit manuscript relationship tied to source, configuration and unit identities.", "",
             "## Availability map", "", "| Item | Shipped material | Execution products | Scope |", "|---|---|---|---|"]
    for row in catalog["entries"]:
        shipped = "; ".join(CLASS_LABELS[asset["artifact_class"]] + ": " + ", ".join(f"`{p}`" for p in asset["patterns"])
                            for asset in row["bundled"])
        outputs = "; ".join(f"`{output['root']}/` " + ", ".join(f"`{p}`" for p in output["patterns"])
                            for output in row["generated"]) or "Declared inputs and implementation"
        if row["provider_data"]:
            outputs = "Official data from the providers below; " + outputs
        lines.append("| " + " | ".join(_cell(s) for s in (row["title"], shipped, outputs, row["scope"])) + " |")
    lines += ["", "## Producing and checking commands", "",
              "Paths in angle brackets are inputs supplied by the operator. These recipes are listed for inspection; "
              "catalog generation runs none of them. `make reconstruct` reports the supported outputs of each display; "
              "it does not turn related plots into exact replicas of every printed panel.", "",
              "| Command | Purpose | Required inputs |", "|---|---|---|"]
    for command in catalog["commands"].values():
        argv = shlex.join(command["argv"])
        lines.append("| " + " | ".join(_cell(value) for value in (f"`{argv}`", command["scope"], "; ".join(command["requires"]) or "Checkout")) + " |")
    lines += ["", "## Dataset providers", "",
              "URLs and adapter identities come from [data/sources.json](../data/sources.json). "
              "The catalog verifies the recorded loader hashes, without making network requests or asserting provider uptime.", "",
              "| Dataset | Provider documentation | Statistical unit |", "|---|---|---|"]
    for row in catalog["provider_datasets"]:
        lines.append(f"| {_cell(row['dataset'])} | [{_cell(row['source_documentation'])}]({row['source_documentation']}) | {_cell(row['statistical_unit'])} |")
    lines += ["", "## Execution identity and data roles", "",
              "Every run uses source, resolved-configuration and unit-membership identities. The prepared split arrays "
              "record the executed memberships; registered design manifests specify the intended seeds and rules. "
              "The roles are `fit`, `explore`, `select`, `confirm`, `compare`, `admit` and `test`. "
              "Optimization seeds, ensemble members, resampling seeds, evaluator and statistical unit must remain attached "
              "to each relevant result. [The registration guide](EVIDENCE_REGISTRATION.md) explains the original-run declarations.", "",
              f"Current runtime identity: `{catalog['runtime_source_sha256']}`.", "",
              "| Bundled synthetic run | Source identity | Sealed stages |", "|---|---|---|"]
    for row in catalog["synthetic_examples"]:
        lines.append(f"| `{row['path']}` | `{row['identities']['source_sha256']}` | {row['sealed_stage_count']} |")
    lines += ["", "## Registered experimental artifacts", "",
              "[runs/index.json](../runs/index.json) and [records/index.json](../records/index.json) are the registration "
              "sources. Present archive payloads are checked against their sealed stage chains and file inventories; "
              "measurement sets are checked against their manifests and raw-value traces. An absent archive with a deposit "
              "reference requires a matching committed registration manifest. Public retrieval remains a separate check.", ""]
    if catalog["registered_runs"]:
        lines += ["| Dataset | Materialization | Declared manuscript relationship | Source identity |", "|---|---|---|---|"]
        for row in catalog["registered_runs"]:
            relationship = row["manuscript_relationship"]["declared_relationship"] or "Not declared"
            lines.append("| " + " | ".join(_cell(s) for s in (row["dataset"], row["materialization"], relationship,
                         ", ".join(f"`{x}`" for x in row["registration"]["run_source_sha256"]))) + " |")
    else:
        lines += ["This code release describes the generated benchmark artifacts above. Original completed runs or new "
                  "reproductions can be registered with their actual source identities and an explicit relationship to the paper."]
    if catalog["registered_measurement_sets"]:
        lines += ["", "Validated local measurement sets: " + ", ".join(f"`{row['registration']['set']}`" for row in catalog["registered_measurement_sets"]) + "."]
    lines += ["", "An `exact_result_producing_run` declaration identifies an author-declared original run; "
              "`post_result_revision` identifies a later reproduction, and `refactor_with_prediction_equivalence` "
              "identifies the declared equivalence case. This catalog checks declaration identities. It does not "
              "authenticate historical chronology, recompute result bindings or replace the scientific evidence audit.", "",
              "## Regeneration and verification", "", "```bash", "python -B scripts/artifact_catalog.py",
              "python -B scripts/artifact_catalog.py --check", "python -B scripts/artifact_catalog.py --output-dir /abs/catalog-review", "```", "",
              "The specification is [artifact_catalog_spec.json](../release/artifact_catalog_spec.json). "
              "`--check` validates its types, required coverage, safe paths, producer references, file hashes and registrations, "
              "then checks that both generated files are current. It executes no training, shell recipe or downloaded code.", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        root = args.root.resolve()
        catalog = build(root)
        outputs = {CATALOG: json.dumps(catalog, indent=2, sort_keys=True, allow_nan=False) + "\n", GUIDE: render(catalog)}
        for relative_path, content in outputs.items():
            path = args.output_dir / Path(relative_path).name if args.output_dir else root / relative_path
            require(not path.is_symlink() and not any(parent.is_symlink() for parent in path.parents), "Catalog output traverses a symlink")
            if args.check:
                require(path.is_file() and path.read_text(encoding="utf-8") == content,
                        f"{path} is stale; run python -B scripts/artifact_catalog.py")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
        print(json.dumps({"status": "PASS", "availability_items": len(catalog["entries"]),
                          "mode": "check" if args.check else "write", "scientific_evidence_verified": False}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
