#!/usr/bin/env python3
"""Inventory instantiated plants and keep the seven-dataset documentation current.

No data download, training or checkpoint deserialization. Every released full
profile is instantiated at its official input/output dimensions and compared
with the recorded source counts and with the Table S5 count at the printed
precision; ``--default-widths`` instantiates the plant with the default widths
(the released default configuration of Table S5d). Equal counts are a necessary
identity check; archived checkpoints establish the executed model. Documentation
modes use only the standard library and do not instantiate a model.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts._paper_common import (DATASETS, STEMS, load_registered, printed_half_unit,
                                   runtime_source_fingerprint, sha256_file)

BEGIN = "<!-- BEGIN MODEL PARAMETER INVENTORY -->"
END = "<!-- END MODEL PARAMETER INVENTORY -->"
WIDTH_KEYS = ("encoder_hidden", "head_hidden", "feedforward_hidden", "block_mlp_hidden", "block_mlp_layers")


def count_agrees_at_printed_precision(count, target):
    """Nonnegative counts round half up; the upper half-unit belongs to the next value."""
    if type(count) is not int or count < 0:
        raise ValueError("Parameter count must be a nonnegative integer")
    if isinstance(target, bool) or not isinstance(target, (int, float)) or not math.isfinite(target) or target <= 0:
        raise ValueError("Invalid manuscript parameter count")
    printed = Decimal(str(target))
    unit = Decimal(10) ** printed.as_tuple().exponent
    actual = Decimal(count) / Decimal(1_000_000)
    return printed - unit / 2 <= actual < printed + unit / 2


def configured_architecture(config):
    """Normalize the documented factory settings without importing PyTorch.

    Instantiation checks this declaration against the model's actual metadata;
    checkpoint review compares it with the sealed model configuration.
    """
    model = config["model"]
    dataset = config.get("benchmark_dataset", config.get("dataset"))
    default = {"ogbn-proteins": "edge_weighted_transport", "Roman-empire": "heterophily",
               "PascalVOC-SP": "gps", "COCO-SP": "gps", "Peptides-func": "linear_gine"}
    backbone = model.get("backbone", default.get(dataset, "shared_transport"))
    classes = {"shared_transport": "FrozenPlant", "edge_weighted_transport": "EdgeWeightedPlant",
               "heterophily": "HeterophilyPlant", "gps": "DesignatedSitePlant", "linear_gine": "DesignatedSitePlant"}
    if backbone not in classes:
        raise ValueError(f"Unsupported backbone: {backbone}")
    def integer(key, default=None, minimum=1, optional=False):
        value = model.get(key, default)
        value = default if optional and value is None else value
        if type(value) is not int or value < minimum:
            raise ValueError(f"Invalid model {key}: expected an integer >= {minimum}")
        return value
    hidden = integer("hidden_dim")
    layers = integer("encoder_layers")
    encoder = integer("encoder_hidden", hidden, optional=True)
    head = integer("head_hidden", 0, minimum=0)
    depth = integer("depth")
    if layers == 1 and encoder != hidden:
        raise ValueError("A one-layer encoder has no separate encoder_hidden width")
    widths = {"hidden_dim": hidden, "encoder_layers": layers, "encoder_hidden": encoder,
              "head_hidden": head, "head_layers": 2 if head else 1}
    if backbone == "gps":
        widths.update(native_layers=integer("native_layers", 6), attention_heads=integer("attention_heads", 4),
                      feedforward_hidden=integer("feedforward_hidden", 2 * hidden, optional=True))
        if hidden % widths["attention_heads"]:
            raise ValueError("GPS hidden width must be divisible by attention_heads")
        if depth < widths["native_layers"] + 2 or (depth - 2) % widths["native_layers"]:
            raise ValueError("GPS depth must equal native_layers * substeps + two boundary sites")
    elif backbone == "linear_gine":
        integer("attention_heads", 4)
        widths.update(native_layers=integer("native_layers", 8), block_mlp_hidden=integer("block_mlp_hidden", hidden, optional=True),
                      block_mlp_layers=integer("block_mlp_layers", 2))
        if widths["block_mlp_layers"] not in (2, 3):
            raise ValueError("Linear-GINE block_mlp_layers must be 2 or 3")
        if depth % widths["native_layers"]:
            raise ValueError("Linear GINE depth must be a multiple of native_layers")
    if backbone != "gps" and model.get("feedforward_hidden") is not None:
        raise ValueError("feedforward_hidden is a GPS-only setting")
    if backbone != "linear_gine" and (model.get("block_mlp_hidden") is not None
                                      or type(model.get("block_mlp_layers", 2)) is not int
                                      or model.get("block_mlp_layers", 2) != 2):
        raise ValueError("Block MLP widths/layers require a linear-GINE backbone")
    return {"architecture": classes[backbone], "architecture_module": "cross_morse.core", "architecture_widths": widths}


def default_width_config(config):
    """The same configuration with the optional width keys removed (default widths)."""
    stripped = dict(config)
    stripped["model"] = {key: value for key, value in config["model"].items() if key not in WIDTH_KEYS}
    return stripped


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def reference_profiles(registered):
    """Validate the complete source-record schema without importing the runtime."""
    from scripts.check_paper_constants import source_dimensions
    profiles = registered.get("backbones_source_status", {}).get("released_full_profiles", {})
    targets = registered.get("backbones_table_S5", {})
    for label, rows in (("released_full_profiles", profiles), ("backbones_table_S5", targets)):
        if not isinstance(rows, dict) or set(rows) != set(DATASETS):
            raise ValueError(f"{label} must cover exactly the seven canonical datasets")
    dimensions = source_dimensions(ROOT)
    for dataset in DATASETS:
        row = profiles[dataset]
        if not isinstance(row, dict) or row.get("configuration") != f"configs/{STEMS[dataset]}_full.json":
            raise ValueError(f"Invalid full configuration reference for {dataset}")
        for key in ("input_dim", "num_outputs", "parameter_count", "buffer_elements"):
            value = row.get(key)
            if type(value) is not int or value < (0 if key == "buffer_elements" else 1):
                raise ValueError(f"Invalid {key} reference for {dataset}")
        if not isinstance(row.get("dimension_source"), str) or not row["dimension_source"].strip():
            raise ValueError(f"Missing dimension_source for {dataset}")
        if (row["input_dim"], row["num_outputs"]) != dimensions[dataset]:
            raise ValueError(f"Reference dimensions differ from the adapter/task contracts for {dataset}")
        components = row.get("parameter_components")
        if (not isinstance(components, dict) or not components
                or any(type(v) is not int or v < 0 for v in components.values())
                or sum(components.values()) != row["parameter_count"]):
            raise ValueError(f"Parameter components must be nonnegative integers that sum to the count for {dataset}")
        if not isinstance(row.get("architecture_widths"), dict) or not row["architecture_widths"]:
            raise ValueError(f"Missing architecture_widths reference for {dataset}")
        config = json.loads((ROOT / row["configuration"]).read_text(encoding="utf-8"))
        if config.get("dataset") != dataset or config.get("profile") != "full":
            raise ValueError(f"Full-profile routing mismatch: {row['configuration']}")
        expected = configured_architecture(config)
        if row["architecture_widths"] != expected["architecture_widths"]:
            raise ValueError(f"Architecture widths differ from the full configuration for {dataset}")
        if any(type(v) is not int for v in row["architecture_widths"].values()):
            raise ValueError(f"Architecture widths must use integers for {dataset}")
        value = targets[dataset].get("parameters_M")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid Table S5 parameter count for {dataset}")
    return profiles


def _width_summary(widths):
    parts = []
    if "feedforward_hidden" in widths:
        parts.append(f"{widths['native_layers']} GPS blocks, feed-forward {widths['feedforward_hidden']}")
    elif "block_mlp_hidden" in widths:
        parts.append(f"{widths['native_layers']} linear-GINE blocks, {widths['block_mlp_layers']}-layer MLP {widths['block_mlp_hidden']}")
    else:
        parts.append(f"{widths['encoder_layers']}-layer encoder, width {widths['encoder_hidden']}")
    parts.append(f"head {widths['head_hidden']}" if widths.get("head_hidden") else "linear head")
    return "; ".join(parts)


def render_parameter_table(registered):
    """Render the exact full-profile inventory next to the Table S5 counts."""
    profiles = reference_profiles(registered)
    targets = registered["backbones_table_S5"]
    lines = ["| Dataset | Table S5 (M) | Released full profile | Inputs / outputs | Explicit internal widths |",
             "|---|---:|---:|---:|---|"]
    for dataset in DATASETS:
        row = profiles[dataset]
        target = targets[dataset]["parameters_M"]
        mark = "" if count_agrees_at_printed_precision(row["parameter_count"], target) else " (differs)"
        lines.append(f"| {dataset} | {target} | {row['parameter_count']:,}{mark} | {row['input_dim']} / {row['num_outputs']} "
                     f"| {_width_summary(row['architecture_widths'])} |")
    return "\n".join(lines)


def document_with_parameter_table(text, registered):
    if text.count(BEGIN) != 1 or text.count(END) != 1 or text.index(BEGIN) > text.index(END):
        raise ValueError("Documentation must contain one ordered model-parameter inventory marker pair")
    start, end = text.index(BEGIN) + len(BEGIN), text.index(END)
    return text[:start] + "\n" + render_parameter_table(registered) + "\n" + text[end:]


def resolve_dimensions(config, registered, input_dim=None, num_outputs=None):
    """Infer only dimensions declared for a supported profile or synthetic data."""
    if config.get("dataset") == "synthetic":
        defaults = (config.get("data", {}).get("synthetic_features"),
                    config.get("data", {}).get("synthetic_classes"))
    elif config.get("dataset") in DATASETS and config.get("profile") in {"full", "pilot"}:
        row = reference_profiles(registered)[config["dataset"]]
        defaults = (row["input_dim"], row["num_outputs"])
    else:
        defaults = (None, None)
    values = (input_dim if input_dim is not None else defaults[0],
              num_outputs if num_outputs is not None else defaults[1])
    if any(value is None for value in values):
        raise ValueError("Custom or unknown profiles require explicit --input-dim and --outputs")
    if any(type(value) is not int or value < 1 for value in values):
        raise ValueError("Model dimensions must be positive integers")
    return values


def inventory(config, input_dim: int, num_outputs: int) -> dict:
    from cross_morse.core import build_plant
    from cross_morse.tasks import task_spec
    if any(type(value) is not int or value < 1 for value in (input_dim, num_outputs)):
        raise ValueError("Model dimensions must be positive integers")
    spec = task_spec(config, metadata={"num_outputs": num_outputs})
    model = build_plant(input_dim, num_outputs, config["model"], spec)
    parameters = [{"name": name, "shape": list(tensor.shape), "elements": tensor.numel(),
                   "requires_grad": bool(tensor.requires_grad)} for name, tensor in model.named_parameters()]
    buffers = [{"name": name, "shape": list(tensor.shape), "elements": tensor.numel(),
                "dtype": str(tensor.dtype)} for name, tensor in model.named_buffers()]
    components = {}
    for row in parameters:
        component = row["name"].split(".", 1)[0]
        components[component] = components.get(component, 0) + row["elements"]
    widths = model.architecture_widths() if hasattr(model, "architecture_widths") else {}
    return {"schema": "cross-morse-instantiated-model-inventory-1", "dataset": config["dataset"],
            "profile": config["profile"], "input_dim": input_dim, "num_outputs": num_outputs,
            "architecture": type(model).__name__, "architecture_module": type(model).__module__,
            "architecture_widths": widths,
            "task": spec.to_dict(), "model_configuration": config["model"],
            "config_sha256": _identity(config), "source_sha256": runtime_source_fingerprint(ROOT),
            "parameter_count": sum(row["elements"] for row in parameters),
            "trainable_parameter_count": sum(row["elements"] for row in parameters if row["requires_grad"]),
            "buffer_elements": sum(row["elements"] for row in buffers),
            "parameter_components": components, "parameters": parameters, "buffers": buffers,
            "count_scope": "Unique named model parameters; trainable means requires_grad at instantiation, before training or freezing. Buffers are separate; controllers and ensemble copies are excluded.",
            "scope": "Instantiated released implementation; count agreement alone does not establish archived benchmark identity"}


def compare_manuscript_count(result, registered):
    """Attach the Table S5 comparison at the printed precision."""
    target = registered["backbones_table_S5"][result["dataset"]]["parameters_M"]
    tolerance = printed_half_unit(target) * 1_000_000
    result["manuscript_reference"] = {
        "table": "S5", "parameters_M": target, "count_tolerance": tolerance,
        "count_agrees_at_printed_precision": count_agrees_at_printed_precision(result["parameter_count"], target),
        "rounding_interval": "lower inclusive, upper exclusive (round half up)",
        "scope": "Count agreement at the printed precision; make compare-runs checks complete sealed official archives"}
    return result


def compare_source_record(result, registered):
    """Attach the exact released-record comparison and the Table S5 comparison."""
    dataset = result["dataset"]
    row = reference_profiles(registered)[dataset]
    keys = ("input_dim", "num_outputs", "parameter_count", "buffer_elements")
    result["source_record"] = {key: row[key] for key in ("configuration", "dimension_source", *keys)}
    result["source_record"].update(parameter_components=row.get("parameter_components"),
                                   architecture_widths=row.get("architecture_widths"))
    result["source_record_mismatches"] = [key for key in keys if result[key] != row[key]]
    for key in ("parameter_components", "architecture_widths"):
        if row.get(key) is not None and result.get(key) != row[key]:
            result["source_record_mismatches"].append(key)
    expected = configured_architecture({"dataset": dataset, "model": result["model_configuration"]})
    for key, value in expected.items():
        if result.get(key) != value:
            result["source_record_mismatches"].append("configured_" + key)
    result["source_record_matches"] = not result["source_record_mismatches"]
    return compare_manuscript_count(result, registered)


def all_full_inventory(registered=None):
    from cross_morse.tasks import task_spec
    registered = load_registered() if registered is None else registered
    references = reference_profiles(registered)
    rows = []
    for dataset in DATASETS:
        reference = references[dataset]
        path = ROOT / reference["configuration"]
        config = json.loads(path.read_text(encoding="utf-8"))
        if config.get("dataset") != dataset or config.get("profile") != "full":
            raise ValueError(f"Full-profile routing mismatch: {reference['configuration']}")
        if task_spec(dataset).num_outputs != reference["num_outputs"]:
            raise ValueError(f"Reference num_outputs differs from the official task for {dataset}")
        dimensions = resolve_dimensions(config, registered)
        row = compare_source_record(inventory(config, *dimensions), registered)
        row.update(configuration=reference["configuration"], config_file_sha256=sha256_file(path))
        rows.append(row)
    matches = all(row["source_record_matches"] for row in rows)
    agree = all(row["manuscript_reference"]["count_agrees_at_printed_precision"] for row in rows)
    return {"schema": "cross-morse-full-model-inventories-1", "profiles": rows,
            "source_records_match": matches, "table_S5_counts_agree": agree,
            "status": "SOURCE_RECORD_MISMATCH" if not matches else "TABLE_S5_COUNT_MISMATCH" if not agree else "SOURCE_RECORDS_MATCH",
            "registered_constants_canonical_sha256": _identity(registered),
            "scope": "Seven source instantiations; no data, training or checkpoint loading. This report is not the benchmark evidence gate."}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--all-full", action="store_true", help="instantiate all seven released full profiles")
    modes.add_argument("--check-docs", action="store_true", help="check the generated parameter table without PyTorch")
    modes.add_argument("--write-docs", action="store_true", help="update only the marked parameter table without PyTorch")
    parser.add_argument("--docs-path", type=Path, default=ROOT / "docs/DATASETS.md")
    parser.add_argument("--config", type=Path, help="default: configs/arxiv_full.json")
    parser.add_argument("--input-dim", type=int, help="override the declared dataset input dimension")
    parser.add_argument("--outputs", type=int, help="override the declared task output dimension")
    parser.add_argument("--expect-parameters", type=int)
    parser.add_argument("--default-widths", action="store_true",
                        help="instantiate the configuration with the default widths (the released default plant of Table S5d)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.default_widths and (args.all_full or args.check_docs or args.write_docs):
        parser.error("--default-widths applies to one configuration")
    if (args.all_full or args.check_docs or args.write_docs) and any(
            value is not None for value in (args.config, args.input_dim, args.outputs, args.expect_parameters)):
        parser.error("Single-model arguments cannot be combined with all-full or documentation modes")
    if (args.check_docs or args.write_docs) and args.output is not None:
        parser.error("Documentation modes use --docs-path, not --output")
    try:
        registered = load_registered()
        if args.check_docs or args.write_docs:
            original = args.docs_path.read_text(encoding="utf-8")
            expected = document_with_parameter_table(original, registered)
            if args.write_docs:
                args.docs_path.write_text(expected, encoding="utf-8")
            elif original != expected:
                print(f"Stale model parameter table: {args.docs_path}; run scripts/model_inventory.py --write-docs", file=sys.stderr)
                return 1
            print(f"Model parameter documentation {'updated' if args.write_docs else 'matches the registry'}: {args.docs_path}")
            return 0
        if args.all_full:
            result = all_full_inventory(registered)
        else:
            path = args.config or ROOT / "configs/arxiv_full.json"
            config = json.loads(path.read_text(encoding="utf-8"))
            instantiated = default_width_config(config) if args.default_widths else config
            result = inventory(instantiated, *resolve_dimensions(config, registered, args.input_dim, args.outputs))
            result.update(configuration=str(path), config_file_sha256=sha256_file(path),
                          default_widths=bool(args.default_widths))
            if config["dataset"] in DATASETS and config["profile"] == "full":
                if args.default_widths:
                    compare_manuscript_count(result, registered)
                    default_count = registered["backbones_source_status"].get("default_width_plant", {}).get(config["dataset"])
                    if default_count is not None:
                        result["default_source_record_matches"] = result["parameter_count"] == default_count
                else:
                    compare_source_record(result, registered)
            if args.expect_parameters is not None:
                result["expected_parameters"] = args.expect_parameters
                result["matches_expected"] = result["parameter_count"] == args.expect_parameters
        text = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
        print(text, end="")
        failed = any(result.get(key) is False for key in ("matches_expected", "source_record_matches", "source_records_match",
                                                        "default_source_record_matches", "table_S5_counts_agree"))
        if not args.default_widths and result.get("manuscript_reference", {}).get("count_agrees_at_printed_precision") is False:
            failed = True
        return 1 if failed else 0
    except (OSError, ValueError, KeyError, TypeError, SyntaxError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
