#!/usr/bin/env python3
"""Audit deposited scientific evidence without training, network access or pickle.

    python -B scripts/scientific_evidence_audit.py --output evidence-audit.json

Exit 0 means the declared evidence requirements passed; exit 1 means missing or
invalid evidence: the gate composes the stage, numeric-replay, result-binding,
record and campaign validators and reports every missing or invalid item.

Use --budget-campaign for the complete extracted E3--E6 campaign and
--confirmation-campaign for the original joint S14 tests. Optionally
--check-examples or --software-run to numerically inspect software fixtures.
Existing stage, numeric replay, result binding, record and campaign validators
remain the authorities; this command composes them and makes omissions blocking.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from cm_campaigns.contract import DATASETS, RECORD_SETS, SLUGS
from cross_morse.review import NumericEvidence, assert_matches, json_read, replay_results
from cross_morse.split_contract import DATA_ROLE_PROTOCOL, TRAIN_CUTS, require_reserved_comparison, validate_reserved_roles
from paper.build.evidence import evidence_from_run
from paper.build.selection import apply_reading, selection_readings
from scripts.archive_run import verify_stage_chain
from scripts.check_result_binding import check_binding
from scripts.records import load_specs, read_summary, verify_set
from scripts.summarize_roman_splits import split_summary
from scripts.check_manuscript_sources import inspect_sources
from scripts._paper_common import evaluated_arm_binding_targets, printed_half_unit, resolved_policy_view

SCHEMA = "cross-morse-scientific-evidence-audit-1"
BINDING_POINTERS = {
    "/paired_qD_minus_qC_loss/mean", "/paired_qD_minus_qC_loss/lower", "/paired_qD_minus_qC_loss/upper",
    "/paired_qD_minus_qC_metric_pp/mean", "/paired_qD_minus_qC_metric_pp/lower", "/paired_qD_minus_qC_metric_pp/upper",
    "/models/qD/metric_percent", "/models/qC/metric_percent", "/models/qN/metric_percent",
}


def support_row(row: dict, thresholds: dict | None = None) -> dict:
    """Check one same-cell summary, including the necessary ESS/max-weight bound.

    An empty cell is a legitimate unqualified cell, not ESS=1/0. Legacy summary
    pairs can disprove consistency, but cannot establish the weight identity.
    Raw trajectory reconstruction is a separate check in audit_run().
    """
    flags, missing = [], []
    output = {"evidence_kind": "derived_arithmetic", "flags": flags, "missing": missing}
    def number(name):
        value = row.get(name)
        if value is None:
            missing.append(name)
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            flags.append("nonfinite_or_invalid_" + name)
            return None
        return float(value)
    ess, maximum = number("ess"), number("max_normalized_weight")
    count = number("trajectory_count")
    total, squares = number("sum_normalized_weights"), number("sum_squared_normalized_weights")
    if count is not None and (count < 0 or count != int(count)):
        flags.append("invalid_trajectory_count")
    empty = count == 0
    if empty:
        if ess != 0 or total != 0 or squares != 0 or row.get("qualified") is not False:
            flags.append("invalid_empty_cell")
    else:
        if ess is not None and ess < 1 - 1e-9:
            flags.append("ess_below_one")
        if maximum is not None:
            if not 0 < maximum <= 1:
                flags.append("invalid_max_normalized_weight")
            else:
                bound = 1 / maximum**2
                output["ess_upper_bound_from_max_weight"] = bound
                if ess is not None and ess > bound + 1e-9:
                    flags.append("ess_exceeds_inverse_max_weight_squared")
        if count is not None and ess is not None and ess > count + 1e-9:
            flags.append("ess_exceeds_trajectory_count")
        if total is not None and abs(total - 1) > 1e-9:
            flags.append("normalized_weights_do_not_sum_to_one")
        if squares is not None:
            if not 0 < squares <= 1 + 1e-9:
                flags.append("invalid_sum_squared_weights")
            elif ess is not None and not math.isclose(ess, 1 / squares, rel_tol=1e-9, abs_tol=1e-9):
                flags.append("ess_differs_from_inverse_sum_squared_weights")
        if squares is not None and maximum is not None and squares < maximum**2 - 1e-12:
            flags.append("sum_squared_weights_below_max_weight_squared")
    if thresholds is not None and ess is not None and maximum is not None:
        propensity = number("min_propensity")
        if propensity is not None:
            qualified = (not empty and propensity >= thresholds["p_min"] - 1e-12 and
                         ess >= thresholds["n_min"] - 1e-9 and
                         maximum <= thresholds["max_normalized_weight"] + 1e-12)
            if type(row.get("qualified")) is not bool or row["qualified"] != qualified:
                flags.append("qualification_differs_from_thresholds")
    output["status"] = "FAIL" if flags else "PENDING" if missing else "PASS"
    return output


def validate_split_evidence(metadata, splits, official, graph_splits=None):
    """Require pre-fitting D_compare and preserve official admission/test roles."""
    require_reserved_comparison(metadata, splits)
    roles = splits
    if metadata.get("independent_graphs"):
        if graph_splits is None:
            raise ValueError("Independent-graph data require graph_splits.npz")
        roles = validate_reserved_roles(graph_splits, size=metadata["num_graphs"])
    if set(official) != {"train", "valid", "test"}:
        raise ValueError("Official split identity is incomplete")
    same = lambda a, b: np.array_equal(np.sort(a), np.sort(b))
    training_roles = ("fit", "explore", "select", "confirm", "compare")
    if not same(np.concatenate([roles[r] for r in training_roles]), official["train"]):
        raise ValueError("Training roles differ from official train")
    if not same(roles["admit"], official["valid"]) or not same(roles["test"], official["test"]):
        raise ValueError("Official validation/test were repartitioned")
    cuts = [0, *[int(len(official["train"]) * f) for f in TRAIN_CUTS], len(official["train"])]
    if [len(roles[r]) for r in training_roles] != list(np.diff(cuts)):
        raise ValueError("Training allocation differs from the registered 60/20/10/5/5 protocol")
    return {"status": "PASS", "protocol": DATA_ROLE_PROTOCOL, "reserved_at": "prepare",
            "allocation_unit": "graph" if metadata.get("independent_graphs") else "node",
            "sizes": {r: len(roles[r]) for r in training_roles}}


def reference_selector(root):
    """Replay the registered selector on the printed representative rows."""
    reported = json_read(root / "paper/reported_results.json")
    constants = json_read(root / "paper/registered_constants.json")
    source = json_read(root / "paper/sources.json")
    selector = constants["selection_eq19_algorithm_S4"]
    margin = selector["epsilon_eq_nats"]
    if margin != .002:
        raise ValueError("Manuscript registered epsilon_eq is expected to be .002")
    route_names = {"one-step": "one_step", "MPC-RH": "mpc_rh"}
    records = []
    for dataset, ledger in reported["supplementary_table_S12_selection_ledger"].items():
        epsilon_r = selector["epsilon_R_nats"]["mean_multilabel_BCE" if dataset in
                       selector["datasets_using_BCE_margins"] else "graph_level_CE"]
        rows = [dict(index=i, route=route_names.get(r["route"], r["route"]), feasible=True,
                     select_upper=r["U_s"], select_delta=r["Delta_s_hat"], complexity=r["rank"],
                     ensemble=r["E"], cost=0.) for i, r in enumerate(ledger)]
        replay = apply_reading(rows, "select_upper", margin, epsilon_r)
        choice = "qC" if replay["choice"] is None else ledger[replay["choice"]]["route"]
        selected = [r["route"] for r in ledger if r.get("selected")]
        if len(selected) != 1:
            raise ValueError(f"{dataset}: one reported representative selection is required")
        mismatch = choice != selected[0]
        records.append({"dataset": dataset, "epsilon_eq": margin, "epsilon_r": epsilon_r,
                        "reported_choice": selected[0], "registered_choice": choice,
                        "near_best_routes": [ledger[i]["route"] for i in replay["near_best"]],
                        "representative_rows_match_reported_choice": not mismatch,
                        "note": "representative-row reading differs from the reported full-ledger choice; the complete candidate trace is required to verify it"
                             if mismatch else None,
                        "complete_ledger": "selector trace of the run archive (E15)"})
    return {"evidence_kind": "manuscript_reference", "measured_rerun": False,
            "scope": "arithmetic replay of the Table S12 representative rows; a difference from the reported full-ledger choice does not establish an incorrect executed selector",
            "rows": records, "selection_trace_required_datasets": [r["dataset"] for r in records if not r["representative_rows_match_reported_choice"]],
            "support_summary": reported.get("supplementary_table_S7_support_audit", {}),
            "support_status": "per-cell weight records are read from the run archive (E19)",
            "analytical_or_derived_groups": [k for k, v in source["table_groups"].items()
                                               if v.get("evidence_kind") in ("analytical_sensitivity", "analytical_model", "derived_arithmetic")],
            "source_classifications": {k: v.get("evidence_kind") for k, v in source["table_groups"].items()}}


def audit_support(e, config):
    """Reconstruct same-cell weights from sealed trajectories, never checkpoints."""
    from cross_morse.exploration import load_records
    from cross_morse.learning import _support, calibration_partition
    shards = sorted(e.root.glob("explore/trajectory_*.npz"))
    if not shards:
        raise ValueError("Raw exploration trajectories are missing")
    for path in shards:
        e.file(path.relative_to(e.root).as_posix())
    records = load_records(e.root / "explore")
    outputs = {}
    for profile in config["sensor"]["profiles"]:
        cfg = copy.deepcopy(config)
        cfg["active_profile"] = profile
        audit = e.json(f"fit/{profile}/audit.json")
        roles = e.arrays(f"fit/{profile}/oof_predictions.npz")["role"]
        expected_roles, partition = calibration_partition(records, cfg)
        assert_matches(expected_roles, roles, f"{profile}.calibration_roles")
        assert_matches(partition, audit["calibration_partition"], f"{profile}.calibration_partition")
        recomputed = _support(records, np.flatnonzero(roles == 0), cfg)
        cells = {key: support_row(row, recomputed["thresholds"]) for key, row in audit["support"]["cells"].items()}
        reconstruction_error = None
        try:
            assert_matches(recomputed, audit["support"], f"{profile}.support")
        except ValueError as exc:
            reconstruction_error = str(exc)
        passed = bool(cells) and all(r["status"] == "PASS" for r in cells.values()) and reconstruction_error is None
        outputs[profile] = {"status": "PASS" if passed else "FAIL", "raw_weight_reconstruction": reconstruction_error is None,
                            "reconstruction_error": reconstruction_error, "cells": cells,
                            "qualified_fraction": recomputed["qualified_fraction"],
                            "scope": "support arithmetic; no claim that repeated trajectories are independent graphs"}
    return outputs


def validate_model_inventory(inventory, reported_millions, *, configuration=None):
    """Require the sealed model inventory to agree with the reported S5 count.

    Counts are a necessary identity check, not sufficient architecture parity.
    The tolerance is half the printed unit in millions, not an arbitrary margin.
    """
    if isinstance(reported_millions, bool) or not isinstance(reported_millions, (int, float)) or not math.isfinite(reported_millions) or reported_millions <= 0:
        raise ValueError("Invalid manuscript parameter count")
    from scripts.model_inventory import configured_architecture, count_agrees_at_printed_precision
    if configuration is not None:
        from scripts.check_paper_constants import source_dimensions
        if inventory.get("configuration") != configuration.get("model"):
            raise ValueError("Checkpoint model configuration differs from its sealed run configuration")
        architecture = configured_architecture(configuration)
        for field, value in architecture.items():
            if inventory.get(field) != value:
                raise ValueError(f"Checkpoint {field} differs from its configured architecture")
        if any(type(value) is not int for value in inventory["architecture_widths"].values()):
            raise ValueError("Checkpoint architecture widths must use exact integers")
        dimensions = source_dimensions()[configuration["dataset"]]
        if any(type(inventory.get(field)) is not int or inventory[field] != value
               for field, value in zip(("input_dim", "num_outputs"), dimensions)):
            raise ValueError("Checkpoint input/output dimensions differ from the official adapter/task contract")
    checkpoints = inventory.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError("Actual checkpoint parameter inventories are required")
    counts, buffers = [], []
    for item in checkpoints:
        tensors = item.get("tensors")
        if not isinstance(tensors, list) or not tensors:
            raise ValueError("Named checkpoint tensor inventory is missing")
        if any(type(t.get("elements")) is not int or t["elements"] < 0 for t in tensors):
            raise ValueError("Invalid tensor element count")
        if configuration is not None:
            names = [t.get("name") for t in tensors]
            if any(not isinstance(name, str) or not name for name in names) or len(names) != len(set(names)):
                raise ValueError("Checkpoint tensors require unique nonempty names")
            for tensor in tensors:
                shape = tensor.get("shape")
                if (not isinstance(shape, list) or any(type(size) is not int or size < 0 for size in shape)
                        or math.prod(shape) != tensor["elements"]):
                    raise ValueError("Checkpoint tensor shape differs from its element count")
        total = sum(t["elements"] for t in tensors)
        if type(item.get("named_tensor_elements")) is not int or total != item["named_tensor_elements"]:
            raise ValueError("Checkpoint inventory total differs from named tensor elements")
        typed = (any("kind" in t for t in tensors)
                 or any(k in item for k in ("parameter_count", "buffer_elements", "optimization_active_parameter_count")))
        if configuration is not None and not typed:
            raise ValueError("Configured checkpoint review requires typed parameter/buffer tensors")
        if typed:
            if any(t.get("kind") not in ("parameter", "buffer") for t in tensors):
                raise ValueError("Typed inventory requires a parameter/buffer kind for every tensor")
            count = sum(t["elements"] for t in tensors if t["kind"] == "parameter")
            buffer_count = sum(t["elements"] for t in tensors if t["kind"] == "buffer")
            for field, measured in (("parameter_count", count), ("buffer_elements", buffer_count)):
                if type(item.get(field)) is not int or item[field] != measured:
                    raise ValueError(f"Checkpoint {field} differs from the typed tensor inventory")
            if "parameter_components" in item or configuration is not None:
                components = {}
                for tensor in tensors:
                    if tensor["kind"] == "parameter":
                        key = tensor["name"].split(".", 1)[0]
                        components[key] = components.get(key, 0) + tensor["elements"]
                recorded = item.get("parameter_components")
                if (not isinstance(recorded, dict) or any(type(value) is not int for value in recorded.values())
                        or recorded != components):
                    raise ValueError("Checkpoint parameter components differ from named parameter tensors")
            active = item.get("optimization_active_parameter_count")
            if "optimization_active_parameter_count" in item and (type(active) is not int or not 0 <= active <= count):
                raise ValueError("Optimization-active parameter count must be between zero and the parameter count")
        else:
            # Historical inventories lack kind metadata. Preserve their original
            # all-named-elements check; do not infer away possible parameters.
            count, buffer_count = total, None
        if not count_agrees_at_printed_precision(count, reported_millions):
            raise ValueError(f"Checkpoint count {count:,} disagrees with Table S5 {reported_millions}M at the printed precision")
        counts.append(count)
        buffers.append(buffer_count)
    return {"status": "PASS", "reported_parameters_M": reported_millions, "counts": counts,
            "buffer_elements": buffers,
            "configuration_metadata_verified": configuration is not None,
            "scope": "Necessary model-count agreement at printed precision; full architecture and checkpoint identity remain required"}


def audit_run(run_root, *, official_dataset=None, manuscript_root=ROOT):
    """Validate a completed run with the existing validators, plus new invariants."""
    run_root = Path(run_root)
    chain = verify_stage_chain(run_root)
    e = NumericEvidence(run_root)
    cfg = e.json("config.json")
    meta = e.json("prepared/metadata.json")
    synthetic = cfg["dataset"] == "synthetic" or meta.get("is_benchmark_data") is not True
    if official_dataset is not None and (synthetic or cfg["dataset"] != official_dataset or
            cfg.get("profile") != "full" or cfg.get("scientific_run") is not True):
        raise ValueError("An official full-profile archive is required; synthetic or pilot data are refused")
    splits = e.arrays("prepared/splits.npz")
    graph_splits = e.arrays("prepared/graph_splits.npz") if meta.get("independent_graphs") else None
    split = validate_split_evidence(meta, splits, e.arrays("prepared/official_splits.npz"), graph_splits)
    support = audit_support(e, cfg)
    numeric = replay_results(run_root, selection=True)
    selector = selection_readings(evidence_from_run(run_root))
    if not selector.get("available") or not selector.get("registered_all_configurations_reproduces_selection"):
        raise ValueError("The registered selector does not reproduce the sealed provisional choice")
    model_identity = {"status": "NOT_APPLICABLE", "scope": "Synthetic software run, not a Table S5 model"}
    role_isolation = {"status": "HISTORICAL_SOFTWARE_REPLAY", "scope": "Official evidence requires role-scoped execution; old fixtures remain replayable"}
    if official_dataset is not None:
        from cross_morse.role_evidence import audit_role_isolation
        role_isolation = audit_role_isolation(run_root)
        if not role_isolation["passed"]:
            raise ValueError("Independent-graph role isolation failed: " + "; ".join(role_isolation["errors"]))
        registered = json_read(Path(manuscript_root) / "paper/registered_constants.json")
        model_identity = validate_model_inventory(e.json("train/model_inventory.json"),
            registered["backbones_table_S5"][official_dataset]["parameters_M"], configuration=cfg)
    return {"status": "PASS" if all(r["status"] == "PASS" for r in support.values()) else "FAIL", "path": str(run_root),
            "evidence_kind": "synthetic_software_validation" if synthetic else "measured_official_data",
            "sealed_artifacts": chain["sealed_artifacts"], "split": split, "support": support,
            "numeric_replay": numeric, "selector": selector, "model_identity": model_identity, "role_isolation": role_isolation,
            "scope": "numeric reconstruction and sealed byte integrity; checkpoint re-execution is a separate check (scripts/evaluate_checkpoints.py)"}


def _inside(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Evidence index contains an unsafe path")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError("Evidence path escapes its archive root")
    return path


def _binding_value_matches(entry, expected, maximum_tolerance):
    value, tolerance = entry.get("reported_value"), entry.get("absolute_tolerance")
    return (all(not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)
                for v in (value, tolerance))
            and (value == expected if maximum_tolerance == 0
                 else math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12))
            and 0 <= tolerance <= maximum_tolerance + (0. if maximum_tolerance == 0 else 1e-15))


def binding_coverage(binding, *, campaign=False, dataset=None, manuscript_root=ROOT):
    artifact = "derived/roman_split_aggregate.json" if campaign else "final/results_summary.json"
    bound = {r.get("json_pointer") for r in binding.get("results", [])
             if r.get("artifact") == artifact and r.get("target_role") != "evaluated_arm"}
    missing = sorted(BINDING_POINTERS - bound)
    dataset = dataset or binding.get("run_identities", {}).get("dataset")
    if dataset == "Peptides-func":
        root = Path(manuscript_root)
        reported, registered = (json_read(root / "paper/reported_results.json"),
                                json_read(root / "paper/registered_constants.json"))
        view = resolved_policy_view(reported, dataset, registered)
        if view["comparator_identity"]:
            # A separate radius record cannot justify a nonzero final qD effect
            # after the registered confirmation rule returns the comparator.
            scale = 100. if reported["supplementary_table_S15_method_comparison"][dataset]["score_unit"].startswith("AP") else 1.
            for pointer in sorted(BINDING_POINTERS):
                if pointer.startswith("/paired_"):
                    value, tolerance, kind = 0., 0., "derived_comparator_identity"
                else:
                    model = pointer.split("/")[2]
                    value, tolerance = scale * view["scores"][model]["official_score"], scale * (.0005 if scale == 100 else .005)
                    kind = "derived_comparator_identity" if model == "qD" else "author_reported_score"
                matches = [r for r in binding.get("results", [])
                           if r.get("artifact") == artifact and r.get("json_pointer") == pointer
                           and r.get("target_role") == "served_policy"]
                if (len(matches) != 1 or matches[0].get("target_kind") != kind
                        or not _binding_value_matches(matches[0], value, tolerance)):
                    missing.append("served_policy:" + pointer)
        elif view["deployed_rule"] is None:
            missing.append("served_policy:predecessor_admission_unavailable")
        targets = evaluated_arm_binding_targets(reported, dataset, registered)
        for target in targets:
            matches = [r for r in binding.get("results", [])
                       if r.get("reference_pointer") == target["reference_pointer"]]
            valid = len(matches) == 1
            if valid:
                entry = matches[0]
                valid = (entry.get("target_role") == "evaluated_arm"
                         and entry.get("target_kind") == "author_reported_arm"
                         and isinstance(entry.get("artifact"), str) and bool(entry["artifact"])
                         and entry["artifact"] != "final/results_summary.json"
                         and isinstance(entry.get("json_pointer"), str) and entry["json_pointer"].startswith("/")
                         and _binding_value_matches(entry, target["reported_value"], target["absolute_tolerance"]))
            if not valid:
                missing.append("evaluated_arm:" + target["reference_pointer"])
    return missing


def check_campaign_binding(paths, inspected, binding):
    """Bind a recomputed ten-split aggregate, never an individual split interval.

    A result-binding-1 declaration for a campaign uses aggregation=
    paired_official_split_t95, run_identities={dataset, splits:[{split_index,
    source_sha256, config_sha256, unit_sha256, dataset}, ...]} and the virtual
    artifact derived/roman_split_aggregate.json. The aggregate is recomputed
    here from already replayed per-split observations, not read from a PASS file.
    """
    if binding.get("schema_version") != "result-binding-1" or binding.get("status") != "author_completed":
        raise ValueError("Complete the campaign author declaration; a template is not evidence")
    if binding.get("aggregation") != "paired_official_split_t95":
        raise ValueError("Roman campaign binding requires aggregation=paired_official_split_t95 and all ten split identities")
    if binding.get("relationship_to_manuscript") not in (
            "exact_result_producing_run", "refactor_with_prediction_equivalence", "post_result_revision"):
        raise ValueError("Declare the actual campaign relationship to the manuscript")
    if not isinstance(binding.get("manuscript_identifier"), str) or not binding["manuscript_identifier"].strip():
        raise ValueError("A manuscript identifier is required")
    identities, shared = [], set()
    from cross_morse.review import canonical
    for path in paths:
        evidence = NumericEvidence(path)
        config = evidence.json("config.json")
        split = config["data"]["split_index"]
        identities.append({"split_index": split, **evidence.identities})
        common = copy.deepcopy(config)
        common["data"].pop("split_index")
        shared.add((canonical(common), evidence.identities["source_sha256"]))
    identities.sort(key=lambda row: row["split_index"])
    if len(shared) != 1 or [r["split_index"] for r in identities] != list(range(10)):
        raise ValueError("All ten unique official splits must share source and non-split configuration")
    if binding.get("run_identities") != {"dataset": "Roman-empire", "splits": identities}:
        raise ValueError("Campaign manuscript binding differs from the actual split identities")
    reports = [r["numeric_replay"] for r in inspected]
    aggregate = {
        "paired_qD_minus_qC_loss": split_summary([r["paired_qD_minus_qC_loss"]["mean"] for r in reports]),
        "paired_qD_minus_qC_metric_pp": split_summary([r["paired_qD_minus_qC_metric_pp"]["mean"] for r in reports]),
        "models": {name: {"metric_percent": float(np.mean([r["models"][name]["metric_percent"] for r in reports]))}
                   for name in ("qN", "qC", "qD")},
    }
    results = []
    for entry in binding["results"]:
        if entry.get("artifact") != "derived/roman_split_aggregate.json" or entry.get("json_pointer") not in BINDING_POINTERS:
            raise ValueError("Campaign result must identify a supported recomputed aggregate field")
        if not isinstance(entry.get("manuscript_location"), str) or not entry["manuscript_location"].strip():
            raise ValueError("Campaign result needs a manuscript location")
        value = aggregate
        for part in entry["json_pointer"].split("/")[1:]:
            value = value[part]
        expected, tolerance = entry.get("reported_value"), entry.get("absolute_tolerance")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in (value, expected, tolerance)) or tolerance < 0:
            raise ValueError("Campaign reported values and tolerances must be finite and explicit")
        results.append({**entry, "actual_value": value, "agrees": abs(value - expected) <= tolerance})
    return {"status": "MATCH" if results and all(r["agrees"] for r in results) else "MISMATCH", "results": results,
            "scope": "ten official splits on one shared graph; descriptive paired t interval; no independent-graph claim",
            "aggregate": aggregate, "relationship_to_manuscript": binding["relationship_to_manuscript"]}


def validate_record_coverage(name, rows, spec, contract):
    """Check the table's actual row keys, without inventing dataset coverage."""
    keys = contract.get("coverage_keys")
    if keys is None:
        if {r["dataset"] for r in rows} != set(contract["datasets"]):
            raise ValueError("Auxiliary record set has incomplete or unexpected dataset coverage")
        return {"keys": ["dataset"], "rows": len(contract["datasets"])}
    if spec["keys"] != keys:
        raise ValueError(f"{name}: record schema differs from the table coverage keys")
    observed = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        observed.setdefault(key, set()).add(row["quantity"])
    expected = {tuple(row) for row in contract["coverage_rows"]}
    if set(observed) != expected:
        raise ValueError(f"{name}: incomplete or unexpected table coverage; "
                         f"missing={sorted(expected-set(observed))}, unexpected={sorted(set(observed)-expected)}")
    for key, quantities in observed.items():
        if quantities != set(spec["quantities"]):
            raise ValueError(f"{name}: incomplete quantities for {key}; "
                             f"missing={sorted(set(spec['quantities'])-quantities)}")
    return {"keys": keys, "rows": len(expected), "quantities_per_row": len(spec["quantities"])}


def build_report(root=ROOT, *, budget_campaign=None, software_runs=(), manuscript_pdfs=None,
                 index_path=None, records_root=None, confirmation_campaign=None):
    root = Path(root).resolve()
    runs_index = Path(index_path).resolve() if index_path is not None else root / "runs/index.json"
    records_root = Path(records_root).resolve() if records_root is not None else root / "records"
    blockers = []
    def block(code, location, detail, status="PENDING"):
        blockers.append({"code": code, "location": str(location), "status": status, "detail": str(detail)})
    report = {"schema": SCHEMA, "publication_ready": False, "blockers": blockers,
              "inputs": {"repository_root": str(root), "run_index": str(runs_index), "records_root": str(records_root),
                         "confirmation_campaign": str(Path(confirmation_campaign).resolve()) if confirmation_campaign else None},
              "scope": "seven full official dataset archives, ten auxiliary record sets, original joint S14 confirmation, nine core manuscript numbers per dataset, and declared support/selector/budget invariants",
              "limits": ["Saved observations are numerically reconstructed; no benchmark is rerun by this audit.",
                         "The manuscript-binding gate covers the enumerated core results; auxiliary records trace their declared summary values.",
                         "Local run confirmation and the joint S14 campaign are distinct. The joint gate preserves original heterogeneous raw tests, reconstructs paired effects and replays Holm; it does not rerun arbitrary supplied original test code.",
                         "The E1-E20 scientific acceptance criteria in docs/PAPER_SUPPORT.md extend beyond the enumerated automated gates."]}
    try:
        report["manuscript_documents"] = inspect_sources(root, manuscript_pdfs)
        if report["manuscript_documents"]["status"] != "PASS":
            block("manuscript_document_identity", "paper/sources.json", report["manuscript_documents"]["issues"], "FAIL")
        elif not all(report["manuscript_documents"]["pdf_bytes_verified"].values()):
            report["limits"].append("Manuscript provenance includes reference-only metadata; PDF bytes are verified only when a declared bundled file or a local copy is supplied.")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        block("manuscript_document_identity", "paper/sources.json", exc, "FAIL")
    try:
        report["manuscript_reference"] = reference_selector(root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        block("manuscript_registry", "paper/", exc, "FAIL")
    try:
        index = json_read(runs_index) if runs_index.is_file() else {"schema": "cross-morse-runs-index-1", "runs": []}
        if index.get("schema") != "cross-morse-runs-index-1":
            raise ValueError("Unsupported runs index schema")
        entries = index["runs"]
        if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
            raise ValueError("Run index entries must be objects")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        block("run_index", runs_index, exc, "FAIL"); entries = []
    official, bindings, inspections = {}, {}, {}
    for dataset in DATASETS:
        matches = [r for r in entries if r.get("dataset") == dataset]
        if not matches:
            block("missing_official_archive", dataset, "No ingested full-profile official-data archive")
            continue
        try:
            if len(matches) != 1:
                raise ValueError("Ambiguous dataset archive entries")
            entry = matches[0]
            if entry.get("verified") is not True or not re.fullmatch(r"[0-9a-f]{64}", entry.get("archive_sha256", "")):
                raise ValueError("Archive ingestion identity/verification is missing")
            path = _inside(runs_index.parent, entry["path"])
            paths = sorted(path.glob("split_*")) if entry.get("kind") == "campaign" else [path]
            if dataset == "Roman-empire" and len(paths) != 10:
                raise ValueError("Roman-empire requires all ten official splits")
            verified = [audit_run(p, official_dataset=dataset, manuscript_root=root) for p in paths]
            inspections[dataset] = verified
            if not verified or any(r["status"] != "PASS" for r in verified):
                raise ValueError("Run scientific invariants failed; inspect the row-level run reports")
            official[dataset] = {"status": "PASS", "archive_sha256": entry["archive_sha256"], "runs": verified}
            binding_path = root / "paper/bindings" / f"{SLUGS[dataset]}_full.json"
            binding = json_read(binding_path)
            missing = binding_coverage(binding, campaign=len(paths) != 1, dataset=dataset, manuscript_root=root)
            if missing:
                raise ValueError("Manuscript result bindings missing or invalid: " + ", ".join(missing))
            result = check_campaign_binding(paths, verified, binding) if len(paths) != 1 else check_binding(paths[0], binding)
            bindings[dataset] = result
            if result["status"] != "MATCH":
                block("manuscript_number_mismatch", dataset, result, "FAIL")
        except (OSError, ValueError, KeyError, IndexError, TypeError, RuntimeError) as exc:
            code = "manuscript_numbers_unlinked" if dataset in official else "invalid_official_archive"
            block(code, dataset, exc, "FAIL")
    # Templates are not author declarations even when no run has yet been deposited.
    for dataset in DATASETS:
        if dataset not in official:
            block("manuscript_numbers_unlinked", dataset, "No validated official run to bind to the manuscript")
    report["official_archives"] = {"required": len(DATASETS), "validated": len(official), "datasets": official,
                                    "failed_inspections": {k: v for k, v in inspections.items() if k not in official}}
    report["manuscript_bindings"] = {"required": len(DATASETS), "validated": sum(x["status"] == "MATCH" for x in bindings.values()), "datasets": bindings}
    record_results = {}
    try:
        # Custom evidence locations use the checkout's registered contract, not
        # a substitute specification supplied alongside those records.
        specs = load_specs(records_root, root / "records/specs.json")
        for name, contract in RECORD_SETS.items():
            path = records_root / name
            if not path.is_dir():
                block("missing_auxiliary_set", name, "Required measured record set has not been deposited")
                continue
            try:
                result = verify_set(path, specs["sets"][name], specs)
                if result.get("synthetic"):
                    raise ValueError("Synthetic records are not accepted as auxiliary evidence")
                rows = read_summary(path, specs["sets"][name], specs["summary_columns"])
                coverage = validate_record_coverage(name, rows, specs["sets"][name], contract)
                record_results[name] = {"evidence_kind": "measured_auxiliary_records", "coverage": coverage, **result}
            except (OSError, ValueError, KeyError, TypeError) as exc:
                block("invalid_auxiliary_set", name, exc, "FAIL")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        block("record_specification", "records/specs.json", exc, "FAIL")
    report["auxiliary_records"] = {"required": len(RECORD_SETS), "validated": len(record_results), "sets": record_results}
    if budget_campaign is None:
        report["budget_invariants"] = {"status": "PENDING", "evidence_kind": "pending"}
        block("budget_campaign_missing", "E3-E6", "Supply --budget-campaign with the extracted E3-E6 campaign")
    else:
        try:
            from cm_campaigns.arm_campaign import check_campaign
            campaign = Path(budget_campaign).resolve()
            result = check_campaign(campaign, json_read(campaign / "study.json"), strict=True)
            report["budget_invariants"] = {"status": "PASS", "evidence_kind": "measured_official_data", **result}
        except (OSError, ValueError, KeyError, IndexError, TypeError, RuntimeError) as exc:
            report["budget_invariants"] = {"status": "FAIL", "error": str(exc)}
            block("budget_invariants", budget_campaign, exc, "FAIL")
    if confirmation_campaign is None:
        report["confirmation_family"] = {"status": "PENDING", "family_size": 9,
                                         "original_test_p_values_recomputed": False}
        block("confirmation_campaign_missing", "S14", "Supply --confirmation-campaign with original unrounded nine-test outputs, paired observations and pre-admission locks")
    else:
        try:
            from cm_campaigns.confirmation import check_campaign as check_confirmation_campaign, bind_official_runs
            result = check_confirmation_campaign(Path(confirmation_campaign), manuscript_root=root)
            report["confirmation_family"] = result
            if result["status"] != "PASS":
                block("confirmation_reference_mismatch", "S14", "Original S14 tests differ from printed manuscript values; inspect confirmation_family.manuscript_comparisons", "FAIL")
            linkage = bind_official_runs(result, {dataset: [Path(run["path"]) for run in evidence["runs"]]
                                                  for dataset, evidence in official.items()})
            result["official_run_linkage"] = linkage
            if linkage["status"] != "PASS":
                if result["status"] == "PASS":
                    result["status"] = "PENDING"
                block("confirmation_official_run_linkage", "S14", "Original S14 input identities must match all seven validated official archives, including every Roman split")
        except (OSError, ValueError, KeyError, IndexError, TypeError, RuntimeError) as exc:
            report["confirmation_family"] = {"status": "FAIL", "error": str(exc),
                                             "original_test_p_values_recomputed": False}
            block("confirmation_family", confirmation_campaign, exc, "FAIL")
    software = []
    for path in software_runs:
        try:
            result = audit_run(path)
            result["counts_toward_official_completeness"] = False
            software.append(result)
            if result["status"] != "PASS":
                block("software_run_invariant", path, "Support row invariants failed; inspect software_validation.support", "FAIL")
        except (OSError, ValueError, KeyError, IndexError, TypeError, RuntimeError) as exc:
            software.append({"path": str(path), "status": "FAIL", "evidence_kind": "software_validation", "error": str(exc),
                             "counts_toward_official_completeness": False})
            block("software_run_invariant", path, exc, "FAIL")
    report["software_validation"] = software
    report["publication_ready"] = not blockers
    report["status"] = "PASS" if report["publication_ready"] else "INCOMPLETE_OR_INVALID"
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--index", type=Path, help="run index; archive paths are relative to its parent")
    parser.add_argument("--records", type=Path, help="auxiliary record root; uses this checkout's record specification")
    parser.add_argument("--output", type=Path, help="JSON report path; the full report also goes to stdout")
    parser.add_argument("--budget-campaign", type=Path)
    parser.add_argument("--confirmation-campaign", type=Path, help="original nine-test S14 campaign, observations and policy locks")
    parser.add_argument("--software-run", action="append", type=Path, default=[])
    parser.add_argument("--manuscript-pdf", type=Path, help="optional local main manuscript copy for byte verification")
    parser.add_argument("--supplementary-pdf", type=Path, help="optional local supplement copy for byte verification")
    parser.add_argument("--check-examples", action="store_true", help="Reconstruct all present bundled synthetic runs")
    args = parser.parse_args(argv)
    paths = list(args.software_run)
    if args.check_examples:
        paths.extend(p for p in sorted((args.root / "examples/runs").glob("*")) if p.is_dir())
    local = {name: path for name, path in (("main", args.manuscript_pdf), ("supplementary", args.supplementary_pdf)) if path}
    report = build_report(args.root, budget_campaign=args.budget_campaign, software_runs=paths, manuscript_pdfs=local,
                          index_path=args.index, records_root=args.records, confirmation_campaign=args.confirmation_campaign)
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["publication_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
