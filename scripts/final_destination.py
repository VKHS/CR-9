#!/usr/bin/env python3
"""The final destination: the repository as it will stand once the measured outputs are published.

The registered availability statement and the printed manuscript text are fulfilled when the
reported runs, measurement-record sets and campaigns are published. This tool draws that end
point now, from what the repository already fixes, and replaces each projection by the realized
value as the artifacts appear:

  A. Design forecast. Printed values that the released code and the registered configurations
     determine before any measurement (candidate-family sizes, data-role sizes, statistical
     units, trajectories, resamples, depths, sensor settings), computed from configs/ and the
     runtime source and compared with the printed values now, before publication.
  B. Pre-registered data roles. The exact D_fit ... D_test memberships of the independent-graph
     benchmarks, computed with the runtime's own role functions (lifted from cross_morse/data.py
     without importing PyTorch) and hashed as the executed manifests hash them.
  C. Projected release. The contents, tier sizes and release parts of every run archive, built
     from the runtime's file layout at the registered scale and counted with the release's own
     rules (scripts/paper_artifacts.py: content_counts, is_heavy, PART_BYTES).
  D. Compliance at the final destination. The twelve registered items evaluated on the destination
     tree (scripts/destination.py) with the checks of scripts/statement_check.py, and every printed claim.
  E. The close-out sequence from the current state to the final destination.
  F. The destination tree: every file publication writes into the checkout (destination/), and the
     convergence check that compares each realized file with its projection.

Projected values are marked as projections (red in docs/FINAL_DESTINATION.md, a `projection` block in
every file of destination/) and never enter a comparison with the paper or the statement checks; the
real loaders refuse projected files.

    python -B scripts/final_destination.py                 # print a summary
    python -B scripts/final_destination.py --write         # write release/final_destination.json, docs/FINAL_DESTINATION.md and destination/
    python -B scripts/final_destination.py --check         # all of them are current
    python -B scripts/final_destination.py --runs-root DIR # completed runs against the forecast and the pre-registered roles
    python -B scripts/final_destination.py --converge      # every realized file of the checkout against destination/

NumPy and the standard library only.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_JSON = "release/final_destination.json"
OUT_MD = "docs/FINAL_DESTINATION.md"
SCHEMA = "cross-morse-final-destination-1"
DATASETS = ("ogbn-arxiv", "ogbn-proteins", "ogbn-products", "Roman-empire", "PascalVOC-SP", "COCO-SP", "Peptides-func")
STEMS = {"ogbn-arxiv": "arxiv", "ogbn-proteins": "proteins", "ogbn-products": "products",
         "Roman-empire": "roman_empire", "PascalVOC-SP": "pascalvoc_sp", "COCO-SP": "coco_sp",
         "Peptides-func": "peptides_func"}
SHORT = {"ogbn-arxiv": "arxiv", "ogbn-proteins": "proteins", "ogbn-products": "products", "Roman-empire": "Roman",
         "PascalVOC-SP": "Pascal", "COCO-SP": "COCO", "Peptides-func": "Peptides"}
INDEPENDENT = ("PascalVOC-SP", "COCO-SP", "Peptides-func")
ROMAN_SPLITS = 10
STAGES = ("prepare", "train", "explore", "fit", "select", "confirm", "admit", "evaluate")
GIB = 1024 ** 3
ZENODO_RECORD_DEFAULT = 50 * 10 ** 9      # https://help.zenodo.org/docs/deposit/manage-quota/
ZENODO_RECORD_MAXIMUM = 200 * 10 ** 9     # one-time increase for a single record (Zenodo support)
ZENODO_FILES = 100

# gzip (level 6) ratios assumed for the deterministic release archives, by content kind.
RATIO = {"float": 0.85, "checkpoint": 0.93, "json": 0.20, "text": 0.30, "index": 0.35, "mixed": 0.70}


class DestinationError(ValueError):
    pass


def _json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


# ===================================================================================== inputs
class Inputs:
    def __init__(self, root: Path):
        self.root = root
        self.configs = {d: _json(root / "configs" / f"{STEMS[d]}_full.json") for d in DATASETS}
        if any(c is None for c in self.configs.values()):
            raise DestinationError("every dataset needs configs/<stem>_full.json")
        self.constants = _json(root / "paper" / "registered_constants.json")
        self.reported = _json(root / "paper" / "reported_results.json")
        sources = (_json(root / "data" / "sources.json") or {}).get("datasets", [])
        self.sources = {r["dataset"]: r for r in sources if isinstance(r, dict) and "dataset" in r}
        self.statistics = self.constants["dataset_statistics_table_S6c_S23b"]
        self.profiles = self.constants["backbones_source_status"]["released_full_profiles"]
        self.multi = self.constants["data_roles_table_S2"]["multi_graph_counts"]

    def printed(self, group: str, *path, default=None):
        value = self.reported.get(group, {})
        for key in path:
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        return value


# ============================================================ A. design-determined forecast
def registry_size(config: dict) -> dict:
    """Candidate family of cross_morse/policies.py::make_registry for one configuration.

    The fixed-sequence family holds the first ``max_exploration_sequences`` distinct normalized
    exploration sequences; at the registered trajectory counts (9,600 or more randomized
    trajectories over ten actions) that bound is reached, so the count is exact.
    """
    L = int(config["model"]["depth"])
    depths = sorted(set([1, L] + [2 ** i for i in range(L.bit_length()) if 2 ** i < L]) - {L})
    registry, controller = config["registry"], config["controller"]
    phase = len(registry["phase_boundaries"]) * len(registry["phase_pairs"])
    fixed = int(registry["max_exploration_sequences"])
    profiles = list(config["sensor"]["profiles"])
    horizons, kappas = list(controller["horizons"]), list(controller["kappa_grid"])
    families = {"nominal": 1, "best_depth": len(depths), "learned_stop": 1, "phase": phase, "fixed_sequence": fixed,
                "mpc_ol": len(profiles) * len(horizons), "one_step": len(profiles),
                "radius": len(profiles) * len(kappas), "mpc_rh": len(profiles) * len(horizons)}
    sizes = [e for e in registry["ensemble_sizes"] if e <= len(config["training"]["seeds"])]
    entries = sum(families.values())
    return {"families": families, "entries": entries, "ensemble_sizes": sizes, "candidates": entries * len(sizes),
            "feedback_candidates": (families["one_step"] + families["radius"] + families["mpc_rh"]) * len(sizes)}


def runtime_role_functions(root: Path) -> dict:
    """The role functions of cross_morse/data.py, executed without importing PyTorch."""
    from cross_morse import split_contract
    names = {"_json_bytes", "_array_sha256", "_indices", "_validate_roles", "_make_roles"}
    text = (root / "cross_morse" / "data.py").read_text(encoding="utf-8")
    nodes = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != names:
        raise DestinationError("cross_morse/data.py no longer defines the role functions this forecast executes")
    source = "from __future__ import annotations\n" + "\n\n".join(ast.get_source_segment(text, n) for n in nodes)
    namespace = {"np": np, "json": json, "hashlib": hashlib, "Any": Any, "TRAIN_CUTS": split_contract.TRAIN_CUTS,
                 "ROLE_NAMES": split_contract.ROLE_NAMES, "LEAF_ROLES": split_contract.LEAF_ROLES,
                 "RERUN_GUIDANCE": split_contract.RERUN_GUIDANCE}
    exec(compile(source, "cross_morse/data.py (role functions)", "exec"), namespace)  # noqa: S102 - repository source
    return {**{name: namespace[name] for name in names}, "source_sha256": _sha(source.encode("utf-8"))}


def preregistered_roles(inputs: Inputs, functions: dict) -> dict:
    """Exact graph-level roles of the independent-graph benchmarks from the official split order."""
    out = {}
    for dataset in INDEPENDENT:
        counts = inputs.multi[dataset]
        n_train, n_valid, n_test = counts["official_train"], counts["D_admit"], counts["D_test"]
        total = n_train + n_valid + n_test
        if total != inputs.statistics[dataset]["graphs"]:
            raise DestinationError(f"{dataset}: official split sizes do not cover the registered graph count")
        config = inputs.configs[dataset]
        official = {"train": np.arange(0, n_train, dtype=np.int64),
                    "valid": np.arange(n_train, n_train + n_valid, dtype=np.int64),
                    "test": np.arange(n_train + n_valid, total, dtype=np.int64)}
        roles = functions["_make_roles"](official, total, int(config["seed"]),
                                         float(config["data"]["internal_val_fraction"]))
        hashes = {k: functions["_array_sha256"](v) for k, v in roles.items()}
        official_hashes = {k: functions["_array_sha256"](v) for k, v in official.items()}
        graph_level = dataset == "Peptides-func"
        out[dataset] = {
            "official_order": "LRGB train, val, test files concatenated in order (cross_morse/datasets.py::pack_lrgb)",
            "seed": int(config["seed"]), "internal_val_fraction": float(config["data"]["internal_val_fraction"]),
            "sizes": {k: int(len(v)) for k, v in roles.items()},
            "graph_role_hashes": hashes,
            "official_split_hashes": official_hashes,
            "role_hashes": hashes if graph_level else None,
            "role_hash_scope": ("graph-level labels: the row roles are the graph roles" if graph_level else
                                "node-level labels: row-role hashes need the official graph sizes (prepared graph_ptr) "
                                "and are compared after publication"),
        }
    return out


def forecast(inputs: Inputs, roles: dict) -> list[dict]:
    rows = []

    def row(display, quantity, dataset, value, printed, derivation, unit="", bound=False):
        if printed is None:
            status = "not printed"
        elif bound:
            # A measured quantity that the design bounds from above: the printed value must lie within the bound.
            status = "within bound" if isinstance(printed, (int, float)) and printed <= value else "exceeds bound"
        else:
            status = ("agrees" if (isinstance(value, (int, float)) and isinstance(printed, (int, float))
                                   and math.isclose(float(value), float(printed), rel_tol=0, abs_tol=1e-12))
                      or value == printed else "differs")
        rows.append({"display": display, "quantity": quantity, "dataset": dataset, "forecast": value,
                     "printed": printed, "status": status, "derivation": derivation, "unit": unit,
                     "kind": "bound" if bound else "value"})

    for dataset in DATASETS:
        config = inputs.configs[dataset]
        size = registry_size(config)
        f = size["families"]
        row("Table S20", "|Q| feasible family (registry bound)", dataset, size["candidates"],
            inputs.printed("supplementary_table_S20_selection_theorem_instantiation", dataset, "Q_size"),
            f"|Q| = |Q_feas| of main Eq. (18), the family Algorithm S4 runs over and reports (line 10): measured at "
            f"the select stage (select/decision.json joint_feasible). It is bounded by the registered family of "
            f"make_registry: {size['entries']} route configurations (nominal 1, best depth {f['best_depth']}, learned "
            f"stop 1, phase {f['phase']}, fixed sequence {f['fixed_sequence']}, MPC-OL {f['mpc_ol']}, one-step "
            f"{f['one_step']}, radius {f['radius']}, MPC-RH {f['mpc_rh']}) x {len(size['ensemble_sizes'])} ensemble "
            f"sizes", bound=True)
        row("Table S20", "selection resamples", dataset, int(config["selection"]["bootstrap_reps"]),
            inputs.printed("supplementary_table_S20_selection_theorem_instantiation", dataset, "resamples"),
            "configs: selection.bootstrap_reps")
        row("Table S7", "exploration trajectories", dataset, int(config["exploration"]["trajectories"]),
            inputs.printed("supplementary_table_S7_support_audit", dataset, "trajectories"),
            "configs: exploration.trajectories")
        printed_depth = (inputs.constants["backbones_table_S5"].get(dataset) or {}).get("depth")
        row("Table S5", "controlled depth L", dataset, int(config["model"]["depth"]), printed_depth, "configs: model.depth")
        row("Table S2", "optimization seeds", dataset, len(config["training"]["seeds"]),
            inputs.constants["data_roles_table_S2"]["plant_seeds"], "configs: training.seeds")
        folds = inputs.constants["data_roles_table_S2"]["selection_folds"]
        row("Table S2", "outer / inner selection folds", dataset,
            f"{config['selection']['outer_folds']}/{config['selection']['inner_folds']}",
            f"{folds['outer']}/{folds['inner']}", "configs: selection.outer_folds, selection.inner_folds")
        row("Table S2", "trajectory cross-fitting folds", dataset, int(config["controller"]["crossfit_folds"]),
            inputs.constants["data_roles_table_S2"]["exploration_trajectory_folds"], "configs: controller.crossfit_folds")
        sensor = (int(config["sensor"]["sketch_dim"]), int(config["sensor"]["sensing_period"]))
        printed_sensor = inputs.printed("supplementary_table_S6c_sensing_cost_identity", dataset)
        printed_pair = None
        if isinstance(printed_sensor, dict):
            k, p = printed_sensor.get("executed_sketch_width_k"), printed_sensor.get("executed_sensing_period_P_s")
            printed_pair = f"{k}/{p}" if k is not None and p is not None else None
        row("Table S6c", "sensor (k, P_s)", dataset, f"{sensor[0]}/{sensor[1]}", printed_pair,
            "configs: sensor.sketch_dim, sensor.sensing_period")
        if dataset not in INDEPENDENT:
            units = int(config["data"]["num_clusters"]) if dataset != "Roman-empire" else ROMAN_SPLITS
            printed_units = {"ogbn-arxiv": 32, "ogbn-proteins": 32, "ogbn-products": 64, "Roman-empire": 10}[dataset]
            text = inputs.constants["data_roles_table_S2"]["single_graph_units"][dataset]
            row("Table S1", "statistical units", dataset, units,
                printed_units if str(printed_units) in text else None,
                "configs: data.num_clusters (buffered cores)" if dataset != "Roman-empire" else
                "ten official splits (configs: data.split_index 0-9, one lifecycle each)")
        else:
            r = roles[dataset]["sizes"]
            counts = inputs.multi[dataset]
            for quantity, value, printed, display in (
                    ("D_confirm graphs", r["confirm"], counts["D_confirm"], "Table S2 / Note S26"),
                    ("D_compare graphs", r["compare"], counts["D_compare"], "Table S2 / Table S39"),
                    ("admission graphs J (D_admit)", r["admit"], counts["D_admit"], "Table S13"),
                    ("test graphs (D_test)", r["test"], counts["D_test"], "Table S37")):
                row(display, quantity, dataset, int(value), printed,
                    "runtime _make_roles on the official split (floors at 0.60/0.80/0.90/0.95)")
    return rows


# ====================================================================== C. projected release
def dataset_scale(inputs: Inputs, dataset: str, roles: dict) -> dict:
    stats, profile = inputs.statistics[dataset], inputs.profiles[dataset]
    outputs = int(inputs.sources[dataset]["official_outputs"])
    features = int(profile["input_dim"])
    if dataset in INDEPENDENT:
        graphs = int(stats["graphs"])
        nodes = graphs * float(stats["mean_nodes_per_graph"])
        edges = graphs * float(stats["mean_reported_edges_per_graph"])
        per_graph = 1.0 if dataset == "Peptides-func" else float(stats["mean_nodes_per_graph"])
        sizes = roles[dataset]["sizes"]
        rows = {role: sizes[role] * per_graph for role in ("select", "confirm", "admit", "test")}
        units = {role: sizes[role] for role in ("select", "confirm", "admit", "test")}
    else:
        graphs, nodes = 1, float(stats["nodes"])
        edges = 2.0 * float(stats["reported_edges"])  # loaded bidirectional adjacency
        rows = {"all": nodes}
        units = {"clusters": int(inputs.configs[dataset]["data"]["num_clusters"])}
    edge_features = {"ogbn-proteins": 8, "PascalVOC-SP": 2, "COCO-SP": 2, "Peptides-func": 4}.get(dataset, 0)
    multilabel = inputs.sources[dataset]["task_type"] == "multilabel"
    label_rows = graphs if inputs.sources[dataset]["prediction_axis"] == "graph" else nodes
    return {"nodes": nodes, "edges": edges, "graphs": graphs, "features": features, "outputs": outputs,
            "edge_features": edge_features, "multilabel": multilabel, "label_rows": label_rows,
            "rows": rows, "units": units, "parameters": int(profile["parameter_count"]),
            "q0_parameters": int(sum(v for k, v in profile["parameter_components"].items() if k in ("encoder", "head")))}


def run_families(inputs: Inputs, dataset: str, roles: dict) -> list[dict]:
    """File families of one full lifecycle run directory (one Roman split), with counts and bytes."""
    config = inputs.configs[dataset]
    scale = dataset_scale(inputs, dataset, roles)
    size = registry_size(config)
    L, T = int(config["model"]["depth"]), int(config["exploration"]["trajectories"])
    S, F = len(config["training"]["seeds"]), int(config["controller"]["crossfit_folds"])
    profiles = len(config["sensor"]["profiles"])
    epochs = int(config["training"]["epochs"])
    C, Q, entries = scale["outputs"], size["candidates"], size["entries"]
    independent = dataset in INDEPENDENT
    fam: list[dict] = []

    def add(path, count, each, kind):
        if count > 0 and each > 0:
            fam.append({"path": path, "count": int(round(count)), "bytes": float(each), "kind": kind})

    for name, each in (("config.json", 6e3), ("action_registry.json", 3e3), ("environment.json", 8e3),
                       ("policy_lock.json", 5e3), ("run_state.json", 2e3)):
        add(name, 1, each, "json")
    add("stages/train.json", 1, 30e3, "json")
    add("stages/select.json", len(STAGES) - 1, 4e3, "json")
    add("provenance/stages/04_select.json", len(STAGES), 45e3, "json")
    add("provenance/evaluator.py", 1, 55e3, "text")
    add("provenance/frozen_models.json", 1, 6e3, "json")
    # prepared data and split indices
    graph_bytes = (scale["nodes"] * scale["features"] * 4 + scale["edges"] * 16 + scale["edges"] * scale["edge_features"] * 4
                   + (scale["label_rows"] * scale["outputs"] * 4 if scale["multilabel"] else scale["label_rows"] * 8)
                   + (scale["graphs"] + 1) * 16 * (scale["graphs"] > 1))
    add("prepared/graph.npz", 1, graph_bytes, "mixed")
    unit_count = scale["label_rows"]
    add("prepared/splits.npz", 1, unit_count * 8 * 1.6, "index")
    add("prepared/official_splits.npz", 1, unit_count * 8, "index")
    add("prepared/clusters.npz", 1, unit_count * 8, "index")
    add("prepared/metadata.json", 1, 12e3, "json")
    if independent:
        add("prepared/graph_splits.npz", 1, scale["graphs"] * 8 * 1.6, "index")
    # frozen checkpoints (best: weights; last: weights and optimizer state)
    for prefix, params in (("plant", scale["parameters"]), ("q0", scale["q0_parameters"])):
        add(f"train/{prefix}_seed1_best.pt", S, 4 * params + 8e3, "checkpoint")
        add(f"train/{prefix}_seed1_last.pt", S, 3.9 * 4 * params + 16e3, "checkpoint")
        add(f"train/{prefix}_seed1_history.json", S, 150 * epochs, "json")
        add(f"train/{prefix}_seed1_done.json", S, 1e3, "json")
    add("train/manifest.json", 1, 12e3, "json")
    add("train/model_inventory.json", 1, 25e3, "json")
    add("train/entry_initial.json", 1, 6e3, "json")
    add("train/entry_depth_values.npz", 1, (L + 1) * S * 8 * 4, "float")
    # propensity-logged exploration trajectories
    add("explore/trajectory_000000.npz", T, 4702 + 378 * L, "mixed")
    add("explore/trajectory_000000.json", T, 5573 + 126 * L, "json")
    add("explore/manifest.json", 1, 2e3, "json")
    add("explore/nominal_reference.npz", 1, S * (L + 1) * 8 * 4, "float")
    # controller packages: per sensor profile, the deployed package, the final fit and each cross-fitting fold
    add("fit/full/controller.pt", profiles, 0.60e6, "checkpoint")
    add("fit/full/final_fit/critic.last.pt", profiles * 5, 0.60e6, "checkpoint")
    add("fit/full/folds/fold_00/critic.last.pt", profiles * 5 * F, 0.60e6, "checkpoint")
    add("fit/full/oof_predictions.npz", profiles, T * L * 37, "float")
    add("fit/full/calibration_assessment.npz", profiles, T * L * 16, "float")
    add("fit/full/audit.json", profiles * 3, 50e3, "json")
    add("fit/registry.json", 1, 30e3, "json")
    add("fit/exploration_hashes.json", 1, T * 120, "json")
    # selection, confirmation and admission records
    select_units = scale["units"].get("select", scale["units"].get("clusters"))
    add("select/candidates.json", 1, Q * 730, "json")
    add("select/candidate_ledger.csv", 1, Q * 300, "text")
    add("select/decision.json", 1, Q * 800 + 20e3, "json")
    add("select/unit_scores.npz", 1, select_units * Q * 16, "float")
    add("select/entry_complete.json", 1, 5e3, "json")
    if independent:
        add("select/metric_observations.npz", 1, select_units * Q * 16, "float")
    add("confirm/decision.json", 3, 15e3, "json")
    admit_units = scale["units"].get("admit", scale["units"].get("clusters"))
    add("admit/decision.json", 2, 10e3, "json")
    add("admit/paired_units.npz", 1, admit_units * 48, "float")
    # candidate, member and final predictions
    member_json = (2000 + 1500 * L) if not independent else (600 + 92 * L)
    if not independent:
        array = scale["rows"]["all"] * C * 4 + 128
        add("predictions/0000000000000000000a.npy", Q, array, "float")
        add("predictions/0000000000000000000a.json", Q, 1.8 * member_json, "json")
        add("member_predictions/000000000000000a_seed1.npy", entries * S, array, "float")
        add("member_predictions/000000000000000a_seed1.json", entries * S, member_json, "json")
        final_rows = scale["rows"]["all"]
        test_rows = scale["rows"]["all"] * 0.3
    else:
        for role, candidates, members in (("select", Q, entries * S), ("confirm", 8, 64), ("admit", 2, 24)):
            array = scale["rows"][role] * C * 4 + 128
            add(f"predictions/{role}/0000000000000000000a.npy", candidates, array, "float")
            add(f"predictions/{role}/0000000000000000000a.json", candidates, 1.8 * member_json, "json")
            add(f"member_predictions/{role}/000000000000000a_seed1.npy", members, array, "float")
            add(f"member_predictions/{role}/000000000000000a_seed1.json", members, member_json, "json")
        final_rows = test_rows = scale["rows"]["test"]
    add("final/qD_probabilities.npy", 4, final_rows * C * 4, "float")
    add("final/test_predictions.npz", 1, test_rows * (C * 16 + 32), "float")
    add("final/paired_test_units.npz", 1, max(admit_units, 64) * 64, "float")
    add("final/results_summary.json", 1, 20e3, "json")
    add("final/results.csv", 1, 2e3, "text")
    return fam


def project_run(inputs: Inputs, dataset: str, roles: dict) -> dict:
    from scripts import paper_artifacts
    families = run_families(inputs, dataset, roles)
    copies = ROMAN_SPLITS if dataset == "Roman-empire" else 1
    totals = {"files": 0, "raw_complete": 0.0, "raw_analysis": 0.0, "packed_complete": 0.0, "packed_analysis": 0.0}
    contents = {key: 0 for key in paper_artifacts.content_counts([])}
    largest = []
    for f in families:
        count, raw = f["count"] * copies, f["bytes"] * f["count"] * copies
        packed = raw * RATIO[f["kind"]]
        heavy = paper_artifacts.is_heavy(f["path"])
        totals["files"] += count
        totals["raw_complete"] += raw
        totals["packed_complete"] += packed
        if not heavy:
            totals["raw_analysis"] += raw
            totals["packed_analysis"] += packed
        for key, value in paper_artifacts.content_counts([{"path": f["path"]}]).items():
            contents[key] += value * count
        largest.append((packed, f["path"].split("/")[0], heavy))
    by_area: dict[str, float] = {}
    for packed, area, _ in largest:
        by_area[area] = by_area.get(area, 0.0) + packed
    parts = {tier: max(1, math.ceil(totals[f"packed_{tier}"] / paper_artifacts.PART_BYTES)) for tier in ("analysis", "complete")}
    complete = totals["packed_complete"]
    zenodo = ("fits the default 50 GB record" if complete <= ZENODO_RECORD_DEFAULT else
              "needs the additional quota (one record up to 200 GB)" if complete <= ZENODO_RECORD_MAXIMUM else
              "exceeds the largest single-record Zenodo quota")
    return {"dataset": dataset, "artifact": f"run:{dataset}", "copies": copies, "files": totals["files"],
            "bytes": {k: int(v) for k, v in totals.items() if k != "files"}, "parts": parts, "contents": contents,
            "areas_packed_complete": {k: int(v) for k, v in sorted(by_area.items(), key=lambda kv: -kv[1])},
            "zenodo_complete_tier": zenodo, "projected": True}


def realized_run(entry: dict) -> dict | None:
    """The published values of a run artifact from release/paper_artifacts.json."""
    from scripts import paper_artifacts
    if entry.get("status") != paper_artifacts.PUBLISHED:
        return None
    tiers = entry.get("tiers") or {}
    return {"files": entry.get("files"), "bytes_unpacked": entry.get("bytes"), "contents": entry.get("contents"),
            "packed": {t: (tiers.get(t) or {}).get("bytes") for t in ("analysis", "complete")},
            "parts": {t: len((tiers.get(t) or {}).get("parts") or []) for t in ("analysis", "complete")},
            "projected": False}


# ======================================================== B'. realized design values of runs
def role_rule(sizes: dict, fraction: float) -> dict:
    """The role sizes that the registered floors give for the realized official training count."""
    from cross_morse.split_contract import TRAIN_CUTS
    n = sum(int(sizes[k]) for k in ("fit", "explore", "select", "confirm", "compare"))
    cuts = [0] + [int(n * f) for f in TRAIN_CUTS] + [n]
    fit = cuts[1] - cuts[0]
    val = max(1, int(fit * fraction))
    return {"fit": fit, "explore": cuts[2] - cuts[1], "select": cuts[3] - cuts[2], "confirm": cuts[4] - cuts[3],
            "compare": cuts[5] - cuts[4], "fit_val": val, "fit_train": fit - val}


def runs_report(runs_root: Path, inputs: Inputs, roles: dict, forecast_rows: list[dict]) -> list[dict]:
    """Completed runs against what the design fixes: each run against the registry and role rule of its own
    configuration and, for full benchmark runs, against the registered configuration, the released runtime,
    the pre-registered roles and the printed |Q|."""
    from scripts._paper_common import runtime_source_fingerprint
    from scripts.destination import canonical_sha256, effective_configs, OFFICIAL_NODE_SPLITS
    if not runs_root.is_dir():
        raise DestinationError(f"{runs_root} is not a directory of completed runs")
    printed_q = {r["dataset"]: r["printed"] for r in forecast_rows if r["quantity"].startswith("|Q|")}
    registered = {d: {canonical_sha256(c) for c in configs} for d, configs in effective_configs(inputs.root).items()}
    fingerprint = runtime_source_fingerprint(inputs.root)
    report = []
    for run in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        members = sorted(run.glob("split_*")) if (run / "split_00").is_dir() else [run]
        for member in members:
            config = _json(member / "config.json")
            decision = _json(member / "select" / "decision.json")
            if config is None or decision is None:
                continue
            dataset = config.get("benchmark_dataset") or config.get("dataset")
            if dataset not in DATASETS:
                continue
            benchmark = bool(config.get("dataset") == dataset and config.get("profile") == "full" and config.get("scientific_run"))
            metadata = _json(member / "prepared" / "metadata.json") or {}
            state = _json(member / "run_state.json") or {}
            flags = decision.get("joint_feasible")
            feasible = sum(1 for f in flags if f is True) if isinstance(flags, list) else None
            row = {"dataset": dataset, "run": member.relative_to(runs_root).as_posix(), "profile": config.get("profile"),
                   "benchmark": benchmark, "registered_candidates": decision.get("candidate_count"),
                   "registered_candidates_from_configuration": registry_size(config)["candidates"],
                   "feasible_family": feasible, "printed_Q": printed_q.get(dataset) if benchmark else None,
                   "bootstrap_replicates": decision.get("bootstrap_replicates")}
            row["registry_agrees"] = row["registered_candidates"] == row["registered_candidates_from_configuration"]
            sizes = metadata.get("graph_split_sizes") or metadata.get("split_sizes")
            if isinstance(sizes, dict) and all(k in sizes for k in ("fit", "explore", "select", "confirm", "compare")):
                rule = role_rule(sizes, float((config.get("data") or {}).get("internal_val_fraction", 0.1)))
                row["role_sizes_follow_the_rule"] = all(int(sizes.get(k, -1)) == v for k, v in rule.items())
            if benchmark:
                row["configuration_is_registered"] = canonical_sha256(config) in registered[dataset]
                row["runtime_is_released"] = state.get("source_sha256") == fingerprint
                if dataset in OFFICIAL_NODE_SPLITS and isinstance(sizes, dict):
                    n_train = sum(int(sizes[k]) for k in ("fit", "explore", "select", "confirm", "compare"))
                    row["official_sizes_as_projected"] = (n_train, int(sizes.get("admit", -1)), int(sizes.get("test", -1))) \
                        == OFFICIAL_NODE_SPLITS[dataset]
            if dataset in roles:
                executed = metadata.get("graph_role_hashes")
                row["graph_roles_match_preregistration"] = (
                    "not a benchmark run" if not benchmark else None if executed is None
                    else executed == roles[dataset]["graph_role_hashes"])
            report.append(row)
    return report


# ============================================================ D. compliance at the destination
RELEASE_ROUTE = {"M25", "M29", "M30", "R01", "S05", "S06"}
ZENODO_ROUTE = {"M27", "S03"}


def printed_at_destination(printed: dict) -> list[dict]:
    from scripts import printed_statement_check as psc
    rows = []
    for claim in printed["claims"]:
        if claim["status"] == psc.FULFILLED:
            projected, note = psc.FULFILLED, "fulfilled now"
        elif claim["id"] in RELEASE_ROUTE:
            projected, note = psc.FULFILLED, "published with the release paper-artifacts"
        elif claim["id"] in ZENODO_ROUTE:
            projected, note = ("Route-dependent", "route R: fulfilled with the aligned wording; route Z: fulfilled once "
                               "the run archives are deposited and registered (section E)")
        else:
            projected, note = ("Route-dependent", "route R: fulfilled with the aligned wording; route Z: needs record-set "
                               "deposit registration (section E)")
        rows.append({"id": claim["id"], "text": claim["text"], "now": claim["status"], "projected": projected,
                     "note": note})
    return rows


def compliance(registered: dict, printed: dict, statement: dict) -> dict:
    rows = [{"id": r["id"], "phrase": r["phrase"], "now": r["now"], "projected": r["destination"],
             "note": "fulfilled now" if r["now"] == "fulfilled" else r["evidence"], "problems": r["problems"]}
            for r in statement["items"]]
    return {"registered": rows, "printed": printed_at_destination(printed),
            "now": {"registered": registered["counts"], "printed": printed["counts"]},
            "destination": {"registered": statement["fulfilled"], "of": statement["total"]}}


# ===================================================================================== build
def build_all(root: Path = ROOT, runs_root: Path | None = None) -> tuple[dict, dict]:
    """The final-destination document and the destination tree (scripts/destination.py)."""
    from scripts import paper_artifacts, statement_check, printed_statement_check
    from scripts import destination
    inputs = Inputs(root)
    functions = runtime_role_functions(root)
    roles = preregistered_roles(inputs, functions)
    rows = forecast(inputs, roles)
    manifest = paper_artifacts.load_manifest(root / "release" / "paper_artifacts.json")
    entries = {a["id"]: a for a in manifest["artifacts"]}
    runs = []
    for dataset in DATASETS:
        projection = project_run(inputs, dataset, roles)
        projection["realized"] = realized_run(entries.get(f"run:{dataset}", {}))
        runs.append(projection)
    required = paper_artifacts.required_artifacts(root)
    specs = (_json(root / "records" / "specs.json") or {}).get("sets", {})
    others = []
    for artifact in required:
        if artifact["kind"] == "run":
            continue
        name = artifact["id"].split(":", 1)[1]
        spec = specs.get(name, {})
        others.append({"artifact": artifact["id"], "kind": artifact["kind"],
                       "title": spec.get("title") or {"budget": "E3-E6 budget-matched campaign",
                                                       "confirmation": "S14 joint confirmation campaign",
                                                       "gpu": "observed GPU environment"}.get(name, name),
                       "tables": spec.get("manuscript_tables", []),
                       "status": (entries.get(artifact["id"]) or {}).get("status", paper_artifacts.AWAITING)})
    totals = {tier: sum(r["bytes"][f"packed_{tier}"] for r in runs) for tier in ("analysis", "complete")}
    parts = {tier: sum(r["parts"][tier] for r in runs) for tier in ("analysis", "complete")}
    configs = destination.effective_configs(root)
    node_roles = {d: {"official_sizes": dict(zip(("train", "valid", "test"), destination.OFFICIAL_NODE_SPLITS[d])),
                      "sizes": destination.node_roles(functions, d, configs[d][0])}
                  for d in DATASETS if d not in INDEPENDENT}
    document = {
        "schema": SCHEMA,
        "about": ("Projected end state of the repository after its measured outputs are published. Projected values "
                  "carry projected=true; they are replaced by realized values from release/paper_artifacts.json and "
                  "never enter a comparison with the paper. destination/ holds every file publication writes into the "
                  "checkout, projected (scripts/destination.py)."),
        "inputs": {"configs": "configs/<stem>_full.json", "constants": "paper/registered_constants.json",
                   "printed": "paper/reported_results.json", "datasets": "data/sources.json",
                   "role_functions_sha256": functions["source_sha256"]},
        "forecast": rows,
        "forecast_counts": {s: sum(1 for r in rows if r["status"] == s)
                            for s in ("agrees", "within bound", "differs", "exceeds bound", "not printed")},
        "preregistered_roles": roles,
        "node_roles": node_roles,
        "runs": runs,
        "runs_totals": {"packed_bytes": {k: int(v) for k, v in totals.items()}, "parts": parts, "projected": True},
        "other_artifacts": others,
        "zenodo_limits": {"default_record_bytes": ZENODO_RECORD_DEFAULT, "largest_record_bytes": ZENODO_RECORD_MAXIMUM,
                          "files_per_record": ZENODO_FILES,
                          "sources": ["https://help.zenodo.org/docs/deposit/manage-quota/",
                                      "https://support.zenodo.org/help/en-gb/1-upload-deposit/80-what-are-the-size-limitations-of-zenodo"]},
        "release_limits": {"part_bytes": paper_artifacts.PART_BYTES, "asset_limit": "2 GiB per asset; no total release size limit"},
    }
    tree = destination.build_tree(root, document, functions)
    registered = statement_check.evaluate(root)
    printed = printed_statement_check.evaluate(root, registered=registered)
    statement = destination.projected_statement(root, tree, registered)
    document["compliance"] = compliance(registered, printed, statement)
    ledger = tree["paper/source_data/INDEX.json"]["displays"]
    tree["STATEMENT_COMPLIANCE.md"] = destination.render_compliance(
        statement, registered["statement"], ledger, document["compliance"]["printed"])
    tree["README.md"] = destination.render_readme(tree, statement)
    figures = [r for r in ledger if r["kind"] == "figure" and r["scope"] != "illustration"]
    document["displays"] = {
        "tables": sum(1 for r in ledger if r["kind"] == "table"),
        "table_targets": sum(r["registered_targets"] for r in ledger if r["kind"] == "table"),
        "quantitative_figures": len(figures),
        "figures_with_contracts": sum(1 for r in figures if r["contract"]["now"] == "registered"),
        "figure_targets": sum(r["registered_targets"] for r in figures),
        "figures_to_register": {r["id"]: r["contract"]["work"] for r in figures if r["contract"]["now"] == "to register"},
    }
    from scripts._paper_common import runtime_source_fingerprint
    document["destination"] = {"tree": destination.TREE + "/", "files": sorted(tree),
                               "fingerprint": runtime_source_fingerprint(root),
                               "statement": {"fulfilled": statement["fulfilled"], "of": statement["total"]}}
    if runs_root is not None:
        document["runs_report"] = runs_report(runs_root, inputs, roles, rows)
    return document, tree


def build(root: Path = ROOT, runs_root: Path | None = None) -> dict:
    return build_all(root, runs_root)[0]


# ==================================================================================== render
def red(text) -> str:
    """A projected value: red in GitHub's Markdown math."""
    safe = str(text).replace("\\", "").replace("{", "").replace("}", "").replace("%", r"\%").replace("_", r"\_")
    return rf"$\color{{red}}{{\textsf{{{safe}}}}}$"


def human(n: float) -> str:
    for unit, size in (("TB", 10 ** 12), ("GB", 10 ** 9), ("MB", 10 ** 6), ("kB", 10 ** 3)):
        if n >= size:
            value = n / size
            return f"{value:.2g} {unit}" if value < 10 else f"{value:.0f} {unit}"
    return f"{int(n)} B"


def cell(text) -> str:
    """Text inside a Markdown table cell."""
    return str(text).replace("|", r"\|")


def count(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def render(doc: dict) -> str:
    fc = doc["forecast_counts"]
    comp = doc["compliance"]
    conflicts = [r for r in doc["forecast"] if r["status"] in ("differs", "exceeds bound")]
    label = {"fulfilled": "Fulfilled", "awaiting_measured_outputs": "Awaiting measured outputs", "missing": "Missing"}
    L = ["# The final destination", "",
         "[Statement compliance](../STATEMENT_COMPLIANCE.md) · [destination/](../destination/README.md) · [Publishing](PUBLICATION.md) · [Producing the outputs](PRODUCTION.md) · [Evidence registration](EVIDENCE_REGISTRATION.md)",
         "",
         "This page draws the repository as it will stand once the measured outputs of the reported experiments are "
         "published, from what the repository already fixes: the released code, the registered configurations, the "
         "paper's registered constants and the publication rules. [`destination/`](../destination/README.md) holds "
         "every file that publication writes into the checkout, projected now at the path it will occupy. Values in "
         + red("red") + " are projections. Each one is replaced by the realized value when its artifact is published; "
         "projections never enter `make paper`, the statement checks or any comparison with the paper, and the real "
         "loaders refuse the projected files. Generated by `scripts/final_destination.py`.", ""]
    # ---------------------------------------------------------------- at a glance
    now_r, now_p = comp["now"]["registered"], comp["now"]["printed"]
    dest = comp["destination"]
    proj_p = sum(1 for r in comp["printed"] if r["projected"] == "fulfilled")
    n_printed = len(comp["printed"])
    totals = doc["runs_totals"]
    shown = doc["displays"]
    L += ["## At a glance", "",
          "| | Now | At the final destination |", "|---|---|---|",
          f"| Registered statement (12 items) | {now_r['fulfilled']} fulfilled, {now_r['awaiting_measured_outputs']} awaiting measured outputs, {now_r['missing']} missing | **{red(str(dest['registered']) + ' of ' + str(dest['of']) + ' fulfilled')}** ([destination/STATEMENT_COMPLIANCE.md](../destination/STATEMENT_COMPLIANCE.md)) |",
          f"| Printed manuscript text ({n_printed} claims) | {now_p['fulfilled']} fulfilled, {now_p['awaiting_measured_outputs']} awaiting, {now_p['missing']} missing | {red(str(proj_p) + ' of ' + str(n_printed))} fulfilled with the current wording; {red(str(n_printed) + ' of ' + str(n_printed))} with route R (section E) |",
          f"| Design-determined printed values (section A) | {fc['agrees']} agree, {fc['within bound']} measured within their design bound, {fc['not printed']} not printed | {fc['differs'] + fc['exceeds bound']} conflicts |",
          f"| Numbered displays rebuilt by `make paper` | {shown['tables']} tables and {shown['quantitative_figures']} quantitative figures awaiting their artifacts; {shown['table_targets'] + shown['figure_targets']} printed values registered as comparison targets | {red('all ' + str(shown['tables'] + shown['quantitative_figures'] + 1) + ' rebuilt, every compared value agrees')}; {red(str(len(shown['figures_to_register'])) + ' figure contracts')} registered (section D) |",
          f"| Run archives (seven datasets; Roman ten splits) | awaiting | {red(human(totals['packed_bytes']['analysis']))} analysis tier, {red(human(totals['packed_bytes']['complete']))} complete tier, {red(count(totals['parts']['complete']) + ' release parts')} |",
          ""]
    if conflicts:
        L += [f"**Resolve before publication.** {len(conflicts)} printed values that the released code fixes by design "
              f"conflict with what it will produce ({', '.join(sorted({r['display'] + ' ' + r['quantity'] for r in conflicts}))}). "
              "`make paper` exits 0 only when every compared value agrees (section A).", ""]
    # ---------------------------------------------------------------- A
    L += ["## A. Design forecast: printed values the released code fixes before any measurement", "",
          "Each value below is computed from `configs/<dataset>_full.json`, the runtime source and the official split "
          "sizes, not from a run. A completed run reproduces it exactly; `make final-destination-runs RUNS_ROOT=<runs>` "
          "checks completed runs against this forecast before they are published. A *bound* row is a measured quantity "
          "that the design limits: the printed value must lie within the bound.", "",
          "| Display | Quantity | Dataset | Released code gives | Printed | Agreement |", "|---|---|---|---|---|---|"]
    for r in doc["forecast"]:
        agreement = {"agrees": "agrees", "differs": "**differs**", "not printed": "not printed",
                     "within bound": "within the bound (measured)", "exceeds bound": "**exceeds the bound**"}[r["status"]]
        given = f"at most {r['forecast']}" if r.get("kind") == "bound" else r["forecast"]
        L.append(f"| {r['display']} | {cell(r['quantity'])} | {SHORT[r['dataset']]} | {cell(given)} | "
                 f"{'—' if r['printed'] is None else r['printed']} | {agreement} |")
    q_rows = [r for r in doc["forecast"] if r["quantity"].startswith("|Q|")]
    if q_rows:
        L += ["", "**Table S20 |Q| and r_max.** " + q_rows[0]["derivation"] + ". The Table S20 generator reports |Q_feas| "
              "(`paper/build/displays/run_audits.py::feasible_family_size`, replayed with Algorithm S4) and the registered "
              "family size beside it in its source data. The runtime's `nested_selection` takes the studentized maximum "
              "over Q_feas, as Algorithm S4 does, so r_max, |Q_s| and the slack are those of the printed method; "
              "`select/decision.json` records the family (`max_t_family`), and records sealed before carry none and replay "
              "with every executed configuration. Whether the original runs used Q_feas is decided from their records "
              "([consistency item K1](CONSISTENCY.md)).", ""]
    # ---------------------------------------------------------------- B
    L += ["## B. Pre-registered data roles", "",
          "The official split order and the registered seed fix every role before any label is read. These memberships "
          "are computed with the runtime's own role functions (`_make_roles`, `_array_sha256` of `cross_morse/data.py`) "
          "and hashed exactly as the executed manifests hash them (`prepared/metadata.json`). For the independent-graph "
          "benchmarks the official LRGB file order fixes the memberships themselves:", ""]
    L += ["| Dataset | fit (train / val) | explore | select | confirm | compare | admit | test |", "|---|---|---|---|---|---|---|---|"]
    for dataset, value in doc["preregistered_roles"].items():
        s = value["sizes"]
        L.append(f"| {dataset} | {s['fit']} ({s['fit_train']} / {s['fit_val']}) | {s['explore']} | {s['select']} | "
                 f"{s['confirm']} | {s['compare']} | {s['admit']} | {s['test']} |")
    L += ["", "| Dataset | Role | SHA-256 of the graph-level role array |", "|---|---|---|"]
    for dataset, value in doc["preregistered_roles"].items():
        for role in ("confirm", "compare", "admit", "test", "select", "explore", "fit"):
            L.append(f"| {dataset} | {role} | `{value['graph_role_hashes'][role]}` |")
    L += ["", f"Seed {next(iter(doc['preregistered_roles'].values()))['seed']}; internal fit-validation fraction 0.1. "
          "Peptides-func has graph-level labels, so these are also its row-role hashes; for PascalVOC-SP and COCO-SP the "
          "node-row hashes follow from the official graph sizes after preparation. The same lifted functions reproduce "
          "every role hash that the two bundled runs recorded when they were prepared "
          "(`tests/test_final_destination.py`).", "",
          "For the single-graph benchmarks the role sizes follow from the provider's official split sizes by the same "
          "floors; the node identifiers, and with them the hashes, are fixed when the provider files are prepared:", "",
          "| Dataset | official train / valid / test | fit (train / val) | explore | select | confirm | compare |",
          "|---|---|---|---|---|---|---|"]
    for dataset, value in doc["node_roles"].items():
        o, s = value["official_sizes"], value["sizes"]
        L.append(f"| {dataset}{' (each of ten splits)' if dataset == 'Roman-empire' else ''} | "
                 f"{red(count(o['train']) + ' / ' + count(o['valid']) + ' / ' + count(o['test']))} | "
                 f"{red(count(s['fit']) + ' (' + count(s['fit_train']) + ' / ' + count(s['fit_val']) + ')')} | "
                 f"{red(count(s['explore']))} | {red(count(s['select']))} | {red(count(s['confirm']))} | {red(count(s['compare']))} |")
    L += ["", "The official sizes are the providers' published splits (OGB time, species and sales-ranking splits; the "
          "ten 50/25/25 masks of Roman-empire); `make final-destination-runs` recomputes the roles from the realized "
          "official sizes of each completed run.", ""]
    # ---------------------------------------------------------------- C
    L += ["## C. The published release", "",
          "Each run archive is projected from the runtime's file layout at the registered scale (seeds, trajectories, "
          "depth, candidate family, controller folds, official dataset sizes) and counted with the release's own rules "
          "(`content_counts`, `is_heavy`, 1 900 MiB parts; each trajectory is one `.npz` and one `.json` file). Sizes "
          "assume gzip ratios by content kind and are accurate to roughly a factor of 1.5; counts follow the layout. "
          "The whole manifest at publication is projected in "
          "[`destination/release/paper_artifacts.json`](../destination/release/paper_artifacts.json).", "",
          "| Artifact | Files | Checkpoints | Prediction files | Trajectory files | Statistical outputs | Analysis tier | Complete tier | Parts | One Zenodo record |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for run in doc["runs"]:
        real = run.get("realized")
        if real:
            L.append(f"| `{run['artifact']}` | {count(real['files'])} | {real['contents']['checkpoints']} | "
                     f"{real['contents']['predictions']} | {real['contents']['propensity_logged_trajectories']} | "
                     f"{real['contents']['statistical_outputs']} | {human(real['packed']['analysis'] or 0)} | "
                     f"{human(real['packed']['complete'] or 0)} | {real['parts']['complete']} | — |")
            continue
        c, b = run["contents"], run["bytes"]
        L.append(f"| `{run['artifact']}` | {red(count(run['files']))} | {red(count(c['checkpoints']))} | "
                 f"{red(count(c['predictions']))} | {red(count(c['propensity_logged_trajectories']))} | "
                 f"{red(count(c['statistical_outputs']))} | {red(human(b['packed_analysis']))} | "
                 f"{red(human(b['packed_complete']))} | {red(run['parts']['complete'])} | {run['zenodo_complete_tier']} |")
    t = doc["runs_totals"]
    L += [f"| **all runs** | | | | | | {red(human(t['packed_bytes']['analysis']))} | {red(human(t['packed_bytes']['complete']))} | {red(count(t['parts']['complete']))} | |", ""]
    biggest = max(doc["runs"], key=lambda r: r["bytes"]["packed_complete"])
    area = next(iter(biggest["areas_packed_complete"]))
    L += [f"The complete tier is dominated by full-graph prediction arrays (`{area}/` of `{biggest['artifact']}`: "
          f"{red(human(biggest['areas_packed_complete'][area]))}); the analysis tier that `make paper` downloads by "
          "default omits checkpoints and prediction arrays and keeps every trajectory, split index and decision record. "
          "GitHub release assets are limited to 2 GiB each with no limit on the release size, which the part splitting "
          "of `scripts/paper_artifacts.py` already respects. A Zenodo record holds 50 GB and 100 files by default and at "
          "most 200 GB after a one-time increase; dividing one dataset across records to avoid that limit is not "
          "accepted by Zenodo's usage policy.", ""]
    L += ["The measurement-record sets and campaigns are published as their own artifacts:", "",
          "| Artifact | Contents | Manuscript displays | Status |", "|---|---|---|---|"]
    for other in doc["other_artifacts"]:
        L.append(f"| `{other['artifact']}` | {other['title']} | {', '.join(other['tables']) or '—'} | "
                 f"{'published' if other['status'] == 'published' else 'awaiting'} |")
    L.append("")
    # ---------------------------------------------------------------- D
    L += ["## D. Compliance at the final destination", "",
          "Each registered item is evaluated on the files of [`destination/`](../destination/README.md) with the checks "
          "of `scripts/statement_check.py`: the release with every required artifact published, the seven executed "
          "manifests, the observed environments, the E12 study view and the source data of a complete reconstruction.", "",
          "| Registered statement item | Now | Final destination | Fulfilled by |", "|---|---|---|---|"]
    for r in comp["registered"]:
        final = label.get(r["projected"], r["projected"])
        L.append(f"| {r['phrase']} | {label[r['now']]} | {final if r['now'] == 'fulfilled' else red(final)} | {cell(r['note'])} |")
    figures = doc["displays"]["figures_to_register"]
    kinds = {}
    for work in figures.values():
        for part in work.split("; "):
            kinds[part] = kinds.get(part, 0) + 1
    L += ["", f"The one-command item needs every numbered display rebuilt and every compared value equal to the printed "
          f"one. {doc['displays']['table_targets']} printed values of the {doc['displays']['tables']} tables and "
          f"{doc['displays']['figure_targets']} of {doc['displays']['figures_with_contracts']} figures are registered "
          f"comparison targets now; the other {len(figures)} quantitative figures receive their dedicated contracts "
          f"before publication ({', '.join(f'{k}: {v}' for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]))}; "
          "[the list with the input that closes each](../destination/STATEMENT_COMPLIANCE.md#every-reported-table-and-figure), "
          "[the plan](FIGURE_REFERENCE_COVERAGE.md)).", "",
          "| Printed claim | Now | Final destination | Note |", "|---|---|---|---|"]
    for r in comp["printed"]:
        if r["now"] == "fulfilled":
            continue
        final = label.get(r["projected"], r["projected"])
        L.append(f"| {r['id']}: {cell(r['text'][:90])}{'…' if len(r['text']) > 90 else ''} | {label[r['now']]} | {red(final)} | {cell(r['note'])} |")
    L += ["", f"The other {sum(1 for r in comp['printed'] if r['now'] == 'fulfilled')} printed claims are fulfilled now "
          "(code, contracts, commands and conventions; see [Statement compliance](../STATEMENT_COMPLIANCE.md)).", ""]
    # ---------------------------------------------------------------- E
    L += ["## E. The close-out sequence", "",
          "From the current state to the final destination, on the cluster that ran the seven lifecycles "
          "(`bash run_cross_morse.sh reproduce`):", "",
          "```bash",
          "make final-destination-runs RUNS_ROOT=<suite>/runs          # completed runs against sections A and B",
          "bash hpc/produce_paper_outputs_arrhenius.sh <suite>          # every other measured output (docs/PRODUCTION.md)",
          "bash hpc/publish_paper_artifacts_arrhenius.sh all <work>/paper-artifacts.map.json   # release paper-artifacts",
          "make statement-fulfilled                                     # the twelve registered items",
          "make final-destination-converge                              # every published file against destination/",
          "make final-destination                                       # projections become realized values",
          "```", "",
          "The printed manuscript text places the run archives and measurement records at deposit DOIs listed in "
          "`runs/index.json` and `records/index.json`, whereas the registered statement and the publication workflow "
          "place them in the repository's release. Two routes close the claims that depend on this:", "",
          "- **Route R (release).** Describe the release as the location of the run archives and measurement records "
          "in the printed paragraphs, as the registered statement already does; `make paper`, `make records-tables`, "
          "`make tables RUNS_ROOT=$ARTIFACTS_ROOT/runs` and `make figures` then operate on the published, verified "
          "assets, and every printed claim is fulfilled at publication.",
          "- **Route Z (Zenodo).** Keep the printed wording, additionally deposit every run archive through the "
          "evidence-registration workflow (`make evidence-deposit`, `evidence-publish PUBLISH=1`, `evidence-register`; "
          "[Evidence registration](EVIDENCE_REGISTRATION.md)) and add record-set deposit registration to the record "
          "contract. The run archives whose complete tier exceeds one Zenodo record (section C) need the analysis tier "
          "on Zenodo with the complete tier in the release, or a quota agreed with Zenodo.", "",
          "`make availability-fulfilled` is the final gate for both texts: the registered statement and every printed claim.", ""]
    # ---------------------------------------------------------------- F
    files = [f for f in doc["destination"]["files"] if f.endswith(".json")]
    L += ["## F. The destination tree and the convergence check", "",
          f"[`destination/`](../destination/README.md) holds {len(files)} projected files at the paths publication "
          "writes, a [compliance report at the destination](../destination/STATEMENT_COMPLIANCE.md) and its README. "
          "Each JSON file classifies its values: *exact* values are fixed by the released runtime "
          "(fingerprint `" + doc["destination"].get("fingerprint", "") + "`), the effective configurations, the seeds and "
          "the official data definitions; *registered* values follow the registered site profile (one NVIDIA GH200 per "
          "job, the CUDA 13 PyTorch wheel, Python 3.13.5 on aarch64); *size* and *count* values are estimates with a "
          "stated tolerance; *pending* values are supplied by the realized output.", "",
          "| Projected file | Projects |", "|---|---|"]
    for f in files:
        L.append(f"| [`destination/{f}`](../destination/{f}) | `{f}` |")
    L += ["", "After publication, `make final-destination-converge` compares every realized file of the checkout with "
          "its projection: exact values must be identical, registered values must match the site, sizes must fall "
          "within a factor of two and counts within 25 %. It exits non-zero when an exact value differs, so the "
          "double-check of the realized outputs is one command.", ""]
    return "\n".join(L)


def render_runs(rows: list[dict]) -> list[str]:
    out = []
    for r in rows:
        q = "" if r["printed_Q"] is None else f"; |Q_feas| {r['feasible_family']} vs printed {r['printed_Q']}"
        extra = []
        for key, text in (("configuration_is_registered", "registered configuration"), ("runtime_is_released", "released runtime"),
                          ("role_sizes_follow_the_rule", "role sizes follow the floors"),
                          ("official_sizes_as_projected", "official sizes as projected"),
                          ("graph_roles_match_preregistration", "graph roles = pre-registration")):
            if key in r:
                extra.append(f"{text}: {r[key]}")
        out.append(f"  {r['run']} ({r['profile']}): registry {r['registered_candidates']} "
                   f"({'agrees' if r['registry_agrees'] else 'DIFFERS'} with its configuration){q}; "
                   f"resamples {r['bootstrap_replicates']}; " + "; ".join(extra))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--runs-root", type=Path, help="completed runs to compare with the design forecast (not written)")
    parser.add_argument("--converge", action="store_true",
                        help="compare every realized file of the checkout with its projection in destination/")
    args = parser.parse_args(argv)
    from scripts import destination
    try:
        doc, tree = build_all(ROOT, args.runs_root.resolve() if args.runs_root else None)
    except (OSError, ValueError, KeyError, TypeError, SyntaxError) as exc:
        print(f"final destination: {exc}", file=sys.stderr)
        return 2
    if args.converge:
        report = destination.converge(ROOT, tree)
        if args.json:
            print(json.dumps(report, indent=1, ensure_ascii=False, default=str))
        else:
            for row in report["files"]:
                counts = ", ".join(f"{k} {v}" for k, v in sorted(row["counts"].items()))
                print(f"{row['status']:<22} destination/{row['file']}{(': ' + counts) if counts else ''}")
                for finding in row["findings"][:20]:
                    print(f"    {finding['outcome']:<8} {finding['class']:<10} {finding['path']}: projected "
                          f"{json.dumps(finding['projected'], default=str)[:80]}, realized "
                          f"{json.dumps(finding['realized'], default=str)[:80]}")
            s = report["summary"]
            print(f"{s['converged']} converged, {s['converged with notes']} converged with notes, {s['differs']} differ, "
                  f"{s['pending']} pending of {len(report['files'])} projected files")
        return 1 if report["summary"]["differs"] else 0
    text = render(doc)
    if args.json:
        print(json.dumps(doc, indent=1, ensure_ascii=False))
    else:
        fc = doc["forecast_counts"]
        print(f"design forecast: {fc['agrees']} agree, {fc['within bound']} measured within their design bound, "
              f"{fc['differs'] + fc['exceeds bound']} conflicts, {fc['not printed']} not printed")
        for r in doc["forecast"]:
            if r["status"] in ("differs", "exceeds bound"):
                print(f"  {r['status']}  {r['display']:<10} {r['quantity']:<28} {SHORT[r['dataset']]:<9} released code "
                      f"{r['forecast']} vs printed {r['printed']}")
        t = doc["runs_totals"]
        print(f"projected run archives: {human(t['packed_bytes']['analysis'])} analysis tier, "
              f"{human(t['packed_bytes']['complete'])} complete tier, {t['parts']['complete']} release parts")
        d = doc["destination"]
        print(f"destination tree: {len(d['files'])} files; registered statement at the destination: "
              f"{d['statement']['fulfilled']} of {d['statement']['of']} fulfilled")
        if args.runs_root:
            print(f"completed runs under {args.runs_root}:")
            for line in render_runs(doc.get("runs_report", [])):
                print(line)
    if args.runs_root:
        return 0
    stored = json.dumps(doc, indent=1, ensure_ascii=False) + "\n"
    if args.write:
        (ROOT / OUT_JSON).write_text(stored, encoding="utf-8")
        (ROOT / OUT_MD).write_text(text + "\n", encoding="utf-8")
        destination.write(ROOT, tree)
    if args.check:
        stale = [p for p, v in ((OUT_JSON, stored), (OUT_MD, text + "\n"))
                 if not (ROOT / p).is_file() or (ROOT / p).read_text(encoding="utf-8") != v]
        stale += destination.stale(ROOT, tree)
        if stale:
            print(f"{', '.join(stale)} not current; run: python -B scripts/final_destination.py --write", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
