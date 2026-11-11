#!/usr/bin/env python3
"""The destination tree: every file that publication writes into the checkout, projected now.

`scripts/final_destination.py` draws the end state of the repository; this module writes it out file by file
under `destination/`, at the relative paths publication uses (`release/paper_artifacts.json`,
`manifests/published/`, `environment/observed/`, `experiments/baselines/`, `paper/source_data/`), evaluates the
twelve registered statement items on that tree, and compares every realized file with its projection once the
measured outputs exist (`make final-destination-converge`).

Every JSON file of the tree carries a top-level ``projection`` block that names the real file it projects and
classifies its values:

  exact       fixed by the released code, the registered configurations and the official data definitions
              (runtime fingerprint, configuration hashes, seeds, role sizes, pre-registered role hashes,
              artifact set); a realized value must be identical
  registered  the registered execution site (Arrhenius GH200 profile of run_cross_morse.sh); a realized value
              differs only if the runs execute elsewhere
  size        projected bytes, within a factor of SIZE_FACTOR
  count       projected file counts, within COUNT_SHARE
  minimum     a lower bound the realized value must reach
  pending     supplied only by the realized output (digests, archive names, timestamps, data-dependent values)

Real loaders refuse any file carrying the ``projection`` block (scripts/paper_artifacts.py,
scripts/statement_evidence.py), so a projected file can never stand in for a published one.
"""
from __future__ import annotations

import ast
import fnmatch
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

TREE = "destination"
MARK = "projection"
SCHEMA = "cross-morse-destination-1"
PENDING = "pending"
SIZE_FACTOR = 2.0
COUNT_SHARE = 0.25
DATASETS = ("ogbn-arxiv", "ogbn-proteins", "ogbn-products", "Roman-empire", "PascalVOC-SP", "COCO-SP", "Peptides-func")
STEMS = {"ogbn-arxiv": "arxiv", "ogbn-proteins": "proteins", "ogbn-products": "products",
         "Roman-empire": "roman_empire", "PascalVOC-SP": "pascalvoc_sp", "COCO-SP": "coco_sp",
         "Peptides-func": "peptides_func"}
INDEPENDENT = ("PascalVOC-SP", "COCO-SP", "Peptides-func")
GRAPH_LABELS = ("Peptides-func",)
ROMAN_SPLITS = 10

# Provider split sizes (train, valid, test) of the single-graph benchmarks: the OGB time, species and
# sales-ranking splits (Hu et al., NeurIPS 2020) and each of the ten official 50/25/25 masks of Roman-empire
# (Platonov et al., ICLR 2023). The realized prepared/official_splits.npz replaces them; the role sizes follow
# from them by the registered floors of cross_morse/data.py::_make_roles.
OFFICIAL_NODE_SPLITS = {"ogbn-arxiv": (90941, 29799, 48603), "ogbn-proteins": (86619, 21236, 24679),
                        "ogbn-products": (196615, 39323, 2213091), "Roman-empire": (11331, 5665, 5666)}

# The registered execution site: SITE SETTINGS of run_cross_morse.sh and Supplementary Table S5 (one NVIDIA
# GH200 96 GB per job, aarch64 nodes, the CUDA 13 PyTorch wheel, the cluster's Python 3.13.5 module).
SITE_KEYS = {"python": "CM_PYTHON_MODULE", "torch": "CM_TORCH_SPEC", "machine": "CM_EXPECT_ARCH",
             "build": "CM_BUILD_MODULE"}

# Work that closes each quantitative figure still without a dedicated printed-value contract
# (docs/FIGURE_REFERENCE_COVERAGE.md). A figure leaves this table when its contract group is registered in
# paper/printed_display_values.json; tests/test_destination.py keeps the two in step.
FIGURE_WORK = {
    "main:Figure 2": ("scenario clarification; original plotted data",
                      "the reference-ESS scenario and candidate/stage population; raw P coordinates where S14 prints a bound"),
    "main:Figure 3": ("original plotted data",
                      "mechanism-contrast families, within-depth action entropies and action-origin fractions of the original records"),
    "main:Figure 4": ("original plotted data; reference work",
                      "panel 4b feedback risk and work by ensemble size from the original selection arrays; other panels from S16, S15, S23 and S41"),
    "main:Figure 6": ("scenario clarification", "the declared settings of the decoding-cost scenario of S47"),
    "supplementary:Figure S2": ("original plotted data", "per-instance processed edges, wall time and peak memory of runtime_scaling"),
    "supplementary:Figure S3": ("original plotted data", "the complete comparator-grid losses of the ensemble grid"),
    "supplementary:Figure S4": ("original plotted data; scope clarification",
                                "scores and provenance of GRAMA/arxiv, AMP/Roman and AMP/Pascal outside the fourteen E12 pairs"),
    "supplementary:Figure S14": ("original plotted data", "depth curves, effects by depth, quartiles and phase allocations of depth_diagnostics"),
    "supplementary:Figure S16": ("original plotted data", "horizon coverage, NIS coverage and false-safe/harm-miss rates of the reliability records"),
    "supplementary:Figure S17": ("original plotted data; scenario clarification", "the held-out critic-RMSE band endpoints"),
    "supplementary:Figure S18": ("scenario clarification; original analytical grid", "the nominal baseline and density-scale grid of panel c"),
}


class DestinationError(ValueError):
    pass


def pending(what: str) -> str:
    return f"{PENDING}: {what}"


def is_pending(value) -> bool:
    return isinstance(value, str) and value.startswith(PENDING)


def is_projection(value) -> bool:
    return isinstance(value, dict) and isinstance(value.get(MARK), dict) and value[MARK].get("projected") is True


def refuse_projection(value, where) -> None:
    """Called by the real loaders: a projected destination file is never a publication record."""
    if isinstance(value, dict) and MARK in value:
        raise ValueError(f"{where} is a projection from destination/ (scripts/destination.py), not a published file")


def marker(replaces: str, about: str, classes: dict[str, list[str]]) -> dict:
    return {"projected": True, "schema": SCHEMA, "replaces": replaces,
            "generated_by": "scripts/final_destination.py --write", "about": about, "classes": classes,
            "rule": ("Every value in this file is a projection of the file that publication writes at `replaces`. "
                     "Real loaders refuse it. `make final-destination-converge` compares the realized file with it: "
                     "exact values must be identical, registered values follow the registered site profile, sizes "
                     f"must fall within a factor of {SIZE_FACTOR:g} and counts within {COUNT_SHARE:.0%}, minimums "
                     "must be reached, and pending values are supplied by the realized output.")}


def canonical_sha256(value) -> str:
    """The configuration identity of the sealed stage records (cross_morse/provenance.py::_json_hash)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


# ===================================================================== design identities
def effective_configs(root: Path) -> dict[str, list[dict]]:
    """The effective configuration of every reported run (run_cross_morse.py::resolve_configuration)."""
    import run_cross_morse as entry
    out = {}
    for dataset in DATASETS:
        path = root / "configs" / f"{STEMS[dataset]}_full.json"
        splits = range(ROMAN_SPLITS) if dataset == "Roman-empire" else [None]
        out[dataset] = [entry.resolve_configuration(config_path=path, dataset=dataset, profile="full", split_index=s)
                        for s in splits]
    return out


def provider_shapes(root: Path) -> dict:
    """The provider shapes the data adapters enforce (cross_morse/data.py: expected, graph_counts)."""
    found = {}
    for node in ast.walk(ast.parse((root / "cross_morse" / "data.py").read_text(encoding="utf-8"))):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in ("expected", "graph_counts", "names") and isinstance(node.value, ast.Dict)):
            value = ast.literal_eval(node.value)
            if node.targets[0].id != "names" or "metis" in value:
                found[node.targets[0].id] = value
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == "split_rule" and isinstance(value, ast.Constant):
                    found["split_rule"] = value.value
    if set(found) != {"expected", "graph_counts", "names", "split_rule"}:
        raise DestinationError("cross_morse/data.py no longer declares the provider shapes and unit names this "
                               "projection reads")
    return found


def site_profile(root: Path) -> dict:
    """The registered execution site from the SITE SETTINGS block of run_cross_morse.sh."""
    import re
    text = (root / "run_cross_morse.sh").read_text(encoding="utf-8")
    values = {}
    for key, name in SITE_KEYS.items():
        match = re.search(r'\$\{' + name + r'=([^}]*)\}', text)
        if not match:
            raise DestinationError(f"run_cross_morse.sh no longer sets {name} in its SITE SETTINGS")
        values[key] = match.group(1).strip()
    python = values["python"].split("/")[-1].split("-")[0]
    torch = values["torch"].split("==", 1)[-1]
    cuda = torch.split("+cu", 1)[1] if "+cu" in torch else None
    return {"python": python, "torch": torch, "machine": values["machine"],
            "cuda_runtime": f"{cuda[:-1]}.{cuda[-1]}" if cuda else None,
            "gpu": "NVIDIA GH200", "gpu_memory": "96 GB", "modules": [values["build"], values["python"]],
            "source": "SITE SETTINGS of run_cross_morse.sh; Supplementary Table S5 (one NVIDIA GH200 96 GB)"}


def node_roles(functions: dict, dataset: str, config: dict) -> dict:
    n_train, n_valid, n_test = OFFICIAL_NODE_SPLITS[dataset]
    total = n_train + n_valid + n_test
    official = {"train": np.arange(0, n_train, dtype=np.int64),
                "valid": np.arange(n_train, n_train + n_valid, dtype=np.int64),
                "test": np.arange(n_train + n_valid, total, dtype=np.int64)}
    roles = functions["_make_roles"](official, total, int(config["seed"]), float(config["data"]["internal_val_fraction"]))
    return {key: int(len(value)) for key, value in roles.items()}


class Context:
    """Everything the tree is built from: the final-destination document and the repository's registrations."""

    def __init__(self, root: Path, doc: dict, functions: dict):
        from scripts import paper_artifacts
        from scripts._paper_common import runtime_source_fingerprint
        self.root = root
        self.doc = doc
        self.functions = functions
        self.configs = effective_configs(root)
        self.fingerprint = runtime_source_fingerprint(root)
        self.shapes = provider_shapes(root)
        self.site = site_profile(root)
        self.required = paper_artifacts.required_artifacts(root)
        self.release = paper_artifacts.load_manifest(root / "release" / "paper_artifacts.json")
        self.specs = json.loads((root / "records" / "specs.json").read_text(encoding="utf-8"))["sets"]
        self.runs = {r["dataset"]: r for r in doc["runs"]}
        self.roles = doc["preregistered_roles"]
        sources = json.loads((root / "data" / "sources.json").read_text(encoding="utf-8"))["datasets"]
        self.sources = {r["dataset"]: r for r in sources}
        constants = json.loads((root / "paper" / "registered_constants.json").read_text(encoding="utf-8"))
        self.profiles = constants["backbones_source_status"]["released_full_profiles"]


# ================================================================ release/paper_artifacts.json
def run_identity(ctx: Context, dataset: str) -> dict:
    def one(config):
        return {"source_sha256": ctx.fingerprint, "config_sha256": canonical_sha256(config),
                "unit_sha256": pending("SHA-256 of prepared/splits.npz"), "profile": "full", "scientific_run": True}
    configs = ctx.configs[dataset]
    if dataset == "Roman-empire":
        return {"campaign": True, "splits": {f"split_{i:02d}": one(c) for i, c in enumerate(configs)}}
    return {"campaign": False, **one(configs[0])}


def tier(stem: str, name: str, packed: int, parts: int) -> dict:
    return {"archive": pending(f"{stem}.{name}-<SHA-256>.tar.gz, named by its content"), "bytes": int(packed),
            "sha256": pending("SHA-256 of the archive"),
            "parts": {"count": int(parts), "each_at_most_bytes": 1900 * 1024 * 1024,
                      "asset": pending(f"{stem}.{name}-<SHA-256>.tar.gz" + (".partNNN" if parts > 1 else "")),
                      "url": pending("https://github.com/VKHS/CROSS-MORSE/releases/download/paper-artifacts/<asset>")}}


def record_identity(ctx: Context, name: str) -> dict:
    from cm_campaigns.contract import EXTERNAL, RECORD_SETS
    identity = {"manifest_sha256": pending("SHA-256 of the record set MANIFEST.json"),
                "rows": pending("rows of the exported summary")}
    contract = RECORD_SETS.get(name)
    spec = ctx.specs.get(name, {})
    if contract is None:
        return identity
    if contract.get("analyzer") == "baselines":
        # cm_campaigns/baselines.py::check_coverage
        identity["coverage"] = {"keys": ["dataset", "method", "seed"], "datasets": len(contract["datasets"]),
                                "internal_rows": pending("internal method rows of the seven runs"),
                                "external_comparisons": sum(len(v) for v in EXTERNAL.values()),
                                "seeds_per_row": pending("seeds per method row")}
    elif contract.get("coverage_keys") is None:
        identity["coverage"] = {"keys": ["dataset"], "rows": len(contract["datasets"])}
    else:
        identity["coverage"] = {"keys": contract["coverage_keys"], "rows": len(contract["coverage_rows"]),
                                "quantities_per_row": len(spec.get("quantities", {}))}
    return identity


def release_manifest(ctx: Context) -> dict:
    from scripts import paper_artifacts as pa
    artifacts = []
    for row in ctx.required:
        entry = {key: row[key] for key in ("id", "kind", "name", "asset_stem", "statement_items", "displays")}
        entry.update(status=pa.PUBLISHED, relationship="exact_result_producing_run", scientific_only_verification=True,
                     inventory_sha256=pending("SHA-256 of the packed file inventory"),
                     published_utc=pending("time of upload"))
        kind, name = row["kind"], row["name"]
        if kind == "run":
            run = ctx.runs[name]
            entry["identity"] = run_identity(ctx, name)
            entry["files"] = int(run["files"])
            entry["bytes"] = int(run["bytes"]["raw_complete"])
            entry["contents"] = dict(run["contents"])
            entry["tiers"] = {t: tier(row["asset_stem"], t, run["bytes"][f"packed_{t}"], run["parts"][t])
                              for t in ("analysis", "complete")}
        else:
            if kind == "records":
                entry["identity"] = record_identity(ctx, name)
            elif kind == "campaign" and name == "confirmation":
                entry["identity"] = {"verification_status": "PASS", "post_hoc": False,
                                     "input_identity": pending("identity of the verified campaign inputs")}
            elif kind == "campaign":
                entry["identity"] = {"verification_status": "PASS",
                                     "coverage": pending("dataset-arm contrasts of the verified campaign")}
            else:
                entry["identity"] = {"observed_environment_sha256": pending("SHA-256 of observed_environment.json"),
                                     "gpu": ctx.site["gpu"], "python": ctx.site["python"], "torch": ctx.site["torch"]}
            entry["files"] = 4 if kind == "environment" else pending("files of the packed set")
            entry["bytes"] = pending("bytes of the packed set")
            entry["contents"] = pending("statement categories of the packed set (scripts/paper_artifacts.py::content_counts)")
            entry["tiers"] = {"complete": {"archive": pending(f"{row['asset_stem']}.complete-<SHA-256>.tar.gz"),
                                           "bytes": pending("bytes of the archive"),
                                           "sha256": pending("SHA-256 of the archive"),
                                           "parts": {"count": pending("parts of the archive"),
                                                     "each_at_most_bytes": 1900 * 1024 * 1024}}}
        artifacts.append(entry)
    classes = {
        "exact": ["schema", "repository", "release_tag", "release_url", "part_bytes", "statement", "status",
                  "artifacts/*/id", "artifacts/*/kind", "artifacts/*/name", "artifacts/*/asset_stem",
                  "artifacts/*/status", "artifacts/*/relationship", "artifacts/*/scientific_only_verification",
                  "artifacts/*/identity/source_sha256", "artifacts/*/identity/config_sha256",
                  "artifacts/*/identity/profile", "artifacts/*/identity/scientific_run", "artifacts/*/identity/campaign",
                  "artifacts/*/identity/splits/*/source_sha256", "artifacts/*/identity/splits/*/config_sha256",
                  "artifacts/*/identity/splits/*/profile", "artifacts/*/identity/splits/*/scientific_run",
                  "artifacts/*/identity/coverage/*", "artifacts/*/identity/verification_status",
                  "artifacts/*/identity/post_hoc",
                  "artifacts/run:*/contents/propensity_logged_trajectories"],
        "registered": ["artifacts/environment:gpu/identity/gpu", "artifacts/environment:gpu/identity/python",
                       "artifacts/environment:gpu/identity/torch"],
        "size": ["artifacts/run:*/bytes", "artifacts/run:*/tiers/*/bytes"],
        "count": ["artifacts/run:*/files", "artifacts/run:*/contents/*", "artifacts/run:*/tiers/*/parts/count",
                  "artifacts/environment:gpu/files"],
        "pending": ["artifacts/*/inventory_sha256", "artifacts/*/published_utc", "artifacts/*/tiers/*/archive",
                    "artifacts/*/tiers/*/sha256", "artifacts/*/identity/unit_sha256",
                    "artifacts/*/identity/splits/*/unit_sha256", "artifacts/*/identity/manifest_sha256",
                    "artifacts/*/identity/rows"],
    }
    value = {MARK: marker("release/paper_artifacts.json",
                          "The release paper-artifacts with every required artifact published: run archives sized from the "
                          "runtime's file layout at the registered scale (docs/FINAL_DESTINATION.md, section C), identities "
                          "from the released runtime and the effective configurations. `parts` is summarized as a count; "
                          "the realized manifest lists one entry per release asset.", classes),
             **{k: v for k, v in pa.blank_manifest().items() if k != "artifacts"},
             "status": pa.PUBLISHED, "artifacts": artifacts}
    return value


# ================================================================= manifests/published/<stem>.json
def manifest_part(ctx: Context, dataset: str, config: dict, run_name: str) -> dict:
    from cross_morse import split_contract
    expected, graphs = ctx.shapes["expected"], ctx.shapes["graph_counts"]
    seed = int(config["seed"])
    if dataset in INDEPENDENT:
        roles = ctx.roles[dataset]
        sizes = roles["sizes"] if dataset in GRAPH_LABELS else pending("node-row role sizes; follow from the official graph sizes")
        role_hashes = roles["graph_role_hashes"] if dataset in GRAPH_LABELS else pending("node-row role hashes")
        graph_hashes = roles["graph_role_hashes"]
        counts = roles["official_split_hashes"]
        source = {"source": pending("provider record of the LRGB archive"), "source_array_hashes": pending("provider array hashes"),
                  "num_nodes": pending("total nodes of the official graphs"), "num_graphs": graphs[dataset],
                  "num_features": int(ctx.profiles[dataset]["input_dim"]),
                  "num_outputs": int(ctx.sources[dataset]["official_outputs"])}
        unit, statistical = "graph", "complete graph"
    else:
        sizes = node_roles(ctx.functions, dataset, config)
        graph_hashes = None
        role_hashes = pending("node role hashes; follow from the official node identifiers")
        counts = pending("hashes of the official node splits")
        n, features, outputs = expected[dataset]
        source = {"source": pending("provider record of the official release"), "source_array_hashes": pending("provider array hashes"),
                  "num_nodes": int(n), "num_graphs": 1, "num_features": int(features), "num_outputs": int(outputs)}
        unit, statistical = "node", ctx.shapes["names"][config["data"]["partitioner"]]
    split_index = (config.get("data") or {}).get("split_index")
    return {
        "run_name": run_name,
        "identities": {"source_sha256": ctx.fingerprint, "config_sha256": canonical_sha256(config),
                       "prepared_fingerprint": pending("prepared-data fingerprint"),
                       "unit_sha256": pending("SHA-256 of prepared/splits.npz")},
        "seeds": {"partition_seed": seed, "training_seeds": list(config["training"]["seeds"]),
                  "ensemble_seeds": pending("seeds of the selected ensembles"),
                  "bootstrap_seeds": {"selection": seed + 4, "admission": seed + 6, "final": seed + 7},
                  "split_index": split_index},
        "splits": {"split_protocol": split_contract.DATA_ROLE_PROTOCOL, "comparison_reserved_at": "prepare",
                   "unit": unit, "rule": ctx.shapes["split_rule"], "sizes": sizes, "role_hashes": role_hashes,
                   "graph_role_hashes": graph_hashes, "official_split_hashes": counts,
                   "official_split_axis": unit, "independent_graphs": dataset in INDEPENDENT,
                   "statistical_unit": statistical,
                   "arrays": pending("splits.npz, official_splits.npz and clusters.npz with SHA-256 and bytes")},
        "source_data": source,
    }


def run_manifest(ctx: Context, dataset: str) -> dict:
    stem = STEMS[dataset]
    configs = ctx.configs[dataset]
    precision = configs[0]["model"]["precision"]
    environment = {"python": ctx.site["python"], "python_implementation": "CPython", "torch": ctx.site["torch"],
                   "cuda_runtime": ctx.site["cuda_runtime"], "gpu": ctx.site["gpu"], "device": "cuda",
                   "platform": pending("platform string of the compute node"), "machine": ctx.site["machine"],
                   "numpy": pending("NumPy of the run environment"), "precision": precision,
                   "deterministic_algorithms_enabled": pending("recorded by the run"),
                   "cudnn_deterministic": True, "tf32_matmul": False}
    base = {"schema": "cross-morse-run-manifest-1", "dataset": dataset, "benchmark_dataset": dataset,
            "data_kind": dataset, "is_benchmark_data": True, "profile": "full", "scientific_run": True}
    if dataset == "Roman-empire":
        parts = [manifest_part(ctx, dataset, config, f"split_{i:02d}") for i, config in enumerate(configs)]
        value = {**base, **{k: v for k, v in parts[0].items() if k != "run_name"}, "run_name": stem,
                 "campaign": {"splits": list(range(ROMAN_SPLITS)),
                              "per_split": [{"run_name": p["run_name"], "split_index": p["seeds"]["split_index"],
                                             "identities": p["identities"], "sizes": p["splits"]["sizes"],
                                             "arrays": pending("membership arrays of the split")} for p in parts]}}
    else:
        value = {**base, **manifest_part(ctx, dataset, configs[0], stem)}
    value["policy_lock"] = pending("comparator, deployed-rule and frozen-model hashes of policy_lock.json")
    value["environment"] = environment
    value["packages"] = pending("installed distributions of the run environment")
    value["index_arrays_copied"] = pending("membership arrays copied next to this manifest")
    classes = {
        "exact": ["schema", "dataset", "benchmark_dataset", "data_kind", "is_benchmark_data", "profile", "scientific_run",
                  "identities/source_sha256", "identities/config_sha256", "seeds/partition_seed", "seeds/training_seeds",
                  "seeds/bootstrap_seeds/*", "seeds/split_index", "splits/split_protocol", "splits/comparison_reserved_at",
                  "splits/unit", "splits/rule", "splits/sizes/*", "splits/role_hashes/*",
                  "splits/graph_role_hashes/*", "splits/official_split_hashes/*", "splits/official_split_axis",
                  "splits/independent_graphs", "splits/statistical_unit", "source_data/num_*",
                  "campaign/splits", "campaign/per_split/*/split_index", "campaign/per_split/*/identities/source_sha256",
                  "campaign/per_split/*/identities/config_sha256", "campaign/per_split/*/sizes/*",
                  "environment/precision", "environment/device", "environment/cudnn_deterministic",
                  "environment/tf32_matmul"],
        "registered": ["environment/python", "environment/python_implementation", "environment/torch",
                       "environment/cuda_runtime", "environment/gpu", "environment/machine"],
        "pending": ["identities/prepared_fingerprint", "identities/unit_sha256", "seeds/ensemble_seeds",
                    "campaign/per_split/*/identities/prepared_fingerprint", "campaign/per_split/*/identities/unit_sha256"],
    }
    return {MARK: marker(f"manifests/published/{stem}.json",
                         "The executed split and seed manifest of the reported run: identities, seeds and role sizes from "
                         "the released runtime and the effective configuration; graph-level role hashes of the LRGB "
                         "benchmarks from the official file order (section B of docs/FINAL_DESTINATION.md).", classes),
            **value}


# ================================================================ environment/observed/<stem>.json
def observed_environment(ctx: Context, dataset: str) -> dict:
    stem = STEMS[dataset]
    precision = ctx.configs[dataset][0]["model"]["precision"]
    runs = [f"{stem}/split_{i:02d}" for i in range(ROMAN_SPLITS)] if dataset == "Roman-empire" else [stem]
    environment = {"python": ctx.site["python"], "python_implementation": "CPython", "torch": ctx.site["torch"],
                   "cuda_runtime": ctx.site["cuda_runtime"], "device": "cuda", "gpu": ctx.site["gpu"],
                   "machine": ctx.site["machine"], "precision": precision, "torch_threads": 16,
                   "cudnn_deterministic": True, "tf32_matmul": False,
                   "platform": pending("platform string of the compute node"), "numpy": pending("NumPy of the run"),
                   "packages": pending("installed distributions"), "version": pending("runtime version string")}
    classes = {"exact": ["schema", "dataset_run", "members/*/run", "members/*/environment/device",
                         "members/*/environment/precision", "members/*/environment/cudnn_deterministic",
                         "members/*/environment/tf32_matmul"],
               "registered": ["members/*/environment/python", "members/*/environment/python_implementation",
                              "members/*/environment/torch", "members/*/environment/cuda_runtime",
                              "members/*/environment/gpu", "members/*/environment/machine",
                              "members/*/environment/torch_threads"]}
    return {MARK: marker(f"environment/observed/{stem}.json",
                         f"The environment every reported {dataset} run records in environment.json, as the registered "
                         f"site profile sets it ({ctx.site['source']}).", classes),
            "schema": "cross-morse-observed-environment-1", "dataset_run": stem,
            "members": [{"run": run, "environment": dict(environment)} for run in runs]}


# =================================================================== experiments/baselines/INDEX.json
def baseline_view(ctx: Context) -> dict:
    from cm_campaigns.baselines import SLUGS, method_directory
    from cm_campaigns.contract import EXTERNAL
    from scripts.paper_artifacts import BASELINE_SCHEMA
    studies = []
    for dataset, methods in EXTERNAL.items():
        for method in methods:
            folder = f"{SLUGS[dataset]}/{method_directory(method)}"
            splits = list(range(ROMAN_SPLITS)) if dataset == "Roman-empire" else [0]
            studies.append({
                "dataset": dataset, "method": method,
                "source_commit": pending("pinned upstream commit"), "adapter_commit": pending("pinned adapter commit"),
                "upstream_repository": pending("upstream repository of the method"),
                "plan": {"file": f"{folder}/plan.json", "sha256": pending("SHA-256 of the study plan")},
                "upstream": {"file": f"{folder}/upstream.json", "sha256": pending("SHA-256 of the upstream identity")},
                "adapter_archive": {"file": pending(f"{folder}/adapter-<commit[:12]>.tar"), "sha256": pending("SHA-256")},
                "study_manifest": {"file": f"{folder}/study_files.json", "sha256": pending("SHA-256")},
                "study_archive": {"artifact": "records:baselines", "file": f"raw/{folder}/study.tar",
                                  "manifest_file": f"raw/{folder}/study_files.json",
                                  "bytes": pending("bytes of the sealed study"), "sha256": pending("SHA-256")},
                "registered_study": {"hpo_trials": 48, "optimization_seeds": 12, "official_splits": splits,
                                     "final_fits": 12 * len(splits), "selection_role": "internal_training_validation",
                                     "metric_unit": "native_0_to_1",
                                     "rule": "cm_experiments/upstream.py::validate_plan; scripts/statement_evidence.py::check_baseline_studies"},
            })
    classes = {"exact": ["schema", "source", "studies/*/dataset", "studies/*/method", "studies/*/plan/file",
                         "studies/*/upstream/file", "studies/*/study_manifest/file", "studies/*/study_archive/artifact",
                         "studies/*/study_archive/file", "studies/*/study_archive/manifest_file"]}
    return {MARK: marker("experiments/baselines/INDEX.json",
                         "The E12 study plans and adapter code of the fourteen external comparisons, copied from the "
                         "published records:baselines set (scripts/paper_artifacts.py derive). `registered_study` states "
                         "what each sealed study must contain; it is checked against the study inventory, not stored in "
                         "the realized view.", classes),
            "schema": BASELINE_SCHEMA, "source": "records:baselines",
            "description": ("E12 study plans and adapter code copied from the published baselines record set; the full "
                            "sealed studies, checkpoints and predictions remain in that release asset"),
            "studies": studies}


# ===================================================================== paper/source_data/INDEX.json
def display_ledger(root: Path) -> list[dict]:
    """Every numbered display: its generator, the artifacts it reads and the printed values it compares."""
    from paper.build.displays import framework
    framework._load_generators()
    registry, metadata, _ = framework.load_references(root)
    manifest = json.loads((root / "paper" / "display_manifest.json").read_text(encoding="utf-8"))
    rows = []
    for display in manifest["displays"]:
        generator = framework.GENERATORS.get(display["id"])
        scope = generator.scope if generator else "measurement"
        _, coverage = framework.comparison_coverage(display["id"], [], registry, metadata, scope=scope)
        targets = int(coverage["expected_values"])
        if display["kind"] == "figure" and scope != "illustration" and targets == 0:
            work, closes = FIGURE_WORK.get(display["id"], ("unclassified", "a dedicated printed-value contract"))
            contract = {"now": "to register", "work": work, "closes": closes}
        else:
            contract = {"now": "registered" if targets else "illustration"}
        rows.append({"id": display["id"], "kind": display["kind"], "pages": display.get("pages"),
                     "generator": generator.function.__module__.rsplit(".", 1)[-1] if generator else None,
                     "scope": scope, "needs": sorted(generator.needs) if generator else [],
                     "registered_targets": targets, "contract": contract})
    return rows


def source_data(ctx: Context, ledger: list[dict]) -> dict:
    displays = len(ledger)
    registered = sum(r["registered_targets"] for r in ledger)
    to_register = [r["id"] for r in ledger if r["contract"]["now"] == "to register"]
    summary = {"displays": displays, "complete": displays, "incomplete": [],
               "values_exact": registered, "values_differ": 0, "values_not_comparable": 0, "values_no_reference": 0,
               "values_missing": 0, "comparison_coverage_complete": True, "display_content_incomplete": [],
               "generator_errors": {}, "module_errors": {}, "artifacts_awaiting_publication": []}
    classes = {"exact": ["schema", "reconstruction/displays", "reconstruction/complete", "reconstruction/incomplete",
                         "reconstruction/values_differ", "reconstruction/values_not_comparable",
                         "reconstruction/values_no_reference", "reconstruction/values_missing",
                         "reconstruction/comparison_coverage_complete", "reconstruction/display_content_incomplete",
                         "reconstruction/artifacts_awaiting_publication"],
               "minimum": ["reconstruction/values_exact"]}
    return {MARK: marker("paper/source_data/INDEX.json",
                         "The source data of every numbered display after `make paper` has rebuilt them from the published "
                         f"artifacts: all {displays} displays complete and every compared value equal to the printed one. "
                         f"`values_exact` counts the {registered} printed values registered now; the "
                         f"{len(to_register)} figure contracts still to be registered add theirs.", classes),
            "schema": "cross-morse-source-data-1", "label": "paper-artifacts",
            "reconstruction": summary,
            "reports": pending("reconstruction, workspace inventory and status reports with SHA-256"),
            "files": pending("the CSV and JSON source files of every display"),
            "displays": ledger}


# ======================================================================== the twelve items on the tree
def projected_statement(root: Path, tree: dict, now: dict) -> dict:
    """The twelve registered items evaluated on the destination tree, with the checks of scripts/statement_check.py."""
    from scripts import statement_check as sc
    from scripts.statement_evidence import reconstruction_problems
    from cm_campaigns.contract import EXTERNAL
    from scripts import paper_artifacts as pa
    release = tree["release/paper_artifacts.json"]
    artifacts = {a["id"]: a for a in release["artifacts"]}
    published = {i for i, a in artifacts.items() if a.get("status") == pa.PUBLISHED}
    required = {a["id"] for a in pa.required_artifacts(root)}
    runs = [f"run:{d}" for d in DATASETS]
    view = tree["experiments/baselines/INDEX.json"]
    pairs = {(s["dataset"], s["method"]) for s in view["studies"]}
    expected_pairs = {(d, m) for d, methods in EXTERNAL.items() for m in methods}
    studies_ok = (pairs == expected_pairs and "records:baselines" in published
                  and all(s["registered_study"]["hpo_trials"] == 48 and s["registered_study"]["optimization_seeds"] == 12
                          for s in view["studies"]))
    manifests = {d: tree.get(f"manifests/published/{STEMS[d]}.json") for d in DATASETS}
    manifests_ok = all(m is not None and m["scientific_run"] is True and m["profile"] == "full"
                       and len(m["seeds"]["training_seeds"]) == 12
                       and (d != "Roman-empire" or len(m["campaign"]["per_split"]) == ROMAN_SPLITS)
                       for d, m in manifests.items()) and set(runs) <= published
    observed = {d: tree.get(f"environment/observed/{STEMS[d]}.json") for d in DATASETS}
    observed_ok = (all(o is not None and len(o["members"]) == (ROMAN_SPLITS if d == "Roman-empire" else 1)
                       and all(all(m["environment"].get(k) for k in ("python", "torch", "device")) for m in o["members"])
                       for d, o in observed.items()) and "environment:gpu" in published)
    contents = {key: all(r in published and (artifacts[r].get("contents") or {}).get(key, 0) > 0 for r in runs)
                for key in ("checkpoints", "predictions", "propensity_logged_trajectories")}
    statistical = (required - {"environment:gpu"}) <= published
    source = tree["paper/source_data/INDEX.json"]
    displays = len(json.loads((root / "paper" / "display_manifest.json").read_text(encoding="utf-8"))["displays"])
    problems = reconstruction_problems(source["reconstruction"], displays)
    current = {row["id"]: row for row in now["items"]}
    checks = {
        "experiment_code": (studies_ok, "E12 adapter code of the fourteen external comparisons at their pinned commits "
                                        "(experiments/baselines/, from records:baselines)"),
        "configurations": (studies_ok, "E12 study plans: 48 HPO configurations and 12 optimization seeds per comparison"),
        "source_data": (required <= published and not problems, "paper/source_data/ written from the published outputs"),
        "split_seed_manifests": (manifests_ok, "manifests/published/: executed memberships of the seven runs "
                                               "(Roman-empire: ten splits)"),
        "environment": (observed_ok, "environment/observed/ of the seven runs and the release asset environment:gpu"),
        "checkpoints": (contents["checkpoints"], "checkpoints in the complete tier of every run"),
        "predictions": (contents["predictions"], "prediction arrays in the complete tier of every run"),
        "trajectories": (contents["propensity_logged_trajectories"], "propensity-logged trajectories of every run"),
        "statistical_outputs": (statistical, "every run, measurement-record set and campaign published"),
        "generators": (current["generators"]["status"] == sc.FULFILLED, "a generator for each numbered display (now)"),
        "one_command": (required <= published and not problems,
                        "`make paper`: every display rebuilt and every compared value agrees"),
        "guidance": (current["guidance"]["status"] == sc.FULFILLED, "REPRODUCIBILITY.md (now)"),
    }
    rows = []
    for item in now["items"]:
        passed, evidence = checks[item["id"]]
        rows.append({"id": item["id"], "phrase": item["phrase"], "now": item["status"],
                     "destination": sc.FULFILLED if passed else sc.MISSING, "evidence": evidence,
                     "problems": problems if item["id"] in ("one_command", "source_data") and problems else []})
    return {"items": rows, "fulfilled": sum(1 for r in rows if r["destination"] == sc.FULFILLED), "total": len(rows)}


# ================================================================================= the tree
def build_tree(root: Path, doc: dict, functions: dict) -> dict[str, Any]:
    """{relative path under destination/: JSON value or Markdown text}."""
    ctx = Context(root, doc, functions)
    ledger = display_ledger(root)
    tree: dict[str, Any] = {"release/paper_artifacts.json": release_manifest(ctx)}
    for dataset in DATASETS:
        tree[f"manifests/published/{STEMS[dataset]}.json"] = run_manifest(ctx, dataset)
    for dataset in DATASETS:
        tree[f"environment/observed/{STEMS[dataset]}.json"] = observed_environment(ctx, dataset)
    tree["experiments/baselines/INDEX.json"] = baseline_view(ctx)
    tree["paper/source_data/INDEX.json"] = source_data(ctx, ledger)
    return tree


def serialize(value) -> str:
    return (json.dumps(value, indent=1, ensure_ascii=False) + "\n") if not isinstance(value, str) else value


def write(root: Path, files: dict[str, Any]) -> None:
    base = root / TREE
    for relative, value in files.items():
        path = base / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialize(value), encoding="utf-8")
    keep = {(base / p).resolve() for p in files}
    for path in sorted(base.rglob("*")) if base.is_dir() else []:
        if path.is_file() and path.resolve() not in keep:
            path.unlink()


def stale(root: Path, files: dict[str, Any]) -> list[str]:
    base = root / TREE
    out = [f"{TREE}/{p}" for p, v in files.items()
           if not (base / p).is_file() or (base / p).read_text(encoding="utf-8") != serialize(v)]
    extra = sorted(p.relative_to(base).as_posix() for p in base.rglob("*")
                   if p.is_file() and p.relative_to(base).as_posix() not in files) if base.is_dir() else []
    return out + [f"{TREE}/{p} (not generated)" for p in extra]


# ================================================================================= convergence
MISSING = object()


def normalize(value):
    """Lists of records become maps keyed by their identity, so projections and realizations align by name."""
    if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
        for keys in (("id",), ("run",), ("run_name",), ("dataset", "method")):
            if all(all(k in v for k in keys) for v in value):
                names = ["/".join(str(v[k]) for k in keys) for v in value]
                if len(set(names)) == len(names):
                    return {name: normalize(v) for name, v in zip(names, value)}
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items()}
    return value


def leaves(value, path=()):
    if isinstance(value, dict) and value:
        for key, child in value.items():
            yield from leaves(child, (*path, str(key)))
    else:
        yield path, value


def lookup(value, path):
    for key in path:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return MISSING
    return value


def classify(path: tuple, classes: dict) -> str | None:
    text = "/".join(path)
    for name in ("pending", "minimum", "size", "count", "registered", "exact"):
        if any(fnmatch.fnmatchcase(text, pattern) for pattern in classes.get(name, [])):
            return name
    return None


def realized_view(relative: str, value):
    """Realized files in the shape of their projection (release parts become a count)."""
    value = normalize(value)
    if relative == "release/paper_artifacts.json":
        for entry in (value.get("artifacts") or {}).values():
            for spec in (entry.get("tiers") or {}).values():
                if isinstance(spec, dict) and isinstance(spec.get("parts"), list):
                    spec["parts"] = {"count": len(spec["parts"])}
    return value


def compare_leaf(kind: str, projected, realized) -> str:
    if realized is MISSING:
        return "absent"
    if kind == "pending" or is_pending(projected):
        return "supplied"
    if kind in ("exact", "registered"):
        if kind == "registered" and isinstance(projected, str) and isinstance(realized, str):
            return "agrees" if projected.lower() in realized.lower() else "differs"
        return "agrees" if realized == projected else "differs"
    if not isinstance(realized, (int, float)) or isinstance(realized, bool):
        return "differs"
    if kind == "minimum":
        return "agrees" if realized >= projected else "differs"
    if kind == "size":
        return "within" if projected > 0 and 1 / SIZE_FACTOR <= realized / projected <= SIZE_FACTOR else "outside"
    if kind == "count":
        return ("within" if (projected == realized or projected > 0 and abs(realized - projected) <= COUNT_SHARE * projected)
                else "outside")
    return "differs"


def converge_file(relative: str, projected: dict, realized, *, publication_complete: bool = True) -> dict:
    classes = projected[MARK]["classes"]
    body = normalize({k: v for k, v in projected.items() if k != MARK})
    report = {"file": relative, "realized": realized is not None, "counts": {}, "findings": []}
    if realized is None or (relative == "paper/source_data/INDEX.json" and not publication_complete):
        # The source data describe a complete reconstruction; before every artifact is published they are pending.
        report["status"] = "pending"
        return report
    real = realized_view(relative, realized)
    skip = set()
    if relative == "release/paper_artifacts.json":
        # An artifact that is not yet published is pending as a whole.
        skip = {name for name, entry in (real.get("artifacts") or {}).items()
                if isinstance(entry, dict) and entry.get("status") != "published"}
        missing = set(body.get("artifacts", {})) - set(real.get("artifacts") or {})
        skip |= missing
    for path, value in leaves(body):
        kind = classify(path, classes)
        if kind is None:
            continue
        if (len(path) > 1 and path[0] == "artifacts" and path[1] in skip) or (path == ("status",) and skip):
            outcome = "pending"
        else:
            outcome = compare_leaf(kind, value, lookup(real, path))
        report["counts"][f"{kind}:{outcome}"] = report["counts"].get(f"{kind}:{outcome}", 0) + 1
        if outcome in ("differs", "outside", "absent") and kind != "pending":
            report["findings"].append({"path": "/".join(path), "class": kind, "outcome": outcome,
                                       "projected": value, "realized": None if lookup(real, path) is MISSING
                                       else lookup(real, path)})
    failures = [f for f in report["findings"] if f["class"] in ("exact", "minimum") and f["outcome"] in ("differs", "absent")]
    report["status"] = "differs" if failures else ("converged" if not report["findings"] else "converged with notes")
    return report


def converge(root: Path, tree: dict[str, Any]) -> dict:
    """Compare every realized file of the checkout with its projection in destination/."""
    reports = []
    release_path = root / "release" / "paper_artifacts.json"
    release = json.loads(release_path.read_text(encoding="utf-8")) if release_path.is_file() else {}
    published = {a.get("id") for a in release.get("artifacts", []) if a.get("status") == "published"}
    required = {a["id"] for a in tree["release/paper_artifacts.json"]["artifacts"]}
    complete = required <= published
    for relative, projected in tree.items():
        if not isinstance(projected, dict) or not is_projection(projected):
            continue
        path = root / relative
        realized = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        if relative == "release/paper_artifacts.json" and not published:
            realized = None
        reports.append(converge_file(relative, projected, realized, publication_complete=complete))
    summary = {s: sum(1 for r in reports if r["status"] == s)
               for s in ("converged", "converged with notes", "differs", "pending")}
    return {"schema": "cross-morse-destination-convergence-1", "files": reports, "summary": summary}


# ================================================================================= rendering
def red(text) -> str:
    safe = str(text).replace("\\", "").replace("{", "").replace("}", "").replace("%", r"\%").replace("_", r"\_")
    return rf"$\color{{red}}{{\textsf{{{safe}}}}}$"


def cell(text) -> str:
    return str(text).replace("|", r"\|")


def render_readme(tree: dict, statement: dict) -> str:
    L = ["# destination/ — the repository at its final destination", "",
         "[The final destination](../docs/FINAL_DESTINATION.md) · [Statement compliance now](../STATEMENT_COMPLIANCE.md) · "
         "[Statement compliance at the destination](STATEMENT_COMPLIANCE.md)", "",
         "Every file publication writes into the checkout, drawn now at the path it will occupy. Each JSON file "
         "carries a top-level `projection` block naming the real file it projects and the class of each value; "
         "values in " + red("red") + " on the Markdown pages are projections. Real loaders refuse these files "
         "(`scripts/paper_artifacts.py`, `scripts/statement_evidence.py`), so none of them can stand in for a "
         "published record. Generated by `scripts/final_destination.py --write`.", "",
         "| Projected file | Real file | Written by |", "|---|---|---|"]
    writers = {"release/": "`bash hpc/publish_paper_artifacts_arrhenius.sh` (`scripts/paper_artifacts.py publish`)",
               "manifests/": "`scripts/paper_artifacts.py derive` (`scripts/run_manifests.py`)",
               "environment/": "`scripts/paper_artifacts.py derive` (the runs' `environment.json`)",
               "experiments/": "`scripts/paper_artifacts.py derive` (from `records:baselines`)",
               "paper/": "`make paper`, then `scripts/paper_artifacts.py derive`"}
    for relative in tree:
        if not relative.endswith(".json"):
            continue
        writer = next(v for k, v in writers.items() if relative.startswith(k))
        L.append(f"| [`{relative}`]({relative}) | `{relative}` | {writer} |")
    L += ["", "## Value classes", "",
          "| Class | Meaning | Convergence rule |", "|---|---|---|",
          "| exact | fixed by the released code, the effective configurations and the official data definitions | identical |",
          "| registered | the registered execution site (Arrhenius, one GH200 per job, CUDA 13 PyTorch wheel) | identical, or the projected text contained in the realized one |",
          f"| size | projected bytes | within a factor of {SIZE_FACTOR:g} |",
          f"| count | projected file counts | within {COUNT_SHARE:.0%} |",
          "| minimum | a lower bound | reached |",
          "| pending | supplied only by the realized output (digests, archive names, timestamps, data-dependent values) | supplied |",
          "", "## Double-checking the realized outputs", "",
          "```bash",
          "make final-destination-runs RUNS_ROOT=<suite>/runs   # completed runs, before publication",
          "make final-destination-converge                      # every published file against its projection",
          "```", "",
          f"The registered statement on this tree: {red(str(statement['fulfilled']) + ' of ' + str(statement['total']) + ' fulfilled')} "
          "([STATEMENT_COMPLIANCE.md](STATEMENT_COMPLIANCE.md)).", ""]
    return "\n".join(L)


def render_compliance(statement: dict, registered_text: str, ledger: list[dict], printed: list[dict]) -> str:
    label = {"fulfilled": "Fulfilled", "awaiting_measured_outputs": "Awaiting measured outputs", "missing": "Missing"}
    to_register = [r for r in ledger if r["contract"]["now"] == "to register"]
    registered = sum(r["registered_targets"] for r in ledger)
    L = ["# Compliance with the data-availability statement — at the final destination", "",
         "[Now](../STATEMENT_COMPLIANCE.md) · [The final destination](../docs/FINAL_DESTINATION.md) · [destination/](README.md)", "",
         "> " + registered_text, "",
         "This page is the projection of `STATEMENT_COMPLIANCE.md` once the measured outputs are published: each item is "
         "evaluated on the files of `destination/` with the checks of `scripts/statement_check.py`. Projected states are "
         "in " + red("red") + ". `make statement-fulfilled` decides the real state after publication.", "",
         "| Statement item | Now | At the destination | Fulfilled by |", "|---|---|---|---|"]
    for row in statement["items"]:
        state = label[row["destination"]]
        state = state if row["now"] == "fulfilled" and row["destination"] == "fulfilled" else red(state)
        L.append(f"| {row['phrase']} | {label[row['now']]} | {state} | {cell(row['evidence'])} |")
    L += ["", f"**{red(str(statement['fulfilled']) + ' of ' + str(statement['total']))}** registered items are fulfilled "
          "on the destination tree.", "",
          "## Every reported table and figure", "",
          f"`make paper` rebuilds {len(ledger)} numbered displays. {registered} printed values are registered as "
          f"comparison targets now; at the destination every one of them agrees, and the {len(to_register)} "
          "quantitative figures below carry their own registered contracts as well:", "",
          "| Figure | Work that closes the contract | Input |", "|---|---|---|"]
    for row in to_register:
        L.append(f"| {row['id']} | {cell(row['contract']['work'])} | {cell(row['contract']['closes'])} |")
    ledger = json.loads((ROOT / "paper" / "consistency_ledger.json").read_text(encoding="utf-8"))
    L += ["", "## Scientific consistency", "",
          "Each open question between the printed methods or values and the code is decided by an original record "
          "and a registered rule ([docs/CONSISTENCY.md](../docs/CONSISTENCY.md); `make consistency-audit "
          "RUNS_ROOT=<runs>` evaluates the checkable rules). At the destination every item is decided:", "",
          "| Item | Question | Now | At the destination |", "|---|---|---|---|"]
    words = {"aligned": "code aligned; records decide the printed values", "pending_records": "awaiting original records",
             "design_difference": "design difference; records decide", "needs_declaration": "needs an author declaration",
             "resolved": "resolved"}
    for item in ledger["items"]:
        final = "resolved" if item["status"] == "resolved" else red("decided by " + ("the author's declaration"
                                                                                    if item["status"] == "needs_declaration"
                                                                                    else "the original records"))
        L.append(f"| {item['id']} | {cell(item['title'])} | {words[item['status']]} | {final} |")
    L += ["", "## The printed availability text", "",
          "| Claim | Now | At the destination | Route |", "|---|---|---|---|"]
    for row in printed:
        if row["now"] == "fulfilled":
            continue
        L.append(f"| {row['id']}: {cell(row['text'][:80])}{'…' if len(row['text']) > 80 else ''} | {label.get(row['now'], row['now'])} | "
                 f"{red(label.get(row['projected'], row['projected']))} | {cell(row['note'])} |")
    L.append("")
    return "\n".join(L)
