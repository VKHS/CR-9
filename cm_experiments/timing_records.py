"""E17 timing records and the E14 official scores of the comparator ensemble grid.

Three measurement-record sets (contracts in ``records/specs.json``; displays in
``paper/build/displays/records_campaigns.py``):

``runtime_scaling``       Supplementary Figure S2. Per (dataset, instance): processed edges, wall time and
                          peak accelerator memory of one execution of the deployed rule.
``timing_anchors``        Supplementary Table S23b. Per dataset: the throughput anchor and the low/high
                          training-time anchors that bracket the per-input nominal execution time t_nom.
``ensemble_grid_scores``  Supplementary Table S26. Per (dataset, ensemble): the official test score of the
                          comparator's E-member ensemble of the ``ensemble_grid`` set.

Workflow (``scripts/run_timing_records.py``):

    scaling plan|run|export   plan registers the timed instances of each sealed parent; run times every
                              instance with the E17 TimingSession of ``cm_campaigns.timing`` (one job per
                              instance; ``--job K`` for array tasks); export binds the jobs to their
                              completed GPU allocations (sacct exports) and writes records/runtime_scaling
    anchors plan|run|export   training-time anchors from the sealed train-stage records (or scheduler
                              accounting); throughput anchor from the nominal timing rows of a completed
                              scaling experiment; run is torch-free
    grid-scores export        reads a completed ``scripts/run_experiment.py ensemble-grid`` output

Instances of the scaling sweep (declared choices; the manuscript does not specify the sweep):

* a transductive parent: its full benchmark graph ("full") and node-induced subgraphs on the first
  f N nodes (nearest integer, halves up) of one breadth-first order ("bfs-<f>"), f in {1/2, 1/4, 1/16, 1/64}
  by default. The
  order starts at a node drawn by a seeded permutation (default seed: the run's configuration seed),
  visits each level in ascending node id and continues from the next unvisited node of the permutation
  when a component is exhausted, so the subgraphs are nested and reproducible. Their node ids are written
  into the experiment directory and registered in the plan by SHA-256;
* an independent-graph collection: complete test-role graphs at the edge-count quantiles {0, 0.5, 1}
  (ties by node count, then graph id), "graph-<id>". No label is read.

One execution of the deployed rule on an instance follows the runtime's per-input scheduling: the E
members of the locked rule (``policy_lock.json``) in seed order, each through
``cross_morse.runtime.predict_dataset`` (transductive graph: one operator per instance, the input moved to
the device per member call) or through the per-graph body of that function (collection: operator and input
built per member execution), with the float64 ensemble mean on the host. The nominal execution is one
member (the first seed) on the registered nominal route (the frozen plant at full depth). Each repeat of
the TimingSession times one nominal execution and one deployed execution in alternating order.

Nothing in ``cross_morse`` is modified: the runtime is imported and called; ``InstanceContext`` is the
frozen-parent context of ``cm_experiments.ensemble_grid`` without its eager full-graph operator, so the
peak memory of a subgraph or a single graph is not inflated by the full graph. A timing job verifies the
parent's stage chain through ``fit`` against the current runtime source (plant checkpoints, controllers,
exploration reference) and checks the candidate registry and the policy lock against their seals; the
large prediction files of later stages are not reread. Planning, statistics and export are torch-free;
execution imports the runtime lazily.
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .artifacts import sha256, write_json
from .ensemble_grid import (OFFICIAL_SCORE, STAGES, ParentContext, RunnerError, SealedRun, candidate_identity,
                            check_identity, check_role, code_files, collect_ensemble_grid, create_plan,
                            environment_record, json_safe, load_plan, member_runs, provenance_hardware,
                            provenance_period, public_plan, read_json, read_member, registered_constants,
                            registered_grid, runtime_source_sha256, selected_jobs, slug, utc_now, write_record_set)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_FILES = ("cm_experiments/timing_records.py", "scripts/run_timing_records.py")
GIB = float(2 ** 30)
SCALING_SCHEMA = "cm-runtime-scaling-instance-1"
ANCHORS_SCHEMA = "cm-timing-anchors-dataset-1"
SCORES_SCHEMA = "cm-ensemble-grid-scores-1"
BINDING_SCHEMA = "cm-timing-allocation-binding-1"
DEFAULT_FRACTIONS = (0.5, 0.25, 0.0625, 0.015625)
DEFAULT_GRAPH_QUANTILES = (0.0, 0.5, 1.0)
DEFAULT_WARMUP, DEFAULT_REPETITIONS = 2, 5
# Forward-equivalent executions per training input and epoch: a training step costs between 1.5 and 3
# forward executions (declared bracket; the low anchor divides by the larger factor).
FORWARD_EQUIVALENTS = (1.5, 3.0)
NATIVE_DENSE_BACKBONES = ("shared_transport", "edge_weighted_transport", "heterophily")
DEFAULT_BACKBONES = {"ogbn-proteins": "edge_weighted_transport", "Roman-empire": "heterophily",
                     "PascalVOC-SP": "gps", "COCO-SP": "gps", "Peptides-func": "linear_gine"}
QUANTITIES = {
    "runtime_scaling": {"processed_edges_millions": "million edges", "wall_time_seconds": "s", "peak_memory_gib": "GiB"},
    "timing_anchors": {"throughput_anchor_seconds": "s", "training_anchor_low_seconds": "s",
                       "training_anchor_high_seconds": "s"},
    "ensemble_grid_scores": {"official_score": None},          # unit of the dataset's metric (OFFICIAL_SCORE)
}
PREDICTION_ENCODING = ("SHA-256 of the NumPy .npy v1.0 serialization (np.save, allow_pickle=False) of the float32 "
                       "C-order ensemble probabilities on the test-role rows in their sealed order")
SCALING_DEFINITIONS = {
    "processed_edges_millions": (
        "GraphOperator.num_edges of the instance's operator / 10^6: the operator's own count of the directed, "
        "de-duplicated nonzero entries (self-loops only for isolated vertices) that each sparse product of the "
        "execution visits; one operator per instance, as the runtime executes one graph at a time"),
    "wall_time_seconds": (
        "mean over the post-warm-up repeats of the E17 TimingSession 'seconds' column: synchronized wall time of one "
        "execution of the deployed rule on the instance (its E members in seed order through the runtime's per-input "
        "path, ensemble mean on the host)"),
    "peak_memory_gib": (
        "maximum over the post-warm-up repeats of the largest peak_cuda_allocated_bytes of the execution's member "
        "executions (cross_morse.policies.execute_policy synchronizes and resets the CUDA peak statistics before each "
        "member execution and reads torch.cuda.max_memory_allocated at its end), / 2^30; unavailable on CPU, never "
        "recorded as zero"),
}
ANCHOR_DEFINITIONS = {
    "training_anchor_low_seconds": (
        "q / c_high, with q the mean over the plant seeds of T_s / (e_s n_s): T_s the measured training time of seed s "
        "(the runtime's elapsed time recorded in train/plant_seed<s>_done.json, or the scheduler allocation seconds the "
        "job map attributes to its backbone training when accounting is supplied), e_s its executed epochs "
        "(epochs_completed), n_s the training inputs per epoch (one full transductive graph; |D_fit,train| complete "
        "graphs of a collection); c_high the upper forward-equivalent factor"),
    "training_anchor_high_seconds": "q / c_low with the lower forward-equivalent factor c_low",
    "throughput_anchor_seconds": (
        "kappa W_nom: W_nom = L (N d^2 + E d) of Supplementary Table S23b (registered dataset statistics, configured "
        "depth and width) at the sustained throughput 1/kappa of the plant, kappa the slope of the least-squares fit "
        "t = t0 + kappa W of the post-warm-up nominal wall times against the runtime-counted multiply-adds W of the "
        "nominal execution over the plant's runtime_scaling instances; full-batch plants with a native dense layer only"),
}


# ============================================================================= shared helpers
def experiment_source_identity() -> dict:
    return {name: sha256(ROOT / name) for name in EXPERIMENT_FILES if (ROOT / name).is_file()}


def _dump(path, value) -> str:
    """Atomic JSON write that replaces a file left by an interrupted attempt; returns its SHA-256."""
    return _write_bytes(path, (json.dumps(json_safe(value), indent=1, sort_keys=True, allow_nan=False) + "\n").encode())


def _write_bytes(path, data: bytes) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return sha256(path)


def _copy(source, destination) -> str:
    """Copy a sealed or completed file unchanged; returns its SHA-256."""
    source, destination = Path(source), Path(destination)
    if source.is_symlink() or not source.is_file():
        raise RunnerError(f"missing input file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    digest = sha256(destination)
    if digest != sha256(source):
        raise RunnerError(f"{source} changed while it was copied")
    return digest


class _HashWriter:
    def __init__(self):
        self.digest = hashlib.sha256()

    def write(self, data) -> int:
        self.digest.update(data)
        return len(data)


def npy_sha256(array) -> str:
    """SHA-256 of the .npy bytes ``np.save`` writes for this array, computed without writing a file."""
    writer = _HashWriter()
    np.lib.format.write_array(writer, np.ascontiguousarray(array), version=(1, 0), allow_pickle=False)
    return writer.digest.hexdigest()


def npy_bytes(array) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array), allow_pickle=False)
    return buffer.getvalue()


def sealed_output(run: SealedRun, stage: str, relative: str) -> str:
    """SHA-256 of a file sealed by a lifecycle stage, checked against that stage's manifest (torch-free)."""
    run.require(stage)
    manifest = read_json(run.root / "provenance" / "stages" / f"{STAGES.index(stage):02d}_{stage}.json")
    recorded = {item["path"]: item["sha256"] for item in manifest.get("outputs", [])}
    if relative not in recorded:
        raise RunnerError(f"{run.root.name}: {relative} is not an output sealed by the {stage} stage")
    path = run.root / relative
    if not path.is_file():
        raise RunnerError(f"missing input file: {path}")
    digest = sha256(path)
    if digest != recorded[relative]:
        raise RunnerError(f"{run.root.name}: {relative} differs from the hash sealed by the {stage} stage")
    run.read[relative] = digest
    return digest


def _number(value, where: str) -> float:
    if value is None or isinstance(value, bool):
        raise RunnerError(f"{where}: value is missing")
    value = float(value)
    if not math.isfinite(value):
        raise RunnerError(f"{where}: value is not finite")
    return value


def _period(times) -> str | None:
    times = sorted(t for t in times if t)
    return f"{times[0]}/{times[-1]}" if times else None


@contextlib.contextmanager
def registered_coverage(set_name: str, contract: dict):
    """Coverage contract for ``write_record_set``, which looks it up in cm_campaigns.contract.RECORD_SETS.

    A set registered only in records/specs.json has no entry there; for the duration of one export the
    contract derived from the plan is supplied in-process (the module file is not changed), and an entry
    that the repository already defines takes precedence.
    """
    from cm_campaigns.contract import RECORD_SETS
    if set_name in RECORD_SETS:
        yield RECORD_SETS[set_name]
        return
    RECORD_SETS[set_name] = contract
    try:
        yield contract
    finally:
        if RECORD_SETS.get(set_name) is contract:
            del RECORD_SETS[set_name]


def deposit(set_name, rows, files, provenance, *, coverage, reasons, output, root=None, replace=False,
            allow_partial=False, command=None, inputs=(), generated=None) -> Path:
    """Write the set with ``ensemble_grid.write_record_set`` (staging, private scan, manifest, verify_set).

    ``reasons`` lists what keeps the measurement from being complete; such a set is written only with
    ``allow_partial`` and never into records/ (write_record_set refuses a partial set there).
    """
    reasons = list(dict.fromkeys(reasons))
    if reasons and not allow_partial:
        raise RunnerError(f"{set_name}: the measurement is incomplete ({'; '.join(reasons)}); pass --allow-partial "
                          "to write it to another root")
    files = dict(files)
    files.update(code_files(output))
    for name in EXPERIMENT_FILES:
        files[f"code/{name}"] = ROOT / name
    generated = dict(generated or {})
    generated.setdefault("code/plan.json", public_plan(output))
    record = dict(provenance, measurement_complete=not reasons, incomplete_measurement=reasons,
                  experiment_code_sha256=experiment_source_identity())
    with registered_coverage(set_name, coverage):
        return write_record_set(set_name, rows, files, record, root=root, replace=replace, allow_partial=allow_partial,
                                command=command, inputs=inputs, generated=generated)


def value_row(keys: dict, quantity: str, value, unit: str, source_file: str, locator: str) -> dict:
    return {**keys, "quantity": quantity, "value": _number(value, f"{keys} {quantity}"), "unit": unit,
            "source_file": source_file, "locator": locator}


def plant_layout(config: dict, metadata: dict, dataset: str) -> dict:
    """Execution layout of the frozen plant: full batch or graph episodes, native dense layer or inserted sites."""
    model = config["model"]
    backbone = model.get("backbone", DEFAULT_BACKBONES.get(dataset, "shared_transport"))
    depth = int(model["depth"])
    native_layers = model.get("native_layers")
    collection = bool(metadata.get("independent_graphs"))
    inserted = backbone in ("gps", "linear_gine") or (native_layers is not None and int(native_layers) < depth)
    dense = backbone in NATIVE_DENSE_BACKBONES and not inserted
    return {"backbone": backbone, "depth": depth, "hidden_dim": int(model["hidden_dim"]), "native_layers": native_layers,
            "independent_graphs": collection, "full_batch": not collection, "inserted_site_plant": inserted,
            "native_dense_layer_at_every_site": dense, "throughput_anchor_applicable": dense and not collection,
            "precision": model.get("precision")}


def nominal_work_statistics(run: SealedRun, metadata: dict, layout: dict) -> dict:
    """W_nom = L (N d^2 + E d) of Supplementary Table S23b for one nominal member execution."""
    L, d = layout["depth"], layout["hidden_dim"]
    if run.synthetic:
        nodes, edges = int(metadata["num_nodes"]), int(metadata["undirected_edges"])
        source = "prepared synthetic graph: num_nodes and undirected_edges of prepared/metadata.json (software fixture)"
    else:
        stats = registered_constants()["dataset_statistics_table_S6c_S23b"].get(run.dataset) or {}
        if "nodes" not in stats or "reported_edges" not in stats:
            raise RunnerError(f"{run.dataset}: no registered node and edge counts for W_nom")
        nodes, edges = int(stats["nodes"]), int(stats["reported_edges"])
        source = ("paper/registered_constants.json dataset_statistics_table_S6c_S23b (nodes, reported_edges), as "
                  "Supplementary Table S23b prints W_nom")
    return {"L": L, "d": d, "N": nodes, "E": edges, "W_nom": float(L * (nodes * d * d + edges * d)),
            "definition": "W_nom = L (N d^2 + E d)", "source": source}


def nominal_candidate(candidates: list[dict], deployed: dict) -> dict:
    """The registered nominal route (frozen plant at full depth), at the deployed ensemble size when registered."""
    nominal = [dict(c, registry_index=i) for i, c in enumerate(candidates) if c.get("route") == "nominal"]
    same = [c for c in nominal if c.get("ensemble") == deployed.get("ensemble")]
    chosen = same or nominal
    if not chosen:
        raise RunnerError("the run registers no nominal (full-depth plant) candidate")
    return chosen[0]


def locked_candidates(run: SealedRun, entry: dict) -> tuple[dict, dict]:
    deployed = run.deployed()
    if deployed is None:
        raise RunnerError(f"{run.root.name}: policy_lock.json is absent; the deployed rule is not locked")
    nominal = nominal_candidate(run.candidates(), deployed)
    if candidate_identity(deployed) != entry["deployed"] or candidate_identity(nominal) != entry["nominal"]:
        raise RunnerError(f"{run.root.name}: the locked deployed or nominal candidate differs from the plan")
    return deployed, nominal


# ============================================================================= timing rows (torch-free)
def read_timing_rows(path) -> list[dict]:
    from cm_campaigns.timing import COLUMNS
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(COLUMNS):
            raise RunnerError(f"{Path(path).name}: timing rows must have the E17 columns {list(COLUMNS)}")
        return list(reader)


def write_timing_rows(path, rows) -> str:
    """The session's rows in cm_campaigns.timing layout, written once (an interrupted attempt is replaced)."""
    from cm_campaigns.timing import write_rows
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".csv.tmp")
    os.close(fd)
    try:
        write_rows(rows, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return sha256(path)


def timing_means(rows) -> dict:
    """Post-warm-up means of the method and nominal wall seconds (protocol E17)."""
    measured = [r for r in rows if int(r["warmup"]) == 0]
    if len(measured) < 3:
        raise RunnerError("protocol E17 needs at least three post-warm-up repeats")
    seconds = [_number(r["seconds"], "timing row seconds") for r in measured]
    nominal = [_number(r["nominal_seconds"], "timing row nominal_seconds") for r in measured]
    return {"wall_time_seconds": math.fsum(seconds) / len(seconds),
            "nominal_wall_time_seconds": math.fsum(nominal) / len(nominal),
            "measured_repeats": len(measured), "warmup_repeats": len(rows) - len(measured)}


def timing_job_key(explicit=None) -> str:
    """Key carried by the timing rows: an explicit sacct key, else the Slurm job ID, bound to sacct at export."""
    if explicit:
        return str(explicit)
    env = os.environ
    if env.get("SLURM_ARRAY_JOB_ID") and env.get("SLURM_ARRAY_TASK_ID"):
        return f"slurm:{env['SLURM_ARRAY_JOB_ID']}_{env['SLURM_ARRAY_TASK_ID']}"
    if env.get("SLURM_JOB_ID"):
        return f"slurm:{env['SLURM_JOB_ID']}"
    return "unscheduled"


# ============================================================================= runtime_scaling: instances (torch-free)
def bfs_order(edge_index, num_nodes: int, *, seed: int, limit: int) -> np.ndarray:
    """The first ``limit`` nodes of one deterministic breadth-first order (see the module docstring)."""
    edge_index = np.asarray(edge_index, dtype=np.int64)
    limit = int(limit)
    if not 1 <= limit <= num_nodes:
        raise RunnerError("the breadth-first prefix must hold between one node and every node")
    src, dst = edge_index
    order = np.lexsort((dst, src))
    indices = dst[order]
    indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=num_nodes), out=indptr[1:])
    visited = np.zeros(num_nodes, dtype=bool)
    roots = np.random.default_rng(int(seed)).permutation(num_nodes)
    levels, count, cursor = [], 0, 0
    while count < limit:
        while visited[roots[cursor]]:
            cursor += 1
        frontier = np.asarray([roots[cursor]], dtype=np.int64)
        visited[frontier] = True
        while len(frontier) and count < limit:
            levels.append(frontier)
            count += len(frontier)
            starts = indptr[frontier]
            lengths = indptr[frontier + 1] - starts
            total = int(lengths.sum())
            if not total:
                break
            offsets = np.repeat(starts - np.concatenate(([0], np.cumsum(lengths)[:-1])), lengths) + np.arange(total)
            neighbours = indices[offsets]
            frontier = np.unique(neighbours[~visited[neighbours]])
            visited[frontier] = True
    return np.concatenate(levels)[:limit]


def induced_mask(edge_index, nodes, num_nodes: int) -> np.ndarray:
    """Mask of the canonical edge entries with both endpoints in ``nodes``."""
    member = np.zeros(num_nodes, dtype=bool)
    member[nodes] = True
    return member[edge_index[0]] & member[edge_index[1]]


def induced_edges(edge_index, nodes, num_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Mask of the canonical edge entries with both endpoints in ``nodes`` and those entries relabelled."""
    edge_index = np.asarray(edge_index, dtype=np.int64)
    keep = induced_mask(edge_index, nodes, num_nodes)
    position = np.full(num_nodes, -1, dtype=np.int64)
    position[nodes] = np.arange(len(nodes), dtype=np.int64)
    return keep, position[edge_index[:, keep]]


def half_up(value: float) -> int:
    """Nearest integer with halves rounded up (Python's round() rounds halves to even)."""
    return int(math.floor(value + 0.5))


def subgraph_instances(edge_index, num_nodes: int, fractions, seed: int) -> list[dict]:
    fractions = sorted({float(f) for f in fractions}, reverse=True)
    order = bfs_order(edge_index, num_nodes, seed=seed, limit=max(1, half_up(fractions[0] * num_nodes)))
    instances, sizes = [], set()
    for fraction in fractions:
        size = max(1, half_up(fraction * num_nodes))
        if size in sizes:
            raise RunnerError(f"subgraph fractions {fractions} give equal node counts on a {num_nodes}-node graph")
        sizes.add(size)
        nodes = np.sort(order[:size])
        instances.append({"instance": f"bfs-{fraction:g}", "kind": "subgraph", "fraction": fraction, "nodes": size,
                          "edge_entries": int(induced_mask(edge_index, nodes, num_nodes).sum()), "node_ids": nodes})
    return instances


def graph_instances(graph_ptr, edge_ptr, test_graphs, quantiles) -> list[dict]:
    """Complete test-role graphs at the edge-count quantiles (ties by node count, then graph id)."""
    graph_ptr, edge_ptr = np.asarray(graph_ptr, dtype=np.int64), np.asarray(edge_ptr, dtype=np.int64)
    ids = np.unique(np.asarray(test_graphs, dtype=np.int64))
    if not len(ids):
        raise RunnerError("the test role holds no complete graph")
    edges, nodes = edge_ptr[ids + 1] - edge_ptr[ids], graph_ptr[ids + 1] - graph_ptr[ids]
    ranked = ids[np.lexsort((ids, nodes, edges))]
    instances, seen = [], set()
    for quantile in sorted({float(q) for q in quantiles}):
        rank = half_up(quantile * (len(ranked) - 1))
        graph = int(ranked[rank])
        if graph in seen:
            continue
        seen.add(graph)
        instances.append({"instance": f"graph-{graph}", "kind": "graph", "graph_id": graph, "quantile": quantile,
                          "rank": rank, "graphs_ranked": int(len(ranked)),
                          "nodes": int(graph_ptr[graph + 1] - graph_ptr[graph]),
                          "edge_entries": int(edge_ptr[graph + 1] - edge_ptr[graph])})
    return instances


def instance_directory(output, job: dict) -> Path:
    return Path(output) / "jobs" / job["slug"] / slug(job["member"]) / slug(job["instance"])


def plan_runtime_scaling(parents, output, *, fractions=None, graph_quantiles=None, subgraph_seed=None,
                         warmup=DEFAULT_WARMUP, repetitions=DEFAULT_REPETITIONS) -> Path:
    if not parents:
        raise RunnerError("give at least one --parent run")
    fractions = tuple(sorted({float(f) for f in (fractions or DEFAULT_FRACTIONS)}, reverse=True))
    if any(not 0 < f < 1 for f in fractions):
        raise RunnerError("subgraph fractions must lie strictly between 0 and 1")
    quantiles = tuple(sorted({float(q) for q in (graph_quantiles or DEFAULT_GRAPH_QUANTILES)}))
    if any(not 0 <= q <= 1 for q in quantiles):
        raise RunnerError("graph quantiles must lie in [0, 1]")
    if int(warmup) < 1 or int(repetitions) < 3:
        raise RunnerError("protocol E17 needs at least one warm-up repeat and three measured repeats")
    jobs, datasets, arrays = [], [], {}
    for parent in parents:
        members = member_runs(parent)
        run = SealedRun(members[0])
        if any(d["dataset"] == run.dataset for d in datasets):
            raise RunnerError(f"{parent}: one parent (run or campaign) per dataset")
        run.require("admit")
        deployed = run.deployed()
        if deployed is None:
            raise RunnerError(f"{run.root.name}: policy_lock.json is absent; the deployed rule is not locked")
        nominal = nominal_candidate(run.candidates(), deployed)
        sealed_output(run, "prepare", "prepared/metadata.json")
        metadata = read_json(run.root / "prepared" / "metadata.json")
        layout = plant_layout(run.config, metadata, run.dataset)
        sealed_output(run, "prepare", "prepared/graph.npz")
        seed = int(run.config["seed"]) if subgraph_seed is None else int(subgraph_seed)
        with np.load(run.root / "prepared" / "graph.npz", allow_pickle=False) as graph:
            if layout["independent_graphs"]:
                sealed_output(run, "prepare", "prepared/graph_splits.npz")
                with np.load(run.root / "prepared" / "graph_splits.npz", allow_pickle=False) as roles:
                    test_graphs = roles["test"]
                instances = graph_instances(graph["graph_ptr"], graph["edge_ptr"], test_graphs, quantiles)
            else:
                edge_index, nodes = graph["edge_index"], int(metadata["num_nodes"])
                instances = [{"instance": "full", "kind": "full", "nodes": nodes, "edge_entries": int(edge_index.shape[1])}]
                instances += subgraph_instances(edge_index, nodes, fractions, seed)
        member_seeds = [int(s) for s in run.config["training"]["seeds"][:int(deployed["ensemble"])]]
        split = run.root.name if len(members) > 1 else "official"
        for instance in instances:
            node_ids = instance.pop("node_ids", None)
            if node_ids is not None:
                data = npy_bytes(np.asarray(node_ids, dtype=np.int64))
                instance["node_ids_file"] = f"subgraphs/{slug(run.dataset)}/{slug(instance['instance'])}.npy"
                instance["node_ids_sha256"] = hashlib.sha256(data).hexdigest()
                arrays[instance["node_ids_file"]] = data
            jobs.append({"job": len(jobs), "dataset": run.dataset, "slug": slug(run.dataset), "member": run.root.name,
                         "run": str(run.root), "split": split, "instance": instance["instance"], "definition": instance,
                         "identity": run.identity(), "deployed": candidate_identity(deployed),
                         "nominal": candidate_identity(nominal), "member_seeds": member_seeds, "layout": layout})
        datasets.append({"dataset": run.dataset, "slug": slug(run.dataset), "parent": str(Path(parent).resolve()),
                         "member": run.root.name, "members_available": len(members), "synthetic": run.synthetic,
                         "subgraph_seed": None if layout["independent_graphs"] else seed,
                         "instances": [i["instance"] for i in instances], "parent_files_read": run.inputs()})
    path = create_plan(output, {
        "experiment": "E17", "record_set": "runtime_scaling", "datasets": datasets, "jobs": jobs,
        "fractions": list(fractions), "graph_quantiles": list(quantiles), "warmup": int(warmup),
        "repetitions": int(repetitions), "experiment_code_sha256": experiment_source_identity(),
        "definitions": SCALING_DEFINITIONS,
        "sweep_status": ("declared choice: the manuscript names full benchmark graphs and registered subgraphs but not "
                         "the sweep; breadth-first node-induced subgraphs of transductive graphs and edge-count "
                         "quantiles of a collection's test graphs, as stated in cm_experiments/timing_records.py"),
        "protocol": ("E17 TimingSession per instance: warm-up and measured repeats alternate one nominal member "
                     "execution and one execution of the locked deployed rule, synchronized before and after each "
                     "call; processed edges are the instance operator's own edge count")})
    for relative, data in arrays.items():
        target = path.parent / relative
        if target.exists():
            raise RunnerError(f"{target} exists; use a new experiment directory")
        if _write_bytes(target, data) != hashlib.sha256(data).hexdigest():
            raise RunnerError(f"{target} was not written intact")
    return path


# ============================================================================= runtime_scaling: statistics (torch-free)
def member_record(trace: dict, edges: int, nodes: int) -> dict:
    """The parts of one member execution's runtime trace that the measurement uses."""
    graph_ops = trace.get("graph_operations") or {}
    peak = trace.get("peak_cuda_allocated_bytes")
    return {"wall_seconds": float(trace["wall_seconds"]), "multiply_add_work": int(trace["multiply_add_work"]),
            "deployed_multiply_add_work": int(trace.get("deployed_multiply_add_work", trace["multiply_add_work"])),
            "dense_multiply_adds": int((trace.get("plant_operations") or {}).get("dense_multiply_adds", 0)),
            "sparse_products": int(graph_ops.get("sparse_products", 0)),
            "sparse_multiply_adds": int(graph_ops.get("sparse_multiply_adds", 0)),
            "controller_dense_multiply_adds": int((trace.get("controller_operations") or {}).get("approx_dense_multiply_adds", 0)),
            "sensor_dense_multiply_adds": int((trace.get("sensor_operations") or {}).get("dense_multiply_adds", 0)),
            "peak_cuda_allocated_bytes": None if peak is None else int(peak),
            "operator_edges": int(edges), "operator_nodes": int(nodes), "actions": len(trace.get("actions") or []),
            "terminal_depth": trace.get("terminal_depth"), "feasible": bool(trace.get("feasible", True)),
            "support_violations": int(trace.get("support_violations", 0))}


def summarize_instance(rows: list[dict], records: dict, counted: dict, *, device: str) -> dict:
    """Processed edges, wall time and peak memory of one instance from its timing rows and execution records.

    ``records[kind][k]`` is the execution record of the k-th call of the nominal or deployed callable
    within the session (repeat k); ``counted`` the untimed counting execution before the session.
    """
    timing = timing_means(rows)
    if len(records["nominal"]) != len(rows) or len(records["deployed"]) != len(rows):
        raise RunnerError("every timed repeat needs its execution record")
    if [int(r["repeat"]) for r in rows] != list(range(len(rows))):
        raise RunnerError("timing rows are not the consecutive repeats of one session")
    deployed = [counted["deployed"], *records["deployed"]]
    nominal = [counted["nominal"], *records["nominal"]]
    members = [m for execution in deployed for m in execution["members"]]
    edges = {m["operator_edges"] for m in members} | {n["operator_edges"] for n in nominal}
    nodes = {m["operator_nodes"] for m in members} | {n["operator_nodes"] for n in nominal}
    if len(edges) != 1 or len(nodes) != 1:
        raise RunnerError("the instance operator changed between executions")
    sparse = sum(m["sparse_products"] for m in counted["deployed"]["members"])
    if sparse < 1:
        raise RunnerError("the deployed rule executed no sparse product on the instance")
    measured = [i for i, r in enumerate(rows) if int(r["warmup"]) == 0]
    peaks, missing = [], False
    for i in measured:
        readings = [m["peak_cuda_allocated_bytes"] for m in records["deployed"][i]["members"]]
        if any(r is None for r in readings):
            missing = True
            break
        peaks.append(max(readings))
    if missing or not str(device).startswith("cuda"):
        peak_bytes, status = None, f"unavailable: no accelerator memory reading on device {device}"
    else:
        peak_bytes, status = max(peaks), "measured"
    work = [sum(m["deployed_multiply_add_work"] for m in execution["members"]) for execution in deployed]
    nominal_work = [n["multiply_add_work"] for n in nominal]
    edge_count, node_count = edges.pop(), nodes.pop()
    return {"processed_edges": edge_count, "processed_edges_millions": edge_count / 1e6, "processed_nodes": node_count,
            "sparse_products_per_execution": sparse, **timing,
            "peak_memory_bytes": peak_bytes, "peak_memory_gib": None if peak_bytes is None else peak_bytes / GIB,
            "peak_memory_status": status, "peak_memory_per_repeat_bytes": peaks if peak_bytes is not None else None,
            "deployed_multiply_add_work": work[0], "nominal_multiply_add_work": nominal_work[0],
            "nominal_dense_multiply_adds": counted["nominal"]["dense_multiply_adds"],
            "nominal_sparse_multiply_adds": counted["nominal"]["sparse_multiply_adds"],
            "work_constant_across_executions": len(set(work)) == 1 and len(set(nominal_work)) == 1}


def write_scaling_result(output, plan: dict, entry: dict, directory, *, rows, records, counted, device, job_key,
                         environment, parent_files, started, session_started, session_finished,
                         resident_bytes=None) -> Path:
    """The job's raw outputs; scaling.json is written last and marks the job complete.

    ``resident_bytes``: device memory allocated before the session (resident members, operator, input
    tensors), informational; the peak readings include it.
    """
    from cm_campaigns.timing import COLUMNS
    directory = Path(directory)
    summary = summarize_instance(rows, records, counted, device=device)
    for row in rows:
        if row["dataset"] != entry["dataset"] or row["input_id"] != entry["instance"]:
            raise RunnerError("timing rows belong to another dataset or instance")
    _dump(directory / "environment.json", environment)
    timing_sha = write_timing_rows(directory / "timing.csv", rows)
    executions_sha = _dump(directory / "executions.json", {
        "schema": "cm-runtime-scaling-executions-1", "counting_execution": counted,
        "repeats": [{"repeat": i, "warmup": int(rows[i]["warmup"]), "nominal": records["nominal"][i],
                     "deployed": records["deployed"][i]} for i in range(len(rows))]})
    log = (f"E17 TimingSession (cm_campaigns.timing)\ndataset: {entry['dataset']}\ninstance: {entry['instance']}\n"
           f"member run: {entry['member']}\ndevice: {device}\nwarm-up repeats: {plan['warmup']}\n"
           f"measured repeats: {plan['repetitions']}\njob key: {job_key}\nstarted: {session_started}\n"
           f"finished: {session_finished}\n")
    log_sha = _write_bytes(directory / "session.log", log.encode())
    write_json(directory / "scaling.json", json_safe({
        "schema": SCALING_SCHEMA, "experiment": "E17", "record_set": "runtime_scaling", "dataset": entry["dataset"],
        "member": entry["member"], "split": entry["split"], "instance": entry["instance"],
        "definition": entry["definition"], "deployed": entry["deployed"], "nominal": entry["nominal"],
        "member_seeds": entry["member_seeds"], "device": str(device), "job_key": job_key, **summary,
        "resident_device_bytes_before_session": resident_bytes,
        "timing_rows": {"file": "timing.csv", "sha256": timing_sha, "columns": list(COLUMNS)},
        "executions": {"file": "executions.json", "sha256": executions_sha},
        "session_log": {"file": "session.log", "sha256": log_sha},
        "protocol": {"warmup": plan["warmup"], "repetitions": plan["repetitions"],
                     "session": "cm_campaigns.timing.TimingSession.measure: alternating nominal/method order, "
                                "synchronized before and after each call; one untimed counting execution first"},
        "definitions": SCALING_DEFINITIONS,
        "identities": {"runtime_source_sha256": runtime_source_sha256(), "parent": entry["identity"],
                       "plan_sha256": sha256(Path(output) / "plan.json"), "parent_files_read": parent_files,
                       "experiment_code_sha256": experiment_source_identity()},
        "started_utc": started, "finished_utc": utc_now()}))
    return directory / "scaling.json"


# ============================================================================= runtime_scaling: execution (torch)
class InstanceContext(ParentContext):
    """The frozen-parent context of ``ensemble_grid`` without its eager full-graph operator.

    Each timed instance builds its own operator on the device, so the device memory of a subgraph or
    a single graph holds that instance only.
    """

    def __init__(self, run: SealedRun, device, *, through: str):
        from cross_morse.data import load_data
        from cross_morse.provenance import current_source_sha256, verify_stage_chain
        from cross_morse.runtime import is_graph_collection
        verify_stage_chain(run.root, through, expected_source_sha256=current_source_sha256())
        self.run, self.config, self.device = run, run.config, str(device)
        self.data = load_data(run.root).protect_labels()
        self.collection = is_graph_collection(self.data)
        self.graph = None
        self.reference = dict(np.load(run.root / "explore" / "nominal_reference.npz", allow_pickle=False))
        self._controllers, self._labels = {}, {}


def instance_input(output, ctx: InstanceContext, definition: dict, device) -> SimpleNamespace:
    """The registered instance: its input view and operator, verified against the plan."""
    import torch
    from cross_morse.runtime import graph_operator
    data, kind = ctx.data, definition["kind"]
    if kind == "graph":
        graph = int(definition["graph_id"])
        if not ctx.collection or graph not in set(np.asarray(data.graph_splits["test"]).tolist()):
            raise RunnerError(f"graph {graph} is not a test-role graph of this collection")
        nodes = int(data.graph_ptr[graph + 1] - data.graph_ptr[graph])
        edges = int(data.edge_ptr[graph + 1] - data.edge_ptr[graph])
        if (nodes, edges) != (definition["nodes"], definition["edge_entries"]):
            raise RunnerError(f"graph {graph} differs from its registered size")
        return SimpleNamespace(kind=kind, graph_id=graph, view=None, operator=None)
    if ctx.collection:
        raise RunnerError("a collection is timed one complete graph at a time")
    if kind == "full":
        if (data.num_nodes, int(data.edge_index.shape[1])) != (definition["nodes"], definition["edge_entries"]):
            raise RunnerError("the prepared graph differs from its registered size")
        return SimpleNamespace(kind=kind, graph_id=None, view=data, operator=graph_operator(data, device))
    if kind != "subgraph":
        raise RunnerError(f"unknown instance kind {kind!r}")
    path = Path(output) / definition["node_ids_file"]
    if not path.is_file() or sha256(path) != definition["node_ids_sha256"]:
        raise RunnerError(f"{definition['node_ids_file']} is missing or differs from its registered SHA-256")
    nodes = np.load(path, allow_pickle=False)
    keep, pairs = induced_edges(data.edge_index.numpy(), nodes, data.num_nodes)
    if (len(nodes), int(keep.sum())) != (definition["nodes"], definition["edge_entries"]):
        raise RunnerError(f"{definition['instance']}: the induced subgraph differs from its registered size")
    mask = torch.from_numpy(keep)
    view = SimpleNamespace(x=data.x[torch.from_numpy(nodes)], edge_index=torch.from_numpy(pairs),
                           edge_weight=None if data.edge_weight is None else data.edge_weight[mask],
                           num_nodes=int(len(nodes)), graph_ptr=None)
    return SimpleNamespace(kind=kind, graph_id=None, view=view, operator=graph_operator(view, device))


def run_runtime_scaling(output, *, job=None, device="cuda", trust=False, job_key=None) -> list[Path]:
    if not trust:
        raise RunnerError("loading local checkpoints requires explicit trust (--trust-local-checkpoints)")
    output = Path(output).resolve()
    plan = load_plan(output, "runtime_scaling")
    key = timing_job_key(job_key)
    done = []
    for entry in selected_jobs(plan, job):
        directory = instance_directory(output, entry)
        if not (directory / "scaling.json").is_file():
            _run_scaling_job(output, plan, entry, directory, device, key)
        done.append(directory / "scaling.json")
    return done


def _run_scaling_job(output, plan, entry, directory, device, job_key):
    import gc
    import torch
    from cross_morse.policies import PolicyExecutionError, execute_policy
    from cross_morse.runtime import graph_operator, predict_dataset
    run = SealedRun(entry["run"])
    check_identity(run, entry["identity"])
    # The job reads the plant checkpoints, controllers and reference (verified through fit) and the
    # candidate registry and policy lock (checked against their seals); later predictions are not reread.
    sealed_output(run, "select", "select/candidates.json")
    sealed_output(run, "admit", "policy_lock.json")
    deployed, nominal = locked_candidates(run, entry)
    started = utc_now()
    environment = environment_record(device)
    directory.mkdir(parents=True, exist_ok=True)
    cuda = str(device).startswith("cuda")
    gc.collect()                        # nothing of an earlier job of this process stays allocated
    if cuda:
        torch.cuda.empty_cache()
    ctx = InstanceContext(run, device, through="fit")
    config, reference = ctx.config, ctx.reference
    instance = instance_input(output, ctx, entry["definition"], device)
    models = [ctx.plant(seed) for seed in entry["member_seeds"]]
    controller = ctx.controller_for(deployed)

    def execute(model, candidate, policy_controller):
        with torch.no_grad():               # predict_dataset's own no-grad scope, also on the per-graph path
            if instance.graph_id is not None:     # the per-graph body of cross_morse.runtime.predict_dataset
                episode = ctx.data.graph(instance.graph_id)
                operator = graph_operator(episode, device)
                x = episode.x.to(device)
                probabilities, trace = execute_policy(model, x, operator, config, candidate, policy_controller,
                                                      reference)
                edges, nodes = operator.num_edges, operator.num_nodes
                del x, operator, episode
            else:
                probabilities, trace = predict_dataset(model, instance.view, instance.operator, config, candidate,
                                                       policy_controller, reference, device=device)
                edges, nodes = instance.operator.num_edges, instance.operator.num_nodes
        return probabilities, member_record(trace, edges, nodes)

    resident = int(torch.cuda.memory_allocated(device)) if cuda else None
    try:
        session = time_instance(entry, plan, job_key=job_key,
                                execute_nominal=lambda: execute(models[0], nominal, None)[1],
                                execute_members=lambda: (execute(model, deployed, controller) for model in models))
    except PolicyExecutionError as exc:
        raise RunnerError(f"{entry['dataset']}/{entry['instance']}: the locked rule failed numerically "
                          f"({exc.reason}); the instance is not timed") from None
    write_scaling_result(output, plan, entry, directory, device=device, job_key=job_key, environment=environment,
                         parent_files=run.inputs(), started=started, resident_bytes=resident, **session)


def time_instance(entry: dict, plan: dict, *, job_key: str, execute_nominal, execute_members) -> dict:
    """The E17 session of one instance (torch-agnostic orchestration).

    ``execute_nominal()`` executes one nominal member and returns its member record;
    ``execute_members()`` returns an iterator that executes the deployed rule's members one at a time,
    yielding (probabilities, member record). One untimed counting execution of each precedes the
    session; the k-th record of each callable belongs to repeat k of the session's rows.
    """
    from cm_campaigns.timing import TimingSession
    records = {"nominal": [], "deployed": []}

    def nominal_call():
        records["nominal"].append(execute_nominal())

    def deployed_call():
        aggregate, members = None, []
        for probabilities, record in execute_members():
            aggregate = np.array(probabilities, dtype=np.float64) if aggregate is None else aggregate + probabilities
            members.append(record)
        if aggregate is None:
            raise RunnerError("the deployed rule has no member")
        ensemble = (aggregate / len(members)).astype(np.float32)      # the runtime's ensemble mean
        records["deployed"].append({"members": members, "finite": bool(np.isfinite(ensemble).all())})

    nominal_call()          # untimed counting execution: work, operator size, lazy initialisation
    deployed_call()
    counted = {kind: values.pop() for kind, values in records.items()}
    session = TimingSession(dataset=entry["dataset"], method="deployed rule", seed=int(entry["member_seeds"][0]),
                            split=entry["split"], job_key=job_key, log="session.log")
    session_started = utc_now()
    rows = session.measure(input_id=entry["instance"], nominal=nominal_call, method=deployed_call,
                           nominal_work=counted["nominal"]["multiply_add_work"],
                           method_work=sum(m["deployed_multiply_add_work"] for m in counted["deployed"]["members"]),
                           warmup=plan["warmup"], repetitions=plan["repetitions"])
    session_finished = utc_now()
    if not all(execution["finite"] for execution in [counted["deployed"], *records["deployed"]]):
        raise RunnerError(f"{entry['dataset']}/{entry['instance']}: nonfinite ensemble output")
    return {"rows": rows, "records": records, "counted": counted, "session_started": session_started,
            "session_finished": session_finished}


# ============================================================================= runtime_scaling: export (torch-free)
def collect_runtime_scaling(output) -> tuple[dict, list[dict]]:
    """Every planned instance's completed result, re-verified against its timing rows and execution records."""
    output = Path(output).resolve()
    plan = load_plan(output, "runtime_scaling")
    plan_sha = sha256(output / "plan.json")
    collected = []
    for entry in plan["jobs"]:
        directory = instance_directory(output, entry)
        path = directory / "scaling.json"
        if not path.is_file():
            raise RunnerError(f"{entry['dataset']}/{entry['instance']}: job {entry['job']} has not completed")
        result = read_json(path)
        if (result.get("schema") != SCALING_SCHEMA or result["dataset"] != entry["dataset"]
                or result["instance"] != entry["instance"] or result["member"] != entry["member"]):
            raise RunnerError(f"{path}: the result belongs to another instance")
        if result["identities"]["plan_sha256"] != plan_sha:
            raise RunnerError(f"{path}: executed under another plan")
        for field in ("timing_rows", "executions", "session_log"):
            if sha256(directory / result[field]["file"]) != result[field]["sha256"]:
                raise RunnerError(f"{path}: {result[field]['file']} differs from its recorded SHA-256")
        rows = read_timing_rows(directory / "timing.csv")
        executions = read_json(directory / "executions.json")
        records = {kind: [r[kind] for r in executions["repeats"]] for kind in ("nominal", "deployed")}
        summary = summarize_instance(rows, records, executions["counting_execution"], device=result["device"])
        for key, value in summary.items():
            recorded = result.get(key)
            same = (recorded == value if not isinstance(value, float)
                    else recorded is not None and math.isclose(recorded, value, rel_tol=1e-12, abs_tol=0.0))
            if not same:
                raise RunnerError(f"{path}: {key} differs from the value recomputed from its raw files")
        collected.append({"entry": entry, "directory": directory, "path": path, "result": result})
    return plan, collected


def bind_allocations(collected: list[dict], accounting) -> dict:
    """Bind each timing job to one completed GPU allocation of the sacct exports (cm_campaigns.accounting)."""
    from cm_campaigns.accounting import read_sacct
    from cm_campaigns.common import EvidenceError
    allocations, sources = {}, []
    for path in accounting:
        path = Path(path).resolve()
        try:
            ledger = read_sacct(path)
        except (EvidenceError, OSError) as exc:
            raise RunnerError(f"{path.name}: {exc}") from None
        for key, row in ledger["allocations"].items():
            if key in allocations and allocations[key]["gpu_hours"] != row["gpu_hours"]:
                raise RunnerError(f"conflicting scheduler records for allocation {key}")
            allocations[key] = row
        sources.append({"file": path.name, "sha256": ledger["source_sha256"]})
    jobs = []
    for item in collected:
        result = item["result"]
        key = str(result["job_key"])
        if key in allocations:
            matched = [key]
        elif key.startswith("slurm:"):
            ident = key.split(":", 1)[1]
            matched = sorted(k for k, r in allocations.items() if ident in (str(r["job_id"]), str(r["job_id_raw"])))
        else:
            matched = []
        record = {"dataset": result["dataset"], "instance": result["instance"], "job_key": key,
                  "device": result["device"], "bound": False}
        if len(matched) != 1:
            record["reason"] = ("no allocation of the sacct exports carries this job" if not matched
                                else f"{len(matched)} allocations carry this job")
        else:
            row = allocations[matched[0]]
            completed = row["state"].split()[0] == "COMPLETED" if row["state"] else False
            gpu = int(row["allocated_gpus"]) >= 1
            record.update(allocation=matched[0], job_id=row["job_id"], state=row["state"],
                          allocated_gpus=row["allocated_gpus"], elapsed_seconds=row["seconds"],
                          bound=completed and gpu,
                          reason=None if completed and gpu else "the allocation is not a completed GPU allocation")
        jobs.append(record)
    return {"schema": BINDING_SCHEMA, "sources": sources, "jobs": jobs,
            "rule": ("a timing job is bound when its key names exactly one allocation of the exports (the sacct key, or "
                     "the Slurm job ID as JobID/JobIDRaw) whose state is COMPLETED and which allocated at least one GPU")}


def export_runtime_scaling(output, *, accounting=(), root=None, replace=False, allow_partial=False, hardware=None,
                           period=None, command=None) -> Path:
    output = Path(output).resolve()
    plan, collected = collect_runtime_scaling(output)
    files, rows, inputs, reasons, generated = {}, [], [], [], {}
    for item in collected:
        entry, directory, result = item["entry"], item["directory"], item["result"]
        base = f"raw/{entry['slug']}/{slug(entry['member'])}/{slug(entry['instance'])}"
        for name in ("scaling.json", "timing.csv", "executions.json", "environment.json", "session.log"):
            files[f"{base}/{name}"] = directory / name
        keys = {"dataset": entry["dataset"], "instance": entry["instance"]}
        source = f"{base}/scaling.json"
        rows.append(value_row(keys, "processed_edges_millions", result["processed_edges_millions"], "million edges",
                              source, "json:/processed_edges_millions"))
        rows.append(value_row(keys, "wall_time_seconds", result["wall_time_seconds"], "s", source, "json:/wall_time_seconds"))
        if result["peak_memory_gib"] is not None:
            rows.append(value_row(keys, "peak_memory_gib", result["peak_memory_gib"], "GiB", source, "json:/peak_memory_gib"))
        else:
            reasons.append(f"{entry['dataset']}/{entry['instance']}: peak memory {result['peak_memory_status']}")
        if not str(result["device"]).startswith("cuda"):
            reasons.append(f"{entry['dataset']}/{entry['instance']}: timed on {result['device']}, not on the accelerator")
        inputs.append({"file": str(item["path"].relative_to(output)), "sha256": sha256(item["path"])})
    if accounting:
        binding = bind_allocations(collected, accounting)
        for path in accounting:
            files[f"raw/accounting/{slug(Path(path).name)}"] = Path(path).resolve()
        generated["raw/accounting_binding.json"] = json.dumps(binding, indent=1, sort_keys=True) + "\n"
        reasons += [f"{j['dataset']}/{j['instance']}: {j['reason']}" for j in binding["jobs"] if not j["bound"]]
    else:
        reasons.append("no scheduler accounting export binds the timing jobs to completed GPU allocations")
    synthetic = {d["synthetic"] for d in plan["datasets"]}
    if len(synthetic) != 1:
        raise RunnerError("synthetic and benchmark parents cannot share one record set")
    results = [item["result"] for item in collected]
    provenance = {
        "experiment": "E17 runtime and memory of the deployed rule against processed graph size (Supplementary Figure S2)",
        "produced_by": "scripts/run_timing_records.py scaling plan/run/export (cm_experiments/timing_records.py)",
        "hardware": hardware or provenance_hardware([read_json(i["directory"] / "environment.json") for i in collected]),
        "period": period or provenance_period(results),
        "protocol": (f"cm_campaigns.timing.TimingSession per instance with {plan['warmup']} warm-up and "
                     f"{plan['repetitions']} measured repeats alternating one nominal member execution and one "
                     "execution of the locked deployed rule; instances: full benchmark graphs, breadth-first "
                     "node-induced subgraphs and test graphs of collections (plan.json)"),
        "synthetic": synthetic.pop(), "definitions": SCALING_DEFINITIONS, "sweep": {
            "fractions": plan["fractions"], "graph_quantiles": plan["graph_quantiles"],
            "status": plan["sweep_status"]},
        "plan_sha256": sha256(output / "plan.json")}
    coverage = {"tables": ["Figure S2"], "datasets": sorted({e["dataset"] for e in plan["jobs"]}),
                "coverage_keys": ["dataset", "instance"],
                "coverage_rows": [[e["dataset"], e["instance"]] for e in plan["jobs"]]}
    return deposit("runtime_scaling", rows, files, provenance, coverage=coverage, reasons=reasons, output=output,
                   root=root, replace=replace, allow_partial=allow_partial, command=command, inputs=inputs,
                   generated=generated)


# ============================================================================= timing_anchors (torch-free)
def anchor_directory(output, job: dict) -> Path:
    return Path(output) / "jobs" / job["slug"] / slug(job["member"])


def training_inputs(metadata: dict) -> dict:
    """Inputs one training epoch processes: the full transductive graph, or the D_fit,train graphs."""
    if metadata.get("independent_graphs"):
        sizes = metadata.get("graph_split_sizes") or {}
        train, validation = sizes.get("fit_train"), sizes.get("fit_val")
        unit = "complete graphs of D_fit,train per epoch (whole-graph episodes); validation: D_fit,val graphs"
    else:
        train, validation = 1, 1
        unit = "the full transductive graph: one full-batch training pass and one validation pass per epoch"
    if not isinstance(train, int) or train < 1 or not isinstance(validation, int) or validation < 1:
        raise RunnerError("prepared/metadata.json does not state the training inputs per epoch")
    return {"train_inputs_per_epoch": train, "validation_inputs_per_epoch": validation, "unit": unit}


def throughput_fit(work, seconds) -> dict:
    """Ordinary least squares of t = t0 + kappa W; the sustained throughput is 1 / kappa."""
    x, y = np.asarray(work, dtype=np.float64), np.asarray(seconds, dtype=np.float64)
    if len(x) < 2 or float(np.ptp(x)) == 0.0:
        return {"estimable": False, "reason": "fewer than two distinct nominal work levels", "points": int(len(x))}
    xm, ym = float(x.mean()), float(y.mean())
    slope = float(((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum())
    intercept = ym - slope * xm
    residuals = y - (intercept + slope * x)
    total = float(((y - ym) ** 2).sum())
    fit = {"points": int(len(x)), "slope_seconds_per_multiply_add": slope, "intercept_seconds": intercept,
           "r_squared": None if total == 0 else 1.0 - float((residuals ** 2).sum()) / total,
           "max_abs_residual_seconds": float(np.abs(residuals).max())}
    if not slope > 0:
        return dict(fit, estimable=False, reason="the fitted time per multiply-add is not positive")
    return dict(fit, estimable=True, sustained_multiply_adds_per_second=1.0 / slope)


def plan_timing_anchors(parents, output, *, scaling_output=None, accounting=None, job_map=None, accounting_method=None,
                        forward_equivalents=FORWARD_EQUIVALENTS) -> Path:
    if not parents:
        raise RunnerError("give at least one --parent run")
    factors = sorted(float(f) for f in forward_equivalents)
    if len(factors) != 2 or not 0 < factors[0] < factors[1] or not all(math.isfinite(f) for f in factors):
        raise RunnerError("give two positive, distinct forward-equivalent factors (low and high)")
    if (accounting is None) != (job_map is None):
        raise RunnerError("scheduler accounting needs both the sacct export and the job map")
    scaling, scaling_plan = None, None
    if scaling_output is not None:
        scaling_output = Path(scaling_output).resolve()
        scaling_plan = load_plan(scaling_output, "runtime_scaling")
        scaling = {"output": str(scaling_output), "plan_sha256": sha256(scaling_output / "plan.json")}
    jobs, datasets, parent_runs = [], [], []
    for parent in parents:
        runs = [SealedRun(m) for m in member_runs(parent)]
        dataset = runs[0].dataset
        if len({r.dataset for r in runs}) != 1 or any(d["dataset"] == dataset for d in datasets):
            raise RunnerError(f"{parent}: one parent (run or campaign) per dataset")
        for run in runs:
            run.require("train")
        sealed_output(runs[0], "prepare", "prepared/metadata.json")
        metadata = read_json(runs[0].root / "prepared" / "metadata.json")
        layout = plant_layout(runs[0].config, metadata, dataset)
        work = nominal_work_statistics(runs[0], metadata, layout) if layout["throughput_anchor_applicable"] else None
        instances = []
        if layout["throughput_anchor_applicable"] and scaling_plan is not None:
            members = {r.root.name: r.identity()["config_sha256"] for r in runs}
            instances = [{"job": j["job"], "member": j["member"], "instance": j["instance"]} for j in scaling_plan["jobs"]
                         if j["dataset"] == dataset and members.get(j["member"]) == j["identity"]["config_sha256"]]
        jobs.append({"job": len(jobs), "dataset": dataset, "slug": slug(dataset), "member": Path(parent).resolve().name,
                     "runs": [{"run": str(r.root), "name": r.root.name, "identity": r.identity(),
                               "seeds": [int(s) for s in r.config["training"]["seeds"]],
                               "seed_label": "{seed}" if len(runs) == 1 else f"{r.root.name}/{{seed}}"} for r in runs],
                     "layout": layout, "nominal_work": work, "throughput_instances": instances})
        parent_runs += [str(r.root) for r in runs]
        datasets.append({"dataset": dataset, "slug": slug(dataset), "synthetic": runs[0].synthetic, "runs": len(runs)})
    record = None
    if accounting is not None:
        record = {"sacct": {"path": str(Path(accounting).resolve()), "sha256": sha256(accounting)},
                  "job_map": {"path": str(Path(job_map).resolve()), "sha256": sha256(job_map)},
                  "method": accounting_method, "phase": "backbone_training"}
    return create_plan(output, {
        "experiment": "E17", "record_set": "timing_anchors", "parent_runs": parent_runs, "datasets": datasets,
        "jobs": jobs, "forward_equivalents": factors, "scaling": scaling, "accounting": record,
        "experiment_code_sha256": experiment_source_identity(), "definitions": ANCHOR_DEFINITIONS,
        "protocol": ("training-time anchors from the sealed plant seed markers of the train stage (training time, "
                     "executed epochs) and the prepared training inputs per epoch, with the scheduler accounting of "
                     "each seed's backbone training when supplied; throughput anchor from the post-warm-up nominal "
                     "E17 timing rows of the plant's runtime_scaling instances"),
        "anchor_status": ("declared choices: forward-equivalent factors, executed rather than configured epochs, the "
                          "runtime's D_fit,train inputs, the mean over plant seeds and the least-squares throughput fit "
                          "are stated in cm_experiments/timing_records.py; the manuscript does not define them")})


def _accounting_ledger(output, plan: dict):
    """The job map's attribution of the sacct export (cm_campaigns.accounting.allocate), copied into the output."""
    from cm_campaigns.accounting import allocate
    from cm_campaigns.common import EvidenceError
    record = plan.get("accounting")
    if not record:
        return None, {}
    copies = {}
    for kind in ("sacct", "job_map"):
        source = Path(record[kind]["path"])
        if not source.is_file() or sha256(source) != record[kind]["sha256"]:
            raise RunnerError(f"{source.name}: the scheduler accounting input changed after planning")
        destination = Path(output) / "accounting" / slug(source.name)
        if not destination.is_file():
            _copy(source, destination)
        if sha256(destination) != record[kind]["sha256"]:
            raise RunnerError(f"{destination}: differs from the planned accounting input")
        copies[kind] = destination
    try:
        ledger = allocate(copies["sacct"], copies["job_map"])
    except (EvidenceError, OSError) as exc:
        raise RunnerError(f"scheduler accounting: {exc}") from None
    return ledger, copies


def accounting_seconds(ledger: dict, dataset: str, label: str, method=None) -> tuple[float, list[dict]]:
    """Allocation seconds the job map attributes to one seed's backbone training (fraction-weighted)."""
    rows = [r for r in ledger["detail"] if r["dataset"] == dataset and r["phase"] == "backbone_training"
            and str(r["seed"]) == label and (method is None or r["method"] == method)]
    if not rows:
        raise RunnerError(f"{dataset} seed {label}: the job map attributes no backbone_training allocation")
    if len({r["method"] for r in rows}) > 1:
        raise RunnerError(f"{dataset} seed {label}: several methods carry backbone training; pass --accounting-method")
    seconds = math.fsum(float(r["seconds"]) * float(r["allocation_fraction"]) for r in rows)
    detail = [{"job_key": r["job_key"], "method": r["method"], "allocation_fraction": float(r["allocation_fraction"]),
               "allocation_seconds": float(r["seconds"]), "state": r["state"], "allocated_gpus": r["allocated_gpus"]}
              for r in rows]
    return seconds, detail


def throughput_record(output, plan: dict, entry: dict, directory: Path) -> tuple[dict, list[str], list[str]]:
    """Throughput anchor of one dataset from the nominal timing rows of its completed scaling instances."""
    layout = entry["layout"]
    if not layout["throughput_anchor_applicable"]:
        return {"applicable": False, "throughput_anchor_seconds": None,
                "status": "not applicable: no native dense layer at every site of a full-batch plant (inserted-site or "
                          "independent-graph plant)"}, [], []
    if not plan.get("scaling") or not entry["throughput_instances"]:
        return {"applicable": True, "throughput_anchor_seconds": None, "nominal_work": entry["nominal_work"],
                "status": "not measured: no runtime_scaling instance of this plant was declared"}, [], []
    scaling_output = Path(plan["scaling"]["output"])
    if sha256(scaling_output / "plan.json") != plan["scaling"]["plan_sha256"]:
        raise RunnerError("the runtime_scaling plan changed after the anchors were planned")
    scaling_plan = load_plan(scaling_output, "runtime_scaling")
    identities = {run["name"]: run["identity"] for run in entry["runs"]}
    points, files, times = [], [], []
    for item in entry["throughput_instances"]:
        job = scaling_plan["jobs"][item["job"]]
        if (job["member"], job["instance"]) != (item["member"], item["instance"]):
            raise RunnerError("the runtime_scaling plan no longer lists the declared instance")
        source = instance_directory(scaling_output, job)
        if not (source / "scaling.json").is_file():
            raise RunnerError(f"{entry['dataset']}/{item['instance']}: the runtime_scaling job has not completed")
        result = read_json(source / "scaling.json")
        timed, own = result["identities"]["parent"], identities.get(item["member"], {})
        same = all(timed.get(k) == own.get(k) for k in ("config_sha256", "units_sha256", "runtime_source_sha256"))
        same &= all(timed["stage_manifests"].get(m) == own.get("stage_manifests", {}).get(m)
                    for m in ("00_prepare.json", "01_train.json"))
        if not same:
            raise RunnerError(f"{entry['dataset']}/{item['instance']}: timed on another parent run or plant")
        if sha256(source / "timing.csv") != result["timing_rows"]["sha256"]:
            raise RunnerError(f"{source / 'timing.csv'} differs from its recorded SHA-256")
        rows = read_timing_rows(source / "timing.csv")
        measured = [r for r in rows if int(r["warmup"]) == 0]
        works = {float(r["nominal_work"]) for r in measured}
        if len(works) != 1:
            raise RunnerError(f"{item['instance']}: the nominal work differs between repeats")
        means = timing_means(rows)
        base = f"throughput/{slug(item['member'])}/{slug(item['instance'])}"
        for name in ("scaling.json", "timing.csv", "environment.json"):
            _copy(source / name, directory / base / name)
            files.append(f"{base}/{name}")
        points.append({"instance": item["instance"], "member": item["member"], "nominal_work_multiply_adds": works.pop(),
                       "nominal_seconds_mean": means["nominal_wall_time_seconds"],
                       "measured_repeats": means["measured_repeats"], "device": result["device"],
                       "timing_rows": f"{base}/timing.csv", "timing_rows_sha256": result["timing_rows"]["sha256"]})
        times += [result.get("started_utc"), result.get("finished_utc")]
    fit = throughput_fit([p["nominal_work_multiply_adds"] for p in points], [p["nominal_seconds_mean"] for p in points])
    work = entry["nominal_work"]
    anchor = work["W_nom"] * fit["slope_seconds_per_multiply_add"] if fit["estimable"] else None
    devices = sorted({p["device"] for p in points})
    status = "measured" if fit["estimable"] else f"not estimable: {fit['reason']}"
    return {"applicable": True, "throughput_anchor_seconds": anchor, "status": status, "fit": fit, "points": points,
            "nominal_work": work, "devices": devices,
            "deployment_device": all(str(d).startswith("cuda") for d in devices)}, files, times


def run_timing_anchors(output, *, job=None) -> list[Path]:
    output = Path(output).resolve()
    plan = load_plan(output, "timing_anchors")
    ledger, _ = _accounting_ledger(output, plan)
    done = []
    for entry in selected_jobs(plan, job):
        directory = anchor_directory(output, entry)
        if not (directory / "anchors.json").is_file():
            _run_anchor_job(output, plan, entry, directory, ledger)
        done.append(directory / "anchors.json")
    return done


def _run_anchor_job(output, plan, entry, directory, ledger):
    started = utc_now()
    directory.mkdir(parents=True, exist_ok=True)
    low_factor, high_factor = plan["forward_equivalents"]
    method = (plan.get("accounting") or {}).get("method")
    seeds, files, times, inputs = [], [], [], {}
    for item in entry["runs"]:
        run = SealedRun(item["run"])
        check_identity(run, item["identity"])
        sealed_output(run, "prepare", "prepared/metadata.json")
        inputs = training_inputs(read_json(run.root / "prepared" / "metadata.json"))
        manifest = read_json(run.root / "provenance" / "stages" / f"{STAGES.index('train'):02d}_train.json")
        times.append(manifest.get("created_utc"))
        sealed_output(run, "prepare", "environment.json")       # the training run's device and software
        files.append(f"environment/{slug(item['name'])}.json")
        _copy(run.root / "environment.json", directory / files[-1])
        for seed in item["seeds"]:
            relative = f"train/plant_seed{seed}_done.json"
            digest = sealed_output(run, "train", relative)
            marker = read_json(run.root / relative)
            if int(marker.get("seed", -1)) != int(seed) or marker.get("baseline") is not False:
                raise RunnerError(f"{run.root.name}: {relative} is not the plant marker of seed {seed}")
            epochs = int(_number(marker.get("epochs_completed"), f"{relative} epochs_completed"))
            if epochs < 1:
                raise RunnerError(f"{run.root.name}: {relative} records no executed epoch")
            copy = f"train/{slug(item['name'])}/plant_seed{seed}_done.json"
            _copy(run.root / relative, directory / copy)
            files.append(copy)
            label = item["seed_label"].format(seed=seed)
            if ledger is None:
                seconds, source, detail = _number(marker.get("seconds_this_attempt"), f"{relative} seconds"), "seed_marker", None
            else:
                seconds, detail = accounting_seconds(ledger, entry["dataset"], label, method)
                source = "scheduler_accounting"
            if seconds <= 0:
                raise RunnerError(f"{run.root.name} seed {seed}: the training time is not positive")
            per_input = seconds / (epochs * inputs["train_inputs_per_epoch"])
            seeds.append({"run": item["name"], "seed": int(seed), "seed_label": label, "marker": copy,
                          "marker_sha256": digest, "training_seconds": seconds, "time_source": source,
                          "seed_marker_seconds_this_attempt": marker.get("seconds_this_attempt"),
                          "epochs_completed": epochs, "configured_epochs": int(run.config["training"]["epochs"]),
                          "train_inputs_per_epoch": inputs["train_inputs_per_epoch"],
                          "validation_inputs_per_epoch": inputs["validation_inputs_per_epoch"],
                          "execution": marker.get("execution"), "accounting": detail,
                          "seconds_per_epoch_per_input": per_input})
    if not seeds:
        raise RunnerError(f"{entry['dataset']}: no plant seed to time")
    quotient = math.fsum(s["seconds_per_epoch_per_input"] for s in seeds) / len(seeds)
    throughput, copied, measured_times = throughput_record(output, plan, entry, directory)
    files += copied
    write_json(directory / "anchors.json", json_safe({
        "schema": ANCHORS_SCHEMA, "experiment": "E17", "record_set": "timing_anchors", "dataset": entry["dataset"],
        "layout": entry["layout"],
        "training": {"seeds": seeds, "time_source": seeds[0]["time_source"], "inputs_unit": inputs.get("unit"),
                     "seconds_per_epoch_per_input_mean": quotient,
                     "seconds_per_epoch_per_input_range": [min(s["seconds_per_epoch_per_input"] for s in seeds),
                                                           max(s["seconds_per_epoch_per_input"] for s in seeds)],
                     "forward_equivalents": [low_factor, high_factor],
                     "training_anchor_low_seconds": quotient / high_factor,
                     "training_anchor_high_seconds": quotient / low_factor,
                     "time_scope": ("the runtime's elapsed time of the attempt that completed the seed (epochs, "
                                    "validation passes and epoch checkpoints included; a resumed seed's earlier "
                                    "attempts are not in it)" if ledger is None else
                                    "allocation seconds x the job map's fraction, over every allocation of the seed's "
                                    "backbone training (failed and resumed attempts included)")},
        "throughput": throughput, "files": files, "definitions": ANCHOR_DEFINITIONS,
        "period_sources": {"train_stage_sealed_utc": [t for t in times if t], "throughput_runs_utc": [t for t in measured_times if t]},
        "identities": {"plan_sha256": sha256(Path(output) / "plan.json"), "runtime_source_sha256": runtime_source_sha256(),
                       "parents": [item["identity"] for item in entry["runs"]],
                       "accounting": None if not plan.get("accounting") else {
                           kind: {"file": Path(plan["accounting"][kind]["path"]).name,
                                  "sha256": plan["accounting"][kind]["sha256"]} for kind in ("sacct", "job_map")},
                       "experiment_code_sha256": experiment_source_identity()},
        "started_utc": started, "finished_utc": utc_now()}))


def export_timing_anchors(output, *, root=None, replace=False, allow_partial=False, hardware=None, period=None,
                          command=None) -> Path:
    from cm_campaigns.contract import ADMITTED
    output = Path(output).resolve()
    plan = load_plan(output, "timing_anchors")
    plan_sha = sha256(output / "plan.json")
    files, rows, inputs, reasons, environments, times = {}, [], [], [], [], []
    for entry in plan["jobs"]:
        directory = anchor_directory(output, entry)
        path = directory / "anchors.json"
        if not path.is_file():
            raise RunnerError(f"{entry['dataset']}: job {entry['job']} has not completed")
        result = read_json(path)
        if result.get("schema") != ANCHORS_SCHEMA or result["dataset"] != entry["dataset"]:
            raise RunnerError(f"{path}: the result belongs to another dataset")
        if result["identities"]["plan_sha256"] != plan_sha:
            raise RunnerError(f"{path}: computed under another plan")
        base = f"raw/{entry['slug']}"
        files[f"{base}/anchors.json"] = path
        for relative in result["files"]:
            files[f"{base}/{relative}"] = directory / relative
            if relative.startswith("environment/") or relative.endswith("/environment.json"):
                environments.append(read_json(directory / relative))
        for seed in result["training"]["seeds"]:
            if sha256(directory / seed["marker"]) != seed["marker_sha256"]:
                raise RunnerError(f"{path}: {seed['marker']} differs from the sealed seed marker")
        keys = {"dataset": entry["dataset"]}
        source = f"{base}/anchors.json"
        for quantity in ("training_anchor_low_seconds", "training_anchor_high_seconds"):
            rows.append(value_row(keys, quantity, result["training"][quantity], "s", source, f"json:/training/{quantity}"))
        throughput = result["throughput"]
        if throughput["throughput_anchor_seconds"] is not None:
            rows.append(value_row(keys, "throughput_anchor_seconds", throughput["throughput_anchor_seconds"], "s", source,
                                  "json:/throughput/throughput_anchor_seconds"))
            if not throughput["deployment_device"]:
                reasons.append(f"{entry['dataset']}: throughput timed on {throughput['devices']}, not on the accelerator")
        elif throughput["applicable"]:
            reasons.append(f"{entry['dataset']}: throughput anchor {throughput['status']}")
        times += result["period_sources"]["train_stage_sealed_utc"] + result["period_sources"]["throughput_runs_utc"]
        inputs.append({"file": str(path.relative_to(output)), "sha256": sha256(path)})
    if plan.get("accounting"):
        for kind in ("sacct", "job_map"):
            name = slug(Path(plan["accounting"][kind]["path"]).name)
            files[f"raw/accounting/{name}"] = output / "accounting" / name
    planned = {e["dataset"] for e in plan["jobs"]}
    missing = [d for d in ADMITTED if d not in planned]
    if missing:
        reasons.append(f"datasets without anchors: {missing}")
    synthetic = {d["synthetic"] for d in plan["datasets"]}
    if len(synthetic) != 1:
        raise RunnerError("synthetic and benchmark parents cannot share one record set")
    if not hardware and not environments:
        raise RunnerError("the anchors carry no environment record; pass --hardware")
    period = period or _period(times)
    if not period:
        raise RunnerError("the anchors carry no timestamp; pass --period")
    provenance = {
        "experiment": "E17 timing anchors that bracket the per-input nominal execution time (Supplementary Table S23b)",
        "produced_by": "scripts/run_timing_records.py anchors plan/run/export (cm_experiments/timing_records.py)",
        "hardware": hardware or provenance_hardware(environments),
        "period": period,
        "protocol": plan["protocol"], "definitions": ANCHOR_DEFINITIONS, "anchor_status": plan["anchor_status"],
        "forward_equivalents": plan["forward_equivalents"], "synthetic": synthetic.pop(),
        "kept_apart_from": "the measured t_nom of the nominal_time set", "plan_sha256": plan_sha}
    coverage = {"tables": ["S23b"], "datasets": sorted(set(ADMITTED) | planned)}
    return deposit("timing_anchors", rows, files, provenance, coverage=coverage, reasons=reasons, output=output,
                   root=root, replace=replace, allow_partial=allow_partial, command=command, inputs=inputs)


# ============================================================================= ensemble_grid_scores (torch-free)
def test_observations(run: SealedRun) -> dict:
    """Sealed test labels, units and row order of the evaluate stage (final/test_predictions.npz)."""
    summary_sha = sealed_output(run, "evaluate", "final/results_summary.json")
    record_sha = sealed_output(run, "evaluate", "final/test_predictions.npz")
    if read_json(run.root / "final" / "results_summary.json").get("test_prediction_record_sha256") != record_sha:
        raise RunnerError(f"{run.root.name}: the test prediction record differs from the one the results summary names")
    with np.load(run.root / "final" / "test_predictions.npz", allow_pickle=False) as record:
        labels, units, ids = record["labels"], record["cluster_ids"], record["sample_ids"]
    sealed_output(run, "prepare", "prepared/splits.npz")
    with np.load(run.root / "prepared" / "splits.npz", allow_pickle=False) as splits:
        order = np.asarray(splits["test"], dtype=np.int64)
    if not np.array_equal(order, np.asarray(ids, dtype=np.int64)) or len(labels) != len(ids) or len(units) != len(ids):
        raise RunnerError(f"{run.root.name}: the sealed test labels are not in the test role's row order")
    return {"labels": labels, "units": units, "sample_ids": ids, "record_sha256": record_sha,
            "summary_sha256": summary_sha, "labels_sha256": npy_sha256(labels), "units_sha256": npy_sha256(units)}


def score_grid_member(plan: dict, entry: dict, grid: dict, directory) -> dict:
    """Official score of every prefix ensemble of one grid job, recomputed from its saved member predictions."""
    from cross_morse.evaluation import score_observations
    directory = Path(directory)
    run = SealedRun(entry["run"])
    check_identity(run, entry["identity"])
    check_role(run, "test")
    observed = test_observations(run)
    members = grid.get("members") or []
    if [int(m["seed"]) for m in members] != [int(s) for s in entry["seed_order"]]:
        raise RunnerError(f"{entry['member']}: the saved member predictions do not follow the planned seed order")
    sizes = sorted(int(e) for e in plan["grid"])
    by_size = {int(r["ensemble"]): r for r in grid["ensembles"]}
    expected = {"candidate": entry["comparator"], "role": plan["role"]}
    aggregate, count, hashes, ensembles = None, 0, [], []
    for item in members:
        path = directory / item["file"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise RunnerError(f"{path}: missing or differs from the SHA-256 recorded in grid.json")
        member = read_member(path, expected)
        if member.ndim != 2 or member.shape[0] != len(observed["labels"]):
            raise RunnerError(f"{path}: the predictions do not cover the sealed test rows")
        aggregate = np.array(member, dtype=np.float64) if aggregate is None else aggregate + member
        count += 1
        hashes.append(item["sha256"])
        if count not in sizes:
            continue
        ensemble = (aggregate / count).astype(np.float32)    # the runtime's ensemble (ensemble_probabilities)
        score = score_observations(ensemble, observed["labels"], observed["units"], run.config)
        scale, unit = OFFICIAL_SCORE[score["metric_name"]]
        official = scale * score["metric"]
        row = by_size.get(count)
        if row is None or row["seeds"] != [int(s) for s in entry["seed_order"][:count]]:
            raise RunnerError(f"{entry['member']}: grid.json lacks the E={count} ensemble of the planned seed prefix")
        if not (math.isclose(official, row["official_score"], rel_tol=1e-9, abs_tol=1e-12)
                and math.isclose(score["node_ce"], row["loss_nats"], rel_tol=1e-9, abs_tol=1e-12)):
            raise RunnerError(f"{entry['member']} E={count}: the score recomputed from the saved predictions differs "
                              f"from grid.json ({official!r} vs {row['official_score']!r})")
        ensembles.append({"ensemble": count, "official_score": official, "unit": unit, "metric": score["metric"],
                          "metric_name": score["metric_name"], "loss_nats": score["node_ce"],
                          "loss_name": score["loss_name"], "units": int(len(score["core_ids"])),
                          "rows": int(ensemble.shape[0]), "seeds": [int(s) for s in entry["seed_order"][:count]],
                          "prediction_sha256": npy_sha256(ensemble), "prediction_encoding": PREDICTION_ENCODING,
                          "member_prediction_sha256": list(hashes), "grid_official_score": row["official_score"],
                          "grid_loss_nats": row["loss_nats"]})
        if count == sizes[-1]:
            break
    if [e["ensemble"] for e in ensembles] != sizes:
        raise RunnerError(f"{entry['member']}: the saved member predictions do not reach every ensemble size {sizes}")
    return {"member": entry["member"], "run": run.root.name, "ensembles": ensembles,
            "test_record": {k: observed[k] for k in ("record_sha256", "summary_sha256", "labels_sha256", "units_sha256")},
            "test_rows": int(len(observed["labels"])), "parent_files_read": run.inputs()}


def export_ensemble_grid_scores(grid_output, *, root=None, replace=False, allow_partial=False, hardware=None,
                                period=None, command=None) -> Path:
    from cm_campaigns.contract import ADMITTED
    grid_output = Path(grid_output).resolve()
    plan = load_plan(grid_output, "ensemble_grid")
    if plan["role"] != "test":
        raise RunnerError(f"Table S26 reports test-role scores; this grid was executed on the {plan['role']} role")
    datasets = collect_ensemble_grid(grid_output)
    staging = Path(tempfile.mkdtemp(prefix="cm-e14-scores-"))
    try:
        files, rows, inputs, reasons, environments, results = {}, [], [], [], [], []
        for dataset in datasets:
            base = f"raw/{dataset['slug']}"
            scored = []
            for member in dataset["members"]:
                entry = next(j for j in plan["jobs"] if j["dataset"] == dataset["dataset"] and j["member"] == member["member"])
                scored.append(score_grid_member(plan, entry, member["result"], Path(member["grid"]).parent))
                name = slug(member["member"])
                files[f"{base}/{name}/grid.json"] = member["grid"]
                files[f"{base}/{name}/environment.json"] = member["environment"]
                environments.append(read_json(member["environment"]))
                results.append(member["result"])
                inputs.append({"file": str(Path(member["grid"]).relative_to(grid_output)), "sha256": member["grid_sha256"]})
            ensembles = []
            for i, size in enumerate(plan["grid"]):
                values = [m["ensembles"][i]["official_score"] for m in scored]
                mean = math.fsum(values) / len(values)
                if not math.isclose(mean, dataset["ensembles"][i]["official_score"], rel_tol=1e-9, abs_tol=1e-12):
                    raise RunnerError(f"{dataset['dataset']} E={size}: the recomputed score differs from the grid's")
                ensembles.append({"ensemble": int(size), "official_score": mean, "unit": scored[0]["ensembles"][i]["unit"],
                                  "metric_name": scored[0]["ensembles"][i]["metric_name"],
                                  "members": [dict(m["ensembles"][i], member=m["member"]) for m in scored]})
            summary = {"schema": SCORES_SCHEMA, "dataset": dataset["dataset"], "role": plan["role"],
                       "aggregation": dataset["aggregation"], "ensembles": ensembles,
                       "identities": {"runtime_source_sha256": runtime_source_sha256(),
                                      "grid_plan_sha256": sha256(grid_output / "plan.json"),
                                      "experiment_code_sha256": experiment_source_identity(),
                                      "grid_results": [{"member": m["member"], "sha256": m["grid_sha256"]}
                                                       for m in dataset["members"]]},
                       "members": [{k: m[k] for k in ("member", "run", "test_record", "test_rows", "parent_files_read")}
                                   for m in scored],
                       "definition": ("official test metric of the comparator's E-member prefix ensemble (the runtime's "
                                      "float64 member mean in seed order, cast to float32), recomputed from the grid's "
                                      "saved member predictions with cross_morse.evaluation.score_observations and the "
                                      "sealed test labels; with several split runs, the mean of their scores")}
            path = staging / dataset["slug"] / "grid_scores.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(json_safe(summary), indent=1, sort_keys=True, allow_nan=False) + "\n",
                            encoding="utf-8")
            files[f"{base}/grid_scores.json"] = path
            for i, row in enumerate(ensembles):
                rows.append({"dataset": dataset["dataset"], "ensemble": str(row["ensemble"]), "quantity": "official_score",
                             "value": row["official_score"], "unit": row["unit"],
                             "source_file": f"{base}/grid_scores.json", "locator": f"json:/ensembles/{i}/official_score"})
        if [int(e) for e in plan["grid"]] != registered_grid():
            reasons.append(f"grid {plan['grid']} differs from the registered E14 grid {registered_grid()}")
        planned = {d["dataset"] for d in datasets}
        missing = [d for d in ADMITTED if d not in planned]
        if missing:
            reasons.append(f"datasets without scores: {missing}")
        synthetic = {d["synthetic"] for d in datasets}
        if len(synthetic) != 1:
            raise RunnerError("synthetic and benchmark parents cannot share one record set")
        provenance = {
            "experiment": "E14 official scores of the comparator ensemble grid (Supplementary Table S26)",
            "produced_by": ("scripts/run_timing_records.py grid-scores export (cm_experiments/timing_records.py) on a "
                            "completed scripts/run_experiment.py ensemble-grid output"),
            "hardware": hardware or provenance_hardware(environments), "period": period or provenance_period(results),
            "protocol": ("companion of the ensemble_grid set: the same checkpoints, seed order, grid and test units; each "
                         "prefix ensemble's official score recomputed from the saved member predictions and checked "
                         "against the grid job's own record; the SHA-256 of every ensemble's predictions is in raw/"),
            "synthetic": synthetic.pop(), "role": plan["role"], "grid": plan["grid"],
            "grid_plan_sha256": sha256(grid_output / "plan.json")}
        coverage = {"tables": ["S26"], "datasets": sorted(set(ADMITTED) | planned)}
        return deposit("ensemble_grid_scores", rows, files, provenance, coverage=coverage, reasons=reasons,
                       output=grid_output, root=root, replace=replace, allow_partial=allow_partial, command=command,
                       inputs=inputs)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
