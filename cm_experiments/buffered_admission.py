"""E11 (Supplementary Table S27; record set ``buffer_sensitivity``): descriptive admission as the cluster
buffer grows.

The released runtime records the k-hop buffer of every core only as a context diagnostic
(``cross_morse.data._buffers``: ``buffer_role`` ``context_diagnostic_only``) and
``cross_morse.statistics.admission`` resamples the cores themselves, so ``data.buffer_hops`` changes no
admission statistic. This module defines how the buffer enters the resampling units and measures the
descriptive admission at every buffer of the registered grid on a sealed lifecycle run. Nothing is
re-executed: the per-core paired losses sealed in ``admit/paired_units.npz`` are regrouped and the
runtime's own admission function is applied to the regrouped units.

Definition at buffer k (registered buffer k0 = the run's ``data.buffer_hops``)
-----------------------------------------------------------------------------
The registered units are the cores c of ``admit/paired_units.npz`` (cores holding at least one scored
D_admit node), with clipped paired difference x_c = clip(candidate_loss - comparator_loss, -B, B) and
weight w_c = the core's scored-node count, exactly as the admit stage passes them to the runtime.

1. Walk dependence. E_k(c <- c') is the probability that a simple random walk on the prepared graph
   (the symmetric, loop-free adjacency of ``data._buffers``; a walk at an isolated node stays there),
   started at a uniformly drawn D_admit node of core c', enters core c within k steps. A D_admit node
   of c' can reach c within k steps only if it lies in the k-hop buffered cluster of c, so E_k is the
   k-hop coverage of c' by the buffered cluster of c with every covered node weighted by the
   probability that a k-step walk realizes the connection. E_k is non-decreasing in k.
   s_k(c, c') = max(E_k(c <- c'), E_k(c' <- c)).
2. Threshold. tau = max over core pairs of s_k0: the strongest dependence that the registered design
   already treats as separate resampling units.
3. Dependency graph G_k: cores c != c' are linked when s_k(c, c') > tau. G_k0 has no links and G_k only
   gains links as k grows.
4. Resampling units. The roots are the lexicographically first maximum independent set of G_k (by core
   order); its size, the independence number alpha(G_k), is the largest number of cores that are
   pairwise independent at buffer k. Every other core joins the root on which it depends most (largest
   s_k, ties to the lower core); that root is linked to it because a maximum independent set dominates
   G_k. A block is a root with the cores that joined it. resampling_units = alpha(G_k).
5. Statistic. ``cross_morse.statistics.admission`` is called on the blocks: a block's difference is the
   w-weighted mean of its cores' clipped differences (the core's own x_c for a one-core block) and its
   weight is the block's scored-node count; replicates, alpha, clip, epsilon_a, epsilon_metric, the
   bootstrap seed and the metric summary are the sealed admission's. Its whole-unit bootstrap then
   resamples whole blocks with the runtime's ratio estimator (a block bootstrap of the weighted mean).
   Reported: D_bar = the registered clipped mean (the block mean is the same weighted mean regrouped;
   checked to 1e-12 nats); U_desc = the bootstrap's one-sided (1 - alpha) upper limit; half-width =
   U_desc - D_bar; admitted = the runtime's mean and upper gates (D_bar <= -epsilon_a and U_desc < 0).
   One block leaves no interval: U_desc and the half-width are undefined and admitted = 0.

Properties checked on every parent: at k0 every block is one core, so the call is the sealed admission
and must reproduce admit/decision.json exactly (D_bar, U_desc, units, both gates and the decision; a
parent that does not reproduce is refused); D_bar is the same at every buffer; resampling_units is
non-increasing in k (alpha cannot grow when links are added). The contexts of ``data._buffers`` at k0
are recomputed and must equal the sealed buffer audit of prepared/metadata.json, which ties the graph
read here to the graph the runtime prepared.

Why not blocks = connected components of a k-hop coverage graph (cores linked when the k-hop context of
one covers more of the other than any pair at k0): (a) k-hop contexts saturate within a few hops on
small-world graphs, so on a compact 32-core partition of a random geometric graph with 2% long-range
edges every pair passes the threshold at 4 hops (one block), and on a stochastic-block graph some pair
is already fully covered at 2 hops (tau = 1), so the buffer never changes anything; (b) the transitive
closure chains neighbouring cores, so even with the walk dependence the components fall to one block
inside the grid (4 hops on the stochastic-block graph, 16-32 hops on the geometric ones). With the walk
dependence and the independence-number count the units fall gradually on the same graphs
(tests/test_sweeps.py, BufferUnits). Cores that are exchangeable rather than local (a degree-stratified
partition interleaves every core over the whole graph) are all about equally dependent, so every pair
crosses tau at about the same buffer and the units drop from J to one there; that is a property of such
partitions, not of the rule.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import sparse

from .sweeps import (ROOT, SealedRun, SweepError, canonical_sha256, read_json, safe_label, sha256_file,
                     write_json)

RECORD_SET = "buffer_sensitivity"
JOB_SCHEMA = "cm-buffer-sensitivity-job-1"
DEFINITION_ID = "walk-dependence-independent-set-blocks-v1"
DEFINITION = ("resampling units at buffer k: the cores of admit/paired_units.npz grouped into blocks; cores c, c' are "
              "linked when s_k(c,c') = max(E_k(c<-c'), E_k(c'<-c)) exceeds tau = max s_k0 over core pairs, where "
              "E_k(c<-c') is the probability that a simple random walk on the prepared graph started at a D_admit "
              "node of c' enters c within k steps; the roots are the lexicographically first maximum independent "
              "set of the link graph and every other core joins the root it depends on most; "
              "cross_morse.statistics.admission on the blocks (block difference = scored-node-weighted mean of its "
              "cores' clipped differences, block weight = scored-node count) with the sealed admission's "
              "replicates, alpha, clip, epsilon_a and seed")
RUNNER_FILES = ("cm_experiments/buffered_admission.py", "cm_experiments/sweeps.py", "scripts/run_sweep.py")
PARENT_FILES = ("admit/decision.json", "admit/paired_units.npz", "prepared/metadata.json", "prepared/graph.npz",
                "prepared/clusters.npz", "prepared/splits.npz")
MEAN_TOLERANCE = 1e-12          # nats: regrouping changes only the floating-point summation order


# ============================================================================= graph
def symmetric_adjacency(edge_index, n: int) -> sparse.csr_matrix:
    """Boolean, symmetric, loop-free CSR adjacency (the pattern ``data._buffers`` walks)."""
    edges = np.asarray(edge_index)
    if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype.kind not in "iu":
        raise SweepError("edge_index must be an integer [2, E] array")
    if edges.size and (edges.min() < 0 or edges.max() >= n):
        raise SweepError("edge_index has endpoints outside the prepared node range")
    adjacency = sparse.coo_matrix((np.ones(edges.shape[1], dtype=bool), (edges[0], edges[1])),
                                  shape=(n, n)).tocsr()
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    adjacency.setdiag(False)
    adjacency.eliminate_zeros()
    adjacency.sort_indices()
    return adjacency


def context_sizes(adjacency: sparse.csr_matrix, clusters, hops: int) -> list[int]:
    """|core + k-hop halo| of every core: the breadth-first contexts of ``cross_morse.data._buffers``."""
    clusters = np.asarray(clusters)
    n = len(clusters)
    sizes = []
    for cluster in range(int(clusters.max()) + 1):
        core = np.flatnonzero(clusters == cluster)
        visited = np.zeros(n, dtype=bool)
        visited[core] = True
        frontier = core
        for _ in range(int(hops)):
            if not len(frontier):
                break
            reached = np.zeros(n, dtype=bool)
            reached[adjacency[frontier].indices] = True
            frontier = np.flatnonzero(reached & ~visited)
            visited[frontier] = True
        sizes.append(int(visited.sum()))
    return sizes


def walk_entry(adjacency: sparse.csr_matrix, clusters, core_ids, sources, steps) -> dict[int, np.ndarray]:
    """E[t][a, b]: probability that a simple random walk from a uniform D_admit node of core b enters core a
    within t steps, for every t in ``steps``.

    Backward recursion on absorbing targets: h_0 = 1[v in core a]; h_t = 1[v in core a] or (P h_{t-1})(v),
    with P the row-normalized adjacency (an isolated node has an all-zero row: its walk stays put).
    """
    steps = sorted({int(s) for s in steps})
    if not steps or steps[0] < 0:
        raise SweepError("walk lengths must be nonnegative integers")
    clusters = np.asarray(clusters, dtype=np.int64)
    core_ids = np.asarray(core_ids, dtype=np.int64)
    n, units = adjacency.shape[0], len(core_ids)
    if len(sources) != units or any(not len(s) for s in sources):
        raise SweepError("every resampling core needs at least one D_admit node")
    unit = np.full(int(clusters.max()) + 1, -1, dtype=np.int64)
    unit[core_ids] = np.arange(units)
    column = unit[clusters]
    rows = np.flatnonzero(column >= 0)
    cols = column[rows]
    counts = np.diff(adjacency.indptr)
    scale = np.zeros(n, dtype=np.float64)
    np.divide(1.0, counts, out=scale, where=counts > 0)
    walk = sparse.csr_matrix((np.repeat(scale, counts), adjacency.indices, adjacency.indptr), shape=(n, n))
    hit = np.zeros((n, units), dtype=np.float64)
    hit[rows, cols] = 1.0

    def summary(h: np.ndarray) -> np.ndarray:
        out = np.empty((units, units), dtype=np.float64)
        for b, nodes in enumerate(sources):
            out[:, b] = h[nodes].mean(axis=0)
        return out

    entry = {}
    if steps[0] == 0:
        entry[0] = summary(hit)
    for t in range(1, steps[-1] + 1):
        hit = walk @ hit
        hit[rows, cols] = 1.0
        if t in steps:
            entry[t] = summary(hit)
    return entry


def dependence(entry: np.ndarray) -> np.ndarray:
    """s(c, c') = max(E(c <- c'), E(c' <- c)) with a zero diagonal."""
    strength = np.maximum(entry, entry.T).astype(np.float64)
    np.fill_diagonal(strength, 0.0)
    return strength


# ============================================================================= independent sets
def _bitsets(links: np.ndarray) -> list[int]:
    links = np.asarray(links, dtype=bool)
    if links.ndim != 2 or links.shape[0] != links.shape[1] or (links != links.T).any() or links.diagonal().any():
        raise SweepError("the link matrix must be square, symmetric and loop-free")
    return [sum(1 << int(j) for j in np.flatnonzero(row)) for row in links]


def _independence_number(neighbours: list[int], candidates: int) -> int:
    """Largest independent set inside ``candidates`` (a bitset): maximum clique of the complement graph,
    branch and bound with the greedy-colouring bound (each colour class is a clique of the link graph)."""
    full = (1 << len(neighbours)) - 1
    others = [full & ~neighbours[v] & ~(1 << v) for v in range(len(neighbours))]    # complement graph
    best = 0

    def colour(pool: int):
        order, bound, uncoloured, used = [], [], pool, 0
        while uncoloured:
            used += 1
            free = uncoloured
            while free:
                low = free & -free
                v = low.bit_length() - 1
                free &= ~low & ~others[v]
                uncoloured &= ~low
                order.append(v)
                bound.append(used)
        return order, bound

    def expand(size: int, pool: int) -> None:
        nonlocal best
        order, bound = colour(pool)
        for i in range(len(order) - 1, -1, -1):
            if size + bound[i] <= best:
                return
            v = order[i]
            inner = pool & others[v]
            if inner:
                expand(size + 1, inner)
            elif size + 1 > best:
                best = size + 1
            pool &= ~(1 << v)

    if candidates:
        expand(0, candidates)
    return best


def independence_number(links) -> int:
    neighbours = _bitsets(links)
    return _independence_number(neighbours, (1 << len(neighbours)) - 1)


def maximum_independent_set(links) -> list[int]:
    """The lexicographically first maximum independent set (sorted core positions)."""
    neighbours = _bitsets(links)
    count = len(neighbours)
    target = _independence_number(neighbours, (1 << count) - 1)
    chosen, candidates = [], (1 << count) - 1
    for v in range(count):
        if len(chosen) == target:
            break
        if not (candidates >> v) & 1:
            continue
        rest = candidates & ~((1 << (v + 1)) - 1) & ~neighbours[v]
        if len(chosen) + 1 + _independence_number(neighbours, rest) == target:
            chosen.append(v)
            candidates = rest
        else:
            candidates &= ~(1 << v)
    if len(chosen) != target:
        raise SweepError("internal error: no maximum independent set was constructed")
    return chosen


def resampling_blocks(strength: np.ndarray, tau: float) -> dict:
    """Blocks of core positions at one buffer: roots of a maximum independent set and the cores they absorb."""
    links = np.asarray(strength) > tau
    np.fill_diagonal(links, False)
    roots = maximum_independent_set(links)
    members = {r: [r] for r in roots}
    for core in range(len(links)):
        if core in members:
            continue
        linked = [r for r in roots if links[core, r]]
        if not linked:
            raise SweepError("internal error: a maximum independent set must dominate the link graph")
        members[max(linked, key=lambda r: (strength[core, r], -r))].append(core)
    blocks = sorted((sorted(b) for b in members.values()), key=lambda b: b[0])
    return {"links": int(np.triu(links, 1).sum()), "roots": roots, "blocks": blocks}


def block_values(clipped, weights, blocks) -> tuple[np.ndarray, np.ndarray]:
    """Block difference (weighted mean of the cores' clipped differences) and block weight (scored nodes)."""
    clipped, weights = np.asarray(clipped, dtype=np.float64), np.asarray(weights)
    values, totals = [], []
    for block in blocks:
        if len(block) == 1:
            values.append(clipped[block[0]])
            totals.append(weights[block[0]])
        else:
            w = weights[block].astype(np.float64)
            values.append(float(np.dot(w, clipped[block]) / w.sum()))
            totals.append(weights[block].sum())
    return np.asarray(values, dtype=np.float64), np.asarray(totals, dtype=weights.dtype)


# ============================================================================= buffered admission on arrays
def admission_settings(config: dict, decision: dict) -> dict:
    """The sealed admission's arguments; the configuration and the sealed decision must agree."""
    ac = config["admission"]
    loss = decision["loss"]
    if (float(decision["clip"]) != float(ac["clip"]) or float(decision["epsilon_a"]) != float(ac["epsilon_a"])
            or int(loss["bootstrap_replicates"]) != int(ac["bootstrap_reps"])
            or float(loss["one_sided_alpha"]) != float(ac["alpha"])):
        raise SweepError("admit/decision.json was not produced with the run's admission configuration")
    return {"reps": int(loss["bootstrap_replicates"]), "alpha": float(ac["alpha"]), "epsilon_a": float(ac["epsilon_a"]),
            "epsilon_metric": float(ac["epsilon_metric"]), "clip": float(ac["clip"]),
            "seed": int(loss["bootstrap_seed"]), "metric_summary": decision["metric"],
            "execution_valid": decision.get("execution_gate")}


def _admission(values, weights, settings: dict) -> dict:
    from cross_morse.statistics import admission
    return admission(values, weights=weights, reps=settings["reps"], alpha=settings["alpha"],
                     epsilon_a=settings["epsilon_a"], epsilon_metric=settings["epsilon_metric"],
                     clip=settings["clip"], seed=settings["seed"], metric_summary=settings["metric_summary"],
                     independent_graphs=False, execution_valid=settings["execution_valid"])


def reproduction(replay: dict, sealed: dict) -> dict:
    """Field-by-field comparison of the registered-buffer replay with admit/decision.json (exact equality)."""
    fields = {"clipped_mean": (replay["loss"]["mean"], sealed["loss"]["mean"]),
              "upper": (replay["loss"]["upper"], sealed["loss"]["upper"]),
              "units": (replay["loss"]["units"], sealed["loss"]["units"]),
              "mean_gate": (replay["mean_gate"], sealed["mean_gate"]),
              "upper_gate": (replay["upper_gate"], sealed["upper_gate"]),
              "metric_gate": (replay["metric_gate"], sealed["metric_gate"]),
              "passed": (replay["passed"], sealed["passed"])}
    return {"fields": {k: {"replay": a, "sealed": b, "equal": a == b} for k, (a, b) in fields.items()},
            "reproduced": all(a == b for a, b in fields.values())}


def evaluate_buffers(differences, weights, entry: dict, grid, registered: int, settings: dict,
                     sealed: dict | None = None) -> dict:
    """Descriptive admission at every buffer of ``grid`` from per-core paired differences and walk entries.

    ``entry`` maps every buffer of the grid and the registered buffer to its E_k matrix. With ``sealed``
    (admit/decision.json) the registered-buffer replay must reproduce it exactly, otherwise SweepError.
    """
    differences = np.asarray(differences, dtype=np.float64)
    weights = np.asarray(weights)
    grid = sorted({int(k) for k in grid})
    registered = int(registered)
    if registered not in grid:
        raise SweepError(f"the registered buffer {registered} must lie on the grid {grid}")
    missing = [k for k in grid if k not in entry]
    if missing:
        raise SweepError(f"walk entries are missing for buffers {missing}")
    clipped = np.clip(differences, -settings["clip"], settings["clip"])
    strength = {k: dependence(entry[k]) for k in grid}
    tau = float(strength[registered].max()) if len(differences) > 1 else 0.0
    if len(differences) > 1:
        a, b = np.unravel_index(int(np.argmax(strength[registered])), strength[registered].shape)
        tau_pair = sorted((int(a), int(b)))
    else:
        tau_pair = []
    rows, registered_row = [], None
    for k in grid:
        structure = resampling_blocks(strength[k], tau)
        values, totals = block_values(clipped, weights, structure["blocks"])
        result = _admission(values, totals, settings)
        off = strength[k][~np.eye(len(differences), dtype=bool)]
        row = {"buffer_hops": k, **structure, "resampling_units": int(result["loss"]["units"]),
               "block_mean_nats": result["loss"]["mean"], "upper_nats": result["loss"]["upper"],
               "lower_nats": result["loss"]["lower"], "se_nats": result["loss"]["se"],
               "valid_interval": bool(result["loss"]["valid_interval"]),
               "mean_gate": bool(result["mean_gate"]), "upper_gate": bool(result["upper_gate"]),
               "admitted": float(result["mean_gate"] and result["upper_gate"]),
               "dependence": ({"min": float(off.min()), "median": float(np.median(off)), "max": float(off.max())}
                              if len(off) else None),
               "_result": result}
        if row["resampling_units"] != len(structure["blocks"]):
            raise SweepError("internal error: the bootstrap did not resample the blocks")
        if k == registered:
            if any(len(block) != 1 for block in structure["blocks"]):
                raise SweepError("internal error: the registered buffer must leave every core its own unit")
            registered_row = row
        rows.append(row)
    d_bar = registered_row["block_mean_nats"]
    for row in rows:
        if abs(row["block_mean_nats"] - d_bar) > MEAN_TOLERANCE:
            raise SweepError(f"buffer {row['buffer_hops']}: the block mean differs from the registered D_bar")
        if row["mean_gate"] != bool(d_bar <= -settings["epsilon_a"]):
            raise SweepError(f"buffer {row['buffer_hops']}: the regrouped mean crosses -epsilon_a by rounding")
        row["clipped_effect_nats"] = d_bar
        row["half_width_nats"] = None if row["upper_nats"] is None else row["upper_nats"] - d_bar
    units = [row["resampling_units"] for row in rows]
    if any(later > earlier for earlier, later in zip(units, units[1:])):
        raise SweepError(f"internal error: resampling units {units} increase with the buffer")
    check = reproduction(registered_row["_result"], sealed) if sealed is not None else None
    if check is not None and not check["reproduced"]:
        failed = sorted(k for k, v in check["fields"].items() if not v["equal"])
        raise SweepError(f"the registered-buffer row does not reproduce admit/decision.json ({', '.join(failed)})")
    for row in rows:
        row.pop("_result")
    return {"tau": tau, "tau_pair": tau_pair, "rows": rows, "registered": check, "D_bar": d_bar}


# ============================================================================= sealed parents
def check_parent(run: SealedRun, spec: dict) -> None:
    """A sealed single-graph run whose descriptive admission is a cluster bootstrap at the registered buffer."""
    from cross_morse.evaluation import independent_graphs
    run.require("admit")
    if run.dataset not in spec["datasets"]:
        raise SweepError(f"{run.root.name}: {run.dataset} is not a dataset of the {RECORD_SET} contract "
                         f"({', '.join(spec['datasets'])})")
    if independent_graphs(run.config) or run.config["admission"].get("rule") != "cluster_bootstrap":
        raise SweepError(f"{run.root.name}: admission is not a cluster bootstrap over the cores of one graph")
    for relative in PARENT_FILES:
        if not (run.root / relative).is_file():
            raise SweepError(f"{run.root.name}: missing sealed file {relative}")
    registered = int(spec["registered"]["buffer_hops"])
    prepared = run.json("prepared/metadata.json")["preparation_config"]["data"]["buffer_hops"]
    if int(prepared) != registered or int(run.config["data"].get("buffer_hops", prepared)) != registered:
        raise SweepError(f"{run.root.name}: the run was prepared with a {prepared}-hop buffer, not the registered "
                         f"{registered} hops")


def _npz_member(run: SealedRun, relative: str, key: str) -> np.ndarray:
    """One array of a sealed NPZ (the graph archive also holds features); the whole file is hashed."""
    path = run.root / relative
    with np.load(path, allow_pickle=False) as archive:
        if key not in archive.files:
            raise SweepError(f"{run.root.name}: {relative} has no array {key}")
        value = archive[key]
    run.read[relative] = sha256_file(path)
    return value


def parent_inputs(run: SealedRun, registered: int) -> dict:
    """Read and cross-check every sealed input of one parent (records each file's SHA-256 in ``run.read``)."""
    decision = run.json("admit/decision.json")
    units = run.npz("admit/paired_units.npz")
    metadata = run.json("prepared/metadata.json")
    edge_index = _npz_member(run, "prepared/graph.npz", "edge_index")
    clusters = np.asarray(run.npz("prepared/clusters.npz")["cluster_ids"], dtype=np.int64)
    admit = np.asarray(run.npz("prepared/splits.npz")["admit"], dtype=np.int64)
    for name in ("graph.npz", "clusters.npz", "splits.npz"):
        if metadata.get("artifact_sha256", {}).get(name) != run.read[f"prepared/{name}"]:
            raise SweepError(f"{run.root.name}: prepared/{name} differs from its prepared/metadata.json hash")
    n = int(metadata["num_nodes"])
    if clusters.shape != (n,) or (admit.size and (admit.min() < 0 or admit.max() >= n)):
        raise SweepError(f"{run.root.name}: prepared cluster or role arrays do not match the graph")
    core_ids = np.asarray(units["core_ids"], dtype=np.int64)
    counts = np.asarray(units["counts"])
    if (core_ids.ndim != 1 or not len(core_ids) or counts.shape != core_ids.shape or (np.diff(core_ids) <= 0).any()
            or (counts <= 0).any() or counts.dtype.kind not in "iu"):
        raise SweepError(f"{run.root.name}: admit/paired_units.npz has malformed core identities or counts")
    sources = []
    for core, count in zip(core_ids, counts):
        nodes = admit[clusters[admit] == core]
        if len(nodes) < count:
            raise SweepError(f"{run.root.name}: core {core} scores more D_admit nodes than the role holds")
        sources.append(nodes)
    audit = metadata.get("buffer_audit") or {}
    if int(audit.get("buffer_hops", -1)) != registered or not audit.get("core_plus_halo_sizes"):
        raise SweepError(f"{run.root.name}: prepared/metadata.json has no buffer audit at {registered} hops")
    differences = np.asarray(units["candidate_loss"], dtype=np.float64) - np.asarray(units["comparator_loss"],
                                                                                    dtype=np.float64)
    return {"decision": decision, "metadata": metadata, "edge_index": edge_index, "clusters": clusters,
            "admit": admit, "core_ids": core_ids, "counts": counts, "sources": sources, "differences": differences,
            "audit_sizes": [int(s) for s in audit["core_plus_halo_sizes"]], "num_nodes": n}


def measure_parent(run: SealedRun, grid, registered: int) -> tuple[dict, dict]:
    """The buffer measurement of one sealed run: (job record without identities, arrays)."""
    inputs = parent_inputs(run, registered)
    settings = admission_settings(run.config, inputs["decision"])
    adjacency = symmetric_adjacency(inputs.pop("edge_index"), inputs["num_nodes"])
    sizes = context_sizes(adjacency, inputs["clusters"], registered)
    if sizes != inputs["audit_sizes"]:
        raise SweepError(f"{run.root.name}: the {registered}-hop contexts recomputed from prepared/graph.npz differ "
                         "from the sealed buffer audit; the graph is not the one the runtime prepared")
    grid = sorted({int(k) for k in grid})
    entry = walk_entry(adjacency, inputs["clusters"], inputs["core_ids"], inputs["sources"], grid + [registered])
    measured = evaluate_buffers(inputs["differences"], inputs["counts"], entry, grid, registered, settings,
                                sealed=inputs["decision"])
    core_ids = inputs["core_ids"]
    for row in measured["rows"]:                       # positions -> sealed core identities
        row["roots"] = [int(core_ids[i]) for i in row["roots"]]
        row["blocks"] = [[int(core_ids[i]) for i in block] for block in row["blocks"]]
    degree = np.diff(adjacency.indptr)
    record = {
        "settings": {"grid": grid, "registered_buffer_hops": registered,
                     **{k: v for k, v in settings.items() if k not in ("metric_summary",)}},
        "graph": {"nodes": inputs["num_nodes"], "undirected_edges": int(adjacency.nnz // 2),
                  "isolated_nodes": int((degree == 0).sum()), "cores": len(sizes),
                  "resampling_cores": [int(c) for c in core_ids],
                  "d_admit_nodes_per_core": [int(len(s)) for s in inputs["sources"]],
                  "scored_counts": [int(c) for c in inputs["counts"]],
                  "registered_context_sizes": sizes, "registered_context_sizes_match_buffer_audit": True},
        "threshold": {"tau": measured["tau"], "pair": [int(core_ids[i]) for i in measured["tau_pair"]],
                      "rule": "max over core pairs of s at the registered buffer; link when s > tau"},
        "registered": measured["registered"],
        "sealed_admission": {"passed": inputs["decision"]["passed"], "metric_gate": inputs["decision"]["metric_gate"],
                             "D_bar": inputs["decision"]["loss"]["mean"], "U_desc": inputs["decision"]["loss"]["upper"],
                             "units": inputs["decision"]["loss"]["units"]},
        "D_bar": measured["D_bar"], "buffers": measured["rows"]}
    steps = sorted(entry)
    arrays = {"core_ids": core_ids, "steps": np.asarray(steps, dtype=np.int64),
              "entry": np.stack([entry[t] for t in steps]), "tau": np.asarray(measured["tau"])}
    return record, arrays


# ============================================================================= plan, run, collect (run_sweep.py)
def plan_settings(entries: list[dict], spec: dict, buffers=None) -> dict:
    """Jobs (one per sealed parent) and the buffer grid of a buffer_sensitivity plan."""
    from .sweeps import check_admission_replay
    registered = int(spec["registered"]["buffer_hops"])
    grid = sorted({int(k) for k in (spec["grid"]["buffer_hops"] if buffers is None else buffers)})
    if not grid or grid[0] < 1 or registered not in grid:
        raise SweepError(f"the buffer grid {grid} must hold positive hop counts including the registered {registered}")
    jobs, datasets = [], set()
    for entry in entries:
        if len(entry["members"]) != 1:
            raise SweepError(f"{entry['root']}: a campaign of split runs has no buffered cores")
        run = SealedRun(Path(entry["members"][0]))
        check_parent(run, spec)
        if run.dataset in datasets:
            raise SweepError(f"two parent runs of {run.dataset}; the record set holds one run per dataset")
        datasets.add(run.dataset)
        check_admission_replay(run)                     # the sealed admission is reproducible at all
        identifier = safe_label(f"{run.dataset}_{run.root.name}")
        jobs.append({"id": identifier, "dataset": run.dataset, "run": str(run.root),
                     "config_sha256": canonical_sha256(run.config), "output": f"jobs/{identifier}"})
    return {"grid": grid, "registered_buffer_hops": registered, "definition": DEFINITION_ID,
            "definition_text": DEFINITION, "jobs": jobs}


def _environment() -> dict:
    from .ensemble_grid import environment_record
    import scipy
    return {**environment_record("cpu"), "scipy": scipy.__version__}


def run_buffer_jobs(plan_path, only=None) -> list[Path]:
    """Measure every planned parent (resume-safe: a completed job with the same identities is kept)."""
    from .sweeps import PLAN_SCHEMA, _runtime_sha256
    plan_path = Path(plan_path).resolve()
    plan = read_json(plan_path)
    if plan.get("schema") != PLAN_SCHEMA or plan.get("sweep") != RECORD_SET:
        raise SweepError("not a buffer_sensitivity sweep plan")
    base, written = plan_path.parent, []
    settings = plan["settings"]
    for job in settings["jobs"]:
        if only and job["id"] not in only:
            continue
        directory = base / job["output"]
        if (directory / "buffered_admission.json").is_file():
            load_job(plan_path, job)                     # refuses a stale or foreign job output
            written.append(directory / "buffered_admission.json")
            continue
        run = SealedRun(Path(job["run"]))
        if canonical_sha256(run.config) != job["config_sha256"]:
            raise SweepError(f"{job['id']}: the parent configuration changed after planning")
        started, clock = datetime.now(timezone.utc), time.perf_counter()
        record, arrays = measure_parent(run, settings["grid"], settings["registered_buffer_hops"])
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "dependence.npz", **arrays)
        record = {"schema": JOB_SCHEMA, "record_set": RECORD_SET, "job": job["id"], "dataset": job["dataset"],
                  "definition": {"id": DEFINITION_ID, "text": DEFINITION}, "plan_sha256": sha256_file(plan_path),
                  "parent": {"root": str(run.root), "config_sha256": canonical_sha256(run.config),
                             "runtime_source_sha256": read_json(run.root / "run_state.json").get("source_sha256"),
                             "synthetic": run.synthetic, "inputs": run.inputs()},
                  "runtime_source_sha256": _runtime_sha256(),
                  "runner_source_sha256": {name: sha256_file(ROOT / name) for name in RUNNER_FILES},
                  "environment": _environment(), "arrays": {"file": "dependence.npz",
                                                            "sha256": sha256_file(directory / "dependence.npz")},
                  "started_utc": started.isoformat(timespec="seconds"),
                  "seconds": round(time.perf_counter() - clock, 3), **record}
        write_json(directory / "buffered_admission.json", record)
        written.append(directory / "buffered_admission.json")
    return written


def load_job(plan_path, job: dict) -> dict:
    """A completed job of this plan whose parent files are unchanged since it ran."""
    plan_path = Path(plan_path).resolve()
    directory = plan_path.parent / job["output"]
    path = directory / "buffered_admission.json"
    if not path.is_file():
        raise SweepError(f"buffer job {job['id']} has not been run (run_sweep.py run --plan {plan_path})")
    record = read_json(path)
    if (record.get("schema") != JOB_SCHEMA or record.get("job") != job["id"]
            or record.get("plan_sha256") != sha256_file(plan_path)):
        raise SweepError(f"{path} does not belong to this plan; remove it and run the job again")
    if record["arrays"]["sha256"] != sha256_file(directory / record["arrays"]["file"]):
        raise SweepError(f"{directory / record['arrays']['file']} changed after the job ran")
    if record["parent"]["config_sha256"] != job["config_sha256"] or not record["registered"]["reproduced"]:
        raise SweepError(f"{job['id']}: the job does not carry a reproduced registered admission of the planned parent")
    run = Path(job["run"])
    for item in record["parent"]["inputs"]:
        if sha256_file(run / item["file"]) != item["sha256"]:
            raise SweepError(f"{job['id']}: the parent's {item['file']} changed after the job ran")
    return record


def collect_buffer(plan_path, plan: dict, collector) -> None:
    """Record rows of every job: D_bar, half-width, U_desc, decision and resampling units per buffer."""
    for job in plan["settings"]["jobs"]:
        record = load_job(plan_path, job)
        run = SealedRun(Path(job["run"]))
        for item in record["parent"]["inputs"]:
            run.read[item["file"]] = item["sha256"]          # verified against the files by load_job
        directory = Path(plan_path).resolve().parent / job["output"]
        files = {}
        for name in ("buffered_admission.json", record["arrays"]["file"]):
            digest = sha256_file(directory / name)
            collector.deposits.append({"path": f"{job['output']}/{name}", "from": str(directory / name),
                                       "sha256": digest})
            files[name] = {"file": f"raw/{job['output']}/{name}", "sha256": digest}
        for row in record["buffers"]:
            k = row["buffer_hops"]
            values = {"clipped_effect_nats": row["clipped_effect_nats"], "half_width_nats": row["half_width_nats"],
                      "upper_nats": row["upper_nats"], "admitted": row["admitted"],
                      "resampling_units": float(row["resampling_units"])}
            detail = {"definition": record["definition"], "job": files["buffered_admission.json"],
                      "dependence_arrays": files[record["arrays"]["file"]],
                      "registered_buffer_hops": record["settings"]["registered_buffer_hops"],
                      "threshold": record["threshold"], "buffer": row,
                      "registered_reproduction": record["registered"],
                      "sealed_admission": record["sealed_admission"],
                      "interval": ("one-sided (1 - alpha) block-bootstrap upper limit of the clipped mean"
                                   if row["upper_nats"] is not None else
                                   "undefined: a single resampling unit has no bootstrap interval"),
                      "decision_rule": "D_bar <= -epsilon_a and U_desc < 0 (metric gate not part of Table S27)"}
            collector.add(f"{job['dataset']}_buffer_{k}", {"dataset": job["dataset"], "buffer_hops": k},
                          values, detail, [run])
