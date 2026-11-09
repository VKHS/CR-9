#!/usr/bin/env python3
"""Reduced official-split graphs for the CPU demonstration.

Except Roman-empire, the official graphs exceed what a hosted runner (4 CPUs, 16 GB) completes
in a few hours for the 24-trajectory exploration and the repeated selection and admission
statistics of every lifecycle. The demonstration runs those datasets on a deterministic
subset of the official data:

* node-level graphs: the nodes of ``keep`` parts of a ``parts``-way METIS partition of the
  symmetrized official graph (pymetis, fixed seed) with every induced edge; every kept node
  keeps its official role (train, valid or test). The parts are the first one (in part order)
  that holds the largest class label, then the lowest-numbered others, so that the subset
  declares the benchmark's number of classes. For ogbn-proteins,
  whose official roles are species, the kept parts are, for each role, the part holding
  most of that role's nodes;
* graph-level datasets: the first ``keep`` graphs of each official split, in the
  official order.

The official data are loaded with the runtime's own adapters (cross_morse.datasets,
download authorized), so features, labels and the official splits are exactly those of
a full run. The result is the trusted raw graph NPZ that a demonstration configuration
names in ``data.graph_npz``; the run records its SHA-256 (``source: trusted_raw_npz``).
``<output>.json`` records the selection rule, the sizes and the official source files.

    python -B demo/data.py --dataset ogbn-products --data-root DIR --output DIR/products_demo.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "cross-morse-demo-graph-1"
# Selection rules. Node-level: METIS parts kept of a partition; graph-level: graphs kept per official split.
RULES = {
    "ogbn-arxiv": {"kind": "metis_parts", "parts": 8, "keep": 1, "seed": 0},
    # Species define the official roles of ogbn-proteins, so one part can miss a role: keep, for each
    # official role, the part holding most of its nodes.
    "ogbn-proteins": {"kind": "metis_parts_per_role", "parts": 96, "seed": 0},
    "ogbn-products": {"kind": "metis_parts", "parts": 128, "keep": 1, "seed": 0},
    "PascalVOC-SP": {"kind": "first_graphs", "keep": {"train": 480, "valid": 60, "test": 60}},
    "COCO-SP": {"kind": "first_graphs", "keep": {"train": 480, "valid": 60, "test": 60}},
    "Peptides-func": {"kind": "first_graphs", "keep": {"train": 1200, "valid": 150, "test": 150}},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _metis_parts(edges: np.ndarray, n: int, parts: int, seed: int) -> np.ndarray:
    import pymetis
    from scipy import sparse
    adjacency = sparse.coo_matrix((np.ones(edges.shape[1], dtype=np.float32), (edges[0], edges[1])), shape=(n, n)).tocsr()
    adjacency = (adjacency + adjacency.T).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    options = pymetis.Options(seed=seed)
    if hasattr(pymetis, "CSRAdjacency"):
        dtype = pymetis.zero_copy_dtype()
        csr = pymetis.CSRAdjacency(np.asarray(adjacency.indptr, dtype=dtype), np.asarray(adjacency.indices, dtype=dtype))
        _, membership = pymetis.part_graph(parts, adjacency=csr, options=options)
    else:
        _, membership = pymetis.part_graph(parts, xadj=adjacency.indptr, adjncy=adjacency.indices, options=options)
    return np.asarray(membership, dtype=np.int64)


def node_subset(raw, rule: dict) -> tuple[dict, dict]:
    edges = np.asarray(raw.edge_index, dtype=np.int64)
    n = int(len(raw.x))
    membership = _metis_parts(edges, n, int(rule["parts"]), int(rule["seed"]))
    if rule["kind"] == "metis_parts_per_role":
        chosen = set()
        for role in ("train", "valid", "test"):
            counts = np.bincount(membership[np.asarray(raw.official_splits[role], dtype=np.int64)],
                                 minlength=int(rule["parts"]))
            chosen.add(int(np.argmax(counts)))
        keep_mask = np.isin(membership, sorted(chosen))
    else:
        parts = list(range(int(rule["parts"])))
        y = np.asarray(raw.y)
        if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
            # A fixture's class count is its largest label plus one (cross_morse/data.py), so the kept parts
            # must hold the benchmark's largest class: the first part (in part order) that does, then the lowest.
            top = np.unique(membership[y.reshape(-1) == y.max()])
            first = int(top.min())
            parts = [first] + [part for part in parts if part != first]
        keep_mask = np.isin(membership, parts[: int(rule["keep"])])
    kept = np.flatnonzero(keep_mask)
    remap = np.full(n, -1, dtype=np.int64)
    remap[kept] = np.arange(len(kept), dtype=np.int64)
    inside = keep_mask[edges[0]] & keep_mask[edges[1]]
    arrays = {"x": np.asarray(raw.x)[kept], "edge_index": remap[edges[:, inside]], "y": np.asarray(raw.y)[kept]}
    for role, ids in raw.official_splits.items():
        local = remap[np.asarray(ids, dtype=np.int64)]
        arrays[f"{role}_idx"] = np.sort(local[local >= 0])
    if raw.edge_attr is not None:
        arrays["edge_attr"] = np.asarray(raw.edge_attr)[inside]
    if raw.edge_weight is not None:
        arrays["edge_weight"] = np.asarray(raw.edge_weight, dtype=np.float32)[inside]
    for role in ("train", "valid", "test"):
        if not len(arrays[f"{role}_idx"]):
            raise ValueError(f"the reduced graph has no official {role} node; choose another selection rule")
    sizes = {"nodes": int(len(kept)), "directed_edges": int(inside.sum()), "full_nodes": n,
             "full_directed_edges": int(edges.shape[1]),
             "official_roles": {role: int(len(arrays[f"{role}_idx"])) for role in ("train", "valid", "test")},
             "full_official_roles": {role: int(len(ids)) for role, ids in raw.official_splits.items()}}
    return arrays, sizes


def graph_subset(name: str, data_root: Path, rule: dict) -> tuple[dict, dict, dict]:
    """First graphs of each official split, packed by the runtime's own LRGB packer."""
    from cross_morse import datasets
    directory = datasets._lrgb_raw_directory(data_root, name, True)
    suffix = datasets.LRGB_SOURCES[name][2]
    files = {role: directory / f"{raw}.{suffix}" for role, raw in (("train", "train"), ("valid", "val"), ("test", "test"))}
    selected, full = {}, {}
    for role, path in files.items():
        tuples = datasets.read_lrgb_tuples(path)
        full[role] = len(tuples)
        selected[role] = list(tuples[: int(rule["keep"][role])])
        del tuples
    source = {"source": "official_lrgb_raw_tensor_tuples", "source_url": datasets.LRGB_SOURCES[name][0],
              "source_files_sha256": {path.name: sha256_file(path) for path in files.values()}}
    raw = datasets.pack_lrgb(name, selected, source)
    arrays = {"x": raw.x, "edge_index": raw.edge_index, "y": raw.y, "graph_ptr": raw.graph_ptr, "edge_ptr": raw.edge_ptr,
              **{f"{role}_idx": raw.official_splits[role] for role in ("train", "valid", "test")}}
    if raw.edge_attr is not None:
        arrays["edge_attr"] = raw.edge_attr
    if raw.edge_weight is not None:
        arrays["edge_weight"] = raw.edge_weight
    sizes = {"graphs": {role: len(graphs) for role, graphs in selected.items()}, "full_graphs": full,
             "nodes": int(len(raw.x)), "directed_edges": int(raw.edge_index.shape[1])}
    return arrays, sizes, source


def build(dataset: str, data_root: Path, output: Path) -> dict:
    from cross_morse import datasets
    rule = RULES[dataset]
    started = time.time()
    if rule["kind"] in ("metis_parts", "metis_parts_per_role"):
        raw = datasets.load_benchmark(data_root, dataset, allow_download=True)
        source = {k: v for k, v in raw.metadata.items() if k.startswith("source") or k in ("ogb_dataset_version", "official_split")}
        arrays, sizes = node_subset(raw, rule)
        del raw
    else:
        arrays, sizes, source = graph_subset(dataset, data_root, rule)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial.npz")
    np.savez(temporary, **arrays)
    temporary.replace(output)
    record = {"schema": SCHEMA, "dataset": dataset, "rule": rule, "sizes": sizes, "official_source": source,
              "npz": output.name, "npz_sha256": sha256_file(output), "npz_bytes": output.stat().st_size,
              "builder": "demo/data.py", "seconds": round(time.time() - started, 1),
              "scope": "Deterministic subset of the official data for the CPU demonstration; not the paper's dataset scale."}
    output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(RULES))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    record = build(args.dataset, args.data_root, args.output.resolve())
    print(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
