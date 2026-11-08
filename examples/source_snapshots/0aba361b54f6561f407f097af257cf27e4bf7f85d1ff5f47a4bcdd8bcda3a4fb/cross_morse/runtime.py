"""Graph-local execution shared by training-independent policy and replay paths."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from .common import precision_context, stop_at_boundary
from .core import GraphOperator
from .policies import execute_policy, PolicyExecutionError
from .tasks import probabilities, task_spec
from .training import native_forward


def is_graph_collection(data):
    ptr = getattr(data, "graph_ptr", None)
    return ptr is not None and len(ptr) > 2


def graph_operator(data, device):
    return GraphOperator(data.edge_index, data.num_nodes, device,
                         edge_weight=getattr(data, "edge_weight", None),
                         graph_ptr=getattr(data, "graph_ptr", None))


def prediction_rows(data):
    if data.metadata.get("label_axis", "node") == "graph":
        return len(data.graph_ptr)-1
    return data.num_nodes


@torch.no_grad()
def predict_dataset(model, data, graph, config, candidate=None, controller=None,
                    reference=None, donors=None, device="cpu", output_path=None, role=None):
    """Execute each independent graph with fresh sensor, observer and policy state.

No label accessor is called here. An optional output path streams corpus
predictions into an atomic NumPy memmap rather than retaining every graph's
probabilities in accelerator or host memory. Checkpoint replay uses the same
episode boundaries and ordering. For independent collections, ``role``
restricts execution itself (and all cost/feasibility metadata) to the declared
complete graphs. Other corpus rows remain NaN and must never be scored. The
unrestricted default is retained only for explicit label-free diagnostics.
    """
    if not is_graph_collection(data):
        graph = graph if graph is not None else graph_operator(data, device)
        if candidate is not None:
            return execute_policy(model, data.x.to(device), graph, config, candidate,
                                  controller, reference, donors=donors)
        graph.reset_counters()
        model.reset_counters()
        started = time.perf_counter()
        with precision_context(config, device):
            p = probabilities(native_forward(model, data.x.to(device), graph, 0).float(),
                              getattr(model, "task", task_spec(config))).cpu().numpy()
        return p, {"wall_seconds": time.perf_counter()-started, "actions": [],
                   "plant_operations": dict(model.counters),
                   "multiply_add_work": model.counters["dense_multiply_adds"]}

    rows, outputs = prediction_rows(data), model.num_classes
    if role is None:
        graph_ids = np.arange(len(data.graph_ptr)-1, dtype=np.int64)
    else:
        if role not in data.graph_splits:
            raise ValueError(f"Unknown independent-graph execution role: {role}")
        values = np.asarray(data.graph_splits[role])
        if (values.ndim != 1 or values.dtype.kind not in "iu" or not len(values)
                or np.any(values < 0) or np.any(values >= len(data.graph_ptr)-1)
                or len(np.unique(values)) != len(values)):
            raise ValueError("Execution role must contain unique valid complete graph IDs")
        graph_ids = np.sort(values.astype(np.int64, copy=False))
    temporary = None
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=path.name+".", suffix=".tmp", dir=path.parent)
        os.close(fd)
        result = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(rows, outputs))
    else:
        result = np.empty((rows, outputs), dtype=np.float32)
    result[:] = np.nan
    totals = {key: 0 for key in ("multiply_add_work", "deployed_multiply_add_work",
              "qualification_multiply_add_work", "support_violations", "shuffle_matched_sites",
              "shuffle_missing_sites", "a9_executions", "native_site_count",
              "inserted_site_count", "inserted_a0_bypass_count")}
    totals.update(feasible=True, state_control_valid=True,
                  native_site_action_counts=[0]*10, inserted_site_action_counts=[0]*10)
    action_digest, written = hashlib.sha256(), 0
    covered = np.zeros(rows, dtype=bool)
    completed_graph_ids = []
    started = time.perf_counter()
    try:
        for graph_id in graph_ids.tolist():
            episode = data.graph(graph_id)
            op = graph_operator(episode, device)
            x = episode.x.to(device)
            if candidate is None:
                op.reset_counters()
                model.reset_counters()
                with precision_context(config, device):
                    p = probabilities(native_forward(model, x, op, 0).float(), getattr(model, "task", task_spec(config))).cpu().numpy()
                trace = {"multiply_add_work": model.counters["dense_multiply_adds"], "actions": []}
            else:
                try:
                    p, trace = execute_policy(model, x, op, config, candidate, controller, reference, donors=donors)
                except PolicyExecutionError as error:
                    failed = dict(error.trace, execution_role=role or "all_graphs_diagnostic",
                                  failed_graph_id=graph_id, completed_graph_ids=completed_graph_ids.copy(),
                                  requested_graph_ids_sha256=hashlib.sha256(graph_ids.astype("<i8").tobytes()).hexdigest())
                    for key in ("multiply_add_work", "deployed_multiply_add_work",
                                "qualification_multiply_add_work", "support_violations",
                                "shuffle_matched_sites", "shuffle_missing_sites"):
                        partial = error.trace.get(key, error.trace.get("multiply_add_work", 0)
                                                  if key == "deployed_multiply_add_work" else 0)
                        failed[key] = totals.get(key, 0) + partial
                    failed.update(wall_seconds=time.perf_counter()-started, feasible=False,
                                  execution_failure=True, cost_is_partial=True)
                    raise PolicyExecutionError(error.reason, failed) from error
            ids = np.asarray(episode.global_sample_ids, dtype=np.int64)
            if p.shape != (len(ids), outputs) or not np.isfinite(p).all():
                raise FloatingPointError(f"Invalid predictions for graph {graph_id}")
            if (np.any(ids < 0) or np.any(ids >= rows)
                    or len(np.unique(ids)) != len(ids) or covered[ids].any()):
                raise ValueError("Graph prediction rows overlap or escape the prepared label axis")
            covered[ids] = True
            result[ids] = p
            written += len(ids)
            completed_graph_ids.append(graph_id)
            for key in ("multiply_add_work", "qualification_multiply_add_work", "support_violations",
                        "shuffle_matched_sites", "shuffle_missing_sites", "native_site_count",
                        "inserted_site_count", "inserted_a0_bypass_count"):
                totals[key] += trace.get(key, 0)
            totals["deployed_multiply_add_work"] += trace.get("deployed_multiply_add_work", trace.get("multiply_add_work", 0))
            totals["a9_executions"] += trace.get("actions", []).count(9)
            for key in ("native_site_action_counts", "inserted_site_action_counts"):
                totals[key] = [a+b for a,b in zip(totals[key], trace.get(key, [0]*10))]
            totals["feasible"] &= trace.get("feasible", True)
            control_valid = trace.get("state_control_valid")
            totals["state_control_valid"] &= True if control_valid is None else bool(control_valid)
            action_digest.update(json.dumps({"graph_id": graph_id, "actions": trace.get("actions", []),
                                "safe_depths": trace.get("safe_depths", [])}, sort_keys=True).encode())
            del x, op, episode
            stop_at_boundary()
        expected_ids = (np.arange(rows, dtype=np.int64) if role is None
                        else np.asarray(data.splits[role], dtype=np.int64))
        if (written != len(expected_ids) or not np.array_equal(np.flatnonzero(covered), np.sort(expected_ids))):
            raise ValueError("Graph prediction rows do not cover exactly the declared execution role")
        totals.update(wall_seconds=time.perf_counter()-started, graphs=len(graph_ids),
                      prediction_rows=written, corpus_prediction_rows=rows,
                      execution_role=role or "all_graphs_diagnostic",
                      graph_ids_sha256=hashlib.sha256(graph_ids.astype("<i8").tobytes()).hexdigest(),
                      unexecuted_rows="NaN; excluded from every score and metadata reduction",
                      actions_sha256=action_digest.hexdigest(),
                      execution_unit="one complete graph with fresh controller state",
                      trace_scope="summed work and failures; ordered per-graph actions and anchors bound by digest",
                      labels_accessed_during_prediction=False,
                      transport_history_definition="site-aware kernel mass and expected kernel hops before source retention; inserted A0 is zero")
        if temporary is not None:
            result.flush()
            del result
            os.replace(temporary, path)
            temporary = None
            result = np.load(path, mmap_mode="r", allow_pickle=False)
        return result, totals
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
