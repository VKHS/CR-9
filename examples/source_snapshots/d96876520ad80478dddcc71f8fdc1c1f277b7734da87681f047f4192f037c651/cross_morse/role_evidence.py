"""Read-only audit of independent-graph execution-role provenance.

No checkpoint loading or PyTorch import is used. This verifies the sealed
execution records against the declared graph/sample splits. Like the rest of
the archive checks, it checks integrity and internal consistency, not an
independent attestation that a historical computation occurred.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from .evaluation import independent_graphs
from .review import NumericEvidence, EvidenceError, json_read
from .split_contract import LEAF_ROLES, ROLE_NAMES

PROTOCOL = "complete-graph-role-execution-v1"


def _require(condition, reason):
    if not condition:
        raise EvidenceError(reason)


def audit_role_isolation(run_root):
    """Return a report; old unscoped independent archives fail this gate.

    Independent benchmark runs require role-local select, confirmation,
    admission and test execution metadata. Transductive runs are explicitly
    inapplicable because their inputs are one fixed graph, not fresh graphs.
    Existing archives remain readable by numerical replay; passing numerical
    replay alone does not satisfy this independent-graph provenance check.
    """
    root = Path(run_root)
    report = {"protocol": PROTOCOL, "applicable": True, "passed": False,
              "errors": [], "verified_candidate_role_records": 0,
              "scope": "sealed role metadata and split consistency; not historical attestation"}
    try:
        config = json_read(root / "config.json")
        report["dataset"] = config.get("benchmark_dataset", config.get("dataset"))
        if not independent_graphs(config):
            report.update(applicable=False, passed=True,
                          reason="one transductive graph; no independent-graph admission claim")
            return report
        e = NumericEvidence(root)
        summary = e.json("final/results_summary.json")
        _require(summary.get("prediction_storage_scope") == "test_role"
                 and summary.get("holdout_covariates_influence_selection") is False,
                 "Independent archive lacks role-scoped execution provenance; a new lifecycle run is required")
        metadata = e.json("prepared/metadata.json")
        graph_splits = e.arrays("prepared/graph_splits.npz")
        splits = e.arrays("prepared/splits.npz")
        clusters = e.arrays("prepared/clusters.npz")["cluster_ids"]
        _require(set(graph_splits) == set(ROLE_NAMES), "Unexpected graph-role manifest")
        leaves = []
        identities = {}
        for role in ROLE_NAMES:
            ids = graph_splits[role]
            _require(ids.ndim == 1 and ids.dtype.kind in "iu" and len(ids) > 0
                     and np.all(ids >= 0) and len(np.unique(ids)) == len(ids),
                     f"Invalid graph IDs in {role}")
            sample_ids = splits[role]
            _require(sample_ids.ndim == 1 and sample_ids.dtype.kind in "iu"
                     and np.all(sample_ids >= 0) and np.all(sample_ids < len(clusters))
                     and len(np.unique(sample_ids)) == len(sample_ids), f"Invalid sample IDs in {role}")
            _require(np.array_equal(np.sort(ids), np.unique(clusters[sample_ids])),
                     f"Graph and sample unit identities differ in {role}")
            identities[role] = {"graphs": len(ids), "prediction_rows": len(sample_ids),
                "graph_ids_sha256": hashlib.sha256(np.sort(ids).astype("<i8").tobytes()).hexdigest()}
            if role in LEAF_ROLES:
                leaves.append(ids)
        all_graphs = np.concatenate(leaves)
        _require(len(np.unique(all_graphs)) == len(all_graphs)
                 and np.array_equal(np.sort(all_graphs), np.arange(metadata["num_graphs"])),
                 "Leaf graph roles overlap or do not cover the complete graph collection")
        _require(np.array_equal(np.sort(graph_splits["fit"]),
                               np.sort(np.r_[graph_splits["fit_train"], graph_splits["fit_val"]])),
                 "Fit graph role differs from its internal train/validation union")
        report["role_units"] = identities

        def trace_scope(trace, role, location):
            _require(trace.get("execution_role") == role, f"{location}: wrong execution role")
            if trace.get("execution_failure"):
                completed = trace.get("completed_graph_ids", [])
                failed = trace.get("failed_graph_id")
                allowed = set(map(int, graph_splits[role]))
                _require(isinstance(completed, list) and len(set(completed)) == len(completed)
                         and set(completed) <= allowed and failed in allowed
                         and failed not in completed,
                         f"{location}: failed execution reached outside its graph role")
                _require(trace.get("requested_graph_ids_sha256") == identities[role]["graph_ids_sha256"],
                         f"{location}: failed execution requested a different graph role")
            else:
                for key, value in identities[role].items():
                    _require(trace.get(key) == value, f"{location}: {key} differs from declared {role} graphs")
                _require(trace.get("corpus_prediction_rows") == len(clusters),
                         f"{location}: corpus output axis differs from prepared samples")

        def candidate_metadata(candidate, role, *, allow_failure=False, embedded=None):
            cid = candidate["candidate_id"]
            path = f"predictions/{role}/{cid}.json"
            if embedded is None:
                if path not in e.sealed and allow_failure:
                    path = f"prediction_failures/{role}/{cid}.json"
                value = e.json(path)
            else:
                value = embedded
            _require(value.get("candidate", {}).get("hash") == candidate["hash"]
                     and value.get("candidate", {}).get("candidate_id") == cid,
                     f"{path}: policy identity differs from the locked registry")
            _require(value.get("execution_role") == role, f"{path}: unscoped/wrong candidate role")
            failed = bool(value.get("execution_failure", False))
            _require(allow_failure or not failed, f"{path}: unavailable prediction used as observed execution")
            members = value.get("members", [])
            expected_seeds = config["training"]["seeds"][:candidate["ensemble"]]
            _require(bool(members) and len(members) <= len(expected_seeds)
                     and (failed or len(members) == len(expected_seeds)), f"{path}: incomplete ensemble trace")
            for i, member in enumerate(members):
                _require(member.get("seed") == expected_seeds[i], f"{path}: wrong ensemble seed order")
                trace_scope(member, role, f"{path} member {i}")
            for field in ("multiply_add_work", "deployed_multiply_add_work", "qualification_multiply_add_work",
                          "support_violations", "shuffle_matched_sites", "shuffle_missing_sites"):
                total = sum(t.get(field, t.get("multiply_add_work", 0)
                                  if field == "deployed_multiply_add_work" else 0) for t in members)
                _require(isinstance(value.get(field), (int, float)) and math.isfinite(value[field])
                         and math.isclose(value[field], total, rel_tol=1e-12, abs_tol=1e-9),
                         f"{path}: {field} was not reduced from role-local member traces")
            expected_feasible = not failed and all(t.get("feasible", True) for t in members)
            _require(value.get("feasible") is expected_feasible, f"{path}: feasibility does not match role-local traces")
            report["verified_candidate_role_records"] += 1
            return value

        candidates = e.json("select/candidates.json")
        selected = e.json("select/decision.json")
        nominal_work = None
        for candidate in candidates:
            value = candidate_metadata(candidate, "select", allow_failure=bool(candidate.get("execution_failure")))
            if candidate["route"] == "nominal" and candidate["ensemble"] == 1:
                nominal_work = value["deployed_multiply_add_work"]
            _require(nominal_work is not None, "Selection registry does not begin with the E=1 nominal reference")
            expected_cost = value["deployed_multiply_add_work"] / max(nominal_work, 1)
            _require(math.isclose(candidate["cost"], expected_cost, rel_tol=1e-12, abs_tol=1e-12),
                     f"{candidate['candidate_id']}: selection cost did not use selection graphs only")
            if candidate.get("feasible"):
                _require(value["feasible"] and expected_cost <= config["selection"]["max_normalized_work"],
                         f"{candidate['candidate_id']}: feasible selection ignores role-local execution/cost")
        qC = selected["comparator_index"]
        candidate_metadata(candidates[qC], "confirm")
        closure = e.json("confirm/contrast_registry.json")["candidate_indices"]
        for i in closure:
            c = candidates[i]
            candidate_metadata(c, "confirm", allow_failure=True)
            control = dict(c, state_control="matched_shuffle", candidate_id="shuffle_" + c["candidate_id"])
            candidate_metadata(control, "confirm", allow_failure=True)
        confirmed = e.json("confirm/decision.json")
        qM = confirmed["confirmed_index"]
        reference = candidate_metadata(candidates[qC], "admit")
        policy = candidate_metadata(candidates[qM], "admit", allow_failure=True)
        guard = e.json("admit/prediction_execution.json")
        _require(guard.get("candidate_metadata") == f"predictions/admit/{candidates[qM]['candidate_id']}.json"
                 and guard.get("comparator_metadata") == f"predictions/admit/{candidates[qC]['candidate_id']}.json"
                 and guard.get("execution_role") == "admit", "Admission execution gate names the wrong scoped records")
        valid = policy["feasible"] and not policy.get("execution_failure", False)
        admission = e.json("admit/decision.json")
        _require(guard.get("valid") is valid and admission.get("execution_gate") is valid,
                 "Admission execution gate does not match its recorded role")
        _require(reference["feasible"] and (valid or not admission["passed"]),
                 "Invalid role-local execution was admitted")
        lock = e.json("policy_lock.json")
        E = candidates[qC]["ensemble"]
        nominal = next(c for c in candidates if c["route"] == "nominal" and c["ensemble"] == E)
        test_metadata = summary.get("test_role_execution", {})
        for candidate in (candidates[qC], candidates[lock["deployed_index"]], nominal):
            _require(candidate["candidate_id"] in test_metadata, "Missing locked-predictor test-role metadata")
            candidate_metadata(candidate, "test", embedded=test_metadata[candidate["candidate_id"]])
        q0 = summary.get("test_reference_execution", {}).get("q0", {})
        _require(q0.get("execution_role") == "test" and len(q0.get("members", [])) == E,
                 "Missing feature-only-reference test-role trace")
        for trace, seed in zip(q0["members"], config["training"]["seeds"][:E]):
            _require(trace.get("seed") == seed, "Feature-only-reference ensemble seed differs")
            trace_scope(trace, "test", "feature-only reference")
        report.update(passed=True, verified_metadata_files=sorted(e.checked))
    except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration) as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    return report
