"""Trajectory-tube audit (records:tube_audit) and depth diagnostics (records:depth_diagnostics).

cm_experiments/trajectory_audit.py, scripts/run_trajectory_audit.py.  Planning, the Proposition 2
arithmetic, the statistics and the export run without PyTorch on the bundled synthetic example runs:
job outputs are written with the module's own raw-file writers, exported to temporary roots, traced by
scripts.records.verify_set and read by the Table S22 and Figure S14 generators.  Execution (smoke
lifecycles, both audits end to end) needs PyTorch and runs in CI.
"""
from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cm_experiments import trajectory_audit as ta
from cm_experiments.artifacts import write_json
from cm_experiments.ensemble_grid import RunnerError, job_directory, registered_constants, utc_now
from scripts.records import RECORDS, load_specs, read_summary, verify_set

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "runs"
TORCH = importlib.util.find_spec("torch") is not None
ENVIRONMENT = {"python": "3.x", "machine": "x86_64", "torch": None}
REGISTRY = json.loads((EXAMPLES / "arxiv" / "action_registry.json").read_text())


def specs():
    return load_specs(RECORDS, RECORDS / "specs.json")


def executed_hops(action: dict) -> float:
    """cross_morse.core.executed_hops (that module imports torch)."""
    if action["mode"] != "transport":
        return 0.0
    if action["eta"] is None:
        return action["nu"]
    eta = action["eta"]
    resolvent = sum(t * eta * (1 - eta) ** t for t in range(action["order"] + 1))
    return action["nu"] * ((1 - action["rho"]) + action["rho"] * resolvent)


class Workspace(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cm-trajectory-audit-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.records = self.root / "records"


# ============================================================================= synthetic job outputs
SHARED_PLANT = {"backbone": "shared_transport", "family": "shared", "activation": "relu", "activation_lipschitz": 1.0,
                "weight_norm": 1.3, "weight_rule": "||W_shared||_2", "layer_norm": {"gamma_max": 1.0, "eps": 1e-5},
                "head_norm": 2.0, "head_rule": "||W_head||_2 (linear head)"}
DESIGNATED_PLANT = {"backbone": "linear_gine", "family": "designated", "activation": "relu",
                    "activation_lipschitz": 1.0, "weight_norm": None, "weight_rule": None, "layer_norm": None,
                    "head_norm": 1.5, "head_rule": "||W_head||_2 (linear head)"}


def synthetic_trajectory(item, depth, designated, rng, *, covered=True, stable=True, realized_ok=True,
                         replay_ok=True):
    """Raw measurements of one audited trajectory in the format of trajectory_audit.audit_trajectory."""
    actions = rng.integers(0, 10, depth)
    kind, anchor = np.zeros(depth, dtype=np.int64), np.zeros(depth, dtype=np.int64)
    native = np.zeros(depth, dtype=bool)
    current = 0
    for l, a in enumerate(actions):
        if rng.random() < 0.3:
            current = l
        anchor[l] = current
        if REGISTRY[a]["mode"] == "source":
            kind[l] = ta.KINDS.index("source_reset")
        elif REGISTRY[a]["mode"] == "last_safe":
            kind[l] = ta.KINDS.index("anchor_reset")
        elif not designated:
            kind[l] = ta.KINDS.index("formula")
        elif l % 4 == 0:
            native[l] = True
            kind[l] = ta.KINDS.index("native_secant")
        else:
            kind[l] = ta.KINDS.index("inserted_identity" if a == 0 else "inserted_linear")
    hops = np.array([0.0 if designated and kind[l] == ta.KINDS.index("inserted_identity") else
                     (1.0 if designated and native[l] and a == 0 else executed_hops(REGISTRY[a]))
                     for l, a in enumerate(actions)])
    arrays = {"kind": kind, "action": actions.astype(np.int64), "anchor": anchor, "native_site": native,
              "alpha": np.array([REGISTRY[a]["alpha"] for a in actions]),
              "nu": np.array([REGISTRY[a]["nu"] for a in actions]), "hops": hops,
              "M": rng.uniform(1.0, 3.0, depth),
              "l_ln": np.where(kind == ta.KINDS.index("formula"), rng.uniform(0.6, 1.2, depth), np.nan),
              "beta_secant": np.where(kind == ta.KINDS.index("native_secant"), rng.uniform(0.8, 1.1, depth), np.nan),
              "ref_secant": np.where(kind == ta.KINDS.index("native_secant"), rng.uniform(0.0, 0.3, depth), np.nan)}
    record = {"trajectory": int(item["trajectory"]), "seed": int(item["seed"]), "graph_id": item.get("graph_id"),
              "delta_p": 0.01, "e0": 0.02, "source_norm": 2.0, "started_utc": utc_now(),
              "predicted_status": "ok", "predicted_terminal_depth": depth,
              "replay_identity": {"predicted_max_abs": 0.0, "realized_max_abs": 0.0, "tolerance": 1e-4,
                                  "ok": replay_ok}}
    plant = DESIGNATED_PLANT if designated else SHARED_PLANT
    terms = ta.site_terms(arrays, plant, record["delta_p"])
    tube, _ = ta.forward_tube(terms["beta"], arrays["M"], terms["eps_op"], terms["R"], terms["ref"], anchor,
                              record["e0"])
    error = 0.5 * tube
    if not covered:
        error[-1] = 2.0 * tube[-1] + 1.0
    trace = {"actions": actions.tolist(), "origins": ["policy"] * depth, "safe_depths": anchor.tolist(),
             "terminal_depth": depth}
    realized = dict(trace, origins=["policy"] * (depth - 1) + ["recovery_no_support"]) if not stable else dict(trace)
    arrays.update(norm_predicted=rng.uniform(1.0, 3.0, depth + 1), norm_realized=rng.uniform(1.0, 3.0, depth + 1),
                  state_error=error)
    loss_p = float(rng.uniform(0.5, 1.0))
    if realized_ok:
        loss_r = loss_p + float(rng.uniform(-0.01, 0.01))
        record.update(realized_status="ok", realized_terminal_depth=depth, predicted=trace, realized=realized,
                      mode_stable=stable, predicted_loss_nats=loss_p, realized_loss_nats=loss_r,
                      observed_error_nats=abs(loss_p - loss_r), realized_failure=None)
    else:
        record.update(realized_status="failed", realized_terminal_depth=None, predicted=trace, realized=None,
                      mode_stable=False, predicted_loss_nats=loss_p, realized_loss_nats=None,
                      observed_error_nats=None, realized_failure={"reason": "numerical_execution_failure"})
    record["finished_utc"] = utc_now()
    return record, arrays


def synthetic_tube_jobs(output, *, mismatch=()):
    plan = json.loads((output / "plan.json").read_text())
    for entry in plan["jobs"]:
        directory = job_directory(output, entry)
        directory.mkdir(parents=True)
        write_json(directory / "environment.json", ENVIRONMENT)
        designated = entry["dataset"] == "Peptides-func"
        rng = np.random.default_rng(entry["job"] + 11)
        for n, item in enumerate(entry["trajectories"]):
            replay_ok = (entry["dataset"], item["trajectory"]) not in set(mismatch)
            record, arrays = synthetic_trajectory(item, entry["depth"], designated, rng, covered=n != 1,
                                                  stable=n != 1, realized_ok=n != 2, replay_ok=replay_ok)
            ta.write_tube_trajectory(directory, record, arrays)
        plants = {int(t["seed"]): DESIGNATED_PLANT if designated else SHARED_PLANT for t in entry["trajectories"]}
        ta.finalize_tube_job(directory, plan, entry, started=utc_now(), parent_files=[], plants=plants)
    return plan


def synthetic_depth_jobs(output):
    plan = json.loads((output / "plan.json").read_text())
    for entry in plan["jobs"]:
        directory = job_directory(output, entry)
        directory.mkdir(parents=True)
        write_json(directory / "environment.json", ENVIRONMENT)
        rng = np.random.default_rng(entry["job"] + 5)
        depth, grid = entry["depth"], entry["depth_grid"]
        inputs = entry["inputs"] if entry["inputs"] is not None else [0]
        executions = {"input": [], "seed": [], "terminal": [], "actions": []}
        for graph in inputs:
            for k, seed in enumerate(entry["member_seeds"]):
                actions = [0] * (k + 1) + [int(a) for a in rng.integers(1, 10, depth - k - 1)]
                executions["input"].append(graph)
                executions["seed"].append(seed)
                executions["terminal"].append(depth)
                executions["actions"].append(actions)
        units = np.asarray(inputs if entry["inputs"] is not None else [0, 2, 5])
        losses = 1.0 + 0.01 * np.arange(len(grid))[:, None] + rng.uniform(0, 0.01, (len(grid), len(units)))
        alignment = (rng.uniform(0.5, 1.0, (len(entry["alignment_seeds"]), len(grid), 3))
                     if entry["alignment_seeds"] else None)
        ta.write_depth_job(directory, plan, entry, unit_ids=units, unit_counts=np.arange(1, len(units) + 1),
                           independent=entry["inputs"] is not None, unit_depth_loss=losses, executions=executions,
                           alignment=alignment, entry_dir=Path(entry["run"]), started=utc_now(), parent_files=[])
    return plan


def summary_rows(target, name):
    with (Path(target) / "summary.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def record_view(target, name):
    from paper.build.displays.framework import RecordSetView
    all_specs = specs()
    spec = all_specs["sets"][name]
    rows = read_summary(Path(target), spec, all_specs["summary_columns"])
    for row in rows:
        row["value"] = float(row["value"])
    return RecordSetView(name, Path(target), spec, rows, synthetic=True)


# ============================================================================= contracts
class ContractTests(unittest.TestCase):
    def test_quantities_and_keys_match_the_contracts_and_the_display_generators(self):
        import inspect
        from paper.build.displays import main_paper, run_audits
        sets = specs()["sets"]
        self.assertEqual(set(ta.TUBE_UNITS), set(sets["tube_audit"]["quantities"]))
        self.assertEqual(set(ta.TUBE_UNITS), set(run_audits.S22_QUANTITIES))
        source = inspect.getsource(main_paper.audit_ratios)      # main Figure 2a: the Proposition 2 ratio
        for quantity in ("final_bound_nats", "observed_error_p95_nats"):
            self.assertIn(f'"{quantity}"', source)
            self.assertIn(quantity, ta.TUBE_UNITS)
        self.assertEqual(sets["tube_audit"]["keys"], ["dataset"])
        self.assertEqual(set(ta.DEPTH_UNITS), set(sets["depth_diagnostics"]["quantities"]))
        self.assertEqual(sets["depth_diagnostics"]["keys"], ["dataset", "depth", "phase"])
        self.assertEqual(set(ta.TUBE_DEFINITIONS), set(ta.TUBE_UNITS))
        for name in ta.CODE_FILES:
            self.assertTrue((ROOT / name).is_file(), name)

    def test_default_disturbance_is_the_registered_relative_state_tolerance(self):
        registered = registered_constants()["precision_tolerances_table_S5c"]
        self.assertEqual(registered["bf16"]["state_relative"], 0.008)
        self.assertEqual(registered["fp32"]["state_relative"], 1e-5)


# ============================================================================= Proposition 2 arithmetic
class TubeArithmeticTests(unittest.TestCase):
    def test_activation_constants(self):
        self.assertEqual(ta.activation_lipschitz("relu"), 1.0)
        x = np.linspace(-12, 12, 4_000_001)
        silu = x / (1 + np.exp(-x))
        self.assertAlmostEqual(ta.activation_lipschitz("silu"), float(np.max(np.gradient(silu, x))), places=5)
        self.assertAlmostEqual(ta.activation_lipschitz("silu"), 1.0998393, places=6)
        self.assertAlmostEqual(ta.activation_lipschitz("gelu"), 1.1289041, places=6)
        with self.assertRaises(RunnerError):
            ta.activation_lipschitz("tanh")

    def test_layer_norm_bound_dominates_the_row_jacobian(self):
        rng = np.random.default_rng(3)
        d, eps = 16, 1e-5
        gamma = rng.uniform(0.5, 1.5, d)

        def layer_norm(u):
            centered = u - u.mean()
            return gamma * centered / math.sqrt(centered.var() + eps)

        for _ in range(20):
            u = rng.normal(0, rng.uniform(0.2, 3.0), d)
            jacobian = np.empty((d, d))
            for j in range(d):
                step = np.zeros(d)
                step[j] = 1e-6
                jacobian[:, j] = (layer_norm(u + step) - layer_norm(u - step)) / 2e-6
            bound = ta.layer_norm_lipschitz(float(np.abs(gamma).max()), float(u.var()), eps)
            self.assertLessEqual(np.linalg.norm(jacobian, 2), bound * (1 + 1e-5))
        with self.assertRaises(RunnerError):
            ta.layer_norm_lipschitz(1.0, 0.1, 0.0)

    def test_site_terms_of_every_site_kind(self):
        kinds = ["formula", "source_reset", "anchor_reset", "inserted_identity", "inserted_linear",
                 "native_secant", "native_secant"]
        raw = {"kind": [ta.KINDS.index(k) for k in kinds], "action": [5, 8, 9, 0, 3, 0, 6],
               "alpha": [0.2, 1.0, 1.0, 0.25, 0.25, 0.25, 0.35], "nu": [0.7, 0, 0, 0.7, 0.7, 0.7, 0.6],
               "hops": [1.5, 0, 0, 0, 0.9, 1.0, 1.2], "l_ln": [0.8] + [np.nan] * 6,
               "beta_secant": [np.nan] * 5 + [1.05, 0.95], "ref_secant": [np.nan] * 5 + [0.0, 0.2]}
        plant = {"activation_lipschitz": 1.1, "weight_norm": 2.0}
        terms = ta.site_terms(raw, plant, 0.01)
        np.testing.assert_allclose(terms["beta"], [0.8 * 0.8 * 1.1 * 2.0, 0, 0, 1, 1 - 0.7 * 0.25, 1.05, 0.95])
        np.testing.assert_allclose(terms["eps_op"], [0.015, 0, 0, 0, 0.75 * 0.9 * 0.01 / (1 - 0.175), 0.01,
                                                     0.65 * 1.2 * 0.01])
        np.testing.assert_allclose(terms["R"], [0.8 * 0.2, 1, 1, 0, 0.7 * 0.25, 0.0, 0.2])
        self.assertEqual(terms["ref"].tolist(), [1, 1, 2, 0, 1, 1, 1])
        # The inserted linear form: beta * M * eps equals (1 - alpha) h ||P - P~|| M exactly.
        self.assertAlmostEqual(terms["beta"][4] * terms["eps_op"][4], 0.75 * 0.9 * 0.01)
        broken = dict(raw, l_ln=[np.nan] * 7)
        with self.assertRaises(RunnerError):
            ta.site_terms(broken, plant, 0.01)

    def test_forward_tube_is_the_product_sum_bound_and_the_registered_recursion(self):
        from cm_theory.stability import perturbation_recursion, unrolled_bound
        rng = np.random.default_rng(1)
        L = 12
        beta, M, eps, R = rng.uniform(0.5, 1.2, L), rng.uniform(1, 3, L), rng.uniform(0, 0.02, L), rng.uniform(0, 0.4, L)
        ref = np.full(L, ta.REF_SOURCE)
        tau, e_ref = ta.forward_tube(beta, M, eps, R, ref, np.zeros(L, dtype=int), 0.05)
        np.testing.assert_allclose(tau, perturbation_recursion(beta, M, eps, R, e_ref, 0.05))
        self.assertAlmostEqual(unrolled_bound(beta, M, eps, R, e_ref, 0.05), tau[-1])
        # A last-safe reset returns the tube to its value at the recorded anchor depth.
        beta[7], R[7], ref[7] = 0.0, 1.0, ta.REF_ANCHOR
        anchors = np.zeros(L, dtype=int)
        anchors[7] = 3
        tau, e_ref = ta.forward_tube(beta, M, eps, R, ref, anchors, 0.05)
        self.assertEqual(tau[8], tau[3])
        self.assertEqual(e_ref[7], tau[3])
        self.assertAlmostEqual(unrolled_bound(beta, M, eps, R, e_ref, 0.05), tau[-1])
        anchors[7] = 9
        with self.assertRaises(RunnerError):
            ta.forward_tube(beta, M, eps, R, ref, anchors, 0.05)

    def test_trajectory_audit_bound_coverage_and_failures(self):
        rng = np.random.default_rng(4)
        item = {"trajectory": 6, "seed": 1, "graph_id": None}
        record, arrays = synthetic_trajectory(item, 6, False, rng)
        audit = ta.trajectory_audit(record, arrays, SHARED_PLANT, relative_tolerance=1e-5)
        self.assertTrue(audit["covered"])
        self.assertTrue(audit["mode_stable"])
        self.assertAlmostEqual(audit["bound_nats"], math.sqrt(2) * 2.0 * audit["tube"][-1])
        self.assertAlmostEqual(audit["unrolled"], audit["tube"][-1])
        record, arrays = synthetic_trajectory(item, 6, False, rng, covered=False, stable=False)
        audit = ta.trajectory_audit(record, arrays, SHARED_PLANT, relative_tolerance=1e-5)
        self.assertFalse(audit["covered"])
        self.assertFalse(audit["mode_stable"])
        record, arrays = synthetic_trajectory(item, 6, False, rng, realized_ok=False)
        audit = ta.trajectory_audit(record, arrays, SHARED_PLANT, relative_tolerance=1e-5)
        self.assertFalse(audit["covered"])
        self.assertIsNone(audit["observed_error_nats"])
        # The tolerance is relative to the larger state norm, never absolute slack.
        record, arrays = synthetic_trajectory(item, 6, False, rng)
        tube = ta.trajectory_audit(record, arrays, SHARED_PLANT, relative_tolerance=0)["tube"]
        arrays["state_error"] = tube + 1e-7
        self.assertFalse(ta.trajectory_audit(record, arrays, SHARED_PLANT, relative_tolerance=0)["covered"])
        self.assertTrue(ta.trajectory_audit(record, arrays, SHARED_PLANT, relative_tolerance=1e-5)["covered"])

    def test_tube_summary_quantities(self):
        audits = []
        for k, (covered, stable, error) in enumerate(((True, True, 0.01), (True, False, 0.03), (False, False, None))):
            audits.append({"beta": np.array([0.5, 1.0, 1.5]) * (k + 1), "m_eps_op": np.array([0.1, 0.2, 0.3]) * (k + 1),
                           "reference_term": np.array([0.01, 0.02, 0.03]) * (k + 1),
                           "tube": np.array([0.0, 1.0, 2.0, 3.0]),
                           "bound_nats": 0.1 * (k + 1), "covered": covered, "mode_stable": stable,
                           "observed_error_nats": error, "e0": 0.02, "head_norm": 2.0})
        summary = ta.tube_summary(audits)
        q = summary["quantities"]
        betas = np.concatenate([a["beta"] for a in audits])
        self.assertAlmostEqual(q["beta_median"], float(np.median(betas)))
        self.assertAlmostEqual(q["beta_p95"], float(np.percentile(betas, 95)))
        self.assertAlmostEqual(q["m_eps_op"], 0.4)                 # median over depths of [0.2, 0.4, 0.6]
        self.assertAlmostEqual(q["reference_term"], 0.04)          # median of [0.02, 0.04, 0.06]
        self.assertAlmostEqual(q["tube_coverage"], 2 / 3)
        self.assertAlmostEqual(q["mode_stable_coverage"], 1 / 3)
        self.assertAlmostEqual(q["final_bound_nats"], float(np.percentile([0.1, 0.2, 0.3], 95)))
        self.assertAlmostEqual(q["observed_error_median_nats"], 0.02)
        self.assertAlmostEqual(q["observed_error_p95_nats"], 0.01 + 0.95 * 0.02)
        self.assertEqual(summary["counts"], {"audited": 3, "covered": 2, "mode_stable": 1, "with_realized_output": 2,
                                             "sites": 9})
        self.assertGreater(summary["envelope_bound_nats"], 0)
        with self.assertRaises(RunnerError):
            ta.tube_summary([])

    def test_disturbances(self):
        x = np.random.default_rng(0).normal(size=(30, 7))
        x[4] = 0.0
        out = ta.input_perturbation(x, 0.008, np.random.default_rng(1))
        ratio = np.linalg.norm(out - x, axis=1) / np.maximum(np.linalg.norm(x, axis=1), 1e-300)
        np.testing.assert_allclose(np.delete(ratio, 4), 0.008)
        np.testing.assert_array_equal(out[4], 0.0)
        np.testing.assert_array_equal(out, ta.input_perturbation(x, 0.008, np.random.default_rng(1)))
        np.testing.assert_array_equal(out, ta.input_perturbation(x, 0.008, np.random.default_rng(1), chunk=7))
        np.testing.assert_array_equal(ta.input_perturbation(x, 0.0, np.random.default_rng(1)), x)
        self.assertEqual(ta.input_perturbation(x.astype(np.float32), 0.008, np.random.default_rng(1)).dtype,
                         np.float32)
        u = ta.pair_uniform(np.arange(20000), np.arange(20000)[::-1], 123)
        self.assertTrue(np.all((u >= 0) & (u < 1)))
        np.testing.assert_array_equal(u, u[::-1])                      # (i, j) and (j, i) share one draw
        self.assertLess(abs(float(u.mean()) - 0.5), 0.01)
        self.assertFalse(np.array_equal(u, ta.pair_uniform(np.arange(20000), np.arange(20000)[::-1], 124)))
        edges = np.array([[0, 1, 1, 2, 2, 3, 0, 0], [1, 0, 2, 1, 3, 2, 0, 1]])
        base = np.array([1.0, 1.0, 2.0, 2.0, 0.5, 0.5, 1.0, 1.0])
        weights = ta.edge_weight_perturbation(edges, base, 0.1, np.random.default_rng(2))
        factor = weights / base
        self.assertTrue(np.all((factor >= 0.9) & (factor <= 1.1)))
        self.assertEqual(factor[0], factor[1])      # one draw for both directions (and the duplicate pair)
        self.assertEqual(factor[0], factor[7])
        self.assertEqual(factor[2], factor[3])
        self.assertEqual(factor[4], factor[5])
        np.testing.assert_array_equal(ta.edge_weight_perturbation(edges, None, 0.0, np.random.default_rng(2)),
                                      np.ones(8))
        with self.assertRaises(RunnerError):
            ta.edge_weight_perturbation(edges, None, 1.0, np.random.default_rng(2))

    def test_holdout_groups_follow_the_runtime_partition(self):
        entries = [{"trajectory": t, "seed_index": t % 2, "graph_id": None} for t in range(6)]
        # (0,0) (0,2) (0,4) (1,1) (1,3) (1,5)
        self.assertEqual(ta.holdout_groups(entries, False), [0, 3, 1, 4, 2, 5])
        graphs = [{"trajectory": t, "seed_index": t % 2, "graph_id": g} for t, g in enumerate((9, 4, 9, 7))]
        self.assertEqual(ta.holdout_groups(graphs, True), [2, 0, 2, 1])


# ============================================================================= depth statistics
class DepthStatisticsTests(unittest.TestCase):
    def test_first_intervention_and_empirical_quartiles(self):
        self.assertEqual(ta.first_intervention_depth([0, 0, 3, 0, 9]), 3.0)
        self.assertEqual(ta.first_intervention_depth([7, 0]), 1.0)
        self.assertTrue(math.isinf(ta.first_intervention_depth([0, 0, 0])))
        self.assertTrue(math.isinf(ta.first_intervention_depth([0, 0, -1, -1])))
        values = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual([ta.empirical_quantile(values, q) for q in (0.25, 0.5, 0.75)], [1.0, 2.0, 3.0])
        censored = [2.0, 3.0, math.inf, math.inf]
        self.assertEqual(ta.empirical_quantile(censored, 0.25), 2.0)
        self.assertEqual(ta.empirical_quantile(censored, 0.5), 3.0)
        self.assertIsNone(ta.empirical_quantile(censored, 0.75))
        with self.assertRaises(RunnerError):
            ta.empirical_quantile([], 0.5)

    def test_depth_phases_and_transport_counts(self):
        self.assertEqual([ta.phase_index(s, 6) for s in range(6)], [0, 0, 1, 1, 2, 2])
        self.assertEqual([sum(ta.phase_index(s, 128) == k for s in range(128)) for k in range(3)], [43, 43, 42])
        actions = np.array([[0, 1, 8, 7, 0, 9], [3, 0, 0, 2, -1, -1]])
        counts = ta.phase_counts(actions, 6)
        self.assertEqual(counts["executed"].tolist(), [4, 4, 2])
        self.assertEqual(counts["transport"].tolist(), [2, 2, 0])   # resets A8/A9 are not transport

    def test_weighted_curve_and_entry_units(self):
        losses = np.array([[1.0, 2.0], [3.0, 5.0]])
        np.testing.assert_allclose(ta.weighted_curve(losses, [1, 3]), [1.75, 4.5])
        with self.assertRaises(RunnerError):
            ta.weighted_curve(losses, [1, 0])
        samples = np.array([[1.0, 2.0, 3.0, np.nan], [2.0, 2.0, 5.0, np.nan]])
        means, units, weights = ta.unit_depth_losses(samples, [4, 4, 7, 7], independent=False)
        np.testing.assert_allclose(means, [[1.5, 3.0], [2.0, 5.0]])
        self.assertEqual(units.tolist(), [4, 7])
        self.assertEqual(weights.tolist(), [2.0, 1.0])
        _, _, weights = ta.unit_depth_losses(samples, [4, 4, 7, 7], independent=True)
        self.assertEqual(weights.tolist(), [1.0, 1.0])

    def test_supported_deterioration_is_the_first_depth_with_an_interval_above_zero(self):
        rng = np.random.default_rng(9)
        grid = [1, 2, 4, 8, 16, 32]
        base = rng.uniform(0.8, 1.2, 40)
        shifts = [0.20, 0.05, 0.0, 0.0002, 0.03, 0.06]           # best at depth 4; small, then clear increases
        losses = np.array([base + s + rng.normal(0, 0.002, 40) for s in shifts])
        losses[2] = base
        weights = rng.integers(5, 50, 40).astype(float)
        result = ta.supported_deterioration(losses, weights, grid, 2, reps=200, seed=1)
        self.assertEqual(result["best_depth"], 4)
        self.assertEqual(result["supported_depth"], 16)
        self.assertEqual([t["depth"] for t in result["tests"]], [8, 16, 32])
        self.assertFalse(result["tests"][0]["supported"])
        flat = np.array([base + rng.normal(0, 0.01, 40) for _ in grid])
        flat[2] = base
        self.assertIsNone(ta.supported_deterioration(flat, weights, grid, 2, reps=200, seed=1)["supported_depth"])
        self.assertEqual(ta.supported_deterioration(losses, weights, grid, 5, reps=200, seed=1)["tests"], [])

    def test_split_runs_are_the_units_of_the_campaign_interval(self):
        from scipy import stats
        curves = np.array([[1.0, 0.9, 0.95, 1.10], [1.0, 0.88, 0.97, 1.12], [1.0, 0.92, 0.93, 1.08]])
        result = ta.supported_deterioration_splits(curves, [1, 2, 4, 8])
        self.assertEqual(result["best_depth"], 2)
        d = curves[:, 2] - curves[:, 1]
        half = stats.t.ppf(0.975, 2) * d.std(ddof=1) / math.sqrt(3)
        self.assertAlmostEqual(result["tests"][0]["lower"], d.mean() - half)
        self.assertEqual(result["supported_depth"], 8)
        with self.assertRaises(RunnerError):
            ta.supported_deterioration_splits(curves[:1], [1, 2, 4, 8])

    def test_alignment_curve(self):
        values = np.array([[[1.0, 0.8, 0.6], [0.5, 0.5, 0.5]], [[0.9, 0.9, 0.9], [0.4, 0.2, 0.3]]])
        np.testing.assert_allclose(ta.alignment_curve(values), [0.85, 0.4])
        self.assertEqual(ta.first_plotted_depth([1, 2, 4]), 2)
        self.assertEqual(ta.registered_grid(6), [1, 2, 4, 6])
        self.assertEqual(ta.registered_grid(128), [1, 2, 4, 8, 16, 32, 64, 128])


# ============================================================================= planning
class PlanningTests(Workspace):
    def test_tube_plan_audits_the_assessment_partition(self):
        path = ta.plan_tube_audit([EXAMPLES / "arxiv", EXAMPLES / "peptides_func"], self.root / "tube",
                                  max_trajectories=0)
        plan = json.loads(path.read_text())
        self.assertEqual(plan["record_set"], "tube_audit")
        arxiv, peptides = plan["jobs"]
        # arxiv: groups are sorted (seed_index, trajectory) pairs; assessment groups 3 and 10 -> trajectories 6, 9
        self.assertEqual([(t["trajectory"], t["seed"]) for t in arxiv["trajectories"]], [(6, 1), (9, 2)])
        # Peptides: groups are sorted graph ids; assessment groups 2 and 9 -> graphs 18 and 72
        self.assertEqual([(t["trajectory"], t["graph_id"]) for t in peptides["trajectories"]], [(1, 72), (2, 18),
                                                                                                (11, 72)])
        self.assertEqual(arxiv["profile"], "full")
        self.assertEqual(arxiv["policy"]["route"], "learned_stop")
        self.assertEqual(arxiv["executed_precision"], "fp32")
        self.assertEqual(arxiv["coverage_relative_tolerance"], 1e-5)
        self.assertEqual(plan["perturbation"]["input_relative"], 0.008)
        self.assertEqual(plan["perturbation"]["edge_weight_relative"], 0.008)
        self.assertEqual(arxiv["audit_seed"], json.loads((EXAMPLES / "arxiv/config.json").read_text())["seed"]
                         + ta.SEED_OFFSET)
        self.assertEqual(arxiv["selection"]["assessment_trajectories"], 2)
        self.assertIn("trajectory_audit.py", " ".join(plan["trajectory_audit_source_sha256"]))
        self.assertTrue(plan["datasets"][0]["synthetic"])

    def test_tube_plan_subsample_and_refusals(self):
        path = ta.plan_tube_audit([EXAMPLES / "peptides_func"], self.root / "sub", max_trajectories=2)
        chosen = [t["trajectory"] for t in json.loads(path.read_text())["jobs"][0]["trajectories"]]
        self.assertEqual(len(chosen), 2)
        self.assertEqual(chosen, sorted(chosen))
        self.assertTrue(set(chosen) <= {1, 2, 11})
        again = ta.plan_tube_audit([EXAMPLES / "peptides_func"], self.root / "sub2", max_trajectories=2)
        self.assertEqual(chosen, [t["trajectory"] for t in json.loads(again.read_text())["jobs"][0]["trajectories"]])
        for kwargs, message in ((dict(policy="best"), "policy"), (dict(precision="bf16"), "precision"),
                                (dict(input_relative=0, edge_relative=0), "nonzero"),
                                (dict(edge_relative=1.0), "edge_relative"), (dict(probes=0), "probes")):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(RunnerError, message):
                ta.plan_tube_audit([EXAMPLES / "arxiv"], self.root / "x", **kwargs)
        with self.assertRaisesRegex(RunnerError, "one parent"):
            ta.plan_tube_audit([EXAMPLES / "arxiv", EXAMPLES / "arxiv"], self.root / "y")
        with self.assertRaisesRegex(RunnerError, "outside the repository"):
            ta.plan_tube_audit([EXAMPLES / "arxiv"], ROOT / "tube-output")
        ta.plan_tube_audit([EXAMPLES / "arxiv"], self.root / "z")
        with self.assertRaisesRegex(RunnerError, "exists"):
            ta.plan_tube_audit([EXAMPLES / "arxiv"], self.root / "z")

    def test_tube_plan_detects_a_partition_it_cannot_reproduce(self):
        run = self.root / "arxiv"
        shutil.copytree(EXAMPLES / "arxiv", run)
        path = run / "fit/full/calibration_partition.json"
        partition = json.loads(path.read_text())
        partition["groups"]["assessment"] = [0, 1, 2]
        path.write_text(json.dumps(partition))
        with self.assertRaisesRegex(RunnerError, "does not reproduce"):
            ta.plan_tube_audit([run], self.root / "bad")

    def test_depth_plan(self):
        path = ta.plan_depth_diagnostics([EXAMPLES / "arxiv", EXAMPLES / "peptides_func"], self.root / "depth")
        plan = json.loads(path.read_text())
        self.assertEqual(plan["reference"], "ogbn-arxiv")
        self.assertEqual(plan["role"], "fit_val")
        self.assertFalse(plan["truncated"])
        arxiv, peptides = plan["jobs"]
        self.assertEqual(arxiv["depth_grid"], [1, 2, 4, 6])
        self.assertEqual(arxiv["first_plotted_depth"], 2)
        self.assertTrue(arxiv["reference"])
        self.assertEqual(arxiv["alignment_seeds"], [1, 2])
        self.assertIsNone(arxiv["inputs"])
        self.assertEqual(arxiv["member_seeds"], [1, 2])                  # learned_stop at E = 2
        self.assertFalse(peptides["reference"])
        self.assertEqual(peptides["alignment_seeds"], [])
        self.assertEqual(peptides["inputs"], [24, 52])                   # the fit_val graphs
        self.assertEqual(set(arxiv["entry_files"]), {"train/entry_initial.json", "train/entry_depth_values.npz"})
        config = json.loads((EXAMPLES / "arxiv/config.json").read_text())
        self.assertEqual(arxiv["bootstrap"], {"reps": config["selection"]["bootstrap_reps"], "seed": config["seed"] + 1,
                                              "alpha": 0.025})
        truncated = ta.plan_depth_diagnostics([EXAMPLES / "peptides_func"], self.root / "trunc", max_inputs=1)
        self.assertTrue(json.loads(truncated.read_text())["truncated"])
        for kwargs, message in ((dict(role="test"), "role"), (dict(role="select"), "role"),
                                (dict(reference="COCO-SP"), "reference"), (dict(max_inputs=-1), "max_inputs")):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(RunnerError, message):
                ta.plan_depth_diagnostics([EXAMPLES / "arxiv", EXAMPLES / "peptides_func"], self.root / "x", **kwargs)
        relabeled = self.root / "roman"
        shutil.copytree(EXAMPLES / "arxiv", relabeled)
        config = json.loads((relabeled / "config.json").read_text())
        config["benchmark_dataset"] = "Roman-empire"
        (relabeled / "config.json").write_text(json.dumps(config))
        with self.assertRaisesRegex(RunnerError, "declare --reference"):
            ta.plan_depth_diagnostics([EXAMPLES / "peptides_func", relabeled], self.root / "noref")


# ============================================================================= export
class TubeExportTests(Workspace):
    def setUp(self):
        super().setUp()
        self.output = self.root / "tube"
        ta.plan_tube_audit([EXAMPLES / "arxiv", EXAMPLES / "peptides_func"], self.output, max_trajectories=0)

    def test_export_verifies_and_feeds_table_s22(self):
        from paper.build.displays.testing import build_one, context, read_table
        synthetic_tube_jobs(self.output)
        target = ta.export_tube_audit(self.output, root=self.records)
        verify_set(target, specs()["sets"]["tube_audit"], specs())
        rows = summary_rows(target, "tube_audit")
        self.assertEqual({(r["dataset"], r["quantity"]) for r in rows},
                         {(d, q) for d in ("ogbn-arxiv", "Peptides-func") for q in ta.TUBE_UNITS})
        self.assertTrue(all(r["source_file"].endswith("tube_audit.json") for r in rows))
        document = json.loads((target / "raw/Peptides-func/tube_audit.json").read_text())
        self.assertEqual(document["counts"]["audited"], 3)
        self.assertEqual(document["counts"]["with_realized_output"], 2)
        self.assertAlmostEqual(document["quantities"]["mode_stable_coverage"], 1 / 3)
        self.assertAlmostEqual(document["quantities"]["tube_coverage"], 1 / 3)
        provenance = json.loads((target / "provenance.json").read_text())
        self.assertTrue(provenance["synthetic"])
        self.assertTrue((target / "code/cm_experiments/trajectory_audit.py").is_file())
        self.assertTrue((target / "raw/ogbn-arxiv/arxiv/tube_arrays.npz").is_file())
        source = json.loads((target / "code/SOURCE.json").read_text())
        self.assertEqual(source["exporter"], "scripts/run_trajectory_audit.py")
        from cm_campaigns.contract import RECORD_SETS
        self.assertNotIn("tube_audit", RECORD_SETS)              # the plan's coverage is scoped to the export
        view = record_view(target, "tube_audit")
        out = self.root / "s22"
        build_one("supplementary:Table S22", context(records={"tube_audit": view}), out)
        table = read_table(out, "S22")
        self.assertEqual([r["Dataset"] for r in table], ["arxiv", "Peptides"])

    def test_trajectory_slices_pool_into_one_dataset_row(self):
        output = self.root / "sliced"
        ta.plan_tube_audit([EXAMPLES / "peptides_func"], output, max_trajectories=0, chunk_size=2)
        plan = json.loads((output / "plan.json").read_text())
        self.assertEqual([len(j["trajectories"]) for j in plan["jobs"]], [2, 1])
        self.assertEqual({j["run_member"] for j in plan["jobs"]}, {"peptides_func"})
        synthetic_tube_jobs(output)
        target = ta.export_tube_audit(output, root=self.records)
        verify_set(target, specs()["sets"]["tube_audit"], specs())
        document = json.loads((target / "raw/Peptides-func/tube_audit.json").read_text())
        self.assertEqual(document["counts"]["audited"], 3)
        self.assertEqual([m["member"] for m in document["members"]], ["peptides_func.part000", "peptides_func.part001"])

    def test_export_refusals(self):
        with self.assertRaisesRegex(RunnerError, "has not completed"):
            ta.export_tube_audit(self.output, root=self.records)
        synthetic_tube_jobs(self.output, mismatch=[("ogbn-arxiv", 9), ("Peptides-func", 1), ("Peptides-func", 2),
                                                   ("Peptides-func", 11)])
        with self.assertRaisesRegex(RunnerError, "replay"):
            ta.export_tube_audit(self.output, root=self.records)
        target = ta.export_tube_audit(self.output, root=self.records, allow_partial=True)
        document = json.loads((target / "raw/ogbn-arxiv/tube_audit.json").read_text())
        self.assertEqual(document["excluded"]["arxiv"]["replay_mismatch"], [9])
        self.assertEqual(document["counts"]["audited"], 1)
        provenance = json.loads((target / "provenance.json").read_text())
        self.assertEqual(provenance["skipped_datasets"], ["Peptides-func"])     # no trajectory left to audit
        self.assertFalse(provenance["coverage_complete"])

    def test_synthetic_sets_never_enter_records_and_jobs_belong_to_their_plan(self):
        synthetic_tube_jobs(self.output)
        with self.assertRaisesRegex(RunnerError, "complete measured sets"):
            ta.export_tube_audit(self.output)
        plan = json.loads((self.output / "plan.json").read_text())
        marker = job_directory(self.output, plan["jobs"][0]) / "tube.json"
        result = json.loads(marker.read_text())
        result["identities"]["plan_sha256"] = "0" * 64
        marker.write_text(json.dumps(result))
        with self.assertRaisesRegex(RunnerError, "another plan"):
            ta.export_tube_audit(self.output, root=self.records)


class DepthExportTests(Workspace):
    def setUp(self):
        super().setUp()
        self.output = self.root / "depth"
        ta.plan_depth_diagnostics([EXAMPLES / "arxiv", EXAMPLES / "peptides_func"], self.output)

    def test_export_verifies_and_feeds_figure_s14(self):
        from paper.build.displays.framework import RunView
        from paper.build.displays.testing import build_one, context
        synthetic_depth_jobs(self.output)
        target = ta.export_depth_diagnostics(self.output, root=self.records)
        verify_set(target, specs()["sets"]["depth_diagnostics"], specs())
        rows = summary_rows(target, "depth_diagnostics")
        alignment = [r for r in rows if r["quantity"] == "source_alignment_change"]
        self.assertEqual({r["dataset"] for r in alignment}, {"ogbn-arxiv"})
        self.assertEqual([r["depth"] for r in alignment], ["2", "4", "6"])
        self.assertEqual(float(alignment[0]["value"]), 0.0)
        policy = [r for r in rows if r["quantity"] == "policy_loss_change"]
        self.assertEqual({(r["dataset"], r["depth"]) for r in policy},
                         {("ogbn-arxiv", d) for d in ("2", "4", "6")} | {("Peptides-func", d) for d in ("2", "4", "8")})
        self.assertTrue(all(r["phase"] == ta.ALL for r in alignment + policy))
        phases = [(r["dataset"], r["phase"]) for r in rows if r["quantity"] == "transport_fraction"]
        self.assertEqual(phases, [(d, p) for d in ("ogbn-arxiv", "Peptides-func") for p in ta.PHASES])
        quartiles = {r["quantity"]: float(r["value"]) for r in rows
                     if r["dataset"] == "ogbn-arxiv" and r["quantity"].startswith("first_intervention")}
        self.assertEqual(quartiles, {"first_intervention_depth_q25": 2.0, "first_intervention_depth_median": 2.0,
                                     "first_intervention_depth_q75": 3.0})
        for row in rows:
            if row["quantity"] in ("source_alignment_change", "policy_loss_change"):
                self.assertNotEqual(row["depth"], ta.ALL)
            else:
                self.assertEqual(row["depth"], ta.ALL)
        document = json.loads((target / "raw/Peptides-func/depth_diagnostics.json").read_text())
        self.assertEqual(document["supported_deterioration"]["best_depth"], 8)
        self.assertIsNone(document["supported_deterioration"]["supported_depth"])
        self.assertNotIn("supported_deterioration_depth",
                         {r["quantity"] for r in rows if r["dataset"] == "Peptides-func"})
        self.assertTrue((target / "raw/ogbn-arxiv/arxiv/entry/entry_depth_values.npz").is_file())
        ctx = context(runs={"ogbn-arxiv": RunView("ogbn-arxiv", EXAMPLES / "arxiv")},
                      records={"depth_diagnostics": record_view(target, "depth_diagnostics")})
        writer, _ = build_one("supplementary:Figure S14", ctx, self.root / "s14")
        self.assertIn("figures/FigS14.png", writer.files)
        panel = (self.root / "s14/data/FigS14/panel_b.csv").read_text().splitlines()
        self.assertEqual(len(panel), 7)

    def test_slices_of_one_run_merge_before_the_statistics(self):
        whole, sliced = self.root / "whole", self.root / "sliced"
        ta.plan_depth_diagnostics([EXAMPLES / "peptides_func"], whole)
        ta.plan_depth_diagnostics([EXAMPLES / "peptides_func"], sliced, chunk_size=1)
        plan = json.loads((sliced / "plan.json").read_text())
        self.assertEqual([j["member"] for j in plan["jobs"]], ["peptides_func.part000", "peptides_func.part001"])
        self.assertEqual([j["inputs"] for j in plan["jobs"]], [[24], [52]])
        self.assertEqual([j["alignment_seeds"] for j in plan["jobs"]], [[1, 2], []])   # reference: first slice only
        self.assertFalse(plan["truncated"])
        for output in (whole, sliced):
            plan = json.loads((output / "plan.json").read_text())
            for entry in plan["jobs"]:
                directory = job_directory(output, entry)
                directory.mkdir(parents=True)
                write_json(directory / "environment.json", ENVIRONMENT)
                executions = {"input": [], "seed": [], "terminal": [], "actions": []}
                for graph in entry["inputs"]:
                    for seed in entry["member_seeds"]:
                        executions["input"].append(graph)
                        executions["seed"].append(seed)
                        executions["terminal"].append(8)
                        executions["actions"].append([0, 0, graph % 7 + 1, 0, 4, 0, 8, 0])
                losses = np.array([[1.0 + 0.01 * k + 0.001 * g for g in entry["inputs"]] for k in range(4)])
                alignment = np.full((len(entry["alignment_seeds"]), 4, 3), 0.7) if entry["alignment_seeds"] else None
                if alignment is not None:
                    alignment[:, 2:] = 0.6
                ta.write_depth_job(directory, plan, entry, unit_ids=np.asarray(entry["inputs"]),
                                   unit_counts=np.ones(len(entry["inputs"])), independent=True, unit_depth_loss=losses,
                                   executions=executions, alignment=alignment, entry_dir=Path(entry["run"]),
                                   started=utc_now(), parent_files=[])
        values = []
        for output, root in ((whole, self.records), (sliced, self.root / "records2")):
            target = ta.export_depth_diagnostics(output, root=root)
            verify_set(target, specs()["sets"]["depth_diagnostics"], specs())
            values.append({(r["depth"], r["phase"], r["quantity"]): r["value"] for r in summary_rows(target, ta.DEPTH)})
        self.assertEqual(values[0], values[1])
        self.assertEqual(values[0][("4", ta.ALL, "source_alignment_change")], "-0.1")

    def test_depth_export_refusals(self):
        with self.assertRaisesRegex(RunnerError, "has not completed"):
            ta.export_depth_diagnostics(self.output, root=self.records)
        synthetic_depth_jobs(self.output)
        plan = json.loads((self.output / "plan.json").read_text())
        copy = job_directory(self.output, plan["jobs"][0]) / "entry" / "entry_initial.json"
        copy.write_text(copy.read_text().replace("1.8853938579559326", "1.9"))
        with self.assertRaisesRegex(RunnerError, "SHA-256"):
            ta.export_depth_diagnostics(self.output, root=self.records)
        truncated = self.root / "trunc"
        ta.plan_depth_diagnostics([EXAMPLES / "peptides_func"], truncated, max_inputs=1)
        with self.assertRaisesRegex(RunnerError, "truncates"):
            ta.export_depth_diagnostics(truncated, root=self.records)

    def test_entry_audit_must_reproduce_its_depth_risk(self):
        synthetic_depth_jobs(self.output)
        plan = json.loads((self.output / "plan.json").read_text())
        directory = job_directory(self.output, plan["jobs"][0])
        result = json.loads((directory / "depth.json").read_text())
        curves = ta.entry_audit_curves(directory, result)
        self.assertEqual(curves["grid"], [1, 2, 4, 6])
        self.assertEqual(curves["best_index"], 2)
        initial = json.loads((directory / "entry/entry_initial.json").read_text())
        initial["depth_loss"][0] += 0.01
        (directory / "entry/entry_initial.json").write_text(json.dumps(initial))
        with self.assertRaisesRegex(RunnerError, "do not reproduce"):
            ta.entry_audit_curves(directory, result)


class CommandLineTests(Workspace):
    def call(self, *argv):
        from scripts import run_trajectory_audit
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = run_trajectory_audit.main([str(a) for a in argv])
        return code, out.getvalue(), err.getvalue()

    def test_plan_run_and_export_commands(self):
        output = self.root / "cli-tube"
        code, out, _ = self.call("tube", "plan", "--parent", EXAMPLES / "arxiv", "--output", output,
                                 "--max-trajectories", "1", "--input-relative", "0.01")
        self.assertEqual(code, 0)
        plan = json.loads((output / "plan.json").read_text())
        self.assertEqual(plan["perturbation"]["input_relative"], 0.01)
        self.assertEqual(len(plan["jobs"][0]["trajectories"]), 1)
        code, _, err = self.call("tube", "run", "--output", output)
        self.assertEqual(code, 2)
        self.assertIn("trust", err)
        code, _, err = self.call("tube", "export", "--output", output, "--root", self.records)
        self.assertEqual(code, 2)
        self.assertIn("has not completed", err)
        code, _, err = self.call("depth", "plan", "--parent", EXAMPLES / "peptides_func", "--output",
                                 self.root / "cli-depth", "--role", "fit_val")
        self.assertEqual(code, 0)
        code, _, err = self.call("depth", "export", "--output", output)
        self.assertEqual(code, 2)
        self.assertIn("not a depth_diagnostics plan", err)


# ============================================================================= execution (PyTorch; CI)
@unittest.skipUnless(TORCH, "PyTorch is required to execute the audits")
class ExecutionTests(unittest.TestCase):
    """Smoke lifecycles on synthetic data, then both audits end to end (software checks, not measurements)."""

    @classmethod
    def setUpClass(cls):
        import torch
        from cross_morse.pipeline import Pipeline
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory(prefix="cm-trajectory-runs-")
        cls.root = Path(cls.temporary.name)
        cls.runs = {}
        for name, stem in (("arxiv", "arxiv_smoke"), ("peptides", "peptides_func_smoke")):
            config = json.loads((ROOT / "configs" / f"{stem}.json").read_text())
            if name == "arxiv":
                config["training"].update(epochs=1, patience=1)
            config["controller"].update(epochs=1, patience=1)
            config["selection"]["bootstrap_reps"] = 20
            config["admission"]["bootstrap_reps"] = 20
            run = cls.root / "runs" / name
            with contextlib.redirect_stdout(io.StringIO()):
                Pipeline(config, cls.root / "data", run, "cpu").run()
            cls.runs[name] = run

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.work = Path(tempfile.mkdtemp(prefix="cm-trajectory-", dir=self.root))
        self.records = self.work / "records"

    def test_grouping_matches_the_runtime(self):
        from cross_morse.exploration import load_records
        from cross_morse.learning import _holdout_groups
        from cm_experiments.ensemble_grid import SealedRun
        for run in self.runs.values():
            sealed = SealedRun(run)
            entries = ta.explore_entries(sealed)
            records = load_records(run / "explore")
            groups = _holdout_groups(records)
            by_trajectory = {int(t): int(g) for t, g in zip(records["trajectory"], groups)}
            collection = "graph_id" in records
            self.assertEqual(ta.holdout_groups(entries, collection), [by_trajectory[e["trajectory"]] for e in entries])

    def test_zero_disturbance_replay_is_exact_and_inside_the_tube(self):
        from cm_experiments.ensemble_grid import ParentContext, SealedRun
        from cross_morse.policies import execute_policy
        for name, run in self.runs.items():
            sealed = SealedRun(run)
            candidate, _ = ta.policy_candidate(sealed, "deployed")
            ctx = ParentContext(sealed, "cpu", through="admit")
            controller = ctx.controller_for(candidate)
            seed = ctx.config["training"]["seeds"][0]
            model = ctx.plant(seed)
            plant = ta.plant_constants(model)
            if ctx.collection:
                from cross_morse.runtime import graph_operator
                gid = int(ctx.data.graph_splits["explore"][0])
                episode = ctx.data.graph(gid)
                graph = graph_operator(episode, "cpu")
            else:
                episode, graph = ctx.data, ctx.graph
            x = episode.x
            probs, trace = execute_policy(model, x, graph, ctx.config, candidate, controller, ctx.reference)
            summary = ta._trace_summary(trace)
            raw, replayed = ta.joint_replay(model, x, x, graph, graph, ctx.config, summary, summary, plant,
                                            {"random": 1, "relative_radius": 1e-3}, np.random.default_rng(0), "cpu")
            self.assertEqual(float(np.abs(replayed["predicted"] - probs).max()), 0.0, name)
            self.assertEqual(replayed["e0"], 0.0)
            self.assertTrue(np.all(raw["state_error"][np.isfinite(raw["state_error"])] == 0.0), name)
            terms = ta.site_terms(raw, plant, 0.0)
            tube, _ = ta.forward_tube(terms["beta"], raw["M"], terms["eps_op"], terms["R"], terms["ref"],
                                      raw["anchor"], 0.0)
            self.assertTrue(np.all(tube == 0.0))
            self.assertTrue(np.all(np.isfinite(terms["beta"])))

    def test_tube_audit_end_to_end(self):
        output = self.work / "tube"
        ta.plan_tube_audit([self.runs["arxiv"], self.runs["peptides"]], output, max_trajectories=2, probes=1)
        ta.run_tube_audit(output, device="cpu", trust=True)
        plan = json.loads((output / "plan.json").read_text())
        for entry in plan["jobs"]:
            result = json.loads((job_directory(output, entry) / "tube.json").read_text())
            for record in result["trajectories"]:
                if record["predicted_status"] == "ok":
                    self.assertTrue(record["replay_identity"]["ok"], record["replay_identity"])
                    self.assertGreater(record["e0"], 0.0)
                    self.assertGreater(record["delta_p"], 0.0)
        target = ta.export_tube_audit(output, root=self.records, allow_partial=True)
        verify_set(target, specs()["sets"]["tube_audit"], specs())
        rows = summary_rows(target, "tube_audit")
        self.assertEqual({r["dataset"] for r in rows}, {"ogbn-arxiv", "Peptides-func"})

    def test_depth_diagnostics_end_to_end(self):
        output = self.work / "depth"
        ta.plan_depth_diagnostics([self.runs["arxiv"], self.runs["peptides"]], output, reference="ogbn-arxiv",
                                  reps=20)
        ta.run_depth_diagnostics(output, device="cpu", trust=True)
        plan = json.loads((output / "plan.json").read_text())
        arxiv = json.loads((job_directory(output, plan["jobs"][0]) / "depth.json").read_text())
        self.assertNotEqual(arxiv["sealed_actions_agree"], False)
        target = ta.export_depth_diagnostics(output, root=self.records, allow_partial=True)
        verify_set(target, specs()["sets"]["depth_diagnostics"], specs())
        rows = summary_rows(target, "depth_diagnostics")
        self.assertEqual({r["dataset"] for r in rows if r["quantity"] == "source_alignment_change"}, {"ogbn-arxiv"})
        self.assertEqual({r["dataset"] for r in rows if r["quantity"] == "policy_loss_change"},
                         {"ogbn-arxiv", "Peptides-func"})


if __name__ == "__main__":
    unittest.main()
