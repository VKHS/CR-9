"""Decision-log audits of the deployed rules (cm_experiments/decision_audit.py, scripts/run_decision_audit.py):
records action_gap_audit, decision_traces and reliability_diagnostics.

Planning, statistics and export run without PyTorch: plans on the bundled synthetic example runs, statistics on
small hand-made arrays, export on synthetic job outputs of all seven datasets written to temporary roots, traced by
scripts.records.verify_set and read back by the display generators. Execution (smoke lifecycles, the instrumented
execution and its replay parity) needs PyTorch and runs in CI.
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

from cm_experiments import decision_audit as da
from cm_experiments.ensemble_grid import PLAN_SCHEMA, RunnerError, job_directory, save_arrays
from scripts.records import RECORDS, load_specs, read_summary, verify_set

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "runs"
TORCH = importlib.util.find_spec("torch") is not None


def specs():
    return load_specs(RECORDS, RECORDS / "specs.json")


def summary(target):
    with (Path(target) / "summary.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def view(name, target):
    from paper.build.displays.testing import fake_records
    return fake_records(name, [dict(r, value=float(r["value"])) for r in summary(target)], root=Path(target))


# ============================================================================= contracts and consumers
class ContractTests(unittest.TestCase):
    def test_quantities_match_the_registered_contracts(self):
        sets = specs()["sets"]
        for name in da.SETS:
            self.assertEqual(set(da.DEFINITIONS[name]), set(sets[name]["quantities"]), name)
            for quantity in sets[name]["quantities"]:
                self.assertIn(quantity, da.UNITS)
        self.assertEqual(sets["action_gap_audit"]["keys"], ["dataset"])
        self.assertEqual(sets["decision_traces"]["keys"], ["dataset"])
        self.assertEqual(sets["reliability_diagnostics"]["keys"], ["diagnostic", "level"])

    def test_keys_match_what_the_generators_read(self):
        from paper.build.displays import main_paper, run_audits
        from paper.build.displays.framework import DATASETS
        self.assertEqual(tuple(k for k, _ in main_paper.ORIGINS), da.CATEGORIES)
        self.assertEqual(DATASETS, da.DATASETS)
        self.assertTrue(set(run_audits.S21_QUANTITIES) < set(da.DEFINITIONS["action_gap_audit"]))
        self.assertEqual(da.OBSERVER_LEVELS[0], "raw")         # Figure S16b normalizes by the raw observer
        self.assertEqual(da.registered_horizons(), [1, 4, 8, 16, 32])
        self.assertEqual(set(da.MODULE_ARMS.values()),
                         set(next(m for m in specs()["sets"]["module_ablations"]["modules"]
                                  if m["module"] == "observer")["arms"]))

    def test_registered_threshold_and_tolerance(self):
        from cm_experiments.budget import Certificate
        self.assertEqual(da.certificate_floor(), Certificate(gap=1., lipschitz_bound=0., radius=0.).floor)
        self.assertEqual(da.IDENTITY_TOLERANCE, 1e-7)


# ============================================================================= statistics
class StatisticsTests(unittest.TestCase):
    def test_origin_categories_are_exhaustive_and_exclusive(self):
        self.assertEqual(da.origin_category("fixed_schedule", "phase"), "schedule_nominal")
        self.assertEqual(da.origin_category("nominal", "best_depth"), "schedule_nominal")
        self.assertEqual(da.origin_category("policy", "radius"), "state_conditioned")
        self.assertEqual(da.origin_category("policy", "one_step"), "state_conditioned")
        self.assertEqual(da.origin_category("policy", "mpc_rh"), "exclusive_sequence")
        for origin in da.ORIGINS[3:]:
            self.assertEqual(da.origin_category(origin, "mpc_rh"), "recovery")
        self.assertEqual(da.origin_category("policy", "radius", fallback=True), "input_fallback")
        with self.assertRaisesRegex(RunnerError, "no registered category"):
            da.origin_category("replay", "radius")
        with self.assertRaisesRegex(RunnerError, "no registered category"):
            da.origin_category("policy", "phase")

    def test_decision_mass_fractions_and_entropy(self):
        mass = da.DecisionMass()
        mass.add_trace([1, 2, 0], ["policy", "policy", "recovery_observer"], "radius", 0.5, fallback=False)
        mass.add_trace([3, 2, 0], ["policy", "policy", "policy"], "radius", 0.5, fallback=False)
        mass.add_trace([4, 4, 4], ["fixed_schedule"] * 3, "phase", 1.0, fallback=True)
        fractions = mass.fractions()
        self.assertAlmostEqual(sum(fractions.values()), 1.0, places=12)
        self.assertAlmostEqual(fractions["state_conditioned"], 2.5 / 6)
        self.assertAlmostEqual(fractions["recovery"], 0.5 / 6)
        self.assertAlmostEqual(fractions["input_fallback"], 3 / 6)
        # depth 0: {1: .5, 3: .5, 4: 1}; depth 1: {2: 1, 4: 1}; depth 2: {0: 1, 4: 1}; equal weights of 2
        h0 = -(2 * 0.25 * math.log(0.25) + 0.5 * math.log(0.5))
        self.assertAlmostEqual(mass.entropy(), (h0 + 2 * math.log(2)) / 3, places=12)
        again = da.DecisionMass.from_json(json.loads(json.dumps(mass.to_json())))
        self.assertEqual(again.categories, mass.categories)
        self.assertAlmostEqual(again.entropy(), mass.entropy(), places=14)
        with self.assertRaisesRegex(RunnerError, "one origin per"):
            mass.add_trace([0, 1], ["policy"], "radius", 1.0, fallback=False)
        with self.assertRaisesRegex(RunnerError, "no executed decision"):
            da.DecisionMass().fractions()

    def test_entropy_weights_depths_by_their_decision_mass(self):
        value = da.same_depth_entropy([(0, 1, 1.0), (0, 2, 1.0), (1, 3, 2.0)])
        self.assertAlmostEqual(value, (2 * math.log(2) + 0) / 4)
        self.assertEqual(da.same_depth_entropy([(0, 5, 3.0), (1, 5, 1.0)]), 0.0)

    def test_gap_summary(self):
        gaps = np.array([0.0005, 0.001, 0.003, 0.004, 0.006, 0.01, 0.02])
        s = da.gap_summary(gaps, 0.002)
        self.assertEqual(s["median"], 0.004)
        self.assertAlmostEqual(s["p5"], float(np.quantile(gaps, 0.05)))
        self.assertAlmostEqual(s["near_zero_fraction"], 2 / 7)
        self.assertEqual(s["sites"], 7)
        with self.assertRaisesRegex(RunnerError, "nonnegative"):
            da.gap_summary([-1e-3, 1.0], 0.002)
        with self.assertRaisesRegex(RunnerError, "nonempty finite"):
            da.gap_summary([], 0.002)

    def test_chi_square_region(self):
        low, high = da.chi_square_region(2, 0.05)
        self.assertAlmostEqual(low, -2 * math.log(1 - 0.025), places=10)      # chi2(2) is exponential(1/2)
        self.assertAlmostEqual(high, -2 * math.log(0.025), places=10)

    def test_perturbation_operator_is_the_observer_noise_model_without_history_rows(self):
        rng = np.random.default_rng(3)
        k, names = 3, ["entropy", "margin", "source_js", "effective_rank_entropy", "log1p_dirichlet",
                       "relative_depth", "normalized_mass", "normalized_hops", "log_state_energy"]
        basis, _ = np.linalg.qr(rng.normal(size=(len(names), k)))
        a = rng.normal(size=(k, k))
        r = a @ a.T + 0.1 * np.eye(k)
        scale = rng.uniform(0.5, 2.0, len(names))
        operator = da.perturbation_operator(r, basis, scale, names)
        history = [names.index(n) for n in ("relative_depth", "normalized_mass", "normalized_hops")]
        self.assertTrue(np.all(operator[history] == 0))
        sensed = [i for i in range(len(names)) if i not in history]
        expected = (np.diag(scale) @ basis @ r @ basis.T @ np.diag(scale))[np.ix_(sensed, sensed)]
        np.testing.assert_allclose((operator @ operator.T)[np.ix_(sensed, sensed)], expected, rtol=1e-9, atol=1e-12)
        np.testing.assert_array_equal(da.perturbation_draw(operator, [5, 1, 2, 3]),
                                      da.perturbation_draw(operator, [5, 1, 2, 3]))
        self.assertFalse(np.allclose(da.perturbation_draw(operator, [5, 1, 2, 3]),
                                     da.perturbation_draw(operator, [5, 1, 2, 4])))
        np.testing.assert_array_equal(da.perturbation_operator(r, basis, scale, names, 0.0), 0)
        with self.assertRaisesRegex(RunnerError, "do not match"):
            da.perturbation_operator(r, basis, scale, names[:-1])

    def test_sequence_radii_follow_the_runtime_rule_without_the_A0_identity(self):
        errors = np.array([0.1, 0.4, 0.2, 0.3, 0.05, 0.5])
        units = np.array([0, 0, 1, 1, 2, 2])
        actions = np.array([0, 0, 0, 0, 3, 3])
        strata = np.array([1, 1, 1, 1, 1, 1])
        radii = da.sequence_radii(errors, units, strata, actions, alpha=0.5, minimum=2)
        self.assertEqual(radii["actions"]["0"]["radius"], 0.4)       # higher quantile of unit maxima {0.4, 0.3}
        self.assertEqual(radii["actions"]["3"]["radius"], 0.5)
        self.assertIn("1:0", radii["cells"])                        # two trajectories: a cell
        self.assertNotIn("1:3", radii["cells"])                     # one trajectory: action-pooled only
        self.assertEqual(da.radius_lookup(radii, 1, 0), 0.4)
        self.assertEqual(da.radius_lookup(radii, 1, 3), 0.5)
        self.assertEqual(da.radius_lookup(radii, 1, 7), math.inf)

    def test_harm_and_coverage_indicators(self):
        delta = np.array([0.002, -0.001, 0.003, 0.0, np.nan])
        mu = np.array([-0.004, -0.002, 0.001, 0.0, -0.01])
        radius = np.array([0.003, 0.0015, 0.001, 0.0, 0.01])
        found = da.harm_indicators(delta, mu, radius, np.ones(5, bool))
        safe_harmful, valid = found["false_safe"]
        self.assertEqual(valid[:, 0].tolist(), [True, True, True, True, False])
        self.assertEqual(safe_harmful[:, 0].tolist(), [True, False, False, False, False])
        self.assertEqual(found["harm_miss"][1][:, 0].tolist(), [True, False, True, False, False])
        self.assertEqual(found["coverage_1"][0][:, 0].tolist(), [False, True, False, True, False])
        self.assertEqual(found["false_safe_point"][0][:, 0].tolist(), [True, False, False, False, False])
        covered, evaluable = da.sequence_indicators(np.array([0.1, 0.2, np.nan]), np.array([0.12, 0.5, 0.0]),
                                                    np.array([0.05, 0.1, 1.0]))
        self.assertEqual(covered[:, 0].tolist(), [True, False, False])
        self.assertEqual(evaluable[:, 0].tolist(), [True, True, False])

    def shared(self, rng, sites=40, units=6):
        horizons = [1, 4]
        counts = rng.integers(3, 9, units).astype(float)
        mu = rng.normal(-0.002, 0.003, sites)
        delta = mu[:, None] + rng.normal(0, 0.004, (sites, units))
        realized = rng.normal(0, 0.01, (2, sites, units))
        realized[1, -3:] = np.nan
        return {"mode": np.asarray("shared"), "horizons": np.asarray(horizons), "unit_ids": np.arange(units),
                "unit_counts": counts, "mu": mu, "radius1": np.abs(rng.normal(0.004, 0.002, sites)),
                "valid": np.ones(sites, bool), "delta_sums": delta * counts, "prediction": rng.normal(0, 0.01, (2, sites)),
                "radius_h": np.full((2, sites), 0.012), "realized_sums": realized * counts}

    def test_shared_unit_totals_and_the_stratified_interval(self):
        rng = np.random.default_rng(11)
        arrays = self.shared(rng)
        stratum = da.SharedUnits(arrays)
        pooled = arrays["delta_sums"].sum(1) / arrays["unit_counts"].sum()
        expected = da.harm_indicators(pooled, arrays["mu"], arrays["radius1"], arrays["valid"])
        for name in ("false_safe", "harm_miss", "coverage_1"):
            n, d = stratum.totals(name, np.ones((1, stratum.units)))
            self.assertEqual((n[0], d[0]), (expected[name][0].sum(), expected[name][1].sum()))
        n, d = stratum.totals("sequence_4", np.ones((1, stratum.units)))
        self.assertEqual(d[0], 37)
        result = da.rate_interval([stratum], "coverage_1", reps=300, seed=[1, 2])
        self.assertLessEqual(result["low"], result["value"])
        self.assertLessEqual(result["value"], result["high"])
        self.assertEqual(result, da.rate_interval([stratum], "coverage_1", reps=300, seed=[1, 2]))
        self.assertIsNone(da.rate_interval([], "coverage_1", reps=10, seed=1))

    def test_own_unit_totals_resample_whole_graphs(self):
        rng = np.random.default_rng(4)
        sites = 30
        merged = {"unit": np.repeat([3, 7, 9], 10), "delta": rng.normal(0, 0.004, sites),
                  "mu": rng.normal(-0.001, 0.003, sites), "radius1": np.full(sites, 0.004),
                  "valid": np.ones(sites, bool), "prediction": rng.normal(0, 0.01, (1, sites)),
                  "radius_h": np.full((1, sites), 0.01), "realized": rng.normal(0, 0.01, (1, sites))}
        totals = da.own_unit_totals(merged, [1])
        self.assertEqual(totals["unit_ids"].tolist(), [3, 7, 9])
        self.assertEqual(totals["false_safe_den"].tolist(), [10, 10, 10])
        own = da.OwnUnits(totals)
        n, d = own.totals("harm_miss", np.array([[2.0, 0.0, 1.0]]))
        expected = da.harm_indicators(merged["delta"], merged["mu"], merged["radius1"], merged["valid"])["harm_miss"]
        self.assertEqual(n[0], 2 * expected[0][:10].sum() + expected[0][20:].sum())
        self.assertEqual(d[0], 2 * expected[1][:10].sum() + expected[1][20:].sum())
        with self.assertRaisesRegex(RunnerError, "not among"):
            own.totals("sequence_8", np.ones((1, 3)))


# ============================================================================= planning
class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cm-decision-plan-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_plan_declares_the_deployed_rules_and_the_registered_settings(self):
        path = da.plan_decision_audit([EXAMPLES / "arxiv", EXAMPLES / "peptides_func"], self.root / "a",
                                      stress_dataset="ogbn-arxiv", reps=40)
        plan = json.loads(path.read_text())
        self.assertEqual(plan["record_set"], da.PLAN_KIND)
        self.assertEqual(plan["horizons"], [1, 4, 8, 16, 32])
        self.assertEqual(plan["near_zero_threshold_nats"], da.certificate_floor())
        self.assertEqual(plan["perturbation"]["factor"], 1.0)
        self.assertEqual(plan["bootstrap"]["reps"], 40)
        arxiv, peptides = plan["jobs"]
        self.assertEqual((arxiv["route"], arxiv["global_return"], arxiv["feedback"]), ("learned_stop", True, False))
        self.assertTrue(arxiv["stress"] and arxiv["gate"]["needed"])
        self.assertEqual(arxiv["evaluation_inputs"], 1)
        self.assertEqual((peptides["route"], peptides["gate"]["needed"]), ("phase", False))
        self.assertEqual(peptides["evaluation_inputs"], 16)
        self.assertEqual(arxiv["perturbation_seed"], json.loads((EXAMPLES / "arxiv/config.json").read_text())["seed"] + 13)
        self.assertEqual(set(plan["audit_code_sha256"]), set(da.AUDIT_FILES + da.REUSED_FILES))

    def test_fixture_override_and_declared_audit_subset(self):
        path = da.plan_decision_audit([EXAMPLES / "arxiv", EXAMPLES / "peptides_func"], self.root / "b",
                                      fixture_route={"ogbn-arxiv": "radius", "Peptides-func": "mpc_rh"},
                                      audit_inputs=5, audit_seed=2)
        arxiv, peptides = json.loads(path.read_text())["jobs"]
        self.assertEqual((arxiv["route"], arxiv["admitted"], arxiv["global_return"]), ("radius", True, False))
        self.assertEqual(peptides["route"], "mpc_rh")
        self.assertEqual(len(peptides["audited_inputs"]), 5)
        self.assertIsNone(arxiv["audited_inputs"])
        self.assertTrue(set(peptides["audited_inputs"]) <= set(np.load(EXAMPLES / "peptides_func/prepared/graph_splits.npz")["test"].tolist()))
        self.assertIsNotNone(arxiv["fixture_override"])

    def test_plan_refusals(self):
        for kwargs, message in ((dict(role="select"), "decision roles"), (dict(fixture_route="phase"), "feedback route"),
                                (dict(stress_dataset="COCO-SP"), "not among"), (dict(ema_weight=0.0), "EMA"),
                                (dict(audit_inputs=-1), "nonnegative")):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(RunnerError, message):
                da.plan_decision_audit([EXAMPLES / "arxiv"], self.root / "x", **kwargs)
        with self.assertRaisesRegex(RunnerError, "outside the repository"):
            da.plan_decision_audit([EXAMPLES / "arxiv"], ROOT / "decision-audit-output")
        with self.assertRaisesRegex(RunnerError, "one parent"):
            da.plan_decision_audit([EXAMPLES / "arxiv", EXAMPLES / "arxiv"], self.root / "y")
        real = self.root / "real"
        shutil.copytree(EXAMPLES / "arxiv", real)
        config = json.loads((real / "config.json").read_text())
        config["scientific_run"], config["dataset"] = True, "ogbn-arxiv"
        (real / "config.json").write_text(json.dumps(config))
        with self.assertRaisesRegex(RunnerError, "synthetic smoke"):
            da.plan_decision_audit([real], self.root / "z", fixture_route="radius")
        da.plan_decision_audit([EXAMPLES / "arxiv"], self.root / "once")
        with self.assertRaisesRegex(RunnerError, "exists"):
            da.plan_decision_audit([EXAMPLES / "arxiv"], self.root / "once")

    def test_a_sealed_input_gate_is_reused_with_its_own_profile_and_level(self):
        from cm_experiments.artifacts import canonical, seal, sha256
        config = json.loads((EXAMPLES / "arxiv/config.json").read_text())
        state = json.loads((EXAMPLES / "arxiv/run_state.json").read_text())

        def gate_run(name, config_sha):
            directory = self.root / name
            (directory / "raw").mkdir(parents=True)
            (directory / "raw" / "gate.json").write_text(json.dumps({"schema": "cm-static-input-gate-1", "threshold": 1.0}))
            (directory / "plan.json").write_text(json.dumps({"gate": {"alpha": 0.1, "profile": "lean"}}))
            (directory / "experiment.json").write_text(json.dumps({"experiment": "input_gate", "parent": {
                "config_sha256": config_sha, "source_sha256": state["source_sha256"]}}))
            seal(directory)
            return directory
        good = gate_run("gate", canonical(config))
        path = da.plan_decision_audit([EXAMPLES / "arxiv"], self.root / "g", gate_from={"ogbn-arxiv": good},
                                      stress_dataset="ogbn-arxiv")
        gate = json.loads(path.read_text())["jobs"][0]["gate"]
        self.assertEqual((gate["profile"], gate["alpha"]), ("lean", 0.1))
        self.assertEqual(gate["reused"]["gate_sha256"], sha256(good / "raw" / "gate.json"))
        other = gate_run("other", "0" * 64)
        with self.assertRaisesRegex(RunnerError, "sealed input_gate run"):
            da.plan_decision_audit([EXAMPLES / "arxiv"], self.root / "h", gate_from={"ogbn-arxiv": other})

    def test_run_requires_explicit_trust(self):
        path = da.plan_decision_audit([EXAMPLES / "arxiv"], self.root / "t")
        with self.assertRaisesRegex(RunnerError, "explicit trust"):
            da.run_decision_audit(path.parent, device="cpu", trust=False)


# ============================================================================= export (synthetic job outputs)
JOBS = (("ogbn-arxiv", "arxiv", "radius", "shared"), ("ogbn-proteins", "proteins", "one_step", "shared"),
        ("ogbn-products", "products", "phase", None), ("Roman-empire", "split_00", "mpc_rh", "shared"),
        ("Roman-empire", "split_01", "mpc_rh", "shared"), ("PascalVOC-SP", "pascal", "mpc_rh", "own"),
        ("COCO-SP", "coco", "mpc_rh", "own"), ("Peptides-func", "peptides", "radius", "own"))
DEPTH, MEMBERS = 40, 2


def synthetic_plan(output, stress="PascalVOC-SP"):
    output.mkdir(parents=True)
    jobs = []
    for index, (dataset, member, route, mode) in enumerate(JOBS):
        jobs.append({"job": index, "dataset": dataset, "slug": da.slug(dataset), "member": member,
                     "run": f"/runs/{member}", "synthetic": True, "route": route,
                     "feedback": route in da.FEEDBACK_ROUTES, "plans": route in da.PLANNING_ROUTES,
                     "global_return": route == "phase", "admitted": route != "phase", "fixture_override": None,
                     "stress": dataset == stress, "mode": mode,
                     "identity": {"name": member, "dataset": dataset, "synthetic": True}})
    datasets = []
    for dataset in dict.fromkeys(j["dataset"] for j in jobs):
        members = [j["member"] for j in jobs if j["dataset"] == dataset]
        datasets.append({"dataset": dataset, "slug": da.slug(dataset), "members": members, "synthetic": True})
    plan = {"schema": PLAN_SCHEMA, "record_set": da.PLAN_KIND, "role": "test", "horizons": [1, 4, 8, 16, 32],
            "near_zero_threshold_nats": 0.002, "perturbation": {"factor": 1.0}, "bootstrap": {"reps": 200, "seed": 5},
            "tolerance_nats": da.IDENTITY_TOLERANCE, "stress_dataset": stress, "datasets": datasets, "jobs": jobs}
    (output / "plan.json").write_text(json.dumps(plan, indent=1))
    return plan


def fabricate_job(output, entry, rng):
    """Raw files of one completed job in the layout execution_phase and _run_job write (synthetic values)."""
    directory = job_directory(output, entry)
    directory.mkdir(parents=True)
    files = {"environment.json": da._dump(directory / "environment.json",
                                          {"python": "3.x", "machine": "x86_64", "torch": None})}
    route, units = entry["route"], (1 if entry["mode"] == "own" else 6)
    inputs = 8 if entry["mode"] == "own" else 1
    mass = da.DecisionMass()
    unit_rows = []
    for input_id in range(inputs):
        for member in range(MEMBERS):
            if entry["global_return"]:
                mass.add_trace([4] * DEPTH, ["fixed_schedule"] * DEPTH, route, units / MEMBERS, fallback=True)
                continue
            actions = rng.integers(0, 8, DEPTH).tolist()
            origins = ["policy"] * (DEPTH - 1) + ["recovery_observer"]
            mass.add_trace(actions, origins, route, units / MEMBERS, fallback=False)
        for u in range(units):
            unit_rows.append({"unit_id": f"graph:{input_id}" if entry["mode"] == "own" else f"core:{u}",
                              "input_id": input_id, "g_input": int(rng.random() > 0.1), "gate_score": "1.0",
                              "served_by": "deployed", "stress_served": 1, "stress_active": int(rng.random() > 0.3)})
    files["decisions.json"] = da._dump(directory / "decisions.json", {"mass": mass.to_json()})
    fields = ["unit_id", "input_id", "g_input", "gate_score", "served_by", "stress_served", "stress_active"]
    files["units.csv"] = da._write_csv(directory / "units.csv", unit_rows, fields)
    files["executions.csv"] = da._write_csv(directory / "executions.csv", [{"input_id": 0, "rule": "deployed"}],
                                            ["input_id", "rule"])
    files["parity.json"] = da._dump(directory / "parity.json", {"probabilities": [], "digests": {}})
    if entry["feedback"]:
        sites = inputs * MEMBERS * DEPTH
        audited = sites - inputs * MEMBERS
        files["gaps.npz"] = save_arrays(
            directory / "gaps.npz", gap=rng.lognormal(math.log(0.005), 0.9, audited),
            kind=np.full(audited, 2 if route == "mpc_rh" else 1), switch=(rng.random(audited) < 0.05).astype(int),
            inversion=np.zeros(audited, int), input_id=np.zeros(audited, int), member=np.zeros(audited, int),
            depth=np.zeros(audited, int), plan_stable=(rng.random(audited) < 0.9).astype(int) if route == "mpc_rh"
            else np.zeros(0, int), anchor_stable=(rng.random(inputs * MEMBERS) < 0.95).astype(int),
            excluded_origin=np.full(inputs * MEMBERS, da.ORIGINS.index("recovery_observer")),
            route=np.asarray(route), feedback=np.asarray(1))
        files["calibration.json"] = da._dump(directory / "calibration.json", {"status": "CALIBRATED", "radii": {}})
        horizons = [1, 4, 8, 16, 32]
        mu = rng.normal(-0.002, 0.003, sites)
        prediction = rng.normal(0, 0.01, (len(horizons), sites))
        radius_h = np.full((len(horizons), sites), 0.02)
        depth = np.tile(np.arange(DEPTH), inputs * MEMBERS)
        if entry["mode"] == "shared":
            counts = rng.integers(3, 9, units).astype(float)
            delta = mu[:, None] + rng.normal(0, 0.004, (sites, units))
            realized = prediction[:, :, None] + rng.normal(0, 0.008, (len(horizons), sites, units))
            for k, h in enumerate(horizons):
                realized[k, depth + h > DEPTH] = np.nan
            files["reliability.npz"] = save_arrays(
                directory / "reliability.npz", mode=np.asarray("shared"), horizons=np.asarray(horizons),
                unit_ids=np.arange(units), unit_counts=counts, mu=mu, radius1=np.abs(rng.normal(0.004, 0.002, sites)),
                valid=np.ones(sites, bool), prediction=prediction, radius_h=radius_h, delta_sums=delta * counts,
                realized_sums=realized * counts)
        else:
            realized = prediction + rng.normal(0, 0.008, prediction.shape)
            for k, h in enumerate(horizons):
                realized[k, depth + h > DEPTH] = np.nan
            merged = {"unit": np.repeat(np.arange(inputs), MEMBERS * DEPTH), "delta": mu + rng.normal(0, 0.004, sites),
                      "mu": mu, "radius1": np.abs(rng.normal(0.004, 0.002, sites)), "valid": np.ones(sites, bool),
                      "prediction": prediction, "radius_h": radius_h, "realized": realized}
            files["reliability.npz"] = save_arrays(directory / "reliability.npz", mode=np.asarray("own"),
                                                   horizons=np.asarray(horizons), **da.own_unit_totals(merged, horizons))
    variants = {}
    for level, scale in zip(da.OBSERVER_LEVELS, (1.6, 1.3, 1.1, 1.0)):
        rows = 500
        variants[level] = {"assessment_rows": rows, "squared_error_sum": rows * (0.006 * scale) ** 2,
                           "nis_events": 200, "nis_in_region": int(200 * (0.95 - 0.1 * (scale - 1)))}
    files["observers.json"] = da._dump(directory / "observers.json", {"status": "MEASURED", "variants": variants})
    da._dump(directory / "audit.json", {"schema": da.JOB_SCHEMA, "files": files, "started_utc": "2026-11-01T00:00:00+00:00",
                                        "finished_utc": "2026-11-01T01:00:00+00:00"})
    return directory


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cm-decision-export-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "audit"
        self.plan = synthetic_plan(self.output)
        rng = np.random.default_rng(20261107)
        self.jobs = [fabricate_job(self.output, entry, rng) for entry in self.plan["jobs"]]
        self.records = self.root / "records"

    def export(self, **kwargs):
        options = dict(root=self.records, hardware="synthetic test fixture", period="2026-11-01")
        options.update(kwargs)
        return {Path(p).name: Path(p) for p in da.export_decision_audit(self.output, **options)}

    def test_three_sets_trace_every_value_and_feed_their_generators(self):
        targets = self.export()
        self.assertEqual(set(targets), set(da.SETS))
        for name, target in targets.items():
            entry = verify_set(target, specs()["sets"][name], specs())
            self.assertTrue(entry["synthetic"])
            self.assertTrue((target / "code" / "cm_experiments" / "decision_audit.py").is_file())
            self.assertTrue((target / "code" / "plan.json").is_file())
        from paper.build.displays import main_paper, run_audits, analytical
        from paper.build.displays.framework import DisplayContext
        traces = main_paper.trace_rows(view("decision_traces", targets["decision_traces"]))
        self.assertEqual([r["dataset"] for r in traces], list(da.DATASETS))
        products = next(r for r in traces if r["dataset"] == "ogbn-products")
        self.assertEqual((products["input_fallback"], products["entropy"]), (1.0, 0.0))
        gaps = view("action_gap_audit", targets["action_gap_audit"])
        values = run_audits.record_rows(gaps, run_audits.S21_QUANTITIES + ("near_zero_threshold_nats",))
        self.assertNotIn("ogbn-products", values)                 # open-loop deployed rule: no scored decision
        self.assertEqual(set(values), set(da.DATASETS) - {"ogbn-products"})
        self.assertEqual(values["ogbn-arxiv"]["beam_sequence_stability"], 1.0)
        ctx = DisplayContext(records={"action_gap_audit": gaps}, synthetic=True)
        for dataset in ("ogbn-arxiv", "Roman-empire", "COCO-SP"):
            model = analytical.gap_model(ctx, dataset)
            self.assertTrue(0 < model["p5"] < model["median"])
        rows = summary(targets["reliability_diagnostics"])
        coverage = sorted({r["level"] for r in rows if r["diagnostic"] == "coverage"}, key=float)
        self.assertEqual(coverage, ["1", "4", "8", "16", "32"])
        self.assertEqual([r["level"] for r in rows if r["quantity"] == "risk_rmse"], list(da.OBSERVER_LEVELS))
        harm = {r["level"] for r in rows if r["diagnostic"] == "harm"}
        self.assertEqual(harm, set(da.DATASETS) - {"ogbn-products"})

    def test_values_are_recomputed_from_the_raw_job_files(self):
        targets = self.export(sets=("action_gap_audit", "decision_traces"))
        lines = {(r["dataset"], r["quantity"]): float(r["value"]) for r in summary(targets["action_gap_audit"])}
        roman = [np.load(job_directory(self.output, e) / "gaps.npz") for e in self.plan["jobs"]
                 if e["dataset"] == "Roman-empire"]
        gaps = np.concatenate([z["gap"] for z in roman])
        self.assertAlmostEqual(lines[("Roman-empire", "median_gap_nats")], float(np.median(gaps)), places=9)
        self.assertAlmostEqual(lines[("Roman-empire", "near_zero_fraction")], float(np.mean(gaps < 0.002)), places=9)
        self.assertEqual(lines[("Roman-empire", "decision_sites")], len(gaps))
        self.assertAlmostEqual(lines[("Roman-empire", "beam_sequence_stability")],
                               float(np.mean(np.concatenate([z["plan_stable"] for z in roman]))), places=9)
        traces = {(r["dataset"], r["quantity"]): float(r["value"]) for r in summary(targets["decision_traces"])}
        for dataset in da.DATASETS:
            self.assertAlmostEqual(sum(traces[(dataset, f"fraction_{c}")] for c in da.CATEGORIES), 1.0, places=9)
        self.assertAlmostEqual(traces[("ogbn-arxiv", "fraction_recovery")], 1 / DEPTH, places=9)
        document = json.loads((targets["action_gap_audit"] / "raw" / "summary.json").read_text())
        self.assertEqual(document["details"]["ogbn-products"]["status"], "NO_SCORED_DECISIONS")

    def test_reliability_pools_runs_and_reports_the_registered_levels(self):
        targets = self.export(sets=("reliability_diagnostics",))
        rows = {(r["diagnostic"], r["level"], r["quantity"]): float(r["value"])
                for r in summary(targets["reliability_diagnostics"])}
        self.assertAlmostEqual(rows[("observer", "raw", "risk_rmse")], 0.006 * 1.6, places=9)
        self.assertAlmostEqual(rows[("observer", "conditioned KF", "nis_coverage")], 0.95, places=9)
        for level in ("1", "4", "8", "16", "32"):
            self.assertLessEqual(rows[("coverage", level, "coverage_ci_low")], rows[("coverage", level, "coverage")])
            self.assertLessEqual(rows[("coverage", level, "coverage")], rows[("coverage", level, "coverage_ci_high")])
        units = []
        units += da._read_csv(job_directory(self.output, self.plan["jobs"][5]) / "units.csv")
        self.assertAlmostEqual(rows[("stress", "clean", "input_fallback")],
                               float(np.mean([1 - int(u["g_input"]) for u in units])), places=9)
        strata = [s for _, s in da.reliability_strata(da.collect_jobs(self.output, self.plan))]
        n = sum(s.totals("false_safe", np.ones((1, s.units)))[0][0] for s in strata[2:3])
        self.assertGreaterEqual(n, 0)
        from paper.build.displays.testing import build_one, context, fake_records
        controls = fake_records("negative_controls", [
            {"control": c, "quantity": q, "value": v}
            for c in ("graph irrelevant", "misleading graph", "sensor bias", "support shift")
            for q, v in (("input_fallback", 0.9), ("residual_activity", 0.05), ("worst_pre_fallback_harm", 0.01))])
        ctx = context(records={"reliability_diagnostics": view("reliability_diagnostics", targets["reliability_diagnostics"]),
                               "negative_controls": controls})
        writer, _ = build_one("supplementary:Figure S16", ctx, self.root / "displays")
        self.assertTrue(writer.files)

    def test_refuses_official_synthetic_incomplete_and_altered_inputs(self):
        with self.assertRaisesRegex(RunnerError, "complete measured sets"):
            da.export_decision_audit(self.output, root=RECORDS, hardware="h", period="p")
        for name in da.SETS:
            self.assertFalse((RECORDS / name).exists())
        (self.jobs[0] / "audit.json").unlink()
        with self.assertRaisesRegex(RunnerError, "have not completed"):
            self.export()
        targets = self.export(allow_partial=True, sets=("decision_traces",))
        provenance = json.loads((targets["decision_traces"] / "provenance.json").read_text())
        self.assertFalse(provenance["coverage_complete"])
        with self.assertRaisesRegex(RunnerError, "exists"):
            self.export(allow_partial=True, sets=("decision_traces",))
        self.export(allow_partial=True, sets=("decision_traces",), replace=True)
        np.savez(self.jobs[3] / "gaps.npz", gap=np.ones(3))
        with self.assertRaisesRegex(RunnerError, "SHA-256"):
            self.export(allow_partial=True)

    def test_incomplete_coverage_needs_allow_partial(self):
        plan = json.loads((self.output / "plan.json").read_text())
        plan["datasets"] = [d for d in plan["datasets"] if d["dataset"] != "COCO-SP"]
        plan["jobs"] = [j for j in plan["jobs"] if j["dataset"] != "COCO-SP"]
        (self.output / "plan.json").write_text(json.dumps(plan))
        with self.assertRaisesRegex(RunnerError, "does not cover"):
            self.export(sets=("decision_traces",))
        target = self.export(sets=("decision_traces",), allow_partial=True)["decision_traces"]
        self.assertNotIn("COCO-SP", {r["dataset"] for r in summary(target)})


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cm-decision-cli-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def call(self, *argv):
        from scripts import run_decision_audit
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = run_decision_audit.main([str(a) for a in argv])
        return code, out.getvalue(), err.getvalue()

    def test_plan_run_and_export_commands(self):
        code, out, _ = self.call("plan", "--parent", EXAMPLES / "arxiv", "--output", self.root / "plan",
                                 "--stress-dataset", "ogbn-arxiv", "--gate-alpha", "0.1")
        self.assertEqual(code, 0, out)
        self.assertEqual(json.loads((self.root / "plan" / "plan.json").read_text())["jobs"][0]["gate"]["alpha"], 0.1)
        code, _, err = self.call("run", "--output", self.root / "plan", "--device", "cpu")
        self.assertEqual(code, 2)
        self.assertIn("explicit trust", err)
        code, _, err = self.call("plan", "--parent", EXAMPLES / "arxiv", "--output", self.root / "p2", "--gate-from", "x")
        self.assertEqual(code, 2)
        self.assertIn("--gate-from", err)
        output = self.root / "fabricated"
        plan = synthetic_plan(output)
        rng = np.random.default_rng(1)
        for entry in plan["jobs"]:
            fabricate_job(output, entry, rng)
        code, out, err = self.call("export", "--output", output, "--root", self.root / "records", "--set",
                                   "decision_traces", "--hardware", "h", "--period", "p")
        self.assertEqual(code, 0, err)
        self.assertTrue((self.root / "records" / "decision_traces" / "MANIFEST.json").is_file())
        code, _, err = self.call("export", "--output", output, "--hardware", "h", "--period", "p")
        self.assertEqual(code, 2)
        self.assertIn("complete measured sets", err)


# ============================================================================= execution (PyTorch; CI)
@unittest.skipUnless(TORCH, "PyTorch is required to execute the audit")
class AuditExecutionTests(unittest.TestCase):
    """Smoke lifecycles on synthetic data, then the instrumented execution end to end (software checks)."""

    @classmethod
    def setUpClass(cls):
        import torch
        from cross_morse.pipeline import Pipeline
        torch.set_num_threads(2)
        cls.temporary = tempfile.TemporaryDirectory(prefix="cm-decision-audit-runs-")
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
        cls.output = cls.root / "audit"
        da.plan_decision_audit([cls.runs["arxiv"], cls.runs["peptides"]], cls.output, reps=40,
                               fixture_route={"ogbn-arxiv": "radius", "Peptides-func": "mpc_rh"},
                               stress_dataset="ogbn-arxiv", audit_inputs=4, calibration_trajectories=2)
        with contextlib.redirect_stdout(io.StringIO()):
            da.run_decision_audit(cls.output, device="cpu", trust=True)
        cls.plan = json.loads((cls.output / "plan.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def job(self, dataset):
        entry = next(e for e in self.plan["jobs"] if e["dataset"] == dataset)
        return entry, job_directory(self.output, entry)

    def test_the_replayed_decision_log_and_the_sealed_actions_agree(self):
        entry, directory = self.job("ogbn-arxiv")
        parity = json.loads((directory / "parity.json").read_text())
        deployed = [p for p in parity["probabilities"] if p["rule"] == "deployed"]
        self.assertTrue(deployed)
        self.assertTrue(all(p["actions_equal"] for p in deployed))
        self.assertTrue(all(p["max_abs_probability_deviation"] < 1e-5 for p in parity["probabilities"]))
        gaps = np.load(directory / "gaps.npz")
        self.assertTrue(np.all(gaps["kind"] == 1))
        self.assertTrue(np.all(gaps["gap"] >= 0))
        trace = json.loads((self.runs["arxiv"] / "member_predictions" /
                            f"{entry['deployed']['behavior_id']}_seed1.json").read_text())
        recorded = sorted(d["score_gap"] for d in trace["decisions"] if d.get("score_gap") is not None)
        sites = np.load(directory / "sites.npz")
        mine = sorted(float(g) for g, m in zip(sites["gap"], sites["member"]) if m == 1 and np.isfinite(g))
        np.testing.assert_allclose(mine, recorded, rtol=1e-9, atol=1e-12)

    def test_planning_gaps_and_independent_graph_outputs(self):
        entry, directory = self.job("Peptides-func")
        gaps = np.load(directory / "gaps.npz")
        self.assertTrue(set(gaps["kind"].tolist()) <= {2, 3})
        self.assertEqual(set(np.load(directory / "sites.npz")["input_id"].tolist()), set(entry["audited_inputs"]))
        reliability = np.load(directory / "reliability.npz")
        self.assertEqual(str(reliability["mode"]), "own")
        self.assertEqual(sorted(reliability["unit_ids"].tolist()), sorted(entry["audited_inputs"]))
        decisions = json.loads((directory / "decisions.json").read_text())
        mass = da.DecisionMass.from_json(decisions["mass"])
        self.assertAlmostEqual(sum(mass.fractions().values()), 1.0, places=12)
        depth = json.loads((self.runs["peptides"] / "config.json").read_text())["model"]["depth"]
        self.assertLessEqual(mass.total, entry["evaluation_inputs"] * depth + 1e-9)    # unit mass per (graph, depth)
        self.assertGreater(mass.categories["exclusive_sequence"] + mass.categories["recovery"]
                           + mass.categories["input_fallback"], 0)

    def test_observer_variants_and_calibration(self):
        for dataset in ("ogbn-arxiv", "Peptides-func"):
            _, directory = self.job(dataset)
            observers = json.loads((directory / "observers.json").read_text())
            self.assertEqual(list(observers["variants"]), list(da.OBSERVER_LEVELS))
            for level in da.OBSERVER_LEVELS:
                self.assertTrue(math.isfinite(observers["variants"][level]["rmse"]))
            self.assertIsNotNone(observers["conditioned_parity"])
            calibration = json.loads((directory / "calibration.json").read_text())
            self.assertEqual(calibration["status"], "CALIBRATED")
            self.assertLessEqual(calibration["trajectories"], 2)

    def test_export_traces_the_measured_part(self):
        records = self.root / "records"
        for target in da.export_decision_audit(self.output, root=records, allow_partial=True):
            target = Path(target)
            verify_set(target, specs()["sets"][target.name], specs())
            rows = read_summary(target, specs()["sets"][target.name], specs()["summary_columns"])
            self.assertTrue(rows)

    def test_a_zero_perturbation_never_switches(self):
        output = self.root / "zero"
        da.plan_decision_audit([self.runs["arxiv"], self.runs["peptides"]], output, perturbation_factor=0.0,
                               fixture_route={"ogbn-arxiv": "radius", "Peptides-func": "mpc_rh"}, audit_inputs=3,
                               calibration_trajectories=1, observer_study=False)
        with contextlib.redirect_stdout(io.StringIO()):
            da.run_decision_audit(output, device="cpu", trust=True)
        for entry in json.loads((output / "plan.json").read_text())["jobs"]:
            sites = np.load(job_directory(output, entry) / "sites.npz")
            self.assertEqual(int(sites["switch"].sum()), 0)
            self.assertEqual(int(sites["anchor_flip"].sum()), 0)
            self.assertTrue(np.all(sites["plan_stable"][sites["plan_stable"] >= 0] == 1))
            np.testing.assert_array_equal(sites["perturbed_origin"], sites["origin"])


if __name__ == "__main__":
    unittest.main()
