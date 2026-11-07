"""Confirmation analyses (main Figure 3a-b, Supplementary Table S14b, Figures S9/S18b) and the
admission-split oracle (Supplementary Table S30): cm_experiments/confirmation_analysis.py,
cm_experiments/admission_oracle.py, scripts/run_confirmation_analysis.py.

Planning, statistics, enumeration and export run without PyTorch on the bundled synthetic example
runs and on fabricated job outputs; every exported set is written to a temporary root and traced by
scripts.records.verify_set.  Execution on smoke lifecycles needs PyTorch and runs in CI.
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
from statistics import NormalDist

import numpy as np

from cm_experiments import admission_oracle as ao
from cm_experiments import confirmation_analysis as ca
from cm_experiments.ensemble_grid import RunnerError, SealedRun, create_plan, job_directory, utc_now
from scripts.records import RECORDS, load_specs, read_summary, verify_set

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "runs"
TORCH = importlib.util.find_spec("torch") is not None
ENVIRONMENT = {"python": "3.x", "machine": "x86_64", "torch": None}
NORMAL = NormalDist()
S14_FIXTURE = "test_complete_family_reconstructs_effects_and_returns_failed_peptides_to_comparator"


def specs():
    return load_specs(RECORDS, RECORDS / "specs.json")


def summary(target):
    with (Path(target) / "summary.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def values(target):
    return {tuple(v for k, v in row.items() if k not in ("quantity", "value", "unit", "source_file", "locator"))
            + (row["quantity"],): float(row["value"]) for row in summary(target)}


class Workspace(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cm-confirmation-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.records = self.root / "records"

    def copy_example(self, name, **state):
        target = self.root / "parents" / name
        shutil.copytree(EXAMPLES / name, target)
        if state:
            path = target / "run_state.json"
            path.write_text(json.dumps({**json.loads(path.read_text()), **state}))
        return target

    def s14_fixture(self):
        from tests.test_s14_confirmation import S14CampaignTests
        fixture = S14CampaignTests(S14_FIXTURE)
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        return fixture


# ============================================================================= contracts
class ContractTests(unittest.TestCase):
    def test_exported_quantities_and_keys_are_the_registered_contracts(self):
        sets = specs()["sets"]
        self.assertEqual(sets["mechanism_contrasts"]["keys"], ["dataset", "contrast"])
        self.assertEqual(set(sets["mechanism_contrasts"]["quantities"]),
                         {"effect_nats", "interval_low_nats", "interval_high_nats"})
        self.assertEqual(sets["confirmation_contrasts"]["keys"], ["contrast"])
        self.assertEqual(set(sets["confirmation_contrasts"]["quantities"]),
                         {"effect_nats", "interval_low_nats", "interval_high_nats", "p_value_zero", "unit_count"})
        self.assertEqual(set(sets["confirmation_sensitivity"]["quantities"]),
                         {"effect_nats", "ci_low_nats", "ci_high_nats", "units_old", "graph_variance_share", "holm_p",
                          "label_survival_probability"})
        self.assertEqual(set(sets["oracle_dispersion"]["quantities"]), set(ao.QUANTITY_UNITS))

    def test_key_values_match_the_display_generators(self):
        from paper.build.displays import main_paper
        from paper.build.displays.framework import DATASETS, SHORT
        self.assertEqual(set(ca.FIGURE3), set(main_paper.FAMILY_KINDS.values()))
        self.assertEqual({v: k for k, v in ca.FIGURE3.items()}, main_paper.FAMILY_KINDS)
        self.assertEqual(ca.DATASETS, DATASETS)
        self.assertEqual(ca.multigraph_contrasts(),
                         {"PascalVOC-SP": "Pascal/RH", "COCO-SP": "COCO/RH", "Peptides-func": "Peptides/radius"})
        # Table S43 maps a contrast to its dataset by the short name before the slash.
        for dataset, key in ca.multigraph_contrasts().items():
            self.assertEqual(SHORT[dataset], key.split("/")[0])

    def test_coverage_is_registered_only_for_the_export(self):
        from cm_campaigns.contract import RECORD_SETS
        self.assertNotIn("oracle_dispersion_probe", RECORD_SETS)
        with ca.registered_coverage("oracle_dispersion_probe", {"datasets": ["x"]}) as contract:
            self.assertIs(RECORD_SETS["oracle_dispersion_probe"], contract)
        self.assertNotIn("oracle_dispersion_probe", RECORD_SETS)
        with ca.registered_coverage("inserted_site", {"datasets": []}) as contract:
            self.assertEqual(contract["analyzer"], "inserted_site")      # a registered entry governs


# ============================================================================= statistics
class TwoWayTests(unittest.TestCase):
    def test_additive_array_recovers_the_components_exactly(self):
        rng = np.random.default_rng(3)
        u, v = rng.normal(0, 0.01, 40), rng.normal(0, 0.003, 6)
        fit = ca.two_way_fit(-0.005 + u[:, None] + v[None, :])
        self.assertAlmostEqual(fit["sigma2_graph"], float(np.var(u, ddof=1)), places=14)
        self.assertAlmostEqual(fit["sigma2_seed"], float(np.var(v, ddof=1)), places=14)
        self.assertAlmostEqual(fit["sigma2_residual"], 0.0, places=14)
        self.assertAlmostEqual(fit["mean"], -0.005 + u.mean() + v.mean(), places=14)
        self.assertEqual((fit["df_graph"], fit["df_seed"], fit["df_residual"]), (39, 5, 195))

    def test_unbiased_variance_of_the_mean_and_truncation(self):
        rng = np.random.default_rng(5)
        d = rng.normal(0, 1e-3, (25, 4)) + rng.normal(0, 2e-3, 4)[None, :]
        fit = ca.two_way_fit(d)
        j, s = d.shape
        self.assertAlmostEqual(fit["variance_of_mean_unbiased"],
                               (fit["ms_graph"] + fit["ms_seed"] - fit["ms_residual"]) / (j * s), places=18)
        if fit["sigma2_graph_untruncated"] >= 0 and fit["sigma2_seed_untruncated"] >= 0:
            self.assertAlmostEqual(ca.variance_of_mean(fit, j, s), fit["variance_of_mean_unbiased"], places=18)
        noise = ca.two_way_fit(rng.normal(0, 1e-3, (30, 5)))      # no graph or seed effect: small/truncated
        self.assertGreaterEqual(noise["sigma2_graph"], 0.0)
        self.assertGreaterEqual(noise["sigma2_seed"], 0.0)
        for bad in (np.zeros((1, 3)), np.zeros((3, 1)), np.array([[0.0, np.nan], [1.0, 2.0]])):
            with self.assertRaises(RunnerError):
                ca.two_way_fit(bad)

    def test_rescaling_reproduces_the_model_variance_and_the_generators_formula(self):
        fit = {"graphs": 1428, "seeds": 12, "mean": -0.009, "sigma2_graph": 2.0e-4, "sigma2_seed": 6.0e-5,
               "sigma2_residual": 3.0e-5, "ms_graph": 0.0, "ms_seed": 0.0, "ms_residual": 0.0,
               "df_graph": 1427, "df_seed": 11, "df_residual": 1427 * 11}
        out = ca.sensitivity(fit, 425)
        v_old = 2.0e-4 / 1428 + 6.0e-5 / 12 + 3.0e-5 / (1428 * 12)
        v_new = 2.0e-4 / 425 + 6.0e-5 / 12 + 3.0e-5 / (425 * 12)
        g = (2.0e-4 / 1428 + 3.0e-5 / (1428 * 12)) / v_old
        self.assertAlmostEqual(out["se_old"], math.sqrt(v_old), places=15)
        self.assertAlmostEqual(out["se_new"], math.sqrt(v_new), places=15)
        self.assertAlmostEqual(out["graph_variance_share"], g, places=15)
        # Figures S9/S18b and Table S14b: each side widens by [(1 - g) + g J_old/J_new]^(1/2) about the fixed effect.
        low, high = out["interval_low_nats"], out["interval_high_nats"]
        factor = math.sqrt(g * 1428 / 425 + 1.0 - g)
        self.assertAlmostEqual(-0.009 - (-0.009 - low) * factor, out["interval_new_low_nats"], places=15)
        self.assertAlmostEqual(-0.009 + (high + 0.009) * factor, out["interval_new_high_nats"], places=15)
        se_from_interval = (high - low) / (2 * NORMAL.inv_cdf(0.975))     # analytical.table_s14b
        self.assertAlmostEqual(se_from_interval, out["se_old"], places=15)
        self.assertAlmostEqual(2 * (1 - NORMAL.cdf(0.009 / (se_from_interval * factor))), out["p_value_zero_new"],
                               places=14)

    def test_holm_family_ranks_and_label_survival(self):
        keys = list(ca._family())
        p = dict(zip(keys, (0.0014, 0.0030, 0.0016, 0.0034, 0.42, 0.0047, 0.00016, 0.000004, 0.035)))
        holm = ca.holm_family(p, 0.05)
        from cross_morse.statistics import holm_adjust
        expected = holm_adjust(np.asarray([p[k] for k in keys]))
        for key, value in zip(keys, expected):
            self.assertAlmostEqual(holm[key]["holm_p"], float(value), places=15)
        self.assertEqual(holm["COCO/RH"]["rank"], 1)
        self.assertEqual(holm["COCO/RH"]["multiplier"], 9)
        self.assertEqual(holm["Peptides/radius"]["multiplier"], 2)
        self.assertAlmostEqual(holm["Peptides/radius"]["holm_p"], 0.07, places=12)
        with self.assertRaises(RunnerError):
            ca.holm_family({k: p[k] for k in keys[:8]}, 0.05)
        # Survival: P(new estimate Holm-significant at its step, beneficial sign, beyond the margin).
        se, effect = 0.0008, -0.0017
        out = ca.label_survival(effect, se, -1, 0.0015, 2, 0.05)
        critical = NORMAL.inv_cdf(1 - 0.05 / 4)
        self.assertAlmostEqual(out["probability"], NORMAL.cdf((0.0017 - critical * se) / se), places=14)
        self.assertEqual(out["binding"], "significance")
        wide_margin = ca.label_survival(effect, se, -1, 0.0019, 2, 0.05)
        self.assertEqual(wide_margin["binding"], "margin")
        self.assertAlmostEqual(wide_margin["probability"], NORMAL.cdf((0.0017 - 0.0019) / se), places=14)
        self.assertLess(ca.label_survival(+0.004, 0.001, -1, 0.002, 9, 0.05)["probability"], 1e-6)

    def test_t_interval_over_splits(self):
        from scipy.stats import t as student
        x = [-0.006, -0.004, -0.0075, -0.005]
        out = ca.t_interval(x)
        se = float(np.std(x, ddof=1)) / 2
        self.assertAlmostEqual(out["lower"], np.mean(x) - student.ppf(0.975, 3) * se, places=15)
        with self.assertRaises(RunnerError):
            ca.t_interval([0.1])


# ============================================================================= mechanism_contrasts
def fake_score(loss, counts=None):
    loss = np.asarray(loss, dtype=np.float64)
    return {"loss": loss, "counts": np.ones(len(loss), dtype=np.int64) if counts is None else np.asarray(counts),
            "core_ids": np.arange(len(loss))}


VALID = {"feasible": True, "state_control_valid": True, "shuffle_matched_sites": 6, "shuffle_missing_sites": 0}


class MechanismTests(Workspace):
    def plan(self, parents=("arxiv", "peptides_func"), **kwargs):
        path = ca.plan_mechanism([EXAMPLES / p for p in parents], self.root / "mech", reps=40, **kwargs)
        return json.loads(path.read_text())

    def test_figure3_rule_selects_the_cells_neither_s14_nor_the_deployed_rule_supply(self):
        arxiv, peptides = SealedRun(EXAMPLES / "arxiv"), SealedRun(EXAMPLES / "peptides_func")
        self.assertTrue(ca.s14_supplies("ogbn-arxiv", "state"))
        self.assertFalse(ca.s14_supplies("Peptides-func", "state"))
        self.assertFalse(ca.run_supplies([arxiv], "mpc"))                   # q_C returned: nothing supplied
        self.assertEqual(ca.required_cells([arxiv], "ogbn-arxiv"), ["rh_minus_ol"])
        self.assertEqual(ca.required_cells([peptides], "Peptides-func"), ["state", "rh_minus_ol"])

    def test_strongest_feedback_configuration_and_replanning_pair(self):
        run = SealedRun(EXAMPLES / "arxiv")
        candidates, selection = run.candidates(), run.json("select/decision.json")
        anchor, rule = ca.strongest_feedback(run)
        feedback = [i for i, c in enumerate(candidates) if c["route"] in ("one_step", "radius", "mpc_rh")]
        self.assertEqual(selection["upper"][anchor["registry_index"]], min(selection["upper"][i] for i in feedback))
        self.assertIn("lowest D_select upper limit", rule)
        rh, ol, _ = ca.replanning_pair(run, anchor)
        self.assertEqual((rh["route"], ol["route"]), ("mpc_rh", "mpc_ol"))
        for key in ("ensemble", "profile", "horizon"):
            self.assertEqual(rh[key], ol[key])
        self.assertEqual(rh["ensemble"], anchor["ensemble"])
        pool = [i for i, c in enumerate(candidates) if c["route"] == "mpc_rh" and c["ensemble"] == anchor["ensemble"]
                and c["profile"] == anchor["profile"]]
        self.assertEqual(selection["upper"][rh["registry_index"]], min(selection["upper"][i] for i in pool))
        declared, how = ca.strongest_feedback(run, 34)
        self.assertEqual((declared["registry_index"], how), (34, "declared in the plan (--anchor)"))
        with self.assertRaisesRegex(RunnerError, "not a registered feedback"):
            ca.strongest_feedback(run, 9)
        same, _, rule = ca.replanning_pair(run, dict(candidates[36], registry_index=36))
        self.assertEqual((same["registry_index"], rule), (36, "the anchor itself"))

    def test_plan_declares_policies_seeds_and_refuses_incomplete_parents(self):
        plan = self.plan()
        jobs = {j["dataset"]: j for j in plan["jobs"]}
        self.assertEqual([c["contrast"] for c in jobs["Peptides-func"]["cells"]], ["state", "rh_minus_ol"])
        state = jobs["Peptides-func"]["cells"][0]
        self.assertEqual(state["reference"]["state_control"], "matched_shuffle")
        self.assertEqual(state["reference_key"], "shuffle_" + state["candidate_key"])
        policies = {p["key"]: p for p in jobs["Peptides-func"]["policies"]}
        self.assertTrue(policies[state["reference_key"]]["uses_donors"])
        self.assertEqual(policies[state["candidate_key"]]["seeds"], [1, 2][:state["candidate"]["ensemble"]])
        self.assertEqual(jobs["ogbn-arxiv"]["bootstrap"], {"reps": 40, "seed": SealedRun(EXAMPLES / "arxiv").config["seed"] + 5})
        self.assertEqual({d["dataset"]: d["unit"] for d in plan["datasets"]},
                         {"ogbn-arxiv": "buffered cluster", "Peptides-func": "graph"})
        explicit = ca.plan_mechanism([EXAMPLES / "arxiv"], self.root / "explicit", cells={"ogbn-arxiv": ["state"]},
                                     anchors={"ogbn-arxiv": 34})
        explicit = json.loads(explicit.read_text())
        self.assertEqual(explicit["datasets"][0]["cell_rule"], "declared in the plan (--cell)")
        self.assertEqual(explicit["jobs"][0]["cells"][0]["candidate"]["registry_index"], 34)
        running = self.copy_example("arxiv", status="RUNNING")
        with self.assertRaisesRegex(RunnerError, "not complete"):
            ca.plan_mechanism([running], self.root / "x")
        with self.assertRaisesRegex(RunnerError, "one parent"):
            ca.plan_mechanism([EXAMPLES / "arxiv", EXAMPLES / "arxiv"], self.root / "y")
        with self.assertRaisesRegex(RunnerError, "outside the repository"):
            ca.plan_mechanism([EXAMPLES / "arxiv"], ROOT / "mechanism-output")

    def test_statistics_follow_the_confirm_stage_rules(self):
        from cross_morse.statistics import confirmation
        plan = self.plan(parents=("peptides_func",))
        entry = plan["jobs"][0]
        state, mpc = entry["cells"]
        rng = np.random.default_rng(1)
        intact, destroyed = rng.uniform(0.6, 0.7, 12), rng.uniform(0.6, 0.7, 12)
        rh, ol = rng.uniform(0.5, 0.6, 12), rng.uniform(0.5, 0.6, 12)
        scores = {state["candidate_key"]: fake_score(intact), state["reference_key"]: fake_score(destroyed),
                  mpc["candidate_key"]: fake_score(rh), mpc["reference_key"]: fake_score(ol)}
        executions = {key: dict(VALID) for key in scores}
        cells = ca.mechanism_statistics(entry["cells"], scores, executions, entry["selection"], collection=True,
                                        reps=40, seed=9)
        direct = confirmation(entry["selection"]["provisional_index"], entry["selection"]["comparator_index"],
                              entry["selection"]["eligible_indices"],
                              [{"kind": "state", "candidate_index": state["candidate"]["registry_index"],
                                "predecessor_index": entry["selection"]["comparator_index"],
                                "differences": destroyed - intact, "weights": np.ones(12)},
                               {"kind": "mpc", "candidate_index": mpc["candidate"]["registry_index"],
                                "predecessor_index": mpc["reference"]["registry_index"],
                                "differences": rh - ol, "weights": np.ones(12)}], reps=40, seed=9)
        for cell, row in zip(cells, direct["contrasts"]):
            self.assertTrue(cell["valid"])
            self.assertEqual((cell["effect_nats"], cell["interval_low_nats"], cell["interval_high_nats"]),
                             (row["mean"], row["lower"], row["upper"]))
            self.assertEqual(cell["units"], 12)
        missing = dict(executions, **{state["reference_key"]: dict(VALID, shuffle_missing_sites=2)})
        cells = ca.mechanism_statistics(entry["cells"], scores, missing, entry["selection"], collection=True,
                                        reps=40, seed=9)
        self.assertFalse(cells[0]["valid"])
        self.assertIn("donors", cells[0]["invalid_reason"])
        self.assertIsNone(cells[0]["effect_nats"])
        failed = dict(executions, **{mpc["reference_key"]: {"execution_failure": True, "feasible": False}})
        cells = ca.mechanism_statistics(entry["cells"], scores, failed, entry["selection"], collection=True,
                                        reps=40, seed=9)
        self.assertTrue(cells[0]["valid"])
        self.assertFalse(cells[1]["valid"])
        self.assertIn("simpler policy", cells[1]["invalid_reason"])
        # A transductive graph keeps the runtime's default feasibility of the intact policy.
        infeasible = dict(executions, **{state["candidate_key"]: dict(VALID, feasible=False)})
        self.assertTrue(ca.mechanism_statistics(entry["cells"], scores, infeasible, entry["selection"],
                                                collection=False, reps=40, seed=9)[0]["valid"])
        self.assertFalse(ca.mechanism_statistics(entry["cells"], scores, infeasible, entry["selection"],
                                                 collection=True, reps=40, seed=9)[0]["valid"])
        shifted = dict(scores, **{mpc["reference_key"]: dict(fake_score(ol), core_ids=np.arange(1, 13))})
        with self.assertRaisesRegex(RunnerError, "different units"):
            ca.mechanism_statistics(entry["cells"], shifted, executions, entry["selection"], collection=True,
                                    reps=40, seed=9)

    def fabricate(self, plan, *, invalid=()):
        output = self.root / "mech"
        rng = np.random.default_rng(4)
        for entry in plan["jobs"]:
            scores = {}
            for policy in entry["policies"]:
                scores[policy["key"]] = fake_score(rng.uniform(0.4, 0.8, 10), rng.integers(3, 9, 10))
            for policy in entry["policies"]:
                scores[policy["key"]]["counts"] = scores[entry["policies"][0]["key"]]["counts"]
            executions = {key: dict(VALID) for key in scores}
            for dataset, contrast in invalid:
                if dataset == entry["dataset"]:
                    cell = next(c for c in entry["cells"] if c["contrast"] == contrast)
                    executions[cell["reference_key"]] = {"execution_failure": True, "feasible": False}
            cells = ca.mechanism_statistics(entry["cells"], scores, executions, entry["selection"],
                                            collection=entry["dataset"] == "Peptides-func",
                                            reps=entry["bootstrap"]["reps"], seed=entry["bootstrap"]["seed"])
            ca.write_mechanism_result(job_directory(output, entry), entry, cells=cells, policies={},
                                      executions=executions,
                                      arrays={"unit_id": np.arange(10), **{f"{k}_unit_loss": s["loss"]
                                                                           for k, s in scores.items()}},
                                      unit_kind="test", parent_files=[], started=utc_now(), environment=ENVIRONMENT)
        return output

    def test_export_traces_every_value(self):
        plan = self.plan()
        output = self.root / "mech"
        with self.assertRaisesRegex(RunnerError, "has not completed"):
            ca.export_mechanism(output, root=self.records, allow_partial=True)
        self.fabricate(plan)
        with self.assertRaisesRegex(RunnerError, "complete measured sets|all seven"):
            ca.export_mechanism(output, root=RECORDS)
        self.assertFalse((RECORDS / "mechanism_contrasts").exists())
        target = ca.export_mechanism(output, root=self.records, command=["scripts/run_confirmation_analysis.py"])
        spec = specs()["sets"]["mechanism_contrasts"]
        entry = verify_set(target, spec, specs())
        self.assertTrue(entry["synthetic"])
        got = values(target)
        self.assertEqual({(k[0], k[1]) for k in got},
                         {("ogbn-arxiv", "rh_minus_ol"), ("Peptides-func", "state"), ("Peptides-func", "rh_minus_ol")})
        for dataset in plan["datasets"]:
            job = next(j for j in plan["jobs"] if j["dataset"] == dataset["dataset"])
            result = json.loads((job_directory(output, job) / "mechanism.json").read_text())
            for cell in result["cells"]:
                for quantity in ("effect_nats", "interval_low_nats", "interval_high_nats"):
                    self.assertAlmostEqual(got[(dataset["dataset"], cell["contrast"], quantity)], cell[quantity],
                                           places=9)
                self.assertLessEqual(cell["interval_low_nats"], cell["effect_nats"])
                self.assertLessEqual(cell["effect_nats"], cell["interval_high_nats"])
        provenance = json.loads((target / "provenance.json").read_text())
        self.assertTrue(provenance["coverage_complete"])
        self.assertTrue((target / "code" / "cm_experiments" / "confirmation_analysis.py").is_file())
        self.assertNotIn("/", json.loads((target / "code" / "plan.json").read_text())["jobs"][0]["run"])

    def test_invalid_cells_are_never_filled(self):
        plan = self.plan()
        output = self.fabricate(plan, invalid=[("Peptides-func", "rh_minus_ol")])
        with self.assertRaisesRegex(RunnerError, "without a valid measurement"):
            ca.export_mechanism(output, root=self.records)
        target = ca.export_mechanism(output, root=self.records, allow_partial=True)
        verify_set(target, specs()["sets"]["mechanism_contrasts"], specs())
        self.assertNotIn(("Peptides-func", "rh_minus_ol", "effect_nats"), values(target))
        self.assertFalse(json.loads((target / "provenance.json").read_text())["coverage_complete"])

    def test_split_campaign_aggregates_with_a_t_interval(self):
        results = [(f"split_{i}", {"cells": [{"contrast": "state", "valid": True, "effect_nats": e,
                                              "interval_low_nats": e - 0.001, "interval_high_nats": e + 0.001,
                                              "units": 32, "invalid_reason": None, "candidate": {}, "reference": {}}]})
                   for i, e in enumerate((-0.004, -0.006, -0.005))]
        cell = ca.aggregate_cells("Roman-empire", results, ["state"])[0]
        expected = ca.t_interval([-0.004, -0.006, -0.005])
        self.assertEqual((cell["effect_nats"], cell["interval_low_nats"], cell["units"]),
                         (expected["mean"], expected["lower"], 3))
        results[1][1]["cells"][0]["valid"] = False
        results[1][1]["cells"][0]["invalid_reason"] = "probe"
        self.assertFalse(ca.aggregate_cells("Roman-empire", results, ["state"])[0]["valid"])


# ============================================================================= confirmation_contrasts / confirmation_sensitivity
SIZES = {"PascalVOC-SP": 30, "COCO-SP": 36, "Peptides-func": 24}
# Fixture admission variances S^2 and Table 1 test intervals (synthetic values; graph shares well inside (0, 1)).
PARENTS = {"PascalVOC-SP": {"s2": 2.0e-4, "test": (-0.020, -0.004)},
           "COCO-SP": {"s2": 3.0e-4, "test": (-0.024, -0.002)},
           "Peptides-func": {"s2": 1.0e-4, "test": (-0.012, 0.002)}}


def analytical_share(dataset, root):
    """paper.build.displays.analytical.graph_variance_share on a sealed run (the Table S14b generator's g)."""
    from paper.build.displays import analytical
    from paper.build.displays.framework import RunView

    class Context:
        def require_run(self, name):
            return RunView(name, root)
    return analytical.graph_variance_share(Context(), dataset)


def s14b_family(contrasts, tests, shares, j_old):
    """Table S14b's arithmetic (analytical.table_s14b): J_new rows, raw P and the re-formed Holm family."""
    from cm_theory.inference import holm_adjust
    family, z = ca._family(), NORMAL.inv_cdf(0.975)
    raw, se_new = {}, {}
    for key, spec in family.items():
        effect, low, high = (contrasts[(key, q)] for q in ("effect_nats", "interval_low_nats", "interval_high_nats"))
        if spec[2] != "graph":
            raw[key] = contrasts[(key, "p_value_zero")]
            continue
        dataset = spec[0]
        g = shares[dataset]
        se = (high - low) / (2 * z) * math.sqrt((1 - g) + g * j_old[dataset] / spec[3])
        se_new[key] = se
        raw[key] = 2 * (1 - NORMAL.cdf(abs(effect) / se))
    holm = dict(zip(family, holm_adjust([raw[k] for k in family])))
    return raw, holm, se_new


class ConfirmationTests(Workspace):
    def test_plan_identifies_the_pair_and_refuses_unidentified_ones(self):
        path = ca.plan_confirmation([EXAMPLES / "peptides_func"], self.root / "conf", allow_fixture=True, reps=40)
        plan = json.loads(path.read_text())
        dataset = plan["datasets"][0]
        self.assertEqual((dataset["contrast"], dataset["J_new"], dataset["practical_margin_nats"]),
                         ("Peptides/radius", 544, 0.0015))
        self.assertEqual((dataset["left"]["route"], dataset["right"]["route"]), ("radius", "one_step"))
        self.assertEqual([j["seed"] for j in plan["jobs"]], [1, 2])
        self.assertEqual(plan["record_sets"], ["confirmation_contrasts", "confirmation_sensitivity"])
        self.assertTrue(dataset["pair_source"].startswith("fixture"))
        with self.assertRaisesRegex(RunnerError, "--campaign"):
            ca.plan_confirmation([EXAMPLES / "peptides_func"], self.root / "x")
        with self.assertRaisesRegex(RunnerError, "J_old contrasts are defined"):
            ca.plan_confirmation([EXAMPLES / "arxiv"], self.root / "y", allow_fixture=True)
        with self.assertRaisesRegex(RunnerError, "two-way model needs"):
            ca.plan_confirmation([EXAMPLES / "peptides_func"], self.root / "z", allow_fixture=True, seeds=[1])

    def parent(self, dataset, *, s2, test):
        """A copy of the multigraph example whose admission moments and Table 1 test interval are fixture values."""
        target = self.root / "parents" / dataset
        shutil.copytree(EXAMPLES / "peptides_func", target)
        admit = json.loads((target / "admit" / "decision.json").read_text())
        admit["empirical_bernstein"].update(units=SIZES[dataset], sample_variance=s2, mean=-0.006)
        (target / "admit" / "decision.json").write_text(json.dumps(admit))
        final = json.loads((target / "final" / "results_summary.json").read_text())
        final["paired_qD_minus_qC_loss"].update(mean=(test[0] + test[1]) / 2, lower=test[0], upper=test[1])
        (target / "final" / "results_summary.json").write_text(json.dumps(final))
        return SealedRun(target)

    def fabricate(self, datasets=tuple(SIZES)):
        """A synthetic plan over the multigraph benchmarks and per-seed outputs from a known two-way model."""
        left, right, source = ca.policy_pair(SealedRun(EXAMPLES / "peptides_func"), "Peptides/radius",
                                             allow_fixture=True)
        seeds = [1, 2, 3, 4, 5]
        rng = np.random.default_rng(17)
        family, entries, jobs = ca._family(), [], []
        for dataset in datasets:
            key = ca.multigraph_contrasts()[dataset]
            run = self.parent(dataset, **PARENTS[dataset])
            _, kind, _, j_new, _, direction, margin = family[key]
            entries.append({"dataset": dataset, "slug": dataset, "contrast": key, "kind": kind, "synthetic": True,
                            "member": run.root.name, "parent": str(run.root), "seeds": seeds,
                            "seed_source": "test", "J_new": j_new, "beneficial_direction": direction,
                            "practical_margin_nats": margin, "left": ca.candidate_identity(left),
                            "right": ca.candidate_identity(right), "pair_source": source,
                            "campaign_binding": "not bound (synthetic software fixture)",
                            "bootstrap": {"reps": 40, "seed": 11}})
            for seed in seeds:
                jobs.append({"job": len(jobs), "dataset": dataset, "slug": dataset, "contrast": key,
                             "member": f"{run.root.name}__seed{seed}", "run": str(run.root),
                             "identity": run.identity(), "seed": seed, "role": "admit", "left": left, "right": right})
        output = self.root / "conf"
        create_plan(output, {"experiment": "E10", "record_set": "confirmation_contrasts",
                             "record_sets": ["confirmation_contrasts", "confirmation_sensitivity"], "role": "admit",
                             "datasets": entries, "jobs": jobs, "campaign": None, "alpha": 0.05})
        plan = json.loads((output / "plan.json").read_text())
        truth = {}
        for entry in entries:
            n = SIZES[entry["dataset"]]
            graph, seed_effect = rng.normal(0, 0.01, n), rng.normal(0, 0.004, len(seeds))
            base = rng.uniform(0.5, 0.9, n)
            truth[entry["dataset"]] = []
            for s, job in enumerate(j for j in plan["jobs"] if j["dataset"] == entry["dataset"]):
                d = -0.006 + graph + seed_effect[s] + rng.normal(0, 0.002, n)
                truth[entry["dataset"]].append(d)
                ca.write_pair_result(job_directory(output, job), job, unit_id=np.arange(100, 100 + n),
                                     left_loss=base + d, right_loss=base, executions={}, reproduction={},
                                     parent_files=[], started=utc_now(), environment=ENVIRONMENT)
        return output, plan, {k: np.stack(v, axis=1) for k, v in truth.items()}

    def test_one_analysis_feeds_both_sets(self):
        output, plan, truth = self.fabricate()
        fixture = self.s14_fixture()
        with self.assertRaisesRegex(RunnerError, "campaign"):
            ca.export_confirmation(output, root=self.records)
        written = ca.export_confirmation(output, campaign=fixture.root, manuscript_root=fixture.manuscript,
                                         root=self.records, command=["scripts/run_confirmation_analysis.py"])
        sets = specs()["sets"]
        for name, target in written.items():
            entry = verify_set(target, sets[name], specs())
            self.assertTrue(entry["synthetic"])
        contrasts, sensitivity = values(written["confirmation_contrasts"]), values(written["confirmation_sensitivity"])
        analysis = json.loads((written["confirmation_sensitivity"] / "raw" / "confirmation_analysis.json").read_text())
        self.assertEqual(analysis, json.loads((written["confirmation_contrasts"] / "raw" /
                                               "confirmation_analysis.json").read_text()))
        family = ca._family()
        self.assertEqual({k[0] for k in contrasts}, set(family))
        # J_old rows: all nine contrasts carry the campaign's original tests, unchanged (the Table S14 values).
        for key, spec in family.items():
            test = fixture.results[key]
            self.assertEqual(contrasts[(key, "effect_nats")], test["effect_nats"])
            self.assertEqual((contrasts[(key, "interval_low_nats")], contrasts[(key, "interval_high_nats")]),
                             tuple(test["interval_nats"]))
            self.assertEqual(contrasts[(key, "p_value_zero")], test["p_value_zero"])
            # J_old: a single-graph row keeps its unit; a multigraph row is the validation split of its run.
            self.assertEqual(contrasts[(key, "unit_count")], spec[3] if spec[2] != "graph" else SIZES[spec[0]])
        shares, j_old = {}, {}
        for dataset, key in ca.multigraph_contrasts().items():
            # Figures S9/S18b read the same J_old row as Table S14b.
            for q2, q3 in (("effect_nats", "effect_nats"), ("interval_low_nats", "ci_low_nats"),
                           ("interval_high_nats", "ci_high_nats"), ("unit_count", "units_old")):
                self.assertEqual(contrasts[(key, q2)], sensitivity[(key, q3)])
            # g is the Table S14b generator's own graph share of the same sealed run.
            parent = self.root / "parents" / dataset
            expected = analytical_share(dataset, parent)
            row = next(c for c in analysis["contrasts"] if c["contrast"] == key)
            self.assertEqual(row["graph_variance_share"], expected["share"])            # raw: the identical float
            self.assertAlmostEqual(sensitivity[(key, "graph_variance_share")], expected["share"], places=9)
            hw_graph = NORMAL.inv_cdf(0.975) * math.sqrt(PARENTS[dataset]["s2"] / SIZES[dataset])
            hw_test = (PARENTS[dataset]["test"][1] - PARENTS[dataset]["test"][0]) / 2
            self.assertAlmostEqual(expected["share"], (hw_graph / hw_test) ** 2, places=14)
            shares[dataset], j_old[dataset] = expected["share"], SIZES[dataset]
        raw, holm, se_new = s14b_family(contrasts, fixture.results, shares, j_old)
        for dataset, key in ca.multigraph_contrasts().items():
            self.assertAlmostEqual(sensitivity[(key, "holm_p")], holm[key], places=9)
            row = next(c for c in analysis["contrasts"] if c["contrast"] == key)
            self.assertAlmostEqual(row["p_value_zero_new"], raw[key], places=12)
            self.assertAlmostEqual(row["se_new"], se_new[key], places=15)
            multiplier = ca.holm_family(raw, 0.05)[key]["multiplier"]
            survival = ca.label_survival(contrasts[(key, "effect_nats")], se_new[key], family[key][5], family[key][6],
                                         multiplier, 0.05)["probability"]
            self.assertAlmostEqual(sensitivity[(key, "label_survival_probability")], survival, places=9)
            # Figure S9/S18b rescale each side about the effect; Table S14b uses the symmetric interval.
            factor = math.sqrt((1 - shares[dataset]) + shares[dataset] * j_old[dataset] / family[key][3])
            self.assertAlmostEqual(row["interval_new_sidewise_nats"][0], contrasts[(key, "effect_nats")]
                                   - (contrasts[(key, "effect_nats")] - contrasts[(key, "interval_low_nats")]) * factor,
                                   places=15)
        # The E16 direct estimate is deposited, labelled, and used by no exported value.
        e16 = {e["contrast"]: e for e in analysis["e16_direct_estimate"]}
        for dataset, key in ca.multigraph_contrasts().items():
            fit = ca.two_way_fit(truth[dataset])
            self.assertAlmostEqual(e16[key]["fit"]["sigma2_graph"], fit["sigma2_graph"], places=15)
            self.assertAlmostEqual(e16[key]["own_estimate"]["effect_nats"], float(truth[dataset].mean()), places=12)
            self.assertIn("not used", e16[key]["label"])
            self.assertIn("campaign_row_rescaled_with_own_share", e16[key])
            self.assertNotEqual(e16[key]["own_share"], sensitivity[(key, "graph_variance_share")])
        for name in ("PascalVOC-SP/paired_per_graph.npz", "PascalVOC-SP/parent/admit_decision.json",
                     "PascalVOC-SP/parent/results_summary.json", "campaign/Pascal_RH/test.json",
                     "campaign/arxiv_state/test.json"):
            self.assertTrue((written["confirmation_sensitivity"] / "raw" / name).is_file(), name)
        self.assertEqual(analysis["campaign"]["agreement"]["Pascal/RH"]["E16_policy_pair_is_the_S14_test_pair"], False)
        self.assertEqual(analysis["refused"], [])

    def test_an_incomplete_family_is_partial_and_never_reaches_records(self):
        output, _, _ = self.fabricate(datasets=("Peptides-func",))
        fixture = self.s14_fixture()
        with self.assertRaisesRegex(RunnerError, "incomplete contrasts"):
            ca.export_confirmation(output, campaign=fixture.root, manuscript_root=fixture.manuscript, root=self.records)
        self.assertFalse((self.records / "confirmation_contrasts").exists())
        with self.assertRaisesRegex(RunnerError, "complete measured sets"):
            ca.export_confirmation(output, campaign=fixture.root, manuscript_root=fixture.manuscript, root=RECORDS,
                                   allow_partial=True)
        self.assertFalse((RECORDS / "confirmation_contrasts").exists())
        written = ca.export_confirmation(output, campaign=fixture.root, manuscript_root=fixture.manuscript,
                                         root=self.records, allow_partial=True)
        sensitivity, contrasts = values(written["confirmation_sensitivity"]), values(written["confirmation_contrasts"])
        self.assertNotIn(("Peptides/radius", "holm_p"), sensitivity)
        self.assertIn(("Peptides/radius", "graph_variance_share"), sensitivity)
        self.assertEqual(len({k[0] for k in contrasts}), 9)          # the campaign's nine original tests
        self.assertNotIn(("Pascal/RH", "unit_count"), contrasts)       # J_old needs the Pascal parent run
        self.assertIn(("Peptides/radius", "unit_count"), contrasts)

    def test_a_row_the_campaign_lacks_is_refused_not_substituted(self):
        output, plan, _ = self.fabricate()
        fixture = self.s14_fixture()
        tests = ca.campaign_tests(ca.campaign_family(fixture.root, fixture.manuscript))
        del tests["COCO/RH"]
        analysis, _ = ca.analyze_confirmation(output, plan, tests, alpha=0.05)
        self.assertEqual([r["contrast"] for r in analysis["refused"]], ["COCO/RH"])
        self.assertNotIn("COCO/RH", [c["contrast"] for c in analysis["contrasts"]])
        self.assertNotIn("COCO/RH", [r["contrast"] for r in analysis["family"]])
        self.assertFalse(analysis["family_complete"])
        self.assertTrue(all(c["holm_p"] is None for c in analysis["contrasts"]))
        self.assertIn("COCO/RH", [e["contrast"] for e in analysis["e16_direct_estimate"]])

    def test_graph_share_needs_a_proper_variance_share(self):
        run = self.parent("PascalVOC-SP", s2=2.0e-4, test=(-0.008, -0.008))
        with self.assertRaisesRegex(RunnerError, "zero-width"):
            ca.deployment_share(run)
        shutil.rmtree(run.root)
        run = self.parent("PascalVOC-SP", s2=2.0e-2, test=(-0.009, -0.007))
        with self.assertRaisesRegex(RunnerError, "not a variance share"):
            ca.deployment_share(run)

    def test_ensemble_level_campaign_uses_the_parents_registered_training_seeds(self):
        key, run = "Peptides/radius", SealedRun(EXAMPLES / "peptides_func")
        built = {"production_kind": "repository_built", "rows": {key: {"seed_ids": ["ensemble"]}}}
        self.assertTrue(ca.ensemble_level(built, key))
        self.assertIsNone(ca.campaign_seeds(built, key))
        seeds, source = ca.e16_seed_population(built, key, run)
        self.assertEqual(seeds, sorted(int(s) for s in run.config["training"]["seeds"]))
        self.assertIn("registered training seeds", source)
        # Per-seed campaigns keep their own population; an imported campaign has no ensemble level.
        per_seed = {"production_kind": "repository_built", "rows": {key: {"seed_ids": ["2", "1"]}}}
        self.assertFalse(ca.ensemble_level(per_seed, key))
        self.assertEqual(ca.e16_seed_population(per_seed, key, run), ([1, 2], "the S14 test's seed population"))
        imported = {"production_kind": "imported_original", "rows": {key: {"seed_ids": ["ensemble"]}}}
        with self.assertRaisesRegex(RunnerError, "not checkpoint seeds"):
            ca.campaign_seeds(imported, key)
        # Planning: the E16 array of an ensemble-level campaign runs over the parent's registered seeds.
        from unittest import mock
        left, right, _ = ca.policy_pair(run, key, allow_fixture=True)
        with mock.patch.object(ca, "campaign_family", return_value={**built, "root": self.root, "status": "PASS",
                                                                    "input_identity": "0" * 64}), \
                mock.patch.object(ca, "policy_pair", return_value=(left, right, "joint S14 campaign policy hashes")), \
                mock.patch.object(ca, "check_binding", return_value="bound to the campaign's original run identity"):
            path = ca.plan_confirmation([EXAMPLES / "peptides_func"], self.root / "conf", campaign=self.root, reps=40)
        dataset = json.loads(path.read_text())["datasets"][0]
        self.assertEqual(dataset["seeds"], seeds)
        self.assertIn("registered training seeds", dataset["seed_source"])

    def test_export_records_the_ensemble_seed_axis_of_a_repository_built_campaign(self):
        from tests.test_s14_confirmation import S14CampaignTests
        output, _, _ = self.fabricate()
        fixture = S14CampaignTests(S14_FIXTURE)
        fixture.SEEDS = ("ensemble",)
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        fixture.declare_repository_built()
        written = ca.export_confirmation(output, campaign=fixture.root, manuscript_root=fixture.manuscript,
                                         root=self.records)
        analysis = json.loads((written["confirmation_sensitivity"] / "raw" / "confirmation_analysis.json").read_text())
        agreement = analysis["campaign"]["agreement"]["Pascal/RH"]
        self.assertEqual(agreement["S14_seed_axis"], "ensemble")
        self.assertEqual(agreement["E16_seed_reference"], "the parent run's registered training seeds")
        self.assertFalse(agreement["E16_seeds_are_the_S14_seed_population"])     # fixture array: seeds 1..5, parent: 1, 2
        for name, target in written.items():
            verify_set(target, specs()["sets"][name], specs())

    def test_paired_array_refuses_misaligned_seeds(self):
        output, plan, _ = self.fabricate(datasets=("Peptides-func",))
        job = plan["jobs"][1]
        directory = job_directory(output, job)
        (directory / "pair.json").unlink()
        ca.write_pair_result(directory, job, unit_id=np.arange(200, 200 + SIZES["Peptides-func"]),
                             left_loss=np.ones(SIZES["Peptides-func"]), right_loss=np.ones(SIZES["Peptides-func"]),
                             executions={}, reproduction={}, parent_files=[], started=utc_now())
        with self.assertRaisesRegex(RunnerError, "different validation graphs"):
            ca.paired_array(output, plan, plan["datasets"][0])


# ============================================================================= oracle_dispersion
def configuration(index, route, losses, *, actions, comparator=False, terminal=None, support=None, complete=None,
                  weights=None):
    """In-memory arrays of one enumerated configuration: losses [depth, inputs]; actions [members, inputs, depth]."""
    losses = np.asarray(losses, dtype=np.float64)
    depth, inputs = losses.shape
    actions = np.asarray(actions, dtype=np.int16)
    members = actions.shape[0]
    terminal = np.full((members, inputs), depth, dtype=np.int16) if terminal is None else np.asarray(terminal)
    support = np.full((members, inputs), depth, dtype=np.int16) if support is None else np.asarray(support)
    if complete is None:
        complete = losses[terminal.min(axis=0) - 1, np.arange(inputs)]
    return {"result": {"is_comparator": comparator, "route": route, "failures": [],
                       "candidate": {"registry_index": index, "candidate_id": f"c{index}"}},
            "unit_id": np.arange(inputs), "prefix_loss": losses, "complete_loss": np.asarray(complete, dtype=float),
            "terminal_depth": terminal, "supported_prefix": support, "actions": actions,
            "safe_depths": np.zeros_like(actions), "unit_weight": np.ones(inputs) if weights is None else weights}


class OracleTests(Workspace):
    def test_family_is_the_resource_feasible_registry_at_the_comparator_size(self):
        run = SealedRun(EXAMPLES / "peptides_func")
        family, scope = ao.oracle_family(run)
        comparator = run.comparator()
        self.assertEqual(scope["ensemble"], comparator["ensemble"])
        self.assertIn(comparator["registry_index"], [c["registry_index"] for c in family])
        for c in family:
            self.assertEqual(c["ensemble"], comparator["ensemble"])
            if c["registry_index"] != comparator["registry_index"]:
                self.assertLessEqual(c["cost"], run.config["selection"]["max_normalized_work"])
        self.assertTrue(all("normalized cost" in e["reason"] for e in scope["excluded"]))
        only, _ = ao.oracle_family(run, routes={"nominal"})
        self.assertEqual(sorted(c["route"] for c in only), sorted({"nominal", comparator["route"]}))

    def test_plan_one_job_per_configuration(self):
        path = ao.plan_oracle([EXAMPLES / "arxiv", EXAMPLES / "peptides_func"], self.root / "oracle")
        plan = json.loads(path.read_text())
        for dataset in plan["datasets"]:
            member = dataset["members"][0]
            jobs = [j for j in plan["jobs"] if j["dataset"] == dataset["dataset"]]
            self.assertEqual(len(jobs), member["family_size"])
            self.assertEqual(sum(j["is_comparator"] for j in jobs), 1)
            self.assertEqual(jobs[0]["seeds"], [1, 2][:member["ensemble"]])
        self.assertEqual({d["dataset"]: d["input_unit"] for d in plan["datasets"]},
                         {"ogbn-arxiv": "buffered cluster", "Peptides-func": "complete graph"})
        missing = self.copy_example("peptides_func")
        (missing / "stages" / "admit.json").unlink()
        with self.assertRaises(RunnerError):
            ao.plan_oracle([missing], self.root / "x")

    def test_support_rule_of_the_runtime(self):
        decisions = [{"action": 0, "supported_actions": list(range(10))}, {"action": 8, "supported_actions": [0]},
                     {"action": 3, "supported_actions": [0, 1]}, {"action": 0, "supported_actions": [0]}]
        self.assertEqual(ao.supported_prefix(decisions), 2)
        self.assertEqual(ao.supported_prefix(decisions[:2]), 2)
        self.assertEqual(ao.supported_prefix([]), 0)

    def test_enumeration_depth_disagreement_ties_support_and_complete_executions(self):
        qc = configuration(3, "fixed_sequence", [[0.9, 0.8, 0.70], [0.7, 0.75, 0.50], [0.6, 0.70, 0.60]],
                           actions=[[[1, 1, 1]] * 3], comparator=True)
        other = configuration(5, "phase", [[0.95, 0.5, 0.9], [0.65, 0.9, 0.9], [0.55, 0.6, 0.9]],
                              actions=[[[0, 4, 4]] * 3], support=[[3, 3, 3]])
        oracle = ao.enumerate_oracle([other, qc])
        rows = oracle["per_input"]
        self.assertEqual([(r["oracle_depth"], r["differs_from_qc"]) for r in rows], [(3, True), (1, True), (2, True)])
        self.assertEqual(rows[2]["oracle_registry_index"], 3)        # a truncated q_C prefix differs from q_C
        self.assertAlmostEqual(oracle["delta_oracle_mean_nats"], np.mean([0.55 - 0.6, 0.5 - 0.7, 0.5 - 0.6]))
        unsupported = configuration(5, "phase", other["prefix_loss"], actions=[[[0, 4, 4]] * 3], support=[[0, 3, 3]])
        self.assertEqual(ao.enumerate_oracle([unsupported, qc])["per_input"][0]["oracle_element"], "q_C")
        twin = configuration(1, "nominal", qc["prefix_loss"], actions=qc["actions"])
        tied = ao.enumerate_oracle([twin, qc])["per_input"]
        self.assertEqual([r["differs_from_qc"] for r in tied], [False, False, True])
        self.assertEqual(tied[0]["oracle_element"], "q_C")
        stop = configuration(7, "learned_stop", [[0.9, 0.9, 0.9], [0.9, 0.9, 0.9], [np.nan, np.nan, np.nan]],
                             actions=[[[0, 0, -1]] * 3, [[0, 0, 0]] * 3], terminal=[[2, 2, 2], [3, 3, 3]],
                             complete=[0.4, 0.95, 0.95])
        qc2 = configuration(3, "fixed_sequence", qc["prefix_loss"], actions=[[[1, 1, 1]] * 3, [[1, 1, 1]] * 3],
                            comparator=True)
        rows = ao.enumerate_oracle([stop, qc2])["per_input"]
        self.assertEqual((rows[0]["oracle_depth"], rows[0]["oracle_element"]), (3, "complete execution"))
        with self.assertRaisesRegex(RunnerError, "q_C exactly once"):
            ao.enumerate_oracle([other])
        misaligned = dict(other, unit_id=np.arange(1, 4))
        with self.assertRaisesRegex(RunnerError, "different admission inputs"):
            ao.enumerate_oracle([misaligned, qc])
        # A configuration that failed on input 1 (absent from its arrays) contributes nothing there.
        partial = configuration(5, "phase", other["prefix_loss"][:, [0, 2]], actions=[[[0, 4, 4]] * 2])
        partial["unit_id"] = np.array([0, 2])
        rows = ao.enumerate_oracle([partial, qc])["per_input"]
        self.assertEqual([(r["oracle_registry_index"], r["oracle_depth"]) for r in rows], [(5, 3), (3, 3), (3, 2)])

    def test_dispersion_quantiles(self):
        out = ao.dispersion([4, 8, 8, 16, 32], [True, False, True, False, False])
        self.assertEqual((out["median_oracle_depth"], out["oracle_depth_q25"], out["oracle_depth_q75"]), (8.0, 8.0, 16.0))
        self.assertEqual((out["fraction_differs_from_qc"], out["inputs"]), (0.4, 5))
        with self.assertRaises(RunnerError):
            ao.dispersion([], [])

    def fabricate(self, *, max_graphs=0):
        output = self.root / "oracle"
        ao.plan_oracle([EXAMPLES / "arxiv", EXAMPLES / "peptides_func"], output, max_graphs=max_graphs)
        plan = json.loads((output / "plan.json").read_text())
        rng = np.random.default_rng(23)
        for entry in plan["jobs"]:
            depth = SealedRun(entry["run"]).config["model"]["depth"]
            members, count = len(entry["seeds"]), 8 if entry["dataset"] == "ogbn-arxiv" else 11
            stop = entry["candidate"]["route"] == "learned_stop"
            inputs = []
            for u in range(count):
                records = []
                for m in range(members):
                    t = int(rng.integers(2, depth + 1)) if stop else depth
                    records.append({"actions": rng.integers(0, 10, t).tolist(), "safe_depths": [0] * t,
                                    "supported_prefix": int(rng.integers(1, t + 1)), "terminal": t})
                reach = min(r["terminal"] for r in records)
                prefix = np.full(depth, np.nan)
                prefix[:reach] = rng.uniform(0.3, 0.9, reach)
                inputs.append({"unit_id": u, "prefix_loss": prefix, "complete_loss": float(rng.uniform(0.3, 0.9)),
                               "weight": int(rng.integers(2, 6)), "members": records})
            ao.write_oracle_result(job_directory(output, entry), entry, inputs=inputs, failures=[], depth=depth,
                                   members=members, reproduction={}, parent_files=[], started=utc_now(),
                                   environment=ENVIRONMENT)
        return output, plan

    def test_export_traces_every_value(self):
        output, plan = self.fabricate()
        with self.assertRaisesRegex(RunnerError, "coverage"):
            ao.export_oracle(output, root=self.records)
        with self.assertRaisesRegex(RunnerError, "complete measured sets"):
            ao.export_oracle(output, root=RECORDS, allow_partial=True)
        self.assertFalse((RECORDS / "oracle_dispersion").exists())
        target = ao.export_oracle(output, root=self.records, allow_partial=True)
        verify_set(target, specs()["sets"]["oracle_dispersion"], specs())
        got = values(target)
        for dataset in plan["datasets"]:
            configurations = [ao.load_configuration(job_directory(output, j)) for j in plan["jobs"]
                              if j["dataset"] == dataset["dataset"]]
            oracle = ao.enumerate_oracle(configurations)
            expected = ao.dispersion([r["oracle_depth"] for r in oracle["per_input"]],
                                     [r["differs_from_qc"] for r in oracle["per_input"]])
            for quantity in ao.QUANTITY_UNITS:
                self.assertAlmostEqual(got[(dataset["dataset"], quantity)], expected[quantity], places=9)
            for row in oracle["per_input"]:
                self.assertLessEqual(row["oracle_loss"], row["qc_loss"])
        rows = read_summary(target, specs()["sets"]["oracle_dispersion"], specs()["summary_columns"])
        self.assertEqual({r["unit"] for r in rows}, set(ao.QUANTITY_UNITS.values()))

    def test_a_truncated_enumeration_is_a_software_check(self):
        output, _ = self.fabricate(max_graphs=3)
        with self.assertRaisesRegex(RunnerError, "software check"):
            ao.export_oracle(output, root=RECORDS)
        target = ao.export_oracle(output, root=self.records)
        self.assertTrue(json.loads((target / "provenance.json").read_text())["truncated"])


# ============================================================================= command line
class CommandLineTests(Workspace):
    def call(self, *argv):
        from scripts.run_confirmation_analysis import main
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main([str(a) for a in argv])
        return code, out.getvalue(), err.getvalue()

    def test_plan_run_export_dispatch(self):
        code, out, err = self.call("mechanism", "plan", "--parent", EXAMPLES / "peptides_func", "--output",
                                   self.root / "m", "--reps", "40", "--cell", "Peptides-func=rh_minus_ol")
        self.assertEqual(code, 0, err)
        plan = json.loads((self.root / "m" / "plan.json").read_text())
        self.assertEqual([c["contrast"] for c in plan["jobs"][0]["cells"]], ["rh_minus_ol"])
        code, _, err = self.call("mechanism", "run", "--output", self.root / "m")
        self.assertEqual(code, 2)
        self.assertIn("trust", err)
        code, _, err = self.call("mechanism", "export", "--output", self.root / "m", "--root", self.records)
        self.assertEqual(code, 2)
        self.assertIn("has not completed", err)
        code, _, err = self.call("mechanism", "plan", "--parent", EXAMPLES / "arxiv", "--output", self.root / "bad",
                                 "--cell", "ogbn-arxiv=radius")
        self.assertEqual(code, 2)
        self.assertIn("--cell", err)
        code, _, err = self.call("confirmation", "plan", "--parent", EXAMPLES / "peptides_func", "--output",
                                 self.root / "c", "--allow-fixture")
        self.assertEqual(code, 0, err)
        code, _, err = self.call("confirmation", "export", "--output", self.root / "c", "--root", self.records)
        self.assertEqual(code, 2)
        self.assertIn("campaign", err)
        code, _, err = self.call("oracle", "plan", "--parent", EXAMPLES / "arxiv", "--output", self.root / "o",
                                 "--routes", "nominal", "best_depth")
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads((self.root / "o" / "plan.json").read_text())["routes"], ["best_depth", "nominal"])
        code, _, err = self.call("oracle", "export", "--output", self.root / "o", "--root", self.records)
        self.assertEqual(code, 2)
        self.assertIn("has not completed", err)


# ============================================================================= execution (PyTorch; CI)
@unittest.skipUnless(TORCH, "PyTorch is required to execute the analyses")
class ExecutionTests(unittest.TestCase):
    """Smoke lifecycles on synthetic data, then each analysis end to end (software checks, not measurements)."""

    @classmethod
    def setUpClass(cls):
        import torch
        from cross_morse.pipeline import Pipeline
        torch.set_num_threads(2)
        cls.temporary = tempfile.TemporaryDirectory(prefix="cm-confirmation-runs-")
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
        self.work = Path(tempfile.mkdtemp(prefix="cm-analysis-", dir=self.root))
        self.records = self.work / "records"

    def test_mechanism_cells_execute_with_the_confirm_stage_code(self):
        output = self.work / "mech"
        cells = {"ogbn-arxiv": ["state", "rh_minus_ol"], "Peptides-func": ["state", "rh_minus_ol"]}
        ca.plan_mechanism([self.runs["arxiv"], self.runs["peptides"]], output, cells=cells, reps=20)
        with contextlib.redirect_stdout(io.StringIO()):
            ca.run_mechanism(output, device="cpu", trust=True)
        plan = json.loads((output / "plan.json").read_text())
        valid = 0
        for entry in plan["jobs"]:
            result = json.loads((job_directory(output, entry) / "mechanism.json").read_text())
            self.assertEqual([c["contrast"] for c in result["cells"]], ["state", "rh_minus_ol"])
            for cell in result["cells"]:
                if cell["valid"]:
                    valid += 1
                    self.assertLessEqual(cell["interval_low_nats"], cell["interval_high_nats"])
                else:
                    self.assertTrue(cell["invalid_reason"])
        if not valid:
            self.skipTest("no smoke cell had a valid interval")
        target = ca.export_mechanism(output, root=self.records, allow_partial=True)
        verify_set(target, specs()["sets"]["mechanism_contrasts"], specs())

    def test_state_contrast_reproduces_the_sealed_confirm_stage(self):
        for name, run in self.runs.items():
            sealed = json.loads((run / "confirm" / "paired_contrasts.json").read_text())
            # An invalid intervention carries the comparator's fallback bytes: no state contrast to reproduce.
            rows = [c for c in sealed if c["kind"] == "state" and c.get("differences") is not None
                    and c.get("intervention_valid", True)]
            if not rows:
                continue
            dataset = SealedRun(run).dataset
            output = self.work / f"repro-{name}"
            ca.plan_mechanism([run], output, cells={dataset: ["state"]}, anchors={dataset: rows[0]["candidate_index"]},
                              reps=20)
            with contextlib.redirect_stdout(io.StringIO()):
                ca.run_mechanism(output, device="cpu", trust=True)
            entry = json.loads((output / "plan.json").read_text())["jobs"][0]
            cell = entry["cells"][0]
            with np.load(job_directory(output, entry) / "unit_losses.npz") as arrays:
                differences = arrays[f"{cell['reference_key']}_unit_loss"] - arrays[f"{cell['candidate_key']}_unit_loss"]
            np.testing.assert_allclose(differences, np.asarray(rows[0]["differences"], dtype=float), rtol=0, atol=1e-6)
            return
        self.skipTest("neither smoke run confirmed a state contrast")

    def test_validation_split_pairs_feed_the_two_way_sets(self):
        output = self.work / "conf"
        ca.plan_confirmation([self.runs["peptides"]], output, allow_fixture=True, reps=20)
        with contextlib.redirect_stdout(io.StringIO()):
            ca.run_confirmation(output, device="cpu", trust=True)
        plan = json.loads((output / "plan.json").read_text())
        for entry in plan["jobs"]:
            pair = json.loads((job_directory(output, entry) / "pair.json").read_text())
            self.assertGreaterEqual(pair["graphs"], 2)
            for check in pair["reproduction"].values():
                self.assertLess(check["max_abs_difference"], 1e-6)
        from tests.test_s14_confirmation import S14CampaignTests
        fixture = S14CampaignTests(S14_FIXTURE)
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        try:
            ca.deployment_share(SealedRun(self.runs["peptides"]))
        except RunnerError as exc:
            # A smoke run that returns q_C has a zero-width test interval: g is undefined and nothing is exported.
            self.assertRegex(str(exc), "zero-width|not a variance share")
            with self.assertRaises(RunnerError):
                ca.export_confirmation(output, campaign=fixture.root, manuscript_root=fixture.manuscript,
                                       root=self.records, allow_partial=True)
            self.assertFalse((self.records / "confirmation_contrasts").exists())
            return
        written = ca.export_confirmation(output, campaign=fixture.root, manuscript_root=fixture.manuscript,
                                         root=self.records, allow_partial=True)
        for name, target in written.items():
            verify_set(target, specs()["sets"][name], specs())
        analysis = json.loads((written["confirmation_sensitivity"] / "raw" / "confirmation_analysis.json").read_text())
        self.assertEqual([e["contrast"] for e in analysis["e16_direct_estimate"]], ["Peptides/radius"])

    def test_admission_oracle_reproduces_the_admission_comparator(self):
        output = self.work / "oracle"
        ao.plan_oracle([self.runs["arxiv"], self.runs["peptides"]], output)
        with contextlib.redirect_stdout(io.StringIO()):
            ao.run_oracle(output, device="cpu", trust=True)
        plan = json.loads((output / "plan.json").read_text())
        for entry in plan["jobs"]:
            result = json.loads((job_directory(output, entry) / "behavior.json").read_text())
            if result["reproduction"]["max_abs_difference"] is not None:
                self.assertLess(result["reproduction"]["max_abs_difference"], 1e-6)
            if entry["is_comparator"]:
                self.assertFalse(result["failures"])
                self.assertGreater(result["reproduction"]["units_compared"], 0)
                self.assertLess(result["reproduction"]["max_abs_loss_difference"], 1e-6)
        target = ao.export_oracle(output, root=self.records, allow_partial=True)
        verify_set(target, specs()["sets"]["oracle_dispersion"], specs())
        for dataset in plan["datasets"]:
            configurations = [ao.load_configuration(job_directory(output, j)) for j in plan["jobs"]
                              if j["dataset"] == dataset["dataset"]]
            for row in ao.enumerate_oracle(configurations)["per_input"]:
                self.assertLessEqual(row["oracle_loss"], row["qc_loss"])


if __name__ == "__main__":
    unittest.main()
