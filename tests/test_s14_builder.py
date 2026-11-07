"""Joint S14 campaign builder: cm_campaigns/s14_builder.py, scripts/build_s14_campaign.py.

Torch-free: planning, the sealed-contrast verification and statistics replay, the small-sample test,
assembly and check_campaign's validation run on fixture runs -- fabricated sealed files of the seven
benchmarks at the S14 unit counts, never benchmark evidence -- and on fabricated job outputs.  Execution on
smoke lifecycles needs PyTorch and runs in CI.
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import math
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from cm_campaigns import s14_builder as sb
from cm_campaigns.common import sha256
from cm_campaigns.confirmation import DATASETS, FAMILY, check_campaign, composite_policy_hash
from cm_experiments import confirmation_analysis as ca
from cm_experiments.ensemble_grid import RunnerError, SealedRun, job_directory, slug

ROOT = Path(__file__).resolve().parents[1]
TORCH = importlib.util.find_spec("torch") is not None
ENVIRONMENT = {"python": "3.x", "machine": "x86_64", "torch": None}
SOURCE = "5" * 64
STAGES = ("prepare", "train", "explore", "fit", "select", "confirm", "admit", "evaluate")
REPS = 400
SEED = 11
NOW = datetime.now(timezone.utc).replace(microsecond=0)
T_SEALED = NOW - timedelta(days=30)         # prepare .. confirm of every fixture run
T_PLAN = NOW - timedelta(days=2)            # registration
T_JOB = NOW - timedelta(days=1)             # job starts (finishes: the writers' clock)
T_LOCK = NOW + timedelta(minutes=5)         # assembly and the joint lock
T_ADMIT_BEFORE = NOW - timedelta(days=20)   # post hoc: the lifecycle admitted before the joint lock
T_ADMIT_AFTER = NOW + timedelta(days=1)     # protocol position: admission after the joint lock


def lock_clock():
    """Assembly and joint-lock time: five minutes after the later of import time and now. The run step stamps
    each test with the real clock, and a full suite can reach these tests long after this module was imported."""
    return max(T_LOCK, datetime.now(timezone.utc).replace(microsecond=0) + timedelta(minutes=5))
VALID = {"feasible": True, "state_control_valid": True, "shuffle_matched_sites": 6, "shuffle_missing_sites": 0}
# dataset -> (unit layout, provisional index or None for q_C, state effect, route effect)
LAYOUT = {"ogbn-arxiv": ("clusters", 32, 4, 0.006, -0.005), "ogbn-proteins": ("clusters", 32, 4, 0.005, -0.004),
          "ogbn-products": ("clusters", 64, None, 0.0002, None), "Roman-empire": ("roman", 4, 5, 0.006, -0.006),
          "PascalVOC-SP": ("graphs", 425, 5, 0.006, -0.009), "COCO-SP": ("graphs", 5664, None, None, -0.009),
          "Peptides-func": ("graphs", 544, 4, 0.006, 0.0)}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def ledger(split=None) -> list[dict]:
    """A registry of the runtime's shape: q_C is the fixed sequence (split-specific on Roman-empire)."""
    items = [dict(route="nominal", complexity=0, is_open_loop=True, profile="none", depth=4),
             dict(route="fixed_sequence", complexity=4, is_open_loop=True, profile="none", depth=4,
                  sequence=[0 if split is None else int(split), 1, 2, 3]),
             dict(route="mpc_ol", complexity=5, is_open_loop=True, profile="lean", horizon=2, depth=4),
             dict(route="one_step", complexity=6, is_open_loop=False, profile="lean", depth=4),
             dict(route="radius", complexity=7, is_open_loop=False, profile="lean", kappa=0.5, depth=4),
             dict(route="mpc_rh", complexity=8, is_open_loop=False, profile="lean", horizon=2, depth=4)]
    out = []
    for item in items:
        item = dict(item, behavior_id=digest(item)[:16], ensemble=1)
        item["hash"] = digest(item)
        item["candidate_id"] = item["hash"][:20]
        out.append(item)
    return out


def key_of(candidate, shuffle=False) -> str:
    return slug(("shuffle_" if shuffle else "") + candidate["candidate_id"])


class Fixture:
    """Fixture runs of the seven benchmarks: sealed role, selection, confirm and admission files."""

    def __init__(self, root: Path, *, admitted_at: datetime, rng_seed=5, provisional=None, roman_routes=None,
                 confirm_units=None):
        self.root, self.admitted_at = Path(root), admitted_at
        self.rng = np.random.default_rng(rng_seed)
        self.provisional = dict(provisional or {})
        self.roman_routes = roman_routes
        self.confirm_units = dict(confirm_units or {})
        self.losses = {}           # run directory -> {"units", "counts", "policies": {key: loss}}
        self.parents = {}
        for dataset in DATASETS:
            self.parents[dataset] = self.dataset(dataset)

    # -- layouts --------------------------------------------------------------------------------
    def roles_for(self, kind, count, split=None):
        if kind == "graphs":
            others = 9                  # one graph in each other leaf role and an extra fit_train graph
            graphs = count + others
            order = np.arange(graphs)
            roles = {"fit_train": order[[0, 1]], "fit_val": order[[2]], "explore": order[[3]], "select": order[[4]],
                     "compare": order[[5]], "admit": order[[6]], "test": order[[7, 8]], "confirm": order[others:]}
            roles["fit"] = np.sort(np.r_[roles["fit_train"], roles["fit_val"]])
            clusters = np.arange(graphs)
            return roles, clusters, graphs
        clusters_n = count
        per = 10
        nodes = clusters_n * per
        clusters = np.repeat(np.arange(clusters_n), per)
        roles = {name: [] for name in ("fit_train", "fit_val", "explore", "select", "confirm", "compare", "admit", "test")}
        shift = 0 if split is None else int(split)
        for c in range(clusters_n):
            members = list(range(c * per, (c + 1) * per))
            members = members[shift % per:] + members[:shift % per]
            confirm = 1 + c % 3                 # unequal cluster weights
            roles["confirm"].extend(members[:confirm])
            rest = members[confirm:]
            for name, node in zip(("fit_train", "fit_val", "explore", "select", "compare", "admit", "test"), rest):
                roles[name].append(node)
            roles["fit_train"].extend(rest[7:])
        roles = {k: np.asarray(sorted(v), dtype=np.int64) for k, v in roles.items()}
        roles["fit"] = np.sort(np.r_[roles["fit_train"], roles["fit_val"]])
        return roles, clusters, nodes

    def dataset(self, dataset):
        kind = LAYOUT[dataset][0]
        if kind != "roman":
            return self.run(self.root / "parents" / slug(dataset), dataset, None)
        campaign = self.root / "parents" / "roman_campaign"
        for k in range(sb.ROMAN_SPLITS):
            self.run(campaign / f"split_{k:02d}", dataset, k)
        return campaign

    # -- one sealed run --------------------------------------------------------------------------
    def run(self, root: Path, dataset: str, split):
        kind, count, provisional, state_effect, route_effect = LAYOUT[dataset]
        provisional = self.provisional.get(dataset, provisional)
        count = self.confirm_units.get(dataset, count)
        roles, clusters, samples = self.roles_for("clusters" if kind == "roman" else kind, count, split)
        candidates = ledger(split)
        if kind == "roman" and self.roman_routes is not None:
            candidates = ledger(split)
            if self.roman_routes[split] == "phase":
                candidates[1] = dict(candidates[1], route="phase")
        config = {"schema_version": 1, "dataset": dataset, "profile": "full", "scientific_run": True, "seed": SEED,
                  "training": {"seeds": [1, 2]}, "selection": {"bootstrap_reps": REPS},
                  "registry": {"ensemble_sizes": [1]}, "data": {"split_index": split} if kind == "roman" else {}}
        write(root / "config.json", config)
        prepared = root / "prepared"
        prepared.mkdir(parents=True, exist_ok=True)
        if kind == "graphs":
            np.savez(prepared / "graph_splits.npz", **roles)
            np.savez(prepared / "splits.npz", **roles)
        else:
            np.savez(prepared / "splits.npz", **roles)
        np.savez(prepared / "clusters.npz", cluster_ids=clusters)
        write(prepared / "metadata.json", {"is_benchmark_data": True})
        qc = 1
        eligible = [] if provisional is None else [provisional]
        upper = [0.0, 0.0, -0.001, -0.002, -0.004, -0.003]
        write(root / "select" / "candidates.json", candidates)
        write(root / "select" / "decision.json", {"comparator_index": qc, "provisional_index": qc if provisional is None
                                                  else provisional, "eligible_indices": eligible, "upper": upper,
                                                  "joint_feasible": [True] * len(candidates)})
        # D_confirm per-unit losses of every policy the S14 tests can involve.
        confirm = roles["confirm"]
        units, counts = np.unique(clusters[confirm], return_counts=True)
        if kind == "graphs":
            counts = np.ones(len(units), dtype=np.int64)
        n = len(units)
        policies = {}
        target = provisional if provisional is not None else 4
        base = self.rng.uniform(0.4, 0.9, n)
        policies[key_of(candidates[target])] = base
        policies[key_of(candidates[target], shuffle=True)] = base + (state_effect or 0.0) + self.rng.normal(0, 4e-4, n)
        rh_loss = base if target == 5 else self.rng.uniform(0.4, 0.9, n)
        policies[key_of(candidates[5])] = rh_loss
        policies[key_of(candidates[2])] = rh_loss - (route_effect or 0.0) + self.rng.normal(0, 4e-4 if route_effect else 3e-3, n)
        radius_loss = base if target == 4 else self.rng.uniform(0.4, 0.9, n)
        policies[key_of(candidates[4])] = radius_loss
        policies[key_of(candidates[3])] = radius_loss - (route_effect or 0.0) + self.rng.normal(
            0, 4e-4 if route_effect else 3e-3, n)
        self.losses[str(root.resolve())] = {"units": units, "counts": counts, "policies": policies}
        # The run's own confirm stage (cross_morse.pipeline._stage_confirm on these per-unit losses).
        from cross_morse.statistics import confirmation
        closure = [] if provisional is None else [provisional]
        sealed = []
        for i in closure:
            intact = policies[key_of(candidates[i])]
            sealed.append({"kind": "state", "candidate_index": i, "predecessor_index": qc,
                           "differences": policies[key_of(candidates[i], shuffle=True)] - intact, "weights": counts})
            simpler = 3 if candidates[i]["route"] == "radius" else 2
            sealed.append({"kind": "radius" if candidates[i]["route"] == "radius" else "mpc", "candidate_index": i,
                           "predecessor_index": simpler, "differences": intact - policies[key_of(candidates[simpler])],
                           "weights": counts})
        decision = confirmation(qc if provisional is None else provisional, qc, eligible, sealed, reps=REPS, seed=SEED + 5)
        write(root / "confirm" / "contrast_registry.json", {"candidate_indices": closure, "selection_fixed": True})
        write(root / "confirm" / "paired_contrasts.json", ca.json_safe(sealed))
        write(root / "confirm" / "decision.json", ca.json_safe(decision))
        confirmed = decision["confirmed_index"]
        lock = {"comparator_index": qc, "confirmed_index": confirmed, "comparator_hash": candidates[qc]["hash"],
                "confirmed_hash": candidates[confirmed]["hash"], "config_sha256": canonical_sha(config),
                "test_labels_accessed": False}
        write(root / "admit" / "pre_admission_lock.json", lock)
        write(root / "policy_lock.json", dict(lock, deployed_index=qc, deployed_hash=candidates[qc]["hash"]))
        state = {"status": "COMPLETE", "last_complete_stage": "evaluate", "source_sha256": SOURCE,
                 "config_sha256": canonical_sha(config), "dataset": dataset, "scientific_run": True}
        write(root / "run_state.json", state)
        units_sha = sha256(prepared / "splits.npz")
        times = {stage: T_SEALED + timedelta(hours=i) for i, stage in enumerate(STAGES)}
        times["admit"] = self.admitted_at
        times["evaluate"] = self.admitted_at + timedelta(hours=1)
        for i, stage in enumerate(STAGES):
            write(root / "stages" / f"{stage}.json", {"stage": stage, "status": "COMPLETE", "seconds_this_attempt": 60.0})
            write(root / "provenance" / "stages" / f"{i:02d}_{stage}.json",
                  {"stage": stage, "dataset": dataset, "created_utc": times[stage].isoformat(), "source_sha256": SOURCE,
                   "config_sha256": canonical_sha(config), "unit_sha256": units_sha})
        return root

    # -- fabricated job outputs -----------------------------------------------------------------
    def scores(self, run_root, keys):
        data = self.losses[str(Path(run_root).resolve())]
        return {key: {"loss": np.asarray(data["policies"][key], dtype=np.float64), "counts": data["counts"],
                      "core_ids": data["units"]} for key in keys}

    def execute(self, output, plan, *, invalid=()):
        """What `run` writes, from the fixture losses: read jobs verify the sealed contrasts with the builder's
        own torch-free code; compute jobs use the mechanism-contrast statistics."""
        for entry in plan["jobs"]:
            directory = job_directory(output, entry)
            keys = [p["key"] for p in entry["policies"]]
            scores = self.scores(entry["run"], keys)
            if entry["kind"] == "read":
                run = SealedRun(entry["run"])
                sealed, statistics = sb.replay_sealed_statistics(run)
                cells = sb.sealed_cells(entry["cells"], sealed, statistics, scores, entry["selection"]["comparator_index"])
                arrays = {"unit_id": scores[keys[0]]["core_ids"], "unit_weight": scores[keys[0]]["counts"],
                          **{f"{k}_unit_loss": s["loss"] for k, s in scores.items()}}
                sb.write_read_result(directory, entry, cells=cells, arrays=arrays, predictions={}, parent_files=[],
                                     started=T_JOB.isoformat(), replay={"rows": len(statistics)},
                                     environment=ENVIRONMENT)
            else:
                self.compute(entry, directory, invalid=invalid)

    def compute(self, entry, directory, *, invalid=(), started=None):
        """A compute job's outputs: the mechanism-contrast statistics of the fixture losses."""
        keys = [p["key"] for p in entry["policies"]]
        scores = self.scores(entry["run"], keys)
        executions = {key: dict(VALID) for key in keys}
        for contrast in invalid:
            cell = next((c for c in entry["cells"] if c["contrast"] == contrast), None)
            if cell is not None:
                executions[cell["reference_key"]] = {"execution_failure": True, "feasible": False}
        collection = entry["dataset"] in sb.GRAPH_DATASETS
        cells = ca.mechanism_statistics(entry["cells"], scores, executions, entry["selection"], collection=collection,
                                        reps=entry["bootstrap"]["reps"], seed=entry["bootstrap"]["seed"])
        arrays = {"unit_id": scores[keys[0]]["core_ids"], "unit_weight": scores[keys[0]]["counts"],
                  **{f"{k}_unit_loss": s["loss"] for k, s in scores.items()}}
        ca.write_mechanism_result(directory, entry, cells=cells, policies={}, executions=executions, arrays=arrays,
                                  unit_kind="graph" if collection else "buffered cluster", parent_files=[],
                                  started=started or T_JOB.isoformat(), environment=ENVIRONMENT)


def canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class Workspace(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cm-s14-builder-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def fixture(self, **kwargs) -> Fixture:
        kwargs.setdefault("admitted_at", T_ADMIT_AFTER)
        return Fixture(self.root / "fixture", **kwargs)

    def plan(self, fixture, name="build", **kwargs):
        path = sb.plan_s14(list(fixture.parents.values()), self.root / name, clock=lambda: T_PLAN, **kwargs)
        return self.root / name, json.loads(path.read_text())

    def build(self, *, admitted_at=T_ADMIT_AFTER):
        fixture = self.fixture(admitted_at=admitted_at)
        output, plan = self.plan(fixture)
        fixture.execute(output, plan)
        verdict = sb.assemble_s14(output, self.root / "campaign", clock=lock_clock)
        return fixture, output, plan, verdict


# ============================================================================= planning
class PlanningTests(Workspace):
    def test_targets_sources_and_registered_protocol(self):
        fixture = self.fixture()
        output, plan = self.plan(fixture)
        modes = {f["contrast_id"]: sorted({m["mode"] for m in f["members"]}) for f in plan["family"]}
        self.assertEqual(modes, {"arxiv/state": ["read"], "arxiv/radius": ["read"], "proteins/state": ["read"],
                                 "proteins/radius": ["read"], "products/state": ["compute"], "Roman/RH": ["read"],
                                 "Pascal/RH": ["read"], "COCO/RH": ["compute"], "Peptides/radius": ["read"]})
        family = {f["contrast_id"]: f for f in plan["family"]}
        arxiv = family["arxiv/state"]["members"][0]
        self.assertEqual(arxiv["right"]["registry_index"], 4)           # the selected radius, intact
        self.assertEqual(arxiv["left"]["state_control"], "matched_shuffle")
        self.assertEqual(family["arxiv/state"]["left_policy_sha256"], sb.state_control_hash(arxiv["right"]["hash"]))
        self.assertIn("provisional policy", arxiv["anchor_rule"])
        radius = family["arxiv/radius"]["members"][0]
        self.assertEqual((radius["left"]["route"], radius["right"]["route"]), ("radius", "one_step"))
        products = family["products/state"]["members"][0]
        self.assertEqual(products["right"]["registry_index"], 4)        # lowest D_select U among feedback routes
        self.assertIn("q_C selected", products["anchor_rule"])
        coco = family["COCO/RH"]["members"][0]
        self.assertEqual((coco["left"]["registry_index"], coco["right"]["registry_index"]), (5, 2))
        self.assertEqual(len(family["Roman/RH"]["members"]), 10)
        self.assertEqual([m["split_index"] for m in family["Roman/RH"]["members"]], list(range(10)))
        # One read and/or compute job per run, in the mechanism-contrast job format.
        self.assertEqual(len(plan["jobs"]), 16)
        compute = next(j for j in plan["jobs"] if j["kind"] == "compute" and j["dataset"] == "ogbn-products")
        self.assertEqual(compute["bootstrap"], {"reps": REPS, "seed": SEED + 5})
        self.assertEqual([p["seeds"] for p in compute["policies"]], [[1], [1]])
        self.assertTrue(any(p["uses_donors"] for p in compute["policies"]))
        # Registration: protocol with the registration time, policies, units, weights and hashed test code.
        protocol = json.loads((output / "registration" / "protocol.json").read_text())
        self.assertEqual(protocol["registered_at"], T_PLAN.isoformat())
        self.assertEqual([c["contrast_id"] for c in protocol["contrasts"]], list(FAMILY))
        for spec in protocol["contrasts"]:
            key = spec["contrast_id"]
            self.assertEqual(spec["seed_ids"], [sb.SEED_LEVEL])
            self.assertEqual(spec["unit_count"], FAMILY[key][3])
            self.assertEqual(spec["method"]["tail"], "greater" if FAMILY[key][5] > 0 else "less")
            code = output / "registration" / spec["method"]["analysis_code"]["path"]
            self.assertEqual(sha256(code), spec["method"]["analysis_code"]["sha256"])
            self.assertEqual(sha256(code), sha256(ROOT / sb.BUILDER_FILE))
            if FAMILY[key][2] == "buffered_cluster":
                weights = json.loads((output / "registration" / spec["weighting"]["unit_weights"]["path"]).read_text())
                self.assertEqual(len(weights), FAMILY[key][3])
                self.assertEqual(sorted(set(weights.values())), [1, 2, 3])
            else:
                self.assertEqual(spec["weighting"], {"scheme": "equal_unit"})
        roman = next(d for d in plan["datasets"] if d["dataset"] == "Roman-empire")
        aggregate = json.loads((output / "registration" / roman["candidates"]["path"]).read_text())
        selection = json.loads((output / "registration" / roman["selection"]["path"]).read_text())
        comparator = aggregate[selection["comparator_index"]]
        self.assertEqual(comparator["route"], "fixed_sequence")
        self.assertEqual(len({p["hash"] for p in comparator["split_policy_hashes"]}), 10)   # split-specific q_C
        self.assertEqual(comparator["hash"], composite_policy_hash(comparator["split_policy_hashes"]))
        self.assertEqual(aggregate[selection["provisional_index"]]["route"], "mpc_rh")
        self.assertEqual([aggregate[i]["route"] for i in selection["eligible_indices"]], ["mpc_rh"])
        self.assertEqual(family["Roman/RH"]["left_policy_sha256"], aggregate[selection["provisional_index"]]["hash"])
        roles = json.loads((output / "registration" / "roles" / "Roman-empire.json").read_text())
        self.assertEqual(sorted(roles["confirmation_units"]), [f"split={k}" for k in range(10)])
        self.assertEqual(roles["reserved_at"], (T_SEALED).isoformat())

    def test_refusals(self):
        fixture = self.fixture()
        parents = dict(fixture.parents)
        with self.assertRaisesRegex(RunnerError, "all seven datasets"):
            sb.plan_s14([p for d, p in parents.items() if d != "COCO-SP"], self.root / "a", clock=lambda: T_PLAN)
        with self.assertRaisesRegex(RunnerError, "one parent"):
            sb.plan_s14([*parents.values(), parents["COCO-SP"]], self.root / "b", clock=lambda: T_PLAN)
        with self.assertRaisesRegex(RunnerError, "outside the repository"):
            sb.plan_s14(list(parents.values()), ROOT / "s14-build", clock=lambda: T_PLAN)
        with self.assertRaisesRegex(RunnerError, "must follow"):
            sb.plan_s14(list(parents.values()), self.root / "c", clock=lambda: T_SEALED)
        with self.assertRaisesRegex(RunnerError, "software check"):
            sb.plan_s14(list(parents.values()), self.root / "d", force_compute=True, clock=lambda: T_PLAN)
        state = parents["ogbn-arxiv"] / "run_state.json"
        state.write_text(json.dumps({**json.loads(state.read_text()), "status": "RUNNING"}))
        with self.assertRaisesRegex(RunnerError, "not complete"):
            sb.plan_s14(list(parents.values()), self.root / "e", clock=lambda: T_PLAN)
        partial = sb.plan_s14([parents["Peptides-func"]], self.root / "f", allow_partial=True, clock=lambda: T_PLAN)
        self.assertTrue(json.loads(partial.read_text())["partial"])
        with self.assertRaisesRegex(RunnerError, "partial plan"):
            sb.assemble_s14(self.root / "f", self.root / "f-campaign", clock=lock_clock)

    def test_target_is_the_selected_policy_even_after_a_lifecycle_downgrade(self):
        # The run's confirm stage downgraded radius to its one-step predecessor, which was then deployed:
        # strongest_feedback would anchor on the deployed one-step; the S14 map tests the selected radius.
        fixture = self.fixture()
        run = Path(fixture.parents["ogbn-arxiv"])
        for name, field in (("policy_lock.json", "deployed_index"), ("confirm/decision.json", "confirmed_index")):
            data = json.loads((run / name).read_text())
            data[field] = 3
            (run / name).write_text(json.dumps(data))
        anchor, _ = ca.strongest_feedback(SealedRun(run))
        self.assertEqual(anchor["registry_index"], 3)
        target, rule = sb.s14_target(SealedRun(run))
        self.assertEqual((target["registry_index"], target["route"]), (4, "radius"))
        self.assertIn("provisional", rule)

    def test_roman_splits_choose_their_source_separately(self):
        fixture = self.fixture()
        split = Path(fixture.parents["Roman-empire"]) / "split_09"
        write(split / "confirm" / "contrast_registry.json", {"candidate_indices": [], "selection_fixed": True})
        output, plan = self.plan(fixture)
        roman = next(f for f in plan["family"] if f["contrast_id"] == "Roman/RH")
        self.assertEqual([m["mode"] for m in roman["members"]], ["read"] * 9 + ["compute"])
        job = plan["jobs"][roman["members"][9]["job"]]
        self.assertEqual((job["kind"], job["split_index"], job["run_name"]), ("compute", 9, "split_09"))
        self.assertEqual([c["contrast"] for c in job["cells"]], ["Roman/RH"])
        # The aggregate identity is the same whichever source a split uses.
        self.assertEqual(roman["left_policy_sha256"], composite_policy_hash(
            [{"split_index": k, "hash": ledger(k)[5]["hash"]} for k in range(10)]))

    def test_a_selected_route_the_row_cannot_test_is_refused(self):
        # Pascal selects a radius: its S14 row tests MPC-RH, which is then not the selected policy.
        fixture = Fixture(self.root / "x", admitted_at=T_ADMIT_AFTER, provisional={"PascalVOC-SP": 4})
        with self.assertRaisesRegex(RunnerError, "must test the selected policy"):
            sb.plan_s14(list(fixture.parents.values()), self.root / "a", clock=lambda: T_PLAN)

    def test_unit_counts_and_roman_routes_must_fit_the_family(self):
        fixture = Fixture(self.root / "x", admitted_at=T_ADMIT_AFTER, confirm_units={"PascalVOC-SP": 400})
        with self.assertRaisesRegex(RunnerError, "S14 registers 425"):
            sb.plan_s14(list(fixture.parents.values()), self.root / "a", clock=lambda: T_PLAN)
        routes = ["fixed_sequence"] * 9 + ["phase"]
        fixture = Fixture(self.root / "y", admitted_at=T_ADMIT_AFTER, roman_routes=routes)
        with self.assertRaisesRegex(RunnerError, "differ in route"):
            sb.plan_s14(list(fixture.parents.values()), self.root / "b", clock=lambda: T_PLAN)


# ============================================================================= sealed contrasts and statistics
class SealedContrastTests(Workspace):
    def setUp(self):
        super().setUp()
        self.fx = self.fixture()
        self.output, self.planned = self.plan(self.fx)
        self.entry = next(j for j in self.planned["jobs"] if j["dataset"] == "ogbn-arxiv")
        self.run = SealedRun(self.entry["run"])

    def test_replay_and_exact_reproduction(self):
        sealed, statistics = sb.replay_sealed_statistics(self.run)
        scores = self.fx.scores(self.run.root, [p["key"] for p in self.entry["policies"]])
        cells = sb.sealed_cells(self.entry["cells"], sealed, statistics, scores, 1)
        self.assertEqual([c["contrast"] for c in cells], ["arxiv/state", "arxiv/radius"])
        decision = json.loads((self.run.root / "confirm" / "decision.json").read_text())
        for cell, row in zip(cells, decision["contrasts"]):
            self.assertTrue(cell["valid"])
            self.assertEqual((cell["effect_nats"], cell["interval_low_nats"], cell["interval_high_nats"]),
                             (row["mean"], row["lower"], row["upper"]))
            self.assertEqual(cell["statistics"]["p_value_zero"], row["p_value_zero"])
            self.assertNotIn("holm_p_value_zero", cell["statistics"])
            self.assertEqual(cell["verification"]["differences_reproduced"], "exact")

    def test_any_difference_from_the_sealed_contrast_is_refused(self):
        sealed, statistics = sb.replay_sealed_statistics(self.run)
        keys = [p["key"] for p in self.entry["policies"]]
        scores = self.fx.scores(self.run.root, keys)
        scores[keys[0]] = dict(scores[keys[0]], loss=scores[keys[0]]["loss"] + 1e-12)
        with self.assertRaisesRegex(RunnerError, "do not reproduce the sealed"):
            sb.sealed_cells(self.entry["cells"], sealed, statistics, scores, 1)
        path = self.run.root / "confirm" / "decision.json"
        decision = json.loads(path.read_text())
        decision["contrasts"][0]["p_value_zero"] = 0.5
        path.write_text(json.dumps(decision))
        with self.assertRaisesRegex(RunnerError, "do not replay"):
            sb.replay_sealed_statistics(SealedRun(self.run.root))

    def test_an_invalid_sealed_intervention_is_reported_not_repaired(self):
        sealed, statistics = sb.replay_sealed_statistics(self.run)
        sealed = json.loads(json.dumps(sealed))
        sealed[0].update(intervention_valid=False, invalid_reason="missing/mismatched state donors", p_value=1.0)
        scores = self.fx.scores(self.run.root, [p["key"] for p in self.entry["policies"]])
        cells = sb.sealed_cells(self.entry["cells"], sealed, statistics, scores, 1)
        self.assertFalse(cells[0]["valid"])
        self.assertIsNone(cells[0]["effect_nats"])
        self.assertIn("donors", cells[0]["invalid_reason"])
        self.assertTrue(cells[1]["valid"])

    def test_small_sample_test_over_splits(self):
        from scipy.stats import t as student
        d = [-0.006, -0.0052, -0.0071, -0.0049, -0.0063, -0.0058, -0.0066, -0.0044, -0.0061, -0.0057]
        out = sb.small_sample_test(d, -1, 0.002)
        mean, se = float(np.mean(d)), float(np.std(d, ddof=1) / math.sqrt(10))
        self.assertAlmostEqual(out["effect_nats"], mean, places=15)
        self.assertAlmostEqual(out["interval_nats"][0], mean - student.ppf(0.975, 9) * se, places=15)
        self.assertAlmostEqual(out["p_value_zero"], student.cdf(mean / se, 9), places=15)
        self.assertAlmostEqual(out["p_value_margin"], student.sf((-mean - 0.002) / se, 9), places=15)
        self.assertEqual(out["degrees_of_freedom"], 9)
        self.assertAlmostEqual(sb.small_sample_test([-x for x in d], 1, 0.002)["p_value_zero"], out["p_value_zero"],
                               places=15)
        with self.assertRaisesRegex(RunnerError, "zero variance"):
            sb.small_sample_test([-0.005] * 10, -1, 0.002)


# ============================================================================= assembly and check_campaign
class AssemblyTests(Workspace):
    def test_campaign_in_protocol_position_passes_check_campaign_structure(self):
        fixture, output, plan, verdict = self.build(admitted_at=T_ADMIT_AFTER)
        campaign = self.root / "campaign"
        # Accepted structurally; the comparison with the printed S14 values is a separate verdict.
        self.assertTrue(verdict["accepted"], verdict.get("error"))
        self.assertIn(verdict["status"], ("PASS", "REFERENCE_MISMATCH"))
        self.assertEqual(verdict["findings"], [])
        self.assertEqual(json.loads((output / "campaign_check.json").read_text())["status"], verdict["status"])
        report = check_campaign(campaign, manuscript_root=ROOT)
        rows = {r["contrast_id"]: r for r in report["rows"]}
        self.assertEqual(set(rows), set(FAMILY))
        self.assertEqual((report["production_kind"], report["post_hoc"]), ("repository_built", False))
        self.assertEqual(set(report["admission_findings"]), set(DATASETS))
        for finding in report["admission_findings"].values():
            self.assertTrue(finding["admission_governed_by_joint_lock"] and finding["admission_after_joint_lock"]
                            and finding["admission_predictor_is_joint_confirmed_policy"]
                            and finding["lifecycle_lock_agrees_with_joint_decision"], finding)
        self.assertEqual(len(report["admission_findings"]["Roman-empire"]["run_locks"]), 10)
        lock = json.loads((campaign / "joint_decision_lock.json").read_text())
        manifest = json.loads((campaign / "campaign.json").read_text())
        self.assertEqual((manifest["status"], manifest["evidence_kind"], manifest["synthetic"]),
                         ("author_completed", "original_confirmation_tests", False))
        self.assertEqual(manifest["production"]["kind"], "repository_built")
        for key, spec in FAMILY.items():
            row = rows[key]
            result = json.loads((campaign / "raw" / sb.key_slug(key) / "test.json").read_text())
            self.assertEqual((row["unit_count"], row["seed_count"]), (spec[3], 1))
            self.assertEqual(row["effect_nats"], result["effect_nats"])
            self.assertTrue(math.isclose(row["reconstructed_effect_nats"], result["effect_nats"], rel_tol=1e-9))
            self.assertEqual(row["retained"], lock["mechanism_retained"][key])
            self.assertEqual(row["method_family"], {"graph": "graph_bootstrap", "official_split": "paired_small_sample"}
                             .get(spec[2], "paired_cluster" if spec[1] == "state" else "paired_unit"))
            self.assertLessEqual(result["interval_nats"][0], result["effect_nats"])
        self.assertEqual({k for k, r in rows.items() if r["retained"]},
                         {"arxiv/state", "arxiv/radius", "proteins/state", "proteins/radius", "Roman/RH", "Pascal/RH",
                          "COCO/RH"})
        decisions = report["policy_decisions"]
        self.assertEqual(decisions["Peptides-func"]["confirmed_hash"], decisions["Peptides-func"]["comparator_hash"])
        self.assertEqual(decisions["COCO-SP"]["confirmed_hash"], decisions["COCO-SP"]["comparator_hash"])
        self.assertFalse(rows["COCO/RH"]["tested_policy_selected"])
        self.assertNotEqual(decisions["ogbn-arxiv"]["confirmed_hash"], decisions["ogbn-arxiv"]["comparator_hash"])
        # Read tests are the sealed confirm-stage rows; compute tests the mechanism-contrast statistics.
        arxiv = json.loads((campaign / "raw" / "arxiv_state" / "test.json").read_text())
        sealed = json.loads((Path(fixture.parents["ogbn-arxiv"]) / "confirm" / "decision.json").read_text())
        state_row = next(r for r in sealed["contrasts"] if r["kind"] == "state")
        self.assertEqual((arxiv["effect_nats"], arxiv["p_value_zero"], arxiv["interval_nats"]),
                         (state_row["mean"], state_row["p_value_zero"], [state_row["lower"], state_row["upper"]]))
        self.assertEqual(arxiv["members"][0]["mode"], "read")
        # Roman: one row per split; the t test over the ten split contrasts.
        with (campaign / "raw" / "Roman_RH" / "paired.csv").open() as handle:
            roman = list(csv.DictReader(handle))
        self.assertEqual([r["unit_id"] for r in roman], [f"split={k}" for k in range(10)])
        d = [float(r["left_loss_nats"]) - float(r["right_loss_nats"]) for r in roman]
        result = json.loads((campaign / "raw" / "Roman_RH" / "test.json").read_text())
        self.assertAlmostEqual(result["p_value_zero"], sb.small_sample_test(d, -1, 0.002)["p_value_zero"], places=15)
        for member, value in zip(sorted(result["members"], key=lambda m: m["split_index"]), d):
            self.assertAlmostEqual(member["split_contrast_runtime_mean_nats"], value, places=13)
        # Raw job outputs travel with each test.
        self.assertTrue((campaign / "raw" / "COCO_RH" / "members").is_dir())
        self.assertTrue(any((campaign / "raw" / "COCO_RH" / "members").rglob("mechanism.json")))
        self.assertTrue(any((campaign / "raw" / "arxiv_state" / "members").rglob("read.json")))

    def test_post_hoc_campaign_is_accepted_with_reported_admission_findings(self):
        # Every complete run admitted before the joint lock: check_campaign reports it (post_hoc) and the
        # status remains the separate comparison with the printed values.
        _, output, _, verdict = self.build(admitted_at=T_ADMIT_BEFORE)
        self.assertTrue(verdict["accepted"], verdict.get("error"))
        self.assertEqual(verdict["status"], "REFERENCE_MISMATCH")
        report = verdict["report"]
        self.assertTrue(report["post_hoc"])
        self.assertEqual(report["production_kind"], "repository_built")
        for dataset, finding in report["admission_findings"].items():
            self.assertFalse(finding["admission_governed_by_joint_lock"], dataset)
            self.assertFalse(finding["admission_after_joint_lock"], dataset)
            self.assertTrue(finding["lifecycle_lock_agrees_with_joint_decision"], dataset)
        self.assertTrue(any("Repository-built campaign" in limit for limit in report["limits"]))
        self.assertEqual(len(verdict["findings"]), len(DATASETS))
        self.assertTrue(all("before the joint lock" in f for f in verdict["findings"]))
        link = json.loads((self.root / "campaign" / "admission" / "ogbn-arxiv.json").read_text())
        self.assertIsNone(link["joint_lock_sha256"])
        self.assertFalse(link["admission_after_joint_lock"])
        production = json.loads((self.root / "campaign" / "provenance.json").read_text())
        self.assertIn("post hoc", production["protocol_position"])
        # The same files declared as an imported original campaign keep the strict admission checks.
        manifest_path = self.root / "campaign" / "campaign.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.pop("production")
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "admission did not use the joint decision lock"):
            check_campaign(self.root / "campaign", manuscript_root=ROOT)

    def test_disagreement_with_the_lifecycle_decision_is_reported(self):
        fixture = self.fixture(admitted_at=T_ADMIT_AFTER)
        path = Path(fixture.parents["Peptides-func"]) / "admit" / "pre_admission_lock.json"
        lock = json.loads(path.read_text())
        candidates = json.loads((path.parent.parent / "select" / "candidates.json").read_text())
        lock.update(confirmed_index=4, confirmed_hash=candidates[4]["hash"])
        path.write_text(json.dumps(lock))
        output, plan = self.plan(fixture)
        fixture.execute(output, plan)
        verdict = sb.assemble_s14(output, self.root / "campaign", clock=lock_clock)
        self.assertTrue(verdict["accepted"], verdict.get("error"))
        report = verdict["report"]
        self.assertFalse(report["post_hoc"])
        peptides = report["admission_findings"]["Peptides-func"]
        self.assertFalse(peptides["lifecycle_lock_agrees_with_joint_decision"])
        self.assertFalse(peptides["admission_predictor_is_joint_confirmed_policy"])
        self.assertEqual(peptides["lifecycle_confirmed_hash"], candidates[4]["hash"])
        self.assertTrue(report["admission_findings"]["COCO-SP"]["lifecycle_lock_agrees_with_joint_decision"])
        self.assertTrue(any(f.startswith("Peptides-func: the lifecycle's pre-admission lock") for f in verdict["findings"]))

    def test_invalid_tests_are_refused_not_substituted(self):
        fixture = self.fixture()
        output, plan = self.plan(fixture)
        fixture.execute(output, plan, invalid=("COCO/RH",))
        with self.assertRaisesRegex(RunnerError, "no value is substituted"):
            sb.assemble_s14(output, self.root / "campaign", clock=lock_clock)
        self.assertFalse((self.root / "campaign").exists())

    def test_registration_and_job_outputs_are_immutable(self):
        fixture = self.fixture()
        output, plan = self.plan(fixture)
        with self.assertRaisesRegex(RunnerError, "has not completed"):
            sb.assemble_s14(output, self.root / "campaign", clock=lock_clock)
        fixture.execute(output, plan)
        weights = output / "registration" / "weights" / "arxiv_state.json"
        original = weights.read_bytes()
        weights.write_text(original.decode().replace(": 1", ": 5", 1))
        with self.assertRaisesRegex(RunnerError, "changed after registration"):
            sb.assemble_s14(output, self.root / "campaign", clock=lock_clock)
        weights.write_bytes(original)
        with self.assertRaisesRegex(RunnerError, "clock is behind"):
            sb.assemble_s14(output, self.root / "campaign", clock=lambda: T_PLAN)
        entry = next(j for j in plan["jobs"] if j["kind"] == "compute")
        losses = job_directory(output, entry) / "unit_losses.npz"
        data = dict(np.load(losses))
        np.savez(losses, **data)
        with self.assertRaisesRegex(RunnerError, "recorded SHA-256"):
            sb.assemble_s14(output, self.root / "campaign", clock=lock_clock)
        with self.assertRaisesRegex(RunnerError, "outside the checkout"):
            sb.assemble_s14(output, ROOT / "s14-campaign", clock=lock_clock)

    def test_job_outputs_bind_the_plan_and_the_registered_policies(self):
        fixture = self.fixture()
        output, plan = self.plan(fixture)
        fixture.execute(output, plan)
        entry = next(j for j in plan["jobs"] if j["kind"] == "read" and j["dataset"] == "PascalVOC-SP")
        marker = job_directory(output, entry) / "read.json"
        result = json.loads(marker.read_text())
        result["cells"][0]["candidate"]["hash"] = "0" * 64
        marker.write_text(json.dumps(result))
        with self.assertRaisesRegex(RunnerError, "other policies"):
            sb.assemble_s14(output, self.root / "campaign", clock=lock_clock)
        result["identities"]["plan_sha256"] = "0" * 64
        marker.write_text(json.dumps(result))
        with self.assertRaisesRegex(RunnerError, "another plan"):
            sb.assemble_s14(output, self.root / "campaign", clock=lock_clock)


# ============================================================================= execution wiring (torch-free)
class ExecutionWiringTests(Workspace):
    """run_s14 and the read job with only the label/model layer replaced (RoleContext, score_observations,
    the compute job's runtime execution): dispatch, sealed-prediction verification, outputs and resumption.
    The runtime execution itself runs in CI (ExecutionTests)."""

    def sealed_predictions(self, fixture, plan, *, fallback=()):
        """Sealed D_confirm prediction files of every read policy; each file's constant value names it.

        ``fallback``: (job index, policy key) pairs whose sealed file holds the comparator's fallback bytes."""
        lookup = {}
        for entry in plan["jobs"]:
            if entry["kind"] != "read":
                continue
            run = Path(entry["run"])
            folder = run / "predictions" / ("confirm" if entry["dataset"] in sb.GRAPH_DATASETS else "")
            folder.mkdir(parents=True, exist_ok=True)
            for policy in entry["policies"]:
                code = float(len(lookup) + 1)
                lookup[code] = (str(run.resolve()), policy["key"])
                path = folder / f"{policy['candidate']['candidate_id']}.npy"
                np.save(path, np.full((2, 2), code, dtype=np.float32))
                write(path.with_suffix(".json"), {"candidate": policy["candidate"], "prediction_sha256": sha256(path),
                                                  "execution_role": "confirm", "feasible": True,
                                                  "fallback_applied": (entry["job"], policy["key"]) in fallback})
        return lookup

    def execute(self, fixture, output, lookup):
        from unittest import mock

        class Context:
            def __init__(self, run, device, *, through):
                self.run, self.config = run, run.config
                self.collection = run.dataset in sb.GRAPH_DATASETS

            def role_ids(self, role):
                return np.arange(2)

            def labels(self, role):
                return None

            def units(self, role):
                return None

        def score(probabilities, labels, units, config):
            run, key = lookup[float(probabilities[0, 0])]
            return fixture.scores(run, [key])[key]

        def compute(plan, entry, directory, device):
            fixture.compute(entry, directory, started=ca.utc_now())

        with mock.patch.object(ca, "RoleContext", Context), \
                mock.patch("cross_morse.evaluation.score_observations", score), \
                mock.patch.object(ca, "_run_mechanism_job", compute):
            return sb.run_s14(output, device="cpu", trust=True)

    def test_run_dispatch_read_verification_and_resume(self):
        fixture = self.fixture()
        output, plan = self.plan(fixture)
        lookup = self.sealed_predictions(fixture, plan)
        with self.assertRaisesRegex(RunnerError, "trust"):
            sb.run_s14(output, device="cpu")
        done = self.execute(fixture, output, lookup)
        self.assertEqual(len(done), len(plan["jobs"]))
        self.assertEqual(sum(p.name == "read.json" for p in done), 14)
        read = json.loads(done[0].read_text())
        self.assertEqual(read["schema"], "cm-s14-sealed-contrasts-1")
        self.assertTrue(all(c["verification"]["differences_reproduced"] == "exact" for c in read["cells"]))
        self.assertEqual(len(read["sealed_predictions"]), 3)
        before = {p: sha256(p) for p in done}
        self.assertEqual(self.execute(fixture, output, {}), done)       # completed jobs are not re-executed
        self.assertEqual({p: sha256(p) for p in done}, before)
        verdict = sb.assemble_s14(output, self.root / "campaign", clock=lock_clock)
        self.assertTrue(verdict["accepted"], verdict.get("error"))

    def test_a_fallback_shuffle_of_an_invalid_sealed_contrast_is_reported_not_loaded(self):
        from cross_morse.statistics import confirmation
        fixture = self.fixture()
        run = Path(fixture.parents["ogbn-arxiv"])
        sealed = json.loads((run / "confirm" / "paired_contrasts.json").read_text())
        sealed[0].update(p_value=1.0, intervention_valid=False, invalid_reason="numerical failure; comparator bytes")
        selection = json.loads((run / "select" / "decision.json").read_text())
        decision = confirmation(selection["provisional_index"], 1, selection["eligible_indices"], sealed, reps=REPS,
                                seed=SEED + 5)
        write(run / "confirm" / "paired_contrasts.json", sealed)
        write(run / "confirm" / "decision.json", ca.json_safe(decision))
        output, plan = self.plan(fixture)
        entry = next(j for j in plan["jobs"] if j["dataset"] == "ogbn-arxiv")
        shuffle = entry["cells"][0]["reference_key"]
        lookup = self.sealed_predictions(fixture, plan, fallback={(entry["job"], shuffle)})
        self.execute(fixture, output, lookup)
        result = json.loads((job_directory(output, entry) / "read.json").read_text())
        state, radius = result["cells"]
        self.assertFalse(state["valid"])
        self.assertIn("numerical failure", state["invalid_reason"])
        self.assertTrue(radius["valid"])
        self.assertNotIn(shuffle, result["sealed_predictions"])
        with self.assertRaisesRegex(RunnerError, "no value is substituted(.|\n)*arxiv/state: .*numerical failure"):
            sb.assemble_s14(output, self.root / "campaign", clock=lock_clock)
        self.assertFalse((self.root / "campaign").exists())


# ============================================================================= command line
class CommandLineTests(Workspace):
    def call(self, *argv):
        from scripts.build_s14_campaign import main
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main([str(a) for a in argv])
        return code, out.getvalue(), err.getvalue()

    def test_plan_run_assemble_dispatch(self):
        fixture = self.fixture()
        parents = [arg for p in fixture.parents.values() for arg in ("--parent", p)]
        code, out, err = self.call("plan", *parents, "--output", self.root / "build")
        self.assertEqual(code, 0, err)
        self.assertTrue((self.root / "build" / "plan.json").is_file())
        self.assertTrue((self.root / "build" / "registration" / "protocol.json").is_file())
        code, _, err = self.call("run", "--output", self.root / "build")
        self.assertEqual(code, 2)
        self.assertIn("trust", err)
        code, _, err = self.call("assemble", "--output", self.root / "build", "--campaign", self.root / "campaign")
        self.assertEqual(code, 2)
        self.assertIn("has not completed", err)
        code, _, err = self.call("plan", "--parent", fixture.parents["COCO-SP"], "--output", self.root / "other")
        self.assertEqual(code, 2)
        self.assertIn("all seven", err)


# ============================================================================= execution (PyTorch; CI)
@unittest.skipUnless(TORCH, "PyTorch is required to execute the S14 jobs")
class ExecutionTests(unittest.TestCase):
    """Smoke lifecycles on synthetic data, then read and compute jobs (software checks, not measurements)."""

    @classmethod
    def setUpClass(cls):
        import torch
        from cross_morse.pipeline import Pipeline
        torch.set_num_threads(2)
        cls.temporary = tempfile.TemporaryDirectory(prefix="cm-s14-runs-")
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

    def plannable(self) -> list:
        """The smoke runs whose selection the S14 rows can test (a smoke run may select, e.g., an open-loop route)."""
        keep, reasons = [], []
        for name, run in self.runs.items():
            probe = Path(tempfile.mkdtemp(prefix="cm-s14-probe-", dir=self.root)) / "probe"
            try:
                sb.plan_s14([run], probe, allow_partial=True, reps=20)
                keep.append(run)
            except RunnerError as exc:
                reasons.append(f"{name}: {exc}")
        if not keep:
            self.skipTest("no smoke selection is testable by the S14 rows: " + "; ".join(reasons))
        return keep

    def execute(self, runs, **kwargs):
        output = Path(tempfile.mkdtemp(prefix="cm-s14-", dir=self.root)) / "build"
        sb.plan_s14(runs, output, allow_partial=True, reps=20, **kwargs)
        with contextlib.redirect_stdout(io.StringIO()):
            sb.run_s14(output, device="cpu", trust=True)
        return output, json.loads((output / "plan.json").read_text())

    def test_jobs_execute_and_sealed_contrasts_reproduce(self):
        runs = self.plannable()
        output, plan = self.execute(runs)
        self.assertTrue(plan["synthetic"])
        with self.assertRaisesRegex(RunnerError, "partial plan"):
            sb.assemble_s14(output, output.parent / "campaign")
        for entry in plan["jobs"]:
            marker = sb.done_marker(job_directory(output, entry), entry)
            result = json.loads(marker.read_text())
            self.assertEqual([c["contrast"] for c in result["cells"]], [c["contrast"] for c in entry["cells"]])
            for cell in result["cells"]:
                if entry["kind"] == "read" and cell["valid"]:
                    self.assertEqual(cell["verification"]["differences_reproduced"], "exact")
                if cell["valid"]:
                    member = sb.member_input(output, plan, entry, cell["contrast"])
                    self.assertGreaterEqual(member["units"], 2)
        # Forced re-execution of the same targets agrees with the sealed confirm stage where the run sealed them.
        forced, forced_plan = self.execute(runs, force_compute=True)
        for key in (f["contrast_id"] for f in plan["family"]):
            first = next(f for f in plan["family"] if f["contrast_id"] == key)["members"][0]
            second = next(f for f in forced_plan["family"] if f["contrast_id"] == key)["members"][0]
            self.assertEqual((first["left"], first["right"]), (second["left"], second["right"]))
            if first["mode"] != "read":
                continue
            a = sb.member_input(output, plan, plan["jobs"][first["job"]], key)
            b = sb.member_input(forced, forced_plan, forced_plan["jobs"][second["job"]], key)
            if a["valid"] and b["valid"]:
                np.testing.assert_allclose(a["left_loss"] - a["right_loss"], b["left_loss"] - b["right_loss"],
                                           rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
