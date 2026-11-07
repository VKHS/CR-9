"""E13: attribution of inserted-site transport (Supplementary Table S45, Figure S18a).

On the designated-site plants (GraphGPS-style PascalVOC-SP and COCO-SP, linear-GINE-style
Peptides-func; Supplementary Table S5b) the controller acts at native designated sites,
where an action replaces the native aggregation, and at parameter-free inserted sites,
where A0 is a literal bypass.  E13 separates the two by executing, on the same units
and with the same frozen ensemble members, four arms (the record-set ``arm`` key):

    native, open loop                    q_C with every inserted site held at the A0 bypass
    inserted, open loop (q_C identity)   q_C itself (paired effect zero by construction)
    native, feedback                     the confirmed feedback route q_M, inserted sites at A0
    inserted, feedback (deployed)        q_M itself

The native-site restriction is executed with the released runtime: a fixed schedule has
its inserted-site actions replaced by A0, and the controller is wrapped so that A0 is the
only admissible action at an inserted site -- for the decision, for MPC planning and for
recovery -- so the trace, the site accounting and the observer all see the executed
action.  Each arm's ensemble is scored on the declared role (default: the reserved
comparison units D_compare, the unit of Supplementary Table S42); the record holds the
paired effect against q_C and its graph-unit standard error.  Work ratios, activity,
native/inserted action counts, bootstrap intervals and the native-site fraction are kept
in the raw result next to them.
"""
from __future__ import annotations

import copy
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from .artifacts import canonical, sha256, write_json
from .ensemble_grid import (FEEDBACK_ROUTES, ParentContext, RunnerError, SealedRun, candidate_identity,
                            check_identity, check_role, code_files, create_plan, ensemble_probabilities,
                            environment_record, job_directory, json_safe, load_plan, member_runs, paired_effect,
                            provenance_hardware, provenance_period, read_json, read_member, runtime_source_sha256,
                            save_arrays, selected_jobs, slug, utc_now, write_member, write_record_set)

ARMS = ("native, open loop", "inserted, open loop (q_C identity)", "native, feedback", "inserted, feedback (deployed)")
ARM_DESIGN = ({"arm": ARMS[0], "policy": "comparator", "sites": "native"},
              {"arm": ARMS[1], "policy": "comparator", "sites": "all"},
              {"arm": ARMS[2], "policy": "feedback", "sites": "native"},
              {"arm": ARMS[3], "policy": "feedback", "sites": "all"})
DATASET_BACKBONES = {"PascalVOC-SP": "gps", "COCO-SP": "gps", "Peptides-func": "linear_gine"}
RESTRICTION = "native_designated_sites"


# ----------------------------------------------------------------------------- site layout (torch-free)
def designated_sites(config: dict) -> dict:
    """Native/inserted layout of a designated-site plant from its configuration.

    The same rule as cross_morse.core.DesignatedSitePlant and the configuration branch of
    cross_morse.core.site_transport_accounting; execution checks it against the loaded plant.
    """
    model = config["model"]
    dataset = config.get("benchmark_dataset") or config.get("dataset")
    backbone = model.get("backbone", DATASET_BACKBONES.get(dataset, "shared_transport"))
    depth = int(model["depth"])
    if backbone == "gps":
        layers = int(model.get("native_layers", 6))
        if depth < layers + 2 or (depth - 2) % layers:
            raise RunnerError("invalid GPS control-site layout")
        stride = (depth - 2) // layers
        native = [1 + i * stride for i in range(layers)]
    elif backbone == "linear_gine":
        layers = int(model.get("native_layers", 8))
        if depth < layers or depth % layers:
            raise RunnerError("invalid linear-GINE control-site layout")
        native = [i * (depth // layers) for i in range(layers)]
    else:
        raise RunnerError(f"{dataset}: backbone {backbone!r} has no inserted sites; E13 needs a designated-site plant")
    return {"backbone": backbone, "control_depth": depth, "native_sites": native,
            "inserted_sites": depth - len(native), "native_site_fraction": len(native) / depth}


def native_only(candidate: dict, native_sites) -> dict:
    """The same policy with control removed from every inserted site (A0 bypass there)."""
    native = set(int(s) for s in native_sites)
    restricted = copy.deepcopy(candidate)
    restricted["site_restriction"] = RESTRICTION
    if restricted.get("sequence") is not None:
        restricted["sequence"] = [int(a) if i in native else 0 for i, a in enumerate(restricted["sequence"])]
    identity = {k: v for k, v in restricted.items() if k not in ("hash", "candidate_id", "registry_index")}
    restricted["hash"] = canonical(identity)
    restricted["candidate_id"] = restricted["hash"][:20]
    return restricted


class NativeSiteController:
    """A fitted controller whose admissible set at an inserted site is the literal A0 bypass.

    Every other method (observer, critic, consequence models, counters) is the wrapped
    controller's own.  ``cross_morse.policies`` asks ``supported`` for the decision, the
    beam search and the recovery choice, so no other action can reach an inserted site.
    """

    def __init__(self, controller, native_sites, control_depth):
        self._controller = controller
        self._native = frozenset(int(s) for s in native_sites)
        self._depth = int(control_depth)

    def __getattr__(self, name):
        return getattr(self._controller, name)

    def supported(self, y, depth):
        allowed = self._controller.supported(y, depth)
        if int(depth) in self._native or int(depth) >= self._depth:
            return allowed
        return [0]


def feedback_policy(run: SealedRun, comparator: dict, *, allow_fixture: bool) -> tuple[dict, dict]:
    """The confirmed feedback route q_M; a synthetic fixture may name the registered MPC-RH instead."""
    confirmed = run.confirmed()
    deployed = run.deployed()
    record = {"confirmed_index": confirmed["registry_index"], "fixture_feedback_override": False,
              "feedback_is_deployed": bool(deployed is not None and deployed["registry_index"] == confirmed["registry_index"])}
    if confirmed["route"] in FEEDBACK_ROUTES:
        return confirmed, record
    if not (allow_fixture and run.synthetic):
        raise RunnerError(f"{run.root.name}: the confirmed policy is open loop ({confirmed['route']}); "
                          "E13's feedback arms need a confirmed feedback route")
    candidates = run.candidates()
    for route in ("mpc_rh", "radius", "one_step"):
        for index, c in enumerate(candidates):
            if c["route"] == route and c["ensemble"] == comparator["ensemble"] and c.get("profile") == "full":
                record.update(fixture_feedback_override=True, feedback_is_deployed=False)
                return dict(c, registry_index=index), record
    raise RunnerError(f"{run.root.name}: no registered feedback candidate at the comparator's ensemble size")


# ----------------------------------------------------------------------------- planning
def plan_attribution(parents, output, *, role="compare", reps=None, allow_fixture=False) -> Path:
    if not parents:
        raise RunnerError("give at least one --parent run")
    jobs, datasets = [], []
    for parent in parents:
        runs = [SealedRun(m) for m in member_runs(parent)]
        if len(runs) != 1:
            raise RunnerError(f"{parent}: E13 runs on the independent-graph benchmarks, one run per dataset")
        run = runs[0]
        if run.dataset not in DATASET_BACKBONES:
            raise RunnerError(f"{run.dataset}: E13 covers {sorted(DATASET_BACKBONES)}")
        if any(d["dataset"] == run.dataset for d in datasets):
            raise RunnerError(f"{run.dataset}: one parent run per dataset")
        check_role(run, role)
        layout = designated_sites(run.config)
        comparator = run.comparator()
        feedback, record = feedback_policy(run, comparator, allow_fixture=allow_fixture)
        if int(feedback["ensemble"]) != int(comparator["ensemble"]):
            raise RunnerError(f"{run.dataset}: the matched-ensemble attribution needs equal q_C and q_M member counts")
        policies = {"comparator": comparator, "feedback": feedback}
        arms = []
        for design in ARM_DESIGN:
            base = policies[design["policy"]]
            candidate = native_only(base, layout["native_sites"]) if design["sites"] == "native" else base
            arms.append({**design, "candidate": candidate, "member_key": slug(design["arm"])})
        seeds = [int(s) for s in run.config["training"]["seeds"][:int(comparator["ensemble"])]]
        jobs.append({"job": len(jobs), "dataset": run.dataset, "slug": slug(run.dataset), "member": run.root.name,
                     "run": str(run.root), "identity": run.identity(), "layout": layout, "arms": arms,
                     "comparator": candidate_identity(comparator), "feedback": candidate_identity(feedback),
                     "feedback_record": record, "member_seeds": seeds,
                     "bootstrap": {"reps": int(reps or run.config["admission"]["bootstrap_reps"]),
                                   "seed": int(run.config["seed"]) + 13}})
        datasets.append({"dataset": run.dataset, "slug": slug(run.dataset), "synthetic": run.synthetic})
    return create_plan(output, {
        "experiment": "E13", "record_set": "inserted_site", "role": role, "datasets": datasets, "jobs": jobs,
        "protocol": ("four arms on the same units and frozen members: q_C and the confirmed feedback route q_M, each "
                     "with all controlled sites and with control restricted to the native designated sites "
                     "(inserted sites held at the A0 bypass); paired effect against q_C per unit")})


# ----------------------------------------------------------------------------- statistics (torch-free)
def arm_results(arm_probabilities: dict, labels, units, config, *, reps: int, seed: int) -> tuple[list[dict], dict]:
    """Paired effect of every arm's ensemble against q_C (the identity arm) on the scored units."""
    from cross_morse.evaluation import score_observations
    if set(arm_probabilities) != set(ARMS):
        raise RunnerError(f"E13 needs exactly the arms {list(ARMS)}")
    reference = score_observations(arm_probabilities[ARMS[1]], labels, units, config)
    arrays = {"unit_id": reference["core_ids"], "unit_weight": reference["counts"], "qC_unit_loss": reference["loss"]}
    rows = []
    for index, arm in enumerate(ARMS):
        score = score_observations(arm_probabilities[arm], labels, units, config)
        if not np.array_equal(score["core_ids"], reference["core_ids"]):
            raise RunnerError(f"{arm}: scored units differ from q_C's")
        effect = paired_effect(score["loss"] - reference["loss"], reference["counts"], reps=reps, seed=seed)
        arrays[f"arm{index}_unit_loss"] = score["loss"]
        rows.append({"arm": arm, "effect_nats": effect["mean"], "standard_error": effect["standard_error"],
                     "interval_low_nats": effect["lower"], "interval_high_nats": effect["upper"], "paired": effect,
                     "loss_nats": score["node_ce"], "metric": score["metric"], "metric_name": score["metric_name"],
                     "units": int(len(score["core_ids"]))})
    return rows, arrays


def retained_share(rows: list[dict]):
    by_arm = {r["arm"]: r["effect_nats"] for r in rows}
    full = by_arm[ARMS[3]]
    return None if full == 0 else by_arm[ARMS[2]] / full


# ----------------------------------------------------------------------------- execution (torch)
def run_attribution(output, *, job=None, device="cuda", trust=False) -> list[Path]:
    if not trust:
        raise RunnerError("loading local checkpoints requires explicit trust (--trust-local-checkpoints)")
    output = Path(output).resolve()
    plan = load_plan(output, "inserted_site")
    done = []
    for entry in selected_jobs(plan, job):
        directory = job_directory(output, entry)
        if not (directory / "attribution.json").is_file():
            _run_attribution_job(plan, entry, directory, device)
        done.append(directory / "attribution.json")
    return done


def _run_attribution_job(plan, entry, directory, device):
    run = SealedRun(entry["run"])
    check_identity(run, entry["identity"])
    role = plan["role"]
    check_role(run, role)
    started = utc_now()
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / "environment.json").exists():
        write_json(directory / "environment.json", environment_record(device))
    ctx = ParentContext(run, device, through="evaluate" if role == "test" else "confirm")
    layout = entry["layout"]
    models = {}
    for seed in entry["member_seeds"]:
        model = ctx.plant(seed)
        if (getattr(model, "backbone", None) != layout["backbone"]
                or list(getattr(model, "native_sites", ())) != layout["native_sites"]
                or int(model.control_depth) != layout["control_depth"]):
            raise RunnerError("the loaded plant's designated-site layout differs from the planned layout")
        models[seed] = model
    probabilities, ledgers = {}, {}
    for arm in entry["arms"]:
        candidate = arm["candidate"]
        controller = ctx.controller_for(candidate)
        if controller is not None and arm["sites"] == "native":
            controller = NativeSiteController(controller, layout["native_sites"], layout["control_depth"])
        expected = {"candidate": candidate_identity(candidate), "role": role, "arm": arm["arm"]}
        members, ledger = [], {"deployed_work": 0.0, "executed_work": 0.0, "wall_seconds": 0.0, "feasible": True,
                               "native_site_action_counts": [0] * 10, "inserted_site_action_counts": [0] * 10,
                               "members": []}
        for seed in entry["member_seeds"]:
            path = directory / "members" / arm["member_key"] / f"seed{seed}.npy"
            if not path.exists():
                rows, totals = ctx.predict(models[seed], candidate, controller, role)
                write_member(path, rows, {**expected, "seed": int(seed), "trace": totals,
                                          "checkpoint_sha256": sha256(run.root / "train" / f"plant_seed{seed}_best.pt")})
            members.append(read_member(path, expected))
            trace = read_json(path.with_suffix(".json"))["trace"]
            ledger["deployed_work"] += float(trace.get("deployed_multiply_add_work", trace["multiply_add_work"]))
            ledger["executed_work"] += float(trace["multiply_add_work"])
            ledger["wall_seconds"] += float(trace["wall_seconds"])
            ledger["feasible"] &= bool(trace.get("feasible", True))
            for key in ("native_site_action_counts", "inserted_site_action_counts"):
                ledger[key] = [a + int(b) for a, b in zip(ledger[key], trace[key])]
            ledger["members"].append({"seed": int(seed), "file": str(path.relative_to(directory)), "sha256": sha256(path)})
        if arm["sites"] == "native" and sum(ledger["inserted_site_action_counts"][1:]):
            raise RunnerError(f"{arm['arm']}: an inserted site executed a non-bypass action")
        executed = sum(ledger["native_site_action_counts"]) + sum(ledger["inserted_site_action_counts"])
        ledger["activity"] = (executed - ledger["native_site_action_counts"][0]
                              - ledger["inserted_site_action_counts"][0]) / executed if executed else 0.0
        probabilities[arm["arm"]] = ensemble_probabilities(members)
        ledgers[arm["arm"]] = ledger
    rows, arrays = arm_results(probabilities, ctx.labels(role), ctx.units(role), ctx.config,
                               reps=entry["bootstrap"]["reps"], seed=entry["bootstrap"]["seed"])
    write_attribution_result(directory, entry, role=role, rows=rows, arrays=arrays, ledgers=ledgers,
                             parent_files=run.inputs(), started=started)


def write_attribution_result(directory, entry, *, role, rows, arrays, ledgers, parent_files, started,
                             environment=None) -> Path:
    """The job's raw outputs: unit_losses.npz and attribution.json (written last: the completion marker)."""
    directory = Path(directory)
    if environment is not None and not (directory / "environment.json").exists():
        write_json(directory / "environment.json", json_safe(environment))
    layout = entry["layout"]
    reference_work = ledgers[ARMS[1]]["deployed_work"]
    design = {a["arm"]: a for a in entry["arms"]}
    for row in rows:
        ledger = ledgers[row["arm"]]
        row.update(policy=design[row["arm"]]["policy"], sites=design[row["arm"]]["sites"],
                   candidate=candidate_identity(design[row["arm"]]["candidate"]),
                   work_ratio=ledger["deployed_work"] / reference_work if reference_work else None, **ledger)
    unit_file = save_arrays(directory / "unit_losses.npz", **arrays)
    write_json(directory / "attribution.json", json_safe({
        "schema": "cm-inserted-site-attribution-1", "experiment": "E13", "dataset": entry["dataset"],
        "member": entry["member"], "role": role, "layout": layout, "member_seeds": entry["member_seeds"],
        "comparator": entry["comparator"], "feedback": entry["feedback"], "feedback_record": entry["feedback_record"],
        "arms": rows, "retained_share_native_feedback": retained_share(rows),
        "unit_losses": {"file": "unit_losses.npz", "sha256": unit_file},
        "identities": {"runtime_source_sha256": runtime_source_sha256(), "parent": entry["identity"],
                       "plan_sha256": sha256(directory.parents[2] / "plan.json"), "parent_files_read": parent_files},
        "definitions": {"effect_nats": "unit-weighted mean of per-unit loss(arm) - loss(q_C) on the role",
                        "standard_error": "linearized unit standard error of that mean (sd/sqrt(n) for graphs)",
                        "work_ratio": "deployed multiply-add work of the arm's members over q_C's (matched-cost check)",
                        "activity": "share of executed actions other than A0 over all sites, members and inputs"},
        "started_utc": started, "finished_utc": utc_now()}))
    return directory / "attribution.json"


# ----------------------------------------------------------------------------- export (torch-free)
def export_attribution(output, *, root=None, replace=False, allow_partial=False, hardware=None, period=None,
                       command=None) -> Path:
    output = Path(output).resolve()
    plan = load_plan(output, "inserted_site")
    files, rows, results, environments, inputs = code_files(output), [], [], [], []
    for entry in plan["jobs"]:
        directory = job_directory(output, entry)
        path = directory / "attribution.json"
        if not path.is_file():
            raise RunnerError(f"{entry['dataset']}: job {entry['job']} has not completed")
        result = read_json(path)
        if [a["arm"] for a in result["arms"]] != list(ARMS) or result["role"] != plan["role"]:
            raise RunnerError(f"{path}: arms or role differ from the plan")
        base = f"raw/{entry['slug']}"
        files[f"{base}/attribution.json"] = path
        files[f"{base}/unit_losses.npz"] = directory / "unit_losses.npz"
        files[f"{base}/environment.json"] = directory / "environment.json"
        for i, arm in enumerate(result["arms"]):
            for quantity in ("effect_nats", "standard_error"):
                rows.append({"dataset": entry["dataset"], "arm": arm["arm"], "quantity": quantity,
                             "value": arm[quantity], "unit": "nats", "source_file": f"{base}/attribution.json",
                             "locator": f"json:/arms/{i}/{quantity}"})
        results.append(result)
        environments.append(read_json(directory / "environment.json"))
        inputs.append({"file": str(path.relative_to(output)), "sha256": sha256(path)})
    synthetic = {d["synthetic"] for d in plan["datasets"]}
    if len(synthetic) != 1:
        raise RunnerError("synthetic and benchmark parents cannot share one record set")
    provenance = {
        "experiment": "E13 three-arm attribution of inserted-site transport (Supplementary Table S45, Figure S18a)",
        "produced_by": "scripts/run_experiment.py inserted-site plan/run/export (cm_experiments/attribution.py)",
        "hardware": hardware or provenance_hardware(environments), "period": period or provenance_period(results),
        "protocol": (f"q_C and the confirmed feedback route executed on the {plan['role']} role with the same frozen "
                     "members, with all controlled sites and with control restricted to the native designated sites "
                     "(inserted sites held at the literal A0 bypass by the schedule and by the controller's admissible "
                     "set); paired per-unit effect against q_C with its unit standard error; work ratios in raw/"),
        "synthetic": synthetic.pop(), "role": plan["role"], "plan_sha256": sha256(output / "plan.json")}
    return write_record_set("inserted_site", rows, files, provenance, root=root, replace=replace,
                            allow_partial=allow_partial, command=command, inputs=inputs)
