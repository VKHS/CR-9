"""Produce the joint S14 confirmation campaign (Supplementary Table S14) from sealed lifecycle runs.

The nine tests of ``cm_campaigns.confirmation.FAMILY`` are computed with the released runtime's own
confirm-stage code (``cross_morse.pipeline.Pipeline._stage_confirm`` for the paired per-unit contrasts,
``cross_morse.statistics.confirmation`` for their statistics) on D_confirm of the seven sealed, complete
lifecycle runs (Roman-empire: its campaign of ten official-split runs), and written in the campaign format
that ``cm_campaigns.confirmation.check_campaign`` validates (``campaign:confirmation``).

Target of each dataset (fixed from the selection record; nothing measured on D_confirm enters it)
    q        the provisional policy of Eq. (19) when the run selected a feedback route (the S14 map tests the
             selected policy); when the run selected q_C, the strongest feedback configuration of
             cm_experiments.confirmation_analysis.strongest_feedback (q_C selected: the feedback candidate
             with the lowest D_select upper limit U, joint-feasible first, registered tie order), i.e. an
             unselected, prespecified feedback target as in Table S14's products row.
    state    left = the matched-shuffle state control of q (controller observations destroyed), right = q
    radius   left = q when q is a radius, else the radius at q's ensemble size and profile with the lowest U;
             right = its registered one-step predecessor (cross_morse.pipeline.Pipeline.predecessor)
    mpc      left = MPC-RH (confirmation_analysis.replanning_pair: q itself when q is MPC-RH), right = its
             MPC-OL predecessor at the same horizon, profile and ensemble size

Source of each contrast (per run; a Roman split decides for itself)
    read     the target belongs to the run's confirmation closure (confirm/contrast_registry.json): the sealed
             confirm/paired_contrasts.json row of that kind and candidate IS the contrast.  Its two sides are
             rescored from the sealed predictions on D_confirm exactly as the confirm stage's role_scores
             scores them and must reproduce the sealed per-unit differences and weights bit for bit; its
             statistics are the sealed confirm/decision.json row, replayed from the sealed contrasts with
             cross_morse.statistics.confirmation.
    compute  otherwise: the contrast is executed on D_confirm with the mechanism-contrast machinery of
             cm_experiments.confirmation_analysis (RoleContext, member-by-member runtime execution,
             mechanism_statistics -> cross_morse.statistics.confirmation), job by job.

Tests (unit counts are those of FAMILY: 32/32/64 buffered clusters, 10 official splits, 425/5,664/544 graphs)
    clusters, graphs   the confirm stage's paired whole-unit bootstrap of the runtime's per-unit differences
                       (cluster weights = scored-sample counts, graphs equal): effect = weighted mean, 95%
                       interval = 2.5%/97.5% bootstrap percentiles, P = one-sided null-centred bootstrap P
                       against zero in the beneficial direction (p_greater_zero for state, p_less_zero for
                       radius/RH), P vs margin = the runtime's one-sided bootstrap P against the margin.
    Roman splits       d_k = loss_k(MPC-RH) - loss_k(MPC-OL), the D_confirm loss of split k being the
                       count-weighted mean of its per-core losses; effect = mean of the ten d_k; one-sided
                       paired Student t test against zero (9 df) and the two-sided 95% small-sample t
                       interval over splits (confirmation_analysis.t_interval); P vs margin: one-sided t.

The runtime's contrasts are contrasts of E-member ensemble predictions: one paired observation per unit.
The campaign's seed axis therefore has one level, ``ensemble`` (the policy's registered members, training
seeds[:E]); seeds are neither resampled nor counted as units.

Workflow (scripts/build_s14_campaign.py)
    plan      sealed parents -> <output>/plan.json and <output>/registration/ (protocol.json with registration
              time, target policies, units, weights, methods and the hashed test code; reserved-role
              manifests; the Roman aggregate ledger).  Label-free.
    run       one job per (run, source): read jobs rescore sealed predictions, compute jobs execute policies.
    assemble  -> the campaign directory: original test records (raw/<key>/test.json, paired.csv), roles,
              selections, original run bindings, the joint Holm decision lock and the admission links;
              then check_campaign's verdict is written next to it (<output>/campaign_check.json).

Protocol position.  Built on complete runs, the joint lock is fixed after the lifecycle's own admission
stages ran.  The admission links record that history: their admission start precedes the lock and they name
no governing joint lock.  The campaign declares production.kind = "repository_built", for which
check_campaign evaluates the admission-link conditions and reports them (admission_findings, post_hoc) instead
of requiring them; every other check is the one imported campaigns pass.  Admission cannot follow the joint
lock with the released runtime: the joint lock binds each run's admit/pre_admission_lock.json (through the run
bindings) and the runtime's admit stage writes that file in the same call that opens the D_admit labels.  The
links also record whether the lifecycle's own pre-admission lock (the run's local confirmation family) names
the joint decision's policies.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from cm_campaigns.common import canonical, sha256, write_bytes, write_json
from cm_campaigns.confirmation import (DATASETS, ENSEMBLE_SEED_LEVEL, FAMILY, LOCK_SCHEMA, PROTOCOL_SCHEMA,
                                       REPOSITORY_BUILT, RESULT_SCHEMA, ROLE_SCHEMA, ROLES, SCHEMA,
                                       composite_policy_hash)
from cm_experiments import confirmation_analysis as ca
from cm_experiments.ensemble_grid import (FEEDBACK_ROUTES, RunnerError, SealedRun, candidate_identity, check_identity,
                                          create_plan, environment_record, job_directory, json_safe, load_plan,
                                          member_runs, public_plan, read_json, runtime_source_sha256, save_arrays,
                                          selected_jobs, slug, utc_now)

ROOT = Path(__file__).resolve().parents[1]
RECORD_SET = "campaign:confirmation"
EXPERIMENT = "S14 joint confirmation"
CONFIRM = "confirm"
SEED_LEVEL = ENSEMBLE_SEED_LEVEL
STATE_SCOPE = "controller_observations_only"
BUILDER_FILE = "cm_campaigns/s14_builder.py"
CODE_FILES = (BUILDER_FILE, "scripts/build_s14_campaign.py", "cm_experiments/confirmation_analysis.py",
              "cm_experiments/ensemble_grid.py", "cm_campaigns/confirmation.py", "cm_campaigns/common.py")
ENVIRONMENT_FILES = ("requirements.txt", "environment/requirements-cpu.lock", "environment/apptainer.def")
GRAPH_DATASETS = ("PascalVOC-SP", "COCO-SP", "Peptides-func")
ROMAN = "Roman-empire"
ROMAN_SPLITS = 10
ROUTE_OF = {"radius": "radius", "mpc": "mpc_rh"}
SIMPLER_OF = {"radius": "one_step", "mpc": "mpc_ol"}
METHOD_NAMES = {
    "paired_cluster": "paired cluster bootstrap of the confirm stage, one-sided",
    "paired_unit": "paired cluster bootstrap of the confirm stage, one-sided",
    "graph_bootstrap": "graph bootstrap of the confirm stage, one-sided",
    "paired_small_sample": "paired small-sample t test over the official splits, one-sided"}
COMMAND = ("python -B scripts/build_s14_campaign.py plan --parent <run> ... --output <dir>; "
           "python -B scripts/build_s14_campaign.py run --output <dir> [--job K] --device cuda "
           "--trust-local-checkpoints; python -B scripts/build_s14_campaign.py assemble --output <dir> "
           "--campaign <campaign dir>")
SEED_AXIS = ("one level, 'ensemble': the runtime's confirm-stage contrast is a contrast of E-member ensemble "
             "predictions (mean member probabilities over training seeds[:E]), one paired observation per unit; "
             "seeds are an averaging axis inside the ensemble, never resampled and never counted as units")
TARGET_RULE = ("Target q of a dataset, fixed from the selection record before D_confirm is read: the provisional "
               "policy of Eq. (19) when the run selected a feedback route; when it selected q_C, the feedback "
               "candidate with the lowest D_select upper limit U (joint-feasible first, registered tie order; "
               "cm_experiments.confirmation_analysis.strongest_feedback).  state: matched-shuffle control of q minus "
               "q.  radius: q (or the lowest-U radius at q's ensemble size and profile) minus its registered "
               "one-step predecessor.  RH: MPC-RH (q, or the lowest-U MPC-RH at q's ensemble size and profile; "
               "confirmation_analysis.replanning_pair) minus its MPC-OL predecessor.  Roman-empire: per official "
               "split, aggregated by composite_policy_hash.")


# ============================================================================= small helpers
def _now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _time(value) -> datetime:
    if isinstance(value, datetime):
        result = value
    else:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() is None:
        raise RunnerError(f"timestamp without a time zone: {value!r}")
    return result


def _iso(value: datetime) -> str:
    return _time(value).isoformat()


def _clock(clock):
    stamp = _time((clock or _now)())
    return stamp.replace(microsecond=0)


def key_slug(key: str) -> str:
    return slug(key)


def code_identity() -> dict:
    """The registered test code: the builder, its CLI and the reused campaign/runner modules."""
    return {name: sha256(ROOT / name) for name in CODE_FILES if (ROOT / name).is_file()}


def state_control_hash(intact_sha256: str) -> str:
    """Identity of the matched-shuffle state control of a registered feedback policy.

    The runtime's control carries the intact candidate's hash (cross_morse.pipeline: dict(c,
    state_control="matched_shuffle", candidate_id="shuffle_" + id)); S14 names the two policies of a state
    contrast separately, so the destroyed policy is identified by this derived hash.
    """
    return hashlib.sha256(canonical({"intervention": "matched_shuffle", "intervention_scope": STATE_SCOPE,
                                     "intact_policy_sha256": intact_sha256})).hexdigest()


def _ref(path: str, data: bytes) -> dict:
    return {"path": path, "sha256": hashlib.sha256(data).hexdigest()}


def _json_bytes(value) -> bytes:
    return canonical(json_safe(value))


def _compact_bytes(value) -> bytes:
    return (json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                       allow_nan=False) + "\n").encode()


def stage_time(run: SealedRun, stage: str) -> datetime:
    """Seal time of a lifecycle stage (its manifest's created_utc)."""
    from cm_experiments.ensemble_grid import STAGES
    value = run.json(f"provenance/stages/{STAGES.index(stage):02d}_{stage}.json").get("created_utc")
    if not value:
        raise RunnerError(f"{run.root.name}: the {stage} stage manifest records no seal time")
    return _time(value)


def run_binding_identity(run: SealedRun) -> dict:
    """source/config/unit identities as the sealed stage manifests record them (NumericEvidence.identities)."""
    manifest = run.json("provenance/stages/05_confirm.json")
    identity = run.identity()
    out = {"source_sha256": manifest.get("source_sha256"), "config_sha256": manifest.get("config_sha256"),
           "unit_sha256": manifest.get("unit_sha256")}
    if (out["source_sha256"], out["config_sha256"], out["unit_sha256"]) != (
            identity["runtime_source_sha256"], identity["config_sha256"], identity["units_sha256"]):
        raise RunnerError(f"{run.root.name}: the confirm-stage manifest identities differ from the run's files")
    return out


def split_index(run: SealedRun):
    return run.config.get("data", {}).get("split_index") if run.dataset == ROMAN else None


# ============================================================================= targets and pairs (label-free)
def s14_target(run: SealedRun) -> tuple[dict, str]:
    """The dataset's S14 target q (TARGET_RULE).

    confirmation_analysis.strongest_feedback looks at the deployed and confirmed policies first; those are
    outcomes of D_confirm/D_admit and, after a downgrade to a feedback predecessor, differ from the selected
    policy that the S14 map (check_campaign._policy_decision) requires to be tested.  The selection-time part
    of that rule is used: the provisional policy as its declared anchor, or its D_select rule when q_C was
    selected (then the deployed and confirmed policies are q_C as well).
    """
    candidates = run.candidates()
    selection = run.json("select/decision.json")
    qc, provisional = int(selection["comparator_index"]), int(selection["provisional_index"])
    if provisional != qc:
        chosen = candidates[provisional]
        if chosen["route"] not in FEEDBACK_ROUTES or chosen.get("execution_failure"):
            raise RunnerError(f"{run.root.name}: the provisional policy {provisional} ({chosen['route']}) is not an "
                              "executed feedback route; the S14 state/radius/RH tests of the selected policy are undefined")
        anchor, _ = ca.strongest_feedback(run, provisional)
        return anchor, "provisional policy of Eq. (19) (selection record; the S14 map tests the selected policy)"
    anchor, rule = ca.strongest_feedback(run)
    if not rule.startswith("lowest D_select upper limit"):
        raise RunnerError(f"{run.root.name}: q_C was selected but a {anchor['route']} policy was confirmed or deployed; "
                          "the run's records are inconsistent")
    return anchor, "q_C selected; unselected prespecified feedback target: " + rule


def radius_pair(run: SealedRun, anchor: dict) -> tuple[dict, dict, str]:
    """A radius configuration at the anchor's ensemble size and profile and its one-step predecessor.

    The radius counterpart of cm_experiments.confirmation_analysis.replanning_pair (same rule)."""
    candidates = run.candidates()
    if anchor["route"] == "radius":
        target, rule = anchor, "the anchor itself"
    else:
        selection = run.json("select/decision.json")
        upper = selection["upper"]
        joint = selection.get("joint_feasible") or [False] * len(candidates)
        pool = [i for i, c in enumerate(candidates)
                if c["route"] == "radius" and c["ensemble"] == anchor["ensemble"] and c["profile"] == anchor["profile"]
                and not c.get("execution_failure") and upper[i] is not None]
        if not pool:
            raise RunnerError(f"{run.root.name}: no executed radius at E={anchor['ensemble']}, profile {anchor['profile']}")
        best = min(pool, key=lambda i: (not joint[i], float(upper[i]), ca._tie_key(candidates[i], i)))
        target = dict(candidates[best], registry_index=best)
        rule = ("radius at the anchor's ensemble size and profile with the lowest D_select upper limit U "
                "(joint-feasible first; registered tie order)")
    one = ca.predecessor(candidates, target["registry_index"], route="one_step")
    if one is None:
        raise RunnerError(f"{run.root.name}: radius {target['registry_index']} has no registered one-step predecessor")
    return target, dict(candidates[one], registry_index=one), rule


def family_pair(run: SealedRun, key: str, anchor: dict, anchor_rule: str) -> dict:
    """The (candidate, reference) policies of one FAMILY test on one run, as a mechanism-contrast cell."""
    dataset, kind = FAMILY[key][:2]
    selection = run.json("select/decision.json")
    selected = int(selection["provisional_index"]) != int(selection["comparator_index"])
    if kind == "state":
        cell = {"candidate": anchor, "reference": ca.shuffle_control(anchor), "policies": ["intact", "destroyed"],
                "pair_rule": "matched-shuffle state control of the target (confirm stage)",
                "definition": "loss(matched-shuffle state control) - loss(intact) per unit"}
    else:
        if kind == "mpc":
            target, simpler, rule = ca.replanning_pair(run, anchor)
        else:
            target, simpler, rule = radius_pair(run, anchor)
        if selected and target["registry_index"] != anchor["registry_index"]:
            raise RunnerError(f"{key}: {run.root.name} selected a {anchor['route']} policy; the S14 {kind} test must "
                              f"test the selected policy, which is not a {ROUTE_OF[kind]} route")
        if (target["route"], simpler["route"]) != (ROUTE_OF[kind], SIMPLER_OF[kind]):
            raise RunnerError(f"{key}: the pair is not {ROUTE_OF[kind]} and its registered {SIMPLER_OF[kind]}")
        cell = {"candidate": target, "reference": simpler, "policies": [ROUTE_OF[kind], SIMPLER_OF[kind]],
                "pair_rule": rule, "definition": f"loss({ROUTE_OF[kind]}) - loss({SIMPLER_OF[kind]}) per unit, "
                                                 "same profile and ensemble size"}
    return {"contrast": key, "kind": kind, "anchor_rule": anchor_rule, **cell}


def left_right(cell: dict) -> tuple[dict, dict]:
    """(left, right) as S14 orders them: state = destroyed minus intact; radius/RH = candidate minus predecessor."""
    return (cell["reference"], cell["candidate"]) if cell["kind"] == "state" else (cell["candidate"], cell["reference"])


def policy_hash(policy: dict) -> str:
    return state_control_hash(policy["hash"]) if policy.get("state_control") else policy["hash"]


# ============================================================================= reserved roles and units (label-free)
def _role_ids(run: SealedRun, graphs: bool) -> dict:
    arrays = run.npz("prepared/graph_splits.npz" if graphs else "prepared/splits.npz")
    missing = [role for role in ROLES if role not in arrays]
    if missing:
        raise RunnerError(f"{run.root.name}: prepared roles lack {missing}")
    return {role: np.asarray(arrays[role], dtype=np.int64) for role in ROLES}


def confirmation_units(run: SealedRun, dataset: str) -> dict:
    """{unit ID: [observation IDs]} of D_confirm and the prespecified unit weights (None: equal)."""
    if dataset in GRAPH_DATASETS:
        graphs = _role_ids(run, True)["confirm"]
        return {"units": {str(int(g)): [str(int(g))] for g in sorted(graphs)}, "weights": None}
    ids = _role_ids(run, False)["confirm"]
    clusters = np.asarray(run.npz("prepared/clusters.npz")["cluster_ids"], dtype=np.int64)[ids]
    units = {}
    for node, cluster in sorted(zip(ids.tolist(), clusters.tolist()), key=lambda pair: (pair[1], pair[0])):
        units.setdefault(str(int(cluster)), []).append(str(int(node)))
    weights = {unit: len(members) for unit, members in units.items()}
    return {"units": units, "weights": weights}


def role_manifest(dataset: str, runs: list, reserved_at: datetime) -> dict:
    """cm-s14-data-roles-1 of a dataset (Roman: split-qualified IDs over the ten split runs)."""
    graphs = dataset in GRAPH_DATASETS
    sources = {}
    if dataset == ROMAN:
        roles = {role: [] for role in ROLES}
        units = {}
        for run in runs:
            k = split_index(run)
            arrays = _role_ids(run, False)
            for role in ROLES:
                roles[role].extend(f"split={k}:{int(i)}" for i in arrays[role])
            units[f"split={k}"] = [f"split={k}:{int(i)}" for i in arrays["confirm"]]
            sources[f"split={k}"] = {"prepared/splits.npz": sha256(run.root / "prepared" / "splits.npz")}
        id_rule = "split-qualified decimal node IDs 'split=<k>:<node>' of prepared/splits.npz; unit = official split"
    else:
        run = runs[0]
        roles = {role: [str(int(i)) for i in ids] for role, ids in _role_ids(run, graphs).items()}
        units = confirmation_units(run, dataset)["units"]
        files = ["prepared/graph_splits.npz"] if graphs else ["prepared/splits.npz", "prepared/clusters.npz"]
        sources[run.root.name] = {name: sha256(run.root / name) for name in files}
        id_rule = ("decimal graph IDs of prepared/graph_splits.npz; unit = confirmation graph" if graphs else
                   "decimal sample (node) IDs of prepared/splits.npz; unit = prepared cluster of prepared/clusters.npz")
    return {"schema": ROLE_SCHEMA, "dataset": dataset, "reserved_at_stage": "prepare", "reserved_at": _iso(reserved_at),
            "roles": roles, "confirmation_units": units,
            "identifiers": id_rule, "source_files_sha256": sources,
            "derivation": "the lifecycle's prepared role arrays (prepare stage), read without labels; fit = fit_train + "
                          "fit_val; reserved_at = the prepare stage's seal time (latest over split runs)"}


# ============================================================================= Roman aggregate ledger (label-free)
def _aggregate_entry(parts: list[tuple[int, dict]]) -> dict:
    routes = {c["route"] for _, c in parts}
    loops = {bool(c["is_open_loop"]) for _, c in parts}
    if len(routes) != 1 or len(loops) != 1:
        raise RunnerError("the ten Roman split policies of one S14 identity differ in route or open-loop status "
                          f"({sorted(routes)}); check_campaign's aggregate identity needs one route")
    split_hashes = [{"split_index": int(k), "hash": c["hash"]} for k, c in parts]
    entry = {"hash": composite_policy_hash(split_hashes), "route": routes.pop(), "is_open_loop": loops.pop(),
             "split_policy_hashes": split_hashes}
    for field in ("profile", "ensemble", "horizon", "kappa", "complexity"):
        values = {json.dumps(c.get(field), sort_keys=True) for _, c in parts}
        if len(values) == 1 and parts[0][1].get(field) is not None:
            entry[field] = parts[0][1][field]
    return entry


def roman_aggregate(members: list[dict]) -> tuple[list, dict, dict]:
    """Identity wrapper of the ten split ledgers (check_campaign._run_bindings).

    Aggregate candidates: every registry identity present in all ten split ledgers, plus the split-wise
    comparator, provisional and tested tuples.  An aggregate is eligible when every split component lies in its
    split's Q_s.  Returns (ledger, selection record, {role: aggregate hash}).
    """
    ledgers = [m["run"].candidates() for m in members]
    selections = [m["run"].json("select/decision.json") for m in members]
    splits = [m["split_index"] for m in members]
    tuples = {
        "comparator": [(k, ledger[int(s["comparator_index"])]) for k, ledger, s in zip(splits, ledgers, selections)],
        "provisional": [(k, ledger[int(s["provisional_index"])]) for k, ledger, s in zip(splits, ledgers, selections)],
        "left": [(k, m["left"]) for k, m in zip(splits, members)],
        "right": [(k, m["right"]) for k, m in zip(splits, members)]}
    shared = [c for c in ledgers[0] if all(any(o["hash"] == c["hash"] for o in other) for other in ledgers[1:])]
    entries, order = {}, []
    for candidate in shared:
        parts = [(k, next(o for o in ledger if o["hash"] == candidate["hash"])) for k, ledger in zip(splits, ledgers)]
        entry = _aggregate_entry(parts)
        if entry["hash"] not in entries:
            entries[entry["hash"]] = entry
            order.append(entry["hash"])
    roles = {}
    for role, parts in tuples.items():
        entry = _aggregate_entry(parts)
        roles[role] = entry["hash"]
        if entry["hash"] not in entries:
            entries[entry["hash"]] = entry
            order.append(entry["hash"])
    ledger = [entries[h] for h in order]
    eligible = []
    for i, entry in enumerate(ledger):
        ok = True
        for part, selection, split_ledger in zip(entry["split_policy_hashes"], selections, ledgers):
            allowed = {split_ledger[int(j)]["hash"] for j in selection["eligible_indices"]}
            ok &= part["hash"] in allowed
        if ok:
            eligible.append(i)
    index = {h: i for i, h in enumerate(order)}
    decision = {"schema": "cm-s14-roman-aggregate-selection-1",
                "comparator_index": index[roles["comparator"]], "provisional_index": index[roles["provisional"]],
                "eligible_indices": eligible,
                "split_selection_sha256": {str(m["split_index"]): sha256(m["run"].root / "select" / "decision.json")
                                           for m in members},
                "split_candidates_sha256": {str(m["split_index"]): sha256(m["run"].root / "select" / "candidates.json")
                                            for m in members},
                "construction": "identity wrapper of the ten split selections: comparator and provisional are the "
                                "split-wise tuples; an aggregate is eligible when every split component lies in its "
                                "split's Q_s (cm_campaigns.confirmation.composite_policy_hash)"}
    return ledger, decision, roles


# ============================================================================= planning
def _registration_code(staging: Path, registered: str) -> dict:
    files = {}
    for name in CODE_FILES:
        path = ROOT / name
        if path.is_file():
            data = path.read_bytes()
            write_bytes(staging / "code" / name, data)
            files[name] = _ref(f"code/{name}", data)
    environment = {"schema": "cm-s14-environment-1", "declared_at": registered,
                   "planning_host": environment_record("cpu"),
                   "repository_environment_files": {name: sha256(ROOT / name) for name in ENVIRONMENT_FILES
                                                    if (ROOT / name).is_file()},
                   "execution_environments": "recorded per job (jobs/<dataset>/<member>/environment.json) and "
                                             "deposited under raw/<test>/members/ with each test record"}
    data = _json_bytes(environment)
    write_bytes(staging / "code" / "environment.json", data)
    files["environment"] = _ref("code/environment.json", data)
    return files


def _job(jobs, kind, run, dataset, cells, *, reps, bootstrap_seed):
    selection = run.json("select/decision.json")
    policies = {}
    for cell in cells:
        for candidate in (cell["candidate"], cell["reference"]):
            entry = {"key": ca._policy_key(candidate), "candidate": candidate,
                     "seeds": [int(s) for s in run.config["training"]["seeds"][:int(candidate["ensemble"])]],
                     "uses_donors": bool(candidate.get("state_control"))}
            policies[entry["key"]] = entry
    job = {"job": len(jobs), "kind": kind, "dataset": dataset, "slug": slug(dataset),
           "member": f"{run.root.name}__{kind}", "run_name": run.root.name, "run": str(run.root),
           "identity": run.identity(), "split_index": split_index(run), "role": CONFIRM,
           "cells": [ca._plan_cell(c) for c in cells], "policies": list(policies.values()),
           "selection": {k: selection[k] for k in ("comparator_index", "provisional_index", "eligible_indices")},
           "bootstrap": {"reps": int(reps or run.config["selection"]["bootstrap_reps"]), "seed": int(bootstrap_seed)}}
    jobs.append(job)
    return job


def plan_s14(parents, output, *, allow_partial=False, force_compute=False, reps=None, clock=None) -> Path:
    """Register the S14 protocol and the execution jobs from sealed, complete parent runs (label-free)."""
    if not parents:
        raise RunnerError("give the seven sealed parents with --parent (Roman-empire: its campaign directory)")
    groups = {}
    for parent in parents:
        runs = [SealedRun(m) for m in member_runs(parent)]
        dataset = runs[0].dataset
        if len({r.dataset for r in runs}) != 1 or dataset in groups:
            raise RunnerError(f"{parent}: one parent (run or campaign) per dataset")
        if dataset not in DATASETS:
            raise RunnerError(f"{dataset} has no test in the S14 family")
        for run in runs:
            ca.require_complete(run)
            ca.check_label_role(run, CONFIRM)
        if dataset == ROMAN:
            runs = sorted(runs, key=lambda r: -1 if split_index(r) is None else int(split_index(r)))
        groups[dataset] = (Path(parent).resolve(), runs)
    missing = [d for d in DATASETS if d not in groups]
    if missing and not allow_partial:
        raise RunnerError(f"the joint family needs all seven datasets; missing {missing} (--allow-partial: software check)")
    synthetic = {r.synthetic for _, runs in groups.values() for r in runs}
    if len(synthetic) != 1:
        raise RunnerError("synthetic and benchmark parents cannot share one campaign")
    synthetic = synthetic.pop()
    if force_compute and not synthetic:
        raise RunnerError("--force-compute is a software check on synthetic parents: a sealed contrast is read")
    if ROMAN in groups:
        indices = [split_index(r) for r in groups[ROMAN][1]]
        if len(set(indices)) != len(indices) or any(i is None for i in indices):
            raise RunnerError("Roman-empire split runs need distinct official split indices")
        if not synthetic and not allow_partial and sorted(indices) != list(range(ROMAN_SPLITS)):
            raise RunnerError(f"Roman-empire needs the ten official splits 0..9; found {sorted(indices)}")
    latest = max(stage_time(r, stage) for _, runs in groups.values() for r in runs for stage in ("prepare", "select"))
    if clock is None:
        _after_registration(_iso(latest))      # registration time has one-second resolution
    registered = _clock(clock)
    jobs, datasets, family = [], [], []
    staging = Path(tempfile.mkdtemp(prefix="cm-s14-registration-"))
    try:
        code = _registration_code(staging, _iso(registered))
        pairs_by_key = {}
        for dataset in DATASETS:
            if dataset not in groups:
                continue
            parent, runs = groups[dataset]
            keys = [k for k, spec in FAMILY.items() if spec[0] == dataset]
            members = []
            for run in runs:
                anchor, anchor_rule = s14_target(run)
                closure = {int(i) for i in run.json("confirm/contrast_registry.json")["candidate_indices"]}
                cells = [family_pair(run, key, anchor, anchor_rule) for key in keys]
                read = [c for c in cells if not force_compute and int(c["candidate"]["registry_index"]) in closure]
                compute = [c for c in cells if c not in read]
                bootstrap_seed = int(run.config["seed"]) + 5
                job_ids = {}
                if read:
                    job_ids["read"] = _job(jobs, "read", run, dataset, read, reps=None, bootstrap_seed=bootstrap_seed)["job"]
                if compute:
                    job_ids["compute"] = _job(jobs, "compute", run, dataset, compute, reps=reps,
                                              bootstrap_seed=bootstrap_seed)["job"]
                for cell in cells:
                    left, right = left_right(cell)
                    mode = "read" if cell in read else "compute"
                    pairs_by_key.setdefault(cell["contrast"], []).append({
                        "member": run.root.name, "split_index": split_index(run), "mode": mode, "job": job_ids[mode],
                        "left": dict(candidate_identity(left), state_control=left.get("state_control")),
                        "right": dict(candidate_identity(right), state_control=right.get("state_control")),
                        "left_policy_sha256": policy_hash(left), "right_policy_sha256": policy_hash(right),
                        "anchor_rule": anchor_rule, "pair_rule": cell["pair_rule"], "run": run, "cell": cell})
                members.append({"run": run, "split_index": split_index(run), "anchor": anchor})
            reserved = max(stage_time(r, "prepare") for r in runs)
            locked = max(stage_time(r, "select") for r in runs)
            for stamp, what in ((reserved, "prepare"), (locked, "select")):
                if registered <= stamp:
                    raise RunnerError(f"{dataset}: registration ({_iso(registered)}) must follow the {what} seal ({_iso(stamp)})")
            roles = role_manifest(dataset, runs, reserved)
            role_bytes = _compact_bytes(roles)
            write_bytes(staging / "roles" / f"{dataset}.json", role_bytes)
            entry = {"dataset": dataset, "slug": slug(dataset), "parent": str(parent), "synthetic": synthetic,
                     "members": [{"name": m["run"].root.name, "run": str(m["run"].root), "split_index": m["split_index"],
                                  "identity": m["run"].identity(), "binding": run_binding_identity(m["run"])}
                                 for m in members],
                     "role_manifest": _ref(f"roles/{dataset}.json", role_bytes),
                     "reserved_at": _iso(reserved), "selection_locked_at": _iso(locked),
                     "unit_count": len(roles["confirmation_units"])}
            if dataset == ROMAN:
                ordered = [{"split_index": p["split_index"], "run": p["run"], "left": left_right(p["cell"])[0],
                            "right": left_right(p["cell"])[1]} for p in pairs_by_key["Roman/RH"]]
                ledger, decision, aggregate = roman_aggregate(ordered)
                ledger_bytes, decision_bytes = _json_bytes(ledger), _json_bytes(decision)
                write_bytes(staging / "selection" / dataset / "candidates.json", ledger_bytes)
                write_bytes(staging / "selection" / dataset / "decision.json", decision_bytes)
                entry.update(selection=_ref(f"selection/{dataset}/decision.json", decision_bytes),
                             candidates=_ref(f"selection/{dataset}/candidates.json", ledger_bytes),
                             aggregate=aggregate, selection_source="registration")
            else:
                run = runs[0]
                entry.update(selection={"path": f"selection/{dataset}/decision.json",
                                        "sha256": sha256(run.root / "select" / "decision.json")},
                             candidates={"path": f"selection/{dataset}/candidates.json",
                                         "sha256": sha256(run.root / "select" / "candidates.json")},
                             selection_source="sealed run (select/decision.json, select/candidates.json)")
            datasets.append(entry)
        protocol = {"schema": PROTOCOL_SCHEMA, "family": "Supplementary Table S14", "registered_at": _iso(registered),
                    "alpha": float(ca._holm_alpha()), "producer": BUILDER_FILE, "contrasts": [],
                    "seed_axis": SEED_AXIS, "target_rule": TARGET_RULE,
                    "test_code": code}
        for key, spec in FAMILY.items():
            if key not in pairs_by_key:
                continue
            dataset, kind, unit, count, loss, direction, margin = spec
            members = pairs_by_key[key]
            entry = next(d for d in datasets if d["dataset"] == dataset)
            if not synthetic and entry["unit_count"] != count:
                raise RunnerError(f"{key}: D_confirm has {entry['unit_count']} {unit} units; S14 registers {count}")
            if dataset == ROMAN:
                left_hash, right_hash = entry["aggregate"]["left"], entry["aggregate"]["right"]
            else:
                left_hash, right_hash = members[0]["left_policy_sha256"], members[0]["right_policy_sha256"]
            if left_hash == right_hash:
                raise RunnerError(f"{key}: the two policies of the contrast share one identity")
            weighting = {"scheme": "equal_unit"}
            if unit == "buffered_cluster":
                weights = confirmation_units(members[0]["run"], dataset)["weights"]
                data = _json_bytes(weights)
                write_bytes(staging / "weights" / f"{key_slug(key)}.json", data)
                weighting = {"scheme": "fixed_unit_weights",
                             "unit_weights": _ref(f"weights/{key_slug(key)}.json", data)}
            method_family = {"graph": "graph_bootstrap", "official_split": "paired_small_sample"}.get(
                unit, "paired_cluster" if kind == "state" else "paired_unit")
            tail = "greater" if direction > 0 else "less"
            config = {"schema": "cm-s14-analysis-config-1", "contrast_id": key, "dataset": dataset, "kind": kind,
                      "statistical_unit": unit, "unit_count": count, "loss": loss, "beneficial_direction": direction,
                      "practical_margin_nats": margin, "tail": tail, "method_family": method_family,
                      "method": METHOD_NAMES[method_family], "seed_axis": SEED_AXIS,
                      "statistics": _statistics_definition(unit),
                      "weighting": "scored-sample counts per cluster (runtime score_observations)"
                                   if unit == "buffered_cluster" else "equal",
                      "members": [{k: v for k, v in m.items() if k not in ("run", "cell")} for m in members],
                      "test_code": {name: ref for name, ref in code.items()},
                      "runtime_source_sha256": runtime_source_sha256(), "registered_at": _iso(registered)}
            config_bytes = _json_bytes(config)
            write_bytes(staging / "code" / key_slug(key) / "analysis_config.json", config_bytes)
            protocol["contrasts"].append({
                "contrast_id": key, "dataset": dataset, "kind": kind, "statistical_unit": unit, "unit_count": count,
                "loss": loss, "beneficial_direction": direction, "practical_margin_nats": margin,
                "data_role": "D_confirm", "seed_ids": [SEED_LEVEL], "weighting": weighting,
                "left_policy_sha256": left_hash, "right_policy_sha256": right_hash,
                "method": {"name": METHOD_NAMES[method_family], "family": method_family, "tail": tail, "null": "zero",
                           "command": COMMAND, "analysis_code": code[BUILDER_FILE],
                           "analysis_config": _ref(f"code/{key_slug(key)}/analysis_config.json", config_bytes),
                           "environment": code["environment"]}})
            family.append({"contrast_id": key, "dataset": dataset, "kind": kind, "statistical_unit": unit,
                           "unit_count": count, "observed_unit_count": entry["unit_count"], "loss": loss,
                           "beneficial_direction": direction, "practical_margin_nats": margin,
                           "method_family": method_family, "tail": tail,
                           "left_policy_sha256": left_hash, "right_policy_sha256": right_hash,
                           "members": [{k: v for k, v in m.items() if k not in ("run", "cell")} for m in members]})
        protocol_bytes = _json_bytes(protocol)
        write_bytes(staging / "protocol.json", protocol_bytes)
        registration = {"protocol": _ref("protocol.json", protocol_bytes),
                        "files": {p.relative_to(staging).as_posix(): sha256(p) for p in sorted(staging.rglob("*"))
                                  if p.is_file()}}
        plan = create_plan(output, {
            "experiment": EXPERIMENT, "record_set": RECORD_SET, "role": CONFIRM, "registered_at": _iso(registered),
            "synthetic": synthetic, "partial": bool(missing), "force_compute": bool(force_compute),
            "datasets": datasets, "family": family, "jobs": jobs, "registration": registration,
            "builder_code_sha256": code_identity(), "alpha": float(ca._holm_alpha()),
            "parent_runs": [m["run"] for d in datasets for m in d["members"]],
            "protocol": ("joint S14 family: the confirm stage's paired D_confirm contrasts of each dataset's S14 target "
                         "(sealed contrast when the run's confirmation closure holds it, else re-executed with the "
                         "mechanism-contrast machinery); confirm-stage bootstrap statistics, Roman small-sample t over "
                         "splits; one Holm family of nine, strict margins, S11 map")})
        target = plan.parent / "registration"
        shutil.move(str(staging), str(target))
        return plan
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _statistics_definition(unit: str) -> dict:
    if unit == "official_split":
        return {"effect": "mean over the ten splits of d_k = loss_k(left) - loss_k(right), loss_k the split's "
                          "D_confirm loss (count-weighted mean of its per-core losses, runtime score_observations)",
                "interval": "mean +- t_{0.975, n-1} sd(d)/sqrt(n) (confirmation_analysis.t_interval)",
                "p_value_zero": "one-sided Student t (n-1 df) against zero in the beneficial direction",
                "p_value_margin": "one-sided Student t (n-1 df) of b*mean - margin"}
    return {"effect": "unit-weighted mean of the per-unit paired differences (cross_morse.statistics.paired_bootstrap)",
            "interval": "2.5% and 97.5% percentiles of the paired whole-unit bootstrap means (alpha = 0.025 each side)",
            "p_value_zero": "one-sided null-centred bootstrap P against zero in the beneficial direction "
                            "(cross_morse.statistics.confirmation: p_greater_zero for state, p_less_zero for radius/RH)",
            "p_value_margin": "cross_morse.statistics.confirmation p_value_margin (bootstrap of b*d - margin)",
            "bootstrap": "replicates = the run's selection.bootstrap_reps (read: as sealed); seed = run seed + 5 + "
                         "contrast position (read: in the run's confirm family; compute: in the job's cells)"}


# ============================================================================= execution: shared checks
def check_registration(output, plan) -> None:
    """The registered protocol files and test code are those of the plan (never edited afterwards)."""
    output = Path(output)
    for relative, digest in plan["registration"]["files"].items():
        path = output / "registration" / relative
        if not path.is_file() or sha256(path) != digest:
            raise RunnerError(f"registration/{relative} changed after registration; plan again")
    current = code_identity()
    if current != plan["builder_code_sha256"]:
        changed = sorted(k for k in set(current) | set(plan["builder_code_sha256"])
                         if current.get(k) != plan["builder_code_sha256"].get(k))
        raise RunnerError(f"the registered test code changed after registration ({changed}); plan again")
    if runtime_source_sha256() != plan["runtime_source_sha256"]:
        raise RunnerError("the runtime source changed after registration; plan again")


def _after_registration(registered: str) -> None:
    """Recorded times have one-second resolution: continue strictly after the given time's second."""
    target = _time(registered).replace(microsecond=0) + timedelta(seconds=1)
    while datetime.now(timezone.utc) < target:
        time.sleep(min(1.0, max(0.05, (target - datetime.now(timezone.utc)).total_seconds())))


def done_marker(directory: Path, entry: dict) -> Path:
    return Path(directory) / ("read.json" if entry["kind"] == "read" else "mechanism.json")


# ============================================================================= execution: sealed contrasts (torch-free parts)
def replay_sealed_statistics(run: SealedRun) -> tuple[list, list]:
    """The run's sealed contrasts and confirm/decision.json rows, replayed with cross_morse.statistics.confirmation."""
    from cross_morse.review import EvidenceError, assert_matches
    from cross_morse.statistics import confirmation
    sealed = run.json("confirm/paired_contrasts.json")
    decision = run.json("confirm/decision.json")
    selection = run.json("select/decision.json")
    replay = confirmation(selection["provisional_index"], selection["comparator_index"], selection["eligible_indices"],
                          sealed, reps=int(run.config["selection"]["bootstrap_reps"]), seed=int(run.config["seed"]) + 5)
    try:
        assert_matches(json_safe(replay["contrasts"]), decision["contrasts"], "confirm/decision.json:/contrasts")
    except EvidenceError as exc:
        raise RunnerError(f"{run.root.name}: the sealed confirm statistics do not replay from the sealed contrasts: "
                          f"{exc}") from None
    return sealed, decision["contrasts"]


def sealed_row(cell: dict, sealed: list) -> tuple[int, dict, bool]:
    """(index, row, usable) of a planned cell's contrast in the run's sealed confirm/paired_contrasts.json."""
    kind, candidate = cell["kind"], int(cell["candidate"]["registry_index"])
    index = next((k for k, row in enumerate(sealed)
                  if row.get("kind") == kind and int(row.get("candidate_index", -1)) == candidate), None)
    if index is None:
        raise RunnerError(f"{cell['contrast']}: the sealed confirm stage holds no {kind} contrast of candidate "
                          f"{candidate}")
    row = sealed[index]
    return index, row, row.get("available") is not False and row.get("intervention_valid") is not False


def sealed_cells(cells: list, sealed: list, statistics: list, scores: dict, comparator_index: int) -> list[dict]:
    """Each planned cell's sealed confirm-stage contrast, verified against the rescored sealed predictions.

    ``scores``: {policy key: score_observations result of its sealed D_confirm prediction}; needed only for
    cells whose sealed row is usable (an unavailable or invalid sealed contrast is reported, never repaired).
    """
    out = []
    for cell in cells:
        kind, candidate = cell["kind"], int(cell["candidate"]["registry_index"])
        index, row, _ = sealed_row(cell, sealed)
        stats = statistics[index]
        expected = comparator_index if kind == "state" else int(cell["reference"]["registry_index"])
        if int(row.get("predecessor_index", -1)) != expected or int(stats.get("candidate_index", -1)) != candidate:
            raise RunnerError(f"{cell['contrast']}: the sealed contrast's policies differ from the plan")
        reason = None
        verification = {"sealed_index": index}
        if row.get("available") is False:
            reason = f"sealed contrast unavailable: {row.get('unavailable_reason')}"
        elif row.get("intervention_valid") is False:
            reason = f"sealed state intervention invalid: {row.get('invalid_reason')}"
        else:
            a, b = scores[cell["candidate_key"]], scores[cell["reference_key"]]
            if not np.array_equal(a["core_ids"], b["core_ids"]):
                raise RunnerError(f"{cell['contrast']}: the two sealed predictions were scored on different units")
            differences = (b["loss"] - a["loss"]) if kind == "state" else (a["loss"] - b["loss"])
            sealed_d = np.asarray(row["differences"], dtype=np.float64)
            sealed_w = np.asarray(row["weights"], dtype=np.float64)
            same_d = differences.shape == sealed_d.shape and np.array_equal(differences, sealed_d)
            same_w = sealed_w.shape == np.shape(a["counts"]) and np.array_equal(np.asarray(a["counts"], float), sealed_w)
            if not (same_d and same_w):
                gap = float(np.max(np.abs(differences - sealed_d))) if differences.shape == sealed_d.shape else None
                raise RunnerError(f"{cell['contrast']}: the rescored sealed predictions do not reproduce the sealed "
                                  f"confirm-stage contrast (differences equal: {same_d}, max |gap| {gap}; weights "
                                  f"equal: {same_w})")
            verification.update(differences_reproduced="exact", weights_reproduced="exact", units=int(len(sealed_d)))
            if not stats.get("valid_interval"):
                reason = "the sealed contrast has no valid interval (fewer than two units)"
        valid = reason is None
        statistics_row = {k: v for k, v in stats.items() if k not in ("holm_p_value_zero", "passes")}
        out.append({"contrast": cell["contrast"], "kind": kind, "valid": valid, "invalid_reason": reason,
                    "effect_nats": stats.get("mean") if valid else None,
                    "interval_low_nats": stats.get("lower") if valid else None,
                    "interval_high_nats": stats.get("upper") if valid else None,
                    "units": stats.get("units"), "statistics": statistics_row, "verification": verification,
                    "source": "sealed confirm stage (confirm/paired_contrasts.json, confirm/decision.json)"})
    return out


def write_read_result(directory, entry, *, cells, arrays, predictions, parent_files, started, replay,
                      environment=None) -> Path:
    """A read job's raw outputs: unit_losses.npz and read.json (written last: the completion marker)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if environment is not None and not (directory / "environment.json").exists():
        write_json(directory / "environment.json", json_safe(environment))
    planned = {c["contrast"]: c for c in entry["cells"]}
    if [c["contrast"] for c in cells] != list(planned):
        raise RunnerError("the verified cells differ from the planned cells")
    for cell in cells:
        plan_cell = planned[cell["contrast"]]
        cell.update(candidate=plan_cell["candidate"], reference=plan_cell["reference"],
                    anchor_rule=plan_cell.get("anchor_rule"), pair_rule=plan_cell.get("pair_rule"),
                    definition=plan_cell.get("definition"))
    digest = save_arrays(directory / "unit_losses.npz", replace=True, **arrays)
    payload = {"schema": "cm-s14-sealed-contrasts-1", "dataset": entry["dataset"], "member": entry["member"],
               "split_index": entry.get("split_index"), "role": entry["role"], "cells": cells,
               "sealed_predictions": predictions, "statistics_replay": replay,
               "unit_losses": {"file": "unit_losses.npz", "sha256": digest},
               "identities": {"runtime_source_sha256": runtime_source_sha256(), "parent": entry["identity"],
                              "plan_sha256": sha256(directory.parents[2] / "plan.json"),
                              "parent_files_read": parent_files, "test_code_sha256": code_identity()},
               "statistics_code": "cross_morse.statistics.confirmation (as sealed by the run's confirm stage)",
               "started_utc": started, "finished_utc": utc_now()}
    path = directory / "read.json"
    write_json(path, json_safe(payload))
    return path


# ============================================================================= execution (torch)
def run_s14(output, *, job=None, device="cuda", trust=False) -> list[Path]:
    if not trust:
        raise RunnerError("loading local checkpoints requires explicit trust (--trust-local-checkpoints)")
    output = Path(output).resolve()
    plan = load_plan(output, RECORD_SET)
    check_registration(output, plan)
    done = []
    for entry in selected_jobs(plan, job):
        directory = job_directory(output, entry)
        marker = done_marker(directory, entry)
        if not marker.is_file():
            _after_registration(plan["registered_at"])
            if entry["kind"] == "read":
                _run_read_job(entry, directory, device)
            else:
                ca._run_mechanism_job(plan, entry, directory, device)
        done.append(marker)
    return done


def sealed_prediction(run: SealedRun, candidate: dict, collection: bool) -> tuple[Path, dict]:
    """The confirm stage's sealed ensemble prediction of a policy, verified as Pipeline.verify_prediction does."""
    folder = run.root / "predictions" / (CONFIRM if collection else "")
    path = folder / f"{candidate['candidate_id']}.npy"
    meta_path = path.with_suffix(".json")
    if not path.is_file() or not meta_path.is_file():
        raise RunnerError(f"{run.root.name}: sealed prediction {path.relative_to(run.root)} is missing")
    meta = read_json(meta_path)
    identity = meta.get("candidate", {})
    if (identity.get("candidate_id") != candidate["candidate_id"] or identity.get("hash") != candidate["hash"]
            or identity.get("state_control") != candidate.get("state_control")
            or (collection and meta.get("execution_role") != CONFIRM) or sha256(path) != meta.get("prediction_sha256")):
        raise RunnerError(f"{path}: sealed prediction failed identity/hash verification")
    if meta.get("fallback_applied") or meta.get("execution_failure"):
        raise RunnerError(f"{path}: the sealed prediction is a numerical fallback or failure, not the policy's output")
    run.read[path.relative_to(run.root).as_posix()] = sha256(path)
    return path, meta


def _run_read_job(entry, directory, device):
    from cross_morse.evaluation import score_observations
    run = SealedRun(entry["run"])
    check_identity(run, entry["identity"])
    ca.require_complete(run)
    ca.check_label_role(run, CONFIRM)
    started = utc_now()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / "environment.json").exists():
        write_json(directory / "environment.json", environment_record(device))
    sealed, statistics = replay_sealed_statistics(run)
    # An unavailable or invalid sealed contrast may carry the comparator's fallback bytes: not rescored.
    needed = {key for cell in entry["cells"] if sealed_row(cell, sealed)[2]
              for key in (cell["candidate_key"], cell["reference_key"])}
    ctx = ca.RoleContext(run, device, through="evaluate")
    ids, labels, units = ctx.role_ids(CONFIRM), ctx.labels(CONFIRM), ctx.units(CONFIRM)
    scores, predictions = {}, {}
    for policy in entry["policies"]:
        if policy["key"] not in needed:
            continue
        path, meta = sealed_prediction(run, policy["candidate"], ctx.collection)
        probabilities = np.load(path, mmap_mode="r", allow_pickle=False)[ids]
        if not np.isfinite(probabilities).all():
            raise RunnerError(f"{path}: nonfinite D_confirm predictions")
        scores[policy["key"]] = score_observations(probabilities, labels, units, ctx.config)
        predictions[policy["key"]] = {"file": path.relative_to(run.root).as_posix(), "sha256": sha256(path),
                                      "metadata_sha256": sha256(path.with_suffix(".json")),
                                      "feasible": meta.get("feasible"),
                                      "state_control_valid": meta.get("state_control_valid")}
    cells = sealed_cells(entry["cells"], sealed, statistics, scores, int(entry["selection"]["comparator_index"]))
    arrays = {}
    for key, score in scores.items():
        arrays.setdefault("unit_id", score["core_ids"])
        arrays.setdefault("unit_weight", score["counts"])
        arrays[f"{key}_unit_loss"] = score["loss"]
    write_read_result(directory, entry, cells=cells, arrays=arrays, predictions=predictions,
                      parent_files=run.inputs(), started=started,
                      replay={"rows": len(statistics), "matches_sealed_decision": True,
                              "reps": int(run.config["selection"]["bootstrap_reps"]),
                              "seed": int(run.config["seed"]) + 5})


# ============================================================================= assembly: member inputs (torch-free)
def member_input(output, plan, entry, key) -> dict:
    """One job's contribution to one FAMILY test: per-unit sides, weights and the runtime's statistics."""
    directory = job_directory(output, entry)
    marker = done_marker(directory, entry)
    if not marker.is_file():
        raise RunnerError(f"{key}: job {entry['job']} ({entry['member']}) has not completed")
    result = read_json(marker)
    identities = result.get("identities", {})
    if identities.get("parent") != entry["identity"] or identities.get("plan_sha256") != sha256(Path(output) / "plan.json"):
        raise RunnerError(f"{marker}: the job output belongs to another plan or parent identity")
    if identities.get("runtime_source_sha256") != plan["runtime_source_sha256"]:
        raise RunnerError(f"{marker}: executed with another runtime source")
    cell = next((c for c in result["cells"] if c["contrast"] == key), None)
    plan_cell = next((c for c in entry["cells"] if c["contrast"] == key), None)
    if cell is None or plan_cell is None:
        raise RunnerError(f"{key}: missing from job {entry['job']}")
    if cell.get("candidate") != plan_cell["candidate"] or cell.get("reference") != plan_cell["reference"]:
        raise RunnerError(f"{key}: job {entry['job']} tested other policies than the registered ones")
    left_key, right_key = ((plan_cell["reference_key"], plan_cell["candidate_key"]) if plan_cell["kind"] == "state"
                           else (plan_cell["candidate_key"], plan_cell["reference_key"]))
    statistics = cell.get("statistics") or {}
    out = {"key": key, "member": entry["run_name"], "split_index": entry.get("split_index"), "mode": entry["kind"],
           "job": entry["job"], "valid": bool(cell.get("valid")), "invalid_reason": cell.get("invalid_reason"),
           "statistics": statistics, "started_utc": result.get("started_utc"), "finished_utc": result.get("finished_utc"),
           "directory": directory, "files": [marker.name, "environment.json"], "cell": cell}
    if not out["valid"]:
        return out
    if not result.get("unit_losses"):
        raise RunnerError(f"{marker}: a valid contrast without its per-unit losses")
    losses = directory / result["unit_losses"]["file"]
    if sha256(losses) != result["unit_losses"]["sha256"]:
        raise RunnerError(f"{losses} differs from its recorded SHA-256")
    out["files"].insert(1, result["unit_losses"]["file"])
    with np.load(losses, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    try:
        unit_id = [str(int(u)) for u in arrays["unit_id"]]
        weight = np.asarray(arrays["unit_weight"], dtype=np.float64)
        left = np.asarray(arrays[f"{left_key}_unit_loss"], dtype=np.float64)
        right = np.asarray(arrays[f"{right_key}_unit_loss"], dtype=np.float64)
    except KeyError as exc:
        raise RunnerError(f"{losses}: missing array {exc}") from None
    n = len(unit_id)
    if not (weight.shape == left.shape == right.shape == (n,)) or n < 2 or len(set(unit_id)) != n:
        raise RunnerError(f"{key}/{entry['run_name']}: unit arrays are misaligned or have fewer than two units")
    if not (np.isfinite(left).all() and np.isfinite(right).all() and (left >= 0).all() and (right >= 0).all()
            and np.isfinite(weight).all() and (weight > 0).all()):
        raise RunnerError(f"{key}/{entry['run_name']}: losses must be finite and nonnegative, weights positive")
    mean = float(np.dot(weight, left - right) / weight.sum())
    if statistics.get("mean") is None or not math.isclose(mean, float(statistics["mean"]), rel_tol=1e-9, abs_tol=1e-12):
        raise RunnerError(f"{key}/{entry['run_name']}: the per-unit sides do not reproduce the contrast's mean")
    if FAMILY[key][1] == "state" and statistics.get("intervention_valid") is False:
        raise RunnerError(f"{key}/{entry['run_name']}: invalid state intervention")
    out.update(unit_id=unit_id, unit_weight=weight, left_loss=left, right_loss=right, units=n)
    return out


def small_sample_test(differences, direction: int, margin: float) -> dict:
    """One-sided paired Student t test over splits, its 95% t interval and the one-sided t against the margin."""
    from scipy.stats import t as student
    d = np.asarray(differences, dtype=np.float64)
    interval = ca.t_interval(d)
    n, mean, se = interval["units"], interval["mean"], interval["se"]
    if not se > 0 or float(np.ptp(d)) == 0.0:
        raise RunnerError("the split contrasts have zero variance: the small-sample test is undefined")
    statistic = mean / se
    p_zero = float(student.cdf(statistic, n - 1)) if direction < 0 else float(student.sf(statistic, n - 1))
    margin_statistic = (direction * mean - margin) / se
    return {"effect_nats": mean, "interval_nats": [interval["lower"], interval["upper"]], "p_value_zero": p_zero,
            "p_value_margin": float(student.sf(margin_statistic, n - 1)), "t_statistic": statistic,
            "t_statistic_margin": margin_statistic, "degrees_of_freedom": n - 1, "se": se,
            "t_quantile": interval["t_quantile"], "units": n, "interval_method": interval["method"],
            "test": "one-sided paired Student t test of the split contrasts against zero"}


def _plan_family(plan, key):
    return next(f for f in plan["family"] if f["contrast_id"] == key)


def _registered(output, plan, key) -> tuple[dict, dict | None]:
    """The registered confirmation units of a test's dataset and its prespecified unit weights (None: equal)."""
    dataset = FAMILY[key][0]
    entry = next(d for d in plan["datasets"] if d["dataset"] == dataset)
    path = Path(output) / "registration" / entry["role_manifest"]["path"]
    units = read_json(path)["confirmation_units"]
    weights = None
    if FAMILY[key][2] == "buffered_cluster":
        weights = read_json(Path(output) / "registration" / "weights" / f"{key_slug(key)}.json")
    return units, weights


def family_tests(output, plan) -> dict:
    """Every FAMILY test from the completed jobs: per-unit rows and the test statistics (no substitution)."""
    jobs = {j["job"]: j for j in plan["jobs"]}
    tests, refused = {}, []
    for key, spec in FAMILY.items():
        dataset, kind, unit, count, loss, direction, margin = spec
        planned = _plan_family(plan, key)
        members = [member_input(output, plan, jobs[m["job"]], key) for m in planned["members"]]
        invalid = [f"{m['member']}: {m['invalid_reason']}" for m in members if not m["valid"]]
        if invalid:
            refused.append(f"{key}: " + "; ".join(invalid))
            continue
        units, weights = _registered(output, plan, key)
        if unit == "official_split":
            if sorted(f"split={m['split_index']}" for m in members) != sorted(units):
                refused.append(f"{key}: the executed splits differ from the registered confirmation units")
                continue
        else:
            member = members[0]
            if set(member["unit_id"]) != set(units):
                refused.append(f"{key}: the scored D_confirm units differ from the registered confirmation units "
                               f"({len(member['unit_id'])} scored, {len(units)} registered)")
                continue
            expected = ({u: float(weights[u]) for u in member["unit_id"]} if weights is not None
                        else {u: 1.0 for u in member["unit_id"]})
            if {u: float(w) for u, w in zip(member["unit_id"], member["unit_weight"])} != expected:
                refused.append(f"{key}: the runtime's unit weights differ from the prespecified weights")
                continue
        if unit == "official_split":
            rows, d = [], []
            for m in sorted(members, key=lambda x: x["split_index"]):
                left = float(np.average(m["left_loss"], weights=m["unit_weight"]))
                right = float(np.average(m["right_loss"], weights=m["unit_weight"]))
                rows.append((f"split={m['split_index']}", left, right, 1))
                d.append(left - right)
                m["split_contrast_nats"] = left - right
                m["split_contrast_runtime_mean_nats"] = float(m["statistics"]["mean"])
            stats = small_sample_test(d, direction, margin)
        else:
            member = members[0]
            rows = list(zip(member["unit_id"], member["left_loss"].tolist(), member["right_loss"].tolist(),
                            [int(w) if float(w).is_integer() else float(w) for w in member["unit_weight"]]))
            s = member["statistics"]
            if not s.get("valid_interval") or s.get("lower") is None or s.get("upper") is None:
                refused.append(f"{key}: no valid interval")
                continue
            stats = {"effect_nats": float(s["mean"]), "interval_nats": [float(s["lower"]), float(s["upper"])],
                     "p_value_zero": float(s["p_value_zero"]),
                     "p_value_margin": None if s.get("p_value_margin") is None else float(s["p_value_margin"]),
                     "units": int(s["units"]), "test": "confirm-stage paired whole-unit bootstrap",
                     "bootstrap": (member["cell"].get("bootstrap") or
                                   {"reps": jobs[member["job"]]["bootstrap"]["reps"],
                                    "seed": jobs[member["job"]]["bootstrap"]["seed"],
                                    "contrast_offset": member["cell"].get("verification", {}).get("sealed_index")})}
        tests[key] = {"rows": rows, "statistics": stats, "members": members, "planned": planned}
    return {"tests": tests, "refused": refused}


# ============================================================================= assembly: campaign (torch-free)
def _paired_csv(rows) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["seed_id", "unit_id", "left_loss_nats", "right_loss_nats", "unit_weight"])
    for unit, left, right, weight in rows:
        writer.writerow([SEED_LEVEL, unit, repr(float(left)), repr(float(right)), weight])
    return buffer.getvalue().encode()


def admission_start(run: SealedRun) -> tuple[datetime, str]:
    """When the run's lifecycle admission (its first D_admit label access) started: no earlier than the confirm
    seal, and no later than the admit seal minus the admit stage's recorded duration."""
    confirm, admit = stage_time(run, "confirm"), stage_time(run, "admit")
    marker = run.json("stages/admit.json")
    seconds = marker.get("seconds_this_attempt")
    if isinstance(seconds, (int, float)) and math.isfinite(seconds) and seconds >= 0:
        start = max(confirm, admit - timedelta(seconds=float(seconds)))
        how = "admit stage seal time minus its recorded duration (stages/admit.json), not before the confirm seal"
    else:
        start = confirm
        how = "confirm stage seal time (the admit stage records no duration)"
    return start, how


def _copy(source: Path, campaign: Path, relative: str, expected: str | None = None) -> dict:
    data = Path(source).read_bytes()
    reference = _ref(relative, data)
    if expected is not None and reference["sha256"] != expected:
        raise RunnerError(f"{source} changed after registration")
    write_bytes(campaign / relative, data)
    return reference


def _decisions(plan, tests_by_key, ledgers, alpha) -> tuple[dict, dict, dict]:
    """Joint Holm, strict margins and the S11 map over the nine tests (the decision rules check_campaign replays)."""
    from cm_campaigns.confirmation import _policy_decision
    from cross_morse.statistics import holm_adjust
    keys = list(FAMILY)
    adjusted = holm_adjust(np.asarray([tests_by_key[k]["statistics"]["p_value_zero"] for k in keys], dtype=np.float64))
    holm, tested = {}, {}
    for key, p in zip(keys, adjusted):
        _, _, _, _, _, direction, margin = FAMILY[key]
        stats, planned = tests_by_key[key]["statistics"], _plan_family(plan, key)
        beyond = direction * stats["effect_nats"] > margin
        retained = bool(float(p) < alpha and beyond)
        holm[key] = {"raw_p": stats["p_value_zero"], "holm_p": float(p), "point_beyond_margin": bool(beyond),
                     "retained": retained}
        tested[key] = {"left_policy_sha256": planned["left_policy_sha256"],
                       "right_policy_sha256": planned["right_policy_sha256"], "retained": retained}
    decisions = {}
    for dataset in DATASETS:
        ledger, selection = ledgers[dataset]
        hashes = [c["hash"] for c in ledger]
        view = {"qC": hashes[int(selection["comparator_index"])], "provisional": hashes[int(selection["provisional_index"])],
                "eligible": {hashes[int(i)] for i in selection["eligible_indices"]}, "candidates": dict(zip(hashes, ledger))}
        try:
            decisions[dataset] = _policy_decision(dataset, view, tested)
        except ValueError as exc:
            raise RunnerError(f"{dataset}: the S11 map cannot be applied to the tested policies: {exc}") from None
    selected = {key: row["tested_policy_selected"] for key, row in tested.items()}
    return holm, decisions, selected


def assemble_s14(output, campaign, *, manuscript_root=ROOT, clock=None) -> dict:
    """Write the campaign directory from the completed jobs, then record check_campaign's verdict."""
    output = Path(output).resolve()
    plan = load_plan(output, RECORD_SET)
    if plan.get("partial"):
        raise RunnerError("a partial plan is a software check; the joint family needs all seven datasets")
    check_registration(output, plan)
    campaign = Path(campaign).expanduser().absolute()
    if any(p.is_symlink() for p in (campaign, *campaign.parents)):
        raise RunnerError("the campaign directory must not traverse symbolic links")
    campaign = campaign.resolve()
    if campaign.exists():
        raise RunnerError(f"{campaign} exists; a campaign is written once into a new directory")
    if campaign.is_relative_to(ROOT) or output.is_relative_to(campaign):
        raise RunnerError("write the campaign outside the checkout, in a directory that does not contain the build")
    for dataset in plan["datasets"]:
        for member in dataset["members"]:
            run = Path(member["run"])
            if campaign == run or campaign.is_relative_to(run) or run.is_relative_to(campaign):
                raise RunnerError("the campaign directory must be separate from every parent run")
    built = family_tests(output, plan)
    if built["refused"]:
        raise RunnerError("tests without a valid measurement (no value is substituted):\n  " + "\n  ".join(built["refused"]))
    tests = built["tests"]
    finished = [_time(m["finished_utc"]) for t in tests.values() for m in t["members"]]
    started = [_time(m["started_utc"]) for t in tests.values() for m in t["members"]]
    registered = _time(plan["registered_at"])
    if min(started) <= registered:
        raise RunnerError("a test started before its registration")
    now = _clock(clock)
    if now < max(finished):
        raise RunnerError("the clock is behind the recorded test completion; assemble later")
    sources = read_json(Path(manuscript_root) / "paper" / "sources.json")
    staging = Path(tempfile.mkdtemp(prefix=".cm-s14-campaign-", dir=campaign.parent))
    try:
        # Registration, unchanged.
        for relative in plan["registration"]["files"]:
            _copy(output / "registration" / relative, staging, relative, plan["registration"]["files"][relative])
        protocol_ref = plan["registration"]["protocol"]
        protocol = read_json(staging / "protocol.json")
        alpha = float(protocol["alpha"])
        # Original test records and paired observations.
        test_entries, result_bytes = [], {}
        entries = {d["dataset"]: d for d in plan["datasets"]}
        for key, spec in FAMILY.items():
            dataset, kind, unit, count, loss, direction, margin = spec
            test, planned = tests[key], tests[key]["planned"]
            base = f"raw/{key_slug(key)}"
            paired = _paired_csv(test["rows"])
            write_bytes(staging / base / "paired.csv", paired)
            paired_ref = _ref(f"{base}/paired.csv", paired)
            members = []
            for m in test["members"]:
                files = {}
                for name in m["files"]:
                    path = m["directory"] / name
                    if path.is_file():
                        files[name] = _copy(path, staging, f"{base}/members/{slug(m['member'])}/{name}")
                members.append({"member": m["member"], "split_index": m["split_index"], "mode": m["mode"],
                                "job": m["job"], "units": m["units"], "files": files,
                                "source": "sealed confirm-stage contrast" if m["mode"] == "read"
                                          else "confirm-stage contrast re-executed on D_confirm",
                                "runtime_statistics": m["statistics"],
                                "split_contrast_nats": m.get("split_contrast_nats"),
                                "split_contrast_runtime_mean_nats": m.get("split_contrast_runtime_mean_nats"),
                                "started_utc": m["started_utc"], "finished_utc": m["finished_utc"]})
            stats = test["statistics"]
            if unit == "official_split":
                completed = now
            else:
                completed = max(_time(m["finished_utc"]) for m in test["members"])
            method_family = planned["method_family"]
            result = {"schema": RESULT_SCHEMA, "contrast_id": key, "dataset": dataset,
                      "evidence_kind": "original_confirmation_test", "p_value_kind": "unadjusted_against_zero",
                      "numeric_precision": "unrounded", "synthetic": bool(plan["synthetic"]), "data_role": "D_confirm",
                      "statistical_unit": unit, "unit_count": count if not plan["synthetic"] else len(test["rows"]),
                      "loss": loss, "method_name": METHOD_NAMES[method_family], "method_family": method_family,
                      "tail": planned["tail"], "interval_confidence": 0.95, "seed_ids": [SEED_LEVEL],
                      "left_policy_sha256": planned["left_policy_sha256"],
                      "right_policy_sha256": planned["right_policy_sha256"],
                      "protocol_sha256": protocol_ref["sha256"], "paired_losses_sha256": paired_ref["sha256"],
                      "role_manifest_sha256": entries[dataset]["role_manifest"]["sha256"],
                      "started_at": _iso(min(_time(m["started_utc"]) for m in test["members"])),
                      "completed_at": _iso(completed),
                      "effect_nats": stats["effect_nats"], "p_value_zero": stats["p_value_zero"],
                      "p_value_margin": stats.get("p_value_margin"), "valid_interval": True,
                      "interval_nats": stats["interval_nats"],
                      "test_statistics": {k: v for k, v in stats.items()
                                          if k not in ("effect_nats", "p_value_zero", "interval_nats", "p_value_margin")},
                      "producer": BUILDER_FILE, "seed_axis": SEED_AXIS, "members": members,
                      "observed_unit_count": len(test["rows"])}
            if kind == "state":
                result.update(intervention_valid=True, intervention_scope=STATE_SCOPE)
            data = _json_bytes(result)
            write_bytes(staging / base / "test.json", data)
            result_bytes[key] = data
            test_entries.append({"contrast_id": key, "result": _ref(f"{base}/test.json", data),
                                 "paired_losses": paired_ref})
        # Datasets: roles, selections and original run bindings (sealed bytes, copied unchanged).
        dataset_entries, ledgers, run_locks, admissions = [], {}, {}, {}
        for dataset in DATASETS:
            entry = entries[dataset]
            runs = [SealedRun(m["run"]) for m in entry["members"]]
            for run, member in zip(runs, entry["members"]):
                check_identity(run, member["identity"])
            bindings = []
            if dataset == ROMAN:
                for run, member in zip(runs, entry["members"]):
                    k = member["split_index"]
                    base = f"original/{dataset}/{k}"
                    bindings.append({"split_index": k, **member["binding"],
                                     "selection": _copy(run.root / "select" / "decision.json", staging,
                                                        f"{base}/select/decision.json"),
                                     "candidates": _copy(run.root / "select" / "candidates.json", staging,
                                                         f"{base}/select/candidates.json"),
                                     "pre_admission_lock": _copy(run.root / "admit" / "pre_admission_lock.json", staging,
                                                                 f"{base}/admit/pre_admission_lock.json")})
                selection_ref, candidates_ref = entry["selection"], entry["candidates"]
                ledger = read_json(staging / candidates_ref["path"])
                selection = read_json(staging / selection_ref["path"])
                locks = [read_json(run.root / "admit" / "pre_admission_lock.json") for run in runs]
                parts = lambda field: [{"split_index": m["split_index"], "hash": lock[field]}
                                       for m, lock in zip(entry["members"], locks)]
                aggregate_lock = {"schema": "cm-s14-roman-aggregate-lock-1",
                                  "comparator_hash": composite_policy_hash(parts("comparator_hash")),
                                  "confirmed_hash": composite_policy_hash(parts("confirmed_hash")),
                                  "split_lock_sha256": {str(b["split_index"]): b["pre_admission_lock"]["sha256"]
                                                        for b in bindings},
                                  "construction": "identity wrapper of the ten split pre-admission locks "
                                                  "(composite_policy_hash of their comparator and confirmed hashes)"}
                data = _json_bytes(aggregate_lock)
                write_bytes(staging / "original" / dataset / "aggregate_pre_admission_lock.json", data)
                original_lock_ref = _ref(f"original/{dataset}/aggregate_pre_admission_lock.json", data)
                run_locks[dataset] = aggregate_lock
            else:
                run = runs[0]
                selection_ref = _copy(run.root / "select" / "decision.json", staging, entry["selection"]["path"],
                                      entry["selection"]["sha256"])
                candidates_ref = _copy(run.root / "select" / "candidates.json", staging, entry["candidates"]["path"],
                                       entry["candidates"]["sha256"])
                original_lock_ref = _copy(run.root / "admit" / "pre_admission_lock.json", staging,
                                          f"original/{dataset}/admit/pre_admission_lock.json")
                bindings.append({"split_index": None, **entry["members"][0]["binding"], "selection": selection_ref,
                                 "candidates": candidates_ref, "pre_admission_lock": original_lock_ref})
                ledger = read_json(staging / candidates_ref["path"])
                selection = read_json(staging / selection_ref["path"])
                run_locks[dataset] = read_json(staging / original_lock_ref["path"])
            ledgers[dataset] = (ledger, selection)
            starts = [admission_start(run) for run in runs]
            admissions[dataset] = (min(s for s, _ in starts), starts[0][1], original_lock_ref)
            dataset_entries.append({"dataset": dataset, "role_manifest": entry["role_manifest"],
                                    "selection": selection_ref, "candidates": candidates_ref,
                                    "selection_locked_at": entry["selection_locked_at"],
                                    "admission_link": {"path": f"admission/{dataset}.json", "sha256": None},
                                    "run_bindings": bindings})
        # Joint decision lock over the nine original tests.
        holm, decisions, selected = _decisions(plan, tests, ledgers, alpha)
        digest = hashlib.sha256(canonical({"protocol": protocol_ref, "tests": test_entries,
                                           "datasets": [{k: v for k, v in e.items() if k != "admission_link"}
                                                        for e in dataset_entries]})).hexdigest()
        policies = {d: {k: decisions[d][k] for k in ("comparator_hash", "provisional_hash", "confirmed_hash")}
                    for d in DATASETS}
        lock = {"schema": LOCK_SCHEMA, "input_sha256": digest, "locked_at": _iso(now), "alpha": alpha,
                "mechanism_retained": {k: holm[k]["retained"] for k in FAMILY}, "policies": policies,
                "holm": holm, "tested_policy_selected": selected,
                "decision_reasons": {d: decisions[d]["reason"] for d in DATASETS}, "producer": BUILDER_FILE,
                "rule": "Holm over the nine unadjusted P against zero (FAMILY order); retained iff Holm P < alpha and "
                        "the point effect lies strictly beyond the signed practical margin; Supplementary S11 map "
                        "(cm_campaigns.confirmation._policy_decision)"}
        lock_bytes = _json_bytes(lock)
        write_bytes(staging / "joint_decision_lock.json", lock_bytes)
        lock_ref = _ref("joint_decision_lock.json", lock_bytes)
        # Admission links: the lifecycle admission as it happened.
        findings = []
        for entry in dataset_entries:
            dataset = entry["dataset"]
            start, how, original_lock_ref = admissions[dataset]
            governed = start > now
            run_confirmed = run_locks[dataset]["confirmed_hash"]
            agrees = run_confirmed == decisions[dataset]["confirmed_hash"] and \
                run_locks[dataset]["comparator_hash"] == decisions[dataset]["comparator_hash"]
            if not governed:
                findings.append(f"{dataset}: lifecycle admission started {_iso(start)}, before the joint lock "
                                f"({_iso(now)}); no joint lock governed it")
            if not agrees:
                findings.append(f"{dataset}: the lifecycle's pre-admission lock (confirm stage) confirmed "
                                f"{run_confirmed[:12]}, the joint S14 decision confirms "
                                f"{decisions[dataset]['confirmed_hash'][:12]} ({decisions[dataset]['reason']})")
            link = {"dataset": dataset, "joint_lock_sha256": lock_ref["sha256"] if governed else None,
                    "confirmed_hash": run_confirmed, "role_manifest_sha256": entry["role_manifest"]["sha256"],
                    "admission_started_at": _iso(start), "admission_started_at_derivation": how,
                    "original_pre_admission_lock": original_lock_ref,
                    "admission_after_joint_lock": governed, "lifecycle_lock_agrees_with_joint_decision": agrees,
                    "joint_decision": policies[dataset],
                    "scope": "the lifecycle's own admission stage (runtime admit, driven by the run's confirm-stage "
                             "pre-admission lock); the runtime never reads the joint lock"}
            data = _json_bytes(link)
            write_bytes(staging / "admission" / f"{dataset}.json", data)
            entry["admission_link"] = _ref(f"admission/{dataset}.json", data)
        provenance = {"schema": "cm-s14-campaign-production-1", "experiment": EXPERIMENT,
                      "producer": {"builder": BUILDER_FILE, "cli": "scripts/build_s14_campaign.py",
                                   "test_code_sha256": plan["builder_code_sha256"]},
                      "runtime_source_sha256": plan["runtime_source_sha256"], "registered_at": plan["registered_at"],
                      "assembled_at": _iso(now), "plan_sha256": sha256(output / "plan.json"),
                      "jobs": [{"job": j["job"], "kind": j["kind"], "dataset": j["dataset"], "member": j["run_name"],
                                "split_index": j.get("split_index")} for j in plan["jobs"]],
                      "findings": findings,
                      "protocol_position": ("admission after the joint lock" if not any("before the joint lock" in f
                                                                                        for f in findings)
                                            else "post hoc: the joint lock follows the lifecycle admissions")}
        data = _json_bytes(provenance)
        write_bytes(staging / "provenance.json", data)
        plan_text = public_plan(output).encode()
        write_bytes(staging / "code" / "plan.json", plan_text)
        manifest = {"schema": SCHEMA, "status": "author_completed", "synthetic": bool(plan["synthetic"]),
                    "evidence_kind": "original_confirmation_tests",
                    "manuscript_sha256": {k: v["sha256"] for k, v in sources["documents"].items()},
                    "protocol": protocol_ref, "tests": test_entries, "datasets": dataset_entries,
                    "joint_decision_lock": lock_ref,
                    "production": {"kind": REPOSITORY_BUILT, "builder": BUILDER_FILE,
                                   "provenance": _ref("provenance.json", data),
                                   "plan": _ref("code/plan.json", plan_text),
                                   "protocol_position": provenance["protocol_position"]}}
        write_json(staging / "campaign.json", manifest)
        os.replace(staging, campaign)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    verdict = verify(campaign, manuscript_root=manuscript_root)
    verdict["findings"] = findings
    write_json(output / "campaign_check.json", json_safe(verdict), replace=True)
    return verdict


def verify(campaign, *, manuscript_root=ROOT) -> dict:
    """check_campaign's verdict on a campaign: its report, or the first requirement it refuses."""
    from cm_campaigns.confirmation import check_campaign
    try:
        report = check_campaign(Path(campaign), manuscript_root=Path(manuscript_root))
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        return {"schema": "cm-s14-builder-check-1", "status": "FAIL", "campaign": str(campaign), "error": str(exc),
                "accepted": False}
    return {"schema": "cm-s14-builder-check-1", "status": report["status"], "campaign": str(campaign),
            "accepted": True, "report": report}
