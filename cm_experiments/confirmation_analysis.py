"""E10 confirmation analyses: held-out mechanism contrasts (main Figure 3a-b), the nine-test
confirmation family at its original sizes (Supplementary Table S14b) and its confirmation-size
sensitivity (Supplementary Figures S9 and S18b).

Record sets
-----------
``mechanism_contrasts``
    For every (dataset, contrast) cell of Figure 3a-b that neither the joint S14 family
    (cm_campaigns.confirmation.FAMILY) nor the deployed rule's own confirm/decision.json supplies
    (the rule of paper/build/displays/main_paper.mechanism_rows), the paired D_confirm contrast of
    the dataset's strongest feedback configuration, computed exactly as the lifecycle's confirm stage
    computes it (cross_morse.pipeline.Pipeline._stage_confirm, cross_morse.statistics.confirmation):

        state        loss(matched-shuffle state control of q) - loss(q) per unit
        rh_minus_ol  loss(MPC-RH) - loss(MPC-OL at the same horizon, profile and ensemble size) per unit

    with the paired whole-unit bootstrap 95% interval on buffered clusters or graphs; a Roman-empire
    campaign gives the mean of the ten split contrasts with the small-sample t interval over splits
    (its registered unit).  The strongest feedback configuration q is the deployed rule if it is a
    feedback route, else the confirmed and then the provisional policy, else the feedback candidate
    with the lowest D_select upper limit U (joint-feasible first; registered tie order).  The MPC-RH
    of an rh_minus_ol cell is q itself when q is MPC-RH, else the MPC-RH configuration at q's
    ensemble size and profile with the lowest D_select U; its predecessor is the runtime's
    (cross_morse.pipeline.Pipeline.predecessor).  Both are fixed from D_select before D_confirm is read.

``confirmation_contrasts`` and ``confirmation_sensitivity`` (one raw computation)
    Single-graph rows of the nine-test family are the original D_confirm tests of the joint S14
    campaign (validated by cm_campaigns.confirmation.check_campaign; their test records are deposited
    unchanged).  For the three multigraph contrasts (Pascal/RH, COCO/RH, Peptides/radius) the S14
    policy pair is executed on the official validation split (role ``admit``, J_old complete graphs)
    with every optimization seed s of the S14 seed population, one checkpoint at a time, giving the
    paired per-graph, per-seed array D_js = loss_js(left) - loss_js(right).  The two-way crossed
    random-effects model of Note S24, D_js = mu + u_j + v_s + e_js, is fitted by the ANOVA method of
    moments (components truncated at zero), and

        effect          D_bar = mean of D_js over graphs and seeds
        V(J, S)         s_u^2 / J + s_v^2 / S + s_e^2 / (J S)
        interval (J)    D_bar -/+ z_0.975 V(J, S)^(1/2)          (normal approximation)
        graph share g   [s_u^2 / J_old + s_e^2 / (J_old S)] / V(J_old, S)   (the part that scales with 1/J)
        P (J)           2 [1 - Phi(|D_bar| / V(J, S)^(1/2))]

    At J_new (the D_confirm size of FAMILY) the means are held fixed, V(J_new, S) =
    V(J_old, S) [(1 - g) + g J_old / J_new] (the rescaling the display generators apply), the Holm
    family is re-formed over the six original single-graph P values and the three rescaled P values,
    and the label-survival probability of a contrast at Holm rank r is
        Phi[(b D_bar - max(margin, z_{1 - alpha / (2 (m - r + 1))} V_new^(1/2))) / V_new^(1/2)]
    with b its beneficial direction: the probability that a new D_confirm estimate is Holm-significant
    at that step in the beneficial direction and beyond the practical margin (the E10 label rule).

Workflow (``scripts/run_confirmation_analysis.py {mechanism|confirmation} plan|run|export``) as in
cm_experiments.ensemble_grid: plan -> <output>/plan.json; run (one job with --job K) ->
<output>/jobs/...; export -> records/<set>.  Planning, statistics and export are torch-free.
"""
from __future__ import annotations

import contextlib
import json
import math
import shutil
import tempfile
from pathlib import Path
from statistics import NormalDist

import numpy as np

from .artifacts import sha256, write_json
from .ensemble_grid import (FEEDBACK_ROUTES, ROLE_STAGES, ParentContext, RunnerError, SealedRun, candidate_identity,
                            check_identity, create_plan, ensemble_probabilities, environment_record, job_directory,
                            json_safe, load_plan, member_runs, provenance_hardware, provenance_period, public_plan,
                            read_json, read_member, runtime_source_sha256, save_arrays, selected_jobs, slug, utc_now,
                            write_member, write_record_set)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_FILES = ("cm_experiments/confirmation_analysis.py", "cm_experiments/ensemble_grid.py",
                    "scripts/run_confirmation_analysis.py")
DATASETS = ("ogbn-arxiv", "ogbn-proteins", "ogbn-products", "Roman-empire", "PascalVOC-SP", "COCO-SP", "Peptides-func")
NORMAL = NormalDist()
Z975 = NORMAL.inv_cdf(0.975)
FIGURE3 = {"state": "state", "rh_minus_ol": "mpc"}          # record contrast key -> confirm-stage contrast kind
KIND_ROUTES = {"mpc": ("mpc_rh", "mpc_ol"), "radius": ("radius", "one_step")}
UNIT_WORDS = {"buffered_cluster": "clusters", "official_split": "splits", "graph": "graphs"}
CONFIRM_ROLE, ADMIT_ROLE = "confirm", "admit"
LABEL_STAGES = {CONFIRM_ROLE: "confirm", ADMIT_ROLE: "admit"}   # label role -> stage that owns and seals it


def _family():
    from cm_campaigns.confirmation import FAMILY
    return FAMILY


def multigraph_contrasts() -> dict:
    """{dataset: S14 contrast id} of the three multigraph benchmarks (unit = graph)."""
    return {spec[0]: key for key, spec in _family().items() if spec[2] == "graph"}


# ============================================================================= shared: roles, identities, export
def require_complete(run: SealedRun) -> None:
    """Every decision of the lifecycle is fixed: all eight stages sealed and the run COMPLETE."""
    run.require("evaluate")
    if run.state.get("status") != "COMPLETE":
        raise RunnerError(f"{run.root.name}: the lifecycle run is not complete")
    if not (run.root / "policy_lock.json").is_file():
        raise RunnerError(f"{run.root.name}: policy_lock.json is missing; the deployment is not fixed")


def check_label_role(run: SealedRun, role: str) -> None:
    """A label role is read only after the stage that owns it is sealed (confirm, admit)."""
    if role in ROLE_STAGES:
        from .ensemble_grid import check_role
        check_role(run, role)
        return
    if role not in LABEL_STAGES:
        raise RunnerError(f"label role must be one of {sorted(LABEL_STAGES)} (or compare/test)")
    run.require(LABEL_STAGES[role])


def experiment_code_identity() -> dict:
    return {name: sha256(ROOT / name) for name in EXPERIMENT_FILES if (ROOT / name).is_file()}


def experiment_code_files() -> dict:
    return {f"code/{name}": ROOT / name for name in EXPERIMENT_FILES if (ROOT / name).is_file()}


@contextlib.contextmanager
def registered_coverage(set_name: str, contract: dict):
    """The coverage the export declares, for scripts.scientific_evidence_audit.validate_record_coverage.

    write_record_set looks the set up in cm_campaigns.contract.RECORD_SETS.  A registered entry
    governs; otherwise the declared contract is registered for the duration of the export only.
    """
    from cm_campaigns.contract import RECORD_SETS
    if set_name in RECORD_SETS:
        yield RECORD_SETS[set_name]
        return
    RECORD_SETS[set_name] = contract
    try:
        yield contract
    finally:
        RECORD_SETS.pop(set_name, None)


def install_set(set_name, rows, files, provenance, *, coverage, root, replace, allow_partial, command, inputs,
                generated):
    with registered_coverage(set_name, coverage):
        return write_record_set(set_name, rows, files, provenance, root=root, replace=replace,
                                allow_partial=allow_partial, command=command, inputs=inputs, generated=generated)


def t_interval(values) -> dict:
    """Small-sample t interval over official splits (the Roman-empire unit)."""
    from scipy.stats import t as student
    x = np.asarray(values, dtype=np.float64)
    n = len(x)
    if n < 2 or not np.isfinite(x).all():
        raise RunnerError("a split interval needs at least two finite split contrasts")
    mean = float(math.fsum(x) / n)
    se = float(np.std(x, ddof=1) / math.sqrt(n))
    q = float(student.ppf(0.975, n - 1))
    return {"mean": mean, "lower": mean - q * se, "upper": mean + q * se, "se": se, "units": n,
            "t_quantile": q, "method": "mean of the split contrasts; two-sided 95% small-sample t interval over splits"}


def predecessor(candidates, current, route=None, profile=None):
    """cross_morse.pipeline.Pipeline.predecessor: the registered simpler policy of a candidate."""
    c = candidates[current]
    desired_route, desired_profile = route or c["route"], profile or c["profile"]
    for i, other in enumerate(candidates):
        if other["route"] != desired_route or other["profile"] != desired_profile or other["ensemble"] != c["ensemble"]:
            continue
        if desired_route in ("mpc_ol", "mpc_rh") and other.get("horizon") != c.get("horizon"):
            continue
        if desired_route == "radius" and other.get("kappa") != c.get("kappa"):
            continue
        return i
    return None


def _tie_key(candidate, index):
    from cross_morse.statistics import _candidate_key      # the registered deterministic tie order
    return _candidate_key(candidate, index)


# ============================================================================= execution helpers (torch)
class RoleContext(ParentContext):
    """A sealed parent opened with the confirmation and admission label roles.

    Labels are read inside the runtime's own label scope of the stage that owns the role, only
    after that stage is sealed (check_label_role).  Predictions of independent graphs are produced
    graph by graph with cross_morse.policies.execute_policy in the order and with the fresh policy
    state of cross_morse.runtime.predict_dataset; only the role's rows are kept in memory.
    """

    def labels(self, role):
        if role in ROLE_STAGES:
            return super().labels(role)
        check_label_role(self.run, role)
        if role not in self._labels:
            with self.data._label_scope(LABEL_STAGES[role]):
                self._labels[role] = self.data.labels(role)
        return self._labels[role]

    def execute(self, model, candidate, controller, role, *, donors=None):
        """Role rows of one member's predictions and predict_dataset's summed execution record."""
        from cross_morse.common import stop_at_boundary
        from cross_morse.policies import execute_policy
        sums = ("multiply_add_work", "deployed_multiply_add_work", "qualification_multiply_add_work",
                "support_violations", "shuffle_matched_sites", "shuffle_missing_sites", "wall_seconds")
        totals = {key: 0 for key in sums}
        totals.update(feasible=True, state_control_valid=True, inputs=0)
        rows = None
        for _, x, graph, local, positions in self.episodes(role):
            p, trace = execute_policy(model, x, graph, self.config, candidate, controller, self.reference,
                                      donors=donors)
            p = np.asarray(p, dtype=np.float32)[local]
            if not np.isfinite(p).all():
                raise RunnerError(f"nonfinite predictions on the {role} role")
            if rows is None:
                rows = np.full((len(self.role_ids(role)), p.shape[1]), np.nan, dtype=np.float32)
            rows[positions] = p
            for key in sums:
                totals[key] += float(trace.get(key, trace.get("multiply_add_work", 0) if key == "deployed_multiply_add_work"
                                               else 0) or 0)
            totals["feasible"] &= bool(trace.get("feasible", True))
            valid = trace.get("state_control_valid")
            totals["state_control_valid"] &= True if valid is None else bool(valid)
            totals["inputs"] += 1
            stop_at_boundary()
        if rows is None or not np.isfinite(rows).all():
            raise RunnerError(f"the {role} role was not executed completely")
        totals["execution_role"] = role if self.collection else "transductive_full_graph"
        return rows, totals


def _member_failure(error, seed):
    trace = getattr(error, "trace", {}) or {}
    return {"seed": int(seed), "reason": getattr(error, "reason", str(error)),
            "failure_stage": trace.get("failure_stage"), "failed_depth": trace.get("failed_depth")}


def execute_members(ctx, directory, key, candidate, seeds, role, *, donors=None):
    """Execute one policy's members (resume-safe: completed members are reused after verification).

    Returns (the ensemble prediction -- the runtime's float64 running sum over members in seed order
    divided by E, as float32 -- or None after a numerical failure, per-member records, failure record
    or None).  A numerical policy failure is recorded; no comparator bytes are substituted.
    """
    from cross_morse.policies import PolicyExecutionError
    controller = ctx.controller_for(candidate)
    expected = {"candidate": candidate_identity(candidate), "role": role,
                "state_control": candidate.get("state_control")}
    paths, records = [], []
    for seed in seeds:
        path = Path(directory) / "members" / key / f"seed{seed}.npy"
        if not path.with_suffix(".json").exists():
            model = ctx.plant(seed)
            try:
                rows, totals = ctx.execute(model, candidate, controller, role, donors=donors)
            except PolicyExecutionError as error:
                return None, records, _member_failure(error, seed)
            finally:
                del model
            write_member(path, rows, {**expected, "seed": int(seed), "trace": totals,
                                      "checkpoint_sha256": sha256(ctx.run.root / "train" / f"plant_seed{seed}_best.pt")})
            del rows
        record = read_json(path.with_suffix(".json"))
        paths.append(path)
        records.append({"seed": int(seed), "file": path.relative_to(directory).as_posix(), "sha256": sha256(path),
                        "trace": record["trace"]})
    return ensemble_probabilities(read_member(p, expected) for p in paths), records, None


def summed_execution(records) -> dict:
    """predict_candidate's member summary: summed counters, feasibility and state-control validity."""
    traces = [r["trace"] for r in records]
    return {"feasible": all(bool(t.get("feasible", True)) for t in traces),
            "state_control_valid": all(bool(t.get("state_control_valid", True)) for t in traces),
            "shuffle_matched_sites": sum(float(t.get("shuffle_matched_sites", 0)) for t in traces),
            "shuffle_missing_sites": sum(float(t.get("shuffle_missing_sites", 0)) for t in traces),
            "support_violations": sum(float(t.get("support_violations", 0)) for t in traces),
            "deployed_multiply_add_work": sum(float(t.get("deployed_multiply_add_work", 0)) for t in traces),
            "wall_seconds": sum(float(t.get("wall_seconds", 0)) for t in traces), "members": len(traces)}


# ============================================================================= Figure 3: which cells, which policies
def s14_supplies(dataset, kind) -> bool:
    return any(spec[0] == dataset and spec[1] == kind for spec in _family().values())


def run_supplies(runs, kind) -> bool:
    """main_paper._run_contrast: the deployed rule's own confirm/decision.json contrast of that kind."""
    for run in runs:
        admit = run.json("admit/decision.json")
        deployed = admit.get("deployed_index")
        if admit.get("Gadmit") != 1 or deployed == admit.get("comparator_index"):
            return False
        rows = [r for r in run.json("confirm/decision.json").get("contrasts") or []
                if r.get("kind") == kind and r.get("candidate_index") == deployed and r.get("mean") is not None]
        if not rows:
            return False
        if len(runs) == 1 and (rows[0].get("lower") is None or rows[0].get("upper") is None):
            return False
    return True


def required_cells(runs, dataset) -> list[str]:
    return [contrast for contrast, kind in FIGURE3.items()
            if not s14_supplies(dataset, kind) and not run_supplies(runs, kind)]


def strongest_feedback(run: SealedRun, override=None) -> tuple[dict, str]:
    """The dataset's strongest feedback configuration, fixed from the selection record."""
    candidates = run.candidates()
    selection = run.json("select/decision.json")
    if override is not None:
        index = int(override)
        if not 0 <= index < len(candidates) or candidates[index]["route"] not in FEEDBACK_ROUTES:
            raise RunnerError(f"{run.root.name}: anchor {override} is not a registered feedback candidate")
        return dict(candidates[index], registry_index=index), "declared in the plan (--anchor)"
    order = (("deployed rule", run.json("policy_lock.json")["deployed_index"]),
             ("confirmed policy q_M", run.json("confirm/decision.json")["confirmed_index"]),
             ("provisional policy of Eq. (19)", selection["provisional_index"]))
    for source, index in order:
        if candidates[index]["route"] in FEEDBACK_ROUTES and not candidates[index].get("execution_failure"):
            return dict(candidates[index], registry_index=index), source
    upper = selection["upper"]
    joint = selection.get("joint_feasible") or [False] * len(candidates)
    pool = [i for i, c in enumerate(candidates)
            if c["route"] in FEEDBACK_ROUTES and not c.get("execution_failure") and upper[i] is not None]
    if not pool:
        raise RunnerError(f"{run.root.name}: no executed feedback candidate in the selection record")
    best = min(pool, key=lambda i: (not joint[i], float(upper[i]), _tie_key(candidates[i], i)))
    scope = "joint-feasible" if joint[best] else "no joint-feasible feedback candidate; all executed ones"
    return (dict(candidates[best], registry_index=best),
            f"lowest D_select upper limit U among feedback candidates ({scope}; registered tie order)")


def replanning_pair(run: SealedRun, anchor: dict) -> tuple[dict, dict, str]:
    """MPC-RH at the anchor's registered ensemble size and profile, and its MPC-OL predecessor."""
    candidates = run.candidates()
    if anchor["route"] == "mpc_rh":
        rh, rule = anchor, "the anchor itself"
    else:
        selection = run.json("select/decision.json")
        upper = selection["upper"]
        joint = selection.get("joint_feasible") or [False] * len(candidates)
        pool = [i for i, c in enumerate(candidates)
                if c["route"] == "mpc_rh" and c["ensemble"] == anchor["ensemble"] and c["profile"] == anchor["profile"]
                and not c.get("execution_failure") and upper[i] is not None]
        if not pool:
            raise RunnerError(f"{run.root.name}: no executed MPC-RH at E={anchor['ensemble']}, profile {anchor['profile']}")
        best = min(pool, key=lambda i: (not joint[i], float(upper[i]), _tie_key(candidates[i], i)))
        rh = dict(candidates[best], registry_index=best)
        rule = ("MPC-RH at the anchor's ensemble size and profile with the lowest D_select upper limit U "
                "(joint-feasible first; registered tie order)")
    ol = predecessor(candidates, rh["registry_index"], route="mpc_ol")
    if ol is None:
        raise RunnerError(f"{run.root.name}: MPC-RH {rh['registry_index']} has no registered MPC-OL predecessor")
    return rh, dict(candidates[ol], registry_index=ol), rule


def shuffle_control(candidate: dict) -> dict:
    """The confirm stage's conditional state destruction of a feedback candidate."""
    return dict(candidate, state_control="matched_shuffle", candidate_id="shuffle_" + candidate["candidate_id"])


def mechanism_cells(run: SealedRun, contrasts, anchor_override=None) -> list[dict]:
    anchor, anchor_rule = strongest_feedback(run, anchor_override)
    cells = []
    for contrast in contrasts:
        if contrast == "state":
            cells.append({"contrast": "state", "kind": "state", "candidate": anchor, "reference": shuffle_control(anchor),
                          "policies": ["intact", "destroyed"], "anchor_rule": anchor_rule,
                          "definition": "loss(matched-shuffle state control) - loss(intact) per unit"})
        elif contrast == "rh_minus_ol":
            rh, ol, rule = replanning_pair(run, anchor)
            cells.append({"contrast": "rh_minus_ol", "kind": "mpc", "candidate": rh, "reference": ol,
                          "policies": ["mpc_rh", "mpc_ol"], "anchor_rule": anchor_rule, "pair_rule": rule,
                          "anchor": candidate_identity(anchor),
                          "definition": "loss(MPC-RH) - loss(MPC-OL) per unit, same horizon, profile and ensemble"})
        else:
            raise RunnerError(f"unknown Figure 3 contrast {contrast!r}; expected {sorted(FIGURE3)}")
    return cells


def _policy_key(candidate) -> str:
    """Unique per registered candidate (behavior and ensemble size); the state control is prefixed."""
    return slug(candidate["candidate_id"])


def _plan_cell(cell) -> dict:
    reference = candidate_identity(cell["reference"])
    if cell["kind"] == "state":
        reference["state_control"] = "matched_shuffle"
    out = {k: v for k, v in cell.items() if k not in ("candidate", "reference")}
    out.update(candidate=candidate_identity(cell["candidate"]), reference=reference,
               candidate_key=_policy_key(cell["candidate"]), reference_key=_policy_key(cell["reference"]))
    return out


# ============================================================================= mechanism_contrasts: planning
def plan_mechanism(parents, output, *, cells=None, anchors=None, reps=None) -> Path:
    """cells: optional {dataset: [contrast, ...]} replacing the Figure 3 rule; anchors: {dataset: index}."""
    if not parents:
        raise RunnerError("give at least one --parent run")
    datasets, jobs = [], []
    for parent in parents:
        runs = [SealedRun(m) for m in member_runs(parent)]
        dataset = runs[0].dataset
        if len({r.dataset for r in runs}) != 1 or any(d["dataset"] == dataset for d in datasets):
            raise RunnerError(f"{parent}: one parent (run or campaign) per dataset")
        for run in runs:
            require_complete(run)
            check_label_role(run, CONFIRM_ROLE)
        rule = "Figure 3 rule"
        wanted = required_cells(runs, dataset)
        if cells is not None and dataset in cells:
            wanted, rule = list(dict.fromkeys(cells[dataset])), "declared in the plan (--cell)"
        members = []
        for run in runs:
            if not wanted:
                break
            planned = mechanism_cells(run, wanted, (anchors or {}).get(dataset))
            policies = {}
            for cell in planned:
                for candidate in (cell["candidate"], cell["reference"]):
                    seeds = [int(s) for s in run.config["training"]["seeds"][:int(candidate["ensemble"])]]
                    policies[_policy_key(candidate)] = {"key": _policy_key(candidate), "candidate": candidate,
                                                        "seeds": seeds,
                                                        "uses_donors": bool(candidate.get("state_control"))}
            selection = run.json("select/decision.json")
            jobs.append({"job": len(jobs), "dataset": dataset, "slug": slug(dataset), "member": run.root.name,
                         "run": str(run.root), "identity": run.identity(), "role": CONFIRM_ROLE,
                         "cells": [_plan_cell(c) for c in planned], "policies": list(policies.values()),
                         "selection": {k: selection[k] for k in ("comparator_index", "provisional_index",
                                                                 "eligible_indices")},
                         "bootstrap": {"reps": int(reps or run.config["selection"]["bootstrap_reps"]),
                                       "seed": int(run.config["seed"]) + 5}})
            members.append(run.root.name)
        unit = ("official split (t interval over splits)" if len(runs) > 1 else
                "graph" if dataset in multigraph_contrasts() else "buffered cluster")
        datasets.append({"dataset": dataset, "slug": slug(dataset), "parent": str(Path(parent).resolve()),
                         "synthetic": runs[0].synthetic, "members": members, "cells": wanted, "cell_rule": rule,
                         "unit": unit,
                         "aggregation": "mean over the official split runs" if len(runs) > 1 else "single run"})
    if not jobs:
        raise RunnerError("no Figure 3 cell is left to measure: the S14 family and the deployed rules supply all")
    return create_plan(output, {
        "experiment": "Figure 3 held-out mechanism contrasts", "record_set": "mechanism_contrasts",
        "role": CONFIRM_ROLE, "datasets": datasets, "jobs": jobs, "experiment_code_sha256": experiment_code_identity(),
        "protocol": ("confirm-stage contrasts of cross_morse.pipeline on D_confirm of each sealed run for the Figure 3a-b "
                     "cells that neither the joint S14 family nor the deployed rule's confirm/decision.json supplies; "
                     "policies re-executed member by member with the released runtime; statistics by "
                     "cross_morse.statistics.confirmation (paired whole-unit bootstrap)")})


# ============================================================================= mechanism_contrasts: statistics (torch-free)
def mechanism_statistics(cells, scores, executions, selection, *, collection, reps, seed) -> list[dict]:
    """Each cell's paired contrast with the confirm stage's validity rules and statistics.

    ``scores``: {policy key: score_observations result or None (failed)}; ``executions``: {policy key:
    summed_execution(...) or {"execution_failure": True, ...}}.
    """
    from cross_morse.statistics import confirmation
    contrasts, notes = [], []
    for cell in cells:
        intact, other = cell["candidate_key"], cell["reference_key"]
        a, b = scores.get(intact), scores.get(other)
        ea, eb = executions.get(intact, {}), executions.get(other, {})
        row = {"kind": cell["kind"], "candidate_index": int(cell["candidate"]["registry_index"]),
               "predecessor_index": int(cell["reference"].get("registry_index", selection["comparator_index"]))}
        note = {"contrast": cell["contrast"], "reason": None}
        if cell["kind"] == "state":
            row["predecessor_index"] = int(selection["comparator_index"])
            if a is None or b is None:
                row.update(available=False, unavailable_reason="a member execution failed numerically")
                note["reason"] = row["unavailable_reason"]
            else:
                if not np.array_equal(a["core_ids"], b["core_ids"]):
                    raise RunnerError("state-control and intact policies were scored on different units")
                row.update(differences=b["loss"] - a["loss"], weights=a["counts"])
                intact_ok = True if not collection else (ea.get("feasible", False) and not ea.get("execution_failure"))
                if (eb.get("shuffle_missing_sites") or not eb.get("feasible") or not eb.get("shuffle_matched_sites")
                        or not eb.get("state_control_valid") or not intact_ok):
                    row.update(p_value=1.0, intervention_valid=False,
                               invalid_reason="missing/mismatched state donors or infeasible execution")
                    note["reason"] = row["invalid_reason"]
        else:
            ol_bad = eb.get("execution_failure") or (collection and not eb.get("feasible", False))
            if a is None or b is None or ol_bad:
                row.update(available=False, unavailable_reason="required simpler policy is absent or failed execution"
                           if (b is None or ol_bad) else "MPC-RH execution failed numerically")
                note["reason"] = row["unavailable_reason"]
            else:
                if not np.array_equal(a["core_ids"], b["core_ids"]):
                    raise RunnerError("MPC-RH and MPC-OL were scored on different units")
                row.update(differences=a["loss"] - b["loss"], weights=a["counts"])
        contrasts.append(row)
        notes.append(note)
    result = confirmation(selection["provisional_index"], selection["comparator_index"], selection["eligible_indices"],
                          contrasts, reps=reps, seed=seed)
    out = []
    for cell, row, contrast, note in zip(cells, result["contrasts"], contrasts, notes):
        valid = bool(row["available"] and row.get("intervention_valid", True) and row["valid_interval"])
        out.append({"contrast": cell["contrast"], "kind": cell["kind"], "valid": valid,
                    "invalid_reason": None if valid else (note["reason"] or "no valid interval (fewer than two units)"),
                    "effect_nats": row["mean"] if valid else None,
                    "interval_low_nats": row["lower"] if valid else None,
                    "interval_high_nats": row["upper"] if valid else None,
                    "units": row["units"], "weighting": None if "weights" not in contrast else
                    ("equal_unit" if np.all(np.asarray(contrast["weights"]) == np.asarray(contrast["weights"])[0])
                     else "fixed_unit_weight_ratio"),
                    "statistics": {k: v for k, v in row.items() if k != "holm_p_value_zero" and k != "passes"},
                    "bootstrap": {"reps": int(reps), "seed": int(seed), "contrast_offset": len(out),
                                  "interval": "central 95% paired whole-unit bootstrap (alpha = 0.025 each side)"}})
    return out


# ============================================================================= mechanism_contrasts: execution (torch)
def run_mechanism(output, *, job=None, device="cuda", trust=False) -> list[Path]:
    if not trust:
        raise RunnerError("loading local checkpoints requires explicit trust (--trust-local-checkpoints)")
    output = Path(output).resolve()
    plan = load_plan(output, "mechanism_contrasts")
    done = []
    for entry in selected_jobs(plan, job):
        directory = job_directory(output, entry)
        if not (directory / "mechanism.json").is_file():
            _run_mechanism_job(plan, entry, directory, device)
        done.append(directory / "mechanism.json")
    return done


def _run_mechanism_job(plan, entry, directory, device):
    from cross_morse.evaluation import score_observations
    run = SealedRun(entry["run"])
    check_identity(run, entry["identity"])
    require_complete(run)
    check_label_role(run, CONFIRM_ROLE)
    started = utc_now()
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / "environment.json").exists():
        write_json(directory / "environment.json", environment_record(device))
    ctx = RoleContext(run, device, through="evaluate")
    donors = None
    if any(p["uses_donors"] for p in entry["policies"]):
        from cross_morse.exploration import load_records
        from cross_morse.policies import DonorBank
        donors = DonorBank(load_records(run.root / "explore"))
    predictions, executions, records = {}, {}, {}
    for policy in entry["policies"]:
        ensemble, member_records, failure = execute_members(ctx, directory, policy["key"], policy["candidate"],
                                                            policy["seeds"], CONFIRM_ROLE,
                                                            donors=donors if policy["uses_donors"] else None)
        records[policy["key"]] = {"candidate": candidate_identity(policy["candidate"]),
                                  "state_control": policy["candidate"].get("state_control"),
                                  "members": member_records, "failure": failure}
        if failure is not None:
            executions[policy["key"]] = {"execution_failure": True, "feasible": False, **failure}
            continue
        predictions[policy["key"]] = ensemble
        executions[policy["key"]] = summed_execution(member_records)
    labels, units = ctx.labels(CONFIRM_ROLE), ctx.units(CONFIRM_ROLE)
    scores = {key: score_observations(p, labels, units, ctx.config) for key, p in predictions.items()}
    cells = mechanism_statistics(entry["cells"], scores, executions, entry["selection"], collection=ctx.collection,
                                 reps=entry["bootstrap"]["reps"], seed=entry["bootstrap"]["seed"])
    arrays = {}
    for key, score in scores.items():
        arrays.setdefault("unit_id", score["core_ids"])
        arrays.setdefault("unit_weight", score["counts"])
        arrays[f"{key}_unit_loss"] = score["loss"]
    write_mechanism_result(directory, entry, cells=cells, policies=records, executions=executions, arrays=arrays,
                           unit_kind="graph" if ctx.collection else "buffered cluster", parent_files=run.inputs(),
                           started=started)


def write_mechanism_result(directory, entry, *, cells, policies, executions, arrays, unit_kind, parent_files, started,
                           environment=None) -> Path:
    """The job's raw outputs: unit_losses.npz and mechanism.json (written last: the completion marker)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if environment is not None and not (directory / "environment.json").exists():
        write_json(directory / "environment.json", json_safe(environment))
    planned = {c["contrast"]: c for c in entry["cells"]}
    if [c["contrast"] for c in cells] != list(planned):
        raise RunnerError("the computed cells differ from the planned cells")
    for cell in cells:
        plan_cell = planned[cell["contrast"]]
        cell.update(candidate=plan_cell["candidate"], reference=plan_cell["reference"],
                    anchor_rule=plan_cell.get("anchor_rule"), pair_rule=plan_cell.get("pair_rule"),
                    definition=plan_cell.get("definition"), unit=unit_kind)
    unit_file = save_arrays(directory / "unit_losses.npz", replace=True, **arrays) if arrays else None
    write_json(directory / "mechanism.json", json_safe({
        "schema": "cm-mechanism-contrasts-member-1", "dataset": entry["dataset"], "member": entry["member"],
        "role": entry["role"], "cells": cells, "policies": policies, "executions": executions,
        "unit_losses": None if unit_file is None else {"file": "unit_losses.npz", "sha256": unit_file},
        "identities": {"runtime_source_sha256": runtime_source_sha256(), "parent": entry["identity"],
                       "plan_sha256": sha256(directory.parents[2] / "plan.json"), "parent_files_read": parent_files,
                       "experiment_code_sha256": experiment_code_identity()},
        "statistics_code": "cross_morse.statistics.confirmation (paired_bootstrap, alpha = 0.025 each side)",
        "started_utc": started, "finished_utc": utc_now()}))
    return directory / "mechanism.json"


# ============================================================================= mechanism_contrasts: export (torch-free)
def aggregate_cells(dataset, results, cells) -> list[dict]:
    """One value per planned cell: the single run's contrast, or the split mean with its t interval."""
    out = []
    for contrast in cells:
        members = []
        for member, result in results:
            cell = next((c for c in result["cells"] if c["contrast"] == contrast), None)
            if cell is None:
                raise RunnerError(f"{dataset}/{member}: cell {contrast} is missing from the job result")
            members.append({"member": member, "valid": cell["valid"], "effect_nats": cell["effect_nats"],
                            "interval_low_nats": cell["interval_low_nats"],
                            "interval_high_nats": cell["interval_high_nats"], "units": cell["units"],
                            "invalid_reason": cell["invalid_reason"], "candidate": cell["candidate"],
                            "reference": cell["reference"]})
        valid = all(m["valid"] for m in members)
        entry = {"dataset": dataset, "contrast": contrast, "valid": valid, "members": members,
                 "invalid_reason": None if valid else "; ".join(f"{m['member']}: {m['invalid_reason']}"
                                                                for m in members if not m["valid"])}
        if valid and len(members) == 1:
            m = members[0]
            entry.update(effect_nats=m["effect_nats"], interval_low_nats=m["interval_low_nats"],
                         interval_high_nats=m["interval_high_nats"], units=m["units"],
                         interval="paired whole-unit bootstrap of the confirm stage (central 95%)")
        elif valid:
            interval = t_interval([m["effect_nats"] for m in members])
            entry.update(effect_nats=interval["mean"], interval_low_nats=interval["lower"],
                         interval_high_nats=interval["upper"], units=interval["units"], interval=interval["method"],
                         t_quantile=interval["t_quantile"], split_se=interval["se"])
        out.append(entry)
    return out


def export_mechanism(output, *, root=None, replace=False, allow_partial=False, hardware=None, period=None,
                     command=None) -> Path:
    from scripts.records import RECORDS, is_official_root
    output = Path(output).resolve()
    plan = load_plan(output, "mechanism_contrasts")
    staging = Path(tempfile.mkdtemp(prefix="cm-mechanism-export-"))
    try:
        files, rows, results_all, environments, inputs, expected = experiment_code_files(), [], [], [], [], []
        missing = []
        for dataset in plan["datasets"]:
            entries = [j for j in plan["jobs"] if j["dataset"] == dataset["dataset"]]
            if not entries:
                continue
            results, base = [], f"raw/{dataset['slug']}"
            for entry in entries:
                directory = job_directory(output, entry)
                path = directory / "mechanism.json"
                if not path.is_file():
                    raise RunnerError(f"{dataset['dataset']}/{entry['member']}: job {entry['job']} has not completed")
                result = read_json(path)
                name = slug(entry["member"])
                files[f"{base}/{name}/mechanism.json"] = path
                files[f"{base}/{name}/environment.json"] = directory / "environment.json"
                if (directory / "unit_losses.npz").is_file():
                    files[f"{base}/{name}/unit_losses.npz"] = directory / "unit_losses.npz"
                results.append((entry["member"], result))
                results_all.append(result)
                environments.append(read_json(directory / "environment.json"))
                inputs.append({"file": str(path.relative_to(output)), "sha256": sha256(path)})
            cells = aggregate_cells(dataset["dataset"], results, dataset["cells"])
            summary = {"schema": "cm-mechanism-contrasts-dataset-1", "dataset": dataset["dataset"], "role": "D_confirm",
                       "unit": dataset["unit"], "aggregation": dataset["aggregation"], "cells": cells,
                       "definition": "paired D_confirm contrast of the dataset's strongest feedback configuration "
                                     "(nats): state = matched-shuffle state destruction minus intact; rh_minus_ol = "
                                     "MPC-RH minus MPC-OL at the registered horizon and ensemble size"}
            path = staging / dataset["slug"] / "mechanism_contrasts.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(json_safe(summary), indent=1, sort_keys=True, allow_nan=False) + "\n")
            files[f"{base}/mechanism_contrasts.json"] = path
            for i, cell in enumerate(cells):
                expected.append([dataset["dataset"], cell["contrast"]])
                if not cell["valid"]:
                    missing.append(f"{dataset['dataset']}/{cell['contrast']}: {cell['invalid_reason']}")
                    continue
                for quantity in ("effect_nats", "interval_low_nats", "interval_high_nats"):
                    rows.append({"dataset": dataset["dataset"], "contrast": cell["contrast"], "quantity": quantity,
                                 "value": cell[quantity], "unit": "nats",
                                 "source_file": f"{base}/mechanism_contrasts.json",
                                 "locator": f"json:/cells/{i}/{quantity}"})
        if missing and not allow_partial:
            raise RunnerError("cells without a valid measurement (no value is substituted):\n  " + "\n  ".join(missing))
        covered = {d["dataset"] for d in plan["datasets"]}
        target_root = Path(root or RECORDS).resolve()
        if is_official_root(target_root) and covered != set(DATASETS):
            raise RunnerError("records/mechanism_contrasts must be planned from all seven benchmark runs (the Figure 3 "
                              f"cells depend on every run); missing {sorted(set(DATASETS) - covered)}")
        synthetic = {d["synthetic"] for d in plan["datasets"]}
        if len(synthetic) != 1:
            raise RunnerError("synthetic and benchmark parents cannot share one record set")
        provenance = {
            "experiment": "Held-out mechanism contrasts outside the nine-test family (main Figure 3a-b)",
            "produced_by": "scripts/run_confirmation_analysis.py mechanism plan/run/export "
                           "(cm_experiments/confirmation_analysis.py)",
            "hardware": hardware or provenance_hardware(environments), "period": period or provenance_period(results_all),
            "protocol": ("confirm-stage contrasts (cross_morse.pipeline._stage_confirm rules, "
                         "cross_morse.statistics.confirmation) on D_confirm of each sealed run for every Figure 3a-b cell "
                         "that neither the joint S14 family nor the deployed rule's confirm/decision.json supplies; "
                         "strongest feedback configuration = deployed, confirmed or provisional feedback route, else the "
                         "lowest D_select U; MPC-RH/MPC-OL at its ensemble size, profile and the lowest-U horizon; "
                         "intervals: paired whole-unit bootstrap (clusters, graphs) or t over Roman splits"),
            "synthetic": synthetic.pop(), "role": "confirm", "plan_sha256": sha256(output / "plan.json"),
            "experiment_code_sha256": experiment_code_identity(),
            "cells": [{"dataset": d["dataset"], "cells": d["cells"], "rule": d["cell_rule"]} for d in plan["datasets"]]}
        coverage = {"tables": ["Figure 3"], "datasets": sorted(covered), "coverage_keys": ["dataset", "contrast"],
                    "coverage_rows": expected, "analyzer": "mechanism_contrasts"}
        return install_set("mechanism_contrasts", rows, files, provenance, coverage=coverage, root=root,
                           replace=replace, allow_partial=allow_partial, command=command, inputs=inputs,
                           generated={"code/plan.json": public_plan(output)})
    finally:
        shutil.rmtree(staging, ignore_errors=True)


# ============================================================================= S14b / S18b: the joint S14 campaign
def campaign_family(campaign, manuscript_root=ROOT) -> dict:
    """The validated joint S14 campaign: its nine rows, original test records and run bindings."""
    from cm_campaigns.common import EvidenceError, confined, directory_inventory
    from cm_campaigns.confirmation import check_campaign
    campaign = Path(campaign).resolve()
    try:
        before = directory_inventory(campaign)
        checked = check_campaign(campaign, manuscript_root=Path(manuscript_root))
        manifest = read_json(campaign / "campaign.json")
        tests = {}
        for entry in manifest["tests"]:
            tests[entry["contrast_id"]] = {"result": confined(campaign, entry["result"]["path"]),
                                           "paired": confined(campaign, entry["paired_losses"]["path"])}
        if directory_inventory(campaign) != before:
            raise EvidenceError("campaign files changed during validation")
    except (EvidenceError, OSError, KeyError) as exc:
        raise RunnerError(f"joint S14 campaign failed validation: {exc}") from None
    rows = {row["contrast_id"]: row for row in checked["rows"]}
    for key, files in tests.items():
        files["result_json"] = read_json(files["result"])
    return {"root": campaign, "status": checked["status"], "input_identity": checked["input_identity"],
            "rows": rows, "tests": tests, "bindings": checked["official_run_bindings"]}


def check_binding(run: SealedRun, family: dict, dataset: str) -> str:
    """The run whose validation split is analyzed is the original run the S14 test was taken from."""
    bindings = family["bindings"].get(dataset) or []
    identity = run.identity()
    for row in bindings:
        ids = row["identities"]
        if (ids["config_sha256"], ids["unit_sha256"], ids["source_sha256"]) == (
                identity["config_sha256"], identity["units_sha256"], identity["runtime_source_sha256"]):
            return "bound to the campaign's original run identity"
    if run.synthetic:
        return "not bound (synthetic software fixture)"
    raise RunnerError(f"{dataset}: the parent run's configuration/unit/source identity is not the joint campaign's "
                      "original run of this dataset")


def policy_pair(run: SealedRun, key: str, family=None, *, allow_fixture=False) -> tuple[dict, dict, str]:
    """The S14 test's (left, right) policies in this run's candidate ledger."""
    dataset, kind = _family()[key][:2]
    left_route, right_route = KIND_ROUTES[kind]
    candidates = run.candidates()
    if family is not None:
        row = family["rows"][key]
        by_hash = {c["hash"]: i for i, c in enumerate(candidates)}
        li, ri = by_hash.get(row["left_policy_sha256"]), by_hash.get(row["right_policy_sha256"])
        if li is None or ri is None:
            raise RunnerError(f"{key}: the S14 test policies are absent from this run's candidate ledger")
        source = "joint S14 campaign policy hashes"
    else:
        selection = run.json("select/decision.json")
        rows = [r for r in run.json("confirm/decision.json").get("contrasts") or []
                if r.get("kind") == kind and r.get("candidate_index") == selection["provisional_index"]
                and r.get("available", True)]
        if rows:
            li, ri = int(rows[0]["candidate_index"]), int(rows[0]["predecessor_index"])
            source = "the run's own confirmation contrast of the provisional policy"
        elif allow_fixture and run.synthetic:
            comparator = run.comparator()
            upper = selection["upper"]
            pool = [i for i, c in enumerate(candidates) if c["route"] == left_route and c["profile"] == "full"
                    and c["ensemble"] == comparator["ensemble"] and not c.get("execution_failure")
                    and upper[i] is not None]
            if not pool:
                raise RunnerError(f"{run.root.name}: no registered {left_route} at the comparator's ensemble size")
            li = min(pool, key=lambda i: (float(upper[i]), _tie_key(candidates[i], i)))
            ri = predecessor(candidates, li, route=right_route)
            if ri is None:
                raise RunnerError(f"{run.root.name}: no registered {right_route} predecessor")
            source = "fixture: registered pair at the comparator's ensemble size (synthetic parent only)"
        else:
            raise RunnerError(f"{key}: neither the run's confirmation record nor a campaign identifies the S14 policy "
                              "pair; pass --campaign")
    left, right = candidates[li], candidates[ri]
    if (left["route"], right["route"]) != (left_route, right_route) or left["ensemble"] != right["ensemble"] \
            or predecessor(candidates, li, route=right_route) != ri:
        raise RunnerError(f"{key}: the policy pair is not {left_route} and its registered {right_route} predecessor")
    return dict(left, registry_index=li), dict(right, registry_index=ri), source


def campaign_seeds(family, key) -> list[int]:
    values = family["rows"][key]["seed_ids"]
    try:
        return sorted(int(v) for v in values)
    except (TypeError, ValueError):
        raise RunnerError(f"{key}: the campaign's seed IDs are not checkpoint seeds: {values}") from None


# ============================================================================= S14b / S18b: planning
def plan_confirmation(parents, output, *, campaign=None, manuscript_root=ROOT, seeds=None, reps=None,
                      allow_fixture=False) -> Path:
    """One job per (multigraph benchmark, optimization seed): both S14 policies on the validation split."""
    if not parents:
        raise RunnerError("give the PascalVOC-SP, COCO-SP and Peptides-func runs with --parent")
    family = campaign_family(campaign, manuscript_root) if campaign is not None else None
    contrasts = multigraph_contrasts()
    datasets, jobs = [], []
    for parent in parents:
        runs = [SealedRun(m) for m in member_runs(parent)]
        if len(runs) != 1 or runs[0].dataset not in contrasts:
            raise RunnerError(f"{parent}: J_old contrasts are defined for one run of {sorted(contrasts)}")
        run = runs[0]
        if any(d["dataset"] == run.dataset for d in datasets):
            raise RunnerError(f"{run.dataset}: one parent run per dataset")
        require_complete(run)
        check_label_role(run, ADMIT_ROLE)
        key = contrasts[run.dataset]
        left, right, source = policy_pair(run, key, family, allow_fixture=allow_fixture)
        trained = [int(s) for s in run.config["training"]["seeds"]]
        seed_source = "the run's training seeds"
        chosen = [int(s) for s in seeds] if seeds is not None else None
        if chosen is None and family is not None:
            chosen, seed_source = campaign_seeds(family, key), "the S14 test's seed population"
            if not set(chosen) <= set(trained):
                if not run.synthetic:
                    raise RunnerError(f"{key}: S14 seeds {chosen} are not checkpoints of this run ({trained})")
                chosen, seed_source = trained, "the run's training seeds (fixture: campaign seeds not checkpoints)"
        chosen = sorted(set(chosen or trained))
        if len(chosen) < 2 or not set(chosen) <= set(trained):
            raise RunnerError(f"{key}: the two-way model needs at least two trained checkpoint seeds; got {chosen}")
        binding = check_binding(run, family, run.dataset) if family is not None else "no campaign at planning"
        dataset, kind, _, j_new, _, direction, margin = _family()[key]
        datasets.append({"dataset": run.dataset, "slug": slug(run.dataset), "contrast": key, "kind": kind,
                         "synthetic": run.synthetic, "member": run.root.name, "parent": str(run.root),
                         "seeds": chosen, "seed_source": seed_source, "J_new": int(j_new),
                         "beneficial_direction": int(direction), "practical_margin_nats": float(margin),
                         "left": candidate_identity(left), "right": candidate_identity(right), "pair_source": source,
                         "campaign_binding": binding,
                         "bootstrap": {"reps": int(reps or run.config["admission"]["bootstrap_reps"]),
                                       "seed": int(run.config["seed"]) + 5}})
        for seed in chosen:
            jobs.append({"job": len(jobs), "dataset": run.dataset, "slug": slug(run.dataset), "contrast": key,
                         "member": f"{run.root.name}__seed{seed}", "run": str(run.root), "identity": run.identity(),
                         "seed": int(seed), "role": ADMIT_ROLE, "left": left, "right": right})
    return create_plan(output, {
        "experiment": "E10", "record_set": "confirmation_contrasts",
        "record_sets": ["confirmation_contrasts", "confirmation_sensitivity"], "role": ADMIT_ROLE,
        "datasets": datasets, "jobs": jobs, "experiment_code_sha256": experiment_code_identity(),
        "campaign": None if family is None else {"path": str(family["root"]), "input_identity": family["input_identity"],
                                                 "status": family["status"]},
        "alpha": _holm_alpha(),
        "protocol": ("S14 policy pair of each multigraph contrast executed on the official validation split (J_old "
                     "graphs) with every seed of the S14 seed population, one checkpoint at a time; paired per-graph, "
                     "per-seed losses; two-way seed-by-graph variance model of Note S24 fitted by the method of moments")})


def _holm_alpha() -> float:
    constants = json.loads((ROOT / "paper" / "registered_constants.json").read_text(encoding="utf-8"))
    return float(constants["confirmation_eq20_table_S11"]["holm_alpha"])


# ============================================================================= S14b / S18b: execution (torch)
def run_confirmation(output, *, job=None, device="cuda", trust=False) -> list[Path]:
    if not trust:
        raise RunnerError("loading local checkpoints requires explicit trust (--trust-local-checkpoints)")
    output = Path(output).resolve()
    plan = load_plan(output, "confirmation_contrasts")
    done = []
    for entry in selected_jobs(plan, job):
        directory = job_directory(output, entry)
        if not (directory / "pair.json").is_file():
            _run_confirmation_job(plan, entry, directory, device)
        done.append(directory / "pair.json")
    return done


def _sealed_member(run, candidate, seed, role):
    """The lifecycle's own member prediction of a policy on a role, when the run stored one."""
    path = run.root / "member_predictions" / role / f"{candidate['behavior_id']}_seed{seed}.npy"
    return path if path.is_file() else None


def _run_confirmation_job(plan, entry, directory, device):
    from cross_morse.evaluation import score_observations
    from cross_morse.policies import PolicyExecutionError
    run = SealedRun(entry["run"])
    check_identity(run, entry["identity"])
    require_complete(run)
    check_label_role(run, ADMIT_ROLE)
    started = utc_now()
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / "environment.json").exists():
        write_json(directory / "environment.json", environment_record(device))
    ctx = RoleContext(run, device, through="evaluate")
    if not ctx.collection:
        raise RunnerError("J_old contrasts are defined on the independent-graph benchmarks")
    seed = int(entry["seed"])
    model = ctx.plant(seed)
    predictions, executions, reproduction = {}, {}, {}
    for side in ("left", "right"):
        candidate = entry[side]
        try:
            rows, totals = ctx.execute(model, candidate, ctx.controller_for(candidate), ADMIT_ROLE)
        except PolicyExecutionError as error:
            raise RunnerError(f"{entry['contrast']}: the {side} policy failed numerically at seed {seed} "
                              f"({error.reason}); the two-way array cannot be completed") from None
        predictions[side], executions[side] = rows, totals
        sealed = _sealed_member(run, candidate, seed, ADMIT_ROLE)
        if sealed is not None:
            stored = np.load(sealed, mmap_mode="r", allow_pickle=False)[ctx.role_ids(ADMIT_ROLE)]
            reproduction[side] = {"sealed_member_prediction": sealed.relative_to(run.root).as_posix(),
                                  "max_abs_difference": float(np.max(np.abs(np.asarray(stored, dtype=np.float64) - rows)))}
    labels, units = ctx.labels(ADMIT_ROLE), ctx.units(ADMIT_ROLE)
    left = score_observations(predictions["left"], labels, units, ctx.config)
    right = score_observations(predictions["right"], labels, units, ctx.config)
    if not np.array_equal(left["core_ids"], right["core_ids"]):
        raise RunnerError("the two policies were scored on different graphs")
    write_pair_result(directory, entry, unit_id=left["core_ids"], left_loss=left["loss"], right_loss=right["loss"],
                      executions=executions, reproduction=reproduction, parent_files=run.inputs(), started=started)


def write_pair_result(directory, entry, *, unit_id, left_loss, right_loss, executions, reproduction, parent_files,
                      started, environment=None) -> Path:
    """One seed's per-graph losses of both policies (losses.npz) and pair.json (the completion marker)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if environment is not None and not (directory / "environment.json").exists():
        write_json(directory / "environment.json", json_safe(environment))
    left_loss, right_loss = np.asarray(left_loss, dtype=np.float64), np.asarray(right_loss, dtype=np.float64)
    if left_loss.shape != right_loss.shape or left_loss.shape != np.shape(unit_id) or not np.isfinite(
            np.r_[left_loss, right_loss]).all():
        raise RunnerError("per-graph losses must be finite and aligned with the graph IDs")
    digest = save_arrays(directory / "losses.npz", replace=True, unit_id=np.asarray(unit_id), left_loss=left_loss,
                         right_loss=right_loss)
    write_json(directory / "pair.json", json_safe({
        "schema": "cm-confirmation-jold-seed-1", "experiment": "E10", "dataset": entry["dataset"],
        "contrast": entry["contrast"], "member": entry["member"], "seed": int(entry["seed"]), "role": entry["role"],
        "left": candidate_identity(entry["left"]), "right": candidate_identity(entry["right"]),
        "graphs": int(len(left_loss)), "losses": {"file": "losses.npz", "sha256": digest},
        "executions": executions, "reproduction": reproduction,
        "loss_definition": "per-graph mean CE/BCE of one checkpoint's prediction (cross_morse.evaluation.score_observations)",
        "identities": {"runtime_source_sha256": runtime_source_sha256(), "parent": entry["identity"],
                       "plan_sha256": sha256(directory.parents[2] / "plan.json"), "parent_files_read": parent_files,
                       "experiment_code_sha256": experiment_code_identity()},
        "started_utc": started, "finished_utc": utc_now()}))
    return directory / "pair.json"


# ============================================================================= S14b / S18b: statistics (torch-free)
def two_way_fit(differences) -> dict:
    """ANOVA method-of-moments fit of D_js = mu + u_j + v_s + e_js (graphs j crossed with seeds s).

    E[MS_graph] = s_e^2 + S s_u^2, E[MS_seed] = s_e^2 + J s_v^2, E[MS_resid] = s_e^2; negative
    component estimates are truncated at zero (the untruncated values are kept).
    """
    d = np.asarray(differences, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] < 2 or d.shape[1] < 2 or not np.isfinite(d).all():
        raise RunnerError("the two-way fit needs a finite [graphs >= 2, seeds >= 2] array")
    j, s = d.shape
    grand = float(d.mean())
    graph_means, seed_means = d.mean(axis=1), d.mean(axis=0)
    residual = d - graph_means[:, None] - seed_means[None, :] + grand
    ms_graph = float(s * np.sum((graph_means - grand) ** 2) / (j - 1))
    ms_seed = float(j * np.sum((seed_means - grand) ** 2) / (s - 1))
    ms_resid = float(np.sum(residual ** 2) / ((j - 1) * (s - 1)))
    raw = {"graph": (ms_graph - ms_resid) / s, "seed": (ms_seed - ms_resid) / j, "residual": ms_resid}
    return {"graphs": j, "seeds": s, "mean": grand, "ms_graph": ms_graph, "ms_seed": ms_seed, "ms_residual": ms_resid,
            "df_graph": j - 1, "df_seed": s - 1, "df_residual": (j - 1) * (s - 1),
            "sigma2_graph_untruncated": raw["graph"], "sigma2_seed_untruncated": raw["seed"],
            "sigma2_graph": max(0.0, raw["graph"]), "sigma2_seed": max(0.0, raw["seed"]),
            "sigma2_residual": max(0.0, raw["residual"]),
            "variance_of_mean_unbiased": (ms_graph + ms_seed - ms_resid) / (j * s),
            "estimator": "ANOVA method of moments, crossed graphs x seeds, one observation per cell"}


def variance_of_mean(fit, graphs, seeds) -> float:
    return fit["sigma2_graph"] / graphs + fit["sigma2_seed"] / seeds + fit["sigma2_residual"] / (graphs * seeds)


def sensitivity(fit, j_new) -> dict:
    """J_old interval, graph share and the D_confirm-size variance with fixed means (Note S24)."""
    j_old, s = fit["graphs"], fit["seeds"]
    v_old = variance_of_mean(fit, j_old, s)
    if not v_old > 0:
        raise RunnerError("the two-way fit gives zero variance; the interval and the shares are undefined")
    graph_part = fit["sigma2_graph"] / j_old + fit["sigma2_residual"] / (j_old * s)
    share = graph_part / v_old
    v_new = variance_of_mean(fit, int(j_new), s)
    factor = math.sqrt((1.0 - share) + share * j_old / int(j_new))
    if not math.isclose(math.sqrt(v_new), math.sqrt(v_old) * factor, rel_tol=1e-9):
        raise RunnerError("internal error: the rescaling identity of the two-way model does not hold")
    se_old, se_new, effect = math.sqrt(v_old), math.sqrt(v_new), fit["mean"]
    return {"J_old": j_old, "J_new": int(j_new), "seeds": s, "effect_nats": effect, "se_old": se_old,
            "interval_low_nats": effect - Z975 * se_old, "interval_high_nats": effect + Z975 * se_old,
            "p_value_zero_old": 2.0 * (1.0 - NORMAL.cdf(abs(effect) / se_old)),
            "graph_variance_share": share, "seed_variance_share": fit["sigma2_seed"] / s / v_old,
            "rescaling_factor": factor, "se_new": se_new,
            "interval_new_low_nats": effect - Z975 * se_new, "interval_new_high_nats": effect + Z975 * se_new,
            "p_value_zero_new": 2.0 * (1.0 - NORMAL.cdf(abs(effect) / se_new)),
            "satterthwaite_df_unbiased": _satterthwaite(fit)}


def _satterthwaite(fit):
    """Approximate degrees of freedom of the unbiased V estimate (diagnostic; None when undefined)."""
    terms = [(fit["ms_graph"], fit["df_graph"]), (fit["ms_seed"], fit["df_seed"]), (-fit["ms_residual"], fit["df_residual"])]
    total = sum(m for m, _ in terms)
    denominator = sum(m * m / df for m, df in terms)
    return None if total <= 0 or denominator <= 0 else float(total * total / denominator)


def holm_family(p_values, alpha) -> dict:
    """Holm step-down over the nine tests in FAMILY order; stable ranks, multipliers m - r + 1."""
    from cross_morse.statistics import holm_adjust
    keys = list(_family())
    if set(p_values) != set(keys):
        raise RunnerError("the Holm family needs all nine P values")
    p = np.asarray([float(p_values[k]) for k in keys], dtype=np.float64)
    adjusted = holm_adjust(p)
    order = np.argsort(p, kind="stable")
    rank = np.empty(len(p), dtype=np.int64)
    rank[order] = np.arange(1, len(p) + 1)
    m = len(p)
    return {key: {"raw_p": float(p[i]), "holm_p": float(adjusted[i]), "rank": int(rank[i]),
                  "multiplier": int(m - rank[i] + 1), "step_level": float(alpha / (m - rank[i] + 1))}
            for i, key in enumerate(keys)}


def label_survival(effect, se_new, direction, margin, multiplier, alpha) -> dict:
    """Probability that a new estimate ~ N(effect, se_new^2) keeps the Holm-significant label.

    The label requires two-sided significance at the contrast's Holm step (alpha / multiplier) in
    the beneficial direction and a point estimate beyond the practical margin.
    """
    critical = NORMAL.inv_cdf(1.0 - alpha / (2.0 * multiplier))
    threshold = max(float(margin), critical * se_new)
    return {"probability": NORMAL.cdf((direction * effect - threshold) / se_new), "critical_z": critical,
            "threshold_nats": threshold, "binding": "margin" if threshold == float(margin) else "significance"}


# ============================================================================= S14b / S18b: analysis and export (torch-free)
def paired_array(output, plan, dataset) -> dict:
    """The [graphs, seeds] paired array of one multigraph contrast from its per-seed job outputs."""
    entries = [j for j in plan["jobs"] if j["dataset"] == dataset["dataset"]]
    if [j["seed"] for j in entries] != dataset["seeds"]:
        raise RunnerError(f"{dataset['dataset']}: the planned seeds have no complete job list")
    ids, left, right, results = None, [], [], []
    for entry in entries:
        directory = job_directory(output, entry)
        path = directory / "pair.json"
        if not path.is_file():
            raise RunnerError(f"{dataset['dataset']}/seed {entry['seed']}: job {entry['job']} has not completed")
        result = read_json(path)
        if sha256(directory / "losses.npz") != result["losses"]["sha256"]:
            raise RunnerError(f"{directory}/losses.npz differs from its recorded SHA-256")
        if (result["left"] != candidate_identity(entry["left"]) or result["right"] != candidate_identity(entry["right"])
                or result["seed"] != entry["seed"]):
            raise RunnerError(f"{path}: policies or seed differ from the plan")
        with np.load(directory / "losses.npz", allow_pickle=False) as data:
            unit, l_loss, r_loss = data["unit_id"], data["left_loss"], data["right_loss"]
        if ids is None:
            ids = unit
        elif not np.array_equal(ids, unit):
            raise RunnerError(f"{dataset['dataset']}: seeds were scored on different validation graphs")
        left.append(l_loss)
        right.append(r_loss)
        results.append((entry, directory, result))
    left, right = np.stack(left, axis=1), np.stack(right, axis=1)
    return {"unit_id": ids, "seeds": np.asarray(dataset["seeds"], dtype=np.int64), "left_loss": left,
            "right_loss": right, "differences": left - right, "results": results}


def single_graph_tests(family) -> dict:
    """The six original D_confirm tests of the single-graph benchmarks, as validated by the auditor."""
    out = {}
    for key, spec in _family().items():
        if spec[2] == "graph":
            continue
        row, files = family["rows"][key], family["tests"][key]
        result = files["result_json"]
        for field in ("effect_nats", "p_value_zero", "unit_count"):
            if result.get(field) != row.get(field):
                raise RunnerError(f"{key}: the original test record and the validated campaign row differ in {field}")
        out[key] = {"effect_nats": float(result["effect_nats"]), "interval_nats": [float(v) for v in result["interval_nats"]],
                    "p_value_zero": float(result["p_value_zero"]), "unit_count": int(result["unit_count"]),
                    "tail": result.get("tail"), "method_name": result.get("method_name"),
                    "statistical_unit": result.get("statistical_unit"), "result_file": files["result"],
                    "paired_file": files["paired"]}
    return out


def analyze_confirmation(output, plan, single, *, alpha) -> tuple[dict, dict]:
    """The one raw computation behind both record sets: arrays, two-way fits, J_new family."""
    family_keys = list(_family())
    contrasts, arrays = [], {}
    for dataset in plan["datasets"]:
        array = paired_array(output, plan, dataset)
        fit = two_way_fit(array["differences"])
        derived = sensitivity(fit, dataset["J_new"])
        graph_means = array["differences"].mean(axis=1)
        from cross_morse.statistics import paired_bootstrap
        fixed = paired_bootstrap(graph_means, reps=dataset["bootstrap"]["reps"], alpha=0.025,
                                 seed=dataset["bootstrap"]["seed"])
        contrasts.append({"contrast": dataset["contrast"], "dataset": dataset["dataset"], "kind": dataset["kind"],
                          "left": dataset["left"], "right": dataset["right"], "pair_source": dataset["pair_source"],
                          "seeds": dataset["seeds"], "seed_source": dataset["seed_source"],
                          "campaign_binding": dataset["campaign_binding"],
                          "beneficial_direction": dataset["beneficial_direction"],
                          "practical_margin_nats": dataset["practical_margin_nats"], "fit": fit, **derived,
                          "units_old": derived["J_old"],
                          "fixed_seed_graph_bootstrap": {"mean": fixed["mean"], "lower": fixed["lower"],
                                                         "upper": fixed["upper"], "units": fixed["units"],
                                                         "reps": fixed["bootstrap_replicates"],
                                                         "seed": fixed["bootstrap_seed"],
                                                         "scope": "graphs resampled, seeds fixed (diagnostic)"}})
        arrays[dataset["slug"]] = array
    by_key = {c["contrast"]: c for c in contrasts}
    p_values = {key: single[key]["p_value_zero"] for key in single}
    p_values.update({key: c["p_value_zero_new"] for key, c in by_key.items()})
    complete = set(p_values) == set(family_keys)
    holm = holm_family(p_values, alpha) if complete else None
    for contrast in contrasts:
        if holm is None:
            contrast.update(holm_p=None, label_survival_probability=None, holm=None, survival=None)
            continue
        row = holm[contrast["contrast"]]
        survival = label_survival(contrast["effect_nats"], contrast["se_new"], contrast["beneficial_direction"],
                                  contrast["practical_margin_nats"], row["multiplier"], alpha)
        contrast.update(holm_p=row["holm_p"], label_survival_probability=survival["probability"], holm=row,
                        survival=survival)
    family_rows = []
    for key in family_keys:
        if key in single:
            s = single[key]
            family_rows.append({"contrast": key, "source": "joint S14 campaign original D_confirm test",
                                "effect_nats": s["effect_nats"], "interval_low_nats": s["interval_nats"][0],
                                "interval_high_nats": s["interval_nats"][1], "p_value_zero": s["p_value_zero"],
                                "unit_count": s["unit_count"], "tail": s["tail"], "method": s["method_name"],
                                "raw_p_at_J_new": s["p_value_zero"]})
        elif key in by_key:
            c = by_key[key]
            family_rows.append({"contrast": key, "source": "validation-split two-way analysis at J_old",
                                "effect_nats": c["effect_nats"], "interval_low_nats": c["interval_low_nats"],
                                "interval_high_nats": c["interval_high_nats"], "p_value_zero": c["p_value_zero_old"],
                                "unit_count": c["J_old"], "tail": "two_sided", "method": "two-way normal approximation",
                                "raw_p_at_J_new": c["p_value_zero_new"]})
        if holm is not None and family_rows and family_rows[-1]["contrast"] == key:
            family_rows[-1].update(holm_p_at_J_new=holm[key]["holm_p"], rank=holm[key]["rank"])
    analysis = {"schema": "cm-confirmation-sensitivity-1", "experiment": "E10", "alpha": alpha,
                "family_complete": complete, "family": family_rows, "contrasts": contrasts,
                "definitions": {
                    "effect_nats": "mean over the J_old validation graphs and the S seeds of D_js = loss(left) - loss(right)",
                    "interval": "D_bar -/+ z_0.975 V(J_old, S)^(1/2), V = s_u^2/J + s_v^2/S + s_e^2/(J S) (Note S24)",
                    "graph_variance_share": "[s_u^2/J_old + s_e^2/(J_old S)] / V(J_old, S): the part of V that scales "
                                            "with 1/J",
                    "p_value_zero_old": "two-sided normal P of D_bar against zero at J_old",
                    "p_value_zero_new": "two-sided normal P at J_new with V(J_new, S) = V(J_old, S)[(1 - g) + g J_old/J_new]",
                    "holm_p": "Holm over the six original single-graph D_confirm P values and the three J_new P values",
                    "label_survival_probability": "Phi[(b D_bar - max(margin, z_{1-alpha/(2 k)} se_new)) / se_new], "
                                                  "k = the contrast's Holm multiplier at J_new"}}
    return analysis, arrays


def export_confirmation(output, *, campaign=None, manuscript_root=ROOT, root=None, replace=False, allow_partial=False,
                        hardware=None, period=None, command=None) -> dict:
    """Write confirmation_contrasts and confirmation_sensitivity from one analysis of the same raw arrays."""
    output = Path(output).resolve()
    plan = load_plan(output, "confirmation_contrasts")
    if campaign is None and plan.get("campaign"):
        campaign = plan["campaign"]["path"]
    if campaign is None:
        raise RunnerError("the single-graph rows and the Holm family need the joint S14 campaign (--campaign)")
    family = campaign_family(campaign, manuscript_root)
    if plan.get("campaign") and plan["campaign"]["input_identity"] != family["input_identity"]:
        raise RunnerError("the joint S14 campaign changed after planning")
    agreement = {}
    for dataset in plan["datasets"]:
        row = family["rows"][dataset["contrast"]]
        same_pair = (dataset["left"]["hash"], dataset["right"]["hash"]) == (row["left_policy_sha256"],
                                                                          row["right_policy_sha256"])
        try:
            population = sorted(int(v) for v in row["seed_ids"])
        except (TypeError, ValueError):
            population = None
        same_seeds = population == [int(s) for s in dataset["seeds"]]
        agreement[dataset["contrast"]] = {"policy_pair_is_the_S14_test_pair": same_pair,
                                          "seeds_are_the_S14_seed_population": same_seeds}
        if dataset["synthetic"]:
            continue
        if not same_pair:
            raise RunnerError(f"{dataset['contrast']}: the J_old policies are not the joint campaign's test policies")
        if not same_seeds:
            raise RunnerError(f"{dataset['contrast']}: the J_old seeds differ from the S14 test's seed population")
        check_binding(SealedRun(dataset["parent"]), family, dataset["dataset"])
    single = single_graph_tests(family)
    alpha = float(plan.get("alpha") or _holm_alpha())
    analysis, arrays = analyze_confirmation(output, plan, single, alpha=alpha)
    analysis["campaign"] = {"status": family["status"], "input_identity": family["input_identity"],
                            "agreement": agreement}
    staging = Path(tempfile.mkdtemp(prefix="cm-confirmation-export-"))
    try:
        files, environments, results, inputs = experiment_code_files(), [], [], []
        analysis_path = staging / "confirmation_analysis.json"
        analysis_path.write_text(json.dumps(json_safe(analysis), indent=1, sort_keys=True, allow_nan=False) + "\n")
        files["raw/confirmation_analysis.json"] = analysis_path
        for dataset in plan["datasets"]:
            array = arrays[dataset["slug"]]
            path = staging / dataset["slug"] / "paired_per_graph.npz"
            save_arrays(path, unit_id=array["unit_id"], seeds=array["seeds"], left_loss=array["left_loss"],
                        right_loss=array["right_loss"], differences=array["differences"])
            files[f"raw/{dataset['slug']}/paired_per_graph.npz"] = path
            for entry, directory, result in array["results"]:
                base = f"raw/{dataset['slug']}/seed{entry['seed']}"
                files[f"{base}/pair.json"] = directory / "pair.json"
                files[f"{base}/losses.npz"] = directory / "losses.npz"
                files[f"{base}/environment.json"] = directory / "environment.json"
                environments.append(read_json(directory / "environment.json"))
                results.append(result)
                inputs.append({"file": str((directory / "pair.json").relative_to(output)),
                               "sha256": sha256(directory / "pair.json")})
        for key, test in single.items():
            base = f"raw/campaign/{slug(key)}"
            files[f"{base}/test.json"] = test["result_file"]
            files[f"{base}/paired.csv"] = test["paired_file"]
            inputs.append({"file": f"campaign:{test['result_file'].relative_to(family['root']).as_posix()}",
                           "sha256": sha256(test["result_file"])})
        family_index = {row["contrast"]: i for i, row in enumerate(analysis["family"])}
        contrast_index = {c["contrast"]: i for i, c in enumerate(analysis["contrasts"])}
        set2, set3 = [], []
        for key, spec in _family().items():
            unit = UNIT_WORDS[spec[2]]
            if key in single:
                source = f"raw/campaign/{slug(key)}/test.json"
                locators = {"effect_nats": "json:/effect_nats", "interval_low_nats": "json:/interval_nats/0",
                            "interval_high_nats": "json:/interval_nats/1", "p_value_zero": "json:/p_value_zero",
                            "unit_count": "json:/unit_count"}
                values = {"effect_nats": single[key]["effect_nats"], "interval_low_nats": single[key]["interval_nats"][0],
                          "interval_high_nats": single[key]["interval_nats"][1],
                          "p_value_zero": single[key]["p_value_zero"], "unit_count": single[key]["unit_count"]}
            elif key in contrast_index:
                i = family_index[key]
                row = analysis["family"][i]
                source = "raw/confirmation_analysis.json"
                locators = {q: f"json:/family/{i}/{q}" for q in ("effect_nats", "interval_low_nats",
                                                                 "interval_high_nats", "p_value_zero", "unit_count")}
                values = {q: row[q] for q in locators}
            else:
                continue
            units = {"effect_nats": "nats", "interval_low_nats": "nats", "interval_high_nats": "nats",
                     "p_value_zero": "probability", "unit_count": unit}
            for quantity in ("effect_nats", "interval_low_nats", "interval_high_nats", "p_value_zero", "unit_count"):
                set2.append({"contrast": key, "quantity": quantity, "value": values[quantity], "unit": units[quantity],
                             "source_file": source, "locator": locators[quantity]})
        for key, i in contrast_index.items():
            c = analysis["contrasts"][i]
            quantities = {"effect_nats": ("effect_nats", "nats"), "ci_low_nats": ("interval_low_nats", "nats"),
                          "ci_high_nats": ("interval_high_nats", "nats"), "units_old": ("units_old", "graphs"),
                          "graph_variance_share": ("graph_variance_share", "fraction"),
                          "holm_p": ("holm_p", "probability"),
                          "label_survival_probability": ("label_survival_probability", "probability")}
            for quantity, (field, unit) in quantities.items():
                if c[field] is None:
                    continue
                set3.append({"contrast": key, "quantity": quantity, "value": c[field], "unit": unit,
                             "source_file": "raw/confirmation_analysis.json", "locator": f"json:/contrasts/{i}/{field}"})
        synthetic = {d["synthetic"] for d in plan["datasets"]}
        if len(synthetic) != 1:
            raise RunnerError("synthetic and benchmark parents cannot share one record set")
        synthetic = synthetic.pop()
        if not allow_partial:
            # Both sets come from one analysis: refuse before installing either when one would be incomplete.
            for name, rows, keys, count in (("confirmation_contrasts", set2, list(_family()), 5),
                                            ("confirmation_sensitivity", set3, list(multigraph_contrasts().values()), 7)):
                present = {}
                for row in rows:
                    present.setdefault(row["contrast"], set()).add(row["quantity"])
                lacking = [k for k in keys if len(present.get(k, ())) != count]
                if lacking:
                    raise RunnerError(f"{name}: incomplete contrasts {lacking} (all three multigraph benchmarks and the "
                                      "complete nine-test family are required; --allow-partial writes elsewhere)")
        common = {"hardware": hardware or provenance_hardware(environments),
                  "period": period or provenance_period(results), "synthetic": synthetic,
                  "plan_sha256": sha256(output / "plan.json"), "experiment_code_sha256": experiment_code_identity(),
                  "campaign": {"status": family["status"], "input_identity": family["input_identity"]},
                  "analysis_sha256": sha256(analysis_path), "family_complete": analysis["family_complete"]}
        protocol = ("validation-split (D_admit, J_old graphs) paired per-graph, per-seed contrasts of the S14 policy "
                    "pairs, one checkpoint per seed of the S14 seed population; two-way seed-by-graph model of Note S24 "
                    "by the method of moments; J_old interval D_bar -/+ z V^(1/2); J_new = FAMILY D_confirm sizes with "
                    "fixed means; Holm re-formed over the six original single-graph D_confirm P values and the three "
                    "J_new P values")
        generated = {"code/plan.json": public_plan(output)}
        coverage2 = {"tables": ["S14b"], "datasets": [], "coverage_keys": ["contrast"],
                     "coverage_rows": [[k] for k in _family()], "analyzer": "confirmation_contrasts"}
        coverage3 = {"tables": ["S18b", "Figure S9"], "datasets": [], "coverage_keys": ["contrast"],
                     "coverage_rows": [[k] for k in multigraph_contrasts().values()], "analyzer": "confirmation_sensitivity"}
        written = {}
        written["confirmation_contrasts"] = install_set(
            "confirmation_contrasts", set2, files,
            {**common, "experiment": "E10 nine-test confirmation family at the original sizes (Supplementary Table S14b)",
             "produced_by": "scripts/run_confirmation_analysis.py confirmation plan/run/export "
                            "(cm_experiments/confirmation_analysis.py); single-graph rows: joint S14 campaign",
             "protocol": protocol + "; single-graph rows are the original D_confirm tests of the joint S14 campaign"},
            coverage=coverage2, root=root, replace=replace, allow_partial=allow_partial, command=command,
            inputs=inputs, generated=generated)
        written["confirmation_sensitivity"] = install_set(
            "confirmation_sensitivity", set3, files,
            {**common, "experiment": "E10 confirmation-size sensitivity of the multigraph mechanism contrasts "
                                     "(Supplementary Figures S9, S18b)",
             "produced_by": "scripts/run_confirmation_analysis.py confirmation plan/run/export "
                            "(cm_experiments/confirmation_analysis.py)",
             "protocol": protocol},
            coverage=coverage3, root=root, replace=replace, allow_partial=allow_partial, command=command,
            inputs=inputs, generated=generated)
        return written
    finally:
        shutil.rmtree(staging, ignore_errors=True)
