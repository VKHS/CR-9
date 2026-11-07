"""E9: search versus model error by exact enumeration at planning horizon H <= 4.

Main Proposition 8 bounds the true-objective regret of a planner by
2 e_H + zeta_H, where e_H bounds |J_hat(a) - J(a)| over the supported sequence
family S and zeta_H is the planner's suboptimality in its own objective J_hat
(Note S20). For each planning instance (an input, a frozen ensemble member and
a planning depth) this runner

* reaches the planning state with the released runtime: the initial measurement
  for MPC-OL; for MPC-RH the parent's registered MPC-RH policy is executed by
  ``cross_morse.policies.execute_policy`` and its recorded actions and last-safe
  anchors are replayed with the same sensor and observer calls, so the observer
  state equals the one the policy planned from (the replayed anchors are checked);
* enumerates the complete supported family S: every sequence the registered
  beam (``cross_morse.policies.beam_plan``) could expand if its top-k and beam
  width were unbounded -- at each model-predicted state the supported finite-score
  actions, or the registered recovery action -- and evaluates J_hat(a) with
  exactly the beam's arithmetic (Eq. 17: stage cost, critic sigma, transport and
  hop penalties, discounted terminal value of the nominal suffix);
* evaluates J(a) on the frozen plant: the same objective with the stage costs and
  the terminal value replaced by the plant's measured loss changes on the declared
  label role (the targets the consequence models were fitted to estimate), the
  actual action consequence including the feedback anchor rule for A9, and the
  nominal completion to the controlled depth;
* runs the registered beam itself and reads its sequence's J_hat from the table.

Per instance e_H = max_S |J_hat - J|, zeta_H = J_hat(beam) - min_S J_hat and the
realized regret J(beam) - min_S J. The record reports e_H and zeta_H as the stated
confidence quantile over instances. The critic-sigma term is a model-defined
penalty: by default it enters J and J_hat identically, so e_H measures the
stage-cost and terminal-value errors; J with zero uncertainty is recorded too.
"""
from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import numpy as np

from .artifacts import write_json
from .predictability import (ExperimentError, ParentRun, RunDirectory, create_plan, export_record_set,
                             open_plan, quantile, read_csv, read_json, role_labels, role_units, summary_row,
                             write_csv)

MAX_HORIZON = 4
PLANNERS = {"mpc_rh": False, "mpc_ol": True}  # planner -> open_loop flag of beam_plan
RAW_EXPORTS = ("raw/instances.csv",)
DEFINITIONS = {
    "two_e_H_nats": "2 e_H: twice the stated-confidence quantile over planning instances of max over the enumerated "
                    "supported family of |J_hat(a) - J(a)| (critic sigma retained in both objectives; nats)",
    "zeta_H_nats": "stated-confidence quantile over planning instances of J_hat(beam sequence) - min over the "
                   "enumerated family of J_hat (nats)",
    "realized_regret_nats": "stated-confidence quantile over planning instances of J(beam sequence) - min over the "
                            "enumerated family of J (nats)",
    "bound_coverage": "fraction of planning instances whose realized regret is at most 2 e_H + zeta_H (pooled values)",
    "beam_optimal_fraction": "fraction of planning instances on which the beam returns a minimizer of J_hat over the family",
    "two_e_H_zero_uncertainty_nats": "2 e_H with the critic sigma term set to zero in the true objective J (nats)",
    "mean_family_size": "mean number of enumerated supported sequences per planning instance",
    "planning_instances": "planning instances with a complete enumeration and a returned beam sequence",
}


class FamilyTooLarge(RuntimeError):
    pass


def plan_search_enumeration(parent, output, *, horizons=(1, 2, 3, 4), planner="mpc_rh", profile="full",
                            role=None, depths=None, members=None, max_graphs=16, confidence=.95,
                            max_sequences=20000, allow_fixture=False):
    parent = parent if isinstance(parent, ParentRun) else ParentRun(parent)
    if parent.synthetic and not allow_fixture:
        raise ExperimentError("a synthetic fixture parent requires --allow-fixture")
    if planner not in PLANNERS:
        raise ExperimentError(f"planner must be one of {sorted(PLANNERS)}")
    horizons = sorted({int(h) for h in horizons})
    if not horizons or horizons[0] < 1 or horizons[-1] > MAX_HORIZON:
        raise ExperimentError("exact enumeration is registered for planning horizons 1 <= H <= 4")
    independent = parent.dataset in ("PascalVOC-SP", "COCO-SP", "Peptides-func")
    role = role or ("select" if independent else "explore")
    checked = parent.check("fit")
    parent.require_role(role)
    if profile not in parent.config["sensor"]["profiles"]:
        raise ExperimentError(f"profile {profile!r} is not a fitted sensor profile of the parent")
    L = parent.depth
    if depths is None:
        depths = [0] if planner == "mpc_ol" else sorted({0, L // 4, L // 2, (3 * L) // 4})
    depths = sorted({int(d) for d in depths})
    if planner == "mpc_ol" and depths != [0]:
        raise ExperimentError("MPC-OL plans once, at depth zero")
    depths = [d for d in depths if 0 <= d <= L - horizons[-1]]
    if not depths:
        raise ExperimentError("no planning depth leaves room for the largest horizon")
    seeds = parent.seeds
    members = [int(s) for s in (members if members is not None else seeds[:1])]
    if not members or not set(members) <= set(seeds) or len(set(members)) != len(members):
        raise ExperimentError("members must be distinct training seeds of the parent")
    if not 0 < confidence < 1 or type(max_sequences) is not int or max_sequences < 1:
        raise ExperimentError("invalid confidence or sequence ceiling")
    if type(max_graphs) is not int or max_graphs < 0:
        raise ExperimentError("max_graphs must be a nonnegative integer")
    prefix = None
    if planner == "mpc_rh" and any(d > 0 for d in depths):
        registry = parent.json("fit/registry.json")["candidates"]
        options = sorted((c for c in registry if c["route"] == "mpc_rh" and c["profile"] == profile
                          and c["ensemble"] == 1), key=lambda c: (c["horizon"], c["hash"]))
        if not options:
            raise ExperimentError(f"the parent registers no MPC-RH policy at profile {profile}")
        prefix = options[0]
    body = {"planner": planner, "open_loop": PLANNERS[planner], "horizons": horizons, "profile": profile,
            "label_role": role, "depths": depths, "members": members, "max_graphs": max_graphs,
            "confidence": float(confidence), "max_sequences": max_sequences, "prefix_policy": prefix,
            "parent_check": checked, "definitions": DEFINITIONS,
            "objective": "Eq. (17) as implemented by cross_morse.policies.beam_plan (kappa = 1)",
            "uncertainty_convention": "critic sigma retained identically in J_hat and J; zero-sigma J recorded",
            "family": "all supported finite-score actions (or the registered recovery action) at every "
                      "model-predicted state; no top-k or beam-width truncation",
            "true_objective": "measured stage loss changes and nominal-suffix terminal loss change on the label "
                              "role's rows of the input; A9 uses the feedback anchor rule"}
    return create_plan("search_enumeration", parent, output, body)


def _reach(model, x, graph, config, controller, profile, depth, prefix, open_loop):
    """Planning state at ``depth``, replaying a recorded feedback trace with the runtime's calls."""
    from cross_morse.core import site_transport_accounting
    from cross_morse.exploration import sensor_bank
    from cross_morse.recovery import label_free_safe
    L = int(config["model"]["depth"])
    model.bind_graph(graph)
    h0 = model.encode(x)
    logits0 = model.logits(h0)
    bank = sensor_bank(graph, model, config, profile)
    bank.prepare(h0, logits0)
    y = bank.measure(h0, logits0, 0, L).cpu().numpy()
    state = controller.start(y)
    h, last_safe, last_depth, mass, hops, previous, anchors = h0, h0, 0, 0., 0., -1, []
    for d in range(depth):
        if d > 0:
            y = bank.measure(h, model.logits(h), d, L, mass, hops).cpu().numpy()
            state = controller.observe(state, y, d, previous)
        if not open_loop and label_free_safe(y):
            last_safe, last_depth = h.clone(), d
        anchors.append(last_depth)
        action = int(prefix["actions"][d])
        accounting = site_transport_accounting(action, d, model)
        h = model.step(h, h0, graph, action, last_safe, site=d)
        mass, hops, previous = mass + accounting["mass"], hops + accounting["hops"], action
    if anchors != [int(v) for v in (prefix or {}).get("safe_depths", [])[:depth]]:
        raise ExperimentError("the replayed last-safe anchors differ from the executed policy trace")
    if depth > 0:
        y = bank.measure(h, model.logits(h), depth, L, mass, hops).cpu().numpy()
        state = controller.observe(state, y, depth, previous)
    if not open_loop and label_free_safe(y):
        last_safe = h.clone()
    return {"h0": h0, "h": h, "last_safe": last_safe, "state": state, "bank": bank, "mass": mass, "hops": hops}


def enumerate_family(model, graph, config, controller, start, horizon, *, open_loop, loss, max_sequences):
    """All sequences of the supported family with J_hat (beam arithmetic) and the plant's J."""
    from cross_morse.core import ACTIONS, site_transport_accounting
    from cross_morse.policies import recovery_action
    from cross_morse.recovery import label_free_safe
    c = config["controller"]
    L = int(config["model"]["depth"])
    h0 = start["h0"]
    rows = []

    def finite_cost(value, component):
        if not np.isfinite(value):
            raise FloatingPointError(f"Nonfinite {component} in enumerated planning")
        return float(value)

    def visit(ms, h, last_safe, bank, mass, hops, cost_hat, cost_true, cost_zero, sequence, current):
        offset = len(sequence)
        if offset == horizon:
            tail = finite_cost(controller.terminal_value(ms), "terminal value")
            j_hat = cost_hat + c["gamma"]**offset * c["lambda_terminal"] * tail
            depth = ms.depth
            if depth < L:
                continuation = h
                for site in range(depth, L):
                    continuation = model.native_step(continuation, h0, graph, site=site)
                value = loss(continuation) - current
            else:
                value = 0.
            j_true = cost_true + c["gamma"]**offset * c["lambda_terminal"] * value
            j_zero = cost_zero + c["gamma"]**offset * c["lambda_terminal"] * value
            rows.append((tuple(sequence), j_hat, j_true, j_zero, tail, value))
            if len(rows) > max_sequences:
                raise FamilyTooLarge(f"more than {max_sequences} supported sequences")
            return
        supported = controller.supported(ms.y, ms.depth)
        if open_loop:
            supported = [a for a in supported if a != 9]
        branch = None
        if supported:
            scores = controller.score(ms, supported, kind="radius", kappa=1., lambda_rho=c["lambda_rho"])
            finite_actions = [a for a in supported if np.isfinite(scores[a]["score"])]
            if finite_actions:
                branch = [(a, scores[a]) for a in sorted(finite_actions, key=lambda a: (scores[a]["score"], a))]
        if branch is None:
            branch = [(recovery_action(supported, last_safe_available=not open_loop), None)]
        site = ms.depth
        for action, score in branch:
            if score is None:  # beam_plan's recovery branch: stage cost only
                value = finite_cost(controller.stage_cost(ms, action), "recovery stage cost")
                penalty, sigma = 0., 0.
            else:
                stage = controller.stage_cost(ms, action)
                transport = site_transport_accounting(action, ms.depth, config)
                stage += score["sigma"] + c["lambda_rho"] * ACTIONS[action].rho + c["lambda_transport"] * transport["hops"]
                value = finite_cost(stage, "stage cost")
                sigma = float(score["sigma"])
                penalty = c["lambda_rho"] * ACTIONS[action].rho + c["lambda_transport"] * transport["hops"]
            following = controller.predict(ms, action)
            h_next = model.step(h, h0, graph, action, last_safe, site=site)
            observed = loss(h_next)
            change = observed - current
            accounting = site_transport_accounting(action, site, model)
            next_mass, next_hops = mass + accounting["mass"], hops + accounting["hops"]
            next_safe, next_bank = last_safe, bank
            if not open_loop and offset + 1 < horizon:
                # The anchor of a later A9 follows the label-free rule on the actual next measurement.
                next_bank = copy.copy(bank)
                y = next_bank.measure(h_next, model.logits(h_next), site + 1, L, next_mass, next_hops).cpu().numpy()
                if label_free_safe(y):
                    next_safe = h_next.clone()
            weight = c["gamma"]**offset
            visit(following, h_next, next_safe, next_bank, next_mass, next_hops,
                  cost_hat + weight * value,
                  cost_true + weight * (change + sigma + penalty),
                  cost_zero + weight * (change + penalty),
                  sequence + [action], observed)

    visit(copy.deepcopy(start["state"]), start["h"], start["last_safe"], start["bank"], start["mass"],
          start["hops"], 0., 0., 0., [], loss(start["h"]))
    return rows


def instance_values(rows, beam):
    """Per-instance e_H, zeta_H and realized regret from enumerated objective rows."""
    table = {tuple(int(a) for a in s): (h, t, z) for s, h, t, z in rows}
    if len(table) != len(rows):
        raise ExperimentError("the enumerated family repeats a sequence")
    j_hat = np.array([v[0] for v in table.values()])
    j_true = np.array([v[1] for v in table.values()])
    j_zero = np.array([v[2] for v in table.values()])
    result = {"family_size": len(table), "e_H": float(np.max(np.abs(j_hat - j_true))),
              "e_H_zero_uncertainty": float(np.max(np.abs(j_hat - j_zero))),
              "j_hat_min": float(j_hat.min()), "j_true_min": float(j_true.min())}
    if beam is None:
        return dict(result, zeta=None, regret=None, regret_zero_uncertainty=None, beam_optimal=None)
    beam = tuple(int(a) for a in beam)
    if beam not in table:
        raise ExperimentError("the beam sequence lies outside the enumerated supported family")
    h, t, z = table[beam]
    return dict(result, zeta=float(h - j_hat.min()), regret=float(t - j_true.min()),
                regret_zero_uncertainty=float(z - j_zero.min()), beam_optimal=int(h <= j_hat.min()))


def run_search_enumeration(plan_path, *, device="cpu", trust=False):
    import torch
    from cross_morse.common import precision_context
    from cross_morse.data import load_data
    from cross_morse.learning import Controller
    from cross_morse.policies import PlanningBudgetExceeded, PolicyExecutionError, beam_plan, execute_policy
    from cross_morse.provenance import sha256_state_dict
    from cross_morse.runtime import graph_operator
    from cross_morse.tasks import probabilities, sample_losses
    from cross_morse.training import load_plant

    plan, parent = open_plan(plan_path, "search_enumeration")
    parent.verify("fit")
    run = RunDirectory(plan_path, plan, parent, device=device, trust=trust)
    root, config = parent.root, parent.config
    open_loop, role, L = plan["open_loop"], plan["label_role"], parent.depth
    data = load_data(root).protect_labels()
    controller = Controller.load(root / f"fit/{plan['profile']}/controller.pt", "cpu")
    if not controller.sequence_models_available:
        raise ExperimentError("the fitted controller has no transition/stage/terminal models to plan with")
    instances, arrays = [], {h: {"instance": [], "sequence": [], "j_hat": [], "j_true": [],
                                 "j_true_zero_uncertainty": []} for h in plan["horizons"]}
    for seed in plan["members"]:
        model = load_plant(data, config, root / "train", seed, device)
        before = sha256_state_dict(model)
        for eid, episode in role_units(data, role, plan["max_graphs"]):
            ids = np.asarray(episode.splits[role], dtype=np.int64)
            if not len(ids):
                continue
            labels = role_labels(data, episode, role, ids)
            graph = graph_operator(episode, device)
            x = episode.x.to(device)
            prefix = None
            if plan["prefix_policy"] is not None:
                try:
                    _, trace = execute_policy(model, x, graph, config, plan["prefix_policy"], controller, None)
                except PolicyExecutionError as error:
                    instances.extend({"input_id": eid, "member_seed": seed, "depth": depth, "horizon": horizon,
                                      "planner": plan["planner"], "status": "PREFIX_POLICY_FAILED",
                                      "reason": error.reason}
                                     for depth in plan["depths"] for horizon in plan["horizons"])
                    continue
                prefix = {"actions": trace["actions"], "safe_depths": trace["safe_depths"]}
            task = model.task

            def loss(h):
                p = probabilities(model.logits(h).float(), task).cpu().numpy()[ids]
                values = sample_losses(p, labels, task)
                return float(np.mean(values[np.isfinite(values)]))

            for depth in plan["depths"]:
                with torch.no_grad(), precision_context(config, torch.device(device)):
                    start = _reach(model, x, graph, config, controller, plan["profile"], depth, prefix, open_loop)
                    for horizon in plan["horizons"]:
                        began = time.perf_counter()
                        record = {"input_id": eid, "member_seed": seed, "depth": depth, "horizon": horizon,
                                  "planner": plan["planner"]}
                        try:
                            rows = enumerate_family(model, graph, config, controller, start, horizon,
                                                    open_loop=open_loop, loss=loss,
                                                    max_sequences=plan["max_sequences"])
                        except FamilyTooLarge as exc:
                            instances.append(dict(record, status="FAMILY_TOO_LARGE", reason=str(exc)))
                            continue
                        except FloatingPointError as exc:
                            instances.append(dict(record, status="NONFINITE_OBJECTIVE", reason=str(exc)))
                            continue
                        budget = {"calls": 0}
                        try:
                            beam = beam_plan(controller, start["state"], horizon, config, budget, open_loop=open_loop)
                            status = "COMPLETE"
                        except PlanningBudgetExceeded:
                            beam, status = None, "BEAM_BUDGET_EXHAUSTED"
                        values = instance_values([(s, h, t, z) for s, h, t, z, _, _ in rows], beam)
                        index = len(instances)
                        for s, h, t, z, _, _ in rows:
                            arrays[horizon]["instance"].append(index)
                            arrays[horizon]["sequence"].append(list(s))
                            arrays[horizon]["j_hat"].append(h)
                            arrays[horizon]["j_true"].append(t)
                            arrays[horizon]["j_true_zero_uncertainty"].append(z)
                        instances.append(dict(record, status=status, beam_calls=budget["calls"],
                                              beam_sequence=" ".join(map(str, beam or [])),
                                              seconds=time.perf_counter() - began, **values))
        if sha256_state_dict(model) != before:
            raise RuntimeError("a frozen predictor changed during enumeration")
    if not instances:
        raise ExperimentError("no planning instance was executed")
    parent.verify("fit")
    for horizon, values in arrays.items():
        if values["instance"]:
            np.savez_compressed(run.raw(f"sequences_H{horizon}.npz"),
                                **{k: np.asarray(v, dtype=np.int64 if k in ("instance", "sequence") else np.float64)
                                   for k, v in values.items()})
    fields = ["input_id", "member_seed", "depth", "horizon", "planner", "status", "reason", "family_size",
              "beam_sequence", "beam_calls", "e_H", "e_H_zero_uncertainty", "zeta", "regret",
              "regret_zero_uncertainty", "beam_optimal", "j_hat_min", "j_true_min", "seconds"]
    write_csv(run.raw("instances.csv"), [{k: _text(r.get(k)) for k in fields} for r in instances], fields)
    write_json(run.raw("enumeration.json"), {"prefix_policy": plan["prefix_policy"], "label_role": role,
                                             "labels": "read inside the role's lifecycle label scope; planning is label-free"})
    summary = summarize_search_enumeration([run.root])
    summary["truncated"] = bool(plan["max_graphs"])
    return run.finish(summary)


def _text(value):
    if value is None:
        return ""
    if isinstance(value, float):
        # float() first: repr of a NumPy scalar is 'np.float64(...)' under NumPy 2
        return repr(float(value))
    return value


def _number(text):
    return None if text in ("", None) else float(text)


def summarize_search_enumeration(dirs):
    """Recompute E9 values from instances.csv, checked against the sealed objective arrays."""
    pooled, notes = {}, {}
    for d in dirs:
        d = Path(d)
        exp, plan = read_json(d / "experiment.json"), read_json(d / "plan.json")
        instances = read_csv(d / "raw/instances.csv")
        _check_instances(d, instances)
        for row in instances:
            key = (exp["parent"]["dataset"], row["planner"], row["horizon"])
            pooled.setdefault(key, {"rows": [], "confidence": plan["confidence"]})["rows"].append(row)
    rows = []
    for (dataset, planner, horizon), entry in sorted(pooled.items()):
        keys = {"dataset": dataset, "planner": planner, "horizon": horizon}
        level = entry["confidence"]
        complete = [r for r in entry["rows"] if r["status"] == "COMPLETE"]
        enumerated = [r for r in entry["rows"] if r["status"] in ("COMPLETE", "BEAM_BUDGET_EXHAUSTED")]
        notes["/".join(map(str, keys.values()))] = {
            "instances": len(entry["rows"]), "complete": len(complete), "confidence": level,
            "family_too_large": sum(r["status"] == "FAMILY_TOO_LARGE" for r in entry["rows"]),
            "beam_budget_exhausted": sum(r["status"] == "BEAM_BUDGET_EXHAUSTED" for r in entry["rows"])}
        if not complete:
            continue
        e_h = quantile([_number(r["e_H"]) for r in enumerated], level)
        e_zero = quantile([_number(r["e_H_zero_uncertainty"]) for r in enumerated], level)
        zeta = quantile([_number(r["zeta"]) for r in complete], level)
        regret = [_number(r["regret"]) for r in complete]
        rows += [summary_row(keys, "two_e_H_nats", 2 * e_h, "nats", DEFINITIONS["two_e_H_nats"]),
                 summary_row(keys, "zeta_H_nats", zeta, "nats", DEFINITIONS["zeta_H_nats"]),
                 summary_row(keys, "realized_regret_nats", quantile(regret, level), "nats",
                             DEFINITIONS["realized_regret_nats"]),
                 summary_row(keys, "bound_coverage", float(np.mean([r <= 2 * e_h + zeta + 1e-12 for r in regret])),
                             "fraction", DEFINITIONS["bound_coverage"]),
                 summary_row(keys, "beam_optimal_fraction",
                             float(np.mean([int(r["beam_optimal"]) for r in complete])), "fraction",
                             DEFINITIONS["beam_optimal_fraction"]),
                 summary_row(keys, "two_e_H_zero_uncertainty_nats", 2 * e_zero, "nats",
                             DEFINITIONS["two_e_H_zero_uncertainty_nats"]),
                 summary_row(keys, "mean_family_size", float(np.mean([int(r["family_size"]) for r in enumerated])),
                             "sequences", DEFINITIONS["mean_family_size"]),
                 summary_row(keys, "planning_instances", len(complete), "instances", DEFINITIONS["planning_instances"])]
    return {"rows": rows, "notes": notes}


def _check_instances(d, instances):
    """Each per-instance value must follow from the sealed objective arrays, when present."""
    for horizon in sorted({int(r["horizon"]) for r in instances}):
        path = d / f"raw/sequences_H{horizon}.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as z:
            index, sequences = z["instance"], z["sequence"]
            j_hat, j_true, j_zero = z["j_hat"], z["j_true"], z["j_true_zero_uncertainty"]
        for number, row in enumerate(instances):
            if int(row["horizon"]) != horizon or row["status"] not in ("COMPLETE", "BEAM_BUDGET_EXHAUSTED"):
                continue
            mask = index == number
            beam = [int(a) for a in row["beam_sequence"].split()] if row["status"] == "COMPLETE" else None
            values = instance_values([(tuple(s), h, t, z) for s, h, t, z in
                                      zip(sequences[mask], j_hat[mask], j_true[mask], j_zero[mask])], beam)
            for name in ("e_H", "e_H_zero_uncertainty", "zeta", "regret"):
                if values[name] is None and row[name] == "":
                    continue
                if not math.isclose(values[name], float(row[name]), rel_tol=1e-12, abs_tol=1e-15):
                    raise ExperimentError(f"{d.name}: instance {number} {name} differs from its objective arrays")


def export_records(results_dirs, **kwargs):
    protocol = ("E9 (Note S20): exact enumeration of the supported sequence family at H <= 4 on states reached by "
                "the released runtime; J_hat with cross_morse.policies.beam_plan arithmetic, J from the frozen "
                "plant's measured stage and terminal loss changes; e_H and zeta_H as stated-confidence quantiles")
    return export_record_set("search_enumeration", results_dirs, summarize=summarize_search_enumeration,
                             raw_exports=RAW_EXPORTS, protocol=protocol, **kwargs)
