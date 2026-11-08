"""Selection, support and audit displays computed from sealed lifecycle runs.

Every run-derived quantity is read from, or replayed with the released runtime
code from, the stage outputs of the seven lifecycle runs: the entry records
(``train/entry_initial.json``, ``select/entry_complete.json``), the support audit
of the fitted controller (``fit/full/audit.json``), the selection record
(``select/decision.json``, ``select/unit_scores.npz``, ``select/candidates.json``),
the confirmation and admission records, the final paired test units and the
stage timing markers. The joint S14 family comes from the validated
confirmation campaign.

Quantities that no run holds are read from measurement-record sets whose
contracts are registered in records/specs.json (action_gap_audit, tube_audit,
oracle_dispersion, depth_diagnostics); a display that needs such a set waits for
it rather than estimating its cells.

Roman-empire is a campaign of ten split runs. Unless a display states otherwise,
effects and statistics are averaged over the splits (as main Table 1 does), gate
statistics take their least favourable split, counts that are fixed by the
configuration are shown once, and every per-split value is kept in the source data.
"""
from __future__ import annotations

import functools
import json
import math
from pathlib import Path

import numpy as np

from .framework import (DATASETS, GRAPH_LEVEL, SHORT, DisplayInputError, DisplayValidationError,
                        display, integer, interval, not_applicable, num, pyplot)

GROUP = {
    "T1": "main_table_1_policy_and_deployment_ledger",
    "T2": "main_table_2_gain_and_cost_ledger",
    "S4": "supplementary_table_S4_target_qualification",
    "S7": "supplementary_table_S7_support_audit",
    "S12": "supplementary_table_S12_selection_ledger",
    "S12b": "supplementary_table_S12b_nested_open_loop_comparison",
    "S12c": "supplementary_table_S12c_selection_rule_audit",
    "S13": "supplementary_table_S13_independent_graph_admission",
    "S14": "supplementary_table_S14_mechanism_contrasts_submitted_protocol",
    "S14b": "supplementary_table_S14b_separated_data_sensitivity",
    "S14b_unchanged": "supplementary_table_S14b_unchanged_contrasts",
    "S16": "supplementary_table_S16_arxiv_factorial",
    "S20": "supplementary_table_S20_selection_theorem_instantiation",
    "S21": "supplementary_table_S21_action_gap_audit",
    "S22": "supplementary_table_S22_trajectory_tube_audit",
    "S30": "supplementary_table_S30_oracle_dispersion",
    "S41": "supplementary_table_S41_qualification_cost_components",
}
S20_GROUP = GROUP["S20"]
FINALISTS = 12
ROUTE_LABELS = {"nominal": "nominal", "best_depth": "best-depth", "learned_stop": "learned-stop", "phase": "phase",
                "fixed_sequence": "fixed-sequence", "mpc_ol": "MPC-OL", "one_step": "one-step", "radius": "radius",
                "mpc_rh": "MPC-RH"}
# Supplementary Table S12 prints q_C and then every route family except the nominal one, in registry order.
S12_FAMILIES = ("best_depth", "learned_stop", "phase", "fixed_sequence", "mpc_ol", "one_step", "radius", "mpc_rh")
OPEN_LOOP_FAMILIES = S12_FAMILIES[:5]
FEEDBACK_FAMILIES = S12_FAMILIES[5:]
PROFILE_LABELS = {"lean": "Lean", "full": "Full", "none": "none"}
LOSS_LABELS = {"ce": "CE", "bce": "BCE"}
METRIC_UNITS = {"accuracy": "pp", "rocauc": "AUC pp", "macro_f1": "F1 pp", "ap": "AP"}
QUALIFICATION_STAGES = (("explore", "exploration logging", "exploration_logging"),
                        ("fit", "observer, critic and sequence-model fitting", "fitting"),
                        ("select", "nested comparator and candidate selection", "nested_selection"),
                        ("confirm", "confirmation evaluation", "confirmation"),
                        ("admit", "admission evaluation", "admission"))
PASS = {True: "pass", False: "fail", None: None}


# ---------------------------------------------------------------------------
# Run records
# ---------------------------------------------------------------------------

class RunRecord:
    """The sealed decision files of one lifecycle run that the audit displays read."""

    def __init__(self, view):
        self.view = view

    @functools.cached_property
    def config(self) -> dict:
        return self.view.json("config.json")

    @functools.cached_property
    def candidates(self) -> list:
        return self.view.json("select/candidates.json")

    @functools.cached_property
    def decision(self) -> dict:
        return self.view.json("select/decision.json")

    @functools.cached_property
    def confirmation(self) -> dict:
        return self.view.json("confirm/decision.json")

    @functools.cached_property
    def admission(self) -> dict:
        return self.view.json("admit/decision.json")

    @functools.cached_property
    def summary(self) -> dict:
        return self.view.json("final/results_summary.json")

    @property
    def qc(self) -> int:
        return self.decision["comparator_index"]

    @property
    def provisional(self) -> int:
        return self.decision["provisional_index"]

    @property
    def confirmed(self) -> int:
        return self.confirmation["confirmed_index"]

    @property
    def deployed(self) -> int:
        return self.admission["deployed_index"]

    def label(self, j: int) -> str:
        return ROUTE_LABELS.get(self.candidates[j]["route"], self.candidates[j]["route"])

    def rule(self, j: int) -> str:
        return "qC" if j == self.qc else self.label(j)


def records(view) -> list[RunRecord]:
    return [RunRecord(member) for member in view.members()]


def independent(dataset: str) -> bool:
    return dataset in GRAPH_LEVEL


def task_labels(dataset: str) -> tuple[str, str]:
    from cross_morse.tasks import task_spec
    spec = task_spec({"dataset": dataset})
    return LOSS_LABELS[spec.loss_name], spec.metric


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def least_favourable(values, pick):
    """Gate statistic of a campaign: its least favourable split; missing when any split lacks it."""
    values = list(values)
    return None if any(v is None for v in values) else pick(values)


def common(values, fmt=str):
    """The value shared by every split, otherwise the range of the splits as text."""
    values = [v for v in values if v is not None]
    if not values:
        return None
    low, high = min(values), max(values)
    return low if low == high else f"{fmt(low)}–{fmt(high)}"


def wrap(text, width: int) -> str:
    import textwrap
    return textwrap.fill(str(text), width)


def tally(texts):
    """A text shared by every split, or each distinct text with its split count."""
    texts = [t for t in texts if t is not None]
    if not texts:
        return None
    distinct = list(dict.fromkeys(texts))
    if len(distinct) == 1:
        return distinct[0]
    return "; ".join(f"{t} ({texts.count(t)}/{len(texts)})" for t in distinct)


# ---------------------------------------------------------------------------
# Selection replay (Algorithm S4)
# ---------------------------------------------------------------------------

def selection_inputs(view):
    """The exact arrays and settings the select stage passed to ``nested_selection``."""
    data = view.npz("select/unit_scores.npz")
    candidates = view.json("select/candidates.json")
    decision = view.json("select/decision.json")
    config = view.json("config.json")
    metric_evaluator = None
    if "accuracy" not in data:
        from cross_morse.evaluation import MetricEvidence
        raw = view.npz("select/metric_observations.npz")
        example = next(raw[f"candidate_{j}"] for j, c in enumerate(candidates) if not c.get("execution_failure"))
        values = [np.full_like(example, np.nan) if c.get("execution_failure") else raw[f"candidate_{j}"]
                  for j, c in enumerate(candidates)]
        metric_evaluator = MetricEvidence(values, raw["labels"], raw["sample_units"],
                                          data["core_ids"], data["counts"], config)
    return data, candidates, decision, config, metric_evaluator


def screened_radius(view, size: int = FINALISTS) -> dict:
    """Simultaneous radius when the max-t maximum runs over ``size`` finalists only.

    Finalists: the ``size`` resource/support-feasible candidates with the smallest
    simultaneous upper bounds ``U`` of the registered full-family selection (ties by
    registry index). The bootstrap is the released Algorithm S4 replayed with the
    stage's own arrays, folds, margins, weights and seed; only the set of columns
    entering the studentized maximum changes.
    """
    from cross_morse import statistics
    data, candidates, decision, config, metric_evaluator = selection_inputs(view)
    upper = decision["U"]
    feasible = [j for j, c in enumerate(candidates)
                if c.get("feasible", True) and not c.get("execution_failure") and upper[j] is not None]
    finalists = sorted(feasible, key=lambda j: (upper[j], j))[:size]
    observed = [j for j, c in enumerate(candidates) if not c.get("execution_failure")]
    columns = np.zeros(len(observed), dtype=bool)
    columns[[observed.index(j) for j in finalists]] = True
    original, captured = statistics._max_t_interval, {}

    def restricted(point, draws, alpha):
        if "radius" in captured:                      # metric interval: unchanged
            return original(point, draws, alpha)
        se = np.std(draws, axis=0, ddof=1)
        use = columns & (se > 1e-12)
        maxima = (np.max(np.abs(draws[:, use] - point[use]) / se[use], axis=1)
                  if use.any() else np.zeros(len(draws)))
        critical = float(np.quantile(maxima, 1 - alpha))
        captured["radius"] = float(np.max(se[columns] * critical)) if columns.any() else 0.0
        captured["critical"] = critical
        return original(point, draws, alpha)

    sc = config["selection"]
    statistics._max_t_interval = restricted
    try:
        replay = statistics.nested_selection(
            data["loss"], data.get("accuracy"), candidates, data["outer_fold"], inner_fold_ids=data["inner_fold"],
            reps=int(decision["bootstrap_replicates"]), alpha=sc["alpha"], epsilon_r=sc["epsilon_r"],
            epsilon_eq=sc["epsilon_eq"], epsilon_metric=sc["epsilon_metric"], weights=data["counts"],
            seed=int(decision["bootstrap_seed"]), metric_evaluator=metric_evaluator)
    finally:
        statistics._max_t_interval = original
    # The loop must be the sealed one: the full-family radius has to come back unchanged.
    if not math.isclose(replay["max_radius"], decision["max_radius"], rel_tol=1e-9, abs_tol=1e-12):
        raise DisplayInputError(f"{view.dataset}: Algorithm S4 replay does not reproduce select/decision.json")
    return {"finalists": finalists, "radius": captured["radius"], "critical": captured["critical"]}


def representatives(rec: RunRecord) -> dict:
    """Representative configuration of each S12 route family.

    The selected, confirmed or deployed configuration represents its own family;
    every other family is represented by its configuration with the smallest
    simultaneous upper bound U_s (ties by registry index). None marks a family
    without an observed configuration.
    """
    upper = rec.decision["U"]
    marked = [j for j in (rec.provisional, rec.confirmed, rec.deployed) if j != rec.qc]
    out = {}
    for route in S12_FAMILIES:
        members = [j for j, c in enumerate(rec.candidates) if c["route"] == route and upper[j] is not None]
        chosen = [j for j in marked if j in members]
        out[route] = chosen[0] if chosen else (min(members, key=lambda j: (upper[j], j)) if members else None)
    return out


def support_feasible(rec: RunRecord, j: int):
    """Support gate of a candidate: no support violation in its select-stage execution."""
    candidate = rec.candidates[j]
    if candidate.get("execution_failure"):
        return False
    for relative in (f"predictions/select/{candidate['candidate_id']}.json",
                     f"predictions/{candidate['candidate_id']}.json"):
        meta = rec.view.json(relative, default=None)
        if meta is not None:
            return bool(meta.get("feasible")) and not meta.get("support_violations", 0)
    return None


def gates(rec: RunRecord, j: int) -> tuple:
    """(support, metric, resource) gates of main Eq. (19) for one configuration."""
    candidate = rec.candidates[j]
    metric = rec.decision["metric_feasible"][j] if not candidate.get("execution_failure") else None
    resource = candidate["cost"] <= rec.config["selection"]["max_normalized_work"]
    return support_feasible(rec, j), metric, bool(resource)


def gate_text(values) -> str:
    return "/".join(PASS[v] or "—" for v in values)


def confirmation_output(rec: RunRecord) -> str:
    j = rec.confirmed
    if j == rec.qc:
        return f"qC / {rec.label(j)}"
    return f"{rec.label(j)}, E={rec.candidates[j]['ensemble']}"


def s12_rows(rec: RunRecord) -> list[dict]:
    decision, reps = rec.decision, representatives(rec)
    rows = []
    for family in ("qC",) + S12_FAMILIES:
        comparator = family == "qC"
        j = rec.qc if comparator else reps[family]
        row = {"family": family, "route": "qC" if comparator else ROUTE_LABELS[family], "index": j}
        if j is None:
            rows.append(row | dict.fromkeys(("E", "profile", "delta", "low", "high", "support", "metric", "resource",
                                             "rank", "selected", "confirmation", "candidate_id")))
            continue
        c = rec.candidates[j]
        # The comparator's contrast with itself is identically zero.
        delta, low, high = (0.0, 0.0, 0.0) if comparator else (decision["mean"][j], decision["L"][j],
                                                               decision["U"][j])
        selected = j == rec.provisional and (comparator or j != rec.qc)
        support, metric, resource = gates(rec, j)
        row.update(E=c["ensemble"], profile=PROFILE_LABELS.get(c["profile"], c["profile"]), delta=delta, low=low,
                   high=high, support=support, metric=metric, resource=resource,
                   rank=0 if comparator else c["complexity"], selected=selected,
                   confirmation=confirmation_output(rec) if selected else None, candidate_id=c["candidate_id"])
        rows.append(row)
    return rows


def merge_s12(split_rows: list[list[dict]]) -> list[dict]:
    """Average the per-split rows of a campaign family by family."""
    if len(split_rows) == 1:
        return split_rows[0]
    merged = []
    for parts in zip(*split_rows):
        row = {"family": parts[0]["family"], "route": parts[0]["route"], "index": None}
        for key in ("delta", "low", "high"):
            row[key] = mean(p[key] for p in parts)
        for key in ("E", "rank"):
            row[key] = common([p[key] for p in parts])
        row["profile"] = tally([p["profile"] for p in parts])
        for key in ("support", "metric", "resource"):
            values = [p[key] for p in parts]
            row[key] = None if None in values else all(values)
        chosen = sum(bool(p["selected"]) for p in parts)
        row["selected"] = True if chosen == len(parts) else False if chosen == 0 else f"{chosen}/{len(parts)}"
        row["confirmation"] = tally([p["confirmation"] for p in parts])
        merged.append(row)
    return merged


def nested_family_effects(rec: RunRecord) -> tuple[dict, list]:
    """Algorithm S4, step 10: every open-loop family re-selected inside the nested comparator loop.

    In each outer fold the family's own representative is the configuration with
    the smallest outer-training risk (the within-family rule of the comparator
    reconstruction) and is evaluated against that fold's nested comparator on the
    held-out units. The comparator of every fold is replayed and must equal the
    sealed trace. Returns {family: nested paired effect} and the fold trace.
    """
    from cross_morse import statistics
    data = rec.view.npz("select/unit_scores.npz")
    loss = np.asarray(data["loss"], dtype=np.float64)
    weights = np.asarray(data["counts"], dtype=np.float64)
    candidates = rec.candidates
    observed = [j for j, c in enumerate(candidates) if not c.get("execution_failure")]
    open_ids = [j for j in observed if candidates[j].get("is_open_loop") and candidates[j].get("feasible", True)]
    families: dict[str, list[int]] = {}
    for j in open_ids:
        families.setdefault(candidates[j]["route"], []).append(j)
    outer = np.unique(np.asarray(data["outer_fold"]))
    inner = np.asarray(data["inner_fold"])
    if inner.ndim == 2 and len(inner) == len(outer) + 1:
        inner = inner[:-1]
    sub_candidates = [candidates[j] for j in observed]
    sub_open = [observed.index(j) for j in open_ids]
    totals = {family: 0.0 for family in families}
    trace = []
    for fold in rec.decision["nested_comparator_fold_choices"]:
        held = np.zeros(len(weights), dtype=bool)
        held[fold["held_out_original_units"]] = True
        train = weights * ~held
        k = int(np.flatnonzero(outer == fold["outer_fold"])[0])
        labels = inner if inner.ndim == 1 else inner[k]
        chosen, _ = statistics._reconstruct_comparator(loss[:, observed], train, labels, sub_open, sub_candidates)
        comparator = fold["candidate_index"]
        if observed[chosen] != comparator:
            raise DisplayInputError(f"{rec.view.dataset}: the nested comparator of outer fold {fold['outer_fold']} "
                                    "does not replay from select/unit_scores.npz")
        risk = np.full(loss.shape[1], np.nan)
        risk[observed] = train @ loss[:, observed] / train.sum()
        for family, ids in families.items():
            j = statistics._argmin(risk, ids, candidates)
            totals[family] += float(weights[held] @ (loss[held, j] - loss[held, comparator]))
            trace.append({"outer_fold": fold["outer_fold"], "family": family, "representative": j,
                          "comparator": comparator, "comparator_family": candidates[comparator]["route"]})
    return {family: value / weights.sum() for family, value in totals.items()}, trace


def readings(rec: RunRecord, alternative_margin: float) -> dict:
    """Main Eq. (19) on the representative feedback rows under the three readings of Table S12c."""
    from paper.build.selection import apply_reading, registered_rule_on_all_configurations
    d = rec.decision
    eps_r, eps_eq = d["epsilon_r"], d["epsilon_eq"]

    def row(j):
        c = rec.candidates[j]
        return {"index": j, "route": c["route"], "joint_feasible": d["joint_feasible"][j], "feasible": c["feasible"],
                "select_upper": d["U"][j], "select_delta": d["mean"][j], "complexity": c["complexity"],
                "ensemble": c["ensemble"], "cost": c["cost"], "hash": c["hash"]}

    reps = representatives(rec)
    rows = [row(reps[f]) for f in FEEDBACK_FAMILIES if reps[f] is not None]
    out = {}
    for name, statistic, margin in (("registered", "select_upper", eps_eq), ("point", "select_delta", eps_eq),
                                    ("alternative", "select_delta", alternative_margin)):
        result = apply_reading(rows, statistic, margin, eps_r)
        near = [rec.label(j) for j in result["near_best"]]
        choice = rec.rule(result["choice"]) if result["choice"] is not None else "qC"
        out[name] = {"near": near, "choice": choice, "reference": result["reference"],
                     "text": f"{', '.join(near) or '—'} → {choice}",
                     "compare": f"{', '.join(near)} -> {choice}" if near else choice}
    everything = [row(j) for j, c in enumerate(rec.candidates) if d["U"][j] is not None]
    executed = registered_rule_on_all_configurations(
        {"selection": {"epsilon_r": eps_r, "epsilon_eq": eps_eq, "qC_index": rec.qc}, "candidates": everything})
    out["U_min"] = out["registered"]["reference"]
    out["executed"] = rec.rule(executed)
    out["separated"] = rec.rule(rec.provisional)
    return out


def selection_interpretation(dataset: str, decision: dict, *, campaign: bool) -> str:
    """Registered reading of a selection outcome (Supplementary Table S20).

    A graph-level selection is a boundary selection when the best eligible upper
    bound lies within epsilon_eq of the eligibility threshold -epsilon_R.
    """
    if not decision.get("Q_s"):
        return "comparator/simple route"
    if campaign:
        return "repeated-split selection"
    if dataset in GRAPH_LEVEL:
        upper = [decision["U"][j] for j in decision["Q_s"]]
        boundary = min(upper) > -decision["epsilon_r"] - decision["epsilon_eq"]
        return "boundary graph-level selection" if boundary else "graph-level selection"
    return "conditional near-best selection"


# ---------------------------------------------------------------------------
# Supplementary Table S4: target qualification
# ---------------------------------------------------------------------------

def entry_statistics(view) -> dict:
    """The six gates of main Eq. (5) for one run, with the thresholds of its own configuration."""
    initial = view.json("train/entry_initial.json")
    complete = view.json("select/entry_complete.json")
    support = view.json("fit/full/audit.json")["support"]
    actions = view.json("action_registry.json")
    entry = view.json("config.json")["entry"]
    nominal = next(a["id"] for a in actions if a["name"] == "nominal")
    strata = support["registered_strata"]
    # phi_selected: the nominal action A0 must be supported in every registered stratum (convention C28).
    selected = (sum(bool(support["cells"].get(f"{s}:{nominal}", {}).get("qualified")) for s in strata) / len(strata)
                if strata else None)
    plant = initial["plant_vs_q0"]
    headroom = complete["headroom"]
    pdd = initial["pdd_fraction"]
    value = {"delta_plant": plant["mean"], "U_plant": plant["upper"], "PDD_pp": None if pdd is None else 100 * pdd,
             "DIL_gap": initial["dil_gap"], "phi_registry": complete["registry_support_fraction"],
             "phi_selected": selected, "U_oracle": headroom["upper"], "identity_pass": initial["identity_pass"],
             "runtime_passed": complete["passed"]}
    value["gates"] = {
        "identity": bool(initial["identity_pass"]),
        "nominal graph value": plant["upper"] is not None and plant["upper"] < -entry["plant_margin"],
        "peak-depth deterioration": pdd is not None and pdd <= entry["pdd_max"],
        "depth-integrated gap": initial["dil_gap"] is not None and initial["dil_gap"] <= entry["dil_gap_max"],
        "registry support": complete["registry_support_fraction"] >= entry["registry_fraction_min"],
        "selected strata": selected == 1.0,
        "residual headroom": headroom["upper"] is not None and headroom["upper"] < -entry["headroom_margin"],
    }
    value["passed"] = all(value["gates"].values())
    return value


@display("supplementary:Table S4", stem="S4", title="Target-qualification ledger", needs=("runs",))
def table_s4(ctx, out):
    rows, source = [], []
    for dataset in DATASETS:
        view = ctx.require_run(dataset)
        split = [entry_statistics(m) for m in view.members()]
        # Campaign: mean plant effect; each gate statistic at its least favourable split.
        worst = {"delta_plant": mean(s["delta_plant"] for s in split)}
        for key, pick in (("U_plant", max), ("PDD_pp", max), ("DIL_gap", max), ("phi_registry", min),
                          ("phi_selected", min), ("U_oracle", max)):
            worst[key] = least_favourable((s[key] for s in split), pick)
        decision = "pass" if all(s["passed"] for s in split) else "fail"
        rows.append([dataset, num(worst["delta_plant"], 4), num(worst["U_plant"], 4), num(worst["PDD_pp"], 2),
                     num(worst["DIL_gap"], 3), num(worst["phi_registry"], 2), num(worst["phi_selected"], 2),
                     num(worst["U_oracle"], 4), decision])
        for index, s in enumerate(split):
            failed = [g for g, ok in s["gates"].items() if not ok]
            source.append([dataset, index if view.is_campaign else "", s["delta_plant"], s["U_plant"], s["PDD_pp"],
                           s["DIL_gap"], s["phi_registry"], s["phi_selected"], s["U_oracle"], s["identity_pass"],
                           s["passed"], s["runtime_passed"], " ".join(failed)])
        for key, value in (("Delta_plant_hat", worst["delta_plant"]), ("U_Delta_plant", worst["U_plant"]),
                           ("PDD_pp", worst["PDD_pp"]), ("DIL_gap_nats", worst["DIL_gap"]),
                           ("phi_registry", worst["phi_registry"]), ("selected_strata", worst["phi_selected"]),
                           ("U_Delta_oracle", worst["U_oracle"]), ("decision", decision)):
            out.compare(GROUP["S4"], (dataset, key), value)
        out.uses("run", dataset, ["train/entry_initial.json", "select/entry_complete.json", "fit/full/audit.json",
                                  "action_registry.json", "config.json"])
    out.table(["Dataset", "Delta_plant", "U(Delta_plant)", "PDD (pp)", "DIL gap", "phi_registry", "selected strata",
               "U(Delta_oracle)", "decision"], rows,
              notes=["Gates of main Eq. (5) with the thresholds of each run's configuration (entry); PDD = 100 x "
                     "pdd_fraction. selected strata: fraction of registered strata whose nominal-action cell A0 passes "
                     "the support thresholds (convention C28). decision: pass when E1 identity and all six gates hold.",
                     "Roman-empire: mean plant effect over the ten splits; every other gate statistic at its least "
                     "favourable split; pass only if every split passes."])
    out.data("entry_gates", ["dataset", "split", "Delta_plant", "U_Delta_plant", "PDD_pp", "DIL_gap", "phi_registry",
                             "phi_selected", "U_Delta_oracle", "identity_pass", "passed", "runtime_entry_passed",
                             "failed_gates"], source)


# ---------------------------------------------------------------------------
# Supplementary Table S7: per-cell support audit
# ---------------------------------------------------------------------------

def support_summary(view) -> dict:
    """Extrema over the (stratum, action) cells with logged rows of the full-profile support audit."""
    audit = view.json("fit/full/audit.json")["support"]
    trajectories = view.json("explore/manifest.json")["trajectories"]
    cells = list(audit["cells"].values())
    observed = [c for c in cells if c["row_count"] > 0]
    if not observed:
        raise DisplayInputError(f"{view.dataset}: the support audit holds no observed cell")
    return {"trajectories": trajectories, "min_count": min(c["trajectory_count"] for c in observed),
            "min_propensity": min(c["min_propensity"] for c in observed), "min_ess": min(c["ess"] for c in observed),
            "max_weight": max(c["max_normalized_weight"] for c in observed),
            "qualified": sum(bool(c["qualified"]) for c in cells), "cells": len(cells),
            "qualified_fraction": audit["qualified_fraction"], "empty_cells": len(cells) - len(observed),
            "failed_cells": sum(not c["qualified"] for c in cells), "thresholds": audit["thresholds"]}


@display("supplementary:Table S7", stem="S7", title="Per-cell support audit and registered thresholds",
         needs=("runs",))
def table_s7(ctx, out):
    remediation = ctx.constants["exploration_support_table_S7"]
    rows, source = [], []
    for dataset in DATASETS:
        view = ctx.require_run(dataset)
        split = [support_summary(m) for m in view.members()]
        value = {"trajectories": common([s["trajectories"] for s in split]),
                 "min_count": min(s["min_count"] for s in split),
                 "min_propensity": min(s["min_propensity"] for s in split),
                 "min_ess": min(s["min_ess"] for s in split), "max_weight": max(s["max_weight"] for s in split),
                 "qualified_fraction": sum(s["qualified"] for s in split) / sum(s["cells"] for s in split)}
        rows.append([dataset, value["trajectories"], integer(value["min_count"]), num(value["min_propensity"], 3),
                     integer(value["min_ess"]), num(value["max_weight"], 2), num(value["qualified_fraction"], 2),
                     remediation[dataset]["failure_remediation"]])
        for index, s in enumerate(split):
            source.append([dataset, index if view.is_campaign else "", s["trajectories"], s["min_count"],
                           s["min_propensity"], s["min_ess"], s["max_weight"], s["qualified"], s["cells"],
                           s["qualified_fraction"], s["failed_cells"], s["empty_cells"], s["thresholds"]["p_min"],
                           s["thresholds"]["n_min"], s["thresholds"]["max_normalized_weight"]])
        trajectories = split[0]["trajectories"] if len({s["trajectories"] for s in split}) == 1 else None
        for key, number in (("trajectories", trajectories), ("minimum_count", value["min_count"]),
                            ("minimum_propensity", value["min_propensity"]),
                            ("qualified_fraction", value["qualified_fraction"]),
                            ("per_cell_minimum_ESS", value["min_ess"]),
                            ("per_cell_maximum_normalized_weight", value["max_weight"])):
            out.compare(GROUP["S7"], (dataset, key), number)
        out.uses("run", dataset, ["explore/manifest.json", "fit/full/audit.json"])
    out.table(["Dataset", "trajectories", "minimum count", "minimum propensity", "ESS (per-cell minimum)",
               "max. normalized weight (per cell)", "qualified fraction", "failure/remediation"], rows,
              notes=["Per (stratum, action) cell of fit/full/audit.json: trajectory units, minimum logged propensity, "
                     "ESS and maximum normalized weight of the one weight vector of the cell. Extrema run over the "
                     "cells with logged rows (empty registered cells are counted in the source data); since "
                     "ESS >= n_min bounds the maximum weight by n_min^-1/2, an ESS below n_min or a weight above "
                     "it belongs to a cell that fails the thresholds. "
                     "qualified fraction: passing cells over all registered cells.",
                     "Roman-empire: trajectories per split run; extrema over the cells of all ten splits; qualified "
                     "fraction pooled over the splits. failure/remediation: registered rule "
                     "(registered_constants exploration_support_table_S7)."])
    out.data("support_cells", ["dataset", "split", "trajectories", "minimum_count", "minimum_propensity",
                               "minimum_ESS", "maximum_normalized_weight", "qualified_cells", "cells",
                               "qualified_fraction_runtime", "failed_cells", "empty_cells", "p_min", "n_min",
                               "omega_max"], source)


# ---------------------------------------------------------------------------
# Supplementary Tables S12, S12b, S12c: candidate ledger and selector audit
# ---------------------------------------------------------------------------

def _selected_text(value):
    if isinstance(value, bool):
        return "yes" if value else "no"
    return value


@display("supplementary:Table S12", stem="S12", title="Route-level candidate ledger", needs=("runs",))
def table_s12(ctx, out):
    rows, source = [], []
    for dataset in DATASETS:
        view = ctx.require_run(dataset)
        split_rows = [s12_rows(rec) for rec in records(view)]
        for index, part in enumerate(split_rows):
            for row in part:
                source.append([dataset, index if view.is_campaign else "", row["family"], row["index"],
                               row["candidate_id"], row["E"], row["profile"], row["delta"], row["low"], row["high"],
                               row["support"], row["metric"], row["resource"], row["rank"], row["selected"],
                               row["confirmation"]])
        for i, row in enumerate(merge_s12(split_rows)):
            rows.append([dataset, row["route"], integer(row["E"]) if isinstance(row["E"], int) else row["E"],
                         row["profile"], num(row["delta"], 5), num(row["low"], 5), num(row["high"], 5),
                         gate_text((row["support"], row["metric"], row["resource"])),
                         integer(row["rank"]) if isinstance(row["rank"], int) else row["rank"],
                         _selected_text(row["selected"]), row["confirmation"] if row["confirmation"] is not None
                         else not_applicable("The candidate was not selected for confirmation")])
            for key, value in (("route", row["route"]), ("E", row["E"]), ("profile", row["profile"]),
                               ("Delta_s_hat", row["delta"]), ("L_s", row["low"]), ("U_s", row["high"]),
                               ("rank", row["rank"]), ("selected", row["selected"]),
                               ("confirmation_output", row["confirmation"])):
                out.compare(GROUP["S12"], (dataset, i, key), value)
        out.uses("run", dataset, ["select/decision.json", "select/candidates.json", "confirm/decision.json",
                                  "admit/decision.json", "predictions/*.json", "config.json"])
    out.table(["Dataset", "route", "E", "profile", "Delta_s", "L_s", "U_s", "support / metric / resource", "rank",
               "selected", "confirmation output"], rows,
              notes=["Representative row per route family: the selected, confirmed or deployed configuration in its "
                     "family, otherwise the configuration with the smallest U_s. Delta_s, [L_s, U_s]: paired selection "
                     "effect against the nested comparator and its simultaneous max-t interval (select/decision.json); "
                     "the q_C row is the comparator's zero contrast with itself. Gates: support (no support violation "
                     "in the select-stage execution), metric (metric lower bound >= -eps_metric), resource (cost <= "
                     "max_normalized_work). rank: registry complexity (q_C row 0).",
                     "Roman-empire: effects averaged over the ten splits; selected shows the number of splits when "
                     "they differ; per-split rows in the source data."])
    out.data("representative_rows", ["dataset", "split", "family", "candidate_index", "candidate_id", "E", "profile",
                                     "Delta_s", "L_s", "U_s", "support", "metric", "resource", "rank", "selected",
                                     "confirmation_output"], source)


@display("supplementary:Table S12b", stem="S12b", title="Nested-family comparison", needs=("runs",))
def table_s12b(ctx, out):
    rows, source = [], []
    for dataset in DATASETS:
        view = ctx.require_run(dataset)
        recs = records(view)
        if not any(rec.decision["Q_s"] for rec in recs):
            continue        # comparator outcome: no feedback selection whose comparison could carry optimism
        per_split = []
        for index, rec in enumerate(recs):
            effects, trace = nested_family_effects(rec)
            reps = representatives(rec)
            identity = {family: all(t["comparator_family"] == family for t in trace if t["family"] == family)
                        for family in effects}
            per_split.append({family: (rec.decision["mean"][reps[family]] if reps[family] is not None else None,
                                       effects.get(family), identity.get(family, False))
                              for family in OPEN_LOOP_FAMILIES})
            for t in trace:
                source.append([dataset, index if view.is_campaign else "", t["outer_fold"], t["family"],
                               t["representative"], t["comparator"], t["comparator_family"],
                               effects[t["family"]]])
        for i, family in enumerate(OPEN_LOOP_FAMILIES):
            outer = mean(split[family][0] for split in per_split)
            nested = mean(split[family][1] for split in per_split)
            if nested is None:
                reading = None
            elif all(split[family][2] for split in per_split):
                reading = "identity (qC family)"
            elif outer is not None and nested > outer:
                reading = "re-selection optimism removed"
            else:
                reading = "no re-selection optimism"
            rows.append([dataset, ROUTE_LABELS[family], num(outer, 5), num(nested, 5, signed=True), reading])
            out.compare(GROUP["S12b"], (dataset, i, "route"), ROUTE_LABELS[family])
            out.compare(GROUP["S12b"], (dataset, i, "nested_Delta_s_nats"), nested)
        out.uses("run", dataset, ["select/decision.json", "select/unit_scores.npz", "select/candidates.json"])
    out.table(["Dataset", "route", "outer-fold Delta_s (Table S12)", "nested Delta_s", "reading"], rows,
              notes=["nested Delta_s: Algorithm S4 step 10 with the family's own representative (smallest "
                     "outer-training risk in the family) in place of q, against each outer fold's nested comparator "
                     "replayed from select/unit_scores.npz. Datasets whose selection returned q_C (empty Q_s) carry no "
                     "feedback selection and are omitted.",
                     "Roman-empire: both effects averaged over the ten splits."])
    out.data("nested_family_folds", ["dataset", "split", "outer_fold", "family", "family_representative",
                                     "nested_comparator", "comparator_family", "nested_Delta_s"], source)


@display("supplementary:Table S12c", stem="S12c", title="The registered selector on the representative rows",
         needs=("runs",))
def table_s12c(ctx, out):
    alternative = min(ctx.constants["sensitivity_grids_table_S18"]["equivalence_margin_epsilon_eq"])
    rows, source, eps_eq = [], [], None
    for dataset in DATASETS:
        view = ctx.require_run(dataset)
        recs = records(view)
        split = [readings(rec, alternative) for rec in recs]
        eps_eq = recs[0].decision["epsilon_eq"]
        u_min = mean(s["U_min"] for s in split)
        cells = {name: tally([s[name]["text"] for s in split]) for name in ("registered", "point", "alternative")}
        compared = {name: tally([s[name]["compare"] for s in split]) for name in ("registered", "point", "alternative")}
        executed, separated = tally([s["executed"] for s in split]), tally([s["separated"] for s in split])
        rows.append([SHORT[dataset], num(u_min, 5), cells["registered"], cells["point"], cells["alternative"],
                     executed, separated])
        for index, s in enumerate(split):
            source.append([dataset, index if view.is_campaign else "", s["U_min"], s["registered"]["compare"],
                           s["point"]["compare"], s["alternative"]["compare"], s["executed"], s["separated"]])
        for key, value in (("U_min", u_min), ("registered_reading_eps_eq_0.002_on_U_s", compared["registered"]),
                           ("point_estimate_reading_eps_eq_0.002", compared["point"]),
                           ("point_estimate_reading_eps_eq_0.001", compared["alternative"]), ("selected", executed)):
            out.compare(GROUP["S12c"], (dataset, key), value)
        out.uses("run", dataset, ["select/decision.json", "select/candidates.json", "confirm/decision.json",
                                  "admit/decision.json"])
    out.table(["Dataset", "U_min", f"eps_eq = {eps_eq:g} on U_s: near-best -> choice",
               f"eps_eq = {eps_eq:g} on Delta_s", f"eps_eq = {alternative:g} on Delta_s",
               "executed selection (complete ledger)", "separated-protocol selection"], rows,
              notes=["Main Eq. (19) on the representative feedback rows of Table S12: eligible when jointly feasible "
                     "and U_s < -eps_R, near-best within eps_eq of the best eligible statistic, least-complex choice. "
                     "The alternative margin is the smallest registered sensitivity value (Table S18). Executed "
                     "selection: the registered rule replayed on every configuration of select/decision.json; "
                     "separated-protocol selection: the provisional selection sealed by the run.",
                     "Roman-empire: U_min averaged over the ten splits; readings that differ between splits are listed "
                     "with their split counts."])
    out.data("selector_readings", ["dataset", "split", "U_min", "registered_reading", "point_reading",
                                   "alternative_reading", "executed_selection", "separated_protocol_selection"], source)


# ---------------------------------------------------------------------------
# Supplementary Table S13: admission arithmetic and clipping sensitivity
# ---------------------------------------------------------------------------

def admission_rows(view, sensitivity) -> list[dict]:
    """Main Eq. (27) at the registered clip and at the sensitivity clips, replayed with the runtime bound."""
    from cross_morse import statistics
    rec = RunRecord(view)
    units = view.npz("admit/paired_units.npz")
    differences = np.asarray(units["candidate_loss"], float) - np.asarray(units["comparator_loss"], float)
    ac, admission = rec.config["admission"], rec.admission
    execution = admission.get("execution_gate")
    rows = []
    for clip in [ac["clip"], *sensitivity]:
        bound = statistics.empirical_bernstein(differences, clip=clip, alpha=ac["alpha"], independent_graphs=True)
        mean_gate = bound["mean"] <= -ac["epsilon_a"]
        upper_gate = bound["upper"] is not None and bound["upper"] <= -ac["epsilon_a"]
        admit = int(mean_gate and upper_gate and admission["metric_gate"] and execution is not False)
        rows.append({"J": bound["units"], "B": clip, "delta_a": ac["alpha"], "epsilon_a": ac["epsilon_a"],
                     "D_bar": bound["mean"], "S2": bound["sample_variance"], "penalty": bound["penalty"],
                     "U_a": bound["upper"], "metric_gate": admission["metric_gate"], "G_admit": admit,
                     "clipped_fraction": bound["clipping_fraction"]})
    recorded = admission.get("empirical_bernstein") or {}
    if (not math.isclose(rows[0]["U_a"], recorded.get("upper", math.nan), rel_tol=1e-9, abs_tol=1e-12)
            or rows[0]["G_admit"] != admission["Gadmit"]):
        raise DisplayInputError(f"{view.dataset}: admission replay differs from admit/decision.json")
    return rows


@display("supplementary:Table S13", stem="S13", title="Admission arithmetic and clipping sensitivity",
         needs=tuple(f"run:{d}" for d in GRAPH_LEVEL))
def table_s13(ctx, out):
    sensitivity = ctx.constants["admission_eq27_theorem2_table_S13"]["sensitivity_B"]
    computed = {d: admission_rows(ctx.require_run(d), sensitivity) for d in GRAPH_LEVEL}
    rows, source = [], []

    def line(label, r):
        return [label, integer(r["J"]), f"{r['B']:g}", f"{r['delta_a']:g}", f"{r['epsilon_a']:g}", num(r["D_bar"], 5),
                num(r["S2"], 6), num(r["penalty"], 6), num(r["U_a"], 6), PASS[r["metric_gate"]], r["G_admit"]]

    for dataset in GRAPH_LEVEL:
        rows.append(line(dataset, computed[dataset][0]))
    for dataset in GRAPH_LEVEL:
        for r in computed[dataset][1:]:
            rows.append(line(f"{dataset} (B = {r['B']:.2f})", r))
    for dataset in GRAPH_LEVEL:
        base = computed[dataset][0]
        for key in ("J", "B", "delta_a", "epsilon_a", "D_bar", "S2", "penalty", "U_a", "G_admit"):
            out.compare(GROUP["S13"], (dataset, key), base[key])
        out.compare(GROUP["S13"], (dataset, "metric_gate"), PASS[base["metric_gate"]])
        for r in computed[dataset][1:]:
            for key in ("penalty", "U_a", "G_admit"):
                out.compare(GROUP["S13"], (dataset, "sensitivity", f"B={r['B']:.2f}", key), r[key])
        source += [[dataset, *(r[k] for k in ("B", "J", "delta_a", "epsilon_a", "D_bar", "S2", "penalty", "U_a",
                                              "clipped_fraction", "metric_gate", "G_admit"))]
                   for r in computed[dataset]]
        out.uses("run", dataset, ["admit/paired_units.npz", "admit/decision.json", "config.json"])
    out.table(["Dataset", "J", "B", "delta_a", "epsilon_a", "D_bar", "S2", "penalty", "U_a", "metric gate",
               "G_admit"], rows,
              notes=["cross_morse.statistics.empirical_bernstein on the paired admission units "
                     "(admit/paired_units.npz), re-clipped at each B; the registered-clip row reproduces "
                     "admit/decision.json. G_admit = [D_bar <= -eps_a] and [U_a <= -eps_a] and the recorded metric "
                     "(and execution) gate."])
    out.data("admission_moments", ["dataset", "B", "J", "delta_a", "epsilon_a", "D_bar", "S2", "penalty", "U_a",
                                   "clipped_fraction", "metric_gate", "G_admit"], source)


# ---------------------------------------------------------------------------
# Supplementary Table S14: the nine-test joint confirmation family
# ---------------------------------------------------------------------------

def joint_confirmation(ctx) -> tuple[dict, dict]:
    """Validated joint S14 campaign and each test's original result record.

    The campaign validator replays effects, joint Holm and the frozen decisions;
    its agreement with the printed values is not required here, because the
    printed values are compared cell by cell afterwards.
    """
    from cm_campaigns.common import EvidenceError, directory_inventory, read_json
    from cm_campaigns.confirmation import check_campaign
    path = ctx.require_campaign("confirmation")
    try:
        before = directory_inventory(path)
        checked = check_campaign(path, manuscript_root=ctx.root)
        manifest = read_json(path / "campaign.json")
        results = {entry["contrast_id"]: read_json(path / entry["result"]["path"]) for entry in manifest["tests"]}
        if directory_inventory(path) != before:
            raise EvidenceError("campaign files changed during validation")
    except (EvidenceError, OSError, KeyError) as exc:
        raise DisplayValidationError(f"joint confirmation campaign failed validation: {exc}") from exc
    return checked, results


FLOOR_TEXT = {1e-4: "1e-4", 1e-3: "0.001"}


def p_text(p, floor):
    """Printed P value: two significant digits, or an inequality below ``floor``."""
    if p is None:
        return None
    if p < floor:
        return f"< {FLOOR_TEXT[floor]}"
    digits = max(0, 1 - math.floor(math.log10(p))) if p > 0 else 2
    return f"{p:.{digits}f}"


def p_compare(p, floor):
    return None if p is None else (f"< {FLOOR_TEXT[floor]}" if p < floor else p)


KIND_TEXT = {"state": "conditional destruction minus intact", "radius": "radius minus one-step",
             "mpc": "MPC-RH minus MPC-OL"}
UNIT_TEXT = {"buffered_cluster": "clusters", "official_split": "splits", "graph": "confirmation graphs"}


def interpretation(row, kind, decisions, alpha) -> str:
    """Reading of one joint-family test under the frozen confirmation map (Table S11)."""
    direction, margin = row["beneficial_direction"], row["practical_margin_nats"]
    low, high = row["interval_nats"]
    far = low if direction > 0 else -high          # the interval end nearest to zero, signed beneficially
    if row["retained"]:
        if far > margin:
            return {"state": "useful state dependence", "radius": "radius retained",
                    "mpc": "replanning retained"}[kind]
        subject = {"state": "state dependence", "radius": "radius", "mpc": "replanning"}[kind]
        return f"{subject} retained on the point estimate; interval does not exclude the margin"
    if not row["tested_policy_selected"]:
        return "no feedback label; equivalence not established"
    decision = decisions[row["dataset"]]
    outcome = "qC" if decision["confirmed_hash"] == decision["comparator_hash"] else "predecessor"
    return f"Holm P = {row['holm_p_value_zero']:.3f}; registered alpha = {alpha:g}; frozen-rule outcome {outcome}"


@display("supplementary:Table S14", stem="S14", title="Mechanism statistics and decision definitions",
         needs=("campaign:confirmation",))
def table_s14(ctx, out):
    from cm_campaigns.confirmation import FAMILY
    checked, results = joint_confirmation(ctx)
    confirmation_scope(out, checked)
    by_id = {row["contrast_id"]: row for row in checked["rows"]}
    rows, source = [], []
    for i, key in enumerate(FAMILY):
        row, kind = by_id[key], FAMILY[key][1]
        loss = LOSS_LABELS["bce" if "BCE" in row["loss"] else "ce"]
        margin_p = results[key].get("p_value_margin", ...)      # optional field of the original test record
        if margin_p is not ... and margin_p is not None and (isinstance(margin_p, bool)
                or not isinstance(margin_p, (int, float)) or not math.isfinite(margin_p) or not 0 <= margin_p <= 1):
            raise DisplayValidationError(f"{key}: p_value_margin must be a probability or an explicit not-performed null")
        reading = interpretation(row, kind, checked["policy_decisions"], checked["alpha"])
        effect = row["effect_nats"]
        rows.append([key, KIND_TEXT[kind], f"{num(effect, 5, signed=True)} {loss}",
                     interval(*row["interval_nats"], 5), f"{row['unit_count']} {UNIT_TEXT[row['statistical_unit']]}",
                     row["method_name"], p_text(row["p_value_zero"], 1e-4), p_text(row["holm_p_value_zero"], 1e-3),
                     f"{row['practical_margin_nats']:g}",
                     None if margin_p is ... else ("not tested" if margin_p is None else p_text(margin_p, 1e-3)),
                     reading])
        source.append([key, row["dataset"], kind, effect, *row["interval_nats"], row["statistical_unit"],
                       row["unit_count"], row["seed_count"], row["method_name"], row["p_value_zero"],
                       row["holm_p_value_zero"], row["practical_margin_nats"], row["beneficial_direction"],
                       None if margin_p is ... else margin_p, row["retained"], row["tested_policy_selected"]])
        for field, value in (("effect", effect), ("interval", list(row["interval_nats"])), ("n", row["unit_count"]),
                             ("raw_P", p_compare(row["p_value_zero"], 1e-4)),
                             ("holm_P", p_compare(row["holm_p_value_zero"], 1e-3)),
                             ("margin", row["practical_margin_nats"]), ("interpretation", reading)):
            out.compare(GROUP["S14"], (i, field), value)
        if margin_p is not ...:
            out.compare(GROUP["S14"], (i, "P_vs_margin"), p_compare(margin_p, 1e-3))
        else:
            out.compare_missing(GROUP["S14"], (i, "P_vs_margin"), reason="The supplied test record omits its margin test")
            out.incomplete(f"{key}: the margin-test result is absent", location="P vs margin")
    out.uses("campaign", "confirmation", ["campaign.json", "protocol.json", "raw/*/test.json", "raw/*/paired.csv"])
    out.table(["Dataset/claim", "contrast", "effect", "95% interval", "unit/n", "test", "raw P", "Holm P", "margin",
               "P vs margin", "interpretation"], rows,
              notes=["Joint nine-test family of the validated confirmation campaign (cm_campaigns.confirmation): "
                     "supplied effects, intervals and raw P against zero; Holm over the nine raw P values; retained "
                     "when Holm P < alpha and the point effect lies beyond the signed practical margin.",
                     "P vs margin: the one-sided test against the margin, read from the optional p_value_margin field "
                     "of each supplied test record (null = not performed); an absent field is explicitly incomplete."])
    out.data("joint_family", ["contrast", "dataset", "kind", "effect_nats", "interval_low_nats", "interval_high_nats",
                              "statistical_unit", "units", "seeds", "method", "raw_P_zero", "holm_P_zero",
                              "margin_nats", "beneficial_direction", "P_vs_margin", "retained",
                              "tested_policy_selected"], source)


def confirmation_scope(out, checked):
    """Preserve the distinction between an imported test and a retrospective new family."""
    out.provenance(confirmation={key: checked.get(key) for key in
        ("production_kind", "post_hoc", "admission_findings", "original_test_p_values_recomputed", "scope", "limits")})
    if checked.get("post_hoc"):
        out.note("This supplied family was produced after the lifecycle admissions. Its reconstructed statistics "
                 "do not establish that a prospective joint lock governed the original admission decisions.")


# ---------------------------------------------------------------------------
# Supplementary Table S20: selection radii
# ---------------------------------------------------------------------------

@display("supplementary:Table S20", stem="S20", title="Selection radii and their interpretation",
         needs=("runs",))
def table_s20(ctx, out):
    rows, source = [], []
    for dataset in DATASETS:
        view = ctx.require_run(dataset)
        members = view.members()
        decisions = [m.json("select/decision.json") for m in members]
        screened = [screened_radius(m) for m in members]
        # Theorem 1 holds per split: a campaign reports the largest radius over its splits (the widest slack),
        # the common family size and the eligible-set sizes of its splits.
        r_max = max(d["max_radius"] for d in decisions)
        r12 = max(s["radius"] for s in screened)
        eps_eq = decisions[0]["epsilon_eq"]
        q_sizes = [d["candidate_count"] for d in decisions]
        qs_sizes = [len(d["Q_s"]) for d in decisions]
        resamples = [d["bootstrap_replicates"] for d in decisions]
        reference = max(decisions, key=lambda d: (len(d["Q_s"]), d["max_radius"]))
        reading = selection_interpretation(dataset, reference, campaign=view.is_campaign)
        rows.append([SHORT[dataset], common(q_sizes), common(qs_sizes), num(r_max, 4), num(eps_eq + 2 * r_max, 4),
                     num(eps_eq + 2 * r12, 4), common(resamples), reading])
        for index, (d, s) in enumerate(zip(decisions, screened)):
            source.append([dataset, index if view.is_campaign else "", d["candidate_count"], len(d["Q_s"]),
                           d["max_radius"], d["epsilon_eq"], d["epsilon_eq"] + 2 * d["max_radius"], s["radius"],
                           d["epsilon_eq"] + 2 * s["radius"], s["critical"], d["bootstrap_replicates"],
                           d["bootstrap_seed"], " ".join(str(j) for j in s["finalists"])])
        single = lambda values: values[0] if len(set(values)) == 1 else None
        for key, value in (("Q_size", single(q_sizes)), ("Q_s_size", single(qs_sizes)), ("r_max", r_max),
                           ("eps_eq_plus_2r_max", eps_eq + 2 * r_max), ("slack_with_12_finalists", eps_eq + 2 * r12),
                           ("resamples", single(resamples)), ("interpretation", reading)):
            out.compare(S20_GROUP, (dataset, key), value)
        out.uses("run", dataset, ["select/decision.json", "select/unit_scores.npz", "select/candidates.json",
                                  "select/metric_observations.npz", "config.json"])
    out.table(["Dataset", "|Q|", "|Qs|", "rmax", "eps_eq + 2 rmax", "slack with 12 finalists", "resamples",
               "interpretation"], rows,
              notes=["Twelve finalists: the feasible candidates with the smallest simultaneous upper bounds U; the "
                     "slack replays Algorithm S4 (cross_morse.statistics.nested_selection with the stage's arrays, "
                     "folds, weights and seed) with the max-t maximum over those finalists.",
                     "Roman-empire: the largest radius over the ten splits; |Q| and |Qs| as a range when splits "
                     "differ."])
    out.data("selection_radii", ["dataset", "split", "Q_size", "Q_s_size", "r_max", "epsilon_eq",
                                 "eps_eq_plus_2r_max", "r_max_12_finalists", "slack_12_finalists",
                                 "max_t_critical_12_finalists", "resamples", "bootstrap_seed", "finalist_indices"],
             source)


# ---------------------------------------------------------------------------
# Supplementary Tables S21 and S22: action gaps and the trajectory tube
# ---------------------------------------------------------------------------

def record_rows(rec, quantities) -> dict:
    """{dataset: {quantity: value}} for every dataset present in a per-dataset record set."""
    out = {}
    for dataset in DATASETS:
        if rec.select(dataset=dataset):
            out[dataset] = {q: rec.require(q, dataset=dataset) for q in quantities}
    if not out:
        raise DisplayInputError(f"the {rec.name} record set holds no registered dataset")
    return out


def gap_interpretation(ctx, dataset, switch) -> str:
    """Fixed-mode wording is adequate while the switch fraction stays within the registered calibration alpha."""
    alpha = ctx.config(dataset)["controller"]["calibration_alpha"]
    return "fixed-mode coverage adequate" if switch <= alpha else "boundary; retain fixed-mode-only wording"


S21_QUANTITIES = ("median_gap_nats", "p5_gap_nats", "near_zero_fraction", "switch_fraction",
                  "recovery_index_stability", "beam_sequence_stability")


@display("supplementary:Table S21", stem="S21", title="Action-gap and switching audit",
         needs=("records:action_gap_audit",))
def table_s21(ctx, out):
    rec = ctx.require_records("action_gap_audit")
    values = record_rows(rec, S21_QUANTITIES + ("near_zero_threshold_nats",))
    rows, source = [], []
    for dataset, v in values.items():
        reading = gap_interpretation(ctx, dataset, v["switch_fraction"])
        rows.append([dataset, num(v["median_gap_nats"], 4), num(v["p5_gap_nats"], 4), num(v["near_zero_fraction"], 3),
                     num(v["switch_fraction"], 3), num(v["recovery_index_stability"], 2),
                     num(v["beam_sequence_stability"], 2), reading])
        source.append([dataset, *(v[q] for q in S21_QUANTITIES), v["near_zero_threshold_nats"],
                       ctx.config(dataset)["controller"]["calibration_alpha"], reading])
        for key, q in (("median_gap", "median_gap_nats"), ("p5_gap", "p5_gap_nats"),
                       ("near_zero_fraction", "near_zero_fraction"), ("switch_fraction", "switch_fraction"),
                       ("recovery_index_stability", "recovery_index_stability"),
                       ("beam_sequence_stability", "beam_sequence_stability")):
            out.compare(GROUP["S21"], (dataset, key), v[q])
        out.compare(GROUP["S21"], (dataset, "interpretation"), reading)
    out.uses("records", "action_gap_audit", sorted({row["source_file"] for row in rec.rows}))
    out.table(["Dataset", "median gap", "5th percentile gap", "near-zero fraction", "switch fraction",
               "recovery-index stability", "beam-sequence stability", "interpretation"], rows,
              notes=["Gaps in nats from the action_gap_audit record set. interpretation: fixed-mode coverage is "
                     "adequate while the switch fraction stays within the registered calibration alpha "
                     "(controller.calibration_alpha); otherwise the wording is restricted to fixed modes."])
    out.data("action_gaps", ["dataset", *S21_QUANTITIES, "near_zero_threshold_nats", "calibration_alpha",
                             "interpretation"], source)


S22_QUANTITIES = ("beta_median", "beta_p95", "m_eps_op", "reference_term", "tube_coverage", "mode_stable_coverage",
                  "final_bound_nats", "observed_error_median_nats", "observed_error_p95_nats")


@display("supplementary:Table S22", stem="S22", title="Plug-in tube audit", needs=("records:tube_audit",))
def table_s22(ctx, out):
    rec = ctx.require_records("tube_audit")
    values = record_rows(rec, S22_QUANTITIES)
    rows, source = [], []
    for dataset, v in values.items():
        unit = task_labels(dataset)[0]
        violation = 1.0 - v["mode_stable_coverage"]          # convention C22
        rows.append([SHORT[dataset], f"{num(v['beta_median'], 2)}/{num(v['beta_p95'], 2)}", num(v["m_eps_op"], 4),
                     num(v["reference_term"], 4), num(v["tube_coverage"], 2), num(v["mode_stable_coverage"], 3),
                     num(violation, 3), f"{num(v['final_bound_nats'], 3)} {unit}",
                     f"{num(v['observed_error_median_nats'], 4)}; {num(v['observed_error_p95_nats'], 4)}"])
        source.append([dataset, *(v[q] for q in S22_QUANTITIES), violation, unit])
        for key, value in (("beta_median", v["beta_median"]), ("beta_95", v["beta_p95"]),
                           ("M_eps_op", v["m_eps_op"]), ("reference_term", v["reference_term"]),
                           ("tube_coverage", v["tube_coverage"]), ("mode_stable_coverage", v["mode_stable_coverage"]),
                           ("violation_rate", violation), ("final_bound", v["final_bound_nats"]),
                           ("bound_unit", unit), ("observed_error_median", v["observed_error_median_nats"]),
                           ("observed_error_p95", v["observed_error_p95_nats"])):
            out.compare(GROUP["S22"], (dataset, key), value)
    out.uses("records", "tube_audit", sorted({row["source_file"] for row in rec.rows}))
    out.table(["Dataset", "beta median/95%", "M eps_op", "reference term", "tube coverage", "mode-stable coverage",
               "violation rate", "final bound", "observed error (median; p95)"], rows,
              notes=["Plug-in product-sum tube of main Proposition 2 from the tube_audit record set; violation rate = "
                     "1 - mode-stable coverage (convention C22); bound unit from the dataset's task loss."])
    out.data("tube", ["dataset", *S22_QUANTITIES, "violation_rate", "unit"], source)


# ---------------------------------------------------------------------------
# Supplementary Table S30: oracle dispersion and the retained-share grid
# ---------------------------------------------------------------------------

def retained_share_grid(ctx, rec: RunRecord):
    """Registered phi_x grid of the comparator family (Note S25); none for a comparator deployment."""
    if rec.deployed == rec.qc:
        return None
    grids = ctx.constants["budget_matched_comparison_section4_table_S32_S40"]["retained_share_grid_phi_x"]
    return {"fixed_sequence": grids["fixed_sequence_regimes"],
            "mpc_ol": grids["MPC_OL_regimes"]}.get(rec.candidates[rec.qc]["route"])


@display("supplementary:Table S30", stem="S30", title="Oracle dispersion and the retained-share grid",
         needs=("runs", "records:oracle_dispersion"))
def table_s30(ctx, out):
    oracle = record_rows(ctx.require_records("oracle_dispersion"),
                         ("median_oracle_depth", "oracle_depth_q25", "oracle_depth_q75", "fraction_differs_from_qc"))
    sensitivity = []
    rows, source = [], []
    for dataset in DATASETS:
        view = ctx.require_run(dataset)
        recs = records(view)
        grids = [retained_share_grid(ctx, rec) for rec in recs]
        grid = grids[0] if all(g == grids[0] for g in grids) else None
        sd = None
        if independent(dataset):
            sd = math.sqrt(admission_rows(view, sensitivity)[0]["S2"])
        o = oracle.get(dataset)
        rows.append([dataset, integer(o and o["median_oracle_depth"]),
                     None if o is None else f"{integer(o['oracle_depth_q25'])}–{integer(o['oracle_depth_q75'])}",
                     num(o and o["fraction_differs_from_qc"], 2), num(sd, 4),
                     None if grid is None else f"{grid[0]:.2f}–{grid[-1]:.2f} ({grid[1]:.2f})"])
        quantities = ("median_oracle_depth", "oracle_depth_q25", "oracle_depth_q75", "fraction_differs_from_qc")
        source.append([dataset, *(None if o is None else o[q] for q in quantities),
                       sd, recs[0].candidates[recs[0].qc]["route"], None if grid is None else " ".join(map(str, grid))])
        if o is not None:
            for key, value in (("median_oracle_depth", o["median_oracle_depth"]),
                               ("IQR", [o["oracle_depth_q25"], o["oracle_depth_q75"]]),
                               ("fraction_oracle_differs_from_qC", o["fraction_differs_from_qc"])):
                out.compare(GROUP["S30"], (dataset, key), value)
        out.compare(GROUP["S30"], (dataset, "per_input_SD_clipped_effect"), sd)
        out.compare(GROUP["S30"], (dataset, "phi_x_grid"), None if grid is None else [grid[0], grid[-1]])
        out.compare(GROUP["S30"], (dataset, "phi_x_central"), None if grid is None else grid[1])
        out.uses("run", dataset, ["select/decision.json", "select/candidates.json", "admit/decision.json"]
                 + (["admit/paired_units.npz"] if independent(dataset) else []))
    out.uses("records", "oracle_dispersion", [])
    out.table(["Dataset", "median oracle depth", "IQR", "fraction with oracle != qC",
               "per-input SD of clipped effect (nats)", "phi_x grid (central)"], rows,
              notes=["Oracle depth and disagreement with q_C: oracle_dispersion record set (admission split, after "
                     "every decision is fixed). Per-input SD: square root of the clipped admission moment S^2 at the "
                     "registered B (Table S13); single-graph benchmarks have no per-input admission unit. phi_x grid: "
                     "the registered retained-share grid of the comparator family (fixed-sequence or MPC-OL regime, "
                     "Note S25); none when the comparator is returned."])
    out.data("oracle_dispersion", ["dataset", "median_oracle_depth", "oracle_depth_q25", "oracle_depth_q75",
                                   "fraction_differs_from_qC", "per_input_SD_clipped_effect", "qC_route",
                                   "phi_x_grid"], source)


# ---------------------------------------------------------------------------
# Supplementary Table S41: qualification accounting
# ---------------------------------------------------------------------------

def stage_hours(view) -> dict:
    """Wall-clock hours of the qualification stages and of the backbone checkpoints on the run's device.

    A completion marker rewritten after an interruption carries no timing; its stage is None.
    """
    hours = {}
    for stage, _, _ in QUALIFICATION_STAGES:
        seconds = view.json(f"stages/{stage}.json").get("seconds_this_attempt")
        hours[stage] = None if seconds is None else seconds / 3600.0
    hours["total"] = total(hours[stage] for stage, _, _ in QUALIFICATION_STAGES)
    checkpoints = [c for c in view.json("train/manifest.json")["checkpoints"] if not c["baseline"]]
    hours["seed"] = [c["seconds_this_attempt"] / 3600.0 for c in checkpoints
                     if c.get("seconds_this_attempt") is not None]
    environment = view.json("environment.json")
    hours["device"] = environment.get("gpu") or environment.get("device")
    hours["gpu"] = str(environment.get("device", "")).startswith("cuda")
    return hours


def total(values):
    values = list(values)
    return None if any(v is None for v in values) else float(sum(values))


def comparison_gpu_hours(campaign: Path) -> tuple[dict[str, float], list[list]]:
    """GPU-hours of the E3-E6 comparison per dataset from the campaign's own Slurm accounting.

    Every arm study of the campaign export names its sacct exports; the allocation whose job ID
    the study captured (environment.json, SLURM_JOB_ID) is counted once, allocated GPUs times
    allocation elapsed time (cm_campaigns.accounting.read_sacct), failed attempts included.
    """
    from cm_campaigns.accounting import read_sacct
    from cm_campaigns.common import confined
    study = json.loads((campaign / "study.json").read_text(encoding="utf-8"))
    hours, rows, seen = {}, [], set()
    for item in study.get("studies") or []:
        environment = json.loads((campaign / item["study"] / "environment.json").read_text(encoding="utf-8"))
        job = str((environment.get("slurm") or {}).get("SLURM_JOB_ID") or "")
        if not job:
            raise DisplayInputError(f"{item['study']}: no captured Slurm job ID")
        allocations = {}
        for name in item.get("accounting") or []:
            allocations.update(read_sacct(confined(campaign, name))["allocations"])
        matching = [(key, row) for key, row in allocations.items()
                    if job in {str(row["job_id"]), str(row["job_id_raw"])}]
        if not matching:
            raise DisplayInputError(f"{item['study']}: job {job} is absent from its scheduler export")
        for key, row in matching:
            if key in seen:
                continue
            seen.add(key)
            hours[item["dataset"]] = hours.get(item["dataset"], 0.0) + row["gpu_hours"]
            rows.append([item["dataset"], item["study"], row["job_id"], row["allocated_gpus"], row["seconds"],
                         row["gpu_hours"], row["state"]])
    if not hours:
        raise DisplayInputError("the E3-E6 campaign export lists no arm study with scheduler accounting")
    return hours, rows


@display("supplementary:Table S41", stem="S41", title="Qualification accounting", needs=("runs", "campaign:budget"))
def table_s41(ctx, out):
    measured, source, devices = {}, [], set()
    comparison, allocations = comparison_gpu_hours(ctx.require_campaign("budget"))
    for dataset in DATASETS:
        view = ctx.require_run(dataset)
        split = [stage_hours(m) for m in view.members()]
        deployed = any(rec.deployed != rec.qc for rec in records(view))
        # The ten split runs together qualify the Roman-empire deployment.
        summed = {stage: total(s[stage] for s in split) for stage, _, _ in QUALIFICATION_STAGES}
        summed["total"] = total(summed[stage] for stage, _, _ in QUALIFICATION_STAGES)
        summed["seed"] = mean(h for s in split for h in s["seed"])
        devices.update(s["gpu"] for s in split)
        if deployed:
            measured[dataset] = summed
        for index, s in enumerate(split):
            source.append([dataset, index if view.is_campaign else "", deployed, s["device"],
                           *(s[stage] for stage, _, _ in QUALIFICATION_STAGES), s["total"], mean(s["seed"]),
                           len(s["seed"])])
        out.uses("run", dataset, ["stages/*.json", "train/manifest.json", "environment.json", "select/decision.json",
                                  "admit/decision.json"])
    listed = list(measured)
    unit = "GPU-hours" if devices == {True} else "device-hours"

    def per_dataset(key, fmt=lambda v: num(v, 2)):
        return "; ".join(f"{SHORT[d]} {fmt(measured[d][key]) or '—'}" for d in listed) or None

    def record(stage):
        """Share of the stage in the summed totals of the listed datasets that timed every stage."""
        timed = [d for d in listed if measured[d]["total"]]
        if not timed:
            return None
        return f"share {sum(measured[d][stage] for d in timed) / sum(measured[d]['total'] for d in timed):.2f}"

    # Per-seed hours at two significant digits, the precision of Table S5's time per seed.
    seed = per_dataset("seed", lambda v: None if v is None else f"{v:.2g}")
    rows = [["backbone training (shared by every route)", "per checkpoint; per ensemble of E",
             f"per seed: {seed} h (Table S5); ensembles of E checkpoints" if seed else None, "training logs"]]
    for stage, label, _ in QUALIFICATION_STAGES[:4]:
        rows.append([label, "entire search per dataset", per_dataset(stage), record(stage)])
    rows.append(["independent E3–E6 comparison", "entire comparison per dataset",
                 "; ".join(f"{SHORT[d]} {num(comparison[d], 2)}" for d in DATASETS if d in comparison) or None,
                 "comparison trace on D_compare; scheduler accounting of the arm jobs"])
    rows.append(["admission evaluation", "entire search per dataset", per_dataset("admit"), record("admit")])
    rows.append(["search (horizon and beam)", "per deployment, per input",
                 "charged in the serial-cost ledger (Table S23), not in qualification", "timing traces"])
    rows.append(["total qualification increment of the deployed rule (the E3–E6 comparison is a separate campaign "
                 "cost)", "entire search per dataset", per_dataset("total"), "sum of the five stage components"])
    for d in listed:
        out.compare(GROUP["S41"], (d, "backbone_training_GPU_h_per_seed"), measured[d]["seed"])
        for stage, _, key in QUALIFICATION_STAGES:
            out.compare(GROUP["S41"], (d, key), measured[d][stage])
        out.compare(GROUP["S41"], (d, "total_qualification_increment_GPU_h"), measured[d]["total"])
    for d, value in comparison.items():
        out.compare(GROUP["S41"], (d, "independent_E3_E6_comparison_GPU_h"), value)
    out.table(["component", "unit", unit, "record"], rows,
              notes=[f"{unit}: measured stage wall time (stages/<stage>.json, seconds_this_attempt) on the run's one "
                     "device; backbone training: mean per plant checkpoint (train/manifest.json). Listed datasets "
                     "deploy a rule other than q_C; a comparator deployment has no qualification increment and stays "
                     "in the source data. A stage whose completion marker was rewritten after an interruption has no "
                     "timing and prints as —, as does its total. Share: the component's part of the summed totals of "
                     "the listed datasets. The E3–E6 comparison is a separate campaign: its GPU-hours are the "
                     "Slurm allocations of its arm jobs (allocated GPUs times elapsed time, failed attempts "
                     "included).",
                     "Roman-empire: the ten split runs summed; per-checkpoint training averaged over all splits."])
    out.data("comparison_allocations", ["dataset", "study", "job_id", "allocated_gpus", "elapsed_seconds",
                                        "gpu_hours", "state"], allocations)
    out.uses("campaign", "budget", ["study.json", "*/environment.json", "scheduler exports named by study.json"])
    out.data("stage_hours", ["dataset", "split", "deployed_rule_differs_from_qC", "device",
                             *(f"{stage}_hours" for stage, _, _ in QUALIFICATION_STAGES), "total_hours",
                             "backbone_hours_per_checkpoint", "checkpoints"], source)


# ---------------------------------------------------------------------------
# Supplementary Table S48: policy outcomes and admission criterion
# ---------------------------------------------------------------------------

def outcome(rec: RunRecord, fallback) -> dict:
    """Test-stage paired outcome, expected work (main Eq. (32)) and admission scope of one run."""
    qc, qd = rec.qc, rec.deployed
    c_qc, c_qd = rec.candidates[qc]["cost"], rec.candidates[qd]["cost"]
    if qd == qc:
        expected = c_qc
    else:
        if fallback is None:
            raise DisplayInputError("the input_gate fallback probability is required for a controller deployment")
        expected = (1 - fallback) * c_qd + fallback * c_qc
    loss = rec.summary["paired_qD_minus_qC_loss"]
    metric = rec.summary["paired_qD_minus_qC_metric_pp"]
    eb = rec.admission.get("empirical_bernstein") or {}
    return {"loss": loss["mean"], "loss_low": loss.get("lower"), "loss_high": loss.get("upper"),
            "metric_pp": metric["mean"], "expected": expected, "assure": c_qd + c_qc if qd != qc else c_qc,
            "C_qC": c_qc, "C_qD": c_qd, "G": rec.admission["Gadmit"], "units": rec.admission["loss"]["units"],
            "J": eb.get("units"), "already": rec.admission.get("candidate_already_comparator"),
            "rule": "qC returned" if qd == qc else f"{rec.label(qd)}, E = {rec.candidates[qd]['ensemble']}",
            "qd_route": rec.candidates[qd]["route"], "E": rec.candidates[qd]["ensemble"]}


def admission_text(dataset, view, outcomes) -> str:
    first = outcomes[0]
    if all(o["already"] for o in outcomes):
        return "comparator returned"
    if independent(dataset):
        return f"Theorem 2, J = {first['J']}; Gadmit = {first['G']}"
    admitted = sum(o["G"] for o in outcomes)
    if view.is_campaign:
        text = f"descriptive, {len(outcomes)} official splits"
        return text if admitted == len(outcomes) else f"{text}; Gadmit = 1 in {admitted}"
    return f"descriptive, {first['units']} buffered cores" + ("" if first["G"] else "; Gadmit = 0")


def dataset_outcome(ctx, dataset, fallback_records) -> dict:
    view = ctx.require_run(dataset)
    recs = records(view)
    fallback = None
    if any(rec.deployed != rec.qc for rec in recs):
        fallback = fallback_records.require("fallback_probability", dataset=dataset)
    split = [outcome(rec, fallback) for rec in recs]
    summary = {key: mean(o[key] for o in split) for key in ("loss", "metric_pp", "expected", "assure", "C_qC", "C_qD")}
    if view.is_campaign:
        from paper.build.evidence import _t_interval
        spread = _t_interval([o["loss"] for o in split])
        summary.update(loss_low=spread["lower"], loss_high=spread["upper"])
    else:
        summary.update(loss_low=split[0]["loss_low"], loss_high=split[0]["loss_high"])
    summary.update(rule=tally([o["rule"] for o in split]), admission=admission_text(dataset, view, split),
                   G=min(o["G"] for o in split), route=tally([o["qd_route"] for o in split]),
                   deployed=any(rec.deployed != rec.qc for rec in recs), E=split[0]["E"], fallback=fallback,
                   split=split, recs=recs)
    return summary


def metric_text(dataset, value) -> str | None:
    if value is None:
        return None
    _, metric = task_labels(dataset)
    if value == 0:
        return "0"
    if metric == "ap":                                     # native AP, not AP percentage points (C18)
        return f"{value / 100:+.4f} AP"
    return f"{value:+.2f} {METRIC_UNITS[metric]}"


@display("supplementary:Table S48", stem="S48", title="Policy outcomes and admission criterion",
         needs=("runs", "records:input_gate"))
def table_s48(ctx, out):
    gate = ctx.require_records("input_gate")
    rows, source = [], []
    for dataset in DATASETS:
        s = dataset_outcome(ctx, dataset, gate)
        loss_label, metric = task_labels(dataset)
        loss_text = "0" if s["loss"] == 0 else f"{num(s['loss'], 5)} {loss_label}"
        rows.append([dataset, s["rule"], loss_text, metric_text(dataset, s["metric_pp"]), num(s["expected"], 2),
                     s["admission"]])
        for index, o in enumerate(s["split"]):
            source.append([dataset, index if len(s["split"]) > 1 else "", o["rule"], o["loss"], o["metric_pp"],
                           o["C_qC"], o["C_qD"], s["fallback"], o["expected"], o["G"], o["units"], o["J"]])
        out.compare(GROUP["T1"], (dataset, "loss_effect_nats"), s["loss"])
        if metric == "ap":
            out.compare(GROUP["T1"], (dataset, "official_gain_native"), s["metric_pp"] / 100)
        else:
            out.compare(GROUP["T1"], (dataset, "official_gain_pp"), s["metric_pp"])
        out.compare(GROUP["T1"], (dataset, "G_admit"), s["G"])
        out.compare(GROUP["T2"], (dataset, "C_expected"), s["expected"])
        out.uses("run", dataset, ["final/results_summary.json", "select/candidates.json", "select/decision.json",
                                  "admit/decision.json"])
    out.uses("records", "input_gate", sorted({row["source_file"] for row in gate.rows}))
    out.table(["Dataset", "evaluated rule", "paired loss over qC", "official metric change", "expected work",
               "admission"], rows,
              notes=["Paired test effects of the deployed rule over q_C (final/results_summary.json; AP in native "
                     "units). Expected work: main Eq. (32), (1 - p_fallback) C(q_D) + p_fallback C(q_C), with the "
                     "input_gate fallback probability and the candidate costs of select/candidates.json.",
                     "Roman-empire: means over the ten splits."])
    out.data("outcomes", ["dataset", "split", "evaluated_rule", "paired_loss", "metric_change_pp", "C_qC", "C_qD",
                          "fallback_probability", "expected_work", "G_admit", "admission_units", "J"], source)


# ---------------------------------------------------------------------------
# Supplementary Figure S9: mechanism contrasts and confirmation-size sensitivity
# ---------------------------------------------------------------------------

@display("supplementary:Figure S9", stem="FigS9", title="Mechanism contrasts and confirmation-size sensitivity",
         needs=("campaign:confirmation", "records:confirmation_sensitivity"))
def figure_s9(ctx, out):
    """Grey: the nine contrasts at their original sizes; blue: the multigraph contrasts rescaled to D_confirm.

    The rescaling is the two-way variance model of Note S24 (Table S14b): only the graph
    share g of the variance scales with the graph count, so each side of the J_old
    interval widens by [(1 - g) + g J_old/J_new]^(1/2) about the fixed point estimate.
    """
    from cm_campaigns.confirmation import FAMILY
    checked, _ = joint_confirmation(ctx)
    confirmation_scope(out, checked)
    sizes = ctx.require_records("confirmation_sensitivity")
    by_id = {row["contrast_id"]: row for row in checked["rows"]}
    data, changed, unchanged = [], 0, 0
    for key in FAMILY:
        row = by_id[key]
        j_new, observed = row["unit_count"], (row["effect_nats"], *row["interval_nats"])
        if sizes.select(contrast=key):
            j_old = int(sizes.require("units_old", contrast=key))
            effect = sizes.require("effect_nats", contrast=key)
            low, high = sizes.require("ci_low_nats", contrast=key), sizes.require("ci_high_nats", contrast=key)
            share = sizes.require("graph_variance_share", contrast=key)
            factor = math.sqrt((1.0 - share) + share * j_old / j_new)
            old = (effect, low, high)
            new = (effect, effect - (effect - low) * factor, effect + (high - effect) * factor)
        else:
            j_old, share, old, new = j_new, None, observed, observed
        moved = j_old != j_new
        holm = row["holm_p_value_zero"]
        note = (f"J: {j_old} → {j_new}; Holm P {p_text(holm, 1e-3)}" if moved else f"unit unchanged (n={j_new})")
        data.append([key, row["dataset"], j_old, j_new, *old, *new, share, *observed,
                     row["beneficial_direction"] * row["practical_margin_nats"], holm, note])
        if moved:
            for field, value in (("J_old", j_old), ("J_new", j_new), ("effect", old[0]),
                                 ("interval_old", list(old[1:])), ("interval", list(new[1:])),
                                 ("graph_variance_share", share), ("holm_P", p_compare(holm, 1e-3))):
                out.compare(GROUP["S14b"], (changed, field), value)
            changed += 1
        else:
            for field, value in (("J_old", j_old), ("J_new", j_new), ("effect", old[0]), ("interval", list(old[1:]))):
                out.compare(GROUP["S14b_unchanged"], (unchanged, field), value)
            unchanged += 1
    out.data("panel_a", ["contrast", "dataset", "units_old", "units_new", "effect_old_nats", "low_old_nats",
                         "high_old_nats", "effect_new_nats", "low_new_nats", "high_new_nats", "graph_variance_share",
                         "campaign_effect_nats", "campaign_low_nats", "campaign_high_nats", "signed_margin_nats",
                         "joint_holm_P", "annotation"], data)
    out.uses("campaign", "confirmation", ["campaign.json", "raw/*/test.json"])
    out.uses("records", "confirmation_sensitivity", sorted({r["source_file"] for r in sizes.rows}))
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for y, row in enumerate(data):
        moved = row[2] != row[3]
        ax.plot(row[5:7], [y, y], color="0.35", lw=1.2)
        ax.plot(row[4], y, "o", color="0.25", ms=4.5)
        if moved:
            ax.plot(row[8:10], [y + 0.22, y + 0.22], color="#2864a5", lw=1.6)
            ax.plot(row[7], y + 0.22, "o", color="#2864a5", ms=4.5)
        ax.plot(row[14], y, "|", color="#2a9d6a", ms=11, mew=1.6)
        ax.text(1.02, y, row[16], transform=ax.get_yaxis_transform(), va="center", fontsize=7,
                color="#2864a5" if moved else "0.3")
    ax.axvline(0, color="0.2", lw=0.8)
    ax.set_yticks(range(len(data)), [row[0] for row in data])
    ax.invert_yaxis()
    ax.set_xlabel("contrast (nats); green tick: practical margin of Table S11")
    ax.set_title("Held-out mechanism contrasts and confirmation-size sensitivity")
    ax.plot([], [], "o-", color="0.25", label="contrasts at J_old (95% intervals)")
    ax.plot([], [], "o-", color="#2864a5", label="rescaled to the D_confirm sizes (Table S14b)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False, ncol=2)
    ax.grid(axis="x", alpha=0.2)
    fig.subplots_adjust(right=0.72)
    out.figure(fig)
    out.note("J_old side and graph-variance share: confirmation_sensitivity record set (multigraph contrasts); the "
             "other contrasts keep their unit and are drawn from the joint confirmation campaign, whose joint Holm P "
             "labels the rescaled rows. The campaign's own D_confirm intervals are kept in the source data.")


# ---------------------------------------------------------------------------
# Supplementary Figure S13: motivating settings and schedule adequacy
# ---------------------------------------------------------------------------

@display("supplementary:Figure S13", stem="FigS13", title="Motivating settings and schedule adequacy",
         needs=("runs",))
def figure_s13(ctx, out):
    measured = []
    for dataset in DATASETS:
        view = ctx.require_run(dataset)
        recs = records(view)
        split = [rec.summary["paired_qD_minus_qC_loss"] for rec in recs]
        if view.is_campaign:
            from paper.build.evidence import _t_interval
            spread = _t_interval([s["mean"] for s in split])
            effect, low, high, method = spread["mean"], spread["lower"], spread["upper"], spread["method"]
        else:
            effect, low, high, method = split[0]["mean"], split[0]["lower"], split[0]["upper"], split[0]["method"]
        route = tally(["qC" if rec.deployed == rec.qc else rec.candidates[rec.deployed]["route"] for rec in recs])
        category = {"qC": "open-loop schedule retained", "radius": "uncertainty-radius feedback",
                    "one_step": "one-step feedback", "mpc_rh": "receding-horizon replanning"}.get(route, route)
        rule = tally(["qC returned" if rec.deployed == rec.qc else
                      f"{rec.label(rec.deployed)}, E={rec.candidates[rec.deployed]['ensemble']}" for rec in recs])
        measured.append([dataset, rule, category, effect, low, high, method])
        out.compare(GROUP["T1"], (dataset, "loss_effect_nats"), effect)
        out.compare(GROUP["T1"], (dataset, "loss_effect_interval_nats"),
                    None if low is None or effect == 0 else [low, high])
        out.uses("run", dataset, ["final/results_summary.json", "select/decision.json", "admit/decision.json"])
    settings = [["compiled or certified backbone", "weights, update, head and precision fixed; the adapter reads H_l "
                 "and wraps the aggregation call A_l"],
                ["shared pretrained backbone", "one frozen model reused by downstream users (the evaluated setting)"],
                ["one backbone, heterogeneous inputs", "graphs of different size and degree served by one model"]]
    depth = np.linspace(0.0, 1.0, 101)
    curves = []
    for family, minima in (("schedule adequate", (0.5, 0.5, 0.5)), ("feedback valuable", (0.25, 0.5, 0.78))):
        for k, centre in enumerate(minima):
            curves += [[family, k, float(d), float(0.4 + 0.15 * k + 2.0 * (d - centre) ** 2)] for d in depth]
    out.data("panel_a", ["setting", "description"], settings)
    out.data("panel_b", ["family", "curve", "relative_depth", "risk"], curves)
    out.data("panel_c", ["dataset", "evaluated_rule", "category", "paired_loss_nats", "low_nats", "high_nats",
                         "interval_method"], measured)
    plt = pyplot()
    fig = plt.figure(figsize=(11.0, 4.0))
    grid = fig.add_gridspec(2, 3, width_ratios=(1.25, 0.75, 1.15), wspace=0.45, hspace=0.6)
    ax = fig.add_subplot(grid[:, 0])
    ax.axis("off")
    ax.set_title("a  Deployment settings in which the backbone stays frozen", loc="left")
    for k, (name, text) in enumerate(settings):
        y = 0.83 - 0.33 * k
        ax.text(0.0, y, "frozen\nbackbone", transform=ax.transAxes, fontsize=6.5, va="center", ha="left",
                bbox={"boxstyle": "round", "fc": "#f7d9c4", "ec": "#c46a2c"})
        ax.text(0.2, y, "adapter", transform=ax.transAxes, fontsize=6.5, va="center", ha="left",
                bbox={"boxstyle": "round", "fc": "#d6e4f2", "ec": "#2864a5"})
        ax.text(0.37, y + 0.07, name, transform=ax.transAxes, fontsize=7, fontweight="bold", va="center")
        ax.text(0.37, y - 0.05, wrap(text, 46), transform=ax.transAxes, fontsize=6, va="center")
    for row, family in enumerate(("schedule adequate", "feedback valuable")):
        ax = fig.add_subplot(grid[row, 1])
        for k in range(3):
            values = [c[3] for c in curves if c[0] == family and c[1] == k]
            ax.plot(depth, values, color="#2864a5" if row else "0.4", lw=1)
        ax.set_title(f"{'b  ' if row == 0 else ''}{family} (illustration)", loc="left", fontsize=7)
        ax.set_yticks([])
        ax.set_ylabel("risk", fontsize=7)
        if row:
            ax.set_xlabel("propagation depth (relative)")
    ax = fig.add_subplot(grid[:, 2])
    colors = {"open-loop schedule retained": "#d37a18", "uncertainty-radius feedback": "#2864a5",
              "receding-horizon replanning": "#2a9d6a"}
    for y, row in enumerate(measured):
        color = colors.get(row[2], "0.4")
        if row[4] is not None and row[5] is not None:
            ax.plot([row[4], row[5]], [y, y], color=color, lw=1.2)
        ax.plot(row[3], y, "o", color=color, ms=4)
    for name, color in colors.items():
        ax.plot([], [], "o", color=color, label=name)
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_yticks(range(len(measured)), [f"{SHORT[r[0]]}\n{r[1]}" for r in measured], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("paired CE or BCE effect vs q_C on D_test (nats, 95%)")
    ax.set_title("c  Measured: deployed rule versus q_C", loc="left")
    ax.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    out.figure(fig)
    out.note("Panels a and b are illustrations (settings and stylized risk-depth curves); panel c is measured from the "
             "runs. Roman-empire: mean over the ten splits with the small-sample t interval.")


# ---------------------------------------------------------------------------
# Supplementary Figure S14: depth diagnostics
# ---------------------------------------------------------------------------

@display("supplementary:Figure S14", stem="FigS14", title="Depth diagnostics",
         needs=("runs", "records:depth_diagnostics"))
def figure_s14(ctx, out):
    rec = ctx.require_records("depth_diagnostics")
    alignment = rec.select("source_alignment_change")
    references = list(dict.fromkeys(row["dataset"] for row in alignment))
    if len(references) != 1:
        raise DisplayInputError("depth_diagnostics must name exactly one reference plant for source alignment")
    reference = references[0]
    initial = ctx.require_run(reference).members()[0].json("train/entry_initial.json")
    grid, risk = initial["depth_grid"], initial["depth_loss"]
    first = next(i for i, d in enumerate(grid) if d >= 2)
    best = grid[int(np.argmin(risk))]
    panel_a = [[reference, d, risk[i] - risk[first], None] for i, d in enumerate(grid) if i >= first]
    for row in alignment:
        panel_a.append([reference, int(float(row["depth"])), None, row["value"]])
    intervention = rec.value("first_intervention_depth_median", dataset=reference)
    policy = [[r["dataset"], int(float(r["depth"])), r["value"]] for r in rec.select("policy_loss_change")]
    depths = []
    for dataset in DATASETS:
        values = {q: rec.value(q, dataset=dataset) for q in ("first_intervention_depth_q25",
                                                              "first_intervention_depth_median",
                                                              "first_intervention_depth_q75",
                                                              "supported_deterioration_depth")}
        if any(v is not None for v in values.values()):
            depths.append([dataset, *values.values()])
    transport = [[r["dataset"], r["phase"], r["value"]] for r in rec.select("transport_fraction")]
    out.data("panel_a", ["dataset", "depth", "validation_loss_change", "source_alignment_change"], panel_a)
    out.data("panel_a_markers", ["dataset", "best_validation_loss_depth", "first_major_intervention_depth"],
             [[reference, best, intervention]])
    out.data("panel_b", ["dataset", "depth", "policy_loss_change"], policy)
    out.data("panel_c", ["dataset", "first_intervention_q25", "first_intervention_median", "first_intervention_q75",
                         "supported_deterioration_depth"], depths)
    out.data("panel_d", ["dataset", "phase", "transport_fraction"], transport)
    out.uses("run", reference, ["train/entry_initial.json"])
    out.uses("records", "depth_diagnostics", sorted({r["source_file"] for r in rec.rows}))
    plt = pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.2))
    ax = axes[0, 0]
    loss_rows = [r for r in panel_a if r[2] is not None]
    align_rows = sorted((r for r in panel_a if r[3] is not None), key=lambda r: r[1])
    ax.plot([r[1] for r in loss_rows], [r[2] for r in loss_rows], "o-", ms=3, label="validation loss change")
    ax.plot([r[1] for r in align_rows], [r[3] for r in align_rows], "o-", ms=3, label="source-alignment change")
    ax.axvline(best, ls="--", color="0.4", lw=0.8)
    if intervention is not None:
        ax.axvline(intervention, ls=":", color="0.4", lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("propagation depth")
    ax.set_ylabel(f"change from depth {grid[first]}")
    ax.set_title(f"a  Development plot ({SHORT.get(reference, reference)})", loc="left")
    ax.legend(fontsize=6, frameon=False)
    ax = axes[0, 1]
    for dataset in dict.fromkeys(r[0] for r in policy):
        series = sorted((r for r in policy if r[0] == dataset), key=lambda r: r[1])
        ax.plot([r[1] for r in series], [r[2] for r in series], "o-", ms=3, label=SHORT.get(dataset, dataset))
    ax.set_xscale("log", base=2)
    ax.set_xlabel("propagation depth")
    ax.set_ylabel("paired loss change (nats)")
    ax.set_title("b  Selected policies against depth", loc="left")
    ax.legend(fontsize=6, frameon=False)
    ax = axes[1, 0]
    for y, row in enumerate(depths):
        if row[1] is not None and row[3] is not None:
            ax.plot([row[1], row[3]], [y, y], color="#2864a5", lw=1.2)
        if row[2] is not None:
            ax.plot(row[2], y, "o", color="#2864a5", ms=4)
        if row[4] is not None:
            ax.plot(row[4], y, "x", color="0.3", ms=5)
    ax.plot([], [], "x", color="0.3", label="supported deterioration")
    ax.plot([], [], "o-", color="#2864a5", label="first intervention (IQR)")
    ax.set_yticks(range(len(depths)), [SHORT.get(r[0], r[0]) for r in depths])
    ax.invert_yaxis()
    ax.set_xlabel("depth")
    ax.set_title("c  Intervention precedes supported deterioration", loc="left")
    ax.legend(fontsize=6, frameon=False)
    ax = axes[1, 1]
    phases = list(dict.fromkeys(r[1] for r in transport))
    names = list(dict.fromkeys(r[0] for r in transport))
    width = 0.8 / max(len(phases), 1)
    for k, phase in enumerate(phases):
        values = {r[0]: r[2] for r in transport if r[1] == phase}
        ax.bar(np.arange(len(names)) + k * width, [values.get(n, 0.0) for n in names], width, label=phase)
    ax.set_xticks(np.arange(len(names)) + width * (len(phases) - 1) / 2, [SHORT.get(n, n) for n in names],
                  rotation=30, fontsize=7)
    ax.set_ylabel("non-nominal transport fraction")
    ax.set_title("d  Transport allocation across depth phases", loc="left")
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    out.figure(fig)


# ---------------------------------------------------------------------------
# Supplementary Figure S17: analytical and statistical audits
# ---------------------------------------------------------------------------

def effect_coded(cells: dict, factors: tuple[str, ...], joint: dict | None = None) -> list[list]:
    """Effect-coded point contrasts and, only when supplied, the paired joint intervals."""
    out = []
    for mask in (1, 2, 4, 3, 5, 6, 7):
        chosen = [i for i in range(3) if mask & (1 << i)]
        signs = {cell: math.prod(1 if cell[i] == "1" else -1 for i in chosen) for cell in cells}
        effect = math.fsum(signs[c] * v["mean"] for c, v in cells.items()) / 4
        interval = joint["contrasts"][tuple(chosen)] if joint else {}
        out.append(["".join(factors[i] for i in chosen), effect, interval.get("low"), interval.get("high")])
    return out


def independent_factorial_illustration(cells: dict, factors: tuple[str, ...]) -> list[list]:
    """Figure S17d's explicitly labelled independent-cell approximation.

    This is an analytical illustration, not a reconstruction of paired contrast
    uncertainty. Each of the eight effect-coded coefficients has magnitude 1/4;
    under independence and a common interval multiplier the propagated 95%
    half-width is sqrt(sum(cell_half_width**2)) / 4. Tables S16/S16b and the
    optional joint source data below retain the actual shared-resampling path.
    """
    from .factorial_data import CELLS
    if set(cells) != set(CELLS) or len(factors) != 3:
        raise DisplayValidationError("Figure S17d requires the complete three-factor eight-cell design")
    widths = []
    for cell in CELLS:
        row = cells[cell]
        try:
            values = [float(row[key]) for key in ("mean", "low", "high")]
        except (KeyError, TypeError, ValueError) as exc:
            raise DisplayValidationError(f"Figure S17d cell {cell} lacks its marginal mean/interval") from exc
        if not all(math.isfinite(value) for value in values) or values[1] > values[2]:
            raise DisplayValidationError(f"Figure S17d cell {cell} has an invalid marginal interval")
        widths.append((values[2] - values[1]) / 2)
    half_width = math.sqrt(math.fsum(width * width for width in widths)) / 4
    return [[name, effect, effect - half_width, effect + half_width]
            for name, effect, _, _ in effect_coded(cells, factors)]


@display("supplementary:Figure S17", stem="FigS17", title="Analytical and statistical audits",
         needs=("runs", "records:factorials", "records:input_gate", "records:action_gap_audit",
                "records:tube_audit"))
def figure_s17(ctx, out):
    from cm_theory import admission as theory
    regret = ctx.constants.get("regret_bound_table_S28b")
    if not regret or "sigma1_grid" not in regret:
        raise DisplayInputError("registered constant regret_bound_table_S28b.sigma1_grid (the sigma_1 grid of "
                                "Supplementary Table S28b) is required for panel g")
    sigma_grid = regret["sigma1_grid"]
    gate = ctx.require_records("input_gate")
    outcomes = {d: dataset_outcome(ctx, d, gate) for d in DATASETS}
    for d in DATASETS:
        out.uses("run", d, ["select/decision.json", "select/candidates.json", "select/entry_complete.json",
                            "admit/decision.json", "admit/paired_units.npz", "final/results_summary.json",
                            "fit/full/audit.json"])
    plt = pyplot()
    fig, axes = plt.subplots(3, 3, figsize=(11.0, 10.5))
    axes[2, 2].axis("off")

    # a: empirical-Bernstein penalty against J
    ax = axes[0, 0]
    J = np.unique(np.round(np.logspace(1, 4.5, 120)).astype(int))
    panel, markers = [], []
    for dataset in GRAPH_LEVEL:
        view = ctx.require_run(dataset)
        base = admission_rows(view, [])[0]
        ac = RunRecord(view).config["admission"]
        curve = [theory.bernstein_terms(base["S2"], int(j), ac["clip"], ac["alpha"])["penalty"] for j in J]
        panel += [[dataset, int(j), p, theory.bernstein_terms(0.0, int(j), ac["clip"], ac["alpha"])["penalty"]]
                  for j, p in zip(J, curve)]
        markers.append([dataset, base["J"], base["penalty"], abs(base["D_bar"]) - ac["epsilon_a"],
                        theory.minimum_admission_units(base["D_bar"], ac["epsilon_a"], ac["clip"], ac["alpha"])])
        ax.plot(J, curve, lw=1, label=SHORT[dataset])
        ax.plot(base["J"], base["penalty"], "o", ms=4)
        out.compare(GROUP["S13"], (dataset, "penalty"), base["penalty"])
    floor = [row[3] for row in panel if row[0] == GRAPH_LEVEL[0]]
    ax.plot(J, floor, "--", color="0.3", lw=1, label="variance-free floor")
    band = [m[3] for m in markers if m[3] > 0]
    if band:
        ax.axhspan(min(band), max(band), color="#2a9d6a", alpha=0.15, label="admissible penalty |D| - eps_a")
    for dataset in DATASETS:
        if not independent(dataset):
            units = outcomes[dataset]["split"][0]["units"] if not ctx.require_run(dataset).is_campaign \
                else len(outcomes[dataset]["split"])
            ac = outcomes[dataset]["recs"][0].config["admission"]
            value = theory.bernstein_terms(0.0, int(units), ac["clip"], ac["alpha"])["penalty"] if units > 1 else None
            markers.append([dataset, units, value, None, None])
            if value is not None:
                ax.plot(units, value, "s", color="#c46a2c", ms=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("admission units J")
    ax.set_ylabel("penalty U_a - D_bar (nats)")
    ax.set_title("a  Admission penalty vs sample size", loc="left")
    ax.legend(fontsize=6, frameon=False)
    out.data("panel_a_curves", ["dataset", "J", "penalty", "variance_free_floor"], panel)
    out.data("panel_a_markers", ["dataset", "J", "penalty_at_J", "admissible_penalty", "corollary2_minimum_J"],
             markers)

    # b: realized headroom fraction
    ax = axes[0, 1]
    headroom = []
    for dataset in DATASETS:
        recs = outcomes[dataset]["recs"]
        values = []
        for rec in recs:
            oracle = rec.view.json("select/entry_complete.json")["headroom"]["upper"]
            qd = rec.deployed
            effect = 0.0 if qd == rec.qc else rec.decision["mean"][qd]
            far = 0.0 if qd == rec.qc else rec.decision["L"][qd]
            values.append((effect, far, oracle))
        effect, far, oracle = (mean(v[i] for v in values) for i in range(3))
        ratio = min(1.0, abs(effect) / abs(oracle)) if oracle else None
        conservative = min(1.0, abs(far) / abs(oracle)) if oracle else None
        headroom.append([dataset, effect, far, oracle, ratio, conservative])
    x = np.arange(len(headroom))
    ax.bar(x - 0.2, [r[4] or 0 for r in headroom], 0.4, color="#2864a5", label="point-estimate ratio")
    ax.bar(x + 0.2, [r[5] or 0 for r in headroom], 0.4, color="white", edgecolor="0.3", hatch="///",
           label="far interval end, capped at 1")
    ax.set_xticks(x, [SHORT[r[0]] for r in headroom], rotation=40, fontsize=7)
    ax.set_ylabel("upper bound on the realized fraction")
    ax.set_title("b  Realized headroom: upper bound", loc="left")
    ax.legend(fontsize=6, frameon=False)
    out.data("panel_b", ["dataset", "Delta_sel_qD", "far_interval_end", "U_Delta_oracle", "ratio",
                         "conservative_ratio"], headroom)

    # c: Theorem 1 slack
    ax = axes[0, 2]
    slack = []
    for dataset in DATASETS:
        recs = outcomes[dataset]["recs"]
        r_max = max(rec.decision["max_radius"] for rec in recs)
        eps_eq, eps_r = recs[0].decision["epsilon_eq"], recs[0].decision["epsilon_r"]
        selected = mean(0.0 if rec.provisional == rec.qc else rec.decision["mean"][rec.provisional] for rec in recs)
        slack.append([dataset, eps_eq + 2 * r_max, abs(selected), eps_r,
                      common([len(rec.decision["Q_s"]) for rec in recs]),
                      common([rec.decision["candidate_count"] for rec in recs])])
        out.compare(S20_GROUP, (dataset, "eps_eq_plus_2r_max"), eps_eq + 2 * r_max)
    for y, row in enumerate(slack):
        ax.plot([0, row[1]], [y, y], color="0.85", lw=6, solid_capstyle="butt")
        ax.plot(row[1], y, "|", color="0.1", ms=9)
        ax.plot(row[2], y, "o", color="#2864a5", ms=4)
        ax.plot(row[3], y, "D", color="#2a9d6a", ms=4)
        ax.text(row[1], y - 0.3, f" {row[4]}/{row[5]}", fontsize=6)
    ax.set_yticks(range(len(slack)), [SHORT[r[0]] for r in slack])
    ax.invert_yaxis()
    ax.set_xlabel("nats")
    ax.set_title("c  Selection slack (Theorem 1)", loc="left")
    ax.plot([], [], "|", color="0.1", label="slack eps_eq + 2 r_max")
    ax.plot([], [], "o", color="#2864a5", label="selected |Delta_s|")
    ax.plot([], [], "D", color="#2a9d6a", label="margin eps_R")
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    out.data("panel_c", ["dataset", "slack", "selected_abs_Delta_s", "epsilon_R", "Q_s_size", "Q_size"], slack)

    # d: the printed independence illustration. Actual paired uncertainty is
    # preserved separately and never replaced by this approximation in S16/S16b.
    ax = axes[1, 0]
    factorials = ctx.require_records("factorials")
    cells = {}
    for cell in ("000", "100", "010", "001", "110", "101", "011", "111"):
        cells[cell] = {"mean": factorials.require("ld_integrated_loss", dataset="ogbn-arxiv", cell=cell),
                       "low": factorials.require("interval_low", dataset="ogbn-arxiv", cell=cell),
                       "high": factorials.require("interval_high", dataset="ogbn-arxiv", cell=cell)}
    from .factorial_data import joint_factorial_contrasts
    joint = joint_factorial_contrasts(factorials, "ogbn-arxiv", {
        cell: {"ld_integrated_loss": row["mean"], "interval_low": row["low"], "interval_high": row["high"]}
        for cell, row in cells.items()})
    contrasts = independent_factorial_illustration(cells, ("A", "O", "X"))
    illustration_method = "independence approximation (printed Figure S17d illustration)"
    out.provenance(panel_d_interval_method={
        "method": illustration_method,
        "source": {"document": "supplementary", "page": 36, "display": "Figure S17d"},
        "formula": "h = sqrt(sum(((interval_high - interval_low) / 2)^2)) / 4",
        "assumption": "zero cross-cell covariance and a common 95% interval multiplier",
        "scope": "analytical illustration; does not replace paired inferential intervals in Tables S16/S16b",
        "full_figure_comparison": False,
    })
    if joint is None:
        out.provenance(factorial_joint={"available": False,
            "reason": "No recognized paired factorial source was supplied; joint uncertainty is not inferred from marginal intervals"})
    else:
        joint_rows = effect_coded(cells, ("A", "O", "X"), joint)
        companion = out.data("panel_d_paired_joint",
            ["contrast", "effect_coded_nats", "low_nats", "high_nats", "interval_method"],
            [[*row, joint["method"]] for row in joint_rows])
        out.provenance(factorial_joint={"available": True, "companion_source_data": companion,
            **{k: v for k, v in joint.items() if k != "contrasts"}})
    colors = ["#2864a5"] * 3 + ["#2a9d6a"] * 3 + ["#d37a18"]
    for k, (name, effect, low, high) in enumerate(contrasts):
        ax.bar(k, effect, color=colors[k])
        if low is not None and high is not None:
            ax.plot([k, k], [low, high], color="0.2", lw=0.8)
    ax.axhline(0, color="0.4", lw=0.6)
    ax.set_xticks(range(7), [c[0] for c in contrasts])
    ax.set_ylabel("effect on DIL CE (nats)")
    ax.set_title("d  arxiv factorial (Table S16)\nintervals: independence approximation", loc="left")
    out.data("panel_d", ["contrast", "effect_coded_nats", "low_nats", "high_nats", "interval_method"],
             [[*row, illustration_method] for row in contrasts])
    out.data("panel_d_cells", ["cell", "mean_nats", "marginal_low_nats", "marginal_high_nats", "marginal_half_width_nats"],
             [[cell, row["mean"], row["low"], row["high"], (row["high"] - row["low"]) / 2]
              for cell, row in cells.items()])
    names = {"A": "actions", "O": "observer", "X": "risk_controlled_execution"}
    for name, effect, low, high in contrasts:
        if name in names:
            out.compare(GROUP["S16"], ("effect_coded_main_effects_nats", names[name]), effect)
    out.compare(GROUP["S16"], ("effect_coded_main_effects_nats", "half_width_95"),
                contrasts[0][3] - contrasts[0][1], note=illustration_method)
    for k, (name, effect, _, _) in enumerate(contrasts[3:6]):
        out.compare(GROUP["S16"], ("pairwise_interactions_nats", k), effect)
    out.compare(GROUP["S16"], ("three_way_interaction_nats",), contrasts[6][1])

    # e: plug-in tube against the observed error
    ax = axes[1, 1]
    tube = record_rows(ctx.require_records("tube_audit"), S22_QUANTITIES)
    tube_rows = [[d, v["final_bound_nats"], v["observed_error_median_nats"], v["observed_error_p95_nats"],
                  v["tube_coverage"]] for d, v in tube.items()]
    for k, row in enumerate(tube_rows):
        ax.bar(k, row[1], color="white", edgecolor="0.2")
        ax.plot(k, row[2], "o", color="#2864a5", ms=4)
        ax.plot([k - 0.3, k + 0.3], [row[3], row[3]], color="#d37a18", lw=1.5)
        ax.text(k, row[1], f"cov. {row[4]:.2f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(range(len(tube_rows)), [SHORT[r[0]] for r in tube_rows], rotation=40, fontsize=7)
    ax.set_ylabel("loss error (nats)")
    ax.set_title("e  Plug-in tube vs observed error", loc="left")
    out.data("panel_e", ["dataset", "final_bound_nats", "observed_median_nats", "observed_p95_nats",
                         "tube_coverage"], tube_rows)

    # f: action-gap audit
    ax = axes[1, 2]
    gaps = record_rows(ctx.require_records("action_gap_audit"), S21_QUANTITIES)
    gap_rows = []
    for dataset, v in gaps.items():
        reading = gap_interpretation(ctx, dataset, v["switch_fraction"])
        gap_rows.append([dataset, 1000 * v["p5_gap_nats"], v["switch_fraction"], v["near_zero_fraction"], reading])
        ax.scatter(1000 * v["p5_gap_nats"], v["switch_fraction"], s=2000 * v["near_zero_fraction"] + 5,
                   color="#e76f51" if reading.startswith("boundary") else "#2864a5", alpha=0.8)
        ax.annotate(SHORT[dataset], (1000 * v["p5_gap_nats"], v["switch_fraction"]), fontsize=6)
    ax.set_xlabel("5th-percentile action gap (mnats)")
    ax.set_ylabel("switch fraction under perturbation")
    ax.set_title("f  Action-gap audit (Prop. 3)", loc="left")
    out.data("panel_f", ["dataset", "p5_gap_mnats", "switch_fraction", "near_zero_fraction", "interpretation"],
             gap_rows)

    # g: one-step regret radius
    ax = axes[2, 0]
    n = np.unique(np.round(np.logspace(np.log10(30), 5, 160)).astype(int))
    curves = [["Hoeffding", int(k), 2 * theory.hoeffding_radius(int(k))] for k in n]
    for sigma in (sigma_grid[0], sigma_grid[-1]):
        curves += [[f"Bernstein sigma1={sigma:g}", int(k), 2 * theory.bernstein_radius(int(k), sigma)] for k in n]
    for name in dict.fromkeys(c[0] for c in curves):
        ax.plot([c[1] for c in curves if c[0] == name], [c[2] for c in curves if c[0] == name], lw=1, label=name)
    ess, rmse, effects = [], [], []
    for dataset in DATASETS:
        for rec in outcomes[dataset]["recs"]:
            audit = rec.view.json("fit/full/audit.json")
            qualified = [c["ess"] for c in audit["support"]["cells"].values() if c["qualified"]]
            if qualified:
                ess.append([dataset, min(qualified), 2 * theory.hoeffding_radius(max(1, int(min(qualified))))])
            rmse.append(audit["oof_rmse"])
        if outcomes[dataset]["deployed"]:
            effects.append(abs(outcomes[dataset]["loss"]))
    for row in ess:
        ax.plot(row[1], row[2], "o", color="#e76f51", ms=3)
    if rmse:
        ax.axhspan(min(rmse), max(rmse), color="#2864a5", alpha=0.12, label="critic held-out RMSE")
    needed = []
    if effects:
        smallest = min(effects)
        ax.axhspan(min(effects), max(effects), color="#2a9d6a", alpha=0.15, label="admitted effects")
        needed = [["Hoeffding", smallest, theory.units_for_hoeffding_regret(smallest)]]
        needed += [[f"Bernstein sigma1={s:g}", smallest, theory.units_for_bernstein_regret(smallest, s)]
                   for s in sigma_grid]
        for row in needed:
            ax.axvline(row[2], ls=":", color="0.4", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("independent logged units per (stratum, action)")
    ax.set_ylabel("regret bound 2e(n) (nats)")
    ax.set_title("g  One-step regret radius (Prop. 7)", loc="left")
    ax.legend(fontsize=5, frameon=False)
    out.data("panel_g_curves", ["bound", "n", "regret_bound_nats"], curves)
    out.data("panel_g_reference_ess", ["dataset", "minimum_qualified_cell_ESS", "hoeffding_bound_nats"], ess)
    out.data("panel_g_units_for_smallest_effect", ["bound", "target_nats", "units"], needed)
    out.data("panel_g_bands", ["band", "low", "high"],
             [["critic_held_out_rmse", min(rmse) if rmse else None, max(rmse) if rmse else None],
              ["admitted_effects", min(effects) if effects else None, max(effects) if effects else None]])

    # h: equal-compute break-even under the ensemble law (main Proposition 5)
    ax = axes[2, 1]
    law = ctx.constants["budget_matched_comparison_section4_table_S32_S40"]
    kappas = (law["ensemble_law_kappa_band"][0], law["kappa_midpoint"], law["ensemble_law_kappa_band"][1])
    breakeven = []
    for dataset in DATASETS:
        s = outcomes[dataset]
        if not s["deployed"]:
            continue
        plant = mean(rec.summary["models"]["qN"]["loss_nats"] for rec in s["recs"])
        for name, cost in (("expected", s["expected"]), ("assurance", s["assure"])):
            o = cost / s["C_qC"] - 1
            values = [k * plant * o / (s["E"] * (1 + o)) for k in kappas]
            breakeven.append([dataset, name, o, plant, s["E"], *values, abs(s["loss"])])
    names = list(dict.fromkeys(r[0] for r in breakeven))
    for k, dataset in enumerate(names):
        for offset, name, style in ((-0.2, "expected", {"color": "0.8"}),
                                    (0.2, "assurance", {"color": "white", "edgecolor": "0.3", "hatch": "///"})):
            row = next(r for r in breakeven if r[0] == dataset and r[1] == name)
            ax.bar(k + offset, row[6], 0.38, **style)
            ax.plot([k + offset] * 2, [row[5], row[7]], color="0.2", lw=0.8)
        measured = next(r for r in breakeven if r[0] == dataset)[8]
        ax.plot([k - 0.4, k + 0.4], [measured, measured], color="#2864a5", lw=2)
    ax.set_xticks(range(len(names)), [SHORT[d] for d in names], rotation=40, fontsize=7)
    ax.set_ylabel("nats")
    ax.set_title("h  Break-even at equal compute (Prop. 5)", loc="left")
    out.data("panel_h", ["dataset", "overhead", "o", "plant_loss_nats", "E", "threshold_kappa_low",
                         "threshold_kappa_mid", "threshold_kappa_high", "measured_abs_effect_nats"], breakeven)
    out.uses("records", "factorials", sorted({r["source_file"] for r in factorials.rows}))
    out.uses("records", "input_gate", sorted({r["source_file"] for r in gate.rows}))
    fig.tight_layout()
    out.figure(fig)
    out.note("a: cm_theory.admission bound with the S^2, J and D_bar of each multigraph admission; squares at the "
             "single-graph unit counts on the variance-free floor. b: |Delta_sel(q_D)| / |U(Delta_oracle)| from the "
             "selection record and select/entry_complete.json. d: the printed independence illustration propagates "
             "marginal cell half-widths in quadrature with 1/4 contrast coefficients; verified paired-bootstrap "
             "intervals, when supplied, remain in separate companion source data. These panel checks and the "
             "S28b sigma_1 grid do not establish numerical agreement for all eight panels. h: D_inf = kappa x l_plant with l_plant the q_N test "
             "loss and the registered kappa band; overheads from main Eq. (32) and C_assure = C(q_D) + C(q_C).")


# ---------------------------------------------------------------------------
# Supplementary Figure S19: scope of the guarantees
# ---------------------------------------------------------------------------

@display("supplementary:Figure S19", stem="FigS19", title="Scope of the guarantees, admission criteria and the "
         "supported model class", needs=("runs",))
def figure_s19(ctx, out):
    scope = ctx.constants.get("guarantee_scope_figure_S19")
    if not scope:
        raise DisplayInputError("registered constant guarantee_scope_figure_S19 (statement, class, condition of "
                                "Supplementary Figure S19a) is required for panel a")
    classes = ("exact", "bound", "conditional", "finite-sample", "empirical", "protocol")
    unknown = [row for row in scope if row["class"] not in classes]
    if unknown:
        raise DisplayInputError(f"guarantee_scope_figure_S19 uses unregistered classes: {unknown}")
    delta_s = ctx.constants["selection_eq19_algorithm_S4"]["delta_s"]
    admitted = []
    for dataset in DATASETS:
        view = ctx.require_run(dataset)
        recs = records(view)
        rec = recs[0]
        units = rec.admission["loss"]["units"]
        if independent(dataset):
            unit = f"{rec.admission['empirical_bernstein']['units']:,} i.i.d. graphs (validation split)"
            criterion = "emp. Bernstein U_a <= -eps_a (Theorem 2)"
            coverage = f">= {1 - delta_s - rec.config['admission']['alpha']:.0%}; policy fixed on D_confirm"
        elif view.is_campaign:
            unit, criterion, coverage = f"{len(recs)} official splits", "paired repeated-split interval", "none"
        else:
            unit, criterion, coverage = f"{units} buffered cores", "cluster bootstrap (descriptive)", "none"
        G = min(r.admission["Gadmit"] for r in recs)
        returned = all(r.deployed == r.qc for r in recs)
        admitted.append([SHORT[dataset], unit, criterion, coverage, f"{G} (qC)" if returned else str(G)])
        out.compare(GROUP["T1"], (dataset, "G_admit"), G)
        if independent(dataset):
            out.compare(GROUP["S13"], (dataset, "J"), rec.admission["empirical_bernstein"]["units"])
        out.uses("run", dataset, ["admit/decision.json", "select/decision.json", "config.json"])
    model = ctx.constants["model_class_definition1"]
    backbones = ctx.constants["backbones_table_S5"]
    tested = list(dict.fromkeys(backbones[d]["backbone_family"] for d in DATASETS if d in backbones))
    longest = max(len(model["requirements"]), len(tested), len(model["excluded"]))
    pad = lambda items: list(items) + [None] * (longest - len(items))
    panel_c = [list(row) for row in zip(pad(model["requirements"]), pad(tested), pad(model["excluded"]))]
    out.data("panel_a", ["statement", "class", "holds_under"], [[r["statement"], r["class"], r["holds_under"]]
                                                                for r in scope])
    out.data("panel_b", ["dataset", "unit_J_or_repeats", "criterion", "coverage", "G_admit"], admitted)
    out.data("panel_c", ["requirement", "tested", "not_covered"], panel_c)
    plt = pyplot()
    fig = plt.figure(figsize=(12.0, 8.0))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.3, 1.0), wspace=0.08, hspace=0.25)
    ax = fig.add_subplot(grid[:, 0])
    ax.axis("off")
    ax.set_title("a  The class of each statement and the condition it holds under", loc="left")
    colors = ("#2864a5", "#2864a5", "#2a9d6a", "#2a9d6a", "#d37a18", "#d37a18")
    step = 0.86 / max(len(scope), 1)
    for k, name in enumerate(classes):
        ax.text(0.36 + 0.03 * k, 0.89, name, transform=ax.transAxes, rotation=90, fontsize=6, ha="center",
                va="bottom")
    for y, row in enumerate(scope):
        top = 0.87 - (y + 0.5) * step
        index = classes.index(row["class"])
        ax.text(0.0, top, wrap(row["statement"], 34), transform=ax.transAxes, fontsize=6, va="center")
        ax.plot(0.36 + 0.03 * index, top, "o", ms=3.5, color=colors[index], transform=ax.transAxes)
        ax.text(0.55, top, wrap(row["holds_under"], 52), transform=ax.transAxes, fontsize=5.5, va="center")
    ax = fig.add_subplot(grid[0, 1])
    ax.axis("off")
    ax.set_title("b  Admission criterion and outcome per dataset", loc="left")
    widths = (8, 16, 18, 16, 8)
    table = ax.table(cellText=[[wrap(c, w) if c is not None else "" for c, w in zip(row, widths)]
                               for row in admitted],
                     colLabels=["dataset", "unit; J or repeats", "criterion", "coverage", "G_admit"],
                     colWidths=[w / sum(widths) for w in widths], loc="upper center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.scale(1, 2.3)
    ax = fig.add_subplot(grid[1, 1])
    ax.axis("off")
    ax.set_title("c  Supported model class (Definition 1)", loc="left")
    for k, (title, items, color) in enumerate((("Requirements", model["requirements"], "#d6e4f2"),
                                               ("Tested", tested, "#d8eedf"),
                                               ("Not covered", model["excluded"], "#f7d9c4"))):
        body = "\n".join(wrap(f"- {item}", 30) for item in items)
        ax.text(0.0 + 0.34 * k, 0.97, f"{title}\n{body}", transform=ax.transAxes, fontsize=5.5, va="top",
                bbox={"boxstyle": "round", "fc": color, "ec": "0.6"})
    out.figure(fig)
    out.note("a: registered statement classes (registered_constants guarantee_scope_figure_S19). b: admission unit, "
             "criterion and outcome of every run; coverage of Corollary 1 = 1 - delta_s - delta_a. c: Definition 1 and "
             "the backbone families of registered_constants (model_class_definition1, backbones_table_S5).")
