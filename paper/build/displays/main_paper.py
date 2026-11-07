"""Main-paper displays: Tables 1-2 and Figures 2-6.

Every number is read from, or computed with the released statistics and
analytical code from, the seven sealed lifecycle runs, the verified measurement
record sets and the joint S14 confirmation campaign. Registered protocol
constants (``paper/registered_constants.json``) supply margins, grids and the
dataset statistics of the analytical cost model.

Sources by display
------------------
Table 1   runs (selection, confirmation, admission, final paired test units) and
          ``input_gate`` (serving-time input-fallback fraction of the ledger).
Table 2   runs and ``input_gate``: Eq. (32) expected cost, two-branch assurance
          cost, the Table S44 tail model, Definition 2 exchange rate and
          amortization horizon (qualification hours from the run's stage records).
Figure 2  runs, ``campaign:confirmation`` (the nine-test Holm family),
          ``ensemble_grid`` (Proposition 5), ``input_gate`` and ``tube_audit``
          (Proposition 2 plug-in tube; a record contract of this module).
Figure 3  ``campaign:confirmation``, runs (the deployed rule's own confirmation
          contrasts), ``mechanism_contrasts`` and ``decision_traces`` (record
          contracts of this module).
Figure 4  ``factorials``, runs, ``input_gate`` and ``baselines``.
Figure 5  runs, ``input_gate``, ``baselines`` and ``nominal_time``.
Figure 6  runs, ``input_gate`` and ``budget_matched``.

A Roman-empire campaign of ten official splits is aggregated as the run
evidence does: paired test effects are the split mean with a small-sample t
interval; selection statistics are split means; qualification hours are summed.
"""
from __future__ import annotations

import math
import re
from statistics import NormalDist

import numpy as np

from .framework import (DATASETS, SHORT, DisplayInputError, display, integer, num, pyplot)

T1_GROUP = "main_table_1_policy_and_deployment_ledger"
T2_GROUP = "main_table_2_gain_and_cost_ledger"
S12_GROUP = "supplementary_table_S12_selection_ledger"
S12C_GROUP = "supplementary_table_S12c_selection_rule_audit"
S13_GROUP = "supplementary_table_S13_independent_graph_admission"
S14_GROUP = "supplementary_table_S14_mechanism_contrasts_submitted_protocol"
S15_GROUP = "supplementary_table_S15_method_comparison"
S16_GROUP = "supplementary_table_S16_arxiv_factorial"
S20_GROUP = "supplementary_table_S20_selection_theorem_instantiation"
S22_GROUP = "supplementary_table_S22_trajectory_tube_audit"
S23_GROUP = "supplementary_table_S23_assurance_and_expected_compute"
S23B_GROUP = "supplementary_table_S23b_nominal_execution_time_and_amortization"
S26B_GROUP = "supplementary_table_S26b_ensemble_law_and_break_even"
S41_GROUP = "supplementary_table_S41_qualification_cost_components"
S42_GROUP = "supplementary_table_S42_power_of_budget_matched_test"
S46_GROUP = "supplementary_table_S46_budget_matched_frontier"
S47_GROUP = "supplementary_table_S47_decision_boundary_and_power"

COLORS = {"ogbn-arxiv": "#2a78d6", "ogbn-proteins": "#eb6834", "ogbn-products": "#8a8984",
          "Roman-empire": "#1baf7a", "PascalVOC-SP": "#6a55d0", "COCO-SP": "#d54ea6", "Peptides-func": "#1e7b3c"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d2"
ROUTE_WORDS = {"nominal": "nominal", "best_depth": "best depth", "learned_stop": "learned stop", "phase": "phase",
               "fixed_sequence": "fixed sequence", "mpc_ol": "MPC-OL", "one_step": "one-step", "radius": "radius",
               "mpc_rh": "MPC-RH"}
FEEDBACK = ("one_step", "radius", "mpc_rh")
# Route families of the representative rows of Supplementary Table S12, in its order (row 0 is q_C).
S12_ROUTES = ("best_depth", "learned_stop", "phase", "fixed_sequence", "mpc_ol", "one_step", "radius", "mpc_rh")
# Method rows of Supplementary Table S15: the full-depth plant, the internal routes, then the two
# retrained architectures of cm_campaigns.contract.EXTERNAL in their listed order.
S15_ROUTE_ROWS = {"nominal": 0, "best_depth": 1, "phase": 2, "fixed_sequence": 3, "mpc_ol": 4, "radius": 5, "mpc_rh": 6}
METRIC_LABELS = {"accuracy": ("pp", "accuracy"), "rocauc": ("AUC pp", "ROC-AUC"), "macro_f1": ("F1 pp", "macro F1"),
                 "ap": ("AP", "AP")}
LOSS_LABELS = {"cross_entropy": "CE", "bce": "BCE", "binary_cross_entropy": "BCE"}
ORIGINS = (("schedule_nominal", "schedule/nominal"), ("state_conditioned", "state one-step/radius"),
           ("exclusive_sequence", "exclusive sequence"), ("recovery", "recovery"), ("input_fallback", "input fallback"))
ORIGIN_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#8f6fd1")
ARMS = ("E3", "E4", "E5", "E6")
ARM_LABELS = {"E3": "E3 initial-input predictor", "E4": "E4 distilled policy", "E5": "E5 replanning without observations",
              "E6": "E6 budget-aware q_B"}
ARM_MARKERS = {"E3": "s", "E4": "^", "E5": "v", "E6": "P"}

# Table S44 tail-cost model. The reference resolvent share is a scenario parameter of the
# supplementary model, not a measurement; where the measured per-member multiplier cannot carry
# that share, the largest share compatible with it (Note S19 identity at sensing share s) is used.
TAIL_REFERENCE_SHARE = 0.25
TAIL_SENSING_SHARE = 0.06
# Note S24: share of the unit variance that shrinks with the number of paired seeds in the
# conservative reading of the seed-paired E3 power (single-graph units / fixed-ensemble graph units).
SEED_SHRINKING_SHARE = {"cluster": 0.5, "official_split": 0.5, "graph": 0.0}
# Rows of Figure 5c drawn again at the two-branch assurance cost (dashed).
ASSURANCE_ROWS = ("ogbn-arxiv", "PascalVOC-SP")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _f(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _pct(fraction: float) -> str:
    """Signed whole percent of a fraction, rounded half away from zero at the printed precision."""
    value = 100.0 * fraction
    whole = int(math.floor(abs(value) + 0.5 + 1e-9))
    return f"{'-' if value < 0 and whole else '+'}{whole}%" if whole else "0%"


def _trim(text: str | None, minimum: int = 4) -> str | None:
    """Drop trailing zeros of a fixed-point text, keeping at least ``minimum`` decimals."""
    if text is None or "." not in text:
        return text
    head, tail = text.split(".")
    while len(tail) > minimum and tail.endswith("0"):
        tail = tail[:-1]
    return f"{head}.{tail}"


def _record_value(view, quantity: str, **keys) -> float | None:
    value = view.value(quantity, **keys)
    return None if value is None else float(value)


def _require_value(view, quantity: str, **keys) -> float:
    value = _record_value(view, quantity, **keys)
    if value is None or not math.isfinite(value):
        raise DisplayInputError(f"{view.name}: {quantity} {keys} is not in the record set")
    return value


def _t_interval(values: list[float]) -> dict:
    from paper.build.evidence import _t_interval as interval
    return interval(values)


def _theta(ctx) -> float:
    text = ctx.constants["deployment_economics_definition2"]["amortization"]
    match = re.search(r"theta\s*=\s*([0-9.]+)", text)
    if not match:
        raise DisplayInputError("registered constants do not state the tolerated share theta of Definition 2")
    return float(match.group(1))


def _truncation_order(ctx) -> int:
    orders = [a["T"] for a in ctx.constants["action_registry_table_S6"]["actions"] if a.get("T")]
    return int(max(orders))


def _sparse_to_dense_ratio(ctx, dataset: str) -> float:
    from cm_theory.budget import sparse_to_dense_ratio
    stats = ctx.constants["dataset_statistics_table_S6c_S23b"][dataset]
    width = ctx.constants["backbones_table_S5"][dataset]["hidden"]
    if "nodes" in stats:
        return sparse_to_dense_ratio(stats["reported_edges"], stats["nodes"], width)
    return sparse_to_dense_ratio(stats["mean_reported_edges_per_graph"], stats["mean_nodes_per_graph"], width)


def _phase_boundaries(candidate: dict) -> str | None:
    sequence = candidate.get("sequence") or []
    changes = [str(i) for i in range(1, len(sequence)) if sequence[i] != sequence[i - 1]]
    return "/".join(changes) or None


def route_identity(candidate: dict, *, style: str) -> str:
    """Printed identity of a route: ``table1`` 'MPC-OL H = 8, E = 4', ``table2`` 'MPC-OL H = 8, 4',
    ``registry`` 'MPC-OL, H = 8' (the convention of the Table 1 ledger)."""
    route = candidate["route"]
    word = ROUTE_WORDS.get(route, route)
    detail = None
    if route in ("mpc_ol", "mpc_rh") and candidate.get("horizon") is not None:
        detail = f"H = {candidate['horizon']}"
    elif route == "phase" and _phase_boundaries(candidate):
        detail = _phase_boundaries(candidate)
    elif route == "best_depth" and candidate.get("depth") is not None:
        detail = f"d = {candidate['depth']}"
    if style == "registry":
        if detail is None:
            return word
        return f"{word}, boundaries {detail}" if route == "phase" else f"{word}, {detail}"
    head = word if detail is None else f"{word} {detail}"
    return f"{head}, E = {candidate['ensemble']}" if style == "table1" else f"{head}, {candidate['ensemble']}"


# ---------------------------------------------------------------------------
# Per-dataset deployment summary (runs)
# ---------------------------------------------------------------------------

def deployment(ctx, dataset: str) -> dict:
    """Everything Tables 1-2 and Figures 2-6 read from one dataset's lifecycle run(s)."""
    view = ctx.require_run(dataset)
    evidence = view.evidence
    members = view.members()
    member_evidence = [m.evidence for m in members] if view.is_campaign else [evidence]
    selection, admission, final = evidence["selection"], evidence["admission"], evidence["final"]
    # The registered candidate records keep the route parameters (horizon, schedule) that the
    # evidence summary drops; a split campaign identifies its policies by split 0, as the evidence does.
    registry = members[0].json("select/candidates.json")

    def candidate(block):
        return None if block is None else {**registry[block["index"]], "index": block["index"]}
    qc, provisional, deployed = (candidate(selection.get("qC")), candidate(selection.get("provisional")),
                                 candidate(admission.get("deployed")))
    if qc is None or deployed is None or provisional is None:
        raise DisplayInputError(f"{dataset}: the selection or admission record does not identify q_C and the deployed rule")
    returned = admission.get("Gadmit") != 1 or deployed["index"] == qc["index"]
    metric = (evidence["task"].get("metric") or "").lower()
    if metric not in METRIC_LABELS:
        raise DisplayInputError(f"{dataset}: unrecognized official metric {metric!r}")
    loss_name = LOSS_LABELS.get(evidence["task"].get("loss_name") or "", evidence["task"].get("loss_name") or "loss")
    loss = final.get("paired_loss") or {}
    gain = final.get("paired_metric_pp") or {}
    if not returned and (loss.get("mean") is None or gain.get("mean") is None):
        raise DisplayInputError(f"{dataset}: final/results_summary.json lacks the paired test effects")

    # Admission: Theorem 2 on independent graphs, descriptive clipped mean otherwise. A split
    # campaign aggregates the split admission means with the small-sample interval over splits.
    independent = bool(admission.get("independent_graph_guarantee"))
    eb = admission.get("empirical_bernstein") or {}
    if view.is_campaign:
        means = [(e["admission"].get("loss") or {}).get("mean") for e in member_evidence]
        if any(m is None for m in means):
            raise DisplayInputError(f"{dataset}: a split admission record lacks its clipped mean")
        split_interval = _t_interval([float(m) for m in means])
        d_bar, u_desc, units = split_interval["mean"], split_interval["upper"], len(means)
    else:
        block = admission.get("loss") or {}
        d_bar, u_desc, units = _f(block.get("mean")), _f(block.get("upper")), block.get("units")
    if independent:
        d_bar, units = _f(eb.get("mean")), eb.get("units")
    readings = [selection_readings_of(e) for e in member_evidence]
    choices = [r for r in readings if r is not None]
    reading = max(set(choices), key=choices.count) if choices else None
    models = final.get("models") or {}
    qualification = [(e["cost"] or {}).get("qualification_seconds_total") for e in member_evidence]
    training = [(e["cost"] or {}).get("training_seconds") for e in member_evidence]
    return {
        "dataset": dataset, "short": SHORT[dataset], "view": view, "evidence": evidence,
        "members": members, "member_evidence": member_evidence, "is_campaign": view.is_campaign,
        "qC": qc, "provisional": provisional, "deployed": deployed, "returned": returned,
        "G_admit": 1 if admission.get("Gadmit") == 1 else 0, "metric": metric, "loss_name": loss_name,
        "loss_effect": {k: _f(loss.get(k)) for k in ("mean", "lower", "upper")},
        "gain_pp": {k: _f(gain.get(k)) for k in ("mean", "lower", "upper")},
        "independent": independent, "D_bar": d_bar, "U_desc": None if independent else u_desc,
        "U_a": _f(eb.get("upper")) if independent else None, "J": units,
        "epsilon_a": _f(admission.get("epsilon_a")), "epsilon_r": _f(selection.get("epsilon_r")),
        "epsilon_eq": _f(selection.get("epsilon_eq")), "Q_s": selection.get("Q_s") or [],
        "registered_reading": reading, "reading_agreement": (choices.count(reading), len(choices)),
        "C_qC": _f(qc.get("cost")), "C_ctrl": None if returned else _f(deployed.get("cost")),
        "models": models,
        "qualification_hours": (sum(qualification) / 3600.0) if all(q is not None for q in qualification) else None,
        "training_hours": (sum(training) / 3600.0) if all(t is not None for t in training) else None,
    }


def selection_readings_of(evidence: dict) -> str | None:
    """Route chosen by the registered reading of Eq. (19) on the representative rows (None: q_C)."""
    from paper.build.selection import selection_readings
    audit = selection_readings(evidence)
    if not audit.get("available"):
        return None
    return audit["readings"]["registered"]["choice_route"] or "qC"


def deployments(ctx) -> dict:
    return {dataset: deployment(ctx, dataset) for dataset in DATASETS}


def metric_texts(dep: dict, block: dict) -> tuple[str | None, str | None, str | None]:
    """Official-metric change in the printed unit: pp (two decimals) or native AP (four decimals)."""
    if dep["metric"] == "ap":
        return tuple(num(None if block[k] is None else block[k] / 100.0, 4, signed=(k == "mean"))
                     for k in ("mean", "lower", "upper"))
    return num(block["mean"], 2, signed=True), num(block["lower"], 2), num(block["upper"], 2)


# ---------------------------------------------------------------------------
# Cost ledger: Eq. (32), assurance cost, Table S44 tail model, Definition 2
# ---------------------------------------------------------------------------

def input_fallback(gate, dataset: str) -> float:
    value = _require_value(gate, "fallback_probability", dataset=dataset)
    if not 0.0 <= value <= 1.0:
        raise DisplayInputError(f"input_gate: fallback_probability of {dataset} is not a probability")
    return value


def tail_model(ctx, dep: dict) -> dict:
    """Table S44: C_ctrl(x) = C_qC [1 + o_fix + f_res(x) v] at the low/central/high tail shares."""
    from cm_theory.budget import max_resolvent_fraction, tail_scenarios
    r = _sparse_to_dense_ratio(ctx, dep["dataset"])
    order = _truncation_order(ctx)
    m = dep["C_ctrl"] / dep["C_qC"]
    capacity = max_resolvent_fraction(m, r, TAIL_SENSING_SHARE, order)
    f_bar = min(TAIL_REFERENCE_SHARE, capacity)
    scenarios = tail_scenarios(dep["C_qC"], m - 1.0, f_bar, r, order)
    return {"r": r, "order": order, "o_ctrl": m - 1.0, "f_bar": f_bar, "f_bar_rule": (
        "scenario share" if f_bar == TAIL_REFERENCE_SHARE else "largest share carried by the measured multiplier"),
        **scenarios}


def cost_ledger(ctx, dep: dict, gate) -> dict:
    """Gain/cost ledger of one dataset (Table 2 and the cost panels of Figures 2 and 4-6)."""
    from cm_theory import economics
    c_qc = dep["C_qC"]
    out = {"dataset": dep["dataset"], "C_qC": c_qc, "C_ctrl": dep["C_ctrl"], "p_fallback": None,
           "C_expected": c_qc, "C_assure": c_qc, "expected_overhead": 0.0, "assure_overhead": 0.0,
           "added_work": 0.0, "lambda_star": None, "lambda_range": None, "metric_rate": None,
           "M_star_t_nom": None, "tail": None, "qualification_hours": dep["qualification_hours"]}
    if dep["returned"]:
        return out
    p = input_fallback(gate, dep["dataset"])
    expected = economics.expected_cost(dep["C_ctrl"], c_qc, p)
    assure = economics.assurance_cost(dep["C_ctrl"], c_qc)
    added = expected - c_qc
    out.update(p_fallback=p, C_expected=expected, C_assure=assure, expected_overhead=added / c_qc,
               assure_overhead=assure / c_qc - 1.0, added_work=added, tail=tail_model(ctx, dep))
    effect = dep["loss_effect"]
    if added > 0 and effect["mean"] < 0:
        out["lambda_star"] = 1000.0 * economics.exchange_rate(abs(effect["mean"]), expected, c_qc)
        ends = [min(0.0, effect[k]) for k in ("upper", "lower") if effect[k] is not None]
        if len(ends) == 2:
            out["lambda_range"] = [1000.0 * abs(e) / added for e in ends]
        out["metric_rate"] = dep["gain_pp"]["mean"] / added
        if dep["qualification_hours"] is not None and dep["qualification_hours"] > 0:
            out["M_star_t_nom"] = economics.amortization_horizon(dep["qualification_hours"], _theta(ctx), added, 1.0)
    return out


def ledgers(ctx, deps: dict, gate) -> dict:
    return {dataset: cost_ledger(ctx, deps[dataset], gate) for dataset in DATASETS}


# ---------------------------------------------------------------------------
# Table 1
# ---------------------------------------------------------------------------

def confirmation_text(dep: dict) -> str:
    """Executed confirmation contrasts: effect, Holm P and decision of every contrast of a run; a split
    campaign reports in how many splits each contrast of the selected rule passes."""
    if dep["is_campaign"]:
        counts = {}
        for member, evidence in zip(dep["members"], dep["member_evidence"]):
            provisional, _ = _member_indices(dep, member)
            for row in evidence["confirmation"].get("contrasts") or []:
                if row.get("candidate_index") == provisional:
                    passed, total = counts.get(row.get("kind"), (0, 0))
                    counts[row.get("kind")] = (passed + bool(row.get("passes")), total + 1)
        if counts:
            return "; ".join(f"{kind}: pass in {p} of {n} splits" for kind, (p, n) in counts.items())
    else:
        rows = dep["evidence"]["confirmation"].get("contrasts") or []
        several = len({row.get("candidate_index") for row in rows}) > 1
        parts = []
        for row in rows:
            name = f"{row.get('kind')} (candidate {row.get('candidate_index')})" if several else str(row.get("kind"))
            if row.get("mean") is None:
                parts.append(f"{name}: unavailable")
                continue
            holm = row.get("holm_p_value_zero")
            parts.append(f"{name} {row['mean']:+.5f} (Holm P = {'n/a' if holm is None else format(holm, '.3g')}; "
                         f"{'pass' if row.get('passes') else 'fail'})")
        if parts:
            return "; ".join(parts)
    return "no feedback candidate in Q_s; q_C retained" if not dep["Q_s"] else "no contrasts recorded"


def admission_criterion(dep: dict) -> str:
    if dep["returned"]:
        return "not evaluated (Q_s empty)" if not dep["Q_s"] else "G_admit = 0; q_C returned"
    if dep["independent"]:
        return f"Theorem 2 on J = {dep['J']} independent graphs, epsilon_a = {dep['epsilon_a']:g}"
    unit = "official splits" if dep["is_campaign"] else "buffered cores"
    return f"descriptive, {dep['J']} {unit}"


def scope_text(dep: dict) -> str:
    if dep["returned"]:
        return "comparator regime (exactly zero effect by construction)"
    return "conditional independent-graph admission" if dep["independent"] else "descriptive single-graph admission"


@display("main:Table 1", stem="T1", title="Policy and paired test outcomes",
         needs=("runs", "records:input_gate"))
def table_1(ctx, out):
    gate = ctx.require_records("input_gate")
    rows, ledger_rows, source = [], [], []
    for dataset in DATASETS:
        dep = deployment(ctx, dataset)
        qc, deployed, provisional = dep["qC"], dep["deployed"], dep["provisional"]
        unit, metric_name = METRIC_LABELS[dep["metric"]]
        loss = dep["loss_effect"]
        fallback = None if dep["returned"] else input_fallback(gate, dataset)
        reading = dep["registered_reading"]
        reading_text = "q_C" if reading in (None, "qC") else ROUTE_WORDS[reading]
        if dep["returned"]:
            policy, outcome, admission_cell = "q_C returned", "0 (comparator identity)", "q_C; 0"
        else:
            policy = f"{ROUTE_WORDS[deployed['route']]}, E = {deployed['ensemble']}"
            g, glo, ghi = metric_texts(dep, dep["gain_pp"])
            outcome = (f"{g} [{glo}, {ghi}] {unit}; {num(loss['mean'], 5, signed=True)} "
                       f"[{num(loss['lower'], 4)}, {num(loss['upper'], 4)}] {dep['loss_name']}")
            label = f"G_admit = {dep['G_admit']}" if dep["independent"] else "descriptive"
            admission_cell = f"{label}; {num(loss['mean'], 4)}"
        rows.append([SHORT[dataset], route_identity(qc, style="table1"), policy, outcome, reading_text, admission_cell])

        provisional_text = (f"same {ROUTE_WORDS[qc['route']]}" if provisional["index"] == qc["index"]
                            else f"{ROUTE_WORDS[provisional['route']]}/{provisional['ensemble']}")
        deployed_rule = "qC returned" if dep["returned"] else ROUTE_WORDS[deployed["route"]]
        gain = dep["gain_pp"]
        official = None if dep["returned"] else gain["mean"]
        official_interval = None if dep["returned"] else [gain["lower"], gain["upper"]]
        effect = 0.0 if dep["returned"] else loss["mean"]
        effect_interval = None if dep["returned"] else [loss["lower"], loss["upper"]]
        stats = {"admission_D_bar": None if dep["returned"] else dep["D_bar"],
                 "admission_U_desc": None if dep["returned"] else dep["U_desc"],
                 "admission_U_a": None if dep["returned"] else dep["U_a"],
                 "admission_J": None if dep["returned"] or not dep["independent"] else dep["J"]}
        g_text = None if dep["returned"] else " ".join(filter(None, [num(official, 2, signed=True),
                                                                   f"[{num(gain['lower'], 2)}, {num(gain['upper'], 2)}]", "pp"]))
        e_text = "0" if dep["returned"] else (f"{num(effect, 5, signed=True)} [{num(loss['lower'], 4)}, "
                                              f"{num(loss['upper'], 4)}] {dep['loss_name']}")
        ledger_rows.append([SHORT[dataset], route_identity(qc, style="registry"), integer(qc["ensemble"]),
                            provisional_text, confirmation_text(dep), integer(dep["G_admit"]), admission_criterion(dep),
                            num(stats["admission_D_bar"], 5, signed=True), num(stats["admission_U_desc"], 5, signed=True),
                            num(stats["admission_U_a"], 6, signed=True), integer(stats["admission_J"]),
                            num(fallback, 3), g_text, e_text, scope_text(dep), deployed_rule,
                            integer(deployed["ensemble"])])
        source.append([dataset, qc["route"], qc["ensemble"], qc.get("horizon"), _phase_boundaries(qc), provisional["route"],
                       provisional["ensemble"], deployed["route"], deployed["ensemble"], dep["returned"], dep["G_admit"],
                       dep["independent"], stats["admission_D_bar"], stats["admission_U_desc"], stats["admission_U_a"],
                       dep["J"], dep["epsilon_a"], fallback, dep["metric"], gain["mean"], gain["lower"], gain["upper"],
                       loss["mean"], loss["lower"], loss["upper"], dep["loss_name"], reading,
                       "{}/{}".format(*dep["reading_agreement"])])
        for key, value in (("qC_identity", route_identity(qc, style="registry")), ("qC_E", qc["ensemble"]),
                           ("provisional", provisional_text), ("G_admit", dep["G_admit"]), *stats.items(),
                           ("input_fallback_fraction", fallback), ("official_gain_pp", 0.0 if dep["returned"] else official),
                           ("official_gain_interval_pp", official_interval), ("official_metric", metric_name),
                           ("loss_effect_nats", effect), ("loss_effect_interval_nats", effect_interval),
                           ("loss_name", dep["loss_name"]), ("deployed_rule", deployed_rule),
                           ("deployed_E", deployed["ensemble"])):
            out.compare(T1_GROUP, (dataset, key), value)
        if dep["metric"] == "ap" and not dep["returned"]:
            out.compare(T1_GROUP, (dataset, "official_gain_native"), official / 100.0)
            out.compare(T1_GROUP, (dataset, "official_gain_native_interval"), [v / 100.0 for v in official_interval])
        out.uses("run", dataset, ["select/decision.json", "select/candidates.json", "confirm/decision.json",
                                  "admit/decision.json", "final/results_summary.json"])
    out.uses("records", "input_gate", ["summary.csv"])
    out.table(["Dataset", "strongest open-loop comparator q_C", "deployed policy",
               "official-metric change; paired loss versus q_C (95% test-stage intervals)",
               "registered reading, representative rows", "separated-protocol admission; test effect"], rows,
              notes=["Effects are paired test evaluations of the executed policy against q_C; a Roman-empire campaign "
                     "reports the split mean with a small-sample t interval over its official splits. The registered "
                     "reading applies Eq. (19) to the representative rows (Supplementary Table S12c); for a split "
                     "campaign it is the reading of the most splits."])
    out.table(["Dataset", "q_C identity", "q_C E", "provisional rule", "confirmation output", "G_admit",
               "admission criterion", "D-bar", "U_desc", "U_a", "J", "input-fallback fraction",
               "official gain [95% interval]", "loss effect [95% interval]", "scope", "deployed rule", "deployed E"],
              ledger_rows, name="T1_ledger",
              notes=["Policy and deployment ledger behind Table 1: confirmation contrasts of the run, the admission "
                     "criterion with its statistics (Theorem 2 on independent graphs, descriptive clipped mean on "
                     "dependent units) and the serving-time input-fallback fraction of the input_gate record set."])
    out.data("ledger", ["dataset", "qC_route", "qC_E", "qC_horizon", "qC_phase_boundaries", "provisional_route",
                        "provisional_E", "deployed_route", "deployed_E", "qC_returned", "G_admit", "independent_graphs",
                        "D_bar", "U_desc", "U_a", "J", "epsilon_a", "input_fallback_fraction", "metric",
                        "official_gain_pp", "official_gain_low_pp", "official_gain_high_pp", "loss_effect_nats",
                        "loss_low_nats", "loss_high_nats", "loss", "registered_reading", "reading_splits"], source)


# ---------------------------------------------------------------------------
# Table 2
# ---------------------------------------------------------------------------

@display("main:Table 2", stem="T2", title="Predictive gains and serial-work accounting",
         needs=("runs", "records:input_gate"))
def table_2(ctx, out):
    gate = ctx.require_records("input_gate")
    rows, source = [], []
    for dataset in DATASETS:
        dep = deployment(ctx, dataset)
        led = cost_ledger(ctx, dep, gate)
        qc, deployed = dep["qC"], dep["deployed"]
        unit = METRIC_LABELS[dep["metric"]][0]
        tail = led["tail"]
        if dep["returned"]:
            cells = [f"q_C returned, {qc['ensemble']}", num(led["C_qC"], 1), f"{num(led['C_expected'], 1)} (0%)", None,
                     f"{num(led['C_assure'], 1)} (0%)", None, None, "exactly 0"]
        else:
            loss = dep["loss_effect"]
            gain = metric_texts(dep, dep["gain_pp"])[0]
            rate = None if led["metric_rate"] is None else f"{led['metric_rate']:.2f}" + (" AP" if dep["metric"] == "ap" else "")
            lam = None
            if led["lambda_star"] is not None:
                lam = f"{led['lambda_star']:.2f}"
                if led["lambda_range"]:
                    lam += f" [{led['lambda_range'][0]:.1f}, {led['lambda_range'][1]:.1f}]"
                lam += f" ({rate})"
            cells = [f"{ROUTE_WORDS[deployed['route']]}, {deployed['ensemble']}", num(led["C_qC"], 1),
                     f"{num(led['C_expected'], 2)} ({_pct(led['expected_overhead'])})",
                     f"{tail['C_099'][1]:.1f} [{tail['C_099'][0]:.1f}, {tail['C_099'][2]:.1f}] "
                     f"({_pct(tail['overhead_pct'][1] / 100.0)})",
                     f"{num(led['C_assure'], 1)} ({_pct(led['assure_overhead'])})", lam,
                     integer(None if led["M_star_t_nom"] is None else led["M_star_t_nom"] / 1000.0),
                     f"{gain} {unit}; {_trim(num(loss['mean'], 5, signed=True))} {dep['loss_name']}"]
        rows.append([SHORT[dataset], route_identity(qc, style="table2"), *cells])
        source.append([dataset, led["C_qC"], led["C_ctrl"], led["p_fallback"], led["C_expected"], led["expected_overhead"],
                       None if tail is None else tail["r"], None if tail is None else tail["f_bar"],
                       *(tail["C_099"] if tail else (None, None, None)), led["C_assure"], led["assure_overhead"],
                       led["added_work"], led["lambda_star"], *(led["lambda_range"] or (None, None)), led["metric_rate"],
                       led["qualification_hours"], led["M_star_t_nom"], dep["loss_effect"]["mean"], dep["gain_pp"]["mean"]])
        values = {"C_qC": led["C_qC"], "C_expected": led["C_expected"], "expected_overhead_pct": 100 * led["expected_overhead"],
                  "C_099": None if tail is None else tail["C_099"][1],
                  "C_099_range": None if tail is None else [tail["C_099"][0], tail["C_099"][2]],
                  "C_099_overhead_pct": None if tail is None else tail["overhead_pct"][1],
                  "C_assure": led["C_assure"], "assure_overhead_pct": 100 * led["assure_overhead"],
                  "lambda_star_mnats": led["lambda_star"], "lambda_star_range": led["lambda_range"],
                  "lambda_star_metric_rate": led["metric_rate"],
                  "M_star_t_nom_thousand_input_seconds": None if led["M_star_t_nom"] is None else led["M_star_t_nom"] / 1000.0}
        for key, value in values.items():
            out.compare(T2_GROUP, (dataset, key), value)
        out.uses("run", dataset, ["select/candidates.json", "admit/decision.json", "final/results_summary.json",
                                  "stages/explore.json", "stages/fit.json", "stages/select.json", "stages/confirm.json",
                                  "stages/admit.json"])
    out.uses("records", "input_gate", ["summary.csv"])
    out.table(["Dataset", "q_C identity, E", "deployed route, E", "C_qC", "C_expected (overhead)",
               "C_0.99 [tail scenarios] (overhead)", "C_assure (overhead)", "lambda* [range] (mnats; pt per unit)",
               "M* t_nom (10^3 input s)", "effect over q_C"], rows,
              notes=["Costs in nominal full-depth executions (select/candidates.json). C_expected = (1 - p) C_ctrl + p C_qC "
                     "(Eq. 32) with the input-fallback probability p of the input_gate record set; C_assure = C_ctrl + C_qC. "
                     "C_0.99 is the Table S44 tail model at f99 = f_bar + {0.10, 0.20, 0.35}, with r = E/(N d) from the "
                     f"registered dataset statistics and f_bar = min({TAIL_REFERENCE_SHARE}, largest resolvent share the "
                     f"measured multiplier carries at sensing share {TAIL_SENSING_SHARE}). lambda* = |Delta|/(C_expected - C_qC) "
                     "with its range from the 95% interval of Delta; the bracketed rate is official-metric points per unit "
                     "of added work. M* t_nom = 3600 C_qual/(theta (C_expected - C_qC)) with C_qual the qualification "
                     "stage hours (explore to admit) of the run."])
    out.data("ledger", ["dataset", "C_qC", "C_controller", "p_fallback", "C_expected", "expected_overhead", "r", "f_bar",
                        "C_099_low", "C_099_central", "C_099_high", "C_assure", "assure_overhead", "added_work",
                        "lambda_star_mnats", "lambda_low_mnats", "lambda_high_mnats", "metric_points_per_unit",
                        "qualification_hours", "M_star_t_nom_input_seconds", "loss_effect_nats", "official_gain_pp"], source)


# ---------------------------------------------------------------------------
# Shared figure helpers
# ---------------------------------------------------------------------------

def _style(plt):
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.edgecolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK})


def _panel(ax, letter: str, title: str):
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.text(-0.08, 1.06, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="right")


def _deployed_datasets(deps: dict) -> list[str]:
    return [d for d in DATASETS if not deps[d]["returned"]]


def confirmation_family(ctx) -> dict:
    """The joint nine-test S14 family, validated by its dedicated campaign auditor."""
    from paper.build.supplement import _confirmation_data
    campaign = ctx.require_campaign("confirmation")
    try:
        checked, _ = _confirmation_data(campaign, ctx.root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DisplayInputError(f"confirmation campaign: {exc}") from None
    return {row["contrast_id"]: row for row in checked["rows"]}


def ensemble_fits(ctx) -> dict:
    """Least-squares ensemble law of every admitted dataset (Supplementary Table S26b)."""
    from paper.build.supplement import _ensemble_data
    view = ctx.require_records("ensemble_grid")
    try:
        fits, _ = _ensemble_data(view.rows, {"sets": {"ensemble_grid": view.spec}})
    except (ValueError, KeyError, TypeError) as exc:
        raise DisplayInputError(f"ensemble_grid: {exc}") from None
    return {row["dataset"]: row for row in fits}


def _selection_effect(member, index: int) -> tuple[float, float, float, float]:
    """(mean, U, epsilon_eq + 2 r_max, epsilon_r) of one candidate in a sealed selection record."""
    decision = member.json("select/decision.json")
    mean, upper = _f(decision["mean"][index]), _f(decision["upper"][index])
    if mean is None or upper is None:
        raise DisplayInputError(f"{member.dataset}: candidate {index} has no selection interval")
    return mean, upper, decision["epsilon_eq"] + 2 * decision["max_radius"], decision["epsilon_r"]


def _member_indices(dep: dict, member) -> tuple[int, int]:
    decision = member.json("select/decision.json")
    admit = member.json("admit/decision.json")
    return decision["provisional_index"], admit["deployed_index"]


def _reference_ess(member, profile: str) -> float | None:
    """Smallest ESS over the qualified support cells of the deployed profile (fit/<profile>/audit.json)."""
    audit = member.json(f"fit/{profile}/audit.json", default=None)
    if audit is None:
        return None
    cells = [c for c in (audit.get("support") or {}).get("cells", {}).values() if c.get("qualified")]
    values = [_f(c.get("ess")) for c in cells if _f(c.get("ess"))]
    return min(values) if values else None


# ---------------------------------------------------------------------------
# Figure 2
# ---------------------------------------------------------------------------

AUDIT_ROWS = (
    ("prop2", "Prop. 2: p95 loss error / plug-in tube"),
    ("prop4", "Prop. 4: deployed effect / headroom bound"),
    ("thm1", "Thm. 1: selected effect / near-best slack"),
    ("prop7", "Prop. 7: deployed effect / regret radius 2e(n) at ESS"),
    ("prop5_expected", "Prop. 5: break-even at expected overhead / effect"),
    ("prop5_assure", "Prop. 5: break-even at assurance overhead / effect"),
    ("thm2", "Thm. 2: eps_a / |U_a| (multigraph admission)"),
    ("cor2", "Cor. 2: minimum admission J / available units"),
    ("s44", "Table S44: expected cost / modelled C_0.99"),
)


def audit_ratios(ctx, deps, leds, fits, tube) -> list[dict]:
    """Panel 2a: each measured quantity divided by its analytical bound or threshold."""
    from cm_theory.admission import hoeffding_radius, minimum_admission_units, regret_bound
    from cm_theory.economics import break_even_effect
    adm = ctx.constants["admission_eq27_theorem2_table_S13"]
    clip = ctx.constants["observer_eq12_table_S9"]["loss_clipping_B"]["primary"]
    actions = len(ctx.constants["action_registry_table_S6"]["actions"])
    rows = []

    def add(key, dataset, measured, bound, measured_def, bound_def):
        if measured is None or bound is None or bound == 0:
            return
        rows.append({"row": key, "statement": dict(AUDIT_ROWS)[key], "dataset": dataset, "measured": measured,
                     "bound": bound, "ratio": measured / bound, "measured_definition": measured_def,
                     "bound_definition": bound_def})

    for dataset in _deployed_datasets(deps):
        dep, led = deps[dataset], leds[dataset]
        effect = abs(dep["loss_effect"]["mean"])
        if tube.select(dataset=dataset):
            add("prop2", dataset, _require_value(tube, "observed_error_p95_nats", dataset=dataset),
                _require_value(tube, "final_bound_nats", dataset=dataset), "p95 observed loss error", "plug-in tube bound")
        selected, headroom, slack, ess = [], [], [], []
        for member, evidence in zip(dep["members"], dep["member_evidence"]):
            provisional, deployed = _member_indices(dep, member)
            mean, _, member_slack, _ = _selection_effect(member, provisional)
            selected.append(abs(mean))
            slack.append(member_slack)
            deployed_mean = _selection_effect(member, deployed)[0]
            upper = (evidence["entry"].get("headroom") or {}).get("upper")
            if upper is not None and upper < 0:
                headroom.append((abs(deployed_mean), abs(upper)))
            ess.append(_reference_ess(member, dep["deployed"].get("profile") or "full"))
        if len(headroom) == len(dep["members"]):
            add("prop4", dataset, float(np.mean([h[0] for h in headroom])), float(np.mean([h[1] for h in headroom])),
                "|selection effect| of the deployed configuration", "|U(Delta_oracle)| of the D_explore headroom")
        add("thm1", dataset, float(np.mean(selected)), max(slack), "|Delta_s| of the selected rule", "eps_eq + 2 r_max")
        if all(e is not None for e in ess):
            n = min(ess)
            add("prop7", dataset, effect, regret_bound(hoeffding_radius(int(math.floor(n)), clip, actions)),
                "|Delta| (test)", f"2 e(n) at reference ESS n = {n:.0f}")
        fit = fits.get(dataset)
        if fit is not None:
            E = dep["qC"]["ensemble"]
            add("prop5_expected", dataset, break_even_effect(fit["D_inf"], led["expected_overhead"], E), effect,
                "D_inf o / (E (1 + o)) at expected overhead", "|Delta| (test)")
            add("prop5_assure", dataset, break_even_effect(fit["D_inf"], led["assure_overhead"], E), effect,
                "D_inf o / (E (1 + o)) at assurance overhead", "|Delta| (test)")
        if dep["independent"] and dep["U_a"] is not None:
            add("thm2", dataset, dep["epsilon_a"], abs(dep["U_a"]), "eps_a", "|U_a|")
        if dep["D_bar"] is not None and dep["J"]:
            j_min = minimum_admission_units(dep["D_bar"], dep["epsilon_a"], adm["B"], adm["delta_a"])
            if math.isfinite(j_min):
                add("cor2", dataset, float(j_min), float(dep["J"]), "minimum J of Corollary 2", "available units")
        if led["tail"] is not None:
            add("s44", dataset, led["C_expected"], led["tail"]["C_099"][1], "C_expected (Eq. 32)", "modelled C_0.99 (central)")
    return rows


STAGES = (("selection", "selection\n$U_s/(-\\epsilon_R)$"), ("confirmation", "confirmation\n|effect|/margin"),
          ("admission", "admission\n$-U_a/\\epsilon_a$ or $-\\bar D/\\epsilon_a$"), ("test", "test stage\n$|\\Delta|/\\epsilon_R$"))


def stage_margins(deps, family) -> list[dict]:
    """Panel 2b: criterion statistic over its threshold at selection, confirmation, admission and test."""
    rows = []
    for dataset in _deployed_datasets(deps):
        dep = deps[dataset]
        ratios = []
        for member in dep["members"]:
            provisional, _ = _member_indices(dep, member)
            _, upper, _, eps_r = _selection_effect(member, provisional)
            ratios.append(-upper / eps_r)
        rows.append({"dataset": dataset, "stage": "selection", "ratio": float(np.mean(ratios)), "low": None, "high": None,
                     "statistic": "U_s of the selected rule", "threshold": "epsilon_R"})
        contrast = next((row for key, row in family.items() if row["dataset"] == dataset), None)
        if contrast is not None:
            sign, margin = contrast["beneficial_direction"], contrast["practical_margin_nats"]
            ends = sorted(sign * v / margin for v in contrast["interval_nats"])
            rows.append({"dataset": dataset, "stage": "confirmation", "ratio": sign * contrast["effect_nats"] / margin,
                         "low": ends[0], "high": ends[1], "statistic": contrast["contrast_id"], "threshold": "practical margin"})
        statistic = dep["U_a"] if dep["independent"] else dep["D_bar"]
        rows.append({"dataset": dataset, "stage": "admission", "ratio": -statistic / dep["epsilon_a"], "low": None,
                     "high": None, "statistic": "U_a" if dep["independent"] else "D-bar", "threshold": "epsilon_a"})
        loss = dep["loss_effect"]
        rows.append({"dataset": dataset, "stage": "test", "ratio": -loss["mean"] / dep["epsilon_r"],
                     "low": -loss["upper"] / dep["epsilon_r"], "high": -loss["lower"] / dep["epsilon_r"],
                     "statistic": "paired test effect", "threshold": "epsilon_R"})
    return rows


def _representatives(member) -> list[dict]:
    """One configuration per route family (Table S12): the selected or deployed one, else the smallest U_s."""
    decision = member.json("select/decision.json")
    candidates = member.json("select/candidates.json")
    admit = member.json("admit/decision.json")
    marked = {decision["provisional_index"], admit["deployed_index"]}
    feasible = decision.get("joint_feasible") or [c.get("feasible") for c in candidates]
    rows = []
    for route in S12_ROUTES:
        family = [j for j, c in enumerate(candidates) if c["route"] == route and _f(decision["upper"][j]) is not None]
        if not family:
            continue
        chosen = [j for j in family if j in marked]
        j = chosen[0] if chosen else min(family, key=lambda i: (decision["upper"][i], i))
        rows.append({"route": route, "index": j, "ensemble": candidates[j]["ensemble"],
                     "profile": candidates[j].get("profile"), "lower": decision["lower"][j], "mean": decision["mean"][j],
                     "upper": decision["upper"][j], "feasible": bool(feasible[j]) if route in FEEDBACK else None})
    return rows


def selection_geometry(deps) -> list[dict]:
    """Panel 2c: representative rows per route family, the near-best band and the deployed route."""
    rows = []
    for dataset in DATASETS:
        dep = deps[dataset]
        per_route = {}
        for member in dep["members"]:
            for row in _representatives(member):
                per_route.setdefault(row["route"], []).append(row)
        eps_r, eps_eq = dep["epsilon_r"], dep["epsilon_eq"]
        averaged = []
        for route in S12_ROUTES:
            items = per_route.get(route)
            if not items:
                continue
            averaged.append({"route": route, "ensemble": items[0]["ensemble"], "profile": items[0]["profile"],
                             "lower": float(np.mean([r["lower"] for r in items])),
                             "mean": float(np.mean([r["mean"] for r in items])),
                             "upper": float(np.mean([r["upper"] for r in items])),
                             "feasible": all(r["feasible"] for r in items) if route in FEEDBACK else None})
        eligible = [r["upper"] for r in averaged if r["route"] in FEEDBACK and r["feasible"] and r["upper"] < -eps_r]
        u_min = min(eligible) if eligible else None
        deployed_route = None if dep["returned"] else dep["deployed"]["route"]
        for r in averaged:
            rows.append({"dataset": dataset, **r, "feedback": r["route"] in FEEDBACK,
                         "deployed": r["route"] == deployed_route, "U_min": u_min, "epsilon_eq": eps_eq,
                         "epsilon_R": eps_r, "splits": len(dep["members"])})
    return rows


def holm_family(ctx, family) -> list[dict]:
    """Panel 2d: raw one-sided P values of the nine contrasts against the Holm step thresholds."""
    alpha = ctx.constants["confirmation_eq20_table_S11"]["holm_alpha"]
    ordered = sorted(family.values(), key=lambda row: (row["p_value_zero"], row["contrast_id"]))
    m = len(ordered)
    return [{"rank": i + 1, "contrast": row["contrast_id"], "dataset": row["dataset"], "raw_P": row["p_value_zero"],
             "holm_threshold": alpha / (m - i), "holm_P": row["holm_p_value_zero"], "retained": row["retained"]}
            for i, row in enumerate(ordered)]


@display("main:Figure 2", stem="Fig2", title="Measured evidence and analytical statements",
         needs=("runs", "campaign:confirmation", "records:ensemble_grid", "records:input_gate", "records:tube_audit"))
def figure_2(ctx, out):
    deps = deployments(ctx)
    leds = ledgers(ctx, deps, ctx.require_records("input_gate"))
    family = confirmation_family(ctx)
    fits = ensemble_fits(ctx)
    tube = ctx.require_records("tube_audit")
    panel_a = audit_ratios(ctx, deps, leds, fits, tube)
    panel_b = stage_margins(deps, family)
    panel_c = selection_geometry(deps)
    panel_d = holm_family(ctx, family)
    columns_a = ["row", "statement", "dataset", "measured", "bound", "ratio", "measured_definition", "bound_definition"]
    out.data("panel_a", columns_a, [[r[c] for c in columns_a] for r in panel_a])
    columns_b = ["dataset", "stage", "ratio", "low", "high", "statistic", "threshold"]
    out.data("panel_b", columns_b, [[r[c] for c in columns_b] for r in panel_b])
    columns_c = ["dataset", "route", "ensemble", "profile", "lower", "mean", "upper", "feedback", "feasible", "deployed",
                 "U_min", "epsilon_eq", "epsilon_R", "splits"]
    out.data("panel_c", columns_c, [[r[c] for c in columns_c] for r in panel_c])
    columns_d = ["rank", "contrast", "dataset", "raw_P", "holm_threshold", "holm_P", "retained"]
    out.data("panel_d", columns_d, [[r[c] for c in columns_d] for r in panel_d])
    _figure_2_compare(out, deps, panel_a, panel_c, family, fits)

    plt = pyplot()
    _style(plt)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5), gridspec_kw={"width_ratios": [1.35, 1.0]})
    ax = axes[0, 0]
    order = [key for key, _ in AUDIT_ROWS]
    for row in panel_a:
        y = order.index(row["row"])
        ax.plot(row["ratio"], y, "o", color=COLORS[row["dataset"]], ms=6, mec="white", mew=0.6, zorder=3)
    ax.axvline(1.0, color=INK, lw=1.2)
    ax.set_xscale("log")
    ax.set_yticks(range(len(AUDIT_ROWS)), [label for _, label in AUDIT_ROWS], fontsize=7.5)
    ax.set_ylim(len(AUDIT_ROWS) - 0.5, -0.5)
    ax.set_xlabel("measured quantity / its analytical bound (log scale); the vertical line is the bound")
    single = [r["ratio"] for r in panel_a if r["row"] == "cor2" and not deps[r["dataset"]]["independent"]]
    if single:
        ax.text(0.98, 0.06, f"single-graph units:\n{min(single):.0f}–{max(single):.0f}× below Cor. 2",
                transform=ax.transAxes, ha="right", fontsize=7.5, color=MUTED)
    _panel(ax, "a", "Measured quantities against the analytical statements")

    ax = axes[0, 1]
    stage_index = {name: i for i, (name, _) in enumerate(STAGES)}
    shown = _deployed_datasets(deps)
    offsets = {d: (k - (len(shown) - 1) / 2) * 0.11 for k, d in enumerate(shown)}
    points = [r["ratio"] for r in panel_b if r["ratio"] > 0]
    floor = 0.15 * min([1.0, *points])
    ceiling = 8.0 * max([1.0, *points, *[r["high"] for r in panel_b if r["high"] is not None]])
    for row in panel_b:
        x = stage_index[row["stage"]] + offsets[row["dataset"]]
        if row["low"] is not None:
            ax.plot([x, x], [max(row["low"], floor), row["high"]], color=COLORS[row["dataset"]], lw=1.6)
        ax.plot(x, row["ratio"], "o", color=COLORS[row["dataset"]], ms=6, mec="white", mew=0.6, zorder=3)
    ax.set_ylim(floor, ceiling)
    returned = [d for d in DATASETS if deps[d]["returned"]]
    if returned:
        from matplotlib.transforms import blended_transform_factory
        transform = blended_transform_factory(ax.transData, ax.transAxes)
        for i in range(len(STAGES)):
            ax.plot(i, 0.1, "x", color=MUTED, ms=7, mew=1.5, transform=transform)
    ax.axhline(1.0, color=INK, lw=1.2)
    ax.set_yscale("log")
    ax.set_xticks(range(len(STAGES)), [label for _, label in STAGES], fontsize=7.5)
    ax.set_ylabel("criterion statistic / its threshold (passes above one)")
    handles = [ax.plot([], [], "o", color=COLORS[d], label=SHORT[d])[0] for d in shown]
    if returned:
        handles.append(ax.plot([], [], "x", color=MUTED, label=", ".join(SHORT[d] for d in returned) + ": comparator retained")[0])
    ax.legend(handles=handles, ncol=2, frameon=False, fontsize=7, loc="upper left")
    _panel(ax, "b", "Evidence margin at every lifecycle stage")

    ax = axes[1, 0]
    y = 0
    ticks = []
    for dataset in DATASETS:
        block = [r for r in panel_c if r["dataset"] == dataset]
        block = sorted(block, key=lambda r: (not r["feedback"], r["upper"]))
        if not block:
            continue
        top, bottom = y - 0.5, y + len(block) - 0.5
        first = block[0]
        if first["U_min"] is not None:
            ax.fill_between([first["U_min"], first["U_min"] + first["epsilon_eq"]], top, bottom, color="#e9dcb5",
                            alpha=0.8, lw=0, zorder=1)
        ax.plot([-first["epsilon_R"]] * 2, [top, bottom], color="#e34948", lw=1.0, zorder=2)
        color = COLORS[dataset]
        for r in block:
            line_color = color if r["feedback"] and not deps[dataset]["returned"] else ("#8a8984" if r["feedback"] else "#9a9994")
            ax.plot([r["lower"], r["upper"]], [y, y], color=line_color, lw=2.2 if r["feedback"] else 1.0, zorder=3)
            ax.plot(r["mean"], y, "o", color=line_color if r["feedback"] else "white", mec=line_color, ms=4, zorder=4)
            if r["deployed"]:
                ax.plot(r["mean"], y, "*", color="#eda100", mec=INK, mew=0.6, ms=13, zorder=5)
            y += 1
        label = (f"{SHORT[dataset]}\n→ q_C retained" if deps[dataset]["returned"]
                 else f"{SHORT[dataset]}\n→ {ROUTE_WORDS[deps[dataset]['deployed']['route']]}")
        ticks.append(((top + bottom) / 2, label, color))
        ax.axhline(bottom, color=GRID, lw=0.8)
    for position, label, color in ticks:
        ax.text(1.01, position, label, transform=ax.get_yaxis_transform(), fontsize=7.5, color=color,
                fontweight="bold", va="center")
    ax.set_ylim(y - 0.5, -0.5)
    ax.set_yticks([])
    ax.set_xlabel("simultaneous selection interval $[L_s, U_s]$ of each representative route against the nested $q_C$ (nats)")
    legend = [ax.plot([], [], color=INK, lw=2.2, label="feedback routes (one-step, radius, MPC-RH)")[0],
              ax.plot([], [], color="#9a9994", lw=1.0, label="open-loop routes")[0],
              ax.fill_between([], [], [], color="#e9dcb5", label="near-best band $[U_{min}, U_{min}+\\epsilon_{eq}]$"),
              ax.plot([], [], color="#e34948", lw=1.0, label="$-\\epsilon_R$")[0],
              ax.plot([], [], "*", color="#eda100", mec=INK, ms=11, label="deployed route")[0]]
    ax.legend(handles=legend, ncol=3, frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.09))
    _panel(ax, "c", "Selection geometry: Eq. (19) on the representative rows of Table S12")

    ax = axes[1, 1]
    if panel_d:
        ranks = [r["rank"] for r in panel_d]
        thresholds = [r["holm_threshold"] for r in panel_d]
        ax.step([ranks[0] - 0.5, *[r + 0.5 for r in ranks]], [thresholds[0], *thresholds], where="pre", color=INK, lw=1.4,
                label=f"Holm threshold $\\alpha/(m-i+1)$, $\\alpha$ = {ctx.constants['confirmation_eq20_table_S11']['holm_alpha']:g}, m = {len(panel_d)}")
        for row in panel_d:
            color = COLORS.get(row["dataset"], MUTED)
            ax.plot(row["rank"], max(row["raw_P"], 1e-6), "o", color=color, mec=INK, mew=0.6, ms=9, zorder=3)
            ax.text(row["rank"] + 0.18, max(row["raw_P"], 1e-6), row["contrast"].replace("/", "\n"), ha="left",
                    va="center", fontsize=6.5, color=color)
        ax.set_yscale("log")
        ax.set_xticks(ranks)
        ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_xlabel("rank of the raw one-sided P value (Table S14)")
    ax.set_ylabel("P value")
    _panel(ax, "d", "Holm family of the nine mechanism contrasts")
    if ctx.synthetic:
        fig.text(0.5, 1.0, "SYNTHETIC SOFTWARE EXAMPLE — not benchmark results", ha="center", va="bottom", color="#b00020")
    fig.tight_layout()
    out.figure(fig)
    for dataset in DATASETS:
        out.uses("run", dataset, ["select/decision.json", "select/candidates.json", "select/entry_complete.json",
                                  "admit/decision.json", "final/results_summary.json", "fit/full/audit.json",
                                  "fit/lean/audit.json"])
    out.uses("campaign", "confirmation", ["campaign.json"])
    for name in ("ensemble_grid", "input_gate", "tube_audit"):
        out.uses("records", name, ["summary.csv"])


def _figure_2_compare(out, deps, panel_a, panel_c, family, fits):
    for row in panel_a:
        dataset = row["dataset"]
        if row["row"] == "prop2":
            out.compare(S22_GROUP, (dataset, "observed_error_p95"), row["measured"])
            out.compare(S22_GROUP, (dataset, "final_bound"), row["bound"])
        elif row["row"] == "thm1":
            out.compare(S20_GROUP, (dataset, "eps_eq_plus_2r_max"), row["bound"])
        elif row["row"] == "thm2":
            out.compare(S13_GROUP, (dataset, "U_a"), -row["bound"])
    for dataset, fit in fits.items():
        if dataset in deps and not deps[dataset]["returned"]:
            out.compare(S26B_GROUP, (dataset, "D_inf_hat"), fit["D_inf"])
    for row in panel_c:
        index = 1 + S12_ROUTES.index(row["route"])
        for key, field in (("mean", "Delta_s_hat"), ("lower", "L_s"), ("upper", "U_s")):
            out.compare(S12_GROUP, (row["dataset"], index, field), row[key])
    for dataset in DATASETS:
        u_min = next((r["U_min"] for r in panel_c if r["dataset"] == dataset), None)
        out.compare(S12C_GROUP, (dataset, "U_min"), u_min)
    from cm_campaigns.confirmation import FAMILY
    for index, key in enumerate(FAMILY):
        if key in family:
            row = family[key]
            out.compare(S14_GROUP, (index, "raw_P"), _p_text(row["p_value_zero"], 1e-4, "< 1e-4"))
            out.compare(S14_GROUP, (index, "holm_P"), _p_text(row["holm_p_value_zero"], 1e-3, "< 0.001"))


def _p_text(value: float, floor: float, text: str):
    """A P value below the printed floor of Table S14 is compared as the printed inequality."""
    return text if value < floor else value


# ---------------------------------------------------------------------------
# Figure 3
# ---------------------------------------------------------------------------

FAMILY_KINDS = {"state": "state", "mpc": "rh_minus_ol"}


def _run_contrast(dep: dict, kind: str) -> dict | None:
    """The deployed rule's own D_confirm contrast of one kind; a split campaign gives the t interval over splits."""
    values = []
    for member, evidence in zip(dep["members"], dep["member_evidence"]):
        _, deployed = _member_indices(dep, member)
        rows = [r for r in evidence["confirmation"].get("contrasts") or []
                if r.get("kind") == kind and r.get("candidate_index") == deployed and r.get("mean") is not None]
        if not rows:
            return None
        values.append(rows[0])
    if len(values) == 1:
        row = values[0]
        if row.get("lower") is None or row.get("upper") is None:
            return None
        return {"effect": float(row["mean"]), "low": float(row["lower"]), "high": float(row["upper"]),
                "source": "deployed rule's confirm/decision.json"}
    interval = _t_interval([float(r["mean"]) for r in values])
    return {"effect": interval["mean"], "low": interval["lower"], "high": interval["upper"],
            "source": f"deployed rule's confirm/decision.json, t interval over {len(values)} splits"}


def mechanism_rows(deps, family, records) -> list[dict]:
    """Panels 3a-b: state destruction and MPC-RH minus MPC-OL for every dataset."""
    from cm_campaigns.confirmation import FAMILY
    rows = []
    for kind, record_key in FAMILY_KINDS.items():
        for dataset in DATASETS:
            found = None
            for key, spec in FAMILY.items():
                if spec[0] == dataset and spec[1] == kind and key in family:
                    item = family[key]
                    found = {"effect": item["effect_nats"], "low": item["interval_nats"][0], "high": item["interval_nats"][1],
                             "source": f"joint S14 family ({key})", "family_key": key}
            if found is None and not deps[dataset]["returned"]:
                found = _run_contrast(deps[dataset], kind)
            if found is None:
                keys = {"dataset": dataset, "contrast": record_key}
                if not records.select("effect_nats", **keys):
                    raise DisplayInputError(f"mechanism_contrasts: {record_key} of {dataset} is required for Figure 3")
                found = {"effect": _require_value(records, "effect_nats", **keys),
                         "low": _require_value(records, "interval_low_nats", **keys),
                         "high": _require_value(records, "interval_high_nats", **keys),
                         "source": "mechanism_contrasts record set"}
            if not found["low"] <= found["effect"] <= found["high"]:
                raise DisplayInputError(f"{dataset}: the {record_key} interval does not contain its effect")
            rows.append({"panel": "a" if kind == "state" else "b", "dataset": dataset, "contrast": record_key, **found})
    return rows


def trace_rows(traces) -> list[dict]:
    rows = []
    for dataset in DATASETS:
        fractions = [_require_value(traces, f"fraction_{key}", dataset=dataset) for key, _ in ORIGINS]
        if any(f < -1e-9 for f in fractions) or abs(sum(fractions) - 1.0) > 1e-6:
            raise DisplayInputError(f"decision_traces: the decision-origin fractions of {dataset} do not sum to one")
        rows.append({"dataset": dataset, "entropy": _require_value(traces, "same_depth_action_entropy", dataset=dataset),
                     **{key: f for (key, _), f in zip(ORIGINS, fractions)}})
    return rows


@display("main:Figure 3", stem="Fig3", title="Feedback and replanning identification",
         needs=("runs", "campaign:confirmation", "records:mechanism_contrasts", "records:decision_traces"))
def figure_3(ctx, out):
    deps = deployments(ctx)
    family = confirmation_family(ctx)
    contrasts = mechanism_rows(deps, family, ctx.require_records("mechanism_contrasts"))
    traces = trace_rows(ctx.require_records("decision_traces"))
    margin = ctx.constants["confirmation_eq20_table_S11"]["contrasts"]["MPC-RH"]["practical_margin_nats"]
    columns = ["panel", "dataset", "contrast", "effect", "low", "high", "source"]
    out.data("panel_a", columns, [[r[c] for c in columns] for r in contrasts if r["panel"] == "a"])
    out.data("panel_b", columns + ["equivalence_margin"],
             [[r[c] for c in columns] + [margin] for r in contrasts if r["panel"] == "b"])
    out.data("panel_c", ["dataset", "same_depth_action_entropy"], [[r["dataset"], r["entropy"]] for r in traces])
    out.data("panel_d", ["dataset", *[key for key, _ in ORIGINS]], [[r["dataset"], *[r[k] for k, _ in ORIGINS]] for r in traces])
    from cm_campaigns.confirmation import FAMILY
    index = {key: i for i, key in enumerate(FAMILY)}
    for row in contrasts:
        if "family_key" in row:
            out.compare(S14_GROUP, (index[row["family_key"]], "effect"), row["effect"])
            out.compare(S14_GROUP, (index[row["family_key"]], "interval"), [row["low"], row["high"]])

    plt = pyplot()
    _style(plt)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    labels = [SHORT[d] for d in DATASETS]
    for ax, panel, title, xlabel in ((axes[0, 0], "a", "Performance decreases after conditional state destruction",
                                      "loss increase after conditional state destruction (nats)"),
                                     (axes[0, 1], "b", "Replanning is retained only outside equivalence",
                                      "MPC-RH minus MPC-OL paired loss (nats)")):
        rows = [r for r in contrasts if r["panel"] == panel]
        if panel == "b":
            ax.axvspan(-margin, margin, color="#e6e5e0", lw=0, label="equivalence region (±confirmation margin)")
        for y, row in enumerate(rows):
            ax.errorbar(row["effect"], y, xerr=[[row["effect"] - row["low"]], [row["high"] - row["effect"]]],
                        fmt="o", color=COLORS[row["dataset"]], capsize=3, ms=6)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_ylim(len(labels) - 0.5, -0.5)
        ax.set_xlabel(xlabel)
        if panel == "b":
            ax.legend(frameon=False, fontsize=7, loc="lower left")
        _panel(ax, panel, title)
    ax = axes[1, 0]
    ax.barh(range(len(traces)), [r["entropy"] for r in traces], color=[COLORS[r["dataset"]] for r in traces], height=0.7)
    for y, row in enumerate(traces):
        ax.text(row["entropy"], y, f" {row['entropy']:.2f}", va="center", fontsize=7, color=MUTED)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_xlabel("same-depth action entropy (nats)")
    _panel(ax, "c", "Selected actions vary within depth strata")
    ax = axes[1, 1]
    bottom = np.zeros(len(traces))
    for (key, label), color in zip(ORIGINS, ORIGIN_COLORS):
        values = np.array([r[key] for r in traces])
        ax.bar(range(len(traces)), values, bottom=bottom, color=color, width=0.75, label=label, edgecolor="white", lw=0.8)
        bottom += values
    ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("fraction of decisions")
    ax.legend(ncol=2, frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    _panel(ax, "d", "Decision origins are mutually exclusive")
    if ctx.synthetic:
        fig.text(0.5, 1.0, "SYNTHETIC SOFTWARE EXAMPLE — not benchmark results", ha="center", va="bottom", color="#b00020")
    fig.tight_layout()
    out.figure(fig)
    for dataset in DATASETS:
        out.uses("run", dataset, ["confirm/decision.json", "admit/decision.json"])
    out.uses("campaign", "confirmation", ["campaign.json"])
    out.uses("records", "mechanism_contrasts", ["summary.csv"])
    out.uses("records", "decision_traces", ["summary.csv"])


# ---------------------------------------------------------------------------
# Figure 4
# ---------------------------------------------------------------------------

def ensemble_response(dep: dict) -> list[dict]:
    """Panel 4b: selection-stage effect of the deployed configuration at every registered ensemble size.

    The family holds the deployed configuration's route, profile and route parameters at each E;
    the retained effect is its D_select risk reduction over the nested q_C divided by the largest
    over E, and the normalized work its serial cost divided by the largest over E.
    """
    member = dep["members"][0]
    decision = member.json("select/decision.json")
    candidates = member.json("select/candidates.json")
    _, deployed = _member_indices(dep, member)
    reference = candidates[deployed]
    keys = ("route", "profile", "kappa", "horizon", "sequence", "depth")
    family = [j for j, c in enumerate(candidates)
              if all(c.get(k) == reference.get(k) for k in keys) and _f(decision["mean"][j]) is not None]
    family.sort(key=lambda j: candidates[j]["ensemble"])
    reduction = [-float(decision["mean"][j]) for j in family]
    costs = [float(candidates[j]["cost"]) for j in family]
    top, work = max(reduction, default=0.0), max(costs, default=0.0)
    return [{"dataset": dep["dataset"], "ensemble": candidates[j]["ensemble"], "candidate_index": j,
             "risk_reduction_nats": red, "retained_fraction": red / top if top > 0 else None,
             "serial_cost": cost, "normalized_work": cost / work if work > 0 else None,
             "selected": j == deployed} for j, red, cost in zip(family, reduction, costs)]


def factorial_rows(ctx) -> list[dict]:
    from paper.build.supplement import _factorial_data
    view = ctx.require_records("factorials")
    spec = view.spec if view.spec.get("quantities") else ctx.record_specs["sets"]["factorials"]
    try:
        rows = _factorial_data(view.rows, {"sets": {"factorials": spec}})
    except (ValueError, KeyError, TypeError) as exc:
        raise DisplayInputError(f"factorials: {exc}") from None
    return [r for r in rows if r["dataset"] == "ogbn-arxiv"]


def gpu_hours(ctx, deps, baselines) -> list[dict]:
    """Panel 4e: run training and qualification hours against the retrained architectures' GPU-hours."""
    from cm_campaigns.contract import EXTERNAL
    rows = []
    for dataset in DATASETS:
        dep = deps[dataset]
        retrained = []
        for method in EXTERNAL[dataset]:
            value = _record_value(baselines, "gpu_hours", dataset=dataset, method=method, seed="")
            if value is None:
                raise DisplayInputError(f"baselines: gpu_hours of {method} on {dataset} is required for Figure 4e")
            retrained.append(value)
        if dep["training_hours"] is None or dep["qualification_hours"] is None:
            raise DisplayInputError(f"{dataset}: the run's stage records lack train or qualification durations")
        rows.append({"dataset": dataset, "plant_training_hours": dep["training_hours"],
                     "qualification_hours": dep["qualification_hours"], "retrained_low_gpu_hours": min(retrained),
                     "retrained_high_gpu_hours": max(retrained), "retrained_methods": " / ".join(EXTERNAL[dataset])})
    return rows


@display("main:Figure 4", stem="Fig4", title="Ablations, ensemble response and resource accounting",
         needs=("runs", "records:factorials", "records:input_gate", "records:baselines"))
def figure_4(ctx, out):
    deps = deployments(ctx)
    gate = ctx.require_records("input_gate")
    leds = ledgers(ctx, deps, gate)
    cells = factorial_rows(ctx)
    response = ensemble_response(deps["ogbn-arxiv"])
    hours = gpu_hours(ctx, deps, ctx.require_records("baselines"))
    wall = {}
    for dataset in DATASETS:
        wall[dataset] = 0.0 if deps[dataset]["returned"] else _require_value(
            gate, "deployed_wall_overhead_percent", dataset=dataset)
    columns_a = ["cell", "A", "O", "X", "ld_integrated_loss", "interval_low", "interval_high"]
    out.data("panel_a", columns_a, [[r[c] for c in columns_a] for r in cells])
    columns_b = ["dataset", "ensemble", "candidate_index", "risk_reduction_nats", "retained_fraction", "serial_cost",
                 "normalized_work", "selected"]
    out.data("panel_b", columns_b, [[r[c] for c in columns_b] for r in response])
    out.data("panel_c", ["dataset", "C_qC", "C_expected", "C_assure", "expected_overhead", "assure_overhead"],
             [[d, leds[d]["C_qC"], leds[d]["C_expected"], leds[d]["C_assure"], leds[d]["expected_overhead"],
               leds[d]["assure_overhead"]] for d in DATASETS])
    panel_d = []
    for d in DATASETS:
        dep, loss = deps[d], deps[d]["loss_effect"]
        route = "returns q_C" if dep["returned"] else ROUTE_WORDS[dep["deployed"]["route"]]
        panel_d.append([d, route, 100 * leds[d]["expected_overhead"], 0.0 if dep["returned"] else loss["mean"],
                        None if dep["returned"] else loss["lower"], None if dep["returned"] else loss["upper"],
                        0.0 if dep["returned"] else dep["gain_pp"]["mean"], dep["metric"]])
    out.data("panel_d", ["dataset", "route", "expected_overhead_pct", "loss_effect_nats", "loss_low_nats", "loss_high_nats",
                         "official_gain_pp", "metric"], panel_d)
    columns_e = ["dataset", "plant_training_hours", "qualification_hours", "retrained_low_gpu_hours",
                 "retrained_high_gpu_hours", "retrained_methods"]
    out.data("panel_e", columns_e, [[r[c] for c in columns_e] for r in hours])
    out.data("panel_f", ["dataset", "expected_serial_overhead_pct", "wall_time_overhead_pct"],
             [[d, 100 * leds[d]["expected_overhead"], wall[d]] for d in DATASETS])
    for row in cells:
        out.compare(S16_GROUP, ("cells", row["cell"], "log_depth_CE"), row["ld_integrated_loss"])
        out.compare(S16_GROUP, ("cells", row["cell"], "interval"), [row["interval_low"], row["interval_high"]])
    for d in DATASETS:
        for key, value in (("C_qC", leds[d]["C_qC"]), ("C_expected", leds[d]["C_expected"]), ("C_assure", leds[d]["C_assure"]),
                           ("expected_overhead_pct", 100 * leds[d]["expected_overhead"]),
                           ("assure_overhead_pct", 100 * leds[d]["assure_overhead"])):
            out.compare(T2_GROUP, (d, key), value)
        out.compare(S23_GROUP, (d, "wall_time_overhead_pct"), wall[d])
        if not deps[d]["returned"]:
            out.compare(T1_GROUP, (d, "loss_effect_nats"), deps[d]["loss_effect"]["mean"])
            out.compare(S41_GROUP, (d, "total_qualification_increment_GPU_h"), deps[d]["qualification_hours"])

    plt = pyplot()
    _style(plt)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    ax = axes[0, 0]
    x = range(len(cells))
    for i, row in enumerate(cells):
        ax.errorbar(i, row["ld_integrated_loss"], yerr=[[row["ld_integrated_loss"] - row["interval_low"]],
                                                         [row["interval_high"] - row["ld_integrated_loss"]]],
                    fmt="o", color=COLORS["ogbn-arxiv"], capsize=3, ms=5)
    if cells:
        base = next(r for r in cells if r["cell"] == "000")["ld_integrated_loss"]
        ax.axhline(base, color=MUTED, ls="--", lw=0.9)
        ax.text(0.15, base, "no component", va="bottom", fontsize=7.5, color=MUTED)
        last = next(i for i, r in enumerate(cells) if r["cell"] == "111")
        ax.text(last - 0.15, cells[last]["ld_integrated_loss"], "full\nsystem", ha="right", va="center", fontsize=7.5,
                color=COLORS["ogbn-arxiv"])
    ax.set_xticks(list(x), [r["cell"] for r in cells])
    ax.set_xlabel("cell AOX (actions, observer, risk-controlled execution)")
    ax.set_ylabel("log-depth-integrated CE (nats)")
    _panel(ax, "a", "Equal-budget $2^3$ component factorial (arxiv)")

    ax = axes[0, 1]
    usable = [r for r in response if r["retained_fraction"] is not None]
    if usable:
        ax.plot([r["ensemble"] for r in usable], [r["retained_fraction"] for r in usable], "o-", color=COLORS["ogbn-arxiv"],
                label="retained risk effect")
    ax.plot([r["ensemble"] for r in response], [r["normalized_work"] for r in response], "s-", color=COLORS["ogbn-proteins"],
            label="normalized model work")
    selected = next((r for r in response if r["selected"]), None)
    if selected is not None:
        ax.axvline(selected["ensemble"], color=MUTED, ls="--", lw=0.9)
        ax.text(selected["ensemble"], 0.02, f" selected\n E = {selected['ensemble']}", fontsize=7.5, color=MUTED)
    ax.set_xticks([r["ensemble"] for r in response])
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("ensemble size E")
    ax.set_ylabel("fraction of maximum")
    ax.legend(frameon=False, fontsize=7.5, loc="center right")
    _panel(ax, "b", "Ensemble size: saturation versus work")

    ax = axes[0, 2]
    positions = np.arange(len(DATASETS))
    width = 0.38
    ax.bar(positions - width / 2, [leds[d]["C_qC"] for d in DATASETS], width, color=COLORS["ogbn-proteins"], label="$q_C$ alone")
    ax.bar(positions + width / 2, [leds[d]["C_expected"] for d in DATASETS], width, color=COLORS["ogbn-arxiv"],
           label="CROSS-MORSE, expected (Eq. 32)")
    ax.bar(positions + width / 2, [leds[d]["C_assure"] - leds[d]["C_expected"] for d in DATASETS], width,
           bottom=[leds[d]["C_expected"] for d in DATASETS], color="white", edgecolor=COLORS["ogbn-arxiv"], hatch="///",
           label="to two-branch assurance cost $C_{assure}$")
    for i, d in enumerate(DATASETS):
        returned = deps[d]["returned"]
        ax.text(i + width / 2, leds[d]["C_expected"], _pct(leds[d]["expected_overhead"]), ha="center",
                va="bottom" if returned else "top", fontsize=6.5, color=INK if returned else "white", fontweight="bold")
        if not returned:
            ax.text(i + width / 2, leds[d]["C_assure"], _pct(leds[d]["assure_overhead"]), ha="center",
                    va="bottom", fontsize=6.5, color=MUTED)
    ax.set_xticks(positions, [SHORT[d] for d in DATASETS], rotation=30, ha="right")
    ax.set_ylabel("serial compute / one nominal execution")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    _panel(ax, "c", "Serial compute per dataset")

    ax = axes[1, 0]
    route_colors = {"returns q_C": COLORS["ogbn-proteins"], "radius": COLORS["ogbn-arxiv"], "one-step": "#6a55d0",
                    "MPC-RH": COLORS["Roman-empire"]}
    for row in panel_d:
        d, route, overhead, effect, low, high, gain, metric = row
        color = route_colors.get(route, MUTED)
        if low is not None:
            ax.plot([overhead, overhead], [low, high], color=color, lw=2.0)
        ax.plot(overhead, effect, "o", color=color, ms=7, mec="white", mew=0.6, zorder=3)
        unit = METRIC_LABELS[metric][0]
        text = (f"{SHORT[d]}, 0" if route == "returns q_C" else
                f"{SHORT[d]}, {gain / 100:+.4f} AP" if metric == "ap" else f"{SHORT[d]}, {gain:+.2f} {unit}")
        ax.text(overhead + 1.0, effect, text, fontsize=7, color=MUTED, va="center")
    for route, color in route_colors.items():
        if any(row[1] == route for row in panel_d):
            ax.plot([], [], "o", color=color, label={"returns q_C": "returns $q_C$", "radius": "radius feedback",
                                                     "one-step": "one-step feedback", "MPC-RH": "MPC-RH replanning"}[route])
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlabel("expected overhead over $q_C$ (%); Eq. 32")
    ax.set_ylabel("CE or BCE effect vs $q_C$ (nats)")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    _panel(ax, "d", "Gain and expected overhead, jointly")

    ax = axes[1, 1]
    for i, row in enumerate(hours):
        ax.bar(i, row["plant_training_hours"], 0.6, color="#d6d5cf", label="plant training (run train stage)" if i == 0 else None)
        ax.bar(i, row["qualification_hours"], 0.6, bottom=row["plant_training_hours"], color=COLORS["ogbn-arxiv"],
               label="controller qualification increment" if i == 0 else None)
        ax.text(i, row["plant_training_hours"] + row["qualification_hours"], f"+{row['qualification_hours']:.1f}",
                ha="center", va="bottom", fontsize=6.5, color=MUTED)
        ax.plot([i + 0.36, i + 0.36], [row["retrained_low_gpu_hours"], row["retrained_high_gpu_hours"]],
                color=COLORS["Roman-empire"], lw=4, solid_capstyle="butt",
                label="retrained adaptive architecture (range)" if i == 0 else None)
    ax.set_xticks(range(len(hours)), [SHORT[r["dataset"]] for r in hours], rotation=30, ha="right")
    tallest = max([r["plant_training_hours"] + r["qualification_hours"] for r in hours] +
                  [r["retrained_high_gpu_hours"] for r in hours])
    ax.set_ylim(0, 1.4 * tallest)
    ax.set_ylabel("GPU-hours")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    _panel(ax, "e", "One-off cost: qualification versus retraining")

    ax = axes[1, 2]
    for i, d in enumerate(DATASETS):
        expected = 100 * leds[d]["expected_overhead"]
        ax.bar(i - 0.2, expected, 0.4, color="#d6d5cf", label="expected serial overhead (Table 2)" if i == 0 else None)
        ax.bar(i + 0.2, wall[d], 0.4, color=COLORS["ogbn-arxiv"], label="wall-time overhead" if i == 0 else None)
        ax.text(i + 0.2, wall[d], f"{wall[d]:.0f}%", ha="center", va="bottom", fontsize=6.5, color=MUTED)
    ax.set_xticks(range(len(DATASETS)), [SHORT[d] for d in DATASETS], rotation=30, ha="right")
    ax.set_ylabel("overhead over $q_C$ (%)")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    _panel(ax, "f", "Wall-time overhead per dataset")
    if ctx.synthetic:
        fig.text(0.5, 1.0, "SYNTHETIC SOFTWARE EXAMPLE — not benchmark results", ha="center", va="bottom", color="#b00020")
    fig.tight_layout()
    out.figure(fig)
    for dataset in DATASETS:
        out.uses("run", dataset, ["select/decision.json", "select/candidates.json", "admit/decision.json",
                                  "final/results_summary.json", "stages/train.json", "stages/explore.json",
                                  "stages/fit.json", "stages/select.json", "stages/confirm.json", "stages/admit.json"])
    for name in ("factorials", "input_gate", "baselines"):
        out.uses("records", name, ["summary.csv"])


# ---------------------------------------------------------------------------
# Figure 5
# ---------------------------------------------------------------------------

def _score_points(dep: dict, name: str) -> float:
    """Official test score of a model of the run in points (accuracy/ROC-AUC/F1 percent, AP x 100)."""
    model = dep["models"].get(name) or {}
    value = _f(model.get("metric_percent"))
    if value is None and _f(model.get("metric")) is not None:
        value = 100.0 * float(model["metric"])
    if value is None:
        raise DisplayInputError(f"{dep['dataset']}: final/results_summary.json lacks the {name} official score")
    return value


def method_map(ctx, deps, leds, baselines) -> list[dict]:
    """Panel 5a: official score minus the full-depth plant against serial compute relative to the plant."""
    from cm_campaigns.contract import EXTERNAL
    rows = []
    for dataset in DATASETS:
        dep, led = deps[dataset], leds[dataset]
        candidates = dep["members"][0].json("select/candidates.json")
        plant = next((c for c in candidates if c["route"] == "nominal" and c["ensemble"] == dep["qC"]["ensemble"]), None)
        if plant is None or not _f(plant.get("cost")):
            raise DisplayInputError(f"{dataset}: no nominal full-depth candidate at the comparator's ensemble size")
        c_plant = float(plant["cost"])
        plant_score = _score_points(dep, "qN")
        rows.append({"dataset": dataset, "policy": "qC", "method": route_identity(dep["qC"], style="table2"),
                     "relative_cost": led["C_qC"] / c_plant, "relative_cost_high": None,
                     "score_gain_points": _score_points(dep, "qC") - plant_score})
        if not dep["returned"]:
            rows.append({"dataset": dataset, "policy": "deployed", "method": ROUTE_WORDS[dep["deployed"]["route"]],
                         "relative_cost": led["C_expected"] / c_plant, "relative_cost_high": led["C_assure"] / c_plant,
                         "score_gain_points": _score_points(dep, "qD") - plant_score})
        for method in EXTERNAL[dataset]:
            scores = [row for row in baselines.select("score", dataset=dataset, method=method) if str(row.get("seed", ""))]
            cost = _record_value(baselines, "serial_cost", dataset=dataset, method=method, seed="")
            if not scores or cost is None:
                raise DisplayInputError(f"baselines: per-seed scores and serial_cost of {method} on {dataset} are required")
            scale = 100.0 if str(scores[0].get("unit", "")).strip().upper() == "AP" else 1.0
            mean = float(np.mean([float(row["value"]) for row in scores])) * scale
            rows.append({"dataset": dataset, "policy": "retrained", "method": method, "relative_cost": cost / c_plant,
                         "relative_cost_high": None, "score_gain_points": mean - plant_score, "seeds": len(scores),
                         "score_points": mean})
    for row in rows:
        row.setdefault("seeds", None)
        row.setdefault("score_points", None)
    return rows


@display("main:Figure 5", stem="Fig5", title="Deployment effect and cost",
         needs=("runs", "records:input_gate", "records:baselines", "records:nominal_time"))
def figure_5(ctx, out):
    from cm_theory.economics import amortization_horizon, net_benefit
    deps = deployments(ctx)
    gate = ctx.require_records("input_gate")
    leds = ledgers(ctx, deps, gate)
    baselines = ctx.require_records("baselines")
    timing = ctx.require_records("nominal_time")
    theta = _theta(ctx)
    panel_a = method_map(ctx, deps, leds, baselines)
    shown = [d for d in _deployed_datasets(deps) if leds[d]["lambda_star"] is not None]
    prices = np.linspace(0.0, 10.0, 101)
    panel_c, panel_d, markers = [], [], []
    t_grid = np.logspace(-4, 2, 61)
    for d in shown:
        led, effect = leds[d], abs(deps[d]["loss_effect"]["mean"]) * 1000.0
        for price in prices:
            panel_c.append([d, "expected", price, net_benefit(effect, price, led["added_work"])])
            if d in ASSURANCE_ROWS:
                panel_c.append([d, "assurance", price, net_benefit(effect, price, led["C_assure"] - led["C_qC"])])
        hours = led["qualification_hours"]
        t_nom = _require_value(timing, "t_nom_seconds", dataset=d)
        for t in t_grid:
            panel_d.append([d, float(t), amortization_horizon(hours, theta, led["added_work"], float(t))])
        markers.append([d, t_nom, amortization_horizon(hours, theta, led["added_work"], t_nom), hours, led["added_work"]])
    columns_a = ["dataset", "policy", "method", "relative_cost", "relative_cost_high", "score_gain_points", "score_points", "seeds"]
    out.data("panel_a", columns_a, [[r[c] for c in columns_a] for r in panel_a])
    out.data("panel_b", ["dataset", "lambda_star_mnats", "lambda_low_mnats", "lambda_high_mnats", "metric_points_per_unit"],
             [[d, leds[d]["lambda_star"], *(leds[d]["lambda_range"] or (None, None)), leds[d]["metric_rate"]] for d in shown])
    out.data("panel_c", ["dataset", "cost", "price_mnats_per_nominal_execution", "net_benefit_mnats"], panel_c)
    out.data("panel_d", ["dataset", "t_nom_seconds", "M_star_inputs"], panel_d)
    out.data("panel_d_operating_points", ["dataset", "t_nom_seconds", "M_star_inputs", "qualification_hours", "added_work"],
             markers)
    from cm_campaigns.contract import EXTERNAL
    for row in panel_a:
        d = row["dataset"]
        if row["policy"] == "retrained":
            index = 7 + EXTERNAL[d].index(row["method"])
            raw = row["score_points"] / (100.0 if deps[d]["metric"] == "ap" else 1.0)
            out.compare(S15_GROUP, (d, "rows", index, "official_score"), raw)
        else:
            route = deps[d]["qC"]["route"] if row["policy"] == "qC" else deps[d]["deployed"]["route"]
            if route in S15_ROUTE_ROWS:
                raw = _score_points(deps[d], "qC" if row["policy"] == "qC" else "qD")
                out.compare(S15_GROUP, (d, "rows", S15_ROUTE_ROWS[route], "official_score"),
                            raw / 100.0 if deps[d]["metric"] == "ap" else raw)
    for d in shown:
        out.compare(T2_GROUP, (d, "lambda_star_mnats"), leds[d]["lambda_star"])
        out.compare(T2_GROUP, (d, "lambda_star_range"), leds[d]["lambda_range"])
        out.compare(T2_GROUP, (d, "lambda_star_metric_rate"), leds[d]["metric_rate"])
    for d, t_nom, m_star, *_ in markers:
        out.compare(S23B_GROUP, (d, "M_star_inputs"), _two_significant(m_star))
        if d in ("PascalVOC-SP", "COCO-SP", "Peptides-func"):
            out.compare(S23B_GROUP, (d, "t_nom_measured_ms"), 1000.0 * t_nom)
        else:
            out.compare(S23B_GROUP, (d, "t_nom_measured_s"), t_nom)

    plt = pyplot()
    _style(plt)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    for d in DATASETS:
        rows = [r for r in panel_a if r["dataset"] == d]
        color = COLORS[d]
        qc = next(r for r in rows if r["policy"] == "qC")
        ax.plot(qc["relative_cost"], qc["score_gain_points"], "D", mfc="white", mec=color, mew=1.6, ms=7, zorder=3)
        deployed = next((r for r in rows if r["policy"] == "deployed"), None)
        if deployed is not None:
            ax.plot([qc["relative_cost"], deployed["relative_cost"]], [qc["score_gain_points"], deployed["score_gain_points"]],
                    color=color, lw=1.0, alpha=0.7)
            ax.plot([deployed["relative_cost"], deployed["relative_cost_high"]], [deployed["score_gain_points"]] * 2,
                    color=color, lw=1.0, alpha=0.7)
            ax.plot(deployed["relative_cost_high"], deployed["score_gain_points"], "|", color=color, ms=12, mew=2)
            ax.plot(deployed["relative_cost"], deployed["score_gain_points"], "o", color=color, ms=7, zorder=4)
            ax.text(deployed["relative_cost"] + 0.02, deployed["score_gain_points"], SHORT[d], color=color, fontsize=7.5,
                    va="center")
        else:
            ax.text(qc["relative_cost"], qc["score_gain_points"] - 0.06, SHORT[d], color=color, fontsize=7.5, ha="center",
                    va="top")
        for r in rows:
            if r["policy"] == "retrained":
                ax.plot(r["relative_cost"], r["score_gain_points"], "o", color=color, ms=6, alpha=0.75)
    ax.plot([], [], "o", color=MUTED, label="deployed rule at expected cost; tick: assurance cost")
    ax.plot([], [], "D", mfc="white", mec=MUTED, label="strongest open-loop comparator $q_C$")
    ax.plot([], [], "o", color=MUTED, alpha=0.6, label="retrained architecture (measured serial cost)")
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xlabel("serial compute relative to the full-depth plant")
    ax.set_ylabel("official score minus plant (points; AP in pp)")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    _panel(ax, "a", "Method map: score gain against serial compute")

    ax = axes[0, 1]
    for y, d in enumerate(shown):
        led = leds[d]
        ax.barh(y, led["lambda_star"], color=COLORS[d], height=0.45)
        if led["lambda_range"]:
            lo, hi = led["lambda_range"]
            ax.errorbar(led["lambda_star"], y, xerr=[[led["lambda_star"] - lo], [hi - led["lambda_star"]]], color=MUTED,
                        capsize=4, lw=1.2)
        rate = "" if led["metric_rate"] is None else f"; {led['metric_rate']:.2f} pt per unit"
        right = led["lambda_range"][1] if led["lambda_range"] else led["lambda_star"]
        ax.text(right * 1.03 + 0.2, y, f"{led['lambda_star']:.2f} mnats{rate}", va="center", fontsize=7, color=MUTED)
    ax.set_yticks(range(len(shown)), [SHORT[d] for d in shown])
    ax.set_ylim(len(shown) - 0.5, -0.5)
    ax.set_xlabel("mnats of risk removed per nominal execution\nof added work (bars: range from the 95% interval of $\\Delta$)")
    _panel(ax, "b", "Exchange rate $\\lambda^\\star = |\\Delta|/\\Delta\\mathbb{E}C$")

    ax = axes[1, 0]
    for d in shown:
        series = [(row[2], row[3]) for row in panel_c if row[0] == d and row[1] == "expected"]
        ax.plot([s[0] for s in series], [s[1] for s in series], color=COLORS[d], label=SHORT[d])
        assurance = [(row[2], row[3]) for row in panel_c if row[0] == d and row[1] == "assurance"]
        if assurance:
            ax.plot([s[0] for s in assurance], [s[1] for s in assurance], color=COLORS[d], ls="--", lw=1.2)
    ax.axhline(0, color=INK, lw=1.0)
    values = [row[3] for row in panel_c]
    if values:
        ax.set_ylim(max(min(values), -0.9 * max(values)), 1.35 * max(values))
    ax.set_xlabel("$\\lambda$ (mnats a deployer pays per nominal execution)")
    ax.set_ylabel("net benefit over $q_C$ (mnats)")
    ax.legend(ncol=3, frameon=False, fontsize=7.5, loc="upper right")
    ax.text(0.98, 0.03, "dashed: assurance cost $C_{assure}$ (" + ", ".join(SHORT[d] for d in ASSURANCE_ROWS) + ")",
            transform=ax.transAxes, ha="right", fontsize=7, color=MUTED)
    _panel(ax, "c", "Net benefit $U(q) = |\\Delta| - \\lambda\\,\\Delta\\mathbb{E}C$ against the conversion $\\lambda$")

    ax = axes[1, 1]
    for d in shown:
        series = [(row[1], row[2]) for row in panel_d if row[0] == d]
        ax.plot([s[0] for s in series], [s[1] for s in series], color=COLORS[d], label=SHORT[d])
    groups = {}
    for d, t_nom, m_star, *_ in markers:
        ax.plot(t_nom, m_star, "o", mfc="none", mec=COLORS[d], mew=1.8, ms=9)
        groups.setdefault((f"{t_nom:.3g}", f"{m_star:.1e}"), []).append((d, t_nom, m_star))
    placed = []
    for (_, label), items in sorted(groups.items(), key=lambda item: item[1][0][1]):
        d, t_nom, m_star = items[0]
        mantissa, exponent = label.split("e")
        y = m_star
        if any(abs(math.log10(t_nom / t)) < 0.5 and abs(math.log10(m_star / m)) < 0.5 for t, m in placed):
            y = m_star / 3.0
        placed.append((t_nom, y))
        ax.text(t_nom * 1.6, y, f"{', '.join(SHORT[i[0]] for i in items)} {mantissa} × $10^{{{int(exponent)}}}$",
                fontsize=7, color=COLORS[d], va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$t_{nom}$: wall time per nominal execution (s)")
    ax.set_ylabel("$M^\\star$: inputs served before $C_{qual}/M \\leq \\theta\\,\\Delta\\mathbb{E}C$")
    ax.legend(ncol=2, frameon=False, fontsize=7.5, loc="lower left")
    ax.text(0.98, 0.97, "hollow markers: operating points at the measured $t_{nom}$", transform=ax.transAxes, ha="right",
            va="top", fontsize=7, color=MUTED)
    _panel(ax, "d", f"Amortization horizon $M^\\star$ ($\\theta$ = {theta:g})")
    if ctx.synthetic:
        fig.text(0.5, 1.0, "SYNTHETIC SOFTWARE EXAMPLE — not benchmark results", ha="center", va="bottom", color="#b00020")
    fig.tight_layout()
    out.figure(fig)
    for dataset in DATASETS:
        out.uses("run", dataset, ["select/candidates.json", "admit/decision.json", "final/results_summary.json",
                                  "stages/explore.json", "stages/fit.json", "stages/select.json",
                                  "stages/confirm.json", "stages/admit.json"])
    for name in ("input_gate", "baselines", "nominal_time"):
        out.uses("records", name, ["summary.csv"])


def _two_significant(value: float) -> float:
    if value <= 0 or not math.isfinite(value):
        return value
    exponent = math.floor(math.log10(value)) - 1
    return round(value / 10 ** exponent) * 10 ** exponent


# ---------------------------------------------------------------------------
# Figure 6
# ---------------------------------------------------------------------------

def budget_arms(budget, dataset: str) -> dict:
    arms = {}
    for arm in ARMS:
        keys = {"dataset": dataset, "arm": arm}
        if not budget.select("effect_vs_qc_nats", **keys):
            raise DisplayInputError(f"budget_matched: arm {arm} of {dataset} is not in the record set")
        row = {q: _require_value(budget, q, **keys) for q in
               ("effect_vs_qc_nats", "effect_vs_feedback_nats", "effect_vs_feedback_ci_low_nats",
                "effect_vs_feedback_ci_high_nats", "p_holm", "work_ratio", "ceiling")}
        if not row["effect_vs_feedback_ci_low_nats"] <= row["effect_vs_feedback_nats"] <= row["effect_vs_feedback_ci_high_nats"]:
            raise DisplayInputError(f"budget_matched: the {arm} interval of {dataset} does not contain its effect")
        arms[arm] = row
    ceilings = {round(row["ceiling"], 12) for row in arms.values()}
    if len(ceilings) != 1:
        raise DisplayInputError(f"budget_matched: the four arms of {dataset} do not share one ceiling")
    return arms


def _share_grid(ctx, dep: dict) -> list[float]:
    grids = ctx.constants["budget_matched_comparison_section4_table_S32_S40"]["retained_share_grid_phi_x"]
    route = dep["qC"]["route"]
    if route == "mpc_ol":
        return list(grids["MPC_OL_regimes"])
    if route in ("fixed_sequence", "phase", "best_depth", "learned_stop", "nominal"):
        return list(grids["fixed_sequence_regimes"])
    raise DisplayInputError(f"{dep['dataset']}: no registered retained-share grid for a {route} comparator")


def budget_rows(ctx, deps, leds, budget) -> list[dict]:
    """Every quantity of Figure 6a-c for one admitted dataset."""
    from cm_campaigns.contract import UNITS
    from cm_theory import economics, inference
    constants = ctx.constants
    kappa = constants["budget_matched_comparison_section4_table_S32_S40"]["ensemble_law_kappa_band"]
    seeds = constants["data_roles_table_S2"]["controller_seeds"]
    alpha = 0.05
    z = NormalDist().inv_cdf(0.975)
    rows = []
    for dataset in _deployed_datasets(deps):
        dep, led = deps[dataset], leds[dataset]
        arms = budget_arms(budget, dataset)
        g = -dep["loss_effect"]["mean"]
        if g <= 0:
            continue
        grid = _share_grid(ctx, dep)
        central = grid[len(grid) // 2]
        e3, e6 = arms["E3"], arms["E6"]
        teacher = e3["effect_vs_qc_nats"] - e3["effect_vs_feedback_nats"]
        phi = e3["effect_vs_qc_nats"] / teacher if teacher < 0 else None
        r_b = e6["effect_vs_qc_nats"] / (e6["effect_vs_qc_nats"] - e6["effect_vs_feedback_nats"])
        ceiling = e3["ceiling"]
        o = ceiling - 1.0
        spare_dec, spare_b = max(0.0, ceiling - e3["work_ratio"]), max(0.0, ceiling - e6["work_ratio"])
        l_plant = _f((dep["models"].get("qN") or {}).get("loss_nats"))
        if l_plant is None or l_plant <= 0:
            raise DisplayInputError(f"{dataset}: final/results_summary.json lacks the full-depth plant test loss")
        E = dep["qC"]["ensemble"]
        ensemble = []
        for k in kappa:
            d_inf = k * l_plant
            b_dec = economics.spare_budget_gain(d_inf, E, spare_dec)
            b_b = economics.spare_budget_gain(d_inf, E, spare_b)
            ensemble.append((1.0 - b_dec / g, r_b - (b_dec - b_b) / g))
        se = (e3["effect_vs_feedback_ci_high_nats"] - e3["effect_vs_feedback_ci_low_nats"]) / (2 * z)
        shrink = SEED_SHRINKING_SHARE[UNITS[dataset]]
        se_conservative = se * math.sqrt(1.0 - shrink + shrink / seeds)
        se_optimistic = se / math.sqrt(seeds)
        margins = [(1.0 - share) * g for share in grid]
        margin = (1.0 - central) * g
        power = lambda m, s, family=1: inference.power_of_contrast(m, s, alpha, family, 1)  # noqa: E731
        rows.append({
            "dataset": dataset, "g_nats": g, "ceiling": ceiling, "overhead_o": o,
            "feedback_low": dep["loss_effect"]["lower"] / g, "feedback_high": dep["loss_effect"]["upper"] / g,
            **{f"{arm}_effect_vs_qc_nats": arms[arm]["effect_vs_qc_nats"] for arm in ARMS},
            **{f"{arm}_work_ratio": arms[arm]["work_ratio"] for arm in ARMS},
            "phi_grid_low": min(grid), "phi_grid_central": central, "phi_grid_high": max(grid), "phi_measured": phi,
            "r_B": r_b, "decoding_overhead": e3["work_ratio"] - 1.0, "budget_aware_overhead": e6["work_ratio"] - 1.0,
            "phi_FB_ensemble_low": min(e[0] for e in ensemble), "phi_FB_ensemble_high": max(e[0] for e in ensemble),
            "phi_B_ensemble_low": min(e[1] for e in ensemble), "phi_B_ensemble_high": max(e[1] for e in ensemble),
            "l_plant_nats": l_plant, "paired_se_nats": se, "MDE_nats": inference.minimum_detectable_effect(se, alpha),
            "MDE_holm_nats": inference.minimum_detectable_effect(se, alpha, family_size=len(ARMS)),
            "margin_nats": margin, "margin_low_nats": min(margins), "margin_high_nats": max(margins),
            "power_unit": power(margin, se), "power_unit_low": power(min(margins), se),
            "power_unit_high": power(max(margins), se), "power_holm": power(margin, se, len(ARMS)),
            "seed_shrinking_share": shrink, "power_seed_conservative": power(margin, se_conservative, len(ARMS)),
            "power_seed_optimistic": power(margin, se_optimistic),
            "MDE_seed_conservative_nats": inference.minimum_detectable_effect(se_conservative, alpha, family_size=len(ARMS)),
            "MDE_seed_optimistic_nats": inference.minimum_detectable_effect(se_optimistic, alpha),
            "C_expected": led["C_expected"], "C_qC": led["C_qC"]})
    return rows


@display("main:Figure 6", stem="Fig6", title="Budget-matched comparison",
         needs=("runs", "records:input_gate", "records:budget_matched"))
def figure_6(ctx, out):
    deps = deployments(ctx)
    gate = ctx.require_records("input_gate")
    leds = ledgers(ctx, deps, gate)
    rows = budget_rows(ctx, deps, leds, ctx.require_records("budget_matched"))
    columns_b = ["dataset", "g_nats", "ceiling", "overhead_o", "phi_grid_low", "phi_grid_central", "phi_grid_high",
                 "phi_measured", "r_B", "decoding_overhead", "budget_aware_overhead", "l_plant_nats",
                 "phi_FB_ensemble_low", "phi_FB_ensemble_high", "phi_B_ensemble_low", "phi_B_ensemble_high"]
    out.data("panel_b", columns_b, [[r[c] for c in columns_b] for r in rows])
    columns_c = ["dataset", "g_nats", "margin_nats", "margin_low_nats", "margin_high_nats", "paired_se_nats", "MDE_nats",
                 "MDE_holm_nats", "power_unit", "power_unit_low", "power_unit_high", "power_holm", "seed_shrinking_share",
                 "power_seed_conservative", "power_seed_optimistic", "MDE_seed_conservative_nats", "MDE_seed_optimistic_nats"]
    out.data("panel_c", columns_c, [[r[c] for c in columns_c] for r in rows])
    frontier = []
    for r in rows:
        d = r["dataset"]
        frontier.append([d, "qC", 1.0, 0.0, None, None])
        frontier.append([d, "feedback", leds[d]["C_expected"] / leds[d]["C_qC"], -1.0, r["feedback_low"], r["feedback_high"]])
        for arm in ARMS:
            frontier.append([d, arm, r[f"{arm}_work_ratio"], r[f"{arm}_effect_vs_qc_nats"] / r["g_nats"], None, None])
    out.data("panel_a", ["dataset", "policy", "work_ratio_qC", "effect_over_g", "low_over_g", "high_over_g"], frontier)
    panel_d = []
    for d in _deployed_datasets(deps):
        tail = leds[d]["tail"]
        panel_d.append([d, 100 * leds[d]["expected_overhead"], *tail["overhead_pct"], 100 * leds[d]["assure_overhead"]])
    out.data("panel_d", ["dataset", "expected_overhead_pct", "C099_low_overhead_pct", "C099_central_overhead_pct",
                         "C099_high_overhead_pct", "assurance_overhead_pct"], panel_d)
    for r in rows:
        d = r["dataset"]
        out.compare(S46_GROUP, (d, "ceiling"), r["ceiling"])
        out.compare(S46_GROUP, (d, "feedback_effect_mnats"), -1000.0 * r["g_nats"])
        for arm, key in (("E3", "E3_predictor"), ("E4", "E4_distilled"), ("E5", "E5_no_observation_replanning"),
                         ("E6", "E6_qB_effect")):
            out.compare(S46_GROUP, (d, key), 1000.0 * r[f"{arm}_effect_vs_qc_nats"])
        out.compare(S46_GROUP, (d, "E6_overhead_pct"), 100 * r["budget_aware_overhead"])
        for key, value in (("phi_central", r["phi_measured"]), ("phi_B_contract", r["r_B"]),
                           ("phi_FB_ensemble", [r["phi_FB_ensemble_low"], r["phi_FB_ensemble_high"]]),
                           ("phi_B_ensemble", [r["phi_B_ensemble_low"], r["phi_B_ensemble_high"]]),
                           ("margin_FB_E3_mnats", 1000 * r["margin_nats"]),
                           ("margin_range", [1000 * r["margin_low_nats"], 1000 * r["margin_high_nats"]]),
                           ("MDE_unit", 1000 * r["MDE_nats"]),
                           ("MDE_seed_paired", [1000 * r["MDE_seed_conservative_nats"], 1000 * r["MDE_seed_optimistic_nats"]]),
                           ("power_unit_pct", 100 * r["power_unit"]),
                           ("power_unit_range", [100 * r["power_unit_low"], 100 * r["power_unit_high"]]),
                           ("power_seed_paired_pct", [100 * r["power_seed_conservative"], 100 * r["power_seed_optimistic"]])):
            out.compare(S47_GROUP, (d, key), value)
        for key, value in (("paired_SE_nats", r["paired_se_nats"]), ("MDE_mnats_0.05", 1000 * r["MDE_nats"]),
                           ("MDE_mnats_holm", 1000 * r["MDE_holm_nats"]), ("power_holm_step_pct", 100 * r["power_holm"])):
            out.compare(S42_GROUP, (d, key), value)
    for d, expected, low, central, high, assure in panel_d:
        out.compare(T2_GROUP, (d, "C_099_overhead_pct"), central)
        out.compare(T2_GROUP, (d, "assure_overhead_pct"), assure)

    plt = pyplot()
    _style(plt)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    ax.plot(1.0, 0.0, "D", color=MUTED, ms=7, zorder=4)
    for r in rows:
        d, color = r["dataset"], COLORS[r["dataset"]]
        x_fb = leds[d]["C_expected"] / leds[d]["C_qC"]
        ax.plot([1.0, x_fb], [0.0, -1.0], color=color, lw=0.8, alpha=0.6)
        ax.plot([x_fb, x_fb], [r["feedback_low"], r["feedback_high"]], color=color, lw=1.6)
        ax.plot(x_fb, -1.0, "o", color=color, ms=7, zorder=4)
        for arm in ARMS:
            ax.plot(r[f"{arm}_work_ratio"], r[f"{arm}_effect_vs_qc_nats"] / r["g_nats"], ARM_MARKERS[arm], mfc="white",
                    mec=color, mew=1.5, ms=8, zorder=4)
    ax.axhline(0, color=INK, lw=0.8)
    ax.axhline(-1, color=INK, lw=0.8, ls="--")
    handles = [ax.plot([], [], "o", color=COLORS[r["dataset"]], label=SHORT[r["dataset"]])[0] for r in rows]
    first = ax.legend(handles=handles, ncol=2, frameon=False, fontsize=7, loc="upper right")
    ax.add_artist(first)
    kinds = [ax.plot([], [], "D", color=MUTED, label="$q_C$")[0],
             ax.plot([], [], "o", color=MUTED, label="feedback rule at the ceiling (95% interval)")[0]]
    kinds += [ax.plot([], [], ARM_MARKERS[a], mfc="white", mec=MUTED, label=ARM_LABELS[a])[0] for a in ARMS]
    ax.legend(handles=kinds, ncol=2, frameon=False, fontsize=7, loc="lower left")
    ax.set_xlabel("serial compute relative to $q_C$ (common ceiling $c=(1+o)C_{q_C}$ per dataset)")
    ax.set_ylabel("paired loss effect vs $q_C$, in units of the deployed gain g")
    _panel(ax, "a", "Budget-matched frontier, all admitted datasets (normalized)")

    ax = axes[0, 1]
    for y, r in enumerate(rows):
        color = COLORS[r["dataset"]]
        ax.plot([r["phi_grid_low"], r["phi_grid_high"]], [y, y], color=color, lw=9, alpha=0.3, solid_capstyle="butt")
        if r["phi_measured"] is not None:
            ax.plot(r["phi_measured"], y, "o", color=color, ms=8, zorder=4)
        ax.plot([1.0, 1.0], [y - 0.25, y + 0.25], color=INK, lw=2.4)
        ax.plot([r["phi_FB_ensemble_low"], r["phi_FB_ensemble_high"]], [y + 0.1, y + 0.1], color="#52514e", lw=3.5,
                solid_capstyle="butt")
        ax.plot([r["r_B"], r["r_B"]], [y - 0.25, y + 0.25], color="#1b8a9a", lw=2.4)
    ax.plot([], [], color="#c3c2b7", lw=7, label="E3 retained-share grid (dot: measured share)")
    ax.plot([], [], color=INK, lw=2.4, label="crossover vs feedback, contract ($\\phi^\\star$ = 1)")
    ax.plot([], [], color="#52514e", lw=3.5, label="crossover vs feedback, ensemble-spend model ($\\kappa$ band)")
    ax.plot([], [], color="#1b8a9a", lw=2.4, label="crossover vs $q_B$, contract ($\\phi^\\star = r_B$)")
    ax.set_yticks(range(len(rows)), [SHORT[r["dataset"]] for r in rows])
    ax.set_ylim(-0.6, len(rows) + 1.4)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("$\\phi_x^{gain}$: share of the feedback gain retained by the E3 predictor")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    _panel(ax, "b", "Decision boundary of Prop. 10")

    ax = axes[1, 0]
    for i, r in enumerate(rows):
        color = COLORS[r["dataset"]]
        ax.bar(i - 0.2, 1000 * r["margin_nats"], 0.38, color=color)
        ax.plot([i - 0.2] * 2, [1000 * r["margin_low_nats"], 1000 * r["margin_high_nats"]], color=MUTED, lw=1.2)
        ax.text(i - 0.2, 0.15, f"{100 * r['power_unit']:.0f}%", ha="center", fontsize=7, color="white", fontweight="bold")
        ax.bar(i + 0.2, 1000 * r["MDE_nats"], 0.38, color="white", edgecolor=INK, hatch="///")
        ax.text(i + 0.2, 1000 * r["MDE_nats"], f"{100 * r['power_seed_conservative']:.0f}–{100 * r['power_seed_optimistic']:.0f}%",
                ha="center", va="bottom", fontsize=6.5, color=MUTED)
    from matplotlib.patches import Patch
    handles = [Patch(color=MUTED, label="margin FB−E3 at the ceiling (whiskers: retained-share grid); "
                                        "label: power at the registered unit"),
               Patch(facecolor="white", edgecolor=INK, hatch="///",
                     label=f"minimum detectable margin, 80% power; label: power with {seeds_label(ctx)} paired seeds")]
    ax.set_xticks(range(len(rows)), [SHORT[r["dataset"]] for r in rows])
    tops = [1000 * max(r["margin_high_nats"], r["MDE_nats"]) for r in rows]
    if tops:
        ax.set_ylim(0, 1.3 * max(tops))
    ax.set_ylabel("mnats")
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="upper left")
    _panel(ax, "c", "Power of the E3 contrast against the detectable margin")

    ax = axes[1, 1]
    shades = (0.35, 0.6, 0.8, 1.0)
    for i, (d, expected, low, central, high, assure) in enumerate(panel_d):
        values = (expected, low, central, high)
        for k, value in enumerate(values):
            ax.bar(i + (k - 1.5) * 0.18, value, 0.18, color=COLORS[d] if k else "#d6d5cf", alpha=shades[k] if k else 1.0)
        ax.plot([i - 0.4, i + 0.4], [assure, assure], color=INK, lw=1.8)
    from matplotlib.lines import Line2D
    handles = [Patch(color="#d6d5cf" if k == 0 else MUTED, alpha=1.0 if k == 0 else shades[k], label=label)
               for k, label in enumerate(("expected cost", "$C_{0.99}$, low tail", "$C_{0.99}$, central",
                                          "$C_{0.99}$, high tail"))]
    handles.append(Line2D([], [], color=INK, lw=1.8, label="two-branch assurance configuration $C_{assure}$"))
    ax.set_xticks(range(len(panel_d)), [SHORT[row[0]] for row in panel_d])
    if panel_d:
        ax.set_ylim(0, 1.35 * max(max(row[1:]) for row in panel_d))
    ax.set_ylabel("overhead over $q_C$ (%)")
    ax.legend(handles=handles, ncol=3, frameon=False, fontsize=7, loc="upper left")
    _panel(ax, "d", "Expected cost, modelled tail and the assurance configuration")
    if ctx.synthetic:
        fig.text(0.5, 1.0, "SYNTHETIC SOFTWARE EXAMPLE — not benchmark results", ha="center", va="bottom", color="#b00020")
    fig.tight_layout()
    out.figure(fig)
    for dataset in DATASETS:
        out.uses("run", dataset, ["select/candidates.json", "admit/decision.json", "final/results_summary.json"])
    out.uses("records", "input_gate", ["summary.csv"])
    out.uses("records", "budget_matched", ["summary.csv"])


def seeds_label(ctx) -> str:
    return str(ctx.constants["data_roles_table_S2"]["controller_seeds"])
