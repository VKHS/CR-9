"""Displays built from measurement-record sets and campaign exports.

Supplementary Tables S6c, S15, S15b, S16, S16b, S19, S23, S23b, S25, S26, S26b,
S37, S45, S46 and S46b, and Supplementary Figures S1, S2, S3, S4, S16 and S18.

Sources
-------
* Lifecycle runs: the comparator q_C, the deployed rule q_D and every evaluated
  configuration with its serial cost (``select/candidates.json``, deployed
  multiply-add count in nominal executions), the test-stage paired effects and
  scores (``final/results_summary.json`` through the run evidence summary) and
  the selection-stage paired effects (``select/decision.json``). A Roman-empire
  campaign enters with its costs averaged over the splits and with the campaign's
  small-sample effect summary; per-split values stay in the source data.
* Verified measurement-record sets (``records/specs.json`` and the contracts of
  ``records/specs.d``): baselines, factorials, negative_controls, input_gate,
  nominal_time, sensor_parameters, cross_backbone, ensemble_grid, inserted_site,
  budget_matched, timing_anchors, ensemble_grid_scores, runtime_scaling,
  reliability_diagnostics, confirmation_sensitivity and band_bound_tightness.
* The E3-E6 campaign export (``campaign:budget``), replayed strictly by
  ``paper.build.supplement._budget_data``; the budget_matched record values must
  agree with that replay.
* Registered constants (dataset statistics, kappa band, retained-share grids,
  measured hours per plant seed, theta of Definition 2), the configurations and
  the analytical modules ``cm_theory.economics`` and ``cm_theory.factorial``.
"""
from __future__ import annotations

import math
import re
import statistics
from collections import Counter

import numpy as np

from cm_campaigns.contract import ADMITTED, ARMS, BASELINE_SEEDS, EXTERNAL, HPO_TRIALS, METRICS, RECORD_SETS, UNITS
from cm_theory import economics, factorial
from scripts._paper_common import BCE_DATASETS, ROUTE_NAMES
from scripts.record_mapping import BUDGET_ARMS, INSERTED_SITE_ARMS

from .framework import (DATASETS, GRAPH_LEVEL, NODE_LEVEL, SHORT, DisplayInputError, display, integer, interval,
                        num, pyplot)

Z95 = 1.959963984540054
ROUTE_METHOD = {"nominal": "full-depth plant", "best_depth": "best fixed depth", "learned_stop": "learned stop",
                "phase": "optimized phase", "fixed_sequence": "best fixed sequence", "mpc_ol": "MPC-OL",
                "one_step": "one-step", "radius": "radius", "mpc_rh": "MPC-RH"}
# Internal routes tabulated in Supplementary Table S15 (the comparator's and the deployed rule's
# route are added when they lie outside this set).
S15_ROUTES = ("nominal", "best_depth", "phase", "fixed_sequence", "mpc_ol", "radius", "mpc_rh")
FAMILY_OPEN, FAMILY_FEEDBACK = "internal open loop", "internal feedback"
FAMILY_DEPLOYED, FAMILY_EXTERNAL = "CROSS-MORSE deployment", "external (re-trained)"
ARM_LABELS = {label.split()[0]: label for label in BUDGET_ARMS}
RIGHT = "→"

G_S6C = "supplementary_table_S6c_sensing_cost_identity"
G_S14B = "supplementary_table_S14b_separated_data_sensitivity"
G_S15 = "supplementary_table_S15_method_comparison"
G_S15B = "supplementary_table_S15b_per_rerun_resource_ledger"
G_S16 = "supplementary_table_S16_arxiv_factorial"
G_S16B = "supplementary_table_S16b_pascal_factorial"
G_S23 = "supplementary_table_S23_assurance_and_expected_compute"
G_S23B = "supplementary_table_S23b_nominal_execution_time_and_amortization"
G_S26B = "supplementary_table_S26b_ensemble_law_and_break_even"
G_S37 = "supplementary_table_S37_input_fallback_counts"
G_S45 = "supplementary_table_S45_inserted_site_attribution"
G_S46 = "supplementary_table_S46_budget_matched_frontier"
G_S46B = "supplementary_table_S46b_arm_level_outcomes"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _runs(datasets) -> tuple[str, ...]:
    return tuple(f"run:{d}" for d in datasets)


def _scale(value, factor):
    return None if value is None else value * factor


def _span(low, high, decimals: int, sep: str = "–", collapse: bool = True) -> str | None:
    """'a–b' at the printed precision; with ``collapse`` a single value when both ends print alike."""
    a, b = num(low, decimals), num(high, decimals)
    if a is None or b is None:
        return None
    return a if collapse and a == b else f"{a}{sep}{b}"


def _signed(value: float, decimals: int = 4) -> str:
    return num(round(value, decimals) + 0.0, decimals, signed=True)


def _sci(value, digits: int = 2) -> str | None:
    if value is None:
        return None
    mantissa, exponent = f"{value:.{digits - 1}e}".split("e")
    return f"{mantissa} × 10^{int(exponent)}"


def _round_sig(value, digits: int = 2):
    return None if value is None else float(f"{value:.{digits - 1}e}")


def _by_magnitude(values):
    """Range ends ordered from the end closest to zero, as the S46 brackets print them."""
    return sorted(values, key=abs)


def _uses_records(out, rec) -> None:
    out.uses("records", rec.name, [row.get("source_file", "") for row in rec.rows if row.get("source_file")])


def _cache(ctx) -> dict:
    return ctx.__dict__.setdefault("_records_campaigns_cache", {})


def _loss_label(dataset: str, loss_name: str | None) -> str:
    if loss_name:
        return "BCE" if "binary" in loss_name or loss_name.lower().endswith("bce") else "CE"
    return "BCE" if dataset in BCE_DATASETS else "CE"


def _open_loop_routes(ctx) -> set[str]:
    names = set(ctx.constants["policy_families_eq15_table_S10"]["open_loop"])
    return {route for route, name in ROUTE_NAMES.items() if name in names}


# ---------------------------------------------------------------------------
# Run ledger: comparator, deployed rule, serial costs and test-stage outcomes
# ---------------------------------------------------------------------------

def run_ledger(ctx, dataset: str, gate=None) -> dict:
    """Comparator, deployed rule, serial-cost ledger and test-stage outcomes of one dataset.

    Costs are the deployed multiply-add counts of ``select/candidates.json`` in nominal
    executions. With the input_gate set the expected cost follows main Eq. (32) at the
    measured fallback probability and the assurance cost is the two-branch sum; a
    returned comparator has no controller branch, so both equal C_qC. A split campaign
    enters with every cost averaged over its splits (modal route labels) and with the
    campaign's small-sample summary of the paired effects and scores.
    """
    view = ctx.require_run(dataset)
    splits = []
    for member in view.members():
        evidence = member.evidence
        raw = member.json("select/candidates.json")
        qc, qd = evidence["selection"]["qC_index"], evidence["admission"]["deployed_index"]
        if qc is None or qd is None:
            raise DisplayInputError(f"{dataset}: {member.root.name} has no locked comparator and deployed rule")
        splits.append({"name": member.root.name, "qc": raw[qc], "qd": raw[qd], "qc_index": qc, "qd_index": qd,
                       "returned": qd == qc, "candidates": raw, "evidence": evidence})
    returned = all(s["returned"] for s in splits)
    p = None
    if gate is not None and not returned:
        p = gate.require("fallback_probability", dataset=dataset)
        if not 0 <= p <= 1:
            raise DisplayInputError(f"input_gate: fallback probability of {dataset} is not a probability")
    for s in splits:
        c_qc = float(s["qc"]["cost"])
        if s["returned"]:
            s.update(C_qC=c_qc, C_ctrl=0.0, C_assure=c_qc, C_exp=c_qc)
        else:
            controller = float(s["qd"]["cost"])
            s.update(C_qC=c_qc, C_ctrl=controller, C_assure=economics.assurance_cost(controller, c_qc),
                     C_exp=None if p is None else economics.expected_cost(controller, c_qc, p))
    deployed = [s for s in splits if not s["returned"]] or splits

    def mode(values):
        return Counter(values).most_common(1)[0][0]

    def mean(key):
        values = [s[key] for s in splits]
        return None if any(v is None for v in values) else statistics.fmean(values)

    evidence = view.evidence
    final = evidence["final"]
    models = final["models"]
    metric = evidence["task"]["metric"]
    native = metric == "ap"

    def score(name):
        block = models.get(name) or {}
        return block.get("metric") if native else block.get("metric_percent")

    loss = final.get("paired_loss") or {}
    gain = final.get("paired_metric_pp") or {}
    factor = 0.01 if native else 1.0      # the AP gain is recorded in percentage points; print it in AP
    return {"dataset": dataset, "splits": splits, "returned": returned, "p": p, "campaign": view.is_campaign,
            "qc_route": mode(s["qc"]["route"] for s in splits), "E": mode(int(s["qc"]["ensemble"]) for s in splits),
            "qd_route": mode(s["qd"]["route"] for s in deployed),
            "E_qd": mode(int(s["qd"]["ensemble"]) for s in deployed),
            "C_qC": mean("C_qC"), "C_ctrl": mean("C_ctrl"), "C_assure": mean("C_assure"), "C_exp": mean("C_exp"),
            "effect": loss.get("mean"), "effect_low": loss.get("lower"), "effect_high": loss.get("upper"),
            "gain": _scale(gain.get("mean"), factor), "gain_low": _scale(gain.get("lower"), factor),
            "gain_high": _scale(gain.get("upper"), factor), "metric": metric, "native_metric": native,
            "score_qN": score("qN"), "score_qC": score("qC"), "score_qD": score("qD"),
            "loss_qN": (models.get("qN") or {}).get("loss_nats"),
            "loss_label": _loss_label(dataset, evidence["task"].get("loss_name")),
            "test_units": (models.get("qD") or {}).get("test_units")}


def route_text(ledger: dict) -> str:
    """Route/E label of Table S23: the deployed route, or qC/<route>/E for a returned comparator."""
    if ledger["returned"]:
        return f"qC/{ROUTE_NAMES[ledger['qc_route']]}/{ledger['E']}"
    return f"{ROUTE_NAMES[ledger['qd_route']]}/{ledger['E_qd']}"


def ledger_source(ledger: dict) -> list[list]:
    return [[ledger["dataset"], s["name"], s["qc"]["route"], s["qc"]["ensemble"], s["qd"]["route"], s["qd"]["ensemble"],
             s["returned"], s["C_qC"], s["C_ctrl"], s["C_assure"], s["C_exp"]] for s in ledger["splits"]]


LEDGER_COLUMNS = ["dataset", "run", "qC_route", "qC_ensemble", "qD_route", "qD_ensemble", "comparator_returned",
                  "C_qC", "C_ctrl", "C_assure", "C_expected"]


def per_member_multiplier(ledger: dict) -> float:
    """Serial cost of the deployed rule per ensemble member (q_C when the comparator is returned)."""
    values = []
    for s in ledger["splits"]:
        rule = s["qc"] if s["returned"] else s["qd"]
        values.append(float(rule["cost"]) / int(rule["ensemble"]))
    return statistics.fmean(values)


def lean_share(ledger: dict) -> float | None:
    """Lean-profile effect of the deployed rule as a share of its Full-profile effect.

    Both profiles of the deployed configuration (same route, ensemble, depth, horizon
    and kappa) carry a nested selection-stage paired effect against the comparator
    (select/decision.json); the share is Lean over Full, averaged over splits. A
    returned comparator executes no controller, so both profiles return q_C (share 1).
    None when a split lacks the Full/Lean pair or its effects.
    """
    shares = []
    for s in ledger["splits"]:
        if s["returned"]:
            shares.append(1.0)
            continue
        rule, candidates = s["qd"], s["candidates"]
        profile = rule.get("profile")
        twin_profile = {"full": "lean", "lean": "full"}.get(profile)

        def identity(c):
            return (c.get("route"), c.get("ensemble"), c.get("depth"), c.get("horizon"), c.get("kappa"))
        twins = [i for i, c in enumerate(candidates) if identity(c) == identity(rule) and c.get("profile") == twin_profile]
        if twin_profile is None or len(twins) != 1:
            return None
        effects = {c["index"]: c.get("select_delta") for c in s["evidence"]["candidates"]}
        full_index, lean_index = (s["qd_index"], twins[0]) if profile == "full" else (twins[0], s["qd_index"])
        full, lean = effects.get(full_index), effects.get(lean_index)
        if full is None or lean is None or full == 0:
            return None
        shares.append(lean / full)
    return statistics.fmean(shares)


def sparse_ratio(ctx, dataset: str) -> dict:
    """r = E / (N d) from the registered dataset statistics and the configured hidden width."""
    stats = ctx.constants["dataset_statistics_table_S6c_S23b"][dataset]
    nodes = stats.get("nodes", stats.get("mean_nodes_per_graph"))
    edges = stats.get("reported_edges", stats.get("mean_reported_edges_per_graph"))
    model = ctx.config(dataset)["model"]
    return {"nodes": nodes, "edges": edges, "d": model["hidden_dim"], "L": model["depth"],
            "native_layers": model.get("native_layers"), "r": edges / (nodes * model["hidden_dim"])}


# ---------------------------------------------------------------------------
# Baselines record set
# ---------------------------------------------------------------------------

def method_labels(ledger: dict) -> tuple[str, str | None]:
    """Method keys of the comparator and deployed rows in the baselines set (Table S15b labels)."""
    qc = f"{ROUTE_METHOD[ledger['qc_route']]} (qC)"
    qd = None if ledger["returned"] else f"{ROUTE_METHOD[ledger['qd_route']]} (deployed)"
    return qc, qd


def method_value(base, quantity: str, dataset: str, method: str):
    rows = [r for r in base.select(quantity, dataset=dataset, method=method) if not str(r.get("seed", "")).strip()]
    if len(rows) > 1:
        raise DisplayInputError(f"baselines: {quantity} of {dataset}/{method} is not unique")
    return rows[0]["value"] if rows else None


def seed_scores(base, dataset: str, method: str) -> list[float]:
    return [float(r["value"]) for r in base.select("score", dataset=dataset, method=method)
            if str(r.get("seed", "")).strip()]


def seed_sd(base, dataset: str, method: str):
    """Sample SD of the official score over the optimization seeds, in percentage points (AP points x 100)."""
    scores = seed_scores(base, dataset, method)
    if len(scores) < 2:
        return None
    return statistics.stdev(scores) * (100.0 if METRICS.get(dataset) == "ap" else 1.0)


def plant_parameters(ctx, dataset: str) -> float:
    """Parameters (millions) of one plant checkpoint from the run's model inventory."""
    view = ctx.require_run(dataset).members()[0]
    inventory = view.json("train/model_inventory.json")
    counts = {c["named_tensor_elements"] for c in inventory["checkpoints"] if "plant_seed" in c["path"]}
    if len(counts) != 1:
        raise DisplayInputError(f"{dataset}: plant checkpoints of the model inventory differ or are absent")
    return counts.pop() / 1e6


def route_cost_range(ledger: dict, route: str):
    """Serial-cost range of a route over its evaluated configurations at the comparator's ensemble size."""
    costs = [float(c["cost"]) for s in ledger["splits"] for c in s["candidates"]
             if c.get("route") == route and int(c.get("ensemble", 0)) == ledger["E"] and not c.get("execution_failure")]
    return (min(costs), max(costs)) if costs else (None, None)


def method_rows(ctx, dataset: str, base, gate) -> tuple[dict, list[dict]]:
    """Rows of Supplementary Table S15 for one dataset (also the points of Figure S4)."""
    ledger = run_ledger(ctx, dataset, gate)
    qc_label, qd_label = method_labels(ledger)
    params = plant_parameters(ctx, dataset)
    open_loop = _open_loop_routes(ctx)
    native = METRICS.get(dataset) == "ap"
    routes = [r for r in ROUTE_NAMES if r in S15_ROUTES or r in (ledger["qc_route"], ledger["qd_route"])]
    rows = []
    for route in routes:
        name = ROUTE_METHOD[route]
        is_qc = route == ledger["qc_route"]
        is_qd = route == ledger["qd_route"] and not ledger["returned"]
        returned_qc = is_qc and ledger["returned"]
        internal_family = FAMILY_OPEN if route in open_loop else FAMILY_FEEDBACK
        if returned_qc:
            label, family, key = f"{name} (qC, returned)", FAMILY_DEPLOYED, qc_label
        elif is_qc:
            label, family, key = qc_label, internal_family, qc_label
        elif is_qd:
            label, family, key = qd_label, FAMILY_DEPLOYED, qd_label
        else:
            label, family, key = name, internal_family, name
        if is_qc or is_qd:
            low = high = ledger["C_qC"] if is_qc else ledger["C_ctrl"]
        else:
            low, high = route_cost_range(ledger, route)
        if returned_qc or is_qd:
            c_exp, c_assure, role = ledger["C_exp"], ledger["C_assure"], "deployed"
        elif is_qc:
            c_exp = c_assure = ledger["C_qC"]
            role = "qC"
        else:
            c_exp = c_assure = None
            role = "route"
        if route == "nominal" and not is_qc:
            score, origin = ledger["score_qN"], "run: qN"
        elif is_qc:
            score, origin = ledger["score_qC"], "run: qC"
        elif is_qd:
            score, origin = ledger["score_qD"], "run: qD"
        else:
            scores = seed_scores(base, dataset, key)
            score, origin = (statistics.fmean(scores), f"baselines: mean of {len(scores)} seeds") if scores else (None, "")
        hpo = method_value(base, "hpo_trials", dataset, key)
        rows.append({"dataset": dataset, "route": route, "method": label, "key": key, "family": family, "role": role,
                     "params": params, "params_external": None, "E": ledger["E"],
                     "hpo": HPO_TRIALS if hpo is None else hpo, "gpu_hours": method_value(base, "gpu_hours", dataset, key),
                     "cost_low": low, "cost_high": high, "C_exp": c_exp, "C_assure": c_assure, "score": score,
                     "score_origin": origin, "sd": seed_sd(base, dataset, key), "native": native})
    for method in EXTERNAL[dataset]:
        scores = seed_scores(base, dataset, method)
        cost = method_value(base, "serial_cost", dataset, method)
        rows.append({"dataset": dataset, "route": None, "method": method, "key": method, "family": FAMILY_EXTERNAL,
                     "role": "external", "params": None,
                     "params_external": method_value(base, "params_millions", dataset, method), "E": ledger["E"],
                     "hpo": method_value(base, "hpo_trials", dataset, method),
                     "gpu_hours": method_value(base, "gpu_hours", dataset, method), "cost_low": cost, "cost_high": cost,
                     "C_exp": None, "C_assure": None, "score": statistics.fmean(scores) if scores else None,
                     "score_origin": f"baselines: mean of {len(scores)} seeds" if scores else "",
                     "sd": seed_sd(base, dataset, method), "native": native})
    return ledger, rows


# ---------------------------------------------------------------------------
# Supplementary Table S6c
# ---------------------------------------------------------------------------

@display("supplementary:Table S6c", stem="S6c", title="Sensing-work identity and executed settings",
         needs=_runs(ADMITTED) + ("records:sensor_parameters",))
def table_s6c(ctx, out):
    sensors = ctx.require_records("sensor_parameters")
    rows, source = [], []
    for dataset in ADMITTED:
        ledger = run_ledger(ctx, dataset)
        stats = sparse_ratio(ctx, dataset)
        degree = ctx.config(dataset)["sensor"]["degree"]
        r = stats["r"]
        share = degree * r / (1.0 + r)
        m = per_member_multiplier(ledger)
        residual = m - 1.0 - share
        admissible = min(1.0, (m - 1.0) / share)
        k = sensors.require("sketch_dim", dataset=dataset)
        period = sensors.require("sensing_period", dataset=dataset)
        f_res = sensors.require("residual_fraction", dataset=dataset)
        sensor = "full state at every site" if k == 0 else f"reduced sensor with k = {k:.0f}, P_s = {period:.0f}"
        rows.append([dataset, num(r, 3), num(m, 3), num(share, 3), num(share / 2.0, 3), num(residual, 3, signed=True),
                     f"{num(admissible, 3)} ({'full state' if admissible >= 1.0 else 'reduced sensor'})",
                     f"{k:.0f}, {period:.0f}, {f_res:.2g}"])
        source.append([dataset, stats["edges"], stats["nodes"], stats["d"], r, degree, m, share, share / 2.0, residual,
                       admissible, k, period, f_res])
        for key, value in (("r", r), ("measured_multiplier_m", m), ("full_state_sensing_share_8r_over_1_plus_r", share),
                           ("at_P_s_2", share / 2.0), ("residual", residual), ("admissible_k_over_d_per_P_s", admissible),
                           ("sensor", sensor), ("executed_sketch_width_k", k), ("executed_sensing_period_P_s", period),
                           ("executed_resolvent_fraction_f_res", f_res)):
            out.compare(G_S6C, (dataset, key), value)
        out.uses("run", dataset, ["select/candidates.json", "final/results_summary.json"])
    _uses_records(out, sensors)
    out.table(["Dataset", "r", "m (Table S23)", f"{degree}r/(1 + r)", "at Ps = 2", "residual", "admissible (k/d)/Ps",
               "executed sensor (k, Ps, f_res)"], rows,
              notes=["r = E/(N d) from the registered dataset statistics and the configured hidden width d; m = serial "
                     "cost of the deployed rule per ensemble member (run cost ledger); residual = m - 1 - 8r/(1 + r); "
                     "admissible (k/d)/Ps = min(1, (m - 1)/(8r/(1 + r))); executed (k, Ps, f_res) from the "
                     "sensor_parameters record set."])
    out.data("sensing_identity", ["dataset", "edges", "nodes", "hidden_dim", "r", "sensor_degree", "m_per_member",
                                  "full_state_share", "share_at_Ps_2", "residual", "admissible_ratio", "sketch_dim_k",
                                  "sensing_period_Ps", "residual_fraction_f_res"], source)


# ---------------------------------------------------------------------------
# Supplementary Tables S15 and S15b
# ---------------------------------------------------------------------------

@display("supplementary:Table S15", stem="S15", title="Method scores and resource ranges",
         needs=("runs", "records:baselines", "records:input_gate"))
def table_s15(ctx, out):
    base = ctx.require_records("baselines")
    gate = ctx.require_records("input_gate")
    rows, source = [], []
    for dataset in DATASETS:
        ledger, entries = method_rows(ctx, dataset, base, gate)
        sds = [e["sd"] for e in entries if e["sd"] is not None]
        sd_text = _span(min(sds), max(sds), 2) if sds else None
        for index, e in enumerate(entries):
            score_decimals = 3 if e["native"] else 2
            if e["role"] == "deployed":
                c_exp, c_assure = num(e["C_exp"], 2), num(e["C_assure"], 1)
            elif e["role"] == "qC":
                c_exp, c_assure = num(e["C_exp"], 1), num(e["C_assure"], 1)
            else:
                c_exp = c_assure = "= Cctrl"
            params = num(e["params_external"], 2) if e["role"] == "external" else num(e["params"], 2)
            rows.append([dataset, e["method"], e["family"], params, integer(e["E"]), integer(e["hpo"]),
                         num(e["gpu_hours"], 2), _span(e["cost_low"], e["cost_high"], 1), c_exp, c_assure,
                         num(e["score"], score_decimals), sd_text])
            source.append([dataset, e["method"], e["family"], e["key"], e["params"], e["params_external"], e["E"],
                           e["hpo"], e["gpu_hours"], e["cost_low"], e["cost_high"], e["C_exp"], e["C_assure"],
                           e["score"], e["score_origin"], e["sd"]])
            for key, value in (("method", e["method"]), ("family", e["family"]), ("official_score", e["score"]),
                               ("C_ctrl", _span(e["cost_low"], e["cost_high"], 1, sep="-")),
                               ("GPU_h", e["gpu_hours"]), ("C_expected", e["C_exp"]), ("C_assure", e["C_assure"]),
                               ("params_M", e["params_external"])):
                out.compare(G_S15, (dataset, "rows", index, key), value)
        out.compare(G_S15, (dataset, "plant_params_M"), entries[0]["params"])
        if sds:
            out.compare(G_S15, (dataset, "seed_SD_range_pp"), [min(sds), max(sds)])
        out.uses("run", dataset, ["select/candidates.json", "final/results_summary.json", "train/model_inventory.json"])
    _uses_records(out, base)
    out.table(["Dataset", "method", "family", "params (M)", "E", "HPO trials", "GPU-h", "Cctrl", "Cexp", "Cassure",
               "score", f"SD ({BASELINE_SEEDS} seeds)"], rows,
              notes=["Internal routes: parameters of one plant checkpoint (train/model_inventory.json) and the serial-cost "
                     "range over the route's evaluated configurations at the comparator's ensemble size "
                     "(select/candidates.json); qC, the deployed rule and the full-depth plant carry their test scores "
                     "and the cost ledger of Table S23. Retrained architectures and the other internal routes: "
                     "parameters, HPO trials, GPU-hours, serial cost and the per-seed scores of the baselines record set "
                     "(score = mean over seeds). SD: range over methods of the per-seed score SD (AP in percentage points)."])
    out.data("methods", ["dataset", "method", "family", "baselines_key", "plant_params_M", "external_params_M", "E",
                         "hpo_trials", "gpu_hours", "C_ctrl_low", "C_ctrl_high", "C_expected", "C_assure",
                         "official_score", "score_origin", "seed_sd_pp"], source)


@display("supplementary:Table S15b", stem="S15b", title="Per-rerun resource accounting",
         needs=("runs", "records:baselines"))
def table_s15b(ctx, out):
    base = ctx.require_records("baselines")
    rows, source = [], []
    for dataset in DATASETS:
        ledger = run_ledger(ctx, dataset)
        qc, qd = method_labels(ledger)
        methods = [qc] + ([qd] if qd else []) + list(EXTERNAL[dataset])
        for index, method in enumerate(methods):
            serial = method_value(base, "serial_cost", dataset, method)
            wall = method_value(base, "wall_time", dataset, method)
            gpu = method_value(base, "gpu_hours", dataset, method)
            sd = seed_sd(base, dataset, method)
            rows.append([dataset, method, num(serial, 1), num(wall, 2), num(gpu, 2), num(sd, 2)])
            source.append([dataset, method, serial, wall, gpu, sd, len(seed_scores(base, dataset, method))])
            for key, value in (("method", method), ("serial_cost_C_ctrl", serial), ("wall_time_nominal_1", wall),
                               ("GPU_h", gpu), ("SD_12_seeds", sd)):
                out.compare(G_S15B, (dataset, index, key), value)
        out.uses("run", dataset, ["select/candidates.json"])
    _uses_records(out, base)
    out.table(["Dataset", "method", "serial cost Cctrl", "wall time (nominal = 1)", "GPU-h",
               f"SD ({BASELINE_SEEDS} seeds)"], rows,
              notes=["Rows: the run's comparator and deployed rule (labelled by route) and the registered retrained "
                     "baselines of the dataset; every value from the baselines record set (E17 timing ratios, scheduler "
                     "GPU-hours, sample SD of the per-seed official scores; AP SD in percentage points)."])
    out.data("ledger", ["dataset", "method", "serial_cost", "wall_time", "gpu_hours", "seed_sd_pp", "seeds"], source)


# ---------------------------------------------------------------------------
# Supplementary Tables S16 and S16b
# ---------------------------------------------------------------------------

FACTORIAL_LAYOUT = {
    "ogbn-arxiv": {"stem": "S16", "group": G_S16, "order": ("000", "100", "010", "001", "110", "101", "011", "111"),
                   "names": ("actions", "observer", "risk_controlled_execution"), "metric": "official metric"},
    "PascalVOC-SP": {"stem": "S16b", "group": G_S16B, "order": ("000", "001", "010", "011", "100", "101", "110", "111"),
                     "names": ("spectral_information", "conditioned_observer", "risk_controlled_execution"),
                     "metric": "official metric (F1)"},
}


def factorial_cells(ctx, dataset: str) -> list[dict]:
    """The eight validated cells of one factorial (paper.build.supplement._factorial_data)."""
    from paper.build.supplement import _factorial_data
    records = ctx.require_records("factorials")
    try:
        rows = _factorial_data(records.rows, ctx.record_specs)
    except ValueError as exc:
        raise DisplayInputError(f"factorials: {exc}") from exc
    return [row for row in rows if row["dataset"] == dataset]


def factorial_contrasts(cells: dict, half_widths: dict) -> dict:
    """Effect-coded effects, baseline-anchored differences and the independence half-width (cm_theory.factorial)."""
    effects = factorial.all_effects(cells)
    anchored = {factors: factorial.anchored_difference(cells, factors) for factors in effects if len(factors) > 1}
    return {"effects": effects, "anchored": anchored, "full_minus_none": factorial.full_minus_none(cells),
            "half_width": factorial.effect_half_width(half_widths)}


def factors_names(dataset: str) -> tuple[str, ...]:
    from paper.build.supplement import FACTORIAL_DATASETS
    return FACTORIAL_DATASETS[dataset][2]


def factors_label(factors: tuple[int, ...], labels) -> str:
    return "".join(labels[j] for j in factors)


def _factorial_display(ctx, out, dataset: str):
    layout = FACTORIAL_LAYOUT[dataset]
    data = {row["cell"]: row for row in factorial_cells(ctx, dataset)}
    factors = labels = list(factors_names(dataset))
    rows, source = [], []
    for cell in layout["order"]:
        row = data[cell]
        rows.append([cell, *[str(row[f]) for f in factors], num(row["ld_integrated_loss"], 4),
                     interval(row["interval_low"], row["interval_high"], 4), num(row["official_metric"], 2),
                     num(row["activity"], 2), num(row["relative_time"], 2)])
        source.append([cell, *[row[f] for f in factors], row["ld_integrated_loss"], row["interval_low"],
                       row["interval_high"], row["official_metric"], row["activity"], row["relative_time"],
                       row["relative_time_unit"]])
        for key, value in (("log_depth_CE", row["ld_integrated_loss"]),
                           ("interval", [row["interval_low"], row["interval_high"]]),
                           ("official_metric", row["official_metric"]), ("activity", row["activity"]),
                           ("relative_time", row["relative_time"])):
            out.compare(layout["group"], ("cells", cell, key), value)
    cells = {cell: data[cell]["ld_integrated_loss"] for cell in data}
    half = {cell: (data[cell]["interval_high"] - data[cell]["interval_low"]) / 2.0 for cell in data}
    contrasts = factorial_contrasts(cells, half)
    effects, names = contrasts["effects"], layout["names"]
    for j, name in enumerate(names):
        out.compare(layout["group"], ("effect_coded_main_effects_nats", name), effects[(j,)])
    out.compare(layout["group"], ("effect_coded_main_effects_nats", "half_width_95"), contrasts["half_width"])
    interactions = {f: v for f, v in effects.items() if len(f) > 1}
    if dataset == "ogbn-arxiv":
        out.compare(layout["group"], ("pairwise_interactions_nats",), [effects[(0, 1)], effects[(0, 2)], effects[(1, 2)]])
        out.compare(layout["group"], ("three_way_interaction_nats",), effects[(0, 1, 2)])
        out.compare(layout["group"], ("full_system_minus_no_component_nats",), contrasts["full_minus_none"])
        for combination, value in contrasts["anchored"].items():
            out.compare(layout["group"], ("baseline_anchored_finite_differences",
                                          "I_" + factors_label(combination, labels)), value)
    else:
        largest = max(abs(v) for v in interactions.values())
        text = ("within 1e-4 nats of zero at the tabulated precision" if round(largest, 4) <= 1e-4
                else f"largest |interaction| {largest:.4f} nats")
        out.compare(layout["group"], ("interactions",), text)
    note = ("Effect-coded main effects " + ", ".join(f"{labels[j]} {_signed(effects[(j,)])}" for j in range(3))
            + f" nats (95% half-width {contrasts['half_width']:.4f} from the cell intervals under independent cells); "
            + "interactions " + ", ".join(f"{factors_label(f, labels)} {_signed(v)}" for f, v in interactions.items())
            + "; baseline-anchored differences "
            + ", ".join(f"I_{factors_label(f, labels)} {_signed(v)}" for f, v in contrasts["anchored"].items())
            + f"; full system minus no component {_signed(contrasts['full_minus_none'])} nats.")
    _uses_records(out, ctx.require_records("factorials"))
    out.table(["cell", *factors, "log-depth-integrated CE", "95% interval", layout["metric"], "activity",
               "relative time"], rows, notes=[note])
    out.data("cells", ["cell", *factors, "ld_integrated_loss_nats", "interval_low_nats", "interval_high_nats",
                       "official_metric", "activity", "relative_time", "relative_time_unit"], source)
    out.data("contrasts", ["contrast", "kind", "value_nats"],
             [[factors_label(f, labels), "effect-coded", v] for f, v in effects.items()]
             + [[factors_label(f, labels), "baseline-anchored", v] for f, v in contrasts["anchored"].items()]
             + [["full minus none", "difference", contrasts["full_minus_none"]],
                ["half-width (independent cells)", "interval", contrasts["half_width"]]])


@display("supplementary:Table S16", stem="S16", title="arxiv component factorial", needs=("records:factorials",))
def table_s16(ctx, out):
    _factorial_display(ctx, out, "ogbn-arxiv")


@display("supplementary:Table S16b", stem="S16b", title="Pascal component factorial", needs=("records:factorials",))
def table_s16b(ctx, out):
    _factorial_display(ctx, out, "PascalVOC-SP")


# ---------------------------------------------------------------------------
# Supplementary Table S19
# ---------------------------------------------------------------------------

@display("supplementary:Table S19", stem="S19", title="Negative controls", needs=("records:negative_controls",))
def table_s19(ctx, out):
    controls = ctx.require_records("negative_controls")
    definitions = ctx.constants.get("negative_controls_table_S19", {})
    rows, source = [], []
    for (control,) in RECORD_SETS["negative_controls"]["coverage_rows"]:
        activity = controls.require("residual_activity", control=control)
        fallback = controls.require("input_fallback", control=control)
        harm_rows = controls.select("worst_pre_fallback_harm", control=control)
        if len(harm_rows) != 1:
            raise DisplayInputError(f"negative_controls: worst pre-fallback harm of {control} is absent or not unique")
        harm, unit = harm_rows[0]["value"], (harm_rows[0].get("unit") or "nats").strip()
        definition = definitions.get(control, {})
        rows.append([control, definition.get("construction"), num(activity, 3), num(fallback, 3),
                     f"{num(harm, 4)} {unit}", definition.get("expected_conclusion")])
        source.append([control, activity, fallback, harm, unit])
    _uses_records(out, controls)
    notes = ["Residual activity, input fallback and worst pre-fallback harm from the negative_controls record set."]
    if not definitions:
        notes.append("Construction and expected conclusion are protocol definitions that paper/registered_constants.json "
                     "does not register (key negative_controls_table_S19); they print as missing.")
    out.table(["control", "construction", "residual activity", "input fallback", "worst pre-fallback harm",
               "expected conclusion"], rows, notes=notes)
    out.data("controls", ["control", "residual_activity", "input_fallback", "worst_pre_fallback_harm", "harm_unit"],
             source)


# ---------------------------------------------------------------------------
# Supplementary Tables S23 and S23b
# ---------------------------------------------------------------------------

def qualification_hours(base, ledger: dict):
    """Scheduler GPU-hours of the deployed rule (the returned comparator when no controller is deployed)."""
    qc, qd = method_labels(ledger)
    return method_value(base, "gpu_hours", ledger["dataset"], qd or qc)


@display("supplementary:Table S23", stem="S23", title="Expected and assurance cost",
         needs=("runs", "records:input_gate", "records:baselines"))
def table_s23(ctx, out):
    gate = ctx.require_records("input_gate")
    base = ctx.require_records("baselines")
    rows, source, ledgers = [], [], []
    for dataset in DATASETS:
        ledger = run_ledger(ctx, dataset, gate)
        ledgers += ledger_source(ledger)
        r = sparse_ratio(ctx, dataset)["r"]
        overhead = 0.0 if ledger["returned"] else gate.require("deployed_wall_overhead_percent", dataset=dataset)
        gpu, cpu = gate.value("gpu_memory_gib", dataset=dataset), gate.value("cpu_memory_gib", dataset=dataset)
        hours = qualification_hours(base, ledger)
        lean = lean_share(ledger)
        memory = None if gpu is None or cpu is None else f"{gpu:.1f}/{cpu:.1f}"
        rows.append([dataset, route_text(ledger), num(ledger["C_ctrl"], 3), num(ledger["C_qC"], 3),
                     "— (global return)" if ledger["returned"] else num(ledger["p"], 3), num(ledger["C_assure"], 3),
                     num(ledger["C_exp"], 3), f"{overhead:.0f}%", num(r, 3), memory, num(hours, 1), num(lean, 3)])
        source.append([dataset, route_text(ledger), ledger["C_ctrl"], ledger["C_qC"], ledger["p"], ledger["C_assure"],
                       ledger["C_exp"], overhead, r, gpu, cpu, hours, lean])
        for key, value in (("route_E", route_text(ledger)), ("controller_cost", ledger["C_ctrl"]),
                           ("comparator_cost", ledger["C_qC"]), ("fallback_probability", ledger["p"]),
                           ("assurance_serial_cost", ledger["C_assure"]), ("expected_serial_cost", ledger["C_exp"]),
                           ("wall_time_overhead_pct", overhead), ("r", r), ("GPU_GiB", gpu), ("CPU_GiB", cpu),
                           ("qualification_GPU_h_cross_table", hours), ("lean_effect", lean)):
            out.compare(G_S23, (dataset, key), value)
        out.uses("run", dataset, ["select/candidates.json", "select/decision.json"])
    _uses_records(out, gate)
    _uses_records(out, base)
    out.table(["Dataset", "route/E", "controller cost", "comparator cost", "fallback probability",
               "assurance serial cost", "expected serial cost", "wall-time overhead", "r", "GPU/CPU GiB",
               "qualification GPU-h", "Lean effect"], rows,
              notes=["Serial costs in nominal executions from the runs' cost ledgers; C_assure = C_ctrl + C_qC; "
                     "C_expected = (1 - p) C_ctrl + p C_qC (main Eq. 32) at the fallback probability p of the "
                     "input_gate set; a returned comparator executes no controller. Wall-time overhead and memory from "
                     "input_gate, qualification GPU-hours from the baselines set, r = E/(N d). Lean effect: "
                     "selection-stage paired effect of the Lean profile of the deployed configuration over that of "
                     "its Full profile (select/decision.json)."])
    out.data("ledger", ["dataset", "route_E", "controller_cost", "comparator_cost", "fallback_probability",
                        "assurance_cost", "expected_cost", "wall_time_overhead_percent", "r", "gpu_memory_gib",
                        "cpu_memory_gib", "qualification_gpu_hours", "lean_effect_share"], source)
    out.data("runs", LEDGER_COLUMNS, ledgers)


def inserted_site_plant(stats: dict) -> bool:
    return stats["native_layers"] is not None and stats["native_layers"] < stats["L"]


def anchor_reference(ctx, dataset: str) -> str:
    """Dataset whose timing anchors bracket t_nom: an earlier graph-level plant with the same model configuration
    and a mean graph size within 1% shares its anchor (COCO-SP takes the PascalVOC-SP anchor)."""
    if dataset not in GRAPH_LEVEL:
        return dataset
    stats = ctx.constants["dataset_statistics_table_S6c_S23b"]
    for other in GRAPH_LEVEL[:GRAPH_LEVEL.index(dataset)]:
        same_model = ctx.config(other)["model"] == ctx.config(dataset)["model"]
        a, b = stats[other]["mean_nodes_per_graph"], stats[dataset]["mean_nodes_per_graph"]
        if same_model and abs(a - b) <= 0.01 * max(a, b):
            return other
    return dataset


def amortization_theta(ctx) -> float:
    text = ctx.constants["deployment_economics_definition2"]["amortization"]
    match = re.search(r"theta = ([0-9.]+)", text)
    if not match:
        raise DisplayInputError("registered constants do not state theta of main Definition 2")
    return float(match.group(1))


@display("supplementary:Table S23b", stem="S23b", title="Nominal execution work, timing and amortization",
         needs=_runs(ADMITTED) + ("records:input_gate", "records:baselines", "records:nominal_time",
                                  "records:timing_anchors"))
def table_s23b(ctx, out):
    gate, base = ctx.require_records("input_gate"), ctx.require_records("baselines")
    timing, anchors = ctx.require_records("nominal_time"), ctx.require_records("timing_anchors")
    theta = amortization_theta(ctx)
    rows, source = [], []
    for dataset in ADMITTED:
        ledger = run_ledger(ctx, dataset, gate)
        stats = sparse_ratio(ctx, dataset)
        graph = dataset in GRAPH_LEVEL
        unit, factor = ("ms", 1000.0) if graph else ("s", 1.0)
        w_nom = None if inserted_site_plant(stats) else stats["L"] * (stats["nodes"] * stats["d"] ** 2
                                                                       + stats["edges"] * stats["d"])
        through = anchors.value("throughput_anchor_seconds", dataset=dataset)
        low = anchors.require("training_anchor_low_seconds", dataset=dataset)
        high = anchors.require("training_anchor_high_seconds", dataset=dataset)
        reference = anchor_reference(ctx, dataset)
        ref_through = anchors.value("throughput_anchor_seconds", dataset=reference)
        ref_low = anchors.require("training_anchor_low_seconds", dataset=reference)
        ref_high = anchors.require("training_anchor_high_seconds", dataset=reference)
        bracket = [ref_low, ref_high] + ([ref_through] if ref_through is not None else [])
        t_low, t_high = min(bracket), max(bracket)
        training_gm = math.sqrt(ref_low * ref_high)
        t_reference = math.sqrt(training_gm * ref_through) if ref_through is not None else training_gm
        if reference != dataset:
            label = f"shares {reference} (same backbone and graph size)"
        elif ref_through is not None:
            label = "geometric mean of both anchors"
        else:
            label = "training-time anchor (launch-bound)"
        t_nom = timing.require("t_nom_seconds", dataset=dataset)
        qc_key, qd_key = method_labels(ledger)
        qual_total = method_value(base, "gpu_hours", dataset, qd_key or qc_key)
        plant_hours = method_value(base, "gpu_hours", dataset, qc_key)
        added = None if ledger["C_exp"] is None else ledger["C_exp"] - ledger["C_qC"]
        if qual_total is None or plant_hours is None or not added or added <= 0:
            raise DisplayInputError(f"{dataset}: amortization needs the qualification GPU-hours of q_D and q_C and a "
                                    f"positive added expected work")
        qualification = qual_total - plant_hours
        m_t = 3600.0 * qualification / (theta * added)        # input-seconds
        horizon = economics.amortization_horizon(qualification, theta, added, t_nom)
        h_low, h_high = m_t / t_high, m_t / t_low
        training_text = f"{num(low * factor, 2)}–{num(high * factor, 2)} {unit}"
        if reference != dataset:
            training_text += " (own quotient; not used)"
        rows.append([SHORT[dataset], "— (not used; see caption)" if w_nom is None else _sci(w_nom, 3),
                     "—" if through is None else f"{num(through * factor, 2)} {unit}", training_text,
                     f"{num(t_nom * factor, 2)} {unit} [{num(t_low * factor, 2)}–{num(t_high * factor, 2)} {unit}]",
                     num(m_t / 1000.0, 0), f"{_sci(horizon)} [{_sci(h_low)}, {_sci(h_high)}]", label])
        source.append([dataset, w_nom, through, low, high, reference, t_low, t_high, t_reference, t_nom, qualification,
                       added, theta, m_t, horizon, h_low, h_high])
        out.compare(G_S23B, (dataset, "W_nom_MAC"), None if w_nom is None else _round_sig(w_nom, 3))
        out.compare(G_S23B, (dataset, "throughput_anchor_s"), through)
        out.compare(G_S23B, (dataset, f"training_time_anchor_{unit}"), [low * factor, high * factor])
        out.compare(G_S23B, (dataset, f"t_nom_measured_{unit}"), t_nom * factor)
        out.compare(G_S23B, (dataset, f"t_nom_reference_range_{unit}"), [t_low * factor, t_high * factor])
        out.compare(G_S23B, (dataset, "M_star_t_nom_thousand_input_s"), m_t / 1000.0)
        out.compare(G_S23B, (dataset, "M_star_inputs"), _round_sig(horizon))
        out.compare(G_S23B, (dataset, "M_star_range"), [_round_sig(h_low), _round_sig(h_high)])
        out.uses("run", dataset, ["select/candidates.json"])
    for rec in (gate, base, timing, anchors):
        _uses_records(out, rec)
    out.table(["Dataset", "Wnom (MAC)", "throughput anchor", "training-time anchor", "tnom [range]",
               "M* tnom (10^3 input·s)", "M* (inputs) [range]", "anchor"], rows,
              notes=["W_nom = L (N d^2 + E d) for the full-batch plants (registered dataset statistics, configured depth "
                     "and width); t_nom measured (nominal_time set) with the range spanned by the timing anchors "
                     "(timing_anchors set); M* t_nom = 3600 C_qual / [theta (E C(q_D) - C(q_C))] with C_qual the "
                     "qualification GPU-hours of q_D minus those of q_C (baselines set), theta of main Definition 2 "
                     "and the expected cost of Table S23; M* = (M* t_nom)/t_nom, range over the anchor bracket."])
    out.data("amortization", ["dataset", "W_nom_MAC", "throughput_anchor_s", "training_anchor_low_s",
                              "training_anchor_high_s", "anchor_reference_dataset", "t_nom_range_low_s",
                              "t_nom_range_high_s", "t_nom_analytical_reference_s", "t_nom_measured_s",
                              "qualification_increment_gpu_h", "added_expected_work", "theta", "M_star_t_nom_input_s",
                              "M_star_inputs", "M_star_low", "M_star_high"], source)


# ---------------------------------------------------------------------------
# Supplementary Table S25
# ---------------------------------------------------------------------------

def cross_backbone_regimes() -> list[tuple[str, str, list[str]]]:
    """(regime, dataset, [model A, model B]) in the order of the registered coverage contract."""
    contract = RECORD_SETS["cross_backbone"]
    regimes: dict[str, list[str]] = {}
    for regime, model in contract["coverage_rows"]:
        regimes.setdefault(regime, []).append(model)
    return [(regime, dataset, models) for (regime, models), dataset in zip(regimes.items(), contract["datasets"])]


@display("supplementary:Table S25", stem="S25", title="Cross-backbone substitutions",
         needs=("records:cross_backbone",) + _runs(RECORD_SETS["cross_backbone"]["datasets"]))
def table_s25(ctx, out):
    records = ctx.require_records("cross_backbone")
    rows, source = [], []
    for regime, dataset, models in cross_backbone_regimes():
        ledger = run_ledger(ctx, dataset)
        route = ROUTE_NAMES[ledger["qc_route"] if ledger["returned"] else ledger["qd_route"]]
        values = []
        for role, model in zip("AB", models):
            effect = records.require("effect_nats", regime=regime, model=model)
            stable = records.require("route_stable_folds", regime=regime, model=model)
            folds = records.require("outer_folds", regime=regime, model=model)
            values.append((model, effect, stable, folds))
            source.append([regime, dataset, role, model, route, effect, stable, folds])
        (model_a, eff_a, st_a, f_a), (model_b, eff_b, st_b, f_b) = values
        loss = _loss_label(dataset, None)
        rows.append([regime, f"{model_a}; post-training adapter", f"{model_b}; same adapter interface",
                     f"{route} in {st_a:.0f}/{f_a:.0f} versus {st_b:.0f}/{f_b:.0f} outer folds",
                     f"{num(eff_a, 5)} versus {num(eff_b, 5)} {loss}"])
        out.uses("run", dataset, ["select/candidates.json"])
    _uses_records(out, records)
    out.table(["regime", "model A", "model B", "route stability", "paired effect vs qC"], rows,
              notes=["Model A is the instrumented plant of the regime, model B the frozen substitute behind the same "
                     "adapter interface (order of the registered coverage contract); the route is the run's deployed "
                     "route; folds and paired effects from the cross_backbone record set."])
    out.data("substitutions", ["regime", "dataset", "role", "model", "route", "effect_nats", "route_stable_folds",
                               "outer_folds"], source)


# ---------------------------------------------------------------------------
# Supplementary Tables S26 and S26b
# ---------------------------------------------------------------------------

def ensemble_fits(ctx) -> tuple[dict, dict]:
    """Validated comparator grid and its least-squares law (paper.build.supplement._ensemble_data)."""
    from paper.build.supplement import _ensemble_data
    grid = ctx.require_records("ensemble_grid")
    try:
        fits, points = _ensemble_data(grid.rows, ctx.record_specs)
    except ValueError as exc:
        raise DisplayInputError(f"ensemble_grid: {exc}") from exc
    losses = {}
    for point in points:
        losses.setdefault(point["dataset"], {})[int(point["ensemble"])] = point["loss_nats"]
    return {fit["dataset"]: fit for fit in fits}, losses


def kappa_band(ctx) -> tuple[float, float]:
    low, high = ctx.constants["budget_matched_comparison_section4_table_S32_S40"]["ensemble_law_kappa_band"]
    return float(low), float(high)


def break_even(band, E: int, E_prime: float) -> list[float]:
    """Proposition 5 thresholds D_inf (1/E - 1/E') over the D_inf band."""
    return [economics.break_even_effect(d, E_prime / E - 1.0, E) for d in band]


@display("supplementary:Table S26b", stem="S26b", title="Fitted ensemble law and break-even",
         needs=_runs(ADMITTED) + ("records:input_gate", "records:ensemble_grid"))
def table_s26b(ctx, out):
    gate = ctx.require_records("input_gate")
    fits, losses = ensemble_fits(ctx)
    k_low, k_high = kappa_band(ctx)
    rows, source = [], []
    for dataset in ADMITTED:
        ledger = run_ledger(ctx, dataset, gate)
        plant = ledger["loss_qN"]
        d_hat = fits[dataset]["D_inf"]
        band = (k_low * plant, k_high * plant)
        E = ledger["E"]
        expected = break_even(band, E, E * ledger["C_exp"] / ledger["C_qC"])
        assure = break_even(band, E, E * ledger["C_assure"] / ledger["C_qC"])
        seeds = len(ctx.config(dataset)["training"]["seeds"])
        extra = [e for e in sorted(losses[dataset]) if e > seeds]
        added = max(losses[dataset]) - seeds
        hours = added * ctx.constants["backbones_table_S5"][dataset]["measured_hours_per_seed"]
        delta = abs(ledger["effect"])
        rows.append([SHORT[dataset], num(plant, 2), f"{num(d_hat, 4)} [{num(band[0], 4)}, {num(band[1], 4)}]",
                     f"{num(d_hat / plant, 3)} [{k_low:g}, {k_high:g}]", _span(*expected, 4, collapse=False),
                     _span(*assure, 4, collapse=False), num(delta, 4), f"{added} ({num(hours, 1)})"])
        source.append([dataset, plant, fits[dataset]["l_inf"], d_hat, fits[dataset]["max_abs_residual"], band[0],
                       band[1], d_hat / plant, E, ledger["C_qC"], ledger["C_exp"], ledger["C_assure"], *expected, *assure,
                       delta, added, hours])
        for key, value in (("l_plant_nats", plant), ("D_inf_hat", d_hat), ("D_inf_band", list(band)),
                           ("kappa_hat", d_hat / plant), ("break_even_expected", expected),
                           ("break_even_assurance", assure), ("measured_abs_Delta", delta),
                           ("extra_checkpoints_E16_20", added), ("extra_GPU_h", hours)):
            out.compare(G_S26B, (dataset, key), value)
        out.uses("run", dataset, ["select/candidates.json", "final/results_summary.json"])
    _uses_records(out, gate)
    _uses_records(out, ctx.require_records("ensemble_grid"))
    sizes = ", ".join(str(e) for e in extra)
    out.table(["Dataset", "l_plant (nats)", "D_inf [band]", "kappa", f"break-even at expected overhead "
               f"(kappa in [{k_low:g}, {k_high:g}])", "break-even at assurance overhead", "measured |Delta|",
               f"extra checkpoints for E = {sizes} (GPU-h)"], rows,
              notes=["D_inf: least-squares fit of l(E) = l_inf + D_inf/E over the ensemble_grid set; band = registered "
                     "kappa band x l_plant with l_plant the run's full-depth plant test loss; kappa = D_inf/l_plant; "
                     "break-even thresholds of main Proposition 5, D_inf (1/E - 1/E'), at E' = E C/C_qC for the "
                     "expected and assurance costs; extra checkpoints = largest grid size minus the plant seeds, "
                     "GPU-h = their number x the registered measured hours per plant seed."])
    out.data("ensemble_law", ["dataset", "l_plant_nats", "l_inf_fit", "D_inf_fit", "max_abs_residual", "D_inf_band_low",
                              "D_inf_band_high", "kappa_hat", "E", "C_qC", "C_expected", "C_assure",
                              "break_even_expected_low", "break_even_expected_high", "break_even_assurance_low",
                              "break_even_assurance_high", "measured_abs_delta", "extra_checkpoints",
                              "extra_gpu_hours"], source)


def matched_verdict(low: float, high: float) -> str:
    return "retained" if high < 0 else "comparator ahead" if low > 0 else "break-even"


@display("supplementary:Table S26", stem="S26", title="Integer ensemble grid and equal-cost interpolation",
         needs=_runs(ADMITTED) + ("records:input_gate", "records:ensemble_grid", "records:ensemble_grid_scores"))
def table_s26(ctx, out):
    gate = ctx.require_records("input_gate")
    scores_set = ctx.require_records("ensemble_grid_scores")
    _, losses = ensemble_fits(ctx)
    k_low, k_high = kappa_band(ctx)
    rows, source = [], []
    for dataset in ADMITTED:
        ledger = run_ledger(ctx, dataset, gate)
        E, c_qc = ledger["E"], ledger["C_qC"]
        per_member = c_qc / E
        decimals = 4 if ledger["native_metric"] else 2
        grid = losses[dataset]
        grid_scores = {e: scores_set.require("official_score", dataset=dataset, ensemble=e) for e in grid}
        if E not in grid:
            raise DisplayInputError(f"ensemble_grid: {dataset} lacks the comparator's ensemble size {E}")
        larger = [e for e in sorted(grid) if e > E]
        slope = None
        if larger and grid[E] != grid[larger[0]]:
            slope = (grid_scores[larger[0]] - grid_scores[E]) / (grid[E] - grid[larger[0]])
        band = (k_low * ledger["loss_qN"], k_high * ledger["loss_qN"])
        delta, low, high = ledger["effect"], ledger["effect_low"], ledger["effect_high"]
        split = ledger["splits"][0]
        horizon = split["qc"].get("horizon")
        qc_name = ROUTE_NAMES[ledger["qc_route"]] + (f" H = {horizon}" if horizon is not None else "")
        rows.append([SHORT[dataset], f"qC: {qc_name}", integer(E), num(c_qc, 1), num(ledger["score_qC"], decimals), None,
                     f"{num(delta, 4, signed=True)} [{num(low, 4, signed=True)}, {num(high, 4, signed=True)}] "
                     "(main Table 1)", None])
        source.append([dataset, "qC", E, c_qc, ledger["score_qC"], None, None, delta, low, high, None])
        if not ledger["returned"]:
            route = ROUTE_NAMES[ledger["qd_route"]]
            rows.append([SHORT[dataset], f"deployed {route} at expected cost", integer(ledger["E_qd"]),
                         num(ledger["C_exp"], 2), num(ledger["score_qD"], decimals), None, None, None])
            source.append([dataset, "deployed", ledger["E_qd"], ledger["C_exp"], ledger["score_qD"], None, None, None,
                           None, None, None])
            for label, cost in (("expected", ledger["C_exp"]), ("assurance", ledger["C_assure"])):
                e_prime = E * cost / c_qc
                thresholds = break_even(band, E, e_prime)
                effects = [delta + t for t in thresholds]
                scores = [None if slope is None else grid_scores[E] + slope * t for t in thresholds]
                verdict = matched_verdict(min(effects), max(effects))
                rows.append([SHORT[dataset], f"qC at matched {label} cost", num(e_prime, 1), num(cost, 2),
                             _span(min(scores), max(scores), decimals, collapse=False) if slope is not None else None,
                             _span(*thresholds, 4, collapse=False), f"{num(effects[0], 4, signed=True)} to "
                             f"{num(effects[1], 4, signed=True)}", verdict])
                source.append([dataset, f"matched {label}", e_prime, cost, scores[0], thresholds[0], thresholds[1],
                               effects[0], effects[1], None, verdict])
        for size in [e for e in larger if e <= len(ctx.config(dataset)["training"]["seeds"])]:
            cost = size * per_member
            effect = delta - (grid[size] - grid[E])
            if effect < 0:
                verdict = ("deployed rule retained at lower cost" if ledger["C_exp"] is not None and cost > ledger["C_exp"]
                           else "deployed rule retained")
            else:
                verdict = (f"comparator ahead at this size; cost {cost / ledger['C_exp']:.1f}× the deployed expected cost"
                           if ledger["C_exp"] else "comparator ahead at this size")
            rows.append([SHORT[dataset], f"qC at E = {size} (registered grid)", integer(size), num(cost, 1),
                         num(grid_scores[size], decimals), None, num(effect, 4, signed=True), verdict])
            source.append([dataset, f"grid E = {size}", size, cost, grid_scores[size], None, None, effect, None, None,
                           verdict])
        out.uses("run", dataset, ["select/candidates.json", "final/results_summary.json"])
    for name in ("input_gate", "ensemble_grid", "ensemble_grid_scores"):
        _uses_records(out, ctx.require_records(name))
    out.table(["Dataset", "configuration", "E or E'", "serial cost", "official score", "break-even threshold (nats)",
               "paired loss effect of deployed rule vs this row (nats)", "verdict"], rows,
              notes=["E' = E C/C_qC is the effective size at the matched cost; break-even threshold of main "
                     f"Proposition 5 over kappa in [{k_low:g}, {k_high:g}] (D_inf = kappa l_plant); effect vs a matched row "
                     "= run effect + threshold; effect vs a grid row = run effect - [l(E_grid) - l(E)] from the "
                     "ensemble_grid set; official score at E' = grid score at E + threshold x the grid's score gain per "
                     "nat between E and the next grid size (ensemble_grid_scores set). Registered grid rows: sizes "
                     "above E up to the number of plant seeds."])
    out.data("grid", ["dataset", "row", "E_or_E_prime", "serial_cost", "official_score", "threshold_low",
                      "threshold_high", "effect_or_low", "effect_high_or_ci_low", "ci_high", "verdict"], source)


# ---------------------------------------------------------------------------
# Supplementary Table S37
# ---------------------------------------------------------------------------

UNIT_WORDS = {"cluster": "buffered cores", "official_split": "splits", "graph": "graphs"}


@display("supplementary:Table S37", stem="S37", title="Fallback, false-safe and harm-miss rates",
         needs=("records:input_gate",) + _runs(GRAPH_LEVEL))
def table_s37(ctx, out):
    gate = ctx.require_records("input_gate")
    episodes = " or ".join(dict.fromkeys(UNIT_WORDS[UNITS[d]] for d in NODE_LEVEL))
    rows, source = [], []
    for dataset in DATASETS:
        if dataset in NODE_LEVEL:
            rows.append([SHORT[dataset], f"input fallback (deployment episodes: {episodes}); false-safe and harm-miss "
                         "rates (executed decisions); stepwise support rejection (sites)", None, None,
                         "fallback fraction in main Table 1; rates plotted in Fig. S16c–d"])
            continue
        ledger = run_ledger(ctx, dataset)
        rate = gate.require("fallback_probability", dataset=dataset)
        graphs = ledger["test_units"]
        if not graphs:
            raise DisplayInputError(f"{dataset}: the run records no official test graphs")
        count = int(round(rate * graphs))
        rows.append([SHORT[dataset], "input fallback (deployment graphs; official test split)", integer(count),
                     integer(graphs), f"{num(rate, 3)} (main Table 1)"])
        rows.append([SHORT[dataset], "false-safe rate and harm-miss rate (executed decisions); stepwise support "
                     "rejection (sites)", None, None, "plotted in Fig. S16c–d"])
        source.append([dataset, rate, graphs, rate * graphs, count])
        for key, value in (("numerator", count), ("denominator", graphs), ("rate", rate)):
            out.compare(G_S37, (dataset, key), value)
        out.uses("run", dataset, ["final/results_summary.json"])
    _uses_records(out, gate)
    out.table(["Dataset", "rate (unit)", "numerator", "denominator", "rate"], rows,
              notes=["Multigraph fallback counts: the input_gate fallback fraction times the official test graphs of the "
                     "run (count rounded to the nearest graph); per-decision rates are the reliability diagnostics of "
                     "Figure S16."])
    out.data("fallback_counts", ["dataset", "fallback_fraction", "test_graphs", "fraction_times_graphs", "count"], source)


# ---------------------------------------------------------------------------
# Supplementary Table S45
# ---------------------------------------------------------------------------

def inserted_site_arms(ctx) -> dict:
    """Paired effects and graph-unit SEs of the four E13 arms per dataset (inserted_site set)."""
    records = ctx.require_records("inserted_site")
    out = {}
    for dataset in RECORD_SETS["inserted_site"]["datasets"]:
        model = ctx.config(dataset)["model"]
        out[dataset] = {"native_fraction": model["native_layers"] / model["depth"], "arms": {}}
        for arm in INSERTED_SITE_ARMS:
            out[dataset]["arms"][arm] = (records.require("effect_nats", dataset=dataset, arm=arm),
                                         records.require("standard_error", dataset=dataset, arm=arm))
    return out


@display("supplementary:Table S45", stem="S45", title="Inserted-site attribution", needs=("records:inserted_site",))
def table_s45(ctx, out):
    data = inserted_site_arms(ctx)
    native_ol, _, native_fb, _ = INSERTED_SITE_ARMS
    rows, source = [], []
    for dataset, entry in data.items():
        arms = entry["arms"]
        (e1, se1), (e2, _), (e1p, se1p), (e3, se3) = (arms[a] for a in INSERTED_SITE_ARMS)
        low, high = e1p - Z95 * se1p, e1p + Z95 * se1p
        share = e1p / e3 if e3 else None
        rows.append([SHORT[dataset], num(entry["native_fraction"], 3), num(e1, 4, signed=True),
                     "0 (identity)" if e2 == 0 else num(e2, 4, signed=True),
                     f"{num(e1p, 4, signed=True)} [{num(low, 4, signed=True)}, {num(high, 4, signed=True)}]",
                     num(e3, 4, signed=True), num(se1, 5), num(share, 2)])
        source.append([dataset, entry["native_fraction"], *[v for arm in INSERTED_SITE_ARMS for v in arms[arm]], low,
                       high, share])
        for key, value in (("native_site_fraction", entry["native_fraction"]), ("arm1_native_open_loop", e1),
                           ("arm2_inserted_open_loop", e2), ("arm1prime_native_feedback", e1p),
                           ("arm1prime_range", [low, high]), ("arm3_inserted_feedback_deployed", e3),
                           ("paired_SE", se1), ("retained_share_arm1prime", share)):
            out.compare(G_S45, (dataset, key), value)
    _uses_records(out, ctx.require_records("inserted_site"))
    out.table(["Dataset", "native-site fraction", f"arm 1: {native_ol}", "arm 2: inserted, open loop (= qC; identity)",
               f"arm 1': {native_fb} [range]", "arm 3: inserted, feedback", "paired SE (unit of Table S42)",
               "retained share of arm 1'"], rows,
              notes=["Native-site fraction = native layers / controlled depth of the configured plant. Arm effects and "
                     "graph-unit SEs from the inserted_site record set; arm 1' range = effect +/- 1.96 SE; paired SE = "
                     f"SE of the {native_ol!r} arm against q_C; retained share = arm 1' / arm 3."])
    out.data("arms", ["dataset", "native_site_fraction"] + [f"{a} {q}" for a in INSERTED_SITE_ARMS
                                                            for q in ("effect_nats", "se_nats")]
             + ["arm1prime_low", "arm1prime_high", "retained_share"], source)


# ---------------------------------------------------------------------------
# E3-E6 budget-matched arms
# ---------------------------------------------------------------------------

def arm_id(text: str) -> str:
    match = re.match(r"\s*(E\d+)", str(text))
    if not match:
        raise DisplayInputError(f"budget_matched: arm {text!r} does not name E3-E6")
    return match.group(1)


def strict_budget_replay(ctx) -> dict:
    """Strict replay of the E3-E6 campaign export (paper.build.supplement._budget_data), once per context."""
    cache = _cache(ctx)
    if "budget" not in cache:
        from paper.build.supplement import _budget_data
        try:
            cache["budget"], _ = _budget_data(ctx.require_campaign("budget"))
        except ImportError as exc:
            raise DisplayInputError(f"the strict E3-E6 campaign replay needs the PyTorch runtime ({exc})") from exc
        except (ValueError, KeyError, OSError) as exc:
            raise DisplayInputError(f"the E3-E6 campaign export does not pass the strict replay: {exc}") from exc
    return cache["budget"]


REPLAY_FIELDS = {"effect_vs_qc_nats": "effect_vs_qC", "effect_vs_feedback_nats": "effect",
                 "effect_vs_feedback_ci_low_nats": "ci_low", "effect_vs_feedback_ci_high_nats": "ci_high",
                 "p_holm": "p_holm", "work_ratio": "work_ratio", "ceiling": "ceiling"}


def budget_arms(ctx) -> dict:
    """Arm outcomes per (dataset, arm) from the budget_matched set.

    The 'selected' configuration (each controller replicate's development-selected
    configuration) gives every central value; ranges span every deposited
    configuration of the arm and are configuration ranges, not intervals. The selected
    values must agree with the strict replay of the campaign export.
    """
    records = ctx.require_records("budget_matched")
    replay = {(row["dataset"], arm_id(row["arm"])): row for row in strict_budget_replay(ctx)["summary"]}
    arms: dict = {}
    for row in records.rows:
        entry = arms.setdefault((row["dataset"], arm_id(row["arm"])), {})
        entry.setdefault(row["configuration"], {})[row["quantity"]] = float(row["value"])
    expected = {(d, a) for d in ADMITTED for a in ARMS}
    if set(arms) != expected:
        raise DisplayInputError("budget_matched: the record set does not cover the four arms of every admitted dataset")
    result = {}
    for key, configurations in arms.items():
        selected = configurations.get("selected")
        if selected is None or set(selected) != set(REPLAY_FIELDS):
            raise DisplayInputError(f"budget_matched: {key} lacks the complete 'selected' configuration")
        reference = replay.get(key)
        if reference is None:
            raise DisplayInputError(f"budget_matched: {key} is absent from the strict campaign replay")
        for quantity, field in REPLAY_FIELDS.items():
            if not math.isclose(selected[quantity], float(reference[field]), rel_tol=1e-8, abs_tol=1e-12):
                raise DisplayInputError(f"budget_matched: {key} {quantity} differs from the strict campaign replay")
        effects = [c["effect_vs_qc_nats"] for c in configurations.values() if "effect_vs_qc_nats" in c]
        overheads = [c["work_ratio"] - 1.0 for c in configurations.values() if "work_ratio" in c]
        result[key] = {**selected, "configurations": len(configurations),
                       "effect_range": _by_magnitude([min(effects), max(effects)]) if len(configurations) > 1 else None,
                       "overhead_range": [min(overheads), max(overheads)] if len(configurations) > 1 else None,
                       "teacher_work_ratio": reference.get("reference_work_ratio"), "seeds": reference.get("seeds"),
                       "units": reference.get("units"), "inference_scope": reference.get("inference_scope")}
    return result


def _bracket_mnats(values) -> str:
    return "[—]" if values is None else f"[{num(1000 * values[0], 1)}, {num(1000 * values[1], 1)}]"


def holm_text(p: float):
    return "< 1e-4" if p < 1e-4 else format(p, ".2g")


@display("supplementary:Table S46", stem="S46", title="Budget-matched frontier",
         needs=_runs(ADMITTED) + ("records:input_gate", "records:budget_matched", "campaign:budget"))
def table_s46(ctx, out):
    gate = ctx.require_records("input_gate")
    arms = budget_arms(ctx)
    rows, source = [], []
    for dataset in ADMITTED:
        ledger = run_ledger(ctx, dataset, gate)
        e = {arm: arms[dataset, arm] for arm in ARMS}
        ceiling = e["E3"]["ceiling"]
        feedback = 1000 * ledger["effect"]
        present = 100 * (ledger["C_exp"] / ledger["C_qC"] - 1.0)
        cells = [f"{num(1000 * e[a]['effect_vs_qc_nats'], 1)} {_bracket_mnats(e[a]['effect_range'])}"
                 for a in ("E3", "E4", "E5")]
        e6 = e["E6"]
        overhead = 100 * (e6["work_ratio"] - 1.0)
        ov_range = e6["overhead_range"]
        e6_text = (f"{num(1000 * e6['effect_vs_qc_nats'], 1)}; {overhead:.0f}% "
                   + ("[—]" if ov_range is None else f"[{100 * ov_range[0]:.0f}, {100 * ov_range[1]:.0f}]"))
        rows.append([SHORT[dataset], num(ceiling, 2), num(feedback, 1), *cells, e6_text, f"{present:.0f}%"])
        source.append([dataset, ceiling, feedback] + [1000 * e[a]["effect_vs_qc_nats"] for a in ARMS]
                      + [overhead, present, ledger["C_exp"], ledger["C_qC"]])
        compare = [("ceiling", ceiling), ("feedback_effect_mnats", feedback), ("E6_qB_effect", 1000 * e6["effect_vs_qc_nats"]),
                   ("E6_overhead_pct", overhead), ("present_overhead_pct", present)]
        for arm, name in (("E3", "E3_predictor"), ("E4", "E4_distilled"), ("E5", "E5_no_observation_replanning")):
            compare.append((name, 1000 * e[arm]["effect_vs_qc_nats"]))
            if e[arm]["effect_range"] is not None:
                compare.append((f"{arm}_range", [1000 * v for v in e[arm]["effect_range"]]))
        if ov_range is not None:
            compare.append(("E6_overhead_range", [100 * v for v in ov_range]))
        for key, value in compare:
            out.compare(G_S46, (dataset, key), value)
        out.uses("run", dataset, ["select/candidates.json", "final/results_summary.json"])
    _uses_records(out, gate)
    _uses_records(out, ctx.require_records("budget_matched"))
    out.uses("campaign", "budget", ["study.json", "<arm studies and parents named by study.json>"])
    out.table(["Dataset", "ceiling", "feedback effect", "E3 predictor", "E4 distilled", "E5 no-observation replanning",
               "E6 qB: effect; overhead [range]", "present overhead"], rows,
              notes=["Millinats. Ceiling: registered common ceiling (expected work relative to q_C); feedback effect: "
                     "the run's test-stage paired effect of the deployed rule; arm effects against q_C on D_compare from "
                     "the budget_matched set (selected configuration; brackets span the deposited configurations, "
                     "[—] when only the selected configuration is deposited), checked against the strict campaign "
                     "replay; E6 overhead = executed work relative to q_C minus one; present overhead = "
                     "C_expected/C_qC - 1 of Table S23."])
    out.data("frontier", ["dataset", "ceiling", "feedback_effect_mnats", "E3_mnats", "E4_mnats", "E5_mnats", "E6_mnats",
                          "E6_overhead_percent", "present_overhead_percent", "C_expected", "C_qC"], source)


@display("supplementary:Table S46b", stem="S46b", title="D_compare arm outcomes",
         needs=("records:budget_matched", "campaign:budget"))
def table_s46b(ctx, out):
    arms = budget_arms(ctx)
    rows, source = [], []
    for dataset in ADMITTED:
        for arm in ARMS:
            e = arms[dataset, arm]
            effect = 1000 * e["effect_vs_qc_nats"]
            if arm == "E6":
                first = f"{num(effect, 1)}; overhead {100 * (e['work_ratio'] - 1.0):.0f}%"
            else:
                first = f"{num(effect, 1)} {_bracket_mnats(e['effect_range'])}"
            vs_m = [1000 * e[q] for q in ("effect_vs_feedback_nats", "effect_vs_feedback_ci_low_nats",
                                           "effect_vs_feedback_ci_high_nats")]
            work = e["work_ratio"] / e["ceiling"]
            rows.append([SHORT[dataset], ARM_LABELS[arm], first,
                         f"{num(vs_m[0], 1, signed=True)} [{num(vs_m[1], 1, signed=True)}, {num(vs_m[2], 1, signed=True)}]",
                         holm_text(e["p_holm"]), num(work, 2)])
            source.append([dataset, arm, effect, *vs_m, e["p_holm"], e["work_ratio"], e["ceiling"], work,
                           e["teacher_work_ratio"], e["configurations"], e["seeds"], e["units"], e["inference_scope"]])
            base = (dataset, "arms", arm)
            out.compare(G_S46B, base + ("effect_vs_qC_mnats",), effect)
            out.compare(G_S46B, base + ("effect_vs_qM_mnats",), vs_m[0])
            out.compare(G_S46B, base + ("interval_vs_qM_mnats",), vs_m[1:])
            out.compare(G_S46B, base + ("holm_P",), holm_text(e["p_holm"]) if e["p_holm"] < 1e-4 else e["p_holm"])
            out.compare(G_S46B, base + ("work_over_ceiling",), work)
            if arm == "E6":
                out.compare(G_S46B, base + ("overhead_pct",), 100 * (e["work_ratio"] - 1.0))
            elif e["effect_range"] is not None:
                out.compare(G_S46B, base + ("retained_share_grid_range_vs_qC_mnats",),
                            [1000 * v for v in e["effect_range"]])
    _uses_records(out, ctx.require_records("budget_matched"))
    out.uses("campaign", "budget", ["study.json", "<arm studies and parents named by study.json>"])
    out.table(["Dataset", "arm", "effect vs qC [grid range]", "effect vs qM [95% interval]", "Holm P",
               "work / ceiling"], rows,
              notes=["Millinats on D_compare. Selected configuration of the budget_matched set, checked against the strict "
                     "replay of the campaign export; brackets after the effect against q_C span the deposited "
                     "configurations (configuration ranges, not intervals); the interval against the frozen teacher q_M "
                     "is the 95% crossed seed-by-unit bootstrap interval; Holm within each dataset's four arms; work / "
                     "ceiling = executed work relative to q_C over the common ceiling."])
    out.data("arms", ["dataset", "arm", "effect_vs_qC_mnats", "effect_vs_qM_mnats", "ci_low_vs_qM_mnats",
                      "ci_high_vs_qM_mnats", "p_holm", "work_ratio", "ceiling", "work_over_ceiling",
                      "teacher_work_ratio", "configurations_deposited", "seeds", "units", "inference_scope"], source)


# ---------------------------------------------------------------------------
# Supplementary Figures
# ---------------------------------------------------------------------------

def _panel_letter(ax, letter: str) -> None:
    ax.text(-0.12, 1.06, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def mechanism_label(ledger: dict) -> str:
    if ledger["returned"]:
        return f"comparator/{ROUTE_NAMES[ledger['qc_route']]}"
    return ROUTE_NAMES[ledger["qd_route"]]


@display("supplementary:Figure S1", stem="FigS1", title="Integrated decision surface",
         needs=("runs", "records:input_gate"))
def figure_s1(ctx, out):
    gate = ctx.require_records("input_gate")
    ledgers = [run_ledger(ctx, dataset, gate) for dataset in DATASETS]
    plt = pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4))
    positions = {ledger["dataset"]: len(ledgers) - 1 - i for i, ledger in enumerate(ledgers)}
    panel_a, panel_b, panel_c = [], [], []
    for ledger in ledgers:
        y = positions[ledger["dataset"]]
        for ax, (mean, low, high), rows in ((axes[0], (ledger["gain"], ledger["gain_low"], ledger["gain_high"]), panel_a),
                                            (axes[1], (ledger["effect"], ledger["effect_low"], ledger["effect_high"]),
                                             panel_b)):
            if mean is None:
                continue
            err = None if low is None or high is None else [[mean - low], [high - mean]]
            ax.errorbar(mean, y, xerr=err, fmt="o", color="#1f77b4", capsize=3, ms=5)
            rows.append([ledger["dataset"], mean, low, high,
                         "AP" if ledger["native_metric"] else "pp" if rows is panel_a else ledger["loss_label"]])
    categories = []
    for ledger in sorted(ledgers, key=lambda l: (l["returned"], -list(ROUTE_NAMES).index(l["qd_route"]))):
        if mechanism_label(ledger) not in categories:
            categories.append(mechanism_label(ledger))
    level = {label: len(categories) - 1 - i for i, label in enumerate(categories)}
    for i, ledger in enumerate(ledgers):
        fallback = 1.0 if ledger["returned"] else ledger["p"]
        y = level[mechanism_label(ledger)] + 0.12 * (i % 3 - 1)
        area = 30.0 * ledger["C_exp"]
        axes[2].scatter([fallback], [y], s=area, color="#1f77b4")
        axes[2].annotate(SHORT[ledger["dataset"]], (fallback, y), xytext=(5, -2), textcoords="offset points", fontsize=6)
        panel_c.append([ledger["dataset"], mechanism_label(ledger), fallback, ledger["C_exp"], area])
        out.compare(G_S23, (ledger["dataset"], "fallback_probability"), ledger["p"])
    for ax in axes[:2]:
        ax.axvline(0, color="0.3", lw=0.8)
        ax.set_yticks(list(positions.values()), [SHORT[d] for d in positions])
    axes[2].set_yticks(list(level.values()), list(level))
    axes[2].set_xlim(-0.05, 1.08)
    axes[0].set_xlabel("official-metric effect over $q_C$ (pp; AP for an AP metric)")
    axes[1].set_xlabel("paired CE/BCE effect over $q_C$ (nats)")
    axes[2].set_xlabel("input comparator-fallback fraction (marker area: expected serial cost)")
    for ax, title in zip(axes, ("a  Endpoint effect", "b  Primary loss effect", "c  Selected mechanism and resource scale")):
        ax.set_title(title)
    fig.tight_layout()
    out.figure(fig)
    out.data("panel_a_endpoint_effect", ["dataset", "effect", "low", "high", "unit"], panel_a)
    out.data("panel_b_loss_effect", ["dataset", "effect_nats", "low", "high", "loss"], panel_b)
    out.data("panel_c_mechanism", ["dataset", "mechanism", "fallback_fraction", "expected_serial_cost", "marker_area"],
             panel_c)
    for dataset in DATASETS:
        out.uses("run", dataset, ["select/candidates.json", "final/results_summary.json"])
    _uses_records(out, gate)


@display("supplementary:Figure S2", stem="FigS2", title="Runtime and memory scaling",
         needs=("records:runtime_scaling",))
def figure_s2(ctx, out):
    records = ctx.require_records("runtime_scaling")
    points = []
    for dataset in records.distinct("dataset"):
        for instance in records.distinct("instance"):
            if not records.select(dataset=dataset, instance=instance):
                continue
            points.append([dataset, instance,
                           records.require("processed_edges_millions", dataset=dataset, instance=instance),
                           records.require("wall_time_seconds", dataset=dataset, instance=instance),
                           records.require("peak_memory_gib", dataset=dataset, instance=instance)])
    if not points:
        raise DisplayInputError("runtime_scaling: no timed instance")
    points.sort(key=lambda row: row[2])
    edges = [row[2] for row in points]
    plt = pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
    for ax, column, color, marker, label, title in (
            (axes[0], 3, "#2a7bd6", "o", "measured wall time (s)", "Wall time versus graph size"),
            (axes[1], 4, "#e8642c", "s", "measured peak memory (GiB)", "Peak memory versus graph size")):
        ax.plot(edges, [row[column] for row in points], marker=marker, color=color, lw=1.6, ms=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("processed edges (millions)")
        ax.set_ylabel(label)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    _panel_letter(axes[0], "a")
    _panel_letter(axes[1], "b")
    fig.tight_layout()
    out.figure(fig)
    out.data("panel_a_wall_time", ["dataset", "instance", "processed_edges_millions", "wall_time_seconds"],
             [row[:4] for row in points])
    out.data("panel_b_peak_memory", ["dataset", "instance", "processed_edges_millions", "peak_memory_gib"],
             [row[:3] + row[4:] for row in points])
    _uses_records(out, records)


ARM_MARKERS = {"E3": ("s", "#e8642c"), "E4": ("^", "#e0a800"), "E5": ("v", "#7a5cd6"), "E6": ("P", "#1a9a9a")}
ARM_DODGE = {"E3": -0.06, "E4": -0.03, "E5": 0.03, "E6": 0.0}      # horizontal offset (x C_qC) to separate markers


def retained_share_grid(ctx, ledger: dict):
    """Registered retained-share grid of the comparator's regime (fixed-sequence or MPC-OL)."""
    grids = ctx.constants["budget_matched_comparison_section4_table_S32_S40"]["retained_share_grid_phi_x"]
    return {"fixed_sequence": grids.get("fixed_sequence_regimes"), "mpc_ol": grids.get("MPC_OL_regimes")}.get(
        ledger["qc_route"])


@display("supplementary:Figure S3", stem="FigS3", title="Budget-matched per-dataset frontiers",
         needs=_runs(ADMITTED) + ("records:input_gate", "records:ensemble_grid", "records:budget_matched",
                                  "campaign:budget"))
def figure_s3(ctx, out):
    gate = ctx.require_records("input_gate")
    arms = budget_arms(ctx)
    fits, losses = ensemble_fits(ctx)
    k_low, k_high = kappa_band(ctx)
    plt = pyplot()
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.4))
    for letter, ax, dataset in zip("abcdef", axes.flat, ADMITTED):
        ledger = run_ledger(ctx, dataset, gate)
        E, c_qc = ledger["E"], ledger["C_qC"]
        per_member = c_qc / E
        x_max = max(2.625 * c_qc, 1.05 * ledger["C_assure"])
        xs = np.linspace(c_qc, x_max, 240)
        law = lambda d: -d * (1.0 / E - 1.0 / (E * xs / c_qc))
        band = (k_low * ledger["loss_qN"], k_high * ledger["loss_qN"])
        d_hat = fits[dataset]["D_inf"]
        data = [["band_low", float(x), float(y), None, None, None] for x, y in zip(xs, law(band[0]))]
        data += [["band_high", float(x), float(y), None, None, None] for x, y in zip(xs, law(band[1]))]
        data += [["fitted_law", float(x), float(y), None, None, None] for x, y in zip(xs, law(d_hat))]
        ax.fill_between(xs, law(band[0]), law(band[1]), color="#f4a582", alpha=0.35, lw=0)
        ax.plot(xs, law(d_hat), color="#e8642c", lw=1.4)
        for size in sorted(losses[dataset]):
            if size > E and size * per_member <= x_max:
                y = losses[dataset][size] - losses[dataset][E]
                ax.plot(size * per_member, y, "o", mfc="white", mec="#e8642c", ms=5)
                data.append([f"grid_E{size}", size * per_member, y, None, None, None])
        ax.plot(c_qc, 0.0, "D", color="black", ms=7)
        data.append(["qC", c_qc, 0.0, None, None, None])
        ceiling = arms[dataset, "E3"]["ceiling"] * c_qc
        ax.axvline(ceiling, color="0.75", lw=3, alpha=0.7)
        ax.text(ceiling, 1.0, " common ceiling", transform=ax.get_xaxis_transform(), fontsize=6, va="top")
        data.append(["common_ceiling", ceiling, None, None, None, None])
        for arm in ARMS:
            e = arms[dataset, arm]
            marker, color = ARM_MARKERS[arm]
            x = e["work_ratio"] * c_qc
            y = e["effect_vs_qc_nats"]
            err = None
            if arm != "E6" and e["effect_range"] is not None:
                low, high = min(e["effect_range"]), max(e["effect_range"])
                err = [[y - low], [high - y]]
            else:
                low = high = None
            ax.errorbar(x + ARM_DODGE[arm] * c_qc, y, yerr=err, fmt=marker, mfc="white", mec=color, ecolor=color, ms=7,
                        mew=1.4, capsize=0)
            data.append([arm, x, y, low, high, x + ARM_DODGE[arm] * c_qc])
        if not ledger["returned"]:
            ax.plot([ledger["C_exp"], ledger["C_exp"]], [ledger["effect_low"], ledger["effect_high"]],
                    color="#2a7bd6", lw=2)
            ax.plot(ledger["C_exp"], ledger["effect"], "o", color="#2a7bd6", ms=7)
            ax.plot([ledger["C_exp"], ledger["C_assure"]], [ledger["effect"]] * 2, ":", color="#2a7bd6")
            data += [["deployed_expected", ledger["C_exp"], ledger["effect"], ledger["effect_low"], ledger["effect_high"],
                      None], ["deployed_assurance", ledger["C_assure"], ledger["effect"], None, None, None]]
        ax.axhline(0, color="0.2", lw=0.8)
        e6, e3 = arms[dataset, "E6"], arms[dataset, "E3"]
        teacher = e6["effect_vs_qc_nats"] - e6["effect_vs_feedback_nats"]
        r_b = e6["effect_vs_qc_nats"] / teacher if teacher else None
        grid = retained_share_grid(ctx, ledger)
        text = [] if grid is None else [f"$\\phi_x^{{gain}}$ grid {min(grid):.2f}–{max(grid):.2f}"]
        if r_b is not None:
            text.append(f"$r_B$ = {r_b:.3f}")
        ax.text(0.98, 0.04, "; ".join(text) + f"\nmargin FB−E3 at the ceiling: {1000 * e3['effect_vs_feedback_nats']:.1f} "
                "mnats", transform=ax.transAxes, ha="right", va="bottom", fontsize=6)
        route = ROUTE_NAMES[ledger["qc_route"] if ledger["returned"] else ledger["qd_route"]]
        ax.set_title(f"{SHORT[dataset]} (E = {E}, {route})", fontweight="bold")
        ax.set_xlabel("serial compute (nominal executions)")
        ax.set_ylabel("paired loss effect vs $q_C$ (nats)")
        ax.grid(alpha=0.25)
        _panel_letter(ax, letter)
        out.data(f"panel_{letter}_{SHORT[dataset].lower()}", ["series", "serial_compute", "effect_nats", "range_low",
                                                              "range_high", "plotted_x_offset_marker"], data)
        out.compare(G_S46, (dataset, "ceiling"), arms[dataset, "E3"]["ceiling"])
        out.uses("run", dataset, ["select/candidates.json", "final/results_summary.json"])
    handles = [plt.Line2D([], [], color="#e8642c", lw=6, alpha=0.4,
                          label=f"qC at a larger ensemble: fitted law band, kappa in [{k_low:g}, {k_high:g}] (E14); "
                                "circles: grid sizes"),
               plt.Line2D([], [], marker="D", color="black", ls="", label="qC"),
               plt.Line2D([], [], marker="o", color="#2a7bd6", ls="",
                          label="deployed rule at expected cost (95% test-stage interval); dotted: assurance cost")]
    handles += [plt.Line2D([], [], marker=ARM_MARKERS[a][0], mfc="white", mec=ARM_MARKERS[a][1], ls="",
                           label=ARM_LABELS[a]) for a in ARMS]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out.figure(fig)
    for name in ("input_gate", "ensemble_grid", "budget_matched"):
        _uses_records(out, ctx.require_records(name))
    out.uses("campaign", "budget", ["study.json", "<arm studies and parents named by study.json>"])


FIGURE_S4_DATASETS = ("ogbn-arxiv", "Roman-empire", "PascalVOC-SP")     # the three regimes of main Table 2


@display("supplementary:Figure S4", stem="FigS4", title="Method score and cost maps",
         needs=_runs(FIGURE_S4_DATASETS) + ("records:baselines", "records:input_gate"))
def figure_s4(ctx, out):
    base, gate = ctx.require_records("baselines"), ctx.require_records("input_gate")
    plt = pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
    for ax, dataset in zip(axes, FIGURE_S4_DATASETS):
        ledger, entries = method_rows(ctx, dataset, base, gate)
        data = []
        for index, e in enumerate(entries):
            if e["score"] is None or e["cost_low"] is None:
                continue
            if e["role"] == "deployed" and not ledger["returned"]:
                ax.plot([e["C_exp"], e["C_assure"]], [e["score"]] * 2, color="#2a7bd6", lw=1)
                ax.plot(e["C_assure"], e["score"], "|", color="#2a7bd6", ms=10, mew=1.5)
                ax.plot(e["C_exp"], e["score"], "o", color="#2a7bd6", ms=7)
                ax.annotate(f"CROSS-MORSE {ROUTE_NAMES[e['route']]}", (e["C_exp"], e["score"]), xytext=(6, 4),
                            textcoords="offset points", fontsize=6, color="#2a7bd6")
                x = e["C_exp"]
            elif e["role"] in ("qC", "deployed"):
                x = e["cost_low"] if e["role"] == "qC" else e["C_exp"]
                ax.plot(x, e["score"], "s", color="#e8642c", ms=6)
                ax.annotate(f"$q_C$: {ROUTE_METHOD[e['route']]}", (x, e["score"]), xytext=(-6, 4),
                            textcoords="offset points", fontsize=6, color="#e8642c", ha="right")
            else:
                external = e["role"] == "external"
                color = "#2bb38a" if external else "0.55"
                x = 0.5 * (e["cost_low"] + e["cost_high"])
                if e["cost_high"] > e["cost_low"]:
                    ax.plot([e["cost_low"], e["cost_high"]], [e["score"]] * 2, color=color, lw=3 if external else 2,
                            alpha=0.6)
                ax.plot(x, e["score"], "D" if external else "o", color=color, ms=5 if external else 3)
                if external:
                    ax.annotate(e["method"], (x, e["score"]), xytext=(0, 5), textcoords="offset points", fontsize=6,
                                color=color, ha="center")
            data.append([e["method"], e["family"], e["cost_low"], e["cost_high"], x, e["score"], e["C_assure"]])
            out.compare(G_S15, (dataset, "rows", index, "official_score"), e["score"])
        ax.set_title(dataset, loc="left")
        ax.set_xlabel("serial compute (one nominal execution = 1)")
        ax.set_ylabel({"accuracy": "accuracy (%)", "rocauc": "ROC-AUC (%)", "macro_f1": "macro F1 (%)",
                       "ap": "AP"}.get(METRICS.get(dataset), "official score"))
        ax.grid(alpha=0.25)
        out.data(f"panel_{SHORT[dataset].lower()}", ["method", "family", "cost_low", "cost_high", "x_plotted",
                                                     "official_score", "assurance_cost"], data)
        out.uses("run", dataset, ["select/candidates.json", "final/results_summary.json", "train/model_inventory.json"])
    handles = [plt.Line2D([], [], marker="o", color="#2a7bd6", label="deployed rule: expected cost (marker) to "
                          "assurance cost (tick)"),
               plt.Line2D([], [], marker="s", color="#e8642c", ls="", label="$q_C$"),
               plt.Line2D([], [], marker="D", color="#2bb38a", lw=3, alpha=0.7, label="retrained architectures"),
               plt.Line2D([], [], marker="o", color="0.55", lw=2, alpha=0.7,
                          label="other internal routes (configuration cost range)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out.figure(fig)
    _uses_records(out, base)
    _uses_records(out, gate)


def calibration_target(ctx) -> float:
    alphas = {ctx.config(dataset)["controller"]["calibration_alpha"] for dataset in DATASETS}
    if len(alphas) != 1:
        raise DisplayInputError("the configurations register different calibration levels")
    return 1.0 - alphas.pop()


@display("supplementary:Figure S16", stem="FigS16", title="Reliability, harm and fallback",
         needs=("records:reliability_diagnostics", "records:negative_controls"))
def figure_s16(ctx, out):
    diag, controls = ctx.require_records("reliability_diagnostics"), ctx.require_records("negative_controls")
    target = calibration_target(ctx)
    plt = pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0))
    horizons = sorted(diag.distinct("level", "coverage"), key=float)
    if not horizons:
        raise DisplayInputError("reliability_diagnostics: no coverage rows")
    cov = [[float(h), diag.require("coverage", diagnostic="coverage", level=h),
            diag.require("coverage_ci_low", diagnostic="coverage", level=h),
            diag.require("coverage_ci_high", diagnostic="coverage", level=h)] for h in horizons]
    ax = axes[0, 0]
    ax.errorbar([r[0] for r in cov], [r[1] for r in cov], yerr=[[r[1] - r[2] for r in cov], [r[3] - r[1] for r in cov]],
                fmt="o-", capsize=3, label="executed decision/sequence")
    ax.axhline(target, ls="--", color="0.4", lw=0.8, label="target")
    ax.set_xscale("log", base=2)
    ax.set_xticks([r[0] for r in cov], [f"{r[0]:g}" for r in cov])
    ax.set_xlabel("prediction horizon")
    ax.set_ylabel("coverage")
    ax.set_title("a  Executed action and sequence coverage")
    ax.legend()
    variants = diag.distinct("level", "risk_rmse")
    if not variants:
        raise DisplayInputError("reliability_diagnostics: no observer rows")
    rmse = {v: diag.require("risk_rmse", diagnostic="observer", level=v) for v in variants}
    nis = {v: diag.require("nis_coverage", diagnostic="observer", level=v) for v in variants}
    reference = rmse.get("raw", max(rmse.values()))
    ax = axes[0, 1]
    xs = np.arange(len(variants))
    ax.bar(xs - 0.18, [rmse[v] / reference for v in variants], 0.36, label="risk RMSE (normalized)")
    ax.bar(xs + 0.18, [nis[v] for v in variants], 0.36, label="NIS coverage")
    ax.axhline(target, ls="--", color="0.4", lw=0.8)
    ax.set_xticks(xs, variants, rotation=20)
    ax.set_title("b  Observer accuracy and consistency are distinct")
    ax.legend()
    harm = []
    for dataset in DATASETS:
        if not diag.select(diagnostic="harm", level=dataset):
            continue
        harm.append([dataset] + [diag.require(q, diagnostic="harm", level=dataset)
                                 for q in ("false_safe_rate", "false_safe_ci_low", "false_safe_ci_high",
                                           "harm_miss_rate", "harm_miss_ci_low", "harm_miss_ci_high")])
    ax = axes[1, 0]
    for i, row in enumerate(harm):
        y = len(harm) - 1 - i
        ax.errorbar(row[1], y + 0.1, xerr=[[row[1] - row[2]], [row[3] - row[1]]], fmt="o", color="#1f77b4",
                    label="false-safe rate" if i == 0 else None)
        ax.errorbar(row[4], y - 0.1, xerr=[[row[4] - row[5]], [row[6] - row[4]]], fmt="s", color="#ff7f0e",
                    ecolor="#1f77b4", label="harm-miss rate" if i == 0 else None)
    ax.set_yticks([len(harm) - 1 - i for i in range(len(harm))], [SHORT.get(r[0], r[0]) for r in harm])
    ax.set_xlabel("rate with interval")
    ax.set_title("c  Harm accounting uses explicit denominators")
    ax.legend()
    stress = [["clean", diag.require("input_fallback", diagnostic="stress", level="clean"),
               diag.require("residual_activity", diagnostic="stress", level="clean")]]
    for (control,) in RECORD_SETS["negative_controls"]["coverage_rows"]:
        stress.append([control, controls.require("input_fallback", control=control),
                       controls.require("residual_activity", control=control)])
    ax = axes[1, 1]
    xs = np.arange(len(stress))
    ax.bar(xs - 0.18, [r[1] for r in stress], 0.36, label="input comparator fallback")
    ax.bar(xs + 0.18, [r[2] for r in stress], 0.36, label="residual activity")
    ax.set_xticks(xs, [r[0] for r in stress], rotation=20)
    ax.set_ylabel("fraction of graphs/episodes")
    ax.set_title("d  Static compatibility and dynamic support reject stress inputs")
    ax.legend()
    fig.tight_layout()
    out.figure(fig)
    out.data("panel_a_coverage", ["horizon", "coverage", "ci_low", "ci_high"], cov)
    out.data("panel_b_observer", ["variant", "risk_rmse", "risk_rmse_normalized", "nis_coverage"],
             [[v, rmse[v], rmse[v] / reference, nis[v]] for v in variants])
    out.data("panel_c_harm", ["dataset", "false_safe_rate", "false_safe_low", "false_safe_high", "harm_miss_rate",
                              "harm_miss_low", "harm_miss_high"], harm)
    out.data("panel_d_stress", ["condition", "input_fallback", "residual_activity"], stress)
    out.data("target", ["quantity", "value"], [["coverage target (1 - calibration alpha)", target]])
    _uses_records(out, diag)
    _uses_records(out, controls)


def confirmation_sensitivity(ctx) -> list[dict]:
    """Mechanism contrasts at J_old and, under the two-way variance model with fixed means, at the D_confirm size."""
    from cm_campaigns.confirmation import FAMILY
    records = ctx.require_records("confirmation_sensitivity")
    rows = []
    for contrast, (dataset, _kind, unit, j_new, _loss, direction, margin) in FAMILY.items():
        if unit != "graph" or not records.select(contrast=contrast):
            continue
        effect = records.require("effect_nats", contrast=contrast)
        low, high = records.require("ci_low_nats", contrast=contrast), records.require("ci_high_nats", contrast=contrast)
        j_old = records.require("units_old", contrast=contrast)
        share = records.require("graph_variance_share", contrast=contrast)
        factor = math.sqrt(share * j_old / j_new + 1.0 - share)
        rows.append({"contrast": contrast, "dataset": dataset, "effect": effect, "old": [low, high], "J_old": j_old,
                     "J_new": j_new, "share": share, "factor": factor,
                     "new": [effect - (effect - low) * factor, effect + (high - effect) * factor],
                     "holm_p": records.require("holm_p", contrast=contrast),
                     "survival": records.require("label_survival_probability", contrast=contrast),
                     "signed_margin": direction * margin})
    if not rows:
        raise DisplayInputError("confirmation_sensitivity: no multigraph contrast")
    return rows


@display("supplementary:Figure S18", stem="FigS18", title="Inserted-site attribution and separated protocols",
         needs=("records:inserted_site", "records:confirmation_sensitivity", "records:band_bound_tightness"))
def figure_s18(ctx, out):
    arms = inserted_site_arms(ctx)
    sensitivity = confirmation_sensitivity(ctx)
    bound = ctx.require_records("band_bound_tightness")
    plt = pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))
    ax = axes[0]
    datasets = list(arms)
    width = 0.8 / len(datasets)
    panel_a = []
    for j, dataset in enumerate(datasets):
        heights, errors = [], []
        for arm in INSERTED_SITE_ARMS:
            effect, se = arms[dataset]["arms"][arm]
            heights.append(1000 * effect)
            errors.append(1000 * Z95 * se)
            panel_a.append([dataset, arm, effect, se, effect - Z95 * se, effect + Z95 * se])
        xs = np.arange(len(INSERTED_SITE_ARMS)) + (j - (len(datasets) - 1) / 2) * width
        ax.bar(xs, heights, width, yerr=errors, capsize=3, label=SHORT[dataset], alpha=0.8)
    ax.axhline(0, color="0.2", lw=0.8)
    ax.set_xticks(range(len(INSERTED_SITE_ARMS)),
                  [a.replace(", ", ",\n", 1).replace(" (", "\n(") for a in INSERTED_SITE_ARMS], fontsize=6)
    ax.set_ylabel("paired effect vs $q_C$ (mnats)")
    ax.set_title("E13: attribution of inserted-site transport", fontweight="bold")
    ax.legend()
    _panel_letter(ax, "a")
    ax = axes[1]
    panel_b = []
    for i, row in enumerate(sensitivity):
        y = len(sensitivity) - 1 - i
        for offset, key, color, label in ((0.17, "old", "0.3", "validation-split size ($J_{old}$)"),
                                          (-0.17, "new", "#7a5cd6", "$D_{confirm}$ size, two-way model")):
            low, high = row[key]
            ax.plot([1000 * low, 1000 * high], [y + offset] * 2, color=color, lw=2)
            ax.plot(1000 * row["effect"], y + offset, "o", color=color, label=label if i == 0 else None)
            panel_b.append([row["contrast"], key, row["J_old" if key == "old" else "J_new"], row["effect"], low, high,
                            row["signed_margin"], row["holm_p"], row["survival"]])
        ax.plot([1000 * row["signed_margin"]] * 2, [y - 0.15, y + 0.15], color="#2bb38a", lw=3)
        ax.text(1.02, (y + 0.5) / len(sensitivity), f"J {row['J_old']:.0f}{RIGHT}{row['J_new']}; Holm P="
                f"{holm_text(row['holm_p'])}\nP(label survives)={row['survival']:.2f}", transform=ax.transAxes,
                fontsize=6, va="center")
        for key, value in (("effect", row["effect"]), ("interval", row["new"]), ("interval_old", row["old"]),
                           ("J_old", row["J_old"]), ("J_new", row["J_new"]),
                           ("graph_variance_share", row["share"])):
            out.compare(G_S14B, (i, key), value)
    ax.axvline(0, color="0.2", lw=0.8)
    ax.set_yticks([len(sensitivity) - 1 - i for i in range(len(sensitivity))],
                  [SHORT[r["dataset"]] for r in sensitivity])
    ax.set_xlabel("mechanism contrast (mnats); green tick: margin")
    ax.set_title("E10: confirmation-size sensitivity", fontweight="bold")
    ax.legend(fontsize=6, loc="lower left")
    _panel_letter(ax, "b")
    ax = axes[2]
    panel_c = []
    for action in bound.distinct("action"):
        scales = sorted(bound.distinct("spectral_scale"), key=float)
        points = [(float(s), bound.require("displacement_ratio", action=action, spectral_scale=s)) for s in scales
                  if bound.select("displacement_ratio", action=action, spectral_scale=s)]
        ax.plot([p[0] for p in points], [p[1] for p in points], "o-", label=action, ms=4)
        panel_c += [[action, s, ratio] for s, ratio in points]
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"spectral energy scale $\lambda_0$ (density $\propto e^{-\lambda/\lambda_0}$ on [0, 2])")
    ax.set_ylabel("observed / bound displacement ratio")
    ax.set_title("E18: tightness of the band-limited bound (6.1)", fontweight="bold")
    ax.legend(ncol=4, fontsize=6, loc="lower right")
    _panel_letter(ax, "c")
    fig.tight_layout()
    out.figure(fig)
    out.data("panel_a_attribution", ["dataset", "arm", "effect_nats", "se_nats", "low_nats", "high_nats"], panel_a)
    out.data("panel_b_confirmation_size", ["contrast", "size", "graphs", "effect_nats", "low_nats", "high_nats",
                                           "signed_margin_nats", "holm_p", "label_survival_probability"], panel_b)
    out.data("panel_c_bound_tightness", ["action", "spectral_scale_lambda0", "displacement_ratio"], panel_c)
    for name in ("inserted_site", "confirmation_sensitivity", "band_bound_tightness"):
        _uses_records(out, ctx.require_records(name))
