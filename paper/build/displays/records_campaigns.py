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
G_S14 = "supplementary_table_S14_mechanism_contrasts_submitted_protocol"
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


def _span(low, high, decimals: int, sep: str = "–") -> str | None:
    """'a–b' at the printed precision; a single value when both ends print alike."""
    a, b = num(low, decimals), num(high, decimals)
    if a is None or b is None:
        return None
    return a if a == b else f"{a}{sep}{b}"


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


def lean_share(ledger: dict) -> float:
    """Lean-profile effect of the deployed rule as a share of its Full-profile effect.

    Both profiles of the deployed configuration (same route, ensemble, depth, horizon
    and kappa) carry a nested selection-stage paired effect against the comparator
    (select/decision.json); the share is Lean over Full, averaged over splits. A
    returned comparator executes no controller, so both profiles return q_C (share 1).
    """
    shares = []
    for s in ledger["splits"]:
        if s["returned"]:
            shares.append(1.0)
            continue
        rule, candidates = s["qd"], s["candidates"]
        profile = rule.get("profile")
        if profile not in ("full", "lean"):
            raise DisplayInputError(f"{ledger['dataset']}: the deployed rule has no Full/Lean profile")
        twin_profile = "lean" if profile == "full" else "full"

        def identity(c):
            return (c.get("route"), c.get("ensemble"), c.get("depth"), c.get("horizon"), c.get("kappa"))
        twins = [i for i, c in enumerate(candidates) if identity(c) == identity(rule) and c.get("profile") == twin_profile]
        if len(twins) != 1:
            raise DisplayInputError(f"{ledger['dataset']}: {s['name']} has no unique {twin_profile} twin of the deployed rule")
        effects = {c["index"]: c.get("select_delta") for c in s["evidence"]["candidates"]}
        full_index, lean_index = (s["qd_index"], twins[0]) if profile == "full" else (twins[0], s["qd_index"])
        full, lean = effects.get(full_index), effects.get(lean_index)
        if full is None or lean is None or full == 0:
            raise DisplayInputError(f"{ledger['dataset']}: selection-stage effects of the Full/Lean pair are missing")
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
    factors = list(factors_names(dataset))
    labels = factors
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
    note = ("Effect-coded main effects " + ", ".join(f"{labels[j]} {effects[(j,)]:+.4f}" for j in range(3))
            + f" nats (95% half-width {contrasts['half_width']:.4f} from the cell intervals under independent cells); "
            + "interactions " + ", ".join(f"{factors_label(f, labels)} {v:+.4f}" for f, v in interactions.items())
            + "; baseline-anchored differences "
            + ", ".join(f"I_{factors_label(f, labels)} {v:+.4f}" for f, v in contrasts["anchored"].items())
            + f"; full system minus no component {contrasts['full_minus_none']:+.4f} nats.")
    out.uses("records", "factorials", [])
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
