"""Analytical and arithmetic displays of the supplement.

Every analytical entry is the stated formula, as implemented in ``cm_theory`` (transport
polynomials, spectral responses, regret and admission bounds, the Note S19 cost model, the
Table S44 tail model, power and the decision boundaries of Proposition 10), evaluated at its
inputs:

* registered inputs: the runtime action registry (``cross_morse/core.py``, parsed as literals
  by ``cm_theory.operators``), ``paper/registered_constants.json`` and the configurations;
* measured inputs: the lifecycle runs (Table 1, S4, S13 and S23 quantities summarized by
  ``paper.build.evidence``) and the measurement-record sets ``input_gate``, ``ensemble_grid``
  and ``budget_matched``.

Observations that no artifact holds yet are read from two record contracts,
``action_gap_audit`` (the Table S21 action-gap quantiles and switch fractions) and
``confirmation_contrasts`` (the nine Table S14 contrasts at their original sizes), and from
optional ``budget_matched`` quantities of the E6 ledger. Scenario parameters that the
supplement states with a display, but that the registry does not hold, are the module
constants below, each named with the note that states it. Two Note S25 parameters, phi_OL and
o_dec, are read from the registered constants when present; until then their cells stay empty.

A Roman-empire campaign enters through the mean over its ten split runs (effects through the
campaign summary of ``paper.build.evidence``, costs and headroom as split means; support
counts as split minima); per-split values are kept in the source data.
"""
from __future__ import annotations

import dataclasses
import math
import re
from statistics import NormalDist, fmean

import numpy as np

from cm_campaigns.contract import ADMITTED, ARMS
from cm_theory import admission, budget, economics, inference, operators, spectral

from .framework import DATASETS, GRAPH_LEVEL, SHORT, DisplayInputError, display, integer, not_applicable, num, pyplot
from . import regret_data

NORMAL = NormalDist()
Z975 = NORMAL.inv_cdf(0.975)
Z95 = NORMAL.inv_cdf(0.95)

GAP_SET = "action_gap_audit"
CONTRAST_SET = "confirmation_contrasts"
BUDGET = "budget_matched_comparison_section4_table_S32_S40"

# Scenario parameters stated with the displays and absent from paper/registered_constants.json.
SENSING_SHARE = 0.06                      # s, Note S19 and Table S31: residual sensing-plus-observer share
RESOLVENT_STEP_SHARES = (0.25, 0.50)      # Figure S6a-b: fraction of executed steps using resolvent actions
TRUNCATION_AXIS = 64                      # Figure S6a-b: largest truncation order shown
TAIL_REFERENCE_SHARE = 0.25               # Table S44: reference resolvent share f_bar
RE_SEARCH_SHARES = (0.03, 0.08, 0.15)     # Note S25: re-search share rho of the no-observation arm
PAIRABLE_SHARE = {"single_graph": 0.5, "graph": 0.0}          # Note S24: conservative seed-paired reading
E18_DENSITIES = (0.1, 0.3, 1.0)           # Table S43 E18 and Figure S20b: spectral energy scales lambda_0
TERMINAL_SITE = {"W": 10.0, "W_C": 4.0, "C_pred": 0.5, "w": 6.0, "sensing": 2.0, "action": 2.0}  # Figure S11

COLORS = {"ogbn-arxiv": "#2f7bd9", "ogbn-proteins": "#e8673a", "ogbn-products": "#8c8c8c",
          "Roman-empire": "#1aaf7b", "PascalVOC-SP": "#6c5ad6", "COCO-SP": "#d55aa8", "Peptides-func": "#1d7a3b"}
ACTION_COLORS = ("#111111", "#2f7bd9", "#1aaf7b", "#e8673a", "#d9a400", "#6c5ad6", "#d55aa8", "#1d7a3b")

G_T1 = "main_table_1_policy_and_deployment_ledger"
G_S4 = "supplementary_table_S4_target_qualification"
G_S6C = "supplementary_table_S6c_sensing_cost_identity"
G_S13 = "supplementary_table_S13_independent_graph_admission"
G_S14B = "supplementary_table_S14b_separated_data_sensitivity"
G_S14B_UNCHANGED = "supplementary_table_S14b_unchanged_contrasts"
G_S21 = "supplementary_table_S21_action_gap_audit"
G_S23 = "supplementary_table_S23_assurance_and_expected_compute"
G_S26B = "supplementary_table_S26b_ensemble_law_and_break_even"
G_S34 = "supplementary_table_S34_budget_aware_rule"
G_S42 = "supplementary_table_S42_power_of_budget_matched_test"
G_S44 = "supplementary_table_S44_per_input_cost_model"
G_S47 = "supplementary_table_S47_decision_boundary_and_power"
G_FS8 = "supplementary_figure_S8_clipping_and_uncertainty"
G_FS10 = "supplementary_figure_S10_sensing_cost"

ARM_FAMILY = len(ARMS)                    # each dataset's Holm family of the budget-matched arms
ADMITTED_RUNS = tuple(f"run:{d}" for d in ADMITTED)
GRAPH_RUNS = tuple(f"run:{d}" for d in GRAPH_LEVEL)


# ---------------------------------------------------------------------------
# Registered inputs
# ---------------------------------------------------------------------------

def registry_actions(ctx):
    """The executed action registry, checked against the registered Table S6 constants."""
    actions = operators.load_registry()
    registered = {row["id"]: row for row in ctx.constants["action_registry_table_S6"]["actions"]}
    if set(registered) != {a.id for a in actions}:
        raise DisplayInputError("the runtime action registry and the registered Table S6 list different actions")
    for a in actions:
        row = registered[a.id]
        pairs = [("rho", a.rho), ("nu", a.nu), ("alpha", a.alpha), ("exploration_probability", a.probability)]
        if a.is_resolvent:
            pairs += [("eta", a.eta), ("T", a.order)]
        for field, value in pairs:
            if row.get(field) is not None and not math.isclose(row[field], value, abs_tol=1e-12):
                raise DisplayInputError(f"A{a.id}: runtime {field} = {value} differs from the registered {row[field]}")
    return actions, registered


def kernel_coefficients(action) -> np.ndarray:
    """Coefficients of the kernel K_a in powers of P, from T_a = (1 - nu) I + nu K_a."""
    c = operators.operator_polynomial(action).copy()
    c[0] -= 1.0 - action.nu
    return c / action.nu


def registered_order(actions) -> int:
    return max(a.order for a in actions if a.is_resolvent)


def sparse_ratio(ctx, dataset) -> float:
    """r = E/(N d) from the published dataset statistics and the hidden width of Table S5."""
    stats = ctx.constants["dataset_statistics_table_S6c_S23b"][dataset]
    width = ctx.constants["backbones_table_S5"][dataset]["hidden"]
    return budget.sparse_to_dense_ratio(stats.get("reported_edges", stats.get("mean_reported_edges_per_graph")),
                                        stats.get("nodes", stats.get("mean_nodes_per_graph")), width)


def certificate_floor() -> float:
    """Registered acquisition threshold tau of Proposition 9: the floor of the executed E6 certificate."""
    from cm_experiments.budget import Certificate
    return next(f.default for f in dataclasses.fields(Certificate) if f.name == "floor")


def share_grid(ctx, dep):
    """Registered retained-share grid phi_x (low, central, high) of the comparator's regime."""
    regime = "MPC_OL_regimes" if dep["qC_route"] == "mpc_ol" else "fixed_sequence_regimes"
    return regime, tuple(ctx.constants[BUDGET]["retained_share_grid_phi_x"][regime])


def seeds(ctx, dataset) -> int:
    return len(ctx.config(dataset, "full")["training"]["seeds"])


def pairable_share(dataset) -> float:
    return PAIRABLE_SHARE["graph" if dataset in GRAPH_LEVEL else "single_graph"]


def unit_label(ctx, dataset) -> str:
    """Registered evaluation unit of the comparison (Table S42)."""
    if dataset in GRAPH_LEVEL:
        return "comparison graphs (fixed ensemble)"
    if dataset == "Roman-empire":
        splits = int(re.match(r"\d+", ctx.constants["data_roles_table_S2"]["single_graph_units"][dataset]).group())
        return f"{splits} official splits"
    return f"{ctx.config(dataset, 'full')['data']['num_clusters']} buffered cores"


# ---------------------------------------------------------------------------
# Measured inputs
# ---------------------------------------------------------------------------

def deployment(ctx, dataset) -> dict:
    """Table 1 and Table S23 quantities of one lifecycle run (split means for a campaign)."""
    view = ctx.require_run(dataset)
    record = view.evidence
    paired = (record.get("final") or {}).get("paired_loss") or {}
    if paired.get("mean") is None:
        raise DisplayInputError(f"{dataset}: the final paired loss effect is not recorded")
    members = []
    for member in view.members():
        evidence = member.evidence
        cost = evidence.get("cost") or {}
        if not cost.get("C_qC") or cost.get("C_qD") is None:
            raise DisplayInputError(f"{dataset}: {member.root.name} records no comparator or deployed serial cost")
        members.append({"run": member.root.name, "C_ctrl": float(cost["C_qD"]), "C_qC": float(cost["C_qC"]),
                        "headroom_upper": ((evidence.get("entry") or {}).get("headroom") or {}).get("upper")})
    selection, admitted = record.get("selection") or {}, record.get("admission") or {}
    headroom = [m["headroom_upper"] for m in members]
    c_ctrl, c_qc = fmean(m["C_ctrl"] for m in members), fmean(m["C_qC"] for m in members)
    plant = ((record.get("final") or {}).get("models") or {}).get("qN") or {}
    return {"dataset": dataset, "view": view, "effect": float(paired["mean"]), "lower": paired.get("lower"),
            "upper": paired.get("upper"), "gain": -float(paired["mean"]), "C_ctrl": c_ctrl, "C_qC": c_qc,
            "m": c_ctrl / c_qc, "E": (admitted.get("deployed") or {}).get("ensemble"),
            "qC_route": (selection.get("qC") or {}).get("route"),
            "returns_qC": admitted.get("Gadmit") == 0 or admitted.get("deployed_index") == selection.get("qC_index"),
            "headroom_upper": None if None in headroom else fmean(headroom),
            "l_plant": plant.get("loss_nats"), "members": members}


def overheads(ctx, dep) -> dict:
    """Expected (main Eq. 32) and assurance overheads over q_C with the input-gate fallback probability."""
    p = ctx.require_records("input_gate").require("fallback_probability", dataset=dep["dataset"])
    expected = economics.expected_cost(dep["C_ctrl"], dep["C_qC"], p)
    assurance = economics.assurance_cost(dep["C_ctrl"], dep["C_qC"])
    return {"p_fallback": p, "expected": expected, "assurance": assurance,
            "o": expected / dep["C_qC"] - 1.0, "o_assure": assurance / dep["C_qC"] - 1.0}


def admission_moments(ctx, dataset) -> dict:
    """Clipped admission moments of Table S13 (admit/decision.json, empirical Bernstein)."""
    eb = (ctx.require_run(dataset).evidence.get("admission") or {}).get("empirical_bernstein") or {}
    if any(eb.get(k) is None for k in ("mean", "sample_variance", "units", "clip")):
        raise DisplayInputError(f"{dataset}: the admission moments (D_bar, S^2, J, B) are not recorded")
    return {"D_bar": float(eb["mean"]), "S2": float(eb["sample_variance"]), "J": int(eb["units"]), "B": float(eb["clip"])}


def support_counts(ctx, dataset) -> dict:
    """Minimum trajectory count over all support cells and the minimum ESS over qualified cells (fit audit)."""
    view = ctx.require_run(dataset)
    counts, qualified, per_member = [], [], []
    for member in view.members():
        relative = "fit/full/audit.json" if member.exists("fit/full/audit.json") else "fit/lean/audit.json"
        cells = ((member.json(relative).get("support") or {}).get("cells") or {}).values()
        if not cells:
            raise DisplayInputError(f"{dataset}: {relative} holds no support cells")
        member_counts = [int(c["trajectory_count"]) for c in cells]
        member_ess = [float(c["ess"]) for c in cells if c.get("qualified")]
        counts += member_counts
        qualified += member_ess
        per_member.append((member.root.name, relative, min(member_counts), min(member_ess, default=None)))
    if not qualified:
        raise DisplayInputError(f"{dataset}: no qualified support cell")
    return {"minimum_count": min(counts), "minimum_qualified_cell_ess": min(qualified),
            "per_member": per_member}


def ensemble_fit(ctx, dataset) -> dict:
    """Least-squares ensemble law l(E) = l_inf + D_inf/E on the E14 grid (Proposition 5)."""
    rows = ctx.require_records("ensemble_grid").select("loss_nats", dataset=dataset)
    if len(rows) < 3:
        raise DisplayInputError(f"ensemble_grid: {dataset} needs at least three ensemble sizes")
    sizes = [float(r["ensemble"]) for r in rows]
    return {**economics.fit_ensemble_law(sizes, [r["value"] for r in rows]), "sizes": sizes}


def arm_value(ctx, dataset, arm, quantity, *, required=True):
    value = ctx.require_records("budget_matched").value(quantity, dataset=dataset, arm=arm)
    if value is None and required:
        raise DisplayInputError(f"budget_matched: {quantity} of {dataset}/{arm} is not recorded")
    return value


def paired_se(ctx, dataset) -> float:
    """Paired SE of the E3 contrast: its 95% crossed seed-by-unit interval against q_M over 2 z_0.975."""
    low = arm_value(ctx, dataset, "E3", "effect_vs_feedback_ci_low_nats")
    high = arm_value(ctx, dataset, "E3", "effect_vs_feedback_ci_high_nats")
    return (high - low) / (2.0 * Z975)


def qb_overheads(ctx, dataset) -> dict:
    """Executed E6 work over q_C (Table S31 record: E6 cost ledger); optional abort-free quantities."""
    ratio = arm_value(ctx, dataset, "E6", "work_ratio")
    completed = arm_value(ctx, dataset, "E6", "work_ratio_completed", required=False)
    return {"o_B": ratio - 1.0, "p1": arm_value(ctx, dataset, "E6", "late_abort_fraction", required=False),
            "o_B_p1_zero": None if completed is None else completed - 1.0}


def gap_model(ctx, dataset) -> dict:
    """Log-normal action-gap model of Table S34: skip = Pr(gap > tau), r_B = 1 - skip x switch."""
    records = ctx.require_records(GAP_SET)
    median = records.require("median_gap_nats", dataset=dataset)
    p5 = records.require("p5_gap_nats", dataset=dataset)
    switch = records.require("switch_fraction", dataset=dataset)
    if not 0 < p5 < median:
        raise DisplayInputError(f"{GAP_SET}: {dataset} needs 0 < p5 < median")
    tau = certificate_floor()
    mu, sigma = math.log(median), (math.log(median) - math.log(p5)) / Z95
    skip = 1.0 - NORMAL.cdf((math.log(tau) - mu) / sigma)
    return {"median": median, "p5": p5, "switch": switch, "tau": tau, "mu": mu, "sigma": sigma,
            "skip": skip, "r_B": 1.0 - skip * switch}


def scenario_constant(ctx, key):
    """A Note S25 parameter registered with the budget-matched comparison (None until registered)."""
    return ctx.constants[BUDGET].get(key)


# ---------------------------------------------------------------------------
# Small formatting and power helpers
# ---------------------------------------------------------------------------

def pct(fraction, decimals=1):
    return None if fraction is None else f"{100.0 * fraction:.{decimals}f}%"


def joined(values, decimals, sep=" / "):
    return sep.join(f"{v:.{decimals}f}" for v in values)


def seed_range_text(conservative, optimistic):
    a, b = f"{conservative:.0f}", f"{optimistic:.0f}"
    return f"{a}%" if a == b else f"{a}–{b}%"


def p_text(p, floor):
    """P-value text as Table S14 prints it: '< floor' below the floor, four decimals below 0.01, else three."""
    if p is None:
        return None
    if p < floor:
        return "< 10^-4" if floor == 1e-4 else f"< {floor:g}"
    return f"{p:.4f}" if p < 0.01 else f"{p:.3f}"


def p_compared(p, floor):
    """The printed reading of a P value below its floor ('< 1e-4', '< 0.001'), else the value."""
    if p is None or p >= floor:
        return p
    return "< 1e-4" if floor == 1e-4 else f"< {floor:g}"


def power(margin, se, family=1) -> float:
    """Power (percent) of the two-sided paired test at the true margin, first Holm step of ``family`` tests."""
    return 100.0 * inference.power_of_contrast(margin, se, family_size=family)


def seed_paired_se(se, share, count) -> tuple[float, float]:
    """Conservative (only the pairable share shrinks with the seeds) and optimistic (sigma/sqrt(S)) SEs."""
    return se * math.sqrt((1.0 - share) + share / count), se / math.sqrt(count)


def require_gain(dep) -> dict:
    """The displays of the admitted datasets model a deployed controller with a positive gain over q_C."""
    if dep["returns_qC"] or dep["gain"] <= 0:
        raise DisplayInputError(f"{dep['dataset']}: the run deploys no feedback rule with a positive gain over q_C")
    return dep


def deployed(ctx, dataset) -> dict:
    return require_gain(deployment(ctx, dataset))


def power_inputs(ctx, dataset) -> dict:
    """Margins, SE and seed design of the E3 contrast (Tables S42 and S47, Figures S12 and S20d), in mnats."""
    dep = deployed(ctx, dataset)
    regime, (low, central, high) = share_grid(ctx, dep)
    g = 1e3 * dep["gain"]
    se = 1e3 * paired_se(ctx, dataset)
    count, share = seeds(ctx, dataset), pairable_share(dataset)
    cons, opt = seed_paired_se(se, share, count)
    return {"dep": dep, "regime": regime, "phi": (low, central, high), "g": g, "se": se,
            "margin": (1 - central) * g, "margin_range": ((1 - high) * g, (1 - low) * g),
            "seeds": count, "share": share, "se_cons": cons, "se_opt": opt}



# ---------------------------------------------------------------------------
# Supplementary Table S6 and S6b: the action registry
# ---------------------------------------------------------------------------

@display("supplementary:Table S6", stem="S6", title="Ten-action registry and per-step cost", scope="analytical")
def table_s6(ctx, out):
    actions, registered = registry_actions(ctx)
    rows, source = [], []
    for a in actions:
        reg = registered[a.id]
        transport, resolvent = a.is_transport, a.is_resolvent
        mass = operators.identity_mass(a) if resolvent else None
        products = operators.maximum_reach(a) if transport else 0
        na = not_applicable("This action does not use the transport or resolvent parameter")
        rows.append([f"A{a.id}", reg["role"], num(a.rho, 2) if transport else na,
                     num(a.nu, 2) if transport else na, num(a.alpha, 2) if transport else na,
                     f"η = {a.eta:.2f}" if resolvent else reg["mode"], integer(a.order) if resolvent else na,
                     num(mass, 4) if resolvent else na, integer(products), num(a.probability, 2), reg["interpretation"]])
        coefficients = operators.operator_polynomial(a) if transport else []
        source.append([f"A{a.id}", a.name, a.mode, a.rho, a.nu, a.alpha, a.eta, a.order if resolvent else None,
                       mass, products, a.probability, " ".join(repr(float(c)) for c in coefficients)])
        values = {"rho": a.rho if transport else None, "nu": a.nu if transport else None,
                  "alpha": a.alpha if transport else None, "eta": a.eta if resolvent else None,
                  "T": a.order if resolvent else None, "identity_mass": mass, "sparse_products": products,
                  "exploration_probability": a.probability}
        for key, value in values.items():
            out.compare("supplementary_table_S6_action_registry", (f"A{a.id}", key), value)
    total = math.fsum(a.probability for a in actions)
    out.table(["ID", "role", "ρ", "ν", "α", "mode/range", "T", "identity mass", "sparse products",
               "explore p_ctrl", "exact interpretation"], rows,
              notes=[f"Exploration probabilities sum to {total:.2f}; A8 and A9 are hybrid resets, not transport mixtures.",
                     "Parameters from the runtime registry (cross_morse/core.py), checked against the registered "
                     "Table S6 constants; identity mass (1 - eta)^(T+1) and sparse products per step (the highest power "
                     "of P formed on line) from cm_theory.operators."])
    out.data("action_registry", ["action", "name", "mode", "rho", "nu", "alpha", "eta", "T", "identity_mass",
                                 "sparse_products", "exploration_probability", "operator_coefficients_P0_to_PT"], source)
    out.uses("repository", "cross_morse/core.py", ["ACTIONS"])
    out.uses("repository", "paper/registered_constants.json", ["action_registry_table_S6"])


@display("supplementary:Table S6b", stem="S6b", title="Action range, reach and product identities",
         scope="analytical")
def table_s6b(ctx, out):
    actions, _ = registry_actions(ctx)
    rows, source = [], []
    for a in (x for x in actions if x.is_transport):
        kernel = kernel_coefficients(a)
        hops = float(np.dot(np.arange(len(kernel)), kernel))
        executed = operators.executed_range(a)
        reach = operators.maximum_reach(a)
        coefficient = operators.identity_coefficient(a)
        mass = operators.identity_mass(a) if a.is_resolvent else None
        rows.append([f"A{a.id}", num(a.nu, 2), num(hops, 3) if a.is_resolvent else integer(hops), num(executed, 3),
                     integer(reach), num(mass, 4) if a.is_resolvent else
                     not_applicable("The action has no resolvent kernel identity mass"), num(coefficient, 3), integer(reach)])
        source.append([f"A{a.id}", a.nu, hops, executed, reach, mass, coefficient, reach,
                       operators.executed_range_closed_form(a), operators.identity_coefficient_closed_form(a)])
        for key, value in (("nu", a.nu), ("kernel_hops", hops), ("executed_range", executed), ("max_reach", reach),
                           ("kernel_identity_mass", mass), ("executed_identity_coefficient", coefficient),
                           ("sparse_products", reach)):
            out.compare("supplementary_table_S6b_executed_actions", (f"A{a.id}", key), value)
    nominal = next(a for a in actions if a.id == 0)
    widest = max((a for a in actions if a.is_resolvent), key=operators.identity_mass)
    out.table(["ID", "ν", "kernel hops", "executed range h̄_a (hops/step)", "max. reach (hops)",
               "kernel identity mass", "executed identity coefficient", "sparse products"], rows,
              notes=[f"A{widest.id} moves less mass per step than the nominal action "
                     f"(h̄ = {operators.executed_range(widest):.3f} against {operators.executed_range(nominal):.3f}): its "
                     f"kernel keeps {operators.identity_mass(widest):.2f} of its row mass on the diagonal and its executed "
                     f"operator {operators.identity_coefficient(widest):.2f}.",
                     "Exact registry functions of main Eq. (10) (cm_theory.operators): kernel hops sum_t t k_t of "
                     "K_a = (1 - rho) P + rho K~_eta,T, executed range nu times the kernel hops, maximum reach the highest "
                     "power of P in T_a."])
    out.data("executed_range", ["action", "nu", "kernel_hops", "executed_range", "max_reach", "kernel_identity_mass",
                                "executed_identity_coefficient", "sparse_products", "executed_range_closed_form",
                                "identity_coefficient_closed_form"], source)
    out.uses("repository", "cross_morse/core.py", ["ACTIONS"])


# ---------------------------------------------------------------------------
# Supplementary Table S14b: confirmation-size sensitivity
# ---------------------------------------------------------------------------

def graph_variance_share(ctx, dataset) -> dict:
    """Upper bound on the graph share of the test-stage variance under the two-way model (Note S24).

    rho = hw_test / hw_graph with hw_graph = z_0.975 (S^2/J)^(1/2) of the fixed deployed ensemble
    (Table S13) and hw_test the half-width of the Table 1 interval; the graph share is rho^-2.
    """
    moments = admission_moments(ctx, dataset)
    dep = deployment(ctx, dataset)
    if dep["lower"] is None or dep["upper"] is None:
        raise DisplayInputError(f"{dataset}: the Table 1 interval is not recorded")
    graph = Z975 * math.sqrt(moments["S2"] / moments["J"])
    test = (dep["upper"] - dep["lower"]) / 2.0
    if graph <= 0 or test <= 0:
        raise DisplayInputError(f"{dataset}: a zero-width admission or test interval leaves the two-way shares undefined")
    return {**moments, "hw_graph": graph, "hw_test": test, "ratio": test / graph, "share": (graph / test) ** 2,
            "dep": dep}


@display("supplementary:Table S14b", stem="S14b", title="Confirmation-size sensitivity", scope="analytical",
         needs=(f"records:{CONTRAST_SET}",) + GRAPH_RUNS)
def table_s14b(ctx, out):
    from cm_campaigns.confirmation import FAMILY as CONTRASTS
    records = ctx.require_records(CONTRAST_SET)
    constants = ctx.constants
    alpha = constants["confirmation_eq20_table_S11"]["holm_alpha"]
    confirm_sizes = constants["data_roles_table_S2"]["multi_graph_counts"]
    mechanism = {"state": "state dependence", "radius": "radius", "mpc": "replanning"}
    items, shares = [], {}
    for key, (dataset, kind, _, j_new, _, direction, margin) in CONTRASTS.items():
        effect, low, high, j_old = (records.require(q, contrast=key) for q in
                                    ("effect_nats", "interval_low_nats", "interval_high_nats", "unit_count"))
        j_old = int(j_old)
        if dataset in GRAPH_LEVEL and j_new != confirm_sizes[dataset]["D_confirm"]:
            raise DisplayInputError(f"{key}: the S14 family size {j_new} differs from the registered D_confirm size")
        if dataset not in GRAPH_LEVEL and j_old != j_new:
            raise DisplayInputError(f"{key}: {j_old} original units differ from the registered {j_new}; only "
                                    "graph-unit contrasts are rescaled")
        se_old = (high - low) / (2.0 * Z975)
        rescaled = j_new != j_old
        se = se_old
        if rescaled:
            shares[dataset] = shares.get(dataset) or graph_variance_share(ctx, dataset)
            g = shares[dataset]["share"]
            se = se_old * math.sqrt((1.0 - g) + g * j_old / j_new)
            raw = 2.0 * (1.0 - NORMAL.cdf(abs(effect) / se))
        else:
            raw = records.require("p_value_zero", contrast=key)
        beyond = direction * effect > margin
        p_margin = 1.0 - NORMAL.cdf((direction * effect - margin) / se) if beyond else None
        items.append({"key": key, "dataset": dataset, "kind": kind, "j_old": j_old, "j_new": j_new, "effect": effect,
                      "old": (low, high), "new": (effect - Z975 * se, effect + Z975 * se), "se_old": se_old, "se": se,
                      "raw": raw, "p_margin": p_margin, "beyond": beyond, "rescaled": rescaled, "margin": margin})
    for item, holm in zip(items, inference.holm_adjust([i["raw"] for i in items])):
        item["holm"] = float(holm)
    rows, source = [], []
    unchanged = rescaled = 0
    for item in items:
        if item["rescaled"]:
            low, high = item["new"]
            if item["holm"] < alpha and item["beyond"]:
                reading = f"{mechanism[item['kind']]} retained"
            else:
                side = "excludes" if low > 0 or high < 0 else "includes"
                reading = (f"interval {side} zero; Holm P above {alpha:g} under this model; "
                           "the confirmation record decides the label (Note S26)")
        else:
            reading = "unchanged"
        old = f"[{item['old'][0]:+.5f}, {item['old'][1]:+.5f}]"
        new = f"[{item['new'][0]:+.5f}, {item['new'][1]:+.5f}]" if item["rescaled"] else "same"
        rows.append([item["key"], integer(item["j_old"]), integer(item["j_new"]), f"{item['effect']:+.5f}", old, new,
                     p_text(item["raw"], 1e-4), p_text(item["holm"], 1e-3),
                     p_text(item["p_margin"], 1e-3) if item["beyond"] else "not tested", reading])
        group, index = (G_S14B, rescaled) if item["rescaled"] else (G_S14B_UNCHANGED, unchanged)
        if item["rescaled"]:
            rescaled += 1
            out.compare(group, (index, "interval_old"), list(item["old"]))
            out.compare(group, (index, "graph_variance_share"), shares[item["dataset"]]["share"])
        else:
            unchanged += 1
        out.compare(group, (index, "interpretation"), reading)
        printed_interval = item["new"] if item["rescaled"] else item["old"]
        for field, computed in (("J_old", item["j_old"]), ("J_new", item["j_new"]), ("effect", item["effect"]),
                                ("interval", list(printed_interval)), ("raw_P", p_compared(item["raw"], 1e-4)),
                                ("holm_P", p_compared(item["holm"], 1e-3)),
                                ("P_vs_margin", p_compared(item["p_margin"], 1e-3))):
            out.compare(group, (index, field), computed)
        share = shares.get(item["dataset"], {}).get("share") if item["rescaled"] else None
        source.append([item["key"], item["dataset"], item["j_old"], item["j_new"], item["effect"], *item["old"],
                       item["se_old"], share, item["se"], *item["new"], item["raw"], item["holm"], item["p_margin"],
                       item["margin"]])
    out.table(["Dataset/claim", "J_old", "J_new", "effect", "95% interval (J_old)", "95% interval (J_new)", "raw P",
               "Holm P", "P vs margin", "interpretation"], rows,
              notes=["Only the graph component of the variance scales with the graph count: SE_new = SE_old "
                     "[(1 - g) + g J_old/J_new]^(1/2), g = (hw_graph/hw_test)^2 from the admission moments (Table S13) and "
                     "the Table 1 interval (Note S24). Point estimates are held fixed; rescaled rows read raw P and P vs "
                     "margin from the normal approximation and the Holm family is re-formed over the nine contrasts.",
                     "P vs margin is one-sided beyond the registered practical margin, computed when the point estimate "
                     "lies beyond it."])
    out.data("contrasts", ["contrast", "dataset", "J_old", "J_new", "effect_nats", "interval_old_low", "interval_old_high",
                           "se_old", "graph_variance_share", "se_new", "interval_new_low", "interval_new_high", "raw_P",
                           "holm_P", "P_vs_margin", "practical_margin"], source)
    out.data("graph_variance_share", ["dataset", "J", "S2", "hw_graph_nats", "hw_test_nats", "ratio", "graph_share"],
             [[d, s["J"], s["S2"], s["hw_graph"], s["hw_test"], s["ratio"], s["share"]] for d, s in shares.items()])
    out.uses("records", CONTRAST_SET, [records.name])
    for dataset in GRAPH_LEVEL:
        out.uses("run", dataset, ["admit/decision.json", "final/results_summary.json"])


# ---------------------------------------------------------------------------
# Supplementary Tables S28 and S28b: one-step regret bounds
# ---------------------------------------------------------------------------

@display("supplementary:Table S28", stem="S28", title="Distribution-free one-step regret bound", scope="analytical",
         needs=("runs",))
def table_s28(ctx, out):
    params = regret_data.parameters(ctx)
    inputs = regret_data.reference_support(ctx)
    rows, source, observed = [], [], []
    for dataset, item in inputs.items():
        support = support_counts(ctx, dataset)
        dep = deployment(ctx, dataset)
        effect = None if dep["returns_qC"] else abs(dep["effect"])
        at_count = 2 * admission.hoeffding_radius(item["minimum_count"], **params)
        at_ess = 2 * admission.hoeffding_radius(item["reference_ess"], **params)
        n_effect = None if effect is None else regret_data.hoeffding_equality(effect, params)
        rows.append([dataset, integer(item["minimum_count"]), integer(item["reference_ess"]), num(at_count, 4),
                     num(at_ess, 4), not_applicable("The deployed policy returned qC; no feedback effect defines a target sample size")
                     if n_effect is None else f"{round(n_effect):,}"])
        source.append([dataset, item["minimum_count"], item["reference_ess"], at_count, at_ess, effect, n_effect,
                       None if n_effect is None else round(n_effect),
                       None if effect is None else admission.units_for_hoeffding_regret(effect, **params)])
        observed += [[dataset, run, relative, count, ess] for run, relative, count, ess in support["per_member"]]
        for key, value in (("minimum_count", item["minimum_count"]), ("reference_ess", item["reference_ess"]),
                           ("two_e_minimum_count", at_count), ("two_e_reference_ess", at_ess),
                           ("n_for_effect_rounded", None if n_effect is None else round(n_effect))):
            out.compare(regret_data.S28_TARGETS, ("datasets", dataset, key), value)
        out.uses("run", dataset, [*(r[1] for r in support["per_member"]), "final/results_summary.json"])
    for key, value in (("B", params["B"]), ("actions", params["actions"]), ("delta_sup", params["delta"])):
        out.compare(regret_data.S28_TARGETS, ("parameters", key), value)
    out.table(["Dataset", "minimum count", "reference ESS", "2e at minimum count (nats)", "2e at reference ESS (nats)",
               "n for the dataset's effect"], rows,
              notes=[f"e(n) = 2B [log(2|A|/delta_sup)/(2n)]^(1/2) with B = {params['B']:g}, "
                     f"|A| = {params['actions']} and delta_sup = {params['delta']:g}; the regret bound is 2e(n). "
                     "The first two columns are the explicitly listed analytical scenario of SI Table S28 (pp. 18–19). "
                     "The source caption distinguishes its qualified-cell reference ESS from the Table S7 minima. "
                     "Observed support minima from the supplied runs are retained in observed_support.csv. "
                     "The last column uses the supplied dataset's admitted |Delta| (main Table 1), with real n at "
                     "equality rounded to the nearest integer; the smallest sufficient integer is separately "
                     "exported in the source data. A dataset that returns q_C has no effect target."])
    out.data("regret_bound", ["dataset", "minimum_count", "reference_ess", "two_e_minimum_count", "two_e_reference_ess",
                              "observed_abs_effect_nats", "n_for_effect_at_equality", "n_for_effect_rounded",
                              "minimum_integer_n_for_effect"], source)
    out.data("observed_support", ["dataset", "run", "audit_file", "observed_minimum_trajectory_count",
                                  "observed_minimum_qualified_cell_ESS"], observed)
    out.provenance(regret_reference_scenario={
        "source": {"document": "supplementary", "pages": [18, 19], "display": "Table S28"},
        "counts_role": "Declared analytical inputs; not observations inferred from the supplied runs",
        "effect_role": "Dataset effect read from the supplied lifecycle run",
        "observed_support": "data/S28/observed_support.csv"})
    regret_data.record_parameters(out, params)


@display("supplementary:Table S28b", stem="S28b", title="Variance-aware one-step radius", scope="analytical",
         needs=("static",))
def table_s28b(ctx, out):
    params = regret_data.parameters(ctx)
    sigmas, counts, targets = regret_data.variance_grid(ctx)
    rows, source, requirements = [], [], []
    out.compare(regret_data.S28B_TARGETS, ("sigma1_grid",), sigmas)
    for n in counts:
        values = [2 * admission.hoeffding_radius(n, **params)] + [
            2 * admission.bernstein_radius(n, sigma, **params) for sigma in sigmas]
        rows.append([f"{n:,}"] + [num(v, 4) for v in values])
        source.append(["count", n, *values])
        for key, value in (("count", n), ("hoeffding_two_e", values[0]), ("bernstein_two_e", values[1:])):
            out.compare(regret_data.S28B_TARGETS, ("count_rows", str(n), key), value)
    rows.append(["target (nats)", "Hoeffding n"] + [f"n at σ1 = {sigmas[0]:.3f}"] +
                [f"σ1 = {s:.3f}" for s in sigmas[1:]])
    for item in targets:
        target, label = item["target_nats"], item["label"]
        needed = regret_data.target_requirements(target, sigmas, params)
        rows.append([f"{target:.4f} ({label})", f"{needed['hoeffding_n_rounded']:,}"] +
                    [f"{n:,}" for n in needed["bernstein_n"]])
        source.append(["target", target, needed["hoeffding_n_at_equality"], *needed["bernstein_n"]])
        requirements.append([item["id"], label, target, needed["hoeffding_n_at_equality"],
                             needed["hoeffding_n_rounded"], needed["hoeffding_minimum_integer_n"],
                             *needed["bernstein_n"]])
        for key, value in (("label", label), ("target_nats", target),
                           ("hoeffding_n_rounded", needed["hoeffding_n_rounded"]),
                           ("bernstein_n", needed["bernstein_n"])):
            out.compare(regret_data.S28B_TARGETS, ("targets", item["id"], key), value)
    out.table(["n", "Hoeffding 2e(n)"] + [f"Bernstein, σ1 = {sigmas[0]:.3f}"] +
              [f"σ1 = {s:.3f}" for s in sigmas[1:]], rows,
              notes=["Empirical-Bernstein regret bound 2 e_B(n), e_B(n) = (2 sigma_1^2 L/n)^(1/2) + 7(2B)L/(3(n - 1)), "
                     "L = log(4|A|/delta_sup) (cm_theory.admission). The count grid and four target scales are explicit "
                     "analytical inputs printed in SI Table S28b (p. 19), including its COCO target of 0.0090 nats. "
                     "They are not substituted with support minima or rounded effects from a supplied run. "
                     "The lower block gives the smallest integer n with 2e_B(n) at most the target and the real "
                     "Hoeffding n at equality rounded to the nearest unit. The smallest sufficient Hoeffding integer "
                     "is separately exported in target_requirements.csv. This table establishes arithmetic "
                     "agreement for the stated scenario, not observed benchmark support or effects."])
    out.data("radius_grid", ["block", "n_or_target", "hoeffding", *[f"sigma1_{s:.3f}" for s in sigmas]], source)
    out.data("target_requirements", ["target_id", "label", "target_nats", "hoeffding_n_at_equality",
                                     "hoeffding_n_rounded", "hoeffding_minimum_integer_n",
                                     *[f"bernstein_n_sigma1_{s:.3f}" for s in sigmas]], requirements)
    out.provenance(regret_reference_scenario={
        "source": {"document": "supplementary", "pages": [19], "display": "Table S28b"},
        "role": "Analytical reconstruction from explicitly declared input coordinates; no measured run inputs"})
    regret_data.record_parameters(out, params, variance=True)


# ---------------------------------------------------------------------------
# Supplementary Tables S33 and S34: retained-share and selective-sensing models
# ---------------------------------------------------------------------------

@display("supplementary:Table S33", stem="S33", title="Retained-share model for comparison arms", scope="analytical",
         needs=ADMITTED_RUNS + (f"records:{GAP_SET}", "records:budget_matched"))
def table_s33(ctx, out):
    phi_ol = scenario_constant(ctx, "phi_OL")
    rows, source = [], []
    for dataset in ADMITTED:
        dep = deployed(ctx, dataset)
        regime, grid = share_grid(ctx, dep)
        r_b = gap_model(ctx, dataset)["r_B"]
        o_b = qb_overheads(ctx, dataset)["o_B"]
        g = dep["gain"]
        for label, phi, rho in zip(("low share", "central", "high share"), grid, RE_SEARCH_SHARES):
            if regime == "MPC_OL_regimes":
                no_observation = -rho * g
            else:
                no_observation = None if phi_ol is None else -(rho + phi_ol) * g
            values = [-phi * g, -0.9 * phi * g, no_observation, -r_b * g, -(r_b - phi) * g, -(1 - phi) * g]
            rows.append([SHORT[dataset], label, num(phi, 2), num(rho, 2), num(r_b, 3)] +
                        [num(v, 4) for v in values] + [pct(o_b)])
            source.append([dataset, regime, label, phi, rho, r_b, g, *values, o_b])
        out.uses("run", dataset, ["final/results_summary.json", "select/decision.json"])
    notes = ["Note S25 at the feedback gain g = |Delta| of main Table 1: predictor -phi g, distilled -0.9 phi g, "
             "no-observation replanning -rho g (MPC-OL comparators) or -(rho + phi_OL) g (fixed-sequence comparators), "
             "q_B -r_B g with r_B = 1 - skip x switch of Table S34, q_B minus predictor -(r_B - phi) g, feedback minus "
             "predictor -(1 - phi) g; phi on the registered grid of the comparator's regime, rho on the Note S25 grid. "
             "q_B overhead: executed E6 work over q_C (budget_matched)."]
    if phi_ol is None:
        notes.append("phi_OL is not registered (Note S25 names it without a value): the no-observation cells of the "
                     "fixed-sequence regimes stay empty until registered_constants.json holds it.")
    out.table(["Dataset", "scenario", "φ_x^gain", "re-search share", "r_B", "predictor vs q_C", "distilled vs q_C",
               "no-observation replanning vs q_C", "q_B vs q_C", "q_B minus predictor", "feedback minus predictor",
               "q_B overhead"], rows, notes=notes)
    out.data("retained_share_model", ["dataset", "regime", "scenario", "phi", "rho", "r_B", "gain_nats", "predictor",
                                      "distilled", "no_observation", "qB", "qB_minus_predictor",
                                      "feedback_minus_predictor", "qB_overhead"], source)
    out.uses("records", GAP_SET)
    out.uses("records", "budget_matched")


@display("supplementary:Table S34", stem="S34", title="Selective-sensing and branch-cost model", scope="analytical",
         needs=ADMITTED_RUNS + (f"records:{GAP_SET}", "records:input_gate", "records:budget_matched"))
def table_s34(ctx, out):
    rows, source = [], []
    for dataset in ADMITTED:
        gap = gap_model(ctx, dataset)
        o = overheads(ctx, deployed(ctx, dataset))["o"]
        qb = qb_overheads(ctx, dataset)
        rows.append([SHORT[dataset], f"{1e3 * gap['median']:.1f} / {1e3 * gap['p5']:.1f}", num(gap["switch"], 3),
                     num(gap["skip"], 2), num(gap["r_B"], 3), pct(o), pct(qb["o_B"]), num(qb["p1"], 3),
                     pct(qb["o_B_p1_zero"])])
        source.append([dataset, gap["median"], gap["p5"], gap["mu"], gap["sigma"], gap["tau"], gap["switch"], gap["skip"],
                       gap["r_B"], o, qb["o_B"], qb["p1"], qb["o_B_p1_zero"]])
        for field, computed in (("gap_median_mnats", 1e3 * gap["median"]), ("gap_p5_mnats", 1e3 * gap["p5"]),
                                ("switch_fraction", gap["switch"]), ("skip_rate", gap["skip"]),
                                ("retained_fraction_r_B", gap["r_B"]), ("present_overhead_pct", 100 * o),
                                ("qB_overhead_pct", 100 * qb["o_B"]), ("p1", qb["p1"]),
                                ("qB_overhead_at_p1_zero_pct", None if qb["o_B_p1_zero"] is None else 100 * qb["o_B_p1_zero"])):
            out.compare(G_S34, (dataset, field), computed)
        out.uses("run", dataset, ["select/candidates.json"])
    tau = certificate_floor()
    out.table(["Dataset", "gap median / p5 (mnats)", "switch fraction", "skip rate", "retained fraction",
               "present overhead", "q_B overhead", "p1", "q_B overhead at p1 = 0"], rows,
              notes=[f"Skip rate: Pr(gap > tau) with tau = {tau:g} nats (registered certificate floor) under the "
                     "log-normal gap law fitted to the median and 5th percentile of the action-gap audit; an upper bound "
                     "on certifiable skips (Note S23). r_B = 1 - skip x switch. Present overhead: expected overhead of "
                     "main Eq. (32) with the input-gate fallback probability. q_B overhead, p1 and the abort-free "
                     "overhead: executed E6 work, late-abort fraction and completed-input work of the budget_matched "
                     "record (the last two are optional quantities of that set)."])
    out.data("selective_sensing", ["dataset", "median_gap_nats", "p5_gap_nats", "mu_log", "sigma_log", "tau_nats",
                                   "switch_fraction", "skip_rate", "r_B", "present_overhead", "qB_overhead", "p1",
                                   "qB_overhead_p1_zero"], source)
    for name in (GAP_SET, "input_gate", "budget_matched"):
        out.uses("records", name)


# ---------------------------------------------------------------------------
# Supplementary Tables S42 and S47: power and decision boundaries
# ---------------------------------------------------------------------------

def ensemble_spend_power(ctx, dataset, inputs):
    """Power at the central margin when the predictor spends o - o_dec on members (Proposition 10(ii))."""
    o_dec = scenario_constant(ctx, "decoding_overhead_o_dec")
    if o_dec is None:
        return None, None
    dep = inputs["dep"]
    spare = overheads(ctx, dep)["o"] - o_dec
    gain = 1e3 * economics.spare_budget_gain(ensemble_fit(ctx, dataset)["D_inf"], dep["E"], spare)
    return power(inputs["margin"] - gain, inputs["se"]), gain


@display("supplementary:Table S42", stem="S42", title="Budget-matched test power", scope="analytical",
         needs=ADMITTED_RUNS + ("records:budget_matched", "records:ensemble_grid", "records:input_gate"))
def table_s42(ctx, out):
    rows, source = [], []
    for dataset in ADMITTED:
        x = power_inputs(ctx, dataset)
        se, margin, (low, high) = x["se"], x["margin"], x["margin_range"]
        mde = (inference.minimum_detectable_effect(se), inference.minimum_detectable_effect(se, family_size=ARM_FAMILY))
        unit = (power(margin, se), power(low, se), power(high, se))
        holm = power(margin, se, ARM_FAMILY)
        cons, opt = power(margin, x["se_cons"], ARM_FAMILY), power(margin, x["se_opt"])
        spend, spend_gain = ensemble_spend_power(ctx, dataset, x)
        label = unit_label(ctx, dataset)
        rows.append([SHORT[dataset], label, num(se / 1e3, 5), joined(mde, 2), f"{margin:.2f} [{low:.2f}, {high:.2f}]",
                     f"{unit[0]:.0f}% [{unit[1]:.0f}, {unit[2]:.0f}]", f"{holm:.0f}%", seed_range_text(cons, opt),
                     None if spend is None else f"{spend:.0f}%"])
        source.append([dataset, label, se, x["g"], *x["phi"], margin, low, high, *mde, *unit, holm, x["seeds"],
                       x["share"], x["se_cons"], x["se_opt"], cons, opt, spend_gain, spend])
        for field, computed in (("unit", label), ("paired_SE_nats", se / 1e3), ("MDE_mnats_0.05", mde[0]),
                                ("MDE_mnats_holm", mde[1]), ("reported_margin_mnats", margin),
                                ("margin_range", [low, high]), ("power_at_unit_pct", unit[0]),
                                ("power_range", [unit[1], unit[2]]), ("power_holm_step_pct", holm),
                                ("power_seed_paired_pct", [cons, opt]), ("power_ensemble_spend_pct", spend)):
            out.compare(G_S42, (dataset, field), computed)
        out.uses("run", dataset, ["final/results_summary.json", "select/decision.json"])
    notes = [f"Paired SE of the E3 contrast from its 95% crossed seed-by-unit interval (budget_matched). MDE = "
             f"(z_(1-a/2) + z_0.8) SE at a = 0.05 and at the first Holm step a/{ARM_FAMILY}. Margin FB-E3 = (1 - phi) g on "
             "the registered phi grid. Power of the two-sided paired test at the margin; seed-paired: conservative "
             "(only the pairable share 0.5 of single-graph units, 0 of fixed-ensemble graph units, shrinks with the "
             "seeds; Holm step) to optimistic (SE/sqrt(S), level 0.05) with the configuration's S seeds (Note S24)."]
    if scenario_constant(ctx, "decoding_overhead_o_dec") is None:
        notes.append("Ensemble spend needs the decoding overhead o_dec of Proposition 10(ii), which is not registered.")
    out.table(["Dataset", "unit", "paired SE (nats)", "MDE (mnats): 0.05 / Holm step", "margin FB−E3 [range]",
               "power at the unit [range]", "power, Holm step", "power, seed-paired: cons.–opt.",
               "power, ensemble spend"], rows, notes=notes)
    out.data("power", ["dataset", "unit", "paired_se_mnats", "gain_mnats", "phi_low", "phi_central", "phi_high",
                       "margin_mnats", "margin_low", "margin_high", "mde_005", "mde_holm", "power_unit", "power_low",
                       "power_high", "power_holm", "seeds", "pairable_share", "se_conservative", "se_optimistic",
                       "power_conservative", "power_optimistic", "ensemble_spend_gain_mnats", "power_ensemble_spend"],
             source)
    for name in ("budget_matched", "ensemble_grid", "input_gate"):
        out.uses("records", name)


def ensemble_crossovers(ctx, dataset, dep, r_b, o_b):
    """Proposition 10(ii) crossovers over the registered D_inf band kappa l_plant (None without o_dec)."""
    o_dec = scenario_constant(ctx, "decoding_overhead_o_dec")
    if o_dec is None or dep["l_plant"] is None:
        return None, None, None
    o = overheads(ctx, dep)["o"]
    band = [k * dep["l_plant"] for k in ctx.constants[BUDGET]["ensemble_law_kappa_band"]]
    phi_fb, phi_b = [], []
    for d_inf in band:
        bounds = economics.decision_boundary(dep["gain"], r_b, o, o_dec, o_b, D_inf=d_inf, E=dep["E"])
        phi_fb.append(bounds["phi_FB_ensemble"])
        phi_b.append(bounds["phi_B_ensemble"])
    return sorted(phi_fb), sorted(phi_b), band


@display("supplementary:Table S47", stem="S47", title="Decision boundaries and power", scope="analytical",
         needs=ADMITTED_RUNS + (f"records:{GAP_SET}", "records:input_gate", "records:budget_matched"))
def table_s47(ctx, out):
    rows, source = [], []
    for dataset in ADMITTED:
        x = power_inputs(ctx, dataset)
        dep = x["dep"]
        r_b = gap_model(ctx, dataset)["r_B"]
        o_b = qb_overheads(ctx, dataset)["o_B"]
        contract = economics.decision_boundary(dep["gain"], r_b, 0.0, 0.0, 0.0)
        phi_fb, phi_b, band = ensemble_crossovers(ctx, dataset, dep, r_b, o_b)
        low, central, high = x["phi"]
        margin, (m_low, m_high) = x["margin"], x["margin_range"]
        se = x["se"]
        mde = inference.minimum_detectable_effect(se)
        mde_cons = inference.minimum_detectable_effect(x["se_cons"], family_size=ARM_FAMILY)
        mde_opt = inference.minimum_detectable_effect(x["se_opt"])
        unit = (power(margin, se), power(m_low, se), power(m_high, se))
        cons, opt = power(margin, x["se_cons"], ARM_FAMILY), power(margin, x["se_opt"])
        label = unit_label(ctx, dataset).replace(" (fixed ensemble)", "")
        rows.append([SHORT[dataset], f"{low:.2f}–{high:.2f} ({central:.2f})", integer(contract["phi_FB_contract"]),
                     num(contract["phi_B_contract"], 3),
                     None if phi_fb is None else f"{phi_fb[0]:.2f}–{phi_fb[1]:.2f}",
                     None if phi_b is None else f"{phi_b[0]:.3f}–{phi_b[1]:.3f}",
                     f"{margin:.1f} [{m_low:.1f}, {m_high:.1f}]", f"{mde:.1f} / {mde_cons:.1f}–{mde_opt:.1f}",
                     f"{unit[0]:.0f}% [{unit[1]:.0f}, {unit[2]:.0f}]; seed-paired {seed_range_text(cons, opt)} ({label})"])
        source.append([dataset, low, central, high, contract["phi_FB_contract"], contract["phi_B_contract"],
                       *(phi_fb or (None, None)), *(phi_b or (None, None)), *(band or (None, None)), o_b, margin, m_low,
                       m_high, mde, mde_cons, mde_opt, *unit, cons, opt])
        for field, computed in (("phi_grid", [low, high]), ("phi_central", central),
                                ("phi_FB_contract", contract["phi_FB_contract"]),
                                ("phi_B_contract", contract["phi_B_contract"]), ("phi_FB_ensemble", phi_fb),
                                ("phi_B_ensemble", phi_b), ("margin_FB_E3_mnats", margin),
                                ("margin_range", [m_low, m_high]), ("MDE_unit", mde),
                                ("MDE_seed_paired", [mde_cons, mde_opt]), ("power_unit_pct", unit[0]),
                                ("power_unit_range", [unit[1], unit[2]]), ("power_seed_paired_pct", [cons, opt])):
            out.compare(G_S47, (dataset, field), computed)
        out.uses("run", dataset, ["final/results_summary.json", "select/decision.json"])
    notes = ["Contract crossovers of Proposition 10(i): 1 against full feedback and r_B (Table S34) against q_B. "
             "Ensemble spend, Proposition 10(ii): phi*_FB = 1 - b(o - o_dec)/g and phi*_B = r_B - [b(o - o_dec) - "
             "b(o - o_B)]/g with b(o') = D_inf o'/[E(1 + o')], D_inf over the registered band kappa l_plant (l_plant: "
             "test loss of the nominal plant q_N), o the expected overhead and o_B the executed E6 overhead. Margin, "
             "MDE and power as in Table S42; seed-paired MDE: conservative at the Holm step, optimistic at 0.05."]
    if scenario_constant(ctx, "decoding_overhead_o_dec") is None:
        notes.append("The decoding overhead o_dec is not registered; the ensemble-spend crossovers stay empty.")
    out.table(["Dataset", "φ_x^gain grid (central)", "φ*_FB, contract", "φ*_B, contract", "φ*_FB, ensemble spend",
               "φ*_B, ensemble spend", "margin FB−E3 [range]", "MDE (mnats): unit / seed-paired cons.–opt.",
               "power at the unit [range]; seed-paired range (unit)"], rows, notes=notes)
    out.data("decision_boundaries", ["dataset", "phi_low", "phi_central", "phi_high", "phi_FB_contract",
                                     "phi_B_contract", "phi_FB_ensemble_low", "phi_FB_ensemble_high",
                                     "phi_B_ensemble_low", "phi_B_ensemble_high", "D_inf_band_low", "D_inf_band_high",
                                     "o_B", "margin_mnats", "margin_low", "margin_high", "mde_unit", "mde_conservative",
                                     "mde_optimistic", "power_unit", "power_low", "power_high", "power_conservative",
                                     "power_optimistic"], source)
    for name in (GAP_SET, "input_gate", "budget_matched"):
        out.uses("records", name)


# ---------------------------------------------------------------------------
# Supplementary Table S44: per-input tail-cost model
# ---------------------------------------------------------------------------

@display("supplementary:Table S44", stem="S44", title="Per-input tail-cost model", scope="analytical",
         needs=ADMITTED_RUNS)
def table_s44(ctx, out):
    order = registered_order(registry_actions(ctx)[0])
    rows, source = [], []
    for dataset in ADMITTED:
        dep = deployed(ctx, dataset)
        r = sparse_ratio(ctx, dataset)
        cap = budget.max_resolvent_fraction(dep["m"], r, SENSING_SHARE, order)
        f_bar = min(TAIL_REFERENCE_SHARE, cap)
        identity = cap < TAIL_REFERENCE_SHARE
        model = budget.tail_scenarios(dep["C_qC"], dep["m"] - 1.0, f_bar, r, order)
        rows.append([SHORT[dataset], num(r, 3), num(model["v"], 3), num(dep["m"] - 1.0, 3),
                     f"{f_bar:.3f} (identity)" if identity else num(f_bar, 2), num(model["o_fix"], 3),
                     joined(model["f99"], 2), joined(model["C_099"], 2),
                     joined(model["overhead_pct"], 0) + "%"])
        source.append([dataset, r, model["v"], dep["m"] - 1.0, cap, f_bar, model["o_fix"], *model["f99"],
                       *model["C_099"], *model["overhead_pct"], dep["C_qC"]])
        for field, computed in (("r", r), ("v", model["v"]), ("o_ctrl", dep["m"] - 1.0), ("f_bar", f_bar),
                                ("o_fix", model["o_fix"]), ("f99_low_central_high", model["f99"]),
                                ("C_099_low_central_high", model["C_099"]),
                                ("overhead_pct_low_central_high", model["overhead_pct"])):
            out.compare(G_S44, (dataset, field), computed)
        out.uses("run", dataset, ["select/candidates.json"])
    out.table(["Dataset", "r", "v", "o_ctrl", "f̄", "o_fix", "f99 (low / central / high)",
               "modelled C_0.99 (nominal executions)", "C_0.99/C_qC − 1"], rows,
              notes=[f"C_ctrl(x) = C_qC [1 + o_fix + f_res(x) v], v = r(T - 1)/(1 + r), T = {order}, o_ctrl = C_ctrl/C_qC - 1 "
                     f"(run cost ledger), o_fix = o_ctrl - f_bar v; f99 = f_bar + (0.10, 0.20, 0.35) "
                     "(cm_theory.budget.tail_scenarios). f_bar is the reference share "
                     f"{TAIL_REFERENCE_SHARE:g} unless the Note S19 identity at s = {SENSING_SHARE:g} caps the resolvent "
                     "fraction below it (identity)."])
    out.data("tail_model", ["dataset", "r", "v", "o_ctrl", "identity_cap", "f_bar", "o_fix", "f99_low", "f99_central",
                            "f99_high", "C099_low", "C099_central", "C099_high", "overhead_pct_low",
                            "overhead_pct_central", "overhead_pct_high", "C_qC"], source)


# ---------------------------------------------------------------------------
# Supplementary Figure S6: cost model and admissible truncation order
# ---------------------------------------------------------------------------

@display("supplementary:Figure S6", stem="FigS6", title="Action-cost and sensing-work model", scope="measurement",
         needs=ADMITTED_RUNS)
def figure_s6(ctx, out):
    order = registered_order(registry_actions(ctx)[0])
    plt = pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    grid = np.arange(1, TRUNCATION_AXIS + 1)
    data = {}
    curve, bars = [], []
    for dataset in ADMITTED:
        dep = deployed(ctx, dataset)
        r = sparse_ratio(ctx, dataset)
        data[dataset] = (r, dep["m"])
        values = [budget.per_member_multiplier(r, SENSING_SHARE, RESOLVENT_STEP_SHARES[-1], int(t)) for t in grid]
        axes[0].plot(grid, values, color=COLORS[dataset], lw=1.6, label=SHORT[dataset])
        axes[0].axhline(dep["m"], color=COLORS[dataset], ls=":", lw=1.0)
        curve += [[dataset, int(t), v, dep["m"]] for t, v in zip(grid, values)]
        for share in RESOLVENT_STEP_SHARES:
            t_max = budget.admissible_truncation_order(dep["m"], r, SENSING_SHARE, share)
            label = f">{TRUNCATION_AXIS}" if t_max > TRUNCATION_AXIS else f"{t_max:.0f}"
            bars.append([dataset, share, t_max, min(t_max, TRUNCATION_AXIS), label])
        out.compare(G_S6C, (dataset, "r"), r)
        out.compare(G_S6C, (dataset, "measured_multiplier_m"), dep["m"])
        out.uses("run", dataset, ["select/candidates.json"])
    ax = axes[0]
    ax.set_yscale("log")
    ax.set_xlim(0, TRUNCATION_AXIS)
    ax.set_xlabel("truncation order T")
    ax.set_ylabel("per-member cost multiplier")
    ax.set_title("a  Multiplier vs truncation order")
    ax.legend(ncol=2, frameon=False, loc="upper left")
    ax.text(0.97, 0.45, f"{100 * RESOLVENT_STEP_SHARES[-1]:.0f}% resolvent steps;\ndotted: measured multipliers (Table S23)",
            transform=ax.transAxes, ha="right", fontsize=7, color="#555555")
    ax = axes[1]
    width = 0.38
    for j, share in enumerate(RESOLVENT_STEP_SHARES):
        selected = [row for row in bars if row[1] == share]
        xs = np.arange(len(selected)) + (j - 0.5) * width
        ax.bar(xs, [row[3] for row in selected], width, color=("#d8d6d0", "#2f7bd9")[j],
               label=f"{100 * share:.0f}% resolvent steps" if j == 0 else f"{100 * share:.0f}%")
        for x, row in zip(xs, selected):
            ax.text(x, row[3] + 1.5, row[4], ha="center", va="bottom", rotation=90, fontsize=7, color="#444444")
    ax.axhline(order, color="#888888", ls="--", lw=1)
    ax.text(-0.4, order + 1.5, f"registered T = {order}", fontsize=7, color="#555555")
    ax.set_xticks(range(len(ADMITTED)), [SHORT[d] for d in ADMITTED], rotation=30)
    ax.set_ylim(0, 1.4 * TRUNCATION_AXIS)
    ax.set_ylabel("admissible T")
    ax.set_title("b  Admissible T by resolvent share")
    ax.legend(frameon=False, loc="upper left")
    ax = axes[2]
    r, m = data["ogbn-proteins"]
    ts = np.arange(2, 2 * order + 1)
    fractions = [budget.max_resolvent_fraction(m, r, SENSING_SHARE, int(t)) for t in ts]
    bound = budget.max_resolvent_fraction(m, r, SENSING_SHARE, order)
    ax.plot(ts, fractions, "o-", color=COLORS["ogbn-proteins"], ms=4)
    ax.axvline(order, color="#888888", ls="--", lw=1)
    ax.text(order + 0.3, max(fractions) * 0.55, f"registered T = {order}:\nf_res ≤ {bound:.3f} (model bound)", fontsize=7)
    ax.text(0.97, 0.97, f"sensing share s = {SENSING_SHARE:g};\nT sparse products per\nresolvent step",
            transform=ax.transAxes, ha="right", va="top", fontsize=7, color="#555555")
    ax.set_xlabel("truncation order T (on-line powers)")
    ax.set_ylabel("executed resolvent fraction f_res")
    ax.set_title(f"c  proteins: f_res consistent with m = {m:.2f}")
    for a in axes:
        a.grid(alpha=0.25)
    fig.tight_layout()
    out.figure(fig)
    out.data("panel_a_multiplier", ["dataset", "T", "multiplier_at_half_resolvent_steps", "measured_multiplier"], curve)
    out.data("panel_b_admissible_order", ["dataset", "resolvent_step_share", "admissible_T", "plotted_T", "label"], bars)
    out.data("panel_c_proteins_resolvent_fraction", ["T", "f_res", "r", "m", "s"],
             [[int(t), f, r, m, SENSING_SHARE] for t, f in zip(ts, fractions)])


# ---------------------------------------------------------------------------
# Supplementary Figure S7: acquisition gate
# ---------------------------------------------------------------------------

@display("supplementary:Figure S7", stem="FigS7", title="Acquisition gate", scope="measurement",
         needs=(f"records:{GAP_SET}",))
def figure_s7(ctx, out):
    plt = pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
    ax = axes[0]
    radius = np.array([0.0, 1.0])
    boundary = 2.0 * radius              # Proposition 9: retain when the gap exceeds 2 L_J r
    top = 2.2
    ax.fill_between(radius, boundary, top, color="#cfe0f6")
    ax.fill_between(radius, 0, boundary, color="#f8dccd")
    ax.plot(radius, boundary, color="#2f7bd9", lw=2)
    ax.text(0.1, 1.75, "retain the action\n(support, mode and search\nprocedure invariant on the region)", fontsize=7)
    ax.text(0.5, 0.25, "acquire a sensor profile\nand replan, or reserve q_C", fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, top)
    ax.set_xlabel("predicted-state region radius r_l (units of 1/L_J)")
    ax.set_ylabel("action-score gap m_l/L_J")
    ax.set_title("a  Conditional action-gap certificate (Prop. 9)")
    ax = axes[1]
    gaps = np.geomspace(0.05, 50.0, 400)          # mnats
    curves = []
    tau = certificate_floor()
    for dataset in ADMITTED:
        gap = gap_model(ctx, dataset)
        density = [math.exp(-0.5 * ((math.log(x / 1e3) - gap["mu"]) / gap["sigma"]) ** 2)
                   / (gap["sigma"] * math.sqrt(2 * math.pi)) for x in gaps]
        ax.plot(gaps, density, color=COLORS[dataset], lw=1.6,
                label=f"{SHORT[dataset]}: median {1e3 * gap['median']:.1f}, p5 {1e3 * gap['p5']:.1f}")
        curves += [[dataset, x, y, gap["mu"], gap["sigma"]] for x, y in zip(gaps, density)]
        out.compare(G_S21, (dataset, "median_gap"), gap["median"])
        out.compare(G_S21, (dataset, "p5_gap"), gap["p5"])
    ax.axvline(1e3 * tau, color="#444444", ls="--", lw=1)
    ax.text(1e3 * tau * 1.08, ax.get_ylim()[1] * 0.92, "τ", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("action-score gap (mnats)")
    ax.set_ylabel("density per log-gap")
    ax.set_title("b  Log-normal fit of the action-gap quantiles (Table S21)")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    for a in axes:
        a.grid(alpha=0.25)
    fig.tight_layout()
    out.figure(fig)
    out.data("panel_a_certificate", ["region_radius_over_LJ", "gap_over_LJ_boundary"], zip(radius, boundary))
    out.data("panel_b_gap_density", ["dataset", "gap_mnats", "density_per_log_gap", "mu_log_nats", "sigma_log"], curves)
    out.uses("records", GAP_SET)


# ---------------------------------------------------------------------------
# Supplementary Figure S8: clipping moments and interval units
# ---------------------------------------------------------------------------

@display("supplementary:Figure S8", stem="FigS8", title="Clipping moments and uncertainty units", scope="measurement",
         needs=GRAPH_RUNS)
def figure_s8(ctx, out):
    rows = []
    for dataset in GRAPH_LEVEL:
        s = graph_variance_share(ctx, dataset)
        f_max = admission.clipped_fraction_bound(s["D_bar"], s["S2"], s["J"], s["B"])
        rows.append([dataset, s["J"], s["B"], s["D_bar"], s["S2"], f_max, 1e3 * s["hw_graph"], 1e3 * s["hw_test"], s["ratio"]])
        for field, computed in (("J", s["J"]), ("B", s["B"]), ("D_bar", s["D_bar"]), ("S2", s["S2"])):
            out.compare(G_S13, (dataset, field), computed)
        out.compare(G_T1, (dataset, "loss_effect_interval_nats"), [s["dep"]["lower"], s["dep"]["upper"]])
        out.compare(G_FS8, ("panel_a", dataset, "clipped_fraction_percent"), 100 * f_max)
        for field, computed in (("graph_unit_half_width_mnats", 1e3 * s["hw_graph"]),
                                ("test_stage_half_width_mnats", 1e3 * s["hw_test"]),
                                ("test_to_graph_ratio", s["ratio"])):
            out.compare(G_FS8, ("panel_b", dataset, field), computed,
                        note="Computed from the run's admission moments and test interval; rounded ratio labels "
                             "are comparison targets only, not inputs used to recover hidden precision.")
        out.uses("run", dataset, ["admit/decision.json", "final/results_summary.json"])
    plt = pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
    xs = np.arange(len(rows))
    ax = axes[0]
    ax.bar(xs, [100 * row[5] for row in rows], 0.55, color="#2f7bd9")
    for x, row in zip(xs, rows):
        ax.text(x, 100 * row[5] + 0.05, f"{100 * row[5]:.2f}%", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xs, [SHORT[row[0]] for row in rows])
    ax.set_ylabel("f_max = [(J − 1)S²/J + D̄²]/B² (%)")
    ax.set_title("a  Maximum clipped fraction compatible with Table S13")
    ax.text(0.98, 0.97, "a per-input summary that puts more\ninputs at ±B contradicts Table S13",
            transform=ax.transAxes, ha="right", va="top", fontsize=7, color="#555555")
    ax = axes[1]
    ax.bar(xs - 0.2, [row[6] for row in rows], 0.38, color="#2f7bd9",
           label="graph unit, fixed ensemble: 1.96 S/√J (Table S13)")
    ax.bar(xs + 0.2, [row[7] for row in rows], 0.38, color="#e8673a",
           label="test-stage interval (seed-and-graph resampling)")
    for x, row in zip(xs, rows):
        ax.text(x + 0.2, row[7] + 0.05, f"×{row[8]:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xs, [SHORT[row[0]] for row in rows])
    ax.set_ylabel("95% half-width (mnats)")
    ax.set_title("b  Interval half-widths by statistical unit")
    ax.legend(frameon=False, fontsize=7)
    for a in axes:
        a.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out.figure(fig)
    out.data("panel_a_clipped_fraction", ["dataset", "J", "B", "D_bar", "S2", "f_max"], [row[:6] for row in rows])
    out.data("panel_b_half_widths", ["dataset", "hw_graph_mnats", "hw_test_mnats", "ratio"],
             [[row[0], row[6], row[7], row[8]] for row in rows])
    out.note("The fixed-ensemble graph half-width comes from the recorded admission variance and graph count; "
             "the seed-and-graph half-width comes from the run's test interval. Their ratio is computed directly. "
             "A rounded ratio in the figure is not an exact variance component or an inferred test interval.")


# ---------------------------------------------------------------------------
# Supplementary Figure S10: sensing-cost identity
# ---------------------------------------------------------------------------

@display("supplementary:Figure S10", stem="FigS10", title="Sensing-cost identities", scope="measurement",
         needs=ADMITTED_RUNS)
def figure_s10(ctx, out):
    rows = []
    for dataset in ADMITTED:
        dep = deployed(ctx, dataset)
        r = sparse_ratio(ctx, dataset)
        share = budget.sensing_share(r)
        rows.append([dataset, r, dep["m"], dep["m"] - 1.0, share, dep["m"] - 1.0 - share, dep["C_ctrl"], dep["C_qC"]])
        for field, computed in (("r", r), ("measured_multiplier_m", dep["m"]),
                                ("full_state_sensing_share_8r_over_1_plus_r", share),
                                ("residual", dep["m"] - 1.0 - share)):
            out.compare(G_S6C, (dataset, field), computed)
        out.compare(G_S23, (dataset, "controller_cost"), dep["C_ctrl"])
        out.compare(G_S23, (dataset, "comparator_cost"), dep["C_qC"])
        out.compare(G_FS10, ("panel_a", dataset, "controller_overhead_percent"), 100 * (dep["m"] - 1))
        out.compare(G_FS10, ("panel_a", dataset, "full_state_sensing_share_percent"), 100 * share)
        residual_percent = 100 * (dep["m"] - 1 - share)
        out.compare(G_FS10, ("panel_b", "datasets", dataset, "residual_share_percent"), residual_percent)
        out.compare(G_FS10, ("panel_b", "datasets", dataset, "printed_residual_percent"),
                    float(f"{residual_percent:.0f}"))
        out.uses("run", dataset, ["select/candidates.json"])
    proteins = next(row for row in rows if row[0] == "ogbn-proteins")
    width = ctx.constants["backbones_table_S5"]["ogbn-proteins"]["hidden"]
    sketches = [2 ** j for j in range(2, int(math.log2(width)) + 1)]
    sketch_rows = [[k, 100 * budget.sketch_relative_sd_bound(k), 100 * budget.sensing_share(proteins[1], k / width, 1)]
                   for k in sketches]
    for k, precision, sensing in sketch_rows:
        for field, computed in (("sketch_width", k), ("relative_sd_bound_percent", precision),
                                ("proteins_sensing_share_percent", sensing)):
            out.compare(G_FS10, ("panel_c", "points", str(k), field), computed,
                        note="Analytical coordinate at the source-declared sketch width and period one; "
                             "not an observation of executed sensing work.")
    out.compare(G_FS10, ("panel_c", "sketch_width_domain"), [sketches[0], sketches[-1]])
    out.compare(G_FS10, ("panel_c", "proteins_overhead_ceiling_percent"), 100 * proteins[3])
    out.compare(G_FS10, ("panel_b", "zero_boundary_percent"), 0.0)
    plt = pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    xs = np.arange(len(rows))
    ax = axes[0]
    ax.bar(xs - 0.2, [100 * row[3] for row in rows], 0.38, color=[COLORS[row[0]] for row in rows],
           label="measured controller overhead C_ctrl/C_qC − 1")
    ax.bar(xs + 0.2, [100 * row[4] for row in rows], 0.38, color="white", edgecolor="black", hatch="///",
           label="full-state degree-8 sensing at every site: 8r/(1 + r)")
    ax.set_yscale("log")
    ax.set_ylabel("share of a nominal execution (%)")
    ax.set_title("a  Sensing cost against the measured overhead")
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax = axes[1]
    residual = [100 * row[5] for row in rows]
    ax.bar(xs, residual, 0.55, color=[COLORS[row[0]] for row in rows])
    for x, v in zip(xs, residual):
        ax.text(x, v + (3 if v >= 0 else -3), f"{v:.0f}%", ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("residual share (%)")
    ax.set_title("b  Residual for observer, search and resolvent work\nunder full-state sensing at every site")
    ax = axes[2]
    ax.plot([row[0] for row in sketch_rows], [row[1] for row in sketch_rows], "o-", color="#2f7bd9",
            label="relative SD bound √(2/k) of a sketched band power")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("sketch width k (columns of H_l Ω)")
    ax.set_ylabel("relative SD (%)", color="#2f7bd9")
    twin = ax.twinx()
    twin.plot([row[0] for row in sketch_rows], [row[2] for row in sketch_rows], "s-", color="#e8673a",
              label="proteins sensing share at width k (period 1)")
    twin.axhline(100 * proteins[3], color="#e8673a", ls=":", lw=1)
    twin.text(sketches[0], 100 * proteins[3] * 1.08, f"proteins measured overhead {100 * proteins[3]:.0f}%",
              fontsize=7, color="#e8673a")
    twin.set_yscale("log")
    twin.set_ylabel("sensing share (%)", color="#e8673a")
    ax.set_title("c  Reduced sensor: precision against cost")
    handles = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    ax.legend(handles, [h.get_label() for h in handles], frameon=False, fontsize=6.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.22))
    for a in axes[:2]:
        a.set_xticks(xs, [SHORT[row[0]] for row in rows], rotation=30)
    for a in axes:
        a.grid(alpha=0.25)
    fig.tight_layout()
    out.figure(fig)
    out.data("panel_a_sensing_share", ["dataset", "r", "multiplier", "controller_overhead", "full_state_sensing_share",
                                       "C_ctrl", "C_qC"], [[row[0], row[1], row[2], row[3], row[4], row[6], row[7]]
                                                          for row in rows])
    out.data("panel_b_residual", ["dataset", "residual_share"], [[row[0], row[5]] for row in rows])
    out.data("panel_c_reduced_sensor", ["sketch_width", "relative_sd_bound_pct", "proteins_sensing_share_pct",
                                        "proteins_overhead_pct"], [row + [100 * proteins[3]] for row in sketch_rows])
    out.data("panel_c_boundary", ["minimum_sketch_width", "maximum_sketch_width", "proteins_overhead_pct"],
             [[sketches[0], sketches[-1], 100 * proteins[3]]])
    out.uses("repository", "paper/registered_constants.json",
             ["dataset_statistics_table_S6c_S23b", "backbones_table_S5"])
    out.note("Controller overhead is computed from observed run costs. Full-state and reduced-sensor shares "
             "are analytical identities using the declared benchmark dimensions and hidden widths, with each "
             "provider's reported edge convention. They do not substitute for measured operation counts or "
             "runtime adjacency statistics. The rounded r of Table S6c is a comparison, not an exact input.")


# ---------------------------------------------------------------------------
# Supplementary Figure S11: terminal-site work ledger of Algorithm S3
# ---------------------------------------------------------------------------

def released_reserve_ledger(scenario) -> list[tuple[str, float]]:
    """A rule that releases the reserve to C_pred at the terminal site (the premise Lemma 3 forbids)."""
    W, W_C, C_pred, w = scenario["W"], scenario["W_C"], scenario["C_pred"], scenario["w"]
    steps = [("enter\nterminal site", w)]
    for name, cost in (("sensing", scenario["sensing"]), ("action", scenario["action"])):
        if w + cost + C_pred > W:
            steps.append((f"{name}\nunaffordable:\nabort", w))
            w += W_C
            steps.append(("comparator\nrestart", w))
            return steps
        w += cost
        steps.append((f"pay {name}\n(reserve C_pred)", w))
    return steps + [("final\nprediction", w + C_pred)]


def held_reserve_ledger(scenario) -> list[tuple[str, float]]:
    """Algorithm S3 as executed: ``cm_experiments.budget.WorkLedger`` holds W_C before every paid operation."""
    from cm_experiments.budget import WorkLedger
    W, W_C, w = scenario["W"], scenario["W_C"], scenario["w"]
    ledger = WorkLedger(W, W_C, W_C)
    ledger.pay("work before the terminal site", w, lambda: (None, w))
    steps = [("enter\nterminal site", ledger.spent)]
    for name in ("sensing", "action"):
        cost = scenario[name]
        if not ledger.affordable(cost):
            steps.append((f"{name} check\n{ledger.spent:g} + {cost:g} + {W_C:g} > {W:g}:\nabort before paying", ledger.spent))
            ledger.restart(lambda: (None, W_C))
            steps.append(("comparator restart\nwithin W_C", ledger.spent))
            return steps
        ledger.pay(name, cost, lambda: (None, cost))
        steps.append((f"pay {name}", ledger.spent))
    ledger.pay("prediction", scenario["C_pred"], lambda: (None, scenario["C_pred"]), terminal=True, outcome="completed")
    return steps + [("final\nprediction", ledger.spent)]


@display("supplementary:Figure S11", stem="FigS11", title="Terminal-site budget ledger", scope="analytical")
def figure_s11(ctx, out):
    scenario = TERMINAL_SITE
    for key in ("W", "W_C", "C_pred", "w", "sensing", "action"):
        out.compare("supplementary_figure_S11_terminal_budget", ("scenario", key), scenario[key])
    panels = (("a", "Reserve released to C_pred at the terminal site", released_reserve_ledger(scenario)),
              ("b", "Reserve held: w + W_C ≤ W until the final prediction", held_reserve_ledger(scenario)))
    plt = pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.9), sharey=True)
    for ax, (letter, title, steps) in zip(axes, panels):
        xs = np.arange(len(steps))
        work = [w for _, w in steps]
        out.compare("supplementary_figure_S11_terminal_budget", ("panels", letter, "accumulated_work"), work)
        ax.step(xs, work, where="post", color="#2f7bd9", lw=2)
        ax.plot(xs, work, "o", color="#2f7bd9")
        for x, w in zip(xs, work):
            ax.text(x + 0.05, w + 0.3, f"w = {w:g}", fontsize=7)
        ax.axhline(scenario["W"], color="#d62728", lw=1.5)
        ax.text(0.02, scenario["W"] + 0.25, f"budget W = {scenario['W']:g}", color="#d62728", fontsize=7)
        ax.axhline(scenario["W"] - scenario["W_C"], color="#1a8f99", ls="--", lw=1)
        ax.text(xs[-1], scenario["W"] - scenario["W_C"] - 0.7,
                f"W − W_C = {scenario['W'] - scenario['W_C']:g}: restart still fits", ha="right", color="#1a8f99",
                fontsize=7)
        final = work[-1]
        exceeded = final > scenario["W"]
        verdict = f"{final:g} > W: the budget is exceeded" if exceeded else f"total work {final:g} ≤ W = {scenario['W']:g}"
        ax.text(np.mean(xs), 12.2, verdict, ha="center", color="#d62728" if exceeded else "#1a8f99", fontsize=8)
        ax.set_xticks(xs, [label for label, _ in steps], fontsize=7)
        ax.set_ylim(0, 13)
        ax.set_title(f"{letter}  {title}")
        ax.grid(alpha=0.25)
        out.data(f"panel_{letter}_ledger", ["step", "operation", "accumulated_work", "budget", "restart_reserve",
                                            "prediction_cost"],
                 [[i, label.replace("\n", " "), w, scenario["W"], scenario["W_C"], scenario["C_pred"]]
                  for i, (label, w) in enumerate(steps)])
    axes[0].set_ylabel("accumulated work w")
    fig.tight_layout()
    out.figure(fig)
    out.note(f"Scenario of the caption: W = {scenario['W']:g}, W_C = {scenario['W_C']:g}, C_pred = {scenario['C_pred']:g}, "
             f"w = {scenario['w']:g} at terminal-site entry; sensing and the chosen action cost {scenario['sensing']:g} each. "
             "Panel b runs the executed ledger (cm_experiments.budget.WorkLedger).")
    out.uses("repository", "cm_experiments/budget.py", ["WorkLedger"])


# ---------------------------------------------------------------------------
# Supplementary Figure S12: power of the budget-matched test
# ---------------------------------------------------------------------------

G_FS12 = "supplementary_figure_S12_budget_matched_test_power"


@display("supplementary:Figure S12", stem="FigS12", title="Comparison-test power", scope="analytical",
         needs=ADMITTED_RUNS + ("records:budget_matched",))
def figure_s12(ctx, out):
    margins = np.linspace(0.0, 8.0, 321)
    reference = 80.0                      # dashed reference power of both panels (%)
    anchor_note = ("Functional value of the plotted power curve at a named anchor (domain endpoint); the "
                   "reference evaluates the stated power formula with printed inputs, not a measurement.")
    crossing_note = ("True margin at which the plotted curve reaches the 80% line, found on the plotted function; "
                     "printed as the minimum detectable effect (Tables S42 and S47).")

    def reference_crossing(se, family=1):
        """True margin at which the plotted power curve reaches the reference power (bisection)."""
        low, high = 0.0, 20.0 * se
        for _ in range(100):
            middle = 0.5 * (low + high)
            if power(middle, se, family) < reference:
                low = middle
            else:
                high = middle
        return 0.5 * (low + high)

    plt = pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    curves, points, seed_counts = [], [], set()
    for dataset in ADMITTED:
        x = power_inputs(ctx, dataset)
        color = COLORS[dataset]
        unit = [power(m, x["se"]) for m in margins]
        cons = [power(m, x["se_cons"], ARM_FAMILY) for m in margins]
        opt = [power(m, x["se_opt"]) for m in margins]
        label = f"{SHORT[dataset]} ({unit_label(ctx, dataset).replace(' (fixed ensemble)', ', fixed ensemble')})"
        axes[0].plot(margins, unit, color=color, lw=1.6, label=label)
        central = power(x["margin"], x["se"])
        axes[0].plot([x["margin_range"][0], x["margin_range"][1]], [central, central], color=color, lw=6, alpha=0.3,
                     solid_capstyle="butt")
        axes[0].plot(x["margin"], central, "o", color=color, ms=6)
        axes[1].plot(margins, cons, color=color, lw=1.6)
        axes[1].plot(margins, opt, color=color, lw=1.2, ls=":")
        p_cons, p_opt = power(x["margin"], x["se_cons"], ARM_FAMILY), power(x["margin"], x["se_opt"])
        axes[1].plot(x["margin"], p_cons, "o", color=color, ms=6)
        axes[1].plot(x["margin"], p_opt, "o", mfc="white", color=color, ms=6)
        curves += [[dataset, m, u, c, o] for m, u, c, o in zip(margins, unit, cons, opt)]
        points.append([dataset, x["se"], x["margin"], *x["margin_range"], central, p_cons, p_opt, x["seeds"], x["share"]])
        seed_counts.add(x["seeds"])
        out.compare(G_S42, (dataset, "paired_SE_nats"), x["se"] / 1e3)
        out.compare(G_S42, (dataset, "reported_margin_mnats"), x["margin"])
        out.compare(G_S42, (dataset, "margin_range"), list(x["margin_range"]))
        out.compare(G_S42, (dataset, "power_at_unit_pct"), central)
        out.compare(G_S42, (dataset, "power_seed_paired_pct"), [p_cons, p_opt])
        low, high = x["margin_range"]
        for field, computed, note in (
                ("legend_label", label, ""), ("central_margin_mnats", x["margin"], ""),
                ("power_at_central_margin_percent", central, ""), ("margin_range_mnats", [low, high], ""),
                ("power_at_range_endpoints_percent", [power(low, x["se"]), power(high, x["se"])], ""),
                ("power_at_zero_margin_percent", unit[0], anchor_note),
                ("power_at_domain_end_percent", unit[-1], anchor_note),
                ("margin_at_reference_power_mnats", reference_crossing(x["se"]), crossing_note)):
            out.compare(G_FS12, ("panel_a", "datasets", dataset, field), computed, note=note)
        for field, computed, note in (
                ("pairable_share", x["share"], ""), ("marker_margin_mnats", x["margin"], ""),
                ("conservative_power_at_central_margin_percent", p_cons, ""),
                ("optimistic_power_at_central_margin_percent", p_opt, ""),
                ("conservative_power_at_zero_margin_percent", cons[0], anchor_note),
                ("optimistic_power_at_zero_margin_percent", opt[0], anchor_note),
                ("conservative_power_at_domain_end_percent", cons[-1], anchor_note),
                ("optimistic_power_at_domain_end_percent", opt[-1], anchor_note),
                ("conservative_margin_at_reference_power_mnats", reference_crossing(x["se_cons"], ARM_FAMILY),
                 crossing_note),
                ("optimistic_margin_at_reference_power_mnats", reference_crossing(x["se_opt"]), crossing_note)):
            out.compare(G_FS12, ("panel_b", "datasets", dataset, field), computed, note=note)
        out.uses("run", dataset, ["final/results_summary.json", "select/decision.json"])
    seeds = next(iter(seed_counts)) if len(seed_counts) == 1 else sorted(seed_counts)
    for panel in ("panel_a", "panel_b"):
        out.compare(G_FS12, (panel, "margin_domain_mnats"), [float(margins[0]), float(margins[-1])])
        out.compare(G_FS12, (panel, "reference_power_percent"), reference)
    out.compare(G_FS12, ("panel_b", "paired_seeds"), seeds)
    out.compare(G_FS12, ("panel_b", "holm_family_size"), ARM_FAMILY)
    for ax in axes:
        ax.axhline(reference, color="#444444", ls="--", lw=0.8)
        ax.set_ylim(0, 103)
        ax.grid(alpha=0.25)
    axes[0].text(7.9, reference + 1.5, f"{reference:.0f}%", ha="right", fontsize=7)
    axes[0].set_xlabel("true margin of the feedback rule over the E3 predictor (mnats)")
    axes[0].set_ylabel("power of the two-sided paired test (%)")
    axes[0].set_title("a  Power at the registered unit; dots: central margin of the E2 grid, bars: its range")
    axes[0].legend(frameon=False, fontsize=6.5, loc="lower right")
    axes[1].set_xlabel("true margin (mnats)")
    axes[1].set_ylabel("power (%)")
    axes[1].set_title("b  Seed-paired design: conservative (solid, Holm step) to optimistic (dotted)")
    axes[1].plot([], [], color="#444444", label="conservative: pairable share 0.5 (single-graph units) or 0 "
                                                f"(fixed-ensemble graph units), level α/{ARM_FAMILY}")
    axes[1].plot([], [], color="#444444", ls=":", label=f"optimistic: σ/√{seeds}, level 0.05")
    axes[1].legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.tight_layout()
    out.figure(fig)
    out.data("power_curves", ["dataset", "margin_mnats", "power_unit", "power_conservative", "power_optimistic"], curves)
    out.data("central_margins", ["dataset", "paired_se_mnats", "margin_mnats", "margin_low", "margin_high",
                                 "power_unit", "power_conservative", "power_optimistic", "seeds", "pairable_share"], points)
    out.uses("records", "budget_matched")


# ---------------------------------------------------------------------------
# Supplementary Figure S15: spectral fingerprint, executed range and headroom
# ---------------------------------------------------------------------------

@display("supplementary:Figure S15", stem="FigS15", title="Spectral fingerprint, action range and headroom",
         scope="measurement", needs=("runs",))
def figure_s15(ctx, out):
    actions, _ = registry_actions(ctx)
    transport = [a for a in actions if a.is_transport]
    nominal = transport[0]
    order = registered_order(actions)
    lam = np.linspace(0.0, 2.0, 401)
    responses = {a.id: spectral.action_response(a, lam) for a in transport}
    edges = spectral.band_edges(6)
    sensitivity = np.array([spectral.band_sensitivity(a, nominal) for a in transport[1:]])
    headroom = []
    for dataset in DATASETS:
        dep = deployment(ctx, dataset)
        upper = dep["headroom_upper"]
        if upper is None:
            raise DisplayInputError(f"{dataset}: the oracle headroom bound U(Delta_oracle) is not recorded")
        bound = 1e3 * abs(upper)
        headroom.append([dataset, upper, bound, dep["effect"], bound - 1e3 * abs(dep["effect"]),
                         " ".join(f"{m['run']}:{m['headroom_upper']!r}" for m in dep["members"])])
        out.compare(G_S4, (dataset, "U_Delta_oracle"), upper)
        out.compare(G_T1, (dataset, "loss_effect_nats"), dep["effect"])
        out.uses("run", dataset, ["select/entry_complete.json", "final/results_summary.json"])
    plt = pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4))
    ax = axes[0, 0]
    for a in transport:
        ax.plot(lam, responses[a.id], color=ACTION_COLORS[a.id], lw=2.4 if a.id == 0 else 1.4,
                label="A0 (nominal)" if a.id == 0 else f"A{a.id}")
    ax.axhline(0, color="#555555", lw=0.7)
    ax.set_xlim(0, 2)
    ax.set_xlabel("graph frequency λ of L_sym")
    ax.set_ylabel("response g_a(λ)")
    ax.set_title(f"a  Spectral response of the registered actions (T = {order})")
    ax.legend(ncol=4, frameon=False, fontsize=7, loc="lower left")
    ax = axes[0, 1]
    ax.imshow(sensitivity, cmap="Blues", vmin=0.0, vmax=max(0.65, float(sensitivity.max())), aspect="auto")
    for i in range(sensitivity.shape[0]):
        for j in range(sensitivity.shape[1]):
            ax.text(j, i, f"{sensitivity[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if sensitivity[i, j] > 0.4 else "black")
    ax.set_yticks(range(len(transport) - 1), [f"A{a.id}" for a in transport[1:]])
    ax.set_xticks(range(len(edges)), [f"[{lo:.2f},{hi:.2f}]" for lo, hi in edges], rotation=30, fontsize=7)
    ax.set_xlabel("band I_b (uniform partition of [0, 2])")
    ax.set_title("b  Band sensitivity s_ab = max_{λ∈I_b} |g_a(λ) − g_A0(λ)|")
    ax = axes[1, 0]
    ranges = [operators.executed_range(a) for a in transport]
    reach = [operators.maximum_reach(a) for a in transport]
    identity = [operators.identity_coefficient(a) for a in transport]
    xs = np.arange(len(transport))
    ax.bar(xs, ranges, 0.6, color=[ACTION_COLORS[a.id] for a in transport])
    for x, h, c in zip(xs, ranges, identity):
        ax.text(x, h + 0.02, f"{h:.2f}", ha="center", fontsize=7)
        ax.text(x, 0.2, f"id {c:.2f}", ha="center", rotation=90, fontsize=7, color="white")
    ax.axhline(ranges[0], color="#333333", ls=":", lw=1)
    ax.text(-0.4, 1.15, f"dotted: nominal level h̄_A0 = {ranges[0]:.2f}", fontsize=7, color="#555555")
    ax.set_ylim(0, 1.25)
    ax.set_xticks(xs, [f"A{a.id}" for a in transport])
    ax.set_ylabel("expected hops per executed step")
    ax.set_title("c  Executed range per action (exact)")
    twin = ax.twinx()
    twin.plot(xs, reach, "s", mfc="white", color="#222222")
    twin.set_ylim(0, 1.25 * max(reach) * 1.15)
    twin.set_ylabel("maximum reach T (squares)")
    ax = axes[1, 1]
    xs = np.arange(len(headroom))
    ax.bar(xs - 0.2, [row[2] for row in headroom], 0.38, color="white", edgecolor="black",
           label="q_C → oracle: headroom bound |U(Δ_oracle)|")
    ax.bar(xs + 0.2, [row[4] for row in headroom], 0.38, color="#2f7bd9",
           label="deployed → oracle: unrealized (lower bound)")
    ax.set_xticks(xs, [SHORT[row[0]] for row in headroom], rotation=30)
    ax.set_ylabel("mnats")
    ax.set_title("d  Committing to one schedule: bounds")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    for a in axes.flat:
        a.grid(alpha=0.2)
    fig.tight_layout()
    out.figure(fig)
    out.data("panel_a_response", ["lambda"] + [f"A{a.id}" for a in transport],
             [[float(x)] + [float(responses[a.id][i]) for a in transport] for i, x in enumerate(lam)])
    out.data("panel_b_band_sensitivity", ["action", "band", "lambda_low", "lambda_high", "sensitivity_upper_bound"],
             [[f"A{a.id}", b, edges[b][0], edges[b][1], float(sensitivity[i, b])]
              for i, a in enumerate(transport[1:]) for b in range(len(edges))])
    out.data("panel_c_executed_range", ["action", "executed_range", "max_reach", "executed_identity_coefficient"],
             [[f"A{a.id}", h, t, c] for a, h, t, c in zip(transport, ranges, reach, identity)])
    out.data("panel_d_headroom", ["dataset", "U_Delta_oracle_nats", "headroom_bound_mnats", "deployed_effect_nats",
                                  "unrealized_lower_bound_mnats", "per_run_U_Delta_oracle"], headroom)


# ---------------------------------------------------------------------------
# Supplementary Figure S20: break-even plane, reach, resampling sources, seed design
# ---------------------------------------------------------------------------

def reach_curves(actions, scales, grid=4001):
    """Band-limited bound of (6.1) and exact one-step displacement relative to A0 per unit state norm.

    The state's spectral energy density on [0, 2] is proportional to exp(-lambda/lambda_0); the
    bound weights the squared band sensitivities by the band energies.
    """
    transport = [a for a in actions if a.is_transport]
    nominal = transport[0]
    lam = np.linspace(0.0, 2.0, grid)
    edges = spectral.band_edges(6)
    members = spectral.band_membership(lam, 6)
    rows = []
    for a in transport[1:]:
        diff2 = (spectral.action_response(a, lam) - spectral.action_response(nominal, lam)) ** 2
        s2 = spectral.band_sensitivity(a, nominal) ** 2
        for scale in scales:
            density = np.exp(-lam / scale)
            density /= np.trapezoid(density, lam)
            exact = math.sqrt(float(np.trapezoid(diff2 * density, lam)))
            energy = np.array([np.trapezoid(np.where(members == b, density, 0.0), lam) for b in range(len(edges))])
            rows.append([f"A{a.id}", float(scale), math.sqrt(float(np.sum(s2 * energy))), exact])
    return rows


G_FS20 = "supplementary_figure_S20_analytical_panels"


@display("supplementary:Figure S20", stem="FigS20", title="Break-even, resampling and seed-design power",
         scope="analytical",
         needs=ADMITTED_RUNS + ("records:ensemble_grid", "records:input_gate", "records:budget_matched"))
def figure_s20(ctx, out):
    actions, _ = registry_actions(ctx)
    kappa = ctx.constants[BUDGET]["ensemble_law_kappa_band"]
    reference = 80.0                      # dotted reference power of panel d (%)
    analytical_note = ("Analytical coordinate of the plotted function at a named anchor; the reference evaluates "
                       "the stated formula with printed inputs and is not an observation.")
    plane, design, shares = [], [], {}
    for dataset in ADMITTED:
        dep = deployed(ctx, dataset)
        cost = overheads(ctx, dep)
        fit = ensemble_fit(ctx, dataset)
        if dep["l_plant"] is None:
            raise DisplayInputError(f"{dataset}: the nominal plant loss (q_N) is not recorded")
        band = [k * dep["l_plant"] for k in kappa]
        plane.append([dataset, dep["E"], cost["o"], cost["o_assure"], dep["gain"], fit["D_inf"], dep["gain"] / fit["D_inf"],
                      dep["gain"] / band[1], dep["gain"] / band[0], dep["l_plant"]])
        out.compare(G_S26B, (dataset, "D_inf_hat"), fit["D_inf"])
        out.compare(G_S26B, (dataset, "l_plant_nats"), dep["l_plant"])
        out.compare(G_S26B, (dataset, "D_inf_band"), band)
        out.compare(G_S23, (dataset, "expected_serial_cost"), cost["expected"])
        out.compare(G_S23, (dataset, "assurance_serial_cost"), cost["assurance"])
        x = power_inputs(ctx, dataset)
        shares[dataset] = x["share"]
        for count in range(1, x["seeds"] + 1):
            cons, opt = seed_paired_se(x["se"], x["share"], count)
            design.append([dataset, count, power(x["margin"], cons, ARM_FAMILY), power(x["margin"], opt)])
        first, last = design[-x["seeds"]], design[-1]
        out.compare(G_S42, (dataset, "power_holm_step_pct"), first[2])
        out.compare(G_S42, (dataset, "power_at_unit_pct"), first[3])
        out.compare(G_S42, (dataset, "power_seed_paired_pct"), [last[2], last[3]])
        out.uses("run", dataset, ["final/results_summary.json", "select/candidates.json"])
    units = []
    for dataset in GRAPH_LEVEL:
        s = graph_variance_share(ctx, dataset)
        units.append([dataset, 1e3 * s["hw_graph"], 1e3 * s["hw_test"], s["ratio"], 1.0 - s["ratio"] ** -2])
    sizes = sorted({e for e in ctx.config(ADMITTED[0], "full")["registry"]["ensemble_sizes"]
                    if 1 < e <= max(row[1] for row in plane)})
    # The plotted lambda_0 grid contains the E18 audit densities, so the anchors are points of the plotted curves.
    scales = np.unique(np.r_[np.geomspace(0.05, 2.0, 60), E18_DENSITIES])
    reach = reach_curves(actions, scales)
    plt = pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    o_grid = np.unique(np.r_[np.geomspace(0.08, 2.0, 200), 1.0])
    styles = (":", "-", "--", "-.")
    curves = []
    for size, style in zip(sizes, styles):
        threshold = [economics.break_even_effect(1.0, o, size) for o in o_grid]
        ax.plot(o_grid, threshold, color="#444444", ls=style, lw=1.2, label=f"break-even g/D∞ = o/[E(1+o)], E = {size}")
        curves += [[size, o, t] for o, t in zip(o_grid, threshold)]
        plotted = dict(zip(o_grid.tolist(), threshold))
        for field, o in (("at_domain_start", o_grid[0]), ("at_unit_overhead", 1.0), ("at_domain_end", o_grid[-1])):
            out.compare(G_FS20, ("panel_a", "break_even_curves", f"E{size}", field), plotted[float(o)],
                        note=analytical_note)
    out.compare(G_FS20, ("panel_a", "ensemble_law_kappa_band"), list(kappa))
    out.compare(G_FS20, ("panel_a", "break_even_ensemble_sizes"), list(sizes))
    out.compare(G_FS20, ("panel_a", "overhead_domain"), [float(o_grid[0]), float(o_grid[-1])])
    for dataset, size, o, o_assure, _, _, ratio, kappa_high, kappa_low, _ in plane:
        for field, computed, note in (
                ("ensemble_size", size, ""), ("expected_overhead", o, analytical_note),
                ("assurance_overhead", o_assure, analytical_note), ("gain_over_D_inf", ratio, analytical_note),
                ("whisker_gain_over_D_inf", [kappa_high, kappa_low], analytical_note),
                ("break_even_at_expected_overhead", economics.break_even_effect(1.0, o, size), analytical_note),
                ("break_even_at_assurance_overhead", economics.break_even_effect(1.0, o_assure, size),
                 analytical_note)):
            out.compare(G_FS20, ("panel_a", "datasets", dataset, field), computed, note=note)
    anchors = {float(scales[0]), *map(float, E18_DENSITIES), float(scales[-1])}
    ratios = {}
    for name, scale, bound, exact in reach:
        if scale in anchors:
            for field, computed in (("upper_bound", bound), ("exact_displacement", exact)):
                out.compare(G_FS20, ("panel_b", "actions", name, f"{scale:g}", field), computed, note=analytical_note)
            ratios.setdefault(scale, []).append(exact / bound)
    out.compare(G_FS20, ("panel_b", "printed_ratio_extremes", "minimum_at_lambda0_0.1"), min(ratios[0.1]),
                note="Ratio of the two plotted families of the declared exponential-density scenario (SI p. 20); "
                     "not the plant-specific E18 audit.")
    out.compare(G_FS20, ("panel_b", "printed_ratio_extremes", "maximum_at_lambda0_1"), max(ratios[1.0]),
                note="Ratio of the two plotted families of the declared exponential-density scenario (SI p. 20); "
                     "not the plant-specific E18 audit.")
    out.compare(G_FS20, ("panel_b", "registered_band_count"), len(spectral.band_edges(6)))
    out.compare(G_FS20, ("panel_b", "e18_audit_densities"), list(E18_DENSITIES))
    out.compare(G_FS20, ("panel_b", "spectral_scale_domain"), [float(scales[0]), float(scales[-1])])
    out.compare(G_FS20, ("panel_c", "graph_unit_z_multiplier"), Z975)
    for dataset, hw_graph, hw_test, ratio, seed_share in units:
        for field, computed in (("graph_unit_half_width_mnats", hw_graph), ("test_stage_half_width_mnats", hw_test),
                                ("ratio_label", ratio), ("seed_share_lower_bound_percent", 100.0 * seed_share)):
            out.compare(G_FS20, ("panel_c", "datasets", dataset, field), computed)
    for dataset in ADMITTED:
        out.compare(G_FS20, ("panel_d", "datasets", dataset, "pairable_share"), shares[dataset])
        for _, count, cons, opt in (row for row in design if row[0] == dataset):
            out.compare(G_FS20, ("panel_d", "datasets", dataset, "conservative_power_percent", str(count)), cons)
            out.compare(G_FS20, ("panel_d", "datasets", dataset, "optimistic_power_percent", str(count)), opt)
    counts = [row[1] for row in design]
    out.compare(G_FS20, ("panel_d", "seed_count_domain"), [min(counts), max(counts)])
    out.compare(G_FS20, ("panel_d", "reference_power_percent"), reference)
    out.compare(G_FS20, ("panel_d", "conservative_level"), power(0.0, 1.0, ARM_FAMILY) / 100.0,
                note="Level of the plotted conservative power function: its value at zero margin.")
    out.compare(G_FS20, ("panel_d", "optimistic_level"), power(0.0, 1.0) / 100.0,
                note="Level of the plotted optimistic power function: its value at zero margin.")
    for row in plane:
        color = COLORS[row[0]]
        ax.plot([row[2], row[3]], [row[6], row[6]], ":", color=color, lw=1)
        for o, face in ((row[2], color), (row[3], "white")):
            ax.plot([o, o], [row[7], row[8]], "-", color=color, lw=1.2)
            ax.plot([o, o], [row[7], row[8]], "_", color=color, ms=8)
            ax.plot(o, row[6], "o", color=color, mfc=face, ms=6)
        ax.text(row[2] * 0.9, row[6], SHORT[row[0]], color=color, ha="right", va="center", fontsize=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("serial overhead o over q_C (filled: expected; hollow: assurance)")
    ax.set_ylabel("deployed gain g in units of the ensemble law's D∞")
    ax.set_title("a  Equal-compute break-even plane (Prop. 5; ensemble law)")
    ax.text(0.02, 0.03, f"whiskers: κ ∈ [{kappa[0]:g}, {kappa[1]:g}] of l_plant;\nabove its E curve the rule beats q_C "
            "at equal compute", transform=ax.transAxes, fontsize=7, color="#555555")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax = axes[0, 1]
    for a in (x for x in actions if x.is_transport and x.id > 0):
        rows = [row for row in reach if row[0] == f"A{a.id}"]
        ax.plot([r[1] for r in rows], [r[2] for r in rows], color=ACTION_COLORS[a.id], lw=1.6, label=f"A{a.id}")
        ax.plot([r[1] for r in rows], [r[3] for r in rows], color=ACTION_COLORS[a.id], lw=1.2, ls="--")
    for scale in E18_DENSITIES:
        ax.axvline(scale, color="#bbbbbb", lw=3, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("spectral energy scale λ0 of the state (density ∝ e^(−λ/λ0) on [0, 2])")
    ax.set_ylabel("one-step displacement per unit state norm, relative to A0")
    ax.set_title("b  Band-limited reach: upper-bound score vs exact displacement (Prop. 6)")
    ax.plot([], [], color="#444444", label="upper bound of (6.1), six registered bands")
    ax.plot([], [], color="#444444", ls="--", label="exact displacement for the density")
    ax.legend(ncol=3, frameon=False, fontsize=6.5, loc="upper left")
    ax = axes[1, 0]
    xs = np.arange(len(units))
    for x, row in zip(xs, units):
        color = COLORS[row[0]]
        ax.bar(x - 0.2, row[1], 0.38, color=color, alpha=0.45)
        ax.bar(x + 0.2, row[2], 0.38, color=color)
        ax.text(x + 0.2, row[2] + 0.05, f"ϱ = {row[3]:.1f}", ha="center", va="bottom", fontsize=7)
        ax.text(x, -0.6, f"seed share ≥ {100 * row[4]:.1f}%", ha="center", fontsize=7)
    from matplotlib.patches import Patch
    ax.set_xticks(xs, [SHORT[row[0]] for row in units])
    ax.set_ylim(-1, 1.25 * max(row[2] for row in units) + 0.5)
    ax.set_ylabel("95% half-width of the paired effect (mnats)")
    ax.set_title("c  Two resampling sources in one interval (two-way model, lower bound)")
    ax.legend(handles=[Patch(color="#999999", alpha=0.45, label="graph-unit half-width, fixed deployed ensemble "
                             "(admission split, Table S13)"),
                       Patch(color="#999999", label="test-stage half-width, seeds and graphs resampled (test split, "
                             "Table 1)")], frameon=False, fontsize=6.5, loc="upper right")
    ax = axes[1, 1]
    for dataset in ADMITTED:
        rows = [row for row in design if row[0] == dataset]
        counts = [row[1] for row in rows]
        ax.plot(counts, [row[2] for row in rows], "o-", color=COLORS[dataset], ms=4, label=SHORT[dataset])
        ax.plot(counts, [row[3] for row in rows], ":", color=COLORS[dataset])
        ax.fill_between(counts, [row[2] for row in rows], [row[3] for row in rows], color=COLORS[dataset], alpha=0.08)
    ax.axhline(reference, color="#444444", ls=":", lw=0.8)
    ax.set_ylim(0, 103)
    ax.set_xlabel("optimization seeds paired as replicates of every evaluation unit, S")
    ax.set_ylabel("power of E3 at the central margin (%)")
    ax.set_title("d  Power against the seed design: conservative (solid) to optimistic (dotted)")
    ax.text(0.03, 0.04, "solid: only the pairable share of the variance shrinks with S, level α/4 (Holm first step);\n"
            "dotted: all variance pairs out, α = 0.05; single-graph pairable share 0.5", transform=ax.transAxes,
            fontsize=6.5, color="#555555")
    ax.legend(ncol=2, frameon=False, fontsize=7, loc="center right")
    for a in axes.flat:
        a.grid(alpha=0.2)
    fig.tight_layout()
    out.figure(fig)
    out.data("panel_a_break_even_points", ["dataset", "ensemble_E", "expected_overhead", "assurance_overhead", "gain_nats",
                                           "D_inf_fit", "gain_over_D_inf", "gain_over_D_inf_kappa_high",
                                           "gain_over_D_inf_kappa_low", "l_plant_nats"], plane)
    out.data("panel_a_break_even_curves", ["ensemble_E", "overhead", "gain_over_D_inf"], curves)
    out.data("panel_b_reach", ["action", "lambda_0", "upper_bound", "exact_displacement"], reach)
    out.data("panel_c_half_widths", ["dataset", "hw_graph_mnats", "hw_test_mnats", "ratio", "seed_share_lower_bound"],
             units)
    out.data("panel_d_seed_design", ["dataset", "seeds", "power_conservative", "power_optimistic"], design)
    for name in ("ensemble_grid", "input_gate", "budget_matched"):
        out.uses("records", name)
