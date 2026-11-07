"""Displays of the experiments outside the eight-stage selection ledger.

Supplementary Table S5c reads the E1 nominal-identity record that the train stage
of every sealed run writes (``train/entry_initial.json``). Tables S17, S18, S27,
S29 and Figure S5 read the sweep record sets of ``records/specs.json``
(``module_ablations``, ``sensitivity_grids``, ``buffer_sensitivity``,
``horizon_beam_sweep``) produced by ``cm_experiments/sweeps.py``. Table S43
assembles every experiment's outcome from the sealed runs and the measurement
record sets, with thresholds from the configurations and the registered constants
and the analytical relations of ``cm_theory``.
"""
from __future__ import annotations

import math

from .framework import DATASETS, GRAPH_LEVEL, NODE_LEVEL, SHORT, DisplayInputError, display, integer, num, pyplot

S5C_TITLE = "Nominal-identity test E1 and site accounting"
ADMITTED = tuple(d for d in DATASETS if d != "ogbn-products")
BACKBONE_LABELS = {
    "shared_transport": "MLP{encoder_layers}-{hidden_dim} transport plant",
    "edge_weighted_transport": "MLP{encoder_layers}-{hidden_dim}, edge-weighted",
    "heterophily": "heterophily encoder{encoder_layers}-{hidden_dim}",
    "gps": "GraphGPS-style local branch",
    "linear_gine": "GINE-style, linear aggregation",
}
S18_ORDER = ("p_min", "n_min", "max_normalized_weight", "epsilon_r", "epsilon_eq", "clip")
DASH = "–"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def sci(value: float | None, digits: int = 1) -> str | None:
    """``m × 10^e`` at ``digits`` significant digits (``10^e`` for a unit mantissa)."""
    if value is None:
        return None
    value = float(value)
    if value == 0:
        return "0"
    exponent = math.floor(math.log10(abs(value)))
    scale = 10 ** (digits - 1)
    mantissa = math.copysign(math.floor(abs(value) / 10 ** exponent * scale + 0.5) / scale, value)  # half up
    if abs(mantissa) >= 10:
        mantissa, exponent = mantissa / 10, exponent + 1
    text = f"{mantissa:.{digits - 1}f}" if digits > 1 else f"{int(mantissa)}"
    return f"10^{exponent}" if text == "1" else f"{text} × 10^{exponent}"


def grid_text(values) -> str:
    """A registered grid as printed: two decimals when exact, otherwise the shortest form."""
    out = []
    for v in values:
        v = float(v)
        if v == int(v) and abs(v) >= 1:
            out.append(str(int(v)))
        elif round(v, 2) == v:
            out.append(f"{v:.2f}")
        else:
            out.append(f"{v:g}")
    return ", ".join(out)


def value_of(rec, quantity: str, **keys) -> float | None:
    value = rec.value(quantity, **keys)
    return None if value is None else float(value)


def require_value(rec, quantity: str, **keys) -> float:
    value = value_of(rec, quantity, **keys)
    if value is None:
        raise DisplayInputError(f"{rec.name}: {quantity} {keys} is not in the record set")
    return value


def constant(ctx, pointer: str):
    node = ctx.constants
    for token in pointer.split("/"):
        node = node[token]
    return node


def source(rec, quantity: str, **keys) -> tuple[str, str]:
    rows = rec.select(quantity, **keys)
    return (rows[0].get("source_file", ""), rows[0].get("locator", "")) if rows else ("", "")


# ---------------------------------------------------------------------------
# Supplementary Table S5c: E1 identity and site accounting
# ---------------------------------------------------------------------------

def backbone_label(config: dict) -> str:
    model = config["model"]
    return BACKBONE_LABELS[model["backbone"]].format(**model)


def identity_records(view) -> tuple[list[dict], dict]:
    """Every seed's E1 record over the run (or the ten split runs of a campaign)."""
    rows, instrumentation = [], None
    for member in view.members():
        entry = member.json("train/entry_initial.json")
        config = member.json("config.json")
        instrumentation = entry["instrumentation"]
        if not entry["nominal_identity"]:
            raise DisplayInputError(f"{view.dataset}: the E1 record of {member.root.name} is empty")
        for record in entry["nominal_identity"]:
            rows.append({"split": member.root.name, "seed": record["seed"],
                         "precision": config["model"]["precision"],
                         "state": float(record["max_relative_state_error"]),
                         "loss": float(record["absolute_fit_val_loss_deviation"]),
                         "state_limit": float(record["state_tolerance"]),
                         "loss_limit": float(record["loss_tolerance"]), "pass": bool(record["pass"])})
    return rows, instrumentation


def site_text(instrumentation: dict) -> str:
    native = int(instrumentation["native_site_count"])
    inserted = int(instrumentation["inserted_site_count"])
    controlled = int(instrumentation["controlled_depth"])
    if not inserted:
        return "none"
    if instrumentation["backbone"] == "gps":
        per = (controlled - 2) // native - 1
        layout = f"{per} per block plus 2 boundary sites"
    else:
        per = controlled // native - 1
        layout = f"{per} per native layer"
    return (f"{inserted} inserted ({layout}; {controlled} controlled sites with the {native} native sites); "
            "zero deviation by construction")


def scope_text(instrumentation: dict) -> str:
    if instrumentation["backbone"] == "gps":
        return "E1 record; inserted-site action mix recorded separately"
    if instrumentation["backbone"] == "linear_gine":
        return "E1 record; nonlinear messages excluded"
    return "E1 record"


@display("supplementary:Table S5c", stem="S5c", title=S5C_TITLE, needs=("runs",))
def table_s5c(ctx, out):
    tolerances = ctx.constants["precision_tolerances_table_S5c"]
    rows, source_rows = [], []
    for dataset in DATASETS:
        view = ctx.require_run(dataset)
        records, instrumentation = identity_records(view)
        precision = records[0]["precision"]
        if len({r["precision"] for r in records}) != 1:
            raise DisplayInputError(f"{dataset}: split runs use different precisions")
        state = max(r["state"] for r in records)
        loss = max(r["loss"] for r in records)
        state_limit = max(r["state_limit"] for r in records)
        loss_limit = max(r["loss_limit"] for r in records)
        registered = tolerances[precision]
        if state_limit != registered["state_relative"] or loss_limit != registered["loss_nats"]:
            out.note(f"{SHORT[dataset]}: the run tested against state/loss tolerances {state_limit:g}/{loss_limit:g}; "
                     f"registered for {precision}: {registered['state_relative']:g}/{registered['loss_nats']:g}.")
        designated = int(instrumentation["inserted_site_count"]) > 0
        where = f" at the {instrumentation['native_site_count']} designated sites" if designated else ""
        config = view.members()[0].json("config.json")
        rows.append([f"{dataset} ({backbone_label(config)})",
                     precision, f"{sci(state)}{where} (limit {sci(state_limit)})",
                     f"{sci(loss)} (limit {sci(loss_limit)})", site_text(instrumentation), scope_text(instrumentation)])
        for r in records:
            source_rows.append([dataset, r["split"], r["seed"], precision, r["state"], r["state_limit"], r["loss"],
                                r["loss_limit"], r["pass"], instrumentation["backbone"],
                                instrumentation["controlled_depth"], instrumentation["native_site_count"],
                                instrumentation["inserted_site_count"]])
        out.uses("run", dataset, ["train/entry_initial.json", "config.json"])
    out.table(["Dataset (backbone)", "precision", "max. relative state deviation", "test-loss deviation (nats)",
               "inserted sites", "reporting scope"], rows,
              notes=["Maximum over seeds (and the ten official splits of Roman-empire) of the train-stage E1 record "
                     "(train/entry_initial.json): relative Frobenius state difference between native execution and "
                     "the A0 adapter over all controlled sites, and the absolute loss deviation of the two executions "
                     "on the internal D_fit validation subset, against the tolerances the run tested.",
                     "Site counts from the plant's instrumentation record; at an inserted site A0 is a literal bypass."])
    out.data("identity_records", ["dataset", "split", "seed", "precision", "max_relative_state_error", "state_tolerance",
                                  "loss_deviation_nats", "loss_tolerance", "pass", "backbone", "controlled_sites",
                                  "native_sites", "inserted_sites"], source_rows)


# ---------------------------------------------------------------------------
# Supplementary Table S29 and Figure S5: horizon and beam sweep
# ---------------------------------------------------------------------------

def sweep_points(rec) -> dict:
    spec = rec.spec
    dataset = spec["dataset"]
    horizon, beam = int(spec["registered"]["horizon"]), int(spec["registered"]["beam_width"])
    rows = {"horizon": [], "beam": []}   # the two sweeps; "registered_*" hold the registered configuration
    for kind, grid in (("horizon", spec["grid"]["horizon"]), ("beam", spec["grid"]["beam_width"])):
        for value in grid:
            h, b = (int(value), beam) if kind == "horizon" else (horizon, int(value))
            keys = {"dataset": dataset, "horizon": h, "beam_width": b}
            effect = require_value(rec, "selection_effect_nats", **keys)
            cost = require_value(rec, "expected_serial_cost", **keys)
            rows[kind].append({"horizon": h, "beam_width": b, "effect": effect, "cost": cost,
                               "fraction": value_of(rec, "realized_fraction", **keys),
                               "headroom": value_of(rec, "oracle_headroom_bound_nats", **keys),
                               "source": source(rec, "selection_effect_nats", **keys)})
    return {"dataset": dataset, "registered_horizon": horizon, "registered_beam": beam, **rows}


SWEEP_COLUMNS = ["sweep", "horizon", "beam_width", "selection_effect_nats", "expected_serial_cost",
                 "oracle_headroom_bound_nats", "realized_fraction", "source_file", "locator"]


def sweep_rows(points, kind):
    return [[kind, p["horizon"], p["beam_width"], p["effect"], p["cost"], p["headroom"], p["fraction"], *p["source"]]
            for p in points[kind]]


@display("supplementary:Table S29", stem="S29", title="Planning horizon and beam sweep",
         needs=("records:horizon_beam_sweep",))
def table_s29(ctx, out):
    rec = ctx.require_records("horizon_beam_sweep")
    points = sweep_points(rec)
    rows = []
    for p in points["horizon"]:
        rows.append([f"horizon H = {p['horizon']} (beam {p['beam_width']})", num(p["effect"], 5), num(p["cost"], 1),
                     num(p["fraction"], 2)])
    for p in points["beam"]:
        rows.append([f"beam {p['beam_width']} (H = {p['horizon']})", num(p["effect"], 5), num(p["cost"], 1),
                     num(p["fraction"], 2)])
    out.table(["configuration", "selection effect (nats)", "expected serial cost", "realized fraction"], rows,
              notes=[f"MPC-RH on {points['dataset']}; registered configuration H = {points['registered_horizon']} at beam "
                     f"{points['registered_beam']}. Selection effect: nested D_select effect against the reconstructed q_C; "
                     "expected serial cost in nominal executions; realized fraction: selection effect over the "
                     "registered configuration's headroom bound U(Delta_oracle)."])
    out.data("sweep", SWEEP_COLUMNS, sweep_rows(points, "horizon") + sweep_rows(points, "beam"))
    out.uses("records", "horizon_beam_sweep", sorted({p["source"][0] for k in ("horizon", "beam") for p in points[k]}))


@display("supplementary:Figure S5", stem="FigS5", title="Horizon and beam sensitivity",
         needs=("records:horizon_beam_sweep",))
def figure_s5(ctx, out):
    rec = ctx.require_records("horizon_beam_sweep")
    points = sweep_points(rec)
    plt = pyplot()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.6, 3.4))
    horizon = points["horizon"]
    xs = [p["horizon"] for p in horizon]
    ys = [p["effect"] for p in horizon]
    ax_a.plot(xs, ys, "o-", color="#1b9e77")
    ax_a.set_xscale("log", base=2)
    ax_a.set_xticks(xs, [str(x) for x in xs])
    ax_a.axvline(points["registered_horizon"], color="0.5", ls=":", lw=1)
    ax_a.text(points["registered_horizon"], max(ys), f" registered\n H={points['registered_horizon']}", fontsize=7,
              va="top", color="0.35")
    for p in horizon:
        if p["fraction"] is not None:
            ax_a.annotate(f"{p['fraction']:.2f} of headroom", (p["horizon"], p["effect"]), textcoords="offset points",
                          xytext=(4, 6), fontsize=7)
    ax_a.set_xlabel("horizon H")
    ax_a.set_ylabel("selection effect vs q_C (nats)")
    ax_a.set_title(f"a  Horizon sensitivity on {SHORT.get(points['dataset'], points['dataset'])} "
                   f"(beam {points['registered_beam']})",
                   loc="left")
    beam = points["beam"]
    ax_b.plot([p["cost"] for p in beam], [p["effect"] for p in beam], "o-", color="#1b9e77",
              label=f"beam sweep at H={points['registered_horizon']}")
    for p in beam:
        ax_b.annotate(f"beam {p['beam_width']}", (p["cost"], p["effect"]), textcoords="offset points", xytext=(4, 6),
                      fontsize=7)
    ax_b.plot([p["cost"] for p in horizon], [p["effect"] for p in horizon], "s--", color="0.55", ms=3,
              label="horizon sweep (panel a)")
    ax_b.set_xlabel("expected serial compute (one nominal execution = 1)")
    ax_b.set_ylabel("selection effect vs q_C (nats)")
    ax_b.set_title(f"b  Beam-width sensitivity at H={points['registered_horizon']} and the horizon sweep", loc="left")
    ax_b.legend(frameon=False)
    for ax in (ax_a, ax_b):
        ax.grid(alpha=0.25)
        ax.margins(x=0.12, y=0.14)
    fig.tight_layout()
    out.data("panel_a", SWEEP_COLUMNS, sweep_rows(points, "horizon"))
    out.data("panel_b", SWEEP_COLUMNS, sweep_rows(points, "beam") + sweep_rows(points, "horizon"))
    out.figure(fig)
    out.uses("records", "horizon_beam_sweep", sorted({p["source"][0] for k in ("horizon", "beam") for p in points[k]}))


# ---------------------------------------------------------------------------
# Supplementary Table S27: buffer sensitivity
# ---------------------------------------------------------------------------

MAIN_T1 = "main_table_1_policy_and_deployment_ledger"


@display("supplementary:Table S27", stem="S27", title="Buffer sensitivity on dependent graphs",
         needs=("records:buffer_sensitivity",))
def table_s27(ctx, out):
    rec = ctx.require_records("buffer_sensitivity")
    spec = rec.spec
    registered = int(spec["registered"]["buffer_hops"])
    rows, data = [], []
    for dataset in spec["datasets"]:
        epsilon = ctx.config(dataset, "full")["admission"]["epsilon_a"]
        for hops in spec["grid"]["buffer_hops"]:
            keys = {"dataset": dataset, "buffer_hops": int(hops)}
            mean = require_value(rec, "clipped_effect_nats", **keys)
            # A buffer left with one resampling unit has no interval (the record omits it).
            half = value_of(rec, "half_width_nats", **keys)
            upper = value_of(rec, "upper_nats", **keys)
            admitted = require_value(rec, "admitted", **keys)
            if upper is None:
                decision = "one resampling unit: no interval"
            elif admitted:
                decision = "pass"
            elif upper >= 0:
                decision = "interval includes zero"
            else:
                decision = f"D_bar above -{epsilon:g}"
            rows.append([SHORT[dataset], integer(hops), num(mean, 4), num(half, 4) if half is not None else "—",
                         num(upper, 4, signed=True) if upper is not None else "—", decision])
            data.append([dataset, hops, mean, half, upper, admitted, value_of(rec, "resampling_units", **keys), epsilon,
                         *source(rec, "clipped_effect_nats" if upper is None else "upper_nats", **keys)])
            if int(hops) == registered:
                out.compare(MAIN_T1, (dataset, "admission_D_bar"), mean)
                out.compare(MAIN_T1, (dataset, "admission_U_desc"), upper)
    out.table(["Dataset", "buffer (hops)", "D_bar", "half-width", "U_desc", "decision at eps_a"], rows,
              notes=["Clipped paired effect on D_admit (the registered value at every buffer) and the one-sided "
                     "block-bootstrap half-width and upper limit over the buffer's resampling units "
                     "(cm_experiments/buffered_admission.py); decision by the descriptive rule (D_bar <= -eps_a and "
                     "U_desc < 0). At the registered buffer each unit is one core and the row reproduces the run's "
                     "sealed admission."])
    out.data("buffer_sensitivity", ["dataset", "buffer_hops", "clipped_effect_nats", "half_width_nats", "upper_nats",
                                    "admitted", "resampling_units", "epsilon_a", "source_file", "locator"], data)
    out.uses("records", "buffer_sensitivity", sorted({row[-2] for row in data}))


# ---------------------------------------------------------------------------
# Supplementary Table S18: registered sensitivity grids
# ---------------------------------------------------------------------------

def parameter_grid(ctx, parameter: str, spec: dict, dataset: str) -> list[float]:
    """The registered grid of one parameter; the selection margin has a CE and a mean-BCE grid."""
    grid = constant(ctx, spec["parameters"][parameter]["grid"])
    if isinstance(grid, dict):
        bce = dataset in constant(ctx, "selection_eq19_algorithm_S4/datasets_using_BCE_margins")
        grid = grid["mean_BCE" if bce else "CE"]
    return [float(v) for v in grid]


def at_registered(ctx, values: dict, parameter: str, dataset: str) -> float:
    """The value at the registered setting (the nearest grid point when the registered value is off the grid)."""
    registered = registered_setting(ctx, parameter, dataset)
    return values[min(values, key=lambda s: (abs(s - registered), s))]


def grid_cell(ctx, parameter: str, spec: dict) -> str:
    grid = constant(ctx, spec["parameters"][parameter]["grid"])
    if isinstance(grid, dict):
        return f"CE: {grid_text(grid['CE'])}; mean BCE: {grid_text(grid['mean_BCE'])}"
    return grid_text(grid)


def registered_setting(ctx, parameter: str, dataset: str) -> float:
    section, key = {"p_min": ("support", "p_min"), "n_min": ("support", "n_min"),
                    "max_normalized_weight": ("support", "max_normalized_weight"),
                    "epsilon_r": ("selection", "epsilon_r"), "epsilon_eq": ("selection", "epsilon_eq"),
                    "clip": ("admission", "clip")}[parameter]
    return float(ctx.config(dataset, "full")[section][key])


def grid_values(rec, ctx, parameter: str, quantity: str) -> dict:
    """{dataset: {setting: value}} for one quantity over the registered grid, every dataset required."""
    spec = rec.spec
    out = {}
    for dataset in DATASETS:
        values = {}
        for setting in parameter_grid(ctx, parameter, spec, dataset):
            value = value_of(rec, quantity, parameter=parameter, setting=f"{setting:g}", dataset=dataset)
            if value is None:
                raise DisplayInputError(f"sensitivity_grids: {quantity} at {parameter} = {setting:g} on {dataset} "
                                        "is not in the record set")
            values[setting] = value
        out[dataset] = values
    return out


def observed_effect(ctx, rec, parameter: str) -> tuple[str, dict]:
    """The registered summary of one grid's observations; returns the text and its inputs."""
    if parameter == "p_min":
        simpler = grid_values(rec, ctx, parameter, "route_simpler")
        shares = {}
        for dataset, values in simpler.items():
            registered = registered_setting(ctx, parameter, dataset)
            for setting, value in values.items():
                if setting > registered:
                    shares.setdefault(setting, []).append(value)
        if not shares:
            raise DisplayInputError("sensitivity_grids: the p_min grid has no stricter setting")
        percents = sorted(100.0 * sum(v) / len(v) for v in shares.values())
        span = (f"{percents[0]:.0f}%" if round(percents[0]) == round(percents[-1])
                else f"{percents[0]:.0f}{DASH}{percents[-1]:.0f}%")
        return f"stricter values simplify {span} of selected routes", {"share_simpler_by_setting": shares}
    if parameter == "n_min":
        changed = grid_values(rec, ctx, parameter, "route_changed")
        strata = grid_values(rec, ctx, parameter, "strata_without_transport")
        grid = sorted(next(iter(changed.values())))
        stable = None
        for setting in grid:
            if any(values[setting] > 0 for values in changed.values()):
                break
            stable = setting
        removes = next((s for s in grid if any(
            s > registered_setting(ctx, parameter, d) and strata[d][s] > at_registered(ctx, strata[d], parameter, d)
            for d in DATASETS)), None)
        text = (f"route stable through {stable:g}" if stable is not None else f"route changes at {grid[0]:g}")
        if removes is not None:
            text += f"; {removes:g} removes weak strata"
        return text, {"route_changed": changed, "strata_without_transport": strata}
    if parameter == "max_normalized_weight":
        ess = float(constant(ctx, "support_eq7/n_min_ESS"))
        ess_grid = [float(v) for v in constant(ctx, "sensitivity_grids_table_S18/minimum_ESS")]
        grid = [float(v) for v in constant(ctx, "sensitivity_grids_table_S18/maximum_normalized_weight")]
        bound, loose = 1 / math.sqrt(ess), 1 / math.sqrt(min(ess_grid))
        inert = [g for g in grid if g > bound]
        text = (f"implied by the ESS threshold: ESS ≥ {ess:g} gives ωmax ≤ {bound:.3f}, so the grid points "
                f"{grid_text([inert[0]])}{DASH}{grid_text([inert[-1]])} cannot change any route when ESS ≥ {ess:g} "
                f"is enforced on the same cells; the grid point {grid_text([grid[-1]])} therefore cannot admit "
                "additional radius/MPC routes under that ESS threshold, and the grid is informative only together "
                f"with the ESS grid (at ESS ≥ {min(ess_grid):g}, ωmax ≤ {loose:.3f})")
        return text, {"omega_bound_at_registered_ess": bound, "omega_bound_at_smallest_ess": loose, "inert_points": inert}
    if parameter == "epsilon_r":
        eligible = grid_values(rec, ctx, parameter, "eligible_count")
        drops = {d: values[max(values)] - at_registered(ctx, values, parameter, d) for d, values in eligible.items()}
        text = ("weak routes disappear under stricter margin" if any(v < 0 for v in drops.values())
                else "eligible family unchanged under stricter margin")
        return text, {"eligible_count": eligible, "change_at_strictest": drops}
    if parameter == "epsilon_eq":
        changed = grid_values(rec, ctx, parameter, "route_changed")
        text = ("complexity selection changes near the frontier" if any(v > 0 for values in changed.values()
                                                                        for v in values.values())
                else "selected route unchanged over the grid")
        return text, {"route_changed": changed}
    if parameter == "clip":
        upper = grid_values(rec, ctx, parameter, "admission_upper_nats")
        spread = {d: max(v.values()) - min(v.values()) for d, v in upper.items()}
        most = max(spread, key=lambda d: (spread[d], d))
        return f"{SHORT[most]} most sensitive", {"admission_upper_nats": upper, "spread": spread}
    raise ValueError(parameter)


@display("supplementary:Table S18", stem="S18", title="Registered sensitivity grids and observations",
         needs=("records:sensitivity_grids",))
def table_s18(ctx, out):
    rec = ctx.require_records("sensitivity_grids")
    spec = rec.spec
    rows, data = [], []
    for parameter in S18_ORDER:
        definition = spec["parameters"][parameter]
        text, inputs = observed_effect(ctx, rec, parameter)
        rows.append([definition["label"], grid_cell(ctx, parameter, spec), text, definition["required_report"]])
        data.append([parameter, grid_cell(ctx, parameter, spec), text, repr(inputs)])
    out.table(["parameter", "grid", "observed effect on route/support", "required report"], rows,
              notes=["Grids from paper/registered_constants.json; one parameter varies at a time. Shares count the "
                     "selected route of each dataset (Roman-empire: mean over its ten splits); the maximum-weight row "
                     "is the identity ESS <= 1/omega_max^2 evaluated at the registered ESS threshold."])
    out.data("summaries", ["parameter", "grid", "observed_effect", "inputs"], data)
    out.data("records", [*rec.keys, "quantity", "value", "unit", "source_file", "locator"],
             [[row.get(k) for k in rec.keys] + [row["quantity"], row["value"], row.get("unit"), row.get("source_file"),
                                                row.get("locator")] for row in rec.rows])
    out.uses("records", "sensitivity_grids", sorted({row.get("source_file", "") for row in rec.rows}))


# ---------------------------------------------------------------------------
# Supplementary Table S17: module ablations
# ---------------------------------------------------------------------------

S23_GROUP = "supplementary_table_S23_assurance_and_expected_compute"


def _format(value: float, term: dict, *, signed: bool | None = None) -> str:
    decimals = int(term.get("decimals", 4))
    if term.get("format") == "percent":
        return f"{100 * value:.{decimals}f}"
    return num(value, decimals, signed=term.get("signed", False) if signed is None else signed)


def signature_term(rec, module: str, term: dict) -> tuple[str | None, list]:
    """One printed component of a module's observed signature, with the rows it used."""
    quantity = term["quantity"]
    arms = term.get("arms") or [term["arm"]]
    rows = [r for r in rec.select(quantity, module=module) if r.get("arm") in arms]
    if not rows:
        return None, []
    percent = "%" if term.get("format") == "percent" else ""
    if "arms" in term:
        parts = []
        for arm in arms:
            chosen = [r for r in rows if r["arm"] == arm]
            if len(chosen) != 1:
                return None, rows
            parts.append(_format(float(chosen[0]["value"]), term))
        value = term.get("join", "/").join(parts) + percent
    else:
        pooled = [r for r in rows if r.get("dataset") == "pooled"]
        values = [float(r["value"]) for r in (pooled or rows)]
        if term.get("aggregate") == "range" and not pooled:
            low, high = min(values), max(values)
            first, last = _format(low, term), _format(high, term, signed=False if low * high > 0 else None)
            value = (first if first.lstrip("+") == last.lstrip("+") else f"{first}{DASH}{last}") + percent
        elif len(values) == 1:
            value = _format(values[0], term) + percent
        else:
            raise DisplayInputError(f"module_ablations: {module}/{quantity} needs one pooled value")
        zero = term.get("zero_text")
        if zero and len(values) == 1 and float(_format(values[0], term, signed=False)) == 0:
            value = zero
    label = term.get("label")
    text = f"{label} {value}" if label else value
    return text + term.get("suffix", ""), rows


@display("supplementary:Table S17", stem="S17", title="Module ablations and observed signatures",
         needs=("records:module_ablations",))
def table_s17(ctx, out):
    rec = ctx.require_records("module_ablations")
    rows, data, missing = [], [], []
    for module in rec.spec["modules"]:
        parts, used = [], []
        for term in module["signature"]:
            text, rows_used = signature_term(rec, module["module"], term)
            used += rows_used
            if text is None:
                parts = None
                break
            parts.append((term.get("separator", "; ") if parts else "") + text)
        signature = "".join(parts) if parts else None
        if signature is None:
            missing.append(module["label"])
        rows.append([module["label"], module["failure_targeted"], module["comparison"], signature,
                     module["removal_rule"]])
        for r in used:
            data.append([module["module"], r.get("arm"), r.get("dataset"), r["quantity"], r["value"], r.get("unit"),
                         r.get("source_file"), r.get("locator")])
    for r in rec.select("retained_effect_fraction", module="full_lean_profile", arm="lean"):
        if r.get("dataset") in DATASETS:
            out.compare(S23_GROUP, (r["dataset"], "lean_effect"), float(r["value"]))
    notes = ["A module is credited by its matched comparison only; signatures are read from the module_ablations "
             "record set (ranges span the datasets that hold the contrast)."]
    if missing:
        notes.append("No observations for: " + ", ".join(missing) + " (see the module_ablations contract in "
                     "records/specs.json, field runtime).")
    out.table(["module", "failure targeted", "comparison", "observed signature", "removal/downgrade rule"], rows,
              notes=notes)
    out.data("signatures", ["module", "arm", "dataset", "quantity", "value", "unit", "source_file", "locator"], data)
    out.uses("records", "module_ablations", sorted({r[6] for r in data if r[6]}))


# ---------------------------------------------------------------------------
# Supplementary Table S43: outcomes, estimators and thresholds
# ---------------------------------------------------------------------------

S42_GROUP = "supplementary_table_S42_power_of_budget_matched_test"
S46_GROUP = "supplementary_table_S46_budget_matched_frontier"
S45_GROUP = "supplementary_table_S45_inserted_site_attribution"
S26B_GROUP = "supplementary_table_S26b_ensemble_law_and_break_even"
S23B_GROUP = "supplementary_table_S23b_nominal_execution_time_and_amortization"
S30_GROUP = "supplementary_table_S30_oracle_dispersion"
S16B_GROUP = "supplementary_table_S16b_pascal_factorial"
S6C_GROUP = "supplementary_table_S6c_sensing_cost_identity"
MAIN_T2 = "main_table_2_gain_and_cost_ledger"
S43_NEEDS = ("runs", "records:budget_matched", "records:inserted_site", "records:ensemble_grid",
             "records:input_gate", "records:nominal_time", "records:baselines", "records:factorials",
             "records:confirmation_sensitivity", "records:band_bound_tightness", "records:oracle_dispersion")
ARM_NATIVE_FEEDBACK = "native, feedback"
ARM_DEPLOYED = "inserted, feedback (deployed)"
FACTORS = ("spectral_information", "conditioned_observer", "risk_controlled_execution")
P1_GAP = "the late-abort probability p1 of q_B is not a quantity of the budget_matched record set"


def arm_value(rec, quantity: str, dataset: str, arm: str) -> float:
    """A budget-matched arm value; the arm key is 'E3' (campaign export) or 'E3 <name>' (mapping export)."""
    rows = [r for r in rec.select(quantity, dataset=dataset)
            if r.get("arm") == arm or str(r.get("arm", "")).startswith(arm + " ")]
    if len(rows) != 1:
        raise DisplayInputError(f"budget_matched: {quantity} of {arm} on {dataset} is missing or ambiguous")
    return float(rows[0]["value"])


def short_list(pairs, fmt, *, group: bool = False) -> str:
    """'arxiv 2.9; proteins 2.5'; with ``group`` consecutive equal printed values share one entry ('Pascal, COCO 0.22')."""
    groups = []
    for dataset, value in pairs:
        text = fmt(value)
        if group and groups and groups[-1][1] == text:
            groups[-1][0].append(SHORT[dataset])
        else:
            groups.append(([SHORT[dataset]], text))
    return "; ".join(f"{', '.join(names)} {text}" for names, text in groups)


def factorial_effects(rec, dataset: str = "PascalVOC-SP") -> dict:
    """Effect-coded main effects of the 2^3 factorial (cm_theory.factorial.effect)."""
    from cm_theory.factorial import effect
    cells = {}
    for cell in ("000", "001", "010", "011", "100", "101", "110", "111"):
        cells[cell] = require_value(rec, "ld_integrated_loss", dataset=dataset, cell=cell)
    return {name: effect(cells, (index,)) for index, name in enumerate(FACTORS)}, cells


def admission_statistics(view) -> tuple[float, float, list]:
    """Clipped D_bar and U_desc on D_admit: cluster bootstrap, or the repeated-split t limit for a campaign."""
    per = []
    for member in view.members():
        decision = member.json("admit/decision.json")
        per.append({"split": member.root.name, "mean": float(decision["loss"]["mean"]),
                    "upper": decision["loss"]["upper"]})
    if not view.is_campaign:
        return per[0]["mean"], float(per[0]["upper"]), per
    means = [p["mean"] for p in per]
    n = len(means)
    t = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}.get(n - 1, 1.96)
    mean = sum(means) / n
    sd = math.sqrt(sum((m - mean) ** 2 for m in means) / (n - 1))
    return mean, mean + t * sd / math.sqrt(n), per


def cost_models(ctx) -> dict:
    """Per dataset: the Table S44 central tail overhead and the Table S6c admissible (k/d)/P_s.

    The deployment ledger, the sparse-to-dense ratio, the registered truncation order and the
    tail-model constants are those of paper.build.displays.analytical (Tables S44, S6c), so the
    two displays and this one share one definition.
    """
    from cm_theory import budget as cost
    from .analytical import (SENSING_SHARE, TAIL_REFERENCE_SHARE, deployment, registered_order, registry_actions,
                             sparse_ratio)
    order = registered_order(registry_actions(ctx)[0])
    models = {}
    for dataset in ADMITTED:
        dep = deployment(ctx, dataset)
        r = sparse_ratio(ctx, dataset)
        f_bar = min(TAIL_REFERENCE_SHARE, cost.max_resolvent_fraction(dep["m"], r, SENSING_SHARE, order))
        tail = cost.tail_scenarios(dep["C_qC"], dep["m"] - 1.0, f_bar, r, order)
        models[dataset] = {"m": dep["m"], "r": r, "f_bar": f_bar, "tail_overhead_pct": tail["overhead_pct"][1],
                           "admissible_sketch_ratio": cost.admissible_sketch_ratio(dep["m"], r)}
    return models


def amortization(ctx, dataset: str, gate, baselines, t_nom: float) -> dict:
    """M* of main Definition 2 exactly as Table S23b computes it (paper.build.displays.records_campaigns)."""
    from cm_theory.economics import amortization_horizon
    from .records_campaigns import amortization_theta, method_labels, method_value, run_ledger
    ledger = run_ledger(ctx, dataset, gate)
    qc_key, qd_key = method_labels(ledger)
    total = method_value(baselines, "gpu_hours", dataset, qd_key or qc_key)
    plant = method_value(baselines, "gpu_hours", dataset, qc_key)
    added = None if ledger["C_exp"] is None else ledger["C_exp"] - ledger["C_qC"]
    if total is None or plant is None or not added or added <= 0:
        raise DisplayInputError(f"{dataset}: M* needs the qualification GPU-hours of q_D and q_C and a positive added "
                                "expected work")
    theta = amortization_theta(ctx)
    qualification = float(total) - float(plant)
    return {"qualification_gpu_h": qualification, "added": added, "theta": theta,
            "M_star": amortization_horizon(qualification, theta, added, t_nom)}


def returns_comparator(view) -> bool:
    return all(m.evidence["admission"]["deployed_index"] == m.evidence["selection"]["qC_index"]
               for m in view.members())


def scale_text(value: float) -> str:
    return f"{value:.1f}" if round(value, 1) == value else f"{value:g}"


@display("supplementary:Table S43", stem="S43", title="Experiment outcomes, estimators and thresholds",
         needs=S43_NEEDS)
def table_s43(ctx, out):
    from cm_theory.economics import fit_ensemble_law
    constants = ctx.constants
    budget = ctx.require_records("budget_matched")
    inserted = ctx.require_records("inserted_site")
    grid = ctx.require_records("ensemble_grid")
    gate = ctx.require_records("input_gate")
    nominal = ctx.require_records("nominal_time")
    baselines = ctx.require_records("baselines")
    factorials = ctx.require_records("factorials")
    confirmation = ctx.require_records("confirmation_sensitivity")
    tightness = ctx.require_records("band_bound_tightness")
    oracle = ctx.require_records("oracle_dispersion")
    rows, data, gaps = [], [], []

    def add(experiment, quantity, value, estimator, threshold, printed, gap=None):
        rows.append([experiment, quantity, value, estimator, threshold, printed])
        if gap:
            gaps.append(f"{experiment}: {gap}")

    # E1 -------------------------------------------------------------------
    bf16 = []
    for dataset in DATASETS:
        records, _ = identity_records(ctx.require_run(dataset))
        for r in records:
            data.append(["E1", dataset, r["split"], f"seed {r['seed']}", "max_relative_state_error", r["state"],
                         "train/entry_initial.json"])
            if r["precision"] == "bf16":
                bf16.append(r["state"])
    if not bf16:
        raise DisplayInputError("no bf16 plant among the runs")
    limit = constants["precision_tolerances_table_S5c"]["bf16"]["state_relative"]
    add("E1", "maximum relative state deviation of A0 (bf16 plants)", sci(max(bf16)),
        "maximum over sites of the relative state difference between native execution and A0",
        f"within {sci(limit)}", "Table S5c")

    # E2 -------------------------------------------------------------------
    shares = constants["budget_matched_comparison_section4_table_S32_S40"]["retained_share_grid_phi_x"]
    fixed, mpc = shares["fixed_sequence_regimes"], shares["MPC_OL_regimes"]
    add("E2", "phi_x^gain, fixed-sequence regimes / MPC-OL regimes",
        f"{fixed[1]:.2f} [{fixed[0]:.2f}, {fixed[2]:.2f}] / {mpc[1]:.2f} [{mpc[0]:.2f}, {mpc[2]:.2f}]",
        "paired share of the feedback gain retained by the predicted sequences on D_compare; central value and "
        "grid (Note S25)", "above 1 favours the predictor; above r_B favours it over q_B", "Table S47")
    for dataset in ADMITTED:
        comparator = ctx.require_run(dataset).evidence["selection"].get("qC") or {}
        regime = mpc if comparator.get("route") == "mpc_ol" else fixed
        out.compare(S30_GROUP, (dataset, "phi_x_central"), regime[1])

    # E3-E6 ----------------------------------------------------------------
    margins, distilled, replanning, qb = [], [], [], []
    for dataset in ADMITTED:
        e3 = arm_value(budget, "effect_vs_feedback_nats", dataset, "E3")
        e4 = arm_value(budget, "effect_vs_qc_nats", dataset, "E4")
        e5 = arm_value(budget, "effect_vs_qc_nats", dataset, "E5")
        e6 = arm_value(budget, "effect_vs_qc_nats", dataset, "E6")
        feedback = e6 - arm_value(budget, "effect_vs_feedback_nats", dataset, "E6")
        work = arm_value(budget, "work_ratio", dataset, "E6")
        if feedback >= 0:
            raise DisplayInputError(f"budget_matched: the feedback rule has no gain over q_C on {dataset}")
        retained = e6 / feedback
        margins.append((dataset, 1000 * e3))
        distilled.append((dataset, 1000 * e4))
        replanning.append((dataset, 1000 * e5))
        qb.append((dataset, (retained, 100 * (work - 1))))
        out.compare(S42_GROUP, (dataset, "reported_margin_mnats"), 1000 * e3)
        out.compare(S46_GROUP, (dataset, "E4_distilled"), 1000 * e4)
        out.compare(S46_GROUP, (dataset, "E5_no_observation_replanning"), 1000 * e5)
        out.compare(S46_GROUP, (dataset, "E6_overhead_pct"), 100 * (work - 1))
        out.compare(S46_GROUP, (dataset, "feedback_effect_mnats"), 1000 * feedback)
        data += [["E3", dataset, "", "E3", "feedback_minus_predictor_mnats", 1000 * e3, "budget_matched"],
                 ["E4", dataset, "", "E4", "effect_vs_qc_mnats", 1000 * e4, "budget_matched"],
                 ["E5", dataset, "", "E5", "effect_vs_qc_mnats", 1000 * e5, "budget_matched"],
                 ["E6", dataset, "", "E6", "retained_share", retained, "budget_matched"],
                 ["E6", dataset, "", "E6", "overhead_percent", 100 * (work - 1), "budget_matched"]]
    epsilon_eq = constants["selection_eq19_algorithm_S4"]["epsilon_eq_nats"]
    rh_margin = constants["confirmation_eq20_table_S11"]["contrasts"]["MPC-RH"]["practical_margin_nats"]
    add("E3", "feedback minus predictor at the ceiling (mnats)", short_list(margins, lambda v: f"{v:.1f}"),
        "paired difference on D_compare at the common ceiling", "Holm-significant paired margin > 0 retains feedback",
        "Tables S46\u2013S47")
    add("E4", "distilled policy vs q_C (mnats)", short_list(distilled, lambda v: f"{v:.1f}"),
        "paired difference on D_compare; student executed open loop",
        f"student within eps_eq = {epsilon_eq:g} of feedback removes the online-feedback claim", "Table S46")
    add("E5", "no-observation replanning vs q_C (mnats)", short_list(replanning, lambda v: f"{v:.1f}"),
        "paired difference at matched search calls",
        f"RH minus no-observation beyond {rh_margin:g} retains the observation-value claim", "Table S46")
    add("E6", "q_B: retained share; overhead; p1",
        short_list(qb, lambda v: f"{v[0]:.3f}, {v[1]:.0f}%") + "; p1 = \u2014",
        "acquisition-gated execution of Algorithm S3; retained share = gain of q_B over gain of feedback",
        f"retained gain within eps_eq = {epsilon_eq:g} of feedback at lower overhead qualifies q_B for separate "
        "admission", "Tables S33\u2013S34, S46", gap=P1_GAP)

    # E10 ------------------------------------------------------------------
    survival = []
    for row in confirmation.select("label_survival_probability"):
        label = str(row["contrast"]).split("/")[0]
        dataset = next((d for d in DATASETS if SHORT[d] == label), None)
        if dataset is None:
            raise DisplayInputError(f"confirmation_sensitivity: contrast {row['contrast']} names no dataset")
        survival.append((dataset, float(row["value"])))
        data.append(["E10", dataset, "", row["contrast"], "label_survival_probability", float(row["value"]),
                     "confirmation_sensitivity"])
    if not survival:
        raise DisplayInputError("confirmation_sensitivity holds no label survival probability")
    survival.sort(key=lambda item: DATASETS.index(item[0]))
    add("E10", "mechanism label on the confirmation role", short_list(survival, lambda v: f"{v:.2f}"),
        "probability that the Holm-significant label survives at the D_confirm size under the two-way variance model "
        "(Table S14b)", "Holm-significant contrast beyond the margin retains the label; otherwise the frozen map "
        "returns q_C", "Table S14b, main Table 1")

    # E13 ------------------------------------------------------------------
    attribution = []
    for dataset in GRAPH_LEVEL:
        native = require_value(inserted, "effect_nats", dataset=dataset, arm=ARM_NATIVE_FEEDBACK)
        deployed = require_value(inserted, "effect_nats", dataset=dataset, arm=ARM_DEPLOYED)
        attribution.append((dataset, native / deployed))
        out.compare(S45_GROUP, (dataset, "retained_share_arm1prime"), native / deployed)
        data.append(["E13", dataset, "", ARM_NATIVE_FEEDBACK, "share_of_deployed_effect", native / deployed,
                     "inserted_site"])
    add("E13", "native-only feedback arm as a share of the deployed gain",
        short_list(attribution, lambda v: f"{v:.2f}", group=True), "paired three-arm comparison at matched work",
        f"native-only arm within eps_eq = {epsilon_eq:g} of the deployed rule removes the insertion mechanism",
        "Table S45")

    # E14 ------------------------------------------------------------------
    sizes = constants["observer_eq12_table_S9"]["extended_diagnostic_grid_E14"]
    fits = []
    for dataset in ADMITTED:
        losses = [require_value(grid, "loss_nats", dataset=dataset, ensemble=e) for e in sizes]
        fit = fit_ensemble_law(sizes, losses)
        fits.append((dataset, fit["D_inf"]))
        out.compare(S26B_GROUP, (dataset, "D_inf_hat"), fit["D_inf"])
        data.append(["E14", dataset, "", "", "D_inf_nats", fit["D_inf"], "ensemble_grid"])
    add("E14", "D_inf (nats)", short_list(fits, lambda v: f"{v:.4f}"),
        "least-squares fit of l_inf + D_inf/E on the checkpoint grid",
        "break-even thresholds of main Prop. 5 evaluated at the fit", "Table S26b")

    # E17 tail, E19 ---------------------------------------------------------
    models = cost_models(ctx)
    for dataset, model in models.items():
        out.compare(MAIN_T2, (dataset, "C_099_overhead_pct"), model["tail_overhead_pct"])
        data.append(["E17", dataset, "", "central tail scenario", "C099_overhead_percent", model["tail_overhead_pct"],
                     "runs + registered constants (Table S44 model)"])
    add("E17", "99th-percentile cost C_0.99/C_qC - 1",
        short_list([(d, m["tail_overhead_pct"]) for d, m in models.items()], lambda v: f"{v:.0f}%"),
        "tail-model quantile calculated under the scenario assumptions of Table S44; the observed maximum is a "
        "separate trace statistic", "an observed maximum above C_assure requires the budget mask for any hard "
        "per-input bound", "main Table 2")
    add("E19", "per-cell support statistics ESS_c, omega_max,c", "Table S7",
        "one logged weight vector per (stratum, action) cell",
        "a changed support mask re-derives Q_feas, the selected route, the qualified fractions and the cost ledger",
        "Tables S7, S4, S12")

    # E17 wall time, t_nom and M* ------------------------------------------
    wall, timing = [], []
    for dataset in ADMITTED:
        overhead = require_value(gate, "deployed_wall_overhead_percent", dataset=dataset)
        wall.append((dataset, overhead))
        t_nom = require_value(nominal, "t_nom_seconds", dataset=dataset)
        horizon = amortization(ctx, dataset, gate, baselines, t_nom)
        timing.append((dataset, (t_nom, horizon["M_star"])))
        field = "t_nom_measured_ms" if dataset in GRAPH_LEVEL else "t_nom_measured_s"
        out.compare(S23B_GROUP, (dataset, field), 1000 * t_nom if dataset in GRAPH_LEVEL else t_nom)
        out.compare(S23B_GROUP, (dataset, "M_star_inputs"), float(f"{horizon['M_star']:.2g}"))
        data += [["E17", dataset, "", "", "wall_overhead_percent", overhead, "input_gate"],
                 ["E17", dataset, "", "", "t_nom_seconds", t_nom, "nominal_time"],
                 ["E17", dataset, "", "", "qualification_gpu_hours", horizon["qualification_gpu_h"], "baselines"],
                 ["E17", dataset, "", "", "added_expected_work", horizon["added"], "select/candidates.json + input_gate"],
                 ["E17", dataset, "", "", "M_star_inputs", horizon["M_star"], "main Definition 2"]]
    add("E17", "wall-time overhead (deployed)", short_list(wall, lambda v: f"{v:.0f}%"),
        "synchronized wall time after warm-up on one execution object",
        "printed next to the serial cost; the two are not interchangeable", "main Fig. 4f; Tables S15b, S23")

    def timing_text(v):
        t_nom, m_star = v
        t = f"{t_nom:.1f} s" if t_nom >= 0.1 else f"{1000 * t_nom:.2f} ms"
        return f"{t}, {sci(m_star, 2)}"
    add("E17", "per-input nominal execution time t_nom; amortization horizon M*", short_list(timing, timing_text),
        "per-input nominal execution time after warm-up; M* of main Definition 2",
        "a measured t_nom rescales M* inversely; lambda* is unaffected", "main Fig. 5d, Table S23b")
    reduced = [d for d in ADMITTED if ctx.config(d, "full")["sensor"]["sketch_dim"] > 0]
    for dataset in reduced:
        out.compare(S6C_GROUP, (dataset, "admissible_k_over_d_per_P_s"), models[dataset]["admissible_sketch_ratio"])
        data.append(["E17", dataset, "", "reduced sensor", "admissible_k_over_d_per_P_s",
                     models[dataset]["admissible_sketch_ratio"], "runs + registered constants (Table S6c identity)"])
    add("E17", f"sensing implementation on {' and '.join(reduced)}",
        "reduced sensor: (k/d)/P_s \u2264 " + " and \u2264 ".join(f"{models[d]['admissible_sketch_ratio']:.2g}"
                                                                  for d in reduced),
        "sensing-cost identity (Table S6c)", "the identity holds with the recorded (k, P_s, f_res)", "Table S6c")

    # E18 ------------------------------------------------------------------
    ratios = {}
    for row in tightness.select("displacement_ratio"):
        ratios.setdefault(float(row["spectral_scale"]), []).append(float(row["value"]))
        data.append(["E18", "", "", f"{row['action']} at lambda_0 = {row['spectral_scale']}", "displacement_ratio",
                     float(row["value"]), "band_bound_tightness"])
    if not ratios:
        raise DisplayInputError("band_bound_tightness holds no displacement ratio")
    parts = []
    for k, scale in enumerate(sorted(ratios)):
        low, high = min(ratios[scale]), max(ratios[scale])
        span = f"{low:.2f}" if f"{low:.2f}" == f"{high:.2f}" else f"{low:.2f}\u2013{high:.2f}"
        prefix = "\u03bb0 = " if k == 0 else ""
        parts.append(f"{span} at {prefix}{scale_text(scale)}")
    add("E18", "displacement/bound ratio of (6.1)", "; ".join(parts),
        "exact one-step displacement over the band-limited bound at the spectral energy scale of the state",
        "a ratio above one falsifies an assumption of Prop. 6 for that plant", "Fig. S18c")

    # admission ------------------------------------------------------------
    admission = []
    for dataset in ("ogbn-arxiv", "ogbn-proteins", "Roman-empire"):
        mean, upper, per = admission_statistics(ctx.require_run(dataset))
        admission.append((dataset, (mean, upper)))
        out.compare(MAIN_T1, (dataset, "admission_D_bar"), mean)
        out.compare(MAIN_T1, (dataset, "admission_U_desc"), upper)
        for p in per:
            data.append(["admission", dataset, p["split"], "", "clipped_mean_nats", p["mean"], "admit/decision.json"])
    epsilon_ce = ctx.config("ogbn-arxiv", "full")["admission"]["epsilon_a"]
    epsilon_bce = ctx.config("ogbn-proteins", "full")["admission"]["epsilon_a"]
    add("admission", "descriptive D_bar, U_desc on D_admit",
        short_list(admission, lambda v: f"{v[0]:.4f}, {v[1]:.4f}"),
        "clipped paired effect with the cluster-bootstrap (arxiv, proteins) or repeated-split (Roman) upper limit",
        f"D_bar <= -eps_a (eps_a = {epsilon_ce:g} CE, {epsilon_bce:g} BCE) and U_desc < 0 keep G_admit = 1",
        "main Table 1")

    # oracle ---------------------------------------------------------------
    listed = list(GRAPH_LEVEL) + [d for d in NODE_LEVEL if returns_comparator(ctx.require_run(d))]
    dispersion = []
    for dataset in listed:
        depth = require_value(oracle, "median_oracle_depth", dataset=dataset)
        differs = require_value(oracle, "fraction_differs_from_qc", dataset=dataset)
        dispersion.append((dataset, (depth, differs)))
        out.compare(S30_GROUP, (dataset, "median_oracle_depth"), depth)
        out.compare(S30_GROUP, (dataset, "fraction_oracle_differs_from_qC"), differs)
        data += [["oracle", dataset, "", "", "median_oracle_depth", depth, "oracle_dispersion"],
                 ["oracle", dataset, "", "", "fraction_differs_from_qc", differs, "oracle_dispersion"]]
    add("oracle", "median oracle depth; fraction != q_C", short_list(dispersion, lambda v: f"{v[0]:.0f}, {v[1]:.2f}"),
        "per-input oracle enumerated on the admission split after every decision is fixed",
        "dispersion is necessary, not sufficient, for feedback value", "Table S30")

    # factorial ------------------------------------------------------------
    effects, _ = factorial_effects(factorials)
    for name, value in effects.items():
        out.compare(S16B_GROUP, ("effect_coded_main_effects_nats", name), value)
        data.append(["factorial", "PascalVOC-SP", "", name, "main_effect_nats", value, "factorials"])
    add("factorial", "Pascal main effects (nats)", ", ".join(f"{effects[n]:.4f}" for n in FACTORS),
        "effect-coded contrasts of the 2^3 factorial with the joint resampling covariance",
        "interactions beyond one joint half-width falsify first-order additivity", "Table S16b")

    notes = ["Values from the sealed runs and the measurement-record sets named in the source data; thresholds from "
             "configs/*_full.json and paper/registered_constants.json. The tail and sensing rows share the deployment "
             "ledger and constants of Tables S44 and S6c, and M* the definition of Table S23b. The oracle row lists the "
             "independent-graph benchmarks and every benchmark whose comparator is returned."]
    if gaps:
        notes.append("Not derivable from any artifact: " + "; ".join(gaps) + ".")
    out.table(["experiment", "quantity", "value", "estimator", "threshold and consequence", "printed in"], rows,
              notes=notes)
    out.data("values", ["experiment", "dataset", "split", "arm_or_term", "quantity", "value", "source"], data)
    for dataset in DATASETS:
        out.uses("run", dataset, ["train/entry_initial.json", "admit/decision.json", "select/candidates.json"])
    for name in S43_NEEDS[1:]:
        out.uses("records", name.split(":", 1)[1], [])
