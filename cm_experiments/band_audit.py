"""E18: spectral displacement audit of main Proposition 6 (band-power audit).

The lifecycle records the controller's partial bound statistic -- the seven
``reachability_A1..A7`` coordinates of every logged measurement in the sealed
D_explore trajectories -- but not the displacement it bounds, the omitted-band
and window-approximation terms, or the native-update constants. This audit
therefore reads the sealed trajectories and replays each one label-free on the
frozen plant (recorded actions and last-safe anchors; every replayed measurement
is compared with the logged one) and, at each audited state H_l and each
registered transport action a = A1..A7 against the next site's nominal transport
(shared A0, native mean aggregation or inserted identity), measures

* the transport displacement X = (T_a - T_nominal) H in the executed row-stochastic
  realization P = D^-1 W and Y = D^1/2 X in the symmetric realization of
  L_sym (both through ``cross_morse.core.apply_transport``; their parity is recorded);
* the ideal six-band bound [sum_b s_ab^2 P_b(D^1/2 H)]^1/2 of Prop. 6(i) from the exact
  spectrum of L_sym on graphs up to a declared size (s_ab is the band supremum of
  |g_a - g_nominal|, evaluated on a fine grid and at the actual eigenvalues), and
  the tightness ||Y|| / bound, which can exceed one only if a hypothesis fails;
* the complete computable window bound with the registry's Chebyshev windows,
  [sum_b s_ab^2 (||G_b D^1/2 H||^2 + eps_b ||D^1/2 H||^2)]^1/2, its window-approximation
  share (eps_b of Prop. 6(ii)) and, with the exact spectrum, the omitted-band share;
* the retained-window statistic the controller measures (the three registered bands
  with the bank's own sensitivities) against the transport displacement and against
  the logged reachability coordinate;
* the one-step block displacement ||H_l+1(a) - H_l+1(nominal)|| of the frozen
  update, its gain over the transport displacement (the realized native-update
  constant, per depth), and the action ranking agreement between the statistic and
  the observed block displacement.

The model-density tightness of Fig. S20b/S18c, [int (g_a - g_0)^2 dmu]^1/2 over
[sum_b s_ab^2 mu(I_b)]^1/2 at mu ~ exp(-lambda/lambda0), is evaluated from the
registry responses. Policy value over matched confidence/history controls (the
E7 part of Table S36's readout) is not measured here.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .artifacts import sha256, write_json
from .predictability import (ExperimentError, ParentRun, RunDirectory, create_plan, export_record_set, median,
                             open_plan, quantile, read_csv, read_json, summary_row, write_csv)

BANDS = 6
TRANSPORT_ACTIONS = tuple(range(1, 8))
RAW_EXPORTS = ("raw/states.csv", "raw/rankings.csv", "raw/model_densities.csv", "raw/constants.json",
               "raw/operators.json", "raw/replay.csv")
ACTIVATION_LIPSCHITZ_GRID = np.linspace(-12, 12, 480001)
DEFINITIONS = {
    "tightness_ratio": "transport displacement over the ideal six-band bound of Prop. 6(i) in the symmetric "
                       "realization: exact ratio at a model density exp(-lambda/lambda0); median over audited "
                       "states with an exact spectrum",
    "tightness_ratio_max": "maximum over audited states with an exact spectrum of the same ratio; above one falsifies a "
                           "hypothesis of Prop. 6 for this plant",
    "window_bound_ratio": "median over audited states of the transport displacement over the complete computable "
                          "bound from the six Chebyshev windows with their approximation error",
    "retained_statistic_ratio": "median over audited states of the symmetric-realization transport displacement over "
                                "the retained three-window statistic the controller measures",
    "omitted_band_share": "median over audited states with an exact spectrum of the share of the squared ideal bound "
                          "carried by the three bands outside the registered windows",
    "approximation_error_share": "median over audited states of the share of the squared window bound contributed by "
                                 "the window approximation errors eps_b ||D^1/2 H||^2",
    "degree_factor": "median over audited states of ||X|| / (d_min^-1/2 ||Y||), the used part of the degree "
                     "weighting factor of the row-stochastic realization",
    "update_gain_p95": "95th percentile over audited states of the one-step block displacement over the transport "
                       "displacement (realized native-update and retention constant)",
    "logged_statistic_ratio": "median over audited states of the full-state retained statistic over the reachability "
                              "coordinate logged in the sealed D_explore record (one without sketching)",
    "audited_states": "audited (state, action) pairs",
    "exact_spectrum_states": "audited (state, action) pairs on graphs with an exact spectrum",
    "ranking_spearman": "median over audited states of the Spearman correlation across A1-A7 between the retained "
                        "statistic and the observed block displacement",
    "replay_max_abs_deviation": "maximum absolute difference between a replayed and the logged D_explore measurement",
    "update_lipschitz_max": "maximum over depths of the per-depth native-update constant (largest block over "
                            "transport displacement among actions with the nominal retention)",
    "shared_weight_spectral_norm": "spectral norm ||W||_2 of the frozen shared update (message argument)",
    "activation_lipschitz": "Lipschitz constant of the plant's activation",
}


def plan_band_audit(parent, output, *, max_trajectories=8, trajectories=None, depth_stride=1,
                    lambda0=(0.1, 0.3, 1.0), exact_max_nodes=4096, grid_points=4097, allow_fixture=False):
    parent = parent if isinstance(parent, ParentRun) else ParentRun(parent)
    if parent.synthetic and not allow_fixture:
        raise ExperimentError("a synthetic fixture parent requires --allow-fixture")
    checked = parent.check("explore")
    files = sorted((parent.root / "explore").glob("trajectory_*.npz"))
    available = [int(p.stem.split("_")[1]) for p in files]
    if not available:
        raise ExperimentError("the parent has no sealed D_explore trajectories")
    chosen = sorted(available)[:int(max_trajectories)] if trajectories is None else sorted({int(t) for t in trajectories})
    if not chosen or not set(chosen) <= set(available):
        raise ExperimentError("declare existing trajectory indices")
    lambda0 = sorted({float(v) for v in lambda0})
    if not lambda0 or min(lambda0) <= 0 or type(depth_stride) is not int or depth_stride < 1:
        raise ExperimentError("invalid density scales or depth stride")
    if type(exact_max_nodes) is not int or exact_max_nodes < 0 or type(grid_points) is not int or grid_points < 33:
        raise ExperimentError("invalid exact-spectrum ceiling or grid size")
    body = {"trajectories": chosen, "depth_stride": depth_stride, "lambda0": lambda0,
            "exact_max_nodes": exact_max_nodes, "grid_points": grid_points, "parent_check": checked,
            "trajectory_sha256": {str(t): sha256(parent.root / f"explore/trajectory_{t:06d}.npz") for t in chosen},
            "labels": "none: the audit is label-free", "definitions": DEFINITIONS,
            "registered_bands": list(parent.config["sensor"]["bands"]), "window_degree": parent.config["sensor"]["degree"]}
    return create_plan("band_audit", parent, output, body)


class _Symmetric:
    """The symmetric realization S = D^-1/2 W D^-1/2 behind the runtime's transport code."""

    def __init__(self, graph):
        self.graph = graph

    def apply(self, h):
        return self.graph.symmetric(h)


def _nominal(operator, h, baseline):
    from cross_morse.core import ACTIONS, apply_transport
    if baseline == "shared":
        return apply_transport(operator, h, ACTIONS[0])
    if baseline == "native_mean":
        return operator.apply(h)
    if baseline == "inserted_identity":
        return h
    raise ExperimentError(f"unknown nominal transport baseline {baseline!r}")


def response_difference(baseline, action, eigenvalues):
    import torch
    from cross_morse.core import ACTIONS, action_operator_response, nominal_operator_response
    grid = torch.as_tensor(np.asarray(eigenvalues, dtype=np.float64))
    return (action_operator_response(ACTIONS[action], grid) - nominal_operator_response(grid, baseline)).numpy()


def band_suprema(baseline, action, points):
    return [float(np.max(np.abs(response_difference(baseline, action, np.linspace(2 * b / 6, 2 * (b + 1) / 6, points)))))
            for b in range(BANDS)]


def window_constants(coefficients, cells=60000):
    """eps_b = sup |1_Ib - g_b^2| on a midpoint grid (never a band edge) and the frame bounds."""
    lam = (np.arange(cells) + .5) * 2. / cells
    frame = np.zeros_like(lam)
    errors = []
    for b in range(BANDS):
        g = np.polynomial.chebyshev.chebval(lam - 1., np.asarray(coefficients[b], dtype=np.float64))
        inside = ((lam >= 2 * b / 6) & (lam < 2 * (b + 1) / 6)).astype(np.float64)
        errors.append(float(np.max(np.abs(inside - g**2))))
        frame += g**2
    return errors, float(frame.min()), float(frame.max())


def activation_lipschitz(name):
    x = ACTIVATION_LIPSCHITZ_GRID
    if name == "relu":
        return 1.0
    sigmoid = 1. / (1. + np.exp(-x))
    if name == "silu":
        return float(np.max(np.abs(sigmoid * (1. + x * (1. - sigmoid)))))
    if name == "gelu":
        from math import erf, pi, sqrt
        phi = np.exp(-x**2 / 2) / sqrt(2 * pi)
        cdf = .5 * (1. + np.vectorize(erf)(x / sqrt(2.)))
        return float(np.max(np.abs(cdf + x * phi)))
    raise ExperimentError(f"unknown activation {name!r}")


def model_density_tightness(baseline, action, lam0, suprema, points=600001):
    lam = np.linspace(0., 2., points)
    weight = np.exp(-lam / lam0)
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    numerator = integrate(response_difference(baseline, action, lam)**2 * weight, lam)
    masses = [lam0 * (math.exp(-(2 * b / 6) / lam0) - math.exp(-(2 * (b + 1) / 6) / lam0)) for b in range(BANDS)]
    denominator = sum(s * s * m for s, m in zip(suprema, masses))
    return math.sqrt(numerator / denominator)


def spearman(a, b):
    def ranks(v):
        v = np.asarray(v, dtype=np.float64)
        order = np.argsort(v, kind="stable")
        r = np.empty(len(v))
        r[order] = np.arange(len(v), dtype=np.float64)
        for value in np.unique(v):  # average ranks of ties
            mask = v == value
            r[mask] = r[mask].mean()
        return r
    ra, rb = ranks(a), ranks(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _spectrum(graph, ceiling):
    import torch
    difference = (graph.S - graph.S.t()).coalesce().values()
    asymmetry = float(difference.abs().max()) if difference.numel() else 0.
    info = {"nodes": graph.num_nodes, "edge_entries": graph.num_edges, "symmetric_operator_max_asymmetry": asymmetry,
            "d_min": float(graph.degree.min()), "d_max": float(graph.degree.max()), "exact_spectrum": False}
    if graph.num_nodes > ceiling:
        return None, info
    S = graph.S.to_dense().double()
    laplacian = torch.eye(graph.num_nodes, dtype=torch.float64, device=S.device) - S
    values, vectors = torch.linalg.eigh(.5 * (laplacian + laplacian.T))
    info.update(exact_spectrum=True, eigenvalue_min=float(values.min()), eigenvalue_max=float(values.max()))
    return {"values": values.clamp(0., 2.).cpu().numpy(), "vectors": vectors}, info


def audit_state(model, graph, windows, h, h0, last_safe, depth, baseline, registered, suprema, bank_suprema,
                errors, spectrum, logged_reach):
    """Every audited quantity of one state and the seven transport actions (label-free)."""
    import torch
    from cross_morse.core import ACTIONS, apply_transport
    hf = h.float()
    sqrt_degree = graph.sqrt_degree.float()[:, None]
    weighted = hf * sqrt_degree
    sym = _Symmetric(graph)
    filtered = list(windows[0]._filter(weighted)) + list(windows[1]._filter(weighted))
    powers = [float(g.double().square().sum()) for g in filtered]
    energy = float(weighted.double().square().sum())
    d_min = float(graph.degree.min())
    nominal_p, nominal_s = _nominal(graph, hf, baseline), _nominal(sym, weighted, baseline)
    reference = model.step(h, h0, graph, 0, last_safe, site=depth)
    if spectrum is not None:
        energy_density = (spectrum["vectors"].T @ weighted.double().to(spectrum["vectors"].device)).square().sum(1).cpu().numpy()
        band_index = np.minimum(BANDS - 1, np.floor(spectrum["values"] * 3).astype(np.int64))
        band_power = np.bincount(band_index, weights=energy_density, minlength=BANDS)
    rows = []
    for action in TRANSPORT_ACTIONS:
        x = apply_transport(graph, hf, ACTIONS[action]) - nominal_p
        y = apply_transport(sym, weighted, ACTIONS[action]) - nominal_s
        dx, dy = float(x.double().norm()), float(y.double().norm())
        parity = float((x - y / sqrt_degree).double().norm()) / max(dx, 1e-300)
        block = float((model.step(h, h0, graph, action, last_safe, site=depth) - reference).double().norm())
        s = suprema[action]
        retained = math.sqrt(sum(bank_suprema[action - 1][i]**2 * powers[b] for i, b in enumerate(registered)))
        squared = [s[b]**2 * (powers[b] + errors[b] * energy) for b in range(BANDS)]
        window_bound = math.sqrt(sum(squared))
        row = {"action": action, "transport_displacement": dx, "symmetric_displacement": dy,
               "operator_parity_relative_error": parity, "block_displacement": block,
               "retained_statistic": retained, "window_bound": window_bound,
               "approximation_error_share": sum(s[b]**2 * errors[b] * energy for b in range(BANDS)) / window_bound**2
               if window_bound > 0 else None,
               "degree_factor": dx / (dy / math.sqrt(d_min)) if dy > 0 else None,
               "update_gain": block / dx if dx > 0 else None, "state_energy": energy, "d_min": d_min,
               "logged_statistic": float(logged_reach[action - 1]) if logged_reach is not None else None,
               "same_retention": int(ACTIONS[action].alpha == ACTIONS[0].alpha)}
        if spectrum is not None:
            difference = response_difference(baseline, action, spectrum["values"])
            spectral = math.sqrt(float(np.sum(difference**2 * energy_density)))
            exact = [max(s[b], float(np.max(np.abs(difference[band_index == b]), initial=0.))) for b in range(BANDS)]
            ideal = math.sqrt(sum(exact[b]**2 * band_power[b] for b in range(BANDS)))
            omitted = sum(exact[b]**2 * band_power[b] for b in range(BANDS) if b not in registered)
            row.update(spectral_displacement=spectral, ideal_bound=ideal,
                       tightness_ratio=spectral / ideal if ideal > 0 else None,
                       omitted_band_share=omitted / ideal**2 if ideal > 0 else None,
                       spectral_parity_relative_error=abs(spectral - dy) / max(dy, 1e-300))
        rows.append(row)
    return rows


def run_band_audit(plan_path, *, device="cpu", trust=False):
    import torch
    from cross_morse.common import precision_context
    from cross_morse.core import ACTIONS, MeasurementBank, chebyshev_window_coefficients, site_transport_accounting
    from cross_morse.data import load_data
    from cross_morse.exploration import sensor_bank
    from cross_morse.provenance import sha256_state_dict
    from cross_morse.runtime import graph_operator, is_graph_collection
    from cross_morse.training import load_plant

    plan, parent = open_plan(plan_path, "band_audit")
    parent.verify("explore")
    run = RunDirectory(plan_path, plan, parent, device=device, trust=trust)
    root, config = parent.root, parent.config
    for t, digest in plan["trajectory_sha256"].items():
        if sha256(root / f"explore/trajectory_{int(t):06d}.npz") != digest:
            raise ExperimentError(f"sealed trajectory {t} changed since planning")
    data = load_data(root).protect_labels()
    collection = is_graph_collection(data)
    L, registered = parent.depth, [int(b) for b in config["sensor"]["bands"]]
    seeds = parent.seeds
    coefficients = chebyshev_window_coefficients(int(config["sensor"]["degree"])).double().numpy()
    errors, frame_min, frame_max = window_constants(coefficients)
    models, frozen, spectra, operators, suprema = {}, {}, {}, {}, {}
    states, rankings, replay = [], [], []
    gains = {}
    for t in plan["trajectories"]:
        with np.load(root / f"explore/trajectory_{t:06d}.npz", allow_pickle=False) as z:
            record = {k: z[k] for k in z.files}
        info = read_json(root / f"explore/trajectory_{t:06d}.json")
        seed = seeds[int(record["seed_index"][0])]
        graph_id = int(record["graph_id"][0]) if "graph_id" in record else None
        if seed not in models:
            models[seed] = load_plant(data, config, root / "train", seed, device)
            frozen[seed] = sha256_state_dict(models[seed])
        model = models[seed]
        episode = data.graph(graph_id) if collection else data
        graph = graph_operator(episode, device)
        key = graph_id if collection else "graph"
        if key not in spectra:
            spectra[key], operators[str(key)] = _spectrum(graph, plan["exact_max_nodes"])
        sequence, safe = [int(a) for a in record["sequence"]], [int(d) for d in record["safe_depths"]]
        with torch.no_grad(), precision_context(config, torch.device(device)):
            model.bind_graph(graph)
            x = episode.x.to(device)
            h0 = model.encode(x)
            logits0 = model.logits(h0)
            bank = sensor_bank(graph, model, config)
            if list(bank.names) != list(info["sensor_schema"]):
                raise ExperimentError("the replay sensor schema differs from the logged trajectory schema")
            bank.prepare(h0, logits0)
            windows = [MeasurementBank(graph, model.hidden_dim, bands=bands, degree=int(config["sensor"]["degree"]),
                                       sketch_dim=0, seed=config["seed"], profile="full", task=model.task, plant=model)
                       for bands in ((0, 1, 2), (3, 4, 5))]
            reach_index = [bank.names.index(f"reachability_A{a}") for a in TRANSPORT_ACTIONS]
            h, kept, mass, hops = h0, {0: h0}, 0., 0.
            for d in range(L):
                y = bank.measure(h, logits0 if d == 0 else model.logits(h), d, L, mass, hops).cpu().numpy()
                replay.append({"trajectory": t, "depth": d,
                               "max_abs_deviation": repr(float(np.max(np.abs(y - record["y"][d]))))})
                if d % plan["depth_stride"] == 0:
                    baseline = bank.nominal_baseline(d, L)
                    if (baseline, 1) not in suprema:
                        for a in TRANSPORT_ACTIONS:
                            suprema[(baseline, a)] = band_suprema(baseline, a, plan["grid_points"])
                    bank_suprema = bank.sensitivities_by_baseline[baseline].double().cpu().numpy()
                    rows = audit_state(model, graph, windows, h, h0, kept[safe[d]], d, baseline, registered,
                                       {a: suprema[(baseline, a)] for a in TRANSPORT_ACTIONS}, bank_suprema,
                                       errors, spectra[key], record["y"][d][reach_index])
                    for row in rows:
                        states.append({"trajectory": t, "member_seed": seed, "graph_id": "" if graph_id is None else graph_id,
                                       "depth": d, "baseline": baseline, **row})
                        if row["same_retention"] or baseline != "shared":
                            if row["update_gain"] is not None:
                                gains[d] = max(gains.get(d, 0.), row["update_gain"])
                    rankings.append({"trajectory": t, "depth": d, "baseline": baseline,
                                     "spearman": spearman([r["retained_statistic"] for r in rows],
                                                          [r["block_displacement"] for r in rows])})
                action = sequence[d]
                accounting = site_transport_accounting(action, d, model)
                h = model.step(h, h0, graph, action, kept[safe[d]], site=d)
                kept[d + 1] = h
                mass, hops = mass + accounting["mass"], hops + accounting["hops"]
    if any(sha256_state_dict(model) != frozen[seed] for seed, model in models.items()):
        raise RuntimeError("a frozen predictor changed during the audit")
    parent.verify("explore")
    baselines = sorted({b for b, _ in suprema})
    densities = [{"baseline": b, "action": a, "lambda0": lam0,
                  "tightness_ratio": repr(model_density_tightness(b, a, lam0, suprema[(b, a)]))}
                 for b in baselines for a in TRANSPORT_ACTIONS for lam0 in plan["lambda0"]]
    plant = next(iter(models.values()))
    norm = (float(torch.linalg.matrix_norm(plant.shared.weight.double(), ord=2))
            if hasattr(plant, "shared") else None)
    fields = sorted({k for row in states for k in row})
    write_csv(run.raw("states.csv"), [{k: _text(row.get(k)) for k in fields} for row in states], fields)
    write_csv(run.raw("rankings.csv"), [{k: _text(v) for k, v in row.items()} for row in rankings])
    write_csv(run.raw("model_densities.csv"), densities)
    write_csv(run.raw("replay.csv"), replay)
    write_json(run.raw("operators.json"), {"graphs": operators, "exact_max_nodes": plan["exact_max_nodes"],
                                           "hypotheses": "undirected nonnegative weights (symmetric S), positive degrees "
                                                         "(isolated vertices carry a self-loop), polynomial transports"})
    write_json(run.raw("constants.json"), {
        "window_approximation_errors": errors, "frame_min": frame_min, "frame_max": frame_max,
        "window_coefficients": coefficients.tolist(), "band_suprema": {f"{b}:A{a}": v for (b, a), v in suprema.items()},
        "shared_weight_spectral_norm": norm, "activation": config["model"].get("activation", "relu"),
        "activation_lipschitz": activation_lipschitz(config["model"].get("activation", "relu")),
        "retention": {f"A{a}": ACTIONS[a].alpha for a in range(8)},
        "native_update_constant_by_depth": {str(d): v for d, v in sorted(gains.items())},
        "native_update_constant_definition": "largest block over transport displacement at the depth among actions "
                                             "with the nominal retention (all actions at designated sites)",
        "backbone": getattr(plant, "backbone", None)})
    summary = summarize_band_audit([run.root])
    summary["truncated"] = False
    return run.finish(summary)


def _text(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return value


def _values(rows, name):
    return [float(r[name]) for r in rows if r.get(name, "") not in ("", None)]


def summarize_band_audit(dirs):
    """Recompute E18 values from the sealed per-state tables (no PyTorch)."""
    pooled, rows, notes = {}, [], {}
    for d in dirs:
        d = Path(d)
        exp = read_json(d / "experiment.json")
        dataset = exp["parent"]["dataset"]
        entry = pooled.setdefault(dataset, {"states": [], "rankings": [], "densities": [], "replay": [], "constants": []})
        entry["states"] += read_csv(d / "raw/states.csv")
        entry["rankings"] += read_csv(d / "raw/rankings.csv")
        entry["densities"] += read_csv(d / "raw/model_densities.csv")
        entry["replay"] += read_csv(d / "raw/replay.csv")
        entry["constants"].append(read_json(d / "raw/constants.json"))
    for dataset, entry in sorted(pooled.items()):
        densities = {}
        for row in entry["densities"]:
            densities.setdefault((row["baseline"], row["action"], row["lambda0"]), []).append(float(row["tightness_ratio"]))
        for (baseline, action, lam0), values in sorted(densities.items()):
            if max(values) - min(values) > 1e-12:
                raise ExperimentError("model-density tightness differs between pooled runs")
            keys = {"dataset": dataset, "baseline": baseline, "action": f"A{action}",
                    "spectrum": f"lambda0={float(lam0):g}"}
            rows.append(summary_row(keys, "tightness_ratio", values[0], "ratio", DEFINITIONS["tightness_ratio"]))
        groups = {}
        for row in entry["states"]:
            groups.setdefault((row["baseline"], row["action"]), []).append(row)
        for (baseline, action), group in sorted(groups.items()):
            keys = {"dataset": dataset, "baseline": baseline, "action": f"A{action}", "spectrum": "explore_states"}
            exact = [r for r in group if r.get("tightness_ratio", "") != ""]
            ratio = lambda num, den: [float(r[num]) / float(r[den]) for r in group
                                      if r.get(num, "") != "" and r.get(den, "") not in ("", None) and float(r[den]) > 0]
            measured = {
                "window_bound_ratio": median(ratio("transport_displacement", "window_bound")),
                "retained_statistic_ratio": median(ratio("symmetric_displacement", "retained_statistic")),
                "approximation_error_share": median(_values(group, "approximation_error_share")),
                "degree_factor": median(_values(group, "degree_factor")),
                "update_gain_p95": quantile(_values(group, "update_gain"), .95),
                "logged_statistic_ratio": median(ratio("retained_statistic", "logged_statistic")),
                "audited_states": len(group), "exact_spectrum_states": len(exact)}
            if exact:
                measured.update(tightness_ratio=median(_values(exact, "tightness_ratio")),
                                tightness_ratio_max=max(_values(exact, "tightness_ratio")),
                                omitted_band_share=median(_values(exact, "omitted_band_share")))
            for name, value in measured.items():
                if value is not None:
                    unit = "pairs" if name.endswith("_states") else ("fraction" if name.endswith("share") else "ratio")
                    rows.append(summary_row(keys, name, value, unit, DEFINITIONS[name]))
        keys = {"dataset": dataset, "baseline": "all", "action": "all", "spectrum": "explore_states"}
        spearman_values = _values(entry["rankings"], "spearman")
        constants = entry["constants"]
        overall = {"ranking_spearman": median(spearman_values),
                   "replay_max_abs_deviation": max(_values(entry["replay"], "max_abs_deviation")),
                   "update_lipschitz_max": max((v for c in constants for v in c["native_update_constant_by_depth"].values()),
                                               default=None),
                   "activation_lipschitz": constants[0]["activation_lipschitz"]}
        norms = {c["shared_weight_spectral_norm"] for c in constants}
        if None not in norms and len(norms) == 1:
            overall["shared_weight_spectral_norm"] = norms.pop()
        for name, value in overall.items():
            if value is not None:
                rows.append(summary_row(keys, name, value, "correlation" if name == "ranking_spearman" else
                                        ("measurement units" if name.startswith("replay") else "ratio"),
                                        DEFINITIONS[name]))
        notes[dataset] = {"states": len(entry["states"]), "window_approximation_errors": constants[0]["window_approximation_errors"],
                          "falsified": any(v > 1 + 1e-9 for v in _values(entry["states"], "tightness_ratio"))}
    return {"rows": rows, "notes": notes}


def export_records(results_dirs, **kwargs):
    protocol = ("E18: label-free replay of sealed D_explore trajectories on the frozen plant; transport and block "
                "displacements of A1-A7 against the next site's nominal transport; ideal six-band bound from the exact "
                "spectrum of L_sym where affordable, complete Chebyshev-window bound everywhere; model-density "
                "tightness at exp(-lambda/lambda0) (main Proposition 6, Fig. S18c)")
    return export_record_set("band_audit", results_dirs, summarize=summarize_band_audit,
                             raw_exports=RAW_EXPORTS, protocol=protocol, **kwargs)
