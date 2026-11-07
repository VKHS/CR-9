"""Variant lifecycles: module ablations (Supplementary Table S17; E7, E8), separately fitted factorial cells
(Tables S16, S16b) and the E18 policy value over matched confidence/history controls (Table S36).

Mechanism
---------
A *variant* is a declared change of what the fitted controller may know or do.  It is installed into the
released runtime for one process only (module attributes of ``cross_morse`` are replaced and restored; no
runtime file is modified), exactly as ``cm_experiments.backbones`` installs its substitute plant.  Its
specification (operations, resolved coordinate indices, donor-pool SHA-256 and the SHA-256 of this file)
is written into the variant configuration under ``ablation_variant``, so the run's configuration hash --
part of every stage manifest -- covers the variant, and the installer refuses a specification whose
recorded source differs from this file.

Every variant lifecycle starts from a fresh copy of a sealed parent: the parent's sealed ``prepare`` and
``train`` outputs (prepared data, roles, checkpoints, frozen-model baseline) are copied byte for byte and
sealed again under the variant configuration with the runtime's own ``write_stage_manifest`` (the stage
markers record the parent manifests they were copied from), then ``explore -> fit -> select -> confirm``
are executed by ``python -B -m cm_experiments.ablation_lifecycle <run_cross_morse.py arguments>`` --
``run_cross_morse.main`` with the variant installed.  The frozen plants are therefore the parent's own;
only exploration logging (scramble), the controller fit and the decisions can differ.  The evaluation
role the display needs is read afterwards, by this module, only when the parent stage that owns it is
sealed (``compare``: the parent's confirm; ``test``: its evaluate) and every arm is sealed.

Operations (process-local patches)
----------------------------------
``delete``          the named measurement coordinates are removed from the controller's information:
                    the observer (``learning.ConditionedObserver``, substituted by ``AblationObserver``)
                    holds them at their D_explore fitting median, i.e. at zero in its normalized
                    coordinates, for fitting, filtering, prediction and reconstruction; deleted transport
                    history is also zeroed in the critic's history features (``learning._features``).
                    The measurement bank still computes every coordinate (matched work) and the vector keeps
                    its dimension (matched capacity).  The registered support strata, calibration cells
                    and the A9 anchor rule keep reading the measured entropy (registered conventions, not
                    learned information).
``scramble``        the named coordinates of every measurement are replaced (``sensor_bank`` wrapped in
                    ``exploration`` and ``policies``) by those of a donor trajectory of the parent's sealed
                    D_explore records at the same depth: a permutation across trajectory units with a
                    recorded seed.  The donor of an exploration trajectory is a function of (seed,
                    trajectory index); of any other execution, of (seed, source-state checksum of the input
                    and member).  The donor pool (a seeded subset of the parent's trajectories) is copied
                    into the run directory; its SHA-256 is part of the variant.  Bank work is unchanged.
``observer``        the observer class: ``raw`` (latest latent measurement) and ``ema`` (exponential
                    average, declared weight) updated at the registered observer cadence with a constant
                    innovation covariance (second moment of the model-fit innovations), no dynamics;
                    ``global`` (the conditioned observer's latent space with one global ridge transition
                    and its own residual Q, R = 0.25 Q + 1e-3 I); ``history_mlp`` (an MLP over a fixed window
                    of the last w latent measurement vectors; below); ``memoryless`` (the decision state
                    re-initialised from every measurement, the factorial O = 0).  The critic and the
                    consequence models are fitted on the variant's own features.  The runtime's conditioned
                    observer is the complete arm.  The Controller's charged work counts the conditioned
                    filter for every variant (``Controller.observe``), so the ledger is matched by construction.
``remove_reset``    a direct reset (A8 source reset or A9 last-safe reset) is never admissible in feedback
                    decisions (one-step, radius, MPC-RH planning) and the recovery fallback uses the
                    other reset; registered open-loop schedules and the MPC-OL plan are unchanged.
``action_bank``     feedback decisions (one-step, radius, MPC-RH planning) may not choose the transport actions
                    A1-A7 (``allowed`` = [0, 8, 9]; the factorial A = 0); the resets stay under ``risk_control``.
``risk_control``    risk-controlled execution off (the factorial X = 0): every registered (stratum, action) cell is
                    qualified, fixed at fit time in the sealed controller; the resets A8/A9 are never admissible in
                    feedback decisions or MPC-RH planning and every recovery fallback returns A0.

History-MLP observer (E8; ``{"op": "observer", "kind": "history_mlp", "window": w, "width": h}``)
--------------------------------------------------------------------------------------------------
The conditioned observer's latent space (median/MAD normalization and PCA basis of dimension d =
controller.state_dim), measurement-update rule and cadence (controller.observer_period, P), with the conditioned
ridge transition replaced by an MLP over a fixed window of the last w latent vectors:

    prior  z_{l|l-1} = z_{l-1} + f([z_{l-w}, ..., z_{l-1}], u_{l-1}, onehot(a_{l-1}))

where u = (normalized relative depth, transported mass, hops), the conditioned design's exogenous input, and f has
one hidden SiLU layer of width h.  Window entries are the measured latent vector at update depths (l % P = 0) and
the prior elsewhere (the measurement is not fused between updates); at depth 0 the window repeats the first
measurement.  At an update depth the measurement z~_l corrects the prior with the gain K = Q (Q + R)^-1; the
innovation e = z~_l - prior gives NIS = e' (Q + R)^-1 e, bias and whiteness as in the runtime; the posterior
covariance is (I - K) Q (I - K)' + K R K' (rank + diagonal); between updates the state is the prior with
covariance Q.  Fitting: the conditioned observer's objective, one-step latent prediction of z(y_{l+1}) on the same
model-fit rows, teacher-forced on measured windows exactly as the ridge transition is fitted on measured pairs,
trained with the runtime's recipe for its learned latent transition (cross_morse.learning._train_network: AdamW
with the registered learning rate, weight decay, batch, epochs and patience; standardized increment targets;
internal validation on 10% of the rows' trajectory groups; seed = fold seed + 68, after the four networks of
``_fit_one``); Q = residual covariance of the fitted f on the model-fit rows (rank + diagonal, + 1e-4 I) and
R = 0.25 Q + 1e-3 I (the runtime's identification convention).  Defaults: w = 2 P (two registered observer-update
periods: 4), h = 128 (the registered width of the runtime's learned consequence models, Supplementary Table S9).
At the registered d = 16 the MLP has (4*16 + 13)*128 + 128 + 128*16 + 16 = 12,048 parameters against at most
41*(16 + 4)*16 = 13,120 for the conditioned observer's ridge transitions (one global and one per fitted depth-bin
and action cell); both counts are recorded in the fitted payload (``ablation_fit.history_fit``).

Studies
-------
``modules``  (S17) per parent run: one lifecycle per (module, arm) of spectral_bands {deleted, scrambled},
             source_alignment {no_alignment}, confidence_sensors {no_confidence}, transport_history
             {no_history}, observer {raw, ema, global, history_mlp} and recovery_type {source_only,
             last_safe_only}, plus one replicate lifecycle with no operation (the complete arm;
             ``complete="parent"`` uses the parent instead).  The history-MLP arm is exported as an extra observer
             arm (records/specs.json registers raw, EMA, global and conditioned; the S17 generator prints those).
             The comparator branch (branch on/off) is a serving-time switch outside the lifecycle: it is read from
             sealed ``input_gate`` and ``negative_controls`` runs of the parent.
``e18``      one lifecycle deleting every spectral and reachability coordinate (and the spectral age), the
             matched confidence/history control, against the complete arm.
``factorial`` (S16, S16b) one lifecycle per distinct cell variant (A, O, X on arxiv; O, X on PascalVOC-SP,
             where S selects the registered Full or Lean profile of the same fitted lifecycle): O = 0 the
             memoryless observer, A = 0 ``action_bank``, X = 0 ``risk_control`` (cm_experiments.factorial.
             cell_operations); the cells, with the read-only observer's stopping rule of the O = 1 cells, are
             executed and exported by ``cm_experiments.factorial`` in its ``fitted`` mode.

Measurement (modules and e18; ``measure`` jobs)
-----------------------------------------------
The audited rule c* is q_M when it is a feedback route, else the provisional selection q_s, at the Full
profile; every arm executes c* (its members, the parent's checkpoints) with the arm's own sealed controller
on every input of the held-out role, the arm's variant installed.  Per arm:

``unit loss``        unit-weighted CE/BCE of the ensemble prediction (``score_observations`` units/weights);
``false-safe rate``  executed decisions predicted safe (critic mean + registered calibrated radius <= 0)
                     whose realized one-action consequence Delta = l_{d+h}(a -> nominal) - l_{d+h}(nominal)
                     (h = exploration.critic_horizon, the critic's own target, by exact replay of the executed
                     trajectory on the input's role rows) exceeds 1e-7 nats, over executed decisions with a
                     finite prediction;
``warning lead``     on the nominal trajectory of every (input, member): onset d* = first minimizer over
                     d = 0..L of the role-row loss (no onset when d* = L); first warning w = first controlled
                     depth at which the arm's observer state is valid and its terminal model predicts a positive
                     remaining nominal loss change (w = L without a warning); lead = d* - w, averaged over the
                     pairs with an onset (not measured when an arm's controller has no fitted terminal model);
``action entropy``   entropy (nats) of the mix of executed actions over all decisions;
``risk RMSE``        held-out RMSE of the critic on the sealed assessment partition of D_explore
                     (fit/<profile>/calibration_assessment.npz; the same rows in every arm).

Contrasts (ablated minus complete, paired on the same units, inputs and members): ``loss_change_nats`` (mean
paired unit difference with the registered paired whole-unit bootstrap), ``false_safe_rate_change``,
``warning_lead_change_depths``, ``action_entropy_change``, ``risk_rmse_change``; ``innovation_rmse`` of an
observer arm is its risk RMSE (records/specs.json: reliability_diagnostics notes that the observer risk
RMSE is the measurement module_ablations calls innovation_rmse; the latent innovation RMS is kept in the
raw arm record).  Several split runs of a dataset are averaged; with several datasets a ``pooled`` row holds
the equal-weight mean over datasets (innovation_rmse: square root of the mean of the squares).

A numerical policy failure of any arm stops the job; nothing is filled in.
"""
from __future__ import annotations

import contextlib
import contextvars
import copy
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from .artifacts import canonical, sha256
from .ensemble_grid import (ROOT, STAGES, RunnerError, SealedRun, candidate_identity, check_identity, check_role,
                            create_plan, environment_record, json_safe, load_plan, member_runs, paired_effect,
                            public_plan, read_json, runtime_source_sha256, selected_jobs, slug, utc_now,
                            write_record_set)

MODULE_FILE = "cm_experiments/ablation_lifecycle.py"
SCRIPT_FILE = "scripts/run_ablation.py"
VARIANT_KEY = "ablation_variant"
VARIANT_SCHEMA = "cm-ablation-variant-1"
RESULTS_SCHEMA = "cm-ablation-module-results-1"
STUDY_KIND = "ablation-lifecycles"
STUDIES = {"modules": "module_ablations", "e18": "spectral_policy_value", "factorial": "factorials"}
FEEDBACK_ROUTES = ("one_step", "radius", "mpc_rh")
HISTORY_COORDINATES = ("relative_depth", "normalized_mass", "normalized_hops")
POOL_FILE = "ablation_pool.npz"
IDENTITY_TOLERANCE = 1e-7          # the runtime's nominal-relative identity tolerance (fit_controller)
LIFECYCLE_MODULES = {
    "spectral_bands": ("deleted", "scrambled"),
    "source_alignment": ("no_alignment",),
    "confidence_sensors": ("no_confidence",),
    "transport_history": ("no_history",),
    "observer": ("raw", "ema", "global", "history_mlp"),
    "recovery_type": ("source_only", "last_safe_only"),
}
OBSERVER_KINDS = ("raw", "ema", "global", "history_mlp", "memoryless")
NO_TRANSPORT = [0, 8, 9]           # the factorial A = 0: A0 and the resets (the resets stay under risk_control)
RESET_ACTIONS = (8, 9)
HISTORY_WIDTH = 128                # registered width of the runtime's learned consequence models (Table S9)
HISTORY_SEED_OFFSET = 68           # 17 * 4: after the seeds _fit_one gives its four networks (seed + 17 k)
FINAL_FIT_SEED = 7919              # cross_morse.learning.fit_controller's final fit
COMPLETE_ARMS = {"observer": "conditioned", "recovery_type": "both"}
POSTHOC_MODULES = {"comparator_branch": ("branch_off",)}
E18_MODULE, E18_ARM = "spectral_coordinates", "deleted"
DEFAULT_SCRAMBLE_SEED = 20261107
QUANTITY_UNITS = {"loss_change_nats": "nats", "warning_lead_change_depths": "depths",
                  "false_safe_rate_change": "fraction", "risk_rmse_change": "nats", "action_entropy_change": "nats",
                  "innovation_rmse": "nats", "clean_score_change": "points", "worst_harm_change_nats": "nats"}
DEFINITIONS = {
    "loss_change_nats": "mean over held-out units of the paired unit loss of the audited rule executed with the "
                        "ablated arm's controller minus with the complete arm's controller (unit-weighted CE/BCE)",
    "warning_lead_change_depths": "mean over (input, member) pairs with a nominal deterioration onset of the "
                                  "ablated minus the complete warning lead (onset minus first terminal-model "
                                  "warning on the nominal trajectory, controlled depths)",
    "false_safe_rate_change": "false-safe rate of the ablated arm minus the complete arm (executed decisions "
                              "predicted safe, mean + calibrated radius <= 0, whose realized one-action "
                              "consequence exceeds 1e-7 nats, over executed decisions with a finite prediction)",
    "risk_rmse_change": "held-out RMSE of the one-step risk model (critic) on the sealed D_explore assessment "
                        "partition, ablated minus complete (nats)",
    "action_entropy_change": "entropy (nats) of the executed action mix of the ablated arm minus the complete arm",
    "innovation_rmse": "held-out RMSE of the one-step risk prediction made through the observer variant (the "
                       "variant's sealed assessment partition; records/specs.json reliability_diagnostics)",
    "clean_score_change": "official score (points) of the evaluation role served without the comparator branch "
                          "(q_M on every input; q_C only after a numerical failure) minus with it (g_input gate)",
    "worst_harm_change_nats": "largest per-unit paired harm (served minus q_C loss) without the branch minus "
                              "with it, over the clean evaluation units and the negative-control constructions",
}
E18_CONTRACT = {
    "title": "E18 policy value of the spectral coordinates over matched confidence/history controls",
    "manuscript_tables": ["Supplementary Table S36"],
    "experiments": ["E18"],
    "keys": ["dataset"],
    "quantities": {
        "policy_value_nats": "paired held-out loss of the audited rule under the matched control (controller fitted "
                             "and executed without the 16 spectral and reachability coordinates and the spectral "
                             "age, at matched sensing work) minus under the complete controller (nats; positive "
                             "means the spectral coordinates add policy value)",
        "interval_low_nats": "lower end of the central 95% interval of that value (paired whole-unit bootstrap; "
                             "split runs: small-sample t over the per-split values) (nats)",
        "interval_high_nats": "upper end of that interval (nats)",
        "risk_rmse_change": "held-out RMSE of the one-step risk model, control minus complete (nats)",
        "units": "held-out evaluation units of the contrast (units)",
    },
    "table": "list",
    "notes": "Produced by python -B scripts/run_ablation.py e18-policy plan|run|export "
             "(cm_experiments/ablation_lifecycle.py).",
}
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


# ============================================================================= files and identities
def installer_identity() -> dict:
    return {"module": "cm_experiments.ablation_lifecycle", "source_sha256": sha256(Path(__file__).resolve())}


def code_identity() -> dict:
    names = (MODULE_FILE, SCRIPT_FILE, "cm_experiments/sweeps.py", "cm_experiments/factorial.py",
             "cm_experiments/ensemble_grid.py")
    return {name: sha256(ROOT / name) for name in names if (ROOT / name).is_file()}


def _dump(path, value) -> str:
    """Atomic JSON write that replaces an interrupted attempt; returns the file's SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        Path(temporary).write_text(json.dumps(json_safe(value), indent=1, sort_keys=True, allow_nan=False) + "\n",
                                   encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return sha256(path)


def _save_npz(path, **arrays) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    os.close(fd)
    try:
        np.savez_compressed(temporary, **{k: np.asarray(v) for k, v in arrays.items()})
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return sha256(path)


def _resolved(path) -> dict:
    from .backbones import resolved
    return resolved(Path(path))


# ============================================================================= measurement coordinates
def measurement_names(config: dict) -> list[str]:
    """``cross_morse.core.MeasurementBank.names`` for a configuration (no PyTorch)."""
    bands = [int(b) for b in config["sensor"]["bands"]]
    names = ["entropy", "margin", "source_js", "effective_rank_entropy", "log1p_dirichlet", "relative_depth",
             "normalized_mass", "normalized_hops", "log_state_energy"]
    names += [f"band_power_{b}" for b in bands] + [f"source_alignment_{b}" for b in bands]
    names += [f"degree_weighted_power_{b}" for b in bands] + [f"reachability_A{i}" for i in range(1, 8)]
    if int(config["sensor"].get("sensing_period", 1)) > 1:
        names.append("spectral_age_fraction")
    return names


def module_coordinates(module: str, config: dict) -> list[str]:
    """The measurement coordinates a sensor module consists of (Supplementary Table S8)."""
    bands = [int(b) for b in config["sensor"]["bands"]]
    names = measurement_names(config)
    band_family = ([f"band_power_{b}" for b in bands] + [f"degree_weighted_power_{b}" for b in bands]
                   + [f"reachability_A{i}" for i in range(1, 8)])
    alignment = [f"source_alignment_{b}" for b in bands]
    table = {"spectral_bands": band_family, "source_alignment": alignment,
             "confidence_sensors": ["entropy", "margin"], "transport_history": ["normalized_mass", "normalized_hops"],
             E18_MODULE: band_family + alignment + [n for n in names if n == "spectral_age_fraction"]}
    if module not in table:
        raise RunnerError(f"{module} is not a sensor module")
    return [n for n in names if n in table[module]]


def coordinate_indices(names, config) -> list[int]:
    schema = measurement_names(config)
    unknown = [n for n in names if n not in schema]
    if unknown:
        raise RunnerError(f"coordinates outside the measurement schema: {unknown}")
    return [schema.index(n) for n in names]


# ============================================================================= variant specifications
def history_window(config: dict) -> int:
    """The history-MLP observer's default window: two registered observer-update periods (2 P)."""
    return 2 * int(config.get("controller", {}).get("observer_period", 2))


def history_parameters(window: int, width: int, dimension: int) -> int:
    """Parameters of the history MLP: window * d latent inputs, 3 exogenous, 10 action indicators; d outputs."""
    inputs = int(window) * int(dimension) + 3 + 10
    return inputs * int(width) + int(width) + int(width) * int(dimension) + int(dimension)


def conditioned_parameters(payload: dict) -> int:
    """Parameters of the conditioned observer's ridge transitions: one global and one per fitted cell, (d + 4) x d."""
    dimension = len(payload["basis"][0])
    return (1 + len(payload.get("conditioned") or {})) * (dimension + 4) * dimension


def module_operations(module: str, arm: str, config: dict, *, ema_weight: float = .5, pool=None,
                      scramble_seed: int = DEFAULT_SCRAMBLE_SEED, history_window_length=None,
                      history_width: int = HISTORY_WIDTH) -> list[dict]:
    if module in ("spectral_bands", "source_alignment", "confidence_sensors", "transport_history", E18_MODULE):
        names = module_coordinates(module, config)
        if arm == "scrambled":
            if pool is None:
                raise RunnerError("a scramble needs its donor pool")
            return [{"op": "scramble", "coordinates": names, "indices": coordinate_indices(names, config),
                     "seed": int(scramble_seed), "pool": pool}]
        return [{"op": "delete", "coordinates": names, "indices": coordinate_indices(names, config)}]
    if module == "observer":
        if arm not in LIFECYCLE_MODULES["observer"]:
            raise RunnerError(f"observer arm {arm!r} is not a variant (conditioned is the complete arm)")
        op = {"op": "observer", "kind": arm}
        if arm == "ema":
            if not 0 < float(ema_weight) < 1:
                raise RunnerError("the EMA weight must lie in (0, 1)")
            op["ema_weight"] = float(ema_weight)
        if arm == "history_mlp":
            window = history_window(config) if history_window_length is None else history_window_length
            if type(window) is not int or type(history_width) is not int or window < 1 or history_width < 1:
                raise RunnerError("the history-MLP window and width are positive integers")
            op.update(window=window, width=history_width)
        return [op]
    if module == "recovery_type":
        return [{"op": "remove_reset", "action": {"source_only": 9, "last_safe_only": 8}[arm]}]
    raise RunnerError(f"{module}/{arm} is not a lifecycle variant")


def factorial_operations(cell: str, factors) -> list[dict]:
    """A factorial cell's fitted components (cm_experiments.factorial.cell_operations of the cell's switches):
    O = 0 the memoryless observer, A = 0 no transport actions, X = 0 risk-controlled execution off."""
    from .factorial import cell_operations, cell_switches
    return cell_operations(cell_switches(cell, factors, "full"))


def variant_spec(variant_id: str, operations: list[dict], *, study: str, module=None, arm=None) -> dict:
    return {"schema": VARIANT_SCHEMA, "id": variant_id, "study": study, "module": module, "arm": arm,
            "operations": copy.deepcopy(operations), "installer": installer_identity()}


def validate_spec(spec: dict, config: dict | None = None) -> None:
    if not isinstance(spec, dict) or spec.get("schema") != VARIANT_SCHEMA:
        raise RunnerError("not an ablation variant specification")
    if spec.get("installer") != installer_identity():
        raise RunnerError("the variant names another installer source (cm_experiments/ablation_lifecycle.py changed "
                          "since planning); plan the variant again")
    kinds = [op.get("op") for op in spec["operations"]]
    if len(set(kinds)) != len(kinds):
        raise RunnerError("one operation of each kind per variant")
    for op in spec["operations"]:
        kind = op.get("op")
        if kind in ("delete", "scramble"):
            if config is not None and coordinate_indices(op["coordinates"], config) != list(op["indices"]):
                raise RunnerError("the variant's coordinate indices differ from the configuration's measurement schema")
            if "relative_depth" in op["coordinates"]:
                raise RunnerError("relative depth is part of the observer's exogenous input and is never ablated")
            if kind == "scramble" and not {"file", "sha256", "units", "depths"} <= set(op.get("pool") or {}):
                raise RunnerError("a scramble names its donor pool file, SHA-256 and shape")
        elif kind == "observer":
            if op.get("kind") not in OBSERVER_KINDS:
                raise RunnerError(f"unknown observer variant {op.get('kind')!r}")
            if op["kind"] == "history_mlp" and not all(type(op.get(k)) is int and op[k] >= 1
                                                       for k in ("window", "width")):
                raise RunnerError("a history-MLP observer declares a positive integer window and width")
        elif kind == "remove_reset":
            if op.get("action") not in (8, 9):
                raise RunnerError("only the direct resets A8 and A9 can be removed")
        elif kind == "action_bank":
            if op.get("allowed") != NO_TRANSPORT:
                raise RunnerError(f"the declared action-bank restriction removes the transport actions A1-A7 "
                                  f"(allowed {NO_TRANSPORT})")
        elif kind == "risk_control":
            if op.get("mode") != "off":
                raise RunnerError("the declared risk-control operation is 'off'")
        else:
            raise RunnerError(f"unknown variant operation {kind!r}")
    if {"remove_reset", "risk_control"} <= set(kinds):
        raise RunnerError("risk_control off already removes both resets")


def variant_config(parent_config: dict, spec: dict) -> dict:
    if VARIANT_KEY in parent_config:
        raise RunnerError("the parent is itself a variant lifecycle; plan from the sealed parent run")
    validate_spec(spec, parent_config)
    config = copy.deepcopy(parent_config)
    config[VARIANT_KEY] = copy.deepcopy(spec)
    return config


def observer_ablation(spec: dict | None) -> dict:
    """The spec-level fields an AblationObserver payload must carry under the installed variant."""
    ops = (spec or {}).get("operations", [])
    deleted = sorted(n for op in ops if op["op"] == "delete" for n in op["coordinates"])
    observer = next((op for op in ops if op["op"] == "observer"), {})
    result = {"variant": (spec or {}).get("id"), "deleted": deleted, "observer": observer.get("kind", "conditioned")}
    if "ema_weight" in observer:
        result["ema_weight"] = float(observer["ema_weight"])
    if observer.get("kind") == "history_mlp":
        result["history"] = {"window": int(observer["window"]), "width": int(observer["width"])}
    return result


def _deleted_indices(spec) -> list[int]:
    return sorted(i for op in (spec or {}).get("operations", []) if op["op"] == "delete" for i in op["indices"])


# ============================================================================= donor pool of the scramble (NumPy)
def scramble_pool(parent_root, names, indices, config, path, *, seed: int, units: int) -> dict:
    """Coordinates ``indices`` of a seeded subset of the parent's sealed D_explore trajectories, at depths 0..L."""
    explore = Path(parent_root) / "explore"
    files = sorted(explore.glob("trajectory_*.npz"))
    if not files:
        raise RunnerError(f"{Path(parent_root).name}: the sealed exploration holds no trajectory")
    depth = int(config["model"]["depth"])
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 101]))
    chosen = sorted(rng.choice(len(files), size=min(int(units), len(files)), replace=False).tolist())
    blocks = np.empty((len(chosen), depth + 1, len(indices)), dtype=np.float32)
    for k, number in enumerate(chosen):
        with np.load(files[number], allow_pickle=False) as z:
            order = np.argsort(z["depth"], kind="stable")
            depths, y, y_next = z["depth"][order], z["y"][order], z["y_next"][order]
        if not np.array_equal(depths, np.arange(depth)):
            raise RunnerError(f"{files[number].name}: a donor trajectory needs every depth (record_stride 1)")
        blocks[k, :depth] = y[:, indices]
        blocks[k, depth] = y_next[-1, indices]
    if not np.isfinite(blocks).all():
        raise RunnerError("the donor pool holds nonfinite measurements")
    digest = _save_npz(path, block=blocks, coordinates=np.asarray(names), indices=np.asarray(indices),
                       trajectory_files=np.asarray([files[n].name for n in chosen]))
    manifest = Path(parent_root) / "provenance" / "stages" / "02_explore.json"
    return {"file": POOL_FILE, "sha256": digest, "units": len(chosen), "depths": depth + 1,
            "source": {"run": Path(parent_root).name, "explore_manifest_sha256": sha256(manifest)
                       if manifest.is_file() else None, "subset_seed": [int(seed), 101],
                       "available_trajectories": len(files)}}


def donor_index(seed: int, key, units: int) -> int:
    digest = hashlib.sha256(json.dumps([int(seed), *[str(k) for k in key]]).encode()).digest()
    return int.from_bytes(digest[:8], "big") % int(units)


# ============================================================================= audited rule (NumPy)
def profile_counterpart(candidates, candidate, profile):
    from .factorial import profile_counterpart as counterpart
    return counterpart(candidates, candidate, profile)


def audited_candidate(run: SealedRun, *, allow_fixture: bool) -> tuple[dict, str]:
    """q_M when it is a feedback route, else the provisional selection q_s; at the Full profile."""
    candidates = run.candidates()
    confirmed = run.confirmed()
    decision = run.json("select/decision.json")
    provisional = dict(candidates[decision["provisional_index"]], registry_index=decision["provisional_index"])
    if confirmed["route"] in FEEDBACK_ROUTES:
        chosen, origin = confirmed, "confirmed policy q_M"
    elif provisional["route"] in FEEDBACK_ROUTES:
        chosen, origin = provisional, "provisional selection q_s (q_M is the open-loop comparator)"
    elif allow_fixture and run.synthetic:
        comparator = run.comparator()
        found = [dict(c, registry_index=i) for i, c in enumerate(candidates) if c["route"] == "radius"
                 and c["ensemble"] == comparator["ensemble"] and c.get("profile") == "full"]
        if not found:
            raise RunnerError(f"{run.root.name}: the fixture registers no radius candidate at q_C's ensemble")
        chosen, origin = found[0], "registered radius candidate (synthetic fixture)"
    else:
        raise RunnerError(f"{run.root.name}: neither q_M nor q_s is a feedback route; no controller decision to ablate")
    if chosen.get("profile") != "full":
        chosen = profile_counterpart(candidates, chosen, "full")
        origin += "; Full-profile counterpart"
    return chosen, origin


def resolve_candidate(run: SealedRun, identity: dict) -> dict:
    candidate = dict(run.candidates()[identity["registry_index"]], registry_index=identity["registry_index"])
    if candidate_identity(candidate) != identity:
        raise RunnerError(f"{run.root.name}: the audited rule differs from the sealed registry")
    return candidate


# ============================================================================= planning (torch-free)
def _parent_members(parents, *, role, allow_fixture, allow_source_change):
    found, datasets = [], set()
    for parent in parents:
        runs = [SealedRun(m) for m in member_runs(parent)]
        dataset = runs[0].dataset
        if len({r.dataset for r in runs}) != 1 or dataset in datasets:
            raise RunnerError(f"{parent}: one parent (run or campaign) per dataset")
        datasets.add(dataset)
        for run in runs:
            if VARIANT_KEY in run.config:
                raise RunnerError(f"{run.root.name}: a variant lifecycle cannot be a parent")
            check_role(run, role)
            source = run.state.get("source_sha256")
            if source != runtime_source_sha256() and not allow_source_change:
                raise RunnerError(f"{run.root.name}: sealed by runtime source {source}, this checkout is "
                                  f"{runtime_source_sha256()}; a variant copies the parent's prepare/train outputs and "
                                  "must execute the same runtime (plan from the parent's source snapshot)")
            found.append((Path(parent).resolve(), dataset, run))
    synthetic = {run.synthetic for _, _, run in found}
    if len(synthetic) != 1:
        raise RunnerError("synthetic and benchmark parents cannot share one experiment")
    return found


def _member_slug(dataset, run) -> str:
    return slug(f"{dataset}__{run.root.name}")


def _check_output(output, parents) -> Path:
    """create_plan's location rules, applied before planning writes any file (donor pools)."""
    output = Path(output).expanduser().absolute()
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise RunnerError("the experiment directory must not traverse symbolic links")
    output = output.resolve()
    if output.is_relative_to(ROOT):
        raise RunnerError("write experiment outputs outside the repository checkout")
    for parent in parents:
        parent = Path(parent).resolve()
        if output == parent or output.is_relative_to(parent) or parent.is_relative_to(output):
            raise RunnerError("the experiment directory must be separate from every parent run")
    if (output / "plan.json").exists():
        raise RunnerError(f"{output}/plan.json exists; use a new experiment directory")
    return output


def _lifecycle_job(output, jobs, configs, run, dataset, spec, *, through, pool_file=None) -> dict:
    member = _member_slug(dataset, run)
    name = slug(spec["id"].replace(":", "__"))
    config = variant_config(run.config, spec)
    relative = f"configs/{member}/{name}.json"
    configs[relative] = config
    entry = {"job": len(jobs), "kind": "lifecycle", "slug": f"{member}/{name}", "dataset": dataset,
             "member": run.root.name, "parent": str(run.root), "parent_identity": run.identity(),
             "variant": spec, "config": str(Path(output) / relative), "config_sha256": canonical(config),
             "run": str(Path(output) / "runs" / member / name), "through": through,
             "pool_file": None if pool_file is None else str(pool_file)}
    jobs.append(entry)
    return entry


def _write_configs(output, configs):
    with tempfile.TemporaryDirectory() as probe:
        for relative, config in configs.items():
            path = Path(probe) / slug(relative)
            path.write_text(json.dumps(config), encoding="utf-8")
            if canonical(_resolved(path)) != canonical(config):
                raise RunnerError(f"{relative}: run_cross_morse.py would resolve a different configuration")
    for relative, config in configs.items():
        path = Path(output) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _module_selection(modules):
    """``["spectral_bands", "observer:raw", ...]`` -> {module: [arms]} of lifecycle variants."""
    chosen = {}
    for item in (modules or list(LIFECYCLE_MODULES)):
        module, _, arm = str(item).partition(":")
        if module in POSTHOC_MODULES:
            continue
        if module not in LIFECYCLE_MODULES:
            raise RunnerError(f"unknown module {module!r}; lifecycle modules are {sorted(LIFECYCLE_MODULES)} and the "
                              "comparator branch is read from --input-gate/--negative-controls runs")
        arms = [arm] if arm else list(LIFECYCLE_MODULES[module])
        for a in arms:
            if a not in LIFECYCLE_MODULES[module]:
                raise RunnerError(f"{module} has no variant arm {a!r}")
            chosen.setdefault(module, [])
            if a not in chosen[module]:
                chosen[module].append(a)
    return chosen


def _gate_runs(paths, kind):
    """Sealed input_gate / negative_controls result directories, keyed by parent configuration hash."""
    from .artifacts import verify
    found = {}
    for path in paths or ():
        path = Path(path).resolve()
        try:
            verify(path)
        except (ValueError, OSError) as exc:
            raise RunnerError(f"{path}: not a sealed experiment directory ({exc})") from None
        experiment = read_json(path / "experiment.json")
        if experiment.get("experiment") != kind:
            raise RunnerError(f"{path}: not a {kind} run")
        key = experiment["parent"]["config_sha256"]
        if key in found:
            raise RunnerError(f"two {kind} runs of one parent")
        found[key] = {"path": str(path), "manifest_sha256": sha256(path / "ARTIFACT_MANIFEST.json"),
                      "role": read_json(path / "plan.json").get("evaluation_role"),
                      "parent": experiment["parent"]["name"], "truncated": bool(read_json(path / "results.json")
                                                                                .get("truncated"))}
    return found


def _measure_job(jobs, run, dataset, candidate, origin, arms, *, role, reps, max_graphs, study) -> dict:
    entry = {"job": len(jobs), "kind": "measure", "study": study, "slug": f"{_member_slug(dataset, run)}/measure",
             "dataset": dataset, "member": run.root.name, "parent": str(run.root), "parent_identity": run.identity(),
             "candidate": candidate_identity(candidate), "candidate_origin": origin, "role": role,
             "arms": arms, "max_graphs": int(max_graphs),
             "bootstrap": {"reps": int(reps or run.config["admission"]["bootstrap_reps"]),
                           "seed": int(run.config["seed"]) + 23},
             "critic_horizon": int(run.config["exploration"]["critic_horizon"])}
    jobs.append(entry)
    return entry


def _arm(name, module, arm, lifecycle=None, *, parent=None) -> dict:
    if lifecycle is None:
        return {"name": name, "module": module, "arm": arm, "source": "parent", "run": str(parent.root),
                "variant": None, "config_sha256": canonical(parent.config), "through": None}
    return {"name": name, "module": module, "arm": arm, "source": "lifecycle", "lifecycle_job": lifecycle["job"],
            "run": lifecycle["run"], "variant": lifecycle["variant"], "config_sha256": lifecycle["config_sha256"],
            "through": lifecycle["through"]}


def plan_modules(parents, output, *, modules=None, role="compare", complete="replicate", through="confirm",
                 reps=None, max_graphs=0, ema_weight=.5, scramble_seed=DEFAULT_SCRAMBLE_SEED, scramble_units=1024,
                 input_gate=(), negative_controls=(), allow_fixture=False, allow_source_change=False,
                 study="modules", history_window_length=None, history_width=HISTORY_WIDTH) -> Path:
    """S17 (study ``modules``) or E18 (study ``e18``): variant lifecycles and their measurement jobs.

    ``history_window_length`` (default: two observer-update periods of each parent) and ``history_width``
    (default 128) declare the history-MLP observer arm."""
    if study not in ("modules", "e18"):
        raise RunnerError("plan_modules plans the modules and e18 studies")
    if complete not in ("replicate", "parent") or through not in ("fit", "select", "confirm"):
        raise RunnerError("complete is 'replicate' or 'parent'; through is fit, select or confirm")
    if role not in ("compare", "test"):
        raise RunnerError("the held-out role is compare or test; decision roles are never reused")
    if int(max_graphs) < 0 or int(scramble_units) < 1:
        raise RunnerError("max_graphs is nonnegative and the donor pool holds at least one trajectory")
    if history_window_length is not None and (type(history_window_length) is not int or history_window_length < 1):
        raise RunnerError("the history-MLP window is a positive integer")
    if type(history_width) is not int or history_width < 1:
        raise RunnerError("the history-MLP width is a positive integer")
    if not parents:
        raise RunnerError("give at least one sealed --parent run")
    selection = _module_selection(modules) if study == "modules" else {E18_MODULE: [E18_ARM]}
    members = _parent_members(parents, role=role, allow_fixture=allow_fixture,
                              allow_source_change=allow_source_change)
    output = _check_output(output, [run.root for _, _, run in members])
    gates, controls = _gate_runs(input_gate, "input_gate"), _gate_runs(negative_controls, "negative_controls")
    jobs, configs, measures, posthoc, pools = [], {}, [], [], {}
    for parent, dataset, run in members:
        candidate, origin = audited_candidate(run, allow_fixture=allow_fixture)
        member = _member_slug(dataset, run)
        arms = []
        if complete == "replicate":
            replicate = _lifecycle_job(output, jobs, configs, run, dataset,
                                       variant_spec("complete", [], study=study), through=through)
            arms.append(_arm("complete", None, "complete", replicate))
        else:
            arms.append(_arm("complete", None, "complete", parent=run))
        for module, module_arms in selection.items():
            for arm in module_arms:
                pool, pool_file = None, None
                if arm == "scrambled":
                    pool_file = output / "pools" / f"{member}__{slug(module)}.npz"
                    names = module_coordinates(module, run.config)
                    pool = scramble_pool(run.root, names, coordinate_indices(names, run.config), run.config,
                                         pool_file, seed=scramble_seed, units=scramble_units)
                    pools[str(pool_file)] = pool["sha256"]
                ops = module_operations(module, arm, run.config, ema_weight=ema_weight, pool=pool,
                                        scramble_seed=scramble_seed, history_window_length=history_window_length,
                                        history_width=history_width)
                entry = _lifecycle_job(output, jobs, configs, run, dataset,
                                       variant_spec(f"{module}:{arm}", ops, study=study, module=module, arm=arm),
                                       through=through, pool_file=pool_file)
                arms.append(_arm(f"{module}:{arm}", module, arm, entry))
        measures.append((run, dataset, candidate, origin, arms))
        key = canonical(run.config)
        if study == "modules" and (key in gates or key in controls):
            if key not in gates:
                raise RunnerError(f"{run.root.name}: the comparator branch needs the parent's input_gate run")
            posthoc.append({"dataset": dataset, "member": run.root.name, "input_gate": gates.pop(key),
                            "negative_controls": controls.pop(key, None)})
    if gates or controls:
        raise RunnerError("an --input-gate/--negative-controls run belongs to no planned parent")
    for run, dataset, candidate, origin, arms in measures:
        _measure_job(jobs, run, dataset, candidate, origin, arms, role=role, reps=reps, max_graphs=max_graphs,
                     study=study)
    plan_path = create_plan(output, {
        "kind": STUDY_KIND, "study": study, "record_set": STUDIES[study],
        "experiment": ("Supplementary Table S17 module ablations (E7, E8) by variant lifecycles" if study == "modules"
                       else "E18 policy value over matched confidence/history controls (Table S36)"),
        "role": role, "through": through, "complete": complete, "parent_runs": sorted({str(p) for p, _, _ in members}),
        "datasets": sorted({d for _, d, _ in members}), "synthetic": members[0][2].synthetic,
        "modules": {m: list(a) for m, a in selection.items()}, "posthoc": posthoc, "jobs": jobs,
        "settings": {"ema_weight": float(ema_weight), "scramble_seed": int(scramble_seed),
                     "scramble_units": int(scramble_units), "max_graphs": int(max_graphs),
                     "reps": None if reps is None else int(reps),
                     "history_mlp": {"window": history_window_length, "width": int(history_width),
                                     "window_rule": "2 x controller.observer_period of each parent" if
                                     history_window_length is None else "declared",
                                     "width_rule": "registered width of the runtime's learned consequence models "
                                                   "(Supplementary Table S9)" if history_width == HISTORY_WIDTH
                                     else "declared"}},
        "pools": pools, "installer": installer_identity(), "ablation_code": code_identity(),
        "definitions": DEFINITIONS,
        "protocol": ("variant lifecycles from byte copies of the parent's sealed prepare/train outputs, explore->"
                     f"{through} re-executed with the declared variant installed; the audited rule executed on the "
                     f"{role} role with every arm's sealed controller; paired contrasts ablated minus complete")})
    _write_configs(output, configs)
    return plan_path


def plan_factorial_fit(parents, output, *, role="test", complete="replicate", through="confirm", reps=None,
                       warmup=1, max_graphs=0, allow_fixture=False, allow_source_change=False) -> Path:
    """One lifecycle per distinct factorial cell variant, and the factorial plan in its fitted mode."""
    from .factorial import CELLS, DESIGNS, plan_factorial
    if complete not in ("replicate", "parent") or through not in ("fit", "select", "confirm"):
        raise RunnerError("complete is 'replicate' or 'parent'; through is fit, select or confirm")
    if not parents:
        raise RunnerError("give the sealed ogbn-arxiv and/or PascalVOC-SP parent run(s)")
    members = _parent_members(parents, role=role, allow_fixture=allow_fixture, allow_source_change=allow_source_change)
    output = _check_output(output, [run.root for _, _, run in members])
    jobs, configs, cells = [], {}, []
    for parent, dataset, run in members:
        if dataset not in DESIGNS or len(member_runs(parent)) != 1:
            raise RunnerError(f"{parent}: the factorials run on one ogbn-arxiv or PascalVOC-SP run each")
        factors = DESIGNS[dataset]["factors"]
        by_ops = {}
        for cell in CELLS:
            ops = factorial_operations(cell, factors)
            key = canonical(ops)
            if key not in by_ops:
                if not ops and complete == "parent":
                    by_ops[key] = None
                else:
                    name = "complete" if not ops else "factorial-" + "-".join(
                        f"{f}{b}" for f, b in zip(factors, cell) if f != "S")
                    by_ops[key] = _lifecycle_job(output, jobs, configs, run, dataset,
                                                 variant_spec(name, ops, study="factorial"), through=through)
            entry = by_ops[key]
            cells.append({"dataset": dataset, "member": run.root.name, "cell": cell,
                          "factors": dict(zip(factors, map(int, cell))), "operations": ops,
                          "source": "parent" if entry is None else "lifecycle",
                          "lifecycle_job": None if entry is None else entry["job"]})
    plan_path = create_plan(output, {
        "kind": STUDY_KIND, "study": "factorial", "record_set": "factorials",
        "experiment": "Equal-budget 2^3 factorials with separately fitted cells (Supplementary Tables S16, S16b)",
        "role": role, "through": through, "complete": complete, "parent_runs": sorted({str(p) for p, _, _ in members}),
        "datasets": sorted({d for _, d, _ in members}), "synthetic": members[0][2].synthetic, "jobs": jobs,
        "cells": cells, "installer": installer_identity(), "ablation_code": code_identity(),
        "factorial_plan": "factorial/plan.json",
        "protocol": ("one lifecycle per distinct cell variant (explore->" + through + " re-executed from byte copies "
                     "of the parent's prepare/train outputs, same exploration size, folds and epochs: equal fitting "
                     "budget); cells executed and exported by cm_experiments/factorial.py in its fitted mode")})
    _write_configs(output, configs)
    plan_factorial([p for p, _, _ in members], output / "factorial", role=role, reps=reps, warmup=warmup,
                   max_graphs=max_graphs, allow_fixture=allow_fixture, mode="fitted", fitted=output)
    return plan_path


def load_study(output, study=None) -> dict:
    plan = read_json(Path(output) / "plan.json")
    if plan.get("kind") != STUDY_KIND or (study is not None and plan.get("study") != study):
        raise RunnerError(f"{output}/plan.json is not an ablation {study or ''} plan")
    return load_plan(output, plan["record_set"])


def factorial_cells(output, member: str) -> dict:
    """{cell: fitted cell entry} of one parent run, for cm_experiments.factorial's fitted mode."""
    plan = load_study(output, "factorial")
    jobs = {j["job"]: j for j in plan["jobs"]}
    cells = {}
    for cell in plan["cells"]:
        if cell["member"] != member:
            continue
        if cell["source"] == "parent":
            cells[cell["cell"]] = {"source": "parent", "operations": [], "plan_sha256": sha256(Path(output) / "plan.json")}
            continue
        job = jobs[cell["lifecycle_job"]]
        cells[cell["cell"]] = {"source": "lifecycle", "operations": cell["operations"], "run": job["run"],
                               "variant": job["variant"], "config_sha256": job["config_sha256"],
                               "through": job["through"], "lifecycle_job": job["job"],
                               "plan_sha256": sha256(Path(output) / "plan.json")}
    if not cells:
        raise RunnerError(f"{output}: no fitted cell of parent {member}")
    return cells


# ============================================================================= the installer (PyTorch at call time)
_ACTIVE = {"spec": None, "run_dir": None, "originals": {}, "pool": None, "patches": []}
_ROUTE = contextvars.ContextVar("cm_ablation_route", default=None)
_OPEN_LOOP = contextvars.ContextVar("cm_ablation_open_loop", default=False)
_TRAJECTORY = contextvars.ContextVar("cm_ablation_trajectory", default=None)
_FITTING = contextvars.ContextVar("cm_ablation_fitting", default=False)
_FIT_CONTEXT = contextvars.ContextVar("cm_ablation_fit_context", default=None)
_OBSERVER_CLASS = None
_HISTORY_CLASS = None


def active_spec():
    return _ACTIVE["spec"]


def _ops(kind):
    return [op for op in (_ACTIVE["spec"] or {}).get("operations", []) if op["op"] == kind]


class Installation:
    def __init__(self, spec, run_dir):
        self.spec, self.run_dir = spec, run_dir

    def restore(self) -> None:
        if _ACTIVE["spec"] is not self.spec:
            return
        for owner, name, original in reversed(_ACTIVE["patches"]):
            setattr(owner, name, original)
        _ACTIVE.update(spec=None, run_dir=None, originals={}, pool=None, patches=[])


def install(spec: dict, *, run_dir=None) -> Installation:
    """Install a variant into the released runtime for this process (until ``restore``)."""
    validate_spec(spec)
    if _ACTIVE["spec"] is not None:
        raise RunnerError("another ablation variant is installed in this process")
    import cross_morse.exploration as exploration
    import cross_morse.learning as learning
    import cross_morse.policies as policies
    kinds = {op["op"] for op in spec["operations"]}
    patches = []
    if kinds & {"delete", "observer"}:
        observer = observer_class()
        patches += [(learning, "ConditionedObserver", observer), (learning, "_fit_observer", _fit_observer),
                    (learning, "_features", _features)]
        if any(op["op"] == "observer" and op["kind"] == "history_mlp" for op in spec["operations"]):
            patches.append((learning, "_fit_one", _fit_one))      # the fold's directory, device, seed for the MLP
    pool = None
    if "scramble" in kinds:
        op = next(o for o in spec["operations"] if o["op"] == "scramble")
        if run_dir is None:
            raise RunnerError("a scramble variant is installed for a run directory holding its donor pool")
        path = Path(run_dir) / op["pool"]["file"]
        if not path.is_file() or sha256(path) != op["pool"]["sha256"]:
            raise RunnerError(f"{path}: the donor pool is missing or differs from the variant")
        with np.load(path, allow_pickle=False) as z:
            pool = np.asarray(z["block"], dtype=np.float32)
        if pool.shape != (op["pool"]["units"], op["pool"]["depths"], len(op["indices"])):
            raise RunnerError("the donor pool's shape differs from the variant")
        patches += [(exploration, "sensor_bank", _scrambled_sensor_bank),
                    (policies, "sensor_bank", _scrambled_sensor_bank),
                    (exploration, "collect_trajectory", _collect_trajectory)]
    if kinds & {"remove_reset", "action_bank", "risk_control"}:
        patches += [(learning.Controller, "supported", _supported), (policies, "recovery_action", _recovery_action),
                    (policies, "beam_plan", _beam_plan), (policies, "_execute_policy", _execute_policy)]
    if "risk_control" in kinds:
        patches.append((learning, "_support", _support))
    originals = {}
    applied = []
    for owner, name, new in patches:
        current = owner.__dict__[name] if isinstance(owner, type) else getattr(owner, name)
        originals[f"{getattr(owner, '__name__', owner)}.{name}"] = current
        applied.append((owner, name, current))
    _ACTIVE.update(spec=spec, run_dir=None if run_dir is None else Path(run_dir), originals=originals, pool=pool,
                   patches=applied)
    for owner, name, new in patches:
        setattr(owner, name, new)
    return Installation(spec, run_dir)


@contextlib.contextmanager
def installed(spec, *, run_dir=None):
    """``install`` for a block; no-op for ``spec=None`` (the parent's own runtime)."""
    if spec is None:
        yield None
        return
    handle = install(spec, run_dir=run_dir)
    try:
        yield handle
    finally:
        handle.restore()


def _original(owner_name, name):
    return _ACTIVE["originals"][f"{owner_name}.{name}"]


# ----------------------------------------------------------------------------- history-MLP observer (E8)
def history_inputs(windows, exogenous, actions, dtype=np.float64) -> np.ndarray:
    """Rows [window (oldest first, flattened), u, onehot(action)] of the history MLP's input."""
    windows = np.asarray(windows)
    actions = np.asarray(actions, dtype=np.int64).reshape(-1)
    onehot = np.zeros((len(actions), 10), dtype=dtype)
    known = (actions >= 0) & (actions < 10)
    onehot[np.flatnonzero(known), actions[known]] = 1
    return np.column_stack((windows.reshape(len(windows), -1).astype(dtype, copy=False),
                            np.asarray(exogenous).astype(dtype, copy=False), onehot))


def history_windows(latent, groups, depths, window: int, dtype=np.float64) -> np.ndarray:
    """[rows, window, d] measured latent windows ending at every row's depth along its own trajectory (the rows of a
    trajectory hold consecutive depths; the first measurement repeats before the trajectory's first depth)."""
    latent = np.asarray(latent)
    groups, depths, window = np.asarray(groups), np.asarray(depths, dtype=np.int64), int(window)
    out = np.empty((len(latent), window, latent.shape[1]), dtype=dtype)
    order = np.lexsort((depths, groups))
    ordered = groups[order]
    cuts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1], True])
    for begin, end in zip(cuts[:-1], cuts[1:]):
        rows = order[begin:end]
        if not np.array_equal(depths[rows], np.arange(depths[rows[0]], depths[rows[0]] + len(rows))):
            raise RunnerError("the history window needs every depth of a trajectory (exploration record_stride 1)")
        position = np.arange(len(rows))
        for k in range(window):
            out[rows, k] = latent[rows[np.maximum(position - (window - 1 - k), 0)]]
    return out


class HistoryMLP:
    """The fitted history MLP f (NumPy forward): the latent increment of the next measurement."""

    def __init__(self, model: dict):
        self.window, self.width = int(model["window"]), int(model["width"])
        self.w1, self.b1 = np.asarray(model["w1"], dtype=np.float64), np.asarray(model["b1"], dtype=np.float64)
        self.w2, self.b2 = np.asarray(model["w2"], dtype=np.float64), np.asarray(model["b2"], dtype=np.float64)
        self.mean = np.asarray(model["scaler"]["mean"], dtype=np.float64)
        self.scale = np.asarray(model["scaler"]["scale"], dtype=np.float64)
        self.dimension = int(self.w2.shape[0])
        if (self.w1.shape != (self.width, self.window * self.dimension + 13)
                or self.w2.shape != (self.dimension, self.width)):
            raise ValueError("the history MLP's weights do not match its declared window and width")

    def increments(self, inputs, chunk: int = 65536) -> np.ndarray:
        inputs = np.asarray(inputs)
        out = np.empty((len(inputs), self.dimension))
        for start in range(0, len(inputs), chunk):
            hidden = inputs[start:start + chunk].astype(np.float64) @ self.w1.T + self.b1
            hidden = hidden * 0.5 * (1.0 + np.tanh(0.5 * hidden))          # SiLU, x * sigmoid(x)
            out[start:start + chunk] = (hidden @ self.w2.T + self.b2) * self.scale + self.mean
        return out

    def predict(self, window, exogenous, action) -> np.ndarray:
        window = np.asarray(window, dtype=np.float64)
        return window[-1] + self.increments(history_inputs(window[None], np.asarray(exogenous)[None], [action]))[0]


def history_filter(q, r) -> dict:
    """Constants of the history observer's measurement update: S = Q + R, K = Q S^-1 and the Joseph-form posterior
    covariance (I - K) Q (I - K)' + K R K' (compressed to rank + diagonal by the observer)."""
    q, r = np.asarray(q, dtype=np.float64), np.asarray(r, dtype=np.float64)
    innovation = q + r
    inverse = np.linalg.inv(innovation)
    gain = np.linalg.solve(innovation, q.T).T          # the runtime's gain arithmetic with prior covariance Q
    complement = np.eye(len(q)) - gain
    return {"inverse": inverse, "gain": gain, "posterior": complement @ q @ complement.T + gain @ r @ gain.T,
            "scale": np.sqrt(np.maximum(np.diag(innovation), 1e-12))}


def history_correct(prior, observation, constants: dict, previous_innovation=None) -> tuple:
    """(state, NIS, bias, whiteness, innovation) of one measurement update of the history observer."""
    prior, observation = np.asarray(prior, dtype=np.float64), np.asarray(observation, dtype=np.float64)
    error = observation - prior
    nis = max(0.0, float(error @ constants["inverse"] @ error))
    previous = np.zeros_like(error) if previous_innovation is None else np.asarray(previous_innovation, dtype=np.float64)
    whiteness = float(error @ previous / max(np.linalg.norm(error) * np.linalg.norm(previous), 1e-12))
    return prior + constants["gain"] @ error, nis, float(np.mean(error / constants["scale"])), whiteness, error


class HistoryFilter:
    """Mixin: the history-MLP observer's predictor-corrector on an observer of the runtime's latent space (it uses the
    host's ``normalize``, ``transform``, ``start``, ``q``, ``r``, ``rank``, ``period``, ``profile``, ``dimension``
    and ``nis_threshold``).  The state carries its window (``history_window``); a state without one (a planner's
    predicted state) repeats its latent vector."""

    def _history_setup(self, model: dict) -> None:
        from cross_morse.learning import _low_rank_diagonal
        self.history = HistoryMLP(model)
        if self.history.dimension != self.dimension:
            raise ValueError("the history MLP's latent dimension differs from the observer's")
        self._history = history_filter(self.q, self.r)
        self._history["posterior"] = _low_rank_diagonal(self._history["posterior"], self.rank)

    def _history_window(self, state) -> np.ndarray:
        window = getattr(state, "history_window", None)
        if window is None or np.shape(window) != (self.history.window, self.dimension):
            return np.repeat(np.asarray(state.z, dtype=np.float64)[None], self.history.window, axis=0)
        return np.asarray(window, dtype=np.float64)

    def _history_prior(self, state, action) -> np.ndarray:
        return self.history.predict(self._history_window(state), self.normalize(state.y)[5:8], int(action))

    def _history_start(self, state):
        state.history_window = np.repeat(np.asarray(state.z, dtype=np.float64)[None], self.history.window, axis=0)
        return state

    def _history_transition(self, state, action):
        return self._history_prior(state, action), self.q.copy()

    def _history_observe(self, state, y, depth, previous_action):
        from cross_morse.learning import ControllerState, _profile_y
        raw = _profile_y(y, self.profile)
        finite = bool(np.isfinite(raw).all())
        if depth < state.depth:
            raise ValueError("Observer depth cannot move backwards")
        if depth > state.depth + 1:
            result = self.start(raw, previous_action, depth)
            result.valid = False
            return result
        window = self._history_window(state)
        if depth == state.depth + 1:
            prior, past = self._history_prior(state, previous_action), window[1:]
        else:
            prior, past = np.asarray(state.z, dtype=np.float64).copy(), window[:-1]
        if depth % self.period:
            result = ControllerState(prior, self.q.copy(), state.nis, state.bias, state.whiteness, raw, int(depth),
                                     int(previous_action), finite and bool(np.isfinite(prior).all()) and state.valid,
                                     state.innovation)
            result.history_window = np.vstack((past, prior[None]))
            return result
        observation = self.transform(raw)
        z, nis, bias, whiteness, error = history_correct(prior, observation, self._history, state.innovation)
        valid = finite and bool(np.isfinite(z).all()) and np.isfinite(nis) and nis <= self.nis_threshold
        result = ControllerState(z, self._history["posterior"].copy(), nis, bias, whiteness, raw, int(depth),
                                 int(previous_action), bool(valid), error)
        result.history_window = np.vstack((past, observation[None]))
        return result


def _runtime_observer():
    """The released runtime's ConditionedObserver (also while an AblationObserver is installed in its place)."""
    import cross_morse.learning as learning
    return next(c for c in learning.ConditionedObserver.__mro__
                if c.__module__ == "cross_morse.learning" and c.__name__ == "ConditionedObserver")


def history_observer_class():
    """``HistoryObserver``: the runtime's latent space and update rule with the history-MLP transition, outside any
    installed variant (payload: the conditioned observer's, with ``history_mlp``, ``q`` and ``r`` of the fit)."""
    global _HISTORY_CLASS
    if _HISTORY_CLASS is not None:
        return _HISTORY_CLASS
    base = _runtime_observer()

    class HistoryObserver(HistoryFilter, base):
        def __init__(self, payload):
            super().__init__(payload)
            self.kind = "history_mlp"
            self._history_setup(payload["history_mlp"])

        def start(self, y, previous_action=-1, depth=0):
            return self._history_start(super().start(y, previous_action, depth))

        def transition(self, state, action):
            return self._history_transition(state, action)

        def observe(self, state, y, depth, previous_action):
            return self._history_observe(state, y, depth, previous_action)

    HistoryObserver.__module__ = __name__
    _HISTORY_CLASS = HistoryObserver
    return HistoryObserver


def fit_history_mlp(payload: dict, records, rows, config: dict, *, window: int, width: int, directory=None,
                    device="cpu", seed: int = FINAL_FIT_SEED + HISTORY_SEED_OFFSET, fingerprint=None,
                    stop_requested=None) -> dict:
    """Fit the history MLP f in the latent space of the conditioned observer ``payload`` on ``rows`` (model-fit rows
    of complete trajectories): the module docstring's objective, recipe, internal validation and Q/R convention."""
    import torch
    from torch import nn
    from cross_morse.learning import _groups, _holdout_groups, _low_rank_diagonal, _profile_y, _train_network
    window, width, seed = int(window), int(width), int(seed)
    rows = np.asarray(rows, dtype=np.int64)
    median, scale = np.asarray(payload["median"], dtype=np.float64), np.asarray(payload["scale"], dtype=np.float64)
    center, basis = np.asarray(payload["center"], dtype=np.float64), np.asarray(payload["basis"], dtype=np.float64)

    def normalized(values):
        values = _profile_y(np.asarray(values)[rows], payload["profile"])
        return np.clip((np.where(np.isfinite(values), values, median) - median) / scale, -5, 5)
    now, following = normalized(records["y"]), normalized(records["y_next"])
    latent, latent_next = (now - center) @ basis, (following - center) @ basis
    windows = history_windows(latent, _groups(records)[rows], np.asarray(records["depth"])[rows], window,
                              dtype=np.float32)
    inputs = history_inputs(windows, now[:, 5:8], np.asarray(records["action"])[rows], dtype=np.float32)
    del windows
    holdout = _holdout_groups(records)[rows]
    unique = np.unique(holdout)
    if len(unique) < 2:
        raise RunnerError("the history MLP needs at least two trajectory groups for its internal validation")
    count = min(len(unique) - 1, max(1, int(math.ceil(0.10 * len(unique)))))
    chosen = np.random.default_rng(seed).permutation(unique)[:count]
    validation = np.isin(holdout, chosen)
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    dimension = basis.shape[1]
    net = nn.Sequential(nn.Linear(inputs.shape[1], width), nn.SiLU(), nn.Linear(width, dimension))
    with contextlib.ExitStack() as stack:
        if directory is None:
            directory = stack.enter_context(tempfile.TemporaryDirectory(prefix="cm-history-mlp-"))
        scaler, audit = _train_network(net, inputs, (latent_next - latent).astype(np.float32),
                                       np.flatnonzero(~validation), np.flatnonzero(validation), config,
                                       Path(directory) / "history_mlp.last.pt",
                                       f"{fingerprint or 'history-mlp'}:history_mlp:{window}:{width}", str(device),
                                       stop_requested or (lambda: False), seed)
    if scaler is None:
        raise RunnerError(f"the history MLP has no finite fitting targets ({audit.get('reason')})")
    state = {k: v.detach().cpu().double().numpy() for k, v in net.state_dict().items()}
    model = {"window": window, "width": width, "activation": "SiLU", "w1": state["0.weight"].tolist(),
             "b1": state["0.bias"].tolist(), "w2": state["2.weight"].tolist(), "b2": state["2.bias"].tolist(),
             "scaler": {"mean": [float(v) for v in np.ravel(scaler["mean"])],
                        "scale": [float(v) for v in np.ravel(scaler["scale"])]}}
    residual = latent_next - (latent + HistoryMLP(model).increments(inputs))
    rank = int(payload["rank"])
    q = _low_rank_diagonal(residual.T @ residual / max(len(residual) - 1, 1) + np.eye(dimension) * 1e-4, rank)
    r = _low_rank_diagonal(0.25 * q + np.eye(dimension) * 1e-3, rank)
    return {"model": model, "q": q.tolist(), "r": r.tolist(), "audit": {
        "window": window, "width": width, "latent_dimension": int(dimension),
        "parameters": history_parameters(window, width, dimension),
        "conditioned_transition_parameters": conditioned_parameters(payload),
        "rows": int(len(rows)), "validation_rows": int(validation.sum()), "validation_groups": int(count),
        "seed": seed, "residual_rms": float(np.sqrt(np.mean(residual ** 2))) if residual.size else None,
        "training": {k: v for k, v in audit.items() if k != "history"},
        "objective": "one-step latent prediction of z(y_{l+1}) from the measured window ending at l (teacher-forced), "
                     "u_l and a_l; squared error of standardized increments (cross_morse.learning._train_network)",
        "noise": "Q = residual covariance on the model-fit rows (rank + diagonal, + 1e-4 I); R = 0.25 Q + 1e-3 I"}}


def history_observer(base_payload: dict, records, rows, config: dict, *, window=None, width=HISTORY_WIDTH,
                     directory=None, device="cpu", seed=FINAL_FIT_SEED + HISTORY_SEED_OFFSET, fingerprint=None):
    """A history-MLP observer fitted outside a lifecycle (the decision audit's E8 study): the sealed conditioned
    observer's latent space, the MLP fitted on its model-fit rows, NIS never invalidating (as its global KF)."""
    window = history_window(config) if window is None else int(window)
    fit = fit_history_mlp(base_payload, records, rows, config, window=window, width=width, directory=directory,
                          device=device, seed=seed, fingerprint=fingerprint)
    payload = copy.deepcopy(base_payload)
    payload.update(history_mlp=fit["model"], history_fit=fit["audit"], q=fit["q"], r=fit["r"], conditioned={},
                   nis_threshold=1e12,
                   identification="history MLP transition over the last w latent vectors; R=0.25Q+1e-3I (E8)")
    return history_observer_class()(payload)


# ----------------------------------------------------------------------------- observer layer
def observer_class():
    """``AblationObserver``: the runtime's ConditionedObserver with deletion and observer variants."""
    global _OBSERVER_CLASS
    if _OBSERVER_CLASS is not None:
        return _OBSERVER_CLASS
    base = _runtime_observer()
    from cross_morse.learning import ControllerState, _profile_y

    class AblationObserver(HistoryFilter, base):
        _cm_ablation_observer = True

        def __init__(self, payload):
            super().__init__(payload)
            ablation = payload.get("ablation")
            if ablation is None:
                if not _FITTING.get():
                    raise ValueError("this observer was not fitted under the installed ablation variant")
                self.ablation, self.kind, self.deleted = None, "conditioned", np.zeros(0, dtype=np.int64)
                return
            if ablation != observer_ablation(_ACTIVE["spec"]):
                raise ValueError("the observer's ablation differs from the installed variant; load the controller "
                                 "under its own variant")
            self.ablation, self.kind = ablation, ablation["observer"]
            self.deleted = np.asarray(payload.get("ablation_indices", []), dtype=np.int64)
            if self.kind in ("raw", "ema"):
                fit = payload.get("ablation_fit") or {}
                self.weight = 1.0 if self.kind == "raw" else float(ablation["ema_weight"])
                self.innovation_cov = np.asarray(fit.get("innovation_covariance", np.eye(self.dimension)),
                                                 dtype=np.float64)
                self._inverse = np.linalg.inv(self.innovation_cov)
                self._innovation_scale = np.sqrt(np.maximum(np.diag(self.innovation_cov), 1e-12))
            elif self.kind == "history_mlp":
                fit = (payload.get("ablation_fit") or {}).get("history_mlp")
                if fit is None:
                    raise ValueError("a history-MLP observer payload holds its fitted MLP (ablation_fit.history_mlp)")
                self._history_setup(fit)

        def normalize(self, y):
            if len(self.deleted):
                y = np.array(y, dtype=np.float64, copy=True)
                y[..., self.deleted] = self.median[self.deleted]
            return super().normalize(y)

        def reconstruct(self, z):
            raw = super().reconstruct(z)
            if len(self.deleted):
                raw[..., self.deleted] = self.median[self.deleted]
            return raw

        def start(self, y, previous_action=-1, depth=0):
            state = super().start(y, previous_action, depth)
            if self.kind in ("raw", "ema"):
                state.cov = self.innovation_cov.copy()
            elif self.kind == "history_mlp":
                state = self._history_start(state)
            return state

        def transition(self, state, action):
            if self.kind == "memoryless":
                return np.asarray(state.z, dtype=np.float64).copy(), self.r.copy()
            if self.kind in ("raw", "ema"):
                return np.asarray(state.z, dtype=np.float64).copy(), self.innovation_cov.copy()
            if self.kind == "history_mlp":
                return self._history_transition(state, action)
            return super().transition(state, action)

        def observe(self, state, y, depth, previous_action):
            if self.kind == "memoryless":
                if depth < state.depth:
                    raise ValueError("Observer depth cannot move backwards")
                return self.start(y, previous_action, depth)
            if self.kind == "history_mlp":
                return self._history_observe(state, y, depth, previous_action)
            if self.kind not in ("raw", "ema"):
                return super().observe(state, y, depth, previous_action)
            raw = _profile_y(y, self.profile)
            finite = bool(np.isfinite(raw).all())
            if depth < state.depth:
                raise ValueError("Observer depth cannot move backwards")
            if depth > state.depth + 1:
                result = self.start(raw, previous_action, depth)
                result.valid = False
                return result
            if depth % self.period:
                return ControllerState(state.z, state.cov, state.nis, state.bias, state.whiteness, raw, int(depth),
                                       int(previous_action), finite and state.valid, state.innovation)
            error = self.transform(raw) - state.z
            nis = max(0.0, float(error @ self._inverse @ error))
            previous = state.innovation if state.innovation is not None else np.zeros_like(error)
            whiteness = float(error @ previous / max(np.linalg.norm(error) * np.linalg.norm(previous), 1e-12))
            z = state.z + self.weight * error
            valid = finite and bool(np.isfinite(z).all()) and np.isfinite(nis) and nis <= self.nis_threshold
            return ControllerState(z, self.innovation_cov.copy(), nis, float(np.mean(error / self._innovation_scale)),
                                   whiteness, raw, int(depth), int(previous_action), bool(valid), error)

    AblationObserver.__module__ = __name__
    _OBSERVER_CLASS = AblationObserver
    return AblationObserver


def _fit_observer(records, indices, config):
    """The runtime's observer fit on records with the deleted coordinates held at their fitting median, then
    the declared observer variant fitted on the same model-fit rows."""
    from cross_morse.learning import _low_rank_diagonal, _profile_y
    original = _original("cross_morse.learning", "_fit_observer")
    spec = _ACTIVE["spec"]
    ablation = observer_ablation(spec)
    deleted = _deleted_indices(spec)
    indices = np.asarray(indices, dtype=np.int64)
    fitting = records
    if deleted:
        profile = config.get("active_profile", "full").lower()
        raw = _profile_y(np.asarray(records["y"])[indices], profile)
        with np.errstate(all="ignore"):
            hold = np.nan_to_num(np.nanmedian(np.where(np.isfinite(raw), raw, np.nan), axis=0))[deleted]
        fitting = dict(records)
        for key in ("y", "y_next"):
            values = np.array(records[key], copy=True)
            values[:, deleted] = hold.astype(values.dtype)
            fitting[key] = values
    token = _FITTING.set(True)
    try:
        base = original(fitting, indices, config)
    finally:
        _FITTING.reset(token)
    payload = copy.deepcopy(base.payload)
    payload.update(ablation=ablation, ablation_indices=list(deleted), ablation_fit={})
    kind = ablation["observer"]
    if kind == "global":
        median, scale = np.asarray(payload["median"]), np.asarray(payload["scale"])
        center, basis = np.asarray(payload["center"]), np.asarray(payload["basis"])

        def normalized(values):
            values = _profile_y(np.asarray(values)[indices], payload["profile"])
            values = np.where(np.isfinite(values), values, median)
            return np.clip((values - median) / scale, -5, 5)
        now, following = normalized(fitting["y"]), normalized(fitting["y_next"])
        z, z_next = (now - center) @ basis, (following - center) @ basis
        residual = z_next - np.column_stack((z, now[:, 5:8], np.ones(len(z)))) @ np.asarray(payload["global_coef"])
        k, rank = basis.shape[1], int(payload["rank"])
        q = _low_rank_diagonal(residual.T @ residual / max(len(residual) - 1, 1) + np.eye(k) * 1e-4, rank)
        payload.update(conditioned={}, q=q.tolist(), r=_low_rank_diagonal(0.25 * q + np.eye(k) * 1e-3, rank).tolist(),
                       identification="global ridge transition; R=0.25Q+1e-3I (ablation variant)")
    elif kind in ("raw", "ema"):
        probe = copy.deepcopy(payload)
        probe["ablation_fit"] = {"innovation_covariance": np.eye(len(payload["basis"][0])).tolist()}
        token = _FITTING.set(True)
        try:
            innovations = latent_innovations(observer_class()(probe), fitting, indices)
        finally:
            _FITTING.reset(token)
        dimension = len(payload["basis"][0])
        second = (innovations.T @ innovations / len(innovations) if len(innovations) else np.eye(dimension))
        payload["ablation_fit"] = {"innovation_covariance": (second + 1e-6 * np.eye(dimension)).tolist(),
                                   "innovation_events": int(len(innovations)),
                                   "rule": "second moment of the model-fit innovations plus 1e-6 I"}
    elif kind == "history_mlp":
        context = _FIT_CONTEXT.get() or {}
        fit = fit_history_mlp(payload, fitting, indices, config, window=ablation["history"]["window"],
                              width=ablation["history"]["width"], directory=context.get("directory"),
                              device=context.get("device", "cpu"),
                              seed=int(context.get("seed", FINAL_FIT_SEED)) + HISTORY_SEED_OFFSET,
                              fingerprint=context.get("fingerprint"), stop_requested=context.get("stop_requested"))
        payload.update(conditioned={}, q=fit["q"], r=fit["r"],
                       identification="history MLP transition over the last w latent vectors; R=0.25Q+1e-3I "
                                      "(ablation variant)")
        payload["ablation_fit"] = {"history_mlp": fit["model"], "history_fit": fit["audit"]}
    return observer_class()(payload)


def _fit_one(records, fitting_indices, config, directory, fingerprint, device, stop_requested, seed):
    """The runtime's fold fit with its directory, device, fingerprint and seed visible to the history-MLP fit."""
    token = _FIT_CONTEXT.set({"directory": Path(directory), "fingerprint": fingerprint, "device": str(device),
                              "stop_requested": stop_requested, "seed": int(seed)})
    try:
        return _original("cross_morse.learning", "_fit_one")(records, fitting_indices, config, directory, fingerprint,
                                                             device, stop_requested, seed)
    finally:
        _FIT_CONTEXT.reset(token)


def observer_parameters(observer) -> dict:
    """Latent dimension and transition parameters of a fitted observer (the E8 capacity comparison)."""
    payload, kind = observer.payload, getattr(observer, "kind", "conditioned")
    dimension = int(observer.dimension)
    if kind == "history_mlp":
        fit = (payload.get("ablation_fit") or {}).get("history_mlp") or payload.get("history_mlp")
        transition = history_parameters(fit["window"], fit["width"], dimension)
    elif kind in ("raw", "ema", "memoryless"):
        transition = 0
    elif kind == "global":
        transition = (dimension + 4) * dimension
    else:
        transition = conditioned_parameters(payload)
    return {"kind": kind, "latent_dimension": dimension, "transition_parameters": int(transition)}


def latent_innovations(observer, records, indices) -> np.ndarray:
    """Innovations at every observer update along the given rows' trajectories (rows in depth order)."""
    from cross_morse.learning import _groups
    indices = np.asarray(indices, dtype=np.int64)
    groups, depths = _groups(records)[indices], np.asarray(records["depth"])[indices]
    previous = np.asarray(records["previous_action"])[indices]
    values = np.asarray(records["y"])[indices]
    order = np.lexsort((depths, groups))
    events, state, last = [], None, None
    for position in order:
        group, depth = int(groups[position]), int(depths[position])
        if group != last or state is None or depth != state.depth + 1:
            state = observer.start(values[position], int(previous[position]), depth)
        else:
            state = observer.observe(state, values[position], depth, int(previous[position]))
            if depth % observer.period == 0 and state.innovation is not None:
                events.append(np.asarray(state.innovation, dtype=np.float64))
        last = group
    return np.stack(events) if events else np.zeros((0, observer.dimension))


def _features(state, max_depth):
    out = _original("cross_morse.learning", "_features")(state, max_depth)
    deleted = [i for i in _deleted_indices(_ACTIVE["spec"]) if i in (6, 7)]
    if deleted:
        out = np.array(out, copy=True)
        offset = 2 * len(state.z) + 3
        for index in deleted:
            out[offset + index - 5] = 0.0
    return out


# ----------------------------------------------------------------------------- measurement layer (scramble)
class ScrambledBank:
    """A runtime MeasurementBank whose declared coordinates are taken from a donor D_explore trajectory."""

    def __init__(self, bank, op, pool):
        self.bank, self.op, self.pool = bank, op, pool
        self.indices = np.asarray(op["indices"], dtype=np.int64)
        self.donor, self.key = None, None

    def prepare(self, h0, logits0):
        result = self.bank.prepare(h0, logits0)
        key = _TRAJECTORY.get()
        if key is None:
            state = h0.detach().double()
            key = ("state", "x".join(map(str, state.shape)), f"{float(state.sum()):.10e}",
                   f"{float(state.square().sum()):.10e}")
        self.key, self.donor = key, donor_index(self.op["seed"], key, len(self.pool))
        return result

    def measure(self, h, logits, depth, max_depth, accumulated_mass=0.0, accumulated_hops=0.0, mask=None):
        import torch
        y = self.bank.measure(h, logits, depth, max_depth, accumulated_mass, accumulated_hops, mask)
        if self.donor is None:
            raise RuntimeError("Call prepare(h0, logits0) before measure")
        if not 0 <= int(depth) < self.pool.shape[1]:
            raise ValueError("measurement depth outside the donor pool")
        out = y.clone()
        out[torch.as_tensor(self.indices, device=y.device)] = torch.as_tensor(
            self.pool[self.donor, int(depth)], dtype=y.dtype, device=y.device)
        return out

    def metadata(self):
        meta = dict(self.bank.metadata())
        meta["ablation_scramble"] = {"coordinates": self.op["coordinates"], "seed": self.op["seed"],
                                     "donor_unit": self.donor, "pool_sha256": self.op["pool"]["sha256"]}
        return meta

    def __getattr__(self, name):
        return getattr(self.bank, name)


def _scrambled_sensor_bank(graph, model, config, profile="full"):
    bank = _original("cross_morse.exploration", "sensor_bank")(graph, model, config, profile)
    return ScrambledBank(bank, _ops("scramble")[0], _ACTIVE["pool"])


def _collect_trajectory(model, x, graph, labels, ids, core_ids, config, trajectory, seed_index, device):
    token = _TRAJECTORY.set(("explore", int(trajectory), int(seed_index)))
    try:
        return _original("cross_morse.exploration", "collect_trajectory")(model, x, graph, labels, ids, core_ids,
                                                                           config, trajectory, seed_index, device)
    finally:
        _TRAJECTORY.reset(token)


# ----------------------------------------------------------------------------- decision layer
def _restricted() -> bool:
    return not _OPEN_LOOP.get() and _ROUTE.get() in (None, *FEEDBACK_ROUTES)


def _supported(self, y, depth):
    actions = _original("Controller", "supported")(self, y, depth)
    if not _restricted():
        return actions
    ops = {op["op"]: op for op in _ACTIVE["spec"]["operations"]}
    if "risk_control" in ops:           # every cell admissible (as the sealed unrestricted audit), no reset
        actions = [] if not np.isfinite(np.asarray(y, dtype=np.float64)).all() else [
            a for a in range(10) if a not in RESET_ACTIONS]
    if "remove_reset" in ops:
        actions = [a for a in actions if a != ops["remove_reset"]["action"]]
    if "action_bank" in ops:
        actions = [a for a in actions if a in ops["action_bank"]["allowed"]]
    return actions


def _recovery_action(supported, last_safe_available=True):
    if not _restricted():
        return _original("cross_morse.policies", "recovery_action")(supported, last_safe_available)
    if _ops("risk_control"):
        return 0                        # risk-controlled execution off: every fallback is the nominal step
    removed = {op["action"] for op in _ops("remove_reset")}
    if 0 in supported:
        return 0
    if last_safe_available and 9 not in removed:
        return 9
    return 8 if 8 not in removed else 9


def _beam_plan(controller, state, horizon, config, budget, *, open_loop=False):
    token = _OPEN_LOOP.set(bool(open_loop))
    try:
        return _original("cross_morse.policies", "beam_plan")(controller, state, horizon, config, budget,
                                                               open_loop=open_loop)
    finally:
        _OPEN_LOOP.reset(token)


def _execute_policy(model, x, graph, config, candidate, *args):
    token = _ROUTE.set(candidate.get("route"))
    try:
        return _original("cross_morse.policies", "_execute_policy")(model, x, graph, config, candidate, *args)
    finally:
        _ROUTE.reset(token)


def unrestricted_support(support: dict, config: dict) -> dict:
    support = copy.deepcopy(support)
    controller = config["controller"]
    strata = int(controller.get("depth_bins", 4)) * (len(controller.get("entropy_bins", [1 / 3, 2 / 3])) + 1)
    for stratum in range(strata):
        for action in range(10):
            cell = support["cells"].setdefault(f"{stratum}:{action}", {"stratum": stratum, "action": action,
                                                                       "row_count": 0, "trajectory_count": 0})
            cell["qualified"] = True
    support["qualified_fraction"] = 1.0
    support["switch"] = ("unrestricted: every registered action admissible in every stratum (risk_control off; "
                         "the resets A8/A9 are removed from feedback decisions at execution)")
    return support


def _support(records, indices, config):
    return unrestricted_support(_original("cross_morse.learning", "_support")(records, indices, config), config)


# ============================================================================= lifecycle entry point
def _option(argv, name):
    for i, item in enumerate(argv):
        if item == name and i + 1 < len(argv):
            return argv[i + 1]
        if item.startswith(name + "="):
            return item.split("=", 1)[1]
    return None


def lifecycle_main(argv=None) -> int:
    """run_cross_morse.py (same arguments) with the configuration's ablation variant installed."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    config_path, output = _option(argv, "--config"), _option(argv, "--output-dir")
    if config_path is None or output is None:
        print("ERROR: a variant lifecycle needs --config and --output-dir", file=sys.stderr)
        return 2
    try:
        config = _resolved(config_path)
        spec = config.get(VARIANT_KEY)
        if spec is None:
            raise RunnerError("the configuration declares no ablation_variant; run run_cross_morse.py instead")
        validate_spec(spec, config)
        install(spec, run_dir=Path(output).resolve())
    except RunnerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    import run_cross_morse
    return run_cross_morse.main(argv)


# ============================================================================= variant runs (PyTorch)
def _sealed_through(run_root, through) -> bool:
    root = Path(run_root)
    for stage in STAGES[:STAGES.index(through) + 1]:
        marker = root / "stages" / f"{stage}.json"
        if not marker.is_file() or json.loads(marker.read_text()).get("status") != "COMPLETE":
            return False
    return True


def seed_variant_run(parent_root, run_root, config_path, *, device="cpu", pool_file=None) -> dict:
    """Copy the parent's sealed prepare/train outputs into a new run and seal them under the variant config."""
    from cross_morse.common import atomic_json, file_hash
    from cross_morse.pipeline import Pipeline
    from cross_morse.provenance import (current_source_sha256, stage_manifest_path, verify_stage_chain,
                                        write_stage_manifest)
    parent_root, run_root = Path(parent_root).resolve(), Path(run_root).resolve()
    config = _resolved(config_path)
    source = current_source_sha256()
    try:
        parent_chain = verify_stage_chain(parent_root, "train", expected_source_sha256=source)
    except Exception as exc:
        raise RunnerError(f"{parent_root.name}: the parent's prepare/train chain does not verify against this runtime "
                          f"source ({exc})") from None
    if (run_root / "stages" / "train.json").is_file():
        verify_stage_chain(run_root, "train", expected_source_sha256=source, expected_config_sha256=canonical(config))
        return read_json(run_root / "stages" / "train.json").get("seeded_from_parent", {})
    if run_root.exists() and not (run_root / "provenance" / "stages").is_dir():
        shutil.rmtree(run_root)                     # an attempt interrupted before its first seal
    run_root.mkdir(parents=True, exist_ok=True)
    if pool_file is not None:
        spec = config[VARIANT_KEY]
        op = next(o for o in spec["operations"] if o["op"] == "scramble")
        shutil.copyfile(pool_file, run_root / POOL_FILE)
        if sha256(run_root / POOL_FILE) != op["pool"]["sha256"]:
            raise RunnerError(f"{pool_file}: the donor pool changed after planning")
    pipeline = Pipeline(config, run_root / "data-root-unused", run_root, device,
                        resume=(run_root / "run_state.json").exists())
    copied = []

    def copy_outputs(manifest, skip):
        for record in manifest["outputs"]:
            relative = record["path"]
            if relative in skip:
                continue
            target = run_root / relative
            if not (target.is_file() and file_hash(target) == record["sha256"]):
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(parent_root / relative, target)
            if file_hash(target) != record["sha256"]:
                raise RunnerError(f"{relative}: the copy differs from the parent's sealed output")
            copied.append({"path": relative, "sha256": record["sha256"]})

    def marker(stage, parent_manifest):
        origin = {"run": parent_root.name, "stage_manifest_sha256": file_hash(stage_manifest_path(parent_root, stage)),
                  "files": len([c for c in copied if c["path"] in {o["path"] for o in parent_manifest["outputs"]}]),
                  "rule": "byte-identical copies of the parent's sealed outputs (configuration and environment "
                          "excepted), sealed under the variant configuration by cross_morse.provenance."
                          "write_stage_manifest; the stage was not re-executed"}
        atomic_json(run_root / "stages" / f"{stage}.json", {
            "stage": stage, "status": "COMPLETE", "config_sha256": pipeline.config_hash,
            "source_sha256": pipeline.source_hash, "label_accesses": [], "seeded_from_parent": origin,
            "manifest_sha256": file_hash(stage_manifest_path(run_root, stage))})
        return origin

    origins = {}
    if not stage_manifest_path(run_root, "prepare").is_file():
        copy_outputs(parent_chain[0], {"config.json", "environment.json"})
        write_stage_manifest(run_root, "prepare", dataset=config["dataset"], source_sha256=pipeline.source_hash,
                             config_sha256=pipeline.config_hash,
                             unit_sha256=file_hash(run_root / "prepared" / "splits.npz"), inputs=[],
                             outputs=pipeline._stage_output_paths("prepare", []))
    if not (run_root / "stages" / "prepare.json").is_file():
        origins["prepare"] = marker("prepare", parent_chain[0])
    chain = verify_stage_chain(run_root, expected_source_sha256=pipeline.source_hash,
                               expected_config_sha256=pipeline.config_hash)
    copy_outputs(parent_chain[1], set())
    inputs = sorted({a["path"] for manifest in chain for a in manifest["outputs"]})
    write_stage_manifest(run_root, "train", dataset=config["dataset"], source_sha256=pipeline.source_hash,
                         config_sha256=pipeline.config_hash, unit_sha256=file_hash(run_root / "prepared" / "splits.npz"),
                         inputs=inputs, outputs=pipeline._stage_output_paths("train", chain),
                         model_baseline="provenance/frozen_models.json")
    origins["train"] = marker("train", parent_chain[1])
    state = read_json(run_root / "run_state.json")
    state.update(status="RUNNING", last_complete_stage="train")
    atomic_json(run_root / "run_state.json", state)
    verify_stage_chain(run_root, "train", expected_source_sha256=source, expected_config_sha256=pipeline.config_hash)
    return {"stages": origins, "copied_files": len(copied)}


def lifecycle_command(entry, stage, *, data_root, device, threads, python=sys.executable) -> list[str]:
    return [python, "-B", "-m", "cm_experiments.ablation_lifecycle", "--config", entry["config"],
            "--output-dir", entry["run"], "--data-root", str(data_root), "--device", device,
            "--threads", str(threads), "--stage", stage, "--resume"]


def run_lifecycle_job(output, entry, *, data_root=None, device="cuda", threads=16, python=sys.executable,
                      runner=subprocess.run) -> Path:
    output = Path(output)
    record = output / "jobs" / entry["slug"] / "lifecycle.json"
    if canonical(read_json(entry["config"])) != entry["config_sha256"]:
        raise RunnerError(f"{entry['config']} changed after planning")
    run = Path(entry["run"])
    if _sealed_through(run, entry["through"]) and record.is_file():
        return record
    started = utc_now()
    seeded = seed_variant_run(entry["parent"], run, entry["config"], device=device, pool_file=entry.get("pool_file"))
    commands = []
    data_root = Path(data_root or output / "data-root-unused")
    log = output / "logs" / f"{slug(entry['slug'])}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    for stage in STAGES[STAGES.index("explore"):STAGES.index(entry["through"]) + 1]:
        if _sealed_through(run, stage):
            continue
        command = lifecycle_command(entry, stage, data_root=data_root, device=device, threads=threads, python=python)
        commands.append(command)
        with log.open("a", encoding="utf-8") as handle:
            handle.write("$ " + " ".join(command) + "\n")
            handle.flush()
            code = runner(command, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT, check=False).returncode
        if code == 75:
            raise InterruptedError(f"lifecycle {run} checkpointed at a safe boundary")
        if code != 0:
            raise RunnerError(f"lifecycle {run} stage {stage} exited with {code}; see {log}")
    sealed = SealedRun(run)
    _dump(record, {"schema": "cm-ablation-lifecycle-1", "job": entry["job"], "variant": entry["variant"],
                   "config_sha256": entry["config_sha256"], "run": run.name, "identity": sealed.identity(),
                   "parent_identity": entry["parent_identity"], "seeded_from_parent": seeded,
                   "stages_executed": [c[c.index("--stage") + 1] for c in commands], "through": entry["through"],
                   "environment": environment_record(device), "runtime_source_sha256": runtime_source_sha256(),
                   "started_utc": started, "finished_utc": utc_now()})
    return record


def open_lifecycle_run(entry, parent: SealedRun, *, verify_chain=True) -> SealedRun:
    """A sealed variant lifecycle of the plan, checked against its planned configuration and its parent."""
    run = SealedRun(entry["run"])
    for stage in STAGES[:STAGES.index(entry["through"]) + 1]:
        run.require(stage)
    if canonical(run.config) != entry["config_sha256"] or run.config.get(VARIANT_KEY) != entry["variant"]:
        raise RunnerError(f"{run.root.name}: the executed configuration differs from the planned variant")
    validate_spec(entry["variant"], run.config)
    for relative in ("prepared/splits.npz", "provenance/frozen_models.json"):
        if sha256(run.root / relative) != sha256(parent.root / relative):
            raise RunnerError(f"{run.root.name}: {relative} differs from the parent's")
    if verify_chain:
        from cross_morse.provenance import verify_stage_chain
        try:
            verify_stage_chain(run.root, entry["through"], expected_source_sha256=runtime_source_sha256(),
                               expected_config_sha256=entry["config_sha256"])
        except Exception as exc:
            raise RunnerError(f"{run.root.name}: the variant's stage chain does not verify ({exc})") from None
    return run


def open_fitted_cell(cell: dict, parent: SealedRun):
    """(run root, variant) of a separately fitted factorial cell (``None`` variant: the parent itself)."""
    if cell["source"] == "parent":
        return parent.root, None
    run = open_lifecycle_run(cell, parent)
    return run.root, cell["variant"]


# ============================================================================= measurement (PyTorch)
class RecordingController:
    """A fitted Controller that keeps the observer state of every executed depth (no other change)."""

    def __init__(self, controller):
        self._controller = controller
        self.states = {}

    def __getattr__(self, name):
        return getattr(self._controller, name)

    def start(self, y, previous_action=-1):
        state = self._controller.start(y, previous_action)
        self.states[int(state.depth)] = state
        return state

    def observe(self, state, y, depth, previous_action):
        result = self._controller.observe(state, y, depth, previous_action)
        self.states[int(depth)] = result
        return result


def _role_loss(model, h, rows, target, spec) -> float:
    import torch
    from cross_morse.tasks import loss as task_loss
    logits = model.logits(h)
    index = torch.as_tensor(np.asarray(rows, dtype=np.int64), device=logits.device)
    return float(task_loss(logits[index].float(), target, spec))


def realized_consequences(model, x, graph, config, trace, rows, target, spec, horizon):
    """Delta_d = l_{d+h}(a_d -> nominal) - l_{d+h}(nominal) at every executed site (exact replay of the executed
    action and anchor trace; the exploration target of cross_morse.exploration.collect_trajectory)."""
    import torch
    from cross_morse.common import precision_context
    from cross_morse.tasks import probabilities
    actions, anchors = [int(a) for a in trace["actions"]], [int(s) for s in trace["safe_depths"]]
    depth = int(config["model"]["depth"])
    deltas = np.zeros(len(actions), dtype=np.float64)
    model.bind_graph(graph)
    with torch.no_grad(), precision_context(config, x.device):
        h0 = model.encode(x)
        h, last_safe, last_safe_depth = h0, h0, 0
        for d, a in enumerate(actions):
            if anchors[d] == d:
                last_safe, last_safe_depth = h.clone(), d
            elif anchors[d] != last_safe_depth:
                raise RunnerError("the replay skipped a recorded last-safe update")
            if a != 0:
                actual = model.step(h, h0, graph, a, last_safe, site=d)
                nominal = model.native_step(h, h0, graph, site=d)
                for offset in range(1, min(int(horizon), depth - d)):
                    nominal = model.native_step(nominal, h0, graph, site=d + offset)
                    actual = model.native_step(actual, h0, graph, site=d + offset)
                deltas[d] = _role_loss(model, actual, rows, target, spec) - _role_loss(model, nominal, rows, target, spec)
            h = model.step(h, h0, graph, a, last_safe, site=d)
        final = probabilities(model.logits(h).float(), getattr(model, "task", spec)).cpu().numpy()
    return deltas, final


def warning_trace(model, x, graph, config, controller, rows, target, spec, profile):
    """Per-depth role-row loss of the nominal trajectory and the controller's terminal-value warnings on it."""
    import torch
    import cross_morse.policies as policies
    from cross_morse.common import precision_context
    from cross_morse.core import site_transport_accounting
    depth = int(config["model"]["depth"])
    losses, values = [], []
    model.bind_graph(graph)
    available = bool(controller.terminal_available)
    with torch.no_grad(), precision_context(config, x.device):
        h0 = model.encode(x)
        logits = model.logits(h0)
        bank = policies.sensor_bank(graph, model, config, profile)
        bank.prepare(h0, logits)
        state = controller.start(bank.measure(h0, logits, 0, depth).cpu().numpy())
        losses.append(_role_loss(model, h0, rows, target, spec))
        h, mass, hops = h0, 0.0, 0.0
        for d in range(depth):
            if d:
                state = controller.observe(state, bank.measure(h, logits, d, depth, mass, hops).cpu().numpy(), d, 0)
            values.append(float(controller.terminal_value(state)) if available and state.valid else float("nan"))
            h = model.step(h, h0, graph, 0, None, site=d)
            accounting = site_transport_accounting(0, d, model if hasattr(model, "is_native_site") else config)
            mass, hops = mass + accounting["mass"], hops + accounting["hops"]
            logits = model.logits(h)
            losses.append(_role_loss(model, h, rows, target, spec))
    return np.asarray(losses), np.asarray(values)


def warning_summary(losses, values) -> dict:
    losses, values = np.asarray(losses, dtype=np.float64), np.asarray(values, dtype=np.float64)
    depth = len(values)
    onset = int(np.argmin(losses))
    warned = np.flatnonzero(np.isfinite(values) & (values > 0))
    first = int(warned[0]) if len(warned) else depth
    return {"onset": onset, "has_onset": onset < depth, "first_warning": first, "lead": onset - first}


def assessment_rmse(run_root, profile) -> dict:
    with np.load(Path(run_root) / "fit" / profile / "calibration_assessment.npz", allow_pickle=False) as z:
        delta, mean, rows = np.asarray(z["delta"], np.float64), np.asarray(z["mean"], np.float64), z["row_index"]
    return {"rmse": float(np.sqrt(np.mean((delta - mean) ** 2))), "rows": int(len(rows)),
            "rows_sha256": hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest(),
            "targets_sha256": hashlib.sha256(np.asarray(delta, dtype="<f8").tobytes()).hexdigest()}


def latent_innovation_rms(run_root, controller, profile) -> dict:
    """Diagnostic: RMS of the observer's latent innovations on the sealed assessment trajectories."""
    from cross_morse.exploration import load_records
    from cross_morse.learning import _holdout_groups
    records = load_records(Path(run_root) / "explore")
    partition = read_json(Path(run_root) / "fit" / profile / "calibration_partition.json")
    groups = _holdout_groups(records)
    rows = np.flatnonzero(np.isin(groups, np.asarray(partition["groups"]["assessment"])))
    innovations = latent_innovations(controller.observer, records, rows)
    return {"rms": float(np.sqrt(np.mean(innovations ** 2))) if innovations.size else None,
            "events": int(len(innovations)), "unit": "observer latent coordinates (robust z-score projected)"}


def measure_arm(ctx, models, seeds, candidate, controller, reference, *, role, labels, units, horizon,
                max_graphs=0) -> tuple[dict, dict]:
    """Execute the audited rule with one arm's controller on every input of the role (variant installed)."""
    from cross_morse.evaluation import independent_graphs
    from cross_morse.policies import PolicyExecutionError, execute_policy
    from cross_morse.statistics import cluster_summary
    from cross_morse.tasks import sample_losses, task_spec
    from .ensemble_grid import ensemble_probabilities
    import torch
    config = ctx.config
    spec = task_spec(config)
    profile = "full" if candidate.get("profile", "none") == "none" else candidate["profile"]
    feedback = candidate["route"] in FEEDBACK_ROUTES
    action_counts = np.zeros(10, dtype=np.int64)
    decision_rows, warning_rows, losses_rows = [], [], []
    unit_values, unit_ids = [], []
    totals = {"deployed_work": 0.0, "sensor_dense_multiply_adds": 0.0, "spectral_products": 0, "wall_seconds": 0.0,
              "executions": 0, "replay_max_abs_deviation": 0.0, "inputs": 0}
    truncated = False
    for count, (eid, x, graph, rows, positions) in enumerate(ctx.episodes(role)):
        if max_graphs and count >= max_graphs:
            truncated = True
            break
        target_np = labels[positions]
        target = torch.as_tensor(target_np, device=x.device)
        members = []
        for seed in seeds:
            model = models[seed]
            proxy = RecordingController(controller)
            try:
                probs, trace = execute_policy(model, x, graph, config, candidate, proxy, reference)
            except PolicyExecutionError as error:
                raise RunnerError(f"{candidate['route']} failed numerically on input {eid} (member {seed}): "
                                  f"{error.reason}; no value is filled in") from None
            members.append(np.asarray(probs, dtype=np.float32)[rows])
            totals["deployed_work"] += float(trace.get("deployed_multiply_add_work", trace["multiply_add_work"]))
            sensors = trace.get("sensor_operations") or {}
            totals["sensor_dense_multiply_adds"] += float(sensors.get("dense_multiply_adds", 0))
            totals["spectral_products"] += int(sensors.get("spectral_products", 0))
            totals["wall_seconds"] += float(trace["wall_seconds"])
            totals["executions"] += 1
            actions = [int(a) for a in trace["actions"]]
            np.add.at(action_counts, actions, 1)
            deltas, final = realized_consequences(model, x, graph, config, trace, rows, target, spec, horizon)
            totals["replay_max_abs_deviation"] = max(totals["replay_max_abs_deviation"],
                                                     float(np.max(np.abs(final - np.asarray(probs)))))
            if feedback:
                depths = list(range(len(actions)))
                missing = [d for d in depths if d not in proxy.states]
                if missing:
                    raise RunnerError(f"no observer state at executed depths {missing[:5]}")
                features = np.stack([controller.features(proxy.states[d]) for d in depths])
                mean, sigma = controller._critic_batch(features, actions)
                for d, a in enumerate(actions):
                    state = proxy.states[d]
                    radius = float(controller._radius(controller.stratum(state.y, state.depth), a))
                    upper = float(mean[d]) + radius
                    decision_rows.append((int(eid), int(seed), d, a, float(mean[d]), radius, upper,
                                          float(deltas[d]), int(np.isfinite(upper) and upper <= 0),
                                          int(deltas[d] > IDENTITY_TOLERANCE)))
            losses, values = warning_trace(model, x, graph, config, controller, rows, target, spec, profile)
            summary = warning_summary(losses, values)
            warning_rows.append((int(eid), int(seed), summary["onset"], int(summary["has_onset"]),
                                 summary["first_warning"], summary["lead"]))
            losses_rows.append(np.concatenate([losses, values]))
        ensemble = ensemble_probabilities(members)
        values = sample_losses(ensemble, target_np, spec)
        valid = np.isfinite(values)
        unit_values.append(values[valid])
        unit_ids.append(np.asarray(units)[positions][valid])
        totals["inputs"] += 1
    if not unit_values:
        raise RunnerError(f"the {role} role holds no executed input")
    means, ids, counts = cluster_summary(np.concatenate(unit_values), np.concatenate(unit_ids))
    weights = np.ones(len(ids)) if independent_graphs(config) else counts.astype(np.float64)
    decisions = np.asarray(decision_rows, dtype=np.float64).reshape(-1, 10)
    finite = np.isfinite(decisions[:, 6]) if len(decisions) else np.zeros(0, bool)
    executed = action_counts.sum()
    shares = action_counts[action_counts > 0] / max(executed, 1)
    warnings = np.asarray(warning_rows, dtype=np.int64).reshape(-1, 6)
    with_onset = warnings[:, 3] == 1
    record = {"units": int(len(ids)), "loss_nats": float(np.average(means, weights=weights)),
              "action_counts": action_counts.tolist(), "action_entropy_nats": float(-(shares * np.log(shares)).sum()),
              "decisions": int(len(decisions)), "finite_predictions": int(finite.sum()),
              "false_safe": int((decisions[finite, 8] * decisions[finite, 9]).sum()) if len(decisions) else 0,
              "false_safe_rate": (float((decisions[finite, 8] * decisions[finite, 9]).mean())
                                  if finite.any() else None),
              "warning_pairs": int(len(warnings)), "pairs_with_onset": int(with_onset.sum()),
              "warning_lead_depths": float(warnings[with_onset, 5].mean()) if with_onset.any() else None,
              "terminal_model_available": bool(controller.terminal_available), "truncated": truncated,
              "work": totals, "profile": profile, "feedback_route": feedback}
    arrays = {"unit_id": ids, "unit_weight": weights, "unit_loss": means, "decisions": decisions,
              "warnings": warnings, "nominal_losses_and_warnings": np.asarray(losses_rows, dtype=np.float64),
              "decision_columns": np.asarray(["input", "seed", "depth", "action", "mean", "radius", "upper", "delta",
                                              "predicted_safe", "harmful"]),
              "warning_columns": np.asarray(["input", "seed", "onset", "has_onset", "first_warning", "lead"])}
    return record, arrays


def run_measure_job(plan, entry, output, *, device="cuda") -> Path:
    from cross_morse.learning import Controller
    from cross_morse.provenance import sha256_state_dict
    from .ensemble_grid import ParentContext
    output = Path(output)
    directory = output / "jobs" / entry["slug"]
    marker = directory / "measure.json"
    if marker.is_file():
        return marker
    parent = SealedRun(entry["parent"])
    check_identity(parent, entry["parent_identity"])
    role = entry["role"]
    check_role(parent, role)
    jobs = {j["job"]: j for j in plan["jobs"]}
    runs = {}
    for arm in entry["arms"]:
        runs[arm["name"]] = (parent.root if arm["source"] == "parent"
                             else open_lifecycle_run(jobs[arm["lifecycle_job"]], parent).root)
    started = utc_now()
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / "environment.json").exists():
        _dump(directory / "environment.json", environment_record(device))
    ctx = ParentContext(parent, device, through="evaluate" if role == "test" else "confirm")
    candidate = resolve_candidate(parent, entry["candidate"])
    seeds = [int(s) for s in parent.config["training"]["seeds"][:int(candidate["ensemble"])]]
    models = {s: ctx.plant(s) for s in seeds}
    frozen = {s: sha256_state_dict(m) for s, m in models.items()}
    labels, units = ctx.labels(role), ctx.units(role)
    profile = "full" if candidate.get("profile", "none") == "none" else candidate["profile"]
    records = {}
    for arm in entry["arms"]:
        name, run_root = arm["name"], runs[arm["name"]]
        path = directory / "arms" / f"{slug(name.replace(':', '__'))}.json"
        if path.is_file():
            records[name] = read_json(path)
            continue
        with installed(arm["variant"], run_dir=run_root):
            controller = Controller.load(run_root / "fit" / profile, "cpu")
            reference = dict(np.load(run_root / "explore" / "nominal_reference.npz", allow_pickle=False))
            record, arrays = measure_arm(ctx, models, seeds, candidate, controller, reference, role=role,
                                         labels=labels, units=units, horizon=entry["critic_horizon"],
                                         max_graphs=entry["max_graphs"])
            if arm["module"] == "observer" or name == "complete":
                record["latent_innovation"] = latent_innovation_rms(run_root, controller, profile)
                record["observer_parameters"] = observer_parameters(controller.observer)
        record["risk"] = assessment_rmse(run_root, profile)
        record.update(arm=dict(arm, run=Path(arm["run"]).name), run=Path(run_root).name,
                      controller_sha256=sha256(Path(run_root) / "fit" / profile / "controller.pt"))
        arrays_sha = _save_npz(path.with_suffix(".npz"), **arrays)
        record["arrays"] = {"file": path.with_suffix(".npz").name, "sha256": arrays_sha}
        _dump(path, record)
        records[name] = record
    if any(sha256_state_dict(m) != frozen[s] for s, m in models.items()):
        raise RuntimeError("a frozen plant changed during the measurement")
    contrasts = job_contrasts(directory, entry)
    _dump(directory / "contrasts.json", contrasts)
    _dump(marker, {"schema": "cm-ablation-measure-1", "job": entry["job"], "dataset": entry["dataset"],
                   "member": entry["member"], "role": role, "candidate": entry["candidate"],
                   "candidate_origin": entry["candidate_origin"], "arms": sorted(records),
                   "parent_identity": entry["parent_identity"], "parent_files_read": parent.inputs(),
                   "runtime_source_sha256": runtime_source_sha256(), "ablation_code": code_identity(),
                   "plan_sha256": sha256(output / "plan.json"), "started_utc": started, "finished_utc": utc_now()})
    return marker


# ============================================================================= contrasts and statistics (NumPy)
def _arm_arrays(directory, name):
    path = Path(directory) / "arms" / f"{slug(name.replace(':', '__'))}.json"
    record = read_json(path)
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    if sha256(path.with_suffix(".npz")) != record["arrays"]["sha256"]:
        raise RunnerError(f"{path.with_suffix('.npz')} differs from its recorded SHA-256")
    return record, arrays


def arm_contrast(complete: tuple, ablated: tuple, *, reps: int, seed: int) -> dict:
    """Ablated minus complete on the same units, inputs and members."""
    (c, ca), (a, aa) = complete, ablated
    if not np.array_equal(ca["unit_id"], aa["unit_id"]) or not np.array_equal(ca["unit_weight"], aa["unit_weight"]):
        raise RunnerError("the arms were scored on different units")
    effect = paired_effect(np.asarray(aa["unit_loss"]) - np.asarray(ca["unit_loss"]), ca["unit_weight"],
                           reps=reps, seed=seed)
    values = {"loss_change_nats": effect["mean"],
              "risk_rmse_change": a["risk"]["rmse"] - c["risk"]["rmse"],
              "action_entropy_change": a["action_entropy_nats"] - c["action_entropy_nats"]}
    if a["risk"]["rows_sha256"] != c["risk"]["rows_sha256"]:
        values.pop("risk_rmse_change")      # different assessment rows: no paired RMSE change
    if c.get("false_safe_rate") is not None and a.get("false_safe_rate") is not None:
        values["false_safe_rate_change"] = a["false_safe_rate"] - c["false_safe_rate"]
    cw, aw = ca["warnings"], aa["warnings"]
    keep, mismatches = np.zeros(len(cw), dtype=bool), None
    warned = c.get("terminal_model_available", True) and a.get("terminal_model_available", True)
    if warned and len(cw) and cw.shape == aw.shape and np.array_equal(cw[:, :2], aw[:, :2]):
        keep = (cw[:, 3] == 1) & (aw[:, 3] == 1)
        mismatches = int((cw[:, 2] != aw[:, 2]).sum())     # the onset is arm-independent; recorded, not assumed
        if keep.any():
            values["warning_lead_change_depths"] = float(np.mean(aw[keep, 5] - cw[keep, 5]))
    return {"values": values, "loss_change": effect, "pairs_with_onset": int(keep.sum()),
            "onset_mismatches": mismatches, "work": {"complete": c["work"], "ablated": a["work"]}}


def job_contrasts(directory, entry) -> dict:
    complete = _arm_arrays(directory, "complete")
    contrasts = {}
    for arm in entry["arms"]:
        if arm["name"] == "complete":
            continue
        result = arm_contrast(complete, _arm_arrays(directory, arm["name"]), reps=entry["bootstrap"]["reps"],
                              seed=entry["bootstrap"]["seed"])
        if arm["module"] == "observer":
            result["values"]["innovation_rmse"] = _arm_arrays(directory, arm["name"])[0]["risk"]["rmse"]
        contrasts[arm["name"]] = dict(result, module=arm["module"], arm=arm["arm"])
    if "observer" in {a["module"] for a in entry["arms"]}:
        contrasts["observer:conditioned"] = {"module": "observer", "arm": COMPLETE_ARMS["observer"],
                                             "values": {"innovation_rmse": complete[0]["risk"]["rmse"]},
                                             "definition": "the complete arm's controller (the runtime's observer)"}
    return {"schema": "cm-ablation-contrasts-1", "dataset": entry["dataset"], "member": entry["member"],
            "role": entry["role"], "candidate": entry["candidate"], "contrasts": contrasts,
            "bootstrap": entry["bootstrap"], "definitions": DEFINITIONS,
            "truncated": bool(complete[0].get("truncated"))}


def pool_values(quantity, values) -> float:
    values = [float(v) for v in values]
    if quantity == "innovation_rmse":
        return math.sqrt(math.fsum(v * v for v in values) / len(values))
    return math.fsum(values) / len(values)


def split_interval(values) -> tuple[float, float]:
    """Small-sample t interval over per-split values (the repository's repeated-split convention)."""
    if len(values) < 2:
        raise RunnerError("a repeated-split interval needs at least two splits")
    mean = math.fsum(values) / len(values)
    sd = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (len(values) - 1))
    half = T975.get(len(values) - 1, 1.96) * sd / math.sqrt(len(values))
    return mean - half, mean + half


# ============================================================================= comparator branch (NumPy)
def comparator_branch(entry) -> dict:
    """Branch on/off from a sealed input_gate run (clean inputs) and, when given, a negative_controls run."""
    from cross_morse.tasks import metric_score, task_spec
    from .artifacts import verify
    from .ensemble_grid import OFFICIAL_SCORE
    gate_dir = Path(entry["input_gate"]["path"])
    verify(gate_dir)
    if sha256(gate_dir / "ARTIFACT_MANIFEST.json") != entry["input_gate"]["manifest_sha256"]:
        raise RunnerError(f"{gate_dir}: changed since planning")
    config = read_json(gate_dir / "base_config.json")
    spec = task_spec(config)
    with np.load(gate_dir / "raw" / "predictions.npz", allow_pickle=False) as z:
        y, q_m, q_c, gated = z["y_true"], z["qM"], z["qC"], z["qD_gated"]
    failed = ~np.isfinite(q_m).all(axis=1)
    off = np.where(failed[:, None], q_c, q_m)
    scale, score_unit = OFFICIAL_SCORE[spec.metric]
    score_off, score_on = scale * float(metric_score(off, y, spec)), scale * float(metric_score(gated, y, spec))
    units = []
    with (gate_dir / "raw" / "deployment_units.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            units.append(("clean", row))
    controls = entry.get("negative_controls")
    if controls:
        control_dir = Path(controls["path"])
        verify(control_dir)
        if sha256(control_dir / "ARTIFACT_MANIFEST.json") != controls["manifest_sha256"]:
            raise RunnerError(f"{control_dir}: changed since planning")
        with (control_dir / "raw" / "units.csv").open(encoding="utf-8", newline="") as handle:
            units += [(row["condition"], row) for row in csv.DictReader(handle)]
    off_harm, on_harm = [], []
    for _, row in units:
        if row["comparator_loss"] == "":
            continue
        comparator = float(row["comparator_loss"])
        served = float(row["controller_loss"]) if row["controller_loss"] != "" else comparator
        off_harm.append(served - comparator)
        on_harm.append(float(row["deployed_loss"]) - comparator)
    if not off_harm:
        raise RunnerError("no scored unit for the comparator branch")
    return {"values": {"clean_score_change": score_off - score_on,
                       "worst_harm_change_nats": max(off_harm) - max(on_harm)},
            "clean_score": {"branch_off": score_off, "branch_on": score_on, "unit": score_unit,
                            "numerical_failures_served_by_qC": int(failed.sum())},
            "worst_harm": {"branch_off": max(off_harm), "branch_on": max(on_harm), "units": len(off_harm),
                           "populations": sorted({p for p, _ in units})},
            "inputs": [{"file": "input_gate/raw/predictions.npz", "sha256": sha256(gate_dir / "raw" / "predictions.npz")},
                       {"file": "input_gate/raw/deployment_units.csv",
                        "sha256": sha256(gate_dir / "raw" / "deployment_units.csv")}]
            + ([{"file": "negative_controls/raw/units.csv", "sha256": sha256(Path(controls["path"]) / "raw" / "units.csv")}]
               if controls else [])}


# ============================================================================= collection (NumPy)
def collect_study(output) -> Path:
    """results/<record set>.json: every measured (module, arm, dataset) value with the files it is read from."""
    output = Path(output).resolve()
    plan = load_study(output)
    if plan["study"] not in ("modules", "e18"):
        raise RunnerError("the factorial cells are collected by cm_experiments.factorial")
    groups, deposits = {}, {}

    def deposit(path):
        relative = Path(path).resolve().relative_to(output).as_posix()
        deposits[relative] = sha256(path)
        return {"file": relative, "sha256": deposits[relative]}

    truncated = False
    for entry in [j for j in plan["jobs"] if j["kind"] == "measure"]:
        directory = output / "jobs" / entry["slug"]
        if not (directory / "measure.json").is_file():
            raise RunnerError(f"{entry['dataset']}/{entry['member']}: measurement job {entry['job']} has not completed")
        contrasts = read_json(directory / "contrasts.json")
        truncated |= bool(contrasts.get("truncated"))
        files = [deposit(directory / "measure.json"), deposit(directory / "contrasts.json")]
        for path in sorted((directory / "arms").glob("*")):
            files.append(deposit(path))
        for name, contrast in contrasts["contrasts"].items():
            key = (contrast["module"], contrast["arm"], entry["dataset"])
            item = groups.setdefault(key, {"runs": [], "files": []})
            item["runs"].append({"member": entry["member"], "values": contrast["values"],
                                 "loss_change": contrast.get("loss_change")})
            item["files"] += files
    for entry in plan.get("posthoc", []):
        result = comparator_branch(entry)
        path = output / "posthoc" / f"{slug(entry['dataset'] + '__' + entry['member'])}.json"
        _dump(path, dict(result, schema="cm-ablation-comparator-branch-1", dataset=entry["dataset"],
                         member=entry["member"], role=entry["input_gate"]["role"],
                         sources={k: (None if entry.get(k) is None else {"run": Path(entry[k]["path"]).name,
                                                                         "manifest_sha256": entry[k]["manifest_sha256"]})
                                  for k in ("input_gate", "negative_controls")}))
        key = ("comparator_branch", "branch_off", entry["dataset"])
        item = groups.setdefault(key, {"runs": [], "files": []})
        item["runs"].append({"member": entry["member"], "values": result["values"]})
        item["files"].append(deposit(path))
    rows = []
    datasets = sorted({k[2] for k in groups})
    for (module, arm, dataset), item in sorted(groups.items()):
        quantities = sorted(set.intersection(*[set(r["values"]) for r in item["runs"]]))
        values = {q: math.fsum(r["values"][q] for r in item["runs"]) / len(item["runs"]) for q in quantities}
        detail = {"runs": item["runs"], "aggregation": ("single run" if len(item["runs"]) == 1 else
                                                        "mean over the official split runs")}
        if plan["study"] == "e18":
            loss = [r["loss_change"] for r in item["runs"]]
            if len(loss) == 1:
                low, high, count = loss[0]["lower"], loss[0]["upper"], loss[0]["units"]
            else:
                low, high = split_interval([x["mean"] for x in loss])
                count = sum(x["units"] for x in loss)
            values = {"policy_value_nats": values["loss_change_nats"], "interval_low_nats": low,
                      "interval_high_nats": high, "units": float(count),
                      **({"risk_rmse_change": values["risk_rmse_change"]} if "risk_rmse_change" in values else {})}
        rows.append({"module": module, "arm": arm, "dataset": dataset, "values": values, "detail": detail,
                     "files": sorted({f["file"]: f for f in item["files"]}.values(), key=lambda f: f["file"])})
    if plan["study"] == "modules" and len(datasets) > 1:
        # A signature term over several arms (the observer) reads exactly one row per arm: with several datasets
        # only its pooled rows are exported; the per-dataset values stay in the pooled row's detail.
        contract = json.loads((ROOT / "records" / "specs.json").read_text(encoding="utf-8"))["sets"]["module_ablations"]
        arms_terms = {m["module"] for m in contract["modules"] if any("arms" in t for t in m["signature"])}
        by_arm, kept = {}, []
        for row in rows:
            by_arm.setdefault((row["module"], row["arm"]), []).append(row)
        for (module, arm), members in sorted(by_arm.items()):
            if len(members) < 2:
                kept += members
                continue
            if module not in arms_terms:
                kept += members
            quantities = sorted(set.intersection(*[set(r["values"]) for r in members]))
            kept.append({"module": module, "arm": arm, "dataset": "pooled",
                         "values": {q: pool_values(q, [r["values"][q] for r in members]) for q in quantities},
                         "detail": {"datasets": {r["dataset"]: r["values"] for r in members},
                                    "per_dataset_rows_exported": module not in arms_terms,
                                    "aggregation": "equal-weight mean over datasets (innovation_rmse: square root of "
                                                   "the mean of the squares)"},
                         "files": sorted({f["file"]: f for r in members for f in r["files"]}.values(),
                                         key=lambda f: f["file"])})
        rows = kept
    for row in rows:
        for quantity, value in row["values"].items():
            if value is None or not math.isfinite(float(value)):
                raise RunnerError(f"{row['module']}/{row['arm']}/{row['dataset']}: {quantity} is not finite")
    results = {"schema": RESULTS_SCHEMA, "study": plan["study"], "record_set": plan["record_set"],
               "experiment": output.name, "plan_sha256": sha256(output / "plan.json"), "synthetic": plan["synthetic"],
               "role": plan["role"], "through": plan["through"], "complete": plan["complete"],
               "truncated": truncated, "rows": rows, "deposits": [{"path": k, "sha256": v} for k, v in
                                                                  sorted(deposits.items())],
               "definitions": DEFINITIONS, "units": QUANTITY_UNITS, "installer": plan["installer"],
               "collected_utc": utc_now()}
    path = output / "results" / f"{plan['record_set']}.json"
    _dump(path, results)
    return path


def module_results(path, expected_sha256=None) -> dict:
    """A collected modules study (results/module_ablations.json), with every deposited file verified."""
    path = Path(path)
    if path.is_dir():
        path = path / "results" / "module_ablations.json"
    if expected_sha256 is not None and sha256(path) != expected_sha256:
        raise RunnerError(f"{path} changed since it was attached")
    results = read_json(path)
    if results.get("schema") != RESULTS_SCHEMA or results.get("record_set") != "module_ablations":
        raise RunnerError(f"{path}: not the collected results of a modules study")
    base = path.parent.parent
    if sha256(base / "plan.json") != results["plan_sha256"]:
        raise RunnerError(f"{base}/plan.json changed since collection")
    for item in results["deposits"]:
        if sha256(base / item["path"]) != item["sha256"]:
            raise RunnerError(f"{base / item['path']} changed since collection")
    results["base"] = str(base)
    return results


# ============================================================================= run and export
def run_study(output, *, job=None, data_root=None, device="cuda", threads=16, python=sys.executable,
              trust=False) -> list[str]:
    if not trust:
        raise RunnerError("loading local checkpoints requires explicit trust (--trust-local-checkpoints)")
    output = Path(output).resolve()
    plan = load_study(output)
    done = []
    factorial_jobs = []
    if plan["study"] == "factorial":
        from .factorial import load_plan as factorial_plan, run_factorial
        factorial_jobs = factorial_plan(output / "factorial", "factorials")["jobs"]
    if job is not None and int(job) >= len(plan["jobs"]):        # factorial measurement jobs follow the lifecycles
        index = int(job) - len(plan["jobs"])
        if not 0 <= index < len(factorial_jobs):
            raise RunnerError(f"job index {job} outside 0..{len(plan['jobs']) + len(factorial_jobs) - 1}")
        return [str(p) for p in run_factorial(output / "factorial", job=index, device=device, trust=trust)]
    for entry in selected_jobs(plan, job):
        if entry["kind"] == "lifecycle":
            done.append(str(run_lifecycle_job(output, entry, data_root=data_root, device=device, threads=threads,
                                              python=python)))
        else:
            done.append(str(run_measure_job(plan, entry, output, device=device)))
    if factorial_jobs and job is None:
        done += [str(p) for p in run_factorial(output / "factorial", device=device, trust=trust)]
    return done


def export_e18(output, *, root=None, replace=False, allow_partial=False, hardware=None, period=None,
               command=None) -> Path:
    from .ensemble_grid import SPECS_PATH, provenance_hardware
    output = Path(output).resolve()
    plan = load_study(output, "e18")
    specs = json.loads(Path(SPECS_PATH).read_text(encoding="utf-8"))
    if STUDIES["e18"] not in specs["sets"]:
        raise RunnerError("records/specs.json registers no spectral_policy_value contract; add "
                          "cm_experiments.ablation_lifecycle.E18_CONTRACT as sets.spectral_policy_value first")
    results = read_json(collect_study(output))
    if results["truncated"] and not allow_partial:
        raise RunnerError("a measurement truncated by max_graphs is not a complete measurement")
    staging = Path(tempfile.mkdtemp(prefix="cm-e18-export-"))
    try:
        files = {f"code/{name}": ROOT / name for name in (MODULE_FILE, SCRIPT_FILE, "cm_experiments/ensemble_grid.py")}
        for item in results["deposits"]:
            files[f"raw/{item['path']}"] = output / item["path"]
        summary_path = staging / "e18_policy_value.json"
        summary_path.write_text(json.dumps(json_safe({k: v for k, v in results.items() if k != "deposits"}),
                                           indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        files["raw/e18_policy_value.json"] = summary_path
        rows, environments, starts, ends = [], [], [], []
        for i, row in enumerate(results["rows"]):
            for quantity in E18_CONTRACT["quantities"]:
                if quantity in row["values"]:
                    unit = "units" if quantity == "units" else "nats"
                    rows.append({"dataset": row["dataset"], "quantity": quantity, "value": row["values"][quantity],
                                 "unit": unit, "source_file": "raw/e18_policy_value.json",
                                 "locator": f"json:/rows/{i}/values/{quantity}"})
        for entry in [j for j in plan["jobs"] if j["kind"] == "measure"]:
            measured = read_json(output / "jobs" / entry["slug"] / "measure.json")
            environments.append(read_json(output / "jobs" / entry["slug"] / "environment.json"))
            starts.append(measured["started_utc"])
            ends.append(measured["finished_utc"])
        provenance = {
            "experiment": "E18 policy value over matched confidence/history controls (Supplementary Table S36)",
            "produced_by": "scripts/run_ablation.py e18-policy plan/run/export (cm_experiments/ablation_lifecycle.py); "
                           "variant lifecycles by run_cross_morse.main with the variant installed",
            "hardware": hardware or provenance_hardware(environments), "period": period or f"{min(starts)}/{max(ends)}",
            "protocol": plan["protocol"], "synthetic": plan["synthetic"], "role": plan["role"],
            "complete": plan["complete"], "through": plan["through"], "plan_sha256": sha256(output / "plan.json"),
            "installer": plan["installer"], "definitions": E18_CONTRACT["quantities"]}
        return write_record_set(STUDIES["e18"], rows, files, provenance, root=root, replace=replace,
                                allow_partial=allow_partial or results["truncated"], command=command,
                                inputs=[{"file": f"results/{STUDIES['e18']}.json",
                                         "sha256": sha256(output / "results" / f"{STUDIES['e18']}.json")}],
                                generated={"code/plan.json": public_plan(output)})
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def export_modules(output, sweep, *, root=None, hardware, period, replace=False, parents=None, argv=None) -> Path:
    """The ten-module record set: the sealed-contrast modules of cm_experiments.sweeps on the study's parents and
    the seven measured by the variant lifecycles, in one scripts/export_sweep_records.py export."""
    from scripts import export_sweep_records
    from .sweeps import collect, plan_sweep
    from .ensemble_grid import RECORDS
    output = Path(output).resolve()
    plan = load_study(output, "modules")
    collect_study(output)
    path = plan_sweep("module_ablations", [Path(p) for p in (parents or plan["parent_runs"])], Path(sweep),
                      ablations=[output])
    return export_sweep_records.export(collect(path), root=Path(root or RECORDS), hardware=hardware, period=period,
                                       replace=replace, argv=list(argv or []))


def export_factorial_fit(output, **kwargs) -> Path:
    from .factorial import export_factorial
    output = Path(output).resolve()
    load_study(output, "factorial")
    return export_factorial(output / "factorial", **kwargs)


if __name__ == "__main__":
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from cm_experiments.ablation_lifecycle import lifecycle_main as _lifecycle_main  # the importable module
    raise SystemExit(_lifecycle_main())
