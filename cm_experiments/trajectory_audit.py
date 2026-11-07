"""Trajectory-tube audit of main Proposition 2 and depth diagnostics of the plants and deployed rules.

Record sets ``tube_audit`` (Supplementary Table S22, Supplementary Figure S17e, main Figure 2a)
and ``depth_diagnostics`` (Supplementary Figure S14).  The eight-stage lifecycle records none of
their quantities; this module executes them on sealed lifecycle runs with the released runtime
(``cross_morse.policies.execute_policy``, ``FrozenPlant.step`` and ``native_step``, the
``MeasurementBank`` sensors, the fitted ``Controller``) and exports the record sets.

Workflow (``scripts/run_trajectory_audit.py {tube|depth} {plan|run|export}``):

    plan    sealed parent runs (Roman-empire: the campaign directory of split runs) and the declared
            settings -> <output>/plan.json with every parent identity; nothing is executed
    run     one job per parent run, or per declared slice of its audited trajectories / role graphs
            (``chunk_size``); ``--job K`` for array jobs; the tube audit resumes per trajectory
    export  -> records/<set> (summary.csv, raw/, code/, provenance.json, MANIFEST.json); slices of one
            run are merged before any statistic

Planning, the statistics and the export are torch-free; execution imports the runtime lazily.

tube_audit: the plug-in trajectory tube of main Proposition 2
-------------------------------------------------------------
Audited trajectories.  The D_explore trajectories of the fit stage's assessment partition
(``fit/<profile>/calibration_partition.json``, role ``assessment``: trajectories that fitted no
observer, critic, sequence model or calibration radius), at most ``max_trajectories`` per run by a
seeded subsample.  An audited trajectory names one episode: its plant checkpoint (the trajectory's
``seed_index``) and, on the multigraph benchmarks, its development graph; the episode's D_explore
labels score the loss (read in the explore stage's label scope after that stage is sealed).

Predicted and realized runs.  The run's rule (default the deployed rule q_D; the frozen consequence
models of ``fit/<profile>``) is executed on the episode twice with ``execute_policy``:

    predicted run   the registered system: features x, graph operator P;
    realized run    the same rule under the declared disturbance: feature rows
                    x_r + r_x ||x_r|| z_r / ||z_r|| (z standard normal) and edge weights
                    w_e (1 + r_w xi_e) with xi uniform on [-1, 1), equal on both directions of a pair
                    (a splitmix64 draw keyed by the unordered pair and a per-trajectory key).

Each run decides its own actions, recovery origins and last-safe anchors from its own measurements.
An exact joint replay of the two recorded traces (main Proposition 1, the runtime's replay
semantics) gives the state pairs (H_l, H~_l); its outputs must reproduce both closed-loop outputs.
In the notation of Proposition 2 the tilde is the predicted run, e_l = ||H_l - H~_l||_{inf,2}.

Plug-in constants, evaluated on the predicted run only (so the tube is a prediction that the
realized run is tested against), per executed site l with action a = (rho, nu, alpha):

    shared, edge-weighted and heterophily plants, transport site (the form of Eq. (8)):
        beta_l = L_LN,l (1 - alpha) L_phi ||W||_2,   L_LN,l = ||gamma||_inf / sqrt(min_r var(U_l,r) + eps)
        at the LayerNorm input U_l of the predicted step (read by a forward pre-hook), L_phi the
        activation's global Lipschitz constant, ||W||_2 the spectral norm of the shared update
        (heterophily: ||W_ego||_2 + ||W_shared||_2);
        eps^op_l = h(a, l) ||P - P~||_inf, h(a, l) the executed hop moment (sum_t t c_t of the
        polynomial T_a(P); ``site_transport_accounting``), which bounds ||T_a(P) - T_a(P~)||_inf;
        R_l = L_LN,l alpha, e^ref_l = e_0 (the source reference of every retention action);
    source reset A8:     beta_l = 0, R_l = 1, e^ref_l = e_0;
    last-safe reset A9:  beta_l = 0, R_l = 1, e^ref_l = the tube at the predicted anchor depth;
    designated-site plants (GraphGPS-style, linear GINE), inserted site:
        A0 (literal bypass) beta_l = 1, eps = R = 0;  other actions (exact for the linear form)
        beta_l = 1 - nu alpha, beta_l eps^op_l = (1 - alpha) h(a, l) ||P - P~||_inf, R_l = nu alpha;
    designated-site plants, native block (no closed form): beta_l and R_l are sampled secant
        quotients of the block in the state and in the source reference (``probes`` random
        directions plus one row-coherent direction, at radius ``probe_relative`` x the largest row
        norm of the predicted state, resp. of the predicted source), eps^op_l = ||P - P~|| (A0,
        aggregation P H) or (1 - alpha) h(a, l) ||P - P~||.

M_l = ||H~_l||_{inf,2}.  The tube is tau_0 = e_0, tau_{l+1} = beta_l [tau_l + M_l eps^op_l] +
R_l e^ref_l; ``cm_theory.stability.unrolled_bound`` gives the same tau_L in product-sum form and
``logit_loss_bound`` the per-trajectory loss bound sqrt(2) ||W_head||_2 tau_L (two-layer heads:
||W_out||_2 L_phi ||W_hidden||_2).  Exported per dataset (pooled over its runs):

    beta_median, beta_p95        median / 95th percentile of beta_l over all executed sites of all
                                 audited predicted runs
    m_eps_op, reference_term     median over depths l of the per-depth median over audited
                                 trajectories of M_l eps^op_l, resp. R_l e^ref_l
    tube_coverage                fraction of audited trajectories with e_l <= tau_l + tol_l at every
                                 depth (tol_l = registered relative state tolerance x the larger state
                                 norm; a failed or differently terminated realized run is uncovered)
    mode_stable_coverage         fraction whose two runs have equal actions, decision origins
                                 (policy or recovery path), last-safe anchors and terminal depth
    final_bound_nats             95th percentile over audited trajectories of the per-trajectory bound
    observed_error_*_nats        median / 95th percentile of |loss(predicted) - loss(realized)| at
                                 the final depth over audited trajectories with a realized output

depth_diagnostics: Supplementary Figure S14
-------------------------------------------
Depths are numbers of executed propagation steps on the registered grid {1, 2, 4, ..., L} of
``train/entry_initial.json``; the first plotted depth is the first grid depth >= 2 (as the display).

    source_alignment_change   reference plant only: nominal trajectory (``native_step``) of every
                              registered checkpoint on D_fit validation (``fit_val``), full-state
                              MeasurementBank (sketch 0, period 1) pooled over the fit_val rows
                              (independent graphs: per graph, then averaged), mean over the three
                              retained bands of c_lb, mean over checkpoints (and split runs), minus
                              the value at the first plotted depth
    policy_loss_change        the declared policy (default q_D) at its ensemble size, each member
                              truncated after its first d executed actions (exact replay of its own
                              action/anchor trace; a stopped policy keeps its output), ensemble mean,
                              unit-weighted CE/BCE on the declared role (default fit_val), minus the
                              value at the first plotted depth; split runs averaged
    first_intervention_*      quartiles over executions (ensemble member x input) of the first
                              propagation depth whose executed action is not A0 (site index + 1);
                              an execution without one is +inf, and a quartile at +inf is not exported
    supported_deterioration   from the sealed entry audit (per-unit depth losses of the ensemble plant
                              on fit_val): first grid depth after the best-validation-loss depth
                              whose paired unit increase has a central 95% interval above zero
                              (registered paired whole-unit bootstrap; split runs: t interval over
                              the splits about the split-mean curve's best depth)
    transport_fraction        executed sites of each depth phase (site l in phase floor(3 l / L):
                              early, middle, late) carrying a transport action A1-A7, pooled over
                              executions

Rows without a depth or a phase carry ``all`` in that key (summary.csv keys may not be empty).
"""
from __future__ import annotations

import contextlib
import copy
import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np

from .artifacts import sha256, write_json
from .ensemble_grid import (ROOT, ParentContext, RunnerError, SealedRun, candidate_identity, check_identity,
                            check_role, create_plan, ensemble_probabilities, environment_record, job_directory,
                            json_safe, load_plan, member_runs, provenance_hardware, provenance_period, public_plan,
                            read_json, registered_constants, runtime_source_sha256, save_arrays, selected_jobs, slug,
                            utc_now, write_record_set)

TUBE, DEPTH = "tube_audit", "depth_diagnostics"
CODE_FILES = ("cm_experiments/trajectory_audit.py", "scripts/run_trajectory_audit.py",
              "cm_experiments/ensemble_grid.py", "cm_experiments/artifacts.py",
              "cm_theory/stability.py", "cm_theory/_util.py")
TABLES = {TUBE: ["S22", "Fig. S17e", "Fig. 2a"], DEPTH: ["Fig. S14"]}
POLICIES = {"deployed": "admit", "confirmed": "confirm", "provisional": "select"}
STAGE_ORDER = ("prepare", "train", "explore", "fit", "select", "confirm", "admit", "evaluate")
ALL = "all"
PHASES = ("early", "middle", "late")
SHARED_BACKBONES = ("shared_transport", "edge_weighted_transport", "heterophily")
DESIGNATED_BACKBONES = ("gps", "linear_gine")
KINDS = ("formula", "source_reset", "anchor_reset", "inserted_identity", "inserted_linear", "native_secant")
REF_NONE, REF_SOURCE, REF_ANCHOR = 0, 1, 2
SEED_OFFSET = 4099                   # audit seed = run seed + offset (declared in the plan)
REPLAY_TOLERANCE = {"fp32": 1e-4, "bf16": 1e-2}   # max |p| difference replay vs closed loop

TUBE_UNITS = {"beta_median": "ratio", "beta_p95": "ratio", "m_eps_op": "state row norm",
              "reference_term": "state row norm", "tube_coverage": "fraction", "mode_stable_coverage": "fraction",
              "final_bound_nats": "nats", "observed_error_median_nats": "nats", "observed_error_p95_nats": "nats"}
TUBE_DEFINITIONS = {
    "beta_median": "median of the plug-in beta_l over every executed site of every audited predicted run",
    "beta_p95": "95th percentile (linear) of the same beta_l values",
    "m_eps_op": "median over depths of the per-depth median over audited trajectories of M_l eps^op_l "
                "(M_l = ||H~_l||_inf,2, eps^op_l the plug-in operator error)",
    "reference_term": "median over depths of the per-depth median over audited trajectories of R_l e^ref_l",
    "tube_coverage": "fraction of audited trajectories whose realized state error stays inside their plug-in tube "
                     "at every depth (registered relative state tolerance)",
    "mode_stable_coverage": "fraction of audited trajectories whose predicted and realized runs have equal actions, "
                            "decision origins, last-safe anchors and terminal depth",
    "final_bound_nats": "95th percentile over audited trajectories of the plug-in bound sqrt(2) ||W_head||_2 tau_L "
                        "(unrolled_bound, logit_loss_bound)",
    "observed_error_median_nats": "median over audited trajectories of |predicted minus realized D_explore loss| at "
                                  "the final depth",
    "observed_error_p95_nats": "95th percentile (linear) of the same observed loss error"}
DEPTH_UNITS = {"source_alignment_change": "cosine", "policy_loss_change": "nats",
               "first_intervention_depth_q25": "propagation depth", "first_intervention_depth_median": "propagation depth",
               "first_intervention_depth_q75": "propagation depth", "supported_deterioration_depth": "propagation depth",
               "transport_fraction": "fraction"}


# ============================================================================= shared helpers (torch-free)
def own_source_identity() -> dict:
    return {name: sha256(ROOT / name) for name in CODE_FILES if (ROOT / name).is_file()}


def code_deposit() -> dict:
    return {f"code/{name}": ROOT / name for name in CODE_FILES if (ROOT / name).is_file()}


def policy_candidate(run: SealedRun, policy: str) -> tuple[dict, str]:
    """The declared policy of a sealed run with its registry index, and where it was read."""
    if policy not in POLICIES:
        raise RunnerError(f"policy must be one of {sorted(POLICIES)}")
    run.require(POLICIES[policy])
    if policy == "deployed":
        candidate = run.deployed()
        if candidate is None:
            raise RunnerError(f"{run.root.name}: policy_lock.json is missing")
        return candidate, "policy_lock.json deployed_index (q_D)"
    if policy == "confirmed":
        return run.confirmed(), "confirm/decision.json confirmed_index (q_M)"
    index = int(run.json("select/decision.json")["provisional_index"])
    return dict(run.candidates()[index], registry_index=index), "select/decision.json provisional_index"


def planned_candidate(run: SealedRun, policy: str, planned: dict) -> dict:
    candidate, _ = policy_candidate(run, policy)
    if candidate_identity(candidate) != planned:
        raise RunnerError(f"{run.root.name}: the {policy} policy differs from the plan")
    return candidate


def latest_stage(*stages: str) -> str:
    return max(stages, key=STAGE_ORDER.index)


def first_plotted_depth(grid) -> int:
    """The display's first plotted depth: the first registered grid depth >= 2."""
    for depth in grid:
        if int(depth) >= 2:
            return int(depth)
    raise RunnerError("the depth grid has no depth >= 2 to plot from")


def registered_grid(depth: int) -> list[int]:
    """The geometric depth grid of cross_morse.training.audit_training."""
    depth = int(depth)
    return sorted(set([1, depth] + [2 ** i for i in range(depth.bit_length()) if 2 ** i <= depth]))


def finite_or_none(value):
    value = None if value is None else float(value)
    return value if value is not None and math.isfinite(value) else None


@contextlib.contextmanager
def record_coverage(set_name: str, datasets):
    """The planned dataset coverage for the scientific-audit check of ``write_record_set``.

    ``cm_campaigns.contract.RECORD_SETS`` does not list these sets, and the writer's fallback rule for
    unlisted sets (every key row carries every quantity) contradicts the sparse depth_diagnostics layout
    its contract prescribes.  An entry registered there takes precedence; otherwise the plan's datasets
    are the declared coverage, for this one in-process export only.
    """
    from cm_campaigns.contract import RECORD_SETS
    added = set_name not in RECORD_SETS
    if added:
        RECORD_SETS[set_name] = {"tables": list(TABLES[set_name]), "datasets": sorted(datasets),
                                 "analyzer": set_name}
    try:
        yield
    finally:
        if added:
            RECORD_SETS.pop(set_name, None)


def _dataset_entries(plan: dict) -> list[tuple[dict, list[dict]]]:
    return [(dataset, [j for j in plan["jobs"] if j["dataset"] == dataset["dataset"]]) for dataset in plan["datasets"]]


def _plan_sha(output) -> str:
    return sha256(Path(output) / "plan.json")


def _parents(parents) -> list[list[SealedRun]]:
    if not parents:
        raise RunnerError("give at least one --parent run")
    groups, seen = [], set()
    for parent in parents:
        runs = [SealedRun(m) for m in member_runs(parent)]
        dataset = runs[0].dataset
        if len({r.dataset for r in runs}) != 1 or dataset in seen:
            raise RunnerError(f"{parent}: one parent (run or campaign) per dataset")
        seen.add(dataset)
        groups.append(runs)
    if len({runs[0].synthetic for runs in groups}) != 1:
        raise RunnerError("synthetic and benchmark parents cannot share one plan")
    return groups


def chunked_jobs(job: dict, key: str, size: int, *, first_only=()) -> list[dict]:
    """Split one run's job into consecutive slices of ``job[key]`` (one scheduler task each).

    ``size`` 0 or a list that fits keeps one job.  Slices keep the run's identity under ``run_member``,
    get their own ``member`` (and hence job directory) and keep the fields in ``first_only`` on the first
    slice only (emptied elsewhere).
    """
    items = job[key]
    if not size or items is None or len(items) <= int(size):
        return [dict(job, run_member=job["member"], chunk={"index": 0, "count": 1})]
    parts = [items[i:i + int(size)] for i in range(0, len(items), int(size))]
    out = []
    for k, part in enumerate(parts):
        piece = dict(job, member=f"{job['member']}.part{k:03d}", run_member=job["member"],
                     chunk={"index": k, "count": len(parts)}, **{key: part})
        for field in first_only:
            if k:
                piece[field] = type(job[field])()
        out.append(piece)
    return out


def _dataset_declaration(parent_runs: list[SealedRun]) -> dict:
    dataset = parent_runs[0].dataset
    parent = parent_runs[0].root if len(parent_runs) == 1 else parent_runs[0].root.parent
    return {"dataset": dataset, "slug": slug(dataset), "parent": str(parent), "synthetic": parent_runs[0].synthetic,
            "members": [r.root.name for r in parent_runs],
            "aggregation": "pooled or averaged over the official split runs" if len(parent_runs) > 1 else "single run"}


# ============================================================================= tube_audit: planning (torch-free)
def explore_entries(run: SealedRun) -> list[dict]:
    """One entry per sealed D_explore trajectory, from its metadata file."""
    directory = run.root / "explore"
    entries = []
    for meta in sorted(directory.glob("trajectory_*.json")):
        data = json.loads(meta.read_text(encoding="utf-8"))
        arrays = meta.with_suffix(".npz")
        if not arrays.is_file():
            raise RunnerError(f"{run.root.name}: {arrays.name} is missing")
        entries.append({"trajectory": int(data["trajectory"]), "seed_index": int(data["seed_index"]),
                        "graph_id": None if data.get("graph_id") is None else int(data["graph_id"]),
                        "records": int(data["records"]), "file": f"explore/{arrays.name}",
                        "meta_file": f"explore/{meta.name}"})
    manifest = run.json("explore/manifest.json")
    if not entries or len(entries) != int(manifest["trajectories"]):
        raise RunnerError(f"{run.root.name}: the D_explore trajectories differ from explore/manifest.json")
    return entries


def holdout_groups(entries: list[dict], collection: bool) -> list[int]:
    """``cross_morse.learning._holdout_groups`` with one row per trajectory (its rows share the keys).

    Independent graphs: the sorted unique graph identifiers; otherwise the lexicographically sorted
    unique (seed_index, trajectory) pairs.
    """
    keys = [e["graph_id"] for e in entries] if collection else [(e["seed_index"], e["trajectory"]) for e in entries]
    if collection and any(k is None or k < 0 for k in keys):
        raise RunnerError("independent-graph trajectories need nonnegative graph identifiers")
    order = {key: index for index, key in enumerate(sorted(set(keys)))}
    return [order[key] for key in keys]


def audited_trajectories(run: SealedRun, profile: str, *, limit: int, seed: int) -> tuple[list[dict], dict]:
    """The assessment-partition trajectories of the fitted profile, optionally a seeded subsample."""
    partition_file = f"fit/{profile}/calibration_partition.json"
    partition = run.json(partition_file)
    entries = explore_entries(run)
    collection = any(e["graph_id"] is not None for e in entries)
    groups = holdout_groups(entries, collection)
    assessment = {int(g) for g in partition["groups"]["assessment"]}
    chosen = [dict(e) for e, g in zip(entries, groups) if g in assessment]
    if not chosen:
        raise RunnerError(f"{run.root.name}: the assessment partition holds no trajectory")
    rows = sum(e["records"] for e in chosen)
    if rows != int(partition["rows"]["assessment"]):
        raise RunnerError(f"{run.root.name}: the trajectory grouping does not reproduce the fit-stage partition "
                          f"({rows} rows against {partition['rows']['assessment']})")
    available = len(chosen)
    if limit and available > limit:
        picks = np.sort(np.random.default_rng(seed).choice(available, size=int(limit), replace=False))
        chosen = [chosen[int(i)] for i in picks]
    seeds = [int(s) for s in run.config["training"]["seeds"]]
    for item in chosen:
        item["seed"] = seeds[item["seed_index"]]
        item["sha256"] = sha256(run.root / item["file"])
        item["meta_sha256"] = sha256(run.root / item["meta_file"])
    declaration = {"partition_file": partition_file, "partition_sha256": run.read[partition_file],
                   "partition_protocol": partition.get("protocol"), "group_definition": partition.get("group_definition"),
                   "assessment_groups": len(assessment), "assessment_trajectories": available,
                   "audited_trajectories": len(chosen), "limit": int(limit), "subsample_seed": int(seed),
                   "rule": "assessment role of the fit-stage calibration partition (no model, observer or radius "
                           "was fitted on it); seeded subsample without replacement when above the limit"}
    return chosen, declaration


def plan_tube_audit(parents, output, *, policy="deployed", max_trajectories=64, input_relative=None,
                    edge_relative=None, precision="fp32", probes=4, probe_relative=1e-3, chunk_size=0,
                    seed_offset=SEED_OFFSET) -> Path:
    if policy not in POLICIES:
        raise RunnerError(f"policy must be one of {sorted(POLICIES)}")
    if precision not in ("fp32", "registered"):
        raise RunnerError("precision must be fp32 or registered")
    if int(max_trajectories) < 0 or int(probes) < 1 or not float(probe_relative) > 0 or int(chunk_size) < 0:
        raise RunnerError("max_trajectories must be >= 0 (0: all), probes >= 1, probe_relative > 0 and "
                          "chunk_size >= 0")
    tolerances = registered_constants()["precision_tolerances_table_S5c"]
    default = float(tolerances["bf16"]["state_relative"])
    r_x = default if input_relative is None else float(input_relative)
    r_w = default if edge_relative is None else float(edge_relative)
    if not (0 <= r_x and 0 <= r_w < 1) or (r_x == 0 and r_w == 0):
        raise RunnerError("declare a nonzero disturbance: input_relative >= 0, 0 <= edge_relative < 1, not both zero")
    datasets, jobs = [], []
    for runs in _parents(parents):
        for run in runs:
            run.require("explore")
            run.require("fit")
            candidate, source = policy_candidate(run, policy)
            profile = candidate.get("profile") if candidate.get("profile") in ("full", "lean") else "full"
            audit_seed = int(run.config["seed"]) + int(seed_offset)
            trajectories, selection = audited_trajectories(run, profile, limit=int(max_trajectories), seed=audit_seed)
            executed = "fp32" if precision == "fp32" else str(run.config["model"]["precision"])
            job = {"dataset": run.dataset, "slug": slug(run.dataset), "member": run.root.name,
                   "run": str(run.root), "identity": run.identity(), "policy": candidate_identity(candidate),
                   "policy_source": source, "profile": profile, "depth": int(run.config["model"]["depth"]),
                   "registered_precision": run.config["model"]["precision"], "executed_precision": executed,
                   "coverage_relative_tolerance": float(tolerances[executed]["state_relative"]),
                   "replay_tolerance": REPLAY_TOLERANCE[executed], "audit_seed": audit_seed,
                   "selection": selection, "trajectories": trajectories}
            for piece in chunked_jobs(job, "trajectories", int(chunk_size)):
                jobs.append(dict(piece, job=len(jobs)))
        datasets.append(_dataset_declaration(runs))
    return create_plan(output, {
        "experiment": "Plug-in trajectory-tube audit (main Proposition 2)", "record_set": TUBE,
        "policy": policy, "precision": precision, "datasets": datasets, "jobs": jobs,
        "perturbation": {"input_relative": r_x, "edge_weight_relative": r_w,
                         "input_rule": "x_r + r_x ||x_r|| z_r/||z_r||, z standard normal (numpy, per trajectory)",
                         "edge_rule": "w_e (1 + r_w xi_e), xi uniform on [-1, 1): splitmix64 of the unordered "
                                      "pair and a per-trajectory key, so both directions share one draw",
                         "default_source": "paper/registered_constants.json precision_tolerances_table_S5c.bf16."
                                           "state_relative (used when a magnitude is not declared)",
                         "seed_rule": "numpy SeedSequence([audit_seed, trajectory]).spawn(3): the input, edge-key "
                                      "and secant-probe streams of that trajectory (independent of job slicing)"},
        "probes": {"random": int(probes), "coherent": 1, "relative_radius": float(probe_relative),
                   "scope": "native blocks of the designated-site plants only"},
        "chunk_size": int(chunk_size),
        "protocol": ("the declared rule executed with execute_policy on the episode of every audited D_explore "
                     "trajectory (assessment partition): predicted run on the registered system, realized run under "
                     "the declared input and edge-weight disturbance; exact joint replay of both traces; plug-in "
                     "Proposition 2 constants on the predicted run; tube, coverage, mode stability, bounds and "
                     "D_explore loss errors (module docstring)"),
        "trajectory_audit_source_sha256": own_source_identity()})


# ============================================================================= tube_audit: statistics (torch-free)
def activation_lipschitz(name: str) -> float:
    """Global Lipschitz constant of the plant activation (sup |phi'| on a fine grid; ReLU exactly 1)."""
    if name == "relu":
        return 1.0
    x = np.linspace(-12.0, 12.0, 2_400_001)
    if name == "silu":
        s = 1.0 / (1.0 + np.exp(-x))
        derivative = s * (1.0 + x * (1.0 - s))
    elif name == "gelu":                      # exact (erf) GELU, torch.nn.functional.gelu's default
        from scipy.special import erf
        derivative = 0.5 * (1.0 + erf(x / math.sqrt(2.0))) + x * np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    else:
        raise RunnerError(f"no Lipschitz constant for the activation {name!r}")
    return float(np.max(np.abs(derivative)))


def layer_norm_lipschitz(gamma_max: float, min_variance: float, eps: float) -> float:
    """Local Lipschitz bound of a row LayerNorm at a state: ||gamma||_inf / sqrt(min_r var_r + eps).

    The row Jacobian is diag(gamma)/s (I - 11'/d - u u'/(d s^2)) with s^2 = var + eps; the bracket is a
    projection minus a rank-one term of norm var/(var + eps) <= 1, so its spectral norm is at most
    ||gamma||_inf / s, and the (inf,2) norm takes the largest row value.
    """
    if not (gamma_max >= 0 and min_variance >= 0 and eps > 0):
        raise RunnerError("LayerNorm constants must be nonnegative with eps > 0")
    return float(gamma_max) / math.sqrt(float(min_variance) + float(eps))


def site_terms(raw: dict, plant: dict, delta_p: float) -> dict:
    """Plug-in constants of Proposition 2 for every executed site of one predicted run.

    ``raw`` holds per-site arrays (kind, action, alpha, nu, hops, l_ln, beta_secant, ref_secant, M,
    anchor) and ``plant`` the checkpoint constants (activation_lipschitz, weight_norm).
    """
    kind = np.asarray(raw["kind"], dtype=np.int64)
    n = len(kind)
    alpha, nu, hops = (np.asarray(raw[k], dtype=np.float64) for k in ("alpha", "nu", "hops"))
    action = np.asarray(raw["action"], dtype=np.int64)
    beta, eps, R = np.zeros(n), np.zeros(n), np.zeros(n)
    ref = np.full(n, REF_NONE, dtype=np.int64)
    for l in range(n):
        name = KINDS[int(kind[l])]
        if name == "formula":
            l_ln = float(raw["l_ln"][l])
            if not math.isfinite(l_ln) or plant.get("weight_norm") is None:
                raise RunnerError("a formula site needs its LayerNorm constant and the update's spectral norm")
            beta[l] = l_ln * (1.0 - alpha[l]) * float(plant["activation_lipschitz"]) * float(plant["weight_norm"])
            eps[l] = hops[l] * delta_p
            R[l], ref[l] = l_ln * alpha[l], REF_SOURCE
        elif name == "source_reset":
            R[l], ref[l] = 1.0, REF_SOURCE
        elif name == "anchor_reset":
            R[l], ref[l] = 1.0, REF_ANCHOR
        elif name == "inserted_identity":
            beta[l] = 1.0
        elif name == "inserted_linear":
            beta[l] = 1.0 - nu[l] * alpha[l]
            eps[l] = (1.0 - alpha[l]) * hops[l] * delta_p / beta[l]
            R[l], ref[l] = nu[l] * alpha[l], REF_SOURCE
        elif name == "native_secant":
            beta[l], R[l] = float(raw["beta_secant"][l]), float(raw["ref_secant"][l])
            if not (math.isfinite(beta[l]) and math.isfinite(R[l])):
                raise RunnerError("a native designated site needs its sampled secant constants")
            eps[l] = delta_p if action[l] == 0 else (1.0 - alpha[l]) * hops[l] * delta_p
            ref[l] = REF_SOURCE
        else:  # pragma: no cover - KINDS is exhaustive
            raise RunnerError(f"unknown site kind {kind[l]}")
    return {"beta": beta, "eps_op": eps, "R": R, "ref": ref}


def forward_tube(beta, M, eps, R, ref, anchors, e0: float) -> tuple[np.ndarray, np.ndarray]:
    """tau_0 = e_0, tau_{l+1} = beta_l [tau_l + M_l eps_l] + R_l e^ref_l with the reference resolved per site."""
    beta, M, eps, R = (np.asarray(v, dtype=np.float64) for v in (beta, M, eps, R))
    ref, anchors = np.asarray(ref, dtype=np.int64), np.asarray(anchors, dtype=np.int64)
    n = len(beta)
    tau, e_ref = np.empty(n + 1), np.zeros(n)
    tau[0] = float(e0)
    for l in range(n):
        if ref[l] == REF_SOURCE:
            e_ref[l] = float(e0)
        elif ref[l] == REF_ANCHOR:
            if not 0 <= anchors[l] <= l:
                raise RunnerError("a last-safe reference must be a previously reached depth")
            e_ref[l] = tau[anchors[l]]
        tau[l + 1] = beta[l] * (tau[l] + M[l] * eps[l]) + R[l] * e_ref[l]
    return tau, e_ref


def trajectory_audit(record: dict, arrays: dict, plant: dict, *, relative_tolerance: float) -> dict:
    """Derived audit of one trajectory from its raw measurements (torch-free)."""
    from cm_theory.stability import logit_loss_bound, unrolled_bound
    M = np.asarray(arrays["M"], dtype=np.float64)
    terms = site_terms(arrays, plant, float(record["delta_p"]))
    e0 = float(record["e0"])
    tau, e_ref = forward_tube(terms["beta"], M, terms["eps_op"], terms["R"], terms["ref"], arrays["anchor"], e0)
    unrolled = unrolled_bound(terms["beta"], M, terms["eps_op"], terms["R"], e_ref, e0)
    if not math.isclose(unrolled, tau[-1], rel_tol=1e-9, abs_tol=1e-12):
        raise RunnerError(f"trajectory {record['trajectory']}: product-sum and recursion differ")
    bound = logit_loss_bound(np.array([[float(plant["head_norm"])]]), unrolled)
    realized_ok = record["realized_status"] == "ok"
    error = np.asarray(arrays["state_error"], dtype=np.float64)
    covered = False
    if realized_ok and record["realized_terminal_depth"] == record["predicted_terminal_depth"]:
        steps = len(tau)
        norms = np.maximum(np.asarray(arrays["norm_predicted"], dtype=np.float64)[:steps],
                           np.asarray(arrays["norm_realized"], dtype=np.float64)[:steps])
        e = error[:steps]
        covered = bool(np.all(np.isfinite(e)) and np.all(e <= tau + relative_tolerance * norms))
    return {"trajectory": int(record["trajectory"]), "seed": int(record["seed"]), "graph_id": record.get("graph_id"),
            "beta": terms["beta"], "m_eps_op": M * terms["eps_op"], "reference_term": terms["R"] * e_ref,
            "tube": tau, "bound_nats": float(bound), "unrolled": float(unrolled), "covered": covered,
            "mode_stable": bool(record["mode_stable"]), "observed_error_nats": finite_or_none(record.get("observed_error_nats")),
            "e0": e0, "delta_p": float(record["delta_p"]), "head_norm": float(plant["head_norm"]),
            "final_state_error": finite_or_none(error[-1]) if len(error) else None}


def _padded(rows: list[np.ndarray]) -> np.ndarray:
    width = max(len(r) for r in rows)
    out = np.full((len(rows), width), np.nan)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


def tube_summary(audits: list[dict]) -> dict:
    """The nine quantities of the record set and the per-depth arrays, from per-trajectory audits."""
    from cm_theory.stability import logit_loss_bound, unrolled_bound
    if not audits:
        raise RunnerError("no audited trajectory")
    betas = np.concatenate([a["beta"] for a in audits])
    m_eps, ref_term = _padded([a["m_eps_op"] for a in audits]), _padded([a["reference_term"] for a in audits])
    beta_depth, tubes = _padded([a["beta"] for a in audits]), _padded([a["tube"] for a in audits])
    with np.errstate(all="ignore"):
        per_depth = {"m_eps_op_median": np.nanmedian(m_eps, axis=0), "m_eps_op_p95": np.nanpercentile(m_eps, 95, axis=0),
                     "reference_term_median": np.nanmedian(ref_term, axis=0),
                     "reference_term_p95": np.nanpercentile(ref_term, 95, axis=0),
                     "beta_median": np.nanmedian(beta_depth, axis=0), "beta_p95": np.nanpercentile(beta_depth, 95, axis=0),
                     "tube_median": np.nanmedian(tubes, axis=0)}
    errors = [a["observed_error_nats"] for a in audits if a["observed_error_nats"] is not None]
    bounds = [a["bound_nats"] for a in audits]
    quantities = {
        "beta_median": float(np.median(betas)), "beta_p95": float(np.percentile(betas, 95)),
        "m_eps_op": float(np.median(per_depth["m_eps_op_median"])),
        "reference_term": float(np.median(per_depth["reference_term_median"])),
        "tube_coverage": float(np.mean([a["covered"] for a in audits])),
        "mode_stable_coverage": float(np.mean([a["mode_stable"] for a in audits])),
        "final_bound_nats": float(np.percentile(bounds, 95)),
        "observed_error_median_nats": float(np.median(errors)) if errors else None,
        "observed_error_p95_nats": float(np.percentile(errors, 95)) if errors else None}
    # Raw-only alternative: one product-sum with per-depth 95th percentiles of the constants.
    e0 = float(np.percentile([a["e0"] for a in audits], 95))
    envelope = unrolled_bound(per_depth["beta_p95"], np.ones(len(per_depth["beta_p95"])), per_depth["m_eps_op_p95"],
                              np.ones(len(per_depth["beta_p95"])), per_depth["reference_term_p95"], e0)
    envelope_nats = logit_loss_bound(np.array([[max(a["head_norm"] for a in audits)]]), envelope)
    counts = {"audited": len(audits), "covered": int(sum(a["covered"] for a in audits)),
              "mode_stable": int(sum(a["mode_stable"] for a in audits)), "with_realized_output": len(errors),
              "sites": int(len(betas))}
    return {"quantities": quantities, "per_depth": per_depth, "counts": counts,
            "per_trajectory_bound_median_nats": float(np.median(bounds)),
            "envelope_bound_nats": float(envelope_nats), "envelope_e0_p95": e0}


# ============================================================================= tube_audit: raw job files (torch-free)
def write_tube_trajectory(directory, record: dict, arrays: dict) -> Path:
    """One audited trajectory: <k>.npz, then <k>.json (the trajectory's completion marker)."""
    base = Path(directory) / "trajectories"
    stem = f"{int(record['trajectory']):06d}"
    digest = save_arrays(base / f"{stem}.npz", replace=True, **arrays)
    write_json(base / f"{stem}.json", json_safe(dict(record, arrays={"file": f"{stem}.npz", "sha256": digest})))
    return base / f"{stem}.json"


def finalize_tube_job(directory, plan: dict, entry: dict, *, started: str, parent_files, plants: dict) -> Path:
    """Collect the job's trajectories into tube_arrays.npz and tube.json (the job's completion marker)."""
    directory = Path(directory)
    records, arrays = [], {}
    for item in entry["trajectories"]:
        stem = f"{int(item['trajectory']):06d}"
        path = directory / "trajectories" / f"{stem}.json"
        if not path.is_file():
            raise RunnerError(f"{entry['member']}: trajectory {item['trajectory']} has not completed")
        record = read_json(path)
        npz = directory / "trajectories" / record["arrays"]["file"]
        if sha256(npz) != record["arrays"]["sha256"]:
            raise RunnerError(f"{npz} differs from its recorded SHA-256")
        with np.load(npz, allow_pickle=False) as data:
            for key in data.files:
                arrays[f"t{stem}_{key}"] = data[key]
        records.append(record)
    digest = save_arrays(directory / "tube_arrays.npz", replace=True, **arrays)
    write_json(directory / "tube.json", json_safe({
        "schema": "cm-tube-audit-job-1", "record_set": TUBE, "dataset": entry["dataset"], "member": entry["member"],
        "policy": entry["policy"], "policy_source": entry["policy_source"], "profile": entry["profile"],
        "executed_precision": entry["executed_precision"], "perturbation": plan["perturbation"],
        "probes": plan["probes"], "coverage_relative_tolerance": entry["coverage_relative_tolerance"],
        "replay_tolerance": entry["replay_tolerance"], "selection": entry["selection"],
        "plants": {str(k): v for k, v in plants.items()}, "trajectories": records,
        "arrays": {"file": "tube_arrays.npz", "sha256": digest},
        "identities": {"runtime_source_sha256": runtime_source_sha256(), "parent": entry["identity"],
                       "plan_sha256": sha256(directory.parents[2] / "plan.json"), "parent_files_read": parent_files,
                       "trajectory_audit_source_sha256": own_source_identity()},
        "started_utc": started, "finished_utc": utc_now()}))
    return directory / "tube.json"


def _job_result(output, entry: dict, marker: str, plan_sha: str) -> tuple[Path, dict]:
    directory = job_directory(output, entry)
    path = directory / marker
    if not path.is_file():
        raise RunnerError(f"{entry['dataset']}/{entry['member']}: job {entry['job']} has not completed")
    result = read_json(path)
    if result["identities"]["plan_sha256"] != plan_sha:
        raise RunnerError(f"{path}: the job belongs to another plan")
    if result["policy"] != entry["policy"]:
        raise RunnerError(f"{path}: the executed policy differs from the plan")
    return directory, result


def load_tube_job(output, entry: dict, plan_sha: str) -> tuple[Path, dict, dict]:
    directory, result = _job_result(output, entry, "tube.json", plan_sha)
    npz = directory / result["arrays"]["file"]
    if sha256(npz) != result["arrays"]["sha256"]:
        raise RunnerError(f"{npz} differs from its recorded SHA-256")
    with np.load(npz, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    planned = [int(t["trajectory"]) for t in entry["trajectories"]]
    if [int(r["trajectory"]) for r in result["trajectories"]] != planned:
        raise RunnerError(f"{directory}: the audited trajectories differ from the plan")
    return directory, result, arrays


def job_audits(result: dict, arrays: dict, *, allow_partial: bool) -> tuple[list[dict], dict]:
    audits, excluded = [], {"predicted_run_failed": [], "replay_mismatch": []}
    for record in result["trajectories"]:
        stem = f"{int(record['trajectory']):06d}"
        if record["predicted_status"] != "ok":
            excluded["predicted_run_failed"].append(record["trajectory"])
            continue
        if not record["replay_identity"]["ok"]:
            if not allow_partial:
                raise RunnerError(f"{result['member']} trajectory {record['trajectory']}: the joint replay did not "
                                  "reproduce the closed-loop outputs")
            excluded["replay_mismatch"].append(record["trajectory"])
            continue
        prefix = f"t{stem}_"
        raw = {key[len(prefix):]: value for key, value in arrays.items() if key.startswith(prefix)}
        plant = result["plants"][str(record["seed"])]
        audit = trajectory_audit(record, raw, plant, relative_tolerance=float(result["coverage_relative_tolerance"]))
        audit["member"] = result["member"]
        audits.append(audit)
    return audits, excluded


# ============================================================================= tube_audit: export (torch-free)
def export_tube_audit(output, *, root=None, replace=False, allow_partial=False, hardware=None, period=None,
                      command=None) -> Path:
    output = Path(output).resolve()
    plan = load_plan(output, TUBE)
    plan_sha = _plan_sha(output)
    staging = Path(tempfile.mkdtemp(prefix="cm-tube-export-"))
    try:
        files, rows, results, environments, inputs, skipped = code_deposit(), [], [], [], [], []
        for dataset, entries in _dataset_entries(plan):
            base = f"raw/{dataset['slug']}"
            audits, excluded, members = [], {}, []
            for entry in entries:
                directory, result, arrays = load_tube_job(output, entry, plan_sha)
                name = slug(entry["member"])
                files[f"{base}/{name}/tube.json"] = directory / "tube.json"
                files[f"{base}/{name}/tube_arrays.npz"] = directory / "tube_arrays.npz"
                files[f"{base}/{name}/environment.json"] = directory / "environment.json"
                job, dropped = job_audits(result, arrays, allow_partial=allow_partial)
                audits += job
                excluded[entry["member"]] = dropped
                members.append({"member": entry["member"], "file": f"{base}/{name}/tube.json",
                                "sha256": sha256(directory / "tube.json")})
                results.append(result)
                environments.append(read_json(directory / "environment.json"))
                inputs.append({"file": str((directory / "tube.json").relative_to(output)),
                               "sha256": sha256(directory / "tube.json")})
            if not audits and allow_partial:
                skipped.append(dataset["dataset"])
                continue
            summary = tube_summary(audits)
            missing = [q for q, v in summary["quantities"].items() if v is None]
            if missing and not allow_partial:
                raise RunnerError(f"{dataset['dataset']}: {missing} are not estimable (no realized output)")
            document = {"schema": "cm-tube-audit-dataset-1", "dataset": dataset["dataset"],
                        "aggregation": dataset["aggregation"], "members": members, "policy": plan["policy"],
                        "precision": plan["precision"], "perturbation": plan["perturbation"],
                        "quantities": summary["quantities"], "definitions": TUBE_DEFINITIONS,
                        "units": TUBE_UNITS, "counts": summary["counts"], "excluded": excluded,
                        "per_depth": {"site": list(range(len(summary["per_depth"]["beta_median"]))),
                                      **summary["per_depth"]},
                        "per_trajectory_bound_median_nats": summary["per_trajectory_bound_median_nats"],
                        "envelope_bound_nats": summary["envelope_bound_nats"],
                        "envelope_definition": "raw only: unrolled_bound with per-depth 95th percentiles of beta_l, "
                                               "M_l eps^op_l and R_l e^ref_l, e_0 at its 95th percentile and the "
                                               "largest head norm",
                        "trajectories": [{k: a[k] for k in ("member", "trajectory", "seed", "graph_id", "covered",
                                                            "mode_stable", "bound_nats", "observed_error_nats", "e0",
                                                            "delta_p", "head_norm", "final_state_error")}
                                         | {"final_tube": float(a["tube"][-1])} for a in audits]}
            path = staging / dataset["slug"] / "tube_audit.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(json_safe(document), indent=1, sort_keys=True, allow_nan=False) + "\n",
                            encoding="utf-8")
            files[f"{base}/tube_audit.json"] = path
            for quantity, unit in TUBE_UNITS.items():
                value = summary["quantities"][quantity]
                if value is None:
                    continue
                rows.append({"dataset": dataset["dataset"], "quantity": quantity, "value": value, "unit": unit,
                             "source_file": f"{base}/tube_audit.json", "locator": f"json:/quantities/{quantity}"})
        synthetic = {d["synthetic"] for d in plan["datasets"]}
        provenance = {
            "experiment": "Plug-in trajectory-tube audit of main Proposition 2 (Supplementary Table S22, "
                          "Supplementary Figure S17e, main Figure 2a)",
            "produced_by": "scripts/run_trajectory_audit.py tube plan/run/export (cm_experiments/trajectory_audit.py)",
            "hardware": hardware or provenance_hardware(environments),
            "period": period or provenance_period(results),
            "protocol": plan["protocol"], "synthetic": synthetic.pop(), "policy": plan["policy"],
            "precision": plan["precision"], "perturbation": plan["perturbation"], "probes": plan["probes"],
            "plan_sha256": plan_sha, "definitions": TUBE_DEFINITIONS, "skipped_datasets": skipped}
        with record_coverage(TUBE, [d["dataset"] for d in plan["datasets"]]):
            return write_record_set(TUBE, rows, files, provenance, root=root, replace=replace,
                                    allow_partial=allow_partial, inputs=inputs,
                                    command=command or ["scripts/run_trajectory_audit.py", "tube", "export"],
                                    generated={"code/plan.json": public_plan(output)})
    finally:
        shutil.rmtree(staging, ignore_errors=True)


# ============================================================================= tube_audit: execution (torch)
def input_perturbation(x: np.ndarray, relative: float, rng, *, chunk: int = 65536) -> np.ndarray:
    """x_r + relative ||x_r|| z_r/||z_r|| with z standard normal (rows of zero norm stay unchanged).

    Rows are drawn in consecutive chunks from one stream, so memory stays bounded on the largest graphs;
    the result has the dtype of ``x``.
    """
    x = np.asarray(x)
    if relative == 0:
        return x.copy()
    out = np.empty_like(x)
    for start in range(0, len(x), chunk):
        rows = np.asarray(x[start:start + chunk], dtype=np.float64)
        z = rng.standard_normal(rows.shape)
        norms = np.linalg.norm(z, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        out[start:start + chunk] = rows + float(relative) * np.linalg.norm(rows, axis=1, keepdims=True) * z / norms
    return out


def pair_uniform(first: np.ndarray, second: np.ndarray, key: int) -> np.ndarray:
    """Uniform [0, 1) draw per unordered node pair: splitmix64 of (min, max, key); symmetric, order-free, O(E)."""
    lo = np.minimum(first, second).astype(np.uint64)
    hi = np.maximum(first, second).astype(np.uint64)
    with np.errstate(over="ignore"):
        z = (lo * np.uint64(0x9E3779B97F4A7C15)) ^ (hi * np.uint64(0xC2B2AE3D27D4EB4F))
        z = z + np.uint64(int(key) % (1 << 64))
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def edge_weight_perturbation(edge_index: np.ndarray, base, relative: float, rng) -> np.ndarray:
    """w_e (1 + relative xi_e), xi uniform on [-1, 1) and equal on both directions of an undirected pair."""
    edge_index = np.asarray(edge_index, dtype=np.int64)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise RunnerError("edge_index must have shape [2, E]")
    if not 0 <= relative < 1:
        raise RunnerError("the edge-weight disturbance must lie in [0, 1) to keep every weight positive")
    key = int(rng.integers(0, 2 ** 63))
    xi = 2.0 * pair_uniform(edge_index[0], edge_index[1], key) - 1.0
    weights = np.ones(edge_index.shape[1]) if base is None else np.asarray(base, dtype=np.float64)
    return weights * (1.0 + float(relative) * xi)


def _streams(audit_seed: int, trajectory: int):
    return [np.random.default_rng(s) for s in np.random.SeedSequence([int(audit_seed), int(trajectory)]).spawn(3)]


def _row_norm_max(tensor) -> float:
    return float(tensor.detach().float().norm(dim=1).max())


def _spectral_norm(matrix) -> float:
    import torch
    return float(torch.linalg.matrix_norm(matrix.detach().double(), ord=2))


def plant_constants(model) -> dict:
    """Checkpoint constants of the plug-in tube: family, activation, update and head norms."""
    import torch
    backbone = getattr(model, "backbone", "shared_transport")
    activation = getattr(model, "activation_name", "relu")
    l_phi = activation_lipschitz(activation)
    if backbone in SHARED_BACKBONES:
        family, weight = "shared", _spectral_norm(model.shared.weight)
        if backbone == "heterophily":
            weight += _spectral_norm(model.ego.weight)
        norm = model.norm
        gamma = 1.0 if norm.weight is None else float(norm.weight.detach().abs().max())
        layer_norm = {"gamma_max": gamma, "eps": float(norm.eps)}
    elif backbone in DESIGNATED_BACKBONES:
        family, weight, layer_norm = "designated", None, None
    else:
        raise RunnerError(f"no plug-in constants are defined for the {backbone} plant")
    head = model.head
    if isinstance(head, torch.nn.ModuleDict):
        head_norm = _spectral_norm(head["output"].weight) * l_phi * _spectral_norm(head["hidden"].weight)
        head_rule = "||W_out||_2 L_phi ||W_hidden||_2 (two-layer head)"
    else:
        head_norm, head_rule = _spectral_norm(head.weight), "||W_head||_2 (linear head)"
    return {"backbone": backbone, "family": family, "activation": activation, "activation_lipschitz": l_phi,
            "weight_norm": weight, "weight_rule": ("||W_ego||_2 + ||W_shared||_2" if backbone == "heterophily" else
                                                   "||W_shared||_2" if family == "shared" else None),
            "layer_norm": layer_norm, "head_norm": head_norm, "head_rule": head_rule}


def perturbed_operator(episode, relative: float, rng, device, clean):
    """The disturbed graph operator of an episode (the clean operator itself when relative == 0)."""
    import torch
    from cross_morse.core import GraphOperator
    if relative == 0:
        return clean
    base = getattr(episode, "edge_weight", None)
    weights = edge_weight_perturbation(episode.edge_index.cpu().numpy(),
                                       None if base is None else base.cpu().numpy(), relative, rng)
    return GraphOperator(episode.edge_index, int(episode.x.shape[0]), device,
                         edge_weight=torch.as_tensor(weights, dtype=torch.float32),
                         graph_ptr=getattr(episode, "graph_ptr", None))


def operator_distance(clean, other) -> float:
    """||P - P~||_inf of the two random-walk operators (same coalesced edge set)."""
    import torch
    if other is clean:
        return 0.0
    a, b = clean.P.coalesce(), other.P.coalesce()
    if not torch.equal(a.indices(), b.indices()):
        raise RunnerError("the disturbed operator changed the edge set")
    difference = (a.values().double() - b.values().double()).abs()
    rows = torch.zeros(clean.num_nodes, dtype=torch.float64, device=difference.device)
    return float(rows.index_add_(0, a.indices()[0], difference).max())


def _anchor(requested, depth, h, safe, safe_depth):
    """The runtime's replay rule for a recorded last-safe trace (cross_morse.policies._execute_policy)."""
    requested = int(requested)
    if not 0 <= requested <= depth:
        raise RunnerError("a replay anchor must be a previously reached state")
    if requested == depth:
        return h.clone(), depth
    if requested != safe_depth:
        raise RunnerError("the replay skipped or reversed a recorded last-safe update")
    return safe, safe_depth


def _directions(shape, count: int, rng) -> list[np.ndarray]:
    out = []
    for _ in range(count):
        z = rng.standard_normal(shape)
        out.append(z / max(float(np.linalg.norm(z, axis=1).max()), 1e-300))
    v = rng.standard_normal(shape[1])
    out.append(np.tile(v / max(float(np.linalg.norm(v)), 1e-300), (shape[0], 1)))   # row-coherent direction
    return out


def _secant(step, point, directions, radius, base) -> float:
    import torch
    best = 0.0
    for z in directions:
        delta = torch.as_tensor(z, dtype=point.dtype, device=point.device) * radius
        best = max(best, _row_norm_max(step(point + delta) - base) / radius)
    return best


def joint_replay(model, x_predicted, x_realized, graph_predicted, graph_realized, config, predicted, realized,
                 plant, probes, rng, device) -> tuple[dict, dict]:
    """Exact lockstep replays of the two recorded traces with the plug-in measurements of the predicted run.

    ``predicted``/``realized`` hold actions and safe_depths (``realized`` is None after a failed realized
    run).  Returns the per-site arrays and the replays' final probabilities.
    """
    import torch
    from cross_morse.common import precision_context
    from cross_morse.core import ACTIONS, site_transport_accounting
    from cross_morse.tasks import probabilities, task_spec
    spec = getattr(model, "task", task_spec(config))
    acts_p, safe_p_trace = list(predicted["actions"]), list(predicted["safe_depths"])
    acts_r = list(realized["actions"]) if realized is not None else []
    safe_r_trace = list(realized["safe_depths"]) if realized is not None else []
    Lp, Lr = len(acts_p), len(acts_r)
    raw = {name: np.full(Lp, np.nan) for name in ("M", "l_ln", "beta_secant", "ref_secant", "alpha", "nu", "hops")}
    raw.update(kind=np.zeros(Lp, dtype=np.int64), action=np.asarray(acts_p, dtype=np.int64),
               anchor=np.zeros(Lp, dtype=np.int64), native_site=np.zeros(Lp, dtype=bool))
    norm_p, norm_r = np.full(Lp + 1, np.nan), np.full(Lr + 1, np.nan)
    error = np.full(max(Lp, Lr) + 1, np.nan)
    capture = {"on": False, "input": None}
    handle = None
    if plant["family"] == "shared":
        def hook(module, args):
            if capture["on"]:
                capture["input"] = args[0].detach()
        handle = model.norm.register_forward_pre_hook(hook)
    try:
        with torch.no_grad(), precision_context(config, device):
            model.bind_graph(graph_predicted)
            h0p = model.encode(x_predicted)
            hp, safe_p, depth_p = h0p, h0p, 0
            h0r = model.encode(x_realized)          # the source disturbance exists even if the realized run failed
            e0 = _row_norm_max(h0r - h0p)
            hr, safe_r, depth_r = h0r, h0r, 0
            for l in range(max(Lp, Lr)):
                if l < Lp:
                    norm_p[l] = _row_norm_max(hp)
                if realized is not None and l < Lr:
                    norm_r[l] = _row_norm_max(hr)
                    if l < Lp:
                        error[l] = _row_norm_max(hr - hp)
                if l < Lp:
                    a = ACTIONS[acts_p[l]]
                    safe_p, depth_p = _anchor(safe_p_trace[l], l, hp, safe_p, depth_p)
                    accounting = site_transport_accounting(a.id, l, model)
                    raw["M"][l], raw["alpha"][l], raw["nu"][l] = norm_p[l], a.alpha, a.nu
                    raw["hops"][l], raw["anchor"][l] = accounting["hops"], depth_p
                    raw["native_site"][l] = accounting["site_kind"] == "native"
                    if a.mode == "source":
                        raw["kind"][l] = KINDS.index("source_reset")
                        hp_next = model.step(hp, h0p, graph_predicted, a.id, safe_p, site=l)
                    elif a.mode == "last_safe":
                        raw["kind"][l] = KINDS.index("anchor_reset")
                        hp_next = model.step(hp, h0p, graph_predicted, a.id, safe_p, site=l)
                    elif plant["family"] == "shared":
                        raw["kind"][l] = KINDS.index("formula")
                        capture["on"] = True
                        hp_next = model.step(hp, h0p, graph_predicted, a.id, safe_p, site=l)
                        capture["on"] = False
                        U, capture["input"] = capture["input"], None
                        if U is None or not torch.allclose(model.norm(U).float(), hp_next.float(), rtol=1e-5, atol=1e-6):
                            raise RunnerError("the LayerNorm input of a transport step was not observed")
                        variance = float(U.float().var(dim=-1, unbiased=False).min())
                        raw["l_ln"][l] = layer_norm_lipschitz(plant["layer_norm"]["gamma_max"], variance,
                                                              plant["layer_norm"]["eps"])
                    elif not raw["native_site"][l]:
                        raw["kind"][l] = KINDS.index("inserted_identity" if a.id == 0 else "inserted_linear")
                        hp_next = model.step(hp, h0p, graph_predicted, a.id, safe_p, site=l)
                    else:
                        raw["kind"][l] = KINDS.index("native_secant")
                        hp_next = model.step(hp, h0p, graph_predicted, a.id, safe_p, site=l)
                        count, scale = int(probes["random"]), float(probes["relative_radius"])
                        state = lambda h: model.step(h, h0p, graph_predicted, a.id, safe_p, site=l)
                        source = lambda h0: model.step(hp, h0, graph_predicted, a.id, safe_p, site=l)
                        raw["beta_secant"][l] = _secant(state, hp, _directions(tuple(hp.shape), count, rng),
                                                        scale * max(norm_p[l], 1e-12), hp_next)
                        raw["ref_secant"][l] = _secant(source, h0p, _directions(tuple(h0p.shape), count, rng),
                                                       scale * max(_row_norm_max(h0p), 1e-12), hp_next)
                    hp = hp_next
                if realized is not None and l < Lr:
                    safe_r, depth_r = _anchor(safe_r_trace[l], l, hr, safe_r, depth_r)
                    hr = model.step(hr, h0r, graph_realized, acts_r[l], safe_r, site=l)
            norm_p[Lp] = _row_norm_max(hp)
            probs_p = probabilities(model.logits(hp).float(), spec).cpu().numpy()
            probs_r = None
            if realized is not None:
                norm_r[Lr] = _row_norm_max(hr)
                if Lr == Lp:
                    error[Lp] = _row_norm_max(hr - hp)
                model.bind_graph(graph_realized)
                probs_r = probabilities(model.logits(hr).float(), spec).cpu().numpy()
    finally:
        if handle is not None:
            handle.remove()
    raw.update(norm_predicted=norm_p, norm_realized=norm_r, state_error=error)
    return raw, {"predicted": probs_p, "realized": probs_r, "e0": e0}


def _trace_summary(trace) -> dict:
    keys = ("actions", "origins", "safe_depths", "terminal_depth", "support_violations", "observer_failures",
            "planning_budget_exhausted", "feasible")
    return {k: json_safe(trace.get(k)) for k in keys}


def _mode_stable(predicted: dict, realized: dict | None) -> bool:
    if realized is None:
        return False
    return all(predicted.get(k) == realized.get(k) for k in ("actions", "origins", "safe_depths", "terminal_depth"))


def _loss(probabilities, rows, labels, spec) -> float:
    from cross_morse.tasks import sample_losses
    values = sample_losses(np.asarray(probabilities)[rows], labels, spec)
    values = values[np.isfinite(values)]
    if not len(values):
        raise RunnerError("the episode has no scored D_explore label")
    return float(values.mean())


def audit_trajectory(model, episode, graph, config, candidate, controller, reference, labels, rows, item, plan,
                     entry, plant, device, cache=None) -> tuple[dict, dict]:
    """Predicted and realized closed-loop runs on one episode, their joint replay and raw measurements.

    ``cache`` (a dict) keeps the predicted closed-loop run of the last episode: it is deterministic and the
    same for every audited trajectory of one (checkpoint, graph), so it is executed once per episode.
    """
    import torch
    from cross_morse.policies import PolicyExecutionError, execute_policy
    from cross_morse.tasks import task_spec
    spec = getattr(model, "task", task_spec(config))
    started = utc_now()
    inputs, edges, probes = _streams(entry["audit_seed"], item["trajectory"])
    disturbance = plan["perturbation"]
    x = episode.x.to(device)
    x_realized = torch.as_tensor(input_perturbation(episode.x.cpu().numpy(), disturbance["input_relative"],
                                                    inputs)).to(device=device, dtype=episode.x.dtype)
    graph_realized = perturbed_operator(episode, disturbance["edge_weight_relative"], edges, device, graph)
    delta_p = operator_distance(graph, graph_realized)
    record = {"trajectory": int(item["trajectory"]), "seed": int(item["seed"]), "graph_id": item.get("graph_id"),
              "delta_p": delta_p, "started_utc": started}
    cache = {} if cache is None else cache
    episode_key = (int(item["seed"]), item.get("graph_id"))
    if cache.get("key") != episode_key:
        try:
            cache.update(key=episode_key, value=execute_policy(model, x, graph, config, candidate, controller,
                                                               reference))
        except PolicyExecutionError as error:
            cache.update(key=episode_key, value=error)
    if isinstance(cache["value"], PolicyExecutionError):
        record.update(predicted_status="failed", predicted_failure=cache["value"].reason, realized_status="not_run",
                      mode_stable=False, finished_utc=utc_now())
        return record, {}
    probs_p, trace_p = cache["value"]
    predicted = _trace_summary(trace_p)
    realized, probs_r, failure = None, None, None
    try:
        probs_r, trace_r = execute_policy(model, x_realized, graph_realized, config, candidate, controller, reference)
        realized = _trace_summary(trace_r)
    except PolicyExecutionError as error:
        failure = {"reason": error.reason, "failed_depth": error.trace.get("failed_depth"),
                   "actions": json_safe(error.trace.get("actions"))}
    raw, replayed = joint_replay(model, x, x_realized, graph, graph_realized, config, predicted, realized, plant,
                                 plan["probes"], probes, device)
    tolerance = float(entry["replay_tolerance"])
    identity = {"predicted_max_abs": float(np.max(np.abs(replayed["predicted"] - probs_p))), "tolerance": tolerance}
    if realized is not None:
        identity["realized_max_abs"] = float(np.max(np.abs(replayed["realized"] - probs_r)))
    identity["ok"] = all(v <= tolerance for k, v in identity.items() if k.endswith("max_abs"))
    loss_p = _loss(probs_p, rows, labels, spec)
    loss_r = _loss(probs_r, rows, labels, spec) if realized is not None else None
    record.update(predicted_status="ok", realized_status="ok" if realized is not None else "failed",
                  realized_failure=failure, predicted=predicted, realized=realized,
                  predicted_terminal_depth=int(predicted["terminal_depth"]),
                  realized_terminal_depth=None if realized is None else int(realized["terminal_depth"]),
                  mode_stable=_mode_stable(predicted, realized), e0=float(replayed["e0"]),
                  source_norm=float(raw["norm_predicted"][0]), predicted_loss_nats=loss_p, realized_loss_nats=loss_r,
                  observed_error_nats=None if loss_r is None else abs(loss_p - loss_r), replay_identity=identity,
                  finished_utc=utc_now())
    return record, raw


def run_tube_audit(output, *, job=None, device="cuda", trust=False) -> list[Path]:
    if not trust:
        raise RunnerError("loading local checkpoints requires explicit trust (--trust-local-checkpoints)")
    output = Path(output).resolve()
    plan = load_plan(output, TUBE)
    done = []
    for entry in selected_jobs(plan, job):
        directory = job_directory(output, entry)
        if not (directory / "tube.json").is_file():
            _run_tube_job(plan, entry, directory, device)
        done.append(directory / "tube.json")
    return done


def _run_tube_job(plan, entry, directory, device):
    run = SealedRun(entry["run"])
    check_identity(run, entry["identity"])
    run.require("explore")
    candidate = planned_candidate(run, plan["policy"], entry["policy"])
    started = utc_now()
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / "environment.json").exists():
        write_json(directory / "environment.json", environment_record(device))
    ctx = ParentContext(run, device, through=latest_stage("fit", POLICIES[plan["policy"]]))
    config = copy.deepcopy(ctx.config)
    config["model"]["precision"] = entry["executed_precision"]
    controller = ctx.controller_for(candidate)
    if not ctx.collection:
        with ctx.data._label_scope("explore"):
            labels_all = ctx.data.labels("explore")
        rows_all = np.asarray(ctx.data.splits["explore"], dtype=np.int64)
    plants, constants, cache = {}, {}, {}
    # Episode order (checkpoint, graph) lets consecutive trajectories share the predicted closed-loop run.
    order = sorted(entry["trajectories"], key=lambda t: (int(t["seed"]), -1 if t.get("graph_id") is None
                                                         else int(t["graph_id"]), int(t["trajectory"])))
    for item in order:
        if (directory / "trajectories" / f"{int(item['trajectory']):06d}.json").is_file():
            continue
        if sha256(run.root / item["file"]) != item["sha256"]:
            raise RunnerError(f"{run.root.name}: {item['file']} changed after planning")
        seed = int(item["seed"])
        if seed not in plants:
            plants[seed] = ctx.plant(seed)
            constants[seed] = plant_constants(plants[seed])
        if ctx.collection:
            from cross_morse.runtime import graph_operator
            episode = ctx.data.graph(int(item["graph_id"]))
            graph = graph_operator(episode, ctx.device)
            with ctx.data._label_scope("explore"):
                labels = episode.labels("explore")
            rows = np.asarray(episode.splits["explore"], dtype=np.int64)
        else:
            episode, graph, labels, rows = ctx.data, ctx.graph, labels_all, rows_all
        record, raw = audit_trajectory(plants[seed], episode, graph, config, candidate, controller, ctx.reference,
                                       labels, rows, item, plan, entry, constants[seed], ctx.device, cache=cache)
        write_tube_trajectory(directory, record, raw)
        from cross_morse.common import stop_at_boundary
        stop_at_boundary()          # a signal stops here; the next run resumes after this trajectory
    for item in entry["trajectories"]:      # resumed jobs: constants of every audited checkpoint
        seed = int(item["seed"])
        if seed not in constants:
            constants[seed] = plant_constants(ctx.plant(seed))
    finalize_tube_job(directory, plan, entry, started=started, parent_files=run.inputs(), plants=constants)


# ============================================================================= depth_diagnostics: planning (torch-free)
def check_label_role(run: SealedRun, role: str) -> None:
    """fit_val after the train stage that owns it; compare through the registered reserved-units check."""
    if role == "fit_val":
        run.require("train")
    elif role == "compare":
        check_role(run, "compare")
    else:
        raise RunnerError("the depth diagnostics read fit_val or compare labels; decision and test roles are not "
                          "reused for these displays")


def role_inputs(run: SealedRun, role: str, *, limit: int, seed: int) -> tuple[list[int] | None, int]:
    """Planned graphs of an independent-graph role (None: the one transductive graph)."""
    path = run.root / "prepared" / "graph_splits.npz"
    if not path.is_file():
        return None, 1
    graphs = sorted(int(g) for g in run.npz("prepared/graph_splits.npz")[role])
    available = len(graphs)
    if not graphs:
        raise RunnerError(f"{run.root.name}: the {role} role holds no graph")
    if limit and available > limit:
        picks = np.sort(np.random.default_rng(seed).choice(available, size=int(limit), replace=False))
        graphs = [graphs[int(i)] for i in picks]
    return graphs, available


def plan_depth_diagnostics(parents, output, *, reference=None, policy="deployed", role="fit_val", max_inputs=0,
                           reps=None, chunk_size=0, seed_offset=1) -> Path:
    if policy not in POLICIES:
        raise RunnerError(f"policy must be one of {sorted(POLICIES)}")
    if role not in ("fit_val", "compare"):
        raise RunnerError("role must be fit_val or compare (the test role is reserved for test-outcome displays)")
    if int(max_inputs) < 0 or int(chunk_size) < 0:
        raise RunnerError("max_inputs must be >= 0 (0: every input of the role) and chunk_size >= 0")
    groups = _parents(parents)
    names = [runs[0].dataset for runs in groups]
    if reference is None:
        reference = "ogbn-arxiv" if "ogbn-arxiv" in names else (names[0] if len(names) == 1 else None)
    if reference not in names:
        raise RunnerError(f"declare --reference, one of the planned datasets {names}")
    datasets, jobs = [], []
    for runs in groups:
        for run in runs:
            check_label_role(run, role)
            run.require("train")
            candidate, source = policy_candidate(run, policy)
            config = run.config
            depth = int(config["model"]["depth"])
            entry_initial = run.json("train/entry_initial.json")
            values = run.npz("train/entry_depth_values.npz")
            grid = [int(d) for d in entry_initial["depth_grid"]]
            if grid != registered_grid(depth) or [int(d) for d in values["depth_grid"]] != grid:
                raise RunnerError(f"{run.root.name}: the entry depth grid differs from the registered grid")
            seeds = [int(s) for s in config["training"]["seeds"]]
            inputs, available = role_inputs(run, role, limit=int(max_inputs), seed=int(config["seed"]) + SEED_OFFSET)
            is_reference = run.dataset == reference
            job = {"dataset": run.dataset, "slug": slug(run.dataset), "member": run.root.name,
                   "run": str(run.root), "identity": run.identity(), "policy": candidate_identity(candidate),
                   "policy_source": source, "member_seeds": seeds[:int(candidate["ensemble"])], "depth": depth,
                   "depth_grid": grid, "first_plotted_depth": first_plotted_depth(grid),
                   "reference": is_reference, "alignment_seeds": seeds if is_reference else [],
                   "bands": [int(b) for b in config["sensor"]["bands"]], "degree": int(config["sensor"]["degree"]),
                   "inputs": inputs, "available_inputs": available,
                   "planned_inputs": None if inputs is None else len(inputs),
                   "entry_files": {name: run.read[name] for name in ("train/entry_initial.json",
                                                                       "train/entry_depth_values.npz")},
                   "bootstrap": {"reps": int(reps or config["selection"]["bootstrap_reps"]),
                                 "seed": int(config["seed"]) + int(seed_offset), "alpha": 0.025}}
            # The source alignment of a reference run is computed once, by its first slice.
            for piece in chunked_jobs(job, "inputs", int(chunk_size), first_only=("alignment_seeds",)):
                jobs.append(dict(piece, job=len(jobs)))
        datasets.append(_dataset_declaration(runs))
    truncated = any(j["planned_inputs"] is not None and j["planned_inputs"] < j["available_inputs"] for j in jobs)
    return create_plan(output, {
        "experiment": "Depth diagnostics of the plants and deployed rules (Supplementary Figure S14)",
        "record_set": DEPTH, "policy": policy, "role": role, "reference": reference, "max_inputs": int(max_inputs),
        "chunk_size": int(chunk_size), "truncated": truncated, "datasets": datasets, "jobs": jobs,
        "protocol": ("reference plant: nominal trajectories of every registered checkpoint on fit_val with a full-state "
                     "MeasurementBank (source alignment of the retained bands); every dataset: the declared policy at its "
                     "ensemble size on the declared role, each member truncated after its first d executed actions by "
                     "exact replay; first non-A0 depths and phase transport fractions over executions; supported "
                     "deterioration from the sealed entry audit (module docstring)"),
        "trajectory_audit_source_sha256": own_source_identity()})


# ============================================================================= depth_diagnostics: statistics (torch-free)
def first_intervention_depth(actions) -> float:
    """Propagation depth (site index + 1) of the first executed action other than A0; +inf without one."""
    for site, action in enumerate(actions):
        if int(action) < 0:
            break
        if int(action) != 0:
            return float(site + 1)
    return math.inf


def empirical_quantile(values, q: float):
    """The empirical quantile function (numpy inverted_cdf: an observed value); None when it is +inf."""
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or np.isnan(values).any():
        raise RunnerError("quantiles need a nonempty vector without NaN")
    value = float(np.quantile(values, q, method="inverted_cdf"))
    return value if math.isfinite(value) else None


def phase_index(site: int, depth: int) -> int:
    return min(2, (3 * int(site)) // int(depth))


def phase_counts(actions_matrix, depth: int) -> dict:
    """Executed sites and A1-A7 transport sites per depth phase (actions padded with -1 after the terminal)."""
    executed, transport = np.zeros(3, dtype=np.int64), np.zeros(3, dtype=np.int64)
    for actions in np.asarray(actions_matrix, dtype=np.int64).reshape(-1, int(depth)):
        for site, action in enumerate(actions):
            if action < 0:
                break
            k = phase_index(site, depth)
            executed[k] += 1
            transport[k] += 1 <= action <= 7
    return {"executed": executed, "transport": transport}


def weighted_curve(unit_depth_loss, weights) -> np.ndarray:
    losses = np.asarray(unit_depth_loss, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if losses.ndim != 2 or losses.shape[1] != len(w) or not np.isfinite(losses).all() or (w <= 0).any():
        raise RunnerError("unit depth losses must be finite [depths, units] with positive unit weights")
    return losses @ w / w.sum()


def unit_depth_losses(losses, unit_ids, independent: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-unit mean loss at every depth from per-sample losses [depths, samples] (finite samples only)."""
    from cross_morse.statistics import cluster_summary
    losses = np.asarray(losses, dtype=np.float64)
    finite = np.isfinite(losses).all(axis=0)
    if not finite.any():
        raise RunnerError("the entry audit holds no scored sample")
    means, units, counts = cluster_summary(losses[:, finite].T, np.asarray(unit_ids)[finite])
    weights = np.ones(len(units)) if independent else counts.astype(np.float64)
    return means.T, units, weights


def supported_deterioration(unit_losses, weights, grid, best_index: int, *, reps: int, seed: int,
                            alpha: float = 0.025) -> dict:
    """First grid depth after the best depth whose paired unit increase has its central interval above zero."""
    from cross_morse.statistics import paired_bootstrap
    unit_losses = np.asarray(unit_losses, dtype=np.float64)
    weighted = weights if len(set(np.asarray(weights).tolist())) > 1 else None
    tests, chosen = [], None
    for k in range(best_index + 1, len(grid)):
        result = paired_bootstrap(unit_losses[k] - unit_losses[best_index], weights=weighted, reps=reps, alpha=alpha,
                                  seed=seed)
        supported = result["lower"] is not None and result["lower"] > 0
        tests.append({"depth": int(grid[k]), "mean": result["mean"], "lower": result["lower"], "upper": result["upper"],
                      "units": result["units"], "supported": bool(supported)})
        if supported and chosen is None:
            chosen = int(grid[k])
    return {"best_depth": int(grid[best_index]), "supported_depth": chosen, "tests": tests,
            "interval": f"central {1 - 2 * alpha:.0%} paired whole-unit bootstrap (cross_morse.statistics)",
            "bootstrap_replicates": int(reps), "bootstrap_seed": int(seed)}


def supported_deterioration_splits(split_curves, grid, *, alpha: float = 0.025) -> dict:
    """Split runs as units: t interval over the splits about the split-mean curve's best depth."""
    from scipy import stats
    curves = np.asarray(split_curves, dtype=np.float64)
    if curves.ndim != 2 or curves.shape[0] < 2:
        raise RunnerError("the split-unit interval needs at least two split runs")
    best = int(np.argmin(curves.mean(axis=0)))
    t = float(stats.t.ppf(1 - alpha, curves.shape[0] - 1))
    tests, chosen = [], None
    for k in range(best + 1, len(grid)):
        d = curves[:, k] - curves[:, best]
        half = t * float(np.std(d, ddof=1)) / math.sqrt(len(d))
        lower, upper = float(d.mean()) - half, float(d.mean()) + half
        supported = lower > 0
        tests.append({"depth": int(grid[k]), "mean": float(d.mean()), "lower": lower, "upper": upper,
                      "units": int(len(d)), "supported": bool(supported)})
        if supported and chosen is None:
            chosen = int(grid[k])
    return {"best_depth": int(grid[best]), "supported_depth": chosen, "tests": tests,
            "interval": f"central {1 - 2 * alpha:.0%} small-sample t interval over the split runs"}


def alignment_curve(values) -> np.ndarray:
    """Mean over checkpoints (and split runs) of the band-mean source alignment: [curves, depths, bands] -> [depths]."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise RunnerError("alignment values must be finite [checkpoints, depths, bands]")
    return values.mean(axis=2).mean(axis=0)


# ============================================================================= depth_diagnostics: raw job files (torch-free)
def write_depth_job(directory, plan: dict, entry: dict, *, unit_ids, unit_counts, independent: bool,
                    unit_depth_loss, executions: dict, alignment, entry_dir, started: str, parent_files,
                    sealed_actions_agree=None) -> Path:
    """The job's raw outputs: copies of the entry audit, depth_arrays.npz, then depth.json (completion marker)."""
    directory = Path(directory)
    copies = {}
    for name, digest in entry["entry_files"].items():
        source = Path(entry_dir) / name
        if sha256(source) != digest:
            raise RunnerError(f"{name} changed after planning")
        target = directory / "entry" / Path(name).name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copies[name] = {"file": f"entry/{target.name}", "sha256": sha256(target)}
    arrays = {"unit_ids": np.asarray(unit_ids), "unit_counts": np.asarray(unit_counts, dtype=np.float64),
              "unit_depth_loss": np.asarray(unit_depth_loss, dtype=np.float64),
              "depth_grid": np.asarray(entry["depth_grid"], dtype=np.int64),
              "execution_input": np.asarray(executions["input"], dtype=np.int64),
              "execution_seed": np.asarray(executions["seed"], dtype=np.int64),
              "execution_terminal": np.asarray(executions["terminal"], dtype=np.int64),
              "execution_actions": np.asarray(executions["actions"], dtype=np.int64).reshape(-1, int(entry["depth"]))}
    if alignment is not None:
        arrays["alignment"] = np.asarray(alignment, dtype=np.float64)
    digest = save_arrays(directory / "depth_arrays.npz", replace=True, **arrays)
    write_json(directory / "depth.json", json_safe({
        "schema": "cm-depth-diagnostics-job-1", "record_set": DEPTH, "dataset": entry["dataset"],
        "member": entry["member"], "policy": entry["policy"], "policy_source": entry["policy_source"],
        "role": plan["role"], "member_seeds": entry["member_seeds"], "depth": entry["depth"],
        "depth_grid": entry["depth_grid"], "first_plotted_depth": entry["first_plotted_depth"],
        "reference": entry["reference"], "alignment_seeds": entry["alignment_seeds"], "bands": entry["bands"],
        "unit_weighting": "equal graphs" if independent else "scored samples per unit",
        "independent_graphs": bool(independent), "inputs": entry["inputs"], "available_inputs": entry["available_inputs"],
        "executions": int(len(executions["seed"])), "entry_copies": copies, "bootstrap": entry["bootstrap"],
        "sealed_actions_agree": sealed_actions_agree,
        "arrays": {"file": "depth_arrays.npz", "sha256": digest},
        "identities": {"runtime_source_sha256": runtime_source_sha256(), "parent": entry["identity"],
                       "plan_sha256": sha256(directory.parents[2] / "plan.json"), "parent_files_read": parent_files,
                       "trajectory_audit_source_sha256": own_source_identity()},
        "started_utc": started, "finished_utc": utc_now()}))
    return directory / "depth.json"


def load_depth_job(output, entry: dict, plan_sha: str) -> tuple[Path, dict, dict]:
    directory, result = _job_result(output, entry, "depth.json", plan_sha)
    npz = directory / result["arrays"]["file"]
    if sha256(npz) != result["arrays"]["sha256"]:
        raise RunnerError(f"{npz} differs from its recorded SHA-256")
    with np.load(npz, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    for copy_entry in result["entry_copies"].values():
        if sha256(directory / copy_entry["file"]) != copy_entry["sha256"]:
            raise RunnerError(f"{directory / copy_entry['file']} differs from its recorded SHA-256")
    return directory, result, arrays


def merge_depth_parts(parts: list[tuple[dict, dict, dict, Path]]) -> tuple[dict, dict, dict, Path]:
    """One run's job slices merged into one (entry, result, arrays, directory); the units are disjoint graphs."""
    entry0, result0, arrays0, directory0 = parts[0]
    if len(parts) == 1:
        return parts[0]
    for _, result, arrays, _ in parts[1:]:
        if (result["depth_grid"] != result0["depth_grid"] or result["independent_graphs"] != result0["independent_graphs"]
                or result["policy"] != result0["policy"]):
            raise RunnerError(f"{entry0['run_member']}: its job slices differ in grid, unit type or policy")
    merged = {"depth_grid": arrays0["depth_grid"]}
    merged["unit_ids"] = np.concatenate([a["unit_ids"] for _, _, a, _ in parts])
    if len(np.unique(merged["unit_ids"])) != len(merged["unit_ids"]):
        raise RunnerError(f"{entry0['run_member']}: two job slices scored the same unit")
    merged["unit_counts"] = np.concatenate([a["unit_counts"] for _, _, a, _ in parts])
    merged["unit_depth_loss"] = np.concatenate([a["unit_depth_loss"] for _, _, a, _ in parts], axis=1)
    for key in ("execution_input", "execution_seed", "execution_terminal", "execution_actions"):
        merged[key] = np.concatenate([a[key] for _, _, a, _ in parts], axis=0)
    alignments = [a["alignment"] for _, _, a, _ in parts if "alignment" in a]
    if len(alignments) > 1:
        raise RunnerError(f"{entry0['run_member']}: more than one slice holds the source alignment")
    if alignments:
        merged["alignment"] = alignments[0]
    result = dict(result0, member=entry0["run_member"], executions=int(sum(r["executions"] for _, r, _, _ in parts)),
                  parts=[r["member"] for _, r, _, _ in parts])
    return entry0, result, merged, directory0


def entry_audit_curves(directory, result: dict) -> dict:
    """Per-unit depth losses of the sealed entry audit, checked against its recorded depth risk."""
    initial = read_json(directory / result["entry_copies"]["train/entry_initial.json"]["file"])
    with np.load(directory / result["entry_copies"]["train/entry_depth_values.npz"]["file"], allow_pickle=False) as data:
        losses, units = data["losses"], data["core_ids"]
    means, unit_ids, weights = unit_depth_losses(losses, units, result["independent_graphs"])
    curve = weighted_curve(means, weights)
    recorded = np.asarray(initial["depth_loss"], dtype=np.float64)
    # The entry audit averages float32 per-sample losses; float64 recomputation agrees to float32 precision.
    if not np.allclose(curve, recorded, rtol=1e-5, atol=1e-7):
        raise RunnerError(f"{directory}: the entry audit's per-unit losses do not reproduce its depth risk")
    return {"means": means, "weights": weights, "curve": recorded, "grid": [int(d) for d in initial["depth_grid"]],
            "best_index": int(np.argmin(recorded))}


def depth_summary(jobs: list[tuple[dict, dict, dict, Path]], *, reference: bool) -> dict:
    """Every quantity of one dataset from its jobs' raw outputs; values that do not exist are None."""
    grid = jobs[0][1]["depth_grid"]
    depth = int(jobs[0][1]["depth"])
    first = int(jobs[0][1]["first_plotted_depth"])
    if any(r["depth_grid"] != grid for _, r, _, _ in jobs):
        raise RunnerError("the split runs of one dataset have different depth grids")
    k0 = grid.index(first)
    plotted = [k for k, d in enumerate(grid) if d >= first]
    out = {"depth_grid": grid, "first_plotted_depth": first, "members": [r["member"] for _, r, _, _ in jobs]}
    # policy_loss_change: unit-weighted loss of the truncated ensemble, change from the first plotted depth
    changes = []
    for _, result, arrays, _ in jobs:
        weights = arrays["unit_counts"] if not result["independent_graphs"] else np.ones(len(arrays["unit_ids"]))
        curve = weighted_curve(arrays["unit_depth_loss"], weights)
        changes.append(curve - curve[k0])
    mean_change = np.mean(changes, axis=0)
    out["policy_loss_change"] = [{"depth": int(grid[k]), "value": float(mean_change[k]),
                                  "per_member": [float(c[k]) for c in changes]} for k in plotted]
    # first intervention quartiles over executions
    actions = np.concatenate([a["execution_actions"] for _, _, a, _ in jobs], axis=0)
    firsts = [first_intervention_depth(row) for row in actions]
    quartiles = {name: empirical_quantile(firsts, q) for name, q in (("q25", 0.25), ("median", 0.5), ("q75", 0.75))}
    out["first_intervention"] = {**quartiles, "executions": len(firsts),
                                 "without_intervention": int(sum(math.isinf(f) for f in firsts)),
                                 "method": "numpy inverted_cdf over executions; +inf without a non-A0 action"}
    # transport fraction per phase
    counts = phase_counts(actions, depth)
    ranges = [[s for s in range(depth) if phase_index(s, depth) == k] for k in range(3)]
    out["transport_fraction"] = [{"phase": PHASES[k], "value": float(counts["transport"][k] / counts["executed"][k]),
                                  "executed_sites": int(counts["executed"][k]),
                                  "transport_sites": int(counts["transport"][k]),
                                  "site_range": [ranges[k][0], ranges[k][-1]]}
                                 for k in range(3) if counts["executed"][k]]
    # supported deterioration from the sealed entry audits
    curves = [entry_audit_curves(directory, result) for _, result, _, directory in jobs]
    if len(curves) == 1:
        c = curves[0]
        boot = jobs[0][1]["bootstrap"]
        out["supported_deterioration"] = supported_deterioration(c["means"], c["weights"], c["grid"], c["best_index"],
                                                                 reps=int(boot["reps"]), seed=int(boot["seed"]),
                                                                 alpha=float(boot["alpha"]))
    else:
        out["supported_deterioration"] = supported_deterioration_splits([c["curve"] for c in curves], grid)
    # source alignment on the reference plant
    if reference:
        values = np.concatenate([a["alignment"] for _, _, a, _ in jobs], axis=0)
        curve = alignment_curve(values)
        out["source_alignment_change"] = [{"depth": int(grid[k]), "value": float(curve[k] - curve[k0]),
                                           "alignment": float(curve[k])} for k in plotted]
        out["source_alignment_checkpoints"] = int(values.shape[0])
    return out


# ============================================================================= depth_diagnostics: export (torch-free)
def export_depth_diagnostics(output, *, root=None, replace=False, allow_partial=False, hardware=None, period=None,
                             command=None) -> Path:
    output = Path(output).resolve()
    plan = load_plan(output, DEPTH)
    plan_sha = _plan_sha(output)
    if plan["truncated"] and not allow_partial:
        raise RunnerError("the plan truncates a role (max_inputs); that is not a complete measurement")
    staging = Path(tempfile.mkdtemp(prefix="cm-depth-export-"))
    try:
        files, rows, results, environments, inputs = code_deposit(), [], [], [], []
        for dataset, entries in _dataset_entries(plan):
            base = f"raw/{dataset['slug']}"
            by_run = {}
            for entry in entries:
                directory, result, arrays = load_depth_job(output, entry, plan_sha)
                if result["role"] != plan["role"]:
                    raise RunnerError(f"{directory}: the evaluated role differs from the plan")
                name = slug(entry["member"])
                files[f"{base}/{name}/depth.json"] = directory / "depth.json"
                files[f"{base}/{name}/depth_arrays.npz"] = directory / "depth_arrays.npz"
                files[f"{base}/{name}/environment.json"] = directory / "environment.json"
                for copy_entry in result["entry_copies"].values():
                    files[f"{base}/{name}/{copy_entry['file']}"] = directory / copy_entry["file"]
                by_run.setdefault(entry.get("run_member", entry["member"]), []).append((entry, result, arrays, directory))
                results.append(result)
                environments.append(read_json(directory / "environment.json"))
                inputs.append({"file": str((directory / "depth.json").relative_to(output)),
                               "sha256": sha256(directory / "depth.json")})
            jobs = [merge_depth_parts(parts) for parts in by_run.values()]
            summary = depth_summary(jobs, reference=dataset["dataset"] == plan["reference"])
            document = {"schema": "cm-depth-diagnostics-dataset-1", "dataset": dataset["dataset"],
                        "aggregation": dataset["aggregation"], "policy": plan["policy"], "role": plan["role"],
                        "reference": dataset["dataset"] == plan["reference"], **summary}
            path = staging / dataset["slug"] / "depth_diagnostics.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(json_safe(document), indent=1, sort_keys=True, allow_nan=False) + "\n",
                            encoding="utf-8")
            source = f"{base}/depth_diagnostics.json"
            files[source] = path

            def add(quantity, value, locator, depth=ALL, phase=ALL):
                rows.append({"dataset": dataset["dataset"], "depth": str(depth), "phase": str(phase),
                             "quantity": quantity, "value": value, "unit": DEPTH_UNITS[quantity],
                             "source_file": source, "locator": locator})
            for i, item in enumerate(summary.get("source_alignment_change", [])):
                add("source_alignment_change", item["value"], f"json:/source_alignment_change/{i}/value",
                    depth=item["depth"])
            for i, item in enumerate(summary["policy_loss_change"]):
                add("policy_loss_change", item["value"], f"json:/policy_loss_change/{i}/value", depth=item["depth"])
            for name in ("q25", "median", "q75"):
                if summary["first_intervention"][name] is not None:
                    add(f"first_intervention_depth_{name}", summary["first_intervention"][name],
                        f"json:/first_intervention/{name}")
            if summary["supported_deterioration"]["supported_depth"] is not None:
                add("supported_deterioration_depth", summary["supported_deterioration"]["supported_depth"],
                    "json:/supported_deterioration/supported_depth")
            for i, item in enumerate(summary["transport_fraction"]):
                add("transport_fraction", item["value"], f"json:/transport_fraction/{i}/value", phase=item["phase"])
        synthetic = {d["synthetic"] for d in plan["datasets"]}
        provenance = {
            "experiment": "Depth diagnostics of the plants and deployed rules (Supplementary Figure S14)",
            "produced_by": "scripts/run_trajectory_audit.py depth plan/run/export (cm_experiments/trajectory_audit.py)",
            "hardware": hardware or provenance_hardware(environments),
            "period": period or provenance_period(results),
            "protocol": plan["protocol"], "synthetic": synthetic.pop(), "policy": plan["policy"], "role": plan["role"],
            "reference": plan["reference"], "key_convention": f"'{ALL}' marks a key that a quantity does not have",
            "plan_sha256": plan_sha}
        with record_coverage(DEPTH, [d["dataset"] for d in plan["datasets"]]):
            return write_record_set(DEPTH, rows, files, provenance, root=root, replace=replace,
                                    allow_partial=allow_partial, inputs=inputs,
                                    command=command or ["scripts/run_trajectory_audit.py", "depth", "export"],
                                    generated={"code/plan.json": public_plan(output)})
    finally:
        shutil.rmtree(staging, ignore_errors=True)


# ============================================================================= depth_diagnostics: execution (torch)
def truncated_outputs(model, x, graph, config, candidate, controller, reference, grid, rows):
    """One member execution and its role-row probabilities after the first d executed actions (exact replay)."""
    from cross_morse.policies import PolicyExecutionError, execute_policy
    try:
        final, trace = execute_policy(model, x, graph, config, candidate, controller, reference)
    except PolicyExecutionError as error:
        raise RunnerError(f"the policy failed during execution ({error.reason}); no prediction exists") from None
    actions, anchors = list(trace["actions"]), list(trace["safe_depths"])
    by_depth = []
    for d in grid:
        if d >= len(actions):
            by_depth.append(np.asarray(final[rows]))       # a stopped policy keeps its output
        else:
            replay, _ = execute_policy(model, x, graph, config, candidate, None, reference,
                                       replay_actions=actions[:d], replay_safe_depths=anchors[:d])
            by_depth.append(np.asarray(replay[rows]))
    return by_depth, trace


def role_labels_of(ctx, role: str) -> np.ndarray:
    if role == "compare":
        return ctx.labels("compare")
    with ctx.data._label_scope("train"):
        return ctx.data.labels(role)


def source_alignments(ctx, entry: dict, device) -> np.ndarray:
    """[checkpoints, grid depths, bands] mean source alignment of the nominal trajectory on fit_val."""
    import torch
    from cross_morse.common import precision_context
    from cross_morse.core import MeasurementBank
    config = ctx.config
    depth, grid = int(config["model"]["depth"]), [int(d) for d in entry["depth_grid"]]
    columns = None
    values = np.full((len(entry["alignment_seeds"]), len(grid), 3), np.nan)
    for i, seed in enumerate(entry["alignment_seeds"]):
        model = ctx.plant(seed)
        sums, count = np.zeros((len(grid), 3)), 0
        for _, x, graph, rows, _ in ctx.episodes("fit_val"):
            with torch.no_grad(), precision_context(config, device):
                model.bind_graph(graph)
                mask = None if ctx.collection else torch.as_tensor(rows, dtype=torch.long, device=graph.device)
                bank = MeasurementBank(graph, model.hidden_dim, bands=entry["bands"], degree=entry["degree"],
                                       sketch_dim=0, seed=int(config["seed"]), profile="full", mask=mask,
                                       task=model.task, plant=model, sensing_period=1)
                columns = [bank.names.index(f"source_alignment_{b}") for b in entry["bands"]]
                h0 = model.encode(x)
                bank.prepare(h0, model.logits(h0))
                h = h0
                for d in range(1, depth + 1):
                    h = model.native_step(h, h0, graph, site=d - 1)
                    if d in grid:
                        y = bank.measure(h, model.logits(h), d, depth).cpu().numpy()
                        sums[grid.index(d)] += y[columns]
            count += 1
        values[i] = sums / count
        del model
    return values


def run_depth_diagnostics(output, *, job=None, device="cuda", trust=False) -> list[Path]:
    if not trust:
        raise RunnerError("loading local checkpoints requires explicit trust (--trust-local-checkpoints)")
    output = Path(output).resolve()
    plan = load_plan(output, DEPTH)
    done = []
    for entry in selected_jobs(plan, job):
        directory = job_directory(output, entry)
        if not (directory / "depth.json").is_file():
            _run_depth_job(plan, entry, directory, device)
        done.append(directory / "depth.json")
    return done


def _run_depth_job(plan, entry, directory, device):
    from cross_morse.evaluation import independent_graphs
    from cross_morse.tasks import sample_losses, task_spec
    run = SealedRun(entry["run"])
    check_identity(run, entry["identity"])
    check_label_role(run, plan["role"])
    candidate = planned_candidate(run, plan["policy"], entry["policy"])
    started = utc_now()
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / "environment.json").exists():
        write_json(directory / "environment.json", environment_record(device))
    stage = latest_stage(POLICIES[plan["policy"]], "confirm" if plan["role"] == "compare" else "train")
    ctx = ParentContext(run, device, through=stage)
    config, spec, grid = ctx.config, task_spec(ctx.config), [int(d) for d in entry["depth_grid"]]
    controller = ctx.controller_for(candidate)
    role = plan["role"]
    labels, units = role_labels_of(ctx, role), ctx.units(role)
    unit_ids = np.unique(units)
    unit_index = np.searchsorted(unit_ids, units)
    sums, counts = np.zeros((len(grid), len(unit_ids))), np.zeros(len(unit_ids))
    executions = {"input": [], "seed": [], "terminal": [], "actions": []}
    models = {int(s): ctx.plant(s) for s in entry["member_seeds"]}
    planned = None if entry["inputs"] is None else set(int(g) for g in entry["inputs"])
    sealed_agree = [] if not ctx.collection else None
    for input_id, x, graph, rows, positions in ctx.episodes(role):
        if planned is not None and int(input_id) not in planned:
            continue
        per_member = []
        for seed in entry["member_seeds"]:
            by_depth, trace = truncated_outputs(models[int(seed)], x, graph, config, candidate, controller,
                                                ctx.reference, grid, rows)
            per_member.append(by_depth)
            actions = [int(a) for a in trace["actions"]]
            executions["input"].append(int(input_id))
            executions["seed"].append(int(seed))
            executions["terminal"].append(len(actions))
            executions["actions"].append(actions + [-1] * (int(entry["depth"]) - len(actions)))
            if sealed_agree is not None:     # diagnostic: the select stage's sealed full-graph member trace
                sealed = f"member_predictions/{candidate['behavior_id']}_seed{seed}.json"
                if (run.root / sealed).is_file():
                    sealed_agree.append(run.json(sealed).get("actions") == actions)
        y, index = labels[positions], unit_index[positions]
        valid = None
        for k in range(len(grid)):
            ensemble = ensemble_probabilities(member[k] for member in per_member)
            values = sample_losses(ensemble, y, spec)
            if valid is None:
                valid = np.isfinite(values)
                np.add.at(counts, index[valid], 1)
            np.add.at(sums[k], index[valid], values[valid])
    scored = counts > 0
    if not scored.any():
        raise RunnerError(f"{run.root.name}: the {role} role holds no scored label")
    alignment = source_alignments(ctx, entry, ctx.device) if entry["alignment_seeds"] else None
    write_depth_job(directory, plan, entry, unit_ids=unit_ids[scored], unit_counts=counts[scored],
                    independent=independent_graphs(config), unit_depth_loss=sums[:, scored] / counts[scored],
                    executions=executions, alignment=alignment, entry_dir=run.root, started=started,
                    parent_files=run.inputs(),
                    sealed_actions_agree=None if not sealed_agree else all(sealed_agree))
