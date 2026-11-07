"""Per-input oracle enumerated on the admission split (Supplementary Table S30, Figure S13).

Convention C12: the oracle is a diagnostic enumerated on D_admit (the official validation split)
after every decision of the lifecycle is fixed (the run is complete and policy_lock.json exists).

Enumerated family S(x)
----------------------
Main Proposition 4 defines the per-input oracle over the supported sequences of the registered
family; the lifecycle's own headroom (select/entry_complete.json) is the finite evaluated family of
registered candidates, without per-input sequences or depths.  Here the family of input x is

    { a_{0:d-1} : a is the executed action sequence of a registered configuration c on x, 1 <= d <= |a| }

for every configuration c of select/candidates.json at the comparator's ensemble size E that is
resource-feasible at selection (no execution failure, normalized cost <= max_normalized_work; the
training-only entry-contract mask is a selection gate, not a support property, and is not applied),
plus q_C itself.  Each configuration is executed on x with its E registered checkpoints exactly as
the runtime executes it (cross_morse.policies.execute_policy, fresh policy state per input); the
frozen head is read after every controlled step, so the prefix a_{0:d-1} is the configuration
truncated at depth d (the head is shared at every depth; a truncated prefix is executed identically
because every route is causal).  An E-member element averages the members' readouts at depth d in
float64 (the runtime's ensemble); it exists while every member has reached d.  A prefix is
supported when every executed action of every member lies in that step's supported set or is a
registered recovery action (A8/A9) -- the runtime's support rule; a learned-stop configuration whose
members stop at different depths also contributes its complete execution (depth = deepest member).

Per input, the oracle is the element of least loss (the runtime's per-unit CE/BCE; ties: q_C, then the
shorter depth, then the registry order).  Its depth is the prefix length d; it differs from q_C when
its member action sequences (with their recorded last-safe anchors) are not q_C's complete ones.

Inputs are the runtime's statistical units of the admission split: complete graphs of the
independent-graph benchmarks, buffered clusters (cores) of a transductive graph (whose admission
samples share one execution per member), and the cores of every official split of a Roman-empire
campaign (pooled).

Workflow (``scripts/run_confirmation_analysis.py oracle plan|run|export``): one job per (run,
configuration); export enumerates the oracle from the per-depth loss arrays (torch-free).
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from .artifacts import sha256, write_json
from .confirmation_analysis import (ADMIT_ROLE, DATASETS, RoleContext, check_label_role, install_set,
                                    multigraph_contrasts, require_complete)
from .ensemble_grid import (RunnerError, SealedRun, candidate_identity, check_identity, create_plan,
                            environment_record, job_directory, json_safe, load_plan, member_runs, provenance_hardware,
                            provenance_period, public_plan, read_json, runtime_source_sha256, save_arrays,
                            selected_jobs, slug, utc_now)

ROOT = Path(__file__).resolve().parents[1]
ORACLE_FILES = ("cm_experiments/admission_oracle.py", "cm_experiments/confirmation_analysis.py",
                "cm_experiments/ensemble_grid.py", "scripts/run_confirmation_analysis.py")
RECOVERY_ACTIONS = (8, 9)
QUANTITY_UNITS = {"median_oracle_depth": "controlled steps", "oracle_depth_q25": "controlled steps",
                  "oracle_depth_q75": "controlled steps", "fraction_differs_from_qc": "fraction",
                  "inputs": "admission inputs"}


def oracle_code_identity() -> dict:
    return {name: sha256(ROOT / name) for name in ORACLE_FILES if (ROOT / name).is_file()}


# ============================================================================= family and planning (torch-free)
def oracle_family(run: SealedRun, routes=None) -> tuple[list[dict], dict]:
    """Registered configurations at the comparator's ensemble size; q_C is always a member."""
    candidates = run.candidates()
    comparator = run.comparator()
    ensemble = int(comparator["ensemble"])
    limit = float(run.config["selection"]["max_normalized_work"])
    family, excluded = [], []
    for i, c in enumerate(candidates):
        if int(c["ensemble"]) != ensemble:
            continue
        if i != comparator["registry_index"]:
            if routes is not None and c["route"] not in routes:
                excluded.append({"registry_index": i, "route": c["route"], "reason": "route not declared"})
                continue
            if c.get("execution_failure") or float(c.get("cost", float("inf"))) > limit:
                excluded.append({"registry_index": i, "route": c["route"],
                                 "reason": "execution failure at selection" if c.get("execution_failure")
                                 else f"normalized cost {c.get('cost')} above {limit}"})
                continue
        family.append(dict(c, registry_index=i))
    return family, {"ensemble": ensemble, "comparator_index": int(comparator["registry_index"]),
                    "max_normalized_work": limit, "excluded": excluded}


def plan_oracle(parents, output, *, routes=None, max_graphs=0) -> Path:
    if not parents:
        raise RunnerError("give at least one --parent run")
    if max_graphs < 0:
        raise RunnerError("--max-graphs must be nonnegative")
    datasets, jobs = [], []
    for parent in parents:
        runs = [SealedRun(m) for m in member_runs(parent)]
        dataset = runs[0].dataset
        if len({r.dataset for r in runs}) != 1 or any(d["dataset"] == dataset for d in datasets):
            raise RunnerError(f"{parent}: one parent (run or campaign) per dataset")
        members = []
        for run in runs:
            require_complete(run)
            check_label_role(run, ADMIT_ROLE)
            family, scope = oracle_family(run, routes)
            seeds = [int(s) for s in run.config["training"]["seeds"][:scope["ensemble"]]]
            for candidate in family:
                jobs.append({"job": len(jobs), "dataset": dataset, "slug": slug(dataset),
                             "member": f"{run.root.name}__{candidate['candidate_id']}", "run": str(run.root),
                             "run_name": run.root.name, "identity": run.identity(), "role": ADMIT_ROLE,
                             "candidate": candidate, "seeds": seeds, "max_graphs": int(max_graphs),
                             "is_comparator": candidate["registry_index"] == scope["comparator_index"]})
            members.append({"run": run.root.name, "family_size": len(family), **scope,
                            "comparator": candidate_identity(run.comparator())})
        datasets.append({"dataset": dataset, "slug": slug(dataset), "parent": str(Path(parent).resolve()),
                         "synthetic": runs[0].synthetic, "members": members,
                         "input_unit": "complete graph" if dataset in multigraph_contrasts() else "buffered cluster",
                         "aggregation": "inputs pooled over the official split runs" if len(runs) > 1 else "single run"})
    return create_plan(output, {
        "experiment": "Admission-split oracle (C12)", "record_set": "oracle_dispersion", "role": ADMIT_ROLE,
        "routes": None if routes is None else sorted(routes), "max_graphs": int(max_graphs),
        "truncated": max_graphs > 0, "datasets": datasets, "jobs": jobs,
        "experiment_code_sha256": oracle_code_identity(),
        "protocol": ("per-input oracle on D_admit after every decision is fixed: every depth prefix of the executed "
                     "sequences of the registered, resource-feasible configurations at the comparator's ensemble size "
                     "(and q_C), read out with the frozen head after each controlled step; per-input least loss")})


# ============================================================================= execution (torch)
class DepthReadout:
    """A frozen plant whose every controlled step also reads the frozen head (the prefix prediction).

    Every other attribute is the wrapped plant's own.  The readout repeats the runtime's final
    prediction, probabilities(model.logits(h).float(), task), on the state after each step, and
    restores the plant's operation counters so the executed trace is unchanged.
    """

    def __init__(self, plant, task, *, rows=None, sink=None):
        self._plant, self._task, self._rows, self._sink = plant, task, rows, sink
        self.readouts, self.steps, self.last = [], 0, None

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._plant, name)

    def step(self, h, h0, graph, action_id, last_safe=None, site=None):
        from cross_morse.tasks import probabilities
        state = self._plant.step(h, h0, graph, action_id, last_safe, site=site)
        counters = dict(self._plant.counters)
        p = probabilities(self._plant.logits(state).float(), self._task).cpu().numpy()
        self._plant.counters.clear()
        self._plant.counters.update(counters)
        if self._rows is not None:
            p = p[self._rows]
        self.steps += 1
        self.last = p
        if self._sink is not None:
            self._sink(self.steps, p)
        else:
            self.readouts.append(p)
        return state


def supported_prefix(decisions) -> int:
    """Number of leading executed actions that are supported (the runtime's rule: supported set or A8/A9)."""
    for t, decision in enumerate(decisions):
        allowed = decision.get("supported_actions")
        action = int(decision["action"])
        if allowed is not None and action not in allowed and action not in RECOVERY_ACTIONS:
            return t
    return len(decisions)


def unit_losses(probabilities, labels, unit_ids, config):
    """The loss part of cross_morse.evaluation.score_observations: per-unit mean CE/BCE and unit IDs."""
    from cross_morse.statistics import cluster_summary
    from cross_morse.tasks import sample_losses, task_spec
    values = sample_losses(probabilities, labels, task_spec(config))
    valid = np.isfinite(values)
    if not valid.any():
        return np.empty(0), np.empty(0, dtype=np.asarray(unit_ids).dtype), np.empty(0, dtype=np.int64)
    means, units, counts = cluster_summary(values[valid], np.asarray(unit_ids)[valid])
    return means, units, counts


def run_oracle(output, *, job=None, device="cuda", trust=False) -> list[Path]:
    if not trust:
        raise RunnerError("loading local checkpoints requires explicit trust (--trust-local-checkpoints)")
    output = Path(output).resolve()
    plan = load_plan(output, "oracle_dispersion")
    done = []
    for entry in selected_jobs(plan, job):
        directory = job_directory(output, entry)
        if not (directory / "behavior.json").is_file():
            _run_oracle_job(plan, entry, directory, device)
        done.append(directory / "behavior.json")
    return done


class _Accumulator:
    """Per-depth float64 ensemble sums of one input's prediction rows (the runtime's ensemble arithmetic)."""

    def __init__(self, depth, rows, outputs):
        self.sums = np.zeros((depth, rows, outputs), dtype=np.float64)
        self.final = np.zeros((rows, outputs), dtype=np.float64)
        self.reached = np.zeros(depth + 1, dtype=np.int64)

    def add(self, step, p):
        self.sums[step - 1] += p
        self.reached[step] += 1


def _execute_members(ctx, models, task, candidate, controller, x, graph, *, rows=None, accumulator=None):
    """All members on one input; returns per-member records or a failure."""
    from cross_morse.policies import PolicyExecutionError, execute_policy
    records = []
    for seed, model in models:
        recorder = DepthReadout(model, task, rows=rows, sink=accumulator.add)
        try:
            p, trace = execute_policy(recorder, x, graph, ctx.config, candidate, controller, ctx.reference)
        except PolicyExecutionError as error:
            return None, {"seed": int(seed), "reason": error.reason, "failure_stage": error.trace.get("failure_stage"),
                          "failed_depth": error.trace.get("failed_depth")}
        p = np.asarray(p, dtype=np.float32)
        if rows is not None:
            p = p[rows]
        actions = [int(a) for a in trace["actions"]]
        if recorder.steps != len(actions) or recorder.steps != int(trace["terminal_depth"]):
            raise RunnerError("the depth readout missed controlled steps of the executed trace")
        if recorder.steps and not np.array_equal(recorder.last, p):
            raise RunnerError("the final-depth readout differs from the runtime's prediction")
        accumulator.final += p
        records.append({"seed": int(seed), "actions": actions, "safe_depths": [int(s) for s in trace["safe_depths"]],
                        "supported_prefix": supported_prefix(trace["decisions"]), "terminal": recorder.steps,
                        "feasible": bool(trace.get("feasible", True))})
    return records, None


def _run_oracle_job(plan, entry, directory, device):
    from cross_morse.common import stop_at_boundary
    from cross_morse.tasks import task_spec
    run = SealedRun(entry["run"])
    check_identity(run, entry["identity"])
    require_complete(run)
    check_label_role(run, ADMIT_ROLE)
    started = utc_now()
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / "environment.json").exists():
        write_json(directory / "environment.json", environment_record(device))
    ctx = RoleContext(run, device, through="evaluate")
    candidate = entry["candidate"]
    controller = ctx.controller_for(candidate)
    models = [(seed, ctx.plant(seed)) for seed in entry["seeds"]]
    task = getattr(models[0][1], "task", task_spec(ctx.config))
    members, depth = len(models), int(ctx.config["model"]["depth"])
    # Every decision is fixed (C12); the label-free executions below never read these labels.
    labels, unit_of_row = ctx.labels(ADMIT_ROLE), ctx.units(ADMIT_ROLE)
    role_ids = ctx.role_ids(ADMIT_ROLE)
    sealed, sealed_path = _sealed_prediction(run, candidate, ctx.collection)
    outputs = int(models[0][1].num_classes)
    inputs, failures, max_difference = [], [], 0.0
    if ctx.collection:
        for count, (graph_id, x, graph, local, positions) in enumerate(ctx.episodes(ADMIT_ROLE)):
            if entry["max_graphs"] and count >= entry["max_graphs"]:
                break
            accumulator = _Accumulator(depth, len(local), outputs)
            records, failure = _execute_members(ctx, models, task, candidate, controller, x, graph,
                                                accumulator=accumulator)
            if failure is not None:
                failures.append({"input": int(graph_id), **failure})
                continue
            item = _input_losses(accumulator, records, labels[positions], unit_of_row[positions], ctx.config, members)
            if item is not None:
                if sealed is not None:
                    stored = np.asarray(sealed[role_ids[positions]], dtype=np.float64)
                    ensemble = (accumulator.final / members).astype(np.float32)
                    max_difference = max(max_difference, float(np.max(np.abs(stored - ensemble))))
                inputs.append(item)
            stop_at_boundary()
    else:
        accumulator = _Accumulator(depth, len(role_ids), outputs)
        records, failure = _execute_members(ctx, models, task, candidate, controller, ctx.data.x.to(ctx.device),
                                            ctx.graph, rows=role_ids, accumulator=accumulator)
        if failure is not None:
            failures.append({"input": "transductive graph", **failure})
        else:
            inputs.extend(_core_losses(accumulator, records, labels, unit_of_row, ctx.config, members))
            if sealed is not None:
                stored = np.asarray(sealed[role_ids], dtype=np.float64)
                max_difference = float(np.max(np.abs(stored - (accumulator.final / members).astype(np.float32))))
    reproduction = {"sealed_prediction": sealed_path, "max_abs_difference": None if sealed is None else max_difference}
    if entry["is_comparator"] and not entry["max_graphs"]:
        reproduction.update(_comparator_reproduction(run, inputs))
    write_oracle_result(directory, entry, inputs=inputs, failures=failures, depth=depth, members=members,
                        reproduction=reproduction, parent_files=run.inputs(), started=started)


def _sealed_prediction(run, candidate, collection):
    """The lifecycle's stored ensemble prediction of this configuration (admission stage on independent
    graphs; the transductive select-stage prediction of a single graph), with its run-relative path."""
    path = (run.root / "predictions" / ADMIT_ROLE / f"{candidate['candidate_id']}.npy" if collection
            else run.root / "predictions" / f"{candidate['candidate_id']}.npy")
    if not path.is_file():
        return None, None
    return np.load(path, mmap_mode="r", allow_pickle=False), path.relative_to(run.root).as_posix()


def _input_losses(accumulator, records, labels, units, config, members):
    """One independent graph: per-depth and complete ensemble losses with its member sequences."""
    depth = accumulator.sums.shape[0]
    prefix = np.full(depth, np.nan)
    unit = None
    for d in range(1, depth + 1):
        if accumulator.reached[d] != members:
            break
        means, ids, _ = unit_losses((accumulator.sums[d - 1] / members).astype(np.float32), labels, units, config)
        if not len(means):
            return None
        prefix[d - 1], unit = means[0], ids[0]
    complete, ids, _ = unit_losses((accumulator.final / members).astype(np.float32), labels, units, config)
    if not len(complete):
        return None
    return {"unit_id": int(ids[0]), "prefix_loss": prefix, "complete_loss": float(complete[0]), "members": records}


def _core_losses(accumulator, records, labels, units, config, members):
    """A transductive graph: per-depth and complete per-core ensemble losses; the cores share the sequences."""
    depth = accumulator.sums.shape[0]
    complete, ids, counts = unit_losses((accumulator.final / members).astype(np.float32), labels, units, config)
    prefix = np.full((depth, len(ids)), np.nan)
    for d in range(1, depth + 1):
        if accumulator.reached[d] != members:
            break
        means, check, _ = unit_losses((accumulator.sums[d - 1] / members).astype(np.float32), labels, units, config)
        if not np.array_equal(check, ids):
            raise RunnerError("per-depth readouts were scored on different cores")
        prefix[d - 1] = means
    return [{"unit_id": int(u), "prefix_loss": prefix[:, k], "complete_loss": float(complete[k]),
             "weight": int(counts[k]), "members": records} for k, u in enumerate(ids)]


def _comparator_reproduction(run, inputs) -> dict:
    """q_C's complete per-unit losses against the admission stage's own comparator losses."""
    path = run.root / "admit" / "paired_units.npz"
    if not path.is_file() or not inputs:
        return {}
    with np.load(path, allow_pickle=False) as data:
        sealed = dict(zip(np.asarray(data["core_ids"]).tolist(), np.asarray(data["comparator_loss"]).tolist()))
    differences = [abs(sealed[item["unit_id"]] - item["complete_loss"]) for item in inputs if item["unit_id"] in sealed]
    return {"admission_comparator_loss": "admit/paired_units.npz",
            "units_compared": len(differences),
            "max_abs_loss_difference": max(differences) if differences else None}


def write_oracle_result(directory, entry, *, inputs, failures, depth, members, reproduction, parent_files, started,
                        environment=None) -> Path:
    """One configuration's per-input arrays (oracle_losses.npz) and behavior.json (the completion marker)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if environment is not None and not (directory / "environment.json").exists():
        write_json(directory / "environment.json", json_safe(environment))
    count = len(inputs)
    actions = np.full((members, count, depth), -1, dtype=np.int16)
    safe = np.full((members, count, depth), -1, dtype=np.int16)
    terminal = np.zeros((members, count), dtype=np.int16)
    support = np.zeros((members, count), dtype=np.int16)
    for u, item in enumerate(inputs):
        if len(item["members"]) != members:
            raise RunnerError("every input needs one record per ensemble member")
        for m, record in enumerate(item["members"]):
            t = int(record["terminal"])
            if t > depth or len(record["actions"]) != t or len(record["safe_depths"]) != t:
                raise RunnerError("member sequence and terminal depth disagree")
            actions[m, u, :t] = record["actions"]
            safe[m, u, :t] = record["safe_depths"]
            terminal[m, u] = t
            support[m, u] = int(record["supported_prefix"])
    arrays = {"unit_id": np.asarray([item["unit_id"] for item in inputs], dtype=np.int64),
              "prefix_loss": (np.stack([item["prefix_loss"] for item in inputs], axis=1) if inputs
                              else np.empty((depth, 0))),
              "complete_loss": np.asarray([item["complete_loss"] for item in inputs], dtype=np.float64),
              "unit_weight": np.asarray([item.get("weight", 1) for item in inputs], dtype=np.int64),
              "actions": actions, "safe_depths": safe, "terminal_depth": terminal, "supported_prefix": support}
    digest = save_arrays(directory / "oracle_losses.npz", replace=True, **arrays)
    candidate = entry["candidate"]
    write_json(directory / "behavior.json", json_safe({
        "schema": "cm-admission-oracle-configuration-1", "dataset": entry["dataset"], "run": entry["run_name"],
        "role": entry["role"], "candidate": candidate_identity(candidate), "route": candidate["route"],
        "is_comparator": entry["is_comparator"], "seeds": entry["seeds"], "members": members, "depth": depth,
        "inputs": count, "failures": failures, "truncated_to_graphs": entry["max_graphs"] or None,
        "arrays": {"file": "oracle_losses.npz", "sha256": digest}, "reproduction": reproduction,
        "definitions": {"prefix_loss": "[depth, input] loss of the E-member ensemble read out after d controlled steps "
                                       "(NaN once a member has stopped)",
                        "complete_loss": "loss of the configuration's own ensemble prediction",
                        "supported_prefix": "[member, input] number of leading executed actions in the supported set "
                                            "or A8/A9"},
        "identities": {"runtime_source_sha256": runtime_source_sha256(), "parent": entry["identity"],
                       "plan_sha256": sha256(directory.parents[2] / "plan.json"), "parent_files_read": parent_files,
                       "experiment_code_sha256": oracle_code_identity()},
        "started_utc": started, "finished_utc": utc_now()}))
    return directory / "behavior.json"


# ============================================================================= enumeration (torch-free)
def load_configuration(directory) -> dict:
    directory = Path(directory)
    result = read_json(directory / "behavior.json")
    if sha256(directory / "oracle_losses.npz") != result["arrays"]["sha256"]:
        raise RunnerError(f"{directory}/oracle_losses.npz differs from its recorded SHA-256")
    with np.load(directory / "oracle_losses.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    return {"result": result, **arrays}


def _best_element(config):
    """Per input: (loss, depth, kind, d) of the configuration's best supported element; inf when none."""
    loss, terminal, support = config["prefix_loss"], config["terminal_depth"], config["supported_prefix"]
    depth, inputs = loss.shape
    reach = np.minimum(terminal.min(axis=0), support.min(axis=0))          # longest prefix all members support
    steps = np.arange(1, depth + 1)[:, None]
    valid = np.isfinite(loss) & (steps <= reach[None, :])
    masked = np.where(valid, loss, np.inf)
    best_d = np.argmin(masked, axis=0)                                       # first minimum: the shorter depth
    best = masked[best_d, np.arange(inputs)]
    out_depth = (best_d + 1).astype(np.float64)
    kind = np.zeros(inputs, dtype=np.int8)                                   # 0 prefix, 1 complete execution
    unequal = terminal.min(axis=0) != terminal.max(axis=0)
    complete_ok = unequal & (support >= terminal).all(axis=0) & np.isfinite(config["complete_loss"])
    better = complete_ok & (config["complete_loss"] < best)
    best = np.where(better, config["complete_loss"], best)
    out_depth = np.where(better, terminal.max(axis=0), out_depth)
    kind = np.where(better, 1, kind)
    return best, out_depth, kind, best_d + 1


def _sequence(config, u, kind, d):
    """Member action/anchor sequences of one element on input u."""
    out = []
    for m in range(config["actions"].shape[0]):
        t = int(config["terminal_depth"][m, u]) if kind == 1 else int(d)
        out.append((tuple(config["actions"][m, u, :t].tolist()), tuple(config["safe_depths"][m, u, :t].tolist())))
    return tuple(out)


def _aligned(config, ids):
    """A configuration's arrays on q_C's inputs; an input it did not complete contributes no element."""
    if np.array_equal(config["unit_id"], ids):
        return config
    position = {int(u): k for k, u in enumerate(config["unit_id"].tolist())}
    if not set(position) <= set(ids.tolist()):
        raise RunnerError(f"{config['result']['candidate']['candidate_id']}: scored on different admission inputs")
    take = np.asarray([position.get(int(u), -1) for u in ids.tolist()], dtype=np.int64)
    present = take >= 0
    out = dict(config, unit_id=ids)
    depth, members = config["prefix_loss"].shape[0], config["terminal_depth"].shape[0]
    out["prefix_loss"] = np.full((depth, len(ids)), np.nan)
    out["prefix_loss"][:, present] = config["prefix_loss"][:, take[present]]
    out["complete_loss"] = np.full(len(ids), np.nan)
    out["complete_loss"][present] = config["complete_loss"][take[present]]
    for key, fill in (("terminal_depth", 0), ("supported_prefix", 0)):
        out[key] = np.full((members, len(ids)), fill, dtype=config[key].dtype)
        out[key][:, present] = config[key][:, take[present]]
    for key in ("actions", "safe_depths"):
        out[key] = np.full((members, len(ids), config[key].shape[2]), -1, dtype=config[key].dtype)
        out[key][:, present] = config[key][:, take[present]]
    return out


def enumerate_oracle(configurations) -> dict:
    """The per-input oracle of one run from its configurations' arrays (q_C flagged ``is_comparator``)."""
    comparators = [c for c in configurations if c["result"]["is_comparator"]]
    if len(comparators) != 1:
        raise RunnerError("the family must hold q_C exactly once")
    qc = comparators[0]
    ids = qc["unit_id"]
    if not len(ids):
        raise RunnerError("q_C has no scored admission input")
    order = sorted((_aligned(c, ids) for c in configurations),
                   key=lambda c: int(c["result"]["candidate"]["registry_index"]))
    best = np.full(len(ids), np.inf)
    depth = np.full(len(ids), np.nan)
    owner = np.full(len(ids), -1, dtype=np.int64)
    kinds = np.zeros(len(ids), dtype=np.int8)
    prefix = np.zeros(len(ids), dtype=np.int64)
    for index, config in enumerate(order):
        loss, d, kind, raw_d = _best_element(config)
        better = (loss < best) | ((loss == best) & (d < depth))              # registry order breaks the rest
        best = np.where(better, loss, best)
        depth = np.where(better, d, depth)
        owner = np.where(better, index, owner)
        kinds = np.where(better, kind, kinds)
        prefix = np.where(better, raw_d, prefix)
    qc_loss = qc["complete_loss"]
    qc_depth = qc["terminal_depth"].max(axis=0).astype(np.float64)
    qc_kind = np.where(qc["terminal_depth"].min(axis=0) != qc["terminal_depth"].max(axis=0), 1, 0)
    rows = []
    for u, unit in enumerate(ids.tolist()):
        if not np.isfinite(best[u]) or qc_loss[u] <= best[u]:
            rows.append({"unit_id": unit, "oracle_depth": float(qc_depth[u]), "differs_from_qc": False,
                         "oracle_loss": float(qc_loss[u]), "qc_loss": float(qc_loss[u]),
                         "oracle_registry_index": int(qc["result"]["candidate"]["registry_index"]),
                         "oracle_element": "q_C"})
            continue
        winner = order[int(owner[u])]
        same = _sequence(winner, u, int(kinds[u]), int(prefix[u])) == _sequence(qc, u, int(qc_kind[u]),
                                                                                  int(qc_depth[u]))
        rows.append({"unit_id": unit, "oracle_depth": float(depth[u]), "differs_from_qc": not same,
                     "oracle_loss": float(best[u]), "qc_loss": float(qc_loss[u]),
                     "oracle_registry_index": int(winner["result"]["candidate"]["registry_index"]),
                     "oracle_element": "complete execution" if kinds[u] == 1 else f"prefix d={int(prefix[u])}"})
    weights = qc["unit_weight"].astype(np.float64)
    gap = np.asarray([r["oracle_loss"] - r["qc_loss"] for r in rows])
    return {"per_input": rows,
            "delta_oracle_mean_nats": float(np.mean(gap)),
            "delta_oracle_weighted_nats": float(np.dot(weights, gap) / weights.sum()),
            "family": [{"registry_index": int(c["result"]["candidate"]["registry_index"]),
                        "route": c["result"]["route"], "candidate_id": c["result"]["candidate"]["candidate_id"],
                        "failures": len(c["result"]["failures"])} for c in order]}


def dispersion(depths, differs) -> dict:
    depths = np.asarray(depths, dtype=np.float64)
    if not len(depths):
        raise RunnerError("no admission input to summarize")
    q25, median, q75 = np.percentile(depths, [25, 50, 75])
    return {"median_oracle_depth": float(median), "oracle_depth_q25": float(q25), "oracle_depth_q75": float(q75),
            "fraction_differs_from_qc": float(np.mean(np.asarray(differs, dtype=bool))), "inputs": int(len(depths)),
            "quantile_method": "numpy.percentile, linear interpolation"}


# ============================================================================= export (torch-free)
def export_oracle(output, *, root=None, replace=False, allow_partial=False, hardware=None, period=None,
                  command=None) -> Path:
    from scripts.records import RECORDS, is_official_root
    output = Path(output).resolve()
    plan = load_plan(output, "oracle_dispersion")
    target_root = Path(root or RECORDS).resolve()
    if plan.get("truncated") and is_official_root(target_root):
        raise RunnerError("a truncated (--max-graphs) oracle is a software check; write it to another --root")
    staging = Path(tempfile.mkdtemp(prefix="cm-oracle-export-"))
    try:
        files = {f"code/{name}": ROOT / name for name in ORACLE_FILES if (ROOT / name).is_file()}
        rows, environments, results, inputs = [], [], [], []
        for dataset in plan["datasets"]:
            depths, differs, per_run = [], [], []
            base = f"raw/{dataset['slug']}"
            for member in dataset["members"]:
                entries = [j for j in plan["jobs"] if j["dataset"] == dataset["dataset"] and j["run_name"] == member["run"]]
                configurations = []
                for entry in entries:
                    directory = job_directory(output, entry)
                    if not (directory / "behavior.json").is_file():
                        raise RunnerError(f"{dataset['dataset']}/{entry['member']}: job {entry['job']} has not completed")
                    configuration = load_configuration(directory)
                    if configuration["result"]["candidate"] != json_safe(candidate_identity(entry["candidate"])):
                        raise RunnerError(f"{directory}: the executed configuration differs from the plan")
                    configurations.append(configuration)
                    name = f"{base}/{slug(member['run'])}/{slug(entry['candidate']['candidate_id'])}"
                    files[f"{name}/behavior.json"] = directory / "behavior.json"
                    files[f"{name}/oracle_losses.npz"] = directory / "oracle_losses.npz"
                    files[f"{name}/environment.json"] = directory / "environment.json"
                    environments.append(read_json(directory / "environment.json"))
                    results.append(configuration["result"])
                    inputs.append({"file": str((directory / "behavior.json").relative_to(output)),
                                   "sha256": sha256(directory / "behavior.json")})
                oracle = enumerate_oracle(configurations)
                run_depths = [r["oracle_depth"] for r in oracle["per_input"]]
                run_differs = [r["differs_from_qc"] for r in oracle["per_input"]]
                depths += run_depths
                differs += run_differs
                per_run.append({"run": member["run"], **dispersion(run_depths, run_differs),
                                "comparator": member["comparator"], "ensemble": member["ensemble"],
                                "family_size": member["family_size"], "excluded": member["excluded"], **oracle})
            summary = {"schema": "cm-oracle-dispersion-dataset-1", "dataset": dataset["dataset"],
                       "role": "D_admit (official validation split)", "input_unit": dataset["input_unit"],
                       "aggregation": dataset["aggregation"], **dispersion(depths, differs), "runs": per_run,
                       "truncated": bool(plan.get("truncated")),
                       "definitions": {"oracle_depth": "prefix length d of the per-input least-loss element of the "
                                                       "enumerated family (depth of the deepest member for a complete "
                                                       "learned-stop execution)",
                                       "fraction_differs_from_qc": "share of inputs whose oracle member sequences are "
                                                                   "not q_C's complete sequences"}}
            path = staging / dataset["slug"] / "oracle_dispersion.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(json_safe(summary), indent=1, sort_keys=True, allow_nan=False) + "\n")
            files[f"{base}/oracle_dispersion.json"] = path
            for quantity, unit in QUANTITY_UNITS.items():
                rows.append({"dataset": dataset["dataset"], "quantity": quantity, "value": summary[quantity],
                             "unit": unit, "source_file": f"{base}/oracle_dispersion.json",
                             "locator": f"json:/{quantity}"})
        synthetic = {d["synthetic"] for d in plan["datasets"]}
        if len(synthetic) != 1:
            raise RunnerError("synthetic and benchmark parents cannot share one record set")
        provenance = {
            "experiment": "Per-input oracle enumerated on the admission split (Supplementary Table S30, Figure S13; C12)",
            "produced_by": "scripts/run_confirmation_analysis.py oracle plan/run/export (cm_experiments/admission_oracle.py)",
            "hardware": hardware or provenance_hardware(environments), "period": period or provenance_period(results),
            "protocol": ("D_admit after every decision is fixed; family: every depth prefix of the executed sequences of "
                         "the registered, resource-feasible configurations at the comparator's ensemble size and q_C, "
                         "E-member ensemble readouts of the frozen head after each step, supported prefixes only; "
                         "per-input least CE/BCE (ties: q_C, shorter depth, registry order); inputs: graphs or cores"),
            "synthetic": synthetic.pop(), "role": ADMIT_ROLE, "routes": plan.get("routes"),
            "truncated": bool(plan.get("truncated")), "plan_sha256": sha256(output / "plan.json"),
            "experiment_code_sha256": oracle_code_identity()}
        coverage = {"tables": ["S30", "Figure S13"], "datasets": list(DATASETS), "analyzer": "oracle_dispersion"}
        return install_set("oracle_dispersion", rows, files, provenance, coverage=coverage, root=root, replace=replace,
                           allow_partial=allow_partial or bool(plan.get("truncated")), command=command, inputs=inputs,
                           generated={"code/plan.json": public_plan(output)})
    finally:
        shutil.rmtree(staging, ignore_errors=True)
