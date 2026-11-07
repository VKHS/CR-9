"""E14: the comparator on the extended ensemble grid, plus the runner infrastructure shared by
the four record-set experiments (E13 attribution, E14 grid, cross-backbone substitutions,
equal-budget factorials).

E14 (Supplementary Tables S26 and S26b; record set ``ensemble_grid``) executes the locked
open-loop comparator q_C of a sealed lifecycle run at every size of the extended grid
E in {1, 2, 4, 8, 12, 16, 20} (paper/registered_constants.json,
``observer_eq12_table_S9.extended_diagnostic_grid_E14``).  The E-member ensemble is the
prefix of one seed order -- the parent's registered checkpoints followed by additional
checkpoints that this runner trains with the parent's frozen training configuration
(``cross_morse.training.train_one``) -- so every size uses E distinct checkpoints, all
sizes share their members (paired seeds) and are scored on the same units (paired units).
Each checkpoint is executed once with ``cross_morse.runtime.predict_dataset``; an ensemble
is the mean of its members' probabilities, accumulated exactly as
``cross_morse.pipeline.Pipeline.predict_candidate`` accumulates them.  Repeated evaluation
of one checkpoint never counts as an additional member.

Workflow (``scripts/run_experiment.py ensemble-grid plan|run|export``):

    plan    parents and settings -> <output>/plan.json (identities of every parent run)
    run     one job per parent run (a Roman-empire campaign has one job per official split)
            -> <output>/jobs/<dataset>/<run>/{checkpoints/, members/, unit_losses.npz, grid.json}
    export  -> records/ensemble_grid (summary.csv, raw/, code/, provenance.json, MANIFEST.json)

The infrastructure below (sealed-run identities, plan files, the frozen-parent execution
context, paired statistics and the record-set writer) is imported by
``cm_experiments.attribution``, ``cm_experiments.backbones`` and ``cm_experiments.factorial``.
Planning, statistics and export are torch-free; execution imports the runtime lazily.
"""
from __future__ import annotations

import csv
import json
import math
import os
import platform
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .artifacts import canonical, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "records"
SPECS_PATH = RECORDS / "specs.json"
CONSTANTS_PATH = ROOT / "paper" / "registered_constants.json"
PLAN_SCHEMA = "cm-record-experiment-plan-1"
ROLE_STAGES = {"compare": "compare", "test": "evaluate"}    # label role -> runtime label scope
FEEDBACK_ROUTES = ("one_step", "radius", "mpc_rh")
CONTROLLER_ROUTES = ("phase", "fixed_sequence", "mpc_ol", "mpc_rh", "one_step", "radius")
STAGES = ("prepare", "train", "explore", "fit", "select", "confirm", "admit", "evaluate")
RUNNER_FILES = ("cm_experiments/ensemble_grid.py", "cm_experiments/attribution.py",
                "cm_experiments/backbones.py", "cm_experiments/factorial.py", "scripts/run_experiment.py")
SUMMARY_TAIL = ("quantity", "value", "unit", "source_file", "locator")
# Official scores as Table S26 prints them: percent / F1 / AUC points, AP in native units.
OFFICIAL_SCORE = {"accuracy": (100.0, "%"), "macro_f1": (100.0, "F1 points"), "rocauc": (100.0, "AUC points"),
                  "ap": (1.0, "AP")}


class RunnerError(ValueError):
    """The inputs do not support a complete, traceable execution or export."""


# ============================================================================= shared: files and identities
def read_json(path):
    path = Path(path)
    if not path.is_file():
        raise RunnerError(f"missing input file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slug(text) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")
    if not out:
        raise RunnerError(f"cannot form a file name from {text!r}")
    return out


def json_safe(value):
    """Plain JSON for numpy scalars/arrays and tensors in runtime traces; nonfinite -> None."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return json_safe(value.detach().cpu().numpy())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_array(path, array, *, replace=False) -> str:
    """Atomic .npy write; an existing file is replaced only on request. Returns its SHA-256."""
    path = Path(path)
    if path.exists() and not replace:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npy")
    os.close(fd)
    try:
        np.save(temporary, np.asarray(array), allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return sha256(path)


def save_arrays(path, *, replace=False, **arrays) -> str:
    path = Path(path)
    if path.exists() and not replace:
        raise FileExistsError(path)
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


def runtime_source_sha256() -> str:
    """The sealed runtime's identity (run_cross_morse*.py and cross_morse/*.py)."""
    from cross_morse.provenance import current_source_sha256
    return current_source_sha256()


def runner_source_identity() -> dict:
    return {name: sha256(ROOT / name) for name in RUNNER_FILES if (ROOT / name).is_file()}


def registered_constants() -> dict:
    return read_json(CONSTANTS_PATH)


def member_runs(root) -> list[Path]:
    """The split runs of a Roman-empire campaign directory, otherwise the run itself."""
    root = Path(root).resolve()
    if (root / "config.json").is_file():
        return [root]
    splits = sorted(p for p in root.glob("split_*") if (p / "config.json").is_file())
    if not splits:
        raise RunnerError(f"{root} is neither a lifecycle run nor a campaign of split runs")
    return splits


class SealedRun:
    """Read-only view of one lifecycle run; every file read is identified by SHA-256."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.config = read_json(self.root / "config.json")
        self.state = read_json(self.root / "run_state.json")
        self.read: dict[str, str] = {}

    @property
    def dataset(self) -> str:
        return self.config.get("benchmark_dataset") or self.config["dataset"]

    @property
    def synthetic(self) -> bool:
        return self.config.get("dataset") == "synthetic" or self.config.get("scientific_run") is False

    @property
    def last_stage(self):
        return self.state.get("last_complete_stage")

    def sealed(self, stage: str) -> bool:
        marker = self.root / "stages" / f"{stage}.json"
        manifest = self.root / "provenance" / "stages" / f"{STAGES.index(stage):02d}_{stage}.json"
        return (marker.is_file() and manifest.is_file()
                and json.loads(marker.read_text()).get("status") == "COMPLETE")

    def require(self, stage: str) -> None:
        if not self.sealed(stage):
            raise RunnerError(f"{self.root.name}: lifecycle stage {stage} is not sealed")

    def json(self, relative: str):
        value = read_json(self.root / relative)
        self.read[relative] = sha256(self.root / relative)
        return value

    def npz(self, relative: str) -> dict:
        path = self.root / relative
        if not path.is_file():
            raise RunnerError(f"missing input file: {path}")
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        self.read[relative] = sha256(path)
        return arrays

    def candidates(self) -> list[dict]:
        return self.json("select/candidates.json")

    def comparator(self) -> dict:
        index = self.json("select/decision.json")["comparator_index"]
        return dict(self.candidates()[index], registry_index=index)

    def confirmed(self) -> dict:
        index = self.json("confirm/decision.json")["confirmed_index"]
        return dict(self.candidates()[index], registry_index=index)

    def deployed(self):
        if not (self.root / "policy_lock.json").is_file():
            return None
        index = self.json("policy_lock.json")["deployed_index"]
        return dict(self.candidates()[index], registry_index=index)

    def identity(self) -> dict:
        manifests = {path.name: sha256(path) for path in sorted((self.root / "provenance" / "stages").glob("*.json"))}
        units = self.root / "prepared" / "splits.npz"
        return {"name": self.root.name, "dataset": self.dataset,
                "synthetic": self.synthetic, "config_sha256": canonical(self.config),
                "run_state_config_sha256": self.state.get("config_sha256"),
                "runtime_source_sha256": self.state.get("source_sha256"),
                "units_sha256": sha256(units) if units.is_file() else None,
                "last_complete_stage": self.last_stage, "stage_manifests": manifests}

    def inputs(self) -> list[dict]:
        return [{"file": name, "sha256": digest} for name, digest in sorted(self.read.items())]


def check_role(run: SealedRun, role: str) -> None:
    """compare: the reserved comparison units after the teacher is fixed; test: post hoc only."""
    if role not in ROLE_STAGES:
        raise RunnerError(f"evaluation role must be one of {sorted(ROLE_STAGES)}; decision roles are never reused")
    if role == "test":
        run.require("evaluate")
        if run.state.get("status") != "COMPLETE":
            raise RunnerError(f"{run.root.name}: the test role is opened only after the run is sealed")
    else:
        from cross_morse.split_contract import require_reserved_comparison
        run.require("confirm")
        try:
            require_reserved_comparison(run.json("prepared/metadata.json"), run.npz("prepared/splits.npz"))
        except ValueError as exc:
            raise RunnerError(f"{run.root.name}: {exc}") from None


def check_identity(run: SealedRun, planned: dict) -> None:
    current = run.identity()
    for key in ("config_sha256", "runtime_source_sha256", "units_sha256", "stage_manifests"):
        if current.get(key) != planned.get(key):
            raise RunnerError(f"{run.root.name}: {key} changed after planning; plan the experiment again")


def candidate_identity(candidate: dict) -> dict:
    keys = ("route", "profile", "ensemble", "horizon", "kappa", "depth", "sequence", "candidate_id", "hash",
            "registry_index", "site_restriction", "behavior_id")
    return {k: candidate[k] for k in keys if k in candidate}


def environment_record(device) -> dict:
    record = {"python": platform.python_version(), "platform": platform.platform(),
              "machine": platform.machine(), "device_requested": str(device), "numpy": np.__version__,
              "slurm": {k: os.environ[k] for k in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
                                                   "SLURM_JOB_PARTITION") if k in os.environ}}
    try:
        import torch
        record.update(torch=torch.__version__, cuda_runtime=torch.version.cuda,
                      gpu=torch.cuda.get_device_name(0) if str(device).startswith("cuda") and torch.cuda.is_available() else None,
                      torch_threads=torch.get_num_threads())
    except ImportError:
        record["torch"] = None
    return record


# ============================================================================= shared: plans
def create_plan(output, payload: dict) -> Path:
    output = Path(output).expanduser().absolute()
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise RunnerError("the experiment directory must not traverse symbolic links")
    output = output.resolve()
    if output.is_relative_to(ROOT):
        raise RunnerError("write experiment outputs outside the repository checkout")
    for parent in payload.get("parent_runs") or [job["run"] for job in payload["jobs"]]:
        run = Path(parent).resolve()
        if output == run or output.is_relative_to(run) or run.is_relative_to(output):
            raise RunnerError("the experiment directory must be separate from every parent run")
    if (output / "plan.json").exists():
        raise RunnerError(f"{output}/plan.json exists; use a new experiment directory")
    output.mkdir(parents=True, exist_ok=True)
    plan = {"schema": PLAN_SCHEMA, "created_utc": utc_now(), "runtime_source_sha256": runtime_source_sha256(),
            "runner_source_sha256": runner_source_identity(), **payload}
    write_json(output / "plan.json", json_safe(plan))
    return output / "plan.json"


def load_plan(output, record_set: str) -> dict:
    plan = read_json(Path(output) / "plan.json")
    if plan.get("schema") != PLAN_SCHEMA or plan.get("record_set") != record_set:
        raise RunnerError(f"{output}/plan.json is not a {record_set} plan")
    return plan


def selected_jobs(plan: dict, job=None) -> list[dict]:
    jobs = plan["jobs"]
    if job is None:
        return jobs
    if not 0 <= int(job) < len(jobs):
        raise RunnerError(f"job index {job} outside 0..{len(jobs) - 1}")
    return [jobs[int(job)]]


def job_directory(output, job: dict) -> Path:
    return Path(output) / "jobs" / job["slug"] / slug(job["member"])


# ============================================================================= shared: paired statistics
def weighted_standard_error(values, weights=None) -> float:
    """Linearized standard error of a fixed-weight unit mean (sd/sqrt(n) for equal weights)."""
    d = np.asarray(values, dtype=np.float64)
    w = np.ones(len(d)) if weights is None else np.asarray(weights, dtype=np.float64)
    n = len(d)
    if n < 2:
        return float("nan")
    mean = float(np.dot(w, d) / w.sum())
    return float(math.sqrt(n / (n - 1) * float(np.sum((w * (d - mean)) ** 2))) / w.sum())


def paired_effect(differences, weights=None, *, reps: int, seed: int) -> dict:
    """Mean paired difference over units, its unit standard error and the registered
    paired whole-unit bootstrap 95% interval (cross_morse.statistics.paired_bootstrap)."""
    from cross_morse.statistics import paired_bootstrap
    d = np.asarray(differences, dtype=np.float64)
    if d.ndim != 1 or not len(d) or not np.isfinite(d).all():
        raise RunnerError("paired differences must be a nonempty finite vector")
    boot = paired_bootstrap(d, weights=weights, reps=reps, alpha=0.025, seed=seed)
    return {"mean": boot["mean"], "standard_error": weighted_standard_error(d, weights),
            "lower": boot["lower"], "upper": boot["upper"], "bootstrap_se": boot["se"],
            "units": int(len(d)), "weighting": boot["weighting"], "bootstrap_replicates": int(reps),
            "bootstrap_seed": int(seed), "interval": "central 95% paired whole-unit bootstrap"}


# ============================================================================= shared: record-set writer
def _git_head():
    from scripts.export_records import git_head
    return git_head()


def public_plan(output) -> str:
    """The plan as deposited under code/: absolute locations reduced to their final names."""
    def strip(value):
        if isinstance(value, dict):
            return {k: strip(v) for k, v in value.items()}
        if isinstance(value, list):
            return [strip(v) for v in value]
        if isinstance(value, str) and value.startswith("/") and len(Path(value).parts) > 2:
            return Path(value).name
        return value
    return json.dumps(strip(read_json(Path(output) / "plan.json")), indent=2, sort_keys=True) + "\n"


def write_record_set(set_name: str, rows: list[dict], files: dict, provenance: dict, *, root=None,
                     replace: bool = False, allow_partial: bool = False, command=None,
                     inputs=(), generated=None) -> Path:
    """Write records/<set> exactly as scripts/records.py verifies it, then install it atomically.

    ``rows``: the set's keys, quantity, value (float), unit, source_file (``raw/...`` or ``code/...``)
    and a locator into that file.  ``files``: {set-relative path: original file}, copied
    unchanged; ``generated``: {set-relative path: text} written by the exporter (the public plan).
    The set is staged, scanned for private infrastructure identifiers, hashed (MANIFEST.json),
    traced value by value (``scripts.records.verify_set``) and installed with
    ``scripts.record_mapping.install_verified_set``, which re-verifies the other deposited sets
    without re-indexing them.
    """
    from scripts.records import RECORDS as OFFICIAL, is_official_root, load_specs, verify_set, write_manifest
    from scripts.export_records import fmt
    from scripts.record_mapping import _private_scan, install_verified_set
    generated = dict(generated or {})
    from scripts.scientific_evidence_audit import validate_record_coverage
    from cm_campaigns.contract import RECORD_SETS
    specs = load_specs(OFFICIAL, SPECS_PATH)
    if set_name not in specs["sets"]:
        raise RunnerError(f"{set_name} is not a record set of records/specs.json")
    spec = specs["sets"][set_name]
    root = Path(root or OFFICIAL).resolve()
    official = is_official_root(root)
    synthetic = provenance.get("synthetic") is True
    if official and (synthetic or allow_partial):
        raise RunnerError("records/ holds complete measured sets only; write synthetic or partial sets to another root")
    for field in specs["provenance_fields"]:
        if not str(provenance.get(field) or "").strip():
            raise RunnerError(f"provenance field {field!r} is empty; pass it explicitly")
    keys = list(spec["keys"])
    seen = set()
    for row in rows:
        if set(row) != set(keys) | set(SUMMARY_TAIL):
            raise RunnerError(f"{set_name}: a row must carry exactly {keys + list(SUMMARY_TAIL)}")
        if row["quantity"] not in spec["quantities"]:
            raise RunnerError(f"{set_name}: {row['quantity']} is not a quantity of the contract")
        if not str(row["unit"]).strip() or not all(str(row[k]).strip() for k in keys):
            raise RunnerError(f"{set_name}: empty key or unit in {row}")
        if not math.isfinite(float(row["value"])):
            raise RunnerError(f"{set_name}: nonfinite value in {row}")
        identity = tuple(str(row[k]) for k in keys) + (row["quantity"],)
        if identity in seen:
            raise RunnerError(f"{set_name}: duplicate row {identity}")
        seen.add(identity)
        if row["source_file"] not in files and row["source_file"] not in generated:
            raise RunnerError(f"{set_name}: {row['source_file']} is not among the deposited files")
    if not rows:
        raise RunnerError(f"{set_name}: no values to export")
    try:
        validate_record_coverage(set_name, rows, spec, RECORD_SETS[set_name])
        complete = True
    except ValueError as exc:
        if not allow_partial:
            raise RunnerError(f"{set_name}: coverage required by the scientific audit is incomplete: {exc}") from None
        complete = False
    root.mkdir(parents=True, exist_ok=True)
    staging_parent = ROOT.parent if root.is_relative_to(ROOT) else root.parent
    if staging_parent.stat().st_dev != root.stat().st_dev:
        raise RunnerError("staging and records must share a filesystem for atomic installation")
    staging = Path(tempfile.mkdtemp(prefix=f"cm-record-{set_name}-", dir=staging_parent))
    try:
        target = staging / set_name
        (target / "raw").mkdir(parents=True)
        (target / "code").mkdir()
        copied = []
        for relative in sorted(set(files) | set(generated)):
            parts = Path(relative).parts
            if (Path(relative).is_absolute() or ".." in parts or parts[0] not in ("raw", "code")
                    or relative == "code/SOURCE.json" or (relative in files and relative in generated)):
                raise RunnerError(f"set paths must be distinct and lie under raw/ or code/: {relative}")
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if relative in generated:
                destination.write_text(generated[relative], encoding="utf-8")
                copied.append({"path": relative, "generated": True, "sha256": sha256(destination)})
                continue
            source = Path(files[relative])
            if source.is_symlink() or not source.is_file():
                raise RunnerError(f"original output missing or not a regular file: {source}")
            shutil.copyfile(source, destination)
            if sha256(destination) != sha256(source):
                raise RunnerError(f"{source} changed while it was copied")
            copied.append({"path": relative, "original_name": source.name, "sha256": sha256(destination)})
        with (target / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(keys + list(SUMMARY_TAIL))
            for row in rows:
                writer.writerow([row[k] for k in keys] + [row["quantity"], fmt(row["value"]), row["unit"],
                                                          row["source_file"], row["locator"]])
        (target / "code" / "SOURCE.json").write_text(json.dumps({
            "schema": "cross-morse-record-source-1", "exporter": "scripts/run_experiment.py",
            "command": [Path(a).name if Path(a).is_absolute() else a for a in (command or [])],
            "runtime_source_sha256": runtime_source_sha256(), "runner_source_sha256": runner_source_identity(),
            "repository_commit": _git_head(), "inputs": list(inputs), "deposited_files": copied},
            indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        record = dict(json_safe(provenance), coverage_complete=complete, synthetic=synthetic)
        (target / "provenance.json").write_text(json.dumps(record, indent=1, sort_keys=True, allow_nan=False) + "\n",
                                                encoding="utf-8")
        private = _private_scan(target)
        if private:
            raise RunnerError("redact private infrastructure identifiers before depositing:\n  " + "\n  ".join(private[:10]))
        write_manifest(target)
        verify_set(target, spec, specs)
        return install_verified_set(staging, target, root, specs, replace=replace)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def code_files(output) -> dict:
    """The runner sources deposited under code/ (the public plan is passed as generated text)."""
    return {f"code/{name}": ROOT / name for name in RUNNER_FILES if (ROOT / name).is_file()}


def provenance_period(results: list[dict]) -> str:
    starts = [r["started_utc"] for r in results if r.get("started_utc")]
    ends = [r["finished_utc"] for r in results if r.get("finished_utc")]
    if not starts or not ends:
        raise RunnerError("the job results carry no execution timestamps; pass --period")
    return f"{min(starts)}/{max(ends)}"


def provenance_hardware(environments: list[dict]) -> str:
    seen = set()
    for env in environments:
        device = env.get("gpu") or f"{env.get('machine', '?')} CPU"
        seen.add(f"{device}; torch {env.get('torch', '?')}; python {env.get('python', '?')}")
    if not seen:
        raise RunnerError("the job results carry no environment record; pass --hardware")
    return " | ".join(sorted(seen))


# ============================================================================= shared: frozen-parent execution (torch)
class ParentContext:
    """A sealed parent run opened for post hoc execution with the released runtime.

    The stage chain is verified against the current runtime source before any checkpoint
    is loaded; labels are read only for the declared role, inside the runtime's own label
    scope (``compare`` -> compare, ``test`` -> evaluate).
    """

    def __init__(self, run: SealedRun, device, *, through: str):
        from cross_morse.data import load_data
        from cross_morse.provenance import current_source_sha256, verify_stage_chain
        from cross_morse.runtime import graph_operator, is_graph_collection
        verify_stage_chain(run.root, through, expected_source_sha256=current_source_sha256())
        self.run, self.config, self.device = run, run.config, str(device)
        self.data = load_data(run.root).protect_labels()
        self.collection = is_graph_collection(self.data)
        self.graph = None if self.collection else graph_operator(self.data, self.device)
        self.reference = dict(np.load(run.root / "explore" / "nominal_reference.npz", allow_pickle=False))
        self._controllers, self._labels = {}, {}

    def plant(self, seed, directory=None):
        from cross_morse.training import load_plant
        return load_plant(self.data, self.config, Path(directory or self.run.root / "train"), int(seed), self.device)

    def controller(self, profile: str):
        from cross_morse.learning import Controller
        profile = "full" if profile in (None, "none") else profile
        if profile not in self._controllers:
            self._controllers[profile] = Controller.load(self.run.root / "fit" / profile, "cpu")
        return self._controllers[profile]

    def controller_for(self, candidate: dict):
        if candidate["route"] not in CONTROLLER_ROUTES:
            return None
        controller = self.controller(candidate.get("profile", "full"))
        if candidate["route"].startswith("mpc") and not controller.sequence_models_available:
            raise RunnerError("MPC configuration requires fitted transition/stage/terminal models")
        return controller

    def role_ids(self, role: str) -> np.ndarray:
        return np.asarray(self.data.splits[role], dtype=np.int64)

    def units(self, role: str) -> np.ndarray:
        return np.asarray(self.data.cluster_ids[self.role_ids(role)])

    def labels(self, role: str) -> np.ndarray:
        if role not in self._labels:
            with self.data._label_scope(ROLE_STAGES[role]):
                self._labels[role] = self.data.labels(role)
        return self._labels[role]

    def predict(self, model, candidate: dict, controller, role: str):
        """Role rows of one member's predictions and the runtime's summed execution trace."""
        from cross_morse.runtime import predict_dataset
        probabilities, totals = predict_dataset(model, self.data, self.graph, self.config, candidate, controller,
                                                self.reference, device=self.device,
                                                role=role if self.collection else None)
        rows = np.asarray(probabilities[self.role_ids(role)], dtype=np.float32)
        if not np.isfinite(rows).all():
            raise RunnerError(f"nonfinite predictions on the {role} role")
        return rows, totals

    def episodes(self, role: str):
        """Every input of the role, one at a time: (episode id, x, graph operator, local rows, positions).

        ``local rows`` select the role's rows from that execution's prediction matrix (the role's nodes
        of a transductive graph; every row of an independent graph) and ``positions`` place them in
        the role's label order.
        """
        from cross_morse.runtime import graph_operator
        ids = self.role_ids(role)
        if not self.collection:
            yield 0, self.data.x.to(self.device), self.graph, ids, np.arange(len(ids))
            return
        position = {int(g): i for i, g in enumerate(ids)}
        for graph_id in sorted(int(g) for g in self.data.graph_splits[role]):
            episode = self.data.graph(graph_id)
            rows = np.asarray(episode.global_sample_ids, dtype=np.int64)
            yield (graph_id, episode.x.to(self.device), graph_operator(episode, self.device),
                   np.arange(len(rows)), np.asarray([position[int(r)] for r in rows], dtype=np.int64))


def ensemble_probabilities(members) -> np.ndarray:
    """The runtime's ensemble: float64 sum over members in seed order, divided by E, as float32."""
    aggregate, count = None, 0
    for member in members:
        aggregate = np.array(member, dtype=np.float64) if aggregate is None else aggregate + member
        count += 1
    if not count:
        raise RunnerError("an ensemble needs at least one member")
    return (aggregate / count).astype(np.float32)


def write_member(path, rows, record: dict) -> dict:
    """One member's role-row predictions and its execution record (resume-safe)."""
    path = Path(path)
    meta = path.with_suffix(".json")
    if meta.exists():
        raise RunnerError(f"{path} exists with its record; refusing to replace an executed member")
    digest = save_array(path, rows, replace=True)    # an orphan without its record is an interrupted write
    write_json(meta, json_safe(dict(record, prediction_file=path.name, prediction_sha256=digest)))
    return read_json(meta)


def read_member(path, expected: dict) -> np.ndarray:
    path = Path(path)
    meta = read_json(path.with_suffix(".json"))
    if sha256(path) != meta.get("prediction_sha256"):
        raise RunnerError(f"{path} differs from its recorded SHA-256")
    for key, value in expected.items():
        if json_safe(meta.get(key)) != json_safe(value):
            raise RunnerError(f"{path}: recorded {key} differs from the plan")
    return np.load(path, allow_pickle=False)


# ============================================================================= E14: planning
def registered_grid() -> list[int]:
    return [int(e) for e in registered_constants()["observer_eq12_table_S9"]["extended_diagnostic_grid_E14"]]


def seed_order(parent_seeds, extra_seeds, grid) -> list[int]:
    """Parent checkpoints in registered order, then the additional ones; exactly max(grid) members."""
    order = [int(s) for s in parent_seeds] + [int(s) for s in extra_seeds]
    if len(set(order)) != len(order):
        raise RunnerError("every ensemble member must be a distinct checkpoint seed")
    if max(grid) > len(order):
        raise RunnerError(f"E={max(grid)} needs {max(grid)} distinct checkpoints; {len(order)} are declared")
    if extra_seeds and len(order) > max(grid):
        raise RunnerError(f"declare exactly the {max(max(grid) - len(parent_seeds), 0)} additional checkpoint(s) "
                          f"that E={max(grid)} needs")
    return order[:max(grid)]


def default_extra_seeds(parent_seeds, grid) -> list[int]:
    needed = max(grid) - len(parent_seeds)
    start = max(int(s) for s in parent_seeds) + 1
    return list(range(start, start + max(needed, 0)))


def plan_ensemble_grid(parents, output, *, grid=None, extra_seeds=None, role="test", reps=None) -> Path:
    grid = sorted({int(e) for e in (grid or registered_grid())})
    if not grid or grid[0] < 1:
        raise RunnerError("the ensemble grid must contain positive sizes")
    if not parents:
        raise RunnerError("give at least one --parent run")
    datasets, jobs = [], []
    for parent in parents:
        runs = [SealedRun(m) for m in member_runs(parent)]
        dataset = runs[0].dataset
        if len({r.dataset for r in runs}) != 1 or any(d["dataset"] == dataset for d in datasets):
            raise RunnerError(f"{parent}: one parent (run or campaign) per dataset")
        members = []
        for run in runs:
            check_role(run, role)
            comparator = run.comparator()
            parent_seeds = [int(s) for s in run.config["training"]["seeds"]]
            extra = [int(s) for s in (extra_seeds if extra_seeds is not None else default_extra_seeds(parent_seeds, grid))]
            order = seed_order(parent_seeds, extra, grid)
            extra = [s for s in order if s not in parent_seeds]
            bootstrap = {"reps": int(reps or run.config["admission"]["bootstrap_reps"]),
                         "seed": int(run.config["seed"]) + 7}
            entry = {"job": len(jobs), "dataset": dataset, "slug": slug(dataset), "member": run.root.name,
                     "run": str(run.root), "identity": run.identity(), "comparator": candidate_identity(comparator),
                     "comparator_ensemble": int(comparator["ensemble"]), "parent_seeds": parent_seeds,
                     "extra_seeds": extra, "seed_order": order, "bootstrap": bootstrap}
            jobs.append(entry)
            members.append(run.root.name)
        datasets.append({"dataset": dataset, "slug": slug(dataset), "parent": str(Path(parent).resolve()),
                         "synthetic": runs[0].synthetic, "members": members,
                         "aggregation": "mean over the official split runs" if len(runs) > 1 else "single run"})
    return create_plan(output, {
        "experiment": "E14", "record_set": "ensemble_grid", "role": role, "grid": grid,
        "datasets": datasets, "jobs": jobs,
        "protocol": ("locked comparator q_C executed once per distinct checkpoint; E-member ensembles are prefixes "
                     "of one seed order (parent checkpoints, then additional checkpoints trained with the parent's "
                     "frozen training configuration); every size scored on the same units of the declared role")})


# ============================================================================= E14: computation (torch-free)
def grid_statistics(members, labels, units, config, grid, seeds, *, parent_seeds, comparator_ensemble,
                    deployed=None, reps: int, seed: int) -> tuple[list[dict], dict]:
    """Loss of every E-member prefix ensemble and its paired contrasts, from role-row member predictions.

    ``members`` yields one probability array per seed in ``seeds`` order (memory: one running sum).
    """
    from cross_morse.evaluation import score_observations
    grid = sorted(int(e) for e in grid)
    rows, arrays, aggregate, count = [], {}, None, 0
    losses = {}
    for member in members:
        aggregate = np.array(member, dtype=np.float64) if aggregate is None else aggregate + member
        count += 1
        if count not in grid:
            continue
        probabilities = (aggregate / count).astype(np.float32)
        score = score_observations(probabilities, labels, units, config)
        losses[count] = score
        arrays[f"E{count}_unit_loss"] = score["loss"]
        scale, score_unit = OFFICIAL_SCORE[score["metric_name"]]
        rows.append({"ensemble": count, "seeds": [int(s) for s in seeds[:count]], "distinct_checkpoints": count,
                     "new_checkpoints": sum(int(s) not in set(parent_seeds) for s in seeds[:count]),
                     "loss_nats": score["node_ce"], "loss_name": score["loss_name"],
                     "metric": score["metric"], "metric_name": score["metric_name"],
                     "official_score": scale * score["metric"], "official_score_unit": score_unit,
                     "units": int(len(score["core_ids"]))})
        if count == max(grid):
            break
    if [r["ensemble"] for r in rows] != grid:
        raise RunnerError(f"the member sequence does not reach every ensemble size of {grid}")
    first = losses[grid[0]]
    arrays.update(unit_id=first["core_ids"], unit_weight=first["counts"])
    reference = losses.get(int(comparator_ensemble))
    deployed_score = None
    if deployed is not None:
        deployed_score = score_observations(np.asarray(deployed, dtype=np.float32), labels, units, config)
        if not np.array_equal(deployed_score["core_ids"], first["core_ids"]):
            raise RunnerError("deployed-rule and grid units differ")
        arrays["deployed_unit_loss"] = deployed_score["loss"]
    for row in rows:
        score = losses[row["ensemble"]]
        if not np.array_equal(score["core_ids"], first["core_ids"]):
            raise RunnerError("ensemble sizes were scored on different units")
        if reference is not None:
            row["minus_comparator_ensemble"] = paired_effect(score["loss"] - reference["loss"], score["counts"],
                                                             reps=reps, seed=seed)
        if deployed_score is not None:
            row["deployed_minus_row"] = paired_effect(deployed_score["loss"] - score["loss"], score["counts"],
                                                      reps=reps, seed=seed)
    return rows, arrays


# ============================================================================= E14: execution (torch)
def run_ensemble_grid(output, *, job=None, device="cuda", trust=False) -> list[Path]:
    if not trust:
        raise RunnerError("loading local checkpoints requires explicit trust (--trust-local-checkpoints)")
    output = Path(output).resolve()
    plan = load_plan(output, "ensemble_grid")
    done = []
    for entry in selected_jobs(plan, job):
        directory = job_directory(output, entry)
        if not (directory / "grid.json").is_file():
            _run_grid_job(plan, entry, directory, device)
        done.append(directory / "grid.json")
    return done


def _run_grid_job(plan, entry, directory, device):
    from cross_morse.training import train_one
    run = SealedRun(entry["run"])
    check_identity(run, entry["identity"])
    role = plan["role"]
    check_role(run, role)
    comparator = run.comparator()
    if candidate_identity(comparator) != entry["comparator"]:
        raise RunnerError(f"{run.root.name}: the locked comparator differs from the plan")
    started = utc_now()
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / "environment.json").exists():
        write_json(directory / "environment.json", environment_record(device))
    ctx = ParentContext(run, device, through="evaluate" if role == "test" else "confirm")
    checkpoints = directory / "checkpoints"
    trained = []
    for seed in entry["extra_seeds"]:
        # The parent's frozen training configuration and the D_fit roles; only the seed differs.
        with ctx.data._label_scope("train"):
            result = train_one(ctx.data, ctx.graph, ctx.config, checkpoints, int(seed), ctx.device)
        trained.append({"seed": int(seed), "origin": "extra", "sha256": sha256(checkpoints / f"plant_seed{seed}_best.pt"),
                        "training": json_safe(result)})
    parent_checkpoints = [{"seed": int(s), "origin": "parent", "sha256": sha256(run.root / "train" / f"plant_seed{s}_best.pt")}
                          for s in entry["seed_order"] if s in entry["parent_seeds"]]
    controller = ctx.controller_for(comparator)
    expected = {"candidate": candidate_identity(comparator), "role": role}
    files = []
    for seed in entry["seed_order"]:
        path = directory / "members" / f"seed{seed}.npy"
        source = run.root / "train" if seed in entry["parent_seeds"] else checkpoints
        if not path.with_suffix(".json").exists():
            model = ctx.plant(seed, source)
            rows, totals = ctx.predict(model, comparator, controller, role)
            write_member(path, rows, {**expected, "seed": int(seed),
                                      "checkpoint_sha256": sha256(source / f"plant_seed{seed}_best.pt"),
                                      "trace": totals})
            del model
        files.append({"seed": int(seed), "file": f"members/seed{seed}.npy", "sha256": sha256(path)})
    members = (read_member(directory / f["file"], expected) for f in files)
    labels, units = ctx.labels(role), ctx.units(role)
    deployed = None
    if role == "test":
        deployed = np.load(run.root / "final" / "qD_probabilities.npy", mmap_mode="r")[ctx.role_ids(role)]
    rows, arrays = grid_statistics(members, labels, units, ctx.config, plan["grid"], entry["seed_order"],
                                   parent_seeds=entry["parent_seeds"], comparator_ensemble=entry["comparator_ensemble"],
                                   deployed=deployed, reps=entry["bootstrap"]["reps"], seed=entry["bootstrap"]["seed"])
    reproduction = None
    if role == "test":
        sealed = run.json("final/results_summary.json")["models"]["qC"]["loss_nats"]
        row = next((r for r in rows if r["ensemble"] == entry["comparator_ensemble"]), None)
        if row is not None:
            reproduction = {"sealed_qC_loss_nats": sealed, "grid_loss_nats": row["loss_nats"],
                            "abs_difference": abs(sealed - row["loss_nats"])}
    write_grid_result(directory, entry, role=role, rows=rows, arrays=arrays, checkpoints=parent_checkpoints + trained,
                      members=files, reproduction=reproduction, parent_files=run.inputs(), started=started)


def write_grid_result(directory, entry, *, role, rows, arrays, checkpoints, members, reproduction, parent_files,
                      started, environment=None) -> Path:
    """The job's raw outputs: unit_losses.npz and grid.json (written last: the completion marker)."""
    directory = Path(directory)
    if environment is not None and not (directory / "environment.json").exists():
        write_json(directory / "environment.json", json_safe(environment))
    # Derived files of an interrupted attempt are replaced; grid.json is the completion marker.
    unit_file = save_arrays(directory / "unit_losses.npz", replace=True, **arrays)
    write_json(directory / "grid.json", json_safe({
        "schema": "cm-ensemble-grid-member-1", "experiment": "E14", "dataset": entry["dataset"],
        "member": entry["member"], "role": role, "comparator": entry["comparator"],
        "comparator_ensemble": entry["comparator_ensemble"], "ensembles": rows,
        "checkpoints": checkpoints, "members": members,
        "unit_losses": {"file": "unit_losses.npz", "sha256": unit_file}, "qC_reproduction": reproduction,
        "identities": {"runtime_source_sha256": runtime_source_sha256(), "parent": entry["identity"],
                       "plan_sha256": sha256(directory.parents[2] / "plan.json"), "parent_files_read": parent_files},
        "loss_definition": "unit-weighted mean of per-unit mean CE/BCE (cross_morse.evaluation.score_observations node_ce)",
        "extra_checkpoint_scope": "trained with the parent's configuration and D_fit roles; the E1 adapter audit of "
                                  "cross_morse.training.audit_training is not repeated for them",
        "started_utc": started, "finished_utc": utc_now()}))
    return directory / "grid.json"


# ============================================================================= E14: export (torch-free)
def collect_ensemble_grid(output) -> list[dict]:
    """Per dataset: the member grids and their mean over split runs (single runs: the run itself)."""
    output = Path(output).resolve()
    plan = load_plan(output, "ensemble_grid")
    datasets = []
    for dataset in plan["datasets"]:
        entries = [j for j in plan["jobs"] if j["dataset"] == dataset["dataset"]]
        grids = []
        for entry in entries:
            path = job_directory(output, entry) / "grid.json"
            if not path.is_file():
                raise RunnerError(f"{dataset['dataset']}/{entry['member']}: job {entry['job']} has not completed")
            grid = read_json(path)
            if [r["ensemble"] for r in grid["ensembles"]] != plan["grid"] or grid["role"] != plan["role"]:
                raise RunnerError(f"{path}: grid or role differs from the plan")
            grids.append((entry, path, grid))
        ensembles = []
        for i, e in enumerate(plan["grid"]):
            values = [g["ensembles"][i]["loss_nats"] for _, _, g in grids]
            scores = [g["ensembles"][i]["official_score"] for _, _, g in grids]
            ensembles.append({"ensemble": e, "loss_nats": math.fsum(values) / len(values),
                              "member_losses_nats": values, "loss_name": grids[0][2]["ensembles"][i]["loss_name"],
                              "official_score": math.fsum(scores) / len(scores), "member_official_scores": scores,
                              "official_score_unit": grids[0][2]["ensembles"][i]["official_score_unit"]})
        datasets.append({"dataset": dataset["dataset"], "slug": dataset["slug"], "synthetic": dataset["synthetic"],
                         "aggregation": dataset["aggregation"], "ensembles": ensembles,
                         "members": [{"member": entry["member"], "grid": path, "grid_sha256": sha256(path),
                                      "unit_losses": path.parent / "unit_losses.npz",
                                      "environment": path.parent / "environment.json", "result": grid}
                                     for entry, path, grid in grids]})
    return datasets


def export_ensemble_grid(output, *, root=None, replace=False, allow_partial=False, hardware=None, period=None,
                         command=None) -> Path:
    output = Path(output).resolve()
    plan = load_plan(output, "ensemble_grid")
    datasets = collect_ensemble_grid(output)
    staging = Path(tempfile.mkdtemp(prefix="cm-e14-export-"))
    try:
        files, rows, results, environments = code_files(output), [], [], []
        for dataset in datasets:
            base = f"raw/{dataset['slug']}"
            members = []
            for member in dataset["members"]:
                name = slug(member["member"])
                files[f"{base}/{name}/grid.json"] = member["grid"]
                files[f"{base}/{name}/unit_losses.npz"] = member["unit_losses"]
                files[f"{base}/{name}/environment.json"] = member["environment"]
                members.append({"member": member["member"], "file": f"{base}/{name}/grid.json",
                                "sha256": member["grid_sha256"]})
                results.append(member["result"])
                environments.append(read_json(member["environment"]))
            summary = {"schema": "cm-ensemble-grid-dataset-1", "dataset": dataset["dataset"], "role": plan["role"],
                       "aggregation": dataset["aggregation"], "members": members, "ensembles": dataset["ensembles"],
                       "definition": "comparator loss of the E-member prefix ensemble (nats); with several split "
                                     "runs, the mean of the split runs' losses"}
            path = staging / dataset["slug"] / "ensemble_grid.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(json_safe(summary), indent=1, sort_keys=True, allow_nan=False) + "\n")
            files[f"{base}/ensemble_grid.json"] = path
            for i, row in enumerate(dataset["ensembles"]):
                rows.append({"dataset": dataset["dataset"], "ensemble": str(row["ensemble"]), "quantity": "loss_nats",
                             "value": row["loss_nats"], "unit": "nats", "source_file": f"{base}/ensemble_grid.json",
                             "locator": f"json:/ensembles/{i}/loss_nats"})
        synthetic = {d["synthetic"] for d in datasets}
        if len(synthetic) != 1:
            raise RunnerError("synthetic and benchmark parents cannot share one record set")
        provenance = {
            "experiment": "E14 comparator on the extended ensemble grid (Supplementary Tables S26, S26b)",
            "produced_by": "scripts/run_experiment.py ensemble-grid plan/run/export (cm_experiments/ensemble_grid.py)",
            "hardware": hardware or provenance_hardware(environments), "period": period or provenance_period(results),
            "protocol": (f"locked comparator q_C of each sealed run executed on the {plan['role']} role at E in "
                         f"{plan['grid']}; E-member prefix ensembles of one seed order with E distinct checkpoints "
                         "(registered parent checkpoints, then additional checkpoints trained with the parent's "
                         "frozen configuration); loss = unit-weighted mean CE/BCE"),
            "synthetic": synthetic.pop(), "role": plan["role"], "grid": plan["grid"],
            "plan_sha256": sha256(output / "plan.json")}
        return write_record_set("ensemble_grid", rows, files, provenance, root=root, replace=replace,
                                allow_partial=allow_partial, command=command,
                                generated={"code/plan.json": public_plan(output)},
                                inputs=[{"file": str(Path(m["grid"]).relative_to(output)), "sha256": m["grid_sha256"]}
                                        for d in datasets for m in d["members"]])
    finally:
        shutil.rmtree(staging, ignore_errors=True)
