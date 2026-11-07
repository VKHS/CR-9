#!/usr/bin/env python3
"""Suites, progress and the report of the reproduction launcher (run_cross_morse.sh).

`run_cross_morse.sh reproduce` executes the registered dataset lifecycles as one
numbered *reproduction suite*: the eight-stage lifecycle of every dataset
(Roman-empire on its ten official splits) followed by one report. The shell
script owns the Slurm side (submission, requeue, logs); this module owns the
files: it freezes the runs of a new suite, reads their progress and writes the
report that places every regenerated value beside the value printed in the paper.

Standard library only, so that the login node needs nothing but python3. The
report calls scripts/compare_with_manuscript.py (and, when requested,
scripts/archive_run.py) as subprocesses with the same interpreter; those need
NumPy, which the job environment provides.

  identity  --root DIR                          runtime fingerprint and the registered release fingerprint
  configs   --root DIR --profile P DATASET...   validate the configurations a suite would use
  reference --root DIR DATASET...               compute the paper reports per dataset (Tables S5, S41)
  create    ...                                 freeze a new suite (called by `reproduce`)
  status    SUITE [--squeue FILE]               progress of every planned run
  active    SUITE --squeue FILE                 job ids of the suite that are still pending or running
  report    SUITE [--sacct FILE] [--pack-archives]
                                                REPRODUCTION_REPORT.md, SUITE_STATUS.json and the results ZIP
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

SCHEMA = "cross-morse-reproduction-suite-1"
DATASETS = ("ogbn-arxiv", "ogbn-proteins", "ogbn-products", "Roman-empire", "PascalVOC-SP", "COCO-SP", "Peptides-func")
STEMS = {"ogbn-arxiv": "arxiv", "ogbn-proteins": "proteins", "ogbn-products": "products", "Roman-empire": "roman_empire",
         "PascalVOC-SP": "pascalvoc_sp", "COCO-SP": "coco_sp", "Peptides-func": "peptides_func"}
PROFILES = ("smoke", "pilot", "full")
SUITE_PREFIX = "CROSS_MORSE_REPRODUCTION_"
# Settings that only route one job or carry test fixtures are not suite settings.
TRANSIENT_KEYS = {"CM_PROJECT_DIR", "CM_ENV_FILE", "CM_REPRODUCTION_SUITE", "CM_RUN_KEY", "CM_DATASET",
                  "CM_SPLIT_INDEX", "CM_CONFIG", "CM_STAGE"}
REDACTED_KEYS = {"CM_ACCOUNT", "CM_STORAGE_PROJECT"}
SCHEDULER_ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED", "REQUEUE_FED", "REQUEUE_HOLD",
                    "RESIZING", "SUSPENDED", "STAGE_OUT", "SIGNALING", "RESV_DEL_HOLD", "SPECIAL_EXIT"}
SMALL_RECORDS = ("config.json", "run_state.json", "environment.json", "policy_lock.json", "action_registry.json",
                 "interruption.json", "failure_summary.json", "select/decision.json", "confirm/decision.json",
                 "admit/decision.json", "final/results_summary.json", "final/results.csv",
                 "provenance/frozen_models.json")
SKIPPED_SUFFIXES = (".pt", ".pth", ".npz", ".npy", ".ckpt", ".tar", ".gz")


class SuiteError(ValueError):
    """A suite that cannot be created or read as declared."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False, sort_keys=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------- identity
def runtime_fingerprint(root: Path) -> str:
    """The identity every run seals: sha256 over run_cross_morse*.py and cross_morse/*.py.

    Identical to cross_morse.pipeline.source_fingerprint and scripts/source_fingerprint.py,
    computed without importing NumPy or PyTorch.
    """
    root = Path(root)
    files = sorted(root.glob("run_cross_morse*.py")) + sorted((root / "cross_morse").glob("*.py"))
    if not files:
        raise SuiteError(f"No CROSS-MORSE runtime under {root}")
    table = {str(path.relative_to(root)): sha256_file(path) for path in files}
    return hashlib.sha256(json.dumps(table, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def registered_fingerprint(root: Path) -> str | None:
    scope = read_json(Path(root) / "release" / "release_scope.json", {}) or {}
    value = scope.get("validated_runtime_source_sha256")
    return value if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) else None


def identity(root: Path) -> dict:
    actual = runtime_fingerprint(root)
    registered = registered_fingerprint(root)
    return {"runtime_sha256": actual, "registered_sha256": registered,
            "matches_release": registered is not None and actual == registered}


# --------------------------------------------------------------------------- plans
def _runtime_module(source: Path):
    """Import the launcher helpers of run_cross_morse.py (standard library only)."""
    source = str(Path(source).resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    import run_cross_morse  # noqa: PLC0415
    if Path(run_cross_morse.__file__).resolve().parent != Path(source):
        raise SuiteError(f"run_cross_morse was imported from {run_cross_morse.__file__}, not {source}")
    return run_cross_morse


def validate_datasets(names) -> list[str]:
    names = list(names) or list(DATASETS)
    unknown = [name for name in names if name not in DATASETS]
    if unknown:
        raise SuiteError("Unknown datasets: " + ", ".join(unknown) + "; canonical names: " + ", ".join(DATASETS))
    if len(set(names)) != len(names):
        raise SuiteError("A dataset is listed twice")
    return [name for name in DATASETS if name in names]


def configuration_path(source: Path, dataset: str, profile: str) -> Path:
    return Path(source) / "configs" / f"{STEMS[dataset]}_{profile}.json"


def validate_configs(source: Path, profile: str, datasets) -> dict[str, dict]:
    if profile not in PROFILES:
        raise SuiteError(f"Profile must be one of {', '.join(PROFILES)}")
    module = _runtime_module(source)
    resolved = {}
    for dataset in validate_datasets(datasets):
        config = module.resolve_configuration(config_path=configuration_path(source, dataset, profile), dataset=dataset,
                                              profile=profile, split_index=0 if dataset == "Roman-empire" else None)
        resolved[dataset] = config
    return resolved


def compute_reference(root: Path, datasets) -> dict[str, dict]:
    """The compute the paper reports for the full profile (Supplementary Tables S5 and S41)."""
    reported = read_json(Path(root) / "paper" / "reported_results.json", {}) or {}
    table = reported.get("supplementary_table_S41_qualification_cost_components", {})
    out = {}
    for dataset in validate_datasets(datasets):
        row = table.get(dataset) or {}
        increment = row.get("total_qualification_increment_GPU_h")
        if increment is None:
            increment = row.get("qualification_increment_derived_from_table_S15_GPU_h")
        out[dataset] = {"training_GPU_h_per_seed": row.get("backbone_training_GPU_h_per_seed"),
                        "qualification_increment_GPU_h": increment}
    return out


TOOL_EXCLUDES = shutil.ignore_patterns(".git", "__pycache__", "*.pyc", ".venv", "venv*", "examples", "tests",
                                       "*.env.example.bak", "outputs", "cache", "logs")


def freeze_tools(source: Path, destination: Path) -> dict[str, str]:
    """Copy the release tree (without examples and tests) that the setup and report jobs use.

    Every run keeps its own hashed snapshot; this copy additionally carries the
    table builders (paper/build) that run snapshots leave out. Returns its hash inventory.
    """
    shutil.copytree(source, destination, ignore=TOOL_EXCLUDES, symlinks=False)
    private = destination / "configs" / "arrhenius.env.example"
    private.unlink(missing_ok=True)
    return {path.relative_to(destination).as_posix(): sha256_file(path)
            for path in sorted(destination.rglob("*")) if path.is_file()}


@contextlib.contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def suite_settings(environ=None) -> dict[str, str]:
    environ = os.environ if environ is None else environ
    return {key: environ[key] for key in sorted(environ)
            if key.startswith("CM_") and key not in TRANSIENT_KEYS and not key.startswith("CM_MOCK")
            and not re.search(r"TOKEN|SECRET|PASSWORD", key)}


def create_suite(*, outputs: Path, state_dir: Path, source: Path, profile: str, data_root: Path,
                 runtime_shell: Path, datasets, allow_source_change: bool = False, environ=None) -> Path:
    """Freeze every run of a new suite before any job starts.

    Each single-dataset run gets its own hashed source/configuration snapshot
    (run_cross_morse.create_launcher_snapshot); Roman-empire gets one frozen
    campaign for its ten split runs (create_roman_campaign). All snapshots come
    from the same source tree within one call.
    """
    source, outputs, state_dir = Path(source).resolve(), Path(outputs).resolve(), Path(state_dir).resolve()
    datasets = validate_datasets(datasets)
    ident = identity(source)
    if not ident["matches_release"] and not allow_source_change:
        raise SuiteError("The runtime in " + str(source) + " has fingerprint " + ident["runtime_sha256"]
                         + ", not the released " + str(ident["registered_sha256"]) + ". A reproduction uses the released "
                         "runtime unchanged; set CM_ALLOW_SOURCE_CHANGE=1 to run a modified runtime as a labelled variant.")
    module = _runtime_module(source)
    configs = {dataset: configuration_path(source, dataset, profile) for dataset in datasets}
    validate_configs(source, profile, datasets)
    if outputs.is_relative_to(source) or source.is_relative_to(outputs):
        raise SuiteError("The outputs directory must be outside the source tree and must not contain it")
    outputs.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(state_dir / "reproduction_counter.lock"):
        counter = state_dir / "reproduction_counter"
        try:
            number = max(1, int(counter.read_text().strip()))
        except (OSError, ValueError):
            number = 1
        while (outputs / f"{SUITE_PREFIX}{number:04d}").exists():
            number += 1
        suite = outputs / f"{SUITE_PREFIX}{number:04d}"
        suite.mkdir()
        counter.write_text(f"{number + 1}\n")
    tag = f"CMR_{number:04d}"
    for name in ("000_MANIFEST", "runs", "999_REPORT"):
        (suite / name).mkdir()
    runs = []
    for dataset in datasets:
        key = f"{DATASETS.index(dataset) + 1:02d}_{STEMS[dataset]}"
        path = suite / "runs" / key
        if dataset == "Roman-empire":
            module.create_roman_campaign(source, path, configs[dataset], profile, data_root, runtime_shell)
            kind, splits = "campaign", list(range(10))
        else:
            module.create_launcher_snapshot(source, path, configs[dataset], profile, data_root, runtime_shell, dataset=dataset)
            kind, splits = "single", None
        runs.append({"key": key, "dataset": dataset, "stem": STEMS[dataset], "kind": kind, "path": str(path),
                     "splits": splits, "config": str(configs[dataset].relative_to(source)),
                     "launcher": str(path / "source" / "run_cross_morse.sh")})
    tools = freeze_tools(source, suite / "source")
    write_json(suite / "000_MANIFEST" / "source_sha256.json", tools)
    settings = suite_settings(environ)
    manifest = {"schema": SCHEMA, "tag": tag, "number": number, "suite": str(suite), "profile": profile,
                "created_utc": utc_now(), "datasets": datasets, "data_root": str(Path(data_root).absolute()),
                "source": {"path": str(source), **ident, "launcher_sha256": sha256_file(runtime_shell),
                           "modified_runtime_allowed": bool(allow_source_change and not ident["matches_release"])},
                "report_source": str(suite / "source"),
                "paper_comparison": profile == "full",
                "settings": settings, "runs": runs}
    write_json(suite / "000_MANIFEST" / "suite.json", manifest)
    write_text(suite / "000_MANIFEST" / "plan.tsv", "key\tdataset\tkind\tpath\tlauncher\n" + "".join(
        f"{run['key']}\t{run['dataset']}\t{run['kind']}\t{run['path']}\t{run['launcher']}\n" for run in runs))
    write_text(suite / "000_MANIFEST" / "suite.env",
               "# Effective settings when this suite was created; `reproduce resume` restores them.\n"
               + "".join(f"export {key}={shlex.quote(value)}\n" for key, value in settings.items()))
    write_text(suite / "000_MANIFEST" / "job_ids.tsv", "role\tkey\tjob_id\tsubmitted_utc\n")
    latest = outputs / f"{SUITE_PREFIX}LATEST"
    with contextlib.suppress(OSError):
        if latest.is_symlink() or not latest.exists():
            latest.unlink(missing_ok=True)
            latest.symlink_to(suite.name)
    write_text(state_dir / "latest_reproduction.txt", str(suite) + "\n")
    return suite


def load_suite(suite: Path) -> dict:
    suite = Path(suite).resolve()
    manifest = read_json(suite / "000_MANIFEST" / "suite.json")
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise SuiteError(f"Not a CROSS-MORSE reproduction suite: {suite}")
    original = Path(manifest["suite"])
    if not original.is_absolute() or manifest.get("profile") not in PROFILES:
        raise SuiteError("Invalid reproduction suite location or profile")
    if Path(manifest["report_source"]) != original / "source":
        raise SuiteError("The report source must be the suite's frozen source directory")
    datasets = validate_datasets(manifest["datasets"])
    if not manifest["datasets"] or [r.get("dataset") for r in manifest["runs"]] != datasets:
        raise SuiteError("Suite run entries must match the declared datasets exactly")
    runs = []
    for run in manifest["runs"]:
        dataset = run["dataset"]
        key = f"{DATASETS.index(dataset) + 1:02d}_{STEMS[dataset]}"
        path = original / "runs" / key
        if (run.get("key") != key or Path(run["path"]) != path
                or Path(run["launcher"]) != path / "source/run_cross_morse.sh"
                or run.get("kind") != ("campaign" if dataset == "Roman-empire" else "single")):
            raise SuiteError(f"Invalid or external run path in suite: {key}")
        # A copied suite must read its own runs and tools, never the old paths.
        path = suite / "runs" / key
        runs.append(dict(run, path=str(path), launcher=str(path / "source/run_cross_morse.sh")))
    manifest = dict(manifest, suite=str(suite), report_source=str(suite / "source"), runs=runs)
    return manifest


def verify_frozen_tree(source: Path, inventory_path: Path) -> int:
    """Verify a frozen tree before using its executable analysis tools."""
    source, inventory_path = Path(source), Path(inventory_path)
    inventory = read_json(inventory_path)
    if source.is_symlink() or not source.is_dir() or not isinstance(inventory, dict) or not inventory:
        raise SuiteError(f"Missing or invalid frozen source inventory: {inventory_path}")
    bad, current = [], set()
    for name, expected in inventory.items():
        relative = Path(name)
        if (relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name
                or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected)):
            raise SuiteError(f"Unsafe frozen source inventory entry: {name!r}")
        path = source / relative
        if (path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(source.resolve())
                or sha256_file(path) != expected):
            bad.append(name)
    for path in source.rglob("*"):
        if path.is_symlink():
            bad.append(path.relative_to(source).as_posix())
        elif path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}:
            current.add(path.relative_to(source).as_posix())
    bad.extend(sorted(current - set(inventory)))
    if bad:
        raise SuiteError("Frozen source changed; refusing mixed-code analysis: " + ", ".join(sorted(set(bad))))
    return len(inventory)


def verify_suite_sources(suite: Path) -> dict:
    """Check report tools and every available run snapshot against their inventories.

    These byte checks establish source consistency, not experimental validity.
    Numerical checks are performed by the report's existing evidence builders.
    """
    manifest = load_suite(suite)
    suite = Path(manifest["suite"])
    source = Path(manifest["report_source"])
    checked = verify_frozen_tree(source, suite / "000_MANIFEST/source_sha256.json")
    expected = manifest["source"]["runtime_sha256"]
    observed = identity(source)
    if observed["runtime_sha256"] != expected:
        raise SuiteError("The report source runtime differs from the frozen suite identity")
    snapshots = 0
    for run in manifest["runs"]:
        root = Path(run["path"])
        roots = [root]
        if run["kind"] == "campaign":
            roots += [root / f"split_{i:02d}" for i in range(10) if (root / f"split_{i:02d}").exists()]
        for path in roots:
            checked += verify_frozen_tree(path / "source", path / "launcher/source_sha256.json")
            if runtime_fingerprint(path / "source") != expected:
                raise SuiteError(f"Run snapshot runtime differs from its suite: {path}")
            snapshots += 1
    return {"status": "PASS", "report_source": str(source), "run_snapshots": snapshots,
            "verified_files": checked, **observed}


def verify_completed_run_source(root: Path, expected: str) -> None:
    """Bind completion to sealed stage identities, not only a progress label."""
    state = read_json(Path(root) / "run_state.json", {})
    if not isinstance(state, dict) or state.get("source_sha256") != expected:
        raise SuiteError(f"Completed run has no matching sealed runtime fingerprint: {Path(root).name}")
    source = str(Path(__file__).resolve().parents[1])
    if source not in sys.path:
        sys.path.insert(0, source)
    from cross_morse.review import sealed_inventory
    _, identities = sealed_inventory(root)
    if identities["source_sha256"] != expected:
        raise SuiteError(f"Completed run sealed stage source differs from its suite: {Path(root).name}")


# --------------------------------------------------------------------------- progress
def run_state(path: Path) -> dict:
    """The state of one run directory from its own records (no scheduler query)."""
    path = Path(path)
    state = read_json(path / "run_state.json")
    terminal = read_json(path / "launcher" / "terminal.json")
    restarts = sum(1 for _ in (path / "launcher").glob("status_*.json")) if (path / "launcher").is_dir() else 0
    result = {"state": "NOT_STARTED", "last_complete_stage": None, "detail": "", "attempts": restarts}
    if isinstance(state, dict):
        result["last_complete_stage"] = state.get("last_complete_stage")
        result["scientific_run"] = state.get("scientific_run")
        result["source_sha256"] = state.get("source_sha256")
        if state.get("status") == "COMPLETE":
            result["state"] = "COMPLETE"
            return result
        result["state"] = "IN_PROGRESS"
    if isinstance(terminal, dict) and terminal.get("status") in {"FAILED", "INCOMPLETE"}:
        result["state"] = terminal["status"]
        result["detail"] = f"{terminal.get('failure_class', '')}: {terminal.get('reason', '')}".strip(": ")
    elif (path / "failure_summary.json").is_file():
        failure = read_json(path / "failure_summary.json", {}) or {}
        result["state"] = "FAILED"
        result["detail"] = f"{failure.get('type', 'error')}: {failure.get('reason', '')}"[:300]
    elif (path / "interruption.json").is_file():
        result["state"] = "CHECKPOINTED"
        result["detail"] = "stopped at a safe boundary; continues from its last sealed stage"
    elif result["state"] == "NOT_STARTED" and (path / "launcher").is_dir() and restarts:
        result["state"] = "STARTED"
    return result


def campaign_state(path: Path) -> dict:
    splits = [run_state(Path(path) / f"split_{index:02d}") for index in range(10)]
    complete = sum(split["state"] == "COMPLETE" for split in splits)
    failed = [index for index, split in enumerate(splits) if split["state"] in {"FAILED", "INCOMPLETE"}]
    started = [split for split in splits if split["state"] != "NOT_STARTED"]
    if complete == 10:
        state = "COMPLETE"
    elif failed:
        state = "FAILED"
    elif started:
        state = "IN_PROGRESS"
    else:
        state = "NOT_STARTED"
    detail = f"{complete}/10 splits complete"
    if failed:
        detail += "; failed splits " + ", ".join(f"{index:02d}" for index in failed)
    return {"state": state, "detail": detail, "splits": splits, "complete_splits": complete,
            "last_complete_stage": None, "attempts": sum(split["attempts"] for split in splits)}


def job_table(suite: Path) -> list[dict]:
    rows = []
    path = Path(suite) / "000_MANIFEST" / "job_ids.tsv"
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 3 and parts[2].strip():
            rows.append({"role": parts[0], "key": parts[1], "job_id": parts[2].strip(),
                         "submitted_utc": parts[3] if len(parts) > 3 else ""})
    return rows


def parse_squeue(text: str) -> dict[str, str]:
    """`squeue -h -o '%i %T'` lines → {base job id: most active state}."""
    states: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        base = parts[0].split("_", 1)[0].split(".", 1)[0]
        states[base] = parts[1].upper() if base not in states or parts[1].upper() == "RUNNING" else states[base]
    return states


def active_jobs(suite: Path, squeue_text: str, *, exclude_roles=("report",), exclude_ids=()) -> list[str]:
    states = parse_squeue(squeue_text)
    active = []
    for row in job_table(suite):
        base = row["job_id"].split("_", 1)[0].split(";", 1)[0]
        if row["role"] in exclude_roles or base in exclude_ids or base in active:
            continue
        if states.get(base, "") in SCHEDULER_ACTIVE:
            active.append(base)
    return active


def suite_status(suite: Path, squeue_text: str = "") -> dict:
    manifest = load_suite(suite)
    states = parse_squeue(squeue_text) if squeue_text else {}
    jobs = job_table(Path(manifest["suite"]))
    runs = []
    for run in manifest["runs"]:
        record = campaign_state(Path(run["path"])) if run["kind"] == "campaign" else run_state(Path(run["path"]))
        own = [row for row in jobs if row["key"] == run["key"] or
               (run["kind"] == "campaign" and row["key"].startswith(run["key"] + "/split_"))]
        scheduler = ""
        active = []
        for row in own:
            base = row["job_id"].split("_", 1)[0].split(";", 1)[0]
            state = states.get(base)
            if state:
                scheduler = state
            if state in SCHEDULER_ACTIVE:
                active.append(base)
        record.update({"key": run["key"], "dataset": run["dataset"], "kind": run["kind"],
                       "jobs": [row["job_id"] for row in own], "scheduler_state": scheduler,
                       "scheduler_active": bool(active), "active_jobs": sorted(set(active))})
        if record["state"] != "COMPLETE" and active:
            # A previous failure or one failed Roman split must not hide other
            # allocations that are still writing to this run/campaign.
            record["recorded_state"] = record["state"]
            record["state"] = "QUEUED" if scheduler == "PENDING" else "RUNNING" if scheduler == "RUNNING" else "ACTIVE"
        runs.append(record)
    complete = sum(run["state"] == "COMPLETE" for run in runs)
    return {"suite": manifest["suite"], "tag": manifest["tag"], "profile": manifest["profile"],
            "created_utc": manifest["created_utc"], "runs": runs, "complete": complete, "planned": len(runs)}


def render_status(status: dict) -> str:
    lines = [f"{status['tag']}  profile={status['profile']}  {status['complete']}/{status['planned']} runs complete",
             f"suite: {status['suite']}", "",
             f"{'run':20s} {'dataset':14s} {'state':13s} {'last stage':10s} {'jobs':18s} detail"]
    for run in status["runs"]:
        jobs = ",".join(run["jobs"]) or "-"
        lines.append(f"{run['key']:20s} {run['dataset']:14s} {run['state']:13s} "
                     f"{(run.get('last_complete_stage') or '-'):10s} {jobs:18s} {run.get('detail', '')}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- report
def gpu_hours_from_sacct(text: str, tag: str) -> dict[str, float]:
    """GPU-hours per job role from `sacct --parsable2` with JobName, ElapsedRaw and AllocTRES columns."""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return {}
    header = lines[0].split("|")
    try:
        name_i, elapsed_i, tres_i = header.index("JobName"), header.index("ElapsedRaw"), header.index("AllocTRES")
    except ValueError:
        return {}
    prefix = tag.lower().replace("_", "") + "_"
    hours: dict[str, float] = {}
    for line in lines[1:]:
        cells = line.split("|")
        if len(cells) <= max(name_i, elapsed_i, tres_i) or not cells[name_i].startswith(prefix):
            continue
        match = re.search(r"gres/gpu(?::[^=,]+)?=(\d+)", cells[tres_i])
        try:
            seconds = float(cells[elapsed_i] or 0)
        except ValueError:
            continue
        gpus = int(match.group(1)) if match else 0
        role = cells[name_i][len(prefix):]
        hours[role] = hours.get(role, 0.0) + gpus * seconds / 3600.0
    return hours


def _paper_row(reported: dict, dataset: str) -> dict:
    return (reported.get("main_table_1_policy_and_deployment_ledger") or {}).get(dataset) or {}


def _fmt(value, digits=5, signed=True) -> str:
    if value is None:
        return "—"
    try:
        return format(float(value), f"{'+' if signed else ''}.{digits}f")
    except (TypeError, ValueError):
        return str(value)


def _interval(lower, upper, digits=5) -> str:
    if lower is None or upper is None:
        return ""
    return f" [{_fmt(lower, digits)}, {_fmt(upper, digits)}]"


def run_summary(run: dict) -> dict:
    """Regenerated outcome of one planned run from its sealed final records."""
    path = Path(run["path"])
    if run["kind"] == "campaign":
        return {"dataset": run["dataset"], "kind": "campaign"}
    summary = read_json(path / "final" / "results_summary.json", {}) or {}
    environment = read_json(path / "environment.json", {}) or {}
    loss = summary.get("paired_qD_minus_qC_loss") or {}
    metric = summary.get("paired_qD_minus_qC_metric_pp") or {}
    deployed, comparator = summary.get("deployed") or {}, summary.get("comparator") or {}
    return {"dataset": run["dataset"], "kind": "single",
            "deployed": f"{deployed.get('route', '?')}, E={deployed.get('ensemble', '?')}" if deployed else None,
            "comparator": f"{comparator.get('route', '?')}, E={comparator.get('ensemble', '?')}" if comparator else None,
            "comparator_returned": bool(deployed) and deployed.get("hash") == comparator.get("hash"),
            "admitted": summary.get("admitted"), "loss": loss, "metric": metric,
            "scientific_run": summary.get("scientific_run"),
            "environment": {key: environment.get(key) for key in ("torch", "cuda_runtime", "gpu", "python", "machine", "precision")}}


def _report_process(command, report_dir: Path, *, timeout=3600):
    """Keep builder failures in the diagnostic report, including timeouts."""
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout, cwd=str(report_dir))
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(command, 124, "", f"Report command timed out after {timeout}s: {Path(command[2]).name}")
    except OSError as exc:
        return subprocess.CompletedProcess(command, 127, "", str(exc))


def compare_with_paper(run: dict, report_dir: Path, python: str) -> dict:
    """Run scripts/compare_with_manuscript.py from the run's own frozen source."""
    path = Path(run["path"])
    script = path / "source" / "scripts" / "compare_with_manuscript.py"
    output = report_dir / "comparisons" / f"{run['key']}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    for stale in (output, output.with_suffix(".md")):
        stale.unlink(missing_ok=True)
    command = [python, "-B", str(script), str(path), "--output", str(output)]
    if run["kind"] == "campaign":
        command.append("--roman-campaign")
    result = _report_process(command, report_dir)
    comparison = read_json(output) if output.is_file() else None
    if not isinstance(comparison, dict):
        comparison = None
    outcome = {"exit_code": result.returncode, "json": str(output) if comparison else None,
               "markdown": str(output.with_suffix(".md")) if comparison else None}
    if comparison is None:
        message = (result.stderr or result.stdout).strip()
        try:
            message = json.loads(message.splitlines()[-1]).get("error", message)
        except (ValueError, IndexError, AttributeError):
            pass
        outcome["error"] = "comparison refused: " + str(message)[-1200:]
        return outcome
    if result.returncode not in (0, 1):
        outcome["error"] = f"comparison incomplete or invalid (exit {result.returncode})"
    elif comparison.get("dataset") != run["dataset"]:
        outcome["error"] = "comparison dataset differs from the planned run"
    outcome["levels"] = comparison.get("levels", {})
    rows = [row for row in comparison.get("rows", []) if row.get("section") == "Table 1"]
    outcome["table1"] = [{"quantity": row.get("quantity"), "run_value": row.get("run_value"),
                          "reported_value": row.get("reported_value"), "status": row.get("status")} for row in rows]
    if run["kind"] == "campaign":
        aggregate = comparison.get("campaign_summary") or {}
        outcome["campaign"] = {"loss": aggregate.get("paired_qD_minus_qC_loss"),
                               "metric": aggregate.get("paired_qD_minus_qC_accuracy_pp"),
                               "admitted_split_count": aggregate.get("admitted_split_count")}
    return outcome


def summarize_campaign(run: dict, report_dir: Path, python: str) -> dict:
    """Aggregate the ten sealed split runs of a Roman-empire campaign (scripts/summarize_roman_splits.py)."""
    path = Path(run["path"])
    output = report_dir / "comparisons" / f"{run['key']}_splits.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    command = [python, "-B", str(path / "source" / "scripts" / "summarize_roman_splits.py"), str(path), "--output", str(output)]
    result = _report_process(command, report_dir)
    summary = read_json(output) if output.is_file() else None
    if not isinstance(summary, dict) or result.returncode != 0:
        return {"error": (result.stderr or result.stdout).strip()[-600:], "exit_code": result.returncode}
    return {"json": str(output), "loss": summary.get("paired_qD_minus_qC_loss"),
            "metric": summary.get("paired_qD_minus_qC_accuracy_pp"), "admitted_split_count": summary.get("admitted_split_count")}


def build_tables(suite: Path, report_dir: Path, python: str, source: Path, tag: str) -> dict:
    """Rebuild the paper's run-derived tables from the suite's completed runs (scripts/build_paper.py)."""
    output = report_dir / "paper"
    shutil.rmtree(output, ignore_errors=True)
    outcome = {"output": str(output)}
    for what in ("tables", "figures"):
        command = [python, "-B", str(source / "scripts" / "build_paper.py"), what, "--runs-root", str(suite / "runs"),
                   "--label", f"reproduction {tag}", "--output-dir", str(output)]
        result = _report_process(command, report_dir)
        produced = sorted(path.stem for path in (output / what).glob("*.md" if what == "tables" else "*.png")) \
            if (output / what).is_dir() else []
        outcome[what] = produced
        if result.returncode != 0 or not produced:
            outcome[f"{what}_error"] = ((result.stderr or result.stdout).strip()[-600:]
                                       or f"{what} builder produced no outputs (exit {result.returncode})")
    # The data-role membership, seeds and environment each run actually used.
    manifests = report_dir / "manifests"
    shutil.rmtree(manifests, ignore_errors=True)
    command = [python, "-B", str(source / "scripts" / "run_manifests.py"), "runs", "--runs-root", str(suite / "runs"),
               "--label", "reproduction", "--output-root", str(manifests), "--with-indices"]
    result = _report_process(command, report_dir)
    outcome["manifests"] = sorted(path.name for path in (manifests / "reproduction").glob("*.json")) \
        if (manifests / "reproduction").is_dir() else []
    if result.returncode != 0 or not outcome["manifests"]:
        outcome["manifests_error"] = ((result.stderr or result.stdout).strip()[-600:]
                                      or f"Manifest builder produced no outputs (exit {result.returncode})")
    return outcome


def _loss_status(outcome: dict) -> str:
    for row in outcome.get("table1", []):
        quantity = str(row.get("quantity", ""))
        if "effect" in quantity and quantity.endswith("mean"):
            return str(row.get("status"))
    return "—"


def pack_results(suite: Path, manifest: dict, destination: Path) -> int:
    """A compact ZIP of the report, manifests, logs and the small decision records (no checkpoints)."""
    suite = Path(suite)
    count = 0
    temporary = destination.with_name(destination.name + ".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        def add(path: Path, limit=20 * 1024 * 1024):
            nonlocal count
            if path.is_file() and not path.is_symlink() and not path.name.endswith(SKIPPED_SUFFIXES) and path.stat().st_size <= limit:
                archive.write(path, path.relative_to(suite).as_posix())
                count += 1
        for name in ("REPRODUCTION_REPORT.md", "SUITE_STATUS.json"):
            add(suite / name)
        for path in sorted((suite / "000_MANIFEST").rglob("*")):
            if path.name == "suite.json":
                redacted = dict(manifest, settings={key: ("<redacted>" if key in REDACTED_KEYS and value else value)
                                                    for key, value in manifest.get("settings", {}).items()})
                archive.writestr("000_MANIFEST/suite.json", json.dumps(redacted, indent=2) + "\n")
                count += 1
            elif path.name != "suite.env":
                add(path)
        for path in sorted((suite / "999_REPORT").rglob("*")):
            if "archives" not in path.relative_to(suite / "999_REPORT").parts:
                add(path)
        for run in manifest["runs"]:
            roots = [Path(run["path"]) / f"split_{index:02d}" for index in range(10)] if run["kind"] == "campaign" else [Path(run["path"])]
            if run["kind"] == "campaign":
                add(Path(run["path"]) / "campaign.json")
            for root in roots:
                for relative in SMALL_RECORDS:
                    add(root / relative)
                for pattern in ("stages/*.json", "provenance/stages/*.json", "launcher/status_*.json", "launcher/terminal.json",
                                "launcher/gpu_health_*.json", "launcher/job_*.json", "launcher/source_sha256.json"):
                    for path in sorted(root.glob(pattern)):
                        add(path)
    os.replace(temporary, destination)
    return count


def build_report(suite: Path, *, sacct_text: str = "", pack_archives: bool = False, python: str | None = None) -> dict:
    manifest = load_suite(suite)
    suite = Path(manifest["suite"])
    source_verification = verify_suite_sources(suite)
    python = python or sys.executable
    report_dir = suite / "999_REPORT"
    report_dir.mkdir(exist_ok=True)
    status = suite_status(suite)
    reported = read_json(Path(manifest["report_source"]) / "paper" / "reported_results.json", {}) or {}
    gpu_hours = gpu_hours_from_sacct(sacct_text, manifest["tag"]) if sacct_text else {}
    rows, environments, fingerprints, errors = [], {}, set(), []
    completed_sources_ok = True
    for run, state in zip(manifest["runs"], status["runs"]):
        entry = {"key": run["key"], "dataset": run["dataset"], "kind": run["kind"], "state": state["state"],
                 "detail": state.get("detail", ""), "summary": run_summary(run) if state["state"] == "COMPLETE" else None,
                 "comparison": None, "gpu_hours": round(gpu_hours.get(run["key"], 0.0), 3) if gpu_hours else None}
        roots = [Path(run["path"]) / f"split_{i:02d}" for i in range(10)] if run["kind"] == "campaign" else [Path(run["path"])]
        for root in roots:
            env = read_json(root / "environment.json")
            if isinstance(env, dict):
                environments[(env.get("torch"), env.get("cuda_runtime"), env.get("gpu"), env.get("python"))] = True
            state_record = read_json(root / "run_state.json")
            if isinstance(state_record, dict) and state_record.get("source_sha256"):
                fingerprints.add(state_record["source_sha256"])
            if run_state(root)["state"] == "COMPLETE":
                try:
                    verify_completed_run_source(root, manifest["source"]["runtime_sha256"])
                except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
                    completed_sources_ok = False
                    errors.append({"component": f"source/{run['key']}", "error": str(exc)})
        if state["state"] in {"FAILED", "INCOMPLETE"}:
            errors.append({"component": run["key"], "error": f"Run {state['state']}: {state.get('detail', '')}"})
        if state["state"] == "COMPLETE" and manifest.get("paper_comparison"):
            entry["comparison"] = compare_with_paper(run, report_dir, python)
            if entry["comparison"].get("error"):
                errors.append({"component": f"comparison/{run['key']}", "error": entry["comparison"]["error"]})
        elif state["state"] == "COMPLETE" and run["kind"] == "campaign":
            entry["campaign"] = summarize_campaign(run, report_dir, python)
            if entry["campaign"].get("error"):
                errors.append({"component": f"campaign/{run['key']}", "error": entry["campaign"]["error"]})
        rows.append(entry)
        if pack_archives and state["state"] == "COMPLETE":
            archive = report_dir / "archives" / f"{run['key']}.tar.gz"
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive.unlink(missing_ok=True)
            command = [python, "-B", str(Path(run["path"]) / "source" / "scripts" / "archive_run.py"), run["path"],
                       "--output", str(archive)] + (["--campaign"] if run["kind"] == "campaign" else [])
            result = _report_process(command, report_dir, timeout=6 * 3600)
            entry["archive"] = str(archive) if result.returncode == 0 and archive.is_file() else None
            if entry["archive"] is None:
                entry["archive_error"] = (result.stderr or result.stdout).strip()[-600:]
                errors.append({"component": f"archive/{run['key']}", "error": entry["archive_error"] or "Archive was not produced"})
    tables = (build_tables(suite, report_dir, python, Path(manifest["report_source"]), manifest["tag"])
              if status["complete"] else {"tables": [], "figures": []})
    errors += [{"component": key.removesuffix("_error"), "error": value}
               for key, value in tables.items() if key.endswith("_error")]
    released = source_verification["registered_sha256"]
    sealed_ok = (status["complete"] == status["planned"] and bool(fingerprints)
                 and fingerprints == {manifest["source"]["runtime_sha256"]}
                 and completed_sources_ok)
    report_status = "FAIL" if errors else "PASS" if status["complete"] == status["planned"] else "INCOMPLETE"
    result = {"schema": "cross-morse-reproduction-report-1", "tag": manifest["tag"], "suite": str(suite),
              "report_status": report_status, "report_errors": errors, "source_verification": source_verification,
              "report_scope": "Requested dataset lifecycles and their supported report outputs; auxiliary experiments and all numbered manuscript displays are separate checks. Numerical differences remain visible and are not tool failures.",
              "profile": manifest["profile"], "created_utc": manifest["created_utc"], "reported_utc": utc_now(),
              "runtime_sha256": manifest["source"]["runtime_sha256"], "released_runtime_sha256": released,
              "runtime_matches_release": source_verification["matches_release"],
              "sealed_runtime_fingerprints": sorted(fingerprints), "sealed_fingerprints_match_suite": sealed_ok,
              "complete_runs": status["complete"], "planned_runs": status["planned"],
              "gpu_hours_total": round(sum(gpu_hours.values()), 3) if gpu_hours else None,
              "environments": [dict(zip(("torch", "cuda", "gpu", "python"), key)) for key in environments],
              "runs": rows, "paper_tables": tables,
              "interpretation": ("Agreement levels are those of scripts/compare_with_manuscript.py: exact at the printed "
                                 "precision, inside the printed 95% interval, or outside. A regenerated run is a new "
                                 "execution of the registered protocol with the same seeds; hardware, kernels and library "
                                 "versions can differ from the authors' runs.")}
    write_json(suite / "SUITE_STATUS.json", result)
    text = render_report(result, manifest, reported)
    write_text(suite / "REPRODUCTION_REPORT.md", text)
    zip_path = suite / f"{manifest['tag']}_RESULTS.zip"
    result["results_zip"] = str(zip_path)
    result["results_zip_files"] = pack_results(suite, manifest, zip_path)
    write_json(suite / "SUITE_STATUS.json", result)
    return result


def render_report(result: dict, manifest: dict, reported: dict) -> str:
    profile = result["profile"]
    released = result.get("released_runtime_sha256")
    identity_text = (f"`{result['runtime_sha256']}`, identical to the released runtime" if result.get("runtime_matches_release")
                     else f"`{result['runtime_sha256']}`, a modified runtime (released: `{released}`)")
    sealed = ("every run sealed this fingerprint" if result["sealed_fingerprints_match_suite"]
              else "sealed run fingerprints: " + (", ".join(result["sealed_runtime_fingerprints"]) or "none yet"))
    envs = "; ".join(" · ".join(f"{k} {v}" for k, v in env.items() if v) for env in result["environments"]) or "—"
    lines = [f"# CROSS-MORSE reproduction {result['tag']}", "",
             "| | |", "|---|---|",
             f"| Report build | {result['report_status']} |",
             f"| Profile | {profile} ({'official data, registered full budget' if profile == 'full' else 'official data, reduced budget' if profile == 'pilot' else 'synthetic data; checks the complete orchestration'}) |",
             f"| Created / reported (UTC) | {result['created_utc']} / {result['reported_utc']} |",
             f"| Runtime fingerprint | {identity_text}; {sealed} |",
             f"| Runs complete | {result['complete_runs']} of {result['planned_runs']} |",
             f"| Execution environment | {envs} |"]
    if result.get("gpu_hours_total") is not None:
        lines.append(f"| GPU-hours (Slurm accounting) | {result['gpu_hours_total']:.1f} |")
    lines += ["", f"Suite directory: `{result['suite']}`", "", result["report_scope"], ""]
    if result["report_errors"]:
        lines += ["## Unsuccessful report steps", "", "| Step | Detail |", "|---|---|"]
        for error in result["report_errors"]:
            lines.append("| " + " | ".join(str(error[key]).replace("|", "/").replace("\n", " ")
                                           for key in ("component", "error")) + " |")
        lines.append("")
    if profile == "full":
        lines += ["## Regenerated results beside the paper (main Table 1)", "",
                  "Paired loss effect of the deployed policy against the open-loop comparator q_C on the test split "
                  "(negative favours the deployed policy), with the paired 95% interval.", "",
                  "| Dataset | Deployed policy: run / paper | Paired loss effect, nats: run | Paper | Table 1 effect | All compared quantities |",
                  "|---|---|---|---|---|---|"]
        for row in result["runs"]:
            paper = _paper_row(reported, row["dataset"])
            paper_policy = f"{paper.get('deployed_rule', '?')}, E={paper.get('deployed_E', '?')}" if paper else "—"
            if paper.get("evaluation_status") == "evaluated_arm":
                paper_policy += " (evaluated arm)"
            interval = paper.get("loss_effect_interval_nats") or [None, None]
            paper_effect = _fmt(paper.get("loss_effect_nats")) + _interval(*interval) if paper else "—"
            comparison = row.get("comparison") or {}
            if row["state"] != "COMPLETE":
                lines.append(f"| {row['dataset']} | not complete ({row['state']}) / {paper_policy} | — | {paper_effect} | — | — |")
                continue
            if row["kind"] == "campaign":
                loss = (comparison.get("campaign") or {}).get("loss") or {}
                policy = "ten-split campaign"
            else:
                summary = row.get("summary") or {}
                loss = summary.get("loss") or {}
                policy = (summary.get("deployed") or "?") + (" (q_C returned)" if summary.get("comparator_returned") else "")
            run_effect = _fmt(loss.get("mean")) + _interval(loss.get("lower"), loss.get("upper"))
            levels = comparison.get("levels") or {}
            level_text = ", ".join(f"{name} {levels[name]}" for name in ("exact", "interval", "outside", "match", "differs", "missing")
                                   if levels.get(name)) or (comparison.get("error", "comparison unavailable")[:120] if comparison else "—")
            lines.append(f"| {row['dataset']} | {policy} / {paper_policy} | {run_effect} | {paper_effect} | "
                         f"{_loss_status(comparison) if comparison else '—'} | {level_text} |")
        lines += ["", "Per-dataset comparisons, with every quantity the comparison tool reads (Tables 1, S12c, S13, S14, "
                  "S15 and S20), are in `999_REPORT/comparisons/`. " + result["interpretation"], ""]
    else:
        lines += ["## Regenerated outcomes", "",
                  "The paper reports the full profile; this profile checks the complete execution path. "
                  "Values are the run's own paired test outcomes.", "",
                  "| Dataset | Deployed / comparator | Admitted | Paired loss effect, nats [95%] |", "|---|---|---|---|"]
        for row in result["runs"]:
            summary = row.get("summary") or {}
            if row["state"] != "COMPLETE":
                lines.append(f"| {row['dataset']} | not complete ({row['state']}) | — | — |")
            elif row["kind"] == "campaign":
                campaign = row.get("campaign") or {}
                loss = campaign.get("loss") or {}
                admitted = campaign.get("admitted_split_count")
                effect = (f"{_fmt(loss.get('mean'))}{_interval(loss.get('lower'), loss.get('upper'))} (ten splits)"
                          if loss else "summary unavailable: " + str(campaign.get("error", ""))[:120])
                lines.append(f"| {row['dataset']} | ten-split campaign | "
                             f"{'—' if admitted is None else f'{admitted} of 10 splits'} | {effect} |")
            else:
                loss = summary.get("loss") or {}
                lines.append(f"| {row['dataset']} | {summary.get('deployed')} / {summary.get('comparator')} | "
                             f"{summary.get('admitted')} | {_fmt(loss.get('mean'))}{_interval(loss.get('lower'), loss.get('upper'))} |")
        lines.append("")
    tables = result.get("paper_tables") or {}
    if tables.get("tables"):
        lines += ["## Paper tables rebuilt from this suite", "",
                  "`scripts/build_paper.py` rebuilt the run-derived columns of " + ", ".join(tables["tables"])
                  + " from the completed runs of this suite (LaTeX, Markdown and CSV in `999_REPORT/paper/tables/`"
                  + (", figures in `999_REPORT/paper/figures/`" if tables.get("figures") else "") + ")."
                  + (" `999_REPORT/manifests/reproduction/` records the data-role membership, seeds and environment "
                     "each run used." if tables.get("manifests") else ""), ""]
        for name in ("T1", "T2"):
            table = Path(tables["output"]) / "tables" / f"{name}.md"
            if table.is_file():
                body = table.read_text(encoding="utf-8").strip().splitlines()
                lines += [("### " + body[0].lstrip("# ")) if body and body[0].startswith("#") else f"### {name}", ""]
                lines += body[1:] + [""]
    elif tables.get("tables_error"):
        lines += ["## Paper tables", "", "The table builder reported: " + tables["tables_error"].replace("\n", " ")[:400], ""]
    lines += ["## Every planned run", "", "| Run | Dataset | State | Detail | GPU-hours |", "|---|---|---|---|---|"]
    for row in result["runs"]:
        hours = f"{row['gpu_hours']:.1f}" if row.get("gpu_hours") is not None else "—"
        lines.append(f"| `{row['key']}` | {row['dataset']} | {row['state']} | {str(row.get('detail', '')).replace('|', '/')} | {hours} |")
    lines += ["", "## Files", "",
              "- `SUITE_STATUS.json`: this report in machine-readable form.",
              f"- `{result['tag']}_RESULTS.zip`: this report, the manifests, the comparisons, the scheduler accounting "
              "and every run's decision records (selection, confirmation, admission, policy lock, final summary); "
              "checkpoints and prediction arrays stay in the run directories.",
              "- `000_MANIFEST/`: settings, plan, job ids and the timestamped run registry.",
              "- `999_REPORT/paper/`: the paper's run-derived tables (and figures) rebuilt from these runs.",
              "- `999_REPORT/manifests/`: the role membership, seeds and environment of every run.",
              "- `999_REPORT/logs/`: the ordered log of every job of the suite.",
              "- `runs/<run>/`: the sealed run directories (each with its frozen `source/` snapshot).", "",
              "Rebuild this report with `bash run_cross_morse.sh report " + result["suite"] + "`.", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- command line
def _datasets_argument(values) -> list[str]:
    names = []
    for value in values or []:
        names += [part for part in re.split(r"[,\s]+", value) if part]
    return names


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("identity")
    p.add_argument("--root", type=Path, required=True)
    p = sub.add_parser("configs")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--profile", required=True, choices=PROFILES)
    p.add_argument("datasets", nargs="*")
    p = sub.add_parser("reference")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("datasets", nargs="*")
    p = sub.add_parser("create")
    p.add_argument("--outputs", type=Path, required=True)
    p.add_argument("--state-dir", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--profile", required=True, choices=PROFILES)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--runtime-shell", type=Path, required=True)
    p.add_argument("--allow-source-change", action="store_true")
    p.add_argument("datasets", nargs="*")
    p = sub.add_parser("status")
    p.add_argument("suite", type=Path)
    p.add_argument("--squeue", type=Path)
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("verify")
    p.add_argument("suite", type=Path)
    p = sub.add_parser("active")
    p.add_argument("suite", type=Path)
    p.add_argument("--squeue", type=Path, required=True)
    p.add_argument("--exclude", action="append", default=[])
    p = sub.add_parser("report")
    p.add_argument("suite", type=Path)
    p.add_argument("--sacct", type=Path)
    p.add_argument("--pack-archives", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "identity":
            print(json.dumps(identity(args.root)))
        elif args.command == "configs":
            for dataset, config in validate_configs(args.root, args.profile, _datasets_argument(args.datasets)).items():
                print(f"{dataset}\t{STEMS[dataset]}_{args.profile}.json\tseeds={len(config['training']['seeds'])}"
                      f"\tepochs={config['training']['epochs']}\tprecision={config['model']['precision']}")
        elif args.command == "reference":
            print(json.dumps(compute_reference(args.root, _datasets_argument(args.datasets))))
        elif args.command == "create":
            suite = create_suite(outputs=args.outputs, state_dir=args.state_dir, source=args.source, profile=args.profile,
                                 data_root=args.data_root, runtime_shell=args.runtime_shell,
                                 datasets=_datasets_argument(args.datasets), allow_source_change=args.allow_source_change)
            print(suite)
        elif args.command == "status":
            text = args.squeue.read_text(encoding="utf-8") if args.squeue and args.squeue.is_file() else ""
            status = suite_status(args.suite, text)
            print(json.dumps(status, indent=1) if args.json else render_status(status), end="" if not args.json else "\n")
        elif args.command == "verify":
            verification = verify_suite_sources(args.suite)
            print(json.dumps(dict(load_suite(args.suite), source_verification=verification)))
        elif args.command == "active":
            text = args.squeue.read_text(encoding="utf-8") if args.squeue.is_file() else ""
            print(" ".join(active_jobs(args.suite, text, exclude_ids=tuple(args.exclude))))
        elif args.command == "report":
            text = args.sacct.read_text(encoding="utf-8", errors="replace") if args.sacct and args.sacct.is_file() else ""
            result = build_report(args.suite, sacct_text=text, pack_archives=args.pack_archives)
            print(f"[REPORT] {Path(result['suite']) / 'REPRODUCTION_REPORT.md'}")
            print(f"[RESULTS] {result['results_zip']} ({result['results_zip_files']} files)")
            print(f"[STATUS] {result['complete_runs']}/{result['planned_runs']} runs complete")
            print(f"[REPORT STATUS] {result['report_status']}")
            return {"PASS": 0, "INCOMPLETE": 1, "FAIL": 2}[result["report_status"]]
        return 0
    except (SuiteError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
