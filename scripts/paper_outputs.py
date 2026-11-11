#!/usr/bin/env python3
"""Produce every measured output the paper's data-availability statement promises, as one Slurm workflow.

The statement is fulfilled once the 37 artifacts of ``release/paper_artifacts.json`` -- the seven sealed
lifecycle runs, the measurement-record sets, the E3-E6 and S14 campaigns and the GPU environment -- are
published as assets of the release ``paper-artifacts``. The runs come from ``bash run_cross_morse.sh
reproduce``; everything else is produced from them by the repository's runners. This tool turns the
roughly thirty runner commands of ``release/experiment_code.json`` into one job graph:

    plan     discover the runs (a reproduction suite, its runs/ directory, or one --run DATASET=DIR per
             dataset), verify that each is a sealed full scientific run, execute every runner's own
             torch-free ``plan`` step whose inputs exist and write <work>/plan.json: one node per producing
             step with its exact command lines, its resources (GPU or CPU; array size = the jobs of the
             runner's plan), its dependencies and its outputs, plus the export of every artifact
    submit   write one Slurm batch script per node with the SITE SETTINGS of run_cross_morse.sh
             (``bash run_cross_morse.sh site-env``) and submit them with --dependency=afterok chains,
             then a final job that runs ``collect``; job ids go to <work>/jobs.json (--dry-run prints the
             sbatch commands and submits nothing; --resume resubmits only unfinished nodes)
    status   the state of every node (sacct/squeue when available, the nodes' task markers otherwise)
    collect  run every export into <work>/records (each set verified with scripts/records.py), assemble
             the E3-E6 campaign, and write <work>/paper-artifacts.map.json -- the complete map that
             ``bash hpc/publish_paper_artifacts_arrhenius.sh all <map>`` publishes
    execute  (inside a Slurm job) one task of one node: deferred runner plans, resumable commands,
             checkpoint continuations (exit 75 -> requeue), task markers under <work>/state/

Example (Arrhenius; hpc/produce_paper_outputs_arrhenius.sh wraps exactly this):

    python -B scripts/paper_outputs.py plan --runs $CM_BASE/outputs/CROSS_MORSE_REPRODUCTION_0001 \\
        --work $CM_BASE/paper-outputs/CROSS_MORSE_REPRODUCTION_0001
    python3 -B scripts/paper_outputs.py submit --work $CM_BASE/paper-outputs/CROSS_MORSE_REPRODUCTION_0001
    python3 -B scripts/paper_outputs.py status --work $CM_BASE/paper-outputs/CROSS_MORSE_REPRODUCTION_0001
    bash hpc/publish_paper_artifacts_arrhenius.sh all $CM_BASE/paper-outputs/CROSS_MORSE_REPRODUCTION_0001/paper-artifacts.map.json

Nothing here measures or alters a value: the runners measure, the exporters write the record sets, and an
artifact that cannot be produced stays ``awaiting`` in the map with the reason. ``submit`` and ``status``
use the standard library only (login node); ``plan``, ``collect`` and ``execute`` run the runners, which
need NumPy (and PyTorch for execution).
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "cm-paper-outputs-plan-1"
JOBS_SCHEMA = "cm-paper-outputs-jobs-1"
TASK_SCHEMA = "cm-paper-outputs-task-1"
COLLECT_SCHEMA = "cm-paper-outputs-collect-1"
MAP_NAME = "paper-artifacts.map.json"
DATASETS = ("ogbn-arxiv", "ogbn-proteins", "ogbn-products", "Roman-empire", "PascalVOC-SP", "COCO-SP",
            "Peptides-func")
STEMS = {"ogbn-arxiv": "arxiv", "ogbn-proteins": "proteins", "ogbn-products": "products",
         "Roman-empire": "roman_empire", "PascalVOC-SP": "pascalvoc_sp", "COCO-SP": "coco_sp",
         "Peptides-func": "peptides_func"}
ADMITTED = tuple(d for d in DATASETS if d != "ogbn-products")      # cm_campaigns.contract.ADMITTED
MULTIGRAPH = ("PascalVOC-SP", "COCO-SP", "Peptides-func")
STAGES = ("prepare", "train", "explore", "fit", "select", "confirm", "admit", "evaluate")
PRE_ADMISSION = STAGES[:STAGES.index("confirm") + 1]
GPU, CPU = "gpu", "cpu"
COMPLETE_STATES = ("COMPLETE", "SKIPPED")
CHECKPOINTED = 75                    # a runner stopped at a safe boundary; rerunning the command resumes
S14_SCRIPT = "scripts/build_s14_campaign.py"

# Datasets of every experiment: the configurations of release/experiment_code.json (and, for the group-B
# experiments that list none, cm_campaigns.contract.RECORD_SETS). tests/test_paper_outputs.py keeps them equal.
EXPERIMENT_DATASETS = {
    "inserted_site": MULTIGRAPH, "ensemble_grid": ADMITTED, "cross_backbone": ("ogbn-arxiv", "Roman-empire",
                                                                             "PascalVOC-SP"),
    "factorials": ("ogbn-arxiv", "PascalVOC-SP"), "sensitivity_grids": DATASETS,
    "horizon_beam_sweep": ("PascalVOC-SP",), "module_ablations": DATASETS,
    "buffer_sensitivity": ("ogbn-arxiv", "ogbn-proteins"), "input_gate": ADMITTED, "runtime_scaling": DATASETS,
    "timing_anchors": ADMITTED, "ensemble_grid_scores": ADMITTED, "mechanism_contrasts": DATASETS,
    "oracle_dispersion": DATASETS, "tube_audit": ("ogbn-arxiv", "Roman-empire", "PascalVOC-SP", "COCO-SP",
                                                  "Peptides-func"),
    "depth_diagnostics": DATASETS, "confirmation": MULTIGRAPH, "decision_audit": DATASETS,
    "nominal_time": ADMITTED, "sensor_parameters": ADMITTED, "budget": ADMITTED, "s14_campaign": DATASETS,
    "baselines": DATASETS,
}
# The producing experiment of every artifact the displays need (scripts/paper_artifacts.py plan).
ARTIFACT_EXPERIMENT = {
    "records:action_gap_audit": "decision_audit", "records:decision_traces": "decision_audit",
    "records:reliability_diagnostics": "decision_audit", "records:band_bound_tightness": "band_audit",
    "records:baselines": "baselines", "records:budget_matched": "budget", "campaign:budget": "budget",
    "records:buffer_sensitivity": "buffer_sensitivity", "records:confirmation_contrasts": "confirmation",
    "records:confirmation_sensitivity": "confirmation", "records:cross_backbone": "cross_backbone",
    "records:depth_diagnostics": "depth_diagnostics", "records:ensemble_grid": "ensemble_grid",
    "records:ensemble_grid_scores": "ensemble_grid_scores", "records:factorials": "factorials",
    "records:horizon_beam_sweep": "horizon_beam_sweep", "records:input_gate": "input_gate",
    "records:inserted_site": "inserted_site", "records:mechanism_contrasts": "mechanism_contrasts",
    "records:module_ablations": "module_ablations", "records:negative_controls": "negative_controls",
    "records:nominal_time": "nominal_time", "records:sensor_parameters": "sensor_parameters",
    "records:oracle_dispersion": "oracle_dispersion", "records:runtime_scaling": "runtime_scaling",
    "records:sensitivity_grids": "sensitivity_grids", "records:timing_anchors": "timing_anchors",
    "records:tube_audit": "tube_audit", "campaign:confirmation": "s14_campaign", "environment:gpu": "environment",
}
# Experiments whose nodes need the nodes (not the exports) of other experiments.
NEEDS = {"negative_controls": ("input_gate",), "module_ablations": ("input_gate", "negative_controls"),
         "decision_audit": ("input_gate",), "timing_anchors": ("runtime_scaling",),
         "ensemble_grid_scores": ("ensemble_grid",), "confirmation": ("s14_campaign",)}
# Software-check flags each runner's plan accepts for synthetic smoke parents (plan --allow-fixture only).
FIXTURE_FLAGS = {"inserted-site": ["--allow-fixture"], "factorial-fit": ["--allow-fixture"],
                 "modules": ["--allow-fixture"], "confirmation": ["--allow-fixture"], "group-b": ["--allow-fixture"],
                 "internal-baselines": ["--allow-fixture"]}
# Declared choices the manuscript does not fix; recorded in plan.json and overridable on the command line.
DEFAULT_NEGATIVE_CONTROLS_DATASET = "PascalVOC-SP"   # the example of scripts/run_decision_audit.py
DEFAULT_BAND_AUDIT_DATASET = "ogbn-arxiv"            # the shared-transport plant, against A0 ("shared")
DEFAULT_BAND_AUDIT_BASELINE = "shared"               # release/experiment_code.json, E18
DEFAULT_ARM_SEEDS = tuple(range(12))                 # scripts/arm_campaign.py template seed_ids


class PlanError(ValueError):
    """The inputs do not support the requested plan, submission or collection."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=1, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def slug(text) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")
    if not out:
        raise PlanError(f"cannot form a file name from {text!r}")
    return out


def plain(value):
    """JSON-ready copy: paths become strings."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def py(script: str, *args) -> list[str]:
    """A Python entry point of this checkout; ``{python}`` is the job's interpreter."""
    return ["{python}", "-B", script, *[str(a) for a in args]]


def repeat(flag: str, values) -> list[str]:
    return [x for value in values for x in (flag, str(value))]


# ============================================================================= runs
def runtime_fingerprint(root: Path = ROOT) -> str:
    from scripts.reproduction import runtime_fingerprint as fingerprint
    return fingerprint(root)


def producer_sources(root: Path = ROOT) -> dict:
    """The actual auxiliary code and contracts that a planned job will execute."""
    paths = set(root.glob("run_cross_morse*.py")) | set(root.glob("run_cross_morse*.sh"))
    for folder in ("scripts", "cm_experiments", "cm_campaigns", "cm_theory", "cross_morse", "hpc",
                   "paper/build", "configs", "environment/arrhenius"):
        paths.update(p for p in (root / folder).rglob("*") if p.is_file() and p.suffix in {".py", ".sh", ".json"})
    paths.update(root / name for name in ("records/specs.json", "paper/registered_constants.json", "paper/sources.json"))
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(paths) if p.is_file()}


def verify_plan_source(plan: dict) -> None:
    repository = plan.get("repository") or {}
    if repository.get("runtime_sha256") not in (None, runtime_fingerprint()):
        raise PlanError("the runtime of this checkout changed after planning; restore it or plan again")
    if repository.get("producer_sources") is not None and repository["producer_sources"] != producer_sources():
        raise PlanError("the auxiliary producer code/contracts changed after planning; restore the planned checkout or use a new work directory")


@contextlib.contextmanager
def work_lock(work: Path, name: str):
    """Prevent overlapping submissions, collection or execution of the same task."""
    import fcntl
    directory = Path(work) / "locks"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / (slug(name) + ".lock")).open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise PlanError(f"another {name} operation is using {work}") from None
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _run_dataset(path: Path) -> str | None:
    for name in ("run_state.json", "config.json"):
        try:
            value = read_json(path / name)
        except (OSError, ValueError):
            continue
        dataset = (value.get("task") or {}).get("dataset") if name == "run_state.json" else (
            value.get("benchmark_dataset") or value.get("dataset"))
        if dataset in DATASETS:
            return dataset
    return None


def _entry(path: Path) -> tuple[str, str] | None:
    """(dataset, kind) of a run directory or a Roman-empire campaign directory."""
    path = Path(path)
    if (path / "run_state.json").is_file() and (path / "config.json").is_file():
        dataset = _run_dataset(path)
        return (dataset, "single") if dataset else None
    splits = sorted(p for p in path.glob("split_*") if (p / "run_state.json").is_file())
    if splits:
        datasets = {_run_dataset(p) for p in splits}
        if len(datasets) == 1 and None not in datasets:
            return datasets.pop(), "campaign"
    return None


def discover_runs(runs=None, run_specs=()) -> tuple[dict, dict]:
    """{dataset: {"path", "kind"}} from a suite, its runs/ directory or a directory of runs, and --run overrides."""
    found, suite = {}, {}
    if runs is not None:
        base = Path(runs).expanduser().resolve()
        if not base.is_dir():
            raise PlanError(f"--runs {runs}: not a directory")
        manifest_dir = next((p for p in (base, base.parent) if (p / "000_MANIFEST" / "suite.json").is_file()), None)
        if manifest_dir is not None:
            from scripts.reproduction import SuiteError, load_suite
            try:
                manifest = load_suite(manifest_dir)
            except (SuiteError, OSError, ValueError, KeyError) as exc:
                raise PlanError(f"{manifest_dir}: {exc}") from None
            suite = {"path": str(manifest_dir), "tag": manifest.get("tag"), "profile": manifest.get("profile"),
                     "data_root": manifest.get("data_root"), "runtime_sha256": manifest["source"].get("runtime_sha256")}
            for run in manifest["runs"]:
                found[run["dataset"]] = {"path": str(Path(run["path"]).resolve()), "kind": run["kind"],
                                         "key": run["key"]}
        else:
            for child in sorted(p for p in base.iterdir() if p.is_dir()):
                entry = _entry(child)
                if entry is None:
                    continue
                dataset, kind = entry
                if dataset in found:
                    raise PlanError(f"{base}: two runs of {dataset} ({found[dataset]['path']}, {child})")
                found[dataset] = {"path": str(child.resolve()), "kind": kind, "key": child.name}
    for spec in run_specs or ():
        dataset, sep, directory = str(spec).partition("=")
        if not sep or dataset not in DATASETS or not directory:
            raise PlanError(f"--run takes DATASET=DIR with a canonical dataset name {list(DATASETS)}: {spec!r}")
        path = Path(directory).expanduser().resolve()
        entry = _entry(path)
        if entry is None:
            raise PlanError(f"--run {spec}: neither a lifecycle run nor a Roman-empire campaign of split runs")
        if entry[0] != dataset:
            raise PlanError(f"--run {spec}: the directory holds a {entry[0]} run")
        found[dataset] = {"path": str(path), "kind": entry[1], "key": path.name}
    if not found:
        raise PlanError("no lifecycle run found; give --runs SUITE or --run DATASET=DIR")
    return found, suite


def member_paths(entry: dict) -> list[Path]:
    path = Path(entry["path"])
    if entry["kind"] == "campaign":
        return sorted(p for p in path.glob("split_*") if (p / "run_state.json").is_file())
    return [path]


def verify_runs(found: dict, *, runtime: str, allow_fixture: bool = False) -> dict:
    """Every run (every Roman-empire split) is sealed through evaluate, scientific and of the current runtime."""
    from scripts.archive_run import ArchiveError
    from scripts.paper_artifacts import _verify_run
    verified, problems = {}, []
    for dataset in DATASETS:
        if dataset not in found:
            continue
        entry = found[dataset]
        members = member_paths(entry)
        if dataset == "Roman-empire":
            names = [m.name for m in members]
            if entry["kind"] != "campaign" or names != [f"split_{i:02d}" for i in range(10)]:
                problems.append(f"{dataset}: the campaign needs the ten split runs split_00..split_09, found {names}")
                continue
        elif entry["kind"] != "single":
            problems.append(f"{dataset}: only Roman-empire is a campaign of split runs")
            continue
        identities, failed = [], False
        for index, member in enumerate(members):
            where = f"{dataset} ({member})"
            try:
                state, config = read_json(member / "run_state.json"), read_json(member / "config.json")
                chain = _verify_run(member, dataset, scientific_only=False)
            except (OSError, ValueError, KeyError, ArchiveError) as exc:
                problems.append(f"{where}: not a sealed eight-stage run: {exc}")
                failed = True
                continue
            if state.get("status") != "COMPLETE" or state.get("last_complete_stage") != "evaluate":
                problems.append(f"{where}: run_state.json is not COMPLETE through evaluate")
                failed = True
            if _run_dataset(member) != dataset:
                problems.append(f"{where}: the run is a {_run_dataset(member)} run")
                failed = True
            scientific = config.get("scientific_run") is True and config.get("profile") == "full"
            if not scientific and not allow_fixture:
                problems.append(f"{where}: not a full scientific run (profile {config.get('profile')!r}, "
                                f"scientific_run {state.get('scientific_run')!r}); --allow-fixture is a software check")
                failed = True
            if chain["source_sha256"] != runtime and not allow_fixture:
                problems.append(f"{where}: sealed by runtime source {chain['source_sha256'][:12]}, this checkout is "
                                f"{runtime[:12]}; run the experiments from the runtime that sealed the runs")
                failed = True
            split_id = (config.get("data") or {}).get("split_index")
            if dataset == "Roman-empire" and (type(split_id) is not int or split_id != index):
                problems.append(f"{where}: split index {(config.get('data') or {}).get('split_index')} != {index}")
                failed = True
            identities.append({"run": str(member), "config_sha256": chain["config_sha256"],
                               "source_sha256": chain["source_sha256"], "unit_sha256": chain["unit_sha256"],
                               "profile": config.get("profile"), "scientific_run": bool(state.get("scientific_run")),
                               "split_index": (config.get("data") or {}).get("split_index"),
                               "training_seeds": [int(s) for s in (config.get("training") or {}).get("seeds", [])]})
        if not failed:
            verified[dataset] = dict(entry, members=identities,
                                     synthetic=not all(i["scientific_run"] and i["profile"] == "full"
                                                       for i in identities))
    if problems:
        raise PlanError("the runs do not support the paper's outputs:\n  " + "\n  ".join(problems))
    return verified


# ============================================================================= planning
def _subprocess_executor(argv: list[str]) -> tuple[int, str]:
    """Run one runner's own plan step (torch-free) in this checkout; returns (exit code, output tail)."""
    command = [sys.executable if a == "{python}" else a for a in argv]
    result = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True, check=False)
    return result.returncode, (result.stdout + result.stderr)[-4000:]


class Planner:
    """Builds the node graph; every node is one sbatch submission (an array when it has several tasks)."""

    def __init__(self, work: Path, runs: dict, options: dict, executor=_subprocess_executor, repository=None):
        self.work, self.runs, self.opts, self.executor = Path(work), runs, options, executor
        self.repository = repository or {}
        self.records = self.work / "records"
        self.nodes: dict[str, dict] = {}
        self.exports: list[dict] = []
        self.artifacts: dict[str, dict] = {}

    # -- inputs
    def missing(self, datasets) -> list[str]:
        return [d for d in datasets if d not in self.runs]

    def present(self, datasets) -> list[str]:
        """The datasets with a run (all of them unless --allow-subset let a missing one through)."""
        return [d for d in datasets if d in self.runs]

    def parent(self, dataset) -> str:
        return self.runs[dataset]["path"]

    def parents(self, datasets) -> list[str]:
        return repeat("--parent", [self.parent(d) for d in self.present(datasets)])

    def members(self, dataset) -> list[Path]:
        return [Path(m["run"]) for m in self.runs[dataset]["members"]]

    def label(self, dataset, member: Path) -> str:
        return STEMS[dataset] + (f"_{member.name}" if self.runs[dataset]["kind"] == "campaign" else "")

    def experiment_dir(self, *parts) -> Path:
        return self.work.joinpath("experiments", *parts)

    def fixture(self, key) -> list[str]:
        return list(FIXTURE_FLAGS.get(key, [])) if self.opts.get("allow_fixture") else []

    # -- runner plans
    def runner_plan(self, argv: list[str], output: Path) -> dict:
        """Execute a runner's plan now; return its plan.json (raises PlanError with the runner's message)."""
        if (output / "plan.json").is_file():
            return read_json(output / "plan.json")
        existed = output.exists()
        code, text = self.executor(argv)
        if code != 0 or not (output / "plan.json").is_file():
            if not existed:         # a refused plan leaves nothing behind (some runners write files first)
                shutil.rmtree(output, ignore_errors=True)
            raise PlanError(f"{' '.join(a for a in argv[2:6])}... exited with {code}: "
                            + " ".join(text.strip().splitlines()[-3:]))
        return read_json(output / "plan.json")

    def probe(self, name: str, argv_for, *, required=True) -> dict | None:
        """A runner's plan written into a scratch directory only to size a node planned later."""
        directory = self.work / "probes" / name
        shutil.rmtree(directory, ignore_errors=True)
        try:
            return self.runner_plan(argv_for(directory), directory)
        except PlanError:
            if required:
                raise
            return None
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    # -- graph
    def add_node(self, node_id, *, title, experiment, resource, tasks, deps=(), artifacts=(), outputs=(),
                 plan_step=None, requeue=True, dataset=None, commands=None) -> dict:
        if node_id in self.nodes:
            raise PlanError(f"duplicate node {node_id}")
        deps = [d if isinstance(d, dict) else {"node": d, "type": "afterok"} for d in deps]
        for dep in deps:
            if dep["node"] not in self.nodes:
                raise PlanError(f"{node_id}: dependency {dep['node']} is not planned before it")
        reason = next((self.nodes[d["node"]]["blocked"] for d in deps if self.nodes[d["node"]].get("blocked")), None)
        node = {"id": node_id, "title": title, "experiment": experiment, "resource": resource,
                "array": len(tasks) if len(tasks) > 1 else None, "tasks": tasks, "dependencies": deps,
                "artifacts": list(artifacts), "outputs": [str(o) for o in outputs], "requeue": bool(requeue),
                "dataset": dataset, "plan_step": plan_step,
                "commands": commands or [c for t in tasks[:1] for c in t["commands"][:1]]}
        if reason:
            node["blocked"] = f"needs {', '.join(d['node'] for d in deps if self.nodes[d['node']].get('blocked'))}: " \
                              + reason
        if not tasks and not node.get("blocked"):
            node["blocked"] = "the runner's plan holds no job"
        self.nodes[node_id] = node
        return node

    def add_export(self, export_id, *, artifacts, steps, after=(), provenance=None, outputs=None) -> dict:
        export = {"id": export_id, "artifacts": list(artifacts), "after": list(after), "steps": steps,
                  "provenance": provenance, "outputs": outputs or {}}
        self.exports.append(export)
        for artifact in artifacts:
            self.artifacts[artifact] = {"status": "planned", "path": export["outputs"].get(artifact),
                                        "nodes": list(after), "export": export_id}
        return export

    def block(self, artifacts, reason):
        for artifact in artifacts:
            self.artifacts[artifact] = {"status": "awaiting", "reason": reason}

    def record_set(self, name) -> str:
        return str(self.records / name)


def job_tasks(argv_for, count: int, *, offset: int = 0, plan_file=None, kind=None, bound=False) -> list[dict]:
    """One task per job of a runner's plan; ``argv_for(job)`` is the run command of that job."""
    tasks = []
    for k in range(count):
        task = {"commands": [argv_for(offset + k)]}
        if plan_file is not None:
            task["job_in"] = {"plan": str(plan_file), "index": offset + k, "kind": kind, "upper_bound": bool(bound)}
        tasks.append(task)
    return tasks


def group_b_task(plan_argv: list[str], directory: Path, run_argv: list[str]) -> dict:
    """Group-B runs need a fresh directory: a failed attempt is moved aside and the plan is written again."""
    return {"commands": [run_argv], "fresh": {"directory": str(directory),
                                              "complete": str(directory / "ARTIFACT_MANIFEST.json"),
                                              "keep": ["plan.json"], "plan": plan_argv,
                                              "plan_file": str(directory / "plan.json")}}


def _gpu_run(*argv) -> list[str]:
    return [*argv, "--device", "cuda", "--threads", "{threads}", "--trust-local-checkpoints"]


# ----------------------------------------------------------------------------- experiment builders
def build_input_gate(p: Planner, export: bool):
    datasets = p.present(EXPERIMENT_DATASETS["input_gate"])
    tasks, dirs, failures = [], [], []
    for dataset in datasets:
        for member in p.members(dataset):
            out = p.experiment_dir("input_gate", p.label(dataset, member))
            argv = py("scripts/run_experiment_b.py", "input-gate", "plan", "--parent", member, "--output", out,
                      *p.fixture("group-b"))
            try:
                p.runner_plan(argv, out)
            except PlanError as exc:
                failures.append(f"{dataset}/{member.name}: {exc}")
            tasks.append(group_b_task(argv, out, _gpu_run(*py("scripts/run_experiment_b.py", "input-gate", "run",
                                                                 "--plan", out / "plan.json"))))
            dirs.append(out)
    node = p.add_node("input_gate.run", title="Serving-time input gate on every admitted run (Tables S23, S37)",
                      experiment="input_gate", resource=GPU, tasks=tasks, outputs=dirs,
                      artifacts=["records:input_gate"], requeue=False)
    if failures:
        raise PlanError("; ".join(failures))
    if export:
        p.add_export("export.input_gate", artifacts=["records:input_gate"], after=[node["id"]],
                     outputs={"records:input_gate": p.record_set("input_gate")},
                     provenance={"kind": "group_b", "dirs": [str(d) for d in dirs]},
                     steps=[{"argv": py("scripts/run_experiment_b.py", "input-gate", "export",
                                        *repeat("--results", dirs), "--root", p.records,
                                        "--hardware", "{hardware}", "--period", "{period}")}])
    return {"dirs": {str(m): str(d) for m, d in zip([m for ds in datasets for m in p.members(ds)], dirs)}}


def build_negative_controls(p: Planner, export: bool, gates: dict):
    """Table S19 behind the frozen g_input of the dataset's sealed input-gate run (planned inside the job)."""
    dataset = p.opts["negative_controls_dataset"]
    if p.missing([dataset]):
        raise PlanError(f"the negative-controls dataset {dataset} has no run")
    tasks, dirs, deps = [], [], []
    for member in p.members(dataset):
        out = p.experiment_dir("negative_controls", p.label(dataset, member))
        gate = gates.get(str(member))
        argv = py("scripts/run_experiment_b.py", "negative-controls", "plan", "--parent", member, "--output", out,
                  *(["--gate-from", gate] if gate else []), *p.fixture("group-b"))
        if gate is None:        # a dataset without input gate (ogbn-products): plan now; the gate is refitted
            p.runner_plan(argv, out)
        tasks.append(group_b_task(argv, out, _gpu_run(*py("scripts/run_experiment_b.py", "negative-controls", "run",
                                                             "--plan", out / "plan.json"))))
        dirs.append(out)
    if any(gates.get(str(m)) for m in p.members(dataset)):
        deps.append("input_gate.run")
    node = p.add_node("negative_controls.run", title=f"Negative controls on {dataset} (Table S19), behind the "
                      "sealed input gate", experiment="negative_controls", resource=GPU, tasks=tasks, deps=deps,
                      outputs=dirs, artifacts=["records:negative_controls"], requeue=False, dataset=dataset)
    if export:
        p.add_export("export.negative_controls", artifacts=["records:negative_controls"], after=[node["id"]],
                     outputs={"records:negative_controls": p.record_set("negative_controls")},
                     provenance={"kind": "group_b", "dirs": [str(d) for d in dirs]},
                     steps=[{"argv": py("scripts/run_experiment_b.py", "negative-controls", "export",
                                        *repeat("--results", dirs), "--root", p.records,
                                        "--hardware", "{hardware}", "--period", "{period}")}])
    return {"dirs": [str(d) for d in dirs]}


def build_band_audit(p: Planner, export: bool):
    dataset, baseline = p.opts["band_audit_dataset"], p.opts["band_audit_baseline"]
    if p.missing([dataset]):
        raise PlanError(f"the band-audit dataset {dataset} has no run")
    tasks, dirs = [], []
    for member in p.members(dataset):
        out = p.experiment_dir("band_audit", p.label(dataset, member))
        argv = py("scripts/run_experiment_b.py", "band-audit", "plan", "--parent", member, "--output", out,
                  *p.fixture("group-b"))
        p.runner_plan(argv, out)
        tasks.append(group_b_task(argv, out, _gpu_run(*py("scripts/run_experiment_b.py", "band-audit", "run",
                                                             "--plan", out / "plan.json"))))
        dirs.append(out)
    node = p.add_node("band_audit.run", title=f"E18 band-power audit of the {dataset} plant (Figure S18c)",
                      experiment="band_audit", resource=GPU, tasks=tasks, outputs=dirs, requeue=False,
                      artifacts=["records:band_bound_tightness"], dataset=dataset)
    if export:
        p.add_export("export.band_bound_tightness", artifacts=["records:band_bound_tightness"], after=[node["id"]],
                     outputs={"records:band_bound_tightness": p.record_set("band_bound_tightness")},
                     provenance={"kind": "group_b", "dirs": [str(d) for d in dirs]},
                     steps=[{"argv": py("scripts/run_experiment_b.py", "band-audit", "export-tightness",
                                        *repeat("--results", dirs), "--dataset", dataset, "--baseline", baseline,
                                        "--root", p.records, "--hardware", "{hardware}", "--period", "{period}")}])


def _record_experiment(p: Planner, *, name, script, verb, datasets, title, artifacts, export, plan_extra=(),
                       run_extra=None, export_extra=(), fixture_key=None, resource=GPU, node_id=None,
                       export_steps=None):
    """plan (now) | run --job K (array) | export: the common runner pattern of cm_experiments."""
    out = p.experiment_dir(name)
    argv = py(script, *verb, "plan", *p.parents(datasets), "--output", out, *plan_extra,
              *p.fixture(fixture_key))
    plan = p.runner_plan(argv, out)
    count = len(plan.get("jobs") or [])
    run = run_extra or (lambda job: _gpu_run(*py(script, *verb, "run", "--output", out, "--job", job)))
    node = p.add_node(node_id or f"{name}.run", title=title, experiment=name, resource=resource,
                      tasks=job_tasks(run, count), outputs=[out], artifacts=artifacts)
    if export:
        steps = export_steps or [{"argv": py(script, *verb, "export", "--output", out, "--root", p.records,
                                             *export_extra)}]
        p.add_export(f"export.{name}", artifacts=artifacts, after=[node["id"]], steps=steps,
                     outputs={a: p.record_set(a.split(":", 1)[1]) for a in artifacts})
    return out, plan, node


def build_inserted_site(p, export):
    _record_experiment(p, name="inserted_site", script="scripts/run_experiment.py", verb=["inserted-site"],
                       datasets=EXPERIMENT_DATASETS["inserted_site"], fixture_key="inserted-site",
                       title="E13 inserted-site attribution (Table S45, Figure S18a)",
                       artifacts=["records:inserted_site"], export=export)


def build_ensemble_grid(p, export):
    out, _, node = _record_experiment(p, name="ensemble_grid", script="scripts/run_experiment.py",
                                      verb=["ensemble-grid"], datasets=EXPERIMENT_DATASETS["ensemble_grid"],
                                      title="E14 comparator on the extended ensemble grid (Tables S26, S26b)",
                                      artifacts=["records:ensemble_grid"], export=export)
    return {"output": out, "node": node["id"]}


def build_ensemble_grid_scores(p, export, grid):
    if export:
        p.add_export("export.ensemble_grid_scores", artifacts=["records:ensemble_grid_scores"], after=[grid["node"]],
                     outputs={"records:ensemble_grid_scores": p.record_set("ensemble_grid_scores")},
                     steps=[{"argv": py("scripts/run_timing_records.py", "grid-scores", "export", "--grid-output",
                                        grid["output"], "--root", p.records)}])


def build_cross_backbone(p, export):
    out = p.experiment_dir("cross_backbone")
    _record_experiment(p, name="cross_backbone", script="scripts/run_experiment.py", verb=["cross-backbone"],
                       datasets=EXPERIMENT_DATASETS["cross_backbone"],
                       title="Cross-backbone substitutions: whole lifecycles behind the same adapter (Table S25)",
                       artifacts=["records:cross_backbone"], export=export,
                       run_extra=lambda job: py("scripts/run_experiment.py", "cross-backbone", "run", "--output", out,
                                                "--job", job, "--data-root", "{data_root}", "--device", "cuda",
                                                "--threads", "{threads}"))


def build_factorials(p, export):
    out = p.experiment_dir("factorials")
    argv = py("scripts/run_ablation.py", "factorial-fit", "plan", *p.parents(EXPERIMENT_DATASETS["factorials"]),
              "--output", out, *p.fixture("factorial-fit"))
    plan = p.runner_plan(argv, out)
    lifecycles = len(plan.get("jobs") or [])
    cells = len(read_json(out / "factorial" / "plan.json").get("jobs") or [])

    def run(job):
        return _gpu_run(*py("scripts/run_ablation.py", "factorial-fit", "run", "--output", out, "--job", job))
    first = p.add_node("factorials.lifecycles", title="Factorial cell variants: lifecycles explore->confirm from the "
                       "sealed prepare/train outputs (Tables S16, S16b)", experiment="factorials", resource=GPU,
                       tasks=job_tasks(run, lifecycles), outputs=[out], artifacts=["records:factorials"])
    second = p.add_node("factorials.cells", title="Factorial cells executed on the test role (fitted mode)",
                        experiment="factorials", resource=GPU, tasks=job_tasks(run, cells, offset=lifecycles),
                        deps=[first["id"]], outputs=[out], artifacts=["records:factorials"])
    if export:
        p.add_export("export.factorials", artifacts=["records:factorials"], after=[second["id"]],
                     outputs={"records:factorials": p.record_set("factorials")},
                     steps=[{"argv": py("scripts/run_ablation.py", "factorial-fit", "export", "--output", out,
                                        "--root", p.records)}])


def build_sweep(p, export, sweep, datasets, title, *, resource=GPU, device="cuda"):
    out = p.experiment_dir(sweep)
    argv = py("scripts/run_sweep.py", "plan", "--sweep", sweep, *p.parents(datasets), "--output", out)
    if sweep == "horizon_beam_sweep":
        swept = p.opts.get("horizon_sweep") or {}
        argv += [*(["--profile", swept["profile"]] if swept.get("profile") else []),
                 *(["--ensemble", str(swept["ensemble"])] if swept.get("ensemble") is not None else []),
                 *(["--beam", str(swept["beam"])] if swept.get("beam") is not None else [])]
    plan = p.runner_plan(argv, out)
    variants = [v["id"] for v in plan.get("variants") or []]
    after = []
    if variants:
        tasks = [{"commands": [py("scripts/run_sweep.py", "run", "--plan", out / "plan.json", "--data-root",
                                  "{data_root}", "--device", device, "--threads", "{threads}", "--only", v)]}
                 for v in variants]
        node = p.add_node(f"{sweep}.variants", title=title, experiment=sweep, resource=resource, tasks=tasks,
                          outputs=[out], artifacts=[f"records:{sweep}"])
        after.append(node["id"])
    elif (plan.get("settings") or {}).get("jobs"):
        # buffer_sensitivity: one measurement job per parent, no lifecycle (cm_experiments/buffered_admission.py)
        tasks = [{"commands": [py("scripts/run_sweep.py", "run", "--plan", out / "plan.json", "--only", job["id"])]}
                 for job in plan["settings"]["jobs"]]
        node = p.add_node(f"{sweep}.run", title=title, experiment=sweep, resource=CPU, tasks=tasks, outputs=[out],
                          artifacts=[f"records:{sweep}"], requeue=False)
        after.append(node["id"])
    if export:
        p.add_export(f"export.{sweep}", artifacts=[f"records:{sweep}"], after=after,
                     outputs={f"records:{sweep}": p.record_set(sweep)},
                     provenance={"kind": "sweep", "plan": str(out / "plan.json")},
                     steps=[{"argv": py("scripts/run_sweep.py", "collect", "--plan", out / "plan.json")},
                            {"argv": py("scripts/export_sweep_records.py", "--results", out / "results.json",
                                        "--root", p.records, "--hardware", "{hardware}", "--period", "{period}")}])


def build_module_ablations(p, export, gates: dict, controls: list[str]):
    datasets = EXPERIMENT_DATASETS["module_ablations"]
    out = p.experiment_dir("module_ablations")

    def plan_argv(directory, with_inputs=True):
        extra = (repeat("--input-gate", gates.values()) + repeat("--negative-controls", controls)) if with_inputs else []
        return py("scripts/run_ablation.py", "modules", "plan", *p.parents(datasets), "--output", directory, *extra,
                  *p.fixture("modules"))
    probe = p.probe("module_ablations", lambda d: plan_argv(d, with_inputs=False))
    kinds = [j.get("kind") for j in probe.get("jobs") or []]
    lifecycles = kinds.count("lifecycle")
    if kinds != ["lifecycle"] * lifecycles + ["measure"] * (len(kinds) - lifecycles):
        raise PlanError("module_ablations: the plan does not list its lifecycles before its measurements")
    planner = p.add_node("module_ablations.plan", title="S17 module ablations: plan with the comparator branch of the "
                         "sealed input-gate and negative-control runs", experiment="module_ablations", resource=CPU,
                         tasks=[{"commands": []}], deps=["input_gate.run", "negative_controls.run"], outputs=[out],
                         artifacts=["records:module_ablations"], requeue=False,
                         plan_step={"argv": plan_argv(out), "plan_file": str(out / "plan.json"),
                                    "jobs": len(kinds), "kinds": kinds, "exact": True},
                         commands=[plan_argv(out)])

    def run(job):
        return _gpu_run(*py("scripts/run_ablation.py", "modules", "run", "--output", out, "--job", job))
    first = p.add_node("module_ablations.lifecycles", title="S17 variant lifecycles (deleted/scrambled coordinates, "
                       "observer variants, restricted recovery, complete replicate)", experiment="module_ablations",
                       resource=GPU, tasks=job_tasks(run, lifecycles, plan_file=out / "plan.json", kind="lifecycle"),
                       deps=[planner["id"]], outputs=[out], artifacts=["records:module_ablations"])
    second = p.add_node("module_ablations.measure", title="S17 measurement of every arm on the held-out role",
                        experiment="module_ablations", resource=GPU,
                        tasks=job_tasks(run, len(kinds) - lifecycles, offset=lifecycles, plan_file=out / "plan.json",
                                        kind="measure"),
                        deps=[first["id"]], outputs=[out], artifacts=["records:module_ablations"])
    if export:
        p.add_export("export.module_ablations", artifacts=["records:module_ablations"], after=[second["id"]],
                     outputs={"records:module_ablations": p.record_set("module_ablations")},
                     provenance={"kind": "modules", "output": str(out)},
                     steps=[{"argv": py("scripts/run_ablation.py", "modules", "export", "--output", out, "--sweep",
                                        p.experiment_dir("module_ablations_sweep"), "--root", p.records,
                                        "--hardware", "{hardware}", "--period", "{period}"),
                             "fresh_directory": str(p.experiment_dir("module_ablations_sweep"))}])


def build_runtime_scaling(p, export):
    out = p.experiment_dir("runtime_scaling")
    accounting = p.work / "accounting" / "runtime_scaling"
    if p.opts.get("device") == "cpu":
        # Timings on CPU, without scheduler allocations: a partial measurement (cm_experiments/timing_records
        # records why), written only with --allow-partial and never into records/.
        steps = [{"argv": py("scripts/run_timing_records.py", "scaling", "export", "--output", out, "--root", p.records,
                             "--allow-partial")}]
    else:
        steps = [
            # the terminal sacct records of the timing allocations, exported after every timing job completed
            {"argv": ["bash", "hpc/capture_arrhenius_accounting.sh", "{jobs:runtime_scaling.run}", accounting],
             "skip_if": str(accounting / "sacct_original.psv")},
            {"argv": py("scripts/run_timing_records.py", "scaling", "export", "--output", out, "--accounting",
                        accounting / "sacct_original.psv", "--root", p.records)}]
    _, _, node = _record_experiment(
        p, name="runtime_scaling", script="scripts/run_timing_records.py", verb=["scaling"],
        datasets=EXPERIMENT_DATASETS["runtime_scaling"], artifacts=["records:runtime_scaling"], export=export,
        title="E17 runtime and peak memory of the deployed rule against graph size (Figure S2)", export_steps=steps)
    for task in node["tasks"]:
        task["requeue"] = False        # one allocation per timing job: a requeue would give it two sacct records
    node["requeue"] = False
    return {"output": out, "node": node["id"]}


def build_timing_anchors(p, export, scaling: dict):
    anchors = p.experiment_dir("timing_anchors")
    argv = py("scripts/run_timing_records.py", "anchors", "plan", *p.parents(EXPERIMENT_DATASETS["timing_anchors"]),
              "--output", anchors, "--scaling-output", scaling["output"])
    plan = p.runner_plan(argv, anchors)
    node = p.add_node("timing_anchors.run", title="E17 throughput and training-time anchors (Table S23b; torch-free)",
                      experiment="timing_anchors", resource=CPU, deps=[scaling["node"]], outputs=[anchors],
                      artifacts=["records:timing_anchors"],
                      tasks=job_tasks(lambda job: py("scripts/run_timing_records.py", "anchors", "run", "--output",
                                                     anchors, "--job", job), len(plan.get("jobs") or [])))
    if export:
        p.add_export("export.timing_anchors", artifacts=["records:timing_anchors"], after=[node["id"]],
                     outputs={"records:timing_anchors": p.record_set("timing_anchors")},
                     steps=[{"argv": py("scripts/run_timing_records.py", "anchors", "export", "--output", anchors,
                                        "--root", p.records,
                                        # throughput timed on CPU is a partial measurement
                                        *(["--allow-partial"] if p.opts.get("device") == "cpu" else []))}])


def build_s14_campaign(p, export):
    """The joint S14 campaign from the seven runs (scripts/build_s14_campaign.py plan|run|assemble)."""
    out, campaign = p.experiment_dir("s14_campaign"), p.work / "campaigns" / "confirmation"
    if not (ROOT / S14_SCRIPT).is_file():
        raise PlanError(f"{S14_SCRIPT} is not in this checkout; give --confirmation-campaign DIR or add the builder")
    excluded = list(p.opts.get("s14_exclude") or [])
    partial = ["--allow-partial"] if p.opts.get("allow_fixture") and (p.missing(DATASETS) or excluded) else []
    included = [d for d in EXPERIMENT_DATASETS["s14_campaign"] if d not in excluded]
    plan_argv = py(S14_SCRIPT, "plan", *p.parents(included), "--output", out, *partial)
    check_cli(plan_argv)
    plan = p.runner_plan(plan_argv, out)
    jobs = plan.get("jobs")
    if isinstance(jobs, list) and jobs:
        tasks = job_tasks(lambda job: _gpu_run(*py(S14_SCRIPT, "run", "--output", out, "--job", job)), len(jobs))
    else:
        tasks = [{"commands": [_gpu_run(*py(S14_SCRIPT, "run", "--output", out))]}]
    for task in tasks:
        check_cli(task["commands"][0])
    assemble = py(S14_SCRIPT, "assemble", "--output", out, "--campaign", campaign)
    check_cli(assemble)
    run = p.add_node("s14_campaign.run", title="Joint S14 confirmation tests on the seven sealed runs", tasks=tasks,
                     experiment="s14_campaign", resource=GPU, outputs=[out], artifacts=["campaign:confirmation"])
    # assemble writes the campaign once and exits 1 when check_campaign refuses it (campaign_check.json);
    # a refused campaign is moved aside before the next attempt.
    node = p.add_node("s14_campaign.assemble", title="Assemble the S14 campaign directory and check it "
                      "(cm_campaigns.confirmation.check_campaign)", experiment="s14_campaign", resource=CPU,
                      deps=[run["id"]], outputs=[campaign], artifacts=["campaign:confirmation"], requeue=False,
                      tasks=[{"commands": [assemble],
                              "fresh": {"directory": str(campaign), "complete": str(out / "campaign_check.json"),
                                        "complete_field": "accepted", "keep": []}}])
    if export:
        p.add_export("export.campaign_confirmation", artifacts=["campaign:confirmation"], after=[node["id"]],
                     outputs={"campaign:confirmation": str(campaign)},
                     steps=[{"internal": "accepted", "file": str(out / "campaign_check.json"),
                             "campaign": str(campaign)}])
    return {"campaign": str(campaign), "node": node["id"]}


def build_confirmation(p, export, campaign: dict):
    datasets = EXPERIMENT_DATASETS["confirmation"]
    out = p.experiment_dir("confirmation")
    plan_argv = py("scripts/run_confirmation_analysis.py", "confirmation", "plan", *p.parents(datasets),
                   "--campaign", campaign["campaign"], "--output", out, *p.fixture("confirmation"))

    def run(job):
        return _gpu_run(*py("scripts/run_confirmation_analysis.py", "confirmation", "run", "--output", out,
                            "--job", job))
    deps = []
    if campaign.get("node"):
        # The jobs are (dataset, S14 seed) pairs; the seeds are known once the campaign exists. The array covers
        # every trained checkpoint seed; tasks beyond the plan's jobs exit without work.
        bound = sum(len(m["training_seeds"]) for d in p.present(datasets) for m in p.runs[d]["members"])
        planner = p.add_node("confirmation.plan", title="E10 plan on the assembled S14 campaign", resource=CPU,
                             experiment="confirmation", tasks=[{"commands": []}], deps=[campaign["node"]],
                             outputs=[out], artifacts=["records:confirmation_contrasts",
                                                       "records:confirmation_sensitivity"], requeue=False,
                             plan_step={"argv": plan_argv, "plan_file": str(out / "plan.json"), "jobs": bound,
                                        "exact": False}, commands=[plan_argv])
        deps.append(planner["id"])
        tasks = job_tasks(run, bound, plan_file=out / "plan.json", bound=True)
    else:
        tasks = job_tasks(run, len(p.runner_plan(plan_argv, out).get("jobs") or []))
    node = p.add_node("confirmation.run", title="E10 S14 policy pairs on the validation split, seed by seed (S14b, S18b)",
                      experiment="confirmation", resource=GPU, tasks=tasks, deps=deps, outputs=[out],
                      artifacts=["records:confirmation_contrasts", "records:confirmation_sensitivity"])
    if export:
        artifacts = ["records:confirmation_contrasts", "records:confirmation_sensitivity"]
        p.add_export("export.confirmation", artifacts=artifacts, after=[node["id"]],
                     outputs={a: p.record_set(a.split(":", 1)[1]) for a in artifacts},
                     steps=[{"argv": py("scripts/run_confirmation_analysis.py", "confirmation", "export", "--output",
                                        out, "--root", p.records)}])


def build_decision_audit(p, export_sets: list[str], gates: dict):
    datasets = p.present(EXPERIMENT_DATASETS["decision_audit"])
    stress = p.opts["negative_controls_dataset"]
    provisional = list(p.opts.get("provisional") or [])
    variants = [("decision_audit", [s for s in export_sets if not (provisional and s == "action_gap_audit")], [])]
    if provisional and "action_gap_audit" in export_sets:
        variants.append(("decision_audit_provisional", ["action_gap_audit"], provisional))
    gate_from = []
    for dataset in datasets:
        members = p.members(dataset)
        for member in members:
            if str(member) in gates:
                key = f"{dataset}/{member.name}" if len(members) > 1 else dataset
                gate_from.append(f"{key}={gates[str(member)]}")
    for name, sets, chosen in variants:
        if not sets:
            continue
        out = p.experiment_dir(name)

        def plan_argv(directory, with_gates=True, chosen=chosen):
            return py("scripts/run_decision_audit.py", "plan", *p.parents(datasets), "--output", directory,
                      "--stress-dataset", stress, *(repeat("--gate-from", gate_from) if with_gates else []),
                      *repeat("--provisional", chosen))
        probe = p.probe(name, lambda d: plan_argv(d, with_gates=False))
        count = len(probe.get("jobs") or [])
        deps = ["input_gate.run"] if gate_from else []
        artifacts = [f"records:{s}" for s in sets]
        planner = p.add_node(f"{name}.plan", title="Decision-log audit: plan behind the sealed input gates",
                             experiment="decision_audit", resource=CPU, tasks=[{"commands": []}], deps=deps,
                             outputs=[out], artifacts=artifacts, requeue=False,
                             plan_step={"argv": plan_argv(out), "plan_file": str(out / "plan.json"), "jobs": count,
                                        "exact": True}, commands=[plan_argv(out)])
        node = p.add_node(f"{name}.run", title="Instrumented serving-time execution of the deployed rules "
                          "(Tables S21, S33-S34; Figures 3c-d, S7, S16, S17)", experiment="decision_audit",
                          resource=GPU, deps=[planner["id"]], outputs=[out], artifacts=artifacts,
                          tasks=job_tasks(lambda job, out=out: _gpu_run(*py("scripts/run_decision_audit.py", "run",
                                                                            "--output", out, "--job", job)),
                                          count, plan_file=out / "plan.json"))
        p.add_export(f"export.{name}", artifacts=artifacts, after=[node["id"]],
                     outputs={a: p.record_set(a.split(":", 1)[1]) for a in artifacts},
                     steps=[{"argv": py("scripts/run_decision_audit.py", "export", "--output", out,
                                        *repeat("--set", sets), "--root", p.records)}])


def build_from_runs(p, set_name):
    """nominal_time and sensor_parameters: quantities the sealed runs record about themselves (no job)."""
    runs = []
    for dataset in p.present(EXPERIMENT_DATASETS[set_name]):
        members = p.members(dataset)
        runs.append(members[0])        # Roman-empire: its first split run, as the E17 timing records time it
    p.add_export(f"export.{set_name}", artifacts=[f"records:{set_name}"], after=[],
                 outputs={f"records:{set_name}": p.record_set(set_name)},
                 steps=[{"argv": py("scripts/export_records.py", "from-runs", "--set", set_name,
                                    *repeat("--run", runs), "--root", p.records)}])


def budget_study_command(p) -> list:
    """The batch script of a GPU allocation; outside Slurm (CPU plans) scripts/run_budget_study.py, which keeps
    the same job evidence and records that no scheduler accounting exists."""
    if p.opts.get("device") == "cpu":
        return py("scripts/run_budget_study.py", "--device", "cpu", "--threads", "{threads}",
                  *(["--allow-fixture"] if p.opts.get("allow_fixture") else []))
    return ["bash", "hpc/run_budget_arms_arrhenius.sh"]


def build_budget(p, export):
    """E3-E6: twelve controller seeds per admitted run (each Roman split) on fresh pre-admission parents."""
    if not p.repository.get("git_commit") or p.repository.get("dirty"):
        raise PlanError("the E3-E6 campaign names the source commit that executed it; plan from a clean git "
                        "checkout (or give --budget-campaign DIR)")
    seeds = [int(s) for s in p.opts["arm_seeds"]]
    # A fixture plan's studies cannot form the original campaign (cm_experiments/campaign.export_campaign
    # refuses software fixtures and missing accounting): its export is the partial campaign.
    complete = not p.opts.get("allow_fixture")
    studies, last = [], []
    for dataset in p.present(EXPERIMENT_DATASETS["budget"]):
        for member in p.members(dataset):
            label = p.label(dataset, member)
            parent = p.work / "budget" / "parents" / label
            config = read_json(member / "config.json")
            split = (config.get("data") or {}).get("split_index") if dataset == "Roman-empire" else None
            commands = [py("run_cross_morse.py", "--dataset", dataset, "--config", member / "config.json",
                           *(["--split-index", split] if split is not None else []), "--data-root", "{data_root}",
                           "--output-dir", parent, "--device", "cuda", "--threads", "{threads}", "--stage", stage,
                           "--resume") for stage in PRE_ADMISSION]
            first = p.add_node(f"budget.parent.{label}", title=f"E3-E6 pre-admission parent of {dataset}"
                               + (f" split {split}" if split is not None else "") + ": the sealed run's configuration "
                               "executed prepare->confirm in a fresh directory", experiment="budget", resource=GPU,
                               tasks=[{"commands": commands}], outputs=[parent], artifacts=["campaign:budget"],
                               dataset=dataset)
            tasks = []
            for seed in seeds:
                study = p.work / "budget" / "studies" / label / f"seed-{seed}"
                tasks.append({"commands": [budget_study_command(p)], "requeue": False,
                              "env": {"CM_SOURCE_ROOT": str(ROOT), "CM_BASE_RUN": str(parent),
                                      "CM_ARM_OUTPUT": str(study), "CM_TRUST_CHECKPOINTS": "1",
                                      "CM_ARM_SEED": str(seed)},
                              "fresh": {"directory": str(study), "complete": str(study / "ARTIFACT_MANIFEST.json"),
                                        "keep": []}})
                studies.append({"dataset": dataset, "seed": seed, "split": int(split or 0), "parent": str(parent),
                                "study": str(study)})
            node = p.add_node(f"budget.studies.{label}", title=f"E3-E6 arm studies of {dataset}"
                              + (f" split {split}" if split is not None else "") + f", controller seeds {seeds}",
                              experiment="budget", resource=GPU, tasks=tasks, deps=[first["id"]], requeue=False,
                              outputs=[s["study"] for s in studies[-len(seeds):]], artifacts=["campaign:budget"],
                              dataset=dataset)
            last.append(node["id"])
    campaign = p.work / "campaigns" / "budget"
    if export:
        manifest = p.work / "budget" / "campaign.json"
        p.add_export("export.budget", artifacts=["campaign:budget", "records:budget_matched"], after=last,
                     outputs={"campaign:budget": str(campaign),
                              "records:budget_matched": p.record_set("budget_matched")},
                     provenance={"kind": "budget", "campaign": str(campaign)},
                     steps=[{"internal": "finalize_budget_accounting", "studies": studies},
                            {"internal": "budget_manifest", "manifest": str(manifest), "studies": studies,
                             "seeds": seeds},
                            {"argv": py("scripts/arm_campaign.py", "export", "--manifest", manifest, "--output",
                                        campaign, *(["--complete"] if complete else [])),
                             "skip_if": str(campaign / ("analysis.json" if complete else "partial_analysis.json"))},
                            {"argv": py("scripts/export_records.py", "budget-matched", "--campaign", campaign,
                                        "--root", p.records, *([] if complete else ["--allow-partial"]),
                                        "--hardware", "{hardware}", "--period", "{period}")}])


def build_budget_provided(p, campaign: Path):
    p.add_export("export.budget_matched", artifacts=["records:budget_matched"], after=[],
                 outputs={"records:budget_matched": p.record_set("budget_matched")},
                 provenance={"kind": "budget", "campaign": str(campaign)},
                 steps=[{"argv": py("scripts/export_records.py", "budget-matched", "--campaign", campaign, "--root",
                                    p.records, "--hardware", "{hardware}", "--period", "{period}")}])


def baseline_plans(directory: Path) -> list[Path]:
    from cm_experiments.upstream import validate_plan
    plans = []
    for path in sorted(Path(directory).glob("**/*.json")):
        if len(path.relative_to(directory).parts) > 3:
            continue
        try:
            value = read_json(path)
        except (OSError, ValueError):
            continue
        if isinstance(value, dict) and value.get("schema") == "cm-upstream-study-1":
            try:
                validate_plan(value)
            except ValueError as exc:
                raise PlanError(f"{path}: {exc}") from None
            plans.append(path)
    return plans


def build_baselines(p, export, directory: Path):
    """E12: the pinned-upstream studies the author supplies, through hpc/run_upstream_baseline_arrhenius.sh, and the
    internal rows of Tables S15 and S15b measured on the sealed runs (scripts/run_internal_baselines.py).

    The export archives each study's adapter checkout at its pinned commit and references the upstream source
    (--checkouts: the plans resolve their checkouts relative to their own directory under DIR)."""
    plans = baseline_plans(directory)
    if not plans:
        raise PlanError(f"{directory}: no cm-upstream-study-1 plan (pinned upstream and adapter commits)")
    internal = p.experiment_dir("baselines_internal")
    plan_argv = py("scripts/run_internal_baselines.py", "plan", *p.parents(EXPERIMENT_DATASETS["baselines"]), "--output",
                   internal, *p.fixture("internal-baselines"))
    count = len(p.runner_plan(plan_argv, internal).get("jobs") or [])
    node = p.add_node("baselines.internal.run", title="E12/E17 internal rows of Tables S15 and S15b: every route's "
                      "configuration on D_test per plant checkpoint, E17 timing of q_C and the deployed rule",
                      experiment="baselines", resource=GPU, outputs=[internal], artifacts=["records:baselines"],
                      requeue=False, tasks=job_tasks(lambda job: _gpu_run(*py("scripts/run_internal_baselines.py", "run",
                                                                              "--output", internal, "--job", job)), count))
    for task in node["tasks"]:
        task["requeue"] = False        # one allocation per timing job: a requeue would give it two sacct records
    studies, finals = [], [node["id"]]
    for path in plans:
        plan = read_json(path)
        label = slug(f"{STEMS.get(plan['dataset'], plan['dataset'])}_{plan['method']}")
        study = p.work / "baselines" / "studies" / label
        env = {"CM_SOURCE_ROOT": str(ROOT), "CM_BASELINE_PLAN": str(path), "CM_BASELINE_OUTPUT": str(study),
               "CM_TRUST_UPSTREAM": "1"}
        hpo = p.add_node(f"baselines.{label}.hpo", title=f"E12 {plan['method']} on {plan['dataset']}: 48 HPO attempts",
                         experiment="baselines", resource=GPU, outputs=[study], artifacts=["records:baselines"],
                         requeue=False, dataset=plan["dataset"],
                         tasks=[{"commands": [["bash", "hpc/run_upstream_baseline_arrhenius.sh"]],
                                 "env": dict(env, CM_BASELINE_PHASE="hpo"), "accept": [0, 1], "requeue": False}
                                for _ in range(48)])
        select = p.add_node(f"baselines.{label}.select", title="E12 development selection over all 48 attempts",
                            experiment="baselines", resource=CPU, deps=[{"node": hpo["id"], "type": "afterany"}],
                            outputs=[study], artifacts=["records:baselines"], requeue=False,
                            tasks=[{"commands": [py("scripts/run_upstream_baseline.py", "--plan", path, "--output",
                                                    study, "--phase", "select")]}])
        for split in plan.get("official_splits", [0]):
            final = p.add_node(f"baselines.{label}.final.split{int(split)}", title=f"E12 final fits, twelve seeds, "
                               f"split {int(split)}", experiment="baselines", resource=GPU, deps=[select["id"]],
                               outputs=[study], artifacts=["records:baselines"], requeue=False, dataset=plan["dataset"],
                               tasks=[{"commands": [["bash", "hpc/run_upstream_baseline_arrhenius.sh"]],
                                       "env": dict(env, CM_BASELINE_PHASE="final", CM_BASELINE_SPLIT=str(int(split))),
                                       "requeue": False} for _ in range(12)])
            finals.append(final["id"])
        studies.append(str(study))
    timing = sorted(str(t) for t in Path(directory).glob("timing/*.csv"))
    accounting = sorted(str(a) for a in Path(directory).glob("accounting/*.psv"))
    jobs = ",".join("{jobs:" + n + "}" for n in p.nodes if n.startswith("baselines.") and not n.endswith(".select"))
    if export:
        capture = p.work / "accounting" / "baselines"
        p.add_export("export.baselines", artifacts=["records:baselines"], after=finals,
                     outputs={"records:baselines": p.record_set("baselines")},
                     provenance={"kind": "baselines", "studies": studies, "internal": str(internal),
                                 "accounting": str(capture / "sacct_original.psv")},
                     steps=[{"argv": ["bash", "hpc/capture_arrhenius_accounting.sh", jobs, capture],
                             "skip_if": str(capture / "sacct_original.psv")},
                            {"argv": py("scripts/export_records.py", "baselines", *repeat("--study", studies),
                                        "--internal", internal, "--checkouts", directory,
                                        "--copy-studies", "--copy-predictions",
                                        *repeat("--timing", timing),
                                        *repeat("--accounting", [capture / "sacct_original.psv", *accounting]),
                                        "--root", p.records, "--hardware", "{hardware}", "--period", "{period}")}])


def build_environment(p):
    out = p.work / "environment" / "gpu"
    node = p.add_node("environment.capture", title="Observed GPU environment of the runs' interpreter",
                      experiment="environment", resource=GPU, outputs=[out], artifacts=["environment:gpu"],
                      tasks=[{"commands": [["bash", "environment/arrhenius/capture_environment.sh", "{python}", out]],
                              "skip_if": str(out / "observed_environment.json")}])
    p.add_export("export.environment", artifacts=["environment:gpu"], after=[node["id"]], steps=[],
                 outputs={"environment:gpu": str(out)})


def required_artifact_ids() -> list[str]:
    from scripts.paper_artifacts import required_artifacts
    return [row["id"] for row in required_artifacts()]


def _normalize_only(values) -> set[str]:
    out = set()
    for value in values or ():
        value = str(value)
        if ":" in value:
            out.add(value)
        else:
            out |= {f"records:{value}", f"campaign:{value}", f"environment:{value}"}
    return out


# Record exporters that write the measured part of an incomplete set, with the cells it lacks, when given
# --allow-partial (never into records/). A fixture plan's runs (a software check or the CPU execution) can
# leave cells without a valid measurement, e.g. no state donors where every audited rule is open loop.
PARTIAL_EXPORTERS = {"scripts/run_confirmation_analysis.py", "scripts/run_experiment.py",
                     "scripts/run_trajectory_audit.py", "scripts/run_decision_audit.py", "scripts/run_timing_records.py"}


def declare_partial_exports(exports: list[dict]) -> None:
    """In a fixture plan, every export step of an exporter that can write a partial set passes --allow-partial."""
    for export in exports:
        for step in export["steps"]:
            argv = [str(a) for a in step.get("argv") or []]
            script = next((a for a in argv if a.endswith(".py")), None)
            partial = script in PARTIAL_EXPORTERS or (script == "scripts/run_ablation.py" and "modules" not in argv)
            if partial and "export" in argv and "--allow-partial" not in argv:
                step["argv"] = [*step["argv"], "--allow-partial"]


def build_plan(work, runs: dict, options: dict, *, required: list[str], executor=_subprocess_executor,
               suite=None, repository=None) -> dict:
    """The node graph and exports of every required artifact (see the module docstring)."""
    repository = dict(repository or {})
    repository.setdefault("producer_sources", producer_sources())
    p = Planner(work, runs, options, executor, repository)
    provided = dict(options.get("artifacts") or {})
    for key, value in (("campaign:budget", "budget_campaign"), ("campaign:confirmation", "confirmation_campaign"),
                       ("environment:gpu", "environment")):
        if options.get(value):
            provided[key] = options[value]
    only = _normalize_only(options.get("only"))
    selected = [a for a in required if not only or a in only or a.startswith("run:")]
    for artifact in required:
        if artifact.startswith("run:"):
            dataset = artifact.split(":", 1)[1]
            p.artifacts[artifact] = ({"status": "run", "path": runs[dataset]["path"]} if dataset in runs
                                     else {"status": "awaiting", "reason": f"no sealed {dataset} run was given"})
        elif artifact in provided:
            p.artifacts[artifact] = {"status": "provided", "path": str(Path(provided[artifact]).resolve())}
        elif artifact not in selected:
            p.artifacts[artifact] = {"status": "not_selected", "reason": "outside --only"}
    unknown = [a for a in selected if p.artifacts.get(a) is None and a not in ARTIFACT_EXPERIMENT]
    p.block(unknown, "no producing experiment is registered in scripts/paper_outputs.py")
    wanted = {ARTIFACT_EXPERIMENT[a] for a in selected if p.artifacts.get(a) is None}
    stack = list(wanted)
    while stack:
        for need in NEEDS.get(stack.pop(), ()):
            if need not in wanted and not (need == "s14_campaign" and "campaign:confirmation" in provided):
                wanted.add(need)
                stack.append(need)

    def artifacts_of(name):
        return [a for a in selected if ARTIFACT_EXPERIMENT.get(a) == name and p.artifacts.get(a) is None]

    exporting = {name: bool(artifacts_of(name)) for name in wanted}

    def attempt(name, function, *, needs=None):
        """Build one experiment; a missing run, input or failed runner plan leaves its artifacts awaiting."""
        if name not in wanted:
            return None
        datasets = EXPERIMENT_DATASETS.get(name, ())
        missing = p.missing(datasets)
        if missing and (not options.get("allow_subset") or len(missing) == len(datasets)):
            p.block(artifacts_of(name), f"no sealed run of {', '.join(missing)}")
            return None
        if needs:
            p.block(artifacts_of(name), needs)
            return None
        try:
            return function(p)
        except PlanError as exc:
            p.block(artifacts_of(name), str(exc))
            for node in p.nodes.values():
                if node["experiment"] == name and not node.get("blocked"):
                    node["blocked"] = str(exc)
            return None

    def unavailable(name, result):
        return None if result is not None else f"the {name.replace('_', ' ')} experiment is not planned"

    gates = attempt("input_gate", lambda q: build_input_gate(q, exporting["input_gate"]))
    gate_dirs = gates["dirs"] if gates else None
    controls = attempt("negative_controls", lambda q: build_negative_controls(q, exporting["negative_controls"],
                                                                              gate_dirs),
                       needs=unavailable("input_gate", gates))
    attempt("band_audit", lambda q: build_band_audit(q, exporting["band_audit"]))
    attempt("inserted_site", lambda q: build_inserted_site(q, exporting["inserted_site"]))
    grid = attempt("ensemble_grid", lambda q: build_ensemble_grid(q, exporting["ensemble_grid"]))
    attempt("ensemble_grid_scores", lambda q: build_ensemble_grid_scores(q, True, grid),
            needs=unavailable("ensemble_grid", grid))
    attempt("cross_backbone", lambda q: build_cross_backbone(q, exporting["cross_backbone"]))
    attempt("factorials", lambda q: build_factorials(q, exporting["factorials"]))
    attempt("sensitivity_grids", lambda q: build_sweep(q, exporting["sensitivity_grids"], "sensitivity_grids",
                                                       EXPERIMENT_DATASETS["sensitivity_grids"],
                                                       "E19 sensitivity grids: support-threshold variant lifecycles "
                                                       "prepare->select (Table S18)"))
    attempt("horizon_beam_sweep", lambda q: build_sweep(q, exporting["horizon_beam_sweep"], "horizon_beam_sweep",
                                                        EXPERIMENT_DATASETS["horizon_beam_sweep"],
                                                        "MPC-RH horizon and beam-width variant lifecycles "
                                                        "(Table S29, Figure S5)"))
    attempt("buffer_sensitivity", lambda q: build_sweep(q, exporting["buffer_sensitivity"], "buffer_sensitivity",
                                                        EXPERIMENT_DATASETS["buffer_sensitivity"],
                                                        "E11 descriptive admission at cluster buffers 2-32 hops: the "
                                                        "sealed per-core losses regrouped (Table S27)"))
    attempt("module_ablations", lambda q: build_module_ablations(q, exporting["module_ablations"], gate_dirs,
                                                                 controls["dirs"]),
            needs=unavailable("input_gate", gates) or unavailable("negative_controls", controls))
    scaling = attempt("runtime_scaling", lambda q: build_runtime_scaling(q, exporting["runtime_scaling"]))
    attempt("timing_anchors", lambda q: build_timing_anchors(q, exporting["timing_anchors"], scaling),
            needs=unavailable("runtime_scaling", scaling))
    for name, verb, title in (("mechanism_contrasts", "mechanism", "Figure 3a-b held-out mechanism contrasts on "
                               "D_confirm"),
                              ("oracle_dispersion", "oracle", "Admission-split oracle per input (C12; Tables S30, "
                               "S43)")):
        attempt(name, lambda q, name=name, verb=verb, title=title: _record_experiment(
            q, name=name, script="scripts/run_confirmation_analysis.py", verb=[verb],
            datasets=EXPERIMENT_DATASETS[name], artifacts=[f"records:{name}"], export=exporting[name], title=title))
    attempt("tube_audit", lambda q: _record_experiment(
        q, name="tube_audit", script="scripts/run_trajectory_audit.py", verb=["tube"],
        datasets=EXPERIMENT_DATASETS["tube_audit"], artifacts=["records:tube_audit"], export=exporting["tube_audit"],
        title="Proposition 2 trajectory-tube audit (Table S22, Figures 2a, S17e)"))
    attempt("depth_diagnostics", lambda q: _record_experiment(
        q, name="depth_diagnostics", script="scripts/run_trajectory_audit.py", verb=["depth"],
        datasets=EXPERIMENT_DATASETS["depth_diagnostics"], plan_extra=["--reference", "ogbn-arxiv"],
        artifacts=["records:depth_diagnostics"], export=exporting["depth_diagnostics"],
        title="Depth diagnostics of the plants and deployed rules (Figure S14)"))
    if "campaign:confirmation" in provided:
        campaign = {"campaign": str(Path(provided["campaign:confirmation"]).resolve()), "node": None}
    else:
        campaign = attempt("s14_campaign", lambda q: build_s14_campaign(q, exporting["s14_campaign"]))
    attempt("confirmation", lambda q: build_confirmation(q, exporting["confirmation"], campaign),
            needs=None if campaign else ((p.artifacts.get("campaign:confirmation") or {}).get("reason")
                                         or "the joint S14 campaign is not available"))
    attempt("decision_audit", lambda q: build_decision_audit(q, [a.split(":", 1)[1]
                                                                 for a in artifacts_of("decision_audit")], gate_dirs),
            needs=unavailable("input_gate", gates))
    for name in ("nominal_time", "sensor_parameters"):
        attempt(name, lambda q, name=name: build_from_runs(q, name))
    if "campaign:budget" in provided:
        attempt("budget", lambda q: build_budget_provided(q, Path(provided["campaign:budget"]).resolve()))
    else:
        attempt("budget", lambda q: build_budget(q, True))
    attempt("baselines", lambda q: build_baselines(q, True, Path(options["baselines"]).resolve()),
            needs=None if options.get("baselines") else (
                "E12 needs the author's pinned-upstream study plans (pinned upstream and adapter checkouts): give "
                "--baselines DIR, or a finished set with --artifact records:baselines=DIR"))
    attempt("environment", build_environment)
    if options.get("allow_fixture"):
        declare_partial_exports(p.exports)
    for artifact in required:
        if artifact not in p.artifacts:
            p.artifacts[artifact] = {"status": "awaiting", "reason": "not planned"}
    blocked_nodes = {k: v["blocked"] for k, v in p.nodes.items() if v.get("blocked")}
    for artifact, entry in p.artifacts.items():
        if entry["status"] == "planned":
            reasons = [f"{n}: {v}" for n, v in blocked_nodes.items() if artifact in p.nodes[n]["artifacts"]
                       or n in entry.get("nodes", [])]
            if reasons:
                p.artifacts[artifact] = {"status": "awaiting", "reason": reasons[0]}
    plan = {"schema": SCHEMA, "created_utc": utc_now(), "work": str(Path(work).resolve()),
            "repository": repository or {}, "suite": suite or {}, "options": dict(options),
            "declared_choices": {
                "negative_controls_dataset": options["negative_controls_dataset"],
                "band_audit": {"dataset": options["band_audit_dataset"], "baseline": options["band_audit_baseline"]},
                "arm_seeds": list(options["arm_seeds"]), "provisional": list(options.get("provisional") or []),
                "from_runs_roman_member": "split_00 (the first split run, as the E17 timing records time it)",
                **({"s14_excluded": options["s14_exclude"]} if options.get("s14_exclude") else {}),
                **({"horizon_sweep_configuration": options["horizon_sweep"]} if options.get("horizon_sweep") else {}),
                "budget_parents": "fresh executions prepare->confirm of each sealed run's own configuration; the "
                                  "sealed runs carry admission and test decisions, which the E3-E6 protocol "
                                  "(cm_experiments/runner.py) and the complete campaign export refuse"},
            "runs": runs, "nodes": list(p.nodes.values()), "exports": p.exports,
            "artifacts": {a: p.artifacts[a] for a in required}}
    return plain(plan)


def repository_identity() -> dict:
    identity = {"root": str(ROOT), "runtime_sha256": runtime_fingerprint(), "producer_sources": producer_sources()}
    try:
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=False, timeout=20)
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=False, timeout=60)
        if head.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", head.stdout.strip()):
            identity.update(git_commit=head.stdout.strip(), dirty=bool(dirty.stdout.strip()))
    except (OSError, subprocess.SubprocessError):
        pass
    return identity


def command_plan(args, executor=_subprocess_executor) -> int:
    work = Path(args.work).expanduser().resolve()
    if work.is_relative_to(ROOT):
        raise PlanError("write the work directory outside the repository checkout")
    if (work / "plan.json").exists():
        raise PlanError(f"{work}/plan.json exists; use a new --work directory")
    if getattr(args, "horizon_sweep_beam", None) is not None and not args.allow_fixture:
        raise PlanError("--horizon-sweep-beam sweeps a configuration the parent never planned: a fixture plan "
                        "(--allow-fixture)")
    found, suite = discover_runs(args.runs, args.run)
    runtime = runtime_fingerprint()
    runs = verify_runs(found, runtime=runtime, allow_fixture=args.allow_fixture)
    for entry in runs.values():
        if Path(entry["path"]).is_relative_to(work) or work.is_relative_to(Path(entry["path"])):
            raise PlanError("the work directory must be separate from every run")
    artifacts = {}
    for spec in args.artifact:
        key, sep, value = spec.partition("=")
        if not sep or not value or ":" not in key:
            raise PlanError(f"--artifact takes ID=DIR, e.g. records:buffer_sensitivity=/abs/dir: {spec!r}")
        if not Path(value).is_dir():
            raise PlanError(f"--artifact {spec}: not a directory")
        artifacts[key] = str(Path(value).resolve())
    for option in ("budget_campaign", "confirmation_campaign", "baselines", "environment"):
        value = getattr(args, option)
        if value is not None and not Path(value).is_dir():
            raise PlanError(f"--{option.replace('_', '-')} {value}: not a directory")
    options = {"only": list(args.only or []), "allow_fixture": bool(args.allow_fixture),
               "allow_subset": bool(args.allow_subset or args.allow_fixture),
               "negative_controls_dataset": args.negative_controls_dataset,
               "band_audit_dataset": args.band_audit_dataset, "band_audit_baseline": args.band_audit_baseline,
               "arm_seeds": list(args.arm_seed or DEFAULT_ARM_SEEDS), "provisional": list(args.provisional or []),
               "data_root": str(Path(args.data_root).resolve()) if args.data_root else (suite.get("data_root") or None),
               "artifacts": artifacts,
               **{k: (str(Path(getattr(args, k)).resolve()) if getattr(args, k) else None)
                  for k in ("budget_campaign", "confirmation_campaign", "baselines", "environment")}}
    if len(set(options["arm_seeds"])) != 12:
        raise PlanError("declare twelve distinct controller-optimization seeds (--arm-seed, twelve times)")
    for dataset in (options["negative_controls_dataset"], options["band_audit_dataset"], *options["provisional"]):
        if dataset not in DATASETS:
            raise PlanError(f"unknown dataset {dataset!r}; canonical names: {list(DATASETS)}")
    if args.device != "cuda":
        if not args.allow_fixture:
            raise PlanError("--device cpu executes the runners on a CPU: a fixture plan (--allow-fixture)")
        options["device"] = args.device
    if args.s14_exclude:
        if not args.allow_fixture:
            raise PlanError("--s14-exclude leaves the joint S14 family partial: a fixture plan (--allow-fixture)")
        unknown = sorted(set(args.s14_exclude) - set(DATASETS))
        if unknown:
            raise PlanError(f"--s14-exclude: unknown datasets {unknown}")
        options["s14_exclude"] = sorted(set(args.s14_exclude))
    if args.horizon_sweep_profile or args.horizon_sweep_ensemble is not None or args.horizon_sweep_beam is not None:
        options["horizon_sweep"] = {"profile": args.horizon_sweep_profile, "ensemble": args.horizon_sweep_ensemble,
                                    **({"beam": args.horizon_sweep_beam} if args.horizon_sweep_beam is not None else {})}
    work.mkdir(parents=True, exist_ok=True)
    plan = build_plan(work, runs, options, required=required_artifact_ids(), suite=suite, executor=executor,
                      repository=dict(repository_identity(), runtime_sha256=runtime))
    if options.get("device"):
        plan["nodes"] = with_device(plan["nodes"], options["device"])
    write_json(work / "plan.json", plan)
    print("\n".join(plan_summary(plan)))
    return 0


def with_device(value, device: str):
    """Every runner argument vector with ``--device cuda`` rewritten to ``--device <device>``."""
    if isinstance(value, dict):
        return {key: with_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        items = [with_device(item, device) for item in value]
        if all(isinstance(item, str) for item in items):
            items = [device if index and items[index - 1] == "--device" and item == "cuda" else item
                     for index, item in enumerate(items)]
        return items
    return value


def plan_summary(plan: dict) -> list[str]:
    lines = [f"plan: {plan['work']}/plan.json"]
    for node in plan["nodes"]:
        size = f"array {node['array']}" if node.get("array") else "1 job"
        state = f"  BLOCKED: {node['blocked']}" if node.get("blocked") else ""
        deps = ",".join(d["node"] for d in node["dependencies"]) or "-"
        lines.append(f"  {node['id']:<42} {node['resource']:<3} {size:<10} after {deps}{state}")
    counts = {}
    for entry in plan["artifacts"].values():
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    lines.append("artifacts: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    for artifact, entry in plan["artifacts"].items():
        if entry["status"] in ("awaiting", "not_selected"):
            lines.append(f"  {entry['status'].upper():<12} {artifact}: {entry.get('reason', '')}")
    return lines


# ============================================================================= command-line validation
def _job_tokens(argv) -> list[str]:
    return [m.group(1)[5:] for a in argv for m in PLACEHOLDER.finditer(str(a)) if m.group(1).startswith("jobs:")]


class _Parsed(Exception):
    def __init__(self, namespace):
        super().__init__("parsed")
        self.namespace = namespace


def check_cli(argv: list[str]) -> None:
    """Parse ``argv`` with the entry point's own argparse parser without executing anything."""
    import argparse as _argparse
    import importlib.util
    args = expand(list(argv), {"python": "{python}", "threads": "16", "data_root": "/data-root", "hardware": "hw",
                               "period": "2026-01-01/2026-01-02", **{f"jobs:{n}": "1" for n in _job_tokens(argv)}})
    if args and args[0] == "{python}":
        args = args[2:] if len(args) > 1 and args[1] == "-B" else args[1:]
    script = ROOT / args[0]
    if not script.is_file():
        raise PlanError(f"{args[0]} is not in this checkout")
    original = _argparse.ArgumentParser.parse_args
    saved = sys.argv

    def intercept(self, values=None, namespace=None):
        parsed = original(self, values, namespace)
        raise _Parsed(parsed)
    _argparse.ArgumentParser.parse_args = intercept
    sys.argv = [str(script), *args[1:]]
    try:
        spec = importlib.util.spec_from_file_location(f"_cm_cli_{script.stem}", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        main = getattr(module, "main")
        try:
            main(args[1:]) if main.__code__.co_argcount else main()
        except _Parsed:
            return
        except SystemExit as exc:
            raise PlanError(f"{args[0]} rejects {' '.join(args[1:4])}...: argparse exit {exc.code}") from None
        raise PlanError(f"{args[0]} did not parse its command line")
    finally:
        _argparse.ArgumentParser.parse_args = original
        sys.argv = saved


# ============================================================================= submission
def site_settings(environ=None) -> dict:
    """The effective SITE SETTINGS of run_cross_morse.sh (CM_ENV_FILE and exported CM_* honoured)."""
    result = subprocess.run(["bash", "-c", 'eval "$(bash "$1" site-env)" && env -0', "_",
                             str(ROOT / "run_cross_morse.sh")], capture_output=True, check=False,
                            env=dict(os.environ if environ is None else environ))
    if result.returncode != 0:
        raise PlanError("bash run_cross_morse.sh site-env failed: " + result.stderr.decode(errors="replace")[-500:])
    settings = {}
    for item in result.stdout.split(b"\0"):
        key, sep, value = item.decode(errors="replace").partition("=")
        if sep and key.startswith("CM_"):
            settings[key] = value
    return settings


def _time(settings: dict, resource: str) -> str:
    if resource == "collect":
        return settings.get("CM_PRODUCE_COLLECT_TIME") or "24:00:00"
    if resource == CPU:
        return settings.get("CM_PRODUCE_CPU_TIME") or "04:00:00"
    return settings.get("CM_PRODUCE_GPU_TIME") or settings.get("CM_TIME") or settings.get("CM_TIME_FULL") or "24:00:00"


def sbatch_options(node: dict, settings: dict, work: Path) -> list[str]:
    """The site's scheduler settings for one node (run_cross_morse.sh sbatch_site_args, run or report)."""
    resource = node["resource"]
    gpu = resource == GPU
    threads = int(settings.get("CM_THREADS") or 16)
    options = [f"--account={settings.get('CM_ACCOUNT', '')}",
               f"--partition={settings.get('CM_PARTITION' if gpu else 'CM_REPORT_PARTITION') or settings.get('CM_PARTITION') or 'gpu'}",
               f"--cpus-per-task={threads if gpu else min(threads, 8)}"]
    options += shlex.split(settings.get("CM_GPU_REQUEST" if gpu else "CM_REPORT_GPU_REQUEST", "") or "")
    memory = settings.get("CM_MEM", "")
    if node.get("dataset") in STEMS:
        memory = settings.get(f"CM_MEM_{STEMS[node['dataset']].upper()}") or memory
    if memory:
        options.append(f"--mem={memory}")
    options += shlex.split(settings.get("CM_SBATCH_EXTRA", "") or "")
    name = "cmpo_" + re.sub(r"[^A-Za-z0-9_.-]", "_", node["id"])[:60]
    logs = work / "slurm" / "logs"
    pattern = f"{node['id']}_%A_%a" if node.get("array") else f"{node['id']}_%j"
    options += [f"--time={_time(settings, resource if node['id'] != 'collect' else 'collect')}",
                f"--job-name={name}", f"--output={logs / (pattern + '.out')}", f"--error={logs / (pattern + '.err')}"]
    if node.get("array"):
        throttle = settings.get("CM_PRODUCE_ARRAY_PARALLEL", "")
        options.append(f"--array=0-{node['array'] - 1}" + (f"%{throttle}" if throttle else ""))
    if gpu and node.get("requeue"):
        options += ["--requeue", "--signal=B:USR1@180"]
    return options


BATCH_TEMPLATE = """#!/usr/bin/env bash
# CROSS-MORSE paper outputs -- node {node}
# {title}
# Generated by scripts/paper_outputs.py submit from {plan}; the site settings are those of
# run_cross_morse.sh (bash run_cross_morse.sh site-env), read again when the job starts.
set -Eeuo pipefail
ROOT={root}
WORK={work}
cd "$ROOT"
eval "$(bash "$ROOT/run_cross_morse.sh" site-env)" || {{ echo 'Cannot read the site settings of run_cross_morse.sh' >&2; exit 2; }}
if [[ -n "${{CM_BUILD_MODULE}}${{CM_PYTHON_MODULE}}" ]]; then
  if ! type module >/dev/null 2>&1 && [[ -f "${{MODULESHOME:-/nonexistent}}/init/bash" ]]; then source "${{MODULESHOME}}/init/bash"; fi
  set +u
  module --force purge >/dev/null 2>&1 || module purge >/dev/null 2>&1 || true
  # shellcheck disable=SC2086
  module load ${{CM_BUILD_MODULE}} ${{CM_PYTHON_MODULE}} || {{ set -u; echo 'Cannot load the site modules' >&2; exit 33; }}
  set -u
fi
PYTHON="${{CM_PYTHON:-${{CM_VENV_DIR}}/bin/python}}"
[[ -x "$PYTHON" ]] || {{ echo "Configured interpreter missing: $PYTHON (run: bash run_cross_morse.sh submit setup)" >&2; exit 33; }}
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
unset PYTHONPATH || true
THREADS="${{SLURM_CPUS_PER_TASK:-${{CM_THREADS:-16}}}}"
export OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"
export TORCH_HOME="${{CM_BASE}}/cache/torch" XDG_CACHE_HOME="${{CM_BASE}}/cache/xdg" OGB_HOME="${{CM_DATA_ROOT}}"
{gpu_block}exec "$PYTHON" -B "$ROOT/scripts/paper_outputs.py" {action} --work "$WORK"{node_option}
"""
GPU_BLOCK = """export CUBLAS_WORKSPACE_CONFIG=:4096:8 NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
"$PYTHON" -c 'import os,platform,torch; a=os.environ.get("CM_EXPECT_ARCH",""); assert not a or platform.machine()==a, platform.machine(); assert torch.cuda.is_available(), "CUDA unavailable"; print("[GPU]", torch.__version__, torch.cuda.get_device_name())' || exit 33
"""


def batch_script(node: dict, work: Path) -> str:
    gpu = node["resource"] == GPU
    action = "collect" if node["id"] == "collect" else "execute"
    return BATCH_TEMPLATE.format(node=node["id"], title=node.get("title", ""), plan=work / "plan.json",
                                 root=shlex.quote(str(ROOT)), work=shlex.quote(str(work)),
                                 gpu_block=GPU_BLOCK if gpu else "", action=action,
                                 node_option="" if action == "collect" else f" --node {shlex.quote(node['id'])}")


def load_plan(work: Path) -> dict:
    path = Path(work) / "plan.json"
    if not path.is_file():
        raise PlanError(f"{path} does not exist; run: python -B scripts/paper_outputs.py plan ...")
    plan = read_json(path)
    if plan.get("schema") != SCHEMA:
        raise PlanError(f"{path}: not a {SCHEMA} plan")
    return plan


def load_jobs(work: Path) -> dict:
    path = Path(work) / "jobs.json"
    if not path.is_file():
        return {"schema": JOBS_SCHEMA, "nodes": {}}
    return read_json(path)


def completion_evidence(node: dict, task_index: int) -> dict:
    """Task-specific completion seals, plus its declared output existence contract."""
    task = node["tasks"][task_index] if node.get("tasks") else {}
    paths = []
    if node.get("plan_step"):
        paths.append(node["plan_step"]["plan_file"])
    if task.get("skip_if"):
        paths.append(task["skip_if"])
        if Path(task["skip_if"]).name == "observed_environment.json":
            paths.extend(str(Path(task["skip_if"]).parent / name) for name in
                         ("requirements-gpu.lock", "nvidia-smi.txt", "modules.txt"))
    if task.get("fresh"):
        paths.append(task["fresh"]["complete"])
    seals = {}
    for name in paths:
        path = Path(name)
        if not path.is_file() or path.is_symlink():
            raise PlanError(f"Completion seal is missing or linked: {path}")
        seals[str(path)] = sha256_file(path)
    return seals


def task_complete(work: Path, node: dict, task_index: int) -> bool:
    try:
        record = read_json(task_marker(work, node["id"], task_index))
        if record.get("status") not in COMPLETE_STATES:
            return False
        if record.get("plan_sha256") != sha256_file(Path(work) / "plan.json"):
            return False
        if record.get("node_sha256") != hashlib.sha256(json.dumps(node, sort_keys=True).encode()).hexdigest():
            return False
        if record["status"] == "SKIPPED":
            job_in = node["tasks"][task_index].get("job_in") or {}
            return bool(job_in.get("upper_bound") and job_in["index"] >= len(read_json(Path(job_in["plan"])).get("jobs") or []))
        return (record.get("completion_seals") == completion_evidence(node, task_index)
                and all(Path(name).exists() for name in node.get("outputs", [])))
    except (OSError, ValueError, KeyError):
        return False


def node_complete(work: Path, node: dict) -> bool:
    return (all(task_complete(work, node, index) for index in range(len(node["tasks"]) or 1))
            and all(Path(name).exists() for name in node.get("outputs", [])))


def submission_order(plan: dict, work: Path, *, resume: bool) -> tuple[list[dict], dict, dict]:
    """(nodes to submit, skipped {id: reason}, finished {id}) in dependency order."""
    nodes = {n["id"]: n for n in plan["nodes"]}
    order, skipped, finished = [], {}, set()
    for node in plan["nodes"]:
        if node.get("blocked"):
            skipped[node["id"]] = "blocked: " + node["blocked"]
            continue
        bad = [d["node"] for d in node["dependencies"] if d["node"] in skipped]
        if bad:
            skipped[node["id"]] = f"depends on {', '.join(bad)}, which is not submitted"
            continue
        if any(d["node"] not in nodes for d in node["dependencies"]):
            raise PlanError(f"{node['id']}: unknown dependency")
        if resume and node_complete(work, node):
            finished.add(node["id"])
            continue
        order.append(node)
    return order, skipped, finished


def _sbatch(arguments: list[str]) -> str:
    result = subprocess.run(["sbatch", "--parsable", *arguments], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise PlanError(f"sbatch refused {arguments[-1]}: {result.stderr.strip()[-500:]}")
    job = result.stdout.strip().split(";", 1)[0]
    if not re.fullmatch(r"[0-9]+", job):
        raise PlanError(f"sbatch returned no job id: {result.stdout!r}")
    return job


def command_submit(args) -> int:
    with work_lock(Path(args.work).expanduser().resolve(), "submit"):
        return _command_submit(args)


def _command_submit(args) -> int:
    work = Path(args.work).expanduser().resolve()
    plan = load_plan(work)
    verify_plan_source(plan)
    jobs = load_jobs(work)
    if jobs["nodes"] and jobs.get("plan_sha256") != sha256_file(work / "plan.json"):
        raise PlanError("jobs.json belongs to another plan; retain the existing workspace and plan in a new directory")
    if jobs["nodes"] and not args.resume and not args.dry_run:
        raise PlanError(f"{work}/jobs.json lists submitted jobs; use --resume to submit only unfinished nodes")
    settings = site_settings()
    if not args.dry_run:
        if not settings.get("CM_ACCOUNT"):
            raise PlanError("set CM_ACCOUNT in the SITE SETTINGS of run_cross_morse.sh, the environment or CM_ENV_FILE")
        if shutil.which("sbatch") is None:
            raise PlanError("sbatch is not available in this shell")
    order, skipped, finished = submission_order(plan, work, resume=args.resume)
    (work / "slurm" / "logs").mkdir(parents=True, exist_ok=True)
    ids: dict[str, str] = {}
    reused = {}
    if args.resume and jobs["nodes"]:
        prior = {name: item["job_id"] for name, item in jobs["nodes"].items() if item.get("job_id")}
        scheduler = scheduler_states(list(prior.values()))
        for node in [*order, {"id": "collect"}]:
            name = node["id"]
            if name not in prior:
                continue
            states = (scheduler or {}).get(prior[name]) or []
            if not states:
                raise PlanError(f"Cannot establish whether Slurm job {prior[name]} ({name}) is active; "
                                "resume refuses a duplicate submission until sacct/squeue provides its state")
            if any(state in SACCT_ACTIVE for state in states):
                reused[name] = prior[name]
        replacing = {node["id"] for node in order} - set(reused)
        for node in order:
            if node["id"] in reused and any(dep["node"] in replacing for dep in node["dependencies"]):
                raise PlanError(f"Active job {reused[node['id']]} ({node['id']}) depends on a node that needs retry; "
                                "wait for its terminal state or cancel that dependent job before resuming")
        if "collect" in reused and replacing:
            raise PlanError(f"Collect job {reused['collect']} is still active while upstream nodes need retry; "
                            "wait for its terminal state or cancel that collect job before resuming")
        ids.update(reused)
    printed = []
    for node in [*order, collect_node(plan, order)]:
        if node["id"] in reused:
            print(f"[active, not resubmitted] {node['id']}: job {reused[node['id']]}")
            continue
        script = work / "slurm" / f"{node['id']}.sbatch"
        script.write_text(batch_script(node, work), encoding="utf-8")
        script.chmod(0o755)
        options = sbatch_options(node, settings, work)
        dependencies = {}
        for dep in node["dependencies"]:
            if dep["node"] in finished:
                continue
            job = ids.get(dep["node"]) or (f"<{dep['node']}>" if args.dry_run else None)
            if job is None:
                raise PlanError(f"{node['id']}: dependency {dep['node']} has no job id")
            dependencies.setdefault(dep["type"], []).append(job)
        if dependencies:
            options += ["--dependency=" + ",".join(f"{kind}:{':'.join(v)}" for kind, v in dependencies.items()),
                        "--kill-on-invalid-dep=yes"]
        arguments = [*options, str(script)]
        printed.append("sbatch --parsable " + " ".join(shlex.quote(a) for a in arguments))
        if args.dry_run:
            ids[node["id"]] = f"<{node['id']}>"
            continue
        job = _sbatch(arguments)
        ids[node["id"]] = job
        record = jobs["nodes"].setdefault(node["id"], {"history": []})
        record.update(job_id=job, array=node.get("array"), submitted_utc=utc_now(), script=str(script),
                      sbatch=arguments)
        record["history"].append(job)
        jobs.update(schema=JOBS_SCHEMA, updated_utc=utc_now(), plan_sha256=sha256_file(work / "plan.json"))
        write_json(work / "jobs.json", jobs)
    for line in printed:
        print(line)
    for node, reason in skipped.items():
        print(f"[not submitted] {node}: {reason}", file=sys.stderr)
    if finished:
        print(f"[complete, not resubmitted] {', '.join(sorted(finished))}")
    if not args.dry_run:
        print(f"submitted {len(order)} node(s) and the collect job {ids['collect']}; job ids in {work}/jobs.json")
        print(f"follow:  python3 -B {ROOT / 'scripts/paper_outputs.py'} status --work {work}")
    print(f"publish: bash {ROOT / 'hpc/publish_paper_artifacts_arrhenius.sh'} all {work / MAP_NAME}")
    return 0


def collect_node(plan: dict, order: list[dict]) -> dict:
    return {"id": "collect", "title": "Export every record set, assemble the campaigns, write the publication map",
            "resource": CPU, "array": None, "tasks": [{"commands": []}], "requeue": False, "dataset": None,
            "dependencies": [{"node": n["id"], "type": "afterok"} for n in order]}


# ============================================================================= execution (inside a Slurm job)
PLACEHOLDER = re.compile(r"\{(python|threads|data_root|hardware|period|jobs:[^{}]+)\}")


def expand(argv: list[str], values: dict) -> list[str]:
    def one(token):
        def substitute(match):
            key = match.group(1)
            if key not in values:
                raise PlanError(f"no value for {{{key}}} in {argv[:4]}")
            return str(values[key])
        return PLACEHOLDER.sub(substitute, str(token))
    return [one(a) for a in argv]


def task_marker(work: Path, node_id: str, task: int) -> Path:
    return Path(work) / "state" / node_id / f"task-{task}.json"


def write_marker(work, node_id, task, status, **fields) -> dict:
    path = task_marker(work, node_id, task)
    try:
        previous = read_json(path)
    except (OSError, ValueError):
        previous = {}
    identity = {}
    if (Path(work) / "plan.json").is_file():
        plan = load_plan(work)
        node = next((row for row in plan["nodes"] if row["id"] == node_id), None)
        if node is not None:
            identity = {"plan_sha256": sha256_file(Path(work) / "plan.json"),
                        "node_sha256": hashlib.sha256(json.dumps(node, sort_keys=True).encode()).hexdigest()}
            if status == "COMPLETE":
                identity["completion_seals"] = completion_evidence(node, task)
    record = dict(previous, **identity, schema=TASK_SCHEMA, node=node_id, task=task, status=status, updated_utc=utc_now(),
                  job=os.environ.get("SLURM_JOB_ID"), array_job=os.environ.get("SLURM_ARRAY_JOB_ID"), **fields)
    write_json(path, record)
    return record


def _move_aside(directory: Path) -> Path:
    for n in range(1, 1000):
        target = directory.with_name(f"{directory.name}.attempt-{n}")
        if not target.exists():
            directory.rename(target)
            return target
    raise PlanError(f"{directory}: too many earlier attempts")


def _execute_values(plan: dict) -> dict:
    data_root = (plan.get("options") or {}).get("data_root") or os.environ.get("CM_DATA_ROOT")
    values = {"python": sys.executable,
              "threads": os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("CM_THREADS") or "16"}
    if data_root:
        values["data_root"] = data_root
    return values


class _Child:
    process = None


def _forward(signum, _frame):
    if _Child.process is not None and _Child.process.poll() is None:
        try:
            _Child.process.send_signal(signum)
        except OSError:
            pass


def run_command(argv: list[str], env: dict | None = None) -> int:
    print("[command] " + " ".join(shlex.quote(a) for a in argv), flush=True)
    _Child.process = subprocess.Popen(argv, cwd=str(ROOT), env=dict(os.environ, **(env or {})))
    while True:
        try:
            return _Child.process.wait()
        except InterruptedError:
            continue


def run_plan_step(step: dict, values: dict) -> dict:
    """A deferred runner plan, written once inside the job; its job count must fit the submitted array."""
    plan_file = Path(step["plan_file"])
    if not plan_file.is_file():
        directory = plan_file.parent
        if directory.exists() and any(directory.iterdir()):
            _move_aside(directory)
        code = run_command(expand(step["argv"], values))
        if code != 0 or not plan_file.is_file():
            raise PlanError(f"the runner's plan exited with {code}")
    jobs = read_json(plan_file).get("jobs") or []
    if step.get("exact") and len(jobs) != int(step["jobs"]):
        raise PlanError(f"{plan_file}: {len(jobs)} jobs, the submitted array has {step['jobs']}; plan again")
    if len(jobs) > int(step["jobs"]):
        raise PlanError(f"{plan_file}: {len(jobs)} jobs exceed the submitted array of {step['jobs']}")
    if step.get("kinds") and [j.get("kind") for j in jobs] != list(step["kinds"]):
        raise PlanError(f"{plan_file}: the job kinds differ from the planned ones; plan again")
    return {"jobs": len(jobs)}


def fresh_complete(fresh: dict) -> bool:
    """A non-resumable output is complete: its seal exists (and, with complete_field, records that field true)."""
    path = Path(fresh["complete"])
    if not path.is_file() or not Path(fresh["directory"]).exists():
        return False
    if fresh.get("complete_field"):
        try:
            return read_json(path).get(fresh["complete_field"]) is True
        except (OSError, ValueError):
            return False
    return True


def execute_task(work: Path, plan: dict, node: dict, task_index: int) -> int:
    """One task of one node; returns the process exit status for Slurm."""
    values = _execute_values(plan)
    marker = task_marker(work, node["id"], task_index)
    try:
        previous = read_json(marker)
    except (OSError, ValueError):
        previous = {}
    if task_complete(work, node, task_index):
        print(f"[skip] {node['id']} task {task_index} is {previous['status']}")
        return 0
    attempts = int(previous.get("attempts", 0)) + 1
    write_marker(work, node["id"], task_index, "RUNNING", attempts=attempts, started_utc=utc_now())
    try:
        if node.get("plan_step"):
            detail = run_plan_step(node["plan_step"], values)
            write_marker(work, node["id"], task_index, "COMPLETE", exit_code=0, detail=detail)
            return 0
        task = node["tasks"][task_index]
        if task.get("skip_if") and Path(task["skip_if"]).exists() and previous.get("status") not in COMPLETE_STATES:
            completion_evidence(node, task_index)
            write_marker(work, node["id"], task_index, "COMPLETE", exit_code=0, detail="output present")
            return 0
        job_in = task.get("job_in")
        if job_in:
            jobs = read_json(job_in["plan"]).get("jobs") or []
            if job_in["index"] >= len(jobs):
                if not job_in.get("upper_bound"):
                    raise PlanError(f"{job_in['plan']} has no job {job_in['index']}; plan again")
                write_marker(work, node["id"], task_index, "SKIPPED", exit_code=0,
                             detail=f"the plan holds {len(jobs)} jobs")
                return 0
            if job_in.get("kind") and jobs[job_in["index"]].get("kind") != job_in["kind"]:
                raise PlanError(f"{job_in['plan']}: job {job_in['index']} is not a {job_in['kind']} job")
        fresh = task.get("fresh")
        if fresh:
            directory = Path(fresh["directory"])
            directory.parent.mkdir(parents=True, exist_ok=True)
            if fresh_complete(fresh) and previous.get("status") not in COMPLETE_STATES:
                write_marker(work, node["id"], task_index, "COMPLETE", exit_code=0, detail="sealed output present")
                return 0
            keep = set(fresh.get("keep") or [])
            if directory.exists() and any(p.name not in keep for p in directory.iterdir()):
                moved = _move_aside(directory)
                print(f"[fresh] an earlier attempt was moved to {moved}")
            if fresh.get("plan") and not Path(fresh["plan_file"]).is_file():
                code = run_command(expand(fresh["plan"], values))
                if code != 0:
                    write_marker(work, node["id"], task_index, "FAILED", exit_code=code, detail="runner plan failed")
                    return code
        accept = set(task.get("accept") or [0])
        for argv in task["commands"]:
            code = run_command(expand(argv, values), task.get("env"))
            if code == CHECKPOINTED and task.get("requeue", node.get("requeue")):
                return continuation(work, node, task_index)
            if code not in accept:
                write_marker(work, node["id"], task_index, "FAILED", exit_code=code)
                return code or 1
        write_marker(work, node["id"], task_index, "COMPLETE", exit_code=0, finished_utc=utc_now())
        return 0
    except (PlanError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        write_marker(work, node["id"], task_index, "FAILED", exit_code=2, detail=str(exc)[:500])
        return 2


def continuation(work: Path, node: dict, task_index: int) -> int:
    """A runner checkpointed at its time limit: requeue this job (the command resumes) within the site budget."""
    marker = read_json(task_marker(work, node["id"], task_index))
    count = int(marker.get("continuations", 0)) + 1
    limit = int(os.environ.get("CM_MAX_CONTINUATIONS") or 12)
    array, job = os.environ.get("SLURM_ARRAY_JOB_ID"), os.environ.get("SLURM_JOB_ID")
    target = f"{array}_{os.environ.get('SLURM_ARRAY_TASK_ID')}" if array else job
    if count > limit or not target or shutil.which("scontrol") is None:
        write_marker(work, node["id"], task_index, "FAILED", exit_code=CHECKPOINTED, continuations=count - 1,
                     detail="checkpointed; the continuation budget is exhausted or requeue is unavailable")
        return CHECKPOINTED
    write_marker(work, node["id"], task_index, "REQUEUED", continuations=count, exit_code=CHECKPOINTED)
    if subprocess.run(["scontrol", "requeue", target], check=False).returncode == 0:
        print(f"[continue] {target} requeued ({count} of {limit}); the runner resumes from its last safe boundary")
        return 0
    write_marker(work, node["id"], task_index, "FAILED", exit_code=CHECKPOINTED, detail="scontrol requeue refused")
    return CHECKPOINTED


def command_execute(args) -> int:
    work = Path(args.work).expanduser().resolve()
    plan = load_plan(work)
    nodes = {n["id"]: n for n in plan["nodes"]}
    if args.node not in nodes:
        raise PlanError(f"{args.node} is not a node of {work}/plan.json")
    node = nodes[args.node]
    task = args.task if args.task is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID") or 0)
    if not 0 <= task < max(1, len(node["tasks"])):
        raise PlanError(f"{node['id']} has no task {task}")
    verify_plan_source(plan)
    # Slurm signals the batch shell (--signal=B:USR1@180), which exec'd this process: pass it to the runner.
    previous = {name: signal.signal(getattr(signal, name), _forward) for name in ("SIGUSR1", "SIGTERM", "SIGINT")}
    try:
        with work_lock(work, f"task-{node['id']}-{task}"):
            return execute_task(work, plan, node, task)
    finally:
        for name, handler in previous.items():
            signal.signal(getattr(signal, name), handler)


# ============================================================================= status
SACCT_ACTIVE = ("PENDING", "RUNNING", "REQUEUED", "CONFIGURING", "COMPLETING", "SUSPENDED", "RESIZING")


def scheduler_states(job_ids: list[str]) -> dict[str, list[str]] | None:
    """Accounting plus live queue state: an accounting delay must not duplicate an active job."""
    if not job_ids:
        return {}
    states, available = {}, False
    commands = []
    if shutil.which("sacct"):
        commands.append(["sacct", "-j", ",".join(job_ids), "-X", "-n", "--parsable2", "--format=JobID,State"])
    if shutil.which("squeue"):
        commands.append(["squeue", "--jobs", ",".join(job_ids), "--noheader", "--format=%A|%T"])
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            continue
        available = True
        for line in result.stdout.splitlines():
            parts = line.strip().split("|")
            if len(parts) < 2 or not parts[0]:
                continue
            base = parts[0].split("_", 1)[0].split(".", 1)[0]
            state = parts[1].split()[0].rstrip("+") if parts[1].split() else ""
            if base in job_ids:
                states.setdefault(base, []).append(state)
    return states if available else None


def node_status(work: Path, node: dict, record: dict | None, scheduler: dict | None) -> dict:
    tasks = max(1, len(node["tasks"]))
    markers = []
    for k in range(tasks):
        try:
            markers.append(read_json(task_marker(work, node["id"], k)).get("status"))
        except (OSError, ValueError):
            markers.append(None)
    done = sum(task_complete(work, node, k) for k in range(tasks))
    failed = sum(1 for m in markers if m == "FAILED")
    if node.get("blocked"):
        state = "BLOCKED"
    elif done == tasks:
        state = "COMPLETE"
    elif record is None:
        state = "NOT_SUBMITTED"
    else:
        sched = (scheduler or {}).get(str(record.get("job_id")))
        if sched:
            if any(s in SACCT_ACTIVE for s in sched):
                state = "RUNNING" if any(s == "RUNNING" for s in sched) else "PENDING"
            elif any(s not in ("COMPLETED",) for s in sched):
                state = "FAILED"
            else:
                state = "INCOMPLETE" if not failed else "FAILED"
        else:
            state = "FAILED" if failed else ("RUNNING" if any(m in ("RUNNING", "REQUEUED") for m in markers)
                                              else "SUBMITTED")
    return {"node": node["id"], "state": state, "tasks": tasks, "complete": done, "failed": failed,
            "job_id": (record or {}).get("job_id"), "blocked": node.get("blocked")}


def command_status(args) -> int:
    work = Path(args.work).expanduser().resolve()
    plan = load_plan(work)
    jobs = load_jobs(work)
    records = jobs.get("nodes", {})
    scheduler = scheduler_states(sorted({r["job_id"] for r in records.values() if r.get("job_id")}))
    rows = [node_status(work, node, records.get(node["id"]), scheduler) for node in plan["nodes"]]
    collect_path = work / "collect.json"
    collected = read_json(collect_path) if collect_path.is_file() else None
    collect_row = node_status(work, collect_node(plan, []), records.get("collect"), scheduler)
    if collected is not None:
        collect_row["state"] = "COMPLETE" if collected.get("status") == "COMPLETE" else collected.get("status", "DONE")
    rows.append(collect_row)
    if args.json:
        print(json.dumps({"work": str(work), "nodes": rows, "source": "sacct" if scheduler is not None else "markers",
                          "collect": collected}, indent=1))
        return 0
    print(f"{work} ({'sacct' if scheduler is not None else 'task markers'})")
    for row in rows:
        detail = f"{row['complete']}/{row['tasks']} tasks complete" + (f", {row['failed']} failed" if row["failed"] else "")
        print(f"  {row['node']:<42} {row['state']:<14} {detail}  job {row['job_id'] or '-'}")
    if collected is not None:
        awaiting = [a for a, e in collected.get("artifacts", {}).items() if not e.get("path")]
        print(f"map: {collected.get('map')} ({collected.get('status', 'UNKNOWN').lower()}; see named artifact statuses)")
    return 0


# ============================================================================= collection
def _first(record: dict, *keys):
    for key in keys:
        value = record.get(key)
        if value:
            return value
    return None


def describe_hardware(records: list[dict]) -> str | None:
    """'<gpu>; torch <v>; python <v>' from environment records (cm_experiments.ensemble_grid.provenance_hardware)."""
    seen = set()
    for record in records:
        gpus = record.get("visible_gpus")
        gpu = record.get("gpu") or (", ".join(sorted(set(gpus))) if gpus else None)
        if not gpu:
            gpu = f"{_first(record, 'machine', 'architecture') or '?'} CPU"
        torch = record.get("torch") or (record.get("packages") or {}).get("torch") or "?"
        python = _first(record, "python", "python_version") or "?"
        seen.add(f"{gpu}; torch {torch}; python {str(python).split()[0]}")
    return " | ".join(sorted(seen)) if seen else None


def _period(stamps) -> str | None:
    stamps = sorted(s for s in stamps if isinstance(s, str) and re.match(r"\d{4}-\d\d-\d\dT", s))
    return f"{stamps[0]}/{stamps[-1]}" if stamps else None


def _run_records(runs) -> tuple[list[dict], list[str]]:
    environments, stamps = [], []
    for run in runs:
        run = Path(run)
        if (run / "environment.json").is_file():
            environments.append(read_json(run / "environment.json"))
        for manifest in (run / "provenance" / "stages").glob("*.json"):
            stamps.append(read_json(manifest).get("created_utc"))
    return environments, stamps


def _sacct_times(paths) -> list[str]:
    stamps = []
    for path in paths:
        path = Path(path)
        if not path.is_file():
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if not lines:
            continue
        header = lines[0].split("|")
        if "Start" not in header or "End" not in header:
            continue
        start, end = header.index("Start"), header.index("End")
        for line in lines[1:]:
            parts = line.split("|")
            if len(parts) == len(header):
                stamps += [parts[start], parts[end]]
    return stamps


def derive_provenance(spec: dict | None) -> tuple[str | None, str | None]:
    """Hardware and period of the executed runs behind one export, read from the runs' own records."""
    if not spec:
        return None, None
    kind = spec["kind"]
    environments, stamps = [], []
    if kind == "group_b":
        for directory in map(Path, spec["dirs"]):
            if (directory / "environment.json").is_file():
                environments.append(read_json(directory / "environment.json"))
            if (directory / "experiment.json").is_file():
                stamps.append(read_json(directory / "experiment.json").get("started_utc"))
            if (directory / "results.json").is_file():
                stamps.append(read_json(directory / "results.json").get("finished_utc"))
    elif kind == "sweep":
        plan_path = Path(spec["plan"])
        plan = read_json(plan_path)
        runs = [m for parent in plan.get("parents", []) for m in parent.get("members", [])]
        runs += [plan_path.parent / v["output"] for v in plan.get("variants", [])]
        environments, stamps = _run_records(runs)
        for job in (plan.get("settings") or {}).get("jobs") or []:   # buffer_sensitivity measurement jobs
            record = plan_path.parent / job.get("output", "") / "buffered_admission.json"
            if record.is_file():
                value = read_json(record)
                stamps.append(value.get("started_utc"))
                if isinstance(value.get("environment"), dict):
                    environments.append(value["environment"])
    elif kind == "modules":
        output = Path(spec["output"])
        plan = read_json(output / "plan.json")
        environments, stamps = _run_records(plan.get("parent_runs", []))
        for path in sorted(output.glob("jobs/**/*.json")):
            if path.name in ("lifecycle.json", "measure.json"):
                record = read_json(path)
                stamps += [record.get("started_utc"), record.get("finished_utc")]
                if isinstance(record.get("environment"), dict):
                    environments.append(record["environment"])
            elif path.name == "environment.json":
                environments.append(read_json(path))
    elif kind == "budget":
        campaign = Path(spec["campaign"])
        for path in sorted(campaign.glob("studies/**/environment.json")):
            environments.append(read_json(path))
        stamps = _sacct_times(sorted(campaign.glob("execution/**/accounting-*")))
        if not stamps:
            # studies run without a scheduler (scripts/run_budget_study.py) have no accounting: their own records
            stamps = [read_json(path).get("created_utc") for path in sorted(campaign.glob("studies/**/experiment.json"))]
    elif kind == "baselines":
        for study in map(Path, spec["studies"]):
            for invocation in study.glob("**/invocation.json"):
                record = {"python": str(read_json(invocation).get("python", "?")).split()[0]}
                evidence = _baseline_evidence(invocation)
                if evidence:
                    record.update(evidence)
                environments.append(record)
        if spec.get("internal"):        # the internal-baselines jobs (scripts/run_internal_baselines.py)
            environments += [read_json(path) for path in sorted(Path(spec["internal"]).glob("jobs/*/*/environment.json"))]
        stamps = _sacct_times([spec["accounting"]])
    return describe_hardware(environments), _period(stamps)


def _baseline_evidence(invocation: Path) -> dict:
    """GPU and torch of an E12 attempt from its driver evidence (hpc/run_upstream_baseline_arrhenius.sh)."""
    slurm = read_json(invocation).get("slurm") or {}
    job = slurm.get("SLURM_JOB_ID")
    study = invocation.parents[2] if "hpo" in invocation.parts else invocation.parents[3]
    for evidence in sorted(study.parent.glob(f"{study.name}.job-{job}-*")) if job else []:
        found = {}
        smi = evidence / "nvidia-smi.txt"
        if smi.is_file():
            match = re.search(r"Product Name\s*:\s*(.+)", smi.read_text(encoding="utf-8", errors="replace"))
            if match:
                found["gpu"] = match.group(1).strip()
        freeze = evidence / "pip-freeze.txt"
        if freeze.is_file():
            match = re.search(r"^torch==(\S+)", freeze.read_text(encoding="utf-8", errors="replace"), re.M)
            if match:
                found["torch"] = match.group(1)
        if found:
            return found
    return {}


def unscheduled(evidence: Path) -> bool:
    """A study executed without a scheduler allocation (scripts/run_budget_study.py): it has no accounting."""
    try:
        return read_json(evidence / "scheduler.json").get("scheduler") == "none"
    except (OSError, ValueError):
        return False


def finalize_budget_accounting(step: dict) -> None:
    """The terminal sacct record of every E3-E6 study job (hpc/finalize_arm_accounting.sh)."""
    for study in step["studies"]:
        evidence = budget_evidence(Path(study["study"]))
        if (evidence / "sacct.tsv").is_file() or unscheduled(evidence):
            continue
        job = (evidence / "job_id.txt").read_text(encoding="utf-8").strip()
        code = run_command(["bash", "hpc/finalize_arm_accounting.sh", job, str(evidence)])
        if code != 0:
            raise PlanError(f"{evidence}: hpc/finalize_arm_accounting.sh exited with {code}")


def budget_evidence(study: Path) -> Path:
    """The job-evidence directory of the study's completed execution (<study>.job-<id>, exit status 0)."""
    found = []
    for evidence in sorted(study.parent.glob(f"{study.name}.job-*")):
        code = evidence / "process_exit_code.txt"
        if code.is_file() and code.read_text(encoding="utf-8").strip() == "0":
            found.append(evidence)
    if len(found) != 1:
        raise PlanError(f"{study}: expected one completed job-evidence directory, found {len(found)}")
    return found[0]


def write_budget_manifest(step: dict, repository: dict) -> None:
    """The cm-arm-campaign-1 manifest of the executed studies (scripts/arm_campaign.py template fields)."""
    from scripts._release_common import inventory
    manifest = Path(step["manifest"])
    if manifest.is_file():
        return
    with tempfile.TemporaryDirectory() as scratch:
        template = Path(scratch) / "template.json"
        code = run_command([sys.executable, "-B", "scripts/arm_campaign.py", "template", "--output", str(template)])
        if code != 0:
            raise PlanError("scripts/arm_campaign.py template failed")
        value = read_json(template)
    commit = repository.get("git_commit")
    if not commit or repository.get("dirty"):
        raise PlanError("the E3-E6 campaign names the source commit that executed it; run from a clean git checkout")
    rows = []
    for study in step["studies"]:
        evidence = budget_evidence(Path(study["study"]))
        logs = [evidence / "stdout.log", evidence / "stderr.log"]
        if unscheduled(evidence):
            # outside a scheduler an empty stream is not written over: there is nothing to attach
            logs = [log for log in logs if log.is_file() and log.stat().st_size]
        rows.append({"dataset": study["dataset"], "seed": int(study["seed"]), "split": int(study["split"]),
                     "parent": study["parent"], "study": study["study"],
                     "accounting": [] if unscheduled(evidence) else [str(evidence / "sacct.tsv")],
                     "job_log": [str(log) for log in logs]})
    value.update(seed_ids=[int(s) for s in step["seeds"]], studies=rows,
                 source={"repository": "https://github.com/VKHS/CROSS-MORSE", "git_commit": commit,
                         "tree_sha256": inventory(ROOT)["tree_sha256"]})
    value.pop("instructions", None)
    write_json(manifest, value)


def check_accepted(step: dict) -> None:
    """The assembled S14 campaign is the one check_campaign accepted (scripts/build_s14_campaign.py assemble)."""
    path = Path(step["file"])
    verdict = read_json(path) if path.is_file() else {}
    if verdict.get("accepted") is not True:
        raise PlanError(f"{path}: check_campaign did not accept the assembled campaign "
                        f"({verdict.get('status')}: {verdict.get('error')})")
    if Path(verdict.get("campaign", "")).resolve() != Path(step["campaign"]).resolve():
        raise PlanError(f"{path}: the verdict belongs to another campaign directory")


INTERNAL_STEPS = {"finalize_budget_accounting": lambda step, plan: finalize_budget_accounting(step),
                  "budget_manifest": lambda step, plan: write_budget_manifest(step, plan.get("repository") or {}),
                  "accepted": lambda step, plan: check_accepted(step)}


def verify_record_set(path: Path, *, allow_fixture: bool) -> dict:
    from scripts.records import load_specs, verify_set
    specs = load_specs(ROOT / "records", ROOT / "records" / "specs.json")
    if path.name not in specs["sets"]:
        raise PlanError(f"{path.name} is not a record set of records/specs.json")
    entry = verify_set(path, specs["sets"][path.name], specs)
    if entry.get("synthetic") and not allow_fixture:
        raise PlanError(f"{path.name}: a synthetic record set is not a paper artifact")
    provenance = read_json(path / "provenance.json")
    # A fixture plan (a software check or the CPU execution) may hold a set whose own provenance declares
    # partial coverage, e.g. timings without a peak-memory reading on CPU; a paper artifact never does.
    if provenance.get("coverage_complete") is False and not allow_fixture:
        raise PlanError(f"{path.name}: partial coverage (provenance coverage_complete = false)")
    if not allow_fixture:
        from scripts.paper_artifacts import _verify_source
        _verify_source("records:" + path.name, path, scientific_only=True)
    return entry


def verify_artifact(artifact: str, path: Path, *, allow_fixture: bool) -> dict:
    kind = artifact.split(":", 1)[0]
    if not path.is_dir() or not any(path.iterdir()):
        raise PlanError(f"{artifact}: {path} is missing or empty")
    if kind == "records":
        return verify_record_set(path, allow_fixture=allow_fixture)
    from scripts.paper_artifacts import _verify_source
    return _verify_source(artifact, path, scientific_only=not allow_fixture)


def _outputs_ready(export: dict, work: Path, plan: dict, *, allow_fixture: bool) -> bool:
    for artifact, path in export["outputs"].items():
        try:
            verify_artifact(artifact, Path(path), allow_fixture=allow_fixture)
        except (PlanError, OSError, ValueError, KeyError) as exc:   # scripts.records.RecordsError is a ValueError
            del exc
            return False
    return bool(export["outputs"])


def run_export(export: dict, work: Path, plan: dict, jobs: dict, *, hardware=None, allow_fixture=False,
               ignore_markers=False) -> dict:
    nodes = {n["id"]: n for n in plan["nodes"]}
    if _outputs_ready(export, work, plan, allow_fixture=allow_fixture):
        return {"status": "COMPLETE", "reason": "outputs present and verified"}
    unfinished = [n for n in export["after"] if n not in nodes or not node_complete(work, nodes[n])]
    if unfinished and not ignore_markers:
        return {"status": "AWAITING", "reason": f"nodes not complete: {', '.join(unfinished)}"}
    values = {"python": sys.executable}
    for node, record in (jobs.get("nodes") or {}).items():
        values[f"jobs:{node}"] = ",".join(record.get("history") or [record.get("job_id")])
    for step in export["steps"]:
        if step.get("skip_if") and Path(step["skip_if"]).exists():
            continue
        if "internal" in step:
            INTERNAL_STEPS[step["internal"]](step, plan)
            continue
        if step.get("fresh_directory"):
            directory = Path(step["fresh_directory"])
            if directory.exists():
                _move_aside(directory)
        if any(token in " ".join(step["argv"]) for token in ("{hardware}", "{period}")) and "period" not in values:
            # read after the earlier steps, which may have written the records (the E3-E6 campaign export)
            derived_hardware, period = derive_provenance(export.get("provenance"))
            if hardware or derived_hardware:
                values["hardware"] = hardware or derived_hardware
            if period:
                values["period"] = period
        try:
            argv = expand(step["argv"], values)
        except PlanError as exc:
            hint = ("the nodes' job ids are not in jobs.json (submitted outside scripts/paper_outputs.py submit)"
                    if "{jobs:" in str(exc) else "the runs' records name no hardware or period (collect "
                                                 "--hardware TEXT supplies the hardware)")
            return {"status": "FAILED", "reason": f"{exc}: {hint}"}
        code = run_command(argv)
        if code != 0 and not step.get("advisory"):
            return {"status": "FAILED", "reason": f"{' '.join(argv[2:5])} exited with {code}"}
    for artifact, path in export["outputs"].items():
        try:
            verify_artifact(artifact, Path(path), allow_fixture=allow_fixture)
        except (PlanError, OSError, ValueError, KeyError) as exc:
            return {"status": "FAILED", "reason": f"{artifact}: {exc}"}
    return {"status": "COMPLETE"}


def publication_map(plan: dict, results: dict, work: Path) -> dict:
    """{artifact id: directory} for scripts/paper_artifacts.py pack; awaiting artifacts carry an empty path."""
    mapping = {"_instructions": "Generated by scripts/paper_outputs.py collect. Publish with: bash "
                                "hpc/publish_paper_artifacts_arrhenius.sh all " + str(work / MAP_NAME)
                                + ". An empty path is an artifact still awaiting its measured outputs (_awaiting).",
               "_generated_utc": utc_now(), "_plan_sha256": sha256_file(work / "plan.json"), "_awaiting": {},
               "_relationships": {}}
    for artifact, entry in plan["artifacts"].items():
        path, reason = "", entry.get("reason")
        if entry["status"] in ("run", "provided"):
            path = entry["path"]
        elif entry["status"] == "planned":
            outcome = results.get(entry.get("export"), {})
            if outcome.get("status") == "COMPLETE":
                path = entry["path"]
            else:
                reason = outcome.get("reason") or "the export has not run"
        if path:
            mapping[artifact] = path
            mapping["_relationships"][artifact] = "post_result_revision"
        else:
            mapping[artifact] = ""
            mapping["_awaiting"][artifact] = reason or entry["status"]
    if (plan.get("options") or {}).get("allow_fixture"):
        mapping["_fixture"] = ("planned with --allow-fixture: synthetic or smoke inputs; "
                               "scripts/paper_artifacts.py pack refuses them")
    return mapping


def command_collect(args) -> int:
    with work_lock(Path(args.work).expanduser().resolve(), "collect"):
        return _command_collect(args)


def _command_collect(args) -> int:
    work = Path(args.work).expanduser().resolve()
    plan = load_plan(work)
    verify_plan_source(plan)
    jobs = load_jobs(work)
    if jobs.get("nodes") and jobs.get("plan_sha256") != sha256_file(work / "plan.json"):
        raise PlanError("jobs.json does not belong to this production plan")
    allow_fixture = bool((plan.get("options") or {}).get("allow_fixture"))
    (work / "records").mkdir(parents=True, exist_ok=True)
    only = _normalize_only(args.only)
    results, failures = {}, []
    for export in plan["exports"]:
        if only and not set(export["artifacts"]) & only:
            ready = _outputs_ready(export, work, plan, allow_fixture=allow_fixture)
            results[export["id"]] = {"status": "COMPLETE"} if ready else {"status": "AWAITING",
                                                                          "reason": "not collected (--only)"}
            continue
        try:
            outcome = run_export(export, work, plan, jobs, hardware=args.hardware, allow_fixture=allow_fixture,
                                 ignore_markers=args.ignore_markers)
        except (PlanError, OSError, ValueError, KeyError) as exc:
            outcome = {"status": "FAILED", "reason": str(exc)}
        results[export["id"]] = outcome
        print(f"[{outcome['status']}] {export['id']}" + (f": {outcome['reason']}" if outcome.get("reason") else ""),
              flush=True)
        if outcome["status"] == "FAILED":
            failures.append(export["id"])
    for artifact, entry in plan["artifacts"].items():
        if entry["status"] in ("run", "provided") and entry.get("path"):
            try:
                verify_artifact(artifact, Path(entry["path"]), allow_fixture=allow_fixture)
            except (PlanError, OSError, ValueError, KeyError) as exc:
                failures.append(artifact)
                print(f"[FAILED] {artifact}: {exc}", flush=True)
    mapping = publication_map(plan, results, work)
    for artifact in failures:
        if artifact in plan["artifacts"]:
            mapping[artifact] = ""
            mapping["_awaiting"][artifact] = "the supplied directory does not verify"
    write_json(work / MAP_NAME, mapping)
    artifacts = {a: {"path": mapping[a], "reason": mapping["_awaiting"].get(a)} for a in plan["artifacts"]}
    status = "COMPLETE" if not failures and not mapping["_awaiting"] else "INCOMPLETE"
    write_json(work / "collect.json", {"schema": COLLECT_SCHEMA, "collected_utc": utc_now(), "status": status,
                                       "exports": results, "artifacts": artifacts, "map": str(work / MAP_NAME)})
    print(f"map: {work / MAP_NAME} ({status.lower()}; named artifacts and verification status are recorded in collect.json)")
    print(f"publish: bash hpc/publish_paper_artifacts_arrhenius.sh all {work / MAP_NAME}")
    return 0 if not failures and (not mapping["_awaiting"] or only or (plan.get("options") or {}).get("only")) else 1


# ============================================================================= command line
def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = top.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan", help="discover and verify the runs, plan every node, write <work>/plan.json")
    p.add_argument("--runs", type=Path, default=None,
                   help="reproduction suite (runs/NN_<dataset>; Roman-empire: the campaign of ten split runs), its "
                        "runs/ directory, or a directory holding one run per dataset")
    p.add_argument("--run", action="append", default=[], metavar="DATASET=DIR",
                   help="the sealed run (Roman-empire: campaign directory) of one dataset; overrides --runs")
    p.add_argument("--work", type=Path, required=True, help="new work directory outside the checkout")
    p.add_argument("--budget-campaign", type=Path, default=None,
                   help="complete E3-E6 campaign export (scripts/arm_campaign.py export --complete) to publish "
                        "instead of executing the campaign")
    p.add_argument("--confirmation-campaign", type=Path, default=None,
                   help="assembled joint S14 campaign to publish and to analyse in E10 instead of building it")
    p.add_argument("--baselines", type=Path, default=None,
                   help="directory of the E12 pinned-upstream study plans (cm-upstream-study-1 JSON with their "
                        "upstream and adapter checkouts beside them, as the plans name them; collect archives each "
                        "adapter at its pinned commit from there), optional timing/*.csv (E17 rows of the external "
                        "methods) and accounting/*.psv; the internal rows are measured on the runs")
    p.add_argument("--environment", type=Path, default=None, help="existing GPU environment capture")
    p.add_argument("--artifact", action="append", default=[], metavar="ID=DIR",
                   help="any artifact produced elsewhere, e.g. records:buffer_sensitivity=/abs/dir (verified at collect)")
    p.add_argument("--only", nargs="+", default=None, metavar="SET",
                   help="plan only these artifacts (record-set names or artifact ids) and the nodes they need")
    p.add_argument("--data-root", type=Path, default=None,
                   help="dataset cache of run_cross_morse.py (default: the suite's, else CM_DATA_ROOT in the jobs)")
    p.add_argument("--negative-controls-dataset", default=DEFAULT_NEGATIVE_CONTROLS_DATASET,
                   help="dataset of the Table S19 negative controls and the Figure S16d stress reference "
                        f"(declared; default {DEFAULT_NEGATIVE_CONTROLS_DATASET})")
    p.add_argument("--band-audit-dataset", default=DEFAULT_BAND_AUDIT_DATASET,
                   help=f"plant of the Figure S18c bound tightness (declared; default {DEFAULT_BAND_AUDIT_DATASET})")
    p.add_argument("--band-audit-baseline", default=DEFAULT_BAND_AUDIT_BASELINE,
                   choices=("shared", "native_mean", "inserted_identity"))
    p.add_argument("--arm-seed", type=int, action="append", default=None,
                   help="E3-E6 controller-optimization seed (twelve; default 0..11, the campaign template's)")
    p.add_argument("--provisional", action="append", default=[], metavar="DATASET",
                   help="audit the provisional selection q_s for action_gap_audit on this dataset (a second audit)")
    p.add_argument("--allow-fixture", action="store_true",
                   help="software check only: accept synthetic or smoke runs and pass the runners' fixture flags; "
                        "the outputs can never be published")
    p.add_argument("--allow-subset", action="store_true",
                   help="plan experiments on the runs present when some datasets are missing (outputs are partial)")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                   help="device of every runner command; cpu is the CPU execution (requires --allow-fixture)")
    p.add_argument("--s14-exclude", action="append", default=[], metavar="DATASET",
                   help="leave a dataset out of the joint S14 family (requires --allow-fixture; the family is partial)")
    p.add_argument("--horizon-sweep-profile", default=None,
                   help="profile of the swept MPC-RH configuration of Table S29 (declared; default: the deployed rule)")
    p.add_argument("--horizon-sweep-ensemble", type=int, default=None,
                   help="ensemble size of the swept MPC-RH configuration (declared; default: the deployed rule)")
    p.add_argument("--horizon-sweep-beam", type=int, default=None,
                   help="beam width the Table S29 horizon sweep holds when the parent was planned with another "
                        "(declared; requires --allow-fixture; every point is then a variant lifecycle)")
    p = sub.add_parser("submit", help="write the batch scripts and submit the job graph with afterok dependencies")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true", help="print the sbatch commands; submit nothing")
    p.add_argument("--resume", action="store_true", help="submit only the nodes whose tasks are not all complete")
    p = sub.add_parser("status", help="state of every node")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("collect", help="run every export into <work>/records and write the publication map")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--hardware", default=None,
                   help="hardware text for exports whose runs carry no environment record (normally derived)")
    p.add_argument("--only", nargs="+", default=None, metavar="SET")
    p.add_argument("--ignore-markers", action="store_true",
                   help="attempt every export even when the task markers do not show its nodes complete (nodes "
                        "executed outside this workflow); the exporters still refuse incomplete measurements")
    p = sub.add_parser("execute", help="(inside a Slurm job) one task of one node")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--node", required=True)
    p.add_argument("--task", type=int, default=None, help="default: SLURM_ARRAY_TASK_ID, else 0")
    return top


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        return {"plan": command_plan, "submit": command_submit, "status": command_status,
                "collect": command_collect, "execute": command_execute}[args.command](args)
    except PlanError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
