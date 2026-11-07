"""E2 sequence predictability, and the runner core shared by experiment group B.

Part A (runner core). Experiment group B is E2 (this module), E9
(``search_enumeration``), E18 (``band_audit``), the Table S19 negative controls
(``negative_controls``) and the serving-time input gate (``input_gate``). All
five use the plan/run/export helpers below: a declared plan names one sealed
parent run by its identities; ``run`` executes the released runtime in the plan
directory and seals every output with its SHA-256; ``export`` re-derives every
summary value from the sealed raw files and writes a record set in the layout
that ``scripts/records.py`` verifies (summary.csv, raw/, code/, provenance.json,
MANIFEST.json). This module and ``scripts/run_experiment_b.py`` are part of the
code identity of every group-B record. The parent run is read, never written.

Part B (E2, Supplementary Table S36). The initial-input predictor of the
feedback policy's executed sequences is fitted on D_explore and evaluated on
the reserved D_compare against the same frozen teacher q_M and comparator q_C.
The predictor is the E4 distillation recipe of ``cm_experiments.schedules``
(``fit_decoder(arm='E4')`` on the teacher's executed sequences with their
terminal depths, dynamic A9 rewritten to source-reset A8, the same
capacity-matched decoder, the same support-masked open-loop execution of
``cm_experiments.execution.execute_arm``). Two readouts are kept apart:

* ``phi_seq`` is the share of the teacher's executed (input, member) sequences
  that the decoded sequence reproduces exactly, length included, against the
  A9->A8 normalized teacher sequence (the decoder's target semantics);
* ``phi_gain = [R(q_C) - R(q_pred)] / [R(q_C) - R(q_M)]`` is the paired share
  of the feedback gain retained by executing the predicted sequences, with all
  risks on the same D_compare units (Note S25). It is undefined, and not
  reported, when the measured feedback gain is not positive.

On a single transductive graph the predictor is fitted and evaluated on the
same input (different label rows) and the teacher is deterministic there, so
these readouts do not measure unseen-input generalization; every result says so.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .artifacts import canonical, fresh_output, seal, sha256, verify, write_json

ROOT = Path(__file__).resolve().parents[1]
PLAN_SCHEMA = "cm-experiment-b-plan-1"
RUN_SCHEMA = "cm-experiment-b-run-1"
RESULTS_SCHEMA = "cm-experiment-b-results-1"
SUMMARY_SCHEMA = "cm-experiment-b-summary-1"
LIFECYCLE = ("prepare", "train", "explore", "fit", "select", "confirm", "admit", "evaluate")
FEEDBACK_ROUTES = ("one_step", "radius", "mpc_rh")
# Label scope (cross_morse.data.STAGE_LABEL_ROLES) and the parent stage that
# must be sealed before an out-of-lifecycle experiment may read the role.
ROLE_SCOPE = {"explore": "explore", "select": "select", "confirm": "confirm",
              "compare": "compare", "admit": "admit", "test": "evaluate"}
ROLE_REQUIRES = {"explore": "explore", "select": "select", "confirm": "confirm",
                 "compare": "confirm", "admit": "admit", "test": "evaluate"}
EXPERIMENTS = {
    "predictability": {"module": "cm_experiments/predictability.py", "record_set": "predictability",
                       "title": "E2 predictability of the feedback policy's executed sequences"},
    "search_enumeration": {"module": "cm_experiments/search_enumeration.py", "record_set": "search_enumeration",
                           "title": "E9 search versus model error by exact enumeration at H <= 4"},
    "band_audit": {"module": "cm_experiments/band_audit.py", "record_set": "band_power_audit",
                   "title": "E18 spectral displacement audit of main Proposition 6"},
    "negative_controls": {"module": "cm_experiments/negative_controls.py", "record_set": "negative_controls",
                          "title": "Negative controls and unsupported regimes (Supplementary Table S19)"},
    "input_gate": {"module": "cm_experiments/input_gate.py", "record_set": "input_gate",
                   "title": "Serving-time input gate g_input and deployed latency (Supplementary Table S23)"},
}
CORE_FILES = ("cm_experiments/predictability.py", "scripts/run_experiment_b.py")
SUMMARY_TAIL = ["quantity", "value", "unit", "source_file", "locator"]


class ExperimentError(ValueError):
    """A plan, parent run or sealed output cannot support the declared experiment."""


# ============================================================================ Part A: runner core
def read_json(path):
    path = Path(path)
    if not path.is_file():
        raise ExperimentError(f"missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def finite(value, where):
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ExperimentError(f"{where}: value is missing or not a number")
    value = float(value)
    if not math.isfinite(value):
        raise ExperimentError(f"{where}: value is not finite")
    return value


def quantile(values, level):
    """Empirical upper quantile (numpy method 'higher'); None without finite values."""
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return float(np.quantile(values, min(1.0, max(0.0, float(level))), method="higher"))


def median(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else None


def safe_label(text):
    label = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")
    if not label:
        raise ExperimentError(f"cannot form a file name from {text!r}")
    return label


def code_identity():
    """SHA-256 of the experiment code and the reused cm_experiments modules at run time."""
    paths = sorted(ROOT.glob("cm_experiments/*.py")) + [ROOT / "scripts/run_experiment_b.py"]
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths if p.is_file()}


def stage_manifest(root, stage):
    from cross_morse.provenance import stage_manifest_path
    return stage_manifest_path(root, stage)


class ParentRun:
    """Read-only view of one sealed lifecycle run; the run directory is never written."""

    def __init__(self, path):
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            raise ExperimentError(f"parent run is not a directory: {path}")
        self.root = root
        self.config = read_json(root / "config.json")
        self.state = read_json(root / "run_state.json")

    @property
    def name(self):
        return self.root.name

    @property
    def dataset(self):
        return self.config.get("benchmark_dataset") or self.config["dataset"]

    @property
    def synthetic(self):
        return self.config.get("dataset") == "synthetic" or self.config.get("scientific_run") is False

    @property
    def depth(self):
        return int(self.config["model"]["depth"])

    @property
    def seeds(self):
        return [int(s) for s in self.config["training"]["seeds"]]

    def json(self, relative):
        return read_json(self.root / relative)

    def sealed(self):
        stages = []
        for stage in LIFECYCLE:
            marker = self.root / "stages" / f"{stage}.json"
            if not (stage_manifest(self.root, stage).is_file() and marker.is_file()
                    and read_json(marker).get("status") == "COMPLETE"):
                break
            stages.append(stage)
        return stages

    def require(self, stage):
        if stage not in self.sealed():
            raise ExperimentError(f"{self.name}: lifecycle stage {stage!r} is not sealed")

    def check(self, through):
        """Planning check: sealed stage and the parent's runtime source equal to this checkout's.

        The complete chain (artifact hashes and frozen checkpoints) needs
        PyTorch; it is verified here when PyTorch is installed and always by
        ``verify`` before and after execution.
        """
        import importlib.util
        from cross_morse.provenance import current_source_sha256
        self.require(through)
        if self.state.get("source_sha256") != current_source_sha256():
            raise ExperimentError(f"{self.name}: the parent was produced by another runtime source; "
                                  "execute the experiment from the parent's own source snapshot")
        if importlib.util.find_spec("torch") is None:
            return {"chain_verified": False, "through": through,
                    "reason": "PyTorch unavailable at planning; the complete chain is verified by run"}
        self.verify(through)
        return {"chain_verified": True, "through": through}

    def verify(self, through):
        """Verify the sealed chain through ``through`` against the current runtime source."""
        from cross_morse.provenance import current_source_sha256, verify_stage_chain
        self.require(through)
        try:
            return verify_stage_chain(self.root, through, expected_source_sha256=current_source_sha256())
        except Exception as exc:  # ProvenanceError and malformed manifests alike
            raise ExperimentError(f"{self.name}: sealed lifecycle verification through {through} failed "
                                  f"(the experiment must execute the parent's own runtime source): {exc}") from exc

    def identity(self):
        sealed = self.sealed()
        return {"name": self.name, "dataset": self.dataset, "input_dataset": self.config.get("dataset"),
                "profile": self.config.get("profile"), "scientific_run": bool(self.config.get("scientific_run")),
                "synthetic": self.synthetic, "config_sha256": canonical(self.config),
                "run_state_config_sha256": self.state.get("config_sha256"),
                "source_sha256": self.state.get("source_sha256"), "sealed_stages": sealed,
                "stage_manifest_sha256": {s: sha256(stage_manifest(self.root, s)) for s in sealed},
                "training_seeds": self.seeds, "depth": self.depth,
                "split_index": self.config.get("data", {}).get("split_index")}

    def require_role(self, role):
        if role not in ROLE_REQUIRES:
            raise ExperimentError(f"unknown data role {role!r}; expected one of {sorted(ROLE_REQUIRES)}")
        self.require(ROLE_REQUIRES[role])

    def candidates(self):
        return self.json("select/candidates.json")

    def policies(self, *, allow_fixture=False):
        """Confirmed teacher q_M and locked comparator q_C, as the E3-E6 runner reads them."""
        candidates = self.candidates()
        decision = self.json("confirm/decision.json")
        teacher = copy.deepcopy(candidates[decision["confirmed_index"]])
        comparator = copy.deepcopy(candidates[decision["comparator_index"]])
        override = False
        if teacher["route"] not in FEEDBACK_ROUTES:
            if not (allow_fixture and self.synthetic):
                raise ExperimentError("the confirmed policy is open-loop; a feedback teacher is required "
                                      "(a synthetic fixture may declare --allow-fixture)")
            options = [c for c in candidates if c["route"] == "mpc_rh" and c["ensemble"] == comparator["ensemble"]]
            if not options:
                raise ExperimentError("the fixture registers no MPC-RH candidate at the comparator's ensemble size")
            teacher = copy.deepcopy(options[0])
            override = True
        return teacher, comparator, {"confirmed_index": decision["confirmed_index"],
                                     "comparator_index": decision["comparator_index"],
                                     "fixture_teacher_override": override}

    def admission(self):
        """Global admission G_admit and the deployed index when the admit stage is sealed."""
        if "admit" not in self.sealed():
            return None
        decision = self.json("admit/decision.json")
        return {"passed": bool(decision.get("passed")), "deployed_index": decision.get("deployed_index"),
                "comparator_index": decision.get("comparator_index"),
                "confirmed_index": decision.get("confirmed_index")}


def protocol_scope(parent, *, allow_posthoc):
    later = any(stage in parent.sealed() for stage in ("admit", "evaluate"))
    if later and not allow_posthoc and not parent.synthetic:
        raise ExperimentError("the parent has admission/test decisions; use a confirmed pre-admission parent "
                              "or declare --allow-posthoc")
    return "posthoc" if later else "pre_admission"


def create_plan(experiment, parent, output, body):
    """Write ``<output>/plan.json``; the output directory must be new and outside the parent."""
    if experiment not in EXPERIMENTS:
        raise ExperimentError(f"unknown experiment {experiment!r}")
    out = fresh_output(output, parent.root)
    plan = {"schema": PLAN_SCHEMA, "experiment": experiment, "record_set": EXPERIMENTS[experiment]["record_set"],
            "title": EXPERIMENTS[experiment]["title"], "created_utc": utc_now(),
            "parent": dict(parent.identity(), path=str(parent.root)), **body}
    write_json(out / "plan.json", json.loads(json.dumps(plan, allow_nan=False)))
    return out / "plan.json"


def open_plan(plan_path, experiment):
    """Load a plan and reject a parent whose sealed identity changed after planning."""
    plan_path = Path(plan_path).resolve()
    plan = read_json(plan_path)
    if plan.get("schema") != PLAN_SCHEMA or plan.get("experiment") != experiment:
        raise ExperimentError(f"{plan_path} is not a {experiment} plan of schema {PLAN_SCHEMA}")
    declared = plan["parent"]
    parent = ParentRun(declared["path"])
    now = parent.identity()
    for key in ("config_sha256", "source_sha256", "dataset"):
        if now[key] != declared[key]:
            raise ExperimentError(f"parent {key} changed since planning")
    for stage, digest in declared["stage_manifest_sha256"].items():
        if now["stage_manifest_sha256"].get(stage) != digest:
            raise ExperimentError(f"parent stage manifest {stage} changed since planning")
    return plan, parent


def _environment(device):
    import importlib.metadata
    import torch
    return {"schema": "cm-experiment-b-environment-1", "python_version": platform.python_version(),
            "platform": platform.platform(), "machine": platform.machine(), "device_requested": str(device),
            "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(), "torch_threads": torch.get_num_threads(),
            "visible_gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "numpy": np.__version__,
            "packages": dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions()
                                    if d.metadata.get("Name"))),
            "slurm": {k: os.environ[k] for k in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID")
                      if k in os.environ},
            "scope": "captured for this invocation, not the parent environment"}


class RunDirectory:
    """The plan directory receives every raw output, then one artifact manifest."""

    def __init__(self, plan_path, plan, parent, *, device, trust):
        if not trust:
            raise ExperimentError("explicit trust is required before loading the parent's local checkpoints")
        self.root = Path(plan_path).resolve().parent
        present = sorted(p.name for p in self.root.iterdir() if p.name != "plan.json")
        if present:
            raise ExperimentError(f"{self.root} already holds {present}; plan a new directory for a new execution")
        if parent.config["model"].get("precision") == "bf16" and not str(device).startswith("cuda"):
            raise ExperimentError("the bf16 plant executes on CUDA only; no implicit CPU replacement")
        import torch
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise ExperimentError("CUDA requested but unavailable")
        from cross_morse.provenance import current_source_sha256
        self.plan, self.parent, self.device = plan, parent, str(device)
        (self.root / "raw").mkdir()
        write_json(self.root / "base_config.json", parent.config)
        write_json(self.root / "environment.json", _environment(device))
        write_json(self.root / "experiment.json", {
            "schema": RUN_SCHEMA, "experiment": plan["experiment"], "record_set": plan["record_set"],
            "plan_sha256": sha256(self.root / "plan.json"), "parent": parent.identity(),
            "runtime_source_sha256": current_source_sha256(), "experiment_code": code_identity(),
            "device": self.device, "started_utc": utc_now(),
            "labels_read_by": "cm_experiments group-B runner inside the declared lifecycle label scope only",
            "parent_written": False})

    def raw(self, relative):
        path = self.root / "raw" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise ExperimentError(f"raw output already exists: {relative}")
        return path

    def finish(self, summary):
        write_json(self.root / "results.json", {"schema": RESULTS_SCHEMA, "experiment": self.plan["experiment"],
                                                "record_set": self.plan["record_set"], "finished_utc": utc_now(),
                                                **summary})
        seal(self.root)
        return self.root


def write_csv(path, rows, fields=None):
    if not rows:
        raise ExperimentError(f"no rows for {Path(path).name}")
    fields = list(fields or rows[0])
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summary_row(keys, quantity, value, unit, definition, **extra):
    return {"keys": {k: str(v) for k, v in keys.items()}, "quantity": quantity,
            "value": finite(value, f"{keys}/{quantity}"), "unit": unit, "definition": definition, **extra}


def role_labels(data, episode, role, ids):
    """Labels of ``ids`` (episode-local rows of ``role``) inside the role's lifecycle label scope."""
    allowed = np.asarray(episode.splits[role], dtype=np.int64)
    lookup = {int(n): i for i, n in enumerate(allowed)}
    ids = [int(i) for i in ids]
    if len(set(ids)) != len(ids) or not set(ids) <= set(lookup):
        raise ExperimentError("requested labels lie outside the permitted role")
    with data._label_scope(ROLE_SCOPE[role]):
        labels = episode.labels(role)
    return labels[np.asarray([lookup[i] for i in ids], dtype=np.int64)]


def role_units(data, role, max_graphs=0):
    """(unit id, episode) pairs of a role: complete graphs, or the one transductive graph."""
    from cross_morse.runtime import is_graph_collection
    if is_graph_collection(data):
        ids = sorted(int(i) for i in data.graph_splits[role])
        if max_graphs:
            ids = ids[:int(max_graphs)]
        return [(i, data.graph(i)) for i in ids]
    return [(0, data)]


def compare_rows(saved, recomputed, where):
    """A sealed summary must equal the summary recomputed from its raw files."""
    def key(row):
        return (tuple(sorted(row["keys"].items())), row["quantity"])
    a = {key(r): r for r in saved}
    b = {key(r): r for r in recomputed}
    if set(a) != set(b):
        raise ExperimentError(f"{where}: sealed summary rows differ from the rows recomputed from raw outputs")
    for k, row in a.items():
        x, y = float(row["value"]), float(b[k]["value"])
        if not math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-15):
            raise ExperimentError(f"{where}: {k} sealed {x!r} but recomputed {y!r} from the raw outputs")


def load_contract(record_set):
    """The record contract of records/specs.json, the only contract location."""
    from scripts.records import RECORDS, load_specs
    specs = load_specs(RECORDS, RECORDS / "specs.json")
    if record_set not in specs["sets"]:
        raise ExperimentError(f"{record_set} is not a record set of records/specs.json")
    return specs, specs["sets"][record_set]


def _public_plan(plan):
    """The plan as deposited: every declared filesystem location reduced to its final name."""
    def strip(value):
        if isinstance(value, dict):
            return {k: (Path(v).name if k == "path" and isinstance(v, str) else strip(v)) for k, v in value.items()}
        if isinstance(value, list):
            return [strip(v) for v in value]
        return value
    return strip(copy.deepcopy(plan))


def _private_findings(target):
    from scripts.release_audit import PRIVATE_PATTERNS
    findings = []
    for path in sorted(p for p in target.rglob("*") if p.is_file()):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for pattern in PRIVATE_PATTERNS:
            if pattern.search(text):
                findings.append(path.relative_to(target).as_posix())
    return findings


def export_record_set(experiment, results_dirs, *, summarize, raw_exports, hardware, period, protocol,
                      records_root=None, replace=False, allow_private=False, allow_truncated=False, argv=None):
    """Write the experiment's record set from sealed run directories and verify it.

    Each directory's sealed results are first recomputed from its raw outputs.
    The exported values are recomputed over all directories (pooling repeated
    splits of one dataset) and written to raw/summary.json with the SHA-256 of
    every input; summary.csv points into that file. Only quantities named by
    the record contract (records/specs.json) are exported; others stay in
    raw/summary.json. The verified set is installed with its index entry.
    """
    from scripts.export_records import fmt, git_head
    from scripts.records import RECORDS, is_official_root, verify_set, write_manifest
    from scripts._paper_common import runtime_source_fingerprint
    if not str(hardware or "").strip() or not str(period or "").strip():
        raise ExperimentError("pass the hardware and period of the executed runs explicitly")
    dirs = [Path(d).resolve() for d in results_dirs]
    if not dirs or len(set(dirs)) != len(dirs):
        raise ExperimentError("give one or more distinct sealed result directories")
    record_set = EXPERIMENTS[experiment]["record_set"]
    specs, spec = load_contract(record_set)
    contract_path = "records/specs.json"
    loaded = []
    for d in dirs:
        verify(d)
        exp = read_json(d / "experiment.json")
        results = read_json(d / "results.json")
        plan = read_json(d / "plan.json")
        if exp.get("schema") != RUN_SCHEMA or exp.get("experiment") != experiment or results.get("experiment") != experiment:
            raise ExperimentError(f"{d.name} is not a sealed {experiment} run")
        compare_rows(results["rows"], summarize([d])["rows"], d.name)
        loaded.append((d, exp, results, plan))
    summary = summarize(dirs)
    rows = [r for r in summary["rows"] if r["quantity"] in spec["quantities"]]
    other = [r for r in summary["rows"] if r["quantity"] not in spec["quantities"]]
    if not rows:
        raise ExperimentError(f"no measured value of the {record_set} contract to export")
    for row in rows:
        if set(row["keys"]) != set(spec["keys"]):
            raise ExperimentError(f"row keys {sorted(row['keys'])} differ from the contract keys {spec['keys']}")
    records_root = Path(records_root or RECORDS).resolve()
    official = is_official_root(records_root)
    synthetic = any(exp["parent"]["synthetic"] for _, exp, _, _ in loaded)
    truncated = sorted({d.name for d, _, results, _ in loaded if results.get("truncated")})
    if official and synthetic:
        raise ExperimentError("synthetic fixture runs cannot be deposited under records/; use another --root")
    if official and truncated and not allow_truncated:
        raise ExperimentError(f"truncated runs {truncated} cannot be deposited under records/")
    records_root.mkdir(parents=True, exist_ok=True)
    staging_parent = ROOT.parent if records_root.is_relative_to(ROOT) else records_root.parent
    staging = Path(tempfile.mkdtemp(prefix=f"cm-record-{record_set}-", dir=staging_parent))
    try:
        target = staging / record_set
        (target / "raw").mkdir(parents=True)
        (target / "code").mkdir()
        inputs, labels = [], set()
        for d, exp, results, plan in loaded:
            label = safe_label(f"{exp['parent']['dataset']}-{exp['parent']['name']}")
            if label in labels:
                raise ExperimentError(f"two result directories share the label {label}")
            labels.add(label)
            manifest = read_json(d / "ARTIFACT_MANIFEST.json")
            copied = {"results.json", "experiment.json", *raw_exports}
            for relative in sorted(copied):
                source = d / relative
                if not source.is_file():
                    raise ExperimentError(f"{d.name} lacks {relative}")
                inner = Path(relative)
                destination = target / "raw" / label / (inner.relative_to("raw") if inner.parts[0] == "raw" else inner)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
            write_json(target / "code" / label / "plan.json", _public_plan(plan))
            inputs.append({"run": label, "plan_sha256": sha256(d / "plan.json"),
                           "artifact_manifest_sha256": sha256(d / "ARTIFACT_MANIFEST.json"),
                           "files": [{"path": f["path"], "sha256": f["sha256"], "bytes": f["bytes"],
                                      "copied": f["path"] in copied} for f in manifest["files"]]})
        write_json(target / "raw" / "summary.json", {
            "schema": SUMMARY_SCHEMA, "experiment": experiment, "record_set": record_set,
            "rows": rows, "measurements_outside_contract": other, "notes": summary.get("notes", {}),
            "inputs": inputs, "recomputed_by": f"{EXPERIMENTS[experiment]['module']}:summarize"})
        for relative in sorted({*CORE_FILES, EXPERIMENTS[experiment]["module"]}):
            destination = target / "code" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)
        write_json(target / "code" / "SOURCE.json", {
            "schema": "cross-morse-record-source-1",
            "exporter": f"{EXPERIMENTS[experiment]['module']}:export_records",
            "command": [Path(a).name if Path(a).is_absolute() else a for a in (argv or [])],
            "runtime_source_sha256": runtime_source_fingerprint(ROOT), "repository_commit": git_head(),
            "experiment_code": {k: v for k, v in code_identity().items()}, "record_contract": contract_path,
            "inputs": [{"run": i["run"], "plan_sha256": i["plan_sha256"],
                        "artifact_manifest_sha256": i["artifact_manifest_sha256"]} for i in inputs]})
        with (target / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(list(spec["keys"]) + SUMMARY_TAIL)
            for index, row in enumerate(rows):
                writer.writerow([row["keys"][k] for k in spec["keys"]]
                                + [row["quantity"], fmt(row["value"]), row["unit"], "raw/summary.json",
                                   f"json:/rows/{index}/value"])
        parents = [{k: v for k, v in exp["parent"].items()} for _, exp, _, _ in loaded]
        write_json(target / "provenance.json", {
            "experiment": EXPERIMENTS[experiment]["title"],
            "produced_by": (f"python -B scripts/run_experiment_b.py {experiment.replace('_', '-')} plan/run/export; "
                            f"{EXPERIMENTS[experiment]['module']} on {len(dirs)} sealed run(s)"),
            "hardware": str(hardware), "period": str(period), "protocol": protocol,
            "synthetic": bool(synthetic), "truncated_runs": truncated, "parents": parents,
            "record_contract": contract_path,
            "runtime_source_sha256": sorted({exp["runtime_source_sha256"] for _, exp, _, _ in loaded})})
        findings = _private_findings(target)
        if findings and not allow_private:
            raise ExperimentError("redact private infrastructure identifiers before depositing: " + ", ".join(findings[:10]))
        write_manifest(target)
        entry = verify_set(target, spec, specs)
        from scripts.record_mapping import install_verified_set
        final = install_verified_set(staging, target, records_root, specs, replace=replace)
        return {"set": record_set, "path": str(final), "rows": len(rows), "indexed": True,
                "record_contract": contract_path, "verification": entry,
                "not_exported": sorted({r["quantity"] for r in other})}
    finally:
        shutil.rmtree(staging, ignore_errors=True)


# ============================================================================ Part B: E2
E2_RAW_EXPORTS = ("raw/sequences.csv", "raw/unit_losses.csv", "raw/predictor.json", "raw/targets.json")
E2_DEFINITIONS = {
    "phi_seq": "share of the frozen teacher's executed (input, member) sequences on the evaluation role that the "
               "initial-input predictor reproduces exactly (A9 written as A8, terminal depth included)",
    "phi_gain": "[R(q_C) - R(q_pred)] / [R(q_C) - R(q_M)] on the same evaluation units (Note S25); defined only "
                "for a positive measured feedback gain",
    "feedback_gain_nats": "g = R(q_C) - R(q_M), unit-weighted mean loss of the comparator minus the teacher (nats)",
    "predictor_gain_nats": "R(q_C) - R(q_pred), unit-weighted mean loss of the comparator minus the executed "
                           "predicted sequences (nats)",
    "compared_sequences": "number of (input, ensemble member) teacher sequences compared",
    "evaluation_units": "statistical units scored on the evaluation role (graphs or graph cores)",
}


def plan_predictability(parent, output, *, allow_fixture=False, allow_posthoc=False, max_graphs=0,
                        evaluation_role="compare", training_seed=17, epochs=100, patience=15, lr=1e-3,
                        batch_size=64, capacity_tolerance=.01):
    parent = parent if isinstance(parent, ParentRun) else ParentRun(parent)
    if parent.synthetic and not allow_fixture:
        raise ExperimentError("a synthetic fixture parent requires --allow-fixture")
    if evaluation_role not in ("compare",):
        raise ExperimentError("E2 evaluates on the reserved D_compare, like E3-E6")
    checked = parent.check("confirm")
    parent.require_role(evaluation_role)
    scope = protocol_scope(parent, allow_posthoc=allow_posthoc)
    if type(max_graphs) is not int or max_graphs < 0:
        raise ExperimentError("max_graphs must be a nonnegative integer")
    for name, value in (("epochs", epochs), ("patience", patience), ("batch_size", batch_size)):
        if type(value) is not int or value < 1:
            raise ExperimentError(f"{name} must be a positive integer")
    if not (math.isfinite(lr) and lr > 0 and 0 <= capacity_tolerance <= .1):
        raise ExperimentError("invalid learning rate or capacity tolerance")
    teacher, comparator, record = parent.policies(allow_fixture=allow_fixture)
    if teacher["ensemble"] != comparator["ensemble"]:
        raise ExperimentError("E2 pairs teacher and comparator at one ensemble size")
    seeds = parent.seeds
    body = {"teacher": teacher, "comparator": comparator, **record,
            "fit_role": "explore", "evaluation_role": evaluation_role,
            "target_seeds": seeds, "ensemble_members": seeds[:teacher["ensemble"]],
            "decoder": {"recipe": "cm_experiments.schedules.fit_decoder(arm='E4') on the teacher's executed sequences "
                                  "and terminal depths; capacity matched to the teacher's action critic",
                        "training_seed": int(training_seed), "epochs": epochs, "patience": patience,
                        "lr": float(lr), "batch_size": batch_size, "capacity_tolerance": float(capacity_tolerance)},
            "max_graphs": max_graphs, "protocol": scope, "parent_check": checked, "definitions": E2_DEFINITIONS,
            "input_scope": ("complete D_explore graphs fit the predictor; complete D_compare graphs evaluate it"
                            if parent.dataset in ("PascalVOC-SP", "COCO-SP", "Peptides-func") else
                            "one transductive graph: the predictor is fitted and evaluated on the same input "
                            "(different label rows); unseen-input generalization is not measured")}
    return create_plan("predictability", parent, output, body)


def run_predictability(plan_path, *, device="cpu", trust=False):
    import torch
    from cross_morse.common import precision_context
    from cross_morse.data import load_data
    from cross_morse.learning import Controller
    from cross_morse.policies import execute_policy
    from cross_morse.provenance import sha256_state_dict
    from cross_morse.runtime import graph_operator, is_graph_collection
    from cross_morse.training import load_plant
    from .execution import initial_features
    from .schedules import fit_decoder, save_decoder
    from .study_execution import StudyBackend

    plan, parent = open_plan(plan_path, "predictability")
    parent.verify("confirm")
    run = RunDirectory(plan_path, plan, parent, device=device, trust=trust)
    root, config = parent.root, parent.config
    teacher, comparator = plan["teacher"], plan["comparator"]
    data = load_data(root).protect_labels()
    collection = is_graph_collection(data)
    profile = "full" if teacher.get("profile", "full") == "none" else teacher.get("profile", "full")
    control = Controller.load(root / f"fit/{profile}/controller.pt", "cpu")
    reference = dict(np.load(root / "explore/nominal_reference.npz", allow_pickle=False))
    L = parent.depth
    fit_units = [eid for eid, _ in role_units(data, "explore", plan["max_graphs"])]
    features, targets, lengths, groups, members, episode_ids, traces = ([] for _ in range(7))
    for seed in plan["target_seeds"]:
        model = load_plant(data, config, root / "train", seed, device)
        before = sha256_state_dict(model)
        for eid in fit_units:
            episode = data.graph(eid) if collection else data
            graph = graph_operator(episode, device)
            x = episode.x.to(device)
            with torch.no_grad(), precision_context(config, torch.device(device)):
                model.bind_graph(graph)
                feature = initial_features(model.encode(x), graph)
            _, trace = execute_policy(model, x, graph, config, teacher, control, reference)
            actions = [int(a) for a in trace["actions"]]
            if not 1 <= len(actions) <= L:
                raise ExperimentError("the teacher did not reach a valid terminal depth")
            normalized = [8 if a == 9 else a for a in actions]
            features.append(feature)
            targets.append(normalized + [0] * (L - len(normalized)))
            lengths.append(len(normalized))
            groups.append(f"graph:{eid}" if collection else f"checkpoint:{seed}")
            members.append(seed)
            episode_ids.append(eid)
            traces.append({"episode_id": eid, "seed": seed, "teacher_actions": actions,
                           "teacher_safe_depths": trace["safe_depths"], "a9_to_a8": actions.count(9)})
        if sha256_state_dict(model) != before:
            raise RuntimeError("frozen predictor changed during teacher target collection")
    arrays = {"initial_features": np.stack(features), "targets": np.asarray(targets, dtype=np.int64),
              "lengths": np.asarray(lengths, dtype=np.int64), "group_ids": np.asarray(groups),
              "source_seeds": np.asarray(members, dtype=np.int64), "episode_ids": np.asarray(episode_ids, dtype=np.int64),
              "roles": np.full(len(features), "Dexplore")}
    np.savez_compressed(run.raw("development_targets.npz"), **arrays)
    write_json(run.raw("targets.json"), {"records": traces, "fit_role": "explore", "label_free": True,
                                         "grouping": "complete graph" if collection else "checkpoint seed on one transductive graph"})
    critic_parameters = int(sum(p.numel() for p in control.critic.parameters()))
    d = plan["decoder"]
    net, audit = fit_decoder(arrays["initial_features"], arrays["targets"], arrays["group_ids"], arrays["roles"],
                             arm="E4", critic_parameters=critic_parameters, seed=d["training_seed"],
                             epochs=d["epochs"], patience=d["patience"], lr=d["lr"], batch_size=d["batch_size"],
                             tolerance=d["capacity_tolerance"], device=device, lengths=arrays["lengths"])
    decoder_path = run.raw("predictor.pt")
    save_decoder(net, audit, decoder_path)
    write_json(run.raw("predictor.json"), {"audit": audit, "decoder_sha256": sha256(decoder_path),
                                           "targets_sha256": sha256(run.root / "raw/development_targets.npz"),
                                           "locked_before_evaluation_label_access": True})
    backend = StudyBackend(root, data, config, teacher, comparator, plan["ensemble_members"], device,
                           {"predictor": net})
    frozen = {s: sha256_state_dict(m) for s, m in backend.models.items()}
    role = plan["evaluation_role"]
    sequences, observed = [], {"observation_id": [], "unit_id": [], "input_id": [], "y_true": [],
                               "qC": [], "qM": [], "qpred": []}
    for eid, episode in role_units(data, role, plan["max_graphs"]):
        ids = np.asarray(episode.splits[role], dtype=np.int64)
        if not len(ids):
            continue
        qc = backend.execute(episode, {"kind": "policy", "candidate": comparator})
        qm = backend.execute(episode, {"kind": "policy", "candidate": teacher})
        qp = backend.execute(episode, {"kind": "decoder", "arm": "E4", "decoder_id": "predictor",
                                       "candidate": teacher})
        for k, seed in enumerate(plan["ensemble_members"]):
            taught = [int(a) for a in qm["traces"][k]["actions"]]
            normalized = [8 if a == 9 else a for a in taught]
            predicted = [int(a) for a in qp["traces"][k]["actions"]]
            sequences.append({"input_id": eid, "member_seed": seed, "teacher_length": len(normalized),
                              "predicted_length": len(predicted), "reproduced": int(predicted == normalized),
                              "agreeing_depths": sum(a == b for a, b in zip(predicted, normalized)),
                              "a9_to_a8": taught.count(9),
                              "support_qualified": int(bool(qp["traces"][k].get("actual_state_support_audit", {})
                                                            .get("feasible", True))),
                              "decoder_multiply_adds": qp["traces"][k].get("decoder_multiply_adds"),
                              "teacher_actions": " ".join(map(str, taught)),
                              "predicted_actions": " ".join(map(str, predicted))})
        # Every prediction of this input exists before its evaluation labels are read.
        labels = role_labels(data, episode, role, ids)
        global_ids = ids if episode.global_sample_ids is None else np.asarray(episode.global_sample_ids)[ids]
        observed["observation_id"].append(np.asarray(global_ids, dtype=np.int64))
        observed["unit_id"].append(np.asarray(episode.cluster_ids[ids]))
        observed["input_id"].append(np.full(len(ids), eid, dtype=np.int64))
        observed["y_true"].append(labels)
        for arm, result in (("qC", qc), ("qM", qm), ("qpred", qp)):
            if result["prediction"] is None:
                raise ExperimentError(f"{arm} produced no prediction on input {eid}")
            observed[arm].append(np.asarray(result["prediction"])[ids])
    if not sequences:
        raise ExperimentError("the evaluation role holds no observations")
    if any(sha256_state_dict(m) != frozen[s] for s, m in backend.models.items()):
        raise RuntimeError("a frozen predictor changed during evaluation")
    parent.verify("confirm")
    arrays = {k: np.concatenate(v) for k, v in observed.items()}
    np.savez_compressed(run.raw("predictions.npz"), **arrays)
    write_csv(run.raw("sequences.csv"), sequences)
    _write_unit_losses(run.raw("unit_losses.csv"), arrays, config)
    summary = summarize_predictability([run.root])
    summary["truncated"] = bool(plan["max_graphs"])
    return run.finish(summary)


def _write_unit_losses(path, arrays, config):
    from cross_morse.evaluation import score_observations
    scores = {arm: score_observations(arrays[arm], arrays["y_true"], arrays["unit_id"], config)
              for arm in ("qC", "qM", "qpred")}
    units = scores["qC"]["core_ids"]
    rows = [{"unit_id": int(u) if np.issubdtype(np.asarray(units).dtype, np.integer) else str(u),
             "weight": float(scores["qC"]["counts"][i]),
             **{f"{arm}_loss": repr(float(scores[arm]["loss"][i])) for arm in scores}}
            for i, u in enumerate(units)]
    write_csv(path, rows)


def summarize_predictability(dirs):
    """Recompute the E2 readouts from sealed raw files only (no PyTorch)."""
    from cross_morse.evaluation import score_observations
    by_dataset = {}
    for d in dirs:
        d = Path(d)
        exp, plan = read_json(d / "experiment.json"), read_json(d / "plan.json")
        config = read_json(d / "base_config.json")
        with np.load(d / "raw/predictions.npz", allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        risks = {arm: score_observations(arrays[arm], arrays["y_true"], arrays["unit_id"], config)
                 for arm in ("qC", "qM", "qpred")}
        sequences = read_csv(d / "raw/sequences.csv")
        entry = by_dataset.setdefault(exp["parent"]["dataset"], {"runs": [], "compared": 0, "reproduced": 0,
                                                                  "units": 0, "scope": plan["input_scope"]})
        entry["runs"].append({arm: risks[arm]["node_ce"] for arm in risks})
        entry["compared"] += len(sequences)
        entry["reproduced"] += sum(int(r["reproduced"]) for r in sequences)
        entry["units"] += len(risks["qC"]["core_ids"])
    rows, notes = [], {}
    for dataset, entry in sorted(by_dataset.items()):
        keys = {"dataset": dataset}
        # Repeated official splits of one dataset are pooled by their mean risks.
        r = {arm: float(np.mean([run[arm] for run in entry["runs"]])) for arm in ("qC", "qM", "qpred")}
        gain, predictor = r["qC"] - r["qM"], r["qC"] - r["qpred"]
        rows.append(summary_row(keys, "phi_seq", entry["reproduced"] / entry["compared"], "fraction",
                                E2_DEFINITIONS["phi_seq"]))
        rows.append(summary_row(keys, "feedback_gain_nats", gain, "nats", E2_DEFINITIONS["feedback_gain_nats"]))
        rows.append(summary_row(keys, "predictor_gain_nats", predictor, "nats", E2_DEFINITIONS["predictor_gain_nats"]))
        rows.append(summary_row(keys, "compared_sequences", entry["compared"], "sequences",
                                E2_DEFINITIONS["compared_sequences"]))
        rows.append(summary_row(keys, "evaluation_units", entry["units"], "units", E2_DEFINITIONS["evaluation_units"]))
        status = "MEASURED"
        if gain > 0:
            rows.append(summary_row(keys, "phi_gain", predictor / gain, "fraction", E2_DEFINITIONS["phi_gain"]))
        else:
            status = "FEEDBACK_GAIN_NOT_POSITIVE: phi_gain is undefined and not reported"
        notes[dataset] = {"status": status, "runs_pooled": len(entry["runs"]), "risks": r,
                          "input_scope": entry["scope"]}
    return {"rows": rows, "notes": notes}


def export_records(results_dirs, **kwargs):
    protocol = ("E2: E4-recipe initial-input predictor fitted on the frozen teacher's executed D_explore sequences; "
                "teacher, comparator and predicted sequences executed by the released runtime on D_compare; "
                "phi_seq and phi_gain kept apart (Supplementary Table S36, Note S25)")
    return export_record_set("predictability", results_dirs, summarize=summarize_predictability,
                             raw_exports=E2_RAW_EXPORTS, protocol=protocol, **kwargs)
