#!/usr/bin/env python3
"""Turn experiment outputs into measurement record sets (the deposit layout of records/README.md).

    # E3-E6: a complete campaign export written by `scripts/arm_campaign.py export --complete`
    python -B scripts/export_records.py budget-matched --campaign /abs/arm_campaign_export \
        --hardware "NVIDIA GH200 96 GB, Arrhenius" --period "2026-11-02/2026-11-09"

    # E12/E17: pinned-upstream studies of scripts/run_upstream_baseline.py, E17 timing rows, sacct exports
    python -B scripts/export_records.py baselines --study /abs/studies/arxiv_amp --study /abs/studies/arxiv_admp \
        --timing /abs/timing/arxiv.csv --accounting /abs/accounting/sacct.psv \
        --hardware "NVIDIA GH200 96 GB, Arrhenius" --period "2026-11-02/2026-11-09"

    # Quantities that a sealed eight-stage run records about itself
    python -B scripts/export_records.py from-runs --set sensor_parameters --run /abs/runs/arxiv_full ...
    python -B scripts/export_records.py from-runs --set nominal_time --run /abs/runs/arxiv_full ...

    # Any set from supplied output files in another format: explicit locators and declared derivations
    python -B scripts/export_records.py mapping-template --set factorials --output ../cm-evidence/factorials.mapping.json
    python -B scripts/export_records.py from-mapping --mapping ../cm-evidence/factorials.mapping.json

Each command writes ``<root>/<set>/`` (default root: ``records/``) with ``summary.csv``, ``raw/``,
``code/`` and ``provenance.json``, then runs ``scripts/records.py index`` on the root, which hashes
the set and re-reads every value from its raw file.  Original outputs are copied into ``raw/``
unchanged.  Where a quantity is computed (a test score from saved predictions, GPU-hours from the
scheduler export, a wall ratio from timing rows, a resolvent share from action counts), the exporter
writes the result into a small JSON file under ``raw/`` together with the SHA-256 of every input.
Small inputs are copied next to it; bulky ones (per-fit test predictions, per-unit campaign
observations) stay in the archived studies or campaign (DOI in ``provenance.json``) and are named by
SHA-256 and archive path, unless ``--copy-predictions`` / ``--copy-observations`` is given.

Nothing is filled in: a study without its predictions, a timing row whose scheduler job is absent
from the accounting, or a campaign without its complete analysis stops the export.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._paper_common import runtime_source_fingerprint  # noqa: E402
from scripts.records import RECORDS, cmd_index, load_specs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SPECS_PATH = RECORDS / "specs.json"
RESOLVENT_ACTIONS = (3, 4, 5, 6, 7)          # registry A3-A7 (Supplementary Table S6)
REGISTRY_SIZE = 10                           # A0-A9
SUMMARY_TAIL = ["quantity", "value", "unit", "source_file", "locator"]


class ExportError(ValueError):
    """The inputs do not support a complete, traceable record set."""


# ----------------------------------------------------------------------------- helpers
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ExportError(f"missing input file: {path}") from None


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def fmt(value: float) -> str:
    """Print a value so that it agrees with its raw source at the printed precision."""
    value = float(value)
    if not math.isfinite(value):
        raise ExportError(f"non-finite value {value!r}")
    text = format(value, ".10g")
    return "0" if text == "-0" else text


def safe_name(text: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")
    if not name:
        raise ExportError(f"cannot form a file name from {text!r}")
    return name


def copy_into(target: Path, source: Path, relative: str) -> str:
    """Copy an original output into the set unchanged and return its set-relative path."""
    destination = target / relative
    if destination.exists():
        if sha256_file(destination) != sha256_file(source):
            raise ExportError(f"two different inputs map to {relative}")
        return relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return relative


def git_head() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=False, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    head = out.stdout.strip()
    return head if out.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", head) else None


def start_set(root: Path, name: str, replace: bool) -> Path:
    specs = load_specs(root, SPECS_PATH)
    if name not in specs["sets"]:
        raise ExportError(f"{name} is not a record set of records/specs.json")
    target = root / name
    if target.exists():
        if not replace:
            raise ExportError(f"{target} exists; pass --replace to rebuild it from the inputs")
        shutil.rmtree(target)
    (target / "raw").mkdir(parents=True)
    (target / "code").mkdir()
    return target


def finish_set(root: Path, target: Path, *, keys: list[str], rows: list[dict], provenance: dict,
               argv: list[str], inputs: list[dict]) -> None:
    if not rows:
        raise ExportError(f"{target.name}: no values to export")
    with (target / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(keys + SUMMARY_TAIL)
        for row in rows:
            writer.writerow([row[k] for k in keys] + [row[c] for c in SUMMARY_TAIL])
    dump_json(target / "code" / "SOURCE.json", {
        "schema": "cross-morse-record-source-1",
        "exporter": "scripts/export_records.py",
        "command": [Path(a).name if Path(a).is_absolute() else a for a in argv],
        "runtime_source_sha256": runtime_source_fingerprint(ROOT),
        "repository_commit": git_head(),
        "inputs": inputs,
    })
    for field in ("experiment", "produced_by", "hardware", "period", "protocol"):
        if not str(provenance.get(field, "")).strip():
            raise ExportError(f"{target.name}: provenance field {field!r} is empty; pass it explicitly")
    dump_json(target / "provenance.json", provenance)
    if cmd_index(root, SPECS_PATH) != 0:
        raise ExportError(f"{target.name}: scripts/records.py index failed")


def value_row(keys: dict, quantity: str, value, unit: str, source: str, locator: str) -> dict:
    return {**keys, "quantity": quantity, "value": fmt(value), "unit": unit, "source_file": source, "locator": locator}


# ----------------------------------------------------------------------------- budget-matched (E3-E6)
BUDGET_FIELDS = (
    ("effect_vs_qc_nats", "effect_vs_qC", "nats"),
    ("effect_vs_feedback_nats", "effect", "nats"),
    ("effect_vs_feedback_ci_low_nats", "ci_low", "nats"),
    ("effect_vs_feedback_ci_high_nats", "ci_high", "nats"),
    ("p_holm", "p_holm", "probability"),
    ("work_ratio", "work_ratio", "q_C = 1"),
    ("ceiling", "ceiling", "q_C = 1"),
)


def export_budget_matched(args) -> Path:
    campaign = Path(args.campaign).resolve()
    study = load_json(campaign / "study.json")
    if study.get("schema") != "cm-budget-matched-3" or study.get("record_id") != "budget_matched":
        raise ExportError("expected a campaign export of scripts/arm_campaign.py (schema cm-budget-matched-3)")
    complete = (campaign / "analysis.json").is_file()
    if not complete and not args.allow_partial:
        raise ExportError("the campaign export has no analysis.json; export the complete campaign with "
                          "`scripts/arm_campaign.py export --complete` (a partial export is not a deposit)")
    analysis_name = "analysis.json" if complete else "partial_analysis.json"
    analysis = load_json(campaign / analysis_name)
    if analysis.get("check") != "budget_matched" or analysis.get("passed") is not True:
        raise ExportError(f"{analysis_name} does not record a passed budget-matched check")
    summary = analysis.get("summary") or []
    if complete and len(summary) != 24:
        raise ExportError(f"a complete campaign has 24 dataset-arm contrasts; {analysis_name} has {len(summary)}")
    root = Path(args.root).resolve()
    target = start_set(root, "budget_matched", args.replace)
    inputs = []
    for name in (analysis_name, "study.json", "coverage.json", "seed_manifest.json", "paired_observations.csv"):
        source = campaign / name
        if not source.is_file():
            raise ExportError(f"campaign export lacks {name}")
        # The per-unit observations can hold hundreds of thousands of rows; they stay in the campaign archive,
        # identified here by SHA-256, unless the set should carry them (--copy-observations).
        if name != "paired_observations.csv" or args.copy_observations:
            copy_into(target, source, f"raw/{name}")
        inputs.append({"file": name, "sha256": sha256_file(source), "copied": name != "paired_observations.csv" or args.copy_observations})
    rows = []
    for i, entry in enumerate(summary):
        keys = {"dataset": entry["dataset"], "arm": entry["arm"], "configuration": "selected"}
        for quantity, field, unit in BUDGET_FIELDS:
            if field not in entry:
                if complete:
                    raise ExportError(f"summary entry {i} lacks {field}")
                continue
            rows.append(value_row(keys, quantity, entry[field], unit, f"raw/{analysis_name}", f"json:/summary/{i}/{field}"))
    provenance = {
        "experiment": "E3-E6 budget-matched comparison at the common ceiling",
        "produced_by": "scripts/run_budget_arms.py per controller replicate; scripts/arm_campaign.py export"
                       + (" --complete" if complete else " (partial)") + "; scripts/export_records.py budget-matched",
        "hardware": args.hardware, "period": args.period,
        "protocol": ("development-selected arms evaluated on independent D_compare against the confirmed policy q_M at the registered "
                     "common ceiling; crossed seed-by-unit bootstrap intervals and Holm-adjusted paired sign-flip tests "
                     "within each dataset's family of four arms (cm_campaigns.arm_campaign.check_campaign)"),
        "synthetic": False, "complete_campaign": complete,
        "campaign_archive": {"doi": args.archive_doi, "sha256": args.archive_sha256},
    }
    finish_set(root, target, keys=["dataset", "arm", "configuration"], rows=rows, provenance=provenance,
               argv=sys.argv if args.argv is None else args.argv, inputs=inputs)
    return target


# ----------------------------------------------------------------------------- baselines (E12, E17)
def read_timing(paths: list[Path]) -> dict[tuple[str, str], list[dict]]:
    from cm_campaigns.timing import COLUMNS
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for path in paths:
        with Path(path).open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(COLUMNS):
                raise ExportError(f"{Path(path).name}: timing rows must have the E17 columns {list(COLUMNS)}")
            for row in reader:
                groups[(row["dataset"], row["method"])].append(row)
    return groups


def read_accounting(paths: list[Path]) -> dict[str, dict]:
    from cm_campaigns.accounting import read_sacct
    allocations: dict[str, dict] = {}
    for path in paths:
        for key, row in read_sacct(Path(path))["allocations"].items():
            if key in allocations and allocations[key]["gpu_hours"] != row["gpu_hours"]:
                raise ExportError(f"conflicting scheduler records for allocation {key}")
            allocations[key] = row
    return allocations


def study_plan(study: Path) -> dict:
    from cm_experiments.upstream import validate_plan
    plans = sorted(study.glob("hpo/trial-*/plan.json")) + sorted(study.glob("final/seed-*/split-*/plan.json"))
    if not plans:
        raise ExportError(f"{study.name}: no attempt carries its plan.json")
    plan = load_json(plans[0])
    for other in plans[1:]:
        if load_json(other) != plan:
            raise ExportError(f"{study.name}: attempts mix different plans")
    return validate_plan(plan)


def attempt_job_ids(attempt_dir: Path) -> set[str]:
    invocation = load_json(attempt_dir / "invocation.json")
    slurm = invocation.get("slurm", {})
    ids = set()
    if slurm.get("SLURM_ARRAY_JOB_ID") and slurm.get("SLURM_ARRAY_TASK_ID") is not None:
        ids.add(f"{slurm['SLURM_ARRAY_JOB_ID']}_{slurm['SLURM_ARRAY_TASK_ID']}")
    if slurm.get("SLURM_JOB_ID"):
        ids.add(str(slurm["SLURM_JOB_ID"]))
    return ids


def export_baselines(args) -> Path:
    from cm_campaigns.contract import METRICS, SLUGS
    from cm_campaigns.numeric import probability_scores
    from cm_campaigns.timing import ratios
    studies = [Path(s).resolve() for s in args.study]
    timing_paths = [Path(p).resolve() for p in args.timing]
    accounting_paths = [Path(p).resolve() for p in args.accounting]
    if not studies and not timing_paths:
        raise ExportError("give at least one --study or --timing input")
    allocations = read_accounting(accounting_paths) if accounting_paths else {}
    by_job = defaultdict(list)
    for key, row in allocations.items():
        for job in {row["job_id"], row["job_id_raw"]}:
            by_job[str(job)].append((key, row))
    timing = read_timing(timing_paths) if timing_paths else {}
    root = Path(args.root).resolve()
    target = start_set(root, "baselines", args.replace)
    inputs, rows = [], []
    for path in accounting_paths:
        copy_into(target, path, f"raw/accounting/{safe_name(path.name)}")
        inputs.append({"file": f"accounting/{path.name}", "sha256": sha256_file(path)})
    for path in timing_paths:
        copy_into(target, path, f"raw/timing/{safe_name(path.name)}")
        inputs.append({"file": f"timing/{path.name}", "sha256": sha256_file(path)})

    for study in studies:
        plan = study_plan(study)
        dataset, method, metric = plan["dataset"], plan["method"], METRICS[plan["dataset"]]
        base = f"raw/{SLUGS[dataset]}/{safe_name(method)}"
        keys = {"dataset": dataset, "method": method, "seed": ""}
        jobs: set[str] = set()
        # every HPO attempt, failed and pruned ones included
        attempts = []
        for path in sorted(study.glob("hpo/trial-*/attempt.json")):
            record = load_json(path)
            attempts.append({k: record.get(k) for k in ("index", "status", "development_score", "elapsed_seconds",
                                                        "process_exit_code", "reason")})
            jobs |= attempt_job_ids(path.parent)
        if len(attempts) != len(plan["trials"]):
            raise ExportError(f"{study.name}: {len(attempts)} HPO attempts recorded for {len(plan['trials'])} registered trials")
        dump_json(target / base / "hpo_attempts.json", {"count": len(attempts), "attempts": attempts,
                                                        "completed": sum(a["status"] == "COMPLETE" for a in attempts)})
        rows.append(value_row(keys, "hpo_trials", len(attempts), "trials", f"{base}/hpo_attempts.json", "json:/count"))
        # final fits: score recomputed from the saved predictions
        per_seed: dict[int, list[tuple[int, float, str]]] = defaultdict(list)
        parameter_counts = set()
        for path in sorted(study.glob("final/seed-*/split-*/attempt.json")):
            record = load_json(path)
            if record.get("status") != "COMPLETE":
                raise ExportError(f"{path.parent.relative_to(study)}: final fit is {record.get('status')}; "
                                  "a registered seed cannot be dropped")
            jobs |= attempt_job_ids(path.parent)
            receipt = load_json(path.parent / "adapter_output" / "receipt.json")
            predictions = (path.parent / "adapter_output" / receipt["predictions"]).resolve()
            if not predictions.is_relative_to((path.parent / "adapter_output").resolve()):
                raise ExportError(f"{path.parent.name}: receipt points outside the adapter output")
            scored = probability_scores(predictions, metric, graph_mean_loss=dataset in ("PascalVOC-SP", "COCO-SP", "Peptides-func"))
            seed, split = int(record["seed"]), int(record["split"])
            leaf = f"{base}/seed-{seed}/split-{split}"
            # Test predictions of the large benchmarks run to hundreds of MB per fit: they stay in the study
            # archive, identified by SHA-256 and archive path, unless the set should carry them (--copy-predictions).
            if args.copy_predictions:
                copy_into(target, predictions, f"{leaf}/{predictions.name}")
            reported = scored["score"] * (1.0 if metric == "ap" else 100.0)
            dump_json(target / leaf / "score.json", {
                "metric": metric, "score": scored["score"], "score_reported": reported,
                "unit": "AP" if metric == "ap" else "%", "loss": scored["loss"], "observations": scored["n_observations"],
                "units": scored["n_units"], "predictions_sha256": sha256_file(predictions),
                "predictions_in_study_archive": predictions.relative_to(study).as_posix(),
                "predictions_copied": bool(args.copy_predictions)})
            per_seed[seed].append((split, reported, f"{leaf}/score.json"))
            if receipt.get("parameter_count") is not None:
                parameter_counts.add(int(receipt["parameter_count"]))
        seeds = sorted(per_seed)
        if studies and len(seeds) != len(plan["seed_ids"]):
            raise ExportError(f"{study.name}: final fits cover {len(seeds)} of {len(plan['seed_ids'])} registered seeds")
        for seed in seeds:
            items = sorted(per_seed[seed])
            seed_keys = {**keys, "seed": str(seed)}
            if len(items) == 1:
                rows.append(value_row(seed_keys, "score", items[0][1], "AP" if metric == "ap" else "%",
                                      items[0][2], "json:/score_reported"))
            else:   # Roman-empire: the seed's score is the mean over the ten official splits
                mean = statistics.fmean(v for _, v, _ in items)
                dump_json(target / base / f"seed-{seed}" / "score_mean.json", {
                    "score_reported": mean, "splits": [{"split": s, "score_reported": v, "file": f} for s, v, f in items]})
                rows.append(value_row(seed_keys, "score", mean, "%", f"{base}/seed-{seed}/score_mean.json",
                                      "json:/score_reported"))
        if len(parameter_counts) > 1:
            raise ExportError(f"{study.name}: final fits report different parameter counts {sorted(parameter_counts)}")
        if parameter_counts:
            count = parameter_counts.pop()
            dump_json(target / base / "parameters.json", {"parameter_count": count, "params_millions": count / 1e6})
            rows.append(value_row(keys, "params_millions", count / 1e6, "M", f"{base}/parameters.json", "json:/params_millions"))
        # GPU-hours of every attempt's allocation (HPO and final fits)
        if allocations:
            matched, missing = {}, sorted(j for j in jobs if j not in by_job)
            if missing and not args.allow_missing_accounting:
                raise ExportError(f"{study.name}: allocations {missing[:5]} are absent from the sacct exports")
            for job in jobs:
                for key, row in by_job.get(job, []):
                    matched[key] = row
            total = sum(r["gpu_hours"] for r in matched.values())
            dump_json(target / base / "accounting.json", {
                "gpu_hours": total, "scope": "allocated GPUs x allocation seconds / 3600 over every HPO and final-fit "
                "allocation of the study; allocation hours, not utilisation", "missing_job_ids": missing,
                "allocations": [{"job_id": r["job_id"], "state": r["state"], "exit_code": r["exit_code"],
                                 "seconds": r["seconds"], "allocated_gpus": r["allocated_gpus"],
                                 "gpu_hours": r["gpu_hours"]} for _, r in sorted(matched.items())]})
            rows.append(value_row(keys, "gpu_hours", total, "GPU-h", f"{base}/accounting.json", "json:/gpu_hours"))

    # E17: serial cost and wall time of every timed method (external and internal rows)
    for (dataset, method), group in sorted(timing.items()):
        jobs = sorted({r["job_key"] for r in group})
        if allocations:
            unknown = [j for j in jobs if j not in allocations]
            if unknown:
                raise ExportError(f"{dataset}/{method}: timing rows name allocations {unknown[:3]} absent from the sacct exports")
            bad = [j for j in jobs if allocations[j]["allocated_gpus"] < 1 or not allocations[j]["state"].startswith("COMPLETED")]
            if bad:
                raise ExportError(f"{dataset}/{method}: timing was not taken in a completed GPU allocation: {bad[:3]}")
        measured = ratios(group)
        slug = SLUGS.get(dataset, safe_name(dataset))
        base = f"raw/{slug}/{safe_name(method)}"
        dump_json(target / base / "timing_ratios.json", {
            "wall_ratio": measured["wall_ratio"], "work_ratio": measured["work_ratio"],
            "observations": measured["observations"], "job_keys": jobs,
            "definition": "means over the post-warm-up repeats of method/nominal wall seconds and work (protocol E17)"})
        keys = {"dataset": dataset, "method": method, "seed": ""}
        rows.append(value_row(keys, "serial_cost", measured["work_ratio"], "nominal executions",
                              f"{base}/timing_ratios.json", "json:/work_ratio"))
        rows.append(value_row(keys, "wall_time", measured["wall_ratio"], "nominal executions",
                              f"{base}/timing_ratios.json", "json:/wall_ratio"))
    provenance = {
        "experiment": "E12 retrained external baselines and E17 timing protocol",
        "produced_by": "scripts/run_upstream_baseline.py (pinned upstream code, 48 HPO attempts, registered seeds); "
                       "cm_campaigns.timing.TimingSession; hpc/capture_arrhenius_accounting.sh; "
                       "scripts/export_records.py baselines",
        "hardware": args.hardware, "period": args.period,
        "protocol": ("scores recomputed from each final fit's saved test predictions with the official metric; "
                     "GPU-hours from the original sacct allocations of every attempt; serial cost and wall time as "
                     "post-warm-up ratios to the nominal execution on the same input"),
        "synthetic": False,
        "studies_archive": {"doi": args.archive_doi, "sha256": args.archive_sha256,
                            "studies": [s.name for s in studies]},
    }
    finish_set(root, target, keys=["dataset", "method", "seed"], rows=rows, provenance=provenance,
               argv=sys.argv if args.argv is None else args.argv, inputs=inputs)
    return target


# ----------------------------------------------------------------------------- from sealed runs
def run_identity(run: Path):
    from cross_morse.review import NumericEvidence
    evidence = NumericEvidence(run)
    config = evidence.json("config.json")
    state = load_json(run / "run_state.json")
    if state.get("status") != "COMPLETE" or state.get("last_complete_stage") != "evaluate":
        raise ExportError(f"{run.name}: the run is not sealed through evaluate")
    synthetic = config.get("dataset") == "synthetic" or config.get("scientific_run") is False
    benchmark = config.get("benchmark_dataset") or config.get("dataset")
    label = f"{benchmark} (synthetic)" if synthetic else benchmark
    return config, label, synthetic, evidence


def run_period(runs: list[Path]) -> str:
    stamps = []
    for run in runs:
        for path in (run / "provenance" / "stages").glob("*.json"):
            stamp = load_json(path).get("created_utc")
            if stamp:
                stamps.append(stamp)
    if not stamps:
        raise ExportError("the runs carry no stage timestamps; pass --period")
    return f"{min(stamps)}/{max(stamps)}"


def run_hardware(runs: list[Path]) -> str:
    seen = set()
    for run in runs:
        env = load_json(run / "environment.json")
        device = env.get("gpu") or f"{env.get('machine', '?')} CPU"
        seen.add(f"{device}; torch {env.get('torch', '?')}; python {env.get('python', '?')}")
    return " | ".join(sorted(seen))


def prediction_directory(run: Path, summary: dict) -> str:
    """Use the sealed evaluation layout, never an earlier lifecycle role."""
    scope = summary.get("prediction_storage_scope")
    if scope == "test_role":
        return "predictions/test"
    if scope not in (None, "transductive_full_graph"):
        raise ExportError(f"{run.name}: unknown prediction storage scope {scope!r}")
    if any(p.is_dir() for p in (run / "predictions").iterdir()):
        raise ExportError(f"{run.name}: role-scoped predictions require a sealed test_role summary")
    return "predictions"  # Historical archives and current transductive runs.


def execution_records(run: Path, summary: dict, evidence):
    """Read current traces from the sealed summary; test caches are optional copies."""
    if summary.get("prediction_storage_scope") == "test_role":
        records = summary.get("test_role_execution")
        if not isinstance(records, dict) or not records:
            raise ExportError(f"{run.name}: sealed summary has no test-role execution records")
        source = evidence.file("final/results_summary.json")
        result = []
        for candidate_id, record in records.items():
            if (not isinstance(candidate_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", candidate_id)
                    or record["candidate"].get("candidate_id") != candidate_id):
                raise ExportError(f"{run.name}: invalid test-role candidate identity")
            result.append((source, f"/test_role_execution/{candidate_id}", record))
        return result
    directory = prediction_directory(run, summary)
    return [(evidence.file(relative), "", evidence.json(relative)) for relative in sorted(evidence.sealed)
            if Path(relative).parent.as_posix() == directory and relative.endswith(".json")]


def validate_candidate_record(run: Path, record: dict, expected: dict, summary: dict, model: str) -> None:
    identity = record["candidate"]
    if any(identity.get(k) != expected.get(k) for k in ("candidate_id", "hash", "route", "ensemble")):
        raise ExportError(f"{run.name}: candidate execution differs from the evaluated {model} identity")
    members = record["members"]
    ensemble = identity.get("ensemble")
    if type(ensemble) is not int or ensemble < 1 or len(members) != ensemble:
        raise ExportError(f"{run.name}: candidate member count differs from its ensemble")
    if record.get("execution_failure") or record.get("fallback_applied"):
        raise ExportError(f"{run.name}: a failed candidate is not a completed {model} execution")
    if record.get("prediction_sha256") != summary["models"][model]["probability_sha256"]:
        raise ExportError(f"{run.name}: candidate predictions differ from the evaluated {model} predictions")
    if summary.get("prediction_storage_scope") == "test_role":
        if (record.get("execution_role") != "test" or record.get("role_scoped_execution") is not True
                or any(m.get("execution_role") != "test" for m in members)):
            raise ExportError(f"{run.name}: candidate/member execution must use the test role")
        if summary.get("test_role_execution", {}).get(identity["candidate_id"]) != record:
            raise ExportError(f"{run.name}: candidate differs from the sealed test-role execution summary")
        count = summary["models"][model]["test_units"]
        if type(count) is not int or count < 1 or any(m.get("graphs") != count for m in members):
            raise ExportError(f"{run.name}: candidate members do not cover the evaluated test graphs")


def export_from_runs(args) -> Path:
    runs = [Path(r).resolve() for r in args.run]
    if not runs:
        raise ExportError("give at least one --run")
    identities = {run: run_identity(run) for run in runs}
    flags = {synthetic for _, _, synthetic, _ in identities.values()}
    if len(flags) != 1:
        raise ExportError("synthetic examples and benchmark runs cannot share one record set")
    root = Path(args.root).resolve()
    if True in flags and root == RECORDS.resolve():
        raise ExportError("synthetic example runs cannot be exported into records/; use --root examples/records")
    target = start_set(root, args.set, args.replace)
    rows, inputs = [], []
    for run in runs:
        config, label, synthetic, evidence = identities[run]
        slug = safe_name(run.name)
        keys = {"dataset": label}
        summary_path = evidence.file("final/results_summary.json")
        summary = evidence.json("final/results_summary.json")
        copy_into(target, run / "config.json", f"raw/{slug}/config.json")
        copy_into(target, summary_path, f"raw/{slug}/results_summary.json")
        inputs += [{"file": f"{run.name}/config.json", "sha256": sha256_file(run / "config.json")},
                   {"file": f"{run.name}/final/results_summary.json", "sha256": sha256_file(summary_path)}]
        available = execution_records(run, summary, evidence)
        if args.set == "sensor_parameters":
            deployed = summary["deployed"]
            matches = [row for row in available if row[2]["candidate"]["candidate_id"] == deployed["candidate_id"]]
            if len(matches) != 1:
                raise ExportError(f"{run.name}: expected one evaluated deployed candidate record; found {len(matches)}")
            path, pointer, record = matches[0]
            validate_candidate_record(run, record, deployed, summary, "qD")
            record_source = f"raw/{slug}/results_summary.json" if pointer else f"raw/{slug}/deployed_candidate.json"
            if not pointer:
                copy_into(target, path, record_source)
            inputs.append({"file": f"{run.name}/{path.relative_to(run).as_posix()}",
                           "json_pointer": pointer, "sha256": sha256_file(path)})
            counts = [0] * REGISTRY_SIZE
            for member in record["members"]:
                for field in ("native_site_action_counts", "inserted_site_action_counts"):
                    values = member.get(field) or [0] * REGISTRY_SIZE
                    if len(values) != REGISTRY_SIZE:
                        raise ExportError(f"{run.name}: {field} does not cover the ten registered actions")
                    counts = [a + int(b) for a, b in zip(counts, values)]
            executed = sum(counts)
            if executed <= 0:
                raise ExportError(f"{run.name}: the deployed rule records no executed actions")
            resolvent = sum(counts[i] for i in RESOLVENT_ACTIONS)
            dump_json(target / "raw" / slug / "sensor_parameters.json", {
                "residual_fraction": resolvent / executed, "resolvent_actions": resolvent, "executed_actions": executed,
                "action_counts_A0_A9": counts, "deployed_route": deployed.get("route"),
                "execution_record": {"file": record_source, "json_pointer": pointer},
                "definition": "share of resolvent actions (A3-A7) among all actions executed at native and inserted sites "
                              "by the deployed rule's members, from the run's own execution record"})
            rows.append(value_row(keys, "sketch_dim", config["sensor"]["sketch_dim"], "columns (0 = full state)",
                                  f"raw/{slug}/config.json", "json:/sensor/sketch_dim"))
            rows.append(value_row(keys, "observer_period", config["controller"]["observer_period"], "control depths",
                                  f"raw/{slug}/config.json", "json:/controller/observer_period"))
            if "sensing_period" not in config["sensor"]:
                raise ExportError(f"{run.name}: sensor acquisition period is undeclared; use a run of this runtime")
            rows.append(value_row(keys, "sensing_period", config["sensor"]["sensing_period"], "control depths",
                                  f"raw/{slug}/config.json", "json:/sensor/sensing_period"))
            rows.append(value_row(keys, "residual_fraction", resolvent / executed, "fraction",
                                  f"raw/{slug}/sensor_parameters.json", "json:/residual_fraction"))
        elif args.set == "nominal_time":
            comparator_ensemble = summary["comparator"].get("ensemble")
            nominal = [row for row in available if row[2]["candidate"].get("route") == "nominal"
                       and row[2]["candidate"].get("ensemble") == comparator_ensemble]
            if len(nominal) != 1:
                raise ExportError(f"{run.name}: expected one nominal candidate at comparator ensemble "
                                  f"{comparator_ensemble} in the evaluated records; found {len(nominal)}")
            path, pointer, record = nominal[0]
            validate_candidate_record(run, record, record["candidate"], summary, "qN")
            record_source = f"raw/{slug}/results_summary.json" if pointer else f"raw/{slug}/nominal_candidate.json"
            if not pointer:
                copy_into(target, path, record_source)
            inputs.append({"file": f"{run.name}/{path.relative_to(run).as_posix()}",
                           "json_pointer": pointer, "sha256": sha256_file(path)})
            per_member = []
            for k, member in enumerate(record["members"]):
                count = int(member.get("graphs", 1))
                if count < 1 or not math.isfinite(member["wall_seconds"]) or member["wall_seconds"] < 0:
                    raise ExportError(f"{run.name}: invalid nominal input count or measured wall time")
                per_member.append({"member": k, "seed": member.get("seed"), "wall_seconds": member["wall_seconds"],
                                   "inputs": count, "seconds_per_input": member["wall_seconds"] / count})
            t_nom = statistics.fmean(m["seconds_per_input"] for m in per_member)
            dump_json(target / "raw" / slug / "t_nom.json", {
                "t_nom_seconds": t_nom, "members": per_member,
                "execution_record": {"file": record_source, "json_pointer": pointer},
                "definition": "wall seconds of one nominal member execution divided by its inputs (graphs; one for a "
                              "single-graph benchmark), mean over the nominal candidate's members, from the run's own record"})
            rows.append(value_row(keys, "t_nom_seconds", t_nom, "s", f"raw/{slug}/t_nom.json", "json:/t_nom_seconds"))
        else:
            raise ExportError(f"from-runs does not derive {args.set}")
    synthetic = flags.pop()
    provenance = {
        "experiment": {"sensor_parameters": "executed sensor parameters (Supplementary Table S6c)",
                       "nominal_time": "per-input nominal execution time (Supplementary Table S23b)"}[args.set],
        "produced_by": f"scripts/export_records.py from-runs --set {args.set} on {len(runs)} sealed run(s): "
                       + ", ".join(sorted(r.name for r in runs)),
        "hardware": args.hardware or run_hardware(runs),
        "period": args.period or run_period(runs),
        "protocol": "read from the runs' sealed config.json, results_summary.json and candidate execution records",
        "synthetic": synthetic,
    }
    finish_set(root, target, keys=["dataset"], rows=rows, provenance=provenance,
               argv=sys.argv if args.argv is None else args.argv, inputs=inputs)
    return target


# ----------------------------------------------------------------------------- command line
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, needs_hardware=True):
        p.add_argument("--root", type=Path, default=RECORDS, help="records root (default records/)")
        p.add_argument("--replace", action="store_true", help="rebuild an existing set from the inputs")
        p.add_argument("--hardware", required=needs_hardware, default=None)
        p.add_argument("--period", required=needs_hardware, default=None)

    p = sub.add_parser("budget-matched", help="E3-E6 from a campaign export")
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--allow-partial", action="store_true", help="dry run on a partial export (never a deposit)")
    p.add_argument("--copy-observations", action="store_true", help="also copy paired_observations.csv into raw/")
    p.add_argument("--archive-doi", default=None)
    p.add_argument("--archive-sha256", default=None)
    common(p)
    p = sub.add_parser("baselines", help="E12/E17 from upstream studies, timing rows and sacct exports")
    p.add_argument("--study", type=Path, action="append", default=[])
    p.add_argument("--timing", type=Path, action="append", default=[])
    p.add_argument("--accounting", type=Path, action="append", default=[])
    p.add_argument("--allow-missing-accounting", action="store_true")
    p.add_argument("--copy-predictions", action="store_true", help="also copy every final fit's prediction NPZ into raw/")
    p.add_argument("--archive-doi", default=None, help="DOI of the archived studies")
    p.add_argument("--archive-sha256", default=None)
    common(p)
    p = sub.add_parser("from-runs", help="quantities that sealed runs record about themselves")
    p.add_argument("--set", required=True, choices=("sensor_parameters", "nominal_time"))
    p.add_argument("--run", type=Path, action="append", default=[])
    common(p, needs_hardware=False)
    p = sub.add_parser("mapping-template", help="write the row skeleton of a set for from-mapping")
    p.add_argument("--set", required=True)
    p.add_argument("--output", type=Path, required=True, help="new JSON file outside the checkout")
    p = sub.add_parser("from-mapping", help="any set from original outputs named in a mapping file (scripts/record_mapping.py)")
    p.add_argument("--mapping", type=Path, required=True)
    p.add_argument("--root", type=Path, default=RECORDS, help="records root (default records/)")
    p.add_argument("--replace", action="store_true", help="rebuild an existing set from the mapping")
    p.add_argument("--allow-partial", action="store_true", help="write a set whose coverage the audit will refuse")
    p.add_argument("--allow-private-patterns", action="store_true", help="keep cluster identifiers (non-public sets only)")
    args = parser.parse_args(argv)
    args.argv = ["scripts/export_records.py", *(argv if argv is not None else sys.argv[1:])]
    try:
        if args.command == "mapping-template":
            from scripts.record_mapping import template
            output = args.output.resolve()
            if output.is_relative_to(ROOT) or output.exists():
                raise ExportError("write the mapping template to a new file outside the checkout")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(template(args.set), indent=1) + "\n", encoding="utf-8")
            print(f"wrote {output}")
            return 0
        if args.command == "from-mapping":
            from scripts.record_mapping import build
            target = build(args.mapping, records_root=args.root, replace=args.replace, allow_partial=args.allow_partial,
                           allow_private=args.allow_private_patterns, argv=args.argv)
        else:
            target = {"budget-matched": export_budget_matched, "baselines": export_baselines,
                      "from-runs": export_from_runs}[args.command](args)
    except (ExportError, ValueError, KeyError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
