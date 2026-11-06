#!/usr/bin/env python3
"""Check whether completed official runs can become registered evidence, before packing or depositing.

    python -B scripts/evidence_preflight.py --dataset ogbn-arxiv --run /abs/arxiv_full --output-dir ../cm-evidence/preflight
    python -B scripts/evidence_preflight.py --dataset Roman-empire --campaign /abs/roman_campaign --output-dir ...
    python -B scripts/evidence_preflight.py --plan ../cm-evidence/plan.json --output-dir ../cm-evidence/preflight

Every check is one of the repository's own validators, applied where the run was produced (normally
on the cluster) before anything is packed, uploaded or published:

  stage_chain       scripts/archive_run.verify_stage_chain: eight sealed stages, prepare to evaluate
  scope             config.json and prepared/metadata.json: this dataset, full profile, scientific run, benchmark data
  runtime_source    the run's exact producing source, current checkout or verified historical snapshot
  data_roles        the registered 60/20/10/5/5 protocol with D_compare reserved at preparation
  role_isolation    cross_morse.role_evidence.audit_role_isolation
  model_inventory   sealed train/model_inventory.json against Table S5 at printed precision
  numeric_replay    cross_morse.review.replay_results: metrics, paired units, intervals and decisions recomputed
  selector          the registered selector reproduces the sealed provisional choice
  full_audit        scripts/scientific_evidence_audit.audit_run, including support reconstruction (needs PyTorch)
  manuscript        compare_with_manuscript and a provisional result binding: coverage and agreement

Verdicts: READY (admissible evidence whose numbers bind to the printed manuscript), INTEGRITY_ONLY
(admissible evidence whose numbers do not, or cannot yet, bind to the printed values), PENDING_FULL_AUDIT
(everything that ran passed, but the PyTorch-dependent support reconstruction did not run here) and
BLOCKED (the scientific audit would refuse the run).  Nothing is written into a run directory or the
repository; provisional bindings are written to a temporary directory and only reported.
Exit status: 0 when every examined dataset is READY, 1 otherwise, 2 for invalid input.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

SCHEMA = "cross-morse-evidence-preflight-1"
DATASETS = ("ogbn-arxiv", "ogbn-proteins", "ogbn-products", "Roman-empire", "PascalVOC-SP", "COCO-SP", "Peptides-func")
STEMS = {"ogbn-arxiv": "arxiv", "ogbn-proteins": "proteins", "ogbn-products": "products", "Roman-empire": "roman_empire",
         "PascalVOC-SP": "pascalvoc_sp", "COCO-SP": "coco_sp", "Peptides-func": "peptides_func"}
RELATIONSHIPS = ("exact_result_producing_run", "refactor_with_prediction_equivalence", "post_result_revision")
INTEGRITY = ("stage_chain", "sealed_inventory", "scope", "runtime_source", "data_roles", "role_isolation",
             "model_inventory", "numeric_replay", "selector", "content_stability")
VERDICT_ORDER = ("READY", "INTEGRITY_ONLY", "PENDING_FULL_AUDIT", "BLOCKED")


class PreflightError(ValueError):
    """A run or plan cannot be examined as declared."""


def canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validation_context() -> dict:
    """Bind a verdict to the validators and reference inputs that actually produced it.

    Runtime source/configuration/units identify a protocol, not the resulting observations.
    The separate stage-content binding below covers those observations. Generated reports,
    documentation and release manifests do not change this validation context.
    """
    from scripts.archive_run import sha256_file
    from scripts._paper_common import runtime_source_fingerprint
    paths = sorted((ROOT / "scripts").glob("*.py")) + sorted((ROOT / "paper" / "build").glob("*.py"))
    if any(path.is_symlink() for path in paths):
        raise PreflightError("validation source files cannot be symlinks")
    return {"runtime_source_sha256": runtime_source_fingerprint(ROOT),
            "validation_source_sha256": canonical_sha256(
                {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}),
            "manuscript_inputs": {name: sha256_file(ROOT / "paper" / name)
                                  for name in ("registered_constants.json", "reported_results.json", "sources.json")}}


def stage_content_sha256(path: Path) -> str:
    """Hash all eight stage descriptors, which bind every sealed output and parent.

    This digest is meaningful only together with a successful byte/chain verification;
    both the preflight and the archive verifier perform that check independently.
    """
    from scripts.archive_run import STAGES, sha256_file
    from cross_morse.review import safe_file
    rows = []
    for index, stage in enumerate(STAGES):
        relative = f"provenance/stages/{index:02d}_{stage}.json"
        source = safe_file(path, relative)
        rows.append({"path": relative, "bytes": source.stat().st_size, "sha256": sha256_file(source)})
    return canonical_sha256(rows)


def archive_stage_content_sha256(manifest: dict):
    """The same digest from a byte- and stage-verified archive inventory."""
    from scripts.archive_run import STAGES
    inventory = {row["path"]: row for row in manifest["files"]}

    def one(prefix):
        rows = []
        for index, stage in enumerate(STAGES):
            relative = f"provenance/stages/{index:02d}_{stage}.json"
            row = inventory[prefix + relative]
            rows.append({"path": relative, "bytes": row["bytes"], "sha256": row["sha256"]})
        return canonical_sha256(rows)

    if manifest.get("kind") == "campaign":
        return [{"split": f"split_{i:02d}", "sha256": one(f"split_{i:02d}/")} for i in range(10)]
    return one("")


def plain(value, depth: int = 0):
    """JSON-safe copy of validator output (NumPy scalars, tuples, paths); large arrays are summarized."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - NumPy is a declared dependency
        np = None
    if depth > 12:
        return str(value)
    if np is not None and isinstance(value, np.generic):
        value = value.item()
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist() if value.size <= 64 else {"array_shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): plain(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        return [plain(v, depth + 1) for v in items[:200]] + ([f"... {len(items) - 200} more"] if len(items) > 200 else [])
    return str(value)


def run_check(name: str, function) -> dict:
    """Run one validator and record PASS/FAIL/NOT_RUN without raising."""
    try:
        return {"check": name, "status": "PASS", "detail": plain(function())}
    except ModuleNotFoundError as exc:
        return {"check": name, "status": "NOT_RUN", "detail": f"module {exc.name!r} is not installed in this environment"}
    except Exception as exc:  # every validator raises its own ValueError/EvidenceError subclass
        return {"check": name, "status": "FAIL", "detail": f"{type(exc).__name__}: {exc}"}


def skipped(name: str, reason: str) -> dict:
    return {"check": name, "status": "SKIPPED", "detail": reason}


def tree_size(path: Path) -> dict:
    total = files = 0
    for folder, directories, names in os.walk(path):
        directories.sort()
        if any((Path(folder) / name).is_symlink() for name in directories):
            raise PreflightError(f"symlink directories are not archived: {folder}")
        for name in names:
            item = Path(folder) / name
            if item.is_symlink():
                raise PreflightError(f"symlinks are not archived: {item}")
            total += item.stat().st_size
            files += 1
    return {"bytes": total, "files": files}


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def examine_run(path: Path, dataset: str, *, full_audit: str = "auto") -> dict:
    """Apply the audit's per-run validators to one sealed run directory."""
    path = Path(path).resolve()
    checks: list[dict] = []
    chain: dict = {}
    content_before: str | None = None

    def stage_chain():
        nonlocal content_before
        from scripts.archive_run import verify_stage_chain
        content_before = stage_content_sha256(path)
        chain.update(verify_stage_chain(path))
        return {key: chain[key] for key in ("stages", "sealed_artifacts", "source_sha256", "config_sha256", "unit_sha256")}

    later = ("scope", "runtime_source", "data_roles", "role_isolation", "model_inventory", "numeric_replay",
             "selector", "full_audit", "content_stability")
    checks.append(run_check("stage_chain", stage_chain))
    if checks[-1]["status"] != "PASS":
        checks += [skipped("sealed_inventory", "requires a complete sealed stage chain")]
        checks += [skipped(name, "requires a complete sealed stage chain") for name in later]
        return {"path": str(path), "checks": checks, "identities": None, "size": None}

    opened: dict = {}

    def sealed_inventory():
        from cross_morse.review import NumericEvidence
        opened["evidence"] = NumericEvidence(path)
        opened["config"] = opened["evidence"].json("config.json")
        opened["metadata"] = opened["evidence"].json("prepared/metadata.json")
        return {"sealed_outputs": len(opened["evidence"].sealed), "identities": opened["evidence"].identities}

    checks.append(run_check("sealed_inventory", sealed_inventory))
    if checks[-1]["status"] != "PASS":
        checks += [skipped(name, "requires a readable sealed inventory") for name in later]
        return {"path": str(path), "checks": checks, "identities": None, "size": None}
    evidence, config, metadata = opened["evidence"], opened["config"], opened["metadata"]

    def scope():
        problems = []
        if config.get("dataset") != dataset:
            problems.append(f"config.json names {config.get('dataset')!r}, the plan names {dataset!r}")
        if config.get("profile") != "full":
            problems.append(f"profile is {config.get('profile')!r}; official evidence needs the full profile")
        if config.get("scientific_run") is not True:
            problems.append("scientific_run is not true")
        if metadata.get("is_benchmark_data") is not True:
            problems.append("prepared/metadata.json does not mark official benchmark data")
        if problems:
            raise PreflightError("; ".join(problems))
        return {"dataset": dataset, "profile": "full", "scientific_run": True}

    def runtime_source():
        from scripts._paper_common import runtime_source_fingerprint
        current = runtime_source_fingerprint(ROOT)
        produced = chain["source_sha256"]
        if produced == current:
            return {"run_source_sha256": produced, "location": "current_checkout"}
        from scripts._source_snapshots import source_resolution
        try:
            resolved = source_resolution(produced, ROOT)
        except ValueError:
            raise PreflightError(
                f"run source {produced[:12]} is not this checkout ({current[:12]}) and no verified snapshot of it is "
                "bundled; supply the byte-verified producing runtime under examples/source_snapshots/<sha256>/ "
                "before claiming the run is ready for reconstruction") from None
        return {"run_source_sha256": produced, "location": resolved["source_location"]}

    def data_roles():
        from scripts.scientific_evidence_audit import validate_split_evidence
        graph = evidence.arrays("prepared/graph_splits.npz") if metadata.get("independent_graphs") else None
        return validate_split_evidence(metadata, evidence.arrays("prepared/splits.npz"),
                                       evidence.arrays("prepared/official_splits.npz"), graph)

    def role_isolation():
        from cross_morse.role_evidence import audit_role_isolation
        result = audit_role_isolation(path)
        if not result.get("passed"):
            raise PreflightError("; ".join(result.get("errors", [])) or "role isolation failed")
        return {key: value for key, value in result.items() if key != "errors"}

    def model_inventory():
        from scripts.scientific_evidence_audit import validate_model_inventory
        registered = json.loads((ROOT / "paper" / "registered_constants.json").read_text(encoding="utf-8"))
        reported = registered["backbones_table_S5"][dataset]["parameters_M"]
        result = validate_model_inventory(evidence.json("train/model_inventory.json"), reported, configuration=config)
        return {"reported_parameters_M": result["reported_parameters_M"], "counts": result["counts"]}

    def numeric_replay():
        from scripts._source_snapshots import replay_results_exact
        result = replay_results_exact(path, selection=True, root=ROOT)
        return {"admitted": result["admitted"], "deployed_index": result["deployed_index"],
                "comparator_index": result["comparator_index"],
                "paired_qD_minus_qC_loss": {k: result["paired_qD_minus_qC_loss"].get(k) for k in ("mean", "lower", "upper")},
                "paired_qD_minus_qC_metric_pp": {k: result["paired_qD_minus_qC_metric_pp"].get(k) for k in ("mean", "lower", "upper")},
                "metric_percent": {name: row["metric_percent"] for name, row in result["models"].items()}}

    def selector():
        from paper.build.evidence import evidence_from_run
        from paper.build.selection import selection_readings
        readings = selection_readings(evidence_from_run(path))
        if not readings.get("available") or not readings.get("registered_all_configurations_reproduces_selection"):
            raise PreflightError("the registered selector does not reproduce the sealed provisional choice")
        return {"reproduces_selection": True}

    def audit():
        from scripts.scientific_evidence_audit import audit_run
        result = audit_run(path, official_dataset=dataset, manuscript_root=ROOT)
        if result["status"] != "PASS":
            failing = [profile for profile, row in result["support"].items() if row["status"] != "PASS"]
            raise PreflightError("support reconstruction failed for profile(s): " + ", ".join(failing))
        return {"status": "PASS", "sealed_artifacts": result["sealed_artifacts"],
                "support_profiles": sorted(result["support"])}

    for name, function in (("scope", scope), ("runtime_source", runtime_source), ("data_roles", data_roles),
                           ("role_isolation", role_isolation), ("model_inventory", model_inventory),
                           ("numeric_replay", numeric_replay), ("selector", selector)):
        checks.append(run_check(name, function))
    if full_audit == "never":
        checks.append({"check": "full_audit", "status": "NOT_RUN", "detail": "disabled with --full-audit never"})
    elif full_audit == "auto" and not torch_available():
        checks.append({"check": "full_audit", "status": "NOT_RUN",
                       "detail": "PyTorch is not installed here; run the preflight where the runs were produced"})
    else:
        checks.append(run_check("full_audit", audit))
    def content_stability():
        from scripts.archive_run import verify_stage_chain
        verify_stage_chain(path)
        after = stage_content_sha256(path)
        if after != content_before:
            raise PreflightError("sealed content changed during the preflight; examine an immutable completed run")
        return {"sealed_content_sha256": after, "scope": "all eight stage descriptors and their verified sealed outputs"}

    checks.append(run_check("content_stability", content_stability))
    identities = {key: chain[key] for key in ("source_sha256", "config_sha256", "unit_sha256")}
    return {"path": str(path), "checks": checks, "identities": identities,
            "sealed_content_sha256": content_before, "size": tree_size(path)}


def manuscript_agreement(dataset: str, paths: list[Path], kind: str, *, relationship: str | None,
                         identifier: str, workdir: Path) -> dict:
    """Compare with the printed values and evaluate a provisional binding exactly as the audit would."""
    from scripts._paper_common import load_reported
    from scripts.scientific_evidence_audit import binding_coverage
    reported = load_reported()
    declared = relationship if relationship in RELATIONSHIPS else None
    provisional = declared or "exact_result_producing_run"
    result = {"relationship": declared, "relationship_used_for_check": provisional}
    if kind == "campaign":
        from scripts.compare_with_manuscript import compare_roman_campaign
        from scripts.emit_campaign_binding import build_binding, check as check_campaign, split_identities
        comparison = compare_roman_campaign(paths[0].parent, reported)
        binding = build_binding(split_identities(paths), relationship=provisional, identifier=identifier)
        checked = check_campaign(paths, binding)
        missing, status = checked["coverage_missing"], checked["status"]
        rows = checked["results"]
    else:
        from scripts.check_result_binding import check_binding
        from scripts.compare_with_manuscript import RunReader, compare, emit_binding
        run = RunReader(paths[0])
        run.verify()
        comparison = compare(run, reported)
        target = workdir / f"{STEMS[dataset]}_full.binding.json"
        emit_binding(run, reported, comparison, target, provisional, identifier, 1.0)
        binding = json.loads(target.read_text(encoding="utf-8"))
        missing = binding_coverage(binding, campaign=False, dataset=dataset, manuscript_root=ROOT)
        checked = check_binding(paths[0], binding)
        status = "INCOMPLETE" if missing else checked["status"]
        rows = [{k: r.get(k) for k in ("manuscript_location", "json_pointer", "reported_value", "actual_value",
                                       "absolute_tolerance", "agrees")} for r in checked["results"]]
    result.update({
        "comparison_levels": comparison.get("levels"),
        "comparison_rows_not_exact": [
            {k: row.get(k) for k in ("section", "quantity", "run_value", "reported_value", "status", "manuscript_location")}
            for row in comparison.get("rows", []) if row.get("status") not in ("exact", "match")],
        "binding_status": status, "binding_coverage_missing": missing, "binding_results": rows,
        "note": ("Binding values are the printed manuscript values; MATCH means the run reproduces them at printed "
                 "precision. INCOMPLETE lists binding targets the audit requires but no sealed artifact supplies."),
    })
    return result


def verdict(report: dict) -> tuple[str, list[str], list[str]]:
    blockers, warnings = [], []
    full = []
    runs = report.get("runs", [])
    if not runs:
        blockers.append("no completed runs were examined")
    for run in runs:
        label = Path(run["path"]).name
        names = [check.get("check") for check in run["checks"]]
        for name in INTEGRITY:
            if names.count(name) != 1:
                blockers.append(f"{label}: exactly one {name} check is required")
        if names.count("full_audit") != 1:
            full.append("NOT_RUN")
        for check in run["checks"]:
            if check["check"] in INTEGRITY and check["status"] != "PASS":
                blockers.append(f"{label}: {check['check']} {check['status']}: {check['detail']}")
            elif check["check"] == "full_audit":
                full.append(check["status"])
                if check["status"] == "FAIL":
                    blockers.append(f"{label}: full_audit FAIL: {check['detail']}")
            elif check["status"] == "WARN":
                warnings.append(f"{label}: {check['check']}: {check['detail']}")
    if report.get("structure_error"):
        blockers.append(report["structure_error"])
    if blockers:
        return "BLOCKED", blockers, warnings
    if not full or any(status != "PASS" for status in full):
        warnings.append("support reconstruction (full audit) did not run here; repeat where PyTorch is installed")
        return "PENDING_FULL_AUDIT", blockers, warnings
    manuscript = report.get("manuscript") or {}
    if manuscript.get("error"):
        return "INTEGRITY_ONLY", [f"manuscript comparison could not run: {manuscript['error']}"], warnings
    if manuscript.get("binding_status") != "MATCH" or manuscript.get("binding_coverage_missing"):
        reasons = [f"binding {manuscript.get('binding_status')}"]
        reasons += [f"binding target without sealed artifact: {m}" for m in manuscript.get("binding_coverage_missing", [])]
        reasons += [f"{r['json_pointer']}: run {r['actual_value']} vs printed {r['reported_value']} (tolerance {r['absolute_tolerance']})"
                    for r in manuscript.get("binding_results", []) if r.get("agrees") is False]
        return "INTEGRITY_ONLY", reasons, warnings
    if not manuscript.get("relationship"):
        return "INTEGRITY_ONLY", ["declare relationship_to_manuscript; numerical agreement alone does not establish "
                                  "that these are the manuscript's producing runs"], warnings
    return "READY", [], warnings


def preflight_dataset(dataset: str, root: Path, kind: str, *, relationship: str | None = None,
                      identifier: str = "CROSS-MORSE manuscript identified by SHA-256 in paper/sources.json",
                      full_audit: str = "auto", manuscript: bool = True) -> dict:
    if dataset not in DATASETS:
        raise PreflightError(f"unknown dataset {dataset!r}")
    if relationship is not None and relationship not in RELATIONSHIPS:
        raise PreflightError("declare a supported relationship_to_manuscript")
    if full_audit not in ("auto", "always", "never"):
        raise PreflightError("full_audit must be auto, always or never")
    if kind not in ("run", "campaign") or (kind == "campaign") != (dataset == "Roman-empire"):
        raise PreflightError("Roman-empire is examined as a ten-split campaign; every other dataset as one run")
    root = Path(root).resolve()
    if not root.is_dir():
        raise PreflightError(f"run directory does not exist: {root}")
    report = {"schema": SCHEMA, "dataset": dataset, "kind": kind, "root": str(root),
              "examined_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
              "validation_context": validation_context(), "runs": [], "manuscript": None,
              "relationship_to_manuscript": relationship, "manuscript_identifier": identifier}
    report["checkout_runtime_source_sha256"] = report["validation_context"]["runtime_source_sha256"]
    if kind == "campaign":
        paths = [root / f"split_{index:02d}" for index in range(10)]
        from scripts.emit_campaign_binding import CampaignBindingError, split_paths, split_identities
        try:
            paths = split_paths(root)
        except CampaignBindingError as exc:
            report["structure_error"] = str(exc)
            paths = [p for p in paths if p.is_dir()]
    else:
        paths = [root]
    for path in paths:
        report["runs"].append(examine_run(path, dataset, full_audit=full_audit))
    integrity_ok = not report.get("structure_error") and all(
        check["status"] == "PASS" for run in report["runs"] for check in run["checks"] if check["check"] in INTEGRITY)
    if kind == "campaign" and integrity_ok:
        try:
            split_identities(paths)
        except (CampaignBindingError, OSError, ValueError, KeyError, TypeError) as exc:
            report["structure_error"] = f"campaign identity: {exc}"
            integrity_ok = False
    if manuscript and integrity_ok:
        with tempfile.TemporaryDirectory(prefix="cm-preflight-") as tmp:
            try:
                report["manuscript"] = manuscript_agreement(dataset, paths, kind, relationship=relationship,
                                                            identifier=identifier, workdir=Path(tmp))
            except Exception as exc:  # comparison tooling raises ValueError/EvidenceError subclasses
                report["manuscript"] = {"error": f"{type(exc).__name__}: {exc}"}
    sizes = [run["size"]["bytes"] for run in report["runs"] if run.get("size")]
    report["estimated_archive_input_bytes"] = sum(sizes) if sizes else None
    identities = [run["identities"] for run in report["runs"] if run.get("identities")]
    report["identities"] = identities[0] if kind == "run" and identities else identities
    contents = [{"split": Path(run["path"]).name, "sha256": run.get("sealed_content_sha256")}
                for run in report["runs"]]
    report["sealed_content_sha256"] = (contents[0]["sha256"] if kind == "run" and contents else contents)
    if validation_context() != report["validation_context"]:
        report["structure_error"] = "validation code or manuscript reference inputs changed during preflight"
    report["verdict"], report["blockers"], report["warnings"] = verdict(report)
    return plain(report)


def load_plan(path: Path) -> dict:
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if plan.get("schema") != "cross-morse-evidence-plan-1":
        raise PreflightError("the plan must use schema cross-morse-evidence-plan-1 (scripts/evidence_pipeline.py template)")
    datasets = plan.get("datasets")
    if not isinstance(datasets, dict) or not set(datasets) <= set(DATASETS):
        raise PreflightError("plan.datasets must map official dataset names to their run directories")
    return plan


def _cell(value) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text.replace("|", "\\|").replace("\n", " ")[:300]


def markdown(reports: list[dict]) -> str:
    checkout = reports[0]["checkout_runtime_source_sha256"] if reports else "—"
    lines = ["# Evidence preflight", "", f"Checkout runtime source identity: `{checkout}`.", "",
             "| Dataset | Verdict | Runs | Full audit | Binding | Input size (GB) |", "|---|---|---|---|---|---|"]
    for report in reports:
        full = sorted({c["status"] for run in report["runs"] for c in run["checks"] if c["check"] == "full_audit"})
        manuscript = report.get("manuscript") or {}
        binding = manuscript.get("binding_status") or ("error" if manuscript.get("error") else "—")
        size = report.get("estimated_archive_input_bytes")
        gigabytes = f"{size / 1e9:.2f}" if isinstance(size, (int, float)) else "—"
        lines.append(f"| {report['dataset']} | **{report['verdict']}** | {len(report['runs'])} | "
                     f"{'/'.join(full) or '—'} | {binding} | {gigabytes} |")
    for report in reports:
        lines += ["", f"## {report['dataset']} — {report['verdict']}", "", f"Root: `{report['root']}`", ""]
        if report["blockers"]:
            lines += ["Blocking:", ""] + [f"- {_cell(item)}" for item in report["blockers"]] + [""]
        if report["warnings"]:
            lines += ["Advisory:", ""] + [f"- {_cell(item)}" for item in report["warnings"]] + [""]
        lines += ["| Run | Check | Status | Detail |", "|---|---|---|---|"]
        for run in report["runs"]:
            for check in run["checks"]:
                lines.append(f"| {Path(run['path']).name} | {check['check']} | {check['status']} | {_cell(check['detail'])} |")
        manuscript = report.get("manuscript") or {}
        if manuscript.get("error"):
            lines += ["", f"Manuscript comparison error: {_cell(manuscript['error'])}"]
        elif manuscript.get("comparison_levels") is not None:
            lines += ["", f"Manuscript comparison levels: `{json.dumps(manuscript['comparison_levels'])}`; binding "
                      f"**{manuscript.get('binding_status')}** (relationship used: {manuscript.get('relationship_used_for_check')})."]
            for missing in manuscript.get("binding_coverage_missing", []):
                lines.append(f"- binding target without a sealed artifact: `{missing}`")
            for row in manuscript.get("binding_results", []):
                if row.get("agrees") is False:
                    lines.append(f"- `{row['json_pointer']}`: run {row['actual_value']} vs printed {row['reported_value']}")
    return "\n".join(lines) + "\n"


def write_outputs(reports: list[dict], output_dir: Path | None, *, merge: bool = False) -> None:
    """Write one JSON report per dataset; with merge, rebuild the summary from every report in the directory."""
    if output_dir is None:
        return
    output_dir = Path(output_dir).resolve()
    if output_dir.is_relative_to(ROOT):
        raise PreflightError("write preflight reports outside the repository checkout")
    if any(output_dir.is_relative_to(Path(report["root"]).resolve()) for report in reports):
        raise PreflightError("preflight reports cannot be written inside an examined run or campaign")
    output_dir.mkdir(parents=True, exist_ok=True)
    for report in reports:
        atomic_text(output_dir / f"{STEMS[report['dataset']]}.json", json.dumps(report, indent=1, allow_nan=False) + "\n")
    if merge:
        found = {}
        for path in sorted(output_dir.glob("*.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(value, dict) and value.get("schema") == SCHEMA and value.get("dataset") in DATASETS:
                found[value["dataset"]] = value
        reports = [found[d] for d in DATASETS if d in found]
    atomic_text(output_dir / "PREFLIGHT.md", markdown(reports))
    atomic_text(output_dir / "summary.json", json.dumps(
        {"schema": SCHEMA + "-summary", "datasets": {r["dataset"]: {"verdict": r["verdict"], "identities": r["identities"],
                                                                     "blockers": r["blockers"]} for r in reports}},
        indent=1, allow_nan=False) + "\n")


def atomic_text(path: Path, text: str) -> None:
    """Publish complete reports; interruption leaves the preceding report intact."""
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, help="evidence plan (scripts/evidence_pipeline.py template)")
    parser.add_argument("--dataset", choices=DATASETS, action="append", help="restrict to these datasets (repeatable)")
    parser.add_argument("--run", type=Path, help="one completed run directory (with a single --dataset)")
    parser.add_argument("--campaign", type=Path, help="the Roman-empire campaign directory (split_00 ... split_09)")
    parser.add_argument("--relationship", choices=RELATIONSHIPS, help="declared relationship to the manuscript")
    parser.add_argument("--full-audit", choices=("auto", "always", "never"), default="auto",
                        help="run the PyTorch-dependent support reconstruction (auto: when PyTorch is installed)")
    parser.add_argument("--no-manuscript", action="store_true", help="skip the manuscript comparison and binding check")
    parser.add_argument("--output-dir", type=Path, help="directory (outside the checkout) for JSON and Markdown reports")
    args = parser.parse_args(argv)
    try:
        jobs = []
        if args.plan:
            plan = load_plan(args.plan)
            for dataset, entry in plan["datasets"].items():
                if args.dataset and dataset not in args.dataset:
                    continue
                if not isinstance(entry, dict) or not entry.get("run_root"):
                    raise PreflightError(f"{dataset}: plan entry needs run_root")
                kind = "campaign" if dataset == "Roman-empire" else "run"
                jobs.append((dataset, Path(entry["run_root"]), kind,
                             entry.get("relationship_to_manuscript", plan.get("relationship_to_manuscript")),
                             plan.get("manuscript_identifier") or "CROSS-MORSE manuscript identified by SHA-256 in paper/sources.json"))
        else:
            if not args.dataset or len(args.dataset) != 1 or bool(args.run) == bool(args.campaign):
                parser.error("give --plan, or one --dataset with exactly one of --run / --campaign")
            dataset = args.dataset[0]
            jobs.append((dataset, args.campaign or args.run, "campaign" if args.campaign else "run", args.relationship,
                         "CROSS-MORSE manuscript identified by SHA-256 in paper/sources.json"))
        if not jobs:
            raise PreflightError("nothing to examine")
        reports = []
        for dataset, root, kind, relationship, identifier in jobs:
            print(f"examining {dataset}: {root}", file=sys.stderr)
            reports.append(preflight_dataset(dataset, root, kind, relationship=relationship, identifier=identifier,
                                             full_audit=args.full_audit, manuscript=not args.no_manuscript))
        write_outputs(reports, args.output_dir)
    except (PreflightError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(markdown(reports))
    return 0 if all(report["verdict"] == "READY" for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
