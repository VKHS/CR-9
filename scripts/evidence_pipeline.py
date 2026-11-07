#!/usr/bin/env python3
"""Register the official evidence from one plan: preflight, pack, deposit, publish, register, bind, record, status.

    python -B scripts/evidence_pipeline.py template --output ../cm-evidence/plan.json      # fill in run roots etc.
    python -B scripts/evidence_pipeline.py discover --root /abs/export --output ../cm-evidence/discovery.json
    python -B scripts/evidence_pipeline.py preflight --plan ../cm-evidence/plan.json        # on the cluster (PyTorch)
    python -B scripts/evidence_pipeline.py pack      --plan ...                             # verified .tar.gz per dataset
    python -B scripts/evidence_pipeline.py deposit   --plan ... [--sandbox]                 # Zenodo drafts (reserved DOIs)
    python -B scripts/evidence_pipeline.py publish   --plan ... --confirm                   # permanent public records
    python -B scripts/evidence_pipeline.py register  --plan ...                             # verified ingest + runs/index.json
    python -B scripts/evidence_pipeline.py bind      --plan ...                             # paper/bindings/<stem>_full.json
    python -B scripts/evidence_pipeline.py records   --plan ...                             # records/<set>/ via the exporters
    python -B scripts/evidence_pipeline.py status    --plan ...                             # one table of everything

Every step delegates to the repository's existing tool for that step (evidence_preflight, archive_run,
deposit_zenodo, ingest_run, compare_with_manuscript / emit_campaign_binding, export_records) and
records its outcome in <deposits_dir>/evidence-state.json.  Steps are idempotent: a verified archive
is not repacked, a Zenodo receipt resumes its own deposition, a registered archive is not re-ingested.

Gates.  Publication is permanent, so ``publish`` requires --confirm and a completed preflight for
the same sealed content, archive bytes, validation code and manuscript inputs. INTEGRITY_ONLY
(integrity checks passed, but manuscript agreement/provenance is unresolved) requires the explicit
--allow-integrity-only declaration. BLOCKED and PENDING_FULL_AUDIT never permit production publication.
Sandbox deposits (--sandbox) are rehearsals and are never registered.  Bindings are installed only when
they cover every audit target and match, and only for the relationship the plan
declares.  Without --dataset, a step skips and lists the datasets that are not ready for it (no
run_root yet, no preflight, no archive, a verdict that does not permit publication); a dataset named
with --dataset is acted on or the step stops.  Nothing is filled in from the manuscript registry.
Standard library only (the steps themselves need what their tools need).
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
from functools import wraps
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tarfile
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - POSIX clusters and workstations provide fcntl
    fcntl = None

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

PLAN_SCHEMA = "cross-morse-evidence-plan-1"
STATE_SCHEMA = "cross-morse-evidence-state-1"
DATASETS = ("ogbn-arxiv", "ogbn-proteins", "ogbn-products", "Roman-empire", "PascalVOC-SP", "COCO-SP", "Peptides-func")
STEMS = {"ogbn-arxiv": "arxiv", "ogbn-proteins": "proteins", "ogbn-products": "products", "Roman-empire": "roman_empire",
         "PascalVOC-SP": "pascalvoc_sp", "COCO-SP": "coco_sp", "Peptides-func": "peptides_func"}
RELATIONSHIPS = ("exact_result_producing_run", "refactor_with_prediction_equivalence", "post_result_revision")
RECORD_SETS = ("baselines", "budget_matched", "input_gate", "factorials", "negative_controls", "nominal_time",
               "sensor_parameters", "cross_backbone", "ensemble_grid", "inserted_site")
DEFAULT_IDENTIFIER = "CROSS-MORSE manuscript identified by SHA-256 in paper/sources.json"
LICENSE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+_-]*")


class PipelineError(ValueError):
    """The plan, the state or a step's outcome does not permit the requested action."""


class VerdictRefusal(PipelineError):
    """The preflight verdict does not permit publication without an explicit, recorded override."""


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


# ------------------------------------------------------------------ plan
def template() -> dict:
    return {
        "schema": PLAN_SCHEMA,
        "_comment": ("Copy outside the repository (it names cluster paths) and fill every null. Keys starting with "
                     "'_' are notes. docs/EVIDENCE_REGISTRATION.md explains each field."),
        "deposits_dir": None,
        "_deposits_dir": "absolute directory outside the checkout for archives, Zenodo receipts, preflight reports and state",
        "license": None,
        "_license": "Zenodo licence id that covers the archived material, e.g. cc-by-4.0; check the dataset licences first",
        "code_doi": None,
        "_code_doi": "optional Zenodo version DOI of the code snapshot (make deposit-code), linked from every archive record",
        "relationship_to_manuscript": None,
        "_relationship_to_manuscript": " | ".join(RELATIONSHIPS),
        "manuscript_identifier": DEFAULT_IDENTIFIER,
        "datasets": {dataset: {"run_root": None} for dataset in DATASETS},
        "_datasets": "run_root: the sealed full-profile run directory; for Roman-empire the campaign holding split_00 ... split_09",
        "budget_campaign": None,
        "_budget_campaign": "extracted E3-E6 campaign directory (arm_campaign export --complete); passed to the audit",
        "record_sets": {name: {"exporter": None, "mapping": None} for name in RECORD_SETS},
        "_record_sets": ("per set give either exporter: the argument list of scripts/export_records.py (budget-matched, "
                         "baselines, from-runs), or mapping: a mapping file for export_records.py from-mapping "
                         "(make record-mapping-template SET=...)"),
    }


def load_plan(path: Path) -> dict:
    path = Path(path).resolve()
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema") != PLAN_SCHEMA:
        raise PipelineError(f"{path}: schema must be {PLAN_SCHEMA} (create one with: evidence_pipeline.py template)")
    if path.is_relative_to(ROOT):
        raise PipelineError("keep the plan outside the repository checkout; it names cluster paths")
    deposits = plan.get("deposits_dir")
    if not isinstance(deposits, str) or not Path(deposits).is_absolute():
        raise PipelineError("deposits_dir must be an absolute path")
    if Path(deposits).resolve().is_relative_to(ROOT):
        raise PipelineError("deposits_dir must lie outside the repository checkout")
    datasets = plan.get("datasets")
    if not isinstance(datasets, dict) or not set(datasets) <= set(DATASETS):
        raise PipelineError("datasets must map official dataset names to {run_root: ...}")
    relationship = plan.get("relationship_to_manuscript")
    if relationship is not None and relationship not in RELATIONSHIPS:
        raise PipelineError(f"relationship_to_manuscript must be one of {', '.join(RELATIONSHIPS)}")
    for dataset, entry in datasets.items():
        if not isinstance(entry, dict):
            raise PipelineError(f"{dataset}: entry must be an object")
        override = entry.get("relationship_to_manuscript")
        if override is not None and override not in RELATIONSHIPS:
            raise PipelineError(f"{dataset}: invalid relationship_to_manuscript")
    sets = plan.get("record_sets", {}) or {}
    if not isinstance(sets, dict) or not set(sets) <= set(RECORD_SETS):
        raise PipelineError(f"record_sets keys must be among {', '.join(RECORD_SETS)}")
    for name, entry in sets.items():
        if not isinstance(entry, dict):
            raise PipelineError(f"{name}: record set plan must be an object")
    for dataset, entry in datasets.items():
        root = entry.get("run_root")
        if root is not None and (not isinstance(root, str) or not Path(root).is_absolute()):
            raise PipelineError(f"{dataset}: run_root must be an absolute path or null")
        if root is not None and Path(deposits).resolve().is_relative_to(Path(root).resolve()):
            raise PipelineError("deposits_dir cannot lie inside a source run or campaign")
    plan["_path"] = str(path)
    return plan


def selected(plan: dict, wanted: list[str] | None, *, need_root: bool) -> list[str]:
    """Planned datasets in the fixed order.  Without --dataset, a dataset whose run_root is still null is
    skipped with a note (runs can be registered as they become ready); a named one, or a run_root that is
    set but not an existing absolute directory, stops the step."""
    names = [d for d in DATASETS if d in plan["datasets"] and (not wanted or d in wanted)]
    if wanted and set(wanted) - set(names):
        raise PipelineError(f"not in the plan: {', '.join(sorted(set(wanted) - set(names)))}")
    if need_root:
        kept = []
        for dataset in names:
            root = plan["datasets"][dataset].get("run_root")
            if root is None and not wanted:
                print(f"skip {dataset}: no run_root in the plan yet", file=sys.stderr)
                continue
            if not isinstance(root, str) or not Path(root).is_absolute() or not Path(root).is_dir():
                raise PipelineError(f"{dataset}: run_root must be an existing absolute directory (got {root!r})")
            kept.append(dataset)
        names = kept
    return names


def relationship_for(plan: dict, dataset: str) -> str | None:
    return plan["datasets"][dataset].get("relationship_to_manuscript") or plan.get("relationship_to_manuscript")


def kind_of(dataset: str) -> str:
    return "campaign" if dataset == "Roman-empire" else "run"


def deposits_dir(plan: dict) -> Path:
    path = Path(plan["deposits_dir"]).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def archive_path(plan: dict, dataset: str) -> Path:
    return deposits_dir(plan) / f"{STEMS[dataset]}_full.tar.gz"


# ------------------------------------------------------------------ state
def state_path(plan: dict) -> Path:
    return deposits_dir(plan) / "evidence-state.json"


def load_state(plan: dict) -> dict:
    path = state_path(plan)
    if path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("schema") != STATE_SCHEMA:
            raise PipelineError(f"{path}: unexpected state schema")
        if (not isinstance(state.get("datasets"), dict) or not isinstance(state.get("record_sets"), dict)
                or not isinstance(state.get("history"), list)):
            raise PipelineError(f"{path}: invalid state structure")
        return state
    return {"schema": STATE_SCHEMA, "datasets": {}, "record_sets": {}, "history": []}


def save_state(plan: dict, state: dict) -> None:
    from scripts.evidence_preflight import atomic_text
    atomic_text(state_path(plan), json.dumps(state, indent=1, sort_keys=True, allow_nan=False) + "\n")


@contextmanager
def locked(plan: dict, resource: str = "state"):
    """Serialize state, same-dataset actions, or checkout mutations for this plan."""
    if fcntl is None:
        raise PipelineError("POSIX file locking is required; run the pipeline on Linux or macOS")
    if not re.fullmatch(r"[a-z0-9_-]+", resource):
        raise PipelineError("invalid lock resource")
    lock = deposits_dir(plan) / f".evidence-{resource}.lock"
    if lock.is_symlink():
        raise PipelineError("evidence lock cannot be a symlink")
    with lock.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def serialized_datasets(function):
    """Different datasets can run in parallel; two writers of one archive cannot."""
    @wraps(function)
    def call(plan, datasets, args, **kwargs):
        rows = []
        for dataset in datasets:
            with locked(plan, "dataset-" + STEMS[dataset]):
                rows.extend(function(plan, [dataset], args, **kwargs))
        return rows
    return call


def serialized_checkout(function):
    @wraps(function)
    def call(plan, *args, **kwargs):
        with locked(plan, "checkout"):
            return function(plan, *args, **kwargs)
    return call


def record(plan: dict, dataset: str, step: str, value: dict, *, record_set: bool = False) -> None:
    """Read-modify-write one entry under the lock; other datasets' entries are never overwritten."""
    with locked(plan):
        state = load_state(plan)
        entry = {**value, "updated_utc": now()}
        if record_set:
            state["record_sets"][dataset] = entry
        else:
            state["datasets"].setdefault(dataset, {})[step] = entry
        state["history"].append({"utc": now(), "dataset": dataset, "step": step,
                                 "status": value.get("status") or value.get("verdict")})
        save_state(plan, state)


def current(plan: dict, dataset: str) -> dict:
    with locked(plan):
        return load_state(plan)["datasets"].get(dataset, {})


# ------------------------------------------------------------------ subprocess helper
def tool(argv: list[str], *, check: bool = True, env: dict | None = None) -> subprocess.CompletedProcess:
    command = [sys.executable, "-B", *argv]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, env=env)
    if check and result.returncode != 0:
        tail = "\n".join((result.stdout + "\n" + result.stderr).strip().splitlines()[-20:])
        raise PipelineError(f"{' '.join(argv[:2])} exited {result.returncode}:\n{tail}")
    return result


# ------------------------------------------------------------------ identities
def archive_identities(manifest: dict):
    if manifest.get("kind") == "campaign":
        return [{k: manifest["splits"][name]["chain"][k] for k in ("source_sha256", "config_sha256", "unit_sha256")}
                for name in sorted(manifest["splits"])]
    return {k: manifest["chain"][k] for k in ("source_sha256", "config_sha256", "unit_sha256")}


def preflight_identities(report: dict):
    value = report.get("identities")
    if isinstance(value, list):
        return [{k: row[k] for k in ("source_sha256", "config_sha256", "unit_sha256")} for row in value]
    return {k: value[k] for k in ("source_sha256", "config_sha256", "unit_sha256")} if value else None


def _preflight_context(preflight: dict, plan: dict | None = None, dataset: str | None = None) -> None:
    from scripts.evidence_preflight import validation_context
    if preflight.get("validation_context") != validation_context():
        raise PipelineError("preflight validation code or manuscript reference inputs changed; rerun preflight")
    if plan is not None and (preflight.get("relationship_to_manuscript") != relationship_for(plan, dataset)
                             or preflight.get("manuscript_identifier") != (plan.get("manuscript_identifier") or DEFAULT_IDENTIFIER)):
        raise PipelineError("the plan's manuscript declaration changed; rerun preflight")
    if plan is not None and preflight.get("root") != str(Path(plan["datasets"][dataset]["run_root"]).resolve()):
        raise PipelineError("the planned source run changed; rerun preflight")


def _verified_archive(dataset: str, archive: dict) -> dict:
    """Re-read exact bytes; state metadata alone never authorizes a deposit or registration."""
    from scripts.deposit_zenodo import verified_run_manifest
    from scripts.evidence_preflight import archive_stage_content_sha256
    path = Path(archive["path"])
    if path.is_symlink() or not path.is_file():
        raise PipelineError(f"{dataset}: recorded archive is missing or is a symlink")
    manifest = verified_run_manifest(path)
    if manifest["dataset"] != dataset or manifest["kind"] != kind_of(dataset):
        raise PipelineError(f"{dataset}: archive dataset/kind differs from the plan")
    if (manifest["archive_sha256"] != archive.get("sha256")
            or manifest["archive_bytes"] != archive.get("bytes")):
        raise PipelineError(f"{dataset}: archive bytes changed after packing; preserve the old receipt and pack/review the intended run")
    if archive_identities(manifest) != archive.get("identities"):
        raise PipelineError(f"{dataset}: recorded archive identities do not match its verified bytes")
    if (manifest.get("profile") != archive.get("profile")
            or manifest.get("scientific_run") is not archive.get("scientific_run")):
        raise PipelineError(f"{dataset}: recorded archive scope differs from its verified bytes")
    if archive_stage_content_sha256(manifest) != archive.get("sealed_content_sha256"):
        raise PipelineError(f"{dataset}: recorded archive sealed content does not match its verified bytes")
    return manifest


def _materialized_content(dataset: str, path: Path):
    from scripts.evidence_preflight import stage_content_sha256
    if dataset == "Roman-empire":
        from scripts.emit_campaign_binding import split_paths
        return [{"split": item.name, "sha256": stage_content_sha256(item)} for item in split_paths(path)]
    return stage_content_sha256(path)


def _bound_receipt(archive: Path, archived: dict, sandbox: bool) -> dict:
    from scripts.deposit_zenodo import PRODUCTION, SANDBOX, RECEIPT_SCHEMA
    receipt = _receipt(archive, sandbox)
    if not isinstance(receipt, dict) or receipt.get("schema") != RECEIPT_SCHEMA:
        raise PipelineError("deposit did not produce a recognized receipt")
    if receipt.get("service") != (SANDBOX if sandbox else PRODUCTION):
        raise PipelineError("deposit receipt belongs to a different service")
    if (receipt.get("filename") != archive.name or receipt.get("sha256") != archived["sha256"]
            or receipt.get("bytes") != archived["bytes"]):
        raise PipelineError("deposit receipt does not bind these archive bytes")
    if receipt.get("status") not in ("draft", "publishing", "published") or type(receipt.get("published")) is not bool:
        raise PipelineError("deposit receipt has no completed draft/publication outcome; resume the deposit tool")
    if receipt["published"] != (receipt["status"] == "published"):
        raise PipelineError("deposit receipt has inconsistent publication state")
    prefix = "10.5072" if sandbox else "10.5281"
    if receipt["published"] and (not re.fullmatch(re.escape(prefix) + r"/zenodo\.[1-9][0-9]*", str(receipt.get("doi") or ""))
                                  or receipt.get("doi") != receipt.get("reserved_doi")):
        raise PipelineError("published DOI does not match the receipt's service and reservation")
    return receipt


# ------------------------------------------------------------------ steps
@serialized_datasets
def step_preflight(plan: dict, datasets: list[str], args) -> list[dict]:
    from scripts.evidence_preflight import preflight_dataset, write_outputs
    reports = []
    for dataset in datasets:
        root = Path(plan["datasets"][dataset]["run_root"])
        print(f"preflight {dataset}: {root}", file=sys.stderr)
        report = preflight_dataset(dataset, root, kind_of(dataset), relationship=relationship_for(plan, dataset),
                                   identifier=plan.get("manuscript_identifier") or DEFAULT_IDENTIFIER,
                                   full_audit=args.full_audit)
        reports.append(report)
        with locked(plan):
            write_outputs([report], deposits_dir(plan) / "preflight", merge=True)
        record(plan, dataset, "preflight", {"verdict": report["verdict"], "identities": report["identities"],
                                            "sealed_content_sha256": report["sealed_content_sha256"],
                                            "validation_context": report["validation_context"],
                                            "relationship_to_manuscript": report["relationship_to_manuscript"],
                                            "manuscript_identifier": report["manuscript_identifier"],
                                            "blockers": report["blockers"][:20], "warnings": report["warnings"][:20],
                                            "root": str(root.resolve()), "report": str(deposits_dir(plan) / "preflight" / f"{STEMS[dataset]}.json")})
    return [{"dataset": r["dataset"], "status": r["verdict"]} for r in reports]


def explicit(args) -> bool:
    """Datasets named with --dataset must be acted on; otherwise a dataset that is not ready yet is skipped."""
    return bool(getattr(args, "dataset", None))


@serialized_datasets
def step_pack(plan: dict, datasets: list[str], args) -> list[dict]:
    from scripts.archive_run import pack
    from scripts.deposit_zenodo import verified_run_manifest
    from scripts.evidence_preflight import archive_stage_content_sha256
    rows = []
    for dataset in datasets:
        preflight = current(plan, dataset).get("preflight")
        if preflight is None:
            if not explicit(args):
                rows.append({"dataset": dataset, "status": "SKIPPED", "detail": "no preflight yet"})
                continue
            raise PipelineError(f"{dataset}: run the preflight first; it reports what the audit would refuse before hours of packing")
        if preflight["verdict"] == "BLOCKED" and not args.allow_blocked:
            if explicit(args):
                raise VerdictRefusal(f"{dataset}: preflight BLOCKED; inspect its report before packing a diagnostic archive")
            rows.append({"dataset": dataset, "status": "SKIPPED", "detail": "preflight BLOCKED (use --allow-blocked to pack anyway)"})
            continue
        if preflight["verdict"] == "PENDING_FULL_AUDIT":
            if explicit(args):
                raise VerdictRefusal(f"{dataset}: run the full audit before packing")
            rows.append({"dataset": dataset, "status": "SKIPPED", "detail": "full audit pending"})
            continue
        _preflight_context(preflight, plan, dataset)
        target = archive_path(plan, dataset)
        root = Path(plan["datasets"][dataset]["run_root"])
        if target.exists() or target.is_symlink():
            if target.is_symlink():
                raise PipelineError("archive cannot be a symlink")
            manifest = verified_run_manifest(target)
            status = "VERIFIED_EXISTING"
        else:
            descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".partial", dir=target.parent)
            os.close(descriptor)
            partial = Path(temporary)
            partial.unlink()
            print(f"pack {dataset}: {root} -> {target}", file=sys.stderr)
            try:
                pack(root, partial, campaign=dataset == "Roman-empire")
                manifest = verified_run_manifest(partial)
                _check_packed_content(dataset, preflight, manifest)
                os.replace(partial, target)
            finally:
                partial.unlink(missing_ok=True)
            status = "PACKED"
        _check_packed_content(dataset, preflight, manifest)
        _preflight_context(preflight, plan, dataset)
        identities = archive_identities(manifest)
        value = {"status": status, "path": str(target), "sha256": manifest["archive_sha256"],
                 "bytes": manifest["archive_bytes"], "files": len(manifest["files"]), "identities": identities,
                 "sealed_content_sha256": archive_stage_content_sha256(manifest),
                 "profile": manifest.get("profile"), "scientific_run": manifest.get("scientific_run")}
        record(plan, dataset, "archive", value)
        rows.append({"dataset": dataset, "status": status, "detail": f"{manifest['archive_bytes'] / 1e9:.2f} GB"})
    return rows


def _check_packed_content(dataset: str, preflight: dict, manifest: dict) -> None:
    from scripts.evidence_preflight import archive_stage_content_sha256
    if manifest.get("dataset") != dataset or manifest.get("kind") != kind_of(dataset):
        raise PipelineError(f"{dataset}: packed archive dataset/kind does not match the plan")
    if archive_identities(manifest) != preflight_identities(preflight):
        raise PipelineError(f"{dataset}: the archive and preflight examined different sealed identities")
    if not preflight.get("sealed_content_sha256") or archive_stage_content_sha256(manifest) != preflight["sealed_content_sha256"]:
        raise PipelineError(f"{dataset}: sealed results changed since the preflight; rerun it on the intended archive inputs")


def _gate(dataset: str, preflight: dict | None, archive: dict | None, args, *, publishing: bool) -> list[str]:
    if archive is None:
        raise PipelineError(f"{dataset}: pack the archive first")
    if preflight is None:
        raise PipelineError(f"{dataset}: no preflight recorded")
    _preflight_context(preflight)
    if preflight_identities({"identities": preflight["identities"]}) != archive["identities"]:
        raise PipelineError(f"{dataset}: the archive and the preflight examined different sealed identities")
    if not preflight.get("sealed_content_sha256") or preflight["sealed_content_sha256"] != archive.get("sealed_content_sha256"):
        raise PipelineError(f"{dataset}: the archive and preflight examined different sealed results")
    manifest = _verified_archive(dataset, archive)
    verdict = preflight["verdict"]
    allowed = {"READY": True, "INTEGRITY_ONLY": getattr(args, "allow_integrity_only", False)}
    if publishing and (manifest.get("profile") != "full" or manifest.get("scientific_run") is not True):
        raise VerdictRefusal(f"{dataset}: only full-profile scientific archives permit production publication")
    if publishing and not allowed.get(verdict, False):
        raise VerdictRefusal(f"{dataset}: preflight verdict {verdict} does not permit publication "
                             f"(see {preflight.get('report')}; BLOCKED/PENDING cannot be overridden for production publication)")
    if publishing and verdict == "INTEGRITY_ONLY":
        return ["published with preflight verdict INTEGRITY_ONLY; manuscript agreement/provenance is unresolved"]
    return [] if verdict == "READY" else [f"preflight verdict {verdict}; not evidence of manuscript reproduction"]


def _receipt(archive: Path, sandbox: bool) -> dict | None:
    path = archive.with_name(archive.name + (".sandbox.deposit.json" if sandbox else ".deposit.json"))
    if path.is_symlink():
        raise PipelineError("deposit receipt cannot be a symlink")
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


@serialized_datasets
def step_deposit(plan: dict, datasets: list[str], args, *, publish: bool) -> list[dict]:
    license_id = plan.get("license")
    if not isinstance(license_id, str) or not LICENSE.fullmatch(license_id):
        raise PipelineError("set license in the plan to the Zenodo licence id covering the archived material")
    if publish and not args.confirm:
        raise PipelineError("publication creates permanent public records; repeat with --confirm")
    token = "ZENODO_SANDBOX_TOKEN" if args.sandbox else "ZENODO_TOKEN"
    rows = []
    for dataset in datasets:
        entry = current(plan, dataset)
        if entry.get("archive") is None and not explicit(args):
            verdict = (entry.get("preflight") or {}).get("verdict") or "not run"
            rows.append({"dataset": dataset, "status": "SKIPPED", "detail": f"no archive (preflight {verdict})"})
            continue
        try:
            if entry.get("preflight") is not None:
                _preflight_context(entry["preflight"], plan, dataset)
            notes = _gate(dataset, entry.get("preflight"), entry.get("archive"), args, publishing=publish and not args.sandbox)
        except VerdictRefusal as exc:
            if explicit(args):
                raise
            rows.append({"dataset": dataset, "status": "SKIPPED", "detail": str(exc).split(": ", 1)[1].split(" (see")[0]})
            continue
        if not os.environ.get(token):
            raise PipelineError(f"set {token} (scopes deposit:write and deposit:actions)")
        archive = Path(entry["archive"]["path"])
        command = ["scripts/deposit_zenodo.py", "archive", str(archive), "--license", license_id]
        if plan.get("code_doi"):
            command += ["--code-doi", plan["code_doi"]]
        if args.sandbox:
            command.append("--sandbox")
        if publish:
            command.append("--publish")
        result = tool(command, check=False)
        if result.returncode not in (0, 1):
            tail = "\n".join((result.stdout + result.stderr).strip().splitlines()[-12:])
            raise PipelineError(f"{dataset}: deposit failed:\n{tail}")
        receipt = _bound_receipt(archive, entry["archive"], args.sandbox)
        if result.returncode == 1 and receipt["status"] != "publishing":
            raise PipelineError(f"{dataset}: deposit command did not complete successfully")
        _verified_archive(dataset, entry["archive"])
        value = {"status": receipt.get("status"), "service": receipt.get("service"), "sandbox": bool(args.sandbox),
                 "deposition_id": receipt.get("deposition_id"), "reserved_doi": receipt.get("reserved_doi"),
                 "doi": receipt.get("doi"), "published": receipt.get("published"), "notes": notes,
                 "archive_sha256": entry["archive"]["sha256"], "archive_bytes": entry["archive"]["bytes"],
                 "receipt": str(archive.with_name(archive.name + (".sandbox.deposit.json" if args.sandbox else ".deposit.json")))}
        record(plan, dataset, "sandbox_deposit" if args.sandbox else "deposit", value)
        rows.append({"dataset": dataset, "status": value["status"], "detail": value["doi"] or value["reserved_doi"]})
    return rows


@serialized_datasets
@serialized_checkout
def step_register(plan: dict, datasets: list[str], args) -> list[dict]:
    from scripts.ingest_run import load_index
    rows = []
    for dataset in datasets:
        entry = current(plan, dataset)
        deposit = entry.get("deposit")
        if not deposit or deposit.get("published") is not True or not deposit.get("doi"):
            if explicit(args):
                raise PipelineError(f"{dataset}: no published production deposit")
            rows.append({"dataset": dataset, "status": "SKIPPED", "detail": "no published production deposit"})
            continue
        if not re.fullmatch(r"10\.5281/zenodo\.[1-9][0-9]*", deposit["doi"]):
            raise PipelineError(f"{dataset}: only production Zenodo version DOIs are registered")
        archive = entry["archive"]
        if archive.get("profile") != "full" or archive.get("scientific_run") is not True:
            raise PipelineError(f"{dataset}: only full-profile scientific archives can be registered through this pipeline")
        preflight = entry.get("preflight")
        if preflight is None or preflight.get("verdict") not in ("READY", "INTEGRITY_ONLY"):
            raise PipelineError(f"{dataset}: a completed integrity preflight is required for registration")
        _preflight_context(preflight, plan, dataset)
        _gate(dataset, preflight, archive, args, publishing=False)
        receipt = _bound_receipt(Path(archive["path"]), archive, False)
        if (receipt.get("published") is not True or deposit.get("sandbox") is not False
                or receipt.get("doi") != deposit["doi"] or receipt.get("service") != deposit.get("service")
                or receipt.get("deposition_id") != deposit.get("deposition_id")
                or deposit.get("archive_sha256") != archive["sha256"] or deposit.get("archive_bytes") != archive["bytes"]):
            raise PipelineError(f"{dataset}: production publication state does not bind the current archive receipt")
        registered = [r for r in load_index()["runs"] if r.get("dataset") == dataset]
        if registered and registered[0].get("archive_sha256") == archive["sha256"] and registered[0].get("deposit_doi") == deposit["doi"]:
            status = "ALREADY_REGISTERED"
        elif registered:
            raise PipelineError(f"{dataset}: runs/index.json registers a different archive; resolve it by hand "
                                "(scripts/ingest_run.py --replace) after checking which archive the manuscript uses")
        else:
            # Pass paths as argv data, not Make variable expansions interpreted by a shell.
            tool(["scripts/ingest_run.py", archive["path"], "--doi", deposit["doi"]])
            status = "REGISTERED"
        # Resume safely if ingest succeeded but a subsequent generation step was interrupted.
        tool(["scripts/run_manifests.py", "runs", "--index", str(ROOT / "runs" / "index.json"),
              "--label", "official", "--with-indices", "--allow-deposited-absent"])
        for script in ("availability.py", "claims_ledger.py", "manuscript_availability.py"):
            tool(["scripts/" + script])
        final = [row for row in load_index()["runs"] if row.get("dataset") == dataset]
        if (len(final) != 1 or final[0].get("archive_sha256") != archive["sha256"]
                or final[0].get("deposit_doi") != deposit["doi"]):
            raise PipelineError(f"{dataset}: registration did not produce the expected index entry")
        record(plan, dataset, "registered", {"status": status, "doi": deposit["doi"], "archive_sha256": archive["sha256"]})
        rows.append({"dataset": dataset, "status": status, "detail": deposit["doi"]})
    return rows


@serialized_datasets
@serialized_checkout
def step_bind(plan: dict, datasets: list[str], args) -> list[dict]:
    from scripts.ingest_run import load_index
    rows = []
    index = {r["dataset"]: r for r in load_index()["runs"]}
    for dataset in datasets:
        relationship = relationship_for(plan, dataset)
        if relationship not in RELATIONSHIPS:
            raise PipelineError(f"{dataset}: declare relationship_to_manuscript in the plan before installing a binding")
        if dataset not in index:
            if explicit(args):
                raise PipelineError(f"{dataset}: not registered in runs/index.json")
            rows.append({"dataset": dataset, "status": "SKIPPED", "detail": "not registered in runs/index.json"})
            continue
        entry = current(plan, dataset)
        preflight = entry.get("preflight")
        if preflight is None or preflight.get("verdict") not in ("READY", "INTEGRITY_ONLY"):
            raise PipelineError(f"{dataset}: complete the integrity preflight before installing a binding")
        _preflight_context(preflight, plan, dataset)
        _gate(dataset, preflight, entry.get("archive"), args, publishing=False)
        if entry["archive"].get("profile") != "full" or entry["archive"].get("scientific_run") is not True:
            raise PipelineError(f"{dataset}: only full-profile scientific archives can be bound through this pipeline")
        if index[dataset].get("archive_sha256") != entry["archive"]["sha256"]:
            raise PipelineError(f"{dataset}: registered archive differs from the preflighted archive")
        if index[dataset].get("path") != STEMS[dataset] or index[dataset].get("kind") != kind_of(dataset):
            raise PipelineError(f"{dataset}: unsafe or inconsistent registered run path/kind")
        run = ROOT / "runs" / index[dataset]["path"]
        if _materialized_content(dataset, run) != entry["archive"]["sealed_content_sha256"]:
            raise PipelineError(f"{dataset}: materialized run differs from the preflighted archive; fetch/ingest it again")
        identifier = plan.get("manuscript_identifier") or DEFAULT_IDENTIFIER
        target = ROOT / "paper" / "bindings" / f"{STEMS[dataset]}_full.json"
        with tempfile.TemporaryDirectory(prefix="cm-binding-") as tmp:
            emitted = Path(tmp) / f"{STEMS[dataset]}_full.json"
            if dataset == "Roman-empire":
                result = tool(["scripts/emit_campaign_binding.py", str(run), "--relationship", relationship,
                               "--manuscript-identifier", identifier, "--output", str(emitted), "--check"], check=False)
                status = {0: "MATCH", 1: "MISMATCH"}.get(result.returncode, "ERROR")
                try:
                    missing = json.loads(result.stdout).get("coverage_missing", []) if status != "ERROR" else []
                except ValueError:
                    missing = ["campaign check output was not JSON"]
            else:
                tool(["scripts/compare_with_manuscript.py", str(run), "--emit-binding", str(emitted), "--relationship",
                      relationship, "--manuscript-identifier", identifier], check=False)
                if not emitted.is_file():
                    raise PipelineError(f"{dataset}: compare_with_manuscript.py did not write a binding")
                result = tool(["scripts/check_result_binding.py", str(run), str(emitted)], check=False)
                status = {0: "MATCH", 1: "MISMATCH"}.get(result.returncode, "ERROR")
                from scripts.scientific_evidence_audit import binding_coverage
                missing = binding_coverage(json.loads(emitted.read_text(encoding="utf-8")), campaign=False,
                                           dataset=dataset, manuscript_root=ROOT)
            if status == "ERROR":
                tail = "\n".join((result.stdout + result.stderr).strip().splitlines()[-12:])
                raise PipelineError(f"{dataset}: binding check failed to run:\n{tail}")
            complete = status == "MATCH" and not missing
            if complete:
                from scripts.evidence_preflight import atomic_text
                _preflight_context(preflight, plan, dataset)
                if _materialized_content(dataset, run) != entry["archive"]["sealed_content_sha256"]:
                    raise PipelineError(f"{dataset}: run changed during binding checks")
                if target.is_symlink() or not target.resolve().is_relative_to(ROOT.resolve()):
                    raise PipelineError("binding target must be inside paper/bindings without symlinks")
                atomic_text(target, emitted.read_text(encoding="utf-8"))
                installed = True
            else:
                installed = False
        value = {"status": status if not missing else "INCOMPLETE", "installed": installed, "relationship": relationship,
                 "coverage_missing": missing, "path": str(target.relative_to(ROOT))}
        record(plan, dataset, "binding", value)
        rows.append({"dataset": dataset, "status": value["status"], "detail": "installed" if installed else "not installed"})
    return rows


def _verified_record_set(records_root: Path, name: str) -> dict:
    from cm_campaigns.contract import RECORD_SETS as CONTRACTS
    from scripts.records import load_specs, read_summary, verify_set
    from scripts.scientific_evidence_audit import validate_record_coverage
    specs = load_specs(ROOT / "records")
    target = records_root / name
    if target.is_symlink() or not target.is_dir():
        raise PipelineError(f"{name}: exporter did not produce the intended record set")
    for path in target.rglob("*"):
        if path.is_symlink():
            raise PipelineError(f"{name}: record set cannot contain symlinks")
    result = verify_set(target, specs["sets"][name], specs)
    if result.get("synthetic") is not False:
        raise PipelineError(f"{name}: synthetic records cannot be installed as manuscript measurements")
    rows = read_summary(target, specs["sets"][name], specs["summary_columns"])
    validate_record_coverage(name, rows, specs["sets"][name], CONTRACTS[name], path=target)
    return result


def _record_command(name: str, entry: dict) -> list[str]:
    exporter, mapping = entry.get("exporter"), entry.get("mapping")
    if exporter and mapping:
        raise PipelineError(f"{name}: give exporter or mapping, not both")
    if mapping:
        if not isinstance(mapping, str) or not Path(mapping).is_absolute():
            raise PipelineError(f"{name}: mapping must be an absolute path")
        supplied = json.loads(Path(mapping).read_text(encoding="utf-8"))
        if supplied.get("set") != name:
            raise PipelineError(f"{name}: mapping names a different record set")
        return ["scripts/export_records.py", "from-mapping", "--mapping", mapping]
    if not isinstance(exporter, list) or not exporter or not all(isinstance(x, str) for x in exporter):
        raise PipelineError(f"{name}: exporter must be the argument list of scripts/export_records.py")
    forbidden = {"--root", "--replace", "--allow-partial", "--allow-missing-accounting", "--allow-private-patterns"}
    if any(value.split("=", 1)[0] in forbidden for value in exporter):
        raise PipelineError(f"{name}: exporter may not override the managed output, completeness or privacy checks")
    produced = {"baselines": "baselines", "budget-matched": "budget_matched"}.get(exporter[0])
    if exporter[0] == "from-runs":
        values = [exporter[i + 1] for i, value in enumerate(exporter[:-1]) if value == "--set"]
        values += [value.split("=", 1)[1] for value in exporter if value.startswith("--set=")]
        produced = values[0] if len(values) == 1 else None
    if produced != name:
        raise PipelineError(f"{name}: exporter names a different or unsupported record set")
    return ["scripts/export_records.py", *exporter]


@serialized_checkout
def step_records(plan: dict, args) -> list[dict]:
    rows = []
    for name in RECORD_SETS:
        entry = (plan.get("record_sets") or {}).get(name) or {}
        exporter, mapping = entry.get("exporter"), entry.get("mapping")
        if args.set and name not in args.set:
            continue
        if not exporter and not mapping:
            rows.append({"dataset": name, "status": "NOT_PLANNED", "detail": "no exporter or mapping in the plan"})
            continue
        if exporter and mapping:
            raise PipelineError(f"{name}: give exporter or mapping, not both")
        target = ROOT / "records" / name
        if target.exists() and not args.replace:
            _verified_record_set(ROOT / "records", name)
            rows.append({"dataset": name, "status": "VERIFIED_EXISTING", "detail": "records/" + name + " (--replace rebuilds it)"})
            continue
        if target.is_symlink():
            raise PipelineError(f"{name}: record output cannot be a symlink")
        command = _record_command(name, entry)
        # Existing measurements stay intact until the new set passes tracing and coverage.
        with tempfile.TemporaryDirectory(prefix=".cm-record-export-", dir=ROOT.parent) as temporary:
            from scripts.record_mapping import install_verified_set
            from scripts.records import load_specs
            staging = Path(temporary) / "new"
            staging.mkdir()
            shutil.copyfile(ROOT / "records" / "specs.json", staging / "specs.json")
            tool([*command, "--root", str(staging)])
            result = _verified_record_set(staging, name)
            install_verified_set(staging, staging / name, ROOT / "records", load_specs(ROOT / "records"),
                                 replace=args.replace)
        record(plan, name, "records", {"status": "EXPORTED", "method": "mapping" if mapping else exporter[0]},
               record_set=True)
        rows.append({"dataset": name, "status": "EXPORTED", "detail": f"{result['rows']} traced rows; required coverage verified"})
    if any(row["status"] in ("EXPORTED", "VERIFIED_EXISTING") for row in rows):
        tool(["scripts/records.py", "check"])
        for script in ("availability.py", "claims_ledger.py", "manuscript_availability.py"):
            tool(["scripts/" + script])
    return rows


def status_rows(plan: dict, state: dict) -> dict:
    from scripts.ingest_run import load_index
    index = {r["dataset"]: r for r in load_index()["runs"]}
    records = json.loads((ROOT / "records" / "index.json").read_text(encoding="utf-8"))
    deposited = set()
    for entry in records.get("sets", []):
        name = entry.get("set")
        if name not in RECORD_SETS or entry.get("verified") is not True:
            continue
        try:
            checked = _verified_record_set(ROOT / "records", name)
            if checked["manifest_sha256"] == entry.get("manifest_sha256"):
                deposited.add(name)
        except (OSError, ValueError, KeyError, TypeError):
            pass
    datasets = []
    for dataset in DATASETS:
        entry = state["datasets"].get(dataset, {})
        binding_file = ROOT / "paper" / "bindings" / f"{STEMS[dataset]}_full.json"
        binding = json.loads(binding_file.read_text(encoding="utf-8")) if binding_file.is_file() else {}
        datasets.append({
            "dataset": dataset, "planned": dataset in plan["datasets"],
            "preflight": (entry.get("preflight") or {}).get("verdict"),
            "archive": (entry.get("archive") or {}).get("sha256", "")[:12] or None,
            "sandbox": (entry.get("sandbox_deposit") or {}).get("status"),
            "deposit": (entry.get("deposit") or {}).get("status"),
            "doi": (entry.get("deposit") or {}).get("doi") or (entry.get("deposit") or {}).get("reserved_doi"),
            "registered": index.get(dataset, {}).get("deposit_doi"),
            "binding": binding.get("status"),
        })
    sets = [{"set": name, "deposited": name in deposited, "planned": bool(((plan.get("record_sets") or {}).get(name) or {}).get("exporter")
                                                                          or ((plan.get("record_sets") or {}).get(name) or {}).get("mapping"))}
            for name in RECORD_SETS]
    return {"datasets": datasets, "record_sets": sets,
            "budget_campaign": plan.get("budget_campaign") if plan.get("budget_campaign") and Path(plan["budget_campaign"]).is_dir() else None}


def status_markdown(summary: dict) -> str:
    lines = ["# Evidence registration status", "", f"Generated {now()} by scripts/evidence_pipeline.py status.", "",
             "| Dataset | Preflight | Archive | Sandbox | Deposit | DOI | Registered | Binding |",
             "|---|---|---|---|---|---|---|---|"]
    for row in summary["datasets"]:
        lines.append("| " + " | ".join(str(row[k] if row[k] is not None else "—") for k in
                                       ("dataset", "preflight", "archive", "sandbox", "deposit", "doi", "registered", "binding")) + " |")
    lines += ["", "| Record set | Planned | Deposited and traced |", "|---|---|---|"]
    for row in summary["record_sets"]:
        lines.append(f"| `{row['set']}` | {'yes' if row['planned'] else 'no'} | {'yes' if row['deposited'] else 'no'} |")
    lines += ["", f"E3–E6 campaign for the audit: {summary['budget_campaign'] or 'not available'}.", "",
              "This table records author workflow progress. Saved statuses do not establish current public availability "
              "or scientific completeness. After registration and binding, obtain the decisive fresh checks with "
              "`make review REQUIRE_EVIDENCE=1 BUDGET_CAMPAIGN=... MANUSCRIPT_PDF=... SUPPLEMENTARY_PDF=...`.", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    item = sub.add_parser("template", help="write an empty plan")
    item.add_argument("--output", type=Path, required=True)
    item = sub.add_parser("discover", help="inspect an explicit export root and propose unambiguous official run paths")
    item.add_argument("--root", type=Path, required=True, help="scan this real directory only; symlinks are not followed")
    item.add_argument("--output", type=Path, required=True, help="new private discovery .json outside checkout and source root")
    item.add_argument("--plan-output", type=Path, help="optional new private plan; leaves all author declarations unset")
    item.add_argument("--deposits-dir", type=Path, help="required with --plan-output; outside checkout and scanned sources")
    item.add_argument("--max-depth", type=int, default=6, help="maximum relative directory depth (default: 6)")
    item.add_argument("--max-directories", type=int, default=10000, help="maximum directories to inspect (default: 10000)")
    for name in ("preflight", "pack", "deposit", "publish", "register", "bind", "records", "status"):
        item = sub.add_parser(name)
        item.add_argument("--plan", type=Path, required=True)
        if name not in ("records", "status"):
            item.add_argument("--dataset", action="append", choices=DATASETS, help="restrict to this dataset (repeatable)")
        if name == "preflight":
            item.add_argument("--full-audit", choices=("auto", "always", "never"), default="auto")
        if name == "pack":
            item.add_argument("--allow-blocked", action="store_true", help="pack a BLOCKED run for local diagnosis or sandbox rehearsal")
        if name in ("deposit", "publish"):
            item.add_argument("--sandbox", action="store_true", help="sandbox.zenodo.org rehearsal; never registered")
            item.add_argument("--allow-integrity-only", action="store_true",
                              help="publish evidence whose numbers do not bind to the printed manuscript (recorded)")
        if name == "publish":
            item.add_argument("--confirm", action="store_true", help="acknowledge that publication is permanent")
        if name == "records":
            item.add_argument("--set", action="append", choices=RECORD_SETS, help="restrict to this record set")
            item.add_argument("--replace", action="store_true", help="rebuild existing sets from their inputs")
    args = parser.parse_args(argv)
    try:
        if args.command == "discover":
            from scripts.evidence_discovery import discover, markdown, write_outputs
            if (args.plan_output is None) != (args.deposits_dir is None):
                raise PipelineError("--plan-output and --deposits-dir must be supplied together")
            report = discover(args.root, max_depth=args.max_depth, max_directories=args.max_directories)
            write_outputs(report, args.output, plan_output=args.plan_output, deposits=args.deposits_dir)
            print(markdown(report))
            return 0 if report["status"] == "CANDIDATES_FOUND" else 1
        if args.command == "template":
            if args.output.is_symlink():
                raise PipelineError("the plan path cannot be a symlink")
            target = args.output.resolve()
            if target.is_relative_to(ROOT):
                raise PipelineError("write the plan outside the repository checkout")
            if target.exists():
                raise PipelineError(f"{target} exists; edit it or choose another path")
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(json.dumps(template(), indent=1) + "\n")
            print(f"wrote {target}; fill deposits_dir, license, relationship_to_manuscript, every run_root and the record sets")
            return 0
        plan = load_plan(args.plan)
        if args.command == "status":
            with locked(plan):
                state = load_state(plan)
                text = status_markdown(status_rows(plan, state))
                from scripts.evidence_preflight import atomic_text
                atomic_text(deposits_dir(plan) / "EVIDENCE_STATUS.md", text)
            print(text)
            return 0
        if args.command == "records":
            rows = step_records(plan, args)
        else:
            datasets = selected(plan, args.dataset, need_root=args.command in ("preflight", "pack"))
            if args.command == "preflight":
                rows = step_preflight(plan, datasets, args)
            elif args.command == "pack":
                rows = step_pack(plan, datasets, args)
            elif args.command in ("deposit", "publish"):
                rows = step_deposit(plan, datasets, args, publish=args.command == "publish")
            elif args.command == "register":
                rows = step_register(plan, datasets, args)
            else:
                rows = step_bind(plan, datasets, args)
        for row in rows:
            print(f"{row['status']:20s} {row['dataset']:16s} {row.get('detail', '')}")
        if args.command == "preflight":
            return 0 if rows and all(row["status"] in ("READY", "INTEGRITY_ONLY") for row in rows) else 1
        if not rows or all(row["status"] in ("SKIPPED", "NOT_PLANNED") for row in rows):
            print("No selected step completed; inspect the skipped/not-planned rows.", file=sys.stderr)
            return 1
        if any(row["status"] in ("publishing", "MISMATCH", "INCOMPLETE", "ERROR") for row in rows):
            return 1
        return 0
    except (PipelineError, OSError, ValueError, KeyError, TypeError, tarfile.TarError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
