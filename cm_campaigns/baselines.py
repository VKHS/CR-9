"""Method keys and coverage of the ``baselines`` record set (Supplementary Tables S15 and S15b).

One definition for the code that prints the rows (``paper/build/displays/records_campaigns.py``), the
runner that measures the internal rows (``cm_experiments/internal_baselines.py``), the exporter
(``scripts/export_records.py baselines``) and the record audit (``scripts/scientific_evidence_audit.py``),
so the method keys of the internal rows cannot drift between them.

Internal rows of a dataset: one per route of ``S15_ROUTES`` plus the comparator's and the deployed rule's
route, in the registry order of the routes. The comparator's row is keyed ``'<route method> (qC)'``, the
deployed rule's ``'<route method> (deployed)'`` (absent when every run returned its comparator) and every
other route by its method name.

A complete set holds, for each of the seven datasets,

* every internal row: per-seed ``score`` rows (``BASELINE_SEEDS`` seeds) and ``gpu_hours``; the comparator
  and the deployed rule also ``serial_cost`` and ``wall_time`` (the E17 timing of Table S15b);
* the two external comparisons of ``cm_campaigns.contract.EXTERNAL``: per-seed ``score`` rows,
  ``hpo_trials``, ``params_millions``, ``gpu_hours``, ``serial_cost`` and ``wall_time``;

and, for each of the fourteen external comparisons, the validated study plan, its upstream identity and the
adapter archive: ``raw/<slug>/<method>/plan.json``, ``raw/<slug>/<method>/upstream.json`` and the archive that
``upstream.json`` names under ``code/adapters/<slug>/<method>/``.

Standard library only (the study plans are validated with ``cm_experiments.upstream.validate_plan``).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath

from .contract import BASELINE_SEEDS, DATASETS, EXTERNAL, SLUGS

# Registry order of the routes (scripts/_paper_common.ROUTE_NAMES) and the method names of Table S15.
ROUTES = ("nominal", "best_depth", "learned_stop", "phase", "fixed_sequence", "mpc_ol", "one_step", "radius", "mpc_rh")
ROUTE_METHOD = {"nominal": "full-depth plant", "best_depth": "best fixed depth", "learned_stop": "learned stop",
                "phase": "optimized phase", "fixed_sequence": "best fixed sequence", "mpc_ol": "MPC-OL",
                "one_step": "one-step", "radius": "radius", "mpc_rh": "MPC-RH"}
METHOD_ROUTE = {name: route for route, name in ROUTE_METHOD.items()}
# Internal routes tabulated in Supplementary Table S15 (the comparator's and the deployed rule's route are added
# when they lie outside this set).
S15_ROUTES = ("nominal", "best_depth", "phase", "fixed_sequence", "mpc_ol", "radius", "mpc_rh")
# Policy families of main Eq. (15) (paper/registered_constants.json policy_families_eq15_table_S10).
OPEN_LOOP_ROUTES = ("nominal", "best_depth", "learned_stop", "phase", "fixed_sequence", "mpc_ol")
FEEDBACK_ROUTES = ("one_step", "radius", "mpc_rh")

QC, DEPLOYED, ROUTE, EXTERNAL_METHOD = "qC", "deployed", "route", "external"
SUFFIX = {QC: " (qC)", DEPLOYED: " (deployed)"}
SCORE = "score"
# Unseeded quantities Tables S15 and S15b read for each kind of row (more rows are allowed, never fewer).
REQUIRED = {QC: ("serial_cost", "wall_time", "gpu_hours"),
            DEPLOYED: ("serial_cost", "wall_time", "gpu_hours"),
            ROUTE: ("gpu_hours",),
            EXTERNAL_METHOD: ("hpo_trials", "params_millions", "gpu_hours", "serial_cost", "wall_time")}
UPSTREAM_SCHEMA = "cm-upstream-identity-1"


def method_keys(qc_route: str, qd_route: str | None, returned: bool) -> tuple[str, str | None]:
    """Keys of the comparator row and the deployed row (None when the comparator is returned)."""
    qc = f"{ROUTE_METHOD[qc_route]}{SUFFIX[QC]}"
    qd = None if returned else f"{ROUTE_METHOD[qd_route]}{SUFFIX[DEPLOYED]}"
    return qc, qd


def internal_methods(qc_route: str, qd_route: str | None, returned: bool) -> list[dict]:
    """The internal rows of Table S15 in registry order: {route, key, role} with role qC, deployed or route."""
    if qc_route not in ROUTE_METHOD:
        raise ValueError(f"unknown comparator route {qc_route!r}")
    if not returned and (qd_route not in ROUTE_METHOD or qd_route == qc_route):
        raise ValueError(f"the deployed rule's route {qd_route!r} must be a route other than the comparator's")
    qc_key, qd_key = method_keys(qc_route, qd_route, returned)
    rows = []
    for route in ROUTES:
        if route == qc_route:
            rows.append({"route": route, "key": qc_key, "role": QC})
        elif route == qd_route and not returned:
            rows.append({"route": route, "key": qd_key, "role": DEPLOYED})
        elif route in S15_ROUTES or route == qd_route:
            rows.append({"route": route, "key": ROUTE_METHOD[route], "role": ROUTE})
    return rows


def parse_key(key: str) -> tuple[str, str]:
    """(route, role) of an internal method key; ValueError for any other text."""
    for role in (QC, DEPLOYED):
        if key.endswith(SUFFIX[role]):
            name = key[:-len(SUFFIX[role])]
            break
    else:
        name, role = key, ROUTE
    if name not in METHOD_ROUTE:
        raise ValueError(f"{key!r} is not an internal method key")
    return METHOD_ROUTE[name], role


def dataset_methods(dataset: str, qc_key: str, qd_key: str | None) -> dict[str, str]:
    """{method key: role} of every row a complete set holds for one dataset (internal rows, then EXTERNAL)."""
    qc_route, qc_role = parse_key(qc_key)
    if qc_role != QC:
        raise ValueError(f"{qc_key!r} is not a comparator key")
    qd_route = None
    if qd_key is not None:
        qd_route, qd_role = parse_key(qd_key)
        if qd_role != DEPLOYED:
            raise ValueError(f"{qd_key!r} is not a deployed-rule key")
    methods = {m["key"]: m["role"] for m in internal_methods(qc_route, qd_route, qd_key is None)}
    methods.update({method: EXTERNAL_METHOD for method in EXTERNAL[dataset]})
    return methods


def method_directory(method: str) -> str:
    """Directory name of a method under raw/<slug>/ (scripts/export_records.safe_name)."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(method)).strip("_")
    if not name:
        raise ValueError(f"cannot form a directory name from {method!r}")
    return name


def study_paths(dataset: str, method: str, adapter_commit: str) -> dict[str, str]:
    """Set-relative locations of an external comparison's plan, upstream identity and adapter archive."""
    folder = f"raw/{SLUGS[dataset]}/{method_directory(method)}"
    return {"plan": f"{folder}/plan.json", "upstream": f"{folder}/upstream.json",
            "adapter_archive": f"code/adapters/{SLUGS[dataset]}/{method_directory(method)}/"
                               f"adapter-{adapter_commit[:12]}.tar"}


# ----------------------------------------------------------------------------- the record audit
def check_coverage(rows, *, seeds: int = BASELINE_SEEDS) -> dict:
    """Rows of a complete set (dicts with dataset, method, seed, quantity); ValueError names what is missing."""
    observed: dict[tuple[str, str], dict] = {}
    problems = []
    for row in rows:
        dataset, method, quantity = str(row["dataset"]), str(row["method"]), str(row["quantity"])
        seed = str(row.get("seed") or "").strip()
        entry = observed.setdefault((dataset, method), {"quantities": set(), "seeds": set()})
        if quantity == SCORE:
            if not seed:
                problems.append(f"{dataset}/{method}: a score row without its seed")
            entry["seeds"].add(seed)
        else:
            entry["quantities"].add(quantity)
    present = {dataset for dataset, _ in observed}
    if present != set(DATASETS):
        raise ValueError(f"baselines: datasets missing={sorted(set(DATASETS) - present)}, "
                         f"unexpected={sorted(present - set(DATASETS))}")
    internal_rows = 0
    for dataset in DATASETS:
        methods = {method for d, method in observed if d == dataset}
        qc = sorted(m for m in methods if m.endswith(SUFFIX[QC]))
        qd = sorted(m for m in methods if m.endswith(SUFFIX[DEPLOYED]))
        if len(qc) != 1 or len(qd) > 1:
            problems.append(f"{dataset}: {len(qc)} comparator rows and {len(qd)} deployed rows "
                            "(one '<route> (qC)' row and at most one '<route> (deployed)' row)")
            continue
        try:
            expected = dataset_methods(dataset, qc[0], qd[0] if qd else None)
        except ValueError as exc:
            problems.append(f"{dataset}: {exc}")
            continue
        missing, unexpected = sorted(set(expected) - methods), sorted(methods - set(expected))
        if missing or unexpected:
            problems.append(f"{dataset}: rows missing={missing}, unexpected={unexpected}")
        for method, role in expected.items():
            entry = observed.get((dataset, method))
            if entry is None:
                continue
            lacking = [q for q in REQUIRED[role] if q not in entry["quantities"]]
            if lacking:
                problems.append(f"{dataset}/{method}: no {', '.join(lacking)}")
            if len(entry["seeds"]) != seeds:
                problems.append(f"{dataset}/{method}: scores of {len(entry['seeds'])} seeds, {seeds} required")
        internal_rows += sum(1 for role in expected.values() if role != EXTERNAL_METHOD)
    if problems:
        raise ValueError("baselines: incomplete coverage: " + "; ".join(problems[:12])
                         + (f"; and {len(problems) - 12} more" if len(problems) > 12 else ""))
    return {"keys": ["dataset", "method", "seed"], "datasets": len(DATASETS), "internal_rows": internal_rows,
            "external_comparisons": sum(len(v) for v in EXTERNAL.values()), "seeds_per_row": seeds}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(root: Path, relative, what: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"{what}: not a set-relative path: {relative!r}")
    parts = PurePosixPath(relative).parts
    if PurePosixPath(relative).is_absolute() or ".." in parts or parts[0] not in ("raw", "code"):
        raise ValueError(f"{what}: {relative!r} does not lie under raw/ or code/ of the set")
    path = root.joinpath(*parts)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{what}: {relative} is not a file of the set")
    return path


def check_studies(path) -> dict:
    """The study plan, upstream identity and adapter archive of every external comparison of a set."""
    from cm_experiments.upstream import validate_plan
    root = Path(path)
    problems, studies = [], []
    for dataset in DATASETS:
        for method in EXTERNAL[dataset]:
            where = f"{dataset}/{method}"
            folder = f"raw/{SLUGS[dataset]}/{method_directory(method)}"
            try:
                upstream_path = _inside(root, f"{folder}/upstream.json", where)
                plan_path = _inside(root, f"{folder}/plan.json", where)
                upstream = json.loads(upstream_path.read_text(encoding="utf-8"))
                plan = validate_plan(json.loads(plan_path.read_text(encoding="utf-8")))
                archive = upstream.get("adapter_archive") or {}
                expected = study_paths(dataset, method, plan["adapter_commit"])
                checks = {
                    "schema": upstream.get("schema") == UPSTREAM_SCHEMA,
                    "dataset and method": (upstream.get("dataset"), upstream.get("method"), plan["dataset"],
                                           plan["method"]) == (dataset, method, dataset, method),
                    "plan location": upstream.get("plan") == expected["plan"],
                    "source commit": upstream.get("source_commit") == plan["source_commit"],
                    "adapter commit": upstream.get("adapter_commit") == plan["adapter_commit"] == archive.get("commit"),
                    "archive location": archive.get("file") == expected["adapter_archive"],
                }
                failed = [name for name, ok in checks.items() if not ok]
                if failed:
                    raise ValueError(f"{where}: upstream.json disagrees with its plan ({', '.join(failed)})")
                adapter = _inside(root, archive["file"], where)
                if _sha256(adapter) != archive.get("sha256"):
                    raise ValueError(f"{where}: {archive['file']} differs from the SHA-256 in upstream.json")
            except (ValueError, OSError, KeyError, TypeError) as exc:
                problems.append(str(exc) if str(exc).startswith(where) else f"{where}: {exc}")
                continue
            studies.append({"dataset": dataset, "method": method, "plan": expected["plan"],
                            "adapter_archive": archive["file"], "adapter_commit": plan["adapter_commit"],
                            "source_commit": plan["source_commit"]})
    if problems:
        raise ValueError("baselines: study plans and adapter code incomplete: " + "; ".join(problems[:8])
                         + (f"; and {len(problems) - 8} more" if len(problems) > 8 else ""))
    return {"external_comparisons": len(studies), "studies": studies}


def audit(rows, path=None) -> dict:
    """The record audit of a complete set: row coverage, and with the set directory its study packaging."""
    result = check_coverage(rows)
    if path is not None:
        result["studies"] = check_studies(path)["external_comparisons"]
    return result
