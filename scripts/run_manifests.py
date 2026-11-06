#!/usr/bin/env python3
"""Write split, seed and environment manifests.

Two kinds of manifest are produced under ``manifests/``:

* ``manifests/registered/<stem>_<profile>.json`` — the registered design of every
  configuration in ``configs/``: partition seed, training seeds, data roles and fold
  structure, bootstrap replicates, margins.  These are what a run *will* use.
* ``manifests/<label>/<stem>.json`` — the manifest extracted from a completed run: the
  actual role membership hashes and sizes, official split hashes, the seeds that were used,
  the ensemble members, the bootstrap seeds of every decision stage and the recorded
  execution environment.  With ``--with-indices`` the membership index arrays are copied next
  to the manifest as ``<stem>_splits.npz``.  These are what a run *did* use.

    python -B scripts/run_manifests.py registered
    python -B scripts/run_manifests.py runs --runs-root examples/runs --label examples
    python -B scripts/run_manifests.py runs --index runs/index.json --label official --with-indices
    python -B scripts/run_manifests.py runs --runs-root examples/runs --label examples --check

Standard library plus NumPy; no PyTorch.

``--allow-empty`` is limited to an explicit official index.  An empty, valid index
then reports that no official runs have executed, provided the output directory
contains no stale files.  It does not authorize any paper or scientific-evidence
build, and is not equivalent to accepting an index of unverified or missing runs.

``--allow-deposited-absent`` checks committed registration metadata when payloads are not
in the checkout (runs/ is not tracked). A missing payload needs a production DOI registration
and a matching manifest with archive name/hash/size, source/configuration/unit identities,
scope and all ten split records for Roman. Any copied membership arrays are hash checked.
This mode does not verify that the DOI is public or that its bytes are available. Present
payloads are still checked against their archive inventory and sealed stage chains.
``make fetch-deposits`` and ``make review REQUIRE_EVIDENCE=1`` perform the evidence checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper.build.evidence import DATASET_ORDER, EvidenceError, stem_for  # noqa: E402
from scripts.build_paper import discover, from_index, is_campaign, is_run  # noqa: E402
from scripts.ingest_run import (check_absent_manifest, check_present, check_registration_manifest,
                                deposited_absent as _deposited_absent, registered_path,
                                registration_descriptor)  # noqa: E402

SCHEMA_RUN = "cross-morse-run-manifest-1"
SCHEMA_REGISTERED = "cross-morse-registered-manifest-2"
SCHEMA_INDEX = "cross-morse-runs-index-1"
STEMS = {"ogbn-arxiv": "arxiv", "ogbn-products": "products", "ogbn-proteins": "proteins", "Roman-empire": "roman_empire",
         "PascalVOC-SP": "pascalvoc_sp", "COCO-SP": "coco_sp", "Peptides-func": "peptides_func"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _dump(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def deposited_absent(entry: dict, index_path: Path) -> bool:
    """Registered metadata is eligible for deferral; this does not check the DOI."""
    return _deposited_absent(entry, index_path.parent)


def check_deposited_absent(index_path: Path, out_dir: Path) -> tuple[list[str], list[str]]:
    """Check complete committed registration descriptors, never infer public availability."""
    stems, problems = [], []
    for entry in _json(index_path)["runs"]:
        if not deposited_absent(entry, index_path):
            continue
        stem = entry["stem"]
        stems.append(stem)
        try:
            check_absent_manifest(entry, out_dir)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            problems.append(f"{stem}: {exc}")
    return stems, problems


def manifest_runs_from_index(index_path: Path, *, allow_empty: bool = False,
                             allow_deposited_absent: bool = False) -> list[tuple[str, Path]]:
    """Keep an empty inventory distinct from missing or malformed run evidence."""
    index = _json(index_path)
    if not isinstance(index, dict) or index.get("schema") != SCHEMA_INDEX:
        raise EvidenceError(f"unexpected runs index schema in {index_path}")
    entries = index.get("runs")
    if not isinstance(entries, list):
        raise EvidenceError(f"runs index requires a 'runs' list: {index_path}")
    seen = set()
    for number, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise EvidenceError(f"runs[{number}] must be an object in {index_path}")
        if type(entry.get("verified")) is not bool:
            raise EvidenceError(f"runs[{number}].verified must be a boolean in {index_path}")
        if not isinstance(entry.get("path"), str) or not entry["path"].strip():
            raise EvidenceError(f"runs[{number}].path must be a nonempty string in {index_path}")
        if "stem" in entry and (not isinstance(entry["stem"], str) or not entry["stem"].strip()
                                or entry["stem"] in (".", "..")
                                or "/" in entry["stem"] or "\\" in entry["stem"]):
            raise EvidenceError(f"runs[{number}].stem must be a file-name component in {index_path}")
        if entry["verified"]:
            stem = entry.get("stem") or Path(entry["path"]).name
            if stem in seen:
                raise EvidenceError(f"runs[{number}] duplicates manifest stem {stem!r}")
            seen.add(stem)
            path = (index_path.parent / entry["path"]).resolve()
            if allow_deposited_absent:
                path = registered_path(entry, index_path.parent)
                if deposited_absent(entry, index_path):
                    continue
            if not (is_run(path) or is_campaign(path)):
                raise EvidenceError(f"runs[{number}] is marked verified but has no completed run or campaign: {path}")
        elif allow_deposited_absent:
            raise EvidenceError(f"runs[{number}] is not verified; absent-deposit mode cannot hide it")
    if not entries and allow_empty:
        return []
    if allow_deposited_absent and all(deposited_absent(e, index_path) for e in entries if e["verified"]) \
            and any(e["verified"] for e in entries):
        return []
    # The existing reader remains strict about finding completed, verified runs.
    # In particular, --allow-empty does not hide missing or unverified entries.
    return from_index(index_path)


def require_no_executed_manifests(out_dir: Path) -> None:
    """Never make an empty index look clean while old execution files remain."""
    if out_dir.is_symlink() or (out_dir.exists() and not out_dir.is_dir()):
        raise EvidenceError(f"empty official index requires an absent or empty manifest directory: {out_dir}")
    stale = sorted(str(path.relative_to(out_dir)) for path in out_dir.rglob("*")
                   if path.is_file() or path.is_symlink()) if out_dir.exists() else []
    if stale:
        raise EvidenceError(f"empty official index has stale executed manifest files in {out_dir}: "
                            + ", ".join(stale))


# ------------------------------------------------------------------ registered manifests

def registered_manifest(config_path: Path) -> dict:
    cfg = _json(config_path)
    data, training, sel, adm, ctrl, expl = (cfg.get(k) or {} for k in
                                            ("data", "training", "selection", "admission", "controller", "exploration"))
    return {
        "schema": SCHEMA_REGISTERED,
        "config": config_path.name,
        "config_sha256": sha256_file(config_path),
        "dataset": cfg.get("dataset"),
        "profile": cfg.get("profile"),
        "scientific_run": cfg.get("scientific_run"),
        "partition_seed": cfg.get("seed"),
        "training_seeds": training.get("seeds"),
        "checkpoint_rule": training.get("checkpoint_rule"),
        "data_roles": {"rule": "seeded permutation of the official training units; 60%/20%/10%/5%/5% to "
                               "D_fit/D_explore/D_select/D_confirm/D_compare; official validation = D_admit; official test = D_test",
                       "split_protocol": data.get("split_protocol"), "comparison_reserved_at": "prepare",
                       "internal_val_fraction": data.get("internal_val_fraction"),
                       "num_clusters": data.get("num_clusters"), "buffer_hops": data.get("buffer_hops"),
                       "partitioner": data.get("partitioner"), "split_index": data.get("split_index"),
                       "download": data.get("download")},
        "exploration": {"trajectories": expl.get("trajectories"), "record_stride": expl.get("record_stride")},
        "controller": {"crossfit_folds": ctrl.get("crossfit_folds"), "observer_period": ctrl.get("observer_period"),
                       "horizons": ctrl.get("horizons"), "beam_width": ctrl.get("beam_width"), "top_k": ctrl.get("top_k"),
                       "calibration_protocol": ctrl.get("calibration_protocol")},
        "selection": {"outer_folds": sel.get("outer_folds"), "inner_folds": sel.get("inner_folds"),
                      "bootstrap_reps": sel.get("bootstrap_reps"), "alpha": sel.get("alpha"),
                      "epsilon_r": sel.get("epsilon_r"), "epsilon_eq": sel.get("epsilon_eq"),
                      "epsilon_metric": sel.get("epsilon_metric"), "enforce_entry_contract": sel.get("enforce_entry_contract")},
        "admission": {"rule": adm.get("rule"), "bootstrap_reps": adm.get("bootstrap_reps"), "alpha": adm.get("alpha"),
                      "epsilon_a": adm.get("epsilon_a"), "clip": adm.get("clip")},
        "registry": cfg.get("registry"),
    }


def write_registered(out_dir: Path, configs_dir: Path) -> list[Path]:
    written = []
    for config_path in sorted(configs_dir.glob("*_*.json")):
        manifest = registered_manifest(config_path)
        if manifest["dataset"] not in STEMS:
            continue
        path = out_dir / f"{STEMS[manifest['dataset']]}_{manifest['profile']}.json"
        _dump(manifest, path)
        written.append(path)
    index = {"schema": SCHEMA_REGISTERED, "manifests": [p.name for p in written],
             "note": "registered design of every shipped configuration; what a run will use, not what a run did use"}
    _dump(index, out_dir / "INDEX.json")
    return written


# ------------------------------------------------------------------ run manifests

def run_manifest(run_root: Path, *, with_indices: bool, out_dir: Path, stem: str) -> dict:
    state = _json(run_root / "run_state.json")
    config = _json(run_root / "config.json")
    meta = _json(run_root / "prepared" / "metadata.json")
    summary = _json(run_root / "final" / "results_summary.json")
    env = _json(run_root / "environment.json") if (run_root / "environment.json").is_file() else {}
    select = _json(run_root / "select" / "decision.json")
    admit = _json(run_root / "admit" / "decision.json")
    lock = _json(run_root / "policy_lock.json") if (run_root / "policy_lock.json").is_file() else {}
    arrays = {}
    for name in ("splits.npz", "official_splits.npz", "clusters.npz", "graph_splits.npz"):
        path = run_root / "prepared" / name
        if path.is_file():
            arrays[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    manifest = {
        "schema": SCHEMA_RUN,
        "dataset": (state.get("task") or {}).get("dataset"),
        "benchmark_dataset": meta.get("benchmark_dataset"),
        "data_kind": state.get("dataset"),
        "is_benchmark_data": meta.get("is_benchmark_data"),
        "profile": state.get("profile"),
        "scientific_run": state.get("scientific_run"),
        "run_name": run_root.name,
        "identities": {"source_sha256": state.get("source_sha256"), "config_sha256": state.get("config_sha256"),
                       "prepared_fingerprint": meta.get("fingerprint"),
                       "unit_sha256": (arrays.get("splits.npz") or {}).get("sha256")},
        "seeds": {"partition_seed": meta.get("partition_seed", config.get("seed")),
                  "training_seeds": (config.get("training") or {}).get("seeds"),
                  "ensemble_seeds": summary.get("ensemble_seeds"),
                  "bootstrap_seeds": {"selection": select.get("bootstrap_seed"),
                                      "admission": (admit.get("loss") or {}).get("bootstrap_seed"),
                                      "final": (summary.get("paired_qD_minus_qC_loss") or {}).get("bootstrap_seed")},
                  "split_index": (config.get("data") or {}).get("split_index")},
        "splits": {"split_protocol": meta.get("data_role_protocol"),
                   "comparison_reserved_at": meta.get("comparison_holdout_reserved_at"),
                   "unit": meta.get("split_unit"), "rule": meta.get("split_rule"), "sizes": meta.get("split_sizes"),
                   "role_hashes": meta.get("role_hashes"), "official_split_hashes": meta.get("official_split_hashes"),
                   "official_split_axis": meta.get("official_split_axis"),
                   "graph_role_hashes": meta.get("graph_role_hashes"),
                   "statistical_unit": (meta.get("buffer_audit") or {}).get("statistical_unit"),
                   "independent_graphs": meta.get("independent_graphs"),
                   "arrays": arrays},
        "source_data": {"source": meta.get("source"), "source_array_hashes": meta.get("source_array_hashes"),
                        "num_nodes": meta.get("num_nodes"), "num_graphs": meta.get("num_graphs"),
                        "num_features": meta.get("num_features"), "num_outputs": meta.get("num_outputs")},
        "policy_lock": {"comparator_hash": lock.get("comparator_hash"), "deployed_hash": lock.get("deployed_hash"),
                        "model_hashes": lock.get("model_hashes")},
        "environment": {k: env.get(k) for k in ("python", "python_implementation", "torch", "cuda_runtime", "gpu",
                                                "device", "platform", "machine", "numpy", "precision",
                                                "deterministic_algorithms_enabled", "cudnn_deterministic", "tf32_matmul")},
        "packages": env.get("packages"),
    }
    if with_indices:
        for name in arrays:
            shutil.copyfile(run_root / "prepared" / name, out_dir / f"{stem}_{name}")
        manifest["index_arrays_copied"] = [f"{stem}_{name}" for name in arrays]
    return manifest


def campaign_manifest(root: Path, *, with_indices: bool, out_dir: Path, stem: str) -> dict:
    splits = sorted(p for p in root.glob("split_*") if is_run(p))
    parts = [run_manifest(p, with_indices=with_indices, out_dir=out_dir, stem=f"{stem}_{p.name}") for p in splits]
    base = json.loads(json.dumps(parts[0]))
    base["run_name"] = root.name
    base["campaign"] = {"splits": [m["seeds"]["split_index"] for m in parts],
                        "per_split": [{"run_name": m["run_name"], "split_index": m["seeds"]["split_index"],
                                       "identities": m["identities"], "arrays": m["splits"]["arrays"],
                                       "unit_sha256": m["identities"]["unit_sha256"],
                                       "role_hashes": m["splits"]["role_hashes"],
                                       "sizes": m["splits"]["sizes"]} for m in parts]}
    if with_indices:
        for row, part in zip(base["campaign"]["per_split"], parts):
            row["index_arrays_copied"] = part.get("index_arrays_copied", [])
        base["index_arrays_copied"] = [name for part in parts for name in part.get("index_arrays_copied", [])]
    return base


def write_runs(found, out_dir: Path, *, with_indices: bool, registrations: dict | None = None) -> list[Path]:
    written = []
    if found:
        out_dir.mkdir(parents=True, exist_ok=True)
    for stem, path in found:
        entry = (registrations or {}).get(stem)
        if entry is not None:
            check_present(entry, path)
        if is_campaign(path) and not is_run(path):
            manifest = campaign_manifest(path, with_indices=with_indices, out_dir=out_dir, stem=stem)
        else:
            manifest = run_manifest(path, with_indices=with_indices, out_dir=out_dir, stem=stem)
        if manifest["dataset"] in DATASET_ORDER:
            stem = stem_for(manifest["dataset"])
        if entry is not None:
            manifest["registration"] = registration_descriptor(entry)
            check_registration_manifest(entry, manifest, out_dir)
        target = out_dir / f"{stem}.json"
        _dump(manifest, target)
        written.append(target)
    return written


def check_runs(found, out_dir: Path, *, registrations: dict | None = None) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        fresh = Path(tmp)
        write_runs(found, fresh, with_indices=False, registrations=registrations)
        problems = []
        for path in sorted(fresh.glob("*.json")):
            committed = out_dir / path.name
            if not committed.is_file():
                problems.append(f"missing committed manifest {committed}")
                continue
            a, b = json.loads(path.read_text()), json.loads(committed.read_text())
            a.pop("index_arrays_copied", None); b.pop("index_arrays_copied", None)
            for value in (a, b):
                for part in (value.get("campaign") or {}).get("per_split", []):
                    part.pop("index_arrays_copied", None)
            if a != b:
                problems.append(f"{committed} differs from a fresh extraction")
            if registrations and path.stem in registrations:
                try:
                    check_registration_manifest(registrations[path.stem], _json(committed), out_dir)
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    problems.append(f"{committed}: {exc}")
        for p in problems:
            print("CHECK FAILED:", p, file=sys.stderr)
        if problems:
            return 1
    print(f"{out_dir}: {len(found)} run manifests match a fresh extraction")
    return 0


def check_manifest_files(entries: list[dict], out_dir: Path) -> None:
    """An absent payload must not make an orphaned committed manifest disappear from checks."""
    if not out_dir.exists():
        return
    if out_dir.is_symlink() or not out_dir.is_dir():
        raise EvidenceError(f"expected a regular manifest directory: {out_dir}")
    allowed = {f"{entry['stem']}.json" for entry in entries}
    for entry in entries:
        path = out_dir / f"{entry['stem']}.json"
        if path.is_file() and not path.is_symlink():
            manifest = _json(path)
            allowed.update(manifest.get("index_arrays_copied", []))
            for part in (manifest.get("campaign") or {}).get("per_split", []):
                allowed.update(part.get("index_arrays_copied", []))
    stale = [p.relative_to(out_dir).as_posix() for p in out_dir.rglob("*")
             if p.is_symlink() or (p.is_file() and p.relative_to(out_dir).as_posix() not in allowed)]
    if stale:
        raise EvidenceError("unregistered or stale executed manifest files: " + ", ".join(sorted(stale)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=("registered", "runs"))
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--label", default="examples")
    parser.add_argument("--output-root", type=Path, default=ROOT / "manifests")
    parser.add_argument("--with-indices", action="store_true", help="copy the membership arrays next to the manifest")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--allow-empty", action="store_true",
                        help="report a valid empty official index without creating executed manifests")
    parser.add_argument("--allow-deposited-absent", action="store_true",
                        help="check committed metadata for absent registered payloads; public DOI and payload verification remains deferred")
    args = parser.parse_args(argv)
    for flag, name in ((args.allow_empty, "--allow-empty"), (args.allow_deposited_absent, "--allow-deposited-absent")):
        if flag and not (args.what == "runs" and args.index is not None and args.runs_root is None and args.label == "official"):
            parser.error(f"{name} requires runs --index FILE --label official")
    try:
        if args.what == "registered":
            out = args.output_root / "registered"
            if args.check:
                with tempfile.TemporaryDirectory() as tmp:
                    fresh = write_registered(Path(tmp), ROOT / "configs")
                    for path in fresh:
                        committed = out / path.name
                        if not committed.is_file() or json.loads(path.read_text()) != json.loads(committed.read_text()):
                            print(f"CHECK FAILED: {committed} is missing or stale", file=sys.stderr)
                            return 1
                print(f"{out}: {len(fresh)} registered manifests match configs/")
                return 0
            written = write_registered(out, ROOT / "configs")
            print(f"wrote {len(written)} registered manifests to {out}")
            return 0
        if not (args.runs_root or args.index):
            parser.error("runs requires --runs-root or --index")
        found = (manifest_runs_from_index(args.index, allow_empty=args.allow_empty,
                                          allow_deposited_absent=args.allow_deposited_absent)
                 if args.index else discover(args.runs_root))
        out = args.output_root / args.label
        entries = _json(args.index)["runs"] if args.index else []
        registrations = {entry["stem"]: entry for entry in entries if entry.get("run_identities")}
        if args.allow_deposited_absent:
            held, problems = check_deposited_absent(args.index, out)
            for problem in problems:
                print("CHECK FAILED:", problem, file=sys.stderr)
            if problems:
                return 1
            if held:
                check_manifest_files(entries, out)
                print(f"{out}: {len(held)} absent payload(s) ({', '.join(held)}); committed registration metadata agrees "
                      "with the index. Deposit availability and payload bytes are NOT VERIFIED here; "
                      "make fetch-deposits retrieves and verifies them.")
                if not found:
                    return 0
        if not found:
            require_no_executed_manifests(out)
            print(f"{out}: 0 official runs; no executed manifests")
            return 0
        if args.check:
            return check_runs(found, out, registrations=registrations)
        written = write_runs(found, out, with_indices=args.with_indices, registrations=registrations)
        print(f"wrote {len(written)} run manifests to {out}")
        return 0
    except (EvidenceError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
