#!/usr/bin/env python3
"""Write a measurement-record set from the results of scripts/run_sweep.py.

    python -B scripts/export_sweep_records.py --results /abs/sweeps/hb/results.json \
        --hardware "NVIDIA GH200 96 GB, Arrhenius" --period "2026-11-02/2026-11-09"

The set is ``<root>/<record set>/`` (default root: records/) in the layout of
records/README.md: ``summary.csv`` with the contract's keys followed by
quantity, value, unit, source_file and locator; ``raw/computed/`` holding the
computed JSON files the values are read from, each with the SHA-256 of the sealed
run files it was computed from; the files the sweep deposits (attached ablation
studies under ``raw/ablations/``, buffer measurement jobs under ``raw/jobs/``);
``raw/results.json`` and ``raw/plan.json``;
``code/SOURCE.json``; ``provenance.json`` and ``MANIFEST.json``. The contract is
``records/specs.json`` (``sets[<record set>]``). The set is verified with
scripts/records.py (every value re-read from its raw file) and the root's
index.json is rewritten for every set it holds. Synthetic example sweeps are
refused under records/.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._paper_common import runtime_source_fingerprint  # noqa: E402
from scripts.export_records import ExportError, dump_json, fmt, git_head, sha256_file  # noqa: E402
from scripts.records import (MANIFEST, RECORDS, RecordsError, is_official_root, verify_set,  # noqa: E402
                             write_index, write_manifest)

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_TAIL = ["quantity", "value", "unit", "source_file", "locator"]
RESULTS_SCHEMA = "cm-sweep-results-1"
PROTOCOLS = {
    "horizon_beam_sweep": "MPC-RH configurations of the parent's deployed profile and ensemble; registered horizons "
                          "and beam width read from the parent run, the others from lifecycle runs whose configuration "
                          "differs from the parent only in controller.horizons or controller.beam_width; nested D_select "
                          "effects and work ledger of select/; headroom bound of the parent",
    "sensitivity_grids": "one registered parameter varied at a time; selection margins by main Eq. (19) on the sealed "
                         "simultaneous upper limits, clip level by the admission replay with the stage's seed, support "
                         "masks by the registered rule on the logged per-cell statistics, support routes by lifecycle "
                         "variant runs through select",
    "module_ablations": "matched held-out contrasts sealed in confirm/ and nested D_select effects of select/ for "
                        "radius uncertainty, sequence risk and the Full/Lean profile; for the other modules, variant "
                        "lifecycles of cm_experiments/ablation_lifecycle.py (byte copies of the parent's prepare/train "
                        "outputs, explore->confirm re-executed with the declared variant installed) executing the "
                        "audited rule on the held-out role with every arm's controller, paired ablated minus complete; "
                        "the comparator branch from sealed input_gate and negative_controls runs",
    "buffer_sensitivity": "descriptive admission at each cluster buffer k: the sealed per-core clipped paired losses of "
                          "admit/paired_units.npz regrouped into blocks (cores linked when the k-step random-walk entry "
                          "dependence exceeds its maximum over core pairs at the registered buffer; roots = the "
                          "lexicographically first maximum independent set of the link graph, every other core joins "
                          "the root it depends on most) and cross_morse.statistics.admission applied to the blocks "
                          "with the sealed replicates, alpha, clip and seed; the registered buffer reproduces "
                          "admit/decision.json exactly (cm_experiments/buffered_admission.py)",
}


def merged_specs() -> dict:
    """The registered record contracts (records/specs.json)."""
    return json.loads((RECORDS / "specs.json").read_text(encoding="utf-8"))


def index_root(root: Path, specs: dict) -> list[dict]:
    """scripts/records.py index over every set of the root."""
    entries = []
    for path in sorted(p for p in root.iterdir() if p.is_dir()):
        if path.name not in specs["sets"]:
            raise ExportError(f"{path.name}: not a record set of records/specs.json")
        write_manifest(path)
        entries.append(verify_set(path, specs["sets"][path.name], specs))
    write_index(root, specs, entries)
    return entries


def export(results_path: Path, *, root: Path, hardware: str, period: str, replace: bool,
           argv: list[str]) -> Path:
    results_path = Path(results_path).resolve()
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if results.get("schema") != RESULTS_SCHEMA:
        raise ExportError("expected results.json of scripts/run_sweep.py collect")
    name = results["record_set"]
    specs = merged_specs()
    from cm_experiments.sweeps import SWEEPS
    if name not in specs["sets"] or name not in SWEEPS:
        raise ExportError(f"{name} is not a sweep contract of records/specs.json")
    spec = specs["sets"][name]
    if not results["rows"]:
        raise ExportError(f"{name}: the sweep collected no values")
    root = Path(root).resolve()
    if results.get("synthetic") and is_official_root(root):
        raise ExportError("synthetic example sweeps cannot be exported into records/; pass --root")
    base = results_path.parent
    computed = {}
    for row in results["rows"]:
        if row["source_file"] not in computed:
            computed[row["source_file"]] = json.loads((base / row["source_file"]).read_text(encoding="utf-8"))
        pointer = row["locator"].removeprefix("json:/values/")
        if computed[row["source_file"]]["values"].get(pointer) != row["value"]:
            raise ExportError(f"{name}: {row['keys']} / {row['quantity']} in results.json differs from its computed file")
    target = root / name
    if target.exists():
        if not replace:
            raise ExportError(f"{target} exists; pass --replace to rebuild it")
        shutil.rmtree(target)
    (target / "raw").mkdir(parents=True)
    (target / "code").mkdir()
    for relative in results["computed"]:
        destination = target / "raw" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(base / relative, destination)
    for item in results.get("deposits") or []:                 # files of attached ablation experiments
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] in ("computed", "results.json",
                                                                                     "plan.json"):
            shutil.rmtree(target)
            raise ExportError(f"{name}: deposit path {item['path']!r} is not a relative raw/ path of its own")
        destination = target / "raw" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(item["from"], destination)
        if sha256_file(destination) != item["sha256"]:
            shutil.rmtree(target)
            raise ExportError(f"{name}: {item['from']} changed after collection")
    shutil.copyfile(results_path, target / "raw" / "results.json")
    shutil.copyfile(base / "plan.json", target / "raw" / "plan.json")
    keys = spec["keys"]
    seen = set()
    with (target / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(keys + SUMMARY_TAIL)
        for row in results["rows"]:
            if set(row["keys"]) != set(keys) or row["quantity"] not in spec["quantities"]:
                raise ExportError(f"{name}: row {row['keys']} / {row['quantity']} is outside the contract")
            identity = tuple(row["keys"][k] for k in keys) + (row["quantity"],)
            if identity in seen:
                raise ExportError(f"{name}: duplicate value {identity}")
            seen.add(identity)
            writer.writerow([row["keys"][k] for k in keys] + [row["quantity"], fmt(row["value"]), row["unit"],
                                                              f"raw/{row['source_file']}", row["locator"]])
    experiment_code = ["cm_experiments/sweeps.py", "scripts/run_sweep.py"]
    if results.get("ablations"):
        experiment_code += ["cm_experiments/ablation_lifecycle.py", "scripts/run_ablation.py"]
    if name == "buffer_sensitivity":
        experiment_code.append("cm_experiments/buffered_admission.py")
    dump_json(target / "code" / "SOURCE.json", {
        "schema": "cross-morse-record-source-1", "exporter": "scripts/export_sweep_records.py",
        "experiment_code": experiment_code,
        "command": [Path(a).name if Path(a).is_absolute() else a for a in argv],
        "runtime_source_sha256": runtime_source_fingerprint(ROOT), "repository_commit": git_head(),
        "sweep_runtime_source_sha256": results.get("runtime_source_sha256"),
        "inputs": [{"file": "results.json", "sha256": sha256_file(results_path)},
                   {"file": "plan.json", "sha256": sha256_file(base / "plan.json")}]})
    provenance = {
        "experiment": f"{spec['title']} ({', '.join(spec['manuscript_tables'])})",
        "produced_by": (f"scripts/run_sweep.py plan/run/collect --sweep {results['sweep']} on "
                        f"{len(results['parents'])} parent(s) with {len(results['variants'])} variant lifecycle "
                        "run(s); scripts/export_sweep_records.py"),
        "hardware": hardware, "period": period, "protocol": PROTOCOLS[name],
        "synthetic": bool(results.get("synthetic")),
        "parents": [{"dataset": p["dataset"], "root": Path(p["root"]).name, "config_sha256": p["config_sha256"]}
                    for p in results["parents"]],
        "variants": [{"id": v["id"], "patch": v["patch"], "config_sha256": v["config_sha256"]}
                     for v in results["variants"]],
        "not_measured": results.get("not_measured", results.get("unsupported", [])),
        "ablation_experiments": [{"experiment": Path(a["experiment"]).name, "results_sha256": a["results_sha256"],
                                  "plan_sha256": a["plan_sha256"], "role": a.get("role")}
                                 for a in results.get("ablations", [])],
        "lifecycle_replicates": [{k: v for k, v in r.items() if k != "inputs"} for r in results.get("replicates", [])],
        "parent_runtime_source_mismatch": results.get("parent_runtime_source_mismatch", []),
    }
    for field in specs["provenance_fields"]:
        if not str(provenance.get(field, "")).strip():
            raise ExportError(f"{name}: provenance field {field!r} is empty; pass it explicitly")
    dump_json(target / "provenance.json", provenance)
    try:
        index_root(root, specs)
    except RecordsError as exc:
        (target / MANIFEST).unlink(missing_ok=True)
        raise ExportError(f"{name}: the written set does not verify: {exc}") from None
    return target


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True, help="results.json of run_sweep.py collect")
    parser.add_argument("--root", type=Path, default=RECORDS, help="records root (default records/)")
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--period", required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)
    command = ["scripts/export_sweep_records.py", *(argv if argv is not None else sys.argv[1:])]
    try:
        target = export(args.results, root=args.root, hardware=args.hardware, period=args.period,
                        replace=args.replace, argv=command)
    except (ExportError, RecordsError, OSError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
