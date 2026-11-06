#!/usr/bin/env python3
"""Report repeated patterns in resource and dispersion columns for provenance review.

    python -B scripts/check_measurement_degeneracy.py --records records            # deposited `baselines` set
    python -B scripts/check_measurement_degeneracy.py --registry                   # transcribed Table S15b
    python -B scripts/check_measurement_degeneracy.py --records records --require  # fail when the set is absent
    python -B scripts/check_measurement_degeneracy.py --registry --strict          # optional editorial gate

The per-rerun resource ledger (Supplementary Table S15b) needs the original timing and accounting
records (E17). This diagnostic computes, for the external-baseline rows:

  T1  the serial-cost multiplier of each external method relative to the plant row is identical
      (to 3 decimals) on every dataset where the method appears (>= 2 datasets);
  T2  wall time equals serial cost on every external row (>= 4 rows);
  T3  the two external methods of a dataset report the same score SD on every dataset, and the ratio
      SD_external / SD_internal is the same constant (to 0.02) on >= 4 datasets;
  T4  the GPU-hour multiplier relative to the plant takes <= 3 distinct values over >= 8 external rows.

Repeated or rounded values can arise from genuine observations, shared accounting conventions,
or protocol constants. These patterns cannot distinguish those explanations. Findings request
inspection of the original records and definitions; they neither establish measurement origin
nor require changing measured values. `make records-check` and the scientific audit perform the
separate provenance and coverage checks. Patterns are advisory by default; --strict makes them
an explicit editorial gate. Standard library only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cm_campaigns.contract import EXTERNAL  # noqa: E402

PLANT = re.compile(r"\(qC\)|full-depth plant|best fixed sequence", re.I)


def external_methods(dataset: str) -> set[str]:
    return set(EXTERNAL.get(dataset, []))


# ------------------------------------------------------------------ inputs
def rows_from_registry(root: Path = ROOT) -> list[dict]:
    registry = json.loads((root / "paper" / "reported_results.json").read_text(encoding="utf-8"))
    ledger = registry["supplementary_table_S15b_per_rerun_resource_ledger"]
    rows = []
    for dataset, entries in ledger.items():
        for entry in entries:
            method = entry["method"]
            rows.append({"dataset": dataset, "method": method, "serial_cost": entry.get("serial_cost_C_ctrl"),
                         "wall_time": entry.get("wall_time_nominal_1"), "gpu_hours": entry.get("GPU_h"),
                         "sd": entry.get("SD_12_seeds"), "external": method in external_methods(dataset),
                         "plant": bool(PLANT.search(method))})
    # Table S15 carries the full-depth plant row, the natural reference of every retraining multiplier.
    comparison = registry.get("supplementary_table_S15_method_comparison", {})
    for dataset, block in comparison.items():
        for entry in block.get("rows", []):
            if entry.get("method") == "full-depth plant" and not any(r["dataset"] == dataset and r["method"] == entry["method"] for r in rows):
                try:
                    serial = float(entry.get("C_ctrl"))
                except (TypeError, ValueError):
                    serial = None
                rows.append({"dataset": dataset, "method": "full-depth plant", "serial_cost": serial, "wall_time": None,
                             "gpu_hours": entry.get("GPU_h"), "sd": None, "external": False, "plant": True})
    return rows


def rows_from_records(records_root: Path) -> list[dict] | None:
    summary = records_root / "baselines" / "summary.csv"
    if not summary.is_file():
        return None
    cells: dict[tuple[str, str], dict] = {}
    scores: dict[tuple[str, str], dict[str, float]] = {}
    with summary.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["dataset"], row["method"])
            quantity = row["quantity"].strip()
            value = float(str(row["value"]).replace("−", "-"))
            if not math.isfinite(value):
                raise ValueError(f"Non-finite baseline value for {key}: {quantity}")
            if quantity == "score" and row.get("seed", "").strip():
                seed = row["seed"].strip()
                if seed in scores.setdefault(key, {}):
                    raise ValueError(f"Duplicate score seed for {key}: {seed}")
                scores[key][seed] = value
            else:
                if quantity in cells.setdefault(key, {}):
                    raise ValueError(f"Duplicate baseline quantity for {key}: {quantity}")
                cells[key][quantity] = value
    rows = []
    for dataset, method in sorted(set(cells) | set(scores)):
        quantities = cells.get((dataset, method), {})
        seeds = list(scores.get((dataset, method), {}).values())
        rows.append({"dataset": dataset, "method": method, "serial_cost": quantities.get("serial_cost"),
                     "wall_time": quantities.get("wall_time"), "gpu_hours": quantities.get("gpu_hours"),
                     "sd": statistics.stdev(seeds) if len(seeds) >= 2 else None,
                     "external": method in external_methods(dataset), "plant": bool(PLANT.search(method))})
    return rows


# ------------------------------------------------------------------ tests
def _references(rows: list[dict]) -> dict[str, dict[str, dict]]:
    """Internal reference rows per dataset, keyed by reference name ('qC' or 'full-depth plant')."""
    refs: dict[str, dict[str, dict]] = {}
    for r in rows:
        if r["external"] or not r["plant"]:
            continue
        name = "qC" if "(qC)" in r["method"] else "full-depth plant" if "full-depth plant" in r["method"] else r["method"]
        refs.setdefault(name, {})[r["dataset"]] = r
    return refs


def analyse(rows: list[dict]) -> dict:
    seen = set()
    for row in rows:
        key = (row["dataset"], row["method"])
        if key in seen:
            raise ValueError(f"Duplicate dataset/method row: {key}")
        seen.add(key)
        for quantity in ("serial_cost", "wall_time", "gpu_hours", "sd"):
            value = row.get(quantity)
            if value is not None and (isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0):
                raise ValueError(f"Invalid {quantity} for {key}")
    findings = []
    external = [r for r in rows if r["external"]]
    references = _references(rows)
    # T1: identical serial multiplier per method across datasets, relative to any internal reference
    for ref_name, plants in sorted(references.items()):
        multipliers: dict[str, list[tuple[str, float]]] = {}
        for r in external:
            plant = plants.get(r["dataset"])
            if plant and r.get("serial_cost") and plant.get("serial_cost"):
                multipliers.setdefault(r["method"], []).append((r["dataset"], round(r["serial_cost"] / plant["serial_cost"], 3)))
        for method, values in sorted(multipliers.items()):
            if len(values) >= 2 and len({v for _, v in values}) == 1:
                findings.append({"test": "T1", "method": method,
                                 "detail": f"serial cost is {values[0][1]:.3f} x {ref_name} on all {len(values)} datasets: "
                                           + ", ".join(d for d, _ in values)})
    # T2: wall time identical to serial cost on every external row
    timed = [r for r in external if r.get("wall_time") is not None and r.get("serial_cost") is not None]
    if len(timed) >= 4 and all(abs(r["wall_time"] - r["serial_cost"]) < 1e-9 for r in timed):
        findings.append({"test": "T2", "detail": f"wall time equals serial cost exactly on all {len(timed)} external rows"})
    # T3: shared SD across the external methods and a constant external/internal SD ratio
    internal_sd = {}
    for r in rows:
        if not r["external"] and r.get("sd") is not None:
            internal_sd.setdefault(r["dataset"], r["sd"])
    ratios = []
    shared = 0
    datasets_with_two = 0
    for dataset in sorted({r["dataset"] for r in external}):
        ext = [r for r in external if r["dataset"] == dataset and r.get("sd") is not None]
        if len(ext) >= 2:
            datasets_with_two += 1
            if len({round(r["sd"], 6) for r in ext}) == 1:
                shared += 1
        if ext and internal_sd.get(dataset):
            ratios.append(round(ext[0]["sd"] / internal_sd[dataset], 2))
    if datasets_with_two >= 3 and shared == datasets_with_two:
        findings.append({"test": "T3a", "detail": f"both external methods report the same score SD on all {shared} datasets"})
    if len(ratios) >= 4 and max(ratios) - min(ratios) <= 0.05:
        findings.append({"test": "T3b", "detail": f"SD_external / SD_internal is {statistics.mean(ratios):.2f} (range {min(ratios)}-{max(ratios)}) on {len(ratios)} datasets"})
    # T4: GPU-hour multipliers take very few distinct values, relative to any internal reference
    for ref_name, plants in sorted(references.items()):
        gpu = []
        for r in external:
            plant = plants.get(r["dataset"])
            if plant and r.get("gpu_hours") and plant.get("gpu_hours"):
                gpu.append(round(r["gpu_hours"] / plant["gpu_hours"], 2))
        if len(gpu) >= 8 and len(set(gpu)) <= 3:
            findings.append({"test": "T4", "detail": f"GPU-hours relative to {ref_name} take only {sorted(set(gpu))} over {len(gpu)} external rows"})
    return {"schema": "cross-morse-degeneracy-check-1", "external_rows": len(external), "rows": len(rows),
            "findings": findings,
            "status": "REVIEW_REQUIRED" if findings else "NO_PATTERNS_DETECTED" if external else "NOT_ASSESSED",
            "measurement_origin_verified": False,
            "scope": "Arithmetic pattern diagnostic only; source-record validation and measurement origin are separate."}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--records", type=Path, help="records root holding a deposited `baselines` set")
    source.add_argument("--registry", action="store_true", help="check the transcribed Table S15b of paper/reported_results.json")
    parser.add_argument("--require", action="store_true", help="fail when the record set is absent")
    parser.add_argument("--warn-only", action="store_true", help="report findings but exit 0")
    parser.add_argument("--strict", action="store_true", help="exit 1 on pattern findings (optional editorial gate, not proof of invalid measurements)")
    parser.add_argument("--output", type=Path, help="JSON report path")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    try:
        if args.registry:
            rows, label = rows_from_registry(args.root), "paper/reported_results.json (Table S15b)"
        else:
            rows, label = rows_from_records(args.records), f"{args.records}/baselines/summary.csv"
            if rows is None:
                report = {"schema": "cross-morse-degeneracy-check-1", "status": "NOT_ASSESSED", "source": label,
                          "external_rows": 0, "rows": 0, "findings": [], "measurement_origin_verified": False,
                          "reason": "No baselines record set is available."}
            else:
                report = analyse(rows)
        if args.registry:
            report = analyse(rows)
        report["source"] = label
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"{report['status']}: {label}; {report['external_rows']} external rows examined")
    for f in report["findings"]:
        print(f"  [{f['test']}] {f.get('method', '')} {f['detail']}".replace("  ", " "))
    if report["findings"]:
        print("  Inspect the original E17 traces, scheduler records, rounding and quantity definitions. "
              "A pattern does not establish whether values were measured or specified by protocol.")
    if rows is None:
        print(report["reason"])
        return 1 if args.require else 0
    return 1 if report["findings"] and args.strict and not args.warn_only else 0


if __name__ == "__main__":
    raise SystemExit(main())
