#!/usr/bin/env python3
"""Compare official outcomes and sealed model counts with the manuscript.

An empty index cannot pass this check. Synthetic fixtures never count as an
official benchmark. Each checkpoint inventory must agree with Table S5 at its
printed precision; individual outcome comparisons use the sealed-run validator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts._paper_common import DATASETS, STEMS


def inspect_index(index_path: Path) -> tuple[list[dict], list[str]]:
    index_path = Path(index_path).resolve()
    if not index_path.is_file():
        return [], ["No official run index; ingest verified full-profile archives first."]
    value = json.loads(index_path.read_text())
    if value.get("schema") != "cross-morse-runs-index-1" or not isinstance(value.get("runs"), list):
        raise ValueError("Invalid official run index schema")
    entries, problems, seen, resolved_paths = [], [], set(), set()
    for row in value["runs"]:
        if not isinstance(row, dict):
            problems.append("Run index contains a non-object entry")
            continue
        dataset = row.get("dataset")
        if dataset not in DATASETS or dataset in seen:
            problems.append(f"Unexpected or duplicate dataset: {dataset}")
            continue
        seen.add(dataset)
        if row.get("verified") is not True or row.get("scientific_run") is not True or row.get("profile") != "full":
            problems.append(f"{dataset}: a verified official full-profile archive is required")
            continue
        relative = row.get("path")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            problems.append(f"{dataset}: unsafe archive path")
            continue
        path = index_path.parent / relative
        if path.is_symlink() or not path.resolve().is_relative_to(index_path.parent) or not path.is_dir():
            problems.append(f"{dataset}: archive directory missing or outside index root")
            continue
        resolved = path.resolve()
        if resolved in resolved_paths:
            problems.append(f"{dataset}: duplicate archive directory")
            continue
        resolved_paths.add(resolved)
        if dataset == "Roman-empire":
            children = sorted(resolved.glob("split_*"))
            if len(children) != 10 or {p.name for p in children} != {f"split_{i:02d}" for i in range(10)}:
                problems.append("Roman-empire: all ten official split directories are required")
                continue
        else:
            children = [resolved]
        try:
            for child in children:
                config = json.loads((child / "config.json").read_text())
                if config.get("dataset") != dataset or config.get("profile") != "full" or config.get("scientific_run") is not True:
                    raise ValueError("archive configuration differs from index identity")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            problems.append(f"{dataset}: {exc}")
            continue
        entries.append({"dataset": dataset, "path": path.resolve(), "stem": STEMS[dataset]})
    accepted = {row["dataset"] for row in entries}
    missing = sorted(set(DATASETS) - accepted)
    if missing:
        problems.append("Missing official full-profile comparisons: " + ", ".join(missing))
    return entries, problems


def check_model_inventories(run_root: Path, dataset: str, reported_millions: float) -> dict:
    """Check sealed inventories and checkpoint bytes, without loading models.

    Roman's Table S5 comparison covers all ten split inventories. Agreement in
    parameter count is necessary; it does not establish architecture identity.
    """
    from cross_morse.review import NumericEvidence
    from scripts.scientific_evidence_audit import validate_model_inventory

    root = Path(run_root)
    paths = [root / f"split_{i:02d}" for i in range(10)] if dataset == "Roman-empire" else [root]
    checks = []
    for path in paths:
        label = path.relative_to(root).as_posix()
        try:
            evidence = NumericEvidence(path)
            config = evidence.json("config.json")
            if (config.get("dataset") != dataset or config.get("profile") != "full"
                    or config.get("scientific_run") is not True):
                raise ValueError("Model inventory requires the declared official full-profile run")
            inventory = evidence.json("train/model_inventory.json")
            checkpoints = inventory.get("checkpoints")
            if not isinstance(checkpoints, list) or not checkpoints:
                raise ValueError("Actual checkpoint parameter inventories are required")
            inventory_paths = [item["path"] for item in checkpoints]
            if any(not isinstance(relative, str) or not relative for relative in inventory_paths):
                raise ValueError("Checkpoint inventory paths must be nonempty strings")
            if len(inventory_paths) != len(set(inventory_paths)):
                raise ValueError("Duplicate checkpoint paths in model inventory")
            # The train stage freezes and inventories every direct *_best.pt
            # checkpoint, including q0. Derive membership from sealed outputs,
            # independently of the inventory being checked.
            trained_paths = {relative for relative in evidence.sealed
                             if Path(relative).parent == Path("train") and relative.endswith("_best.pt")}
            if not trained_paths:
                raise ValueError("No sealed trained best checkpoints are available")
            if set(inventory_paths) != trained_paths:
                raise ValueError("Model inventory checkpoint membership differs from sealed training outputs: "
                                 f"missing={sorted(trained_paths - set(inventory_paths))}; "
                                 f"unexpected={sorted(set(inventory_paths) - trained_paths)}")
            baseline = evidence.json("provenance/frozen_models.json")
            frozen = baseline.get("checkpoints")
            if (baseline.get("captured_after_stage") != "train" or not isinstance(frozen, dict)
                    or set(frozen) != trained_paths):
                raise ValueError("Frozen-model baseline membership differs from sealed trained checkpoints")
            result = validate_model_inventory(inventory, reported_millions, configuration=config)
            for checkpoint in checkpoints:
                relative = checkpoint["path"]
                evidence.file(relative)
                if checkpoint.get("sha256") != evidence.sealed[relative]["sha256"]:
                    raise ValueError(f"Checkpoint identity differs from its model inventory: {relative}")
                if frozen[relative].get("artifact") != evidence.sealed[relative]:
                    raise ValueError(f"Checkpoint identity differs from its frozen-model baseline: {relative}")
            checks.append({"run": label, **result, "checkpoint_bytes_verified": True,
                           "checkpoint_inventory_complete": True, "checkpoint_count": len(checkpoints)})
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            checks.append({"run": label, "status": "FAIL", "issue": str(exc)})
    return {"status": "PASS" if all(row["status"] == "PASS" for row in checks) else "FAIL",
            "reported_parameters_M": reported_millions, "runs": checks,
            "scope": "Sealed checkpoint inventories and bytes; Table S5 parameter-count agreement at printed precision. "
                     "Architecture and manuscript-result provenance require separate evidence."}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=ROOT / "runs/index.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "cm-reports/compare")
    args = parser.parse_args(argv)
    try:
        entries, problems = inspect_index(args.index)
        backbones = json.loads((ROOT / "paper/registered_constants.json").read_text())["backbones_table_S5"]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report_dir = Path(tempfile.mkdtemp(prefix="comparison-", dir=args.output_dir))
        results = []
        for row in entries:
            models = check_model_inventories(row["path"], row["dataset"], backbones[row["dataset"]]["parameters_M"])
            if models["status"] != "PASS":
                problems.append(f"{row['dataset']}: sealed model inventory does not satisfy Table S5")
            command = [sys.executable, "-B", str(ROOT / "scripts/compare_with_manuscript.py"), str(row["path"]),
                       "--output", str(report_dir / (row["stem"] + ".json"))]
            if row["dataset"] == "Roman-empire":
                command.append("--roman-campaign")
            result = subprocess.run(command, check=False)
            results.append({"dataset": row["dataset"], "exit_code": result.returncode, "model_inventory": models})
            if result.returncode:
                problems.append(f"{row['dataset']}: sealed manuscript comparison failed")
        report = {"schema": "cross-morse-official-comparison-coverage-1", "status": "PASS" if not problems else "INCOMPLETE_OR_INVALID",
                  "required_datasets": len(DATASETS), "compared_datasets": len(results), "results": results, "issues": problems,
                  "model_inventory_verified_datasets": sum(row["model_inventory"]["status"] == "PASS" for row in results),
                  "report_directory": str(report_dir),
                  "scope": "Official-run coverage, sealed model inventories against Table S5 and descriptive outcome comparison; "
                           "complete scientific evidence is checked separately"}
        (report_dir / "coverage.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 0 if not problems else 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
