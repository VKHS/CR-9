#!/usr/bin/env python3
"""Scientific consistency: decide each open question of paper/consistency_ledger.json from the original records.

The ledger lists the places where the printed methods or printed values and the released code (or two printed
statements) do not yet agree, the original record that decides each one and the rule that decides it. This tool

  * renders the ledger as docs/CONSISTENCY.md (--write; --check verifies it is current), and
  * evaluates the machine-checkable rules on completed runs (--runs-root DIR; --json; --output FILE):
      K1 selection_family           which studentized-maximum family reproduces each recorded selection
      K2 peptides_confirmation      the Peptides confirmation record against the registered frozen map
      K3 complete_ledger_selection  the registered rule on the complete candidate ledger of Roman, Pascal, COCO
      K4 seed_hierarchy             the executed controller-fit and plant-seed structure
      K6 entry_gates                whether any target-regime gate failed

Nothing here changes a reference value or a measurement. Smoke and synthetic runs are evaluated for their
structure only; the verdicts on the printed values need the full scientific runs.

    python -B scripts/consistency_audit.py --write
    python -B scripts/consistency_audit.py --check
    python -B scripts/consistency_audit.py --runs-root <suite>/runs [--output audit.json]

NumPy and the standard library only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LEDGER = "paper/consistency_ledger.json"
DOC = "docs/CONSISTENCY.md"
SCHEMA = "cross-morse-consistency-ledger-1"
DATASETS = ("ogbn-arxiv", "ogbn-proteins", "ogbn-products", "Roman-empire", "PascalVOC-SP", "COCO-SP", "Peptides-func")
LEDGER_DATASETS = ("Roman-empire", "PascalVOC-SP", "COCO-SP")
STATUS_WORDS = {"aligned": "Code aligned; records decide the printed values", "pending_records": "Awaiting original records",
                "design_difference": "Design difference; records decide", "needs_declaration": "Needs an author declaration",
                "resolved": "Resolved"}


def load_ledger(root: Path = ROOT) -> dict:
    ledger = json.loads((root / LEDGER).read_text(encoding="utf-8"))
    if ledger.get("schema") != SCHEMA or not isinstance(ledger.get("items"), list):
        raise ValueError(f"{LEDGER} has the wrong schema")
    displays = {d["id"] for d in json.loads((root / "paper/display_manifest.json").read_text(encoding="utf-8"))["displays"]}
    seen = set()
    for item in ledger["items"]:
        missing = {"id", "title", "displays", "printed", "code_before", "code_now", "decided_by", "rule", "status", "check"} - set(item)
        if missing:
            raise ValueError(f"{item.get('id')}: missing {sorted(missing)}")
        if item["id"] in seen or item["status"] not in ledger["statuses"]:
            raise ValueError(f"{item['id']}: duplicate identifier or unknown status")
        seen.add(item["id"])
        unknown = [d for d in item["displays"] if d not in displays]
        if unknown:
            raise ValueError(f"{item['id']}: unknown displays {unknown}")
        if item["check"] is not None and item["check"] not in CHECKS:
            raise ValueError(f"{item['id']}: no check named {item['check']}")
    return ledger


# ================================================================================= run discovery
def discover(runs_root: Path) -> dict[str, list[dict]]:
    """{dataset: [{"path", "benchmark", "split"}]} for every completed lifecycle under runs_root."""
    found: dict[str, list[dict]] = {}
    for run in sorted(p for p in Path(runs_root).iterdir() if p.is_dir()):
        members = sorted(run.glob("split_*")) if (run / "split_00").is_dir() else [run]
        for member in members:
            config_path = member / "config.json"
            if not config_path.is_file() or not (member / "select" / "decision.json").is_file():
                continue
            config = json.loads(config_path.read_text(encoding="utf-8"))
            dataset = config.get("benchmark_dataset") or config.get("dataset")
            if dataset not in DATASETS:
                continue
            benchmark = bool(config.get("dataset") == dataset and config.get("profile") == "full"
                             and config.get("scientific_run"))
            found.setdefault(dataset, []).append({"path": member, "benchmark": benchmark, "config": config,
                                                  "split": (config.get("data") or {}).get("split_index")})
    return found


def _json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _view(dataset: str, path: Path):
    from paper.build.displays.framework import RunView
    return RunView(dataset, path)


# ================================================================================= the checks
def selection_family(runs: dict, printed: dict) -> dict:
    """K1: replay each recorded selection with both families of the studentized maximum."""
    from cross_morse import statistics
    from paper.build.displays.run_audits import selection_inputs
    rows = []
    for dataset, members in runs.items():
        for member in members:
            data, candidates, decision, config, metric = selection_inputs(_view(dataset, member["path"]))
            sc = config["selection"]
            reproduced = []
            for family in statistics.MAX_T_FAMILIES:
                replay = statistics.nested_selection(
                    data["loss"], data.get("accuracy"), candidates, data["outer_fold"], inner_fold_ids=data["inner_fold"],
                    reps=int(decision["bootstrap_replicates"]), alpha=sc["alpha"], epsilon_r=sc["epsilon_r"],
                    epsilon_eq=sc["epsilon_eq"], epsilon_metric=sc["epsilon_metric"], weights=data["counts"],
                    seed=int(decision["bootstrap_seed"]), metric_evaluator=metric, max_t_family=family)
                same_u = all((a is None and b is None) or (a is not None and b is not None and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12))
                             for a, b in zip(replay["U"], decision["U"]))
                if (math.isclose(replay["max_radius"], decision["max_radius"], rel_tol=1e-9, abs_tol=1e-12) and same_u
                        and replay["joint_feasible"] == decision["joint_feasible"]):
                    reproduced.append(family)
            recorded = decision.get("max_t_family")
            feasible = sum(1 for f in decision.get("joint_feasible", []) if f is True)
            verdict = ("consistent" if reproduced == ["feasible"] or recorded == "feasible" and "feasible" in reproduced
                       else "executed family" if "executed" in reproduced and "feasible" not in reproduced
                       else "both families agree" if len(reproduced) == 2 else "not reproduced")
            rows.append({"dataset": dataset, "run": member["path"].name, "benchmark": member["benchmark"],
                         "recorded_family": recorded, "reproduced_by": reproduced, "feasible_family": feasible,
                         "max_radius": decision["max_radius"], "eligible": len(decision.get("Q_s", [])),
                         "printed_Q": (printed.get(dataset) or {}).get("Q_size") if member["benchmark"] else None,
                         "printed_r_max": (printed.get(dataset) or {}).get("r_max") if member["benchmark"] else None,
                         "verdict": verdict})
    if not rows:
        return {"status": "pending", "detail": "no completed run", "rows": []}
    benchmark = [r for r in rows if r["benchmark"]]
    if not benchmark:
        status = "structure only"
    elif any(r["verdict"] == "not reproduced" for r in benchmark):
        status = "needs attention"
    elif any(r["verdict"] == "executed family" for r in benchmark):
        status = "needs correction"
    else:
        status = "consistent"
    return {"status": status, "rows": rows,
            "detail": "Q_feas reproduces the record: Algorithm S4 as printed; only every executed configuration reproduces "
                      "it: the printed Algorithm S4 text or the selection must change (ledger K1)."}


def peptides_confirmation(runs: dict, printed: dict) -> dict:
    """K2: the Peptides confirmation record against the registered frozen confirmation map."""
    members = runs.get("Peptides-func", [])
    if not members:
        return {"status": "pending", "detail": "no completed Peptides-func run", "rows": []}
    rows = []
    for member in members:
        path = member["path"]
        confirm = _json(path / "confirm" / "decision.json", {})
        candidates = _json(path / "select" / "candidates.json", [])
        provisional, confirmed = confirm.get("provisional_index"), confirm.get("confirmed_index")
        comparator = confirm.get("comparator_index")
        contrasts = [c for c in confirm.get("contrasts", []) if c.get("candidate_index") == provisional]
        holm = [c.get("holm_p_value_zero") for c in contrasts]
        route = lambda j: None if j is None or j >= len(candidates) else ("qC" if j == comparator else candidates[j].get("route"))
        retained = confirmed == provisional and provisional != comparator
        significant = bool(contrasts) and all(c.get("passes") for c in contrasts)
        verdict = ("consistent" if retained and significant else
                   "returns qC" if confirmed == comparator and provisional != comparator else
                   "contradicts the map" if retained and not significant else
                   "no feedback selected")
        rows.append({"run": path.name, "benchmark": member["benchmark"], "provisional": route(provisional),
                     "confirmed": route(confirmed), "holm_p": holm, "alpha": confirm.get("alpha"),
                     "downgrades": confirm.get("downgrades", []), "verdict": verdict})
    benchmark = [r for r in rows if r["benchmark"]]
    status = ("structure only" if not benchmark else
              "consistent" if all(r["verdict"] == "consistent" and r["confirmed"] == "radius" for r in benchmark) else
              "needs correction")
    return {"status": status, "rows": rows,
            "detail": "Table 1 reports radius, E = 4: the record must retain radius with every confirmation contrast "
                      "Holm-significant, or a separate corrected confirmation record must support it (ledger K2)."}


def complete_ledger_selection(runs: dict, printed: dict) -> dict:
    """K3: the registered rule on the complete candidate ledger against the executed selection."""
    from paper.build.displays.run_audits import RunRecord
    from paper.build.selection import registered_rule_on_all_configurations
    rows = []
    for dataset in LEDGER_DATASETS:
        for member in runs.get(dataset, []):
            rec = RunRecord(_view(dataset, member["path"]))
            d = rec.decision
            everything = [{"index": j, "route": c["route"], "joint_feasible": d["joint_feasible"][j],
                           "feasible": c["feasible"], "select_upper": d["U"][j], "select_delta": d["mean"][j],
                           "complexity": c["complexity"], "ensemble": c["ensemble"], "cost": c["cost"], "hash": c["hash"]}
                          for j, c in enumerate(rec.candidates) if d["U"][j] is not None]
            chosen = registered_rule_on_all_configurations(
                {"selection": {"epsilon_r": d["epsilon_r"], "epsilon_eq": d["epsilon_eq"], "qC_index": rec.qc},
                 "candidates": everything})
            rows.append({"dataset": dataset, "run": member["path"].name, "benchmark": member["benchmark"],
                         "registered_rule": rec.rule(chosen), "executed_selection": rec.rule(rec.provisional),
                         "agrees": chosen == rec.provisional})
    if not rows:
        return {"status": "pending", "detail": "no completed Roman-empire, PascalVOC-SP or COCO-SP run", "rows": []}
    benchmark = [r for r in rows if r["benchmark"]]
    status = ("structure only" if not benchmark else
              "consistent" if all(r["agrees"] and r["registered_rule"] == "MPC-RH" for r in benchmark) else
              "needs correction")
    return {"status": status, "rows": rows,
            "detail": "The registered rule on the complete ledger must return MPC-RH, as the executed selection and "
                      "Table 1 do (ledger K3)."}


def seed_hierarchy(runs: dict, printed: dict) -> dict:
    """K4: the executed controller-fit and plant-seed structure of each run."""
    rows = []
    for dataset, members in runs.items():
        for member in members:
            path = member["path"]
            profiles = sorted(p.name for p in (path / "fit").iterdir() if p.is_dir() and (p / "controller.pt").is_file()) \
                if (path / "fit").is_dir() else []
            per_seed = sorted({p.parent.name for p in (path / "fit").rglob("controller.pt")} - set(profiles)) \
                if (path / "fit").is_dir() else []
            rows.append({"dataset": dataset, "run": path.name, "benchmark": member["benchmark"],
                         "plant_seeds": len((member["config"].get("training") or {}).get("seeds", [])),
                         "controller_fits_per_profile": {p: 1 for p in profiles}, "per_seed_controller_packages": per_seed,
                         "selection_plant_refit": False})
    if not rows:
        return {"status": "pending", "detail": "no completed run", "rows": []}
    single = all(not r["per_seed_controller_packages"] for r in rows)
    benchmark = [r for r in rows if r["benchmark"]]
    status = "structure only" if not benchmark else ("needs correction" if single else "needs attention")
    return {"status": status, "rows": rows,
            "detail": ("One controller fit per profile and no plant refit inside selection folds: Table S2's controller- "
                       "and plant-seed columns are corrected to this hierarchy, or the runtime adds the printed loops "
                       "before reproduction (ledger K4).") if single else
                      "Per-seed controller packages exist: compare them with the twelve printed controller seeds."}


def entry_gates(runs: dict, printed: dict) -> dict:
    """K6: did any target-regime gate fail?"""
    rows = []
    for dataset, members in runs.items():
        for member in members:
            entry = _json(member["path"] / "select" / "entry_complete.json")
            if entry is None:
                continue
            initial = entry.get("initial") or {}
            rows.append({"dataset": dataset, "run": member["path"].name, "benchmark": member["benchmark"],
                         "initial_entry_pass": initial.get("initial_entry_pass"),
                         "registry_support_fraction": entry.get("registry_support_fraction"),
                         "selected_strata_support": (entry.get("selected_strata_support") or {}).get("passed"),
                         "passed": entry.get("passed"), "enforced": entry.get("enforced")})
    if not rows:
        return {"status": "pending", "detail": "no completed run with select/entry_complete.json", "rows": []}
    benchmark = [r for r in rows if r["benchmark"]]
    status = ("structure only" if not benchmark else
              "consistent" if all(r["passed"] for r in benchmark) else "needs correction")
    return {"status": status, "rows": rows,
            "detail": "Every gate passed: the failure responses never executed and no reported value depends on them "
                      "(ledger K6)."}


CHECKS = {"selection_family": selection_family, "peptides_confirmation": peptides_confirmation,
          "complete_ledger_selection": complete_ledger_selection, "seed_hierarchy": seed_hierarchy,
          "entry_gates": entry_gates}


def audit(runs_root: Path, root: Path = ROOT) -> dict:
    ledger = load_ledger(root)
    runs = discover(runs_root)
    reported = json.loads((root / "paper/reported_results.json").read_text(encoding="utf-8"))
    s20 = reported.get("supplementary_table_S20_selection_theorem_instantiation", {})
    printed = {d: {k: (v.get("$value") if isinstance(v, dict) and "$value" in v else v) for k, v in row.items()}
               for d, row in s20.items() if isinstance(row, dict)}
    results = []
    for item in ledger["items"]:
        if item["check"] is None:
            results.append({"id": item["id"], "title": item["title"], "status": item["status"], "rows": [],
                            "detail": item["decided_by"]})
            continue
        try:
            outcome = CHECKS[item["check"]](runs, printed)
        except (OSError, KeyError, ValueError, TypeError) as exc:
            outcome = {"status": "needs attention", "rows": [], "detail": f"{type(exc).__name__}: {exc}"}
        results.append({"id": item["id"], "title": item["title"], **outcome})
    return {"schema": "cross-morse-consistency-audit-1", "runs_root": str(runs_root),
            "runs": {d: [m["path"].name for m in members] for d, members in runs.items()},
            "benchmark_runs": sum(1 for members in runs.values() for m in members if m["benchmark"]),
            "items": results}


# ================================================================================= rendering
def cell(text) -> str:
    return str(text).replace("|", r"\|").replace("\n", " ")


def render(ledger: dict) -> str:
    L = ["# Scientific consistency", "",
         "[Statement compliance](../STATEMENT_COMPLIANCE.md) · "
         "[Figure coverage](FIGURE_REFERENCE_COVERAGE.md)", "",
         ledger["about"], "",
         "| Item | Question | Status | Decided by |", "|---|---|---|---|"]
    for item in ledger["items"]:
        L.append(f"| {item['id']} | {cell(item['title'])} | {STATUS_WORDS[item['status']]} | {cell(item['decided_by'])} |")
    L += ["", "On the cluster, after the seven lifecycles have run:", "", "```bash",
          "make consistency-audit RUNS_ROOT=<suite>/runs     # K1, K2, K3, K4 and K6 on the original runs", "```", "",
          "Each check reports `consistent`, `needs correction` (the rule names what changes), `pending` (no such run) or "
          "`structure only` (smoke and synthetic runs, which exercise the check but decide nothing about printed values).", ""]
    for item in ledger["items"]:
        L += [f"## {item['id']}. {item['title']}", "",
              f"**Displays:** {', '.join(item['displays'])}. **Status:** {STATUS_WORDS[item['status']]}.", ""]
        for source in item["printed"]:
            pages = ", ".join(str(p) for p in source["pages"])
            L.append(f"- **Printed** ({source['document']}, p. {pages}): {source['text']}")
        L += [f"- **Code before:** {item['code_before']}", f"- **Code now:** {item['code_now']}",
              f"- **Decided by:** {item['decided_by']}", f"- **Rule:** {item['rule']}"]
        if item["check"]:
            L.append(f"- **Check:** `{item['check']}` in `scripts/consistency_audit.py`")
        L.append("")
    L += ["Generated from `paper/consistency_ledger.json` by `scripts/consistency_audit.py --write`; "
          "`make consistency-check` verifies it is current.", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help=f"write {DOC}")
    parser.add_argument("--check", action="store_true", help=f"{DOC} is current")
    parser.add_argument("--runs-root", type=Path, help="completed runs to audit")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path, help="write the audit as JSON")
    args = parser.parse_args(argv)
    try:
        ledger = load_ledger(ROOT)
    except (OSError, ValueError, KeyError) as exc:
        print(f"consistency ledger: {exc}", file=sys.stderr)
        return 2
    text = render(ledger) + "\n"
    if args.write:
        (ROOT / DOC).write_text(text, encoding="utf-8")
    if args.check:
        if not (ROOT / DOC).is_file() or (ROOT / DOC).read_text(encoding="utf-8") != text:
            print(f"{DOC} is not current; run: python -B scripts/consistency_audit.py --write", file=sys.stderr)
            return 1
    if args.runs_root:
        if not args.runs_root.is_dir():
            print(f"{args.runs_root} is not a directory of completed runs", file=sys.stderr)
            return 2
        report = audit(args.runs_root.resolve())
        if args.output:
            args.output.write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
        if args.json:
            print(json.dumps(report, indent=1, default=str))
        else:
            print(f"{report['benchmark_runs']} full benchmark runs under {args.runs_root}")
            for item in report["items"]:
                print(f"{item['id']:<3} {item['status']:<18} {item['title']}")
                for row in item.get("rows", [])[:12]:
                    shown = {k: v for k, v in row.items() if k not in ("downgrades",)}
                    print(f"      {json.dumps(shown, default=str)[:200]}")
        return 1 if any(i["status"] in ("needs correction", "needs attention") for i in report["items"]) else 0
    if not (args.write or args.check):
        for item in ledger["items"]:
            print(f"{item['id']:<3} {STATUS_WORDS[item['status']]:<52} {item['title']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
