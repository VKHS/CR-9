#!/usr/bin/env python3
"""Check the repository against the paper's registered data-availability statement.

The statement (release/availability_statement.json) promises twelve things. Each is
evaluated from what the repository holds and what its release has published:

    fulfilled                  the repository and its published outputs provide it
    awaiting_measured_outputs  the code and structure are complete; the measured
                               outputs of the reported experiments are not yet
                               published (hpc/produce_paper_outputs_arrhenius.sh,
                               then hpc/publish_paper_artifacts_arrhenius.sh all)
    missing                    a repository-side gap (code, generator, configuration)

    python -B scripts/statement_check.py            # print the evaluation
    python -B scripts/statement_check.py --write    # also write STATEMENT_COMPLIANCE.md
    python -B scripts/statement_check.py --check    # STATEMENT_COMPLIANCE.md matches the repository
    python -B scripts/statement_check.py --require-complete    # nothing missing on the repository side
    python -B scripts/statement_check.py --require-fulfilled   # every item fulfilled

Nothing here downloads or computes results; it inspects files and manifests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STATEMENT = "release/availability_statement.json"
EXPERIMENTS = "release/experiment_code.json"
REPORT = "STATEMENT_COMPLIANCE.md"
FULFILLED, AWAITING, MISSING = "fulfilled", "awaiting_measured_outputs", "missing"
DATASETS = ("ogbn-arxiv", "ogbn-proteins", "ogbn-products", "Roman-empire", "PascalVOC-SP", "COCO-SP", "Peptides-func")
STEMS = {"ogbn-arxiv": "arxiv", "ogbn-products": "products", "ogbn-proteins": "proteins",
         "Roman-empire": "roman_empire", "PascalVOC-SP": "pascalvoc_sp", "COCO-SP": "coco_sp",
         "Peptides-func": "peptides_func"}
LABEL = {FULFILLED: "Fulfilled", AWAITING: "Awaiting measured outputs", MISSING: "Missing"}


def _json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(root: Path = ROOT) -> dict:
    from scripts import paper_artifacts
    from scripts import statement_evidence as evidence
    from paper.build.displays import framework
    framework._load_generators()
    statement = _json(root / STATEMENT)
    manifest = paper_artifacts.load_manifest(root / "release/paper_artifacts.json")
    artifacts = {a["id"]: a for a in manifest["artifacts"]}
    # A missing row or a changed `required` flag cannot shrink the paper's contract.
    required = paper_artifacts.required_artifacts(root)
    fixture_only = {i for i, a in artifacts.items() if a.get("status") == paper_artifacts.PUBLISHED
                    and a.get("scientific_only_verification") is False}
    published = {i for i, a in artifacts.items() if a.get("status") == paper_artifacts.PUBLISHED
                 and i not in fixture_only}
    awaiting = sorted(a["id"] for a in required if a["id"] not in published)
    runs = [f"run:{d}" for d in DATASETS]
    displays = _json(root / "paper/display_manifest.json")["displays"]
    display_ids = [d["id"] for d in displays]
    results = {}

    def item(key, status, where, remaining=""):
        results[key] = {"status": status, "where": where, "remaining": remaining}

    # Complete experiment code
    experiments = _json(root / EXPERIMENTS, {"experiments": []})["experiments"]

    def published_with_outputs(item_id):
        """Code or configurations that travel inside a measured artifact and are copied into the checkout when
        it is published (E12: the study plans and adapter code of the external comparisons, inside
        records:baselines, copied to experiments/baselines/ by scripts/paper_artifacts.py derive)."""
        pending, changed = [], []
        for e in experiments:
            declared = e.get("published_with_outputs")
            if not declared or item_id not in declared.get("items", []):
                continue
            view_path = root / declared["view"]
            view = _json(view_path)
            parts = declared["parts"][item_id] if isinstance(declared["parts"], dict) else declared["parts"]
            if declared["artifact"] not in published or view is None:
                pending.append(f"{e['experiment']}: {parts} (published inside {declared['artifact']} "
                               f"and copied to {Path(declared['view']).parent.as_posix()}/)")
                continue
            try:
                evidence.check_baseline_studies(view_path)
                evidence.check_snapshot(root, view_path.parent, manifest)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                changed.append(f"{declared['view']}: {exc}")
        return pending, changed

    absent = sorted({p for e in experiments for p in e.get("code", []) if not (root / p).exists()})
    produced = {o for e in experiments for o in e.get("outputs", [])}
    needed = {a["id"] for a in required if a["kind"] in ("records", "campaign", "run")}
    unproduced = sorted(i for i in needed if i not in produced and not (i.startswith("run:") and "run:*" in produced))
    # Parts of a reported experiment that the released code cannot execute, declared with the reason.
    gaps = [f"{e['experiment']}: {gap['display']} ({gap['reason']})"
            for e in experiments for gap in e.get("not_executable", [])]
    pending, changed = published_with_outputs("experiment_code")
    where = f"{len(experiments)} experiments mapped to their code in `release/experiment_code.json`"
    if absent or unproduced or gaps or changed or not experiments:
        item("experiment_code", MISSING, "release/experiment_code.json",
             "; ".join(filter(None, [f"code not found: {', '.join(absent)}" if absent else "",
                                     f"no producing experiment for {', '.join(unproduced)}" if unproduced else "",
                                     "not executable with the released code: " + "; ".join(gaps) if gaps else "",
                                     "; ".join(changed),
                                     "no experiment-code map" if not experiments else ""])))
    elif pending:
        item("experiment_code", AWAITING, where, "; ".join(pending))
    else:
        item("experiment_code", FULFILLED, where)

    # Configurations
    configs = [f"configs/{STEMS[d]}_{p}.json" for d in DATASETS for p in ("smoke", "pilot", "full")]
    configs += sorted({c for e in experiments for c in e.get("configurations", [])})
    absent = [c for c in configs if not (root / c).exists()]
    pending, changed = published_with_outputs("configurations")
    if absent or changed:
        item("configurations", MISSING, "`configs/` and the experiment configurations", ", ".join(absent + changed))
    elif pending:
        item("configurations", AWAITING, "`configs/` and the experiment configurations", "; ".join(pending))
    else:
        item("configurations", FULFILLED, "`configs/` and the experiment configurations")

    # Generators
    without = [i for i in display_ids if i not in framework.GENERATORS]
    broken = dict(framework.LOAD_ERRORS)
    duplicate_displays = len(display_ids) != len(set(display_ids))
    if without or broken or not display_ids or duplicate_displays:
        item("generators", MISSING, "`paper/build/displays/`",
             "; ".join(filter(None, [f"{len(without)} displays without a generator: {', '.join(without)}" if without else "",
                                     f"modules that fail to import: {broken}" if broken else "",
                                     "empty or duplicate display inventory" if not display_ids or duplicate_displays else ""])))
    else:
        item("generators", FULFILLED, f"`paper/build/displays/`: a generator for each of the {len(display_ids)} "
                                      "numbered displays")

    # Outputs carried by the published runs
    def run_contents(category):
        missing = [r for r in runs if r not in published]
        empty = [r for r in runs if r in published and not (artifacts[r].get("contents") or {}).get(category)]
        return missing, empty

    for key, category, words in (("checkpoints", "checkpoints", "checkpoints"),
                                 ("predictions", "predictions", "prediction arrays"),
                                 ("trajectories", "propensity_logged_trajectories", "exploration trajectories")):
        missing, empty = run_contents(category)
        if fixture_only & set(runs):
            item(key, MISSING, "release `paper-artifacts`", "software fixtures cannot fulfil benchmark availability: "
                 + ", ".join(sorted(fixture_only & set(runs))))
        elif empty:
            item(key, MISSING, "release `paper-artifacts`", f"published runs without {words}: {', '.join(empty)}")
        elif missing:
            item(key, AWAITING, "release `paper-artifacts` (`release/paper_artifacts.json`)",
                 f"{words} of {len(missing)} reported runs")
        else:
            item(key, FULFILLED, "release `paper-artifacts`; complete tier of every run")

    stats_missing = [a for a in awaiting if not a.startswith("environment:")]
    item("statistical_outputs", MISSING if fixture_only else (AWAITING if stats_missing else FULFILLED),
         "release `paper-artifacts`: runs, measurement-record sets and campaigns",
         ("software fixtures cannot fulfil benchmark availability: " + ", ".join(sorted(fixture_only)))
         if fixture_only else ("awaiting publication: " + ", ".join(stats_missing) if stats_missing else ""))

    # Split and seed manifests
    registered = [f"manifests/registered/{STEMS[d]}_full.json" for d in DATASETS]
    absent = [p for p in registered if not (root / p).is_file()]
    published_manifests = [f"manifests/published/{STEMS[d]}.json" for d in DATASETS]
    pending = [p for dataset, p in zip(DATASETS, published_manifests)
               if not (root / p).is_file() or f"run:{dataset}" not in published]
    stale = []
    for dataset, path in zip(DATASETS, published_manifests):
        entry = artifacts.get(f"run:{dataset}")
        if (root / path).is_file() and f"run:{dataset}" in published:
            try:
                evidence.check_run_manifest(root / "manifests/published", STEMS[dataset], dataset, entry)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                stale.append(f"{path}: {exc}")
    if not pending:
        try:
            evidence.check_snapshot(root, root / "manifests/published", manifest)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            stale.append(str(exc))
    if absent or stale:
        item("split_seed_manifests", MISSING, "`manifests/`", "; ".join(absent + stale))
    elif pending:
        item("split_seed_manifests", AWAITING, "`manifests/registered/` (design); `manifests/published/` (executed)",
             f"executed memberships of {len(pending)} runs")
    else:
        item("split_seed_manifests", FULFILLED, "`manifests/registered/` and `manifests/published/`")

    # Environment specification
    specs = ["environment/requirements-cpu.lock", "environment/apptainer.def", "Dockerfile", "run_cross_morse.sh"]
    absent = [p for p in specs if not (root / p).is_file()]
    observed = [f"environment/observed/{STEMS[d]}.json" for d in DATASETS]
    pending = [p for dataset, p in zip(DATASETS, observed)
               if not (root / p).is_file() or f"run:{dataset}" not in published]
    stale = []
    for dataset, path in zip(DATASETS, observed):
        if (root / path).is_file() and f"run:{dataset}" in published:
            try:
                evidence.check_observed_environment(root / "environment/observed", STEMS[dataset], dataset,
                                                    artifacts[f"run:{dataset}"])
            except (OSError, ValueError, KeyError, TypeError) as exc:
                stale.append(f"{path}: {exc}")
    if "environment:gpu" not in published:
        pending.append("environment:gpu")
    if not pending:
        try:
            evidence.check_snapshot(root, root / "environment/observed", manifest)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            stale.append(str(exc))
    if absent or stale:
        item("environment", MISSING, "`environment/`", "; ".join(absent + stale))
    elif pending:
        item("environment", AWAITING, "`environment/`, `Dockerfile`, SITE SETTINGS of `run_cross_morse.sh`",
             "observed environment of the reported runs: " + ", ".join(pending))
    else:
        item("environment", FULFILLED, "`environment/` specifications and `environment/observed/`")

    # Source data and one-command reconstruction
    index_path = root / "paper/source_data/INDEX.json"
    index, summary, source_errors = None, {}, []
    if not index_path.is_file():
        item("source_data", AWAITING if awaiting else MISSING, "`paper/source_data/` (written by the publication step)",
             "generated from the published outputs")
    else:
        try:
            index, summary = evidence.check_source_data(root, manifest, display_ids)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            source_errors.append(str(exc))
        item("source_data", MISSING if source_errors else (AWAITING if awaiting else FULFILLED),
             "`paper/source_data/`", "; ".join(source_errors) or (", ".join(awaiting) if awaiting else ""))
    paper_target = (root / "Makefile").is_file() and "scripts/reconstruct_paper.py" in (root / "Makefile").read_text(encoding="utf-8")
    if not paper_target or not (root / "scripts/reconstruct_paper.py").is_file():
        item("one_command", MISSING, "`make paper`", "the make target or scripts/reconstruct_paper.py is absent")
    elif source_errors:
        item("one_command", MISSING, "`make paper`", "; ".join(source_errors))
    elif awaiting:
        item("one_command", AWAITING, "`make paper` (scripts/reconstruct_paper.py)",
             f"{len(awaiting)} artifacts unpublished; displays needing them cannot be rebuilt yet")
    elif evidence.reconstruction_problems(summary, len(display_ids)):
        item("one_command", MISSING, "`make paper`",
             "; ".join(evidence.reconstruction_problems(summary, len(display_ids))))
    else:
        item("one_command", FULFILLED, "`make paper`: every display rebuilt and every compared value agrees")

    text = (root / "REPRODUCIBILITY.md").read_text(encoding="utf-8") if (root / "REPRODUCIBILITY.md").is_file() else ""
    item("guidance", FULFILLED if "Pineau" in text else MISSING, "`REPRODUCIBILITY.md`",
         "" if "Pineau" in text else "no mapping of the reproducibility checklist")

    order = [i["id"] for i in statement["items"]]
    rows = [{"id": i["id"], "phrase": i["phrase"], "requirement": i["requirement"], **results[i["id"]]}
            for i in statement["items"]]
    counts = {s: sum(1 for r in rows if r["status"] == s) for s in (FULFILLED, AWAITING, MISSING)}
    return {"schema": "cross-morse-statement-compliance-1", "statement": statement["registered_text"],
            "items": rows, "counts": counts, "order": order, "awaiting_artifacts": awaiting,
            "artifacts_required": len(required), "artifacts_published": len(published & {a["id"] for a in required})}


def render(evaluation: dict, printed: dict | None = None) -> str:
    """STATEMENT_COMPLIANCE.md: the registered statement, then (when given) the printed manuscript text."""
    counts = evaluation["counts"]
    if counts[MISSING]:
        scope = "The checks below identify the files or validation results that need attention."
    elif counts[AWAITING]:
        scope = ("Experiment and reconstruction tooling is included. The measured artifacts identified below "
                 "await publication from the completed runs.")
    else:
        scope = "The published files and reconstruction records satisfy each item checked below."
    lines = ["# Compliance with the data-availability statement", "",
             "The paper's registered data-availability statement:", "",
             f"> {evaluation['statement']}", "",
             scope, "", "This offline check validates the release manifest and its committed views. "
             "`make statement-fulfilled` also downloads and verifies the complete artifacts and executes the "
             "reconstruction before evaluating this statement.", "",
             "| Statement item | Status | Where | Remaining |", "|---|---|---|---|"]
    for row in evaluation["items"]:
        lines.append(f"| {row['phrase']} | {LABEL[row['status']]} | {row['where']} | {row['remaining'] or '—'} |")
    lines += ["", "Items awaiting measured outputs have their code, structure and checks in place. On the cluster, "
              "after `bash run_cross_morse.sh reproduce` has completed the seven lifecycle runs:", "",
              "```bash",
              "bash hpc/produce_paper_outputs_arrhenius.sh <suite>      # every other measured output, one Slurm job graph",
              "bash hpc/publish_paper_artifacts_arrhenius.sh all <work>/paper-artifacts.map.json   # release paper-artifacts",
              "make statement-fulfilled                                 # every item fulfilled",
              "```", "",
              "The first command is described in [docs/PRODUCTION.md](docs/PRODUCTION.md); its last job writes the map "
              "of all artifacts. The second uploads them to the release `paper-artifacts`, downloads and verifies "
              "them, rebuilds every table and figure with `make paper` and writes the manifests, observed "
              "environments and source data into the checkout (`CM_PUBLISH_COMMIT=1` commits and pushes them). "
              "The reconstruction item is fulfilled when every display is rebuilt and every compared value agrees "
              "with the printed one; `paper/source_data/compare/` lists each difference.", "",
              "[The final destination](docs/FINAL_DESTINATION.md) projects the repository after publication: the "
              "contents and sizes of every artifact, the pre-registered data roles, the design-determined values "
              "of the reconstruction and the complete close-out sequence. [`destination/`](destination/README.md) "
              "holds every file that publication writes into the checkout, projected, with the twelve items "
              "evaluated on them ([at the destination](destination/STATEMENT_COMPLIANCE.md)); "
              "`make final-destination-converge` compares each published file with its projection.", ""]
    if printed is not None:
        from scripts import printed_statement_check
        lines += printed_statement_check.render_section(printed, None)
    lines += ["Generated by `scripts/statement_check.py --write`; `make statement-check` verifies it is current.", ""]
    return "\n".join(lines)


def evaluate_printed(evaluation: dict, root: Path = ROOT) -> dict:
    from scripts import printed_statement_check
    return printed_statement_check.evaluate(root, registered=evaluation)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--require-fulfilled", action="store_true")
    parser.add_argument("--require-printed-fulfilled", action="store_true",
                        help="also require every claim of the printed manuscript text (release/printed_availability.json)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        evaluation = evaluate(ROOT)
        printed = evaluate_printed(evaluation, ROOT)
    except (OSError, ValueError, KeyError, TypeError, SyntaxError) as exc:
        print(f"Invalid availability evidence: {exc}", file=sys.stderr)
        return 2
    text = render(evaluation, printed)
    if args.json:
        print(json.dumps({**evaluation, "printed": printed}, indent=1, ensure_ascii=False))
    else:
        for row in evaluation["items"]:
            print(f"{LABEL[row['status']]:<27} {row['phrase']}" + (f" — {row['remaining']}" if row["remaining"] else ""))
        c = printed["counts"]
        print(f"Printed manuscript text: {c[FULFILLED]} fulfilled, {c[AWAITING]} awaiting measured outputs, "
              f"{c[MISSING]} missing of {len(printed['claims'])} claims (scripts/printed_statement_check.py)")
    if args.write:
        (ROOT / REPORT).write_text(text, encoding="utf-8")
    if args.check:
        current = (ROOT / REPORT).read_text(encoding="utf-8") if (ROOT / REPORT).is_file() else ""
        if current != text:
            print(f"{REPORT} is not current; run: python -B scripts/statement_check.py --write", file=sys.stderr)
            return 1
    if (args.require_complete or args.require_fulfilled) and evaluation["counts"][MISSING]:
        return 1
    if args.require_fulfilled and evaluation["counts"][AWAITING]:
        return 1
    if args.require_printed_fulfilled and (printed["counts"][MISSING] or printed["counts"][AWAITING]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
