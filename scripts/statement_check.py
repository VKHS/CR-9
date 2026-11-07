#!/usr/bin/env python3
"""Check the repository against the paper's registered data-availability statement.

The statement (release/availability_statement.json) promises twelve things. Each is
evaluated from what the repository holds and what its release has published:

    fulfilled                  the repository and its published outputs provide it
    awaiting_measured_outputs  the code and structure are complete; the measured
                               outputs of the reported experiments are not yet
                               published (hpc/publish_paper_artifacts_arrhenius.sh)
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
    from paper.build.displays import framework
    framework._load_generators()
    statement = _json(root / STATEMENT)
    manifest = paper_artifacts.load_manifest(root / "release/paper_artifacts.json")
    artifacts = {a["id"]: a for a in manifest["artifacts"]}
    required = [a for a in manifest["artifacts"] if a.get("required", True)]
    published = {i for i, a in artifacts.items() if a.get("status") == paper_artifacts.PUBLISHED}
    awaiting = sorted(a["id"] for a in required if a["id"] not in published)
    runs = [f"run:{d}" for d in DATASETS]
    displays = _json(root / "paper/display_manifest.json")["displays"]
    display_ids = [d["id"] for d in displays]
    results = {}

    def item(key, status, where, remaining=""):
        results[key] = {"status": status, "where": where, "remaining": remaining}

    # Complete experiment code
    experiments = _json(root / EXPERIMENTS, {"experiments": []})["experiments"]
    absent = sorted({p for e in experiments for p in e.get("code", []) if not (root / p).exists()})
    produced = {o for e in experiments for o in e.get("outputs", [])}
    needed = {a["id"] for a in required if a["kind"] in ("records", "campaign", "run")}
    unproduced = sorted(i for i in needed if i not in produced and not (i.startswith("run:") and "run:*" in produced))
    # Parts of a reported experiment that the released code cannot execute, declared with the reason.
    gaps = [f"{e['experiment']}: {gap['display']} ({gap['reason']})"
            for e in experiments for gap in e.get("not_executable", [])]
    if absent or unproduced or gaps or not experiments:
        item("experiment_code", MISSING, "release/experiment_code.json",
             "; ".join(filter(None, [f"code not found: {', '.join(absent)}" if absent else "",
                                     f"no producing experiment for {', '.join(unproduced)}" if unproduced else "",
                                     "not executable with the released code: " + "; ".join(gaps) if gaps else "",
                                     "no experiment-code map" if not experiments else ""])))
    else:
        item("experiment_code", FULFILLED, f"{len(experiments)} experiments mapped to their code in "
                                           "`release/experiment_code.json`")

    # Configurations
    configs = [f"configs/{STEMS[d]}_{p}.json" for d in DATASETS for p in ("smoke", "pilot", "full")]
    configs += sorted({c for e in experiments for c in e.get("configurations", [])})
    absent = [c for c in configs if not (root / c).exists()]
    item("configurations", MISSING if absent else FULFILLED, "`configs/` and the experiment configurations",
         ", ".join(absent))

    # Generators
    without = [i for i in display_ids if i not in framework.GENERATORS]
    broken = dict(framework.LOAD_ERRORS)
    if without or broken:
        item("generators", MISSING, "`paper/build/displays/`",
             "; ".join(filter(None, [f"{len(without)} displays without a generator: {', '.join(without)}" if without else "",
                                     f"modules that fail to import: {broken}" if broken else ""])))
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
        if empty:
            item(key, MISSING, "release `paper-artifacts`", f"published runs without {words}: {', '.join(empty)}")
        elif missing:
            item(key, AWAITING, "release `paper-artifacts` (`release/paper_artifacts.json`)",
                 f"{words} of {len(missing)} reported runs")
        else:
            item(key, FULFILLED, "release `paper-artifacts`; complete tier of every run")

    stats_missing = [a for a in awaiting if not a.startswith("environment:")]
    item("statistical_outputs", AWAITING if stats_missing else FULFILLED,
         "release `paper-artifacts`: runs, measurement-record sets and campaigns",
         f"{len(stats_missing)} of {len([a for a in required if a['kind'] != 'environment'])} artifacts: "
         + ", ".join(stats_missing) if stats_missing else "")

    # Split and seed manifests
    registered = [f"manifests/registered/{STEMS[d]}_full.json" for d in DATASETS]
    absent = [p for p in registered if not (root / p).is_file()]
    published_manifests = [f"manifests/published/{STEMS[d]}.json" for d in DATASETS]
    pending = [p for p in published_manifests if not (root / p).is_file()]
    stale = []
    for dataset, path in zip(DATASETS, published_manifests):
        entry = artifacts.get(f"run:{dataset}")
        if (root / path).is_file() and entry and entry.get("status") == paper_artifacts.PUBLISHED:
            value = _json(root / path)
            units = ([s["unit_sha256"] for s in value["campaign"]["per_split"]] if "campaign" in value
                     else [value["identities"]["unit_sha256"]])
            identity = entry.get("identity") or {}
            expected = ([s["unit_sha256"] for _, s in sorted((identity.get("splits") or {}).items())]
                        if identity.get("campaign") else [identity.get("unit_sha256")])
            if units != expected:
                stale.append(path)
    if absent or stale:
        item("split_seed_manifests", MISSING, "`manifests/`", ", ".join(absent + [f"{p} differs from the published run"
                                                                                  for p in stale]))
    elif pending:
        item("split_seed_manifests", AWAITING, "`manifests/registered/` (design); `manifests/published/` (executed)",
             f"executed memberships of {len(pending)} runs")
    else:
        item("split_seed_manifests", FULFILLED, "`manifests/registered/` and `manifests/published/`")

    # Environment specification
    specs = ["environment/requirements-cpu.lock", "environment/apptainer.def", "Dockerfile", "run_cross_morse.sh"]
    absent = [p for p in specs if not (root / p).is_file()]
    observed = [f"environment/observed/{STEMS[d]}.json" for d in DATASETS]
    pending = [p for p in observed if not (root / p).is_file()]
    if "environment:gpu" not in published:
        pending.append("environment:gpu")
    if absent:
        item("environment", MISSING, "`environment/`", ", ".join(absent))
    elif pending:
        item("environment", AWAITING, "`environment/`, `Dockerfile`, SITE SETTINGS of `run_cross_morse.sh`",
             "observed environment of the reported runs: " + ", ".join(pending))
    else:
        item("environment", FULFILLED, "`environment/` specifications and `environment/observed/`")

    # Source data and one-command reconstruction
    index = _json(root / "paper/source_data/INDEX.json")
    if index is None:
        item("source_data", AWAITING if awaiting else MISSING, "`paper/source_data/` (written by the publication step)",
             "generated from the published outputs")
    else:
        changed = [name for name, digest in index["files"].items()
                   if not (root / "paper/source_data" / name).is_file() or _sha(root / "paper/source_data" / name) != digest]
        item("source_data", MISSING if changed else (AWAITING if awaiting else FULFILLED),
             "`paper/source_data/`", ", ".join(changed) or (", ".join(awaiting) if awaiting else ""))
    paper_target = (root / "Makefile").is_file() and "scripts/reconstruct_paper.py" in (root / "Makefile").read_text(encoding="utf-8")
    summary = (index or {}).get("reconstruction") or {}
    if not paper_target or not (root / "scripts/reconstruct_paper.py").is_file():
        item("one_command", MISSING, "`make paper`", "the make target or scripts/reconstruct_paper.py is absent")
    elif awaiting:
        item("one_command", AWAITING, "`make paper` (scripts/reconstruct_paper.py)",
             f"{len(awaiting)} artifacts unpublished; displays needing them cannot be rebuilt yet")
    elif summary.get("complete") != len(display_ids) or summary.get("values_differ"):
        item("one_command", MISSING, "`make paper`",
             f"last published reconstruction: {summary.get('complete')} of {len(display_ids)} displays complete, "
             f"{summary.get('values_differ')} values differ from the paper")
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


def render(evaluation: dict) -> str:
    counts = evaluation["counts"]
    lines = ["# Compliance with the data-availability statement", "",
             "The paper's registered data-availability statement:", "",
             f"> {evaluation['statement']}", "",
             f"**{counts[FULFILLED]} of {len(evaluation['items'])} items fulfilled**; "
             f"{counts[AWAITING]} awaiting the publication of the measured outputs; {counts[MISSING]} missing. "
             f"Published artifacts: {evaluation['artifacts_published']} of {evaluation['artifacts_required']} "
             "(`release/paper_artifacts.json`).", "",
             "| Statement item | Status | Where | Remaining |", "|---|---|---|---|"]
    for row in evaluation["items"]:
        lines.append(f"| {row['phrase']} | {LABEL[row['status']]} | {row['where']} | {row['remaining'] or '—'} |")
    lines += ["", "Items awaiting measured outputs have their code, structure and checks in place; they are "
              "fulfilled when `hpc/publish_paper_artifacts_arrhenius.sh` uploads the outputs of the reported runs "
              "to the release `paper-artifacts` and the updated manifest is committed. `make paper` then rebuilds "
              "every table and figure from those outputs and compares each computed value with the printed one.", "",
              "Generated by `scripts/statement_check.py --write`; `make statement-check` verifies it is current.", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--require-fulfilled", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    evaluation = evaluate(ROOT)
    text = render(evaluation)
    if args.json:
        print(json.dumps(evaluation, indent=1))
    else:
        for row in evaluation["items"]:
            print(f"{LABEL[row['status']]:<27} {row['phrase']}" + (f" — {row['remaining']}" if row["remaining"] else ""))
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
