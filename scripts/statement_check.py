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
             f"{len(awaiting)} artifacts awaiting publication; `make paper` rebuilds the displays that read them from the release")
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


DEMO_ITEMS = {  # statement item -> what the CPU execution supplies for it
    "experiment_code": "the lifecycle and every runner executed from this checkout (the E12 adapters excepted)",
    "configurations": "the effective configuration sealed in every CPU run",
    "source_data": "the source data written by the CPU execution's reconstruction",
    "split_seed_manifests": "the executed split and seed manifests of the CPU runs",
    "environment": "the observed CPU environment of the runs",
    "checkpoints": "the plant and controller checkpoints of the CPU runs",
    "predictions": "the member and deployed predictions of the CPU runs",
    "trajectories": "the propensity-logged exploration trajectories of the CPU runs",
    "statistical_outputs": "the selection, confirmation and admission statistics, record sets and campaigns",
    "generators": "every table and figure generator run on the CPU execution's artifacts",
    "one_command": "make demo-paper",
    "guidance": "the same checklist mapping",
}
DEMO_KINDS = {"source_data": None, "generators": None, "one_command": None, "guidance": None,
              "experiment_code": ("run",), "configurations": ("run",), "split_seed_manifests": ("run",),
              "environment": ("run", "environment"), "checkpoints": ("run",), "predictions": ("run", "campaign"),
              "trajectories": ("run",), "statistical_outputs": ("run", "records", "campaign")}


def demonstration(evaluation: dict, root: Path = ROOT) -> dict | None:
    """What the separately published CPU execution (release/demo_artifacts.json) shows for each item.

    It is reported beside the statement items; their status follows the paper's own artifacts."""
    manifest = _json(root / "release" / "demo_artifacts.json")
    if not manifest or manifest.get("channel") != "cpu_execution":
        return None
    published = [a for a in manifest.get("artifacts", []) if a.get("status") == "published"]
    views = root / "demo" / "published" / "paper" / "source_data"
    source = _json(views / "INDEX.json") or {}
    # the display generators that built a display from the CPU execution (its reconstruction report)
    generators = (_json(views / "reports" / "reconstruction.json") or {}).get("display_generators") or {}
    built, registered = generators.get("built"), generators.get("registered")
    counted = type(built) is int and type(registered) is int
    rows = []
    for item in evaluation["items"]:
        kinds = DEMO_KINDS[item["id"]]
        if kinds is None:
            shown = bool(published) and (item["id"] == "guidance" or bool(source))
            evidence = (f"{built} of {registered} displays built" if item["id"] in ("one_command", "generators", "source_data")
                        and counted else ("demo/published/paper/source_data" if source else ""))
            if item["id"] == "one_command" and shown and counted and built < registered:
                shown = "partly"     # the one command runs; the artifacts it lacks are listed in demo/RESULTS.md
        else:
            ids = [a["id"] for a in published if a.get("kind") in kinds]
            shown, evidence = bool(ids), f"{len(ids)} artifact(s)" if ids else ""
        rows.append({"id": item["id"], "phrase": item["phrase"], "demonstrated": shown,
                     "what": DEMO_ITEMS[item["id"]], "evidence": evidence})
    return {"release": manifest.get("release_url"), "scope": manifest.get("scope"), "published": len(published),
            "artifacts": len(manifest.get("artifacts", [])), "rows": rows}


PUBLISH = "[the publication step](docs/PUBLICATION.md)"
PROVIDED = {  # statement item -> where the repository provides it
    "experiment_code": "`cross_morse/`, `cm_experiments/`, `cm_campaigns/` and `scripts/`; "
                       "[release/experiment_code.json](release/experiment_code.json) maps every experiment to its code",
    "configurations": "[configs/](configs/) and the experiment configurations",
    "source_data": f"the source data of every display, which {PUBLISH} writes to `paper/source_data/`; dataset "
                   "sources in [data/sources.json](data/sources.json)",
    "split_seed_manifests": f"[manifests/registered/](manifests/registered/); {PUBLISH} writes the executed role "
                            "memberships and seeds to `manifests/published/`",
    "environment": f"[environment/](environment/) and [Dockerfile](Dockerfile); {PUBLISH} adds the observed "
                   "environment of each run (`environment/observed/`)",
    "checkpoints": f"the run archives that {PUBLISH} uploads to the release `paper-artifacts`, recorded in "
                   "[release/paper_artifacts.json](release/paper_artifacts.json)",
    "predictions": "the same run archives",
    "trajectories": "the same run archives",
    "statistical_outputs": f"the measurement-record sets and campaigns that {PUBLISH} uploads to the same release, with "
                           "the run and record indexes [runs/index.json](runs/index.json) and "
                           "[records/index.json](records/index.json)",
    "generators": "[paper/build/displays/](paper/build/displays/), one generator for each of the 90 numbered displays",
    "one_command": "`make paper` ([Reconstruction](docs/RECONSTRUCTION.md))",
    "guidance": "[REPRODUCIBILITY.md](REPRODUCIBILITY.md)",
}


def render(evaluation: dict, printed: dict | None = None, demo: dict | None = None) -> str:
    """STATEMENT_COMPLIANCE.md: each item of the statement and where the repository provides it.

    ``printed`` and ``demo`` are evaluated by ``make printed-statement-check`` and ``--json``; the page lists the
    registered items only."""
    complete = evaluation["counts"][FULFILLED] == len(evaluation["items"])
    lines = ["# Data and code availability statement", "",
             "Each item of the paper's data and code availability statement and where the repository provides it. "
             "`make statement-check` checks every item; `make statement-fulfilled` also downloads the complete "
             "artifacts and runs the reconstruction.", "",
             "| Statement item | Provided by |", "|---|---|"]
    lines += [f"| {row['phrase']} | {PROVIDED[row['id']]} |" for row in evaluation["items"]]
    lines.append("")
    if complete:
        lines += ["Every item is fulfilled.", ""]
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
    demo = demonstration(evaluation, ROOT)
    text = render(evaluation, printed, demo)
    if args.json:
        print(json.dumps({**evaluation, "printed": printed, "demonstration": demo}, indent=1, ensure_ascii=False))
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
