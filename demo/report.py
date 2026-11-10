#!/usr/bin/env python3
"""Write demo/RESULTS.md: what the published CPU demonstration contains and how it was produced.

Every number comes from the sealed runs, the reduced-graph records, the production plan's task
markers, the demonstration manifest and the status of ``make demo-paper``.

    python -B demo/report.py --status ../cm-demo-paper.status.json --graph WORK [--graph WORK2] --runs RUNS \
        --graphs GRAPHS [--pack-failures FAILURES.json]

The first --graph is the main production graph; a further one is a graph planned for one record set alone
(demo-sweep.yml), whose artifacts demo/merge_maps.py added to the main map.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STEMS = {"ogbn-arxiv": "arxiv", "ogbn-proteins": "proteins", "ogbn-products": "products",
         "Roman-empire": "roman_empire", "PascalVOC-SP": "pascalvoc_sp", "COCO-SP": "coco_sp",
         "Peptides-func": "peptides_func"}
STAGES = ("prepare", "train", "explore", "fit", "select", "confirm", "admit", "evaluate")


def read(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def size(path: Path) -> int:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def display_rows(workspace: Path | None, status: dict) -> list[str]:
    """What ``make demo-paper`` built, from its workspace (reconstruction.json): the display generators that ran on
    the release or on the repository's own sources, each display still waiting for an input with what it lacks,
    and the printed-value comparison; without a workspace, the status counts alone."""
    result = read(workspace / "reconstruction.json") if workspace is not None else None
    comparison = (f"- Printed values: {status.get('values_exact', 0)} equal, {status.get('values_differ', 0)} "
                  f"different, {status.get('values_missing', 0)} missing. Differences are expected: these are "
                  "reduced-scale CPU runs, not the runs the paper reports.")
    if not result:
        return [f"- Displays: {status.get('complete', 0)} of {status.get('displays', 0)} complete in content and "
                "comparison coverage; the others name the artifact or printed-value contract they lack.", comparison]
    generators = result.get("display_generators") or {}
    displays = result.get("displays") or []
    kinds = {}
    for output in result.get("outputs") or []:
        for display in output.get("display_ids") or []:
            if str(output.get("id", "")).startswith("display:"):
                kinds[display] = output.get("input_kind")
    built = [d for d in displays if d.get("generator") == "built"]
    own = [d for d in built if kinds.get(d["id"]) == "repository_sources"]
    waiting = [d for d in displays if d.get("generator") == "awaiting_inputs"]
    supplement = result.get("supplementary_build") or {}
    lines = [f"- Display generators: {len(built)} of {len(displays)} built, {len(built) - len(own)} of them from the "
             f"release and {len(own)} from the repository's own sources (protocol and analytical displays); "
             f"{len(waiting)} wait for an input (below). Every display built from the release is labelled as the "
             "demonstration (synthetic software example).",
             f"- Supplementary builds from the release's record sets and campaign: "
             f"{len(supplement.get('outputs') or [])} output(s), status {supplement.get('status', 'not run')}.",
             comparison, "", "Displays waiting for an input, with what they lack:", ""]
    lines += [f"- `{d['id']}`: {'; '.join(str(x) for x in d.get('awaiting') or [])[:300]}" for d in waiting]
    return lines


def run_rows(runs: Path, graphs: Path) -> list[str]:
    lines = ["| Dataset | Data | Profile | Seconds per stage (" + ", ".join(STAGES) + ") | Size |",
             "|---|---|---|---|---|"]
    for dataset, stem in STEMS.items():
        base = runs / stem
        members = sorted(base.glob("split_*")) or [base]
        for member in members:
            config = read(member / "config.json", {}) or {}
            if not config:
                continue
            seconds = []
            for stage in STAGES:
                record = read(member / "stages" / f"{stage}.json", {}) or {}
                seconds.append(f"{record.get('seconds_this_attempt', 0):.0f}")
            built = read(graphs / f"{stem}_demo.json")
            if built:
                sizes = built["sizes"]
                if "graphs" in sizes:
                    data = (f"first {sum(sizes['graphs'].values()):,} of {sum(sizes['full_graphs'].values()):,} "
                            f"official graphs ({sizes['nodes']:,} nodes)")
                else:
                    parts = ("one part per official role" if built["rule"]["kind"] == "metis_parts_per_role"
                             else f"{built['rule']['keep']} part")
                    data = (f"{sizes['nodes']:,} of {sizes['full_nodes']:,} nodes "
                            f"({parts} of a {built['rule']['parts']}-way METIS partition)")
            else:
                data = "complete official graph"
            name = dataset + (f" {member.name}" if member is not base else "")
            lines.append(f"| {name} | {data} | {config.get('profile')} | {' / '.join(seconds)} | "
                         f"{size(member) / 2**20:,.0f} MiB |")
    return lines


def graph_rows(work: Path) -> list[str]:
    """Every planned node with the states its task markers record (scripts/paper_outputs.py execute)."""
    from demo.execute import levels
    plan = read(work / "plan.json")
    level = levels(plan)
    lines = ["| Node | Level | Tasks complete | State |", "|---|---|---|---|"]
    for node in sorted(plan["nodes"], key=lambda n: (level[n["id"]], n["id"])):
        count = max(1, len(node["tasks"]))
        states = [(read(work / "state" / node["id"] / f"task-{k}.json", {}) or {}).get("status") or "NOT RUN"
                  for k in range(count)]
        done = sum(s in ("COMPLETE", "SKIPPED") for s in states)
        if node.get("blocked"):
            state = "blocked at planning: " + str(node["blocked"])[:160].replace("|", "/")
        elif done == count:
            state = "complete"
        else:
            other = sorted({s for s in states if s not in ("COMPLETE", "SKIPPED")})
            state = "incomplete (" + ", ".join(other) + ")"
        lines.append(f"| {node['id']} | {level[node['id']]} | {done}/{count} | {state} |")
    return lines


def artifact_rows(works: list[Path], failures: dict, notes: dict | None = None) -> list[str]:
    """Every artifact that no graph produced, with the main graph's reason (or the publication request's note on
    it, when the graph's reason names a step rather than the cause), and those the packer refused."""
    notes = notes or {}
    plan = read(works[0] / "plan.json", {}) or {}
    mapping = read(works[0] / "paper-artifacts.map.json", {}) or {}
    produced = {key for work in works for key, value in (read(work / "paper-artifacts.map.json", {}) or {}).items()
                if not key.startswith("_") and value}
    lines = []
    for artifact in sorted(plan.get("artifacts", {})):
        if artifact in failures:
            lines.append(f"- `{artifact}`: produced, refused by the packer: {str(failures[artifact])[:400]}")
        elif artifact not in produced:
            reason = (notes.get(artifact) or (mapping.get("_awaiting") or {}).get(artifact)
                      or (plan["artifacts"][artifact] or {}).get("reason") or "")
            lines.append(f"- `{artifact}`: {str(reason)[:600]}")
    return lines


def graph_section(work: Path, runs: list[str], first: bool) -> list[str]:
    plan = read(work / "plan.json", {}) or {}
    links = ", ".join(f"[{run}](https://github.com/VKHS/CR-9/actions/runs/{run})" for run in runs)
    only = (plan.get("options") or {}).get("only") or []
    if first:
        text = ("The repository's own production graph (scripts/paper_outputs.py, planned with `--allow-fixture "
                "--device cpu`) executed every node it could plan on CPU (demo/execute.py"
                + (f"; workflow runs {links}" if links else "") + ").")
    else:
        text = (f"A second graph, planned for {', '.join(f'`{o}`' for o in only) or 'one record set'} alone in its own "
                "work directory (demo-sweep.yml" + (f"; workflow run {links}" if links else "") + "), produced the "
                "artifacts that demo/merge_maps.py added to the main graph's map.")
    return [text + " Its declared choices are recorded in the plan: "
            + json.dumps(plan.get("declared_choices", {}), sort_keys=True)[:1200] + ".", ""] + graph_rows(work)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--status", type=Path, required=True, help="the status JSON written by make demo-paper")
    parser.add_argument("--graph", type=Path, action="append", required=True,
                        help="a production work directory (the main graph first; repeatable)")
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--graph-run", action="append", default=[], metavar="[INDEX=]RUN",
                        help="a workflow run that executed a graph; INDEX (default 0) is the --graph it belongs to")
    parser.add_argument("--pack-failures", type=Path, default=None,
                        help="the artifacts demo/pack_each.py could not pack, with the packer's reasons")
    parser.add_argument("--notes", type=Path, default=None,
                        help="JSON {artifact: text}: the cause of an absence the graph's reason does not state")
    parser.add_argument("--workspace", type=Path, default=None,
                        help="the workspace of make demo-paper (reconstruction.json): the displays it built and awaits")
    parser.add_argument("--output", type=Path, default=ROOT / "demo" / "RESULTS.md")
    args = parser.parse_args(argv)
    status = read(args.status, {}) or {}
    failures = (read(args.pack_failures, {}) or {}) if args.pack_failures else {}
    graph_runs = {}
    for spec in args.graph_run:
        index, _, run = spec.rpartition("=")
        graph_runs.setdefault(int(index or 0), []).append(run)
    manifest = read(ROOT / "release" / "demo_artifacts.json", {}) or {}
    published = [a for a in manifest.get("artifacts", []) if a.get("status") == "published"]
    awaiting = [a for a in manifest.get("artifacts", []) if a.get("status") != "published"]
    lines = ["# CPU demonstration: results", "",
             f"> **Demonstration.** {manifest.get('scope', '')}", "",
             f"Release: [{manifest.get('release_tag', 'demo-artifacts')}]({manifest.get('release_url', '')}) — "
             f"{len(published)} of {len(manifest.get('artifacts', []))} artifacts published "
             f"({sum(a.get('bytes', 0) for a in published) / 2**30:.2f} GiB, "
             f"{sum(a.get('files', 0) for a in published):,} files).", "",
             "## Rebuilt from the release with `make demo-paper`", ""] + display_rows(args.workspace, status) + [
             ""] + [
             f"- Generator error: `{name}`: {str(error)[:400]}"
             for name, error in sorted({**(status.get("generator_errors") or {}),
                                        **(status.get("module_errors") or {})}.items())] + [
             "",
             "## The seven lifecycle runs", ""] + run_rows(args.runs, args.graphs) + [
             "", "Every run executed the eight stages of the lifecycle from this checkout on a hosted 4-CPU runner "
             "(demo-runs.yml). Configurations: demo/configure.py (the pilot settings, fp32 plant).", "",
             "## The production graph", ""]
    for index, work in enumerate(args.graph):
        lines += graph_section(work, graph_runs.get(index, []), index == 0) + [""]
    rows = artifact_rows(args.graph, failures, (read(args.notes, {}) or {}) if args.notes else {})
    lines += ["## Not in the demonstration", "", "Each artifact the demonstration does not hold, with its reason:", ""]
    lines += rows or [f"- `{entry['id']}`" for entry in awaiting]
    lines += ["", "Generated by `demo/report.py` in the publishing workflow.", ""]
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
