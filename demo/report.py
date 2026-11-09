#!/usr/bin/env python3
"""Write demo/RESULTS.md: what the published CPU demonstration contains and how it was produced.

Every number comes from the sealed runs, the reduced-graph records, the production plan's task
markers, the demonstration manifest and the status of ``make demo-paper``.

    python -B demo/report.py --status ../cm-demo-paper.status.json --graph WORK --runs RUNS --graphs GRAPHS
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
                    data = (f"{sizes['nodes']:,} of {sizes['full_nodes']:,} nodes "
                            f"(METIS part 1 of {built['rule']['parts']})")
            else:
                data = "complete official graph"
            name = dataset + (f" {member.name}" if member is not base else "")
            lines.append(f"| {name} | {data} | {config.get('profile')} | {' / '.join(seconds)} | "
                         f"{size(member) / 2**20:,.0f} MiB |")
    return lines


def graph_rows(work: Path) -> list[str]:
    from demo.execute import levels, node_state
    from scripts import paper_outputs as po
    plan = po.load_plan(work)
    level = levels(plan)
    lines = ["| Node | Level | Tasks | State |", "|---|---|---|---|"]
    for node in sorted(plan["nodes"], key=lambda n: (level[n["id"]], n["id"])):
        state = node_state(work, node)
        if node.get("blocked"):
            state += ": " + str(node["blocked"])[:160].replace("|", "/")
        lines.append(f"| {node['id']} | {level[node['id']]} | {max(1, len(node['tasks']))} | {state} |")
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--status", type=Path, required=True, help="the status JSON written by make demo-paper")
    parser.add_argument("--graph", type=Path, required=True, help="the production work directory")
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "demo" / "RESULTS.md")
    args = parser.parse_args(argv)
    status = read(args.status, {}) or {}
    manifest = read(ROOT / "release" / "demo_artifacts.json", {}) or {}
    published = [a for a in manifest.get("artifacts", []) if a.get("status") == "published"]
    awaiting = [a for a in manifest.get("artifacts", []) if a.get("status") != "published"]
    lines = ["# CPU demonstration: results", "",
             f"> **Demonstration.** {manifest.get('scope', '')}", "",
             f"Release: [{manifest.get('release_tag', 'demo-artifacts')}]({manifest.get('release_url', '')}) — "
             f"{len(published)} of {len(manifest.get('artifacts', []))} artifacts published "
             f"({sum(a.get('bytes', 0) for a in published) / 2**30:.2f} GiB, "
             f"{sum(a.get('files', 0) for a in published):,} files).", "",
             "## Rebuilt from the release with `make demo-paper`", "",
             f"- Displays: {status.get('complete', 0)} of {status.get('displays', 0)} complete in content and comparison "
             "coverage; the others name the artifact or printed-value contract they lack.",
             f"- Printed values: {status.get('values_exact', 0)} equal, {status.get('values_differ', 0)} different, "
             f"{status.get('values_missing', 0)} missing. Differences are expected: these are reduced-scale CPU runs, "
             "not the runs the paper reports.", "",
             "## The seven lifecycle runs", ""] + run_rows(args.runs, args.graphs) + [
             "", "Every run executed the eight stages of the lifecycle from this checkout on a hosted 4-CPU runner "
             "(demo-runs.yml). Configurations: demo/configure.py (the pilot settings, fp32 plant).", "",
             "## The production graph", "",
             "The repository's own production graph (scripts/paper_outputs.py, planned with `--allow-fixture "
             "--device cpu`) executed every node it could plan on CPU (demo-graph.yml, demo/execute.py).", ""] + graph_rows(
             args.graph) + ["", "## Not in the demonstration", ""]
    for entry in awaiting:
        lines.append(f"- `{entry['id']}`")
    lines += ["", "Generated by `demo/report.py` in the publishing workflow.", ""]
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
