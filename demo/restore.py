#!/usr/bin/env python3
"""Restore the parts of a production work directory that one job needs, from workflow artifacts.

Each executing shard of the demonstration graph saves everything it wrote under the work directory as
one archive (demo/work_delta.sh save), so a shard's archive holds several experiments. A later job needs
only the outputs of the nodes its own tasks depend on; the complete set is too large for one runner's disk.
This tool downloads the archives one at a time (GitHub REST API, GH_TOKEN), extracts only the members under
the selected nodes' outputs and task markers, and deletes each archive before the next.

    python -B demo/restore.py --work W --run RUN --prefix demo-work-base --all
    python -B demo/restore.py --work W --run RUN --prefix demo-work-L --level 1      # what level 1 depends on
    python -B demo/restore.py --work W --run RUN --prefix demo-work-L --export export.factorials
    python -B demo/restore.py --work W --plan-only --level 1                          # print the selection

Several ``--run``/``--prefix`` pairs are processed in the order given; later archives overwrite earlier files.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def closure(plan: dict, start: set[str]) -> set[str]:
    nodes = {n["id"]: n for n in plan["nodes"]}
    seen, stack = set(), list(start)
    while stack:
        node = stack.pop()
        if node in seen or node not in nodes:
            continue
        seen.add(node)
        stack.extend(d["node"] for d in nodes[node].get("dependencies") or [])
    return seen


def selection(plan: dict, work: Path, *, level: int | None, nodes: list[str], export: str | None) -> tuple[set, list]:
    """The nodes whose outputs are needed and the path prefixes (relative to /) to extract."""
    from demo_levels import levels  # noqa: E402  (demo/execute.py, loaded below)
    by_id = {n["id"]: n for n in plan["nodes"]}
    wanted: set[str] = set(nodes)
    if level is not None:
        depth = levels(plan)
        for node_id, node in by_id.items():
            if depth[node_id] == level:
                wanted.update(d["node"] for d in node.get("dependencies") or [])
    if export is not None:
        entry = next(e for e in plan["exports"] if e["id"] == export)
        wanted.update(entry["after"])
        experiments = {by_id[n]["experiment"] for n in entry["after"] if n in by_id}
        wanted.update(n for n, node in by_id.items() if node.get("experiment") in experiments)
    needed = closure(plan, wanted)
    prefixes = {str(work / "state").lstrip("/") + "/"}
    for node_id in needed:
        for output in by_id[node_id].get("outputs") or []:
            prefixes.add(str(Path(output)).lstrip("/") + "/")
            prefixes.add(str(Path(output)).lstrip("/"))
    return needed, sorted(prefixes)


def artifacts(run: str, prefix: str) -> list[tuple[int, str]]:
    api = f"https://api.github.com/repos/{os.environ['GITHUB_REPOSITORY']}"
    found, page = [], 1
    while True:
        listing = json.loads(subprocess.check_output(
            ["curl", "-fsSL", "--retry", "8", "--retry-delay", "15", "--retry-all-errors", "-H", f"Authorization: Bearer {os.environ['GH_TOKEN']}", "-H",
             "Accept: application/vnd.github+json", f"{api}/actions/runs/{run}/artifacts?per_page=100&page={page}"]))
        rows = listing["artifacts"]
        found += [(a["id"], a["name"]) for a in rows if a["name"].startswith(prefix) and not a["expired"]]
        if len(rows) < 100:
            break
        page += 1
    return sorted(found, key=lambda row: row[1])


EXCLUDED: list[str] = []      # path prefixes (relative to /) left out of every extraction (--exclude)
PATTERNS: list[str] = []      # wildcard patterns of member names left out (--exclude-pattern)


def extract(archive_zip: Path, prefixes: list[str] | None) -> str:
    """Stream every tar.gz inside one artifact zip through unzip and GNU tar (both in C)."""
    members = subprocess.check_output(["unzip", "-Z1", str(archive_zip)], text=True).split()
    report = []
    exclude = ["--anchored", *(f"--exclude={p.rstrip('/')}" for p in EXCLUDED)] if EXCLUDED else []
    if PATTERNS:
        exclude += ["--no-anchored", "--wildcards", *(f"--exclude={p}" for p in PATTERNS)]
    for member in (m for m in members if m.endswith(".tar.gz")):
        names = [p.rstrip("/") for p in prefixes] if prefixes is not None else []
        unzip = subprocess.Popen(["unzip", "-p", str(archive_zip), member], stdout=subprocess.PIPE)
        # A directory name extracts everything below it; a name absent from this archive is not an error here.
        tar = subprocess.run(["tar", "-xzf", "-", "-C", "/", "--no-same-owner", *exclude, *names], stdin=unzip.stdout,
                             capture_output=True, text=True)
        unzip.stdout.close()
        unzip.wait()
        if tar.returncode not in (0, 2) or (tar.returncode == 2 and "Not found in archive" not in tar.stderr):
            raise RuntimeError(f"{archive_zip.name}/{member}: tar failed: {tar.stderr[-800:]}")
        report.append(member)
    return ", ".join(report)


def level_of(name: str) -> str:
    """demo-work-L3-12 -> L3; the archives of one level never write the same task's files."""
    parts = name.split("-")
    return parts[2] if len(parts) > 2 else name


def fetch_and_extract(artifact_id: int, name: str, prefixes) -> str:
    with tempfile.TemporaryDirectory(prefix="restore-", dir=os.environ.get("DEMO_TEMP") or os.environ.get("RUNNER_TEMP")) as temporary:
        archive = Path(temporary) / "artifact.zip"
        subprocess.run(["curl", "-fsSL", "--retry", "8", "--retry-delay", "15", "--retry-all-errors", "-H",
                        f"Authorization: Bearer {os.environ['GH_TOKEN']}", "-o", str(archive),
                        f"https://api.github.com/repos/{os.environ['GITHUB_REPOSITORY']}/actions/artifacts/"
                        f"{artifact_id}/zip"], check=True)
        return f"{name}: {extract(archive, prefixes)}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--run", action="append", default=[])
    parser.add_argument("--prefix", action="append", default=[])
    parser.add_argument("--all", action="store_true", help="extract every member")
    parser.add_argument("--level", type=int, default=None)
    parser.add_argument("--node", action="append", default=[])
    parser.add_argument("--export", default=None)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--parallel", type=int, default=4, help="archives of one level processed at once")
    parser.add_argument("--exclude", action="append", default=[], metavar="PATH",
                        help="leave this path (and everything below it) out of the extraction (repeatable)")
    parser.add_argument("--exclude-from", type=Path, default=None, help="a file of such paths, one per line")
    parser.add_argument("--exclude-pattern", action="append", default=[], metavar="GLOB",
                        help="leave out every member whose name matches this wildcard (repeatable), e.g. '*.pt'")
    args = parser.parse_args(argv)
    PATTERNS[:] = list(args.exclude_pattern)
    EXCLUDED[:] = [p.lstrip("/") for p in args.exclude]
    if args.exclude_from is not None and args.exclude_from.is_file():
        EXCLUDED.extend(line.strip().lstrip("/") for line in args.exclude_from.read_text().splitlines() if line.strip())
    if EXCLUDED:
        print(f"excluded from the extraction: {len(EXCLUDED)} path(s)")
    if len(args.run) != len(args.prefix) and not args.plan_only:
        parser.error("give one --prefix per --run")
    import importlib.util
    spec = importlib.util.spec_from_file_location("demo_levels", ROOT / "demo" / "execute.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["demo_levels"] = module
    spec.loader.exec_module(module)
    prefixes = None
    if not args.all:
        plan = read_json(args.work / "plan.json")
        needed, prefixes = selection(plan, args.work, level=args.level, nodes=args.node, export=args.export)
        print(f"nodes needed: {len(needed)}: {', '.join(sorted(needed))[:2000]}")
        if args.plan_only:
            print("\n".join(prefixes))
            return 0
    import concurrent.futures as futures
    for run, prefix in zip(args.run, args.prefix):
        found = artifacts(run, prefix)
        groups: dict[str, list] = {}
        for artifact_id, name in found:
            groups.setdefault(level_of(name), []).append((artifact_id, name))
        for group in sorted(groups):            # levels in order; the shards of one level in parallel
            with futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
                for line in pool.map(lambda row: fetch_and_extract(row[0], row[1], prefixes), groups[group]):
                    print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
