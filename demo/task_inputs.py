#!/usr/bin/env python3
"""The variant lifecycle runs that one share of a dependency level does not read, for restore.py --exclude-from.

A measurement job of the module ablations (cm_experiments/ablation_lifecycle.py, job kind "measure") reads
the variant lifecycle runs of its own parent run, which its arms name; the node's output directory holds the
variant runs of every parent run, more than one runner's disk. For the tasks that this share executes (the
assignment of demo/execute.py), this tool prints the lifecycle run directories of the same runner plans that
none of them reads, one per line and relative to /; a share without measurement tasks prints nothing.

    python -B demo/task_inputs.py --work W --level 4 --shard 3/16 --salt RUN_ID [--skip-node N ...] > EXCLUDES
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def executor():
    spec = importlib.util.spec_from_file_location("demo_execute", ROOT / "demo" / "execute.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unread_runs(plan: dict, *, level: int, shard: int, shards: int, salt: str | None, skip=()) -> list[str]:
    execute = executor()
    nodes = {n["id"]: n for n in plan["nodes"]}
    depth = execute.levels(plan)
    ordered = [n for n in sorted(nodes, key=lambda n: (depth[n], list(nodes).index(n))) if n not in skip]
    tasks = [(n, k) for n in ordered if depth[n] == level for k in range(max(1, len(nodes[n]["tasks"])))]
    share = execute.shares(tasks, shards, salt)
    runners: dict[str, dict] = {}
    every, read = set(), set()
    for node_id, index in tasks:
        task = nodes[node_id]["tasks"][index] if nodes[node_id]["tasks"] else {}
        job_in = task.get("job_in") or {}
        if job_in.get("kind") != "measure" or not job_in.get("plan"):
            continue
        if job_in["plan"] not in runners:
            runners[job_in["plan"]] = json.loads(Path(job_in["plan"]).read_text(encoding="utf-8"))
        jobs = {int(entry["job"]): entry for entry in runners[job_in["plan"]].get("jobs") or []}
        every.update(entry["run"] for entry in jobs.values() if entry.get("kind") == "lifecycle" and entry.get("run"))
        if share[(node_id, index)] == shard and int(job_in["index"]) in jobs:
            read.update(arm["run"] for arm in jobs[int(job_in["index"])].get("arms") or []
                        if arm.get("source") == "lifecycle" and arm.get("run"))
    return sorted(str(run).lstrip("/") for run in every - read)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--shard", required=True, help="i/n, as demo/execute.py --shard")
    parser.add_argument("--salt", default=None)
    parser.add_argument("--skip-node", action="append", default=[])
    args = parser.parse_args(argv)
    shard, shards = (int(x) for x in args.shard.split("/"))
    plan = json.loads((args.work / "plan.json").read_text(encoding="utf-8"))
    for line in unread_runs(plan, level=args.level, shard=shard, shards=shards, salt=args.salt, skip=args.skip_node):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
