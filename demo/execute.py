#!/usr/bin/env python3
"""Execute a production plan (scripts/paper_outputs.py plan) on one machine, without Slurm.

The Slurm workflow submits one job per node and task. This executor runs the same tasks
with the same entry point (``scripts/paper_outputs.py execute --work W --node N --task K``),
so the task markers, the completion seals and ``collect`` are unchanged. Nodes run level by
level: a node starts after every node it depends on (``afterok``: completed; ``afterany``:
finished). A runner that checkpointed (exit 75) is resumed by running its task again.

    python -B demo/execute.py --work W                        # everything, in dependency order
    python -B demo/execute.py --work W --shard 2/4 --level 0  # one share of the first level
    python -B demo/execute.py --work W --status               # what is complete, failed or pending

``--budget-minutes`` stops starting tasks after that time so that a time-limited job can save
its work; a later call continues where it stopped.

``--shard i/n`` takes every n-th task of a level, counted over all of its tasks (complete or not), so
every share agrees on the assignment whatever it has restored. With ``--salt`` the tasks are counted in
an order drawn from the salt (every job of one workflow run passes the same): the tasks that one call
left unfinished then spread over every share of the next call instead of falling to one share.
``--matrix`` prints, per level, the shares that hold an unfinished task under the same assignment.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import paper_outputs as po  # noqa: E402

CHECKPOINTED = 75


def shares(tasks: list, shards: int, salt: str | None) -> dict:
    """The share (1..shards) of every (node, task): every shards-th task, in an order drawn from the salt."""
    order = list(tasks)
    if salt:
        order.sort(key=lambda t: hashlib.sha256(f"{salt}:{t[0]}:{t[1]}".encode()).hexdigest())
    return {task: index % shards + 1 for index, task in enumerate(order)}


def left_out(work: Path, nodes: dict, skip) -> set:
    """The nodes left to another job (--skip-node) and every node that waits on one of them that has not finished:
    none of them can run in this call, so no share counts or runs their tasks."""
    def finished(node_id: str) -> bool:
        for k in range(max(1, len(nodes[node_id]["tasks"]))):
            try:
                status = po.read_json(po.task_marker(work, node_id, k)).get("status")
            except (OSError, ValueError):
                return False
            if status not in ("COMPLETE", "SKIPPED"):
                return False
        return True
    out, changed = set(n for n in skip if n in nodes), True
    while changed:
        changed = False
        for node_id, node in nodes.items():
            if node_id not in out and any(d["node"] in out and not finished(d["node"])
                                          for d in node.get("dependencies") or []):
                out.add(node_id)
                changed = True
    return out


def levels(plan: dict) -> dict[str, int]:
    nodes = {n["id"]: n for n in plan["nodes"]}
    level: dict[str, int] = {}

    def depth(node_id: str, trail=()) -> int:
        if node_id in level:
            return level[node_id]
        if node_id in trail:
            raise ValueError(f"dependency cycle through {node_id}")
        deps = [d["node"] for d in nodes[node_id].get("dependencies") or [] if d["node"] in nodes]
        level[node_id] = 0 if not deps else 1 + max(depth(d, trail + (node_id,)) for d in deps)
        return level[node_id]

    for node_id in nodes:
        depth(node_id)
    return level


TRUST_MARKERS = {"value": False}


def task_state(work: Path, node: dict, index: int) -> str:
    """COMPLETE when the task's completion seals verify; with --trust-markers, when its marker says so.

    Trusting the marker serves a job that holds the markers but not the outputs of finished tasks (it
    executes only the tasks that never finished); it is never used to decide a dependency's completion."""
    if po.task_complete(work, node, index):
        return "COMPLETE"
    try:
        status = po.read_json(po.task_marker(work, node["id"], index)).get("status") or "PENDING"
    except (OSError, ValueError):
        return "PENDING"
    if TRUST_MARKERS["value"] and status in ("COMPLETE", "SKIPPED") and not node.get("dependencies"):
        return "COMPLETE"
    return status


def node_state(work: Path, node: dict) -> str:
    states = [task_state(work, node, k) for k in range(max(1, len(node["tasks"])))]
    if node.get("blocked"):
        return "BLOCKED"
    if all(s in ("COMPLETE", "SKIPPED") for s in states):
        return "COMPLETE"
    if any(s == "FAILED" for s in states):
        return "FAILED"
    return "PENDING"


def runnable(work: Path, nodes: dict, node: dict) -> tuple[bool, str]:
    if node.get("blocked"):
        return False, "blocked at planning: " + str(node["blocked"])[:300]
    for dep in node.get("dependencies") or []:
        if dep["node"] not in nodes:
            continue
        state = node_state(work, nodes[dep["node"]])
        if dep.get("type") == "afterany":
            if state == "PENDING":
                return False, f"waits for {dep['node']}"
        elif state != "COMPLETE":
            return False, f"needs {dep['node']} ({state})"
    return True, ""


def run_task(work: Path, node_id: str, index: int, threads: int, log_dir: Path, retries: int) -> dict:
    started = time.time()
    env = dict(os.environ, CM_THREADS=str(threads), OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    log = log_dir / f"{node_id}.task{index}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    attempts, code = 0, None
    with log.open("a", encoding="utf-8") as stream:
        while attempts <= retries:
            attempts += 1
            stream.write(f"\n=== attempt {attempts} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
            stream.flush()
            code = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/paper_outputs.py"), "execute", "--work",
                                   str(work), "--node", node_id, "--task", str(index)],
                                  cwd=str(ROOT), env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
            marker = {}
            try:
                marker = po.read_json(po.task_marker(work, node_id, index))
            except (OSError, ValueError):
                pass
            if code == 0 and marker.get("status") in ("COMPLETE", "SKIPPED"):
                break
            if marker.get("exit_code") == CHECKPOINTED or code == CHECKPOINTED:
                # A local rerun resumes the runner from its last safe boundary.
                path = po.task_marker(work, node_id, index)
                record = dict(marker, status="REQUEUED")
                po.write_json(path, record)
                continue
            break
    return {"node": node_id, "task": index, "exit_code": code, "attempts": attempts,
            "seconds": round(time.time() - started, 1), "log": str(log)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--level", type=int, default=None, help="only this dependency level")
    parser.add_argument("--node", action="append", default=None, help="only these nodes (repeatable)")
    parser.add_argument("--skip-node", action="append", default=[], metavar="NODE",
                        help="leave this node's tasks to another job (repeatable); --matrix does not count them")
    parser.add_argument("--salt", default=None,
                        help="count the tasks of a level in an order drawn from this value (the same in every share)")
    parser.add_argument("--shard", default="1/1", help="i/n: the i-th of n interleaved shares of the selected tasks")
    parser.add_argument("--parallel", type=int, default=1, help="tasks run at once (threads are shared)")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--budget-minutes", type=float, default=None)
    parser.add_argument("--retries", type=int, default=12, help="resumptions of a checkpointed task")
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--trust-markers", action="store_true",
                        help="count a dependency-free task as complete when its marker says so (outputs absent)")
    parser.add_argument("--matrix", type=int, default=None, metavar="MAX_SHARDS",
                        help="print, per dependency level, the shard numbers to run (at most MAX_SHARDS) as JSON")
    args = parser.parse_args(argv)
    work = args.work.expanduser().resolve()
    TRUST_MARKERS["value"] = bool(args.trust_markers)
    plan = po.load_plan(work)
    nodes = {n["id"]: n for n in plan["nodes"]}
    level = levels(plan)
    left = left_out(work, nodes, args.skip_node)
    ordered = [n for n in sorted(nodes, key=lambda n: (level[n], list(nodes).index(n))) if n not in left]
    if args.matrix is not None:
        matrix = {}
        for current in sorted({level[n] for n in ordered}):
            tasks = [(n, k) for n in ordered if level[n] == current for k in range(max(1, len(nodes[n]["tasks"])))]
            share = shares(tasks, args.matrix, args.salt)
            busy = sorted({share[(n, k)] for n, k in tasks if not nodes[n].get("blocked")
                           and task_state(work, nodes[n], k) not in ("COMPLETE", "SKIPPED")})
            if busy:
                matrix[f"l{current}"] = busy
        print(json.dumps(matrix))
        return 0
    if args.status:
        for node_id in sorted(nodes, key=lambda n: (level[n], n)):
            node = nodes[node_id]
            states = [task_state(work, node, k) for k in range(max(1, len(node["tasks"])))]
            print(f"L{level[node_id]} {node_state(work, node):9s} {node_id:44s} "
                  f"{sum(s in ('COMPLETE', 'SKIPPED') for s in states)}/{len(states)}")
        return 0
    shard, shards = (int(x) for x in args.shard.split("/"))
    selected = [n for n in ordered if (args.level is None or level[n] == args.level) and (not args.node or n in args.node)]
    deadline = None if args.budget_minutes is None else time.time() + 60 * args.budget_minutes
    threads = max(1, args.threads // max(1, args.parallel))
    results, skipped = [], []
    for current in sorted({level[n] for n in selected}):
        tasks = [(n, k) for n in selected if level[n] == current
                 for k in range(max(1, len(nodes[n]["tasks"])))]
        share = shares(tasks, shards, args.salt)
        tasks = [t for t in tasks if share[t] == shard
                 and task_state(work, nodes[t[0]], t[1]) not in ("COMPLETE", "SKIPPED")]
        pending = []
        for node_id, index in tasks:
            ok, why = runnable(work, nodes, nodes[node_id])
            if not ok:
                skipped.append({"node": node_id, "task": index, "reason": why})
            else:
                pending.append((node_id, index))
        with futures.ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
            running = {}
            queue = list(pending)
            while queue or running:
                while queue and len(running) < max(1, args.parallel):
                    if deadline is not None and time.time() > deadline:
                        skipped.extend({"node": n, "task": k, "reason": "time budget"} for n, k in queue)
                        queue = []
                        break
                    node_id, index = queue.pop(0)
                    print(f"[start] L{current} {node_id} task {index}", flush=True)
                    running[pool.submit(run_task, work, node_id, index, threads, work / "demo-logs",
                                        args.retries)] = (node_id, index)
                if not running:
                    break
                done, _ = futures.wait(running, return_when=futures.FIRST_COMPLETED)
                for future in done:
                    running.pop(future)
                    result = future.result()
                    state = task_state(work, nodes[result["node"]], result["task"])
                    result["state"] = state
                    results.append(result)
                    print(f"[done ] {result['node']} task {result['task']}: {state} exit {result['exit_code']} "
                          f"{result['seconds']} s", flush=True)
    summary = {"work": str(work), "shard": args.shard, "level": args.level, "results": results, "skipped": skipped}
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    failed = [r for r in results if r["state"] not in ("COMPLETE", "SKIPPED")]
    print(f"executed {len(results)} task(s): {len(results) - len(failed)} complete, {len(failed)} not complete; "
          f"{len(skipped)} not started")
    return 0


if __name__ == "__main__":
    sys.exit(main())
