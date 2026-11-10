#!/usr/bin/env python3
"""Train the additional checkpoints of one E14 ensemble-grid job on several runners at once.

An E14 job (cm_experiments/ensemble_grid.py, ``_run_grid_job``) trains its additional checkpoint seeds one
after another with ``cross_morse.training.train_one``, which skips a seed whose completion record
(``plant_seed<s>_done.json``) exists and resumes an interrupted one from its last epoch. The registered grid
reaches E = 20, so a job trains eighteen plants; on a 4-CPU runner the PascalVOC-SP job needs more than the
six hours of one job. This helper makes the job's own call (``train_one`` in the job's parent context, on the
D_fit roles) for one share of those seeds, into the job's own checkpoint directory. The job itself then finds
them complete and computes its member predictions and grid statistics.

    python -B demo/grid_seeds.py --output EXPERIMENT_DIR --job 12 --list            # unfinished seeds (JSON)
    python -B demo/grid_seeds.py --output EXPERIMENT_DIR --job 12 --shard 2/6 --device cpu --threads 4
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def unfinished(output: Path, job: int) -> tuple[dict, dict, Path, list[int]]:
    from cm_experiments import ensemble_grid as eg
    plan = eg.load_plan(output, "ensemble_grid")
    entry, = eg.selected_jobs(plan, job)
    checkpoints = eg.job_directory(output, entry) / "checkpoints"
    seeds = [int(s) for s in entry["extra_seeds"] if not (checkpoints / f"plant_seed{s}_done.json").exists()]
    return plan, entry, checkpoints, seeds


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="the ensemble-grid experiment directory")
    parser.add_argument("--job", type=int, required=True, help="the job index of its plan")
    parser.add_argument("--shard", default="1/1", help="i/n: the i-th of n interleaved shares of the unfinished seeds")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--list", action="store_true", help="print the unfinished seeds as JSON")
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    plan, entry, checkpoints, seeds = unfinished(output, args.job)
    if args.list:
        print(json.dumps(seeds))
        return 0
    shard, shards = (int(x) for x in args.shard.split("/"))
    mine = [s for i, s in enumerate(seeds) if i % shards == shard - 1]
    print(f"job {args.job} ({entry['dataset']}, {entry['member']}): {len(seeds)} unfinished seed(s); "
          f"share {shard}/{shards}: {mine}", flush=True)
    if not mine:
        return 0
    import torch
    from cm_experiments import ensemble_grid as eg
    from cross_morse.training import train_one
    torch.set_num_threads(max(1, args.threads))
    run = eg.SealedRun(entry["run"])
    eg.check_identity(run, entry["identity"])
    eg.check_role(run, plan["role"])
    ctx = eg.ParentContext(run, args.device, through="evaluate" if plan["role"] == "test" else "confirm")
    for seed in mine:
        # The parent's frozen training configuration and the D_fit roles; only the seed differs (as in the job).
        with ctx.data._label_scope("train"):
            train_one(ctx.data, ctx.graph, ctx.config, checkpoints, int(seed), ctx.device)
        print(f"seed {seed}: complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
