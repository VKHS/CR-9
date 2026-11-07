#!/usr/bin/env python3
"""Plan, run and export the E17 timing records and the E14 official scores of the ensemble grid.

    python -B scripts/run_timing_records.py scaling plan|run|export ...   E17 -> records/runtime_scaling (Fig. S2)
    python -B scripts/run_timing_records.py anchors plan|run|export ...   E17 -> records/timing_anchors (Table S23b)
    python -B scripts/run_timing_records.py grid-scores export ...        E14 -> records/ensemble_grid_scores (Table S26)

scaling plan     registers the timed instances of each sealed parent run (Roman-empire: the campaign directory;
                 its first split run is timed): the full benchmark graph and breadth-first node-induced subgraphs
                 of a transductive graph, or test graphs of a collection at edge-count quantiles; writes
                 <output>/plan.json and the subgraphs' node ids
scaling run      times each instance with the E17 TimingSession of cm_campaigns.timing (one job per instance;
                 --job K for an array task); completed jobs are skipped. Needs PyTorch and
                 --trust-local-checkpoints. Run it inside the GPU allocation whose sacct record is exported later.
scaling export   binds every timing job to a completed GPU allocation of the sacct exports (--accounting) and
                 writes the record set; without them, or on CPU (no memory reading), the set is partial
anchors plan     declares the parents, the completed scaling experiment that supplies the throughput fit
                 (--scaling-output) and optionally the scheduler accounting of the plant seeds
anchors run      torch-free: training-time anchors from the sealed seed markers (or accounting), throughput
                 anchor from the scaling experiment's nominal timing rows
anchors export   writes the record set
grid-scores export
                 reads a completed `scripts/run_experiment.py ensemble-grid` output (--grid-output) and writes
                 the official score of every ensemble, recomputed from the saved member predictions

export writes into --root (default records/), traced value by value by scripts/records.py; a synthetic or
partial set can only be written to another root (--allow-partial).

Example (E17 on the sealed arxiv run, then the anchors):

    python -B scripts/run_timing_records.py scaling plan --parent ../cm-runs/arxiv-full --output ../cm-e17-scaling
    python -B scripts/run_timing_records.py scaling run --output ../cm-e17-scaling --device cuda --trust-local-checkpoints
    python -B scripts/run_timing_records.py scaling export --output ../cm-e17-scaling --accounting ../acct/sacct_original.psv
    python -B scripts/run_timing_records.py anchors plan --parent ../cm-runs/arxiv-full --output ../cm-e17-anchors \\
        --scaling-output ../cm-e17-scaling
    python -B scripts/run_timing_records.py anchors run --output ../cm-e17-anchors
    python -B scripts/run_timing_records.py anchors export --output ../cm-e17-anchors
    python -B scripts/run_timing_records.py grid-scores export --grid-output ../cm-e14
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _scaling(args):
    from cm_experiments import timing_records as tr
    if args.action == "plan":
        return tr.plan_runtime_scaling(args.parent, args.output, fractions=args.fractions,
                                       graph_quantiles=args.graph_quantiles, subgraph_seed=args.subgraph_seed,
                                       warmup=args.warmup, repetitions=args.repetitions)
    if args.action == "run":
        if args.threads < 1:
            raise ValueError("--threads must be positive")
        if not args.trust_local_checkpoints:
            raise ValueError("loading local checkpoints requires explicit trust (--trust-local-checkpoints)")
        try:
            import torch
        except ImportError:
            raise RuntimeError("executing the timing sessions requires PyTorch (environment/requirements)") from None
        from cross_morse.common import install_signal_handlers
        torch.set_num_threads(args.threads)
        install_signal_handlers()
        return [str(p) for p in tr.run_runtime_scaling(args.output, job=args.job, device=args.device,
                                                        trust=args.trust_local_checkpoints, job_key=args.job_key)]
    return tr.export_runtime_scaling(args.output, accounting=args.accounting, root=args.root, replace=args.replace,
                                     allow_partial=args.allow_partial, hardware=args.hardware, period=args.period,
                                     command=args.argv)


def _anchors(args):
    from cm_experiments import timing_records as tr
    if args.action == "plan":
        return tr.plan_timing_anchors(args.parent, args.output, scaling_output=args.scaling_output,
                                      accounting=args.accounting, job_map=args.job_map,
                                      accounting_method=args.accounting_method,
                                      forward_equivalents=args.forward_equivalents)
    if args.action == "run":
        return [str(p) for p in tr.run_timing_anchors(args.output, job=args.job)]
    return tr.export_timing_anchors(args.output, root=args.root, replace=args.replace, allow_partial=args.allow_partial,
                                    hardware=args.hardware, period=args.period, command=args.argv)


def _grid_scores(args):
    from cm_experiments import timing_records as tr
    return tr.export_ensemble_grid_scores(args.grid_output, root=args.root, replace=args.replace,
                                          allow_partial=args.allow_partial, hardware=args.hardware,
                                          period=args.period, command=args.argv)


def _export_options(p):
    p.add_argument("--root", type=Path, default=None, help="records root (default records/)")
    p.add_argument("--replace", action="store_true", help="rebuild an existing set")
    p.add_argument("--allow-partial", action="store_true", help="incomplete measurement (never into records/)")
    p.add_argument("--hardware", default=None, help="default: from the jobs' environment records")
    p.add_argument("--period", default=None, help="default: from the recorded execution timestamps")


def parser():
    from cm_experiments.timing_records import (DEFAULT_FRACTIONS, DEFAULT_GRAPH_QUANTILES, DEFAULT_REPETITIONS,
                                               DEFAULT_WARMUP, FORWARD_EQUIVALENTS)
    top = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    records = top.add_subparsers(dest="record", required=True)

    actions = records.add_parser("scaling", help="E17 runtime and memory scaling (records/runtime_scaling)") \
        .add_subparsers(dest="action", required=True)
    p = actions.add_parser("plan", help="register the timed instances of sealed parent runs")
    p.add_argument("--parent", type=Path, action="append", required=True,
                   help="sealed lifecycle run or Roman-empire campaign directory (repeatable, one per dataset)")
    p.add_argument("--output", type=Path, required=True, help="new experiment directory outside the checkout")
    p.add_argument("--fractions", type=float, nargs="+", default=None,
                   help=f"node fractions of the breadth-first subgraphs (default {list(DEFAULT_FRACTIONS)})")
    p.add_argument("--graph-quantiles", type=float, nargs="+", default=None,
                   help=f"edge-count quantiles of a collection's timed test graphs (default {list(DEFAULT_GRAPH_QUANTILES)})")
    p.add_argument("--subgraph-seed", type=int, default=None, help="breadth-first root seed (default: the run's seed)")
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help="E17 warm-up repeats per instance")
    p.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS, help="E17 measured repeats per instance")
    p = actions.add_parser("run", help="time the planned instances with the released runtime")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--job", type=int, default=None, help="time one instance of the plan (default: all)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--job-key", default=None,
                   help="sacct key Cluster|JobIDRaw|Submit|Start of this allocation (default: the Slurm job ID, "
                        "bound to the sacct export at export)")
    p.add_argument("--trust-local-checkpoints", action="store_true",
                   help="authorize loading the parent's local checkpoint files (pickle)")
    p = actions.add_parser("export", help="write records/runtime_scaling from the completed jobs")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--accounting", type=Path, action="append", default=[],
                   help="original sacct export (pipe-delimited) holding the timing allocations (repeatable)")
    _export_options(p)

    actions = records.add_parser("anchors", help="E17 timing anchors (records/timing_anchors)") \
        .add_subparsers(dest="action", required=True)
    p = actions.add_parser("plan", help="declare the parents, the scaling experiment and the accounting inputs")
    p.add_argument("--parent", type=Path, action="append", required=True,
                   help="sealed lifecycle run or Roman-empire campaign directory (repeatable, one per dataset)")
    p.add_argument("--output", type=Path, required=True, help="new experiment directory outside the checkout")
    p.add_argument("--scaling-output", type=Path, default=None,
                   help="runtime_scaling experiment whose nominal timing rows supply the throughput fit")
    p.add_argument("--accounting", type=Path, default=None, help="original sacct export of the plant training")
    p.add_argument("--job-map", type=Path, default=None,
                   help="job map (job_key,dataset,method,phase,seed,allocation_fraction) of cm_campaigns.accounting")
    p.add_argument("--accounting-method", default=None, help="job-map method label of the plant training")
    p.add_argument("--forward-equivalents", type=float, nargs=2, default=list(FORWARD_EQUIVALENTS),
                   metavar=("LOW", "HIGH"), help="forward executions per training input and epoch (bracket)")
    p = actions.add_parser("run", help="compute the anchors (torch-free)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--job", type=int, default=None, help="one dataset of the plan (default: all)")
    p = actions.add_parser("export", help="write records/timing_anchors")
    p.add_argument("--output", type=Path, required=True)
    _export_options(p)

    actions = records.add_parser("grid-scores", help="E14 official scores of the ensemble grid "
                                 "(records/ensemble_grid_scores)").add_subparsers(dest="action", required=True)
    p = actions.add_parser("export", help="recompute the official scores from a completed ensemble-grid output")
    p.add_argument("--grid-output", type=Path, required=True,
                   help="output directory of scripts/run_experiment.py ensemble-grid (plan.json and jobs/)")
    _export_options(p)
    return top


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    args.argv = ["scripts/run_timing_records.py", *(argv if argv is not None else sys.argv[1:])]
    try:
        result = {"scaling": _scaling, "anchors": _anchors, "grid-scores": _grid_scores}[args.record](args)
    except InterruptedError as exc:
        print(f"[checkpoint] {exc}; rerun the same command to resume", file=sys.stderr)
        return 75
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"record": args.record, "action": args.action,
                      "result": result if isinstance(result, (dict, list)) else str(result)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
