#!/usr/bin/env python3
"""Plan, run and export the internal rows of Supplementary Tables S15 and S15b (record set baselines; E12, E17).

    python -B scripts/run_internal_baselines.py plan --parent RUN ... --output DIR
    python -B scripts/run_internal_baselines.py run --output DIR [--job K] --device cuda --trust-local-checkpoints
    python -B scripts/run_internal_baselines.py export --output DIR [options of export_records.py baselines]

plan     reads the sealed lifecycle runs (one per dataset; Roman-empire: the campaign directory of its ten split
         runs, one job per split run) and registers, per run, the Table S15 method rows, the configuration of every
         method with the ranking that chose it, the timed inputs and the stage times of the GPU-hours; torch-free.
         A run whose lifecycle did not run on CUDA is refused (--allow-fixture accepts CPU fixtures as a software
         check; their sets are partial)
run      executes each configuration on D_test once per plant checkpoint as a one-member ensemble (official
         score), times q_C and the deployed rule with the E17 TimingSession against the nominal execution, and
         writes the sealed result directory of the run; completed jobs are skipped and an interrupted job resumes its
         scores. Needs PyTorch and --trust-local-checkpoints. Run it inside the GPU allocation whose sacct record is
         exported for the export (the timing rows carry its key)
export   scripts/export_records.py baselines --internal DIR with the other options of that command (--study,
         --checkouts, --timing, --accounting, --root, --hardware, --period, ...): verifies the sealed results, the plan
         identity and the parent identities and writes the baselines set; the timing rows need --accounting

The definitions of the method rows, configurations, scores, timing and GPU-hours are in
cm_experiments/internal_baselines.py and in the plan.

Example (the seven sealed runs of a reproduction suite):

    python -B scripts/run_internal_baselines.py plan --output ../cm-baselines-internal \\
        --parent ../suite/runs/01_arxiv --parent ../suite/runs/04_roman_empire ...
    python -B scripts/run_internal_baselines.py run --output ../cm-baselines-internal --job 0 --device cuda \\
        --trust-local-checkpoints
    bash hpc/capture_arrhenius_accounting.sh <job ids> ../acct
    python -B scripts/export_records.py baselines --internal ../cm-baselines-internal --study ../studies/arxiv_AMP ... \\
        --checkouts ../baselines --accounting ../acct/sacct_original.psv --hardware "..." --period "..."
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _plan(args):
    from cm_experiments import internal_baselines as ib
    return ib.plan_internal_baselines(args.parent, args.output, warmup=args.warmup, repetitions=args.repetitions,
                                      graph_quantiles=args.graph_quantiles, allow_fixture=args.allow_fixture)


def _run(args):
    from cm_experiments import internal_baselines as ib
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    if not args.trust_local_checkpoints:
        raise ValueError("loading local checkpoints requires explicit trust (--trust-local-checkpoints)")
    try:
        import torch
    except ImportError:
        raise RuntimeError("executing the configurations requires PyTorch (environment/requirements)") from None
    from cross_morse.common import install_signal_handlers
    torch.set_num_threads(args.threads)
    install_signal_handlers()
    return [str(p) for p in ib.run_internal_baselines(args.output, job=args.job, device=args.device,
                                                       trust=args.trust_local_checkpoints, job_key=args.job_key)]


def _export(args):
    from scripts.export_records import ExportError, export_baselines
    args.internal = [args.output, *args.internal]
    try:
        return export_baselines(args)
    except ExportError as exc:
        raise ValueError(str(exc)) from None


def parser():
    from cm_experiments.internal_baselines import DEFAULT_GRAPH_QUANTILES, DEFAULT_REPETITIONS, DEFAULT_WARMUP
    from scripts.export_records import baselines_arguments
    top = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    actions = top.add_subparsers(dest="action", required=True)
    p = actions.add_parser("plan", help="register the method rows, configurations and timed inputs of sealed runs")
    p.add_argument("--parent", type=Path, action="append", required=True,
                   help="sealed lifecycle run or Roman-empire campaign directory (repeatable, one per dataset)")
    p.add_argument("--output", type=Path, required=True, help="new experiment directory outside the checkout")
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help="E17 warm-up repeats per session")
    p.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS, help="E17 measured repeats per session")
    p.add_argument("--graph-quantiles", type=float, nargs="+", default=None,
                   help=f"edge-count quantiles of a collection's timed test graphs (default {list(DEFAULT_GRAPH_QUANTILES)})")
    p.add_argument("--allow-fixture", action="store_true",
                   help="software check: accept runs whose lifecycle ran on CPU (their GPU-hours are not GPU-hours)")
    p = actions.add_parser("run", help="execute the planned configurations and timing sessions")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--job", type=int, default=None, help="one run of the plan (default: all)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--job-key", default=None,
                   help="sacct key Cluster|JobIDRaw|Submit|Start of this allocation (default: the Slurm job ID, bound to "
                        "the sacct export at export)")
    p.add_argument("--trust-local-checkpoints", action="store_true",
                   help="authorize loading the parent's local checkpoint files (pickle)")
    p = actions.add_parser("export", help="write the baselines set with these rows (scripts/export_records.py baselines)")
    p.add_argument("--output", type=Path, required=True, help="the runner output (passed as --internal)")
    baselines_arguments(p)
    return top


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    args.argv = ["scripts/run_internal_baselines.py", *(argv if argv is not None else sys.argv[1:])]
    try:
        result = {"plan": _plan, "run": _run, "export": _export}[args.action](args)
    except InterruptedError as exc:
        print(f"[checkpoint] {exc}; rerun the same command to resume", file=sys.stderr)
        return 75
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"action": args.action, "result": result if isinstance(result, (dict, list)) else str(result)},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
