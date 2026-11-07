#!/usr/bin/env python3
"""Variant lifecycles of the module ablations (S17; E7, E8), the E18 policy value and the separately
fitted factorial cells (S16, S16b): plan, run and export (cm_experiments/ablation_lifecycle.py).

    python -B scripts/run_ablation.py modules       plan|run|collect|export ...   S17 -> records/module_ablations
    python -B scripts/run_ablation.py e18-policy    plan|run|collect|export ...   E18 -> records/spectral_policy_value
    python -B scripts/run_ablation.py factorial-fit plan|run|export ...           S16/S16b -> records/factorials

plan     reads sealed parent runs (a Roman-empire parent is its campaign of split runs) and writes
         <output>/plan.json and one configuration per variant lifecycle (the parent's configuration plus its
         ``ablation_variant``); factorial-fit also writes <output>/factorial/plan.json (fitted mode: one lifecycle
         per distinct set of cell operations -- O = 0 memoryless observer, A = 0 no transport actions A1-A7,
         X = 0 risk-controlled execution off -- and the read-only observer's stopping rule in the O = 1 cells).
         modules: the observer arms are raw, ema, global and history_mlp (the E8 history-MLP observer, window
         --history-window, default two observer-update periods, and width --history-width, default 128).
run      jobs 0..n-1 are the variant lifecycles: the parent's sealed prepare/train outputs are copied and
         sealed under the variant, then explore -> fit (-> select -> confirm) run as
         ``python -B -m cm_experiments.ablation_lifecycle <run_cross_morse.py arguments>`` (variant installed);
         then the measurement jobs (modules, e18: the audited rule on the held-out role with every arm;
         factorial-fit: the cells of cm_experiments.factorial, job indices continuing after the lifecycles).
         Completed jobs are skipped; --job K runs one job (a Slurm array task).  Needs PyTorch and
         --trust-local-checkpoints.
collect  writes <output>/results/<record set>.json from the completed measurement jobs.
export   modules: the ten-module record set -- the sealed-contrast modules of cm_experiments/sweeps.py on the
         same parents and the seven lifecycle-measured ones -- through a new sweep directory (--sweep) and
         scripts/export_sweep_records.py;  e18-policy: records/spectral_policy_value (its contract must be
         registered in records/specs.json: cm_experiments.ablation_lifecycle.E18_CONTRACT);  factorial-fit:
         records/factorials from the fitted-mode factorial jobs.

Example (S17 on the sealed arxiv run, comparator branch from its input-gate and negative-control runs):

    python -B scripts/run_ablation.py modules plan --parent ../cm-runs/arxiv-full --output ../cm-s17 \\
        --input-gate ../cm-gate-arxiv --negative-controls ../cm-controls-arxiv
    python -B scripts/run_ablation.py modules run --output ../cm-s17 --device cuda --trust-local-checkpoints
    python -B scripts/run_ablation.py modules export --output ../cm-s17 --sweep ../cm-s17-sweep \\
        --hardware "NVIDIA GH200 96 GB" --period 2026-11-08/2026-11-20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STUDIES = ("modules", "e18-policy", "factorial-fit")


def _plan(args):
    from cm_experiments import ablation_lifecycle as ablation
    if args.study == "factorial-fit":
        return ablation.plan_factorial_fit(args.parent, args.output, role=args.role or "test", complete=args.complete,
                                           through=args.through, reps=args.reps, warmup=args.warmup,
                                           max_graphs=args.max_graphs, allow_fixture=args.allow_fixture,
                                           allow_source_change=args.allow_source_change)
    return ablation.plan_modules(args.parent, args.output, modules=getattr(args, "module", None) or None,
                                 role=args.role or "compare", complete=args.complete, through=args.through,
                                 reps=args.reps, max_graphs=args.max_graphs, ema_weight=getattr(args, "ema_weight", .5),
                                 scramble_seed=getattr(args, "scramble_seed", ablation.DEFAULT_SCRAMBLE_SEED),
                                 scramble_units=getattr(args, "scramble_units", 1024),
                                 input_gate=getattr(args, "input_gate", ()),
                                 negative_controls=getattr(args, "negative_controls", ()),
                                 allow_fixture=args.allow_fixture, allow_source_change=args.allow_source_change,
                                 study="modules" if args.study == "modules" else "e18",
                                 history_window_length=getattr(args, "history_window", None),
                                 history_width=getattr(args, "history_width", ablation.HISTORY_WIDTH))


def _run(args):
    from cm_experiments.ablation_lifecycle import run_study
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    if not args.trust_local_checkpoints:
        raise ValueError("loading local checkpoints requires explicit trust (--trust-local-checkpoints)")
    try:
        import torch
    except ImportError:
        raise RuntimeError("executing a variant lifecycle requires PyTorch (environment/requirements)") from None
    from cross_morse.common import install_signal_handlers
    torch.set_num_threads(args.threads)
    install_signal_handlers()
    return run_study(args.output, job=args.job, data_root=args.data_root, device=args.device, threads=args.threads,
                     python=args.python, trust=True)


def _collect(args):
    from cm_experiments.ablation_lifecycle import collect_study
    return collect_study(args.output)


def _export(args):
    from cm_experiments import ablation_lifecycle as ablation
    if args.study == "modules":
        if not (args.sweep and args.hardware and args.period):
            raise ValueError("modules export needs --sweep (a new directory), --hardware and --period")
        return ablation.export_modules(args.output, args.sweep, root=args.root, hardware=args.hardware,
                                       period=args.period, replace=args.replace, parents=args.parent or None,
                                       argv=args.argv)
    export = ablation.export_e18 if args.study == "e18-policy" else ablation.export_factorial_fit
    return export(args.output, root=args.root, replace=args.replace, allow_partial=args.allow_partial,
                  hardware=args.hardware, period=args.period, command=args.argv)


def parser():
    top = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    studies = top.add_subparsers(dest="study", required=True)
    for name in STUDIES:
        actions = studies.add_parser(name).add_subparsers(dest="action", required=True)
        p = actions.add_parser("plan", help="write <output>/plan.json and the variant configurations")
        p.add_argument("--parent", type=Path, action="append", required=True,
                       help="sealed lifecycle run or Roman-empire campaign (repeatable, one per dataset)")
        p.add_argument("--output", type=Path, required=True, help="new experiment directory outside the checkout")
        p.add_argument("--role", choices=("compare", "test"), default=None,
                       help="held-out role (default: compare; factorial-fit: test)")
        p.add_argument("--complete", choices=("replicate", "parent"), default="replicate",
                       help="complete arm: a replicate lifecycle without operation (default) or the parent itself")
        p.add_argument("--through", choices=("fit", "select", "confirm"), default="confirm",
                       help="last re-executed lifecycle stage of every variant (default: confirm)")
        p.add_argument("--reps", type=int, default=None, help="bootstrap replicates (default: the run's registered count)")
        p.add_argument("--max-graphs", type=int, default=0, help="software check only: truncate the role")
        p.add_argument("--allow-fixture", action="store_true",
                       help="synthetic smoke parents only: audit a registered feedback candidate when q_M is open loop")
        p.add_argument("--allow-source-change", action="store_true",
                       help="plan on parents sealed by another runtime source (execution still refuses them)")
        if name == "modules":
            p.add_argument("--module", action="append", default=[], metavar="MODULE[:ARM]",
                           help="lifecycle module or one arm (repeatable; default: all seven arms' modules)")
            p.add_argument("--ema-weight", type=float, default=.5, help="EMA observer weight (declared)")
            p.add_argument("--scramble-seed", type=int, default=20261107, help="seed of the band scramble")
            p.add_argument("--scramble-units", type=int, default=1024, help="donor trajectories in the scramble pool")
            p.add_argument("--input-gate", type=Path, action="append", default=[],
                           help="sealed input_gate run of a parent (comparator branch; repeatable)")
            p.add_argument("--negative-controls", type=Path, action="append", default=[],
                           help="sealed negative_controls run of a parent (comparator branch; repeatable)")
            p.add_argument("--history-window", type=int, default=None,
                           help="history-MLP observer window w (default: 2 x controller.observer_period of each parent)")
            p.add_argument("--history-width", type=int, default=128,
                           help="history-MLP hidden width (default 128, the registered consequence-model width)")
        if name == "factorial-fit":
            p.add_argument("--warmup", type=int, default=1, help="discarded warm-up executions per cell")
        p = actions.add_parser("run", help="execute the variant lifecycles, then the measurement")
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--job", type=int, default=None, help="execute one job (default: all, in order)")
        p.add_argument("--data-root", type=Path, default=None,
                       help="unused by the copied lifecycles (prepare is sealed); accepted for symmetry")
        p.add_argument("--device", default="cuda")
        p.add_argument("--threads", type=int, default=16)
        p.add_argument("--python", default=sys.executable, help="interpreter for the lifecycle subprocesses")
        p.add_argument("--trust-local-checkpoints", action="store_true",
                       help="authorize loading the parent's local checkpoint files (pickle)")
        if name != "factorial-fit":
            p = actions.add_parser("collect", help="write <output>/results/<record set>.json")
            p.add_argument("--output", type=Path, required=True)
        p = actions.add_parser("export", help="write the record set")
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--root", type=Path, default=None, help="records root (default records/)")
        p.add_argument("--replace", action="store_true", help="rebuild an existing set")
        p.add_argument("--hardware", default=None)
        p.add_argument("--period", default=None)
        if name == "modules":
            p.add_argument("--sweep", type=Path, default=None, help="new module_ablations sweep directory")
            p.add_argument("--parent", type=Path, action="append", default=[],
                           help="parents of the sealed-contrast modules (default: the study's parents)")
        else:
            p.add_argument("--allow-partial", action="store_true", help="incomplete coverage (never into records/)")
    return top


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    args.argv = ["scripts/run_ablation.py", *(argv if argv is not None else sys.argv[1:])]
    try:
        result = {"plan": _plan, "run": _run, "collect": _collect, "export": _export}[args.action](args)
    except InterruptedError as exc:
        print(f"[checkpoint] {exc}; rerun the same command to resume", file=sys.stderr)
        return 75
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"study": args.study, "action": args.action,
                      "result": result if isinstance(result, (dict, list)) else str(result)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
