#!/usr/bin/env python3
"""Plan, run and export the record-set experiments that the eight-stage lifecycle does not produce.

    python -B scripts/run_experiment.py inserted-site  plan|run|export ...   E13 -> records/inserted_site (S45, Fig. S18a)
    python -B scripts/run_experiment.py ensemble-grid  plan|run|export ...   E14 -> records/ensemble_grid (S26, S26b)
    python -B scripts/run_experiment.py cross-backbone plan|run|export ...   S25 -> records/cross_backbone
    python -B scripts/run_experiment.py factorial      plan|run|export ...   S16/S16b -> records/factorials

plan    reads sealed parent runs (Roman-empire: the campaign directory of ten split runs) and writes
        <output>/plan.json with every parent's identity; nothing is executed.
run     executes the plan with the released runtime (one job with --job K, e.g. a Slurm array task);
        completed jobs are skipped, so an interrupted run resumes.  Loading local checkpoints needs
        --trust-local-checkpoints.  cross-backbone runs whole lifecycles (needs --data-root).
export  writes the record set (summary.csv, raw/, code/, provenance.json, MANIFEST.json) into --root
        (default records/), traced value by value by scripts/records.py.  A synthetic or partial set
        can only be written to another root.

Example (E14 on the sealed arxiv run):

    python -B scripts/run_experiment.py ensemble-grid plan --parent ../cm-runs/arxiv-full --output ../cm-e14
    python -B scripts/run_experiment.py ensemble-grid run --output ../cm-e14 --device cuda --trust-local-checkpoints
    python -B scripts/run_experiment.py ensemble-grid export --output ../cm-e14
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPERIMENTS = ("inserted-site", "ensemble-grid", "cross-backbone", "factorial")


def _json_file(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else None


def _plan(args):
    if args.experiment == "inserted-site":
        from cm_experiments.attribution import plan_attribution
        return plan_attribution(args.parent, args.output, role=args.role or "compare", reps=args.reps,
                                allow_fixture=args.allow_fixture)
    if args.experiment == "ensemble-grid":
        from cm_experiments.ensemble_grid import plan_ensemble_grid
        return plan_ensemble_grid(args.parent, args.output, grid=args.grid, extra_seeds=args.extra_seeds,
                                  role=args.role or "test", reps=args.reps)
    if args.experiment == "cross-backbone":
        from cm_experiments.backbones import plan_cross_backbone
        rows = [tuple(r.split("|", 1)) for r in args.row] if args.row else None
        return plan_cross_backbone(args.parent, args.output, rows=rows, patches=_json_file(args.patches),
                                   rerun_model_a=args.rerun_model_a, reps=args.reps)
    from cm_experiments.factorial import plan_factorial
    return plan_factorial(args.parent, args.output, role=args.role or "test", reps=args.reps, warmup=args.warmup,
                          max_graphs=args.max_graphs, allow_fixture=args.allow_fixture,
                          switches=_json_file(args.switches))


def _run(args):
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    if args.experiment == "cross-backbone":
        from cm_experiments.backbones import run_cross_backbone
        return run_cross_backbone(args.output, job=args.job, data_root=args.data_root, device=args.device,
                                  threads=args.threads, python=args.python)
    if not args.trust_local_checkpoints:
        raise ValueError("loading local checkpoints requires explicit trust (--trust-local-checkpoints)")
    try:
        import torch
    except ImportError:
        raise RuntimeError("executing an experiment requires PyTorch (environment/requirements)") from None
    from cross_morse.common import install_signal_handlers
    torch.set_num_threads(args.threads)
    install_signal_handlers()
    if args.experiment == "inserted-site":
        from cm_experiments.attribution import run_attribution as execute
    elif args.experiment == "ensemble-grid":
        from cm_experiments.ensemble_grid import run_ensemble_grid as execute
    else:
        from cm_experiments.factorial import run_factorial as execute
    return [str(p) for p in execute(args.output, job=args.job, device=args.device, trust=args.trust_local_checkpoints)]


def _export(args):
    from cm_experiments.attribution import export_attribution
    from cm_experiments.backbones import export_cross_backbone
    from cm_experiments.ensemble_grid import export_ensemble_grid
    from cm_experiments.factorial import export_factorial
    export = {"inserted-site": export_attribution, "ensemble-grid": export_ensemble_grid,
              "cross-backbone": export_cross_backbone, "factorial": export_factorial}[args.experiment]
    return export(args.output, root=args.root, replace=args.replace, allow_partial=args.allow_partial,
                  hardware=args.hardware, period=args.period, command=args.argv)


def parser():
    top = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    experiments = top.add_subparsers(dest="experiment", required=True)
    for name in EXPERIMENTS:
        actions = experiments.add_parser(name).add_subparsers(dest="action", required=True)
        p = actions.add_parser("plan", help="write <output>/plan.json from sealed parent runs")
        p.add_argument("--parent", type=Path, action="append", required=True,
                       help="sealed lifecycle run, or a Roman-empire campaign directory (repeatable, one per dataset)")
        p.add_argument("--output", type=Path, required=True, help="new experiment directory outside the checkout")
        p.add_argument("--reps", type=int, default=None, help="bootstrap replicates (default: the run's registered count)")
        if name != "cross-backbone":
            p.add_argument("--role", choices=("compare", "test"), default=None,
                           help="evaluation role (default: compare for inserted-site, test otherwise)")
        if name in ("inserted-site", "factorial"):
            p.add_argument("--allow-fixture", action="store_true",
                           help="synthetic smoke parents only: use the registered feedback candidate when q_M is open loop")
        if name == "ensemble-grid":
            p.add_argument("--grid", type=int, nargs="+", default=None,
                           help="ensemble sizes (default: the registered E14 grid 1 2 4 8 12 16 20)")
            p.add_argument("--extra-seeds", type=int, nargs="*", default=None,
                           help="seeds of the additional checkpoints (default: consecutive after the parent's)")
        if name == "cross-backbone":
            p.add_argument("--row", action="append", default=[], help="'regime|model' (repeatable; default: all six)")
            p.add_argument("--patches", type=Path, default=None,
                           help="JSON {'regime|model': {json-pointer: value}} replacing the default substitutions")
            p.add_argument("--rerun-model-a", action="store_true",
                           help="rerun the paper's plant of each regime instead of reusing the sealed parent")
        if name == "factorial":
            p.add_argument("--warmup", type=int, default=1, help="discarded warm-up executions per cell")
            p.add_argument("--max-graphs", type=int, default=0, help="software check only: truncate the role")
            p.add_argument("--switches", type=Path, default=None,
                           help="JSON {factor: [switch, [off, on]]} overriding the factor-to-switch table")
        p = actions.add_parser("run", help="execute the plan with the released runtime")
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--job", type=int, default=None, help="execute one job of the plan (default: all)")
        p.add_argument("--device", default="cuda")
        p.add_argument("--threads", type=int, default=16)
        if name == "cross-backbone":
            p.add_argument("--data-root", type=Path, default=None, help="dataset cache of run_cross_morse.py")
            p.add_argument("--python", default=sys.executable, help="interpreter for the lifecycle subprocesses")
        else:
            p.add_argument("--trust-local-checkpoints", action="store_true",
                           help="authorize loading the parent's local checkpoint files (pickle)")
        p = actions.add_parser("export", help="write the record set from the completed jobs")
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--root", type=Path, default=None, help="records root (default records/)")
        p.add_argument("--replace", action="store_true", help="rebuild an existing set")
        p.add_argument("--allow-partial", action="store_true", help="incomplete coverage (never into records/)")
        p.add_argument("--hardware", default=None, help="default: from the jobs' environment records")
        p.add_argument("--period", default=None, help="default: from the jobs' execution timestamps")
    return top


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    args.argv = ["scripts/run_experiment.py", *(argv if argv is not None else sys.argv[1:])]
    try:
        result = {"plan": _plan, "run": _run, "export": _export}[args.action](args)
    except InterruptedError as exc:
        print(f"[checkpoint] {exc}; rerun the same command to resume", file=sys.stderr)
        return 75
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"experiment": args.experiment, "action": args.action,
                      "result": result if isinstance(result, (dict, list)) else str(result)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
