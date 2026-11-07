#!/usr/bin/env python3
"""Plan, run and export the trajectory-tube audit and the depth diagnostics on sealed lifecycle runs.

    python -B scripts/run_trajectory_audit.py tube  plan|run|export ...   -> records/tube_audit
                                                                            (Table S22, Fig. S17e, main Fig. 2a)
    python -B scripts/run_trajectory_audit.py depth plan|run|export ...   -> records/depth_diagnostics (Fig. S14)

plan    reads sealed parent runs (Roman-empire: the campaign directory of its split runs; one --parent per
        dataset) and writes <output>/plan.json with every parent's identity and the declared settings;
        nothing is executed.
run     executes the plan with the released runtime, one job per parent run or, with --chunk-size at planning,
        per slice of a run's trajectories or graphs (--job K for array jobs); completed work is skipped (the
        tube audit resumes after its last completed trajectory).  Needs --trust-local-checkpoints.
export  writes the record set (summary.csv, raw/, code/, provenance.json, MANIFEST.json) into --root
        (default records/), traced value by value by scripts/records.py.  Synthetic or partial sets can only
        be written to another root.

Example (tube audit of the sealed arxiv run, then the depth diagnostics with arxiv as reference plant):

    python -B scripts/run_trajectory_audit.py tube plan --parent ../cm-runs/arxiv-full --output ../cm-tube
    python -B scripts/run_trajectory_audit.py tube run --output ../cm-tube --device cuda --trust-local-checkpoints
    python -B scripts/run_trajectory_audit.py tube export --output ../cm-tube
    python -B scripts/run_trajectory_audit.py depth plan --parent ../cm-runs/arxiv-full --reference ogbn-arxiv \
        --output ../cm-depth
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AUDITS = ("tube", "depth")


def _plan(args):
    from cm_experiments import trajectory_audit as audit
    if args.audit == "tube":
        return audit.plan_tube_audit(args.parent, args.output, policy=args.policy, max_trajectories=args.max_trajectories,
                                     input_relative=args.input_relative, edge_relative=args.edge_relative,
                                     precision=args.precision, probes=args.probes, probe_relative=args.probe_relative,
                                     chunk_size=args.chunk_size)
    return audit.plan_depth_diagnostics(args.parent, args.output, reference=args.reference, policy=args.policy,
                                        role=args.role, max_inputs=args.max_inputs, reps=args.reps,
                                        chunk_size=args.chunk_size)


def _run(args):
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    if not args.trust_local_checkpoints:
        raise ValueError("loading local checkpoints requires explicit trust (--trust-local-checkpoints)")
    try:
        import torch
    except ImportError:
        raise RuntimeError("executing an audit requires PyTorch (environment/requirements)") from None
    from cross_morse.common import install_signal_handlers
    from cm_experiments import trajectory_audit as audit
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False      # fp32 audits: no TF32 rounding in dense products
    torch.backends.cudnn.allow_tf32 = False
    install_signal_handlers()
    execute = audit.run_tube_audit if args.audit == "tube" else audit.run_depth_diagnostics
    return [str(p) for p in execute(args.output, job=args.job, device=args.device, trust=True)]


def _export(args):
    from cm_experiments import trajectory_audit as audit
    export = audit.export_tube_audit if args.audit == "tube" else audit.export_depth_diagnostics
    return export(args.output, root=args.root, replace=args.replace, allow_partial=args.allow_partial,
                  hardware=args.hardware, period=args.period, command=args.argv)


def parser():
    top = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    audits = top.add_subparsers(dest="audit", required=True)
    for name in AUDITS:
        actions = audits.add_parser(name).add_subparsers(dest="action", required=True)
        p = actions.add_parser("plan", help="write <output>/plan.json from sealed parent runs")
        p.add_argument("--parent", type=Path, action="append", required=True,
                       help="sealed lifecycle run, or a Roman-empire campaign directory (repeatable, one per dataset)")
        p.add_argument("--output", type=Path, required=True, help="new experiment directory outside the checkout")
        p.add_argument("--policy", choices=("deployed", "confirmed", "provisional"), default="deployed",
                       help="the rule of each run: q_D (default), q_M or the provisional selection")
        p.add_argument("--chunk-size", type=int, default=0,
                       help=("audited trajectories per job" if name == "tube" else "graphs of an independent-graph "
                             "role per job") + " (one scheduler array task each; 0: one job per run)")
        if name == "tube":
            p.add_argument("--max-trajectories", type=int, default=64,
                           help="audited D_explore trajectories per run (seeded subsample of the assessment "
                                "partition; 0: all)")
            p.add_argument("--input-relative", type=float, default=None,
                           help="relative row disturbance of the input features (default: registered bf16 relative "
                                "state tolerance of Table S5c)")
            p.add_argument("--edge-relative", type=float, default=None,
                           help="relative edge-weight disturbance in [0, 1) (default: the same registered tolerance)")
            p.add_argument("--precision", choices=("fp32", "registered"), default="fp32",
                           help="execution precision of both runs (default fp32; registered: the run's own)")
            p.add_argument("--probes", type=int, default=4,
                           help="random secant directions per native designated-site block (plus one coherent)")
            p.add_argument("--probe-relative", type=float, default=1e-3,
                           help="secant radius relative to the predicted state's largest row norm")
        else:
            p.add_argument("--reference", default=None,
                           help="dataset of the reference plant for source alignment (default ogbn-arxiv if planned)")
            p.add_argument("--role", choices=("fit_val", "compare"), default="fit_val",
                           help="label role of the truncated-policy losses (default fit_val, D_fit validation)")
            p.add_argument("--max-inputs", type=int, default=0,
                           help="software check only: seeded subsample of an independent-graph role (0: all)")
            p.add_argument("--reps", type=int, default=None,
                           help="bootstrap replicates of the deterioration interval (default: the run's registered "
                                "selection count)")
        p = actions.add_parser("run", help="execute the plan with the released runtime")
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--job", type=int, default=None, help="execute one job of the plan (default: all)")
        p.add_argument("--device", default="cuda")
        p.add_argument("--threads", type=int, default=16)
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
    args.argv = ["scripts/run_trajectory_audit.py", *(argv if argv is not None else sys.argv[1:])]
    try:
        result = {"plan": _plan, "run": _run, "export": _export}[args.action](args)
    except InterruptedError as exc:
        print(f"[checkpoint] {exc}; rerun the same command to resume", file=sys.stderr)
        return 75
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"audit": args.audit, "action": args.action,
                      "result": result if isinstance(result, (dict, list)) else str(result)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
