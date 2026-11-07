#!/usr/bin/env python3
"""Plan, run and export the decision-log audits of the deployed rules (one instrumented execution, three sets).

    python -B scripts/run_decision_audit.py plan --parent RUN [--parent RUN ...] --output DIR [options]
    python -B scripts/run_decision_audit.py run --output DIR [--job K] --device cuda --trust-local-checkpoints
    python -B scripts/run_decision_audit.py export --output DIR [--set NAME ...] [--root records]

Record sets: action_gap_audit (Supplementary Table S21, Figure S17f; Tables S33-S34, Figure S7),
decision_traces (main Figure 3c-d) and reliability_diagnostics (Supplementary Figure S16; E7, E8).

plan    reads sealed parent runs (Roman-empire: the campaign directory of ten split runs) and writes
        <output>/plan.json with every parent's identity, the deployed rule, the registered perturbation, the
        near-zero threshold, the input-gate settings and the declared audit subset; nothing is executed.
run     executes the plan with the released runtime, one job per lifecycle run (--job K for an array task);
        each job has three resumable phases (h-step radius calibration and observer variants on D_explore,
        then the instrumented execution on the evaluation role) and writes audit.json when complete.
export  recomputes every value from the jobs' raw files and writes each requested set (default: all three)
        into --root (default records/), traced value by value by scripts/records.py. A synthetic, fixture or
        partial set can only be written to another root.

Example (the seven sealed runs, the negative-controls reference on PascalVOC-SP):

    python -B scripts/run_decision_audit.py plan --output ../cm-decisions --stress-dataset PascalVOC-SP \\
        --parent ../cm-runs/arxiv-full --parent ../cm-runs/proteins-full --parent ../cm-runs/products-full \\
        --parent ../cm-runs/roman-empire --parent ../cm-runs/pascalvoc-full --parent ../cm-runs/coco-full \\
        --parent ../cm-runs/peptides-full
    python -B scripts/run_decision_audit.py run --output ../cm-decisions --device cuda --trust-local-checkpoints
    python -B scripts/run_decision_audit.py export --output ../cm-decisions
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SETS = ("action_gap_audit", "decision_traces", "reliability_diagnostics")


def _gate_from(values):
    out = {}
    for value in values or []:
        key, sep, path = value.partition("=")
        if not sep or not key or not path:
            raise ValueError("--gate-from takes DATASET=DIR or DATASET/RUN=DIR (a sealed input_gate run)")
        out[key] = Path(path)
    return out


def _plan(args):
    from cm_experiments.decision_audit import plan_decision_audit
    return plan_decision_audit(args.parent, args.output, role=args.role, reps=args.reps, seed_offset=args.seed_offset,
                               perturbation_factor=args.perturbation_factor,
                               near_zero_threshold=args.near_zero_threshold, audit_inputs=args.audit_inputs,
                               audit_seed=args.audit_seed, gate_from=_gate_from(args.gate_from),
                               gate_alpha=args.gate_alpha, gate_profile=args.gate_profile, ema_weight=args.ema_weight,
                               stress_dataset=args.stress_dataset,
                               calibration_trajectories=args.calibration_trajectories,
                               observer_study=not args.no_observer_study, fixture_route=args.fixture_route,
                               provisional=args.provisional)


def _run(args):
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    if not args.trust_local_checkpoints:
        raise ValueError("loading local checkpoints requires explicit trust (--trust-local-checkpoints)")
    try:
        import torch
    except ImportError:
        raise RuntimeError("executing the audit requires PyTorch (environment/requirements)") from None
    from cross_morse.common import install_signal_handlers
    from cm_experiments.decision_audit import run_decision_audit
    torch.set_num_threads(args.threads)
    install_signal_handlers()
    return [str(p) for p in run_decision_audit(args.output, job=args.job, device=args.device,
                                               trust=args.trust_local_checkpoints)]


def _export(args):
    from cm_experiments.decision_audit import export_decision_audit
    return [str(p) for p in export_decision_audit(args.output, sets=tuple(args.set or SETS), root=args.root,
                                                  replace=args.replace, allow_partial=args.allow_partial,
                                                  hardware=args.hardware, period=args.period, command=args.argv,
                                                  stress_dataset=args.stress_dataset)]


def parser():
    top = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    actions = top.add_subparsers(dest="action", required=True)
    p = actions.add_parser("plan", help="write <output>/plan.json from sealed parent runs")
    p.add_argument("--parent", type=Path, action="append", required=True,
                   help="sealed lifecycle run, or a Roman-empire campaign directory (repeatable, one per dataset)")
    p.add_argument("--output", type=Path, required=True, help="new experiment directory outside the checkout")
    p.add_argument("--role", choices=("test", "compare"), default="test",
                   help="evaluation role (decision_traces requires the test role)")
    p.add_argument("--reps", type=int, default=None, help="bootstrap replicates (default: the registered count)")
    p.add_argument("--seed-offset", type=int, default=13, help="perturbation seed = run seed + offset")
    p.add_argument("--perturbation-factor", type=float, default=1.0,
                   help="multiple of the observer's registered measurement-noise draw (registered: 1)")
    p.add_argument("--near-zero-threshold", type=float, default=None,
                   help="near-zero gap threshold in nats (default: the certificate floor tau)")
    p.add_argument("--audit-inputs", type=int, default=0,
                   help="seeded subset of evaluation graphs for the per-site analysis (0 = all)")
    p.add_argument("--audit-seed", type=int, default=0)
    p.add_argument("--gate-from", action="append", default=[], metavar="DATASET=DIR",
                   help="reuse the frozen g_input of a sealed input_gate run (default: refit, deterministic)")
    p.add_argument("--gate-alpha", type=float, default=.05)
    p.add_argument("--gate-profile", default="full")
    p.add_argument("--ema-weight", type=float, default=.5, help="EMA observer weight on the new measurement")
    p.add_argument("--stress-dataset", default=None,
                   help="dataset of the negative-controls record (clean reference of Figure S16d)")
    p.add_argument("--calibration-trajectories", type=int, default=0,
                   help="seeded subset of radius-fit trajectories for the h-step radii (0 = all)")
    p.add_argument("--no-observer-study", action="store_true", help="skip the E8 observer variants")
    p.add_argument("--fixture-route", choices=("one_step", "radius", "mpc_rh"), default=None,
                   help="synthetic smoke parents only: audit a registered feedback candidate (software check)")
    p.add_argument("--provisional", action="append", default=[], metavar="DATASET",
                   help="audit the provisional selection q_s of Eq. (19) instead of the deployed rule q_D on this "
                        "dataset (repeatable; named in every output)")
    p = actions.add_parser("run", help="execute the plan with the released runtime")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--job", type=int, default=None, help="execute one job of the plan (default: all)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--trust-local-checkpoints", action="store_true",
                   help="authorize loading the parent's local checkpoint files (pickle)")
    p = actions.add_parser("export", help="write the record sets from the completed jobs")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--set", action="append", choices=SETS, default=None, help="record set (repeatable; default all)")
    p.add_argument("--root", type=Path, default=None, help="records root (default records/)")
    p.add_argument("--replace", action="store_true", help="rebuild an existing set")
    p.add_argument("--allow-partial", action="store_true", help="incomplete coverage (never into records/)")
    p.add_argument("--hardware", default=None, help="default: from the jobs' environment records")
    p.add_argument("--period", default=None, help="default: from the jobs' execution timestamps")
    p.add_argument("--stress-dataset", default=None, help="override the planned stress dataset")
    return top


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    args.argv = ["scripts/run_decision_audit.py", *(argv if argv is not None else sys.argv[1:])]
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
