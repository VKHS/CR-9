#!/usr/bin/env python3
"""Plan, run and export the confirmation analyses and the admission-split oracle.

    python -B scripts/run_confirmation_analysis.py mechanism    plan|run|export ...  -> records/mechanism_contrasts (main Fig. 3a-b)
    python -B scripts/run_confirmation_analysis.py confirmation plan|run|export ...  -> records/confirmation_contrasts (S14b)
                                                                                         and records/confirmation_sensitivity (S18b, Fig. S9)
    python -B scripts/run_confirmation_analysis.py oracle       plan|run|export ...  -> records/oracle_dispersion (S30, Fig. S13)

plan    reads sealed, complete parent runs (Roman-empire: the campaign directory of ten split runs) and
        writes <output>/plan.json with every parent's identity; nothing is executed.
            mechanism     every Figure 3a-b cell that neither the joint S14 family nor the deployed
                          rule's confirm/decision.json supplies (or --cell DATASET=CONTRAST)
            confirmation  the PascalVOC-SP, COCO-SP and Peptides-func runs; --campaign identifies the
                          S14 policy pairs and seed population.  Export: the nine J_old rows are the
                          joint campaign's original tests, the graph share comes from each parent's
                          admission moments and Table 1 interval; the executed per-seed validation-split
                          array is deposited as the separate E16 direct estimate
            oracle        one job per (run, registered configuration at the comparator's ensemble size)
run     executes the plan with the released runtime (one job with --job K, e.g. a Slurm array task);
        completed jobs are skipped, so an interrupted run resumes.  Needs --trust-local-checkpoints.
export  writes the record set(s) into --root (default records/), traced value by value by
        scripts/records.py.  A synthetic, truncated or partial set can only be written to another root.

Example (oracle on the sealed PascalVOC-SP run):

    python -B scripts/run_confirmation_analysis.py oracle plan --parent ../cm-runs/pascalvoc_sp-full --output ../cm-oracle
    python -B scripts/run_confirmation_analysis.py oracle run --output ../cm-oracle --job 0 --device cuda --trust-local-checkpoints
    python -B scripts/run_confirmation_analysis.py oracle export --output ../cm-oracle --root ../records-check
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ANALYSES = ("mechanism", "confirmation", "oracle")


def _cells(values):
    if not values:
        return None
    from cm_experiments.confirmation_analysis import FIGURE3
    cells = {}
    for value in values:
        dataset, sep, contrast = value.rpartition("=")
        if not sep or not dataset or contrast not in FIGURE3:
            raise ValueError(f"--cell takes DATASET=CONTRAST with CONTRAST in {sorted(FIGURE3)}: {value!r}")
        cells.setdefault(dataset, []).append(contrast)
    return cells


def _anchors(values):
    out = {}
    for value in values or ():
        dataset, sep, index = value.rpartition("=")
        if not sep or not dataset or not index.isdigit():
            raise ValueError(f"--anchor takes DATASET=REGISTRY_INDEX: {value!r}")
        out[dataset] = int(index)
    return out or None


def _plan(args):
    if args.analysis == "mechanism":
        from cm_experiments.confirmation_analysis import plan_mechanism
        return plan_mechanism(args.parent, args.output, cells=_cells(args.cell), anchors=_anchors(args.anchor),
                              reps=args.reps)
    if args.analysis == "confirmation":
        from cm_experiments.confirmation_analysis import plan_confirmation
        return plan_confirmation(args.parent, args.output, campaign=args.campaign, seeds=args.seeds, reps=args.reps,
                                 allow_fixture=args.allow_fixture)
    from cm_experiments.admission_oracle import plan_oracle
    return plan_oracle(args.parent, args.output, routes=args.routes, max_graphs=args.max_graphs)


def _run(args):
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    if not args.trust_local_checkpoints:
        raise ValueError("loading local checkpoints requires explicit trust (--trust-local-checkpoints)")
    try:
        import torch
    except ImportError:
        raise RuntimeError("executing an analysis requires PyTorch (environment/requirements)") from None
    from cross_morse.common import install_signal_handlers
    torch.set_num_threads(args.threads)
    install_signal_handlers()
    if args.analysis == "mechanism":
        from cm_experiments.confirmation_analysis import run_mechanism as execute
    elif args.analysis == "confirmation":
        from cm_experiments.confirmation_analysis import run_confirmation as execute
    else:
        from cm_experiments.admission_oracle import run_oracle as execute
    return [str(p) for p in execute(args.output, job=args.job, device=args.device, trust=args.trust_local_checkpoints)]


def _export(args):
    options = dict(root=args.root, replace=args.replace, allow_partial=args.allow_partial, hardware=args.hardware,
                   period=args.period, command=args.argv)
    if args.analysis == "mechanism":
        from cm_experiments.confirmation_analysis import export_mechanism
        return export_mechanism(args.output, **options)
    if args.analysis == "confirmation":
        from cm_experiments.confirmation_analysis import export_confirmation
        return {name: str(path) for name, path in
                export_confirmation(args.output, campaign=args.campaign, **options).items()}
    from cm_experiments.admission_oracle import export_oracle
    return export_oracle(args.output, **options)


def parser():
    top = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    analyses = top.add_subparsers(dest="analysis", required=True)
    for name in ANALYSES:
        actions = analyses.add_parser(name).add_subparsers(dest="action", required=True)
        p = actions.add_parser("plan", help="write <output>/plan.json from sealed parent runs")
        p.add_argument("--parent", type=Path, action="append", required=True,
                       help="sealed, complete lifecycle run, or a Roman-empire campaign directory (repeatable)")
        p.add_argument("--output", type=Path, required=True, help="new analysis directory outside the checkout")
        if name in ("mechanism", "confirmation"):
            p.add_argument("--reps", type=int, default=None,
                           help="bootstrap replicates (default: the run's registered count)")
        if name == "mechanism":
            p.add_argument("--cell", action="append", default=[],
                           help="DATASET=CONTRAST (state|rh_minus_ol) replacing the Figure 3 rule for that dataset")
            p.add_argument("--anchor", action="append", default=[],
                           help="DATASET=REGISTRY_INDEX: declare the strongest feedback configuration")
        if name == "confirmation":
            p.add_argument("--campaign", type=Path, default=None,
                           help="joint S14 campaign directory: the test policy pairs and seed population")
            p.add_argument("--seeds", type=int, nargs="+", default=None,
                           help="checkpoint seeds of the two-way array (default: the S14 seed population)")
            p.add_argument("--allow-fixture", action="store_true",
                           help="synthetic smoke parents only: the registered pair when no record identifies it")
        if name == "oracle":
            p.add_argument("--routes", nargs="+", default=None,
                           help="restrict the enumerated family to these routes (q_C is always included)")
            p.add_argument("--max-graphs", type=int, default=0,
                           help="software check only: enumerate the first K admission graphs")
        p = actions.add_parser("run", help="execute the plan with the released runtime")
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--job", type=int, default=None, help="execute one job of the plan (default: all)")
        p.add_argument("--device", default="cuda")
        p.add_argument("--threads", type=int, default=16)
        p.add_argument("--trust-local-checkpoints", action="store_true",
                       help="authorize loading the parent's local checkpoint files (pickle)")
        p = actions.add_parser("export", help="write the record set(s) from the completed jobs")
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--root", type=Path, default=None, help="records root (default records/)")
        p.add_argument("--replace", action="store_true", help="rebuild an existing set")
        p.add_argument("--allow-partial", action="store_true", help="incomplete coverage (never into records/)")
        p.add_argument("--hardware", default=None, help="default: from the jobs' environment records")
        p.add_argument("--period", default=None, help="default: from the jobs' execution timestamps")
        if name == "confirmation":
            p.add_argument("--campaign", type=Path, default=None,
                           help="joint S14 campaign (default: the one given at planning)")
    return top


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    args.argv = ["scripts/run_confirmation_analysis.py", *(argv if argv is not None else sys.argv[1:])]
    try:
        result = {"plan": _plan, "run": _run, "export": _export}[args.action](args)
    except InterruptedError as exc:
        print(f"[checkpoint] {exc}; rerun the same command to resume", file=sys.stderr)
        return 75
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"analysis": args.analysis, "action": args.action,
                      "result": result if isinstance(result, (dict, list)) else str(result)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
