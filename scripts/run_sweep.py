#!/usr/bin/env python3
"""Plan, execute and collect the sensitivity sweeps and module ablations of records/specs.d.

    # S29 / Fig. S5: MPC-RH horizon and beam width on the sealed PascalVOC-SP run
    python -B scripts/run_sweep.py plan --sweep horizon_beam_sweep --parent /abs/runs/pascalvoc_sp --output /abs/sweeps/hb
    python -B scripts/run_sweep.py run --plan /abs/sweeps/hb/plan.json --data-root /abs/datasets --device cuda
    python -B scripts/run_sweep.py collect --plan /abs/sweeps/hb/plan.json

    # S18: support thresholds (variant lifecycle runs), selection margins and clip level (exact replays)
    python -B scripts/run_sweep.py plan --sweep sensitivity_grids --parent /abs/runs/arxiv ... --output /abs/sweeps/grids

    # S17: the module ablations the released runtime supports (sealed outputs only)
    python -B scripts/run_sweep.py plan --sweep module_ablations --parent /abs/runs/arxiv ... --output /abs/sweeps/modules

``plan`` writes plan.json and one configuration per variant (the parent's configuration
with one registered key changed). ``run`` executes the released lifecycle on every
variant, stage by stage from prepare through select, in a fresh output directory
(``run_cross_morse.py --config <variant> --stage <stage> --resume``); it is safe to
repeat. ``collect`` measures every row and writes results.json for
scripts/export_sweep_records.py. A Roman-empire parent is its campaign directory of
split runs. buffer_sensitivity is refused: the released runtime has no buffer-aware
resampling unit (records/specs.d/buffer_sensitivity.json).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cm_experiments.sweeps import SWEEPS, SweepError, collect, plan_sweep, run_plan  # noqa: E402


def _numbers(text: str | None, kind=float):
    return None if text is None else [kind(v) for v in text.split(",") if v.strip()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan", help="write plan.json and the variant configurations")
    p.add_argument("--sweep", required=True, choices=SWEEPS)
    p.add_argument("--parent", type=Path, action="append", default=[], help="sealed run or Roman-empire campaign")
    p.add_argument("--output", type=Path, required=True, help="new sweep directory")
    p.add_argument("--horizon", type=int, default=None, help="registered horizon (default: the contract's)")
    p.add_argument("--horizons", default=None, help="horizon grid, comma separated (default: the contract's)")
    p.add_argument("--beams", default=None, help="beam-width grid, comma separated (default: the contract's)")
    p.add_argument("--profile", default=None, help="profile of the swept MPC-RH configuration (default: deployed)")
    p.add_argument("--ensemble", type=int, default=None, help="ensemble of the swept configuration (default: deployed)")
    p.add_argument("--grid", action="append", default=[], metavar="PARAMETER=V1,V2",
                   help="replace one registered Table S18 grid (smoke runs only)")
    p.add_argument("--no-variants", action="store_true", help="plan only the rows recomputed from the parents")
    p.add_argument("--no-replicate", action="store_true",
                   help="omit the lifecycle replicate of the unchanged parent configuration")
    p.add_argument("--allow-source-change", action="store_true",
                   help="record, instead of refusing, a runtime source that differs from the parents'")
    p = sub.add_parser("run", help="execute the planned variant lifecycles")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--only", action="append", default=None, help="variant identifier (repeatable)")
    p.add_argument("--dry-run", action="store_true", help="print the commands without executing them")
    p = sub.add_parser("collect", help="measure every row and write results.json")
    p.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            grids = {}
            for item in args.grid:
                name, _, values = item.partition("=")
                grids[name.strip()] = _numbers(values)
            path = plan_sweep(args.sweep, args.parent, args.output, horizon=args.horizon,
                              horizons=_numbers(args.horizons, int), beams=_numbers(args.beams, int),
                              profile=args.profile, ensemble=args.ensemble, grids=grids or None,
                              variants=not args.no_variants, replicate=not args.no_replicate,
                              allow_source_change=args.allow_source_change)
            plan = json.loads(path.read_text(encoding="utf-8"))
            print(f"wrote {path} ({len(plan['variants'])} variant lifecycle run(s))")
        elif args.command == "run":
            commands = run_plan(args.plan, data_root=args.data_root, device=args.device, threads=args.threads,
                                python=args.python, only=args.only, dry_run=args.dry_run)
            for command in commands if args.dry_run else ():
                print(" ".join(command))
            print(f"{len(commands)} lifecycle stage(s) {'planned' if args.dry_run else 'executed'}")
        else:
            path = collect(args.plan)
            results = json.loads(path.read_text(encoding="utf-8"))
            print(f"wrote {path} ({len(results['rows'])} values)")
    except (SweepError, OSError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
