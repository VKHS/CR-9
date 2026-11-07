#!/usr/bin/env python3
"""Build the joint S14 confirmation campaign (campaign:confirmation) from the sealed lifecycle runs.

    python -B scripts/build_s14_campaign.py plan --parent <arxiv run> --parent <proteins run> \\
        --parent <products run> --parent <Roman-empire campaign of ten split runs> --parent <PascalVOC-SP run> \\
        --parent <COCO-SP run> --parent <Peptides-func run> --output <dir>
    python -B scripts/build_s14_campaign.py run --output <dir> [--job K] --device cuda --trust-local-checkpoints
    python -B scripts/build_s14_campaign.py assemble --output <dir> --campaign <campaign dir>

plan      reads the sealed, complete parents (no labels) and registers the S14 protocol: each dataset's target
          policy and pair, the confirmation units and weights, the test methods and the hashed test code
          (<output>/registration/), and one job per (run, source) in <output>/plan.json.
run       executes the jobs with the released runtime (one with --job K, e.g. a Slurm array task); completed
          jobs are skipped.  A read job rescores the run's sealed D_confirm predictions and verifies the sealed
          confirm-stage contrast; a compute job executes the contrast with the mechanism-contrast machinery.
          Loading local checkpoints needs --trust-local-checkpoints.
assemble  writes the campaign directory (a new directory outside the checkout) from the completed jobs and
          records check_campaign's verdict in <output>/campaign_check.json.  Exit status 0: the campaign was
          written and check_campaign accepted it (PASS or REFERENCE_MISMATCH, the separate comparison with the
          printed S14 values; post_hoc and admission_findings report a joint lock fixed after the lifecycle's
          own admissions); 1: written, but check_campaign refused it (the verdict names the requirement).

The implementation is cm_campaigns/s14_builder.py.
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
    from cm_campaigns.s14_builder import plan_s14
    return str(plan_s14(args.parent, args.output, allow_partial=args.allow_partial, force_compute=args.force_compute,
                        reps=args.reps))


def _run(args):
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    if not args.trust_local_checkpoints:
        raise ValueError("loading local checkpoints requires explicit trust (--trust-local-checkpoints)")
    try:
        import torch
    except ImportError:
        raise RuntimeError("executing the S14 jobs requires PyTorch (environment/requirements)") from None
    from cross_morse.common import install_signal_handlers
    from cm_campaigns.s14_builder import run_s14
    torch.set_num_threads(args.threads)
    install_signal_handlers()
    return [str(p) for p in run_s14(args.output, job=args.job, device=args.device, trust=args.trust_local_checkpoints)]


def _assemble(args):
    from cm_campaigns.s14_builder import assemble_s14
    verdict = assemble_s14(args.output, args.campaign, manuscript_root=args.root)
    return {"campaign": str(Path(args.campaign).resolve()), "status": verdict["status"],
            "accepted": verdict["accepted"], "error": verdict.get("error"),
            "post_hoc": (verdict.get("report") or {}).get("post_hoc"), "findings": verdict["findings"],
            "check_report": str(Path(args.output).resolve() / "campaign_check.json")}


def parser():
    top = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    actions = top.add_subparsers(dest="action", required=True)
    p = actions.add_parser("plan", help="register the protocol and write <output>/plan.json from sealed parents")
    p.add_argument("--parent", type=Path, action="append", required=True,
                   help="sealed, complete lifecycle run, or the Roman-empire campaign directory (repeatable)")
    p.add_argument("--output", type=Path, required=True, help="new build directory outside the checkout")
    p.add_argument("--reps", type=int, default=None,
                   help="bootstrap replicates of re-executed contrasts (default: the run's registered count)")
    p.add_argument("--allow-partial", action="store_true",
                   help="software check: fewer than seven datasets (assemble refuses a partial plan)")
    p.add_argument("--force-compute", action="store_true",
                   help="software check on synthetic parents: re-execute even contrasts the runs sealed")
    p = actions.add_parser("run", help="execute the registered jobs with the released runtime")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--job", type=int, default=None, help="execute one job of the plan (default: all)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--trust-local-checkpoints", action="store_true",
                   help="authorize loading the parents' local checkpoint files (pickle)")
    p = actions.add_parser("assemble", help="write the campaign directory and check it")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--campaign", type=Path, required=True, help="new campaign directory outside the checkout")
    p.add_argument("--root", type=Path, default=ROOT,
                   help="manuscript root holding paper/sources.json etc. (default: this checkout)")
    return top


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        result = {"plan": _plan, "run": _run, "assemble": _assemble}[args.action](args)
    except InterruptedError as exc:
        print(f"[checkpoint] {exc}; rerun the same command to resume", file=sys.stderr)
        return 75
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"action": args.action, "result": result}, indent=2))
    if args.action == "assemble" and not result["accepted"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
