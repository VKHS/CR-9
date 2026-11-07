#!/usr/bin/env python3
"""Build original-input supplementary tables and figures with a hashed output manifest.

Missing optional inputs are skipped; malformed/synthetic supplied evidence fails.
No measured values are copied from the paper's reference registry.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
from paper.build.supplement import build_supplement


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-index", type=Path)
    parser.add_argument("--confirmation-campaign", type=Path)
    parser.add_argument("--budget-campaign", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT, help="Checkout containing the registered source/record contracts")
    args = parser.parse_args(argv)
    try:
        report = build_supplement(args.output_dir, records_index=args.records_index,
                                  confirmation_campaign=args.confirmation_campaign, budget_campaign=args.budget_campaign,
                                  manuscript_root=args.root)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 1 if report["errors"] else 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
