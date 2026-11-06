#!/usr/bin/env python3
"""Verify original S14 tests and replay their joint nine-contrast Holm family.

    python -B scripts/audit_s14_confirmation.py /path/to/campaign --output report.json
    python -B scripts/audit_s14_confirmation.py --write-template template.json

No original test program is executed. Reports belong outside the campaign root.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
from cm_campaigns.common import write_json
from cm_campaigns.confirmation import check_campaign, template


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", nargs="?", type=Path)
    parser.add_argument("--campaign", dest="campaign_option", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--write-template", type=Path)
    args = parser.parse_args(argv)
    if args.write_template:
        if args.campaign or args.campaign_option or args.output:
            parser.error("--write-template is separate from auditing")
        write_json(args.write_template, template(args.root))
        print(json.dumps({"status": "TEMPLATE_WRITTEN", "path": str(args.write_template), "evidence": False}))
        return 0
    if bool(args.campaign) == bool(args.campaign_option):
        parser.error("supply exactly one campaign directory")
    campaign = (args.campaign or args.campaign_option).resolve()
    if args.output and args.output.resolve().is_relative_to(campaign):
        parser.error("write the report outside the immutable campaign directory")
    try:
        report = check_campaign(campaign, manuscript_root=args.root)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        report = {"schema": "cm-s14-audit-1", "status": "FAIL", "campaign": str(campaign), "error": str(exc),
                  "original_test_p_values_recomputed": False}
    if args.output:
        write_json(args.output, report, replace=True)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
