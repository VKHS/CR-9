#!/usr/bin/env python3
"""Render the registered badge locally, including inside private repositories."""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
COLORS={'blue':'#007ec6','brightgreen':'#4c1','green':'#97ca00','red':'#e05d44','orange':'#fe7d37','yellow':'#dfb317','lightgrey':'#9f9f9f'}

def render_badge(value: dict) -> str:
    if value.get('schemaVersion')!=1 or not all(isinstance(value.get(k),str) and value[k] for k in ('label','message')):
        raise ValueError('A schemaVersion 1 badge with label and message is required')
    label,message=value['label'],value['message']
    if len(label)>200 or len(message)>500:
        raise ValueError('Badge text is too long')
    color=value.get('color','blue')
    color=COLORS.get(color,color)
    if not re.fullmatch(r'#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?',color):
        raise ValueError('Unsupported badge color')
    left=max(50,round(len(label)*6.7)+16)
    right=max(50,round(len(message)*6.7)+16)
    width=left+right
    title=html.escape(label+': '+message,quote=True)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="20" role="img" aria-label="{title}">'
            f'<title>{title}</title><rect width="{left}" height="20" fill="#555"/>'
            f'<rect x="{left}" width="{right}" height="20" fill="{color}"/>'
            '<g fill="#fff" font-family="Verdana,DejaVu Sans,sans-serif" font-size="11" text-anchor="middle">'
            f'<text x="{left/2}" y="14">{html.escape(label)}</text>'
            f'<text x="{left+right/2}" y="14">{html.escape(message)}</text></g></svg>\n')

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json',type=Path,default=ROOT/'docs/badges/evidence.json')
    parser.add_argument('--svg',type=Path,default=ROOT/'docs/badges/evidence.svg')
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args(argv)
    text=render_badge(json.loads(args.json.read_text(encoding='utf-8')))
    if args.check:
        if not args.svg.is_file() or args.svg.read_text(encoding='utf-8')!=text:
            raise SystemExit('Local badge is stale; run make claims')
    else:
        args.svg.parent.mkdir(parents=True,exist_ok=True)
        args.svg.write_text(text,encoding='utf-8')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
