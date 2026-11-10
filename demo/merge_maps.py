#!/usr/bin/env python3
"""Add to the main graph's publication map the artifacts that another demonstration graph produced.

A graph planned for one record set alone (``paper_outputs.py plan --only``, e.g. the Table S29 sweep of
demo-sweep.yml) writes its own map. Every artifact that such a map names, and that the main map leaves
awaiting, is added to the main map with its directory; the main map's other entries are kept.

    python -B demo/merge_maps.py MAIN_MAP EXTRA_MAP [EXTRA_MAP ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def merge(main: dict, extra: dict) -> list[str]:
    added = []
    for artifact, directory in extra.items():
        if artifact.startswith("_") or not directory or main.get(artifact):
            continue
        main[artifact] = directory
        (main.setdefault("_awaiting", {})).pop(artifact, None)
        main.setdefault("_relationships", {})[artifact] = (extra.get("_relationships") or {}).get(
            artifact, "post_result_revision")
        added.append(artifact)
    return added


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    target = Path(argv[0])
    document = json.loads(target.read_text(encoding="utf-8"))
    for path in argv[1:]:
        added = merge(document, json.loads(Path(path).read_text(encoding="utf-8")))
        print(f"{path}: added {', '.join(added) if added else 'nothing'}")
    target.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
