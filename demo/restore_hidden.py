#!/usr/bin/env python3
"""Restore the empty hidden files that a workflow artifact drops from a sealed run.

actions/upload-artifact omits hidden files by default, and a run seals its empty
``prepared/.prepare.lock``. For every run under the given directories, a sealed file that is
missing, hidden and declared empty (0 bytes, the SHA-256 of no bytes) in a stage manifest is
recreated empty, which reproduces the sealed bytes exactly. Any other missing sealed file is
reported and left missing.

    python -B demo/restore_hidden.py RUN_OR_CAMPAIGN_DIR ...
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def restore(run: Path) -> tuple[list[str], list[str]]:
    restored, missing = [], []
    for manifest in sorted((run / "provenance" / "stages").glob("*.json")):
        stage = json.loads(manifest.read_text(encoding="utf-8"))
        for field in ("inputs", "outputs"):
            for declared in stage.get(field) or []:
                relative = declared.get("path") if isinstance(declared, dict) else None
                if not isinstance(relative, str) or ".." in Path(relative).parts:
                    continue
                path = run / relative
                if path.exists():
                    continue
                if Path(relative).name.startswith(".") and declared.get("bytes") == 0 and declared.get("sha256") == EMPTY:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
                    restored.append(relative)
                elif relative not in missing:
                    missing.append(relative)
    return restored, missing


def main(argv=None) -> int:
    roots = [Path(a) for a in (argv if argv is not None else sys.argv[1:])]
    for root in roots:
        runs = sorted(p for p in root.glob("split_*") if (p / "provenance").is_dir()) or [root]
        for run in runs:
            restored, missing = restore(run)
            print(f"{run}: restored {len(restored)} empty hidden file(s)" +
                  (f"; still missing {len(missing)}: {missing[:5]}" if missing else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
