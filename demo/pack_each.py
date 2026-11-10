#!/usr/bin/env python3
"""Pack every artifact of a demonstration map on its own, so that one refusal does not stop the others.

Each mapped artifact is packed by ``scripts/paper_artifacts.py pack --demo`` from a map that names it alone;
the staged entries (``<stem>.entry.json``) accumulate in one directory, which ``publish --demo`` reads. The
artifacts that the packer refuses are written, with its reason, to ``--failures`` (demo/report.py lists them).

    python -B demo/pack_each.py --map WORK/paper-artifacts.map.json --output-dir STAGED --failures FAILURES.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def reason_of(output: str, returncode: int) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[-1][:800] if lines else f"exit status {returncode}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--failures", type=Path, required=True)
    args = parser.parse_args(argv)
    document = json.loads(args.map.read_text(encoding="utf-8"))
    metadata = {key: value for key, value in document.items() if key.startswith("_")}
    mapped = {key: value for key, value in document.items() if not key.startswith("_") and value}
    packed, failures = [], {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pack-each-") as temporary:
        for artifact, directory in sorted(mapped.items()):
            single = Path(temporary) / "map.json"
            single.write_text(json.dumps({**metadata, artifact: directory}), encoding="utf-8")
            result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / "paper_artifacts.py"), "pack",
                                     "--demo", "--map", str(single), "--output-dir", str(args.output_dir)],
                                    capture_output=True, text=True, cwd=ROOT)
            output = (result.stdout or "") + (result.stderr or "")
            if result.returncode == 0:
                packed.append(artifact)
                print(f"packed {artifact}", flush=True)
            else:
                failures[artifact] = reason_of(output, result.returncode)
                print(f"REFUSED {artifact}: {failures[artifact]}\n{output[-3000:]}", flush=True)
    args.failures.write_text(json.dumps(failures, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{len(packed)} packed, {len(failures)} refused (written to {args.failures})")
    return 0 if packed else 1


if __name__ == "__main__":
    sys.exit(main())
