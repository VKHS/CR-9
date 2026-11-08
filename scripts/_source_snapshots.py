"""Resolve an exact runtime identity without relabeling historical artifacts."""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys

from scripts._paper_common import runtime_source_fingerprint

ROOT = Path(__file__).resolve().parents[1]
HASH = re.compile(r"[0-9a-f]{64}\Z")


def runtime_files(root):
    root = Path(root)
    return sorted(root.glob("run_cross_morse*.py")) + sorted((root / "cross_morse").glob("*.py"))


def resolve_runtime_source(expected_sha256, root=ROOT, *, run_root=None):
    """Return current, bundled, or the run's own byte-verified runtime source.

    No unknown source, missing snapshot, symlinked runtime file, or mismatching
    digest is accepted. This does not import or execute the resolved source.
    Original exports may carry ``source/`` inside the run or its campaign parent.
    """
    if not isinstance(expected_sha256, str) or not HASH.fullmatch(expected_sha256):
        raise ValueError("Invalid runtime source SHA-256")
    root = Path(root).resolve()
    candidates = [root, root / "examples/source_snapshots" / expected_sha256]
    if run_root is not None:
        run_root = Path(run_root).absolute()
        if any(p.is_symlink() for p in (run_root, *run_root.parents)):
            raise ValueError("Run source resolution must not traverse a symlink")
        run_root = run_root.resolve()
        candidates.extend((run_root / "source", run_root.parent / "source"))
    for source in dict.fromkeys(candidates):
        if not source.is_dir():
            continue
        if source != root and (any(p.is_symlink() for p in (source, *source.parents))
                               or any(p.is_symlink() for p in source.rglob("*"))):
            raise ValueError("Historical runtime source snapshot must not contain symlink directories")
        paths = runtime_files(source)
        if (not paths or not (source / "cross_morse/pipeline.py").is_file()
                or (source / "cross_morse").is_symlink()):
            continue
        if any(p.is_symlink() or not p.is_file() for p in paths):
            raise ValueError("Runtime source identity cannot be satisfied by symlinked files")
        if runtime_source_fingerprint(source) == expected_sha256:
            return source
    raise ValueError("Run requires its exact runtime source snapshot; no matching verified source is available")


def source_resolution(expected_sha256, root=ROOT, *, run_root=None):
    root = Path(root).resolve()
    source = resolve_runtime_source(expected_sha256, root, run_root=run_root)
    location = ("current_checkout" if source == root else
                source.relative_to(root).as_posix() if source.is_relative_to(root) else str(source))
    return {"runtime_source_sha256": expected_sha256, "verified": True,
            "source_location": location,
            "run_artifacts_rewritten": False}


def replay_results_exact(run_root, *, selection=True, root=ROOT):
    """Recompute with the verified runtime that produced a historical run."""
    run_root = Path(run_root).absolute()
    if any(p.is_symlink() for p in (run_root, *run_root.parents)):
        raise ValueError("Numerical replay must not traverse a symlinked run path")
    run_root = run_root.resolve()
    first = json.loads((run_root / "provenance/stages/00_prepare.json").read_text())
    expected = first["source_sha256"]
    source = resolve_runtime_source(expected, root, run_root=run_root)
    if source == ROOT.resolve():
        from cross_morse.review import replay_results
        return replay_results(run_root, selection=selection)
    code = '''
import json, sys
sys.path.insert(0, sys.argv[1])
from cross_morse.provenance import current_source_sha256
if current_source_sha256() != sys.argv[4]:
    raise ValueError("Historical runtime source changed before numerical replay")
from cross_morse.review import replay_results
print(json.dumps(replay_results(sys.argv[2], selection=json.loads(sys.argv[3])), sort_keys=True, allow_nan=False))
'''
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(source), str(run_root),
                             json.dumps(bool(selection)), expected], text=True, capture_output=True, timeout=300)
    if result.returncode:
        raise ValueError("Exact-source numerical replay failed: " + result.stderr.strip()[-4000:])
    if runtime_source_fingerprint(source) != expected:
        raise ValueError("Historical runtime source changed during numerical replay")
    return json.loads(result.stdout)
