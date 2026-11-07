"""Helpers for testing display generators on small synthetic inputs.

Tests build a temporary run directory holding exactly the files a generator
reads, wrap it in a :class:`RunView`, and call the generator through
:func:`build_one`. Values in such fixtures are synthetic; they exercise the
computation and the layout, never the paper's results.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .framework import (GENERATORS, DisplayContext, DisplayWriter, RecordSetView, RunView,
                        _load_generators, compare)


def write_files(root: Path, files: dict[str, Any]) -> Path:
    """Write ``{relative path: content}``: ``.json`` objects, ``.npz`` array dicts, ``.csv`` row dicts."""
    root = Path(root)
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative.endswith(".json"):
            path.write_text(json.dumps(content, indent=1), encoding="utf-8")
        elif relative.endswith(".npz"):
            np.savez(path, **{k: np.asarray(v) for k, v in content.items()})
        elif relative.endswith(".npy"):
            np.save(path, np.asarray(content))
        elif relative.endswith(".csv"):
            rows = list(content)
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
        else:
            path.write_text(str(content), encoding="utf-8")
    return root


def fake_run(root: Path, dataset: str, files: dict[str, Any], *, evidence: dict | None = None,
             splits: list[RunView] = ()) -> RunView:
    write_files(root, files)
    return RunView(dataset, root, evidence=evidence, splits=splits)


def fake_records(name: str, rows: list[dict], *, root: Path | None = None, spec: dict | None = None,
                 synthetic: bool = True) -> RecordSetView:
    """Record rows use the summary schema: set keys plus quantity, value, unit, source_file, locator."""
    if spec is None:
        specs = json.loads((Path(__file__).resolve().parents[3] / "records/specs.json").read_text(encoding="utf-8"))
        spec = specs["sets"].get(name, {"keys": []})
    normal = []
    for row in rows:
        row = dict(row)
        row.setdefault("unit", "")
        row.setdefault("source_file", "raw/synthetic.json")
        row.setdefault("locator", "")
        row["value"] = float(row["value"])
        normal.append(row)
    return RecordSetView(name, root, spec, normal, synthetic=synthetic)


def context(**kwargs) -> DisplayContext:
    kwargs.setdefault("synthetic", True)
    return DisplayContext(**kwargs)


def build_one(display_id: str, ctx: DisplayContext, out_dir: Path, *, registry: dict | None = None):
    """Run one generator; return its writer and the comparison rows against ``registry``."""
    _load_generators()
    generator = GENERATORS[display_id]
    writer = DisplayWriter(display_id, generator.stem, Path(out_dir), title=generator.title,
                           synthetic=ctx.synthetic, scope=generator.scope)
    generator.function(ctx, writer)
    rows = compare(writer.comparisons, registry or {})
    return writer, rows


def read_table(out_dir: Path, stem: str) -> list[dict[str, str]]:
    with (Path(out_dir) / "tables" / f"{stem}.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
