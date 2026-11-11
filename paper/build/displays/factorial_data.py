"""Joint uncertainty for the paired eight-cell factorial displays.

Marginal cell intervals do not determine the covariance of contrasts. Only a
registered producer's per-unit observations and shared bootstrap stream support
the intervals below. Older summary-only deposits still support point contrasts.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np

from cm_theory.factorial import all_effects, anchored_difference
from .framework import DisplayValidationError

CELLS = ("000", "100", "010", "001", "110", "101", "011", "111")
FACTORS = {"ogbn-arxiv": ("A", "O", "X"), "PascalVOC-SP": ("S", "O", "X")}


def _safe_file(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if (not name or relative.is_absolute() or ".." in relative.parts or "\\" in name):
        raise DisplayValidationError("Factorial raw source path is not confined to its record set")
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise DisplayValidationError("Factorial raw source may not be a symlink")
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise DisplayValidationError(f"Factorial raw source is absent: {name}")
    return path


def _equal(actual, expected, message):
    try:
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        okay = a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a, b, rtol=1e-10, atol=1e-12)
    except (ValueError, TypeError):
        okay = False
    if not okay:
        raise DisplayValidationError(message)


def joint_factorial_contrasts(records, dataset: str, cells: dict) -> dict | None:
    """Replay a deposited ``cm-factorial-2`` joint bootstrap, or report unavailable.

``None`` means that no supported raw joint record was supplied. Inconsistent
recognized records are errors, never a reason to fall back to an independence
assumption. ``cells`` supplies the verified summary means and marginal intervals.
    """
    if records.root is None:
        return None
    sources = {row.get("source_file") for row in records.select("ld_integrated_loss", dataset=dataset)}
    if len(sources) != 1 or None in sources:
        return None
    relative = next(iter(sources))
    path = _safe_file(records.root, relative)
    if path.suffix != ".json":
        return None
    original = path.read_bytes()
    try:
        raw = json.loads(original)
    except (ValueError, UnicodeError) as exc:
        raise DisplayValidationError("Factorial raw source is not valid JSON") from exc
    if not isinstance(raw, dict) or raw.get("schema") != "cm-factorial-2":
        return None
    if raw.get("dataset") != dataset or tuple(raw.get("factors", ())) != FACTORS[dataset]:
        raise DisplayValidationError("Factorial raw dataset/factor identity differs from its summary")
    raw_cells = raw.get("cells", [])
    if [row.get("cell") for row in raw_cells] != list(CELLS) or set(cells) != set(CELLS):
        raise DisplayValidationError("Factorial joint record must bind the same eight ordered cells")
    for row in raw_cells:
        for key in ("ld_integrated_loss", "interval_low", "interval_high"):
            _equal(row.get(key), cells[row["cell"]].get(key), f"Factorial {row['cell']} {key} differs from its summary")
    descriptor = raw.get("unit_dil") or {}
    unit_path = _safe_file(path.parent, descriptor.get("file", ""))
    unit_bytes = unit_path.read_bytes()
    unit_hash = hashlib.sha256(unit_bytes).hexdigest()
    if unit_hash != descriptor.get("sha256"):
        raise DisplayValidationError("Factorial per-unit array hash differs from the raw record")
    with np.load(unit_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    bootstrap = raw.get("bootstrap") or {}
    reps, seed = bootstrap.get("reps"), bootstrap.get("seed")
    if type(reps) is not int or reps < 2 or type(seed) is not int or seed < 0:
        raise DisplayValidationError("Factorial joint bootstrap needs its original replicate count and seed")
    try:
        ids = np.asarray(arrays["unit_id"])
        counts = np.asarray(arrays["unit_weight"], dtype=float)
        values = {cell: np.asarray(arrays[f"cell{cell}_unit_dil"], dtype=float) for cell in CELLS}
        draws = np.asarray(arrays["joint_bootstrap_cell_means"], dtype=float)
    except (KeyError, ValueError) as exc:
        raise DisplayValidationError("Factorial joint source lacks the paired per-unit loss arrays") from exc
    n = len(ids)
    if (ids.ndim != 1 or n < 2 or len(np.unique(ids)) != n or counts.shape != (n,)
            or not np.isfinite(counts).all() or (counts <= 0).any()
            or any(value.shape != (n,) or not np.isfinite(value).all() for value in values.values())
            or draws.shape != (reps, len(CELLS)) or not np.isfinite(draws).all()):
        raise DisplayValidationError("Factorial joint arrays have invalid units, weights, values or shape")
    # Whole-graph Pascal inference gives every graph equal weight. Arxiv's
    # buffered graph clusters retain the number of scored nodes as their weight.
    weights = counts if dataset == "ogbn-arxiv" else np.ones(n, dtype=float)
    means = np.asarray([np.average(values[cell], weights=weights) for cell in CELLS])
    _equal(means, [cells[c]["ld_integrated_loss"] for c in CELLS], "Factorial means do not replay from paired units")
    rng = np.random.default_rng(seed)
    for start in range(0, reps, 1024):
        indices = rng.integers(0, n, size=(min(1024, reps - start), n))
        sampled_weights = weights[indices]
        for j, cell in enumerate(CELLS):
            expected = (sampled_weights * values[cell][indices]).sum(axis=1) / sampled_weights.sum(axis=1)
            _equal(draws[start:start + len(indices), j], expected,
                   "Factorial joint draws do not replay the declared shared resampling stream")
    for j, cell in enumerate(CELLS):
        _equal(np.quantile(draws[:, j], [0.025, 0.975]),
               [cells[cell]["interval_low"], cells[cell]["interval_high"]],
               f"Factorial {cell} marginal interval differs from the joint stream")
    effects = all_effects(dict(zip(CELLS, means)))
    declared = raw.get("contrasts", [])
    labels = FACTORS[dataset]
    if len(declared) != 7 or len({row.get("contrast") for row in declared}) != 7:
        raise DisplayValidationError("Factorial joint record must contain seven unique contrasts")
    declarations = {row["contrast"]: row for row in declared}
    contrasts = {}
    for positions, effect in effects.items():
        signs = np.array([np.prod([1 if cell[p] == "1" else -1 for p in positions]) / 4 for cell in CELLS])
        low, high = np.quantile(draws @ signs, [0.025, 0.975])
        name = ":".join(labels[p] for p in positions)
        stored = declarations.get(name, {})
        _equal([stored.get("effect_coded_nats"), stored.get("interval_low"), stored.get("interval_high"),
                stored.get("baseline_anchored_nats")],
               [effect, low, high, anchored_difference(dict(zip(CELLS, means)), positions)],
               f"Factorial contrast {name} does not replay from the joint stream")
        contrasts[positions] = {"effect": effect, "low": float(low), "high": float(high)}
    if path.read_bytes() != original or unit_path.read_bytes() != unit_bytes:
        raise DisplayValidationError("Factorial joint source changed during its reconstruction")
    return {"contrasts": contrasts, "method": "paired whole-unit bootstrap; one shared resampling stream",
            "source": relative, "source_sha256": hashlib.sha256(original).hexdigest(),
            "unit_source": unit_path.relative_to(records.root).as_posix(), "unit_sha256": unit_hash,
            "units": n, "replicates": reps, "seed": seed, "mode": raw.get("mode")}
