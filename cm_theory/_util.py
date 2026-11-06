"""Shared result type, norms, graph generators and printed-precision helpers."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STATUSES = ("PASS", "FAIL", "ADVISORY", "INFO")


def _plain(value: Any) -> Any:
    """JSON-safe copy of a numeric result (NumPy scalars and arrays become Python values)."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


@dataclass(frozen=True)
class Check:
    """One verified claim.

    ``observed`` and ``bound`` carry the compared quantities (for an inequality,
    the measured left side and the stated right side; for an identity, the two
    sides). ``instances`` counts the randomized instances checked. PASS and FAIL
    describe the stated claim and INFO records a computed reference value without
    a pass/fail criterion. ADVISORY completes the status vocabulary shared with
    ``scripts/check_si_identities.py``, where it marks a printed value that only a
    reading other than the stated one reproduces.
    """
    statement: str
    claim: str
    status: str
    observed: Any = None
    bound: Any = None
    instances: int = 1
    detail: str = ""

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        if not isinstance(self.instances, int) or self.instances < 0:
            raise ValueError("instances must be a nonnegative integer")

    def to_dict(self) -> dict:
        return {"statement": self.statement, "claim": self.claim, "status": self.status,
                "observed": _plain(self.observed), "bound": _plain(self.bound),
                "instances": self.instances, "detail": self.detail}


def passed(condition: bool) -> str:
    return "PASS" if bool(condition) else "FAIL"


# ---------------------------------------------------------------------------- norms
def norm_inf(matrix: np.ndarray) -> float:
    """Induced maximum-row-sum norm ||M||_inf."""
    return float(np.max(np.sum(np.abs(matrix), axis=1)))


def norm_inf2(states: np.ndarray) -> float:
    """||H||_{inf,2}: the largest row Euclidean norm (main Proposition 2)."""
    return float(np.max(np.linalg.norm(states, axis=1)))


def fro(matrix: np.ndarray) -> float:
    return float(np.linalg.norm(matrix))


# ---------------------------------------------------------------------------- graphs
def finish_adjacency(weights: np.ndarray) -> np.ndarray:
    """Symmetric nonnegative adjacency with the runtime's isolated-vertex self-loop.

    ``cross_morse.core.GraphOperator`` adds a unit self-loop only to vertices of
    zero degree; every other convention is left to the caller. This helper
    applies the same rule so dense reference operators match the runtime's.
    """
    w = np.array(weights, dtype=np.float64)
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError("adjacency must be square")
    if (w < 0).any() or not np.isfinite(w).all():
        raise ValueError("adjacency must be finite and nonnegative")
    if not np.allclose(w, w.T):
        raise ValueError("adjacency must be symmetric")
    isolated = w.sum(axis=1) == 0
    w[isolated, isolated] = 1.0
    return w


def random_graph(n: int, rng: np.random.Generator, density: float = 0.3,
                 weighted: bool = False) -> np.ndarray:
    upper = np.triu(rng.random((n, n)) < density, 1).astype(np.float64)
    if weighted:
        upper *= rng.uniform(0.2, 2.0, size=(n, n))
    return finish_adjacency(upper + upper.T)


def path_graph(n: int) -> np.ndarray:
    w = np.zeros((n, n))
    idx = np.arange(n - 1)
    w[idx, idx + 1] = w[idx + 1, idx] = 1.0
    return finish_adjacency(w)


def star_graph(n: int) -> np.ndarray:
    w = np.zeros((n, n))
    w[0, 1:] = w[1:, 0] = 1.0
    return finish_adjacency(w)


def complete_bipartite(a: int, b: int) -> np.ndarray:
    """Bipartite graphs put an eigenvalue of L_sym at exactly 2 (periodic random walk)."""
    n = a + b
    w = np.zeros((n, n))
    w[:a, a:] = 1.0
    w[a:, :a] = 1.0
    return finish_adjacency(w)


def graph_family(rng: np.random.Generator, count: int, n_range: tuple[int, int] = (6, 18)) -> list[np.ndarray]:
    """Random weighted/unweighted graphs plus spectrally extreme structured graphs."""
    graphs = [path_graph(7), star_graph(8), complete_bipartite(3, 5)]
    while len(graphs) < count:
        n = int(rng.integers(n_range[0], n_range[1] + 1))
        graphs.append(random_graph(n, rng, density=float(rng.uniform(0.15, 0.6)),
                                   weighted=bool(rng.integers(2))))
    return graphs[:count]


def random_walk(adjacency: np.ndarray) -> np.ndarray:
    """P = D^{-1} W."""
    return adjacency / adjacency.sum(axis=1, keepdims=True)


def symmetric_normalized(adjacency: np.ndarray) -> np.ndarray:
    """S = D^{-1/2} W D^{-1/2}; the symmetric normalized Laplacian is I - S."""
    d = adjacency.sum(axis=1)
    inv = 1.0 / np.sqrt(d)
    return adjacency * inv[:, None] * inv[None, :]


def laplacian_eigensystem(adjacency: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues in [0, 2] and orthonormal eigenvectors of L_sym = I - S."""
    lam, vectors = np.linalg.eigh(np.eye(len(adjacency)) - symmetric_normalized(adjacency))
    return np.clip(lam, 0.0, 2.0), vectors


# ---------------------------------------------------------------------------- printed values
def printed_half_unit(value: Any) -> float:
    """Half of the last printed decimal, the rounding tolerance of a transcribed value.

    The same convention as ``scripts/_paper_common.printed_half_unit``; kept here so
    the package does not import the scripts tree.
    """
    if isinstance(value, bool):
        raise TypeError("a boolean has no printed precision")
    if isinstance(value, int):
        return 0.5
    value = float(value)
    if value == 0.0:
        return 0.0
    text = repr(value)
    if "e" in text.lower():
        mantissa, exponent = text.lower().split("e")
        decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
        return 0.5 * 10 ** (int(exponent) - decimals)
    decimals = len(text.split(".")[1]) if "." in text else 0
    return 0.5 * 10 ** (-decimals)


def interval_of(value: Any) -> tuple[float, float]:
    """The real interval a printed value can stand for."""
    h = printed_half_unit(value)
    return float(value) - h, float(value) + h


def within_printed(computed: float, printed: Any, extra: float = 0.0) -> bool:
    """True when ``computed`` rounds to ``printed`` (plus a declared propagated tolerance)."""
    return abs(float(computed) - float(printed)) <= printed_half_unit(printed) + extra + 1e-12


def finite(values: Iterable[float]) -> bool:
    return all(math.isfinite(float(v)) for v in values)
