"""Transport operators of main Eqs. (8)-(10), Lemma 1 and Lemma 2 (dense references).

Every registered transport action is a polynomial in the row-stochastic operator
P with nonnegative coefficients that sum to one:

    T_a = (1 - nu) I + nu K_a,   K_a = (1 - rho) P + rho K~_{eta,T}   (resolvent actions)
    K~_{eta,T} = eta sum_{t=0}^{T} (1-eta)^t P^t + (1-eta)^{T+1} I       (Eq. 9)

and K_a = P for the one-step actions A1/A2 (``eta is None``), exactly as
``cross_morse.core.apply_kernel``/``apply_transport`` execute them. The executed
range (Eq. 10), maximum reach, identity mass and identity coefficient are read
off the coefficient vector, which also proves Lemma 1 at the polynomial level:
nonnegative coefficients summing to one turn any row-stochastic P into a
row-stochastic T_a.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._util import ROOT, norm_inf

CORE = ROOT / "cross_morse" / "core.py"


@dataclass(frozen=True)
class Action:
    id: int
    name: str
    rho: float
    nu: float
    alpha: float
    eta: float | None = None
    order: int = 8
    probability: float = 0.0
    mode: str = "transport"

    @property
    def is_transport(self) -> bool:
        return self.mode == "transport"

    @property
    def is_resolvent(self) -> bool:
        return self.is_transport and self.eta is not None


def load_registry(core_path: Path | str = CORE) -> tuple[Action, ...]:
    """The ``ACTIONS`` tuple of the runtime, read as literals without importing it.

    Only literal ``Action(...)`` calls are accepted, so parsing executes no code;
    a changed registry shape raises instead of being guessed.
    """
    path = Path(core_path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    fields = ("id", "name", "rho", "nu", "alpha", "eta", "order", "probability", "mode")
    for node in tree.body:
        targets = getattr(node, "targets", None) or ([node.target] if isinstance(node, ast.AnnAssign) else [])
        if not any(isinstance(t, ast.Name) and t.id == "ACTIONS" for t in targets):
            continue
        if not isinstance(node.value, ast.Tuple):
            raise ValueError("ACTIONS must be a literal tuple")
        actions = []
        for call in node.value.elts:
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "Action"):
                raise ValueError("ACTIONS entries must be literal Action(...) calls")
            if len(call.args) > len(fields):
                raise ValueError("too many positional Action fields")
            keyword_names = [kw.arg for kw in call.keywords]
            if None in keyword_names or len(set(keyword_names)) != len(keyword_names) \
                    or set(fields[:len(call.args)]) & set(keyword_names):
                raise ValueError("Action fields must be unique literal arguments; no ** expansion")
            record = {name: ast.literal_eval(arg) for name, arg in zip(fields, call.args)}
            record.update({kw.arg: ast.literal_eval(kw.value) for kw in call.keywords})
            unknown = set(record) - set(fields)
            if unknown or not {"id", "name", "rho", "nu", "alpha"} <= set(record):
                raise ValueError(f"unexpected Action fields: {sorted(unknown) or 'missing required fields'}")
            actions.append(Action(**record))
        if [a.id for a in actions] != list(range(len(actions))):
            raise ValueError("action identifiers must be 0..N-1 in order")
        return tuple(actions)
    raise ValueError(f"ACTIONS registry not found in {path}")


def transport_actions(registry: tuple[Action, ...] | None = None) -> tuple[Action, ...]:
    registry = load_registry() if registry is None else registry
    return tuple(a for a in registry if a.is_transport)


# ---------------------------------------------------------------------------- polynomial form
def operator_polynomial(action: Action) -> np.ndarray:
    """Coefficients c_t of T_a = sum_t c_t P^t (index t = power of P)."""
    if not action.is_transport:
        raise ValueError("hybrid resets (A8/A9) assign a state; they have no transport polynomial")
    nu, rho = float(action.nu), float(action.rho)
    if action.eta is None:
        return np.array([1.0 - nu, nu])
    eta, order = float(action.eta), int(action.order)
    c = np.zeros(max(order, 1) + 1)
    c[0] += 1.0 - nu
    c[1] += nu * (1.0 - rho)
    c[: order + 1] += nu * rho * eta * (1.0 - eta) ** np.arange(order + 1)
    c[0] += nu * rho * (1.0 - eta) ** (order + 1)
    return c


def executed_range(action: Action) -> float:
    """Main Eq. (10): expected hops per executed step, sum_t t c_t."""
    c = operator_polynomial(action)
    return float(np.dot(np.arange(len(c)), c))


def executed_range_closed_form(action: Action) -> float:
    """Eq. (10) as printed: nu [(1-rho) + rho sum_t t eta (1-eta)^t] (nu for one-step actions)."""
    if action.eta is None:
        return float(action.nu)
    t = np.arange(action.order + 1)
    return float(action.nu * ((1 - action.rho) + action.rho * np.sum(t * action.eta * (1 - action.eta) ** t)))


def identity_coefficient(action: Action) -> float:
    """Coefficient of P^0 in the executed operator."""
    return float(operator_polynomial(action)[0])


def identity_coefficient_closed_form(action: Action) -> float:
    """(1-nu) + nu rho [eta + (1-eta)^{T+1}] as printed below Eq. (10)."""
    if action.eta is None:
        return float(1 - action.nu)
    return float((1 - action.nu) + action.nu * action.rho * (action.eta + (1 - action.eta) ** (action.order + 1)))


def identity_mass(action: Action) -> float:
    """(1-eta)^{T+1}, the truncation tail returned to the identity (zero for one-step actions)."""
    return 0.0 if action.eta is None else float((1 - action.eta) ** (action.order + 1))


def maximum_reach(action: Action) -> int:
    """Largest power of P with positive weight: T for a resolvent action, one otherwise."""
    c = operator_polynomial(action)
    return int(np.max(np.nonzero(c > 0)[0]))


# ---------------------------------------------------------------------------- dense matrices
def matrix_polynomial(P: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    n = len(P)
    out = coefficients[0] * np.eye(n)
    power = np.eye(n)
    for c in coefficients[1:]:
        power = power @ P
        out = out + c * power
    return out


def truncated_resolvent(P: np.ndarray, eta: float, order: int) -> np.ndarray:
    """Eq. (9): K~_{eta,T} = eta sum_{t<=T} (1-eta)^t P^t + (1-eta)^{T+1} I."""
    if not 0 < eta <= 1 or order < 0:
        raise ValueError("Lemma 2 requires 0 < eta <= 1 and T >= 0")
    coefficients = eta * (1 - eta) ** np.arange(order + 1)
    coefficients[0] += (1 - eta) ** (order + 1)
    return matrix_polynomial(P, coefficients)


def exact_resolvent(P: np.ndarray, eta: float) -> np.ndarray:
    """K_eta = eta [I - (1-eta) P]^{-1}, built by a dense solve (references only)."""
    if not 0 < eta <= 1:
        raise ValueError("Lemma 2 requires 0 < eta <= 1")
    n = len(P)
    return eta * np.linalg.solve(np.eye(n) - (1 - eta) * P, np.eye(n))


def kernel_matrix(P: np.ndarray, action: Action) -> np.ndarray:
    """K_a = (1-rho) P + rho K~ (resolvent actions) or P (one-step actions); no lazy mixing."""
    if not action.is_transport:
        raise ValueError("a recovery action is not a transport matrix")
    if action.eta is None:
        return P.copy()
    return (1 - action.rho) * P + action.rho * truncated_resolvent(P, action.eta, action.order)


def transport_matrix(P: np.ndarray, action: Action) -> np.ndarray:
    """Main Eq. (8): T_a = (1 - nu) I + nu K_a."""
    return (1 - action.nu) * np.eye(len(P)) + action.nu * kernel_matrix(P, action)


def inserted_site(H: np.ndarray, H0: np.ndarray, P: np.ndarray, action: Action) -> np.ndarray:
    """Definition 1 inserted site V_a(H, R) with R = H0; A0 is a literal bypass.

    Matches ``cross_morse.core.DesignatedSitePlant.step`` at inserted sites:
    (1-nu) H + nu [(1-alpha) K_a H + alpha H0], K_a the non-lazy kernel.
    Resets assign H0 (A8) or the supplied last-safe state elsewhere.
    """
    if not action.is_transport:
        raise ValueError("resets are hybrid assignments, not inserted transport")
    if action.id == 0:
        return H
    return (1 - action.nu) * H + action.nu * ((1 - action.alpha) * (kernel_matrix(P, action) @ H) + action.alpha * H0)


def nominal_shared_matrix(P: np.ndarray) -> np.ndarray:
    """The native A0 aggregation of the shared-transport plants: 0.3 I + 0.7 P."""
    return 0.3 * np.eye(len(P)) + 0.7 * P


# ---------------------------------------------------------------------------- Lemma 1 and Lemma 2
def is_row_stochastic(matrix: np.ndarray, atol: float = 1e-11) -> bool:
    return bool(np.min(matrix) >= -atol and np.max(np.abs(matrix.sum(axis=1) - 1.0)) <= atol)


def mixture_transport(kernels: list[np.ndarray], weights: np.ndarray, nu: float) -> np.ndarray:
    """Lemma 1's general form T = (1-nu) I + nu sum_m w_m K~_m."""
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (len(kernels),) or (w < 0).any() or not np.isclose(w.sum(), 1.0) or not 0 <= nu <= 1:
        raise ValueError("Lemma 1 requires w in the simplex and 0 <= nu <= 1")
    n = len(kernels[0])
    return (1 - nu) * np.eye(n) + nu * sum(wm * km for wm, km in zip(w, kernels))


def resolvent_truncation_error(P: np.ndarray, eta: float, order: int) -> dict:
    """||K_eta - K~_{eta,T}||_inf with the Lemma 2 bound and the exact tail identity.

    K_eta - K~_{eta,T} = (1-eta)^{T+1} (P^{T+1} K_eta - I), so the error is
    (1-eta)^{T+1} times the distance of the row-stochastic P^{T+1} K_eta from I,
    which is at most two.
    """
    K = exact_resolvent(P, eta)
    Kt = truncated_resolvent(P, eta, order)
    tail = (1 - eta) ** (order + 1)
    identity_form = tail * (np.linalg.matrix_power(P, order + 1) @ K - np.eye(len(P)))
    return {"error": norm_inf(K - Kt), "bound": 2.0 * tail, "tail_mass": tail,
            "identity_residual": norm_inf((K - Kt) - identity_form)}
