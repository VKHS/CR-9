"""Deterministic replay (Proposition 1) and the comparator-relative advantage identity (Proposition 11).

Proposition 1 is exercised on a dense member of the plant class driven by a
deterministic label-free feedback rule; Proposition 11 on finite-horizon
decision processes with history-dependent stochastic policies, where every
expectation is an exact enumeration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .operators import Action, transport_matrix
from .stability import relu_retention_step


# ---------------------------------------------------------------------------- Proposition 1
def _measurements(H: np.ndarray, head: np.ndarray, H0: np.ndarray) -> np.ndarray:
    logits = H @ head
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    entropy = float(np.mean(-np.sum(p * np.log(p + 1e-300), axis=1)))
    return np.array([entropy, float(np.linalg.norm(H)), float(np.sum(H * H0) / (np.linalg.norm(H) * np.linalg.norm(H0) + 1e-12))])


def closed_loop(P, H0, W, b, head, actions: tuple[Action, ...], score_weights: np.ndarray, depth: int) -> dict:
    """Feedback execution: measure, score every action, apply the argmin (lowest id on ties)."""
    H = H0.copy()
    chosen, states = [], [H.copy()]
    for _ in range(depth):
        m = _measurements(H, head, H0)
        scores = score_weights @ m
        a = actions[int(np.argmin(scores))]
        chosen.append(a.id)
        H = relu_retention_step(transport_matrix(P, a), H, W, b, a.alpha, H0)
        states.append(H.copy())
    return {"actions": chosen, "states": states, "logits": H @ head}


def open_loop(P, H0, W, b, head, actions: tuple[Action, ...], sequence) -> dict:
    """Replay a stored action sequence without any measurement."""
    by_id = {a.id: a for a in actions}
    H = H0.copy()
    states = [H.copy()]
    for a_id in sequence:
        a = by_id[a_id]
        H = relu_retention_step(transport_matrix(P, a), H, W, b, a.alpha, H0)
        states.append(H.copy())
    return {"actions": list(sequence), "states": states, "logits": H @ head}


# ---------------------------------------------------------------------------- Proposition 11
@dataclass(frozen=True)
class FiniteProcess:
    """States 0..S-1, actions 0..A-1, horizon L, kernel p[s, a, s'], stage cost c[s, a], terminal cost f[s]."""
    transition: np.ndarray
    cost: np.ndarray
    terminal: np.ndarray
    initial: int
    horizon: int


History = tuple  # (s0, a0, s1, a1, ..., s_l)
Policy = Callable[[History], np.ndarray]  # distribution over actions


def random_process(rng: np.random.Generator, states: int = 3, actions: int = 3, horizon: int = 4) -> FiniteProcess:
    p = rng.random((states, actions, states)) + 0.05
    p /= p.sum(axis=2, keepdims=True)
    return FiniteProcess(p, rng.normal(size=(states, actions)), rng.normal(size=states), 0, horizon)


def random_history_policy(rng: np.random.Generator, actions: int) -> Policy:
    """A stochastic policy that depends on the whole history through a hashed seed."""
    salt = int(rng.integers(1 << 30))

    def policy(history: History) -> np.ndarray:
        local = np.random.default_rng(abs(hash((salt,) + tuple(history))) % (1 << 32))
        w = local.random(actions) + 0.05
        return w / w.sum()
    return policy


def value(process: FiniteProcess, policy: Policy, history: History, cache: dict) -> float:
    """V(h): expected remaining stage plus terminal cost when ``policy`` continues from h."""
    key = history
    if key in cache:
        return cache[key]
    depth = (len(history) - 1) // 2
    s = history[-1]
    if depth == process.horizon:
        out = float(process.terminal[s])
    else:
        out = 0.0
        for a, pa in enumerate(policy(history)):
            if pa == 0:
                continue
            q = process.cost[s, a]
            for s_next, ps in enumerate(process.transition[s, a]):
                q += ps * value(process, policy, history + (a, s_next), cache)
            out += pa * q
    cache[key] = out
    return out


def advantage_identity(process: FiniteProcess, policy: Policy, comparator: Policy) -> dict:
    """Return V^pi_0 - V^C_0 and E_pi[sum_l A^C_l(h_l, a_l)], both by exact enumeration."""
    v_c: dict = {}
    v_pi: dict = {}
    h0: History = (process.initial,)
    left = value(process, policy, h0, v_pi) - value(process, comparator, h0, v_c)

    def q_c(history: History, a: int) -> float:
        s = history[-1]
        return float(process.cost[s, a] + sum(ps * value(process, comparator, history + (a, s_next), v_c)
                                               for s_next, ps in enumerate(process.transition[s, a])))

    def expected_advantage(history: History) -> float:
        depth = (len(history) - 1) // 2
        if depth == process.horizon:
            return 0.0
        s = history[-1]
        total = 0.0
        baseline = value(process, comparator, history, v_c)
        for a, pa in enumerate(policy(history)):
            if pa == 0:
                continue
            step = q_c(history, a) - baseline
            future = sum(ps * expected_advantage(history + (a, s_next))
                         for s_next, ps in enumerate(process.transition[s, a]))
            total += pa * (step + future)
        return total

    right = expected_advantage(h0)
    return {"value_difference": float(left), "accumulated_advantage": float(right),
            "residual": float(abs(left - right))}
