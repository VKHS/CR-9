"""Executable analytical statements of the CROSS-MORSE manuscript.

Every definition, lemma, proposition, theorem and corollary of the main text,
the Supplementary Lemma 3 (work bound of Algorithm S3) and the equations whose
arithmetic the lifecycle executes are implemented here as functions of explicit
inputs, with NumPy and the Python standard library only. ``cm_theory.statements``
lists them with their runtime counterparts; ``cm_theory.verification`` checks
each one on randomized instances of its hypotheses with fixed seeds
(``make theory``); ``scripts/check_si_identities.py`` applies the same formulas
to the values transcribed from the manuscript (``make si-check``).

The package parses the action registry of ``cross_morse/core.py`` as literals
and imports only the PyTorch-free runtime modules it checks for conformance
(``cross_morse.statistics``, ``cm_experiments.budget``, ``cm_campaigns.numeric``);
it never modifies the files that define the sealed runtime source identity.

A sampled numerical check of an inequality shows that the implementation and the
stated assumptions agree on the instances drawn; it does not replace the proofs
of the supplementary notes. ``docs/THEORY.md`` maps each statement to its
implementation, runtime counterpart and checks.
"""
from __future__ import annotations

from ._util import Check, STATUSES
from .statements import STATEMENTS, Statement, statement

__all__ = ["Check", "STATUSES", "STATEMENTS", "Statement", "statement", "run_battery"]


def run_battery(*, quick: bool = True, seed: int | None = None, only=None) -> dict:
    """Run the verification battery (imported lazily so ``import cm_theory`` stays light)."""
    from .verification import DEFAULT_SEED, run_battery as _run
    return _run(quick=quick, seed=DEFAULT_SEED if seed is None else seed, only=only)
