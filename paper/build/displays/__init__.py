"""Generators of every numbered table and figure of the paper and supplement.

See :mod:`paper.build.displays.framework` for the contract. Generator modules in
this package register themselves with :func:`framework.display`;
:func:`framework.build_displays` runs them and compares each declared cell with the
value printed in the paper.
"""
from .framework import (DATASETS, GENERATORS, DisplayContext, DisplayInputError, DisplayWriter,  # noqa: F401
                        RecordSetView, RunView, build_displays, compare_value, display)
