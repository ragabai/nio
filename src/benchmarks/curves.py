"""From saved runs to the numbers a figure draws: errors on a common evaluation grid, and their median and spread.

A run's trace holds (evaluation, best value) at every improvement, so its best value after any number of evaluations is
the last improvement at or before it. Errors below `FLOOR` count as 0, as the CEC reports do, and are drawn at `FLOOR`
on a log axis.
"""

import warnings
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

FLOOR = 1e-8


def best_at(trace: np.ndarray, evaluations: np.ndarray) -> np.ndarray:
    """The best value a run had found after each number of evaluations; +inf before its first evaluation."""
    trace = np.asarray(trace, dtype=float).reshape(-1, 2)
    found = np.searchsorted(trace[:, 0], np.asarray(evaluations), side="right")
    return np.concatenate([[np.inf], trace[:, 1]])[found]


def grid(budget: int, points: int = 200) -> np.ndarray:
    """Evaluation counts from 1 to `budget`, spaced evenly."""
    return np.unique(np.linspace(1, budget, points).round().astype(int))


@dataclass(frozen=True, eq=False)
class Band:
    """The median of several runs' errors at each evaluation count, with the quartiles around it."""

    evaluations: np.ndarray
    median: np.ndarray
    low: np.ndarray  # 25th percentile
    high: np.ndarray  # 75th percentile


def band(traces: Iterable[np.ndarray], optimum: float, budget: int, points: int = 200, floor: float = FLOOR) -> Band:
    """The error band of `traces` (runs of one problem with optimum `optimum`), floored at `floor`.

    With `optimum=0` the band is of the best values themselves. Evaluation counts at which no run has a finite value
    yet are NaN, and left undrawn.
    """
    evaluations = grid(budget, points)
    errors = np.maximum(np.array([best_at(trace, evaluations) - optimum for trace in traces]), floor)
    finite = np.where(np.isfinite(errors), errors, np.nan)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)
        low, median, high = np.nanpercentile(finite, [25, 50, 75], axis=0)
    return Band(evaluations, median, low, high)


def floored(errors, floor: float = FLOOR) -> np.ndarray:
    """Errors with every value below `floor` raised to it, for a log axis."""
    return np.maximum(np.asarray(errors, dtype=float), floor)
