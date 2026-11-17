"""Helpers shared by the problem definitions, which follow the organisers' MATLAB code line by line.

Every problem maps a population x (n × D) to (f, g, h): objective (n,), inequality constraints (n × g) that hold
when <= 0, and equality constraints (n × h) that hold when 0. `c(k)` is MATLAB's x(:, k), 1-based.
"""

from collections.abc import Callable
from functools import cache

import numpy as np

from problems.paths import DATA

type Outcome = tuple[np.ndarray, np.ndarray, np.ndarray]
type Function = Callable[[np.ndarray], Outcome]


def columns(x: np.ndarray) -> Callable[[int], np.ndarray]:
    """MATLAB's column access: `columns(x)(k)` is x(:, k), 1-based."""
    return lambda k: x[:, k - 1]


def stack(n: int, *values) -> np.ndarray:
    """Constraint columns as an (n × k) array; values may be scalars, broadcast to every row."""
    return (
        np.stack([np.broadcast_to(np.asarray(v, dtype=float), (n,)) for v in values], axis=1)
        if values
        else np.zeros((n, 0))
    )


def none(n: int) -> np.ndarray:
    """No constraints of a kind: an (n × 0) array."""
    return np.zeros((n, 0))


def mround(x):
    """MATLAB's round: halves away from zero (NumPy rounds them to even)."""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


@cache
def data() -> dict[str, np.ndarray]:
    """The official input data: power networks (PS*) and feed composition tables (RM_*)."""
    with np.load(DATA / "cec" / "rwc2020.npz") as archive:
        return {name: archive[name] for name in archive.files}
