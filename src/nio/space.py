"""The box NIO searches: one lower and one upper bound per dimension."""

from dataclasses import dataclass
from typing import Self

import numpy as np


@dataclass(frozen=True, eq=False)
class SearchSpace:
    """Per-dimension bounds of the terrain the herd searches.

    Build one with `SearchSpace.of`, which broadcasts scalar bounds and validates them.
    """

    lb: np.ndarray
    ub: np.ndarray

    @classmethod
    def of(cls, lb, ub, dimensions: int) -> Self:
        """Broadcast scalar or per-dimension bounds to `dimensions` and check that they are finite with lb < ub."""
        if dimensions < 1:
            raise ValueError("A search space needs at least 1 dimension.")
        lb = np.broadcast_to(np.asarray(lb, dtype=float), (dimensions,)).copy()
        ub = np.broadcast_to(np.asarray(ub, dtype=float), (dimensions,)).copy()
        if not np.all(np.isfinite(lb) & np.isfinite(ub)):
            raise ValueError("A search space needs finite bounds.")
        if np.any(lb >= ub):
            raise ValueError("A search space needs lb < ub in every dimension.")
        return cls(lb, ub)

    @property
    def dimensions(self) -> int:
        """The number of dimensions."""
        return self.lb.size

    @property
    def span(self) -> np.ndarray:
        """The width of the box in each dimension, ub - lb."""
        return self.ub - self.lb

    def clip(self, x: np.ndarray) -> np.ndarray:
        """Keep positions inside the bounds."""
        return np.clip(x, self.lb, self.ub)
