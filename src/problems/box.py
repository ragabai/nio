"""The box a problem is defined over: one lower and one upper bound per variable.

The problems carry their own box, so they serve any optimizer; NIO and others read its `lb` and `ub`.
"""

from dataclasses import dataclass
from typing import Self

import numpy as np


@dataclass(frozen=True, eq=False)
class Box:
    """Per-variable bounds `lb` and `ub` (arrays of one length). Build one with `Box.of`, which broadcasts scalar
    bounds and checks them."""

    lb: np.ndarray
    ub: np.ndarray

    @classmethod
    def of(cls, lb, ub, dimensions: int) -> Self:
        """Broadcast scalar or per-variable bounds to `dimensions` and check that they are finite with lb < ub."""
        if dimensions < 1:
            raise ValueError("A box needs at least 1 dimension.")
        lb = np.broadcast_to(np.asarray(lb, dtype=float), (dimensions,)).copy()
        ub = np.broadcast_to(np.asarray(ub, dtype=float), (dimensions,)).copy()
        if not np.all(np.isfinite(lb) & np.isfinite(ub)):
            raise ValueError("A box needs finite bounds.")
        if np.any(lb >= ub):
            raise ValueError("A box needs lb < ub in every dimension.")
        return cls(lb, ub)

    @property
    def dimensions(self) -> int:
        """The number of variables."""
        return self.lb.size

    @property
    def span(self) -> np.ndarray:
        """The width of the box in each dimension, ub − lb."""
        return self.ub - self.lb
