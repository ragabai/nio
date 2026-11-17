"""`Function`: a benchmark function and what is known about it, the type every suite of `problems.functions` holds."""

from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np

Bound = float | np.ndarray


@dataclass(frozen=True)
class Function:
    """A benchmark function and what is known about it.

    `key` ("F1", "F2", ...) and `name` identify it; `solve` is the objective itself, taking a position (an array of
    length `dim`) and returning a number, and calling the `Function` calls it; `lb` and `ub` are the bounds (a number
    for every dimension, or an array); `dim` is the dimension and `optimal` the global optimum value, so a result is
    reported as its error `f(best) − optimal`. `resize` builds the same function at another dimension; it is None
    when the dimension is fixed.

        >>> rosenbrock.at(dim=50, lb=-10, ub=10)  # a customised copy
    """

    key: str
    name: str
    solve: Callable[[np.ndarray], float]
    lb: Bound
    ub: Bound
    dim: int
    optimal: float
    resize: "Callable[[int], Function] | None" = field(default=None, compare=False, repr=False)
    fixed_box: bool = field(default=False, compare=False, repr=False)  # True when its optimum was placed for its box

    def __call__(self, x: np.ndarray) -> float:
        return self.solve(x)

    def batch(self, population: np.ndarray) -> np.ndarray:
        """The value of every row of `population`: in one call when the objective scores populations itself."""
        whole = getattr(self.solve, "batch", None)
        if whole is not None:
            return np.asarray(whole(population), dtype=float)
        return np.array([self.solve(x) for x in population], dtype=float)

    def at(self, dim: int | None = None, lb: Bound | None = None, ub: Bound | None = None) -> "Function":
        """This function at `dim` dimensions and over [`lb`, `ub`]; what is left out stays as it is.

        Bounds set earlier carry over to a new dimension when they are numbers; bounds per coordinate must be given
        again. A shifted function keeps its box, since its shift was drawn for it."""
        fn = self
        if dim is not None and dim != self.dim:
            if self.resize is None:
                raise ValueError(f"{self.key} {self.name} has a fixed dimension {self.dim}, not {dim}.")
            if any(np.ndim(bound) for bound, given in ((self.lb, lb), (self.ub, ub)) if given is None):
                raise ValueError(f"{self.key} has bounds per coordinate; give its bounds for {dim} dimensions.")
            fn = self.resize(dim)
            lb = self.lb if lb is None else lb
            ub = self.ub if ub is None else ub
        if lb is None and ub is None:
            return fn
        lb = fn.lb if lb is None else lb
        ub = fn.ub if ub is None else ub
        try:
            low, high = _box(lb, fn.dim), _box(ub, fn.dim)
        except ValueError:
            raise ValueError(f"The bounds of {fn.key} must be numbers or arrays of length {fn.dim}.") from None
        if not np.all(low < high):
            raise ValueError(f"The lower bounds of {fn.key} must lie below its upper bounds.")
        if fn.fixed_box and not (
            np.array_equal(low, _box(fn.lb, fn.dim)) and np.array_equal(high, _box(fn.ub, fn.dim))
        ):
            raise ValueError(f"{fn.key} {fn.name} keeps its box [{fn.lb}, {fn.ub}]: its shift was drawn for it.")
        return replace(fn, lb=lb, ub=ub)

    def shifted(self) -> "Function":
        """This classic function with its optimum moved off the origin, as in the `fair23` suite, at its dimension."""
        from problems.classic import classic  # here, not at the top: both modules build on Function
        from problems.shifted import fair23

        if self.key not in classic or classic[self.key].name != self.name:
            raise ValueError(
                f"{self.key} {self.name} is not one of the 23 classic functions, so it has no shifted form."
            )
        default = classic[self.key].at(dim=self.dim)
        if not (
            np.array_equal(_box(self.lb, self.dim), _box(default.lb, self.dim))
            and np.array_equal(_box(self.ub, self.dim), _box(default.ub, self.dim))
        ):
            raise ValueError(f"{self.key} has custom bounds; a shifted function keeps the classic bounds of its shift.")
        return fair23[self.key].at(dim=self.dim)


def _box(bound: Bound, dim: int) -> np.ndarray:
    return np.broadcast_to(np.asarray(bound, dtype=float), (dim,))
