"""Scoring positions with the objective, with the evaluation count, the budget limit and the best-so-far trace."""

import math
from collections.abc import Callable
from typing import Protocol, runtime_checkable

import numpy as np

type Objective = Callable[[np.ndarray], float]


@runtime_checkable
class Batched(Protocol):
    """An objective that can also score a whole population (rows are positions) in one call."""

    def batch(self, population: np.ndarray) -> np.ndarray: ...


def _number(value) -> float:
    """The objective's value as a float; a 1-element array counts as its single number. None is an error."""
    if value is None:
        raise TypeError("The objective returned None; it must return a single number (a function with no `return`?).")
    array = np.asarray(value, dtype=float)
    if array.size != 1:
        raise ValueError(f"The objective must return a single number, not an array of shape {array.shape}.")
    return float(array.item())


class Evaluator:
    """Scores positions with the objective and counts every evaluation.

    With a `limit`, it stops calling the objective once the limit is reached and scores the rest +inf, which the herd
    never accepts. NaN is scored +inf too. Every value that improves on the best so far is traced, so a run's best
    value is known after any number of evaluations.
    """

    def __init__(self, objective: Objective, limit: int | None = None):
        self.objective = objective
        self.limit = limit
        self.count = 0
        self.best = math.inf
        self._trace: list[tuple[int, float]] = []

    @property
    def trace(self) -> np.ndarray:
        """(evaluation, best value) at every improvement, evaluations numbered from 1: (improvements × 2)."""
        return np.array(self._trace, dtype=float).reshape(-1, 2)

    def one(self, x: np.ndarray) -> float:
        """Score one position, or +inf without calling the objective once the limit is reached."""
        if self.limit is not None and self.count >= self.limit:
            return math.inf
        return self._record(_number(self.objective(x)))

    def many(self, population: np.ndarray) -> np.ndarray:
        """Score every row: in one call when the objective is `Batched`, otherwise row by row in order."""
        if not isinstance(self.objective, Batched):
            return np.array([self.one(x) for x in population], dtype=float)
        scores = np.full(len(population), math.inf)
        allowed = len(population) if self.limit is None else max(0, min(len(population), self.limit - self.count))
        if allowed:
            returned = self.objective.batch(population[:allowed])
            if returned is None:
                raise TypeError("The objective returned None from batch(); it must return one number per row.")
            values = np.asarray(returned, dtype=float).reshape(allowed)
            scores[:allowed] = [self._record(float(v)) for v in values]
        return scores

    def _record(self, value: float) -> float:
        self.count += 1
        if math.isnan(value):
            value = math.inf
        if value < self.best:
            self.best = value
            self._trace.append((self.count, value))
        return value
