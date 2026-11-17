"""When a run stops."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Budget:
    """When a run stops: after `iterations` seasons, or once the objective has been evaluated `evaluations` times.

    Exactly one of the two must be given, and it must be at least 1.
    """

    iterations: int | None = None
    evaluations: int | None = None

    def __post_init__(self):
        if (self.iterations is None) == (self.evaluations is None):
            raise ValueError("A budget needs exactly one of iterations or evaluations.")
        if (self.iterations if self.iterations is not None else self.evaluations) < 1:
            raise ValueError("A budget must allow at least 1 iteration or evaluation.")

    def seasons(self, agents: int) -> int:
        """Seasons to run: the herd's first scoring takes `agents` evaluations and each season `agents` more."""
        if self.iterations is not None:
            return self.iterations
        return math.ceil(self.evaluations / agents) - 1

    def herd(self, dimensions: int) -> int:
        """The herd size NIO takes when none is given: 18 ibex per dimension, but no more than leaves the herd about
        500 seasons (`evaluations / 500`), and at least 30; 30 under an iteration budget, where every ibex adds to
        the cost of the run."""
        if self.evaluations is None:
            return 30
        return max(30, min(18 * dimensions, self.evaluations // 500))

    @property
    def limit(self) -> int | None:
        """The most evaluations the objective may receive, if the budget counts evaluations."""
        return self.evaluations
