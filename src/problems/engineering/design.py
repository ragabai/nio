"""A constrained design problem, and the static penalty that turns it, or any constrained problem, into an objective."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from problems.box import Box

type Vectorised = Callable[[np.ndarray], np.ndarray]  # works on positions stacked along leading axes


@dataclass(frozen=True)
class Design:
    """A constrained engineering design problem: minimise `cost` subject to every `constraints` value <= 0.

    Constraints are normalised by their limits, so one penalty weight treats them alike.
    """

    code: str
    name: str
    variables: tuple[str, ...]
    lb: tuple[float, ...]
    ub: tuple[float, ...]
    cost: Vectorised
    constraints: Vectorised  # (..., constraints)
    best: float  # best known feasible cost in the literature
    discrete: Vectorised | None = None  # snaps positions to the values a variable may take

    @property
    def dimensions(self) -> int:
        """The number of design variables."""
        return len(self.variables)

    @property
    def space(self) -> Box:
        """The box the variables range over."""
        return Box.of(self.lb, self.ub, self.dimensions)

    def repair(self, x: np.ndarray) -> np.ndarray:
        """The positions with their discrete variables snapped to the values they may take."""
        x = np.asarray(x, dtype=float)
        return self.discrete(x) if self.discrete is not None else x

    def violations(self, x: np.ndarray) -> np.ndarray:
        """How far each constraint is exceeded: 0 when it holds."""
        return np.maximum(self.constraints(self.repair(x)), 0.0)

    def measure(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Cost and per-constraint violations, after snapping discrete variables."""
        x = self.repair(x)
        return self.cost(x), self.violations(x)


@dataclass(frozen=True, eq=False)
class Solution:
    """A design as built: its variables after snapping, its cost, each constraint's violation, and whether it holds."""

    x: np.ndarray  # design after snapping discrete variables
    cost: float
    violations: np.ndarray
    feasible: bool


class Constrained(Protocol):
    """A constrained problem a penalty can wrap: Design, or a CEC 2020 RealWorld problem."""

    @property
    def space(self) -> Box: ...

    def repair(self, x: np.ndarray) -> np.ndarray: ...

    def measure(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass(frozen=True)
class Penalized:
    """Static penalty: cost + factor * sum of constraint violations, a minimisation objective over the problem's space.

    With a large enough factor, the linear penalty is exact: the penalised and constrained optima coincide.
    """

    problem: Constrained
    factor: float = 1e6
    tolerance: float = 1e-6  # violation still counted as feasible when reporting

    @property
    def space(self) -> Box:
        """The problem's box."""
        return self.problem.space

    def batch(self, population: np.ndarray) -> np.ndarray:
        """The penalised cost of every row."""
        cost, violations = self.problem.measure(population)
        return cost + self.factor * violations.sum(axis=-1)

    def __call__(self, position: np.ndarray) -> float:
        return float(np.ravel(self.batch(position))[0])

    def evaluate(self, position: np.ndarray) -> Solution:
        """The design at `position`: its cost, its violations and whether it is feasible within `tolerance`."""
        x = self.problem.repair(position)
        cost, violations = self.problem.measure(x)
        violations = np.ravel(violations)  # one position: a RealWorld returns a row of one population
        return Solution(x, float(np.ravel(cost)[0]), violations, bool(np.all(violations <= self.tolerance)))
