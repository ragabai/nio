"""One problem of the CEC 2020 real-world constrained suite, its violation measure and its competition statistics."""

from dataclasses import dataclass

import numpy as np

from problems.box import Box
from problems.engineering.realworld.common import Function, Outcome

TOLERANCE = 1e-4  # an equality constraint holds once |h| is within this


@dataclass(frozen=True, eq=False)
class RealWorld:
    """One problem of the CEC 2020 real-world constrained suite: minimise f subject to g <= 0 and h = 0.

    `evaluate` returns the organisers' (f, g, h) for a population; `violation` is the competition's measure,
    the mean over all constraints of the positive part of g and of |h| beyond the 1e-4 tolerance.
    """

    code: str
    name: str
    category: str
    function: Function
    lb: tuple[float, ...]
    ub: tuple[float, ...]
    best: float  # best known feasible objective value
    inequalities: int
    equalities: int

    @property
    def dimensions(self) -> int:
        """The number of variables."""
        return len(self.lb)

    @property
    def space(self) -> Box:
        """The competition's search range."""
        return Box.of(self.lb, self.ub, self.dimensions)

    @property
    def budget(self) -> int:
        """The competition's evaluation budget, which grows with the dimension."""
        d = self.dimensions
        return (
            100_000 if d <= 10 else 200_000 if d <= 30 else 400_000 if d <= 50 else 800_000 if d <= 150 else 1_000_000
        )

    def evaluate(self, population: np.ndarray) -> Outcome:
        """The organisers' (f, g, h) for every row: objective (n,), inequalities (n × g) and equalities (n × h)."""
        x = np.array(population, dtype=float, ndmin=2)
        with np.errstate(all="ignore"):
            f, g, h = self.function(x)
        return np.asarray(f, dtype=float).reshape(len(x)), g, h

    def violations(self, g: np.ndarray, h: np.ndarray) -> np.ndarray:
        """Each constraint's violation: g where positive, |h| where it exceeds the tolerance."""
        return np.hstack([np.maximum(g, 0.0), np.where(np.abs(h) > TOLERANCE, np.abs(h), 0.0)])

    def violation(self, population: np.ndarray) -> np.ndarray:
        """The competition's mean constraint violation ν; 0 means feasible."""
        _, g, h = self.evaluate(population)
        return self.violations(g, h).sum(axis=1) / max(self.inequalities + self.equalities, 1)

    def repair(self, x: np.ndarray) -> np.ndarray:
        """The positions unchanged: integer variables are rounded inside each function."""
        return np.asarray(x, dtype=float)

    def measure(self, population: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Objective and per-constraint violations for a penalty; undefined values (NaN) count as infinitely bad."""
        f, g, h = self.evaluate(population)
        return np.nan_to_num(f, nan=np.inf), np.nan_to_num(self.violations(g, h), nan=np.inf)

    def statistics(self, solutions: np.ndarray) -> dict[str, float]:
        """The competition's table for the final solutions of independent runs (one per row).

        Runs are ranked feasible first by objective, then infeasible by violation; best, median and worst follow
        that order, mean and std are over all runs, and FR is the share of runs whose final solution is feasible.
        """
        f, _, _ = self.evaluate(solutions)
        v = self.violation(solutions)
        order = np.lexsort((np.where(v > 0, v, f), v > 0))
        best, median, worst = order[0], order[(len(order) - 1) // 2], order[-1]
        return {
            "Best f": f[best],
            "Best ν": v[best],
            "Median f": f[median],
            "Median ν": v[median],
            "Mean f": f.mean(),
            "Mean ν": v.mean(),
            "Worst f": f[worst],
            "Worst ν": v[worst],
            "Std f": f.std(),
            "Std ν": v.std(),
            "FR": float(np.mean(v <= 0)),
        }
