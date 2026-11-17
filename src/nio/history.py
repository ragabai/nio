"""The herd as it stood at the end of each season."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nio.result import Result

if TYPE_CHECKING:
    from nio.evaluator import Evaluator
    from nio.herd import Herd


class History:
    """What the herd did each season; with `keep=False`, only the best value per season (not the herd itself)."""

    def __init__(self, keep: bool = True):
        self.keep = keep
        self.convergence: list[float] = []
        self.positions: list[np.ndarray] = []
        self.fitnesses: list[np.ndarray] = []
        self.trajectory: list[np.ndarray] = []

    def record(self, herd: Herd) -> None:
        """Note the herd at the end of a season."""
        self.convergence.append(herd.elite_fit)
        if self.keep:
            self.positions.append(herd.positions.copy())
            self.fitnesses.append(herd.fitness.copy())
            self.trajectory.append(herd.elite.copy())

    def result(self, herd: Herd, evaluate: Evaluator, runtime: float) -> Result:
        """Assemble the `Result` of a finished run."""
        seasons, dimensions = len(self.convergence), herd.elite.size
        if self.keep:
            positions = np.array(self.positions)
            fitnesses = np.array(self.fitnesses)
            trajectory = np.array(self.trajectory)
        else:
            positions = np.empty((seasons, 0, dimensions))
            fitnesses = np.empty((seasons, 0))
            trajectory = np.empty((0, dimensions))
        return Result(
            fitness=float(herd.elite_fit),
            position=herd.elite,
            convergence=np.array(self.convergence, dtype=float),
            positions=positions,
            fitnesses=fitnesses,
            trajectory=trajectory,
            trace=evaluate.trace,
            runtime=runtime,
            evaluations=evaluate.count,
        )
