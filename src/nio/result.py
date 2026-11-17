"""What a run of NIO returns."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Self

import numpy as np


@dataclass(frozen=True, eq=False)
class Result:
    """What a run of NIO found, and how it got there."""

    fitness: float  # best objective value found
    position: np.ndarray  # where the best value was found
    convergence: np.ndarray  # best value after each season (iteration)
    positions: np.ndarray  # herd positions per season (seasons × agents × dimensions)
    fitnesses: np.ndarray  # herd fitness per season (seasons × agents)
    trajectory: np.ndarray  # best position after each season (seasons × dimensions)
    trace: np.ndarray  # (evaluation, best value) at every improvement of the best (improvements × 2)
    runtime: float  # wall-clock seconds for the run
    evaluations: int  # objective evaluations used

    def best_at(self, evaluations) -> np.ndarray:
        """The best value found within each given number of evaluations; +inf before the first evaluation."""
        found = np.searchsorted(self.trace[:, 0], np.asarray(evaluations), side="right")
        return np.concatenate([[np.inf], self.trace[:, 1]])[found]

    def without_history(self) -> Self:
        """The result without the per-season herd and the trajectory, which are the bulk of it."""
        seasons, dimensions = len(self.convergence), self.position.size
        return replace(
            self,
            positions=np.empty((seasons, 0, dimensions)),
            fitnesses=np.empty((seasons, 0)),
            trajectory=np.empty((0, dimensions)),
        )

    def save(self, path) -> Path:
        """Write every field to a compressed .npz file and return its path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            np.savez_compressed(file, **asdict(self))
        return path

    @classmethod
    def load(cls, path) -> Self:
        """Read a result written by `save`."""
        with np.load(path) as data:
            fields = {name: data[name] for name in data.files}
        scalars = {
            "fitness": float(fields["fitness"]),
            "runtime": float(fields["runtime"]),
            "evaluations": int(fields["evaluations"]),
        }
        return cls(**fields | scalars)
