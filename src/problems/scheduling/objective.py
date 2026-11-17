"""Makespan minimisation as an objective over a continuous position, under one of two encodings."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from problems.box import Box
from problems.scheduling.schedule import Schedule, by_assignment, by_priority
from problems.scheduling.workload import Workload

type Encoding = Literal["priority", "assignment"]


@dataclass(frozen=True, eq=False)
class Scheduling:
    """Makespan minimisation over a continuous position, decoded by one of two encodings.

    priority:   [0, 1]^tasks, sorting the position gives the task order for list scheduling (random keys).
    assignment: [0, machines]^tasks, the integer part of each coordinate is that task's machine.
    """

    workload: Workload
    encoding: Encoding = "priority"

    def __post_init__(self):
        if self.encoding not in ("priority", "assignment"):
            raise ValueError(f"Unknown encoding '{self.encoding}'. Choose 'priority' or 'assignment'.")

    @property
    def space(self) -> Box:
        """[0, 1]^tasks for the priority encoding, [0, machines]^tasks for the assignment encoding."""
        upper = 1 if self.encoding == "priority" else self.workload.count
        return Box.of(0, upper, self.workload.size)

    def machines(self, positions: np.ndarray) -> np.ndarray:
        """Machine index per coordinate under the assignment encoding."""
        return np.clip(np.floor(positions).astype(int), 0, self.workload.count - 1)

    def decode(self, position: np.ndarray) -> Schedule:
        """The schedule a position encodes."""
        position = np.asarray(position, dtype=float)
        if self.encoding == "priority":
            return by_priority(self.workload, np.argsort(position, kind="stable"))
        return by_assignment(self.workload, self.machines(position))

    def batch(self, population: np.ndarray) -> np.ndarray:
        """Makespan of every row; the assignment encoding is computed for the whole population at once."""
        population = np.atleast_2d(np.asarray(population, dtype=float))
        if self.encoding == "priority":
            return np.array([self.decode(position).makespan for position in population])
        return self.workload.loads(self.machines(population)).max(axis=1)

    def __call__(self, position: np.ndarray) -> float:
        return float(self.batch(position)[0])
