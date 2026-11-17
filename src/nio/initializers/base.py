"""What an initializer is: a strategy that scatters the first herd over the box."""

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from nio.space import SearchSpace


class Initializer(ABC):
    """Places the first herd inside a search space.

    Calling an initializer returns an (agents × dimensions) array of positions, clipped to the box. Subclasses draw
    the raw positions in `sample`, from the run's generator `rng`, so a seeded run starts from the same herd.
    """

    name: ClassVar[str]

    def __call__(self, space: SearchSpace, agents: int, rng: np.random.Generator) -> np.ndarray:
        """The first herd: `agents` positions drawn by `sample`, brought inside the box."""
        if agents < 1:
            raise ValueError(f"An initializer needs at least 1 agent, not {agents}.")
        positions = np.asarray(self.sample(space, agents, rng), dtype=float)
        if positions.shape != (agents, space.dimensions):
            raise ValueError(
                f"{self.name} drew positions of shape {positions.shape}, not ({agents}, {space.dimensions})."
            )
        return space.clip(positions)

    @abstractmethod
    def sample(self, space: SearchSpace, agents: int, rng: np.random.Generator) -> np.ndarray:
        """Draw the raw positions; `__call__` checks their shape and clips them."""

    def __str__(self) -> str:
        return self.name


def center(space: SearchSpace) -> np.ndarray:
    """The centre of the box, (lb + ub) / 2."""
    return (space.lb + space.ub) / 2
