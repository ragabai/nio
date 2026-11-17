"""Initializers that place the first herd on a fixed pattern; they ignore the run's generator."""

from dataclasses import dataclass

import numpy as np

from nio.initializers.base import Initializer


@dataclass(frozen=True)
class SineCosine(Initializer):
    """The herd spread along one sine period in the even dimensions and one cosine period in the odd ones."""

    name = "Sine Cosine"

    def sample(self, space, agents, rng):
        t = np.linspace(0, 2 * np.pi, agents)[:, None]
        waves = np.where(np.arange(space.dimensions) % 2 == 0, np.sin(t), np.cos(t))
        return space.lb + space.span * (waves + 1) / 2


@dataclass(frozen=True)
class Fibonacci(Initializer):
    """A golden-ratio lattice: ibex `k` stands at `lb + span ⊙ frac(k · φ^−(1, …, D))`."""

    name = "Fibonacci Lattice"

    def sample(self, space, agents, rng):
        golden = (1 + np.sqrt(5)) / 2
        alpha = golden ** -np.arange(1, space.dimensions + 1)
        return space.lb + space.span * (np.outer(np.arange(1, agents + 1), alpha) % 1)
