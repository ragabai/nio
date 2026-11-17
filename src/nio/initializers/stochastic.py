"""Initializers that draw the first herd at random from the run's generator."""

from dataclasses import dataclass, field

import numpy as np

from nio.initializers.base import Initializer, center


@dataclass(frozen=True)
class RandomUniform(Initializer):
    """Every ibex uniform in the box: `lb + span · U`, the herd NIO starts from by default."""

    name = "Random Uniform"

    def sample(self, space, agents, rng):
        return space.lb + space.span * rng.random((agents, space.dimensions))


@dataclass(frozen=True)
class Opposition(Initializer):
    """Half the herd from `base`, the other half mirrored through the centre of the box (`lb + ub − x`)."""

    name = "Opposition Based"
    base: Initializer = field(default_factory=RandomUniform)

    def sample(self, space, agents, rng):
        half = self.base(space, (agents + 1) // 2, rng)
        return np.vstack([half, space.clip(space.lb + space.ub - half)])[:agents]


@dataclass(frozen=True)
class Sphere(Initializer):
    """Points on the surface of the ellipsoid inscribed in the box: `c + u ⊙ span / 2`, `u` a random unit vector."""

    name = "Sphere Sampling"

    def sample(self, space, agents, rng):
        directions = rng.standard_normal((agents, space.dimensions))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        return center(space) + directions * space.span / 2


@dataclass(frozen=True)
class Centroid(Initializer):
    """A tight cloud around the centre of the box, with a fixed absolute standard deviation `spread`."""

    name = "Centroid"
    spread: float = 0.1

    def sample(self, space, agents, rng):
        return center(space) + rng.normal(0, self.spread, (agents, space.dimensions))


@dataclass(frozen=True)
class Gaussian(Initializer):
    """Normal around the centre of the box, with a standard deviation of `spread` times the span."""

    name = "Gaussian"
    spread: float = 0.25

    def sample(self, space, agents, rng):
        return rng.normal(center(space), space.span * self.spread, (agents, space.dimensions))
