"""Cliff escape: heavy-tailed leaps across the terrain."""

import numpy as np

from nio.herd import Herd
from nio.space import SearchSpace


def cliff_escape(
    herd: Herd, who: np.ndarray, rng: np.random.Generator, space: SearchSpace, reach: float, ledges: float
) -> np.ndarray:
    """An ibex fleeing a threat across the rock is modelled as leaping: mostly short hops, now and then a long bound.

    Formula: `x_i + reach·span ⊙ C ⊙ m`, with C standard Cauchy per dimension (heavy-tailed: mostly short, occasionally
    across the terrain) and m a mask that selects each dimension with probability `min(1, ledges / D)`, plus one
    dimension chosen at random so at least one is always selected. One leap crosses a few ledges, not every direction at
    once. Its size is set by the terrain (`span`), not by the herd, so it still reaches distant ground once the herd has
    gathered.

    Ibex leap by choice, through route preference. With `Settings.CLIFF_PATIENCE` above 0 the optimizer also sends
    an ibex that has stopped improving on this leap once, then lets it settle and gives it its patience again.

    Edge cases: with D = 1 the leap always happens (the mask is 1); a herd standing on one point still leaps by
    `reach·span`, since the size comes from the terrain.
    """
    dimensions = space.dimensions
    x = herd.positions[who]
    mask = rng.random(x.shape) < min(1.0, ledges / dimensions)
    forced = rng.integers(dimensions, size=len(who))
    mask[np.arange(len(who)), forced] = True
    leap = reach * space.span * rng.standard_cauchy(x.shape)
    return x + np.where(mask, leap, 0.0)
