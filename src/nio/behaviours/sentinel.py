"""Sentinel: wide look-outs across the whole bounds."""

import numpy as np

from nio.herd import Herd
from nio.space import SearchSpace


def sentinel(herd: Herd, who: np.ndarray, rng: np.random.Generator, space: SearchSpace, lookouts: float) -> np.ndarray:
    """A sentinel ibex keeping watch from the ridge is modelled as scanning a few lines of terrain far from the herd.

    The sentinel stays where it is in most dimensions. Each dimension is redrawn with probability
    `min(1, lookouts / D)`, with at least one redrawn, uniformly across the whole bounds: `lb_d + span_d · U(0, 1)`.
    The redrawn coordinates are relative to the terrain's edges, not to any point inside it, so the behaviour
    translates with the problem and is unbiased across the bounds. With greedy replacement a sentinel only moves when
    the look-out finds better ground, so a wide look costs an evaluation, never the ibex's place.

    The optimizer lowers the sentinels' share of the herd over the season: it scales their route preference by
    `(1 − τ)^SENTINEL_FADE` (`Settings.SENTINEL_FADE`).

    Edge cases: with `lookouts` ≥ D every dimension is redrawn; with D = 1 the proposal is a fresh uniform draw.
    """
    dimensions = space.dimensions
    x = herd.positions[who]
    mask = rng.random(x.shape) < min(1.0, lookouts / dimensions)
    mask[np.arange(len(who)), rng.integers(dimensions, size=len(who))] = True
    look = space.lb + space.span * rng.random(x.shape)
    return np.where(mask, look, x)
