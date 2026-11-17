"""Home range: pulls toward the centre of the search box."""

import numpy as np

from nio.herd import Herd
from nio.space import SearchSpace


def home_range(herd: Herd, who: np.ndarray, rng: np.random.Generator, space: SearchSpace, depth: float) -> np.ndarray:
    """An ibex returning toward the core of its home range is modelled as moving toward the centre of the search box.

    **This behaviour uses the structure of the search box.** The home range is the search box, and its core is the
    box's centre `c = (lb + ub) / 2`, a fixed point of the box's frame. Unlike the other behaviours, which work on
    differences between ibex, it pulls toward that point. Shifting the bounds together with the objective moves the
    centre with them, so the behaviour keeps the run translation invariant.

    Formula: `c + r ⊙ (x_i − c)`,   r = 10^(−depth·U),   U ~ U(0, 1) per dimension.

    Each visit shrinks the ibex's distance to the centre, coordinate by coordinate, to a random share r of what was
    left: from all of it (r = 1) down to 10^−depth of it, log-uniformly, so repeated visits close in on the centre
    geometrically (`depth`/2 orders of magnitude per visit on average).

    The optimizer makes the visits seasonal. Route preference scales this behaviour by `τ^HOME_RISE`
    (`Settings.HOME_RISE`): rare early in the run, while the herd ranges widely, and more frequent as the run goes on.
    The `depth` is `HOME_DEPTH · min(1, τ / HOME_FULL)` (`Settings.HOME_DEPTH`, `Settings.HOME_FULL`): early on an ibex
    drifts only part of the way back, so the herd is not drawn onto the centre before it has scouted the terrain.
    With `HOME_FULL` 0 the depth is `HOME_DEPTH` all year, from the first season.

    Edge cases: an ibex already on the centre stays there; with `depth` 0 the proposal is the ibex's own position
    (up to rounding in `c + (x_i − c)`).
    """
    x = herd.positions[who]
    centre = (space.lb + space.ub) / 2
    return centre + 10.0 ** (-depth * rng.random(x.shape)) * (x - centre)
