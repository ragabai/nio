"""Ridge walk: steps along the line through the origin in the direction (1, …, 1)."""

import numpy as np

from nio.herd import Herd


def ridge_walk(herd: Herd, who: np.ndarray, rng: np.random.Generator, stride: float) -> np.ndarray:
    """An ibex walking the main ridgeline is modelled as stepping along the line through (1, …, 1).

    **This behaviour uses the coordinate frame, not only differences between positions.** The main ridge is the line
    through the origin along (1, …, 1), on which every coordinate is equal; it is the box's main diagonal only when
    every dimension has the same bounds. Every leg of the walk moves all coordinates the same way. The ibex steps onto
    the ridge at the foot of the leader, the leader's projection `mean(x_L)·(1, …, 1)`, and walks along it by one
    common scalar step on every coordinate.

    It is the one behaviour that breaks translation invariance, because the direction (1, …, 1) is absolute rather than
    a difference between positions.

    Formula: `mean(x_L) · (1, …, 1) + s · (1, …, 1)`,   s = stride · ρ · z,   z ~ N(0, 1) (one per proposal),

    where x_L is the herd's leader, mean(x_L) the mean of its coordinates, and ρ the mean over dimensions of the herd's
    per-dimension spread, so the leg shrinks as the herd gathers.

    Edge cases: on a herd standing on one point ρ = 0 and every proposal is the leader's projection; in D = 1 the
    ridge is the whole line and the walk is a step around the leader.
    """
    foot = float(np.mean(herd.positions[herd.best]))
    leg = stride * float(np.mean(herd.spread())) * rng.standard_normal((len(who), 1))
    return (foot + leg) * np.ones((1, herd.positions.shape[1]))
