"""The terrain's edges: how an ibex that would step off the search space is brought back onto it."""

import numpy as np

from nio.space import SearchSpace


def repair(x: np.ndarray, parent: np.ndarray, space: SearchSpace) -> np.ndarray:
    """Bring each proposal back inside `[lb, ub]` using only the proposal `x`, the ibex it came from (`parent`) and the
    bounds, so the repair moves with the terrain when everything is translated.

    Per coordinate that crosses a bound b: the ibex rebounds off the rock face, `2·b − x`, when that lands between the
    parent and b (a small overshoot keeps its size); otherwise it lands halfway between where it stood and the edge,
    `(parent + b) / 2`. A non-finite coordinate returns to the parent's. Coordinates inside the bounds are unchanged.
    Works on one position or on rows of positions with matching parents.
    """
    x = np.array(x, dtype=float)
    lb, ub = space.lb, space.ub
    if ((x >= lb) & (x <= ub)).all():  # NaN fails both comparisons
        return x
    parent = np.broadcast_to(np.asarray(parent, dtype=float), x.shape)
    x = np.where(np.isfinite(x), x, parent)
    for crossed, bound in ((x > ub, ub), (x < lb, lb)):
        bound = np.broadcast_to(bound, x.shape)
        rebound = 2 * bound - x
        between = (rebound - parent) * (bound - rebound) >= 0
        x = np.where(crossed, np.where(between, rebound, (parent + bound) / 2), x)
    return np.clip(x, lb, ub)
