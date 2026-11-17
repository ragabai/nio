"""Follow a trail: part of the way to a dominant, plus a stride along the line from an old track to a herd-mate."""

import numpy as np

from nio.herd import Herd
from nio.tracks import Tracks


def follow_trail(
    herd: Herd,
    who: np.ndarray,
    rng: np.random.Generator,
    dominants: int,
    stride: float | np.ndarray,
    keep: float | np.ndarray = 1.0,
    tracks: Tracks | None = None,
) -> np.ndarray:
    """An ibex walking along a trail is modelled as heading for a dominant and going on the way the herd has been
    moving.

    In the model a herd spreads out along the good ground it feeds on, and every ibex that moves to better ground
    leaves a track where it stood (`nio.tracks`). The offset from a track, or from where another herd-mate stands, to
    a herd-mate marks the line of a trail across the terrain. The follower picks one of the `dominants` top-ranked
    ibex p at random (never itself: when it is among them, the spare next-ranked ibex stands in), a herd-mate r1 and a
    second place r2, another herd-mate or one of the `tracks`, all different, and lands

        `x + s · (x_p − x) + s · (x_r1 − x_r2)`,

    a stride s of the way to the dominant and the same stride along the trail. `stride` is one number for every
    follower or one per follower (the herd's lore, `nio.lore`, draws them). On a long curved valley the herd so moves
    along its floor, the way its members are strung out, instead of straight across it. This is the
    current-to-pbest/1 move of differential evolution with an archive (JADE, Zhang and Sanderson 2009).

    The follower keeps to the trail in some directions and to its own ground in the others: each coordinate of the
    landing comes from the trail with probability `keep` (one number, or one per follower), and otherwise stays the
    follower's own, with at least one coordinate from the trail (`keep` 1: the whole landing is the trail's). Keeping
    some of its own coordinates, the follower carries what it has already found in those directions onto the trail.
    Only differences between positions and the follower's own coordinates are used, so the behaviour translates with
    the problem.

    The optimizer scales this behaviour's route preference by the size of the herd (`Settings.TRAIL_WEIGHT`). The
    behaviour sees the whole herd, not a subgroup.

    Edge cases: in a herd of two there is no pair of other places, and the trail is the line the follower came along:
    `x + 2s · (x_p − x)`. On a herd standing on one spot the proposal is that spot. `dominants` is clamped to between
    1 and `agents − 1`. In D = 1 the one coordinate is the trail's.
    """
    agents = len(herd.fitness)
    count = min(max(1, dominants), agents - 1)
    top = herd.order[: count + 1]  # one spare in case the follower is itself a dominant
    pick = top[rng.integers(count, size=len(who))]
    dominant = np.where(pick == who, top[count], pick)
    old = 0 if tracks is None else len(tracks)
    ground = herd.positions if old == 0 else np.concatenate((herd.positions, tracks.positions))
    if agents >= 3:
        r1 = rng.integers(agents - 1, size=len(who))
        r1 += r1 >= who
        low, high = np.minimum(who, r1), np.maximum(who, r1)
        r2 = rng.integers(agents - 2 + old, size=len(who))
        r2 += r2 >= low
        r2 += r2 >= high
    else:  # a herd of two: the trail is the follower's own line to the dominant
        r1, r2 = dominant, who
    x = herd.positions[who]
    s = np.broadcast_to(np.reshape(stride, (-1, 1)), (len(who), 1))
    trail = x + s * (herd.positions[dominant] - x) + s * (herd.positions[r1] - ground[r2])
    if np.all(np.asarray(keep) >= 1):
        return trail
    taken = rng.random(x.shape) < np.reshape(keep, (-1, 1))
    taken[np.arange(len(who)), rng.integers(x.shape[1], size=len(who))] = True
    return np.where(taken, trail, x)
