"""Zigzag climb: switchback legs toward better-ranked ibex."""

import numpy as np

from nio.herd import Herd


def movement_angle(rise: float | np.ndarray, run: float | np.ndarray) -> float | np.ndarray:
    """The slope angle θ = atan2(rise, run), in [0, π/2] for rise, run ≥ 0; 0 when both are 0 (ibex on the leader)."""
    return np.arctan2(rise, run)


def zigzag_climb(herd: Herd, who: np.ndarray, rng: np.random.Generator, stride: float) -> np.ndarray:
    """An ibex climbing a steep slope is modelled as moving toward better ground in switchbacks, not straight up.

    Each climber picks a better-ranked ibex t at random and sets `d = x_t − x_i`. The slope's rise is the fitness gap
    scaled by the herd's fitness range, `rise = (f_i − f_t) / (f_worst − f_best)`, and its run is the distance scaled by
    the herd's reach, `run = ‖d‖ / max_k ‖x_k − x_best‖`, both clamped to [0, 1]. With θ = atan2(rise, run) one
    switchback leg is

        `x_i + s·(cos θ·d + sin θ·‖d‖·e)`,   s ~ U(0, 2·stride),

    with e a random unit direction perpendicular to d. On gentle ground (θ → 0) the ibex walks straight at the target;
    on steep ground (θ → π/2) it traverses sideways across the slope.

    Edge cases: the leader, with no one ranked above it, scouts toward the runner-up, so it never proposes its own
    position. The rise is 0 when the fitness range is 0 or not finite, and 1 for a +inf climber below a finite target;
    the run is 0 when the herd's reach is 0. An ibex standing where its target stands (d = 0) has a leg of 0: finite,
    never NaN. With D = 1 there is no direction perpendicular to d (`side` is 0), so the leg is `cos θ · d`.
    """
    dimensions = herd.positions.shape[1]
    x = herd.positions[who]
    ranks = herd.ranks()[who]
    pick = np.floor(rng.random(len(who)) * ranks).astype(int)
    target = herd.order[np.where(ranks > 0, pick, min(1, len(herd.order) - 1))]
    d = herd.positions[target] - x
    distance = np.sqrt(np.add.reduce(d * d, axis=1))  # ‖d‖ per row, as np.linalg.norm(d, axis=1) computes it

    scale = herd.scale()
    f_i, f_t = herd.fitness[who], herd.fitness[target]
    with np.errstate(invalid="ignore", divide="ignore"):
        rise = (f_i - f_t) / scale if scale > 0 else np.zeros(len(who))
    rise = np.where(np.isinf(f_i) & np.isfinite(f_t), 1.0, rise)
    rise = np.minimum(np.maximum(np.where(np.isnan(rise), 0.0, rise), 0.0), 1.0)  # NaN → 0, then into [0, 1]
    reach = herd.reach()
    run = np.minimum(np.maximum(distance / reach, 0.0), 1.0) if reach > 0 else np.zeros(len(who))
    theta = movement_angle(rise, run)

    z = rng.standard_normal((len(who), dimensions))
    unit = np.divide(d, distance[:, None], out=np.zeros_like(d), where=distance[:, None] > 0)
    side = z - np.sum(z * unit, axis=1, keepdims=True) * unit
    length = np.sqrt(np.add.reduce(side * side, axis=1, keepdims=True))
    side = np.divide(side, length, out=np.zeros_like(side), where=length > 1e-300)
    s = rng.uniform(0.0, 2.0 * stride, (len(who), 1))
    leg = np.cos(theta)[:, None] * d + (np.sin(theta) * distance)[:, None] * side
    return x + s * leg
