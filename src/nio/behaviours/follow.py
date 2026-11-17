"""Follow the dominant: steps toward a blend of the best-ranked ibex."""

import numpy as np

from nio.herd import Herd, dominance


def follow_dominant(herd: Herd, who: np.ndarray, rng: np.random.Generator, dominants: int, reach: float) -> np.ndarray:
    """An ibex following the dominant members of its herd is modelled as stepping toward a blend of the herd's best.

    The `dominants` top-ranked ibex carry dominance weights `w_j` that fall with rank (`nio.herd.dominance`: the leader
    weighs most). Each follower mixes them with its own random attention,

        `B = Σ_j w_j u_j x_(j) / Σ_j w_j u_j`,   u_j ~ U(0, 1],

    leaves itself out of its own blend, and steps toward it: `x_i + φ ⊙ (B − x_i)`, φ ~ U(0, 2·reach) per dimension.
    With `reach` 1 it lands, on average, on B, scattered by its own distance to B, so the herd tightens around its
    dominants as it gathers.

    Edge cases: `dominants` is clamped to between 1 and `agents − 1`. The leader, with no one above it, follows the
    runner-up; a follower outside the dominants ignores the spare ibex drawn in case it was itself a dominant. In a
    herd of two each ibex follows the other. With `reach` 0 the proposal is the ibex's own position.
    """
    agents, dimensions = herd.positions.shape
    x = herd.positions[who]
    count = min(max(1, dominants), agents - 1)
    top = herd.order[: count + 1]  # one spare in case the follower is itself a dominant
    attention = dominance(count + 1)[None, :] * (1.0 - rng.random((len(who), count + 1)))  # u in (0, 1]
    own = top[None, :] == who[:, None]
    attention[own] = 0.0
    spare = ~own.any(axis=1)  # a follower outside the dominants ignores the spare
    attention[spare, count] = 0.0
    attention /= attention.sum(axis=1, keepdims=True)
    blend = attention @ herd.positions[top]
    return x + rng.uniform(0.0, 2.0 * reach, (len(who), dimensions)) * (blend - x)
