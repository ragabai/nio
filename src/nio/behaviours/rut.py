"""Rut: crossing two ibex from different subgroups."""

import numpy as np

from nio.herd import Herd


def rut(herd: Herd, who: np.ndarray, rng: np.random.Generator, mix: float, step: float) -> np.ndarray:
    """The breeding season is modelled as a meeting of the separate groups: rivals contest, and the young take after
    both parents.

    Female and male Siberian ibex have different environmental requirements (han2021different), which the model reads
    as separate groups living apart; the rut is where they mix.

    Mate: two ibex from other subgroups than ibex i's (the whole herd but i when there is one group) contest, and the
    better one (lower fitness; ties to the first) is i's mate j. Mix: each coordinate of the young comes from j with
    probability `mix` and otherwise from i, with at least one coordinate from each parent when D ≥ 2. Step: the young
    then takes a short step toward the better parent b,

        `c + η ⊙ (x_b − c)`,   η ~ U(0, step) per dimension,

    where c is the mixed position. The step moves only the coordinates taken from the worse parent. Swapping coordinates
    and stepping along a difference both move with the problem when it is translated.

    Edge cases: in D = 1 the young's coordinate is j's. With `step` 0 the young is the mix itself. With one group the
    rivals are drawn from every ibex but i, so a lone herd still has a mate.
    """
    agents, dimensions = herd.positions.shape
    x = herd.positions[who]
    if herd.groups > 1:
        rivals = np.array([(pool := herd.outside(herd.group[i]))[rng.integers(len(pool), size=2)] for i in who])
    else:  # one herd: any ibex but i
        rivals = rng.integers(agents - 1, size=(len(who), 2))
        rivals += rivals >= who[:, None]
    mate = np.where(herd.fitness[rivals[:, 1]] < herd.fitness[rivals[:, 0]], rivals[:, 1], rivals[:, 0])

    taken = rng.random(x.shape) < mix
    rows, given = np.arange(len(who)), rng.integers(dimensions, size=len(who))
    own = (given + 1 + rng.integers(max(1, dimensions - 1), size=len(who))) % dimensions  # ≠ given when D ≥ 2
    taken[rows, own] = False
    taken[rows, given] = True
    young = np.where(taken, herd.positions[mate], x)

    better = np.where(herd.fitness[mate] < herd.fitness[who], mate, who)
    return young + rng.uniform(0.0, step, x.shape) * (herd.positions[better] - young)
