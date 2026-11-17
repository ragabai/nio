"""Return to spring: going back to the best places the herd has found."""

import numpy as np

from nio.herd import Herd, dominance


def return_to_spring(herd: Herd, who: np.ndarray, rng: np.random.Generator, scatter: float) -> np.ndarray:
    """An ibex going back to known water is modelled as returning to one of the best places the herd has found.

    Drought affects Nubian ibex populations (tichon2023population), so water is the natural picture for a place worth
    remembering. The herd's springs (`nio.springs.Springs`: the best distinct finite positions ever scored, best first)
    carry dominance weights that fall with rank (`nio.herd.dominance`: the best spring is visited most). Each ibex picks
    a spring s better than the ground it stands on (strictly lower fitness) and arrives around it, scattered by how
    far it came,

        `s + scatter · |x_i − s| ⊙ z`,   z ~ N(0, I),

    so an ibex far from the water lands loosely around it and one already close lands close. The springs outlive the
    ibex that found them (an ibex that left for better ground, a regrouping), so the herd can go back to places no
    ibex stands on now.

    Edge cases: an ibex with no better spring (it stands on the best water already, or every better spring is exactly
    where it stands) arrives around the best spring scattered by the herd's spread σ instead, so it never proposes its
    own position. While no finite spring exists yet (every score so far is +inf or NaN) the herd's best position
    stands in for the spring.
    """
    x = herd.positions[who]
    if len(herd.springs.fitness):
        springs, water = herd.springs.positions, herd.springs.fitness
    else:
        springs, water = herd.positions[herd.order[:1]], np.full(1, np.inf)
    weights = dominance(len(springs))
    out = np.empty_like(x)
    for k, (xi, fi) in enumerate(zip(x, herd.fitness[who], strict=True)):
        better = (water < fi) & np.any(springs != xi, axis=1)  # better water, and not where it stands
        p = np.where(better, weights, 0.0)
        if p.sum() > 0:
            s = springs[rng.choice(len(springs), p=p / p.sum())]
            spread = np.abs(xi - s)
        else:
            s, spread = springs[0], herd.spread()
        out[k] = s + scatter * spread * rng.standard_normal(len(xi))
    return out
