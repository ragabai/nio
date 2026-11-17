"""Browse: small steps around where an ibex stands."""

import numpy as np

from nio.herd import Herd


def browse(herd: Herd, who: np.ndarray, rng: np.random.Generator, step: float) -> np.ndarray:
    """An ibex browsing on good ground is modelled as taking small, careful steps around where it stands.

    Formula: `x_i + step · σ ⊙ z`, with z ~ N(0, I) and σ the herd's current spread per dimension (the standard
    deviation of the positions). `step` is the browse step of the season; `Settings.BROWSE_STEP` and
    `Settings.BROWSE_STEP_LATE` make it fall over the run. The step lives on the herd's own scale: wide while the herd
    is scattered, fine once it has gathered.

    Edge cases: in a dimension where every ibex stands at the same coordinate σ is 0, so the ibex does not move
    there (inside a subgroup, `Herd.within` lends the whole herd's spread to such a dimension); with `step` 0 the
    proposal is the ibex's own position.
    """
    x = herd.positions[who]
    return x + step * herd.spread() * rng.standard_normal(x.shape)
