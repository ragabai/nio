"""Walk the ledges: a step shaped like the ground the herd's best stand on."""

import numpy as np

from nio.herd import Herd, dominance


def ledge_shape(herd: Herd, share: float) -> np.ndarray:
    """The run of the ledge under the herd's best: a D × m factor A of the rank-weighted scatter of its `share` best
    ibex (m of them, at least two), A·Aᵀ = Σ_j w_j (x_(j) − c)(x_(j) − c)ᵀ, with `w` the dominance weights
    (`nio.herd.dominance`) and c = Σ_j w_j x_(j) their rank-weighted centre."""
    agents = len(herd.fitness)
    count = min(agents, max(2, round(share * agents)))
    top = herd.positions[herd.order[:count]]
    w = dominance(count)
    return ((top - w @ top) * np.sqrt(w)[:, None]).T


def ledge_walk(herd: Herd, who: np.ndarray, rng: np.random.Generator, share: float, stride: float) -> np.ndarray:
    """An ibex walking the ledges is modelled as stepping along the shelf of rock its herd's best stand on.

    In the model the best feeding on a cliff runs along ledges, and the herd's best ibex stand strung out along them,
    so the shape of where they stand shows the run of the ledge: long along it, narrow across it. The walker steps
    from where it stands with a stride of that shape,

        `x_i + stride · A u`,   u ~ N(0, I_m),

    where A·Aᵀ is the rank-weighted scatter of the `share` best ibex (`ledge_shape`); A u is a random blend of their
    offsets from their centre, so the step has their scatter's covariance without any matrix being factored. On a
    narrow curved or tilted valley the step runs along the valley, whichever way it lies, and it shrinks as the best
    ibex gather. Each walker keeps its own place, so the herd keeps its spread. Only differences between positions are
    used, so the behaviour translates and rotates with the problem.

    The optimizer lets the herd walk the ledges late in the run, once its best have found them: route preference
    scales the behaviour by `τ^LEDGE_RISE` (`Settings.LEDGE_RISE`). The behaviour sees the whole herd, not a subgroup.

    Edge cases: the best ibex are at least two (clamped to the herd). On a herd whose best stand on one spot, or with
    `stride` 0, the proposal is the walker's own position.
    """
    shape = ledge_shape(herd, share)
    return herd.positions[who] + stride * rng.standard_normal((len(who), shape.shape[1])) @ shape.T
