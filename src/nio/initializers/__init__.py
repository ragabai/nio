"""Initializers: how the first herd is scattered over the box.

NIO starts from `RandomUniform()` unless it is given another (`NIO(..., initializer=Sobol())`). `INITIALIZERS` holds
one of each by name, and `get` finds one ignoring case:

    >>> from nio.initializers import get
    >>> get("halton sequence").name
    'Halton Sequence'
"""

from nio.initializers.base import Initializer
from nio.initializers.deterministic import Fibonacci, SineCosine
from nio.initializers.hybrid import Hybrid
from nio.initializers.quasi import Halton, LatinHypercube, QuasiRandom, Sobol
from nio.initializers.stochastic import Centroid, Gaussian, Opposition, RandomUniform, Sphere

INITIALIZERS: dict[str, Initializer] = {
    initializer.name: initializer
    for initializer in (
        RandomUniform(),
        LatinHypercube(),
        Sobol(),
        Opposition(),
        Halton(),
        Sphere(),
        Centroid(),
        SineCosine(),
        Hybrid(),
        Fibonacci(),
    )
}


def get(name: str) -> Initializer:
    """An initializer of `INITIALIZERS` by name, ignoring case and surrounding spaces."""
    wanted = name.strip().lower()
    for key, initializer in INITIALIZERS.items():
        if key.lower() == wanted:
            return initializer
    raise ValueError(f"Unknown initializer {name!r}. Choose from {', '.join(INITIALIZERS)}.")


__all__ = [
    "INITIALIZERS",
    "Centroid",
    "Fibonacci",
    "Gaussian",
    "Halton",
    "Hybrid",
    "Initializer",
    "LatinHypercube",
    "Opposition",
    "QuasiRandom",
    "RandomUniform",
    "SineCosine",
    "Sobol",
    "Sphere",
    "get",
]
