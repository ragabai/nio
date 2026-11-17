"""An initializer that splits the first herd between several others."""

from dataclasses import dataclass, field

import numpy as np

from nio.initializers.base import Initializer
from nio.initializers.quasi import Halton
from nio.initializers.stochastic import Gaussian, RandomUniform


@dataclass(frozen=True)
class Hybrid(Initializer):
    """Equal shares of the herd from each of `parts`, the last taking the remainder (uniform, Halton, Gaussian)."""

    name = "Hybrid"
    parts: tuple[Initializer, ...] = field(default_factory=lambda: (RandomUniform(), Halton(), Gaussian()))

    def sample(self, space, agents, rng):
        share = agents // len(self.parts)
        sizes = [share] * (len(self.parts) - 1) + [agents - share * (len(self.parts) - 1)]
        return np.vstack([part(space, size, rng) for part, size in zip(self.parts, sizes, strict=True) if size])
