"""Low-discrepancy initializers: Latin hypercube, Sobol and Halton points from `scipy.stats.qmc`.

scipy is needed only when one of these draws a herd (`pip install scipy`); importing the module does not load it.
Each sequence is scrambled with a seed drawn from the run's generator, so a seeded run starts from the same herd.
"""

import warnings
from abc import abstractmethod
from dataclasses import dataclass

from nio.initializers.base import Initializer


def _qmc():
    try:
        from scipy.stats import qmc
    except ImportError as error:
        raise ImportError("The quasi-random initializers need scipy: pip install scipy.") from error
    return qmc


class QuasiRandom(Initializer):
    """A low-discrepancy sequence scaled to the box."""

    @abstractmethod
    def engine(self, dimensions: int, seed: int):
        """The scipy QMC engine that draws the points."""

    def sample(self, space, agents, rng):
        engine = self.engine(space.dimensions, int(rng.integers(2**32)))
        return _qmc().scale(engine.random(agents), space.lb, space.ub)


@dataclass(frozen=True)
class LatinHypercube(QuasiRandom):
    """One point in every row and column of an agents × agents grid, per pair of dimensions."""

    name = "Latin Hypercube"

    def engine(self, dimensions, seed):
        return _qmc().LatinHypercube(d=dimensions, rng=seed)


@dataclass(frozen=True)
class Sobol(QuasiRandom):
    """A scrambled Sobol sequence."""

    name = "Sobol Sequence"
    scramble: bool = True

    def engine(self, dimensions, seed):
        return _qmc().Sobol(d=dimensions, scramble=self.scramble, rng=seed)

    def sample(self, space, agents, rng):
        """A herd is rarely a power of two, so scipy's balance warning is expected and silenced here only."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The balance properties of Sobol' points")
            return super().sample(space, agents, rng)


@dataclass(frozen=True)
class Halton(QuasiRandom):
    """A scrambled Halton sequence."""

    name = "Halton Sequence"
    scramble: bool = True

    def engine(self, dimensions, seed):
        return _qmc().Halton(d=dimensions, scramble=self.scramble, rng=seed)
