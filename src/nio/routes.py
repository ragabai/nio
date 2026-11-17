"""Learned route preference: which behaviour each ibex tries."""

import numpy as np


class Routes:
    """What the herd has learned about its moves: each ibex picks a behaviour with probability proportional to that
    behaviour's recent success rate.

    Rates are exponentially smoothed by `memory`, `rate = memory·rate + (1 − memory)·hits/tries`; every behaviour
    keeps at least probability `floor`. The draws come from the generator given to `choose`, never from a global one.
    """

    def __init__(self, count: int, floor: float, memory: float):
        self.count = count
        self.floor = floor
        self.memory = memory
        self.rates = np.full(self.count, 0.5)  # every behaviour starts at the same success rate, 0.5

    def probabilities(self) -> np.ndarray:
        """`floor + (1 − count·floor)·rate / Σ rate`, one probability per behaviour; they sum to 1."""
        return self.floor + (1 - self.count * self.floor) * self.rates / self.rates.sum()

    def choose(self, agents: int, rng: np.random.Generator, weights: np.ndarray | None = None) -> np.ndarray:
        """A behaviour index for each of `agents` ibex. `weights` (one per behaviour, ≥ 0, not all 0) scale the learned
        probabilities before they are renormalised, e.g. to fade the sentinels' share over the season."""
        p = self.probabilities()
        if weights is not None:
            p = p * weights
            p = p / p.sum()
        return rng.choice(self.count, size=agents, p=p)

    def reward(self, hits: np.ndarray, tries: np.ndarray) -> None:
        """Update the success rates of the behaviours tried this season (a rate never falls below 1e-3)."""
        used = tries > 0
        self.rates[used] = self.memory * self.rates[used] + (1 - self.memory) * hits[used] / tries[used]
        self.rates = np.maximum(self.rates, 1e-3)

    def join(self) -> None:
        """A new behaviour joins the repertoire as the last one, at the starting success rate (0.5); the behaviours
        already learned keep their rates."""
        self.count += 1
        self.rates = np.append(self.rates, 0.5)
