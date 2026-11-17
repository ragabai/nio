"""The springs: the few best distinct places the herd has found, modelled as the herd's memory of water."""

import math

import numpy as np


class Springs:
    """An archive of at most `size` springs: the best distinct finite positions ever scored, best first.

    Two positions are the same spring when they lie within `gap` of each other in every dimension (`gap` is a
    per-dimension distance, set by NIO as `SPRING_GAP · span`, so it moves with the terrain). Offering a position:

    * a non-finite fitness (+inf from a spent budget, NaN) is never archived;
    * if a spring within the gap is at least as good, the position is the same water, and is dropped;
    * otherwise it replaces every spring within its gap (all worse), and joins the archive if there is room or it beats
      the worst spring, which then dries up.

    So no two springs ever lie within the gap of each other, and every spring is finite. The archive lives on the herd,
    not on a subgroup, so it is carried across every regrouping.
    """

    def __init__(self, size: int, gap: np.ndarray | float = 0.0):
        self.size = max(1, int(size))
        self.gap = gap
        self.positions = None
        self.fitness = np.empty(0)
        self.bar = math.inf  # a position must score strictly below this to be accepted: the worst spring once full

    def offer(self, positions: np.ndarray, fitness: np.ndarray) -> None:
        """Offer scored positions (rows) to the archive."""
        fitness = np.asarray(fitness, dtype=float).reshape(-1)
        positions = np.asarray(positions, dtype=float).reshape(len(fitness), -1)
        if self.positions is None:
            self.positions = np.empty((0, positions.shape[1]))
        for x, f in zip(positions, fitness.tolist(), strict=True):
            if not f < self.bar:
                continue  # +inf and NaN never pass; once full, nothing that is no better than the worst spring
            near = (np.abs(self.positions - x) <= self.gap).all(axis=1)
            if near.any():
                if (self.fitness[near] <= f).any():
                    continue  # the same water as a spring at least as good
                self.positions, self.fitness = self.positions[~near], self.fitness[~near]
            spot = int(np.searchsorted(self.fitness, f, side="right"))
            self.positions = np.concatenate((self.positions[:spot], x[None, :], self.positions[spot:]))[: self.size]
            self.fitness = np.concatenate((self.fitness[:spot], [f], self.fitness[spot:]))[: self.size]
            self.bar = float(self.fitness[-1]) if len(self.fitness) == self.size else math.inf
