"""The herd: the ibex of one season, ranked by fitness, with their memory of the best ground."""

import functools
import math

import numpy as np

from nio.springs import Springs


def _std(positions: np.ndarray) -> np.ndarray:
    """Per-dimension standard deviation of the rows, computed exactly as `np.std(positions, axis=0)` does (the same
    operations in the same order), without its overhead."""
    count = len(positions)
    deviation = positions - np.add.reduce(positions, axis=0, keepdims=True) / count
    np.square(deviation, out=deviation)
    return np.sqrt(np.add.reduce(deviation, axis=0) / count)


@functools.lru_cache(maxsize=512)
def dominance(count: int) -> np.ndarray:
    """Dominance weights of the `count` top-ranked ibex, best first: `w_j ∝ ln(count + 1) − ln(j + 1)`,
    j = 0 … count − 1, normalised to sum to 1. The weights are cached per count and returned read-only.

    They come from rank only, never from fitness values, so they are finite for a flat or +inf herd; they fall strictly
    with rank (the leader weighs most, the last dominant least), and a single dominant takes the whole weight.
    """
    weights = math.log(count + 1) - np.log(np.arange(1, count + 1))
    weights = weights / weights.sum()
    weights.flags.writeable = False
    return weights


class Herd:
    """The ibex of one season, ranked by fitness (`order`, best first; ties and +inf keep their index order).

    `elite` and `elite_fit` are the best ibex ever seen; `stall` counts, per ibex, the seasons since it last moved to
    better ground. `offer` is the greedy replacement: an ibex moves only to a strictly better proposal. Every scored
    position, accepted or not, is offered to the herd's `springs` (by default an archive of 5 with no gap).

    Fission–fusion: `group` holds each ibex's subgroup (0 … `groups` − 1; one group at first), set by `regroup`.
    `leaders()` are the subgroups' best ibex, and `within(i)` is ibex i's own subgroup seen as a herd of its own, which
    the group moves (browse, follow, zigzag) use, so each subgroup follows its own leader.
    """

    def __init__(self, positions: np.ndarray, fitness: np.ndarray, springs: Springs | None = None):
        self.positions = positions
        self.fitness = np.asarray(fitness, dtype=float)
        self.stall = np.zeros(len(self.fitness), dtype=int)
        self.elite, self.elite_fit = None, math.inf
        self._spread = self._reach = self._scale = self._ranks = None
        self._rank()
        self.springs = Springs(5) if springs is None else springs
        self.springs.offer(self.positions, self.fitness)
        self.regroup(np.zeros(len(self.fitness), dtype=int))

    def regroup(self, group: np.ndarray) -> None:
        """Set each ibex's subgroup, numbered 0 … groups − 1 with none empty."""
        self.group = np.asarray(group, dtype=int)
        self.groups = int(self.group.max()) + 1
        self._members = [np.flatnonzero(self.group == g) for g in range(self.groups)]
        self._outside = [np.flatnonzero(self.group != g) for g in range(self.groups)]
        self._views = {}

    def members(self, g: int) -> np.ndarray:
        """Indices of the ibex of subgroup g."""
        return self._members[g]

    def outside(self, g: int) -> np.ndarray:
        """Indices of the ibex of every subgroup but g."""
        return self._outside[g]

    def leaders(self) -> np.ndarray:
        """Each subgroup's leader, its best ibex (ties keep index order), in group order."""
        _, first = np.unique(self.group[self.order], return_index=True)
        return self.order[first]

    def within(self, i: int) -> tuple["Herd", int]:
        """Ibex i's subgroup as a herd of its own (copies of its members' positions and fitness, ranked), and i's index
        in it. With one group it is the herd itself. Where the subgroup has no spread in a dimension (all its ibex
        agree there) it borrows the whole herd's, so browsing never stands still."""
        if self.groups == 1:
            return self, int(i)
        g = self.group[i]
        members = self._members[g]
        if g in self._views:  # unchanged since it was built: no ibex has moved and the herd has not regrouped
            return self._views[g], int(np.searchsorted(members, i))
        view = Herd.__new__(Herd)
        view.positions, view.fitness, view.stall = self.positions[members], self.fitness[members], self.stall[members]
        view.order = np.argsort(view.fitness, kind="stable")
        view.elite, view.elite_fit = view.positions[view.order[0]], float(view.fitness[view.order[0]])
        spread = _std(view.positions)
        view._spread = np.where(spread > 0, spread, self.spread())
        view._reach = view._scale = view._ranks = None
        view.springs = self.springs
        view.group, view.groups = np.zeros(len(members), dtype=int), 1  # as regroup() sets one group
        view._members, view._outside, view._views = [np.arange(len(members))], [np.empty(0, dtype=np.intp)], {}
        self._views[g] = view
        return view, int(np.searchsorted(members, i))

    def _rank(self) -> None:
        self.order = np.argsort(self.fitness, kind="stable")
        self._ranks = None
        best = self.order[0]
        if self.elite is None or self.fitness[best] < self.elite_fit:
            self.elite, self.elite_fit = self.positions[best].copy(), float(self.fitness[best])

    @property
    def best(self) -> int:
        """Index of the herd's best ibex this season."""
        return int(self.order[0])

    @property
    def best_fit(self) -> float:
        """The fitness of the herd's best ibex this season."""
        return float(self.fitness[self.order[0]])

    def ranks(self) -> np.ndarray:
        """Each ibex's rank, 0 for the best (cached until the herd is re-ranked; read-only)."""
        if self._ranks is None:
            ranks = np.empty(len(self.order), dtype=int)
            ranks[self.order] = np.arange(len(self.order))
            ranks.flags.writeable = False
            self._ranks = ranks
        return self._ranks

    def spread(self) -> np.ndarray:
        """How scattered the herd is in each dimension: the standard deviation of its positions."""
        if self._spread is None:
            self._spread = _std(self.positions)
        return self._spread

    def reach(self) -> float:
        """How far the herd ranges: the largest distance of any ibex from the herd's best."""
        if self._reach is None:
            gap = self.positions - self.positions[self.best]
            self._reach = float(np.sqrt(np.add.reduce(gap * gap, axis=1)).max())  # as np.linalg.norm(gap, axis=1)
        return self._reach

    def scale(self) -> float:
        """The herd's fitness range, best to worst over its finite fitness values (0 when there are none)."""
        if self._scale is None:
            finite = self.fitness[np.isfinite(self.fitness)]
            self._scale = float(finite.max() - finite.min()) if finite.size else 0.0
        return self._scale

    def offer(self, who: np.ndarray, proposals: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Greedy replacement: ibex `who[k]` moves to `proposals[k]` only if `scores[k]` is strictly better. Returns
        which moved, and re-ranks the herd."""
        if len(scores) == 1:
            return self._offer_one(int(who[0]), proposals, scores)
        if np.fmin.reduce(scores) < self.springs.bar:  # NaN-blind minimum: a NaN never hides a good score
            self.springs.offer(proposals, scores)
        accepted = scores < self.fitness[who]
        moved = who[accepted]
        self.positions[moved] = proposals[accepted]
        self.fitness[moved] = scores[accepted]
        self.stall[who] += 1
        self.stall[moved] = 0
        if moved.size:
            self._spread = self._reach = self._scale = None
            self._views = {}
            self._rank()
        return accepted

    def thin(self, count: int) -> np.ndarray:
        """Herd thinning: only the `count` best-ranked ibex stay (ties keep index order), in their index order; the
        weakest leave. Returns the indices (before thinning) of the ibex that stayed. Each keeps its place, fitness and
        stall count; the elite and the springs are the herd's memory and stay as they were. The herd is one group
        again until fission–fusion regroups it."""
        keep = np.sort(self.order[:count])
        self.positions, self.fitness, self.stall = self.positions[keep], self.fitness[keep], self.stall[keep]
        self._spread = self._reach = self._scale = None
        self._rank()
        self.regroup(np.zeros(len(keep), dtype=int))
        return keep

    def retaste(self, i: int, score: float) -> None:
        """Re-check: ibex i has tasted the ground it stands on again and trusts the fresh taste `score`, better or worse
        than the one it remembered. It keeps its place, its patience restarts (`stall` 0), the herd re-ranks, and the
        taste is offered to the springs like every scored position. The best taste ever stays the elite. A non-finite
        taste (a spent budget scores +inf) is not trusted: only the patience restarts."""
        if score < self.springs.bar:  # a NaN or +inf fails
            self.springs.offer(self.positions[i : i + 1], np.array([score]))
        self.stall[i] = 0
        if not math.isfinite(score):
            return
        self.fitness[i] = score
        self._spread = self._reach = self._scale = None
        self._views = {}
        self._rank()

    def _offer_one(self, i: int, proposals: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """`offer` for a single ibex i (one proposal row, one score): the same updates, without the array overhead."""
        score = scores[0]
        if score < self.springs.bar:  # a NaN fails, as in the NaN-blind minimum of `offer`
            self.springs.offer(proposals, scores)
        self.stall[i] += 1
        if not score < self.fitness[i]:
            return np.zeros(1, dtype=bool)
        self.positions[i] = proposals[0]
        self.fitness[i] = score
        self.stall[i] = 0
        self._spread = self._reach = self._scale = None
        self._views = {}
        self._rank()
        return np.ones(1, dtype=bool)
