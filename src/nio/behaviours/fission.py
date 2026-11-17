"""Fission–fusion: the herd's split into subgroups, each with its own leader."""

import math

import numpy as np

from nio.herd import Herd


class FissionFusion:
    """The herd splits into subgroups, each following its own leader, and regroups now and then.

    Female and male Siberian ibex have different environmental requirements (han2021different). The model reads this as
    ibex living in separate groups, and fission–fusion as the herd's way of splitting into groups that explore
    separately and fusing again; the rut (`nio.behaviours.rut`) is where the groups meet.

    At the start of each season the herd has `groups_at(τ)` subgroups: `most = min(groups, agents // 2)` (so every
    subgroup has at least 2 ibex) at τ = 0, falling as `⌈most · (1 − τ / fusion)⌉` to a single herd by τ = `fusion`,
    so the late season contracts around one leader. The herd regroups, with fresh random membership in near-equal
    subgroups (`Herd.regroup`), when

    * the number of subgroups changes (fusion over the season), or
    * a subgroup stalls: its best fitness has not improved for `patience` seasons, or
    * a subgroup has gathered: every member stands within `gather · R` of its leader, where R is the herd's reach, the
      largest distance of any ibex from the herd's best.

    Each subgroup's leader is its best member (`Herd.leaders`). The group behaviours (browse, follow_dominant,
    zigzag_climb) see only the ibex's own subgroup (`Herd.within`), so several leaders pull the herd toward several
    places at once instead of all of it collapsing onto one ibex early. Regrouping mixes ibex from different places, so
    what one subgroup found reaches the others. Membership is random and every test is on fitness or on distances
    between ibex, so translating the problem translates the run. The springs live on the herd, not on a subgroup, so
    they carry across every regrouping.

    Edge cases: with R = 0 (the whole herd on one spot) no subgroup counts as gathered; a herd of fewer than 4 ibex
    has a single group; after the herd is thinned (`resize`) the subgroup count is capped by the new herd size and the
    herd regroups at the next season.
    """

    def __init__(self, agents: int, groups: int, patience: int, gather: float, fusion: float):
        self.most = max(1, min(int(groups), agents // 2))
        self.patience, self.gather, self.fusion = patience, gather, fusion
        self.best = np.empty(0)
        self.stall = np.empty(0, dtype=int)

    def resize(self, agents: int, groups: int) -> None:
        """The herd now has `agents` ibex (herd thinning): at most `agents // 2` subgroups from now on, and the herd
        regroups at the next season."""
        self.most = max(1, min(int(groups), agents // 2))
        self.best = np.empty(0)
        self.stall = np.empty(0, dtype=int)

    def groups_at(self, tau: float) -> int:
        """How many subgroups the herd has at season τ."""
        if self.most == 1 or tau >= self.fusion:
            return 1
        return max(1, min(self.most, math.ceil(self.most * (1 - tau / self.fusion))))

    def regroup(self, herd: Herd, count: int, rng: np.random.Generator) -> None:
        """Split the herd into `count` subgroups of near-equal size, with random membership."""
        group = np.empty(len(herd.fitness), dtype=int)
        group[rng.permutation(len(group))] = np.arange(len(group)) % count
        herd.regroup(group)
        self.track(herd)

    def track(self, herd: Herd) -> None:
        """Start watching the herd's current subgroups for stalls."""
        self.best = np.array([herd.fitness[herd.members(g)].min() for g in range(herd.groups)])
        self.stall = np.zeros(herd.groups, dtype=int)

    def season(self, herd: Herd, tau: float, rng: np.random.Generator) -> bool:
        """At the start of a season: regroup if the count changes, a subgroup stalled, or a subgroup gathered. Returns
        whether the herd regrouped."""
        count = self.groups_at(tau)
        if count != herd.groups or len(self.best) != herd.groups:
            self.regroup(herd, count, rng)
            return True
        if count == 1:
            return False
        best = np.array([herd.fitness[herd.members(g)].min() for g in range(count)])
        improved = best < self.best
        self.best = np.where(improved, best, self.best)
        self.stall = np.where(improved, 0, self.stall + 1)
        if np.any(self.stall >= self.patience) or self._gathered(herd):
            self.regroup(herd, count, rng)
            return True
        return False

    def _gathered(self, herd: Herd) -> bool:
        positions, reach = herd.positions, herd.reach()
        if reach <= 0:
            return False
        for g, leader in enumerate(herd.leaders()):
            members = herd.members(g)
            if np.max(np.linalg.norm(positions[members] - positions[leader], axis=1)) <= self.gather * reach:
                return True
        return False
