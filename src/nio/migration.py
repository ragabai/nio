"""Migration: a herd that has stopped finding better ground leaves for a fresh range and starts a new year there."""

import math


class Migration:
    """When the herd migrates, and the season τ of the year it is living through.

    A herd that has gathered on one patch of ground keeps tasting it: every move costs an evaluation and none finds
    better ground, so the rest of the budget would be spent where the herd already is. In the model the herd then
    migrates. It watches its best ibex each season; once the best has not improved for `patience` seasons in a row,
    the herd leaves for a fresh range (the optimizer scatters a new herd across the box) and lives a new year there,
    from exploring to exploiting again, over what is left of the run. The springs, the best places it has found, go
    with it as the herd's memory, and so does the elite, the best ibex ever seen.

    The run's clock (`clock`) runs from 0 at the start of the run to 1 at its end. A year that starts at clock c0
    has its own season τ = (clock − c0) / (1 − c0), from 0 when the herd arrives to 1 at the end of the run, so each
    year exploits by the end of the run and a later year is shorter.

    The herd migrates only before the clock reaches `until`, so the last year is long enough to gather the herd; with
    `patience` 0 it never migrates, and the run is one year. It never migrates on noisy terrain either, where a best
    that does not improve says nothing about the ground (the optimizer decides that).
    """

    def __init__(self, patience: int, until: float):
        self.patience, self.until = int(patience), float(until)
        self.start = 0.0  # the clock when the herd arrived on its current range
        self.best = math.inf  # the best fitness of the current herd so far
        self.stall = 0  # seasons since the current herd's best last improved
        self.count = 0  # migrations so far

    def watch(self, best: float) -> None:
        """At a season's start: note the herd's best fitness, and count the seasons it has not improved."""
        if best < self.best:
            self.best, self.stall = best, 0
        else:
            self.stall += 1

    def due(self, clock: float) -> bool:
        """Whether the herd migrates now: its best has stalled `patience` seasons and the clock is before `until`."""
        return 0 < self.patience <= self.stall and clock < self.until

    def arrive(self, clock: float, best: float) -> None:
        """The herd has migrated: a new year starts at `clock`, with the fresh herd's best fitness."""
        self.start, self.best, self.stall = clock, best, 0
        self.count += 1

    def season(self, clock: float) -> float:
        """The season τ of the current year at `clock`, in [0, 1]."""
        if self.start <= 0 or self.start >= 1:
            return min(1.0, max(0.0, clock))
        return min(1.0, max(0.0, (clock - self.start) / (1.0 - self.start)))
