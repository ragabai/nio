"""What the herd did during a run, season by season."""

from dataclasses import dataclass

import numpy as np

from nio.springs import Springs


@dataclass(frozen=True, eq=False)
class Diagnostics:
    """How a run of NIO went, set by `NIO.solve()` as `NIO.diagnostics`. Read-only: the fields cannot be assigned and
    the arrays cannot be written.

    * `used` (seasons × behaviours, int): how many ibex used each behaviour in each season, in the order of
      `nio.behaviours.BEHAVIOURS`. Forced cliff escapes count; re-checks of a spot on noisy terrain do not.
    * `groups` (seasons, int): the number of subgroups in each season.
    * `sizes` (seasons, int): the number of ibex in the herd in each season, after thinning.
    * `regroups`: how many times the herd regrouped (fission–fusion).
    * `springs`: the spring archive at the end of the run, the best distinct positions ever scored.
    * `noisy`: whether the two first tastes of the herd's best spot differed, so the terrain was taken as noisy.
    * `rechecks`: how many scored re-checks the ibex made of their own spot (a re-check after the budget is spent
      scores nothing and does not count).
    * `migrations`: how many times the herd migrated (`nio.migration`), each time starting a new year with a fresh
      herd.
    """

    used: np.ndarray
    groups: np.ndarray
    sizes: np.ndarray
    regroups: int
    springs: Springs
    noisy: bool
    rechecks: int
    migrations: int = 0

    def __post_init__(self) -> None:
        for array in (self.used, self.groups, self.sizes):
            array.setflags(write=False)
