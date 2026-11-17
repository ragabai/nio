"""NIO's settings: every tunable number of the herd's behaviour, in one frozen record."""

from dataclasses import dataclass, fields
from dataclasses import replace as _replace
from typing import Self

import numpy as np

from nio.behaviours import BEHAVIOURS


@dataclass(frozen=True)
class Settings:
    """The upper-case settings of NIO, with the default of each. A `Settings` never changes; `replace` gives a changed
    copy.

    Behaviour steps (the season τ runs from 0 at the start of the run to 1 at its end):

    * `BROWSE_STEP` (1.0): browse step, in herd spreads, at the start of the season.
    * `BROWSE_STEP_LATE` (0.1): browse step at the end of the season; it falls linearly in τ between the two.
    * `DOMINANTS` (0.2): share of the herd followed as dominants at the start; the count falls linearly to 1 by the end.
    * `FOLLOW_REACH` (1.0): how far a follower goes toward its blend of dominants (1: lands around it on average).
    * `ZIGZAG_STRIDE` (1.0): mean length of a switchback leg, as a fraction of the distance to the target ibex.
    * `CLIFF_REACH` (0.05): scale of a cliff leap, as a fraction of the terrain's span (Cauchy-distributed).
    * `CLIFF_LEDGES` (2.0): each dimension joins a leap with probability CLIFF_LEDGES / D, and one more is always
      chosen, so a leap crosses 1 + CLIFF_LEDGES × (D − 1) / D dimensions on average (about 3 for the default).
    * `CLIFF_PATIENCE` (0): seasons without improving after which an ibex (not a subgroup's leader) is forced to
      escape by a cliff leap, once: then it settles and its patience restarts (0: no ibex is ever forced; the herd
      still leaps by choice, through route preference).
    * `SENTINEL_LOOKOUTS` (1.0): mean number of dimensions a sentinel redraws across the whole bounds (at least one).
    * `SENTINEL_FADE` (3.0): the sentinels' route preference is scaled by `(1 − τ)^SENTINEL_FADE`.
    * `SPRINGS` (5): most springs (best distinct positions ever scored) the herd remembers, at least 1. It is a size,
      not a switch: a bool is refused.
    * `SPRING_GAP` (0.01): two positions within this share of the span in every dimension are the same spring.
    * `SPRING_SCATTER` (1.0): an ibex returning to a spring lands around it, scattered by this times its distance to it.
    * `RUT_MIX` (0.5): share of the young's coordinates taken from the mate from another subgroup.
    * `RUT_STEP` (0.5): the young steps toward the better parent by η ~ U(0, RUT_STEP) per dimension.

    Subgroups (fission and fusion):

    * `FISSION_GROUPS` (3): subgroups at the start of the run (at most agents // 2, so each has two ibex or more).
    * `FUSION_SEASON` (0.7): τ by which the subgroups have fused into one herd (their count falls linearly until then).
    * `FUSION_PATIENCE` (10): seasons a subgroup's best may go without improving before the herd regroups.
    * `FUSION_GATHER` (0.05): the herd regroups when a subgroup has gathered within this share of the herd's reach
      around its leader.

    Moves that use the search box (`home_range` pulls toward the centre of the box; `ridge_walk` steps along the line
    through the origin along (1, ..., 1), on which every coordinate is equal; it is the box's main diagonal only when
    every dimension has the same bounds. That direction is absolute, so only `ridge_walk` breaks translation
    invariance). Both are always in route preference:

    * `HOME_DEPTH` (8.0): a home-range visit leaves a share 10^(−HOME_DEPTH·U) of the distance to the centre, per
      dimension (HOME_DEPTH / 2 orders of magnitude closer on average).
    * `HOME_RISE` (0.5): seasonal home range; the `home_range` route preference is scaled by `τ^HOME_RISE` (0: the
      same all year).
    * `HOME_FULL` (0.5): in the model an ibex's pull toward home grows over the year: a visit's depth is
      `HOME_DEPTH · τ / HOME_FULL` until τ = `HOME_FULL`, then `HOME_DEPTH` (0: full depth all year). Early visits only
      drift part of the way home.
    * `RIDGE_STRIDE` (3.0): the ridge walk starts from the leader's projection onto the line, and its leg is
      RIDGE_STRIDE × the herd's mean spread × N(0, 1).

    Ledges (`ledge_walk` steps with the shape of the ground the herd's best stand on):

    * `LEDGE_WALK` (True): the `ledge_walk` move joins route preference.
    * `LEDGE_SHARE` (0.5): share of the herd (its best, at least two) whose scatter shapes the ledge step.
    * `LEDGE_STRIDE` (1.0): the step is LEDGE_STRIDE × A u, A·Aᵀ the rank-weighted scatter of those best ibex.
    * `LEDGE_RISE` (2.0): in the model the herd walks the ledges once its best have found them: the `ledge_walk`
      route preference is scaled by `τ^LEDGE_RISE` (0: the same all year).

    Noisy terrain:

    * `RECHECK` (True): ibex re-check a feeding spot before trusting it. When the run starts, the herd tastes its best
      spot a second time (when the budget leaves more than one evaluation for it); if the two tastes differ, the
      terrain is noisy, and from then on an ibex that has gone `RECHECK_PATIENCE` seasons without improving tastes
      its own spot again and trusts the fresh taste. On deterministic terrain the two tastes agree, so it costs one
      evaluation and changes nothing else.
    * `RECHECK_PATIENCE` (10): seasons without improving after which an ibex on noisy terrain re-checks its spot.

    Trails (`follow_trail`, `nio.lore`, `nio.tracks`):

    * `TRAIL_WEIGHT` (0.3): a larger herd wears deeper trails: the `follow_trail` route preference is scaled by
      `TRAIL_WEIGHT` × the herd's size at the start of the run (9 for a herd of 30, 54 for a herd of 180), so the
      larger the herd, the more of it walks its trails and the less it uses its other behaviours.
    * `TRAIL_TRACKS` (1.0): the herd remembers as many tracks (positions its ibex have left) as this times its size.
    * `LORE_SLOTS` (5): pairs of a stride and a share of the trail kept that the herd's lore holds, at least 1.
    * `LORE_STRIDE` (0.5): the stride every slot of the lore starts at.
    * `LORE_KEEP` (0.5): the share of the trail kept that every slot of the lore starts at.

    Thinning:

    * `THIN_LEAST` (12): over the year the weakest ibex leave the herd, so later in the year fewer, fitter ibex share
      the evaluations and each of them moves more often. At each season's start the herd keeps its
      `round(agents − (agents − least)·τ)` best ibex, `least = min(agents, max(4, THIN_LEAST))` (a value of `agents`
      or more: no thinning); the elite and the springs stay.

    Migration (`nio.migration`):

    * `MIGRATE_PATIENCE` (30): seasons the herd's best may go without improving before the herd may migrate: a fresh
      herd is scattered across the box and lives a new year over what is left of the run, from exploring to exploiting,
      with the springs and the elite kept (0: never; the run is one year). Never on noisy terrain (`RECHECK`).
    * `MIGRATE_GATHER` (0.001): the herd migrates only once it has gathered on one spot: its spread is within this
      share of the box's span in every dimension. A herd that is still spread out is still searching, and leaving
      then would throw its search away (0: it migrates on patience alone).
    * `MIGRATE_UNTIL` (0.9): the herd migrates only before this share of the run, so the last year is long enough to
      gather.

    Route preference:

    * `ROUTE_FLOOR` (0.03): least probability of each behaviour in the route preference, between 0 and 1 / 11 (the
      eleven floors must fit in a probability of 1).
    * `ROUTE_MEMORY` (0.7): smoothing of the behaviours' success rates.
    """

    BROWSE_STEP: float = 1.0
    BROWSE_STEP_LATE: float = 0.1
    DOMINANTS: float = 0.2
    FOLLOW_REACH: float = 1.0
    ZIGZAG_STRIDE: float = 1.0
    CLIFF_REACH: float = 0.05
    CLIFF_LEDGES: float = 2.0
    CLIFF_PATIENCE: int = 0
    SENTINEL_LOOKOUTS: float = 1.0
    SENTINEL_FADE: float = 3.0
    SPRINGS: int = 5
    SPRING_GAP: float = 0.01
    SPRING_SCATTER: float = 1.0
    RUT_MIX: float = 0.5
    RUT_STEP: float = 0.5
    FISSION_GROUPS: int = 3
    FUSION_SEASON: float = 0.7
    FUSION_PATIENCE: int = 10
    FUSION_GATHER: float = 0.05
    HOME_DEPTH: float = 8.0
    HOME_RISE: float = 0.5
    HOME_FULL: float = 0.5
    RIDGE_STRIDE: float = 3.0
    LEDGE_WALK: bool = True
    LEDGE_SHARE: float = 0.5
    LEDGE_STRIDE: float = 1.0
    LEDGE_RISE: float = 2.0
    RECHECK: bool = True
    RECHECK_PATIENCE: int = 10
    TRAIL_WEIGHT: float = 0.3
    TRAIL_TRACKS: float = 1.0
    LORE_SLOTS: int = 5
    LORE_STRIDE: float = 0.5
    LORE_KEEP: float = 0.5
    THIN_LEAST: int = 12
    MIGRATE_PATIENCE: int = 30
    MIGRATE_GATHER: float = 0.001
    MIGRATE_UNTIL: float = 0.9
    ROUTE_FLOOR: float = 0.03
    ROUTE_MEMORY: float = 0.7

    def __post_init__(self) -> None:
        if isinstance(self.SPRINGS, bool | np.bool_):
            raise ValueError(
                f"SPRINGS is the size of the spring archive (an integer), not a switch; got {self.SPRINGS!r}."
            )
        if self.SPRINGS < 1:
            raise ValueError(f"SPRINGS is the size of the spring archive, at least 1; got {self.SPRINGS!r}.")
        if self.LORE_SLOTS < 1:
            raise ValueError(f"LORE_SLOTS is the size of the herd's lore, at least 1; got {self.LORE_SLOTS!r}.")
        if not 0 <= self.ROUTE_FLOOR * len(BEHAVIOURS) <= 1:
            raise ValueError(
                f"ROUTE_FLOOR is the least probability of each behaviour, so it must lie between 0 and "
                f"1 / {len(BEHAVIOURS)} = {1 / len(BEHAVIOURS):g}; got {self.ROUTE_FLOOR!r}."
            )

    def replace(self, **settings) -> Self:
        """A copy with the given settings changed. A name that is not a setting raises ValueError, so a typo cannot pass
        silently; the message lists the valid names."""
        valid = [field.name for field in fields(self)]
        unknown = [name for name in settings if name not in valid]
        if unknown:
            raise ValueError(f"Unknown NIO setting(s): {', '.join(unknown)}. The settings are: {', '.join(valid)}.")
        return _replace(self, **settings)
