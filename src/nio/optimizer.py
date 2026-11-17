"""The NIO optimizer: a herd of ibex moving season by season over the objective's terrain."""

import math
import time
from dataclasses import fields
from typing import NamedTuple, Self

import numpy as np

from nio.behaviours import (
    BEHAVIOURS,
    browse,
    cliff_escape,
    follow_dominant,
    follow_trail,
    home_range,
    ledge_walk,
    return_to_spring,
    ridge_walk,
    rut,
    sentinel,
    zigzag_climb,
)
from nio.behaviours.fission import FissionFusion
from nio.budget import Budget
from nio.diagnostics import Diagnostics
from nio.evaluator import Evaluator, Objective
from nio.herd import Herd
from nio.history import History
from nio.initializers import Initializer, RandomUniform
from nio.lore import Lore
from nio.migration import Migration
from nio.result import Result
from nio.routes import Routes
from nio.settings import Settings
from nio.space import SearchSpace
from nio.springs import Springs
from nio.terrain import repair
from nio.tracks import Tracks
from utils.progress import Progress

_CLIFF = BEHAVIOURS.index("cliff_escape")
_SENTINEL = BEHAVIOURS.index("sentinel")
_HOME = BEHAVIOURS.index("home_range")
_RIDGE = BEHAVIOURS.index("ridge_walk")
_LEDGE = BEHAVIOURS.index("ledge_walk")
_TRAIL = BEHAVIOURS.index("follow_trail")
_GROUP_MOVES = ("browse", "follow_dominant", "zigzag_climb")
_SETTINGS = tuple(field.name for field in fields(Settings))


class _Snapshot(NamedTuple):
    """What the history records of a season: every ibex's position and fitness (an ibex that left the herd stays
    where it left, with the fitness it had) and the elite."""

    positions: np.ndarray
    fitness: np.ndarray
    elite: np.ndarray
    elite_fit: float


class NIO:
    """NIO, the Nubian Ibex Optimizer: minimises `objective` over the box `space` with a herd of `agents` ibex.

    The search is modelled as a herd of ibex living through a year on the objective's terrain, where lower ground is
    better feeding. The year is a run of seasons; the season τ runs from 0 at the start of the run to 1 at its end,
    from exploring to exploiting; a herd that migrates starts a new year (below).

    Each season:

    1. **Thinning**: the weakest ibex leave the herd, which shrinks linearly over the year from `agents` to
       `THIN_LEAST`, so later in the year fewer, fitter ibex share the evaluations.
    2. **Fission–fusion** (`nio.behaviours.fission`): the herd lives in subgroups, each with its own leader, that
       regroup with fresh membership when one stalls or gathers, and fuse into one herd by τ = `FUSION_SEASON`.
    3. **Route preference** (`nio.routes`): every ibex picks a behaviour with probability proportional to the
       behaviour's recent success, with a floor, scaled by a seasonal weight: `(1 − τ)^SENTINEL_FADE` for
       `sentinel`, `τ^HOME_RISE` for `home_range`, `τ^LEDGE_RISE` for `ledge_walk`, and `TRAIL_WEIGHT · agents` for
       `follow_trail`, so a large herd mostly walks its trails; a switched-off behaviour has weight 0 and is never
       chosen. With `CLIFF_PATIENCE` above 0, an ibex that has gone that many seasons without improving, other than
       a subgroup's leader, is forced to flee by `cliff_escape` once.
    4. **Moves**: the herd moves together. Every ibex proposes a position with its behaviour from where the herd
       stood when the season began; the proposals are brought back into the box (`nio.terrain.repair`) and scored,
       in one batch when the objective has a `batch` method, and an ibex moves only if its proposal is better. Every
       scored proposal is offered to the herd's springs (`nio.springs`), the best distinct places ever found. Success
       teaches route preference; a forced escape teaches it nothing.
    5. **Trails**: an ibex that moves leaves a track where it stood (`nio.tracks`), which later trail followers
       stride along, and the trail walks that reached better ground teach the herd's lore (`nio.lore`) which strides
       pay.

    Migration (`nio.migration`): when the herd's best has not improved for `MIGRATE_PATIENCE` seasons and the herd has
    gathered on one spot (`MIGRATE_GATHER`), before `MIGRATE_UNTIL` of the run and on deterministic terrain, the herd
    leaves for a fresh range: a new herd is scattered uniformly across the box (whatever the first herd's initializer)
    and lives a new year over what is left of the run, τ running again from 0 to 1. Its springs and elite go with it;
    route preference, fission–fusion, the tracks and the lore start afresh.

    The behaviours (`nio.behaviours`) are `browse`, `follow_dominant`, `zigzag_climb`, `cliff_escape`, `sentinel`,
    `return_to_spring`, `rut`, `home_range`, `ridge_walk`, `ledge_walk` and `follow_trail`. All of them work on
    differences between positions or on the bounds and span of the box, except two that use the structure of the box:
    `home_range` pulls toward the centre of the box and `ridge_walk` steps along the line through the origin in the
    direction (1, …, 1), on which every coordinate is equal; that line is the box's main diagonal only when every
    dimension has the same bounds. Only `ridge_walk` breaks translation invariance.

    Noisy terrain (`RECHECK`): when the run starts, the herd tastes its best spot a second time, if the budget leaves
    more than one evaluation for it. If the two tastes differ (a +inf taste says nothing), the terrain is taken as
    noisy, and from then on an ibex that has gone `RECHECK_PATIENCE` seasons without improving, a leader too, tastes
    its own spot again and trusts the fresh taste instead of moving.

    Search box: `space` is a `nio.SearchSpace`, or any object with per-dimension bounds `lb` and `ub` (such as a
    benchmark problem's own box), which becomes one.

    Settings: `settings` (a `nio.Settings`, which documents every setting and its default), given to the constructor
    or changed with `with_settings`; they are read when `solve()` starts.

    Herd size: `agents`, or when it is left out the size the budget suggests (`Budget.herd`): 18 ibex per dimension,
    no more than `evaluations / 500` and at least 30; 30 under an iteration budget.

    Budget: `iterations` seasons (500 when neither budget is given), or exactly `evaluations` objective calls. The
    first scoring of the herd takes `agents` evaluations (and the second taste of its best spot one more), every
    migration `agents` more, and every season one per ibex in the herd; the seasons run until the budget is spent, and
    the last may be cut short. The run's clock is `season / iterations`, or, with `evaluations`, the share of the run's
    ibex moves made once the herd (as it stands at the season's start) has moved, counting a migrated herd's scoring as
    moves, the run being `agents · seasons` moves with `seasons = ⌈evaluations / agents⌉ − 1`. τ is the clock until
    the herd first migrates, and then runs from 0 to 1 again over each new year.

    First herd: `initializer` (a `nio.initializers.Initializer`) scatters it; by default every ibex is uniform in the
    box (`RandomUniform`).

    Random numbers: every draw of the run comes from one `numpy.random.Generator` seeded with `seed`. With a `seed`,
    `solve()` also seeds the global numpy generator, so an objective that draws its own noise from it (as F7 does) is
    reproducible too.

    After `solve()`, `diagnostics` (a `nio.Diagnostics`) tells how the run went: the behaviours used, the subgroups and
    the herd size in each season, the regroupings, the springs and the re-checks.

    Example:

        >>> import numpy as np
        >>> from nio import NIO, SearchSpace
        >>> result = NIO(lambda x: float(np.sum(x**2)), SearchSpace.of(-5, 5, 10), evaluations=3_000, seed=1).solve()
        >>> result.fitness < 1e-3
        True
    """

    name = "NIO"

    def __init__(
        self,
        objective: Objective,
        space: SearchSpace,
        agents: int | None = None,
        iterations: int | None = None,
        evaluations: int | None = None,
        seed: int | None = None,
        verbose: bool = False,
        history: bool = True,
        settings: Settings | None = None,
        initializer: Initializer | None = None,
    ):
        if not callable(objective):
            raise TypeError(
                f"The objective must be callable, scoring one position (an array) with one number; got "
                f"{type(objective).__name__}. A `batch` method does not replace it: re-checks on noisy terrain score "
                f"one position at a time."
            )
        if not isinstance(space, SearchSpace):  # a problem's own box: any object with per-dimension lb and ub
            if not (hasattr(space, "lb") and hasattr(space, "ub")):
                raise TypeError(
                    f"space must be a nio.SearchSpace or have per-dimension bounds lb and ub, "
                    f"not {type(space).__name__}."
                )
            space = SearchSpace.of(space.lb, space.ub, np.size(space.lb))
        if iterations is not None and evaluations is not None:
            raise ValueError("Give either iterations or evaluations as the budget, not both.")
        if evaluations is not None:
            budget = Budget(evaluations=evaluations)
        else:
            budget = Budget(iterations=500 if iterations is None else iterations)
        if agents is None:
            agents = budget.herd(space.dimensions)
            if evaluations is not None:
                agents = max(4, min(agents, evaluations - 1))  # a small budget still leaves the herd a move
        if agents < 4:
            raise ValueError("NIO requires at least 4 agents.")
        if evaluations is not None and evaluations <= agents:
            raise ValueError(f"An evaluation budget must exceed the {agents} evaluations of the first herd.")
        if settings is not None and not isinstance(settings, Settings):
            raise TypeError(f"settings must be a nio.Settings, not {type(settings).__name__}.")
        if initializer is not None and not isinstance(initializer, Initializer):
            raise TypeError(f"initializer must be a nio.initializers.Initializer, not {type(initializer).__name__}.")
        self.objective = objective
        self.space = space
        self.agents = agents
        self.budget = budget
        self.seed = seed
        self.verbose = verbose
        self.history = history
        self.settings = Settings() if settings is None else settings
        self.initializer = RandomUniform() if initializer is None else initializer
        self.diagnostics: Diagnostics | None = None

    def __setattr__(self, name: str, value) -> None:
        """Settings are not attributes of the optimizer: assigning an upper-case name raises AttributeError, so a
        setting assigned by mistake (`opt.LEDGE_WALK = False`) cannot pass silently."""
        if name.isupper() and not name.startswith("_"):
            raise AttributeError(
                f"{name} cannot be assigned on NIO; the settings live in `settings`, use with_settings(...); "
                f"valid settings: {', '.join(_SETTINGS)}."
            )
        super().__setattr__(name, value)

    def with_settings(self, **settings) -> Self:
        """Change the named settings (`Settings.replace`) and return the optimizer. A name that is not a setting raises
        ValueError, naming the valid settings."""
        self.settings = self.settings.replace(**settings)
        return self

    def solve(self) -> Result:
        """Run the herd until the budget is spent and return the best position found. Sets `diagnostics`."""
        rng = np.random.default_rng(self.seed)
        if self.seed is not None:
            np.random.seed(self.seed)  # for objectives that draw their own noise from the global generator
        started = time.perf_counter()
        settings, agents = self.settings, self.agents
        evaluate = Evaluator(self.objective, self.budget.limit)
        routes = self._routes()
        history = History(self.history)

        herd = self._first_herd(evaluate, rng)
        fission = self._fission()
        self._fresh_trails()
        noisy = self._terrain_is_noisy(herd, evaluate)
        migration = Migration(0 if noisy else settings.MIGRATE_PATIENCE, settings.MIGRATE_UNTIL)

        least = min(agents, max(4, settings.THIN_LEAST))
        positions, fitness = herd.positions.copy(), herd.fitness.copy()  # every ibex, as the history records them
        place = np.arange(agents)  # where each ibex of the herd stands in those arrays
        used, groups, sizes = [], [], []
        regroups = rechecks = 0
        iterations, limit = self.budget.iterations, self.budget.limit
        moves = 0  # ibex moves so far, this season's included: the clock of an evaluation budget
        year = agents * self.budget.seasons(agents)  # the moves of a year with the herd whole
        with Progress(total=iterations, label=self.name, disable=not self.verbose) as bar:
            season = 0
            while (season < iterations) if iterations is not None else (evaluate.count < limit):
                season += 1
                migration.watch(herd.best_fit)
                if migration.due(self._clock(season, moves + len(herd.fitness), year)) and self._has_gathered(herd):
                    herd = self._migrate(herd, evaluate, rng)
                    routes, fission, place = self._routes(), self._fission(), np.arange(agents)
                    self._fresh_trails()
                    moves += agents  # the fresh herd's scoring
                    migration.arrive(self._clock(season, moves, year), herd.best_fit)
                moves += len(herd.fitness)
                tau = migration.season(self._clock(season, moves, year))
                place, regrouped = self._thin_and_regroup(herd, routes, fission, place, least, tau, rng)
                regroups += regrouped
                groups.append(herd.groups)
                sizes.append(len(herd.fitness))
                used.append(np.zeros(len(BEHAVIOURS), dtype=int))
                rechecks += self._season(herd, routes, evaluate, tau, rng, noisy, used[-1])
                if self.history:
                    positions[place], fitness[place] = herd.positions, herd.fitness
                history.record(_Snapshot(positions, fitness, herd.elite, herd.elite_fit))
                bar.note(best=f"{herd.elite_fit:.4e}")
                bar.update()

        self.diagnostics = Diagnostics(
            used=np.array(used, dtype=int).reshape(-1, len(BEHAVIOURS)),
            groups=np.array(groups, dtype=int),
            sizes=np.array(sizes, dtype=int),
            regroups=int(regroups),
            springs=herd.springs,
            noisy=noisy,
            rechecks=rechecks,
            migrations=migration.count,
        )
        return history.result(herd, evaluate, time.perf_counter() - started)

    def _has_gathered(self, herd: Herd) -> bool:
        """`MIGRATE_GATHER`: whether the herd stands on one spot, its spread within that share of the span in every
        dimension (0: always, so patience alone decides)."""
        gather = self.settings.MIGRATE_GATHER
        return gather <= 0 or bool(np.all(herd.spread() <= gather * self.space.span))

    def _clock(self, season: int, moves: int, year: int) -> float:
        """The run's clock, from 0 at its start to 1 at its end: `season / iterations`, or under an evaluation budget
        the share of a year's ibex moves made so far (`moves`, out of `year`)."""
        iterations = self.budget.iterations
        return season / iterations if iterations is not None else min(1.0, moves / year)

    def _routes(self) -> Routes:
        """Route preference as a herd starts its year: every behaviour at the same success rate."""
        return Routes(len(BEHAVIOURS), self.settings.ROUTE_FLOOR, self.settings.ROUTE_MEMORY)

    def _fresh_trails(self) -> None:
        """A herd starting its year has no tracks yet, and its lore is what every herd starts with."""
        settings = self.settings
        self._lore = Lore(settings.LORE_SLOTS, settings.LORE_STRIDE, settings.LORE_KEEP)
        self._tracks = Tracks(round(settings.TRAIL_TRACKS * self.agents), self.space.dimensions)

    def _fission(self) -> FissionFusion:
        """Fission–fusion as a herd of `agents` ibex starts its year."""
        settings = self.settings
        return FissionFusion(
            self.agents,
            settings.FISSION_GROUPS,
            settings.FUSION_PATIENCE,
            settings.FUSION_GATHER,
            settings.FUSION_SEASON,
        )

    def _migrate(self, herd: Herd, evaluate: Evaluator, rng: np.random.Generator) -> Herd:
        """Migration (`nio.migration`): a fresh herd of `agents` ibex, uniform in the box whatever the initializer of
        the first herd, scored. It keeps the springs, and the elite when the elite is better than any fresh ibex."""
        space = self.space
        positions = space.lb + space.span * rng.random((self.agents, space.dimensions))
        fresh = Herd(positions, evaluate.many(positions), herd.springs)
        if herd.elite_fit < fresh.elite_fit:
            fresh.elite, fresh.elite_fit = herd.elite, herd.elite_fit
        return fresh

    def _first_herd(self, evaluate: Evaluator, rng: np.random.Generator) -> Herd:
        """The herd's first scattering by the initializer (uniform in the box by default), scored, with an empty
        spring archive."""
        space, settings = self.space, self.settings
        positions = self.initializer(space, self.agents, rng)
        return Herd(positions, evaluate.many(positions), Springs(settings.SPRINGS, settings.SPRING_GAP * space.span))

    def _terrain_is_noisy(self, herd: Herd, evaluate: Evaluator) -> bool:
        """`RECHECK`: taste the herd's best spot a second time; the terrain is noisy if the two tastes differ (a +inf
        taste says nothing). With one evaluation left the evaluation goes to a move instead, so a budget of
        `agents + 1` still gets a season."""
        room = evaluate.limit is None or evaluate.limit - evaluate.count > 1
        if not (self.settings.RECHECK and room):
            return False
        best = herd.best
        again = evaluate.one(herd.positions[best])
        noisy = math.isfinite(again) and bool(again != herd.fitness[best])
        if noisy:
            herd.retaste(best, again)
        return noisy

    def _thin_and_regroup(
        self,
        herd: Herd,
        routes: Routes,
        fission: FissionFusion,
        place: np.ndarray,
        least: int,
        tau: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, bool]:
        """The herd at a season's start. The weakest ibex leave: the herd keeps its
        `max(least, round(agents − (agents − least)·τ))` best ibex, and as many tracks per ibex as before. Then
        fission–fusion splits or regroups the herd. Returns where each ibex of the herd stands among all ibex, and
        whether the herd regrouped."""
        settings, agents = self.settings, self.agents
        size = max(least, round(agents - (agents - least) * tau))
        if size < len(herd.fitness):
            place = place[herd.thin(size)]
            self._tracks.resize(round(settings.TRAIL_TRACKS * size), rng)
            fission.resize(size, settings.FISSION_GROUPS)
        return place, fission.season(herd, tau, rng)

    def _route_weights(self, tau: float) -> np.ndarray:
        """The weight of each behaviour in route preference at season τ: sentinels fade, the home range and the ledges
        rise, and the trails weigh as much as the herd is large."""
        settings = self.settings
        weights = np.ones(len(BEHAVIOURS))
        weights[_SENTINEL] = (1.0 - tau) ** settings.SENTINEL_FADE
        weights[_HOME] = tau**settings.HOME_RISE
        weights[_LEDGE] = float(bool(settings.LEDGE_WALK)) * tau**settings.LEDGE_RISE
        weights[_TRAIL] = settings.TRAIL_WEIGHT * self.agents
        return weights

    def _season(
        self,
        herd: Herd,
        routes: Routes,
        evaluate: Evaluator,
        tau: float,
        rng: np.random.Generator,
        noisy: bool,
        used: np.ndarray,
    ) -> int:
        """One season of moves: every ibex picks a behaviour and proposes from where the herd stood at the season's
        start, the proposals are scored together, and each ibex moves if its proposal is better. `used` counts the
        behaviours used; returns the number of scored re-checks."""
        settings = self.settings
        agents, count = len(herd.fitness), routes.count
        choice = routes.choose(agents, rng, self._route_weights(tau))
        movers = rng.permutation(agents)  # the order in which the proposals are scored
        fleeing = np.zeros(agents, dtype=bool)
        rechecks = 0
        if noisy:  # a stalled ibex, leader or not, re-checks its spot instead of moving
            stalled = herd.stall >= settings.RECHECK_PATIENCE
            for i in movers[stalled[movers]]:
                rechecks += self._recheck(herd, i, evaluate)
            movers = movers[~stalled[movers]]
        elif settings.CLIFF_PATIENCE > 0:
            fleeing = herd.stall >= settings.CLIFF_PATIENCE
            fleeing[herd.leaders()] = False  # a subgroup's leader is never forced to flee
        if not len(movers):
            return rechecks
        behaviour = np.where(fleeing, _CLIFF, choice)[movers]
        proposals = np.empty((len(movers), herd.positions.shape[1]))
        walks = np.flatnonzero(behaviour == _TRAIL)
        strides, keeps = self._lore.draw(len(walks), rng)
        for b in np.unique(behaviour):
            rows = np.flatnonzero(behaviour == b)
            who, name = movers[rows], BEHAVIOURS[b]
            used[b] += len(rows)
            if name in _GROUP_MOVES and herd.groups > 1:  # these see only the ibex's own subgroup
                for g in np.unique(herd.group[who]):
                    mine = herd.group[who] == g
                    view, _ = herd.within(int(who[mine][0]))
                    inside = np.searchsorted(herd.members(g), who[mine])
                    proposals[rows[mine]] = self._move(name, view, inside, tau, rng)
            elif b == _TRAIL:
                proposals[rows] = follow_trail(herd, who, rng, self._dominants(herd, tau), strides, keeps, self._tracks)
            else:
                proposals[rows] = self._move(name, herd, who, tau, rng)
        x = repair(proposals, herd.positions[movers], self.space)
        before, was = herd.positions[movers].copy(), herd.fitness[movers].copy()
        scores = evaluate.many(x)
        moved = herd.offer(movers, x, scores)
        fled = fleeing[movers]
        herd.stall[movers[fled]] = 0  # it fled once and settles: its patience restarts
        hits, tries = np.zeros(count), np.zeros(count)
        np.add.at(tries, behaviour[~fled], 1)  # a forced escape teaches route preference nothing
        np.add.at(hits, behaviour[~fled], moved[~fled])
        routes.reward(hits, tries)
        for k in np.flatnonzero(moved):
            self._tracks.leave(before[k], rng)
        won = walks[moved[walks]]  # a walk from unscored ground (+inf) has no measurable gain: the lore skips it
        with np.errstate(invalid="ignore"):
            gain = was[won] - scores[won]
        self._lore.note(strides[moved[walks]], keeps[moved[walks]], gain)
        self._lore.learn()
        return rechecks

    @staticmethod
    def _recheck(herd: Herd, i: int, evaluate: Evaluator) -> bool:
        """Ibex i tastes its own spot again and trusts the fresh taste. Returns whether the taste was scored (after
        the budget is spent a taste is +inf and teaches nothing)."""
        scored = evaluate.limit is None or evaluate.count < evaluate.limit
        herd.retaste(i, evaluate.one(herd.positions[i]))
        return scored

    def _move(self, behaviour: str, herd: Herd, who: np.ndarray, tau: float, rng: np.random.Generator) -> np.ndarray:
        """The proposals of `behaviour` for the ibex `who` of `herd`, with its parameters at season τ."""
        settings = self.settings
        match behaviour:
            case "browse":
                step = settings.BROWSE_STEP + (settings.BROWSE_STEP_LATE - settings.BROWSE_STEP) * tau
                return browse(herd, who, rng, step)
            case "follow_dominant":
                return follow_dominant(herd, who, rng, self._dominants(herd, tau), settings.FOLLOW_REACH)
            case "zigzag_climb":
                return zigzag_climb(herd, who, rng, settings.ZIGZAG_STRIDE)
            case "cliff_escape":
                return cliff_escape(herd, who, rng, self.space, settings.CLIFF_REACH, settings.CLIFF_LEDGES)
            case "sentinel":
                return sentinel(herd, who, rng, self.space, settings.SENTINEL_LOOKOUTS)
            case "return_to_spring":
                return return_to_spring(herd, who, rng, settings.SPRING_SCATTER)
            case "rut":
                return rut(herd, who, rng, settings.RUT_MIX, settings.RUT_STEP)
            case "home_range":
                fidelity = min(1.0, tau / settings.HOME_FULL) if settings.HOME_FULL > 0 else 1.0  # grows over the year
                return home_range(herd, who, rng, self.space, settings.HOME_DEPTH * fidelity)
            case "ridge_walk":
                return ridge_walk(herd, who, rng, settings.RIDGE_STRIDE)
            case "ledge_walk":
                return ledge_walk(herd, who, rng, settings.LEDGE_SHARE, settings.LEDGE_STRIDE)
        raise ValueError(f"Unknown behaviour: {behaviour}.")

    def _dominants(self, herd: Herd, tau: float) -> int:
        """How many of the herd's top ibex are followed at season τ: the `DOMINANTS` share, falling to 1 by the end."""
        return max(1, round(self.settings.DOMINANTS * len(herd.fitness) * (1 - tau)))
