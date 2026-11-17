"""Small studies of NIO, as the notebooks run them: how its results change with the herd size, the number of seasons,
the dimension or a setting, on the classical functions F1-F23.

A `Trial` is one run (a function, a herd size, a number of seasons, a seed, settings); `run_all` runs many over worker
processes and returns their `Outcome`s in the trials' order, and `grouped`, `median_curve` and `final_values` sum them
up for a figure or a table:

    trials = [Trial("F5", agents=n, iterations=110, seed=s) for n in (25, 50, 100) for s in range(5)]
    by_size = grouped(run_all(trials), lambda trial: trial.agents)
    curves = {n: median_curve(outcomes) for n, outcomes in by_size.items()}
"""

from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np

from benchmarks.runs import execute
from nio import NIO, SearchSpace
from problems.functions import classic


@dataclass(frozen=True)
class Trial:
    """One run of NIO on `function` (a key of `problems.functions.classic`), in `dim` dimensions (the function's
    own when left out), with `agents` ibex for `iterations` seasons from `seed`, and `settings` changed."""

    function: str
    agents: int = 50
    iterations: int = 100
    seed: int = 0
    dim: int | None = None
    settings: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, eq=False)
class Outcome:
    """What a trial found: the best value, the best after each season, and the seconds it took."""

    trial: Trial
    fitness: float
    convergence: np.ndarray
    runtime: float


def run(trial: Trial) -> Outcome:
    """Run one trial."""
    f = classic[trial.function].at(dim=trial.dim)
    space = SearchSpace.of(f.lb, f.ub, f.dim)
    optimizer = NIO(f, space, agents=trial.agents, iterations=trial.iterations, seed=trial.seed, history=False)
    result = optimizer.with_settings(**trial.settings).solve()
    return Outcome(trial, result.fitness, result.convergence, result.runtime)


def _indexed(task: tuple[int, Trial]) -> tuple[int, Outcome]:
    index, trial = task
    return index, run(trial)


def run_all(trials: Iterable[Trial], workers: int | None = None) -> list[Outcome]:
    """Every trial's outcome, run over `workers` processes (all cores when left out), in the trials' order."""
    done = execute(_indexed, enumerate(trials), workers)
    return [outcome for _, outcome in sorted(done, key=lambda pair: pair[0])]


def grouped[K: Hashable](outcomes: Iterable[Outcome], key: Callable[[Trial], K]) -> dict[K, list[Outcome]]:
    """The outcomes grouped by `key` of their trial, the groups in the order they first appear."""
    groups: dict[K, list[Outcome]] = {}
    for outcome in outcomes:
        groups.setdefault(key(outcome.trial), []).append(outcome)
    return groups


def median_curve(outcomes: Iterable[Outcome]) -> np.ndarray:
    """The median, season by season, of the outcomes' best values (they must have the same number of seasons)."""
    return np.median([outcome.convergence for outcome in outcomes], axis=0)


def final_values(outcomes: Iterable[Outcome]) -> np.ndarray:
    """The best value each outcome found."""
    return np.array([outcome.fitness for outcome in outcomes], dtype=float)
