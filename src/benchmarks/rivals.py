"""The metaheuristics NIO is compared with, run from mealpy on the same objective and the same evaluation budget.

`solve(name, objective, space, agents, evaluations, seed)` runs one of `RIVALS` (mealpy's "Original" version of each,
with mealpy's default parameters) and returns its best position, its fitness, the evaluations it used and its
improvement trace, like `nio.Result`.

The budget is counted in objective evaluations, not iterations: the rivals spend different numbers of evaluations
per iteration (QSA three per agent, NRO and AEO two), so equal iterations would not be a fair comparison. Every call
goes through a counter that records the best position seen and stops the run at exactly `evaluations` calls.

mealpy is an optional dependency, in the `compare` group: `uv sync --group compare`.
"""

import importlib
import time
from dataclasses import dataclass

import numpy as np

# name -> mealpy module and class; the papers are cited in the manuscript
RIVALS = {
    "SCA": ("mealpy.math_based.SCA", "OriginalSCA"),  # Sine Cosine Algorithm, Mirjalili 2016
    "WOA": ("mealpy.swarm_based.WOA", "OriginalWOA"),  # Whale Optimization Algorithm, Mirjalili and Lewis 2016
    "QSA": ("mealpy.human_based.QSA", "OriginalQSA"),  # Queuing Search Algorithm, Zhang et al. 2018
    "NRO": ("mealpy.physics_based.NRO", "OriginalNRO"),  # Nuclear Reaction Optimization, Wei et al. 2019
    "GBO": ("mealpy.math_based.GBO", "OriginalGBO"),  # Gradient-Based Optimizer, Ahmadianfar et al. 2020
    "AEO": ("mealpy.system_based.AEO", "OriginalAEO"),  # Artificial Ecosystem Optimization, Zhao et al. 2020
    "BMO": ("mealpy.bio_based.BMO", "OriginalBMO"),  # Barnacles Mating Optimizer, Sulaiman et al. 2020
    "GJO": ("mealpy.swarm_based.GJO", "OriginalGJO"),  # Golden Jackal Optimization, Chopra and Ansari 2022
    "CDO": ("mealpy.physics_based.CDO", "OriginalCDO"),  # Chernobyl Disaster Optimizer, Shehadeh 2023
}


@dataclass(frozen=True)
class Outcome:
    """One rival's run: the best position and fitness, the evaluations used, the improvement trace (evaluation, best
    value) and the wall time in seconds."""

    position: np.ndarray
    fitness: float
    evaluations: int
    trace: np.ndarray
    runtime: float


class Spent(Exception):
    """Raised by the counter when the evaluation budget is used up; it ends the rival's run."""


class Counter:
    """The objective as a rival sees it: counts the calls, keeps the best position and its trace, and raises `Spent`
    once `budget` calls have been made."""

    def __init__(self, objective, budget: int):
        self.objective = objective
        self.budget = budget
        self.calls = 0
        self.best = np.inf
        self.position = None
        self.trace = []

    def __call__(self, x) -> float:
        if self.calls >= self.budget:
            raise Spent
        value = float(self.objective(np.asarray(x, dtype=float)))
        self.calls += 1
        if value < self.best or self.position is None:
            self.best = value
            self.position = np.array(x, dtype=float)
            self.trace.append((self.calls, value))
        return value


def solve(name: str, objective, space, agents: int, evaluations: int, seed: int) -> Outcome:
    """Run rival `name` on `objective` over `space` (a box with `lb` and `ub`) with `agents` agents for exactly
    `evaluations` objective calls, from `seed`."""
    if name not in RIVALS:
        raise ValueError(f"Unknown rival {name!r}; choose from {', '.join(RIVALS)}.")
    if evaluations <= agents:
        raise ValueError(f"The budget ({evaluations}) must exceed the {agents} evaluations of the first population.")
    try:
        from mealpy import FloatVar
    except ImportError as error:
        raise ImportError("The rivals need mealpy: uv sync --group compare") from error
    module, cls = RIVALS[name]
    optimizer = getattr(importlib.import_module(module), cls)
    bounds = FloatVar(lb=space.lb, ub=space.ub)
    epochs = iterations_for(optimizer, bounds, agents, evaluations)
    counter = Counter(objective, evaluations)
    problem = {"obj_func": counter, "bounds": bounds, "minmax": "min", "log_to": None}
    start = time.perf_counter()
    try:
        with np.errstate(divide="ignore", invalid="ignore"):  # mealpy's diversity log divides 0 by 0 at the end
            optimizer(epoch=epochs, pop_size=agents).solve(problem, seed=seed)
    except Spent:
        pass
    runtime = time.perf_counter() - start
    return Outcome(counter.position, counter.best, counter.calls, np.asarray(counter.trace, dtype=float), runtime)


def iterations_for(optimizer, bounds, agents: int, evaluations: int) -> int:
    """The number of iterations that spends `evaluations` calls, so that the rival's schedules (which shrink its
    steps as the iterations run out) end with the budget: its first population and its calls per iteration are
    measured on a sphere with the same bounds, and the count is rounded up (the counter stops the last iteration)."""

    def calls(epochs: int) -> int:
        counter = Counter(lambda x: float(np.sum(x * x)), 10**9)
        problem = {"obj_func": counter, "bounds": bounds, "minmax": "min", "log_to": None}
        with np.errstate(divide="ignore", invalid="ignore"):
            optimizer(epoch=epochs, pop_size=agents).solve(problem, seed=0)
        return counter.calls

    one, three = calls(1), calls(3)
    per_iteration = (three - one) / 2
    first = one - per_iteration
    return int(min(100_000, max(1, np.ceil((evaluations - first) / per_iteration))))
