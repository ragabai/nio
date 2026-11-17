"""Every continuous test function, by name or by suite, from one place.

    >>> from problems.functions import rosenbrock, classic, fair, fair23, cec, digits
    >>> rosenbrock(x)                         # evaluate: classic F5, 30-D over [-30, 30]
    >>> rosenbrock.at(dim=50, lb=-10, ub=10)  # a customised copy
    >>> classic["F5"], classic.rosenbrock     # the same function, by key or by name
    >>> classic.at(dim=50)                    # the whole suite at 50 dimensions

* `classic`: the 23 classical functions F1-F23, each also importable by name (`problems.classic`);
* `fair` and `fair23`: the same functions with their optima moved off the origin (`problems.shifted`);
  `rosenbrock.shifted()` is `fair23.rosenbrock`;
* `cec(year, dim=None)`: the CEC 2017, 2020, 2022 and 2024 suites (`cec.editions`), with the competition's rules in
  `suite.rules`; `digits`: the CEC 2019 100-Digit Challenge, at its fixed dimensions (`problems.cec`).
"""

from problems.cec import cec
from problems.cec.digits import suite as digits
from problems.classic import (
    ackley,
    branin,
    camel,
    classic,
    foxholes,
    goldstein,
    griewank,
    hartmann3,
    hartmann6,
    kowalik,
    penalized1,
    penalized2,
    quartic,
    rastrigin,
    rosenbrock,
    schwefel12,
    schwefel221,
    schwefel222,
    schwefel226,
    shekel5,
    shekel7,
    shekel10,
    sphere,
    step,
)
from problems.function import Function
from problems.shifted import fair, fair23
from problems.suite import Suite

__all__ = [
    "Function",
    "Suite",
    "ackley",
    "branin",
    "camel",
    "cec",
    "classic",
    "digits",
    "fair",
    "fair23",
    "foxholes",
    "goldstein",
    "griewank",
    "hartmann3",
    "hartmann6",
    "kowalik",
    "penalized1",
    "penalized2",
    "quartic",
    "rastrigin",
    "rosenbrock",
    "schwefel12",
    "schwefel221",
    "schwefel222",
    "schwefel226",
    "shekel5",
    "shekel7",
    "shekel10",
    "sphere",
    "step",
]
