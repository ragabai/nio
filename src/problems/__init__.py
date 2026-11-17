"""Benchmark problems for continuous optimizers, each with its bounds (`Box`), its dimension and its known optimum.

* `functions`: every continuous test function, by name (`rosenbrock`) or by suite: `classic` (F1 to F23, from
  `classic`), `fair` and `fair23` (them moved away from the origin, from `shifted`), `cec(year)` (the CEC 2017, 2020,
  2022 and 2024 suites, from `cec`) and `digits` (the 100-Digit Challenge, from `cec.digits`);
* `engineering`: nine constrained engineering designs, and `engineering.realworld`: the CEC 2020 real-world suite;
* `datasets` and `selection`: wrapper feature selection with k-NN; `scheduling`: cloud task scheduling.

The problems depend on numpy only and serve any optimizer: each objective is callable on one position, and most
also score a population at once with `batch`.
"""

from problems.box import Box

__all__ = ["Box"]
