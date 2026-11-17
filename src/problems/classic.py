"""The 23 classical benchmark functions F1–F23.

* F1–F7: unimodal, scalable (30 dimensions);
* F8–F13: multimodal, scalable (30 dimensions);
* F14–F23: multimodal, of fixed dimension.

Each is a `Function` named after its function (`rosenbrock`), and `classic` gathers them as a `Suite`: the objective,
its bounds, its dimension and the known global optimum value (`optimal`), so a result can be reported as its error
`f(best) − optimal`. The scalable F1-F13 take any dimension (`rosenbrock.at(dim=50)`). F7 adds uniform noise drawn
from the global numpy generator, which `NIO.solve()` seeds from the run's seed.
"""

from functools import partial

import numpy as np

from problems.function import Function
from problems.suite import Suite


def _u(x, a, k, m):
    return k * ((x - a) ** m) * (x > a) + k * ((-x - a) ** m) * (x < -a)


def f1(x):  # Sphere
    return float(np.sum(x**2))


def f2(x):  # Schwefel 2.22
    ax = np.abs(x)
    return float(np.sum(ax) + np.prod(ax))


def f3(x):  # Schwefel 1.2
    return float(np.sum(np.cumsum(x) ** 2))


def f4(x):  # Schwefel 2.21
    return float(np.max(np.abs(x)))


def f5(x):  # Rosenbrock
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2))


def f6(x):  # Step
    return float(np.sum(np.floor(x + 0.5) ** 2))


def f7(x):  # Quartic with noise
    i = np.arange(1, len(x) + 1)
    return float(np.sum(i * x**4) + np.random.rand())


def f8(x):  # Schwefel 2.26
    return float(np.sum(-x * np.sin(np.sqrt(np.abs(x)))))


def f9(x):  # Rastrigin
    return float(np.sum(x**2 - 10 * np.cos(2 * np.pi * x)) + 10 * len(x))


def f10(x):  # Ackley
    n = len(x)
    return float(-20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n)) - np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e)


def f11(x):  # Griewank
    i = np.arange(1, len(x) + 1)
    return float(np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(i))) + 1)


def f12(x):  # Penalized 1
    n = len(x)
    y = 1 + (x + 1) / 4
    return float(
        (np.pi / n)
        * (
            10 * np.sin(np.pi * y[0]) ** 2
            + np.sum((y[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * y[1:]) ** 2))
            + (y[-1] - 1) ** 2
        )
        + np.sum(_u(x, 10, 100, 4))
    )


def f13(x):  # Penalized 2
    return float(
        0.1
        * (
            np.sin(3 * np.pi * x[0]) ** 2
            + np.sum((x[:-1] - 1) ** 2 * (1 + np.sin(3 * np.pi * x[1:]) ** 2))
            + (x[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * x[-1]) ** 2)
        )
        + np.sum(_u(x, 5, 100, 4))
    )


_A14 = np.array(
    [
        [-32, -16, 0, 16, 32] * 5,
        [v for v in (-32, -16, 0, 16, 32) for _ in range(5)],
    ],
    dtype=float,
)


def f14(x):  # Shekel's Foxholes
    j = np.arange(1, 26)
    s = np.sum((x[:, None] - _A14) ** 6, axis=0)
    return float(1.0 / (1.0 / 500 + np.sum(1.0 / (j + s))))


_A15 = np.array([0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246])
_B15 = 1.0 / np.array([0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16])


def f15(x):  # Kowalik
    b = _B15
    return float(np.sum((_A15 - x[0] * (b**2 + b * x[1]) / (b**2 + b * x[2] + x[3])) ** 2))


def f16(x):  # Six-Hump Camel-Back
    a, b = x
    return float(4 * a**2 - 2.1 * a**4 + a**6 / 3 + a * b - 4 * b**2 + 4 * b**4)


def f17(x):  # Branin
    a, b = x
    return float(
        (b - 5.1 / (4 * np.pi**2) * a**2 + 5 / np.pi * a - 6) ** 2 + 10 * (1 - 1 / (8 * np.pi)) * np.cos(a) + 10
    )


def f18(x):  # Goldstein-Price
    a, b = x
    return float(
        (1 + (a + b + 1) ** 2 * (19 - 14 * a + 3 * a**2 - 14 * b + 6 * a * b + 3 * b**2))
        * (30 + (2 * a - 3 * b) ** 2 * (18 - 32 * a + 12 * a**2 + 48 * b - 36 * a * b + 27 * b**2))
    )


_C_H = np.array([1, 1.2, 3, 3.2])
_A19 = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
_P19 = np.array([[0.3689, 0.117, 0.2673], [0.4699, 0.4387, 0.747], [0.1091, 0.8732, 0.5547], [0.03815, 0.5743, 0.8828]])
_A20 = np.array(
    [[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]]
)
_P20 = np.array(
    [
        [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
        [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
        [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
        [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
    ]
)


def f19(x):  # Hartmann 3
    return float(-np.sum(_C_H * np.exp(-np.sum(_A19 * (x - _P19) ** 2, axis=1))))


def f20(x):  # Hartmann 6
    return float(-np.sum(_C_H * np.exp(-np.sum(_A20 * (x - _P20) ** 2, axis=1))))


_A_SH = np.array(
    [
        [4, 4, 4, 4],
        [1, 1, 1, 1],
        [8, 8, 8, 8],
        [6, 6, 6, 6],
        [3, 7, 3, 7],
        [2, 9, 2, 9],
        [5, 5, 3, 3],
        [8, 1, 8, 1],
        [6, 2, 6, 2],
        [7, 3.6, 7, 3.6],
    ]
)
_C_SH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])


def _shekel(m):
    def f(x):
        d = np.sum((x - _A_SH[:m]) ** 2, axis=1) + _C_SH[:m]
        return float(-np.sum(1.0 / d))

    return f


def _zero(dim: int) -> float:
    return 0.0


def _schwefel_optimum(dim: int) -> float:
    return -418.9828872724338 * dim


def _scalable(key, name, solve, lb, ub, dim=30, optimal=_zero) -> Function:
    """A function defined in any dimension: 30 by default, with the same bounds and an optimum that may follow `dim`.
    Its `resize` is a `partial`, so the function pickles."""
    resize = partial(_scalable, key, name, solve, lb, ub, optimal=optimal)
    return Function(key, name, solve, lb, ub, dim, optimal(dim), resize=resize)


sphere = _scalable("F1", "Sphere", f1, -100, 100)
schwefel222 = _scalable("F2", "Schwefel 2.22", f2, -10, 10)
schwefel12 = _scalable("F3", "Schwefel 1.2", f3, -100, 100)
schwefel221 = _scalable("F4", "Schwefel 2.21", f4, -100, 100)
rosenbrock = _scalable("F5", "Rosenbrock", f5, -30, 30)
step = _scalable("F6", "Step", f6, -100, 100)
quartic = _scalable("F7", "Quartic+noise", f7, -1.28, 1.28)
schwefel226 = _scalable("F8", "Schwefel 2.26", f8, -500, 500, optimal=_schwefel_optimum)
rastrigin = _scalable("F9", "Rastrigin", f9, -5.12, 5.12)
ackley = _scalable("F10", "Ackley", f10, -32, 32)
griewank = _scalable("F11", "Griewank", f11, -600, 600)
penalized1 = _scalable("F12", "Penalized 1", f12, -50, 50)
penalized2 = _scalable("F13", "Penalized 2", f13, -50, 50)
foxholes = Function("F14", "Shekel Foxholes", f14, -65.536, 65.536, 2, 0.998003837794449)
kowalik = Function("F15", "Kowalik", f15, -5, 5, 4, 0.0003074859878056)
camel = Function("F16", "Six-Hump Camel", f16, -5, 5, 2, -1.0316284534898776)
branin = Function("F17", "Branin", f17, np.array([-5.0, 0.0]), np.array([10.0, 15.0]), 2, 0.39788735772973816)
goldstein = Function("F18", "Goldstein-Price", f18, -2, 2, 2, 3.0)
hartmann3 = Function("F19", "Hartmann 3", f19, 0, 1, 3, -3.86278214782076)
hartmann6 = Function("F20", "Hartmann 6", f20, 0, 1, 6, -3.32236801141551)
shekel5 = Function("F21", "Shekel 5", _shekel(5), 0, 10, 4, -10.153199679058229)
shekel7 = Function("F22", "Shekel 7", _shekel(7), 0, 10, 4, -10.402940566818662)
shekel10 = Function("F23", "Shekel 10", _shekel(10), 0, 10, 4, -10.536409816692045)

# The name of every function, by which `problems.functions` exports it, and its key.
NAMES = {
    "sphere": "F1",
    "schwefel222": "F2",
    "schwefel12": "F3",
    "schwefel221": "F4",
    "rosenbrock": "F5",
    "step": "F6",
    "quartic": "F7",
    "schwefel226": "F8",
    "rastrigin": "F9",
    "ackley": "F10",
    "griewank": "F11",
    "penalized1": "F12",
    "penalized2": "F13",
    "foxholes": "F14",
    "kowalik": "F15",
    "camel": "F16",
    "branin": "F17",
    "goldstein": "F18",
    "hartmann3": "F19",
    "hartmann6": "F20",
    "shekel5": "F21",
    "shekel7": "F22",
    "shekel10": "F23",
}

classic = Suite(
    "classic",
    [
        sphere,
        schwefel222,
        schwefel12,
        schwefel221,
        rosenbrock,
        step,
        quartic,
        schwefel226,
        rastrigin,
        ackley,
        griewank,
        penalized1,
        penalized2,
        foxholes,
        kowalik,
        camel,
        branin,
        goldstein,
        hartmann3,
        hartmann6,
        shekel5,
        shekel7,
        shekel10,
    ],
    names=NAMES,
)
