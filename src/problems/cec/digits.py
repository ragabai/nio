"""The CEC 2019 100-Digit Challenge: ten functions, each with its optimum value at 1, written from the organisers'
reference C code (`cec19_func.cpp`) and data.

* F1 Storn's Chebyshev polynomial fitting (D = 9, [-8192, 8192]);
* F2 inverse Hilbert matrix (D = 16, [-16384, 16384]);
* F3 Lennard-Jones minimum-energy cluster (D = 18, [-4, 4]);
* F4-F10 shifted and rotated Rastrigin, Griewank, Weierstrass, Schwefel, expanded Schaffer F6, HappyCat and Ackley
  (D = 10, [-100, 100]), with the official shift vectors and rotation matrices.

    >>> from problems.functions import digits
    >>> fn = digits["F1"]
    >>> fn.dim, fn.ub, fn.optimal
    (9, 8192.0, 1.0)

The reference Chebyshev scores the point 1.2 twice and adds p(1.2)² when p(1.2) falls below the target, not
(target − p(1.2))²; this module keeps that, so the zero polynomial scores the optimum 1 as it does there.
"""

from dataclasses import dataclass
from functools import cache

import numpy as np

from problems.cec import data
from problems.cec.basic import BASIC
from problems.function import Function
from problems.suite import Suite

YEAR = 2019
OPTIMUM = 1.0  # every function adds 1 to its value
LENNARD_JONES_OFFSET = 12.7120622568  # minus the least energy of six atoms, as the reference adds it

SHIFTED = {  # number: (basic function, name)
    4: ("rastrigin", "Shifted and Rotated Rastrigin's Function"),
    5: ("griewank", "Shifted and Rotated Griewank's Function"),
    6: ("weierstrass", "Shifted and Rotated Weierstrass Function"),
    7: ("schwefel", "Shifted and Rotated Schwefel's Function"),
    8: ("escaffer6", "Shifted and Rotated Expanded Schaffer's F6 Function"),
    9: ("happycat", "Shifted and Rotated HappyCat Function"),
    10: ("ackley", "Shifted and Rotated Ackley Function"),
}
SHIFTED_DIMENSIONS, SHIFTED_BOUND = 10, 100.0


def _rows(x) -> tuple[np.ndarray, tuple[int, ...]]:
    x = np.asarray(x, dtype=float)
    return x.reshape(-1, x.shape[-1]), x.shape[:-1]


def chebyshev(x: np.ndarray) -> np.ndarray:
    """Storn's Chebyshev fitting (rows × D): the coefficients of a polynomial that must stay in [-1, 1] on [-1, 1]
    (sampled at 32·D + 1 points) and reach a target at 1.2."""
    n = x.shape[1]
    a, b = 1.0, 1.2
    target = b
    for _ in range(n - 2):
        target = 2.4 * b - a
        a, b = b, target
    samples = 32 * n
    points = -1.0 + np.arange(samples + 1) * (2.0 / samples)
    p = np.repeat(x[:, :1], samples + 1, axis=1)
    for j in range(1, n):
        p = points * p + x[:, j : j + 1]
    outside = np.abs(p) > 1
    total = np.sum(np.where(outside, (1.0 - np.abs(p)) ** 2, 0.0), axis=1)
    end = x[:, 0].copy()
    for j in range(1, n):
        end = 1.2 * end + x[:, j]
    return total + 2 * np.where(end < target, end * end, 0.0)


def hilbert(x: np.ndarray) -> np.ndarray:
    """Inverse Hilbert matrix (rows × k²): the summed absolute deviation of H·X from the identity, X the k × k
    matrix of the position read row by row."""
    k = int(np.sqrt(x.shape[1]))
    h = 1.0 / (np.arange(k)[:, None] + np.arange(k)[None, :] + 1.0)
    product = h @ x.reshape(-1, k, k)
    return np.sum(np.abs(product - np.eye(k)), axis=(1, 2))


def lennard_jones(x: np.ndarray) -> np.ndarray:
    """Lennard-Jones energy of D / 3 atoms in space (rows × D), offset so the least energy of six atoms is about 0."""
    atoms = x.reshape(len(x), -1, 3)
    i, j = np.triu_indices(atoms.shape[1], 1)
    squared = np.sum((atoms[:, i] - atoms[:, j]) ** 2, axis=2)
    cubed = squared**3
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        pair = np.where(cubed > 1.0e-10, (1.0 / cubed - 2.0) / cubed, 1.0e20)
    return np.sum(pair, axis=1) + LENNARD_JONES_OFFSET


SPECIAL = {  # number: (function, name, dimensions, bound)
    1: (chebyshev, "Storn's Chebyshev Polynomial Fitting Problem", 9, 8192.0),
    2: (hilbert, "Inverse Hilbert Matrix Problem", 16, 16384.0),
    3: (lennard_jones, "Lennard-Jones Minimum Energy Cluster", 18, 4.0),
}


@dataclass(frozen=True, eq=False)
class Special:
    """F1-F3: a reference function with no shift or rotation."""

    number: int

    def batch(self, population: np.ndarray) -> np.ndarray:
        rows, shape = _rows(population)
        return (SPECIAL[self.number][0](rows) + OPTIMUM).reshape(shape)

    def __call__(self, x: np.ndarray) -> float:
        return float(self.batch(np.asarray(x, dtype=float)[None])[0])


@dataclass(frozen=True, eq=False)
class Shifted:
    """F4-F10: a basic CEC function shifted and rotated by the official data."""

    number: int
    shift: np.ndarray
    rotation: np.ndarray

    def batch(self, population: np.ndarray) -> np.ndarray:
        rows, shape = _rows(population)
        basic = BASIC[SHIFTED[self.number][0]]
        return (basic(rows, self.shift, self.rotation, True, True, np.zeros_like(rows)) + OPTIMUM).reshape(shape)

    def __call__(self, x: np.ndarray) -> float:
        return float(self.batch(np.asarray(x, dtype=float)[None])[0])


def group(number: int) -> str:
    """ "special" for F1-F3 (the reference problems), "multimodal" for the shifted and rotated F4-F10."""
    return "special" if number in SPECIAL else "multimodal"


@cache
def get(number: int) -> Function:
    """Function `number` (1 to 10) at its official dimension and bounds."""
    if number in SPECIAL:
        _, name, dimensions, bound = SPECIAL[number]
        return Function(f"F{number}", name, Special(number), -bound, bound, dimensions, OPTIMUM)
    if number not in SHIFTED:
        raise ValueError(f"CEC 2019 has functions 1 to 10, not {number}.")
    d = SHIFTED_DIMENSIONS
    solve = Shifted(number, data.shifts(YEAR, number, d, 1)[0], data.rotations(YEAR, number, d, 1)[0])
    return Function(f"F{number}", SHIFTED[number][1], solve, -SHIFTED_BOUND, SHIFTED_BOUND, d, OPTIMUM)


suite = Suite("CEC 2019", [get(n) for n in range(1, 11)])  # `problems.functions.digits`

__all__ = ["OPTIMUM", "chebyshev", "get", "group", "hilbert", "lennard_jones", "suite"]
