"""The shifted suites: the 23 benchmark functions with their optima moved away from the origin and the line (1, ..., 1).

In the classic suite F1-F7 and F9-F13 all have their optimum at (or, for F5, F12 and F13, next to) the origin, and
F5, F12 and F13 sit on the line through the origin along (1, ..., 1). A method that pulls its search toward the box
centre or steps along that line looks better there than it is. The shifted suites remove that advantage:

* `fair` shifts F1-F7 and F9-F13: `f_shift(x) = f(x - o)` moves the minimiser from `x*` to `o + x*` and leaves the
  optimal value unchanged. `o` is drawn once, from a fixed seed (2026), in `[0.8 lb, 0.8 ub]` per coordinate, so the
  shifted optimum stays inside the bounds. The other functions are those of the classic suite.
* `fair23` also shifts F8 and F14-F23, with a fixed seed (2027). Their optima lie near the box edge or have a fixed
  position, so the shift is drawn so that every known global minimiser stays inside the box, and the function is
  evaluated at `clip(x - o, lb, ub)`: the box is unchanged and the base function is never read outside its domain.

At another dimension (`fair.at(dim=50)`, `rosenbrock.at(dim=50).shifted()`) a scalable function gets a fresh shift,
drawn by the same rule from a generator seeded with (seed, function number, dimension), so it is the same every time.
"""

import numpy as np

from problems.classic import NAMES, classic
from problems.function import Function
from problems.suite import Suite

SEED = 2026
SHIFTED_KEYS = [f"F{i}" for i in [*range(1, 8), *range(9, 14)]]


def _optimum(key: str, dim: int) -> np.ndarray:
    """The classic minimiser x* of a function `fair` shifts: zeros, ones for F5 and F13, minus ones for F12."""
    return {"F5": np.ones(dim), "F12": -np.ones(dim), "F13": np.ones(dim)}.get(key, np.zeros(dim))


# The classic minimisers x* of the functions `fair` shifts (zeros unless listed).
OPTIMA: dict[str, np.ndarray] = {key: _optimum(key, classic[key].dim) for key in SHIFTED_KEYS}


def _draw() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)
    return {
        key: rng.uniform(0.8 * classic[key].lb, 0.8 * classic[key].ub, size=classic[key].dim) for key in SHIFTED_KEYS
    }


# The shift o of each function `fair` shifts.
SHIFTS: dict[str, np.ndarray] = _draw()


def shift(key: str, dim: int) -> np.ndarray:
    """The shift o of `fair`'s function `key` at `dim` dimensions."""
    fn = classic[key]
    if dim == fn.dim:
        return SHIFTS[key]
    return np.random.default_rng([SEED, int(key[1:]), dim]).uniform(0.8 * fn.lb, 0.8 * fn.ub, size=dim)


def _fair(key: str, dim: int | None = None) -> Function:
    fn = classic[key].at(dim=dim)
    o = shift(key, fn.dim)
    return Function(
        fn.key,
        fn.name + " (shifted)",
        lambda x: fn.solve(x - o),
        fn.lb,
        fn.ub,
        fn.dim,
        fn.optimal,
        resize=lambda d: _fair(key, d),
        fixed_box=True,
    )


fair = Suite("fair", [_fair(f.key) if f.key in SHIFTS else f for f in classic], names=NAMES)


SEED23 = 2027
_SHEKEL_X = np.array([4.00004, 4.00013, 4.00004, 4.00013])
_SCHWEFEL_X = 420.968746  # every coordinate of the Schwefel 2.26 (F8) minimiser

# Every known global minimiser x* of the functions `fair` leaves alone (each matches `optimal` to 1e-4).
MINIMISERS: dict[str, list[np.ndarray]] = {
    "F8": [np.full(30, _SCHWEFEL_X)],
    "F14": [np.full(2, -31.97833)],
    "F15": [np.array([0.1928, 0.1908, 0.1231, 0.1358])],
    "F16": [np.array([0.08984201, -0.7126564]), np.array([-0.08984201, 0.7126564])],
    "F17": [np.array([-np.pi, 12.275]), np.array([np.pi, 2.275]), np.array([9.42478, 2.475])],
    "F18": [np.array([0.0, -1.0])],
    "F19": [np.array([0.114614, 0.555649, 0.852547])],
    "F20": [np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])],
    "F21": [_SHEKEL_X.copy()],
    "F22": [_SHEKEL_X.copy()],
    "F23": [_SHEKEL_X.copy()],
}


def _window(fn: Function, xs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """o per dimension in [max(lb - x*, -0.1 span), min(ub - x*, 0.1 span)], shrunk 1 % of the span, for every x*."""
    lb = np.broadcast_to(np.asarray(fn.lb, dtype=float), (fn.dim,))
    ub = np.broadcast_to(np.asarray(fn.ub, dtype=float), (fn.dim,))
    span = ub - lb
    lo = np.max([np.maximum(lb - x, -0.1 * span) for x in xs], axis=0) + 0.01 * span
    hi = np.min([np.minimum(ub - x, 0.1 * span) for x in xs], axis=0) - 0.01 * span
    return lo, hi


def _draw23() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED23)
    return {key: rng.uniform(*_window(classic[key], xs)) for key, xs in MINIMISERS.items()}


# The shift o of each function `fair23` adds to the `fair` suite.
SHIFTS23: dict[str, np.ndarray] = _draw23()


def shift23(key: str, dim: int) -> np.ndarray:
    """The shift o of `fair23`'s own function `key` (F8 or F14-F23) at `dim` dimensions; only F8 is scalable."""
    fn = classic[key].at(dim=dim)
    if dim == classic[key].dim:
        return SHIFTS23[key]
    return np.random.default_rng([SEED23, int(key[1:]), dim]).uniform(*_window(fn, [np.full(dim, _SCHWEFEL_X)]))


def _fair23(key: str, dim: int | None = None) -> Function:
    """f(clip(x - o, lb, ub)): the box is unchanged and the base function is never read outside its own domain."""
    fn = classic[key].at(dim=dim)
    o = shift23(key, fn.dim)
    lb, ub = np.asarray(fn.lb, dtype=float), np.asarray(fn.ub, dtype=float)
    return Function(
        fn.key,
        fn.name + " (shifted)",
        lambda x: fn.solve(np.clip(x - o, lb, ub)),
        fn.lb,
        fn.ub,
        fn.dim,
        fn.optimal,
        resize=None if fn.resize is None else lambda d: _fair23(key, d),
        fixed_box=True,
    )


fair23 = Suite("fair23", [_fair23(f.key) if f.key in SHIFTS23 else f for f in fair], names=NAMES)
