"""Measures of a run's herd over its seasons, from a `nio.Result` or its arrays.

`positions` is seasons × agents × dimensions and `fitnesses` seasons × agents, as a run with `history=True` records
them. An ibex recorded as NaN (one that left the herd) is skipped by every measure.

* `best_positions`: where the season's leader stood;
* `diversity` and `median_diversity`: how spread out the herd is;
* `step_sizes`: how far each ibex moved from one season to the next;
* `distance_to_best`: how far each ibex stood from the run's best position;
* `balance` and `transition`: the share of exploration against exploitation, and the season they cross;
* `fitness_summary`: the best, worst, average, median and spread of the herd's fitness per season;
* `variance` and `movement`: the spread per dimension, and how far the whole herd moved each season;
* `cumulative_improvement`, `speed` and `top`: how far each ibex and the best have come, and the k best per season.
"""

import warnings

import numpy as np


def _quietly(reduce, *args, **kwargs):
    """A NaN-skipping numpy reduction, silent about seasons with no ibex left (they come out NaN)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"(All-NaN slice|Mean of empty slice|Degrees of freedom)", RuntimeWarning)
        return reduce(*args, **kwargs)


def best_positions(result) -> np.ndarray:
    """The position of each season's best ibex (seasons × dimensions)."""
    fitnesses = np.where(np.isnan(result.fitnesses), np.inf, result.fitnesses)
    leaders = np.argmin(fitnesses, axis=1)
    return result.positions[np.arange(len(leaders)), leaders]


def diversity(positions: np.ndarray) -> np.ndarray:
    """The standard deviation of every coordinate of the herd, per season."""
    return _quietly(np.nanstd, positions.reshape(len(positions), -1), axis=1)


def median_diversity(positions: np.ndarray) -> np.ndarray:
    """The mean distance of the coordinates from the herd's median, per season."""
    centre = _quietly(np.nanmedian, positions, axis=1, keepdims=True)
    return _quietly(np.nanmean, np.abs(positions - centre), axis=(1, 2))


def step_sizes(positions: np.ndarray) -> np.ndarray:
    """How far each ibex moved between consecutive seasons ((seasons - 1) × agents)."""
    return np.linalg.norm(np.diff(positions, axis=0), axis=-1)


def distance_to_best(result) -> np.ndarray:
    """How far each ibex stood from the run's best position, per season (seasons × agents)."""
    return np.linalg.norm(result.positions - result.position, axis=-1)


def smooth(values, window: int = 11) -> np.ndarray:
    """A moving average over `window` seasons (made odd), the ends padded with the end values; same length."""
    values = np.asarray(values, dtype=float)
    if window <= 1:
        return values.copy()
    window += window % 2 == 0
    padded = np.pad(values, window // 2, mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def balance(positions: np.ndarray, percentile: float = 95, window: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """Exploration and exploitation per season, in percent: the herd's spread against its `percentile` spread over
    the run, smoothed over `window` seasons. The two always add up to 100."""
    spread = median_diversity(positions)
    finite = spread[np.isfinite(spread)]
    scale = max(float(np.percentile(finite, percentile)) or float(finite.max()), 1e-12) if finite.size else 1.0
    explore = smooth(np.clip(np.nan_to_num(spread / scale * 100), 0, 100), window)
    return explore, 100 - explore


def transition(explore: np.ndarray) -> int | None:
    """The index of the first season whose exploration is below 50% after a season above it, or None when it never
    falls below; touching 50% is not a crossing."""
    explore = np.asarray(explore, dtype=float)
    above = np.flatnonzero(explore > 50)
    if not above.size:
        return None
    below = np.flatnonzero(explore[above[0] :] < 50)
    return int(above[0] + below[0]) if below.size else None


def fitness_summary(fitnesses: np.ndarray) -> dict[str, np.ndarray]:
    """The best, worst, average, median and standard deviation of the herd's fitness, per season."""
    return {
        "Best": _quietly(np.nanmin, fitnesses, axis=1),
        "Worst": _quietly(np.nanmax, fitnesses, axis=1),
        "Average": _quietly(np.nanmean, fitnesses, axis=1),
        "Median": _quietly(np.nanmedian, fitnesses, axis=1),
        "Deviation": _quietly(np.nanstd, fitnesses, axis=1),
    }


def variance(positions: np.ndarray) -> np.ndarray:
    """The variance of the herd in each dimension, per season (seasons × dimensions)."""
    return _quietly(np.nanvar, positions, axis=1)


def movement(positions: np.ndarray) -> np.ndarray:
    """The total distance the herd moved between consecutive seasons, as the sum of every coordinate's change
    (seasons - 1); an ibex missing from either season counts as still."""
    return np.nansum(np.abs(np.diff(positions, axis=0)), axis=(1, 2))


def cumulative_improvement(fitnesses: np.ndarray) -> np.ndarray:
    """How much each ibex's fitness has fallen since the first season, after each later one ((seasons - 1) × agents)."""
    return fitnesses[:1] - fitnesses[1:]


def speed(convergence) -> np.ndarray:
    """How much the best value fell in each season after the first (seasons - 1)."""
    return -np.diff(np.asarray(convergence, dtype=float))


def top(fitnesses: np.ndarray, k: int = 5) -> np.ndarray:
    """The `k` best fitnesses of each season, best first (k × seasons); missing ibex come last."""
    ordered = np.sort(np.where(np.isnan(fitnesses), np.inf, fitnesses), axis=1)[:, :k]
    return np.where(np.isinf(ordered), np.nan, ordered).T
