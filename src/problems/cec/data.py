"""The official CEC shift vectors, rotation matrices and shuffles, read exactly as the reference C loader reads them.

`data/cec/cec<year>.npz` holds, per data set k, the numbers of each official input file in file order: `M_<k>_D<d>`
(rotations), `shift_<k>` with `shift_<k>_lines` (the number of values on each line), and `shuffle_<k>_D<d>` (1-based).
"""

from functools import cache

import numpy as np

from problems.paths import DATA

CEC = DATA / "cec"


@cache
def archive(year: int) -> dict[str, np.ndarray]:
    """Every array of the year's data file."""
    with np.load(CEC / f"cec{year}.npz") as data:
        return {name: data[name] for name in data.files}


def has(year: int, key: str) -> bool:
    """Whether the year's data file holds the array `key`."""
    return key in archive(year)


def shifts(year: int, data: int, dimensions: int, count: int) -> np.ndarray:
    """`count` shift vectors: each takes the next D values, then the reader skips the rest of that line."""
    values, lines = archive(year)[f"shift_{data}"], archive(year)[f"shift_{data}_lines"]
    starts = np.concatenate([[0], np.cumsum(lines)])
    rows, position = [], 0
    for _ in range(count):
        rows.append(values[position : position + dimensions])
        last = position + dimensions - 1
        position = starts[np.searchsorted(starts, last, side="right")]  # start of the line after the last value read
    return np.array(rows)


def rotations(year: int, data: int, dimensions: int, count: int) -> np.ndarray:
    """`count` D × D rotation matrices, read row by row in sequence."""
    flat = archive(year)[f"M_{data}_D{dimensions}"]
    return flat[: count * dimensions * dimensions].reshape(count, dimensions, dimensions)


def shuffles(year: int, data: int, dimensions: int, count: int) -> np.ndarray:
    """`count` permutations of 0..D−1 (the files store them 1-based)."""
    flat = archive(year)[f"shuffle_{data}_D{dimensions}"]
    return flat[: count * dimensions].reshape(count, dimensions).astype(np.intp) - 1


def dimensions(year: int, data: int) -> tuple[int, ...]:
    """The dimensions with a rotation matrix for data set `data`."""
    prefix = f"M_{data}_D"
    return tuple(sorted(int(key.removeprefix(prefix)) for key in archive(year) if key.startswith(prefix)))
