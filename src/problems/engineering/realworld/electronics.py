"""RC45–RC50: synchronous optimal pulse-width modulation (SOPWM) of multilevel inverters.

The switching angles (degrees, ascending) minimise the weighted total harmonic distortion over the harmonics
5..97 not divisible by 3, while the fundamental matches the modulation index.
"""

import numpy as np

from problems.engineering.realworld.common import stack

HARMONICS = np.array(
    [
        5,
        7,
        11,
        13,
        17,
        19,
        23,
        25,
        29,
        31,
        35,
        37,
        41,
        43,
        47,
        49,
        53,
        55,
        59,
        61,
        65,
        67,
        71,
        73,
        77,
        79,
        83,
        85,
        91,
        95,
        97,
    ]
)


def _sopwm(x: np.ndarray, signs, levels: int, modulation: float):
    n = len(x)
    signs = np.asarray(signs, dtype=float)
    radians = x * np.pi / 180
    harmonics = (signs * np.cos(HARMONICS[None, :, None] * radians[:, None, :])).sum(axis=2)  # (n, harmonics)
    distortion = np.sqrt((harmonics**2 / HARMONICS**4).sum(axis=1)) / np.sqrt((1 / HARMONICS**4).sum())
    g = stack(n, *(x[:, i] - x[:, i + 1] + 1e-6 for i in range(x.shape[1] - 1)))
    h = stack(n, (signs * np.cos(radians)).sum(axis=1) - levels * modulation)
    return distortion / levels, g, h


def rc45(x):
    """SOPWM for 3-level inverters."""
    return _sopwm(x, (-np.ones(25)) ** np.arange(2, 27), 1, 0.32)


def rc46(x):
    """SOPWM for 5-level inverters."""
    return _sopwm(x, [1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1], 2, 0.32)


def rc47(x):
    """SOPWM for 7-level inverters."""
    return _sopwm(x, [1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1], 3, 0.36)


def rc48(x):
    """SOPWM for 9-level inverters."""
    return _sopwm(
        x,
        [1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1, -1, 1],
        4,
        0.32,
    )


def rc49(x):
    """SOPWM for 11-level inverters."""
    return _sopwm(
        x,
        [1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1],
        5,
        0.3333,
    )


def rc50(x):
    """SOPWM for 13-level inverters."""
    return _sopwm(
        x,
        [1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1],
        6,
        0.32,
    )
