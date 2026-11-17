"""Colorscales for contours and surfaces: any of Plotly's (Viridis by default), or one made from a palette.

`colors` is a Plotly colorscale name (`"Viridis"`, `"Ice"`, `"Cividis"`, ...), a list of `[position, colour]` stops,
or the name of a palette (`"soft"`, ...), whose `scale` runs from near white to its first colour. `None` takes the
session's, which `plots.use(scale=...)` sets.
"""

import numpy as np
from plotly.colors import sample_colorscale

from plots.defaults import current_scale
from plots.palettes import PALETTES


def resolve(colors=None) -> tuple[str | list, bool]:
    """The colorscale for `colors`, and whether to reverse it: a palette's scale is reversed on contours so that the
    low ground takes the palette's colour, as it always has."""
    chosen = current_scale() if colors is None else colors
    if isinstance(chosen, str) and chosen in PALETTES:
        return PALETTES[chosen].scale, True
    return chosen, False


def colorscale(colors=None) -> str | list:
    """The colorscale for `colors`: a palette name becomes the palette's scale; anything else is given as it is."""
    return resolve(colors)[0]


LINES_TOP = 0.85


def for_lines(scale: str | list, top: float = LINES_TOP) -> list:
    """`scale` without its last stretch above `top`, which is too pale to read as thin lines on a white page."""
    at = list(np.linspace(0, 1, 9))
    return [[t, c] for t, c in zip(at, sample_colorscale(scale, [top * a for a in at]), strict=True)]


CLEAR = "rgba(255,255,255,0)"


def from_clear(scale: str | list) -> list:
    """`scale` rising from clear: its lowest stop transparent, so ground with nothing on it (a density of 0) stays
    the page's colour."""
    colors = sample_colorscale(scale, list(np.linspace(0, 1, 9)))
    return [[0.0, CLEAR], *([float(t), c] for t, c in zip(np.linspace(0.05, 1, 9), colors, strict=True))]
