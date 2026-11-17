"""Points on a landscape: the herd in the palette's red, its best in its green, and any other set of points.

`points` draws a set of two- or three-dimensional points as markers, under one legend entry; `herd` draws the herd
and its best with it, in two dimensions or, given heights, in three, with the markers of the figure's theme.
"""

import numpy as np
import plotly.graph_objects as go

from plots.series import figure_palette, figure_theme

EDGE = {"color": "white", "width": 1}
SOLID_3D = {"circle", "diamond", "square", "cross", "x"}  # the filled symbols Plotly's 3-D markers have


def points(
    fig: go.Figure,
    xy,
    name: str,
    marker: dict,
    z=None,
    legend: bool = True,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """The points `xy` (n × 2) as markers named `name`; with `z` (one height each) in three dimensions."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    common = {"x": xy[:, 0], "y": xy[:, 1], "mode": "markers", "name": name, "marker": marker, "showlegend": legend}
    if z is None:
        return fig.add_trace(go.Scatter(**common), row=row, col=col)
    return fig.add_trace(go.Scatter3d(z=np.asarray(z, dtype=float), **common))


def herd(
    fig: go.Figure,
    positions=None,
    best=None,
    z=None,
    best_z: float | None = None,
    size: float = 8,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """Positions (n × 2) as dots in the palette's red, and `best` (one point, or several) in its green, as the theme's
    best marker (a diamond, or the classic theme's star-diamond); with `z` (one height per position, and `best_z` for
    the best) they are drawn in three dimensions. Either may be left out."""
    colors, look = figure_palette(fig), figure_theme(fig)
    flat = z is None and best_z is None
    edge = {"line": EDGE} if look.edge else {}
    if positions is not None:
        dots = {"color": colors.alert, "size": size if flat else size * 0.75} | edge
        points(fig, positions, "herd", dots, z=z, row=row, col=col)
    if best is not None:
        symbol = look.best if flat or look.best in SOLID_3D else "diamond"
        mark = {"color": colors.good, "size": size * (look.best_size if flat else 1.2), "symbol": symbol}
        mark |= edge if flat else {}
        heights = None if flat else np.atleast_1d(best_z)
        points(fig, best, "best", mark, z=heights, row=row, col=col)
    return fig
