"""A function of two variables drawn over its box, as contours or as a surface, with the herd on it.

    Contour(f, (-5, 5))                                   # filled, in Viridis
    Contour(f, (-5, 5), fill=False, colors="Ice")         # coloured lines only, in another scale
    Contour(f).paths(run.positions).herd(run.positions[-1], best=run.position)
    Contour(f, theme="classic").arrows(run.positions[0], run.positions[-1]).herd(best=run.position)
    Surface(f, (-5, 5)).herd(run.positions[-1], best=run.position)

How they are drawn by default (filled or lines, log or not, the colours and markers) is the figure's theme
(`plots.themes`); any argument given wins over it.

The box defaults to the function's own `lb` and `ub`. A function of more than two variables that cannot be called on
two (Kowalik, Hartmann, Shekel, ...) is drawn with its other coordinates at the middle of the box, or at `fixed`.
"""

from typing import Self

import numpy as np
import plotly.graph_objects as go

from plots.fields import Bounds, Function, contour, corners, heights, lift, surface
from plots.markers import EDGE, herd, points
from plots.plot import Plot

PATH = "#9aa0a6"


def own_bounds(f, bounds: Bounds | None) -> Bounds:
    """`bounds`, or the function's own (`f.lb`, `f.ub`)."""
    if bounds is not None:
        return bounds
    if not (hasattr(f, "lb") and hasattr(f, "ub")):
        raise TypeError("Give the box as bounds=(lb, ub); the function has no lb and ub of its own.")
    return f.lb, f.ub


class Contour(Plot):
    """Contours of a function over a box, filled (`fill=True`) or as coloured lines, in the colorscale `colors`
    (the theme's: Viridis, or the classic theme's Plasma); `log` spaces the contours on a log scale above the lowest
    value.

    With `rows`/`columns` it is a grid, and `add(f, bounds, row=, col=)` draws a function in each cell.
    """

    DEFAULT_SIZE = "map"
    STATE = (*Plot.STATE, "_style")

    def __init__(
        self,
        f: Function | None = None,
        bounds: Bounds | None = None,
        /,
        *,
        fill: bool | None = None,
        colors=None,
        log: bool | None = None,
        resolution: int = 100,
        levels: int = 18,
        scale: bool = False,
        fixed=None,
        **options,
    ):
        super().__init__(**options)
        look = self.theme
        self._style = {
            "fill": look.fill if fill is None else fill,
            "colors": self.colorscale if colors is None else colors,
            "log": look.log if log is None else log,
            "crop": look.crop,
            "resolution": resolution,
            "levels": levels,
            "showscale": scale,
        }
        if f is not None:
            self.add(f, bounds, fixed=fixed)

    def add(
        self,
        f: Function,
        bounds: Bounds | None = None,
        /,
        *,
        fixed=None,
        row: int | None = None,
        col: int | None = None,
        **style,
    ) -> Self:
        """Contours of `f` over `bounds` (its own when left out), in the cell at `row`, `col`; `style` overrides any
        of the figure's `fill`, `colors`, `log`, `resolution`, `levels` and `scale` for this one."""
        bounds = own_bounds(f, bounds)
        if "scale" in style:
            style["showscale"] = style.pop("scale")
        contour(self, f, bounds, rest=lift(f, bounds, fixed), row=row, col=col, **(self._style | style))
        lb, ub = corners(bounds)
        frame = {"showgrid": False, "zeroline": False, "constrain": "domain", "row": row, "col": col}
        self.update_xaxes(range=[lb[0], ub[0]], **frame)
        self.update_yaxes(range=[lb[1], ub[1]], **frame)
        if row is None and col is None:
            self.update_yaxes(scaleanchor="x", scaleratio=1)
        return self

    def herd(
        self, positions=None, /, best=None, *, size: float = 8, row: int | None = None, col: int | None = None
    ) -> Self:
        """Positions (n × 2) as dots in the palette's red, and `best` (a point, or several such as the last herd) in
        its green, as the theme's best marker; either may be left out."""
        herd(self, positions, best, size=size, row=row, col=col)
        return self

    def arrows(
        self,
        start,
        end,
        /,
        *,
        color: str | None = None,
        width: float = 1.5,
        row: int | None = None,
        col: int | None = None,
    ) -> Self:
        """An arrow in `color` (the palette's red) from each point of `start` (n × 2) to the matching point of `end`,
        such as each agent's first position to its last: the way the herd went. An agent missing from either (NaN,
        having left the herd) gets none."""
        start, end = (np.asarray(p, dtype=float).reshape(-1, 2) for p in (start, end))
        refs = {"xref": "x", "yref": "y"} if row is None and col is None else {"row": row, "col": col}
        for (ax, ay), (x, y) in zip(start, end, strict=True):
            if not np.isfinite([ax, ay, x, y]).all():
                continue
            self.add_annotation(
                x=x,
                y=y,
                ax=ax,
                ay=ay,
                text="",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.2,
                arrowwidth=width,
                arrowcolor=color or self.palette.alert,
                opacity=0.9,
                **refs,
            )
            note = self.layout.annotations[-1]
            note.update(axref=note.xref, ayref=note.yref)
        return self

    def segments(self, start, end, /, *, dash: str = "dash", row: int | None = None, col: int | None = None) -> Self:
        """A straight line in the palette's red, dashed unless `dash` says otherwise, from each point of `start`
        (n × 2) to the matching point of `end`: where each agent began and where it ended."""
        start, end = (np.asarray(p, dtype=float).reshape(-1, 2) for p in (start, end))
        gap = np.full((len(start), 1), None)
        xs = np.hstack([start[:, :1], end[:, :1], gap]).ravel().tolist()
        ys = np.hstack([start[:, 1:], end[:, 1:], gap]).ravel().tolist()
        line = {"color": self.palette.alert, "dash": dash, "width": 1.5}
        lines = go.Scatter(x=xs, y=ys, mode="lines", line=line, opacity=0.8, name="start to end", showlegend=False)
        self.add_trace(lines, row=row, col=col)
        return self

    def trajectories(self, positions, /, *, row: int | None = None, col: int | None = None) -> Self:
        """Each agent's path over the seasons (`positions`: seasons × agents × 2) as a line with a dot at every
        season, agent `k` in the palette's colour `k`, round again after the last."""
        positions = np.asarray(positions, dtype=float)
        colors = self.palette.colors
        for k in range(positions.shape[1]):
            color = colors[k % len(colors)]
            path = go.Scatter(
                x=positions[:, k, 0],
                y=positions[:, k, 1],
                mode="lines+markers",
                line={"color": color, "width": 1.5},
                marker={"color": color, "size": 4},
                name=f"agent {k + 1}",
                legendgroup="trajectories",
                showlegend=False,
                hoverinfo="skip",
            )
            self.add_trace(path, row=row, col=col)
        return self

    def paths(self, positions, /, *, row: int | None = None, col: int | None = None) -> Self:
        """Each agent's path over the seasons (`positions`: seasons × agents × 2), as a thin grey line."""
        positions = np.asarray(positions, dtype=float)
        for k in range(positions.shape[1]):
            path = positions[:, k]
            line = go.Scatter(
                x=path[:, 0],
                y=path[:, 1],
                mode="lines",
                line={"color": PATH, "width": 0.8},
                opacity=0.5,
                showlegend=False,
                hoverinfo="skip",
            )
            self.add_trace(line, row=row, col=col)
        return self

    def visited(self, positions, /, *, row: int | None = None, col: int | None = None) -> Self:
        """Every position the agents stood on (`positions`: seasons × agents × 2), as red dots (small and faded, or
        full-size ones, as the classic theme draws them)."""
        size, opacity = self.theme.visited
        marker = {"color": self.palette.alert, "size": size, "opacity": opacity}
        points(self, positions, "visited", marker, legend=False, row=row, col=col)
        return self

    def start(self, positions, /, *, row: int | None = None, col: int | None = None) -> Self:
        """Where the agents started (n × 2), as dots in the palette's fifth colour, named "first season"."""
        marker = {"color": self.palette.color(4), "size": 7, "line": EDGE}
        points(self, positions, "first season", marker, row=row, col=col)
        return self

    def star(self, point, /, *, row: int | None = None, col: int | None = None) -> Self:
        """A point such as the minimiser, as a star in the palette's red."""
        marker = {"symbol": "star", "size": 13, "color": self.palette.alert, "line": EDGE}
        points(self, [point], "minimiser", marker, legend=False, row=row, col=col)
        return self


class Surface(Plot):
    """The surface of a function over a box in three dimensions, in the colorscale `colors`, with its contours on the
    floor (`floor=False` leaves them out); `camera` is Plotly's scene camera, such as `{"eye": {"x": 1.6, ...}}`."""

    DEFAULT_SIZE = "map"
    STATE = (*Plot.STATE, "_function", "_rest")

    def __init__(
        self,
        f: Function | None = None,
        bounds: Bounds | None = None,
        /,
        *,
        colors=None,
        opacity: float | None = None,
        floor: bool = True,
        resolution: int = 60,
        fixed=None,
        camera: dict | None = None,
        **options,
    ):
        super().__init__(**options)
        self._function, self._rest = f, np.array([])
        if f is not None:
            bounds = own_bounds(f, bounds)
            self._rest = lift(f, bounds, fixed)
            opacity = self.theme.opacity if opacity is None else opacity
            colors = self.colorscale if colors is None else colors
            surface(self, f, bounds, resolution, opacity, floor, colors=colors, rest=self._rest)
        self.update_scenes(xaxis_title="x₁", yaxis_title="x₂", zaxis_title="f")
        self.update_layout(margin={"l": 10, "r": 10, "b": 10})
        if camera:
            self.update_layout(scene_camera=camera)

    def _heights(self, positions) -> np.ndarray:
        """The function's value at each two-dimensional position, on the surface."""
        if self._function is None:
            raise ValueError("The herd sits on the surface: make the Surface with a function first.")
        return heights(self._function, positions, self._rest)

    def herd(self, positions=None, /, best=None, *, size: float = 8) -> Self:
        """Positions (n × 2) on the surface as red dots, and `best` (a point, or several) in green, as the theme's
        best marker; either may be left out."""
        best_z = None if best is None else self._heights(np.reshape(best, (-1, 2)))
        z = None if positions is None else self._heights(positions)
        herd(self, positions, best, z=z, best_z=best_z, size=size)
        return self

    def segments(self, start, end, /, *, dash: str = "dash") -> Self:
        """A line in the palette's red, dashed unless `dash` says otherwise, from each point of `start` (n × 2) to the
        matching point of `end`, each end at the surface's height there: where each agent began and ended."""
        start, end = (np.asarray(p, dtype=float).reshape(-1, 2) for p in (start, end))
        rows = [[*a, self._heights([a])[0]] for a in start], [[*b, self._heights([b])[0]] for b in end]
        xs, ys, zs = [], [], []
        for a, b in zip(*rows, strict=True):
            xs += [a[0], b[0], None]
            ys += [a[1], b[1], None]
            zs += [a[2], b[2], None]
        line = {"color": self.palette.alert, "dash": dash, "width": 4}
        self.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=line, name="start to end", showlegend=False))
        return self

    def visited(self, positions, /) -> Self:
        """Every position the agents stood on (`positions`: seasons × agents × 2), on the surface, as red dots."""
        flat = np.asarray(positions, dtype=float).reshape(-1, 2)
        flat = flat[np.isfinite(flat).all(axis=1)]
        marker = {"color": self.palette.alert, "size": 3}
        points(self, flat, "visited", marker, z=self._heights(flat), legend=False)
        return self

    def paths(self, positions, /) -> Self:
        """Each agent's path over the seasons (`positions`: seasons × agents × 2), on the surface."""
        positions = np.asarray(positions, dtype=float)
        for k in range(positions.shape[1]):
            path = positions[:, k]
            line = go.Scatter3d(
                x=path[:, 0],
                y=path[:, 1],
                z=self._heights(path),
                mode="lines",
                line={"color": self.palette.alert, "width": 3},
                opacity=0.8,
                showlegend=False,
                hoverinfo="skip",
            )
            self.add_trace(line)
        return self
