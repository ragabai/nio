"""Values over a grid: a heatmap of a table, and the density of a cloud of points.

Heatmap(result.fitnesses.T, seasons, ibex, log=True, x="Season", y="Ibex")
Density(xs, ys, bounds=(-30, 30), x="x₁", y="x₂")

Both are in the figure's colorscale (the theme's: Viridis, or the classic theme's Plasma) unless `colors=` says
otherwise.
"""

from typing import Self

import numpy as np
import plotly.graph_objects as go

from plots.plot import Plot
from plots.scales import from_clear, resolve


class Heatmap(Plot):
    """A table of values as coloured cells: `z` is rows × columns, `x` labels the columns and `y` the rows. `log`
    colours by the base-10 logarithm (cells that are not positive are left blank); `fmt` (a d3 format such as `".2f"`
    or `".0%"`) writes each value in its cell."""

    def __init__(self, z=None, x=None, y=None, /, *, log: bool = False, fmt: str | None = None, colors=None, **options):
        super().__init__(**options)
        self._cells = {"log": log, "fmt": fmt, "colors": colors}
        if z is not None:
            self.add(z, x, y)

    def add(self, z, x=None, y=None, /, *, row: int | None = None, col: int | None = None, **cells) -> Self:
        """One table, in the cell at `row`, `col`; `cells` overrides the figure's `log`, `fmt` and `colors`."""
        chosen = self._cells | cells
        z = np.asarray(z, dtype=float)
        if chosen["log"]:
            with np.errstate(divide="ignore", invalid="ignore"):
                z = np.where(z > 0, np.log10(np.where(z > 0, z, 1.0)), np.nan)
        scale, reverse = resolve(self.colorscale if chosen["colors"] is None else chosen["colors"])
        text = {"texttemplate": f"%{{z:{chosen['fmt']}}}"} if chosen["fmt"] else {}
        bar = {"title": {"text": "log₁₀" if chosen["log"] else ""}, "thickness": 12}
        cells = go.Heatmap(z=z, x=x, y=y, colorscale=scale, reversescale=reverse, colorbar=bar, **text)
        self.add_trace(cells, row=row, col=col)
        return self


class Density(Plot):
    """How densely the points (`x`, `y`) lie, counted in `bins` × `bins` bins: as filled contours, or (`contours=False`)
    as the bins themselves; where there are none the page shows through. `bounds` = (low, high) sets the box drawn,
    the same on both axes or one pair per axis."""

    def __init__(
        self,
        x=None,
        y=None,
        /,
        *,
        bins: int = 40,
        bounds=None,
        contours: bool = True,
        colors=None,
        **options,
    ):
        super().__init__(**options)
        self._style = {"bins": bins, "contours": contours, "colors": colors}
        if x is not None:
            self.add(x, y, bounds=bounds)

    def add(self, x, y, /, *, bounds=None, row: int | None = None, col: int | None = None) -> Self:
        """One cloud of points, in the cell at `row`, `col`."""
        style = self._style
        scale, _ = resolve(self.colorscale if style["colors"] is None else style["colors"])
        common = {
            "x": np.asarray(x, dtype=float).ravel(),
            "y": np.asarray(y, dtype=float).ravel(),
            "nbinsx": style["bins"],
            "nbinsy": style["bins"],
            "colorscale": from_clear(scale),
            "showscale": False,
        }
        kind = go.Histogram2dContour if style["contours"] else go.Histogram2d
        self.add_trace(kind(**common), row=row, col=col)
        if bounds is not None:
            (x_low, x_high), (y_low, y_high) = _box(bounds)
            self.update_xaxes(range=[x_low, x_high], row=row, col=col)
            self.update_yaxes(range=[y_low, y_high], row=row, col=col)
        return self


def _box(bounds) -> tuple[tuple[float, float], tuple[float, float]]:
    """The (low, high) range of each axis: `bounds` = (low, high) for both, or ((x low, x high), (y low, y high))."""
    low, high = bounds
    if np.ndim(low) == 0:
        return (float(low), float(high)), (float(low), float(high))
    return (float(low[0]), float(high[0])), (float(low[1]), float(high[1]))
