"""Line charts: series over an axis, a series with a shaded band, and many thin lines under one legend entry.

Line(seasons, result.convergence, title="Best value per season", x="Season", y="f(best)", log=True)

fig = Line(title="The herd's fitness", x="Season", log=True)
for name, values in summary.items():
    fig.add(seasons, values, name=name)
"""

from collections.abc import Sequence
from typing import Self

import numpy as np
import plotly.graph_objects as go

from plots.charts import LINE, agents, band, dual
from plots.plot import Plot
from plots.series import style


def _xy(x, y):
    """`(x, y)` from one or two value sequences: one is y, with x counting from 1."""
    if y is None:
        y = np.asarray(x)
        return np.arange(1, len(y) + 1), y
    return x, y


class Line(Plot):
    """Lines over a shared axis. `Line(y)` or `Line(x, y, name)` draws a first series; `add` draws more.

    Every keyword of `Plot` (`title`, `x`, `y`, `log`, `size`, `rows`, `columns`, `legend`, ...) applies.
    """

    def __init__(self, x=None, y=None, /, name: str | None = None, **options):
        super().__init__(**options)
        if x is not None:
            self.add(x, y, name)

    def add(
        self,
        x,
        y=None,
        /,
        name: str | None = None,
        *,
        color: str | None = None,
        row: int | None = None,
        col: int | None = None,
        **trace,
    ) -> Self:
        """One series, in the next palette colour unless `color` is given; `trace` takes any Plotly scatter option
        (`line_shape="hv"`, `mode="lines+markers"`, `opacity`, ...)."""
        x, y = _xy(x, y)
        chosen = style(self, name, color=color)
        line = {"color": chosen.color, "width": LINE} | dict(trace.pop("line", {}))
        self.add_trace(go.Scatter(x=x, y=y, name=name, line=line, **chosen.trace, **trace), row=row, col=col)
        return self

    def band(
        self,
        x,
        median,
        low,
        high,
        /,
        name: str | None = None,
        *,
        series: int | None = None,
        row: int | None = None,
        col: int | None = None,
    ) -> Self:
        """`median` as a line and the area from `low` to `high` around it, as one legend entry, in palette colour
        `series` (the next one when left out; give it to keep one series' colour in every cell of a grid)."""
        band(self, x, median, low, high, name, series, row=row, col=col)
        return self

    def agents(
        self,
        ys,
        x=None,
        /,
        name: str = "agents",
        *,
        opacity: float = 0.25,
        series: int | None = None,
        row: int | None = None,
        col: int | None = None,
    ) -> Self:
        """One thin line per row of `ys` (series × points), all under one legend entry, in palette colour `series`
        (the next one when left out)."""
        agents(self, ys, x, name, series, opacity, row, col)
        return self


class Dual(Line):
    """Two series on one x axis: `left` = (name, values) on the left y axis, `right` on a second one at the right,
    such as the herd's spread next to its best value. `right_log` makes the right axis logarithmic."""

    def __init__(
        self, x, left: tuple[str, Sequence], right: tuple[str, Sequence], /, right_log: bool = False, **options
    ):
        super().__init__(secondary_y=True, **options)
        dual(self, x, left, right, right_log)
