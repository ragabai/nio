"""Stacked areas and markers.

Area(seasons, {"walk": walked, "jump": jumped}, title="Behaviours used", x="Season", y="Ibex")
Scatter(seasons, scores, "herd of 30", x="Seasons").add(seasons, others, name="herd of 50")
"""

from collections.abc import Mapping, Sequence
from typing import Self

import plotly.graph_objects as go

from plots.plot import Plot
from plots.series import style


class Area(Plot):
    """Series stacked on one another, each a filled band: shares of a whole over an axis. `Area(x, {name: values})`
    stacks the series in the order given; `add` stacks one more on top."""

    def __init__(self, x=None, ys: Mapping[str, Sequence] | None = None, /, **options):
        super().__init__(**options)
        for name, values in (ys or {}).items():
            self.add(x, values, name)

    def add(
        self,
        x,
        y,
        /,
        name: str | None = None,
        *,
        color: str | None = None,
        row: int | None = None,
        col: int | None = None,
        **trace,
    ) -> Self:
        """One more layer, in the next palette colour unless `color` is given."""
        chosen = style(self, name, color=color)
        line = {"color": chosen.color, "width": 0.5}
        layer = go.Scatter(x=x, y=y, name=name, stackgroup="all", line=line, **chosen.trace, **trace)
        self.add_trace(layer, row=row, col=col)
        return self


class Scatter(Plot):
    """Points as markers. `Scatter(x, y, name)` draws a first set; `add` draws more."""

    def __init__(self, x=None, y=None, /, name: str | None = None, **options):
        super().__init__(**options)
        if x is not None:
            self.add(x, y, name)

    def add(
        self,
        x,
        y,
        /,
        name: str | None = None,
        *,
        size: float = 6,
        color: str | None = None,
        row: int | None = None,
        col: int | None = None,
        **trace,
    ) -> Self:
        """One set of points, in the next palette colour unless `color` is given; `trace` takes any Plotly scatter
        option (`opacity`, `marker_symbol`, ...)."""
        chosen = style(self, name, color=color)
        marker = {"color": chosen.color, "size": size} | dict(trace.pop("marker", {}))
        dots = go.Scatter(x=x, y=y, mode="markers", name=name, marker=marker, **chosen.trace, **trace)
        self.add_trace(dots, row=row, col=col)
        return self
