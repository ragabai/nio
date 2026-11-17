"""The statistical charts behind the classes, each more than one Plotly call, added to a figure.

Every function takes the figure first, `row` and `col` to place it in a grid, and returns the figure. Colours come from
the figure's palette (`plots.series`): `series=None` takes the next colour.

* `band`: a line with a shaded band, such as a median and its quartiles;
* `agents`: many thin lines under one legend entry, such as every ibex's fitness;
* `boxes`: a box per group with every value as a dot, grouped by series when called with several names;
* `violins`: a violin per group, side by side by series like `boxes`;
* `bars`: horizontal bars labelled with their values, with an optional (low, high) spread;
* `hist`, and `dual` (a second y axis).
"""

from collections.abc import Mapping, Sequence

import numpy as np
import plotly.graph_objects as go

from plots.axes import decade_step, log_axis
from plots.series import figure_palette, style

LINE = 2.2


def band(
    fig: go.Figure,
    x,
    median,
    low,
    high,
    name: str | None = None,
    series: int | None = None,
    log: bool = False,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """`median` as a line and the area from `low` to `high` around it, as one legend entry; points where the band
    is not finite (such as before a run's first value) are left out."""
    x, median, low, high = (np.asarray(v, dtype=float) for v in (x, median, low, high))
    chosen = style(fig, name, series)
    kept = np.isfinite(low) & np.isfinite(high)
    area = go.Scatter(
        x=np.concatenate([x[kept], x[kept][::-1]]),
        y=np.concatenate([high[kept], low[kept][::-1]]),
        fill="toself",
        fillcolor=chosen.color,
        opacity=0.16,
        line={"width": 0},
        hoverinfo="skip",
        legendgroup=chosen.group,
        showlegend=False,
    )
    fig.add_trace(area, row=row, col=col)
    line = go.Scatter(x=x, y=median, name=name, line={"color": chosen.color, "width": LINE}, **chosen.trace)
    fig.add_trace(line, row=row, col=col)
    return log_axis(fig, "y", row, col) if log else fig


def agents(
    fig: go.Figure,
    ys,
    x=None,
    name: str = "agents",
    series: int | None = None,
    opacity: float = 0.25,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """One thin line per row of `ys` (series × points), all under one legend entry."""
    ys = np.atleast_2d(np.asarray(ys, dtype=float))
    x = np.arange(1, ys.shape[1] + 1) if x is None else np.asarray(x)
    chosen = style(fig, name, series)
    for k, y in enumerate(ys):
        line = go.Scatter(
            x=x,
            y=y,
            name=name,
            legendgroup=chosen.group,
            showlegend=k == 0,
            line={"color": chosen.color, "width": 1},
            opacity=opacity,
            hoverinfo="skip",
        )
        fig.add_trace(line, row=row, col=col)
    return fig


def boxes(
    fig: go.Figure,
    groups: Mapping[str, Sequence[float]],
    name: str | None = None,
    series: int | None = None,
    log: bool = False,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """A box of the values of each group, in the order given, with every value as a dot over it; calls with
    different `name`s set the boxes of each group side by side."""
    chosen = style(fig, name, series)
    labels = [label for label, values in groups.items() for _ in values]
    values = np.concatenate([np.asarray(v, dtype=float) for v in groups.values()]) if groups else []
    box = go.Box(
        x=labels,
        y=values,
        name=name,
        marker={"color": chosen.color, "size": 4},
        line={"width": 1.2},
        boxpoints="all",
        jitter=0.35,
        pointpos=0,
        **chosen.trace,
    )
    fig.add_trace(box, row=row, col=col)
    if name:
        fig.update_layout(boxmode="group")
    return log_axis(fig, "y", row, col) if log else fig


def violins(
    fig: go.Figure,
    groups: Mapping[str, Sequence[float]],
    name: str | None = None,
    series: int | None = None,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """A violin of the values of each group, in the order given, with a box inside; calls with different `name`s set
    the violins of each group side by side."""
    chosen = style(fig, name, series)
    labels = [label for label, values in groups.items() for _ in values]
    values = np.concatenate([np.asarray(v, dtype=float) for v in groups.values()]) if groups else []
    violin = go.Violin(
        x=labels,
        y=values,
        name=name,
        line={"color": chosen.color, "width": 1.2},
        fillcolor=chosen.color,
        opacity=0.6,
        box={"visible": True},
        meanline={"visible": True},
        points=False,
        **chosen.trace,
    )
    fig.add_trace(violin, row=row, col=col)
    if name:
        fig.update_layout(violinmode="group")
    return fig


def bars(
    fig: go.Figure,
    values: Mapping[str, float],
    spread: Mapping[str, tuple[float, float]] | None = None,
    fmt: str = "{:.3f}",
    name: str | None = None,
    series: int | None = None,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """Horizontal bars, the first label on top, each labelled with its value; `spread` adds a (low, high) range."""
    labels = list(values)
    heights = np.array([values[label] for label in labels], dtype=float)
    error = None
    if spread:
        low = np.array([spread[label][0] for label in labels], dtype=float)
        high = np.array([spread[label][1] for label in labels], dtype=float)
        error = {"type": "data", "array": high - heights, "arrayminus": heights - low, "thickness": 1, "width": 0}
    chosen = style(fig, name, series)
    bar = go.Bar(
        x=heights,
        y=labels,
        orientation="h",
        name=name,
        marker={"color": chosen.color, "line": {"width": 0}},
        text=[fmt.format(h) for h in heights],
        textposition="outside",
        cliponaxis=False,
        error_x=error,
        **chosen.trace,
    )
    fig.add_trace(bar, row=row, col=col)
    return fig.update_yaxes(autorange="reversed", row=row, col=col)


def hist(
    fig: go.Figure,
    values,
    name: str | None = None,
    series: int | None = None,
    bins: int | None = None,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """A histogram of `values`, in `bins` bins (Plotly's choice when left out)."""
    chosen = style(fig, name, series)
    marker = {"color": chosen.color, "line": {"width": 0}}
    histogram = go.Histogram(x=np.asarray(values, dtype=float), nbinsx=bins, name=name, marker=marker, **chosen.trace)
    fig.add_trace(histogram, row=row, col=col)
    return fig


def dual(fig: go.Figure, x, left: tuple[str, Sequence], right: tuple[str, Sequence], right_log: bool = False):
    """Two series on one x axis, `left` on the left y axis and `right` on the right one; each is (name, values).
    The figure must have a second y axis, as `Dual` and `Plot(secondary_y=True)` do."""
    second = getattr(fig.layout, "yaxis2", None)
    if second is None or second.overlaying != "y":
        raise ValueError("dual needs a second y axis: use plots.Dual, or Plot(secondary_y=True).")
    colors = figure_palette(fig)
    for k, (name, values) in enumerate((left, right)):
        line = go.Scatter(x=x, y=values, name=name, legendgroup=name, line={"color": colors.color(k), "width": LINE})
        fig.add_trace(line, secondary_y=k == 1)
    fig.update_yaxes(title=left[0], secondary_y=False)
    fig.update_yaxes(title=right[0], secondary_y=True, showgrid=False)
    if right_log:
        fig.update_yaxes(type="log", dtick=decade_step([right[1]]), exponentformat="power", secondary_y=True)
    return fig
