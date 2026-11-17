"""The report's figures, in the report's look (`simple_white` with the `soft` palette), drawn with `plots`' classes.

* `REPORT`: that look, as keywords for any figure class (`Line(size=(640, 300), **REPORT)`);
* `panels`: one convergence chart per problem, in a grid with one legend and shared axis titles;
* `landscape`: a function's log contours over its box, with its minimiser as a star;
* `herd_box`: positions inside the outline of their box.
"""

import math
from collections.abc import Callable, Mapping

import numpy as np

from benchmarks.curves import Band
from plots import Contour, Line

REPORT = {"template": "simple_white", "palette": "soft"}
OUTLINE = "#c8c7c2"
BARE = {"showticklabels": False, "showgrid": False, "ticks": "", "showline": False, "zeroline": False}


def panels(
    charts: Mapping[str, Mapping[str, Band]],
    columns: int = 5,
    width: int = 1100,
    row_height: int = 190,
    log: bool = True,
    references: Mapping[str, float] | None = None,
    ylabel: str = "Error",
    title: str = "",
) -> Line:
    """One convergence chart per entry of `charts` (title → series → band), each series' median and quartiles, with
    a dashed line at `references[title]` where given; one legend above the grid when there are several series."""
    columns = min(columns, len(charts))
    rows = math.ceil(len(charts) / columns)
    fig = Line(
        size=(width, row_height),
        rows=rows,
        columns=columns,
        titles=list(charts),
        **REPORT,
        horizontal_spacing=0.25 / columns,
        vertical_spacing=min(0.45 / rows, 0.2),
    )
    for index, (name, bands) in enumerate(charts.items()):
        row, col = divmod(index, columns)
        for series, (label, curve) in enumerate(bands.items()):
            fig.band(
                curve.evaluations, curve.median, curve.low, curve.high, label, series=series, row=row + 1, col=col + 1
            )
            if log:
                fig.log(row=row + 1, col=col + 1)
        if references and name in references:
            fig.reference(references[name], row + 1, col + 1)
    several = len(next(iter(charts.values()))) > 1
    fig.update_layout(
        showlegend=several,
        legend={"orientation": "h", "x": 0, "y": 1, "yref": "container", "yanchor": "top"},
        margin={"t": 70 if title or several else 40, "l": 70, "b": 60},
        title=title or None,
    )
    fig.update_annotations(font={"size": 12})
    fig.add_annotation(text="Evaluations", x=0.5, y=0, yref="paper", yshift=-45, showarrow=False, xref="paper")
    fig.add_annotation(text=ylabel, x=0, y=0.5, xref="paper", yref="paper", xshift=-60, textangle=-90, showarrow=False)
    return fig


def landscape(
    fig: Contour,
    function: Callable[[np.ndarray], float],
    lb,
    ub,
    minimiser: np.ndarray | None = None,
    resolution: int = 120,
    row: int | None = None,
    col: int | None = None,
) -> Contour:
    """Log contours of `function` over [lb, ub]², with its minimiser as a star, without ticks."""
    fig.add(function, (lb, ub), resolution=resolution, row=row, col=col)
    if minimiser is not None:
        fig.star(minimiser, row=row, col=col)
    fig.update_xaxes(row=row, col=col, **BARE)
    return fig.update_yaxes(row=row, col=col, **BARE)


def herd_box(fig: Contour, points, lb, ub, row: int | None = None, col: int | None = None) -> Contour:
    """Positions in a two-dimensional box, as dots inside its outline, without ticks."""
    lb, ub = np.broadcast_to(np.asarray(lb, dtype=float), (2,)), np.broadcast_to(np.asarray(ub, dtype=float), (2,))
    margin = 0.04 * (ub - lb)
    fig.herd(points, size=6, row=row, col=col)
    fig.add_shape(
        type="rect",
        x0=lb[0],
        y0=lb[1],
        x1=ub[0],
        y1=ub[1],
        line={"color": OUTLINE, "width": 1},
        fillcolor="rgba(0,0,0,0)",
        opacity=1,
        layer="below",
        row=row,
        col=col,
    )
    fig.update_xaxes(range=[lb[0] - margin[0], ub[0] + margin[0]], row=row, col=col, **BARE)
    fig.update_yaxes(range=[lb[1] - margin[1], ub[1] + margin[1]], row=row, col=col, **BARE)
    return fig.update_layout(showlegend=False)
