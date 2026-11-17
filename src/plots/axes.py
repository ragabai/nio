"""Axis helpers: logarithmic axes labelled at powers of ten, and dashed reference lines."""

import numpy as np
import plotly.graph_objects as go

LOG_LABELS = 8
REFERENCE = "#8a8984"


def decade_step(arrays) -> int:
    """How many decades between labels, so that the positive values of `arrays` span at most `LOG_LABELS` labels."""
    positive = []
    for values in arrays:
        try:
            values = np.asarray(values, dtype=float).ravel()
        except (TypeError, ValueError):
            continue
        positive.append(values[np.isfinite(values) & (values > 0)])
    found = np.concatenate(positive) if positive else np.array([])
    if found.size == 0:
        return 1
    decades = np.ceil(np.log10(found.max())) - np.floor(np.log10(found.min()))
    return max(1, int(np.ceil(decades / LOG_LABELS)))


def log_axis(fig: go.Figure, axis: str = "y", row: int | None = None, col: int | None = None) -> go.Figure:
    """Make an axis logarithmic, labelled at powers of ten only: every power over a few decades, every second, fifth,
    ... over many, so that at most about `LOG_LABELS` labels show. Call it after adding the data."""
    traces = fig.data if row is None and col is None else list(fig.select_traces(row=row, col=col))
    update = fig.update_yaxes if axis == "y" else fig.update_xaxes
    step = decade_step(getattr(trace, axis, None) for trace in traces)
    return update(type="log", dtick=step, exponentformat="power", row=row, col=col)


def reference(fig: go.Figure, y: float, row: int | None = None, col: int | None = None) -> go.Figure:
    """A dashed horizontal line at `y`, such as a best known value."""
    return fig.add_hline(y=y, line={"color": REFERENCE, "width": 1, "dash": "dash"}, opacity=1, row=row, col=col)
