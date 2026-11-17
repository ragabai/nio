"""How a series is styled: its colour from the figure's palette, and its legend entry.

Series `k` of a figure takes palette colour `k`; a series is counted by its legend group, so a band's line and its
shaded area, or every thin line of `agents`, are one series. A series drawn again under the same name (in another
cell of a grid, say) keeps its colour and shows in the legend once. An unnamed series does not show in the legend,
but still takes its own colour.
"""

from dataclasses import dataclass

import plotly.graph_objects as go

from plots.defaults import current, current_scale, current_theme
from plots.palettes import Palette, palette
from plots.themes import Theme, theme


@dataclass(frozen=True)
class Style:
    """The colour of a series and the legend options of its traces."""

    color: str
    group: str
    legend: bool

    @property
    def trace(self) -> dict:
        """The legend options, as keywords of a Plotly trace."""
        return {"legendgroup": self.group, "showlegend": self.legend}


def _recorded(fig: go.Figure, key: str, otherwise):
    """What `Plot` recorded on `fig` under `key` in `layout.meta`, or `otherwise` (the session's)."""
    meta = fig.layout.meta
    return meta[key] if isinstance(meta, dict) and key in meta else otherwise


def figure_palette(fig: go.Figure) -> Palette:
    """The palette `Plot` recorded on `fig`, or the session's."""
    return palette(_recorded(fig, "palette", current()[1]))


def figure_theme(fig: go.Figure) -> Theme:
    """The theme `Plot` recorded on `fig`, or the session's."""
    return theme(_recorded(fig, "theme", current_theme()))


def figure_scale(fig: go.Figure):
    """The colorscale `Plot` recorded on `fig` for its contours and surfaces, or the session's."""
    return _recorded(fig, "scale", current_scale())


def groups(fig: go.Figure) -> list[str]:
    """The series (legend groups) `fig` holds, in the order they were drawn."""
    return list(dict.fromkeys(trace.legendgroup for trace in fig.data if trace.legendgroup))


def next_series(fig: go.Figure) -> int:
    """The index of the next series: how many series `fig` holds."""
    return len(groups(fig))


def style(fig: go.Figure, name: str | None, series: int | None = None, color: str | None = None) -> Style:
    """The style of the series `name` on `fig`: palette colour `series` unless `color` is given (when left out, the
    colour `name` already has on `fig`, or the next one), in the legend group `name` (or "series k"), shown in the
    legend when named and not shown already."""
    known = groups(fig)
    if series is None:
        series = known.index(name) if name in known else len(known)
    return Style(
        color or figure_palette(fig).color(series), name or f"series {series}", bool(name) and name not in known
    )
