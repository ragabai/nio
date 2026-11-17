"""The base of every figure class: a Plotly figure that is sized, themed and gridded, with log axes applied when shown.

`Plot` is a `plotly.graph_objects.Figure`, so every Plotly method (`update_layout`, `update_traces`, `add_annotation`,
...) works on it and returns it. The classes built on it (`Line`, `Contour`, ...) add the drawing methods; each method
returns the figure, so calls chain.

Sizes: a chart is 600 × 600 px, a contour or surface 500 × 500, a grid 400 px per cell and at
most 1200 wide. `size=` takes a `LAYOUTS` name or (width, height), the height per row.
"""

from typing import Self

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from plots.axes import log_axis, reference
from plots.defaults import current, current_scale, current_show_title, current_theme, look
from plots.export import FORMATS, save
from plots.layouts import Layout
from plots.layouts import layout as find_layout
from plots.palettes import Palette
from plots.palettes import palette as find_palette
from plots.series import figure_palette, figure_scale, figure_theme
from plots.themes import Theme
from plots.themes import theme as find_theme

GRID_CELL, GRID_WIDTH = 400, 1200
GRID_OPTIONS = {
    "shared_xaxes",
    "shared_yaxes",
    "horizontal_spacing",
    "vertical_spacing",
    "specs",
    "column_widths",
    "row_heights",
}
LEGENDS = {
    "top": {"orientation": "h", "x": 0, "xanchor": "left", "y": 1.02, "yanchor": "bottom"},
    "right": {"orientation": "v", "x": 1.02, "xanchor": "left", "y": 1, "yanchor": "top"},
    "bottom": {"orientation": "h", "x": 0.5, "xanchor": "center", "y": 0, "yanchor": "bottom", "yref": "container"},
    "inside": {"x": 0.98, "xanchor": "right", "y": 0.98, "yanchor": "top", "bgcolor": "rgba(255,255,255,0.7)"},
}
BOTTOM_MARGIN = 110  # room under the x axis title for a legend placed at the bottom
LOG_AXES = {False: "", None: "", True: "y", "y": "y", "x": "x", "xy": "xy", "yx": "xy"}


def _rebuild(cls, data: dict, state: dict):
    """A figure of class `cls` from Plotly's dict of it and its own state; how copies and pickles come back."""
    fig = cls.__new__(cls)
    go.Figure.__init__(fig, data)
    for name, value in state.items():
        setattr(fig, name, value)
    return fig


class Plot(go.Figure):
    """A sized, themed Plotly figure, one panel or a grid, that the figure classes draw on.

    `title`, `x` and `y` are the title and the axis titles (in a grid, `x` goes under the bottom row and `y` beside
    the first column); `titles` names the cells. `log` is `True` (the y axis), `"x"`, `"y"` or `"xy"`; it is applied
    when the figure is shown or saved, over the data it then holds. `legend` is `None` (the template's), `False`
    (hidden), `True`, `"top"`, `"right"`, `"bottom"` (under the x axis, clear of a grid's cell titles) or
    `"inside"`. Grid options of `make_subplots` (`shared_yaxes`, `horizontal_spacing`, ...) and any layout option
    (`hovermode`, `margin`, ...) are taken as keywords. The title is kept on the figure but drawn only when
    `show_title` is True (the session's choice, hidden unless `plots.use(show_title=True)`, when left out).
    `theme` (`plots.THEMES`) is the whole look; `template` and `palette` change part of it. Left out, the figure takes
    the session's (`plots.use(theme=...)`).
    """

    DEFAULT_SIZE: str | tuple[int, int] = "chart"
    STATE: tuple[str, ...] = (
        "_log",
        "_show_title",
        "_grid_ref",
        "_grid_str",
    )  # what a copy keeps besides Plotly's dict

    def __init__(
        self,
        *,
        title: str | None = None,
        x: str | None = None,
        y: str | None = None,
        size: str | tuple[int, int] | Layout | None = None,
        rows: int = 1,
        columns: int = 1,
        titles=None,
        log: bool | str = False,
        legend: bool | str | None = None,
        template: str | None = None,
        palette: str | None = None,
        secondary_y: bool = False,
        show_title: bool | None = None,
        theme: str | None = None,
        **layout,
    ):
        if log not in LOG_AXES:
            raise ValueError(f"log is True, False, 'x', 'y' or 'xy'; got {log!r}.")
        if isinstance(legend, str) and legend not in LEGENDS:
            raise ValueError(f"No legend place {legend!r}; the places are {', '.join(LEGENDS)}.")
        grid = {key: layout.pop(key) for key in GRID_OPTIONS & set(layout)}
        gridded = rows * columns > 1
        if gridded or secondary_y or titles or grid:
            specs = [[{"secondary_y": True}] * columns for _ in range(rows)] if secondary_y else None
            grid.setdefault("specs", specs)
            super().__init__(make_subplots(rows=rows, cols=columns, subplot_titles=titles, **grid))
        else:
            super().__init__()
        self._log = LOG_AXES[log]
        self._show_title = current_show_title() if show_title is None else bool(show_title)
        if size is None:
            size = (min(GRID_WIDTH, GRID_CELL * columns), GRID_CELL) if gridded else self.DEFAULT_SIZE
        dimensions = find_layout(size)
        if theme is None:  # the session's look, as plots.use left it
            look_of = find_theme(current_theme())
            template, palette, scale = template or current()[0], palette or current()[1], current_scale()
        else:
            look_of = find_theme(theme)
            template, palette, scale = template or look_of.template, palette or look_of.palette, look_of.scale
        chosen = find_palette(palette).name
        self.update_layout(
            width=dimensions.width,
            height=dimensions.height * rows,
            template=look(template, chosen),
            meta={"palette": chosen, "theme": look_of.name, "scale": scale},
            title=title,
        )
        if gridded:
            if x:
                self.update_xaxes(title=x, row=rows)
            if y:
                self.update_yaxes(title=y, col=1)
        else:
            self.update_layout(xaxis_title=x, yaxis_title=y)
        if legend is False or legend is True:
            self.update_layout(showlegend=legend)
        elif legend:
            self.update_layout(legend=LEGENDS[legend])
            if legend == "bottom":
                self.update_layout(margin={"b": BOTTOM_MARGIN})
        if layout:
            self.update_layout(**layout)

    @property
    def palette(self) -> Palette:
        """The figure's palette, whose colours its series take in order."""
        return figure_palette(self)

    @property
    def theme(self) -> Theme:
        """The figure's theme: how its contours, surfaces and herd are drawn."""
        return figure_theme(self)

    @property
    def colorscale(self):
        """The figure's colorscale for contours and surfaces."""
        return figure_scale(self)

    def log(self, axis: str = "y", row: int | None = None, col: int | None = None) -> Self:
        """Make `axis` ("x", "y" or "xy") logarithmic now, labelled at powers of ten, over the data added so far."""
        for name in LOG_AXES[axis]:
            log_axis(self, name, row, col)
        return self

    def reference(self, y: float, row: int | None = None, col: int | None = None) -> Self:
        """A dashed horizontal line at `y`, such as a best known value."""
        reference(self, y, row, col)
        return self

    def save(self, path, formats=FORMATS, scale: float = 3):
        """Write the figure as `path` in each format (PNG at `scale` times its size, PDF, SVG); return the paths."""
        return save(self, path, formats, scale)

    def show_title(self, show: bool = True) -> Self:
        """Draw the figure's title (`show=False` hides it again)."""
        self._show_title = bool(show)
        return self

    def to_dict(self) -> dict:
        """Plotly's dict of the figure as it is drawn: the log axes applied, and the title left out unless shown.
        Every way a figure leaves (shown in a notebook, saved, written as JSON) goes through here."""
        drawn = self._whole()
        if not getattr(self, "_show_title", current_show_title()):
            drawn["layout"].get("title", {}).pop("text", None)
        return drawn

    def _whole(self) -> dict:
        """Plotly's dict of the figure with the log axes applied and everything kept, title included."""
        return super(Plot, self._finish()).to_dict()

    def _finish(self) -> Self:
        """Apply the log axes asked for at construction, over the data the figure holds now."""
        deferred = getattr(self, "_log", "")
        return self.log(deferred) if deferred else self

    def _state(self) -> dict:
        """What a copy keeps besides Plotly's dict: the deferred log axes and the grid."""
        return {name: getattr(self, name, None) for name in self.STATE}

    def __reduce__(self):
        return _rebuild, (type(self), self._whole(), self._state())
