"""Themes: a whole look by name, the template and palette of every figure and how landscapes and the herd are drawn.

* `nio`: the `white` template and `soft` palette; filled contours in Viridis on a log scale; the best as a diamond;
* `classic`: Plotly's white template with dark slate grey text, narrow margins and no legend, Plotly's own colours,
  contour lines in Plasma on the raw values, an opaque surface, and the best as a star-diamond.

`plots.use(theme="classic")` makes a theme the session's; `Contour(..., theme="classic")` takes it for one figure. Any
argument given (`fill=`, `colors=`, `palette=`, ...) wins over the theme.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    """A named look. `fill`, `log` and `crop` are how contours are drawn (filled, on a log scale, and as lines without
    the palest end of the scale); `best` and `best_size` the best's marker symbol and its size against a herd dot's;
    `edge` whether markers have a white edge; `visited` the size and opacity of every visited position; `opacity` a
    surface's."""

    name: str
    template: str
    palette: str
    scale: str
    fill: bool
    log: bool
    crop: bool
    best: str
    best_size: float
    edge: bool
    visited: tuple[float, float]
    opacity: float


THEMES: dict[str, Theme] = {
    t.name: t
    for t in (
        Theme("nio", "white", "soft", "Viridis", True, True, True, "diamond", 1.6, True, (3, 0.4), 0.85),
        Theme("classic", "classic", "plotly", "Plasma", False, False, False, "star-diamond", 1.9, False, (8, 1.0), 1.0),
    )
}


def theme(name: str | Theme) -> Theme:
    """The theme called `name` (a `Theme` is returned as it is)."""
    if isinstance(name, Theme):
        return name
    if name not in THEMES:
        raise KeyError(f"No theme {name!r}; known: {', '.join(sorted(THEMES))}")
    return THEMES[name]
