"""The session's look: the `nio` theme (the `white` template, the `soft` palette, the Viridis colorscale) and figure
titles hidden, until `use()` changes them."""

import plotly.io as pio

from plots.palettes import palette as find_palette
from plots.templates import template as find_template
from plots.themes import theme as find_theme

_current = {"theme": "nio", "template": "white", "palette": "soft", "scale": "Viridis", "show_title": False}


def use(
    template: str | None = None,
    palette: str | None = None,
    scale=None,
    show_title: bool = False,
    *,
    theme: str = "nio",
) -> None:
    """Make `theme` the session's look, for the figure classes and for Plotly's own figures, with its template,
    palette and colorscale unless `template`, `palette` or `scale` (a Plotly colorscale or a palette name, for
    contours and surfaces) are given; `show_title` is whether the figure classes draw their titles (hidden by
    default: a paper or a notebook captions its figures)."""
    chosen = find_theme(theme)
    template = chosen.template if template is None else template
    palette = chosen.palette if palette is None else palette
    scale = chosen.scale if scale is None else scale
    find_template(template), find_palette(palette)  # unknown names raise here, naming the known ones
    _check_scale(scale)
    _current.update(theme=chosen.name, template=template, palette=palette, scale=scale, show_title=bool(show_title))
    pio.templates.default = look()


def _check_scale(scale) -> None:
    """Raise ValueError unless `scale` is a palette name or a colorscale Plotly accepts."""
    import plotly.graph_objects as go

    from plots.palettes import PALETTES

    if not (isinstance(scale, str) and scale in PALETTES):
        go.Contour(colorscale=scale)  # Plotly's own check, which names the valid scales


def current_theme() -> str:
    """The session's theme name."""
    return _current["theme"]


def current_show_title() -> bool:
    """Whether the figure classes draw their titles unless told otherwise."""
    return _current["show_title"]


def current_scale():
    """The session's colorscale for contours and surfaces."""
    return _current["scale"]


def current() -> tuple[str, str]:
    """The session's template and palette names."""
    return _current["template"], _current["palette"]


def look(template: str | None = None, palette: str | None = None) -> str:
    """The Plotly template string of a look, `look("white", "soft") == "nio_white+nio_soft"`; `None` takes the
    session's."""
    template = find_template(template or _current["template"]).name
    palette = find_palette(palette or _current["palette"]).name
    return f"nio_{template}+nio_{palette}"
