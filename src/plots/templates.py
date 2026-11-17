"""The templates: clean looks built on Plotly's own, registered in `plotly.io.templates` as `nio_<name>`.

* `white`: like `plotly_white`, a white page with a soft blue-gray grid and no axis lines (the default);
* `simple_white`: like Plotly's `simple_white`, no grid, dark axis lines and ticks outside (the report's look);
* `soft_gray`: like `ggplot2`, a pale gray panel with a white grid;
* `classic`: Plotly's `plotly_white` with dark slate grey text at 14 px, narrow margins and no legend.

Each palette is registered too, as `nio_<palette>`, setting only the colors, so a look is a template and a palette
joined with Plotly's `+`: `look("white", "soft") == "nio_white+nio_soft"`. Plotly's own templates are never changed,
and registering does not change Plotly's default template; `plots.use()` does.
"""

from dataclasses import dataclass, field

import plotly.graph_objects as go
import plotly.io as pio

from plots.palettes import PALETTES, Palette

FONT = "Times New Roman, Times, serif"

COMMON = {
    "font": {"family": FONT, "size": 13},
    "title": {"x": 0, "xref": "paper", "xanchor": "left", "font": {"size": 16, "weight": 600}},
    "legend": {"borderwidth": 0, "bgcolor": "rgba(0,0,0,0)"},
    "margin": {"l": 60, "r": 20, "t": 50, "b": 50},
}


@dataclass(frozen=True)
class Template:
    """A look: Plotly's template `base` with our `layout` on top."""

    name: str
    base: str
    layout: dict = field(default_factory=dict)

    def build(self) -> go.layout.Template:
        """The Plotly template: a copy of `base` updated with the common settings and `layout`."""
        built = go.layout.Template(pio.templates[self.base])
        built.layout.update(COMMON)
        built.layout.update(self.layout)
        return built


def _axes(**settings) -> dict:
    return {"xaxis": settings, "yaxis": settings}


TEMPLATES: dict[str, Template] = {
    t.name: t
    for t in (
        Template(
            "white",
            "plotly_white",
            {"font": {"color": "#2a3f5f"}, **_axes(gridcolor="#ebf0f8", zerolinecolor="#ebf0f8")},
        ),
        Template(
            "simple_white",
            "simple_white",
            {"font": {"color": "#222222"}, **_axes(linecolor="#444444", tickcolor="#444444", ticks="outside")},
        ),
        Template(
            "soft_gray",
            "ggplot2",
            {
                "font": {"color": "#333333"},
                "paper_bgcolor": "#ffffff",
                "plot_bgcolor": "#eef0f4",
                **_axes(gridcolor="#ffffff", linecolor="#eef0f4", ticks="", zeroline=False),
            },
        ),
        Template(
            "classic",
            "plotly_white",
            {
                "font": {"color": "darkslategrey", "size": 14},
                "margin": {"l": 40, "r": 40, "b": 40, "t": 40},
                "showlegend": False,
            },
        ),
    )
}


def template(name: str | Template) -> Template:
    """The template called `name` (a `Template` is returned as it is)."""
    if isinstance(name, Template):
        return name
    if name not in TEMPLATES:
        raise KeyError(f"No template {name!r}; known: {', '.join(sorted(TEMPLATES))}")
    return TEMPLATES[name]


def register_palette(chosen: Palette) -> None:
    """Register `chosen` as the Plotly template `nio_<name>`, holding only its colors."""
    colors = go.layout.Template()
    colors.layout.colorway = list(chosen.colors)
    colors.layout.colorscale.sequential = chosen.scale
    pio.templates[f"nio_{chosen.name}"] = colors


def register_all() -> None:
    """Register every template and palette in `plotly.io.templates`; `import plots` does this."""
    for name, chosen in TEMPLATES.items():
        pio.templates[f"nio_{name}"] = chosen.build()
    for chosen in PALETTES.values():
        register_palette(chosen)
