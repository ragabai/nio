"""Clean, consistent figures for any plot, on Plotly: one class per kind of figure.

    from plots import Line, Contour, Surface, Animate

    Line(seasons, run.convergence, title="Best value per season", x="Season", y="f(best)", log=True)

    fig = Line(title="The herd's fitness", x="Season", log=True)
    for name, values in summary.items():
        fig.add(seasons, values, name=name)
    fig.save("figures/fitness")                       # fitness.png (3x), fitness.pdf, fitness.svg

    Contour(f, (-5, 5), fill=False).paths(run.positions).herd(run.positions[-1], best=run.position)

Every class is a Plotly figure, so any Plotly method (`update_layout`, `update_traces`, ...) works on it; each takes
`title`, `x`, `y`, `size`, `rows`/`columns`, `log`, `legend`, `template` and `palette`. The classes: `Line`, `Area`,
`Scatter`, `Bars`, `Boxes`, `Histogram`, `Dual`, `Contour`, `Surface`, and `Animate` for a herd moving over a
landscape. Charts are 600 x 600 px and landscapes 500 x 500 unless `size` says otherwise. Text is in Times New
Roman; titles are kept but drawn only with `show_title=True` (or `plots.use(show_title=True)` for the session).

Templates: `white` (default), `simple_white`, `soft_gray`. Palettes: `soft` (default), `pastel`, `muted`, `plotly`.
Contours and surfaces are in Viridis unless `colors=` (any Plotly colorscale, or a palette name) says otherwise.
`plots.use("simple_white", "muted", scale="Cividis")` changes the session's look; `plots.notebook()` makes figures
show on GitHub too. Saving images needs Chrome (`plotly_get_chrome` installs one).
"""

from plots.animation import Animate
from plots.defaults import current, current_scale, current_show_title, current_theme, look, use
from plots.distributions import ECDF, Bars, Boxes, Histogram, Violin
from plots.export import browser, has_chrome, save_many
from plots.landscapes import Contour, Surface
from plots.layouts import LAYOUTS, Layout, layout
from plots.lines import Dual, Line
from plots.maps import Density, Heatmap
from plots.marks import Area, Scatter
from plots.notebook import notebook
from plots.palettes import PALETTES, Palette, palette
from plots.plot import Plot
from plots.spaces import Parallel, Paths3D
from plots.templates import TEMPLATES, Template, register_all, template
from plots.themes import THEMES, Theme, theme

register_all()

__all__ = [
    "ECDF",
    "LAYOUTS",
    "PALETTES",
    "TEMPLATES",
    "THEMES",
    "Animate",
    "Area",
    "Bars",
    "Boxes",
    "Contour",
    "Density",
    "Dual",
    "Heatmap",
    "Histogram",
    "Layout",
    "Line",
    "Palette",
    "Parallel",
    "Paths3D",
    "Plot",
    "Scatter",
    "Surface",
    "Template",
    "Theme",
    "Violin",
    "browser",
    "current",
    "current_scale",
    "current_show_title",
    "current_theme",
    "has_chrome",
    "layout",
    "look",
    "notebook",
    "palette",
    "save_many",
    "template",
    "theme",
    "use",
]
