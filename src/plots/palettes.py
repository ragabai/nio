"""The palettes: eight colors each, assigned in order and never cycled, with a red and a green kept by role.

Series `k` always takes color `k`, so a figure with fewer series does not repaint the ones it keeps. `alert` (the
palette's red) and `good` (its green) mark the herd and its best position whatever the palette's order. `scale` is a
sequential colorscale from near white to the first color, for contours and surfaces.
"""

from dataclasses import dataclass

MAX_SERIES = 8


@dataclass(frozen=True)
class Palette:
    """A named set of eight series colors, with its red (`alert`) and green (`good`)."""

    name: str
    colors: tuple[str, ...]
    alert: str
    good: str

    def color(self, k: int) -> str:
        """The color of series `k` (0-based); a ninth series is refused rather than given a made-up hue."""
        if not 0 <= k < min(MAX_SERIES, len(self.colors)):
            raise ValueError(f"A figure takes at most {MAX_SERIES} series; fold the rest into small multiples.")
        return self.colors[k]

    @property
    def scale(self) -> list[list]:
        """A sequential colorscale, from the first color mixed mostly with white to the first color itself."""
        stops = [(0.0, 0.92), (0.25, 0.7), (0.5, 0.45), (0.75, 0.2)]
        return [[at, _toward_white(self.colors[0], share)] for at, share in stops] + [[1, self.colors[0]]]


def _toward_white(color: str, share: float) -> str:
    """`color` mixed with white, `share` of it white."""
    rgb = [int(color[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(c + (255 - c) * share):02x}" for c in rgb)


PALETTES: dict[str, Palette] = {
    p.name: p
    for p in (
        Palette(
            "soft",
            ("#5B8FF9", "#F6903D", "#5AD8A6", "#E86452", "#9270CA", "#F6BD16", "#6DC8EC", "#FF99C3"),
            alert="#E86452",
            good="#5AD8A6",
        ),
        Palette(
            "pastel",
            ("#8CB4F0", "#F4B183", "#93D7B8", "#F2A0A0", "#B9A6E0", "#F7D488", "#9FD8EA", "#F5B8D4"),
            alert="#F2A0A0",
            good="#93D7B8",
        ),
        Palette(
            "muted",
            ("#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"),
            alert="#C44E52",
            good="#55A868",
        ),
        Palette(
            "plotly",
            ("#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880"),
            alert="#EF553B",
            good="#00CC96",
        ),
    )
}


def palette(name: str | Palette) -> Palette:
    """The palette called `name` (a `Palette` is returned as it is)."""
    if isinstance(name, Palette):
        return name
    if name not in PALETTES:
        raise KeyError(f"No palette {name!r}; known: {', '.join(sorted(PALETTES))}")
    return PALETTES[name]


def register(chosen: Palette) -> None:
    """Add a palette of your own, by its name, and register it as the Plotly template `nio_<name>` too."""
    PALETTES[chosen.name] = chosen
    from plots.templates import register_palette  # here: templates imports this module

    register_palette(chosen)
