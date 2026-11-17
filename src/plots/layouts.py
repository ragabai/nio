"""Named figure sizes in pixels; a grid's height is per row. Saved PNGs are `scale` times larger."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Layout:
    """A figure's width and height, in pixels."""

    width: int
    height: int


LAYOUTS: dict[str, Layout] = {
    "paper": Layout(520, 380),  # one journal column
    "wide": Layout(1000, 420),  # both columns
    "notebook": Layout(900, 450),
    "slide": Layout(1280, 720),  # 16:9
    "square": Layout(600, 600),
    "chart": Layout(600, 600),  # the figure classes' charts
    "map": Layout(500, 500),  # contours, surfaces and 3-D paths
}


def layout(spec: str | tuple[int, int] | Layout) -> Layout:
    """The layout named `spec`, or a `(width, height)` in pixels."""
    if isinstance(spec, Layout):
        return spec
    if isinstance(spec, str):
        if spec not in LAYOUTS:
            raise KeyError(f"No layout {spec!r}; known: {', '.join(LAYOUTS)}")
        return LAYOUTS[spec]
    width, height = spec
    return Layout(int(width), int(height))
