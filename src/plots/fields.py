"""A function over a box: sampled on a grid, and drawn as contours or as a surface.

A function of more than two variables that cannot be called on two (Kowalik, Hartmann, Shekel, ...) is sampled with
its other coordinates fixed (`lift`): at the values given, or at the middle of the box.
"""

from collections.abc import Callable, Sequence

import numpy as np
import plotly.graph_objects as go

from plots.scales import for_lines, resolve
from plots.series import figure_palette

Bounds = tuple[float | Sequence[float], float | Sequence[float]]
Function = Callable[[np.ndarray], float]


def corners(bounds: Bounds) -> tuple[np.ndarray, np.ndarray]:
    """The box's lower and upper corners in its first two coordinates."""
    lb, ub = (np.asarray(b, dtype=float).ravel() for b in bounds)
    return tuple(np.broadcast_to(b, (2,)) if b.size == 1 else b[:2] for b in (lb, ub))


def lift(f: Function, bounds: Bounds, fixed=None) -> np.ndarray:
    """The coordinates after the first two at which `f` is drawn: `fixed`, or none when `f` takes two variables, or
    else the middle of the box for each of its other `f.dim - 2` variables."""
    if fixed is not None:
        return np.atleast_1d(np.asarray(fixed, dtype=float))
    dim = int(getattr(f, "dim", 2) or 2)
    if dim <= 2:
        return np.array([])
    lb, ub = (np.broadcast_to(np.asarray(b, dtype=float).ravel(), (dim,)) for b in bounds)
    try:  # a scalable function (Sphere, Rastrigin, ...) is drawn as itself, in two dimensions
        f((lb[:2] + ub[:2]) / 2)
        return np.array([])
    except (ValueError, IndexError, TypeError):
        return (lb[2:] + ub[2:]) / 2


def sample(f: Function, bounds: Bounds, resolution: int, rest=()) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The axes and the values of `f` over the box `bounds`, on a `resolution` × `resolution` grid, the other
    coordinates at `rest`; values that are not finite come back as NaN."""
    lb, ub = corners(bounds)
    xs, ys = np.linspace(lb[0], ub[0], resolution), np.linspace(lb[1], ub[1], resolution)
    rest = np.asarray(rest, dtype=float)
    z = np.array([[f(np.concatenate([[x, y], rest])) for x in xs] for y in ys], dtype=float)
    z[~np.isfinite(z)] = np.nan
    return xs, ys, z


def heights(f: Function, points, rest=()) -> np.ndarray:
    """The value of `f` at each two-dimensional point, the other coordinates at `rest`."""
    rest = np.asarray(rest, dtype=float)
    return np.array([f(np.concatenate([p, rest])) for p in np.asarray(points, dtype=float)], dtype=float)


def _logged(z: np.ndarray) -> np.ndarray:
    """`z` on a log scale above its lowest value, which spreads the contours near the optimum."""
    if not np.isfinite(z).any():
        return z
    low, high = np.nanmin(z), np.nanmax(z)
    return np.log10(z - low + 1e-3 * ((high - low) or 1.0))


def _faded(color: str, alpha: float) -> str:
    """`color` (#rrggbb) at opacity `alpha`, as rgba."""
    red, green, blue = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({red},{green},{blue},{alpha})"


def contour(
    fig: go.Figure,
    f: Function,
    bounds: Bounds,
    resolution: int = 100,
    log: bool = True,
    showscale: bool = False,
    row: int | None = None,
    col: int | None = None,
    colors=None,
    fill: bool = True,
    levels: int = 18,
    rest=(),
    crop: bool = True,
) -> go.Figure:
    """Contours of `f` over the box `bounds` in the colorscale `colors` (the session's when left out): filled with
    faint lines, or (`fill=False`) as coloured lines only, without the palest end of the scale unless `crop` is
    False. `log` draws them on a log scale above the lowest value."""
    xs, ys, z = sample(f, bounds, resolution, rest)
    scale, reverse = resolve(colors)
    if fill:
        edge = _faded(figure_palette(fig).colors[0], 0.45) if reverse else "rgba(255,255,255,0.35)"
        lines = {"width": 0.6, "color": edge}
    else:
        lines = {"width": 1.4}
        scale = for_lines(scale) if crop and not reverse else scale
    trace = go.Contour(
        x=xs,
        y=ys,
        z=_logged(z) if log else z,
        colorscale=scale,
        reversescale=reverse,
        showscale=showscale,
        ncontours=levels,
        contours={"coloring": "fill" if fill else "lines"},
        line=lines,
        hoverinfo="skip",
    )
    fig.add_trace(trace, row=row, col=col)
    return fig


def surface(
    fig: go.Figure,
    f: Function,
    bounds: Bounds,
    resolution: int = 60,
    opacity: float = 0.8,
    contours: bool = True,
    colors=None,
    rest=(),
) -> go.Figure:
    """The surface of `f` over the box `bounds` in three dimensions, in the colorscale `colors`, with its contours
    projected on the floor."""
    xs, ys, z = sample(f, bounds, resolution, rest)
    scale, _ = resolve(colors)
    projected = {"z": {"show": True, "project": {"z": True}, "usecolormap": True}} if contours else None
    trace = go.Surface(
        x=xs, y=ys, z=z, colorscale=scale, opacity=opacity, showscale=False, contours=projected, hoverinfo="skip"
    )
    fig.add_trace(trace)
    return fig
