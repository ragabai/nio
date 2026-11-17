"""The herd animated over a landscape, season by season.

`Animate` takes a function of two variables, its box, and the herd's positions per season (seasons × agents × 2,
such as `nio.Result.positions` of a run in two dimensions). It shows `seasons` of them, spread evenly from the first
to the last, each eased into the next over `smooth` frames, over the contours (`view="contour"`, filled or with
`fill=False` as lines) or the surface (`view="surface"`) of the function, in the colorscale `colors`:

    animation = Animate(f, (lb, ub), result.positions, best=result.position, smooth=4)
    animation.interactive()               # a player with play, pause and a slider, in Jupyter
    animation.gif("herd.gif")             # the frames as a GIF, which also plays on GitHub
"""

import tempfile
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from PIL import Image

from plots.export import save_many
from plots.fields import Bounds, heights, lift
from plots.landscapes import Contour, Surface
from plots.markers import herd

VIEWS = ("contour", "surface")


def frames(seasons: int, count: int) -> list[int]:
    """`count` season indices spread evenly over `seasons`, the first and the last included."""
    return sorted({min(round(i * (seasons - 1) / max(count - 1, 1)), seasons - 1) for i in range(count)})


def tween(positions: np.ndarray, steps: int = 1) -> np.ndarray:
    """`positions` with `steps - 1` evenly eased positions inserted between each one and the next."""
    if steps <= 1:
        return positions
    weights = np.linspace(0, 1, steps, endpoint=False)[:, None, None]
    pairs = [(1 - weights) * a + weights * b for a, b in pairwise(positions)]
    return np.concatenate([*pairs, positions[-1:]])


class Animate:
    """The herd over the landscape of `f` in the box `bounds`, as frames, an interactive player or a GIF."""

    def __init__(
        self,
        f: Callable[[np.ndarray], float],
        bounds: Bounds,
        positions,
        /,
        best=None,
        *,
        view: str = "contour",
        fill: bool = True,
        colors=None,
        seasons: int = 50,
        smooth: int = 1,
        size="map",
        template: str | None = None,
        palette: str | None = None,
        resolution: int | None = None,
    ):
        if view not in VIEWS:
            raise ValueError(f"No view {view!r}; the views are {', '.join(VIEWS)}.")
        positions = np.asarray(positions, dtype=float)
        shown = frames(len(positions), seasons)
        self.f, self.bounds, self.view = f, bounds, view
        self.best = None if best is None else np.asarray(best, dtype=float)
        self.positions = tween(positions[shown], smooth)
        self.labels = np.repeat(np.array(shown) + 1, smooth)[: len(self.positions)]
        self.look = {"size": size, "template": template, "palette": palette, "colors": colors}
        self.fill = fill
        self.resolution = resolution or (100 if view == "contour" else 60)
        self._rest = lift(f, bounds)
        self._backdrop = self._landscape()

    def _landscape(self) -> go.Figure:
        if self.view == "contour":
            fig = Contour(self.f, self.bounds, fill=self.fill, resolution=self.resolution, **self.look)
        else:
            fig = Surface(self.f, self.bounds, opacity=0.6, resolution=self.resolution, **self.look)
        return go.Figure(fig.update_layout(showlegend=False, margin={"t": 60}))

    def _herd(self, fig: go.Figure, index: int) -> go.Figure:
        positions = self.positions[index]
        if self.view == "contour":
            return herd(fig, positions, best=self.best)
        best_z = None if self.best is None else float(heights(self.f, [self.best], self._rest)[0])
        return herd(fig, positions, best=self.best, z=heights(self.f, positions, self._rest), best_z=best_z)

    def frame(self, index: int) -> go.Figure:
        """Frame `index` as a figure of its own, titled with its season."""
        fig = self._herd(go.Figure(self._backdrop), index)
        return fig.update_layout(title={"text": f"Season {self.labels[index]}", "x": 0.5, "xanchor": "center"})

    def figures(self) -> list[go.Figure]:
        """Every frame, as figures."""
        return [self.frame(i) for i in range(len(self.positions))]

    def interactive(self, ms: int = 120) -> go.Figure:
        """One figure that plays the frames, with play and pause buttons and a slider over the seasons."""
        fig = self.frame(0)
        moving = list(range(len(self._backdrop.data), len(fig.data)))
        fig.frames = [
            go.Frame(
                data=list(self._herd(go.Figure(self._backdrop), i).data[len(self._backdrop.data) :]),
                traces=moving,
                name=str(i),
                layout={"title": {"text": f"Season {self.labels[i]}"}},
            )
            for i in range(len(self.positions))
        ]
        play = {"frame": {"duration": ms, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}
        still = {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}
        buttons = [
            {"label": "▶", "method": "animate", "args": [None, play]},
            {"label": "❚❚", "method": "animate", "args": [[None], still]},
        ]
        steps = [
            {"label": str(self.labels[int(f.name)]), "method": "animate", "args": [[f.name], still]} for f in fig.frames
        ]
        return fig.update_layout(
            updatemenus=[
                {"type": "buttons", "direction": "left", "x": 0, "y": 1.12, "xanchor": "left", "buttons": buttons}
            ],
            sliders=[{"currentvalue": {"prefix": "Season "}, "steps": steps, "pad": {"t": 30}}],
        )

    def snapshots(self, directory, prefix: str = "season") -> list[Path]:
        """Every frame as a PNG in `directory`, numbered in order."""
        directory = Path(directory)
        paths = [directory / f"{prefix}-{i + 1:03d}" for i in range(len(self.positions))]
        return save_many(self.figures(), paths, formats=("png",), scale=1)

    def gif(self, path, ms: int = 80, loop: int = 0) -> Path:
        """The frames as a GIF at `path`, `ms` milliseconds each, looping forever by default."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as scratch:
            pictures = [Image.open(p).convert("RGB") for p in self.snapshots(scratch)]
        pictures[0].save(out, save_all=True, append_images=pictures[1:], duration=ms, loop=loop, optimize=True)
        return out
