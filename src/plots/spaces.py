"""Positions in many dimensions, or over time: parallel coordinates, and the agents' paths in space and time.

Parallel({f"x{d + 1}": last[:, d] for d in range(10)}, color=fitness, label="fitness")
Paths3D(run.positions[:, :10])                      # x₁ and x₂ across, the season upward
"""

from collections.abc import Mapping, Sequence
from typing import Self

import numpy as np
import plotly.graph_objects as go

from plots.plot import Plot
from plots.scales import resolve


class Parallel(Plot):
    """Parallel coordinates: one vertical axis per entry of `columns` (name → a value per item), each item a line
    across them. `color` (a value per item, such as its fitness, named `label`) colours the lines in the figure's
    colorscale; `ranges` fixes an axis's (low, high), such as a variable's bounds."""

    def __init__(
        self,
        columns: Mapping[str, Sequence[float]],
        /,
        *,
        color=None,
        label: str = "",
        ranges: Mapping[str, tuple[float, float]] | None = None,
        colors=None,
        **options,
    ):
        super().__init__(**options)
        ranges = ranges or {}
        dimensions = [
            {
                "label": name,
                "values": np.asarray(values, dtype=float),
                **({"range": ranges[name]} if name in ranges else {}),
            }
            for name, values in columns.items()
        ]
        if color is None:
            line = {"color": self.palette.color(0)}
        else:
            scale, reverse = resolve(self.colorscale if colors is None else colors)
            bar = {"title": {"text": label}, "thickness": 12}
            line = {"color": np.asarray(color, dtype=float), "colorscale": scale, "reversescale": reverse}
            line |= {"showscale": True, "colorbar": bar}
        self.add_trace(go.Parcoords(dimensions=dimensions, line=line))


class Paths3D(Plot):
    """Each agent's path in three dimensions, with a dot at every season, agent `k` in palette colour `k`, round again
    after the last. `positions` is seasons × agents × 2, drawn with the season upward (or `z`, one value per season),
    or seasons × agents × 3, drawn through the three coordinates; `labels` names the axes (by default x₁, x₂ and the
    season or x₃)."""

    DEFAULT_SIZE = "map"

    def __init__(self, positions=None, /, *, z=None, labels=None, **options):
        super().__init__(**options)
        through = positions is not None and np.shape(positions)[-1] == 3
        x_title, y_title, z_title = labels or ("x₁", "x₂", "x₃" if through else "Season")
        self.update_scenes(xaxis_title=x_title, yaxis_title=y_title, zaxis_title=z_title)
        self.update_layout(margin={"l": 10, "r": 10, "b": 10})
        if positions is not None:
            self.add(positions, z=z)

    def add(self, positions, /, *, z=None) -> Self:
        """The paths of more agents."""
        positions = np.asarray(positions, dtype=float)
        if z is None and positions.shape[-1] < 3:
            z = np.arange(1, positions.shape[0] + 1)
        colors = self.palette.colors
        for k in range(positions.shape[1]):
            color = colors[k % len(colors)]
            path = go.Scatter3d(
                x=positions[:, k, 0],
                y=positions[:, k, 1],
                z=positions[:, k, 2] if z is None else z,
                mode="lines+markers",
                line={"color": color, "width": 3},
                marker={"color": color, "size": 2},
                name=f"agent {k + 1}",
                legendgroup="paths",
                showlegend=False,
                hoverinfo="skip",
            )
            self.add_trace(path)
        return self
