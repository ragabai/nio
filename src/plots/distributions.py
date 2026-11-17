"""Bars, boxes, histograms, violins and ECDFs: how values are spread, per group or per series.

Bars({"NIO": 1.2, "PSO": 2.5}, fmt="{:.2f}", title="Average rank")
Boxes({"F1": errors_1, "F5": errors_5}, name="NIO", log=True).add({"F1": ..., "F5": ...}, name="PSO")
Histogram(final_values, bins=20, x="Fitness", y="Ibex")
Violin({"25": finals_25, "50": finals_50}, log=True, x="Herd size")
ECDF(errors, name="NIO", log="x", x="Error").add_values(other_errors, name="PSO")
"""

from collections.abc import Mapping, Sequence
from typing import Self

import numpy as np

from plots.charts import bars, boxes, hist, violins
from plots.lines import Line
from plots.plot import Plot


class Bars(Plot):
    """Horizontal bars, the first label on top, each labelled with its value (`fmt`); `spread` maps each label to a
    (low, high) range drawn as a line through its bar."""

    def __init__(
        self,
        values: Mapping[str, float] | None = None,
        /,
        spread: Mapping[str, tuple[float, float]] | None = None,
        fmt: str = "{:.3f}",
        name: str | None = None,
        **options,
    ):
        super().__init__(**options)
        if values is not None:
            self.add(values, spread, fmt, name)

    def add(
        self,
        values: Mapping[str, float],
        /,
        spread: Mapping[str, tuple[float, float]] | None = None,
        fmt: str = "{:.3f}",
        name: str | None = None,
        *,
        series: int | None = None,
        row: int | None = None,
        col: int | None = None,
    ) -> Self:
        """One set of bars, in palette colour `series` (the next one when left out)."""
        bars(self, values, spread, fmt, name, series, row, col)
        return self


class Boxes(Plot):
    """A box per group with every value as a dot over it; sets added under different names stand side by side."""

    def __init__(self, groups: Mapping[str, Sequence[float]] | None = None, /, name: str | None = None, **options):
        super().__init__(**options)
        if groups is not None:
            self.add(groups, name)

    def add(
        self,
        groups: Mapping[str, Sequence[float]],
        /,
        name: str | None = None,
        *,
        series: int | None = None,
        row: int | None = None,
        col: int | None = None,
    ) -> Self:
        """A box for each group, in the order given, in palette colour `series` (the next one when left out)."""
        boxes(self, groups, name, series, row=row, col=col)
        return self


class Histogram(Plot):
    """How values fall into bins: `bins` of them, or Plotly's choice when left out."""

    def __init__(self, values=None, /, name: str | None = None, bins: int | None = None, **options):
        super().__init__(**options)
        if values is not None:
            self.add(values, name, bins)

    def add(
        self,
        values,
        /,
        name: str | None = None,
        bins: int | None = None,
        *,
        series: int | None = None,
        row: int | None = None,
        col: int | None = None,
    ) -> Self:
        """One histogram, in palette colour `series` (the next one when left out)."""
        hist(self, values, name, series, bins, row, col)
        return self


class Violin(Plot):
    """A violin per group, the shape of its values with a box inside; sets added under different names stand side by
    side, each in its palette colour. With `log` the violins are shaped on the base-10 logarithm of the values (a
    violin's shape on a log axis would be drawn from the plain values), labelled 10ⁿ; values that are not positive
    are left out."""

    STATE = (*Plot.STATE, "_logged")

    def __init__(
        self,
        groups: Mapping[str, Sequence[float]] | None = None,
        /,
        name: str | None = None,
        log: bool = False,
        **options,
    ):
        super().__init__(**options)
        self._logged = bool(log)
        if groups is not None:
            self.add(groups, name)

    def add(
        self,
        groups: Mapping[str, Sequence[float]],
        /,
        name: str | None = None,
        *,
        series: int | None = None,
        row: int | None = None,
        col: int | None = None,
    ) -> Self:
        """One set of violins, in palette colour `series` (the next one, or the one `name` already has, when left
        out)."""
        if self._logged:
            groups = {label: _logarithms(values) for label, values in groups.items()}
        violins(self, groups, name, series, row, col)
        if self._logged:
            self._label_decades(row, col)
        return self

    def _label_decades(self, row: int | None, col: int | None) -> None:
        """Ticks at whole powers of ten over the violins drawn, labelled 10ⁿ."""
        found = np.concatenate([np.asarray(t.y, dtype=float) for t in self.data if t.type == "violin"] or [[]])
        if not found.size:
            return
        low, high = int(np.floor(np.nanmin(found))), int(np.ceil(np.nanmax(found)))
        step = max(1, -(-(high - low) // 8))
        ticks = list(range(low, high + 1, step))
        self.update_yaxes(tickvals=ticks, ticktext=[f"10<sup>{k}</sup>" for k in ticks], row=row, col=col)


def _logarithms(values) -> np.ndarray:
    """The base-10 logarithms of the positive `values`."""
    values = np.asarray(values, dtype=float).ravel()
    return np.log10(values[np.isfinite(values) & (values > 0)])


class ECDF(Line):
    """The empirical distribution of each set of values: the share of them at or below every value, as a step line
    climbing to 1, such as the share of runs that reached each error."""

    def __init__(self, values=None, /, name: str | None = None, **options):
        options.setdefault("y", "Share")
        super().__init__(**options)
        if values is not None:
            self.add_values(values, name)

    def add_values(
        self, values, /, name: str | None = None, *, row: int | None = None, col: int | None = None, **trace
    ) -> Self:
        """One more set of values, in the next palette colour; values that are not finite are left out."""
        values = np.asarray(values, dtype=float).ravel()
        values = np.sort(values[np.isfinite(values)])
        share = np.arange(1, len(values) + 1) / max(len(values), 1)
        return self.add(values, share, name=name, row=row, col=col, line_shape="hv", **trace)
