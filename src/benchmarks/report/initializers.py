"""The report of the initializer study, and the gallery of the first herds the initializers scatter.

The pieces are public so a notebook can show them: `results` reads the study, `standing` ranks the initializers of
one suite, `convergence` gives an initializer's error bands from its traces, `ranks_figure` draws the average ranks,
and `herds_figure` the first herd of each initializer.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from benchmarks.curves import Band, band
from benchmarks.report.charts import REPORT, herd_box
from benchmarks.report.tables import ZERO, column, grouped, markdown, ranks, rows, write
from benchmarks.runs import read_traces, trace_key
from nio import SearchSpace
from nio.initializers import INITIALIZERS
from plots import Bars, Contour


def slug(name: str) -> str:
    """The file name of an initializer's results: "Sobol Sequence" is `sobol-sequence`."""
    return name.lower().replace(" ", "-")


SLUGS = {slug(name): name for name in INITIALIZERS}
Rows = list[dict[str, str]]


def results(directory) -> dict[str, dict[str, Rows]]:
    """suite → initializer → rows, from the files `<suite>_<initializer slug>.csv` in `directory`."""
    found: dict[str, dict[str, Rows]] = {}
    for path in sorted(Path(directory).glob("*.csv")):
        suite, _, slug = path.stem.partition("_")
        if slug in SLUGS:
            found.setdefault(suite, {})[SLUGS[slug]] = rows(path)
    return found


@dataclass(frozen=True)
class Standing:
    """How the initializers of one suite compare: the mean error of each on each function (errors below `ZERO`
    count as 0), and their ranks on each function (1 is best, ties share their ranks)."""

    names: list[str]
    functions: list[str]
    errors: np.ndarray  # initializers × functions

    @cached_property
    def ranked(self) -> np.ndarray:
        """The rank of each initializer on each function (functions × initializers)."""
        return ranks(self.errors.T)

    @property
    def mean_ranks(self) -> dict[str, float]:
        """Each initializer's rank averaged over the functions."""
        return dict(zip(self.names, self.ranked.mean(axis=0).tolist(), strict=True))

    @property
    def wins(self) -> np.ndarray:
        """On how many functions each initializer ranks first."""
        return (self.ranked == self.ranked.min(axis=1, keepdims=True)).sum(axis=0)


def standing(by_initializer: Mapping[str, Rows]) -> Standing:
    """The standing of the initializers in `by_initializer` (initializer → rows), in `INITIALIZERS`' order."""
    names = [name for name in INITIALIZERS if name in by_initializer]
    functions = list(grouped(by_initializer[names[0]], "function"))
    errors = np.array(
        [
            [
                np.where(column(r, "error") < ZERO, 0.0, column(r, "error")).mean()
                for r in grouped(found, "function").values()
            ]
            for found in (by_initializer[name] for name in names)
        ]
    )
    return Standing(names, functions, errors)


def convergence(directory, suite: str, name: str, functions) -> dict[str, Band]:
    """The error band of initializer `name` on each of `functions` in `suite`: the median and quartiles of its runs'
    errors at each number of evaluations, from the traces saved beside its results."""
    stem = Path(directory) / f"{suite}_{slug(name)}"
    found = grouped(rows(stem.with_suffix(".csv")), "function")
    traces = read_traces(stem.with_suffix(".npz"))
    bands = {}
    for function in functions:
        runs = found[function]
        optimum = float(runs[0]["best"]) - float(runs[0]["error"])
        budget = int(column(runs, "evaluations").max())
        bands[function] = band([traces[trace_key(function, int(r["seed"]))] for r in runs], optimum, budget)
    return bands


def ranks_figure(mean_ranks: Mapping[str, Mapping[str, float]], size=(900, 320)) -> Bars:
    """The initializers' average ranks, one cell per suite (suite → initializer → rank), best on top."""
    titles = [f"Initializers on {suite}" for suite in mean_ranks]
    fig = Bars(size=size, columns=len(mean_ranks), titles=titles, horizontal_spacing=0.18, legend=False, **REPORT)
    for index, ranked in enumerate(mean_ranks.values()):
        ordered = dict(sorted(ranked.items(), key=lambda item: item[1]))
        fig.add(ordered, fmt="{:.2f}", series=index, row=1, col=index + 1)
        axis = {"title": "Average rank (lower is better)", "range": [0, len(ordered) + 0.8]}
        fig.update_xaxes(row=1, col=index + 1, **axis)
    return fig


def herds_figure(agents: int = 40, columns: int = 5, size=(1000, 220)) -> Contour:
    """The first herd each initializer scatters over [-1, 1]², from the same seed."""
    space = SearchSpace.of(-1, 1, 2)
    rows_needed = -(-len(INITIALIZERS) // columns)
    fig = Contour(size=size, rows=rows_needed, columns=columns, titles=list(INITIALIZERS), **REPORT)
    for index, initializer in enumerate(INITIALIZERS.values()):
        row, col = divmod(index, columns)
        points = initializer(space, agents, np.random.default_rng(1))
        herd_box(fig, points, space.lb, space.ub, row=row + 1, col=col + 1)
    fig.update_layout(title=f"The first herd of {agents} ibex from each initializer", margin={"t": 70})
    return fig.update_annotations(font={"size": 12})


def report(directory, figures) -> list[Path]:
    """`initializers.md` in `directory` with each initializer's average rank per suite, and the ranks as bars."""
    directory, figures = Path(directory), Path(figures)
    by_suite = results(directory)
    if not by_suite:
        return []
    sections, mean_ranks = [], {}
    for suite, by_initializer in by_suite.items():
        table = standing(by_initializer)
        mean_ranks[suite] = table.mean_ranks
        order = np.argsort(table.ranked.mean(axis=0), kind="stable")
        body = [[table.names[k], table.mean_ranks[table.names[k]], int(table.wins[k])] for k in order]
        sections.append(f"## {suite}\n\n" + markdown(["Initializer", "Average rank", "Functions ranked first"], body))
        per_function = [[f, *table.errors[:, i]] for i, f in enumerate(table.functions)]
        sections.append("Mean error per function:\n\n" + markdown(["Function", *table.names], per_function))
    runs = len(grouped(next(iter(next(iter(by_suite.values())).values())), "function")["F1"])
    intro = (
        f"NIO from each of its {len(INITIALIZERS)} initializers, {runs} runs per function (seeds 1000 on), 15 000 "
        "evaluations per run. On each function the initializers are ranked by their mean error (1 is best, ties share "
        "their ranks), and the ranks are averaged over the functions. The `fair23` suite shifts every optimum away "
        "from the origin and the line (1, ..., 1), so an initializer that crowds the herd at the centre of the box "
        "gains nothing from where the optimum sits."
    )
    written = [write(directory / "initializers.md", "NIO from each initializer", intro, *sections)]
    return written + ranks_figure(mean_ranks).save(figures / "initializers_ranks")


def gallery(figures, agents: int = 40) -> list[Path]:
    """The gallery of first herds, saved in `figures`."""
    return herds_figure(agents).save(Path(figures) / "initializers_herds")
