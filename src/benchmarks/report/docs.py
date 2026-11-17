"""Pages in `docs/`: one per benchmark suite and one for the engineering designs, each with NIO's results.

    python -m benchmarks.report.docs

A suite page has one row per function: its group, dimension and optimum f*, the best value any run found, and the
errors f(best) − f* over the runs (best, median, mean ± standard deviation, worst), with errors below 1e-8 counted as
0 as the CEC reports do; **Solved** counts the runs that reached that tolerance. A page whose results are missing is
skipped.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from benchmarks.paths import RESULTS, ROOT
from benchmarks.report.tables import ZERO, column, grouped, markdown, number, rows, summary, write
from problems.cec.digits import group as digits_group
from problems.engineering import DESIGNS
from problems.functions import Suite, cec, classic, digits

DOCS = ROOT / "docs"
HEADER = ["F", "Function", "Group", "D", "f*", "Best f(x)", "Best error", "Median", "Mean ± std", "Worst", "Solved"]


def _classic_group(number: int) -> str:
    return "Unimodal" if number <= 7 else "Multimodal" if number <= 13 else "Fixed-dimension multimodal"


@dataclass(frozen=True)
class Page:
    """One suite page: its file name, title, suite, results CSV (relative to `results/`), group of a function number
    and a sentence on how the runs were set up."""

    name: str
    title: str
    suite: Callable[[], Suite]
    results: str
    group: Callable[[int], str]
    setup: str


RUNS = 10  # each page reports the first ten seeds of its results file
DEFAULTS = "the runner's defaults (a herd of 30)"
CEC_DEFAULTS = "the runner's defaults (a herd of 18 · D) at the competition's budget, 10 000 · D"
PAGES = [
    Page("classic", "Classic suite, F1–F23", lambda: classic, "classic_default_1000.csv", _classic_group, DEFAULTS),
    Page(
        "cec2017",
        "CEC 2017, D = 10",
        lambda: cec(2017, dim=10),
        "cec/cec2017_d10.csv",
        lambda n: cec.rules(2017).group(n).capitalize(),
        CEC_DEFAULTS,
    ),
    Page(
        "cec2019",
        "CEC 2019, the 100-Digit Challenge",
        lambda: digits,
        "cec/cec2019.csv",
        lambda n: digits_group(n).capitalize(),
        DEFAULTS,
    ),
    Page(
        "cec2022",
        "CEC 2022, D = 10",
        lambda: cec(2022, dim=10),
        "cec/cec2022_d10.csv",
        lambda n: cec.rules(2022).group(n).capitalize(),
        "the runner's defaults (a herd of 18 · D) at the competition's budget",
    ),
    Page(
        "cec2024",
        "CEC 2024, D = 30",
        lambda: cec(2024, dim=30),
        "cec/cec2024_d30.csv",
        lambda n: cec.rules(2024).group(n).capitalize(),
        CEC_DEFAULTS,
    ),
]


def _first_runs(path: Path) -> list[dict[str, str]]:
    """The rows of the first `RUNS` seeds of a results file."""
    found = rows(path)
    kept = set(sorted({int(row["seed"]) for row in found})[:RUNS])
    return [row for row in found if int(row["seed"]) in kept]


def _plain(value: float) -> str:
    """An optimum or best value as it is, to 10 significant digits."""
    return "0" if value == 0 else f"{value:.10g}"


def suite_page(page: Page, path: Path, docs: Path) -> Path:
    found = _first_runs(path)
    by_function = grouped(found, "function")
    suite = page.suite()
    body, solved_everywhere = [], 0
    for fn in suite:
        runs = by_function.get(fn.key)
        if not runs:
            continue
        errors = column(runs, "error")
        stats = summary(errors, zero=ZERO)
        solved = int(np.sum(errors < ZERO))
        solved_everywhere += solved == len(runs)
        body.append(
            [
                fn.key,
                fn.name,
                page.group(int(fn.key[1:])),
                fn.dim,
                _plain(fn.optimal),
                _plain(float(column(runs, "best").min())),
                stats["best"],
                stats["median"],
                f"{number(stats['mean'])} ± {number(stats['std'])}",
                stats["worst"],
                f"{solved}/{len(runs)}",
            ]
        )
    seeds = sorted({int(row["seed"]) for row in found})
    evaluations = {int(row["evaluations"]) for row in found}
    budget = " or ".join(f"{e:,}".replace(",", " ") for e in sorted(evaluations))
    intro = (
        f"NIO on {len(body)} functions: {len(seeds)} runs (seeds {seeds[0]}–{seeds[-1]}) of {budget} evaluations, "
        f"with {page.setup}. Results from `results/{page.results}`.\n\n"
        f"Solved in every run: **{solved_everywhere} of {len(body)}** functions."
    )
    notes = (
        "## Reading the table\n\n"
        "- **f\\***: the known optimum. **Best f(x)**: the best value any run found.\n"
        "- **Errors** are f(x) − f\\* of each run's best value; an error below 1e-8 counts as 0, and **Solved** counts "
        "the runs that reached it.\n"
    )
    return write(docs / f"{page.name}.md", page.title, intro, markdown(HEADER, body), notes)


def engineering_page(path: Path, docs: Path) -> Path:
    found = _first_runs(path)
    by_design = grouped(found, "design")
    body = []
    for code, design in DESIGNS.items():
        runs = by_design.get(code)
        if not runs:
            continue
        kept = [row for row in runs if row["feasible"] == "True"]  # an infeasible design's cost means nothing
        feasible = len(kept)
        if not kept:
            body.append(
                [design.name, design.dimensions, _plain(design.best), "—", "—", "—", "—", "—", f"0/{len(runs)}"]
            )
            continue
        stats = summary(column(kept, "cost"))
        gap = (stats["best"] - design.best) / abs(design.best) if design.best else stats["best"]
        body.append(
            [
                design.name,
                design.dimensions,
                _plain(design.best),
                _plain(stats["best"]),
                stats["median"],
                f"{number(stats['mean'], 7)} ± {number(stats['std'])}",
                stats["worst"],
                (f"{gap:.2%}" if design.best else number(gap)).replace("-0.00%", "0.00%"),
                f"{feasible}/{len(runs)}",
            ]
        )
    seeds = sorted({int(row["seed"]) for row in found})
    counts = sorted({len(runs) for runs in by_design.values()})
    runs = " to ".join(map(str, counts[:: max(len(counts) - 1, 1)]))
    intro = (
        f"NIO on {len(body)} constrained engineering designs, each minimised as its cost plus 1e6 times the sum of its "
        f"normalised constraint violations: {runs} runs per design (seeds {seeds[0]}–{seeds[-1]}), with the "
        f"runner's defaults (30 agents, 300 seasons). Results from `results/engineering/{path.name}`."
    )
    header = ["Design", "D", "Best known", "Best", "Median", "Mean ± std", "Worst", "Gap of best", "Feasible"]
    notes = (
        "## Reading the table\n\n"
        "- Costs are of the best design a run found, after snapping its discrete variables, over the feasible runs; "
        "lower is better.\n"
        "- **Gap of best**: how far the best run is above the best known cost. **Feasible** counts the runs whose "
        "best design breaks no constraint.\n"
    )
    return write(docs / "engineering.md", "Constrained engineering designs", intro, markdown(header, body, 7), notes)


def build(results=RESULTS, docs=DOCS) -> list[Path]:
    """Write every page whose results are present; return the files written."""
    results, docs = Path(results), Path(docs)
    written = [suite_page(page, results / page.results, docs) for page in PAGES if (results / page.results).exists()]
    designs = results / "engineering" / "designs.csv"
    if designs.exists():
        written.append(engineering_page(designs, docs))
    return written


if __name__ == "__main__":
    for path in build():
        print(path)
