"""Reports of every study in `results/`: a markdown summary next to each results CSV, figures in `results/figures/`,
and an index of both in `results/README.md`.

    python -m benchmarks.report               # every study whose results are present
    python -m benchmarks.report --landscapes  # also the CEC galleries at D = 2, which take a few minutes

A study whose results are missing is skipped, so the report can be rebuilt while runs are still going.
"""

from pathlib import Path

import plots
from benchmarks.paths import RESULTS
from benchmarks.report import applications, engineering, initializers, landscapes, suites

SUITE_FILES = ("*.csv", "cec/*.csv")


def build(results=RESULTS, galleries: bool = False) -> list[Path]:
    """Write every report whose results are present under `results`, and the index; return the files written."""
    with plots.browser():  # one Chrome draws every figure
        written = _reports(Path(results), galleries)
    written.append(index(results))
    return written


def _reports(results: Path, galleries: bool) -> list[Path]:
    figures = results / "figures"
    written: list[Path] = []
    for pattern in SUITE_FILES:
        for path in sorted(results.glob(pattern)):
            written += suites.report(path, figures / ("cec" if path.parent.name == "cec" else "classic"))
    studies = [
        (results / "engineering" / "designs.csv", engineering.designs, "engineering"),
        (results / "engineering" / "rc2020.csv", engineering.realworld, "engineering"),
        (results / "selection" / "classification.csv", applications.selection, "selection"),
        (results / "selection" / "regression.csv", applications.selection, "selection"),
        (results / "scheduling" / "hcsp.csv", applications.scheduling, "scheduling"),
    ]
    for path, report, folder in studies:
        if path.exists():
            written += report(path, figures / folder)
    if (results / "initializers").is_dir():
        written += initializers.report(results / "initializers", figures / "initializers")
    written += initializers.gallery(figures / "initializers")
    if galleries:
        for suite in ("cec2017", "cec2022"):
            written += landscapes.gallery(suite, figures / "landscapes")
    return written


def index(results=RESULTS) -> Path:
    """`results/README.md`: every summary and every figure under `results`, linked by folder."""
    results = Path(results)
    lines = [
        "# Results",
        "",
        "Every CSV holds one row per run; the `.npz` beside it holds each run's improvement trace (evaluation, best "
        "value), keyed `<problem>/<seed>` (`benchmarks.runs.read_traces`). `python -m benchmarks.report` rebuilds the "
        "summaries and the figures from them.",
        "",
        "## Summaries",
        "",
    ]
    for path in sorted(results.rglob("*.md")):
        if path.name != "README.md":
            relative = path.relative_to(results).as_posix()
            lines.append(f"- [{relative}]({relative})")
    lines += ["", "## Figures", ""]
    for path in sorted((results / "figures").rglob("*.png")):
        relative = path.relative_to(results).as_posix()
        lines.append(f"- [{relative}]({relative})")
    out = results / "README.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return out


__all__ = ["build", "index"]
