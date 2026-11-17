"""The report of a benchmark-suite CSV (`benchmarks.run`): the error statistics per function, the final errors as
boxes and, when the runs' traces were kept, the median convergence of every function."""

import re
from pathlib import Path

from benchmarks.curves import band, floored
from benchmarks.report.charts import REPORT, panels
from benchmarks.report.tables import ZERO, column, grouped, markdown, rows, summary, write
from benchmarks.runs import read_traces, trace_key
from plots import Boxes
from problems.cec import digits
from problems.functions import cec

CLASSIC_GROUPS = {**dict.fromkeys(range(1, 8), "unimodal"), **dict.fromkeys(range(8, 14), "multimodal")}
CEC_NAME = re.compile(r"(cec\d{4})_d(\d+)")


def group_of(name: str, function: str) -> str:
    """The group of a function: from its CEC suite when the file is named after one, else the classic suite's."""
    number = int(function.removeprefix("F"))
    if match := CEC_NAME.fullmatch(name):
        return cec.rules(int(match[1][3:])).group(number)
    if name.startswith("cec2019"):
        return digits.group(number)
    return CLASSIC_GROUPS.get(number, "fixed-dimension")


def title_of(name: str) -> str:
    """A readable title for a results file: "CEC 2017, D = 10" for cec2017_d10."""
    if match := CEC_NAME.fullmatch(name):
        return f"CEC {match[1][3:]}, D = {match[2]}"
    if name.startswith("cec2019"):
        return "CEC 2019" + name.removeprefix("cec2019").replace("_", " ")
    return name.replace("_", " ")


def report(path, figures) -> list[Path]:
    """Write `<name>.md` next to the CSV, and the figures of the suite into `figures`."""
    path, figures = Path(path), Path(figures)
    name = path.stem
    found = rows(path)
    by_function = grouped(found, "function")
    title = title_of(name)
    runs = len(next(iter(by_function.values())))
    budget = int(column(found, "evaluations").max())
    table = []
    for function, runs_of in by_function.items():
        stats = summary(column(runs_of, "error"), ZERO)
        table.append(
            [
                function,
                group_of(name, function),
                stats["best"],
                stats["median"],
                stats["mean"],
                stats["worst"],
                stats["std"],
                float(column(runs_of, "seconds").mean()),
            ]
        )
    solved = sum(summary(column(r, "error"), ZERO)["median"] == 0 for r in by_function.values())
    intro = (
        f"NIO on {title}: {len(by_function)} functions, {runs} runs each (seeds {found[0]['seed']} on), "
        f"{budget:,} evaluations per run. Errors are f(best) − f*, and errors below {ZERO:g} count as 0. "
        f"The median run solves {solved} of {len(by_function)} functions to that tolerance."
    )
    header = ["Function", "Group", "Best", "Median", "Mean", "Worst", "Std", "Seconds per run"]
    written = [write(path.with_suffix(".md"), f"NIO on {title}", intro, markdown(header, table))]

    errors = {f: floored(column(r, "error")) for f, r in by_function.items()}
    fig = Boxes(errors, size=(max(640, 32 * len(by_function) + 150), 340), log=True, **REPORT)
    fig.update_yaxes(title="Final error")
    fig.update_layout(title=f"Final errors, {title} (a dot per run; 1e-8 and below: solved)", showlegend=False)
    written += fig.save(figures / f"{name}_errors")

    traces_path = path.with_suffix(".npz")
    if traces_path.exists():
        traces = read_traces(traces_path)
        charts = {}
        for function, runs_of in by_function.items():
            optimum = float(runs_of[0]["best"]) - float(runs_of[0]["error"])
            kept = [traces[trace_key(function, int(row["seed"]))] for row in runs_of]
            charts[function] = {"NIO": band(kept, optimum, budget)}
        fig = panels(
            charts,
            columns=6,
            width=1200,
            row_height=180,
            ylabel="Error (median and quartiles)",
            title=f"Convergence, {title}",
        )
        written += fig.save(figures / f"{name}_convergence")
    return written
