"""The reports of the engineering studies: the classic designs and the CEC 2020 real-world constrained suite."""

from pathlib import Path

import numpy as np

from benchmarks.curves import band
from benchmarks.report.charts import REPORT, panels
from benchmarks.report.tables import column, grouped, markdown, rows, summary, write
from benchmarks.runs import read_traces, trace_key
from plots import Bars
from problems.engineering import DESIGNS
from problems.engineering.realworld import CATEGORIES, PROBLEMS


def designs(path, figures) -> list[Path]:
    """`designs.md` next to the CSV, and the median convergence of every design."""
    path, figures = Path(path), Path(figures)
    found = rows(path)
    by_design = grouped(found, "design")
    table = []
    for code, runs_of in by_design.items():
        cost = summary(column(runs_of, "cost"))
        feasible = np.mean([row["feasible"] == "True" for row in runs_of])
        table.append(
            [
                DESIGNS[code].name,
                DESIGNS[code].best,
                cost["best"],
                cost["median"],
                cost["mean"],
                cost["worst"],
                cost["std"],
                f"{feasible:.0%}",
            ]
        )
    runs = len(next(iter(by_design.values())))
    intro = (
        f"NIO on the {len(by_design)} constrained engineering designs, {runs} runs each (seeds {found[0]['seed']} "
        f"on), each minimising the cost plus 1e6 times the sum of the normalised constraint violations. Costs are of "
        f"the best design a run found, after snapping its discrete variables."
    )
    header = ["Design", "Best known", "Best", "Median", "Mean", "Worst", "Std", "Feasible"]
    written = [write(path.with_suffix(".md"), "NIO on the engineering designs", intro, markdown(header, table, 7))]
    traces_path = path.with_suffix(".npz")
    if traces_path.exists():
        traces = read_traces(traces_path)
        budget = int(column(found, "evaluations").max())
        charts = {
            DESIGNS[code].name: {
                "NIO": band([traces[trace_key(code, int(r["seed"]))] for r in runs_of], 0.0, budget, floor=1e-14)
            }
            for code, runs_of in by_design.items()
        }
        references = {DESIGNS[code].name: DESIGNS[code].best for code in by_design}
        fig = panels(
            charts,
            columns=4,
            width=1000,
            row_height=220,
            references=references,
            ylabel="Penalised cost",
            title="Convergence on the engineering designs (dashed: best known)",
        )
        written += fig.save(figures / "designs_convergence")
    return written


def standing(f: np.ndarray, violation: np.ndarray) -> np.ndarray:
    """The competition's order of runs: feasible runs first by f, then infeasible runs by their violation."""
    return np.lexsort((np.where(violation > 0, violation, f), violation > 0))


def realworld(path, figures) -> list[Path]:
    """`rc2020.md` next to the CSV with the competition's statistics, and the feasibility rate per category."""
    path, figures = Path(path), Path(figures)
    found = rows(path)
    by_problem = grouped(found, "problem")
    table, feasible_rates = [], {}
    for code, runs_of in by_problem.items():
        f, violation = column(runs_of, "f"), column(runs_of, "violation")
        order = standing(f, violation)
        best, median, worst = order[0], order[(len(order) - 1) // 2], order[-1]
        rate = float(np.mean(violation <= 0))
        feasible_rates[code] = rate
        table.append(
            [
                code,
                PROBLEMS[code].name,
                PROBLEMS[code].dimensions,
                PROBLEMS[code].best,
                f[best],
                violation[best],
                f[median],
                violation[median],
                f[worst],
                violation[worst],
                f.mean(),
                violation.mean(),
                f"{rate:.0%}",
            ]
        )
    runs = len(next(iter(by_problem.values())))
    solved = sum(
        1
        for code, runs_of in by_problem.items()
        if any(
            float(r["violation"]) <= 0
            and float(r["f"]) <= PROBLEMS[code].best + 1e-4 * max(1.0, abs(PROBLEMS[code].best))
            for r in runs_of
        )
    )
    intro = (
        f"NIO on the CEC 2020 real-world constrained suite: {len(by_problem)} problems, {runs} runs each (seeds "
        f"{found[0]['seed']} on), each with the competition's evaluation budget for its dimension, minimising f plus "
        f"1e6 times the sum of the violations. ν is the competition's mean constraint violation (0 is feasible). Runs "
        f"are ordered as the competition orders them: feasible runs by f, then infeasible runs by ν. FR is the share "
        f"of runs ending feasible. A feasible run reaches the best known f (to 1e-4, relative) on {solved} problems."
    )
    header = [
        "Problem",
        "Name",
        "D",
        "Best known f",
        "Best f",
        "Best ν",
        "Median f",
        "Median ν",
        "Worst f",
        "Worst ν",
        "Mean f",
        "Mean ν",
        "FR",
    ]
    written = [write(path.with_suffix(".md"), "NIO on the CEC 2020 real-world suite", intro, markdown(header, table))]

    categories = {
        category: float(np.mean([feasible_rates[code] for code in codes if code in feasible_rates]))
        for category, codes in CATEGORIES.items()
        if any(code in feasible_rates for code in codes)
    }
    fig = Bars(categories, fmt="{:.0%}", size=(640, 260), **REPORT)
    fig.update_xaxes(title="Share of runs ending feasible", range=[0, 1.12])
    fig.update_layout(title="Feasibility on the CEC 2020 real-world suite, by category", showlegend=False)
    written += fig.save(figures / "rc2020_feasibility")
    return written
