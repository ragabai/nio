"""The reports of the feature-selection and the task-scheduling studies."""

from pathlib import Path

import numpy as np

from benchmarks.curves import band
from benchmarks.report.charts import REPORT, panels
from benchmarks.report.tables import column, grouped, markdown, rows, write
from benchmarks.runs import read_traces, trace_key
from plots import Bars, Boxes


def selection(path, figures) -> list[Path]:
    """`<task>.md` next to the CSV, and the test score and the share of features kept per dataset."""
    path, figures = Path(path), Path(figures)
    found = rows(path)
    task = found[0]["task"]
    by_dataset = grouped(found, "dataset")
    classification = task == "classification"
    table, score, spread, kept = [], {}, {}, {}
    for name, runs_of in by_dataset.items():
        test = column(runs_of, "test_error")
        measure = 1 - test if classification else test
        score[name], spread[name] = float(measure.mean()), (float(measure.min()), float(measure.max()))
        kept[name] = float(column(runs_of, "ratio").mean())
        table.append(
            [
                name,
                int(runs_of[0]["samples"]),
                int(runs_of[0]["features"]),
                float(column(runs_of, "selected").mean()),
                float(column(runs_of, "error").mean()),
                float(measure.mean()),
                float(measure.std()),
                float(column(runs_of, "fitness").mean()),
            ]
        )
    runs = len(next(iter(by_dataset.values())))
    measure_name = "test accuracy" if classification else "test relative RMSE"
    validation = (
        "the error rate under 5-fold cross-validation"
        if classification
        else "the RMSE over the targets' standard deviation on one held-out fold"
    )
    intro = (
        f"NIO selecting features for k-NN (k = 5) on {len(by_dataset)} {task} datasets, {runs} runs each (seeds "
        f"{found[0]['seed']} on), {int(column(found, 'evaluations').max()):,} evaluations per run. The fitness is "
        f"0.99 · validation error + 0.01 · share of features kept; the validation error is {validation} of the "
        f"training part. The {measure_name} is measured on a held-out 20 % the search never sees. Across datasets "
        f"NIO keeps {100 * np.mean(list(kept.values())):.0f} % of the features on average."
    )
    header = [
        "Dataset",
        "Samples",
        "Features",
        "Selected (mean)",
        "Validation error",
        f"Mean {measure_name}",
        "Std",
        "Fitness",
    ]
    title = f"NIO for feature selection ({task})"
    written = [write(path.with_suffix(".md"), title, intro, markdown(header, table))]

    titles = (f"{measure_name.capitalize()} of the selected features", "Features kept")
    fig = Bars(size=(900, 17 * len(by_dataset) + 110), columns=2, shared_yaxes=True, titles=titles, **REPORT)
    fig.add(score, spread=spread, fmt="{:.3f}", series=0, row=1, col=1)
    fig.add(kept, fmt="{:.0%}", series=1, row=1, col=2)
    limit = 1.15 if classification else 1.3 * max(score.values())
    fig.update_xaxes(title=f"{measure_name.capitalize()} (mean; line: range over runs)", range=[0, limit], row=1, col=1)
    fig.update_xaxes(title="Share of features kept (mean)", range=[0, 1.15], row=1, col=2)
    written += fig.update_layout(showlegend=False).save(figures / f"selection_{task}")
    return written


def scheduling(path, figures) -> list[Path]:
    """`hcsp.md` next to the CSV, the gaps to the lower bound, and the convergence of both encodings."""
    path, figures = Path(path), Path(figures)
    found = rows(path)
    by_instance = grouped(found, "instance")
    table, gaps = [], {}
    for instance, runs_of in by_instance.items():
        gaps[instance] = {}
        for encoding, of_encoding in grouped(runs_of, "encoding").items():
            makespan, gap = column(of_encoding, "makespan"), column(of_encoding, "gap")
            gaps[instance][encoding] = 100 * gap
            table.append(
                [
                    instance,
                    encoding,
                    makespan.min(),
                    makespan.mean(),
                    makespan.std(),
                    float(of_encoding[0]["bound"]),
                    f"{100 * gap.min():.1f} %",
                    f"{100 * gap.mean():.1f} %",
                    float(column(of_encoding, "utilization").mean()),
                ]
            )
    runs = len(grouped(next(iter(by_instance.values())), "encoding")["priority"])
    intro = (
        f"NIO scheduling 1024 independent tasks on 32 heterogeneous machines (HCSP), {runs} runs per instance and "
        f"encoding (seeds {found[0]['seed']} on), {int(column(found, 'evaluations').max()):,} evaluations per run. The "
        f"gap is the makespan's excess over the lower bound, total work over total speed, which no schedule can reach."
    )
    header = [
        "Instance",
        "Encoding",
        "Best makespan",
        "Mean makespan",
        "Std",
        "Lower bound",
        "Best gap",
        "Mean gap",
        "Mean utilization",
    ]
    written = [write(path.with_suffix(".md"), "NIO for task scheduling", intro, markdown(header, table))]

    fig = Boxes(size=(640, 320), **REPORT)
    encodings = list(dict.fromkeys(encoding for by_encoding in gaps.values() for encoding in by_encoding))
    for series, encoding in enumerate(encodings):
        of_encoding = {instance: by_encoding.get(encoding, []) for instance, by_encoding in gaps.items()}
        fig.add(of_encoding, name=encoding, series=series)
    fig.update_yaxes(title="Gap to the lower bound (%)")
    fig.update_layout(title="Makespan above the lower bound, by encoding")
    written += fig.save(figures / "scheduling_gaps")

    traces_path = path.with_suffix(".npz")
    if traces_path.exists():
        traces = read_traces(traces_path)
        budget = int(column(found, "evaluations").max())
        charts = {}
        for instance, runs_of in by_instance.items():
            bound = float(runs_of[0]["bound"])
            charts[instance] = {
                encoding: band(
                    [traces[trace_key(f"{instance}-{encoding}", int(r["seed"]))] for r in of_encoding],
                    0.0,
                    budget,
                    floor=0.0,
                )
                for encoding, of_encoding in grouped(runs_of, "encoding").items()
            }
            for curve in charts[instance].values():
                for values in (curve.median, curve.low, curve.high):
                    values /= bound
        fig = panels(charts, columns=4, width=1100, row_height=240, log=False, ylabel="Makespan / lower bound")
        written += fig.save(figures / "scheduling_convergence")
    return written
