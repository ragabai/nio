"""Run NIO on wrapper feature selection and write one CSV row per (dataset, run).

The objective is `FeatureSelection(dataset, validation=...)`: k-NN's validation error weighed against the share of
features kept. Every run of a dataset searches the same train/test split (seed 0); run `r` seeds NIO with
`--seed + r`. A row reports the subset NIO found: its size, its validation and test errors (error rate for
classification, RMSE over the targets' standard deviation for regression) and the kept features as a 0/1 string.
`--traces` also keeps every run's improvement trace, keyed "<dataset>/<seed>".

The defaults are the study's: classification with 5-fold cross-validation, 20 agents and 50 seasons; regression with
one held-out fold, 20 agents and 2 000 evaluations.

Usage:
    python -m benchmarks.selection.run --out results/selection/classification.csv
    python -m benchmarks.selection.run --task regression --out results/selection/regression.csv
    python -m benchmarks.selection.run --datasets Sonar Zoo --runs 2 --out quick.csv
"""

import argparse
import os
import time
from functools import cache
from multiprocessing import Pool

from benchmarks.runs import execute, trace_key, write_rows, write_traces
from nio import NIO
from problems import datasets
from problems.selection.objective import FeatureSelection

ALGORITHM = "NIO"
DEFAULTS = {  # validation, iterations, evaluations
    "classification": ("cv", 50, None),
    "regression": ("holdout", None, 2000),
}


@cache
def objective_for(name: str, task: str, validation: str) -> FeatureSelection:
    """The objective of one dataset, built once per worker process."""
    return FeatureSelection(datasets.load(name, task), validation=validation)


def run_one(task):
    """One run: `task` is (dataset, learning task, validation, seed, agents, iterations, evaluations). Returns its
    CSV row and its improvement trace."""
    name, learning, validation, seed, agents, iterations, evaluations = task
    objective = objective_for(name, learning, validation)
    optimizer = NIO(
        objective,
        objective.space,
        agents=agents,
        iterations=iterations,
        evaluations=evaluations,
        seed=seed,
        history=False,
    )
    result = optimizer.solve()
    best = objective.evaluate(result.position)
    row = {
        "dataset": name,
        "task": learning,
        "algorithm": ALGORITHM,
        "seed": seed,
        "evaluations": result.evaluations,
        "samples": objective.dataset.samples,
        "features": objective.dataset.dimensions,
        "selected": len(best.features),
        "ratio": best.ratio,
        "error": best.error,
        "test_error": best.test_error,
        "fitness": result.fitness,
        "mask": "".join("1" if kept else "0" for kept in best.mask),
        "seconds": round(result.runtime, 3),
    }
    return row, result.trace


def build_parser():
    p = argparse.ArgumentParser(prog="python -m benchmarks.selection.run", description="Run NIO on feature selection.")
    p.add_argument("--task", choices=list(DEFAULTS), default="classification", help="(default: classification)")
    p.add_argument("--datasets", nargs="*", metavar="NAME", help="datasets (default: every dataset of the task)")
    p.add_argument("--validation", choices=["holdout", "cv"], help="(default: cv for classification, else holdout)")
    p.add_argument("--agents", type=int, default=20, help="herd size (default: 20)")
    budget = p.add_mutually_exclusive_group()
    budget.add_argument("--iterations", type=int, help="seasons per run (default: 50 for classification)")
    budget.add_argument("--budget", type=int, help="evaluations per run (default: 2000 for regression)")
    p.add_argument("--runs", type=int, default=5, help="runs per dataset (default: 5)")
    p.add_argument("--seed", type=int, default=1000, help="seed of the first run; run r uses seed + r (default: 1000)")
    p.add_argument("--workers", type=int, default=os.cpu_count(), help="worker processes (default: all cores)")
    p.add_argument("--out", required=True, metavar="PATH", help="the results CSV to write")
    p.add_argument("--traces", metavar="PATH", help="also write every run's improvement trace to this .npz")
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validation, iterations, evaluations = DEFAULTS[args.task]
    validation = args.validation or validation
    if args.iterations is not None or args.budget is not None:
        iterations, evaluations = args.iterations, args.budget
    available = datasets.available(args.task)
    try:
        names = [datasets.load(name, args.task).name for name in (args.datasets or available)]
    except ValueError as error:
        parser.error(str(error))
    if not names or args.runs < 1 or args.agents < 4:
        parser.error("give at least one dataset, --runs of at least 1 and --agents of at least 4")
    seeds = range(args.seed, args.seed + args.runs)
    tasks = [
        (name, args.task, validation, seed, args.agents, iterations, evaluations) for name in names for seed in seeds
    ]
    start = time.perf_counter()
    done = execute(run_one, tasks, args.workers, Pool)
    done.sort(key=lambda run: (available.index(run[0]["dataset"]), run[0]["seed"]))
    out = write_rows([row for row, _ in done], args.out)
    if args.traces:
        write_traces({trace_key(row["dataset"], row["seed"]): trace for row, trace in done}, args.traces)
    print(f"{len(done)} runs, {time.perf_counter() - start:.0f}s  ->  {out}")


if __name__ == "__main__":
    main()
