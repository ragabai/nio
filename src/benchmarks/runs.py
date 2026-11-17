"""What every runner shares: running tasks over worker processes, and writing the rows and the runs they return.

A runner turns each (problem, seed) into a task, runs the tasks with `execute`, writes one CSV row per run with
`write_rows`, and keeps every run's improvement trace (evaluation, best value) with `write_traces`, keyed
"<problem>/<seed>", so a convergence curve can be drawn from any run without running it again.
"""

import csv
import pathlib
import sys
from collections.abc import Callable, Iterable, Mapping
from multiprocessing import Pool

import numpy as np

from utils.progress import Progress


def execute[T, R](run: Callable[[T], R], tasks: Iterable[T], workers: int | None, pool=Pool) -> list[R]:
    """Every task's result, from `workers` processes, in the order the tasks finish, with a progress bar.

    `pool` is the pool class, so a caller (or a test) can give its own.
    """
    tasks = list(tasks)
    with pool(workers) as workers_pool:
        done = Progress(workers_pool.imap_unordered(run, tasks), total=len(tasks), label="runs", stream=sys.stdout)
        return list(done)  # progress on stdout, which the campaign logs capture


def write_rows(rows: list[Mapping], path) -> pathlib.Path:
    """Write the rows to a CSV with the first row's keys as its header, creating the directory if needed."""
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return out


def write_traces(traces: Mapping[str, np.ndarray], path) -> pathlib.Path:
    """Write every run's trace, an (improvements × 2) array of (evaluation, best value), to a compressed .npz."""
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        np.savez_compressed(fh, **{key: np.asarray(trace, dtype=float) for key, trace in traces.items()})
    return out


def read_traces(path) -> dict[str, np.ndarray]:
    """The traces `write_traces` wrote, keyed "<problem>/<seed>"."""
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def trace_key(problem: str, seed: int) -> str:
    """The key of one run's trace."""
    return f"{problem}/{seed}"
