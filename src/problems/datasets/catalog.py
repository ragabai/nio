"""The dataset files: `data/datasets/<task>/<name>.csv`, features in every column but the last, the label in the last.

A file may start with a header row (it does when a feature cell of the first row is not a number). Features are read
as floats; labels are read as numbers when every label is one, and as strings otherwise.
"""

import csv
from pathlib import Path

import numpy as np

from problems.datasets.dataset import Dataset, Task
from problems.paths import DATA

DATASETS = DATA / "datasets"
TASKS = ("classification", "regression")


def _is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def read(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """The features and the labels of a dataset file."""
    with Path(path).open(newline="") as fh:
        rows = [row for row in csv.reader(fh) if row]
    if not all(_is_number(value) for value in rows[0][:-1]):
        rows = rows[1:]
    x = np.array([row[:-1] for row in rows], dtype=float)
    labels = [row[-1] for row in rows]
    y = np.array(labels, dtype=float) if all(_is_number(label) for label in labels) else np.array(labels)
    return x, y


def _directory(task: Task) -> Path:
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}. Choose 'classification' or 'regression'.")
    return DATASETS / task


def files(task: Task = "classification") -> dict[str, Path]:
    """Every dataset file of `task`, keyed by name, in case-insensitive name order."""
    return {path.stem: path for path in sorted(_directory(task).glob("*.csv"), key=lambda p: p.stem.lower())}


def available(task: Task = "classification") -> list[str]:
    """The names of the datasets of `task`."""
    return list(files(task))


def load(name: str, task: Task = "classification") -> Dataset:
    """A dataset by name, ignoring case and surrounding spaces."""
    lookup = {stem.lower(): path for stem, path in files(task).items()}
    try:
        path = lookup[name.strip().lower()]
    except KeyError:
        raise ValueError(f"Unknown {task} dataset {name!r}. Choose from {', '.join(available(task))}.") from None
    return Dataset(path.stem, *read(path), task=task)
