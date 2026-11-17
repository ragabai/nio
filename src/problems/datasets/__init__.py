"""The datasets of the feature-selection study: 42 for classification and 7 for regression (`data/datasets/`).

    >>> from problems import datasets
    >>> datasets.load("Sonar")
    Dataset('Sonar', samples=208, dimensions=60, classes=2)
    >>> len(datasets.available()), len(datasets.available("regression"))
    (42, 7)

`data/datasets/sources.json` records where each dataset comes from (UCI or OpenML) and how it was prepared.
"""

from problems.datasets.catalog import DATASETS, TASKS, available, files, load, read
from problems.datasets.dataset import Dataset, Task
from problems.datasets.preprocessing import Split, encode, kfold, split

__all__ = [
    "DATASETS",
    "TASKS",
    "Dataset",
    "Split",
    "Task",
    "available",
    "encode",
    "files",
    "kfold",
    "load",
    "read",
    "split",
]
