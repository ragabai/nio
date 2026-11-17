"""A dataset: features, one label or target per sample, and whether it is a classification or a regression task."""

from dataclasses import dataclass, replace
from typing import Literal, Self

import numpy as np

from problems.datasets.preprocessing import Split, encode, split

type Task = Literal["classification", "regression"]


@dataclass(frozen=True, eq=False)
class Dataset:
    """Features `x` (samples × dimensions) and one label (classification) or numeric target (regression) per sample."""

    name: str
    x: np.ndarray
    y: np.ndarray
    task: Task = "classification"

    def __post_init__(self):
        if self.task not in ("classification", "regression"):
            raise ValueError(f"{self.name}: task must be 'classification' or 'regression', got {self.task!r}.")
        if self.x.ndim != 2 or len(self.x) == 0:
            raise ValueError(f"{self.name}: features must be a non-empty 2-D array, got shape {self.x.shape}.")
        if len(self.x) != len(self.y):
            raise ValueError(f"{self.name}: {len(self.x)} samples but {len(self.y)} labels.")

    @property
    def samples(self) -> int:
        """The number of samples."""
        return self.x.shape[0]

    @property
    def dimensions(self) -> int:
        """The number of features."""
        return self.x.shape[1]

    @property
    def classes(self) -> np.ndarray:
        """The distinct labels, sorted."""
        return np.unique(self.y)

    def split(self, test_size: float = 0.2, seed: int | None = None, stratify: bool = True) -> Split:
        """Train/test split, keeping each class's share when stratified; regression targets are never stratified."""
        return split(self.x, self.y, test_size, seed, stratify and self.task == "classification")

    def encoded(self) -> Self:
        """A copy whose labels are integer codes 0..k-1."""
        return replace(self, y=encode(self.y)[0])

    def __repr__(self) -> str:
        target = f"classes={len(self.classes)}" if self.task == "classification" else "task='regression'"
        return f"Dataset({self.name!r}, samples={self.samples}, dimensions={self.dimensions}, {target})"
