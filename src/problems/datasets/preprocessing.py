"""Seeded train/test splits, stratified k-fold partitions and label encoding."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class Split:
    """A train/test split; it unpacks as `x_train, x_test, y_train, y_test`, the order scikit-learn uses."""

    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray

    def __iter__(self):
        return iter((self.x_train, self.x_test, self.y_train, self.y_test))


def split(
    x: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int | None = None, stratify: bool = True
) -> Split:
    """Shuffle into train and test sets, keeping each class's share in both when stratified."""
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}.")
    if len(x) != len(y):
        raise ValueError(f"x and y lengths must match, got {len(x)} and {len(y)}.")
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == label) for label in np.unique(y)] if stratify else [np.arange(len(y))]
    test = np.zeros(len(y), dtype=bool)
    for group in groups:
        test[rng.permutation(group)[: round(len(group) * test_size)]] = True
    train, test = rng.permutation(np.flatnonzero(~test)), rng.permutation(np.flatnonzero(test))
    return Split(x[train], x[test], y[train], y[test])


def kfold(y: np.ndarray, k: int = 5, seed: int | None = None) -> list[np.ndarray]:
    """Stratified k-fold: `k` disjoint index sets covering every sample, each holding about 1/k of every class.

    The members of each class are dealt round-robin, continuing across classes; for a numeric regression target every
    value is its own class, so the folds spread the target's range evenly.
    """
    if not 2 <= k <= len(y):
        raise ValueError(f"k must be between 2 and the {len(y)} samples, got {k}.")
    rng = np.random.default_rng(seed)
    assignment = np.empty(len(y), dtype=int)
    offset = 0
    for label in np.unique(y):
        members = rng.permutation(np.flatnonzero(y == label))
        assignment[members] = (offset + np.arange(len(members))) % k
        offset += len(members)
    return [np.flatnonzero(assignment == fold) for fold in range(k)]


def encode(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map labels to integer codes 0..k-1; returns the codes and the sorted classes they index."""
    classes, codes = np.unique(np.asarray(labels).ravel(), return_inverse=True)
    return codes, classes
