"""Wrapper feature selection as an objective: k-NN validation error against the share of features kept."""

from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np

from problems.box import Box
from problems.datasets import Dataset, Split, kfold
from problems.selection.knn import classify, error_rate, regress, relative_rmse

type Validation = Literal["holdout", "cv"]


@dataclass(frozen=True, eq=False)
class Selection:
    """The features a position keeps and how well k-NN predicts from them.

    Errors are error rates for classification (1 − accuracy) and relative RMSE for regression.
    """

    mask: np.ndarray  # True for every kept feature
    error: float  # validation error the search optimised
    test_error: float  # error on the held-out test part the search never saw
    fitness: float  # objective value, lower is better

    @property
    def features(self) -> np.ndarray:
        """The indices of the kept features."""
        return np.flatnonzero(self.mask)

    @property
    def ratio(self) -> float:
        """Share of the features kept."""
        return float(self.mask.mean())


@dataclass(frozen=True, eq=False)
class FeatureSelection:
    """Wrapper feature selection as a minimisation objective over [0, 1]^features, for classification or regression.

    The data is split once into a training part and a held-out test part the search never sees. A coordinate
    above `threshold` keeps its feature, and the fitness uses the training part only:
    alpha * validation error + (1 - alpha) * kept / total, where the validation error comes from k-NN on one
    fold ("holdout") or on all `folds` in turn ("cv"). Classification scores the error rate of a k-NN vote;
    regression scores the RMSE of the k-NN mean over the training targets' standard deviation, so in both an
    error of 1 is what keeping no feature costs. `evaluate` also reports the test error, from k-NN trained on
    the whole training part.
    """

    dataset: Dataset
    alpha: float = 0.99
    neighbours: int = 5
    threshold: float = 0.5
    validation: Validation = "holdout"
    folds: int = 5
    test_size: float = 0.2
    seed: int = 0

    def __post_init__(self):
        if not 0 <= self.alpha <= 1:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}.")
        if not 0 < self.threshold < 1:
            raise ValueError(f"threshold must be in (0, 1), got {self.threshold}.")
        if self.validation not in ("holdout", "cv"):
            raise ValueError(f"Unknown validation '{self.validation}'. Choose 'holdout' or 'cv'.")

    @property
    def classification(self) -> bool:
        """Whether the dataset is a classification task."""
        return self.dataset.task == "classification"

    @cached_property
    def split(self) -> Split:
        """Train/test split (stratified for classification) with features min-max scaled on the training part."""
        dataset = self.dataset.encoded() if self.classification else self.dataset
        x_train, x_test, y_train, y_test = dataset.split(self.test_size, self.seed)
        low, high = x_train.min(axis=0), x_train.max(axis=0)
        scale = np.where(high > low, high - low, 1.0)
        return Split((x_train - low) / scale, (x_test - low) / scale, y_train, y_test)

    @cached_property
    def partitions(self) -> list[np.ndarray]:
        """Training indices held out for validation: one fold for holdout, every fold for cv.

        Folds keep each class's share, and for regression spread the target's range evenly.
        """
        folds = kfold(self.split.y_train, self.folds, self.seed)
        return folds[:1] if self.validation == "holdout" else folds

    @property
    def space(self) -> Box:
        """[0, 1]^features: one coordinate per feature."""
        return Box.of(0, 1, self.dataset.dimensions)

    def mask(self, position: np.ndarray) -> np.ndarray:
        """The features a position keeps: those whose coordinate is above `threshold`."""
        return np.asarray(position) > self.threshold

    @cached_property
    def scale(self) -> float:
        """The training targets' standard deviation, which regression errors are measured against."""
        return float(np.std(self.split.y_train)) or 1.0

    def _error(self, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray) -> float:
        if self.classification:
            return error_rate(classify(train_x, train_y, test_x, self.neighbours), test_y)
        return relative_rmse(regress(train_x, train_y, test_x, self.neighbours), test_y, self.scale)

    def error(self, mask: np.ndarray) -> float:
        """Mean validation error over the partitions using only the masked features; 1 when none is kept."""
        if not mask.any():
            return 1.0
        x, y = self.split.x_train[:, mask], self.split.y_train
        errors = []
        for held in self.partitions:
            fit = np.ones(len(y), dtype=bool)
            fit[held] = False
            errors.append(self._error(x[fit], y[fit], x[held], y[held]))
        return float(np.mean(errors))

    def test(self, mask: np.ndarray) -> float:
        """Error on the held-out test part, with k-NN trained on the whole training part; 1 when no feature is kept."""
        if not mask.any():
            return 1.0
        x_train, x_test, y_train, y_test = self.split
        return self._error(x_train[:, mask], y_train, x_test[:, mask], y_test)

    def fitness(self, mask: np.ndarray, error: float) -> float:
        """alpha · error + (1 − alpha) · share kept; 1 when no feature is kept."""
        return float(self.alpha * error + (1 - self.alpha) * mask.mean()) if mask.any() else 1.0

    def evaluate(self, position: np.ndarray) -> Selection:
        """The subset a position keeps, with its validation error, its test error and its fitness."""
        mask = self.mask(position)
        error = self.error(mask)
        return Selection(mask, error, self.test(mask), self.fitness(mask, error))

    def __call__(self, position: np.ndarray) -> float:
        mask = self.mask(position)
        return self.fitness(mask, self.error(mask))
