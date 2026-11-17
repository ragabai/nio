"""Wrapper feature selection with k-NN, for classification and regression.

The fitness of a subset is `0.99 · validation error + 0.01 · share of features kept`; the validation error comes from
k-NN on the training part (one held-out fold, or 5-fold cross-validation), and a held-out test part is never searched.

    >>> import numpy as np
    >>> from problems import datasets
    >>> from problems.selection import FeatureSelection
    >>> objective = FeatureSelection(datasets.load("Iris"), validation="cv")
    >>> objective.space.dimensions, objective(np.zeros(4))
    (4, 1.0)
"""

from problems.selection.knn import classify, error_rate, nearest, regress, relative_rmse
from problems.selection.objective import FeatureSelection, Selection, Validation

__all__ = [
    "FeatureSelection",
    "Selection",
    "Validation",
    "classify",
    "error_rate",
    "nearest",
    "regress",
    "relative_rmse",
]
