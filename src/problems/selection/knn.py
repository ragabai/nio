"""k-nearest-neighbour prediction (Euclidean), the learner the wrapper feature selection scores subsets with."""

import numpy as np


def nearest(train_x: np.ndarray, test_x: np.ndarray, neighbours: int = 5) -> np.ndarray:
    """Indices of the `neighbours` nearest training samples (Euclidean) of every test sample."""
    neighbours = min(neighbours, len(train_x))
    distances = (test_x**2).sum(axis=1)[:, None] + (train_x**2).sum(axis=1)[None, :] - 2 * test_x @ train_x.T
    return np.argpartition(distances, neighbours - 1, axis=1)[:, :neighbours]


def classify(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, neighbours: int = 5) -> np.ndarray:
    """Majority vote of the nearest training samples; labels must be codes 0..k-1, and ties go to the smallest."""
    votes = np.zeros((len(test_x), int(train_y.max()) + 1))
    np.add.at(votes, (np.arange(len(test_x))[:, None], train_y[nearest(train_x, test_x, neighbours)]), 1)
    return votes.argmax(axis=1)


def regress(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, neighbours: int = 5) -> np.ndarray:
    """Mean target of the nearest training samples."""
    return train_y[nearest(train_x, test_x, neighbours)].mean(axis=1)


def error_rate(predicted: np.ndarray, actual: np.ndarray) -> float:
    """The share of wrong predictions, 1 − accuracy."""
    return float(np.mean(predicted != actual))


def relative_rmse(predicted: np.ndarray, actual: np.ndarray, scale: float) -> float:
    """Root mean squared error over `scale` (the target's standard deviation): 0 is exact, 1 no better than the mean."""
    return float(np.sqrt(np.mean((predicted - actual) ** 2)) / scale)
