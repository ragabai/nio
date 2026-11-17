"""The significance mark used to compare two samples of final errors."""

import numpy as np
from scipy.stats import ranksums

TIE = 1e-8  # errors below this are numerical noise around the optimum: they count as 0


def compare(x, y, floor=TIE):
    """'+' if x is significantly better (lower) than y, '-' if significantly worse, '=' otherwise.

    The test is the Wilcoxon rank-sum test at p < 0.05. Errors under `floor` (1e-8 by default, the numerical precision
    of the optimum) count as 0 in the test and in the medians, so a tiny error ties with an exact zero; negative errors
    count as 0 too. `floor=0` ranks the raw errors (negative ones as 0), without the tie floor. When the floored
    samples match within 1e-8 once sorted, the test is not run and the result is '='.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    x, y = np.where(x < floor, 0.0, np.maximum(x, 0.0)), np.where(y < floor, 0.0, np.maximum(y, 0.0))
    if np.allclose(np.sort(x), np.sort(y), atol=TIE, rtol=0):
        return "="
    p = ranksums(x, y).pvalue
    sign = np.sign(np.median(y) - np.median(x)) or np.sign(y.mean() - x.mean())
    if p < 0.05:
        return "+" if sign > 0 else "-"
    return "="
