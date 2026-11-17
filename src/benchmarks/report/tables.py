"""What every study's report shares: reading result CSVs, summary statistics, average ranks and markdown tables."""

import csv
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

ZERO = 1e-8  # errors below this count as 0, as the CEC reports do


def rows(path) -> list[dict[str, str]]:
    """The rows of a result CSV."""
    with Path(path).open(newline="") as fh:
        return list(csv.DictReader(fh))


def column(found: Iterable[Mapping[str, str]], name: str) -> np.ndarray:
    """One column of the rows as floats."""
    return np.array([float(row[name]) for row in found], dtype=float)


def grouped(found: Iterable[Mapping[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    """The rows by the value of `key`, in order of first appearance."""
    groups: dict[str, list[dict[str, str]]] = {}
    for row in found:
        groups.setdefault(row[key], []).append(row)
    return groups


def summary(values: Sequence[float], zero: float = 0.0) -> dict[str, float]:
    """Best, median, mean, worst and standard deviation, with values below `zero` counted as 0."""
    values = np.asarray(values, dtype=float)
    values = np.where(values < zero, 0.0, values)
    return {
        "best": float(values.min()),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "worst": float(values.max()),
        "std": float(values.std()),
    }


def ranks(scores: np.ndarray) -> np.ndarray:
    """Average ranks (1 = lowest) along the last axis, ties sharing the mean of their ranks."""
    scores = np.asarray(scores, dtype=float)
    out = np.empty_like(scores)
    for index in np.ndindex(scores.shape[:-1]):
        row = scores[index]
        order = np.argsort(row, kind="stable")
        ranked = np.empty(len(row))
        ranked[order] = np.arange(1, len(row) + 1)
        for value in np.unique(row):
            tied = row == value
            ranked[tied] = ranked[tied].mean()
        out[index] = ranked
    return out


def number(value, digits: int = 4) -> str:
    """A number for a table: integers as they are, others to `digits` significant digits."""
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(value)
    value = float(value)
    if value == 0 or not np.isfinite(value):
        return "0" if value == 0 else str(value)
    if 1e-3 <= abs(value) < 1e6:
        return f"{value:.{digits}g}"
    return f"{value:.{digits - 1}e}"


def markdown(header: Sequence[str], body: Iterable[Sequence], digits: int = 4) -> str:
    """A markdown table."""
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(number(cell, digits) for cell in line) + " |" for line in body]
    return "\n".join(lines) + "\n"


def write(path, title: str, intro: str, *sections: str) -> Path:
    """A markdown report: a title, an introduction and its sections."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"# {title}\n\n{intro.strip()}\n\n" + "\n".join(section.strip() + "\n" for section in sections)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path
