"""A CEC competition as one object: its functions, the dimensions the official data defines them at, and the rules
its report follows (runs, reported dimensions and evaluation budgets)."""

from dataclasses import dataclass
from functools import cache, partial

import numpy as np

from problems.cec import data
from problems.cec.recipes import Composition, Hybrid, Single
from problems.function import Function

BOUND = 100.0  # every CEC function is searched over [-100, 100]^D
DIMENSIONS = 10  # every function of every suite is defined there


@dataclass(frozen=True)
class Entry:
    """One function of a competition: its name, its recipe, its optimum value (`bias`) and the number of its official
    input files (`data`; CEC 2020 reuses the CEC 2017 and 2014 numbering)."""

    name: str
    recipe: Single | Hybrid | Composition
    bias: float
    data: int


@dataclass(frozen=True, eq=False)
class CECFunction:
    """One CEC function at one dimension with its official data, vectorised over leading axes.

    Calling it on one position returns a float, and `batch` scores the rows of a population in one call.
    """

    entry: Entry
    shift: np.ndarray  # (shift vectors × D)
    rotation: np.ndarray  # (matrices × D × D)
    shuffle: np.ndarray  # (permutations × D), possibly empty

    def batch(self, population: np.ndarray) -> np.ndarray:
        """The value of every row of `population`."""
        x = np.asarray(population, dtype=float)
        flat = x.reshape(-1, x.shape[-1])
        values = self.entry.recipe(flat, self.shift, self.rotation, self.shuffle, np.zeros_like(flat))
        return (values + self.entry.bias).reshape(x.shape[:-1])

    def __call__(self, x: np.ndarray) -> float:
        return float(self.batch(np.asarray(x, dtype=float)[None])[0])


@dataclass(frozen=True, eq=False)
class CECSuite:
    """One CEC bound-constrained competition, with the rules its report follows.

    `functions(dimensions)` gives every function defined at that dimension as a `problems.functions.Function`, keyed
    "F1", "F2", ..., over [-100, 100]^D with the function's bias as its optimum; `budget(dimensions)` is the
    competition's evaluation budget and `runs` the number of independent runs it asks for.
    """

    year: int
    entries: dict[int, Entry]
    runs: int
    official: tuple[int, ...]  # the dimensions the competition reports
    budgets: dict[int, int] | None  # the evaluation budget at each reported dimension; None means 10 000 · D
    unimodal: frozenset[
        int
    ]  # single functions the report groups as unimodal; its other single functions are multimodal
    not_at_two: frozenset[int]  # functions the organisers leave undefined at D = 2
    source: str  # the competition's technical report
    broken: frozenset[tuple[int, int]] = frozenset()  # (function, dimension) where the reference returns NaN everywhere
    archive: int | None = None  # the year of the data files, when the competition reuses an earlier suite's
    default: int = DIMENSIONS  # the dimension `functions()` takes

    @property
    def name(self) -> str:
        return f"CEC {self.year}"

    @property
    def files(self) -> int:
        """The year of the official data files this suite reads."""
        return self.year if self.archive is None else self.archive

    @property
    def numbers(self) -> tuple[int, ...]:
        return tuple(self.entries)

    def budget(self, dimensions: int) -> int:
        """The competition's maximum number of function evaluations at `dimensions`."""
        if self.budgets is None:
            return 10_000 * dimensions
        try:
            return self.budgets[dimensions]
        except KeyError:
            raise ValueError(f"{self.name} sets budgets for D in {tuple(self.budgets)}, not {dimensions}.") from None

    def group(self, number: int) -> str:
        """ "unimodal", "multimodal", "hybrid" or "composition"."""
        recipe = self._entry(number).recipe
        if isinstance(recipe, Hybrid):
            return "hybrid"
        if isinstance(recipe, Composition):
            return "composition"
        return "unimodal" if number in self.unimodal else "multimodal"

    def available(self, number: int) -> tuple[int, ...]:
        """The dimensions at which the official data defines function `number`."""
        entry = self._entry(number)
        dimensions = data.dimensions(self.files, entry.data)
        if entry.recipe.shuffles:
            dimensions = tuple(d for d in dimensions if data.has(self.files, f"shuffle_{entry.data}_D{d}"))
        return tuple(
            d for d in dimensions if not (d == 2 and number in self.not_at_two) and (number, d) not in self.broken
        )

    def get(self, number: int, dimensions: int | None = None) -> Function:
        """Function `number` at `dimensions` (the suite's default if omitted)."""
        dimensions = self.default if dimensions is None else dimensions
        allowed = self.available(number)
        if dimensions not in allowed:
            raise ValueError(f"{self.name} F{number} is defined for D in {allowed}, not {dimensions}.")
        return _build(self, number, dimensions)

    def functions(self, dimensions: int | None = None) -> dict[str, Function]:
        """Every function defined at `dimensions` (the suite's default if omitted), keyed "F<number>"."""
        dimensions = self.default if dimensions is None else dimensions
        return {f"F{n}": self.get(n, dimensions) for n in self.numbers if dimensions in self.available(n)}

    def minimiser(self, number: int, dimensions: int | None = None) -> np.ndarray:
        """Where function `number` takes its optimum: its first shift vector (CEC 2017 F9's Levy sits just beside)."""
        return self.get(number, dimensions).solve.shift[0].copy()

    def _entry(self, number: int) -> Entry:
        try:
            return self.entries[number]
        except KeyError:
            raise ValueError(
                f"{self.name} has functions {self.numbers[0]} to {self.numbers[-1]}, not {number}."
            ) from None


@cache
def _build(suite: CECSuite, number: int, dimensions: int) -> Function:
    entry, year = suite.entries[number], suite.files
    recipe = entry.recipe
    function = CECFunction(
        entry,
        data.shifts(year, entry.data, dimensions, recipe.shifts),
        data.rotations(year, entry.data, dimensions, recipe.rotations),
        data.shuffles(year, entry.data, dimensions, recipe.shuffles)
        if recipe.shuffles
        else np.empty((0, dimensions), dtype=np.intp),
    )
    resize = partial(suite.get, number)  # the same function at another dimension the data defines
    return Function(f"F{number}", entry.name, function, -BOUND, BOUND, dimensions, entry.bias, resize=resize)


__all__ = ["BOUND", "DIMENSIONS", "CECFunction", "CECSuite", "Entry"]
