"""How a CEC function is built from basic functions: single, hybrid (permuted segments) or composition."""

from dataclasses import dataclass
from math import ceil
from typing import Literal

import numpy as np

from problems.cec.basic import BASIC, transform


@dataclass(frozen=True)
class Single:
    """One shifted, rotated basic function."""

    function: str

    shifts: int = 1
    rotations: int = 1
    shuffles: int = 0

    def __call__(self, x, shift, rotation, shuffle, y, shifted=True, rotated=True):
        return BASIC[self.function](x, shift[0], rotation[0], shifted, rotated, y)


@dataclass(frozen=True)
class Hybrid:
    """Shift and rotate, permute the coordinates, then score consecutive segments with different functions.

    Segment sizes are ceil(share · D); the remaining coordinates go to the last segment, or to the first when
    `remainder` is "first" (CEC 2020). `small` fixes the sizes for given dimensions, as CEC 2020 does for D = 5.
    """

    functions: tuple[str, ...]
    shares: tuple[float, ...]
    remainder: Literal["first", "last"] = "last"
    small: tuple[tuple[int, tuple[int, ...]], ...] = ()

    shifts: int = 1
    rotations: int = 1
    shuffles: int = 1

    def sizes(self, n: int) -> list[int]:
        if fixed := dict(self.small).get(n):
            return list(fixed)
        if self.remainder == "last":
            sizes = [ceil(share * n) for share in self.shares[:-1]]
            return [*sizes, n - sum(sizes)]
        sizes = [ceil(share * n) for share in self.shares[1:]]
        return [n - sum(sizes), *sizes]

    def __call__(self, x, shift, rotation, shuffle, y, shifted=True, rotated=True):
        n = x.shape[1]
        z = transform(x, shift[0], rotation[0], 1.0, shifted, rotated, y)
        y[:, :n] = z[:, shuffle[0]]
        total, start = np.zeros(len(x)), 0
        for name, size in zip(self.functions, self.sizes(n), strict=True):
            total = total + BASIC[name](y[:, start : start + size], shift[0], None, False, False, y)
            start += size
        return total


@dataclass(frozen=True)
class Member:
    """One part of a composition: its recipe, spread `sigma`, `bias` and the reference's `scale` / `divisor`."""

    recipe: Single | Hybrid
    sigma: float
    bias: float
    scale: float = 1.0
    divisor: float = 1.0
    rotated: bool = True


@dataclass(frozen=True)
class Composition:
    """Members weighted by closeness to their own optima, the CEC cf_cal rule."""

    members: tuple[Member, ...]

    @property
    def shifts(self) -> int:
        return len(self.members)

    @property
    def rotations(self) -> int:
        return len(self.members)

    @property
    def shuffles(self) -> int:
        return len(self.members) if any(isinstance(member.recipe, Hybrid) for member in self.members) else 0

    def __call__(self, x, shift, rotation, shuffle, y, shifted=True, rotated=True):
        n = x.shape[1]
        fits, weights = [], []
        for index, member in enumerate(self.members):
            part = member.recipe(
                x,
                shift[index : index + 1],
                rotation[index : index + 1],
                shuffle[index : index + 1] if len(shuffle) else shuffle,
                y,
                True,
                rotated and member.rotated,
            )
            fits.append(member.scale * part / member.divisor + member.bias)
            distance = np.sum((x - shift[index]) ** 2.0, axis=1)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
                weight = (1.0 / distance) ** 0.5 * np.exp(-distance / 2.0 / n / member.sigma**2.0)
            weights.append(np.where(distance != 0, weight, 1.0e99))
        weights, fits = np.array(weights), np.array(fits)
        uniform = np.max(weights, axis=0) == 0
        weights = np.where(uniform, 1.0, weights)
        return np.sum(weights / np.sum(weights, axis=0) * fits, axis=0)
