"""`Suite`: an ordered, read-only collection of `Function`s, the one shape every family of test functions takes.

>>> classic["F5"] is classic.F5 is classic.rosenbrock
True
>>> [f.key for f in classic][:3]
['F1', 'F2', 'F3']
>>> classic.at(dim=50)  # every scalable function at 50 dimensions, the fixed ones as they are
"""

from collections.abc import Iterable, Iterator, Mapping
from typing import TYPE_CHECKING

from problems.function import Bound, Function

if TYPE_CHECKING:
    from problems.cec.suite import CECSuite


class Suite:
    """Functions keyed "F1", "F2", ..., also reachable as attributes by key (`suite.F5`) or by name
    (`suite.rosenbrock`, from `names`, which maps a name to a key). Iterating yields the functions in order.

    `rules` is the CEC competition a CEC suite comes from (its budgets, runs and groups); None for the others.
    """

    __slots__ = ("_functions", "_names", "name", "rules")

    def __init__(
        self,
        name: str,
        functions: Iterable[Function],
        names: Mapping[str, str] | None = None,
        rules: "CECSuite | None" = None,
    ):
        self.name = name
        self.rules = rules
        self._functions = {f.key: f for f in functions}
        self._names = dict(names or {})

    def __getitem__(self, key: str) -> Function:
        try:
            return self._functions[key]
        except KeyError:
            raise KeyError(f"{self.name} has no function {key!r}; its keys are {', '.join(self._functions)}.") from None

    def __getattr__(self, name: str) -> Function:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._names:
            return self._functions[self._names[name]]
        if name in self._functions:
            return self._functions[name]
        known = ", ".join(self._names) or ", ".join(self._functions)
        raise AttributeError(f"{self.name} has no function {name!r}; try one of {known}.")

    def __dir__(self):
        """The names and keys too, so tab completion offers `classic.rosenbrock` and `classic.F5`."""
        return [*super().__dir__(), *self._names, *self._functions]

    def __iter__(self) -> Iterator[Function]:
        return iter(self._functions.values())

    def __contains__(self, key: object) -> bool:
        return key in self._functions

    def __len__(self) -> int:
        return len(self._functions)

    def __repr__(self) -> str:
        return f"Suite({self.name!r}, {len(self)} functions)"

    def keys(self):
        return self._functions.keys()

    def items(self):
        return self._functions.items()

    def at(self, dim: int | None = None, lb: Bound | None = None, ub: Bound | None = None) -> "Suite":
        """The suite at `dim` dimensions and over [`lb`, `ub`]. A fixed-dimension function keeps its own dimension;
        a CEC suite is rebuilt from its competition's data, without the functions it leaves undefined at `dim`."""
        if dim is None and lb is None and ub is None:
            return self
        if self.rules is not None:
            functions = self if dim is None else self.rules.functions(dim).values()
            return Suite(self.name, [f.at(lb=lb, ub=ub) for f in functions], self._names, self.rules)
        functions = [f.at(None if f.resize is None else dim, lb, ub) for f in self]
        return Suite(self.name, functions, self._names, self.rules)
