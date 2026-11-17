"""The CEC 2017, 2020, 2022 and 2024 single-objective bound-constrained suites, as the reference dispatchers
define them.

`cec` reaches them by year: `cec(2022)` is the CEC 2022 suite at its default dimension, `cec(2022, dim=20)` at 20,
and `cec.rules(2022)` the competition's rules alone. CEC 2024 is CEC 2017 without F2 at D = 30, and its functions keep
their CEC 2017 numbers.
"""

from problems.cec.recipes import Composition, Hybrid, Member, Single
from problems.cec.suite import CECSuite, Entry
from problems.suite import Suite


def _m(
    function: str | Hybrid, sigma: float, bias: float, scale: float = 1.0, divisor: float = 1.0, rotated: bool = True
) -> Member:
    recipe = Single(function) if isinstance(function, str) else function
    return Member(recipe, sigma, bias, scale, divisor, rotated)


# hybrid functions of CEC 2017; CEC 2020 and 2022 define some of their own below
H17 = {
    1: Hybrid(("zakharov", "rosenbrock", "rastrigin"), (0.2, 0.4, 0.4)),
    2: Hybrid(("ellips", "schwefel", "bent_cigar"), (0.3, 0.3, 0.4)),
    3: Hybrid(("bent_cigar", "rosenbrock", "bi_rastrigin"), (0.3, 0.3, 0.4)),
    4: Hybrid(("ellips", "ackley", "schaffer_f7", "rastrigin"), (0.2, 0.2, 0.2, 0.4)),
    5: Hybrid(("bent_cigar", "hgbat", "rastrigin", "rosenbrock"), (0.2, 0.2, 0.3, 0.3)),
    6: Hybrid(("escaffer6", "hgbat", "rosenbrock", "schwefel"), (0.2, 0.2, 0.3, 0.3)),
    7: Hybrid(("katsuura", "ackley", "grie_rosen", "schwefel", "rastrigin"), (0.1, 0.2, 0.2, 0.2, 0.3)),
    8: Hybrid(("ellips", "ackley", "rastrigin", "hgbat", "discus"), (0.2,) * 5),
    9: Hybrid(("bent_cigar", "rastrigin", "grie_rosen", "weierstrass", "escaffer6"), (0.2,) * 5),
    10: Hybrid(("hgbat", "katsuura", "ackley", "rastrigin", "schwefel", "schaffer_f7"), (0.1, 0.1, 0.2, 0.2, 0.2, 0.2)),
}

C17 = {
    1: Composition((_m("rosenbrock", 10, 0), _m("ellips", 20, 100, 10000, 1e10), _m("rastrigin", 30, 200))),
    2: Composition((_m("rastrigin", 10, 0), _m("griewank", 20, 100, 1000, 100), _m("schwefel", 30, 200))),
    3: Composition(
        (_m("rosenbrock", 10, 0), _m("ackley", 20, 100, 1000, 100), _m("schwefel", 30, 200), _m("rastrigin", 40, 300))
    ),
    4: Composition(
        (
            _m("ackley", 10, 0, 1000, 100),
            _m("ellips", 20, 100, 10000, 1e10),
            _m("griewank", 30, 200, 1000, 100),
            _m("rastrigin", 40, 300),
        )
    ),
    5: Composition(
        (
            _m("rastrigin", 10, 0, 10000, 1e3),
            _m("happycat", 20, 100, 1000, 1e3),
            _m("ackley", 30, 200, 1000, 100),
            _m("discus", 40, 300, 10000, 1e10),
            _m("rosenbrock", 50, 400),
        )
    ),
    6: Composition(
        (
            _m("escaffer6", 10, 0, 10000, 2e7),
            _m("schwefel", 20, 100),
            _m("griewank", 20, 200, 1000, 100),
            _m("rosenbrock", 30, 300),
            _m("rastrigin", 40, 400, 10000, 1e3),
        )
    ),
    7: Composition(
        (
            _m("hgbat", 10, 0, 10000, 1000),
            _m("rastrigin", 20, 100, 10000, 1e3),
            _m("schwefel", 30, 200, 10000, 4e3),
            _m("bent_cigar", 40, 300, 10000, 1e30),
            _m("ellips", 50, 400, 10000, 1e10),
            _m("escaffer6", 60, 500, 10000, 2e7),
        )
    ),
    8: Composition(
        (
            _m("ackley", 10, 0, 1000, 100),
            _m("griewank", 20, 100, 1000, 100),
            _m("discus", 30, 200, 10000, 1e10),
            _m("rosenbrock", 40, 300),
            _m("happycat", 50, 400, 1000, 1e3),
            _m("escaffer6", 60, 500, 10000, 2e7),
        )
    ),
    9: Composition((_m(H17[5], 10, 0), _m(H17[6], 30, 100), _m(H17[7], 50, 200))),
    10: Composition((_m(H17[5], 10, 0), _m(H17[8], 30, 100), _m(H17[9], 50, 200))),
}

NAMES17 = {
    1: "Shifted and Rotated Bent Cigar",
    2: "Shifted and Rotated Sum of Different Power",
    3: "Shifted and Rotated Zakharov",
    4: "Shifted and Rotated Rosenbrock",
    5: "Shifted and Rotated Rastrigin",
    6: "Shifted and Rotated Expanded Schaffer F6",
    7: "Shifted and Rotated Lunacek Bi-Rastrigin",
    8: "Shifted and Rotated Non-Continuous Rastrigin",
    9: "Shifted and Rotated Levy",
    10: "Shifted and Rotated Schwefel",
}
SINGLE17 = {1: "bent_cigar", 2: "sum_diff_pow", 3: "zakharov", 4: "rosenbrock", 5: "rastrigin"}
SINGLE17 |= {6: "schaffer_f7", 7: "bi_rastrigin", 8: "step_rastrigin", 9: "levy", 10: "schwefel"}

CEC2017 = {number: Entry(NAMES17[number], Single(SINGLE17[number]), 100.0 * number, number) for number in range(1, 11)}
CEC2017 |= {10 + k: Entry(f"Hybrid Function {k}", H17[k], 100.0 * (10 + k), 10 + k) for k in range(1, 11)}
CEC2017 |= {20 + k: Entry(f"Composition Function {k}", C17[k], 100.0 * (20 + k), 20 + k) for k in range(1, 11)}

CEC2020 = {
    1: Entry("Shifted and Rotated Bent Cigar", Single("bent_cigar"), 100.0, 1),
    2: Entry("Shifted and Rotated Schwefel", Single("schwefel"), 1100.0, 2),
    3: Entry("Shifted and Rotated Lunacek Bi-Rastrigin", Single("bi_rastrigin"), 700.0, 3),
    4: Entry("Expanded Rosenbrock plus Griewank", Single("grie_rosen"), 1900.0, 7),
    5: Entry("Hybrid Function 1", Hybrid(("schwefel", "rastrigin", "ellips"), (0.3, 0.3, 0.4), "first"), 1700.0, 4),
    6: Entry(
        "Hybrid Function 2",
        Hybrid(("escaffer6", "hgbat", "rosenbrock", "schwefel"), (0.2, 0.2, 0.3, 0.3), "first", ((5, (1, 1, 1, 2)),)),
        1600.0,
        16,
    ),
    7: Entry(
        "Hybrid Function 3",
        Hybrid(
            ("escaffer6", "hgbat", "rosenbrock", "schwefel", "ellips"),
            (0.1, 0.2, 0.2, 0.2, 0.3),
            "first",
            ((5, (1, 1, 1, 1, 1)),),
        ),
        2100.0,
        6,
    ),
    8: Entry("Composition Function 1", C17[2], 2200.0, 22),
    9: Entry("Composition Function 2", C17[4], 2400.0, 24),
    10: Entry("Composition Function 3", C17[5], 2500.0, 25),
}

CEC2022 = {
    1: Entry("Shifted and Full Rotated Zakharov", Single("zakharov"), 300.0, 1),
    2: Entry("Shifted and Full Rotated Rosenbrock", Single("rosenbrock"), 400.0, 2),
    3: Entry("Shifted and Full Rotated Expanded Schaffer F6", Single("schaffer_f7"), 600.0, 3),
    4: Entry("Shifted and Full Rotated Non-Continuous Rastrigin", Single("step_rastrigin"), 800.0, 4),
    5: Entry("Shifted and Full Rotated Levy", Single("levy_2022"), 900.0, 5),
    6: Entry("Hybrid Function 1", Hybrid(("bent_cigar", "hgbat", "rastrigin"), (0.4, 0.4, 0.2)), 1800.0, 6),
    7: Entry(
        "Hybrid Function 2",
        Hybrid(("hgbat", "katsuura", "ackley", "rastrigin", "schwefel", "schaffer_f7"), (0.1, 0.2, 0.2, 0.2, 0.1, 0.2)),
        2000.0,
        7,
    ),
    8: Entry(
        "Hybrid Function 3",
        Hybrid(("katsuura", "happycat", "grie_rosen", "schwefel", "ackley"), (0.3, 0.2, 0.2, 0.1, 0.2)),
        2200.0,
        8,
    ),
    9: Entry(
        "Composition Function 1",
        Composition(
            (
                _m("rosenbrock", 10, 0, 10000, 1e4),
                _m("ellips", 20, 200, 10000, 1e10),
                _m("bent_cigar", 30, 300, 10000, 1e30),
                _m("discus", 40, 100, 10000, 1e10),
                _m("ellips", 50, 400, 10000, 1e10, rotated=False),
            )
        ),
        2300.0,
        9,
    ),
    10: Entry(
        "Composition Function 2",
        Composition((_m("schwefel", 20, 0, rotated=False), _m("rastrigin", 10, 200), _m("hgbat", 10, 100))),
        2400.0,
        10,
    ),
    11: Entry(
        "Composition Function 3",
        Composition(
            (
                _m("escaffer6", 20, 0, 10000, 2e7),
                _m("schwefel", 20, 200),
                _m("griewank", 30, 300, 1000, 100),
                _m("rosenbrock", 30, 400),
                _m("rastrigin", 20, 200, 10000, 1e3),
            )
        ),
        2600.0,
        11,
    ),
    12: Entry(
        "Composition Function 4",
        Composition(
            (
                _m("hgbat", 10, 0, 10000, 1000),
                _m("rastrigin", 20, 300, 10000, 1e3),
                _m("schwefel", 30, 500, 10000, 4e3),
                _m("bent_cigar", 40, 100, 10000, 1e30),
                _m("ellips", 50, 400, 10000, 1e10),
                _m("escaffer6", 60, 200, 10000, 2e7),
            )
        ),
        2700.0,
        12,
    ),
}

CEC2024 = {number: entry for number, entry in CEC2017.items() if number != 2}


CEC2017_SUITE = CECSuite(
    2017,
    CEC2017,
    runs=51,
    official=(10, 30, 50, 100),
    budgets=None,
    unimodal=frozenset({1, 2, 3}),
    not_at_two=frozenset({17, 18, 19, 20, 21, 22, 29, 30}),
    source="Awad, Ali, Liang, Qu and Suganthan, Problem definitions and evaluation criteria for the CEC 2017 special "
    "session and competition on single objective real-parameter numerical optimization, technical report, 2016",
)
CEC2020_SUITE = CECSuite(
    2020,
    CEC2020,
    runs=30,
    official=(5, 10, 15, 20),
    budgets={5: 50_000, 10: 1_000_000, 15: 3_000_000, 20: 10_000_000},
    unimodal=frozenset({1}),
    not_at_two=frozenset({5, 6, 7}),
    source="Yue et al., Problem definitions and evaluation criteria for the CEC 2020 special session and competition "
    "on single objective bound constrained numerical optimization, technical report, 2019",
    # F7 at D = 5 splits into one-coordinate segments, and its ellipsoid segment computes 10^(0/0)
    broken=frozenset({(7, 5)}),
)
CEC2022_SUITE = CECSuite(
    2022,
    CEC2022,
    runs=30,
    official=(10, 20),
    budgets={10: 200_000, 20: 1_000_000},
    unimodal=frozenset({1}),
    not_at_two=frozenset({6, 7, 8}),
    source="Kumar, Price, Mohamed, Hadi and Suganthan, Problem definitions and evaluation criteria for the 2022 "
    "special session and competition on single objective bounded optimization, technical report, 2021",
)
CEC2024_SUITE = CECSuite(
    2024,
    CEC2024,
    runs=25,
    official=(30,),
    budgets=None,
    unimodal=frozenset({1, 3}),
    not_at_two=frozenset({17, 18, 19, 20, 21, 22, 29, 30}),
    source="CEC 2024 competition on single objective bound constrained numerical optimization "
    "(github.com/P-N-Suganthan/2024-CEC): the CEC 2017 functions without F2 at D = 30, 25 runs, ranked by the U-score",
    archive=2017,
    default=30,
)


class Editions:
    """The CEC competitions by year. Calling it builds a competition's functions as a `Suite` whose `rules` is the
    competition (its budgets, runs and groups); `rules(year)` gives the competition without building anything."""

    def __init__(self, *competitions: CECSuite):
        self._by_year = {competition.year: competition for competition in competitions}

    @property
    def editions(self) -> tuple[int, ...]:
        return tuple(self._by_year)

    def rules(self, year: int) -> CECSuite:
        try:
            return self._by_year[year]
        except KeyError:
            if year == 2019:
                raise ValueError("CEC 2019 is the 100-Digit Challenge: use `digits` from problems.functions.") from None
            years = ", ".join(map(str, self.editions))
            raise ValueError(f"There is no CEC {year} suite; the editions are {years}.") from None

    def __call__(self, year: int, dim: int | None = None) -> Suite:
        """The functions of the CEC `year` competition defined at `dim` dimensions (its default if None)."""
        rules = self.rules(year)
        return Suite(rules.name, rules.functions(dim).values(), rules=rules)

    def __repr__(self) -> str:
        return f"cec: editions {self.editions}"


cec = Editions(CEC2017_SUITE, CEC2020_SUITE, CEC2022_SUITE, CEC2024_SUITE)
