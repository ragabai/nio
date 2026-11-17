"""The CEC 2020 real-world constrained suite: 57 problems from chemical processes to livestock feed rations.

Ported from the organisers' MATLAB code (Kumar et al., Swarm and Evolutionary Computation, 2020) and checked
against an independent C++ port of it. Bounds, best known values and evaluation budgets are the competition's.

    problem = realworld.get("RC15")
    result = NIO(Penalized(problem), problem.space, agents=30, evaluations=problem.budget, seed=0).solve()
    problem.violation(result.position)   # the competition's mean constraint violation; 0 is feasible
"""

from problems.engineering.realworld import chemical, electronics, livestock, mechanical, power, process
from problems.engineering.realworld.problem import TOLERANCE, RealWorld


def _full(d: int, value: float) -> list[float]:
    return [value] * d


def _bounds() -> dict[int, tuple[list[float], list[float]]]:
    """Cal_par.m: the search range of every problem."""
    b = {
        1: ([0, 0, 0, 0, 1000, 0, 100, 100, 100], [10, 200, 100, 200, 2000000, 600, 600, 600, 900]),
        2: (
            [1e4, 1e4, 1e4, 0, 0, 0, 100, 100, 100, 100, 100],
            [0.819e6, 1.131e6, 2.05e6, 0.05074, 0.05074, 0.05074, 200, 300, 300, 300, 400],
        ),
        3: ([1000, 0, 2000, 0, 0, 0, 0], [2000, 100, 4000, 100, 100, 20, 200]),
        4: ([0, 0, 0, 0, 1e-5, 1e-5], [1, 1, 1, 1, 16, 16]),
        5: (_full(9, 0), [100, 200, 100, 100, 100, 100, 200, 100, 200]),
        6: (
            _full(38, 0),
            [
                *[90, 150, 90, 150, 90, 90, 150, 90, 90, 90, 150, 150, 90, 90, 150, 90, 150, 90, 150, 90],
                *[1, 1.2, 1, 1, 1, 0.5, 1, 1, 0.5, 0.5, 0.5, 1.2, 0.5, 1.2, 1.2, 0.5, 1.2, 1.2],
            ],
        ),
        8: ([0, -0.51], [1.6, 1.49]),
        9: ([0.5, 0.5, -0.51], [1.4, 1.4, 1.49]),
        10: ([0.2, -2.22554, -0.51], [1, -1, 1.49]),
        11: ([0, 0, 0, 0, -0.51, -0.51, 0], [20, 20, 10, 10, 1.49, 1.49, 40]),
        12: ([0, 0, 0, -0.51, -0.51, -0.51, -0.51], [100, 100, 100, 1.49, 1.49, 1.49, 1.49]),
        13: ([27, 27, 27, 77.51, 32.51], [45, 45, 45, 102.49, 45.49]),
        14: ([0.51, 0.51, 0.51, 250, 250, 250, 6, 4, 40, 10], [3.49, 3.49, 3.49, 2500, 2500, 2500, 20, 16, 700, 450]),
        15: ([2.6, 0.7, 17, 7.3, 7.3, 2.9, 5], [3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5]),
        16: (_full(14, 0.001), _full(14, 5)),
        17: ([0.05, 0.25, 2.00], [2, 1.3, 15.0]),
        18: ([0.51, 0.51, 10, 10], [99.49, 99.49, 200, 200]),
        19: ([0.125, 0.1, 0.1, 0.1], [2, 10, 10, 2]),
        20: (_full(2, 0), _full(2, 1)),
        21: ([60, 90, 1, 0, 2], [80, 110, 3, 1000, 9]),
        22: (
            [16.51, 13.51, 13.51, 16.51, 13.51, 47.51, 0.51, 0.51, 0.51],
            [96.49, 54.49, 51.49, 46.49, 51.49, 124.49, 3.49, 6.49, 6.49],
        ),
        23: ([0, 0, 0, 0, 0], [60, 60, 90, 90, 90]),
        24: ([10, 10, 100, 0, 10, 100, 1], [150, 150, 200, 50, 150, 300, 3.14]),
        25: ([1, 1, 1e-6, 1], [16, 16, 16e-6, 16]),
        26: ([6.51] * 8 + [0.51] * 14, [76.49] * 8 + [4.49] * 4 + [9.49] * 10),
        27: (_full(10, 0.645e-4), _full(10, 50e-4)),
        28: (
            [125, 10.5, 4.51, 0.515, 0.515, 0.4, 0.6, 0.3, 0.02, 0.6],
            [150, 31.5, 50.49, 0.6, 0.6, 0.5, 0.7, 0.4, 0.1, 0.85],
        ),
        29: ([20, 1, 20, 0.1], [50, 10, 50, 60]),
        30: ([0.51, 0.6, 0.51], [70.49, 3, 42.49]),
        31: (_full(4, 12), _full(4, 60)),
        32: ([78, 33, 27, 27, 27], [102, 45, 45, 45, 45]),
        33: (_full(30, 0.001), _full(30, 1)),
        34: (_full(118, -1), _full(118, 1)),
        35: (_full(153, -1), _full(153, 1)),
        36: (_full(158, -1), _full(158, 1)),
        41: (_full(74, -1), _full(74, 1)),
        44: (_full(30, 40), _full(30, 1960)),
        **{k: (_full(25, 0), _full(25, 90)) for k in (45, 46, 47)},
        **{k: (_full(30, 0), _full(30, 90)) for k in (48, 49, 50)},
        **{k: (_full(59, 0), _full(59, 10)) for k in (51, 52, 53, 54)},
        **{k: (_full(64, 0), _full(64, 10)) for k in (55, 56, 57)},
    }
    lb7, ub7 = _full(48, 0), _full(48, 1)
    for i in (24, 26, 28, 31):
        lb7[i - 1] = 0.849999
    ub7[3] = 140
    for i in (25, 27, 32, 35, 37, 29):
        ub7[i - 1] = 30
    for i in (2, 3, 5, 13, 14, 15):
        ub7[i - 1] = 90
    for i in (1, 6, 7, 8, 9, 10, 11, 12, 16, 17, 18, 19, 20):
        ub7[i - 1] = 35
    b[7] = (lb7, ub7)
    for k in (37, 38, 39):
        lb, ub = _full(126, -1), _full(126, 1)
        lb[116:126] = [0] * 10
        b[k] = (lb, ub)
    lb, ub = _full(76, -1), _full(76, 1)
    lb[74:76], ub[74:76] = [0, 0], [2, 2]
    b[40] = (lb, ub)
    for k in (42, 43):
        lb, ub = _full(86, -1), _full(86, 1)
        lb[74:86], ub[74:76], ub[76:86] = [0] * 12, [2, 2], [500] * 10
        b[k] = (lb, ub)
    return b


CHEMICAL, PROCESS, MECHANICAL = (
    "Industrial chemical processes",
    "Process synthesis and design",
    "Mechanical engineering",
)
POWER, ELECTRONICS, LIVESTOCK = "Power systems", "Power electronics", "Livestock feed ration"

# (name, category, best known f, inequality constraints, equality constraints), from the competition's guidelines;
# constraint counts are what the code returns (RC17 and RC21 return one more inequality than the table lists,
# RC31 none, RC44 91 rather than 105)
TABLE = {
    1: ("Heat exchanger network design (case 1)", CHEMICAL, 1.8931162966e02, 0, 8),
    2: ("Heat exchanger network design (case 2)", CHEMICAL, 7.0490369540e03, 0, 9),
    3: ("Optimal operation of alkylation unit", CHEMICAL, -4.5291197395e03, 14, 0),
    4: ("Reactor network design", CHEMICAL, -3.8826043623e-01, 1, 4),
    5: ("Haverly's pooling problem", CHEMICAL, -4.0000560000e02, 2, 4),
    6: ("Blending-pooling-separation problem", CHEMICAL, 1.8638304088e00, 0, 32),
    7: ("Propane, isobutane, n-butane nonsharp separation", CHEMICAL, 1.5670451000e00, 0, 38),
    8: ("Process synthesis problem", PROCESS, 2.0000000000e00, 2, 0),
    9: ("Process synthesis and design problem", PROCESS, 2.5576545740e00, 1, 1),
    10: ("Process flow sheeting problem", PROCESS, 1.0765430833e00, 3, 0),
    11: ("Two-reactor problem", PROCESS, 9.9238463653e01, 4, 4),
    12: ("Process synthesis problem", PROCESS, 2.9248305537e00, 9, 0),
    13: ("Process design problem", PROCESS, 2.6887000000e04, 3, 0),
    14: ("Multi-product batch plant", PROCESS, 5.3638942722e04, 10, 0),
    15: ("Weight minimisation of a speed reducer", MECHANICAL, 2.9944244658e03, 11, 0),
    16: ("Optimal design of an industrial refrigeration system", MECHANICAL, 3.2213000814e-02, 15, 0),
    17: ("Tension/compression spring design (case 1)", MECHANICAL, 1.2665232788e-02, 4, 0),
    18: ("Pressure vessel design", MECHANICAL, 5.8853327736e03, 4, 0),
    19: ("Welded beam design", MECHANICAL, 1.6702177263e00, 5, 0),
    20: ("Three-bar truss design", MECHANICAL, 2.6389584338e02, 3, 0),
    21: ("Multiple disk clutch brake design", MECHANICAL, 2.3524245790e-01, 8, 0),
    22: ("Planetary gear train design", MECHANICAL, 5.2576870748e-01, 10, 1),
    23: ("Step-cone pulley design", MECHANICAL, 1.6069868725e01, 8, 3),
    24: ("Robot gripper design", MECHANICAL, 2.5287918415e00, 7, 0),
    25: ("Hydrostatic thrust bearing design", MECHANICAL, 1.6161197651e03, 7, 0),
    26: ("Four-stage gear box design", MECHANICAL, 3.5359231973e01, 86, 0),
    27: ("Ten-bar truss design", MECHANICAL, 5.2445076066e02, 3, 0),
    28: ("Rolling element bearing design", MECHANICAL, 1.4614135715e04, 9, 0),
    29: ("Gas transmission compressor design", MECHANICAL, 2.9648954173e06, 1, 0),
    30: ("Tension/compression spring design (case 2)", MECHANICAL, 2.6138840583e00, 8, 0),
    31: ("Gear train design", MECHANICAL, 0.0, 0, 0),
    32: ("Himmelblau's function", MECHANICAL, -3.0665538672e04, 6, 0),
    33: ("Topology optimisation", MECHANICAL, 2.6393464970e00, 30, 0),
    34: ("Single-phase distributed generation sizing for phase balancing", POWER, 0.0, 0, 108),
    35: ("Distributed generation sizing for active power loss", POWER, 7.9963854000e-02, 0, 148),
    36: ("Distributed generation and capacitor sizing for reactive power loss", POWER, 4.7733529000e-02, 0, 148),
    37: ("Optimal power flow: active power loss", POWER, 1.8593563000e-02, 0, 116),
    38: ("Optimal power flow: fuel cost", POWER, 2.7139366000e00, 0, 116),
    39: ("Optimal power flow: active power loss and fuel cost", POWER, 2.7515909000e00, 0, 116),
    40: ("Microgrid power flow, islanded", POWER, 0.0, 0, 76),
    41: ("Microgrid power flow, grid-connected", POWER, 0.0, 0, 74),
    42: ("Droop settings for active power loss in an islanded microgrid", POWER, 7.7027102000e-02, 0, 76),
    43: ("Droop settings for reactive power loss in an islanded microgrid", POWER, 7.9835970000e-02, 0, 76),
    44: ("Wind farm layout", POWER, -6.2731715000e03, 91, 0),
    45: ("SOPWM for 3-level inverters", ELECTRONICS, 3.0739360000e-02, 24, 1),
    46: ("SOPWM for 5-level inverters", ELECTRONICS, 2.0240335000e-02, 24, 1),
    47: ("SOPWM for 7-level inverters", ELECTRONICS, 1.2783068000e-02, 24, 1),
    48: ("SOPWM for 9-level inverters", ELECTRONICS, 1.6787535766e-02, 29, 1),
    49: ("SOPWM for 11-level inverters", ELECTRONICS, 9.3118741800e-03, 29, 1),
    50: ("SOPWM for 13-level inverters", ELECTRONICS, 1.5051470000e-02, 29, 1),
    51: ("Beef cattle ration (case 1)", LIVESTOCK, 4.5508511497e03, 14, 1),
    52: ("Beef cattle ration (case 2)", LIVESTOCK, 3.3489821493e03, 14, 1),
    53: ("Beef cattle ration (case 3)", LIVESTOCK, 4.9976069290e03, 14, 1),
    54: ("Beef cattle ration (case 4)", LIVESTOCK, 4.2405482538e03, 14, 1),
    55: ("Dairy cattle ration (case 1)", LIVESTOCK, 6.6964145128e03, 0, 6),
    56: ("Dairy cattle ration (case 2)", LIVESTOCK, 1.4746580000e04, 0, 6),
    57: ("Dairy cattle ration (case 3)", LIVESTOCK, 3.2132917019e03, 0, 6),
}

MODULES = (chemical, process, mechanical, power, electronics, livestock)


def _problem(number: int, bounds: tuple[list[float], list[float]]) -> RealWorld:
    name, category, best, inequalities, equalities = TABLE[number]
    function = next(getattr(module, f"rc{number:02d}") for module in MODULES if hasattr(module, f"rc{number:02d}"))
    return RealWorld(
        f"RC{number:02d}",
        name,
        category,
        function,
        tuple(map(float, bounds[0])),
        tuple(map(float, bounds[1])),
        best,
        inequalities,
        equalities,
    )


PROBLEMS: dict[str, RealWorld] = {
    problem.code: problem for problem in (_problem(k, bounds) for k, bounds in sorted(_bounds().items()))
}
CATEGORIES: dict[str, tuple[str, ...]] = {
    category: tuple(code for code, problem in PROBLEMS.items() if problem.category == category)
    for category in (CHEMICAL, PROCESS, MECHANICAL, POWER, ELECTRONICS, LIVESTOCK)
}


def get(code: str) -> RealWorld:
    """A problem by code, such as "RC15" or "rc15"."""
    try:
        return PROBLEMS[code.strip().upper()]
    except KeyError:
        raise ValueError(f"Unknown problem {code!r}. Choose from RC01 to RC57.") from None


__all__ = ["CATEGORIES", "PROBLEMS", "TOLERANCE", "RealWorld", "get"]
