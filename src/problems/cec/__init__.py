"""The CEC 2017, 2020, 2022 and 2024 bound-constrained suites, written from the organisers' definitions and data;
the CEC 2019 100-Digit Challenge is in `problems.cec.digits`.

    >>> from problems.functions import cec
    >>> suite = cec(2022, dim=20)
    >>> fn = suite["F12"]
    >>> fn.dim, fn.optimal, suite.rules.budget(20), suite.rules.runs
    (20, 2700.0, 1000000, 30)

Each function is a `problems.functions.Function` over [-100, 100]^D, so any optimizer runs it like a classic one
(`python -m benchmarks.run` runs NIO on a suite).
"""

from problems.cec.catalog import Editions, cec
from problems.cec.recipes import Composition, Hybrid, Member, Single
from problems.cec.suite import BOUND, DIMENSIONS, CECFunction, CECSuite, Entry

__all__ = [
    "BOUND",
    "DIMENSIONS",
    "CECFunction",
    "CECSuite",
    "Composition",
    "Editions",
    "Entry",
    "Hybrid",
    "Member",
    "Single",
    "cec",
]
