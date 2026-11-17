"""Run NIO on a benchmark suite and write one CSV row per (function, run).

Each run gets a fixed budget of objective evaluations (or a number of iterations); its score is the best value found,
and its error is that value minus the known optimum. Run `r` of a function uses the seed `--seed + r`, so a CSV can be
regenerated exactly.

Suites: `classic` (F1-F23), `fair` (F1-F7 and F9-F13 shifted away from the origin) and `fair23` (all 23 shifted);
see `problems.shifted`; `cec2019` is the CEC 2019 100-Digit Challenge at its fixed dimensions
(`problems.functions.digits`). The CEC suites `cec2017`, `cec2020`, `cec2022` and `cec2024` (`cec(year)`) run at
`--dim` dimensions, by default with their competition's evaluation budget and number of runs.

`--traces` also keeps every run: its improvement trace (evaluation, best value), keyed "<function>/<seed>"
(`benchmarks.runs.read_traces`), or "<algorithm>/<function>/<seed>" when several algorithms run.

`--algorithms NIO SCA WOA ...` adds NIO's rivals (`benchmarks.rivals`, from mealpy), which need `--agents` and run on
an evaluation budget: `--budget`, or with `--iterations` the agents × (iterations + 1) evaluations NIO would spend.

Usage:
    python -m benchmarks.run --out results/quick.csv --runs 2
    python -m benchmarks.run --suite fair23 --agents 50 --budget 10050 --runs 5 --seed 1000 --out fair23.csv
    python -m benchmarks.run --funcs F5 F8 --set BROWSE_STEP=0.5 LEDGE_WALK=false --out tuned.csv
    python -m benchmarks.run --initializer "Sobol Sequence" --out results/initializers/sobol.csv
    python -m benchmarks.run --suite cec2022 --dim 20 --out results/cec/cec2022_d20.csv --traces cec2022_d20.npz
    python -m benchmarks.run --algorithms NIO SCA WOA --agents 50 --budget 10050 --runs 50 --out compare.csv
"""

import argparse
import os
import time
from multiprocessing import Pool

from benchmarks import rivals
from benchmarks.runs import execute, trace_key, write_rows, write_traces
from nio import NIO, SearchSpace
from nio.initializers import INITIALIZERS, get
from problems.functions import Suite, cec, classic, digits, fair, fair23

ALGORITHM = "NIO"
ALGORITHMS = (ALGORITHM, *rivals.RIVALS)
NAMED = {"classic": classic, "fair": fair, "fair23": fair23, "cec2019": digits}  # the suites of fixed dimensions
CHOICES = [*NAMED, *(f"cec{year}" for year in cec.editions)]
DEFAULT_BUDGET = 15000
DEFAULT_RUNS = 5


def functions(suite: str, dim: int | None = None) -> Suite:
    """The functions of `suite` ("classic", "cec2022", ...); a CEC suite's at `dim` dimensions (its default if None)."""
    if suite not in NAMED:
        return cec(int(suite.removeprefix("cec")), dim)
    if dim is not None:
        raise ValueError(f"--dim applies to the CEC suites; {suite} has fixed dimensions.")
    return NAMED[suite]


def parse_settings(items: list[str]) -> dict:
    """Turn `["NAME=value", ...]` into settings: `true` / `false` (any case) become bool, whole numbers int, the rest
    float."""
    settings = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep or not name:
            raise ValueError(f"A setting is written NAME=value, not {item!r}.")
        if value.lower() in ("true", "false"):
            settings[name] = value.lower() == "true"
            continue
        try:
            settings[name] = int(value)
        except ValueError:
            try:
                settings[name] = float(value)
            except ValueError:
                raise ValueError(f"The value of {name} must be a number or true/false, not {value!r}.") from None
    return settings


def optimizer_for(key, suite, agents, evaluations, iterations, seed, dim=None, initializer=None):
    """The optimizer of one run on function `key` of `suite`, starting from the initializer named `initializer` (the
    uniform herd if None); it refuses a bad herd size or budget."""
    fn = functions(suite, dim)[key]
    return NIO(
        fn,
        SearchSpace.of(fn.lb, fn.ub, fn.dim),
        agents=agents,
        evaluations=evaluations,
        iterations=iterations,
        seed=seed,
        history=False,
        initializer=None if initializer is None else get(initializer),
    )


def run_one(task):
    """One run: `task` is (function key, seed, agents, evaluations, iterations, settings, suite, dim, initializer), and
    optionally the algorithm (NIO if left out). Returns its CSV row and its improvement trace."""
    key, seed, agents, evaluations, iterations, settings, suite, dim, initializer, *rest = task
    algorithm = rest[0] if rest else ALGORITHM
    fn = functions(suite, dim)[key]
    if algorithm == ALGORITHM:
        optimizer = optimizer_for(key, suite, agents, evaluations, iterations, seed, dim, initializer)
        result = optimizer.with_settings(**settings).solve()
    else:
        budget = evaluations if evaluations is not None else agents * (iterations + 1)
        result = rivals.solve(algorithm, fn, SearchSpace.of(fn.lb, fn.ub, fn.dim), agents, budget, seed)
    row = {
        "function": key,
        "algorithm": algorithm,
        "seed": seed,
        "evaluations": result.evaluations,
        "best": result.fitness,
        "error": result.fitness - fn.optimal,
        "seconds": round(result.runtime, 3),
    }
    return row, result.trace


def save(rows, path, seconds):
    out = write_rows(rows, path)
    print(f"{len(rows)} runs, {seconds:.0f}s  ->  {out}")


def build_parser():
    p = argparse.ArgumentParser(
        prog="python -m benchmarks.run", description="Run NIO on a benchmark suite and write a results CSV."
    )
    p.add_argument(
        "--suite",
        choices=CHOICES,
        default="classic",
        help="classic: F1-F23; fair: F1-F7 and F9-F13 shifted away from the origin; fair23: all 23 shifted; "
        "cec2019: the 100-Digit Challenge; cec2017, cec2020, cec2022, cec2024: the CEC suites (default: classic)",
    )
    p.add_argument("--dim", type=int, help="dimensions of a CEC suite (default: 10, and 30 for cec2024)")
    p.add_argument("--funcs", nargs="*", metavar="F", help="functions to run (default: every function of the suite)")
    p.add_argument(
        "--algorithms",
        nargs="*",
        default=[ALGORITHM],
        metavar="NAME",
        help=f"from {', '.join(ALGORITHMS)}; the rivals need --agents (default: NIO)",
    )
    p.add_argument(
        "--agents",
        type=int,
        help="herd size (default: NIO's own for the budget: 18 per dimension, at most budget / 500, at least 30)",
    )
    budget = p.add_mutually_exclusive_group()
    budget.add_argument(
        "--budget",
        type=int,
        help=f"objective evaluations per run (default: {DEFAULT_BUDGET}, or the competition's for a CEC suite)",
    )
    budget.add_argument("--iterations", type=int, help="seasons per run, instead of an evaluation budget")
    p.add_argument(
        "--runs", type=int, help=f"runs per function (default: {DEFAULT_RUNS}, or the competition's for a CEC suite)"
    )
    p.add_argument("--seed", type=int, default=1000, help="seed of the first run; run r uses seed + r (default: 1000)")
    p.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="NAME=VALUE",
        help="NIO settings to change, e.g. BROWSE_STEP=0.5 LEDGE_WALK=false",
    )
    p.add_argument(
        "--initializer",
        choices=list(INITIALIZERS),
        metavar="NAME",
        help=f"how the first herd is scattered: {', '.join(INITIALIZERS)} (default: Random Uniform)",
    )
    p.add_argument("--workers", type=int, default=os.cpu_count(), help="worker processes (default: all cores)")
    p.add_argument("--out", required=True, metavar="PATH", help="the results CSV to write")
    p.add_argument("--traces", metavar="PATH", help="also write every run's improvement trace to this .npz")
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        suite = functions(args.suite, args.dim)
    except ValueError as error:
        parser.error(str(error))
    funcs = list(suite.keys()) if args.funcs is None else args.funcs
    if not funcs:
        parser.error("--funcs needs at least one function")
    unknown = [key for key in funcs if key not in suite]
    if unknown:
        parser.error(f"unknown function(s): {', '.join(unknown)}")
    if not args.algorithms or any(name not in ALGORITHMS for name in args.algorithms):
        parser.error(f"choose algorithms from {', '.join(ALGORITHMS)}")
    if args.agents is None and any(name != ALGORITHM for name in args.algorithms):
        parser.error("the rivals need --agents")
    rules = suite.rules  # the CEC competition, None for the other suites
    runs = args.runs if args.runs is not None else rules.runs if rules else DEFAULT_RUNS
    if runs < 1:
        parser.error("--runs must be at least 1")
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be at least 1")
    evaluations = args.budget
    if evaluations is None and args.iterations is None:
        evaluations = rules.budget(suite[funcs[0]].dim) if rules else DEFAULT_BUDGET
    try:
        settings = parse_settings(args.set)
        # Refuse a bad herd, budget or setting before any run starts: build the first run's optimizer here.
        optimizer = optimizer_for(
            funcs[0], args.suite, args.agents, evaluations, args.iterations, args.seed, args.dim, args.initializer
        )
        optimizer.with_settings(**settings)
    except ValueError as error:
        parser.error(str(error))
    seeds = range(args.seed, args.seed + runs)

    tasks = [
        (key, seed, args.agents, evaluations, args.iterations, settings, args.suite, args.dim, args.initializer, name)
        for name in args.algorithms
        for key in funcs
        for seed in seeds
    ]
    start = time.perf_counter()
    done = execute(run_one, tasks, args.workers, Pool)
    names = list(ALGORITHMS)
    done.sort(  # not the finishing order
        key=lambda run: (
            names.index(run[0]["algorithm"]),
            int(run[0]["function"].removeprefix("F")),
            run[0]["seed"],
        )
    )
    save([row for row, _ in done], args.out, time.perf_counter() - start)
    if args.traces:  # keyed "<function>/<seed>" for one algorithm, "<algorithm>/<function>/<seed>" for several

        def problem(row):
            return row["function"] if len(args.algorithms) == 1 else f"{row['algorithm']}/{row['function']}"

        write_traces({trace_key(problem(row), row["seed"]): trace for row, trace in done}, args.traces)


if __name__ == "__main__":
    main()
