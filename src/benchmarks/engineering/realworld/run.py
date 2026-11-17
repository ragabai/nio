"""Run NIO on the CEC 2020 real-world constrained suite and write one CSV row per (problem, run).

Each problem is minimised as `Penalized(problem)` with the competition's evaluation budget for its dimension. A row
reports the objective `f` and the competition's mean constraint violation `violation` (ν, 0 when feasible) at the
best position, and the gap of `f` to the best known value; `--traces` also keeps every run's improvement trace of
the penalised objective, keyed "<problem>/<seed>". The competition asks for 25 runs.

Usage:
    python -m benchmarks.engineering.realworld.run --out results/engineering/rc2020.csv --traces rc2020.npz
    python -m benchmarks.engineering.realworld.run --problems RC15 RC17 --runs 5 --budget 20000 --out quick.csv
"""

import argparse
import os
import time
from multiprocessing import Pool

from benchmarks.runs import execute, trace_key, write_rows, write_traces
from nio import NIO
from problems.engineering import Penalized
from problems.engineering.realworld import PROBLEMS

ALGORITHM = "NIO"
RUNS = 25


def run_one(task):
    """One run: `task` is (problem code, seed, agents, evaluations or None for the competition's). Returns its CSV
    row and its improvement trace."""
    code, seed, agents, evaluations = task
    problem = PROBLEMS[code]
    objective = Penalized(problem, tolerance=0.0)
    evaluations = problem.budget if evaluations is None else evaluations
    result = NIO(objective, objective.space, agents=agents, evaluations=evaluations, seed=seed, history=False).solve()
    f, _, _ = problem.evaluate(result.position)
    violation = float(problem.violation(result.position)[0])
    row = {
        "problem": code,
        "category": problem.category,
        "algorithm": ALGORITHM,
        "seed": seed,
        "dimensions": problem.dimensions,
        "evaluations": result.evaluations,
        "fitness": result.fitness,
        "f": float(f[0]),
        "violation": violation,
        "feasible": violation <= 0,
        "best_known": problem.best,
        "gap": float(f[0]) - problem.best,
        "seconds": round(result.runtime, 3),
    }
    return row, result.trace


def build_parser():
    p = argparse.ArgumentParser(
        prog="python -m benchmarks.engineering.realworld.run",
        description="Run NIO on the CEC 2020 real-world constrained suite.",
    )
    p.add_argument("--problems", nargs="*", default=list(PROBLEMS), metavar="RC", help="problems (default: all 57)")
    p.add_argument("--agents", type=int, default=30, help="herd size (default: 30)")
    p.add_argument("--budget", type=int, help="evaluations per run (default: the competition's for the dimension)")
    p.add_argument("--runs", type=int, default=RUNS, help=f"runs per problem (default: {RUNS})")
    p.add_argument("--seed", type=int, default=1000, help="seed of the first run; run r uses seed + r (default: 1000)")
    p.add_argument("--workers", type=int, default=os.cpu_count(), help="worker processes (default: all cores)")
    p.add_argument("--out", required=True, metavar="PATH", help="the results CSV to write")
    p.add_argument("--traces", metavar="PATH", help="also write every run's improvement trace to this .npz")
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    codes = [code.strip().upper() for code in args.problems]
    unknown = [code for code in codes if code not in PROBLEMS]
    if not codes or unknown:
        parser.error("choose problems from RC01 to RC57")
    if args.runs < 1 or args.agents < 4 or (args.budget is not None and args.budget <= args.agents):
        parser.error("--runs must be at least 1, --agents at least 4, and --budget above --agents")
    seeds = range(args.seed, args.seed + args.runs)
    tasks = [(code, seed, args.agents, args.budget) for code in codes for seed in seeds]
    start = time.perf_counter()
    done = execute(run_one, tasks, args.workers, Pool)
    done.sort(key=lambda run: (run[0]["problem"], run[0]["seed"]))
    out = write_rows([row for row, _ in done], args.out)
    if args.traces:
        write_traces({trace_key(row["problem"], row["seed"]): trace for row, trace in done}, args.traces)
    print(f"{len(done)} runs, {time.perf_counter() - start:.0f}s  ->  {out}")


if __name__ == "__main__":
    main()
