"""Run NIO, and optionally its rivals, on the constrained engineering designs; one CSV row per (algorithm, design, run).

Each design is minimised as `Penalized(design)`: its cost plus 1e6 times the sum of its normalised constraint
violations. A row reports the design's cost at the best position, whether it is feasible, its total violation and
its gap to the best known cost; `--traces` also keeps every run's improvement trace, keyed "<design>/<seed>".

Usage:
    python -m benchmarks.engineering.run --out results/engineering/designs.csv --traces results/engineering/designs.npz
    python -m benchmarks.engineering.run --designs welded-beam gear-train --runs 5 --out quick.csv
    python -m benchmarks.engineering.run --agents 50 --evaluations 10050 --runs 5 --out designs_10050.csv
    python -m benchmarks.engineering.run --algorithms NIO SCA WOA --agents 50 --evaluations 10050 --out compare.csv

The rivals (`benchmarks.rivals`, from mealpy) always run on an evaluation budget: `--evaluations`, or with
`--iterations` the agents × (iterations + 1) evaluations NIO would spend.
"""

import argparse
import os
import time
from multiprocessing import Pool

from benchmarks import rivals
from benchmarks.runs import execute, trace_key, write_rows, write_traces
from nio import NIO
from problems.engineering import DESIGNS, Penalized

ALGORITHM = "NIO"
ALGORITHMS = (ALGORITHM, *rivals.RIVALS)


def run_one(task):
    """One run: `task` is (algorithm, design code, seed, agents, iterations, evaluations), one of the last two None.
    Returns its CSV row and its improvement trace."""
    algorithm, code, seed, agents, iterations, evaluations = task
    design = DESIGNS[code]
    objective = Penalized(design)
    if algorithm == ALGORITHM:
        result = NIO(
            objective,
            objective.space,
            agents=agents,
            iterations=iterations,
            evaluations=evaluations,
            seed=seed,
            history=False,
        ).solve()
    else:
        budget = evaluations if evaluations is not None else agents * (iterations + 1)
        result = rivals.solve(algorithm, objective, objective.space, agents, budget, seed)
    solution = objective.evaluate(result.position)
    row = {
        "design": code,
        "algorithm": algorithm,
        "seed": seed,
        "evaluations": result.evaluations,
        "fitness": result.fitness,
        "cost": solution.cost,
        "best_known": design.best,
        "gap": (solution.cost - design.best) / abs(design.best),
        "feasible": solution.feasible,
        "violation": float(solution.violations.sum()),
        "x": " ".join(f"{value:.10g}" for value in solution.x),
        "seconds": round(result.runtime, 3),
    }
    return row, result.trace


def build_parser():
    p = argparse.ArgumentParser(
        prog="python -m benchmarks.engineering.run", description="Run NIO and its rivals on the engineering designs."
    )
    p.add_argument("--designs", nargs="*", default=list(DESIGNS), metavar="CODE", help="designs (default: all 9)")
    p.add_argument(
        "--algorithms",
        nargs="*",
        default=[ALGORITHM],
        metavar="NAME",
        help=f"from {', '.join(ALGORITHMS)} (default: NIO)",
    )
    p.add_argument("--agents", type=int, default=30, help="herd size (default: 30)")
    budget = p.add_mutually_exclusive_group()
    budget.add_argument("--iterations", type=int, help="seasons per run (default: 300)")
    budget.add_argument("--evaluations", type=int, help="objective evaluations per run, instead of seasons")
    p.add_argument("--runs", type=int, default=30, help="runs per design (default: 30)")
    p.add_argument("--seed", type=int, default=1000, help="seed of the first run; run r uses seed + r (default: 1000)")
    p.add_argument("--workers", type=int, default=os.cpu_count(), help="worker processes (default: all cores)")
    p.add_argument("--out", required=True, metavar="PATH", help="the results CSV to write")
    p.add_argument("--traces", metavar="PATH", help="also write every run's improvement trace to this .npz")
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    unknown = [code for code in args.designs if code not in DESIGNS]
    if not args.designs or unknown:
        parser.error(f"choose designs from {', '.join(DESIGNS)}")
    if not args.algorithms or any(name not in ALGORITHMS for name in args.algorithms):
        parser.error(f"choose algorithms from {', '.join(ALGORITHMS)}")
    if args.iterations is None and args.evaluations is None:
        args.iterations = 300
    if args.runs < 1 or args.agents < 4 or (args.iterations or 1) < 1:
        parser.error("--runs and --iterations must be at least 1, and --agents at least 4")
    if args.evaluations is not None and args.evaluations <= args.agents:
        parser.error(f"--evaluations must exceed the {args.agents} evaluations of the first herd")
    tasks = [
        (algorithm, code, seed, args.agents, args.iterations, args.evaluations)
        for algorithm in args.algorithms
        for code in args.designs
        for seed in range(args.seed, args.seed + args.runs)
    ]
    start = time.perf_counter()
    done = execute(run_one, tasks, args.workers, Pool)
    order = list(DESIGNS)
    names = list(ALGORITHMS)
    done.sort(key=lambda run: (names.index(run[0]["algorithm"]), order.index(run[0]["design"]), run[0]["seed"]))
    out = write_rows([row for row, _ in done], args.out)
    if args.traces:  # keyed "<design>/<seed>" for one algorithm, "<algorithm>/<design>/<seed>" for several

        def problem(row):
            return row["design"] if len(args.algorithms) == 1 else f"{row['algorithm']}/{row['design']}"

        write_traces({trace_key(problem(row), row["seed"]): trace for row, trace in done}, args.traces)
    print(f"{len(done)} runs, {time.perf_counter() - start:.0f}s  ->  {out}")


if __name__ == "__main__":
    main()
