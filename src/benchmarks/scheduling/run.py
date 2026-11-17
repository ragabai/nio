"""Run NIO on cloud task scheduling and write one CSV row per (instance, encoding, run).

The instances are HCSP-100 to HCSP-400 (1024 tasks on 32 machines), each under the `priority` and the `assignment`
encoding. A row reports the makespan of the best schedule, the lower bound no schedule beats (total work over total
speed), the gap to it, the machines' mean utilization and the load imbalance. `--traces` also keeps every run's
improvement trace, keyed "<instance>-<encoding>/<seed>". The defaults are the study's: 20 agents, 50 seasons.

Usage:
    python -m benchmarks.scheduling.run --out results/scheduling/hcsp.csv --traces results/scheduling/hcsp.npz
    python -m benchmarks.scheduling.run --instances 100 --encodings assignment --runs 2 --out quick.csv
"""

import argparse
import os
import time
from functools import cache
from multiprocessing import Pool

from benchmarks.runs import execute, trace_key, write_rows, write_traces
from nio import NIO
from problems.scheduling.objective import Scheduling
from problems.scheduling.workload import HCSP, hcsp

ALGORITHM = "NIO"
ENCODINGS = ("priority", "assignment")


@cache
def objective_for(instance: str, encoding: str) -> Scheduling:
    """The objective of one instance and encoding, built once per worker process."""
    return Scheduling(hcsp(instance), encoding)


def run_one(task):
    """One run: `task` is (HCSP instance, encoding, seed, agents, iterations). Returns its CSV row and its trace."""
    instance, encoding, seed, agents, iterations = task
    objective = objective_for(instance, encoding)
    result = NIO(objective, objective.space, agents=agents, iterations=iterations, seed=seed, history=False).solve()
    schedule = objective.decode(result.position)
    bound = objective.workload.bound
    row = {
        "instance": objective.workload.name,
        "encoding": encoding,
        "algorithm": ALGORITHM,
        "seed": seed,
        "evaluations": result.evaluations,
        "makespan": schedule.makespan,
        "bound": bound,
        "gap": (schedule.makespan - bound) / bound,
        "utilization": float(schedule.utilization.mean()),
        "imbalance": schedule.imbalance,
        "seconds": round(result.runtime, 3),
    }
    return row, result.trace


def build_parser():
    p = argparse.ArgumentParser(prog="python -m benchmarks.scheduling.run", description="Run NIO on task scheduling.")
    p.add_argument("--instances", nargs="*", default=list(HCSP), choices=HCSP, help="HCSP instances (default: all)")
    p.add_argument("--encodings", nargs="*", default=list(ENCODINGS), choices=ENCODINGS, help="(default: both)")
    p.add_argument("--agents", type=int, default=20, help="herd size (default: 20)")
    p.add_argument("--iterations", type=int, default=50, help="seasons per run (default: 50)")
    p.add_argument("--runs", type=int, default=5, help="runs per instance and encoding (default: 5)")
    p.add_argument("--seed", type=int, default=1000, help="seed of the first run; run r uses seed + r (default: 1000)")
    p.add_argument("--workers", type=int, default=os.cpu_count(), help="worker processes (default: all cores)")
    p.add_argument("--out", required=True, metavar="PATH", help="the results CSV to write")
    p.add_argument("--traces", metavar="PATH", help="also write every run's improvement trace to this .npz")
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.instances or not args.encodings or args.runs < 1 or args.agents < 4 or args.iterations < 1:
        parser.error("give an instance and an encoding, --runs and --iterations of at least 1, --agents of at least 4")
    seeds = range(args.seed, args.seed + args.runs)
    tasks = [
        (instance, encoding, seed, args.agents, args.iterations)
        for instance in args.instances
        for encoding in args.encodings
        for seed in seeds
    ]
    start = time.perf_counter()
    done = execute(run_one, tasks, args.workers, Pool)
    done.sort(key=lambda run: (run[0]["instance"], ENCODINGS.index(run[0]["encoding"]), run[0]["seed"]))
    out = write_rows([row for row, _ in done], args.out)
    if args.traces:
        traces = {trace_key(f"{row['instance']}-{row['encoding']}", row["seed"]): trace for row, trace in done}
        write_traces(traces, args.traces)
    print(f"{len(done)} runs, {time.perf_counter() - start:.0f}s  ->  {out}")


if __name__ == "__main__":
    main()
