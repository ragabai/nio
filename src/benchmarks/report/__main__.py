"""python -m benchmarks.report: rebuild the summaries, the figures and the index of `results/`."""

import argparse

from benchmarks.paths import RESULTS
from benchmarks.report import build

parser = argparse.ArgumentParser(prog="python -m benchmarks.report", description="Rebuild the reports of results/.")
parser.add_argument("--results", default=str(RESULTS), metavar="DIR", help="the results directory (default: results/)")
parser.add_argument("--landscapes", action="store_true", help="also draw the CEC 2017 and 2022 galleries at D = 2")
args = parser.parse_args()
written = build(args.results, galleries=args.landscapes)
print(f"{len(written)} files written under {args.results}")
