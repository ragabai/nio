"""Where the results live: `results/` at the root of the repository."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # src/benchmarks/paths.py
RESULTS = ROOT / "results"
