"""Durations written for people, the clock of the progress bar, and a stopwatch."""

import time


def duration(seconds: float) -> str:
    """A duration as people say it: "0.4s", "42s", "1m 12s", "2h 03m"."""
    if seconds < 10:
        return f"{seconds:.1f}s"
    whole = round(seconds)
    if whole < 60:
        return f"{whole}s"
    if whole < 3600:
        return f"{whole // 60}m {whole % 60:02d}s"
    return f"{whole // 3600}h {whole % 3600 // 60:02d}m"


def clock(seconds: float) -> str:
    """A duration as a clock shows it: "00:12", "12:05", "1:02:03"."""
    whole = int(seconds)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


class Stopwatch:
    """Seconds since the `with` block began; it stops counting when the block ends.

    with Stopwatch() as watch:
        ...
    print(duration(watch.elapsed))
    """

    def __init__(self):
        self._start = time.perf_counter()
        self._stop: float | None = None

    def __enter__(self):
        self._start, self._stop = time.perf_counter(), None
        return self

    def __exit__(self, *exc):
        self._stop = time.perf_counter()
        return False

    @property
    def elapsed(self) -> float:
        """Seconds counted so far, or in all once the block has ended."""
        return (self._stop if self._stop is not None else time.perf_counter()) - self._start
