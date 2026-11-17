"""A small progress bar, written for NIO's runs, one short line:

    NIO  70% |==============>-----| 700/1000 00:12<00:05 best=1.2e-03

The time is the time taken and, while running, the time left. On a terminal it redraws its one line, at most ten times
a second. In a notebook it draws one output and updates it in place, so the saved notebook keeps only the last line.
Anywhere else (a log file, a pipe) it writes a line at every tenth and at the end, so logs stay readable. Without a
total it counts (`NIO 5 00:05`). It is plain ASCII, so every console prints it, and it never warns.

    with Progress(total=seasons, label="NIO") as bar:
        for season in range(seasons):
            ...
            bar.note(best=f"{best:.4e}")
            bar.update()

    for item in Progress(items, label="runs"):
        ...
"""

import math
import sys
import time
from collections.abc import Callable, Iterable, Iterator

from utils.timing import clock as as_clock

CELLS = 20  # the width of the bar, inside its brackets
REFRESH = 0.1  # seconds between redraws of a terminal line or a notebook output
DONE, HEAD, LEFT = "=", ">", "-"  # a cell done, the head, a cell to do
EDGE = "|"  # the ends of the bar


class Progress:
    """Progress through `total` steps (or an unknown number), drawn on `stream` (standard error by default), or in
    the notebook when it runs in one (`notebook=None` finds out; True or False decides)."""

    def __init__(
        self,
        iterable: Iterable | None = None,
        total: int | None = None,
        label: str = "",
        disable: bool = False,
        every: float = 0.1,
        stream=None,
        clock: Callable[[], float] = time.monotonic,
        notebook: bool | None = None,
    ):
        if total is None and iterable is not None and hasattr(iterable, "__len__"):
            total = len(iterable)
        self.iterable, self.total, self.label, self.disable = iterable, total, label, disable
        self.n = 0
        self.fields: dict[str, object] = {}
        self._notebook = in_notebook() and stream is None if notebook is None else notebook
        self._stream = sys.stderr if stream is None else stream
        self._clock = clock
        self._start = clock()
        self._redraws = self._notebook or bool(getattr(self._stream, "isatty", lambda: False)())
        self._step = max(1, math.ceil(total * every)) if total else None
        self._mark = self._step  # the next count that earns a line in a log
        self._printed: int | None = None  # the count of the last line in a log
        self._drawn: float | None = None  # when the redrawn line was last drawn
        self._cell: NotebookLine | None = None
        self._width = 0
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __iter__(self) -> Iterator:
        try:
            for item in () if self.iterable is None else self.iterable:
                yield item
                self.update()
        finally:
            self.close()

    def note(self, **fields) -> None:
        """Set the fields shown at the end of the line, such as `best=1.2e-03`."""
        self.fields.update(fields)

    def update(self, n: int = 1) -> None:
        """Advance by `n` steps."""
        self.n += n
        if self.disable or self._closed:
            return
        if self._redraws:
            now = self._clock()
            if self._drawn is None or now - self._drawn >= REFRESH:
                self._draw(now)
        elif self._step is not None and self.n >= self._mark:
            self._print()
            self._mark = (self.n // self._step + 1) * self._step

    def close(self) -> None:
        """Draw the last line and end it; later calls do nothing."""
        if self._closed:
            return
        self._closed = True
        if self.disable:
            return
        if self._redraws:
            self._draw(self._clock())
            if not self._notebook:
                self._write("\n")
        elif self._printed != self.n:
            self._print()

    def line(self) -> str:
        """The progress as one line of text."""
        elapsed = self._clock() - self._start
        if self.total is None:
            parts = [self.label, str(self.n), as_clock(elapsed)]
        else:
            share = min(self.n / self.total, 1.0) if self.total else 1.0
            if self.n == 0:
                time_ = ""
            elif self.n >= self.total:
                time_ = as_clock(elapsed)
            else:
                time_ = f"{as_clock(elapsed)}<{as_clock(elapsed * (self.total - self.n) / self.n)}"
            parts = [
                f"{self.label} {int(100 * share):3d}%" if self.label else f"{int(100 * share):3d}%",
                bar(share),
                f"{self.n}/{self.total}",
                time_,
            ]
        parts += [f"{key}={value}" for key, value in self.fields.items()]
        return " ".join(part for part in parts if part)

    def _draw(self, now: float) -> None:
        text = self.line()
        if self._notebook:
            if self._cell is None:
                self._cell = NotebookLine(text)
            else:
                self._cell.update(text)
        else:
            self._write("\r" + text + " " * max(0, self._width - len(text)))
            self._width = len(text)
        self._drawn = now

    def _print(self) -> None:
        self._write(self.line() + "\n")
        self._printed = self.n

    def _write(self, text: str) -> None:
        self._stream.write(text)
        flush = getattr(self._stream, "flush", None)
        if flush is not None:
            flush()


def bar(share: float) -> str:
    """The bar for a share of the work done (0 to 1), `CELLS` wide: `|==========>---------|` at 50%, the head moving a
    cell per twentieth done. It is plain ASCII, so every console prints it."""
    done = int(CELLS * min(max(share, 0.0), 1.0))
    inside = DONE * CELLS if done >= CELLS else DONE * done + HEAD + LEFT * (CELLS - done - 1)
    return EDGE + inside + EDGE


def in_notebook() -> bool:
    """Whether this runs in a Jupyter kernel."""
    ipython = sys.modules.get("IPython")
    shell = ipython.get_ipython() if ipython is not None else None
    return getattr(shell, "kernel", None) is not None


class NotebookLine:
    """One output of a notebook cell, drawn once and then updated in place."""

    def __init__(self, text: str):
        from IPython.display import display

        self._handle = display({"text/plain": text}, raw=True, display_id=True)

    def update(self, text: str) -> None:
        self._handle.update({"text/plain": text}, raw=True)
