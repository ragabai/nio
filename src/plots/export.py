"""Saving figures as PNG, PDF and SVG side by side, through kaleido's headless Chrome.

The PNG is `scale` times the figure's size in pixels (3 by default, crisp on slides); PDF and SVG are vector and
written at scale 1. Without Chrome, saving raises one `RuntimeError` saying how to get it.

Each save starts Chrome on its own, which takes seconds; inside `with browser():` every save shares one Chrome.
"""

import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

FORMATS = ("png", "pdf", "svg")
IMAGE_SUFFIXES = {"png", "pdf", "svg", "jpg", "jpeg", "webp"}
NO_BROWSER = {"ChromeNotFoundError", "BrowserFailedError", "BrowserDepsError"}
NO_CHROME = "Saving images needs Chrome; run `plotly_get_chrome` once (or install Chrome)."


def save(fig: go.Figure, path, formats: Sequence[str] = FORMATS, scale: float = 3) -> list[Path]:
    """Write `fig` as `path` with each format's suffix (an image suffix on `path` is dropped); return the paths."""
    return save_many([fig], [path], formats, scale)


def save_many(
    figures: Sequence[go.Figure], paths: Sequence, formats: Sequence[str] = FORMATS, scale: float = 3
) -> list[Path]:
    """Write every figure in every format in one browser session, each at its own width and height."""
    jobs = [
        (fig, base.with_name(f"{base.name}.{fmt}"), fmt)
        for fig, base in zip(figures, map(_base, paths), strict=True)
        for fmt in formats
    ]
    for _, out, _ in jobs:
        out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with quiet():
            pio.write_images(
                fig=[fig for fig, _, _ in jobs],
                file=[out for _, out, _ in jobs],
                format=[fmt for _, _, fmt in jobs],
                scale=[scale if fmt == "png" else 1 for _, _, fmt in jobs],
                width=[fig.layout.width or 900 for fig, _, _ in jobs],
                height=[fig.layout.height or 450 for fig, _, _ in jobs],
            )
    except Exception as err:
        if _no_chrome(err):
            raise RuntimeError(NO_CHROME) from None
        raise
    return [out for _, out, _ in jobs]


def _no_chrome(err: Exception) -> bool:
    """Whether `err` says Chrome is missing: kaleido's own error, or Plotly's wrapping of it."""
    named = any(kind.__name__ in NO_BROWSER for kind in type(err).__mro__)
    return named or (isinstance(err, RuntimeError) and "plotly_get_chrome" in str(err))


@contextmanager
def quiet() -> Iterator[None]:
    """Silence kaleido's notice that a shared browser ignores per-call options; there is nothing to act on."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "The kopts argument is ignored", UserWarning)
        yield


@contextmanager
def browser() -> Iterator[None]:
    """Keep one Chrome open for every figure saved or shown inside the block. Without Chrome it raises at once (a
    shared browser that cannot start would otherwise leave every save waiting)."""
    import kaleido

    if not has_chrome():
        raise RuntimeError(NO_CHROME)
    kaleido.start_sync_server(silence_warnings=True)
    try:
        yield
    finally:
        kaleido.stop_sync_server(silence_warnings=True)


def has_chrome() -> bool:
    """Whether figures can be saved here: a Chrome that kaleido can drive is installed, or was downloaded by
    `plotly_get_chrome`. It looks Chrome up without launching it, so a slow start never reads as missing."""
    from choreographer.browsers.chromium import Chromium

    return Chromium.find_browser(skip_local=False) is not None


def _base(path) -> Path:
    path = Path(path)
    return path.with_suffix("") if path.suffix.lstrip(".").lower() in IMAGE_SUFFIXES else path
