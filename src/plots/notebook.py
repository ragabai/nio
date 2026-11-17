"""Showing figures in notebooks so that GitHub shows them too.

GitHub draws a notebook without running Plotly's JavaScript, so an interactive figure alone shows blank there.
`bundle` puts the interactive figure and a PNG of it in one output: Jupyter draws the first, GitHub the second.
`notebook()` makes every figure of the session show this way, and keeps one Chrome open to draw the PNGs. The PNGs
are reduced to 256 colors, which keeps a notebook of many figures small.
"""

import base64
import sys
from io import BytesIO

import kaleido
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image
from plotly.io._base_renderers import MimetypeRenderer

from plots import export
from plots.export import quiet


def bundle(fig: go.Figure) -> dict[str, object]:
    """The figure's interactive JSON, a PNG of it at its own size, and a line of text, as one output."""
    data = dict(pio.renderers["plotly_mimetype"].to_mimebundle(fig.to_dict()))
    with quiet():
        png = pio.to_image(fig, format="png", scale=1)
    data["image/png"] = base64.b64encode(compact(png)).decode("ascii")
    data["text/plain"] = f"Figure: {fig.layout.title.text or 'untitled'}"
    return data


def compact(png: bytes) -> bytes:
    """A PNG reduced to 256 colors without dithering: the same chart, a fraction of the bytes."""
    with Image.open(BytesIO(png)) as image:
        fewer = image.convert("RGB").quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    out = BytesIO()
    fewer.save(out, format="PNG", optimize=True)
    return out.getvalue()


def show(fig: go.Figure) -> None:
    """Display `fig` in a notebook as `bundle` makes it."""
    try:
        from IPython.display import display
    except ImportError as err:
        raise ImportError("Showing figures needs IPython; install Jupyter (ipykernel) to show figures.") from err
    display(bundle(fig), raw=True)


class _Renderer(MimetypeRenderer):
    """Plotly's renderer protocol, answered with `bundle`."""

    def to_mimebundle(self, fig_dict):
        return bundle(go.Figure(fig_dict))


def notebook() -> None:
    """Show every figure of this session as `bundle` makes it, including a bare `fig` at the end of a cell. Without
    Chrome to draw the PNGs, figures show interactively only, and one line says how to get Chrome."""
    if not export.has_chrome():
        print(f"Figures show interactively only: {export.NO_CHROME}", file=sys.stderr)
        pio.renderers.default = "plotly_mimetype"
        return
    kaleido.start_sync_server(silence_warnings=True)
    pio.renderers["nio"] = _Renderer()
    pio.renderers.default = "nio"
