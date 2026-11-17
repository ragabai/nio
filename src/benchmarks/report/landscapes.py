"""Galleries of the CEC functions the organisers define at D = 2, drawn as contours over [-100, 100]²."""

from pathlib import Path

from benchmarks.report.charts import REPORT, landscape
from plots import Contour
from problems.functions import cec


def gallery(suite: str, figures, columns: int = 6, resolution: int = 90) -> list[Path]:
    """Every function of `suite` ("cec2017", ...) defined at D = 2, with its minimiser marked."""
    rules = cec.rules(int(suite.removeprefix("cec")))
    numbers = [number for number in rules.numbers if 2 in rules.available(number)]
    rows = -(-len(numbers) // columns)
    titles = [f"F{number}" for number in numbers]
    size = (175 * columns, 185)
    fig = Contour(size=size, rows=rows, columns=columns, titles=titles, horizontal_spacing=0.02, **REPORT)
    for index, number in enumerate(numbers):
        row, col = divmod(index, columns)
        fn = rules.get(number, 2)
        minimiser = rules.minimiser(number, 2)
        landscape(fig, fn, fn.lb, fn.ub, minimiser=minimiser, resolution=resolution, row=row + 1, col=col + 1)
    fig.update_layout(title=f"{rules.name} at D = 2 (log contours; star: the minimiser)", showlegend=False)
    return fig.update_annotations(font={"size": 12}).save(Path(figures) / f"{suite}_landscapes")
