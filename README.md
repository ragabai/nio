# NIO

NIO, the Nubian Ibex Optimizer, is a population-based optimizer for continuous problems with bounds. It minimises an
objective over a box with a herd of ibex that lives through a year on the objective's terrain: it explores widely in the
first seasons and gathers around the best ground by the last. This repository holds the implementation and the benchmark
suites.

## Layout

```
src/utils/        small tools: a progress bar, timing, the system a run ran on
src/plots/        figures for anything, on Plotly: templates, palettes, layouts, charts, export, animation
src/nio/          the optimizer: behaviours, settings, initializers
src/problems/     the problems and their data (data/): classical functions, CEC suites, engineering designs,
                  the CEC 2020 real-world suite, feature selection, task scheduling
src/benchmarks/   the studies: runners, statistics, run analysis and reports
```

Each package depends only on the ones above it. The problems never import `nio`: each carries its
own box (`problems.Box`, with `lb` and `ub`), so any optimizer can use them, and NIO takes such a box as its search
space. Neither `utils` nor `plots` imports NIO's packages, so either can serve other work too.

## Install

NIO needs Python 3.12 or later and uses [uv](https://docs.astral.sh/uv/). Saving figures needs Chrome
(`uv run plotly_get_chrome` installs one).

```bash
git clone https://github.com/ragabai/nio.git
cd nio
uv sync
```

## Example

```python
import numpy as np
from nio import NIO, SearchSpace

optimizer = NIO(lambda x: float(np.sum(x**2)), SearchSpace.of(-5, 5, 10), evaluations=3_000, seed=1)
result = optimizer.solve()
print(result.fitness)
```

The search box is a `SearchSpace`, or any object with per-dimension bounds `lb` and `ub`, such as a benchmark
problem's own box (`objective.space`). The budget is `evaluations` objective calls or `iterations` seasons; with
neither, a run lasts 500 seasons. The herd has `agents` ibex; left out, NIO sizes the herd from the budget: 18 ibex
per dimension, at most `evaluations / 500` and at least 30 (30 under an iteration budget). `result.position` is the best point and
`result.convergence` the best value after each season. `history=False` leaves out the per-season herd, which is the
bulk of a result.

An objective that also has a `batch(population)` method, which scores the rows of an array in one call, has its first
herd scored in one `batch` call. Every later proposal is scored on its own with `objective(x)`, so the objective must be
callable too; an object with only a `batch` method is refused.

With a `seed`, `solve()` also seeds the global `np.random`, so an objective that draws its own noise from it (as F7
does) is reproducible too.

## Using NIO from another project

NIO is a package named `nio`. From a project that uses uv, run `uv add git+https://github.com/ragabai/nio`; with pip,
run `pip install git+https://github.com/ragabai/nio`. A local clone works too (`uv add <path-to-repo>`). Then
`from nio import NIO, SearchSpace` works as in the example above, and `problems` comes with it, for use with any
optimizer.

## The behaviours

Each season every ibex picks one of eleven behaviours, with a probability that follows how often that behaviour has
improved the herd recently. The ibex picture is the model's metaphor. In the formulas:

* $`x_i`$ is the ibex that moves and $`\sigma`$ the herd's spread (standard deviation) per dimension;
* $`z \sim \mathcal{N}(0, I)`$ and $`U \sim \mathcal{U}(0, 1)`$ are drawn per dimension;
* $`\mathrm{lb}`$ and $`\mathrm{ub}`$ are the bounds of the box, and $`\mathrm{span} = \mathrm{ub} - \mathrm{lb}`$;
* $`\odot`$ is the element-wise product, and $`\mathbf{1} = (1, \dots, 1)`$;
* $`\tau \in [0, 1]`$ is the season, from 0 at the start of the run to 1 at its end.

| Behaviour | The ibex picture | Formula |
| --- | --- | --- |
| `browse` | small, careful steps around where an ibex stands | $`x_i + \mathrm{step}\,\sigma \odot z`$ |
| `follow_dominant` | steps toward a blend of the herd's best-ranked ibex | $`x_i + \varphi \odot (B - x_i)`$, $`B`$ the rank-weighted blend, $`\varphi \sim \mathcal{U}(0, 2\,\mathrm{reach})`$ |
| `zigzag_climb` | a climb up a steep slope in switchbacks, not straight up | $`x_i + s\,(\cos\theta\, d + \sin\theta\, \lVert d \rVert\, e)`$, $`d = x_t - x_i`$ |
| `cliff_escape` | a leap across the rock: mostly hops, now and then a long bound | $`x_i + \mathrm{reach}\;\mathrm{span} \odot C \odot m`$, $`C`$ standard Cauchy, $`m`$ a mask |
| `sentinel` | a look-out scanning a few lines of terrain far from the herd | a few coordinates redrawn as $`\mathrm{lb}_d + \mathrm{span}_d\, U`$ |
| `return_to_spring` | going back to one of the best places the herd has found | $`s + \mathrm{scatter}\, \lvert x_i - s \rvert \odot z`$, $`s`$ a spring |
| `rut` | the groups meet and the young take after both parents | mix coordinates into $`c`$, then $`c + \eta \odot (x_b - c)`$ |
| `home_range` | returning toward the core of the home range | $`c + 10^{-\mathrm{depth}\,U} \odot (x_i - c)`$, $`c = (\mathrm{lb} + \mathrm{ub})/2`$ |
| `ridge_walk` | walking the main ridgeline | $`\bar{x}_L\,\mathbf{1} + s\,\mathbf{1}`$, $`\bar{x}_L`$ the leader's mean coordinate |
| `ledge_walk` | stepping along the ledge the herd's best stand on | $`x_i + \mathrm{stride}\; A u`$, $`A A^\top`$ the rank-weighted scatter of the best half, $`u \sim \mathcal{N}(0, I)`$ |
| `follow_trail` | heading for a dominant and on along a trail the herd has worn, keeping to its own ground in some directions | $`x_i + s\,(x_p - x_i) + s\,(x_{r_1} - x_{r_2})`$, $`x_{r_2}`$ a herd-mate or an old track, per coordinate with probability `keep` (at least one), else $`x_i`$'s; $`s`$ and `keep` come from the herd's lore |

The full formula of each behaviour, with its edge cases, is in its docstring under `src/nio/behaviours/`. For example,
the dominant blend that `follow_dominant` steps toward is

```math
B = \frac{\sum_j w_j\, u_j\, x_{(j)}}{\sum_j w_j\, u_j}, \qquad u_j \sim \mathcal{U}(0, 1],
```

with $`x_{(j)}`$ the $`j`$-th best ibex and $`w_j`$ its dominance weight, which falls with rank. The slope angle of
`zigzag_climb` is $`\theta = \mathrm{atan2}(\mathrm{rise}, \mathrm{run})`$, where the rise is the fitness gap and
the run the distance, each scaled by the herd's range.

Six more parts of the model shape the herd without being behaviours:

* **Fission–fusion:** the herd lives in subgroups, each with its own leader, that regroup when one stalls or gathers.
  They fuse into one herd by `FUSION_SEASON` ($`\tau_f`$); the number of subgroups is
  $`\lceil \mathrm{most}\,(1 - \tau / \tau_f) \rceil`$.
* **Moving together:** every ibex proposes its move from where the herd stood when the season began, and the
  proposals are scored together (in one call when the objective has a `batch` method).
* **Thinning:** over the year the weakest ibex leave the herd, which shrinks linearly from `agents` to `THIN_LEAST`
  (12), so later in the year fewer, fitter ibex share the evaluations.
* **Tracks and lore:** an ibex that moves leaves a track where it stood, and trail followers stride along the line
  from a track to a herd-mate, the way the herd has been moving. The herd's lore remembers which strides, and how
  much of the trail kept, led to better ground, and the next followers draw theirs around it. A larger herd wears
  deeper trails: the route preference of `follow_trail` is scaled by `TRAIL_WEIGHT` (0.3) times the herd's size.
* **Noisy-terrain re-check:** at the start the herd tastes its best spot a second time. If the two tastes differ the
  terrain is noisy, and a stalled ibex tastes its own spot again instead of moving.
* **Migration:** a herd whose best has not improved for `MIGRATE_PATIENCE` seasons (30) and that has gathered on one
  spot (`MIGRATE_GATHER`) has used up its ground, so it leaves for a fresh range. A new herd is scattered uniformly across the box and lives a new year over
  what is left of the run, from exploring to exploiting, with $`\tau = (c - c_0) / (1 - c_0)`$, where $`c`$ is the
  run's clock and $`c_0`$ the clock when the herd arrived. The springs and the elite go with it, and the herd migrates
  only before `MIGRATE_UNTIL` (0.9) of the run and never on noisy terrain. Without it a herd that gathers early spends
  the rest of the budget tasting the same ground; `MIGRATE_PATIENCE=0` keeps the run to one year.

`home_range` and `ridge_walk` use the structure of the search box. `home_range` pulls toward the centre of the search
box. `ridge_walk` steps along the line through the origin along $`\mathbf{1} = (1, \dots, 1)`$, on which every
coordinate is equal; that line is the box's main diagonal only when every dimension has the same bounds. `home_range`
keeps the run translation invariant, because the centre moves with the bounds; `ridge_walk` is the only behaviour that
breaks translation invariance.

## Settings

Every tunable number is a field of `nio.Settings`, a frozen dataclass whose docstring explains each one with its
default. Give a `Settings` to the constructor (`settings=`) or change values with `with_settings(NAME=value)`. An
unknown name is refused with a message that lists the valid settings; a bool for `SPRINGS`, a `SPRINGS` below 1 and a
`ROUTE_FLOOR` outside [0, 1/11] are refused with a message that says why.

## Initializers

The first herd is uniform in the box by default. `initializer=` scatters it another way: `nio.initializers` has
`RandomUniform`, `LatinHypercube`, `Sobol`, `Halton`, `Opposition`, `Sphere`, `Centroid`, `Gaussian`, `SineCosine`,
`Fibonacci` and `Hybrid`, and `INITIALIZERS` holds one of each by name. The three quasi-random ones need scipy.
`benchmarks.run --initializer NAME` runs a suite from one of them, such as `--initializer "Sobol Sequence"`.

```python
from nio.initializers import Sobol

sobol = NIO(optimizer.objective, optimizer.space, evaluations=3_000, seed=1, initializer=Sobol()).solve()
```

## Diagnostics

After `solve()`, `optimizer.diagnostics` tells how the run went:

```python
from nio.behaviours import BEHAVIOURS

diagnostics = optimizer.diagnostics
print(dict(zip(BEHAVIOURS, diagnostics.used.sum(axis=0).tolist())))  # how often each behaviour was used
print(diagnostics.groups[:3], diagnostics.sizes[-1], diagnostics.regroups, diagnostics.noisy)
```

`used` counts the ibex that used each behaviour in each season, `groups` and `sizes` give the number of subgroups and
the herd size in each season, `regroups` counts the regroupings, `springs` holds the best distinct places found, `noisy`
and `rechecks` report the noisy-terrain check, and `migrations` counts the herd's migrations.

## The test functions

`problems.functions` is the one place to import a continuous test function from, by name or by suite. Each is a
`Function` you call on a position, with its bounds, dimension and optimum; `.at()` returns a customised copy:

```python
from problems.functions import rosenbrock, classic, fair23, cec, digits

rosenbrock(np.ones(30))                       # 0.0: classic F5, 30-D over [-30, 30]
rosenbrock.at(dim=50, lb=-10, ub=10)          # the same function, 50-D over [-10, 10]
rosenbrock.shifted()                          # its optimum moved off (1, ..., 1): fair23.rosenbrock
classic["F5"], classic.rosenbrock             # by key or by name; `for fn in classic` runs through F1 to F23
classic.at(dim=50)                            # every scalable function (F1 to F13) at 50-D
cec(2022, dim=20)["F12"], cec.editions        # a CEC suite by year: (2017, 2020, 2022, 2024)
digits["F1"]                                  # the CEC 2019 100-Digit Challenge
```

## Running the benchmarks

`benchmarks.run` runs NIO on a suite and writes one CSV row per run, in the order of function number and seed. The
runners' defaults are NIO's own herd size for the budget and seeds from 1000: 15 000 evaluations on the classic
suite and the CEC 2019 functions (a herd of 30), and the competition's budget (10 000 · D) on the other CEC suites
(a herd of 18 · D).

```bash
uv run python -m benchmarks.run --suite classic --runs 10 --out results/classic.csv
uv run python -m benchmarks.run --suite cec2019 --runs 10 --out results/cec2019.csv --traces results/cec2019.npz
uv run python -m benchmarks.run --suite cec2022 --runs 10 --out results/cec2022_d10.csv
```

`--algorithms NIO SCA WOA QSA NRO GBO AEO BMO GJO CDO --agents 50 --budget 10050` runs NIO's rivals on the same
functions and budget (see Engineering design for how they run).

Runs are bit-exact on Linux; other platforms may differ in the last digits. Add `--set NAME=value` to `benchmarks.run`
to change a setting.

## The CEC suites

`problems.cec` holds the CEC 2017, 2020, 2022 and 2024 bound-constrained suites, reached as `cec(year)`, written from
the organisers' definitions and their official shift, rotation and shuffle data (`src/problems/data/cec/`). They match
the organisers' compiled C code to within 1e-11. Each function is a `Function` over $`[-100, 100]^D`$ whose optimum is
the function's bias, and each suite knows its competition's budget and number of runs:

```python
from problems.functions import cec

suite = cec(2022, dim=20)
fn = suite["F12"]
run = NIO(fn, SearchSpace.of(fn.lb, fn.ub, fn.dim), evaluations=3_000, seed=1).solve()  # suite.rules.budget(20): 1 000 000
print(run.fitness - fn.optimal, suite.rules.runs, suite.rules.group(12))  # the error, 30 runs, "composition"
```

CEC 2024 is CEC 2017 without F2 at $`D = 30`$, with its functions keeping their CEC 2017 numbers.

`problems.cec.digits` holds the CEC 2019 100-Digit Challenge (`digits` in `problems.functions`), written from the
organisers' `cec19_func.cpp` and data: F1 Chebyshev ($`D = 9`$, $`[-8192, 8192]`$), F2 inverse Hilbert ($`D = 16`$,
$`[-16384, 16384]`$), F3 Lennard-Jones ($`D = 18`$, $`[-4, 4]`$) and the shifted, rotated F4 to F10 ($`D = 10`$,
$`[-100, 100]`$), each with its optimum at 1. Like the reference, F1 adds $`p(1.2)^2`$ (not the gap to the target) when
$`p(1.2)`$ falls short, so the zero polynomial also scores 1. `benchmarks.run --suite cec2019` runs it at those fixed
dimensions.

`benchmarks.run` runs a CEC suite at `--dim` dimensions, by default with the competition's budget and number of
runs; `--traces` also keeps every run's improvement trace, keyed `"<function>/<seed>"`:

```bash
uv run python -m benchmarks.run --suite cec2022 --dim 20 --out results/cec2022_d20.csv --traces results/cec2022_d20.npz
```

## Engineering design

`problems.engineering` holds nine classic constrained designs: pressure vessel, welded beam, tension/compression
spring, speed reducer, three-bar truss, cantilever beam, gear train, multiple-disc clutch brake (as CEC 2020 RC21
states it) and piston lever. Constraints are divided by their limits, and discrete variables (plate thicknesses, teeth)
are snapped before evaluation. `Penalized` turns a design into an
objective: its cost plus $`10^6 \sum_k \max(0, g_k)`$.

```python
from problems.engineering import DESIGNS, Penalized

objective = Penalized(DESIGNS["welded-beam"])
design = objective.evaluate(NIO(objective, objective.space, iterations=300, seed=1).solve().position)
print(design.cost, design.feasible, design.x)  # the best known cost is 1.724852
```

`python -m benchmarks.engineering.run --out results/engineering/designs.csv` runs every design 30 times (30 agents,
300 seasons).

`--algorithms NIO SCA WOA QSA NRO GBO AEO BMO GJO CDO` adds the rivals, run from mealpy (`uv sync --group compare`)
with mealpy's default parameters. They get the same objective and the same budget in evaluations, not iterations,
because they spend different numbers of evaluations per iteration; each rival's iterations are set so that its
schedules end with the budget. `benchmarks.rivals.solve` runs one rival on any objective.

`problems.engineering.realworld` holds the 57 problems of the CEC 2020 real-world constrained competition (RC01 to
RC57): industrial chemical processes, process synthesis, mechanical design, power systems, power electronics and
livestock feed rations, ported from the organisers' MATLAB code and checked against an independent C++ port of it.
Each has the competition's bounds, budget and violation measure $`\nu`$, the
mean over all constraints of $`\max(0, g)`$ and of $`|h|`$ beyond $`10^{-4}`$; $`\nu = 0`$ is feasible.

```python
from problems.engineering import realworld

problem = realworld.get("RC15")
best = NIO(Penalized(problem), problem.space, evaluations=20_000, seed=1).solve().position  # budget: problem.budget
print(problem.evaluate(best)[0], problem.violation(best), problem.best)
```

`python -m benchmarks.engineering.realworld.run --out results/engineering/rc2020.csv` runs every problem with its
budget; the competition asks for `--runs 25`.

## Feature selection

`problems.selection` is wrapper feature selection with k-NN ($`k = 5`$) on the 42 classification and 7 regression
datasets of `problems.datasets` (its `data/datasets/`, from UCI and OpenML). A position in $`[0, 1]^n`$ keeps the
features whose coordinate is above 0.5, and its fitness is $`0.99\,e + 0.01\,r`$, with $`e`$ the validation error and
$`r`$ the share of features kept. The error is the error rate for classification and the RMSE over the targets'
standard deviation for regression, so in both an error of 1 is what keeping no feature costs. The data is split once
into a training part and a held-out test part that the search never sees.

```python
from problems import datasets
from problems.selection import FeatureSelection

objective = FeatureSelection(datasets.load("Sonar"), validation="cv")  # 5-fold cross-validation on the training part
subset = objective.evaluate(NIO(objective, objective.space, agents=20, iterations=5, seed=1).solve().position)
print(subset.features, subset.error, subset.test_error)
```

`python -m benchmarks.selection.run --out results/selection/classification.csv` runs every classification dataset
5 times (5-fold cross-validation, 20 agents, 50 seasons); `--task regression` runs the regression ones (one held-out
fold, 20 agents, 2 000 evaluations).

## Task scheduling

`problems.scheduling` minimises the makespan of independent tasks on heterogeneous machines: the HCSP instances
(1024 tasks on 32 machines, its `data/scheduling/HCSP/`) and GoCJ job sets (`data/scheduling/GoCJ/`). Under the
`priority` encoding a position in $`[0, 1]^n`$ is sorted into a task order and list-scheduled, each task going to the
machine that frees up first; under `assignment` the integer part of each coordinate is that task's machine. The
total work over the total speed is a makespan no schedule can beat (`workload.bound`).

```python
from problems.scheduling import Scheduling, hcsp

objective = Scheduling(hcsp("100"), encoding="priority")
schedule = objective.decode(NIO(objective, objective.space, agents=20, iterations=2, seed=1).solve().position)
print(schedule.makespan, objective.workload.bound, schedule.utilization.mean())
```

`python -m benchmarks.scheduling.run --out results/scheduling/hcsp.csv` runs every HCSP instance under both encodings
5 times (20 agents, 50 seasons).

## Plots

`plots` draws every figure of the repository, and any of your own, on [Plotly](https://plotly.com/python/) in one
clean look, one class per kind of figure:

```python
import numpy as np

from nio import NIO, SearchSpace
from plots import Contour, Line, Surface

sphere = lambda x: float(np.sum(x**2))
run = NIO(sphere, SearchSpace.of(-5, 5, 2), iterations=60, seed=1).solve()

Line(run.convergence, title="Sphere, two dimensions", x="Season", y="f(best)", log=True)

fig = Line(title="The herd's fitness", x="Season", log=True)
fig.add(run.fitnesses.min(axis=1), name="best").add(run.fitnesses.max(axis=1), name="worst")

Contour(sphere, (-5, 5)).paths(run.positions).herd(run.positions[-1], best=run.position)
Surface(sphere, (-5, 5), colors="Cividis").herd(run.positions[-1], best=run.position)
```

| Class | Draws | Methods |
| --- | --- | --- |
| `Line` | series over an axis | `add`, `band` (a median and its quartiles), `agents` (a thin line per ibex) |
| `Area` | stacked areas | `add` |
| `Scatter` | markers | `add` |
| `Bars`, `Boxes`, `Histogram`, `Violin` | how values spread (`Violin(log=True)` shapes on the logarithm) | `add` |
| `ECDF` | the share of values at or below each value | `add_values` |
| `Heatmap` | a table as coloured cells, optionally on a log scale or with its values written | `add` |
| `Density` | how densely a cloud of points lies, as filled contours or bins | `add` |
| `Parallel` | parallel coordinates, a line per item across many axes, coloured by a value | |
| `Paths3D` | each agent's path with the season upward | `add` |
| `Dual` | two series on two y axes | |
| `Contour` | a function over a box, filled or (`fill=False`) as lines | `add`, `herd`, `paths`, `trajectories`, `segments`, `arrows`, `start`, `visited`, `star` |
| `Surface` | a function over a box in 3-D | `herd`, `paths` |
| `Animate` | the herd moving over a landscape | `interactive`, `gif`, `snapshots` |

Every class is a Plotly figure, so Plotly's own methods (`update_layout`, `update_traces`, ...) work on it. Each takes
`title`, `x` and `y` (the axis titles), `log`, `legend` and `size`; `plots.use(...)` sets the session's template,
palette and theme. `fig.save("figures/fitness")` writes a PNG, a PDF and an SVG (saving needs Chrome:
`uv run plotly_get_chrome`).

## Reports and figures

`python -m benchmarks.report` writes a markdown summary next to every results CSV in `results/`, the figures of every
study in `results/figures/` (PNG, PDF and SVG), and an index of both in `results/README.md`. A study whose results are
missing is skipped.

## Comparing with another method

`benchmarks.stats` compares NIO's results CSV with another method's CSV. `compare(x, y)` takes two samples of final
errors, such as the `error` column of one function in each file, and returns `+` if `x` is significantly lower than `y`,
`-` if it is significantly higher and `=` otherwise. The test is the Wilcoxon rank-sum test at p < 0.05, and an error
below 1e-8 counts as 0, so a tiny error ties with an exact zero. It needs scipy, which `nio-benchmarks` depends on.

```python
import csv

from benchmarks.stats import compare


def errors(path, function):
    with open(path, newline="") as fh:
        return [float(row["error"]) for row in csv.DictReader(fh) if row["function"] == function]


nio = errors("results/classic.csv", "F1")
other = errors("results/other.csv", "F1")  # the other method's CSV, with the same columns
print(compare(nio, other))
```

## Citation

If you use NIO in your work, please cite the paper:

Ragab, M. G., Abdulkadir, S. J., Alhussian, H., Alqushaibi, A., Sumiea, E. H., Al-Selwi, S. M., Muneer, A., Ismail, A.
O. A., & Wu, J. (2026). *Nubian Ibex Optimizer: A Nature-inspired Metaheuristic Algorithm* [Preprint]. Research
Square. https://doi.org/10.21203/rs.3.rs-11294841/v1

```bibtex
@misc{ragab2026nubian,
  title={Nubian Ibex Optimizer: A Nature-inspired Metaheuristic Algorithm},
  author={Ragab, Mohammed Gamal and Abdulkadir, Said Jadid and Alhussian, Hitham and Alqushaibi, Alawi and Sumiea, Ebrahim Hamid and Al-Selwi, Safwan Mahmood and Muneer, Amgad and Ismail, Ahmed Omer Ahmed and Wu, Jia},
  year={2026},
  month={oct},
  publisher={Research Square},
  note={Preprint, Version 1},
  doi={10.21203/rs.3.rs-11294841/v1},
  url={https://doi.org/10.21203/rs.3.rs-11294841/v1}
}
```

## Licence

MIT, copyright 2026 Mohammed Gamal. See `LICENSE`.
