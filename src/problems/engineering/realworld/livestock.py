"""RC51–RC57: least-cost livestock feed rations meeting nutrient requirements (beef cattle, then dairy cattle)."""

from problems.engineering.realworld.common import data, none, stack


def _beef(x, limits: list[tuple[float, float]], energy: float):
    """Cost of the ration; each nutrient (rows 5, 4, 6, 7 of the feed table, then the roughage share, the
    share-weighted row 3 and the grain mass) between its bounds; row 2 exactly `energy`."""
    n, table = len(x), data()["RM_feed"]
    total = x.sum(axis=1)
    measures = [
        x @ table[4],
        x @ table[3],
        x @ table[5],
        x @ table[6],
        x[:, :17].sum(axis=1) / total,
        (x @ table[2]) / total,
        x[:, 33:59].sum(axis=1),
    ]
    g = stack(
        n,
        *(
            bound
            for measure, (low, high) in zip(measures, limits, strict=True)
            for bound in (-measure + low, measure - high)
        ),
    )
    return x @ table[0], g, stack(n, x @ table[1] - energy)


def rc51(x):
    """Beef cattle ration (case 1)."""
    return _beef(
        x,
        [(1.090, 2.170), (4.870, 5.200), (0.043, 0.086), (0.023, 0.046), (0.295, 0.36), (0.3, 0.4712), (9.2, 11.5)],
        6.9,
    )


def rc52(x):
    """Beef cattle ration (case 2)."""
    return _beef(
        x, [(1.280, 2.560), (7.300, 7.810), (0.005, 0.094), (0.031, 0.062), (0.2, 0.24), (0.3, 0.4), (9.8, 16.4)], 9.8
    )


def rc53(x):
    """Beef cattle ration (case 3)."""
    return _beef(
        x,
        [(1.170, 2.340), (6.940, 7.430), (0.038, 0.076), (0.034, 0.068), (0.085, 0.111), (0.25, 0.4), (11.6, 14.5)],
        8.7,
    )


def rc54(x):
    """Beef cattle ration (case 4)."""
    return _beef(
        x,
        [(0.56, 1.12), (3.23, 3.46), (0.018, 0.036), (0.0116, 0.040), (0.25, 0.9), (0.3, 0.4384), (7.470, 9.340)],
        5.6,
    )


def _dairy(x, targets: tuple[float, ...]):
    """Cost of the ration with rows 12, 2, 3, 13, 14 of the dairy table and the summed rows 2–11 at their targets."""
    n, table = len(x), data()["RM_dairy"]
    measures = [x @ table[11], x @ table[1], x @ table[2], x @ table[12], x @ table[13], x @ table[1:11].sum(axis=0)]
    return (
        x @ table[0],
        none(n),
        stack(n, *(measure - target for measure, target in zip(measures, targets, strict=True))),
    )


def rc55(x):
    """Dairy cattle ration (case 1)."""
    return _dairy(x, (25.67, 0.0218, 0.062, 0.034, 0.021, 0.999))


def rc56(x):
    """Dairy cattle ration (case 2)."""
    return _dairy(x, (65.24, 0.066, 0.159, 0.103, 0.052, 2.644))


def rc57(x):
    """Dairy cattle ration (case 3)."""
    return _dairy(x, (30.05, 0.0259, 0.077, 0.096, 0.025, 1.214))
