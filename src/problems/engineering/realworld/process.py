"""RC08–RC14: process synthesis and design problems; integer variables are rounded as the organisers do."""

import numpy as np

from problems.engineering.realworld.common import columns, mround, none, stack


def rc08(x):
    """Process synthesis problem."""
    c, n = columns(x), len(x)
    x1, x2 = c(1), mround(c(2))
    return 2 * x1 + x2, stack(n, 1.25 - x1**2 - x2, x1 + x2 - 1.6), none(n)


def rc09(x):
    """Process synthesis and design problem."""
    c, n = columns(x), len(x)
    x1, x2, x3 = c(1), c(2), mround(c(3))
    return -x3 + 2 * x1 + x2, stack(n, -x1 + x2 + x3), stack(n, x1 - 2 * np.exp(-x2))


def rc10(x):
    """Process flow sheeting problem."""
    c, n = columns(x), len(x)
    x1, x2, x3 = c(1), c(2), mround(c(3))
    f = -0.7 * x3 + 5 * (x1 - 0.5) ** 2 + 0.8
    return f, stack(n, -np.exp(x1 - 0.2) - x2, x2 + 1.1 * x3 + 1, x1 - x3 - 0.2), none(n)


def rc11(x):
    """Two-reactor problem."""
    c, n = columns(x), len(x)
    x1, x2, vol1, vol2, y1, y2, total = c(1), c(2), c(3), c(4), mround(c(5)), mround(c(6)), c(7)
    z1 = 0.9 * (1 - np.exp(-0.5 * vol1)) * x1
    z2 = 0.8 * (1 - np.exp(-0.4 * vol2)) * x2
    f = 7.5 * y1 + 5.5 * y2 + 7 * vol1 + 6 * vol2 + 5 * total
    h = stack(n, y1 + y2 - 1, z1 + z2 - 10, x1 + x2 - total, z1 * y1 + z2 * y2 - 10)
    g = stack(n, vol1 - 10 * y1, vol2 - 10 * y2, x1 - 20 * y1, x2 - 20 * y2)
    return f, g, h


def rc12(x):
    """Process synthesis problem."""
    c, n = columns(x), len(x)
    x1, x2, x3 = c(1), c(2), c(3)
    y1, y2, y3, y4 = (mround(c(k)) for k in (4, 5, 6, 7))
    f = (y1 - 1) ** 2 + (y2 - 1) ** 2 + (y3 - 1) ** 2 - np.log(y4 + 1) + (x1 - 1) ** 22 + (x2 - 2) ** 2 + (x3 - 3) ** 2
    g = stack(
        n,
        x1 + x2 + x3 + y1 + y2 + y3 - 5,
        y3**2 + x1**2 + x2**2 + x3**2 - 5.5,
        x1 + y1 - 1.2,
        x2 + y2 - 1.8,
        x3 + y3 - 2.5,
        x1 + y4 - 1.2,
        y2**2 + x2**2 - 1.64,
        y3**2 + x3**2 - 4.25,
        y2**2 + x3**2 - 4.64,
    )
    return f, g, none(n)


def rc13(x):
    """Process design problem."""
    c, n = columns(x), len(x)
    x1, x2, x3, y1, y2 = c(1), c(2), c(3), mround(c(4)), mround(c(5))
    f = -5.357854 * x1**2 - 0.835689 * y1 * x3 - 37.29329 * y1 + 40792.141
    a = [
        85.334407,
        0.0056858,
        0.0006262,
        0.0022053,
        80.51249,
        0.0071317,
        0.0029955,
        0.0021813,
        9.300961,
        0.0047026,
        0.0012547,
        0.0019085,
    ]
    g = stack(
        n,
        a[0] + a[1] * y2 * x3 + a[2] * y1 * x2 - a[3] * y1 * y1 * x3 - 92,
        a[4] + a[5] * y2 * x3 + a[6] * y1 * x2 + a[7] * x1**2 - 90 - 20,
        a[8] + a[9] * y1 * x2 + a[10] * y1 * x1 + a[11] * x1 * x2 - 20 - 5,
    )
    return f, g, none(n)


def rc14(x):
    """Multi-product batch plant."""
    c, n = columns(x), len(x)
    s = np.array([[2, 3, 4], [4, 6, 3]])
    t = np.array([[8, 20, 8], [16, 4, 4]])
    hours, alpha, beta, q1, q2 = 6000, 250, 0.6, 40000, 20000
    n1, n2, n3 = mround(c(1)), mround(c(2)), mround(c(3))
    vol1, vol2, vol3, tl1, tl2, b1, b2 = c(4), c(5), c(6), c(7), c(8), c(9), c(10)
    f = alpha * (n1 * vol1**beta + n2 * vol2**beta + n3 * vol3**beta)
    g = stack(
        n,
        q1 * tl1 / b1 + q2 * tl2 / b2 - hours,
        s[0, 0] * b1 + s[1, 0] * b2 - vol1,
        s[0, 1] * b1 + s[1, 1] * b2 - vol2,
        s[0, 2] * b1 + s[1, 2] * b2 - vol3,
        t[0, 0] - n1 * tl1,
        t[0, 1] - n2 * tl1,
        t[0, 2] - n3 * tl1,
        t[1, 0] - n1 * tl2,
        t[1, 1] - n2 * tl2,
        t[1, 2] - n3 * tl2,
    )
    return f, g, none(n)
