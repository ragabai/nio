"""RC15–RC33: mechanical engineering problems."""

import cmath
import math
from functools import cache

import numpy as np

from problems.engineering.realworld.common import columns, mround, none, stack


def rc15(x):
    """Weight minimisation of a speed reducer."""
    c, n = columns(x), len(x)
    f = (
        0.7854 * c(1) * c(2) ** 2 * (3.3333 * c(3) ** 2 + 14.9334 * c(3) - 43.0934)
        - 1.508 * c(1) * (c(6) ** 2 + c(7) ** 2)
        + 7.477 * (c(6) ** 3 + c(7) ** 3)
        + 0.7854 * (c(4) * c(6) ** 2 + c(5) * c(7) ** 2)
    )
    g = stack(
        n,
        -c(1) * c(2) ** 2 * c(3) + 27,
        -c(1) * c(2) ** 2 * c(3) ** 2 + 397.5,
        -c(2) * c(6) ** 4 * c(3) * c(4) ** (-3) + 1.93,
        -c(2) * c(7) ** 4 * c(3) / c(5) ** 3 + 1.93,
        10 * c(6) ** (-3) * np.sqrt(16.91e6 + (745 * c(4) / (c(2) * c(3))) ** 2) - 1100,
        10 * c(7) ** (-3) * np.sqrt(157.5e6 + (745 * c(5) / (c(2) * c(3))) ** 2) - 850,
        c(2) * c(3) - 40,
        -c(1) / c(2) + 5,
        c(1) / c(2) - 12,
        1.5 * c(6) - c(4) + 1.9,
        1.1 * c(7) - c(5) + 1.9,
    )
    return f, g, none(n)


def rc16(x):
    """Optimal design of an industrial refrigeration system."""
    c, n = columns(x), len(x)
    f = (
        63098.88 * c(2) * c(4) * c(12)
        + 5441.5 * c(2) ** 2 * c(12)
        + 115055.5 * c(2) ** 1.664 * c(6)
        + 6172.27 * c(2) ** 2 * c(6)
        + 63098.88 * c(1) * c(3) * c(11)
        + 5441.5 * c(1) ** 2 * c(11)
        + 115055.5 * c(1) ** 1.664 * c(5)
        + 6172.27 * c(1) ** 2 * c(5)
        + 140.53 * c(1) * c(11)
        + 281.29 * c(3) * c(11)
        + 70.26 * c(1) ** 2
        + 281.29 * c(1) * c(3)
        + 281.29 * c(3) ** 2
        + 14437 * c(8) ** 1.8812 * c(12) ** 0.3424 * c(10) * c(14) ** (-1) * c(1) ** 2 * c(7) * c(9) ** (-1)
        + 20470.2 * c(7) ** 2.893 * c(11) ** 0.316 * c(1) ** 2
    )
    g = stack(
        n,
        1.524 * c(7) ** (-1) - 1,
        1.524 * c(8) ** (-1) - 1,
        0.07789 * c(1) - 2 * c(7) ** (-1) * c(9) - 1,
        7.05305 * c(9) ** (-1) * c(1) ** 2 * c(10) * c(8) ** (-1) * c(2) ** (-1) * c(14) ** (-1) - 1,
        0.0833 / c(13) * c(14) - 1,
        0.04771 * c(10) * c(8) ** 1.8812 * c(12) ** 0.3424 - 1,
        0.0488 * c(9) * c(7) ** 1.893 * c(11) ** 0.316 - 1,
        0.0099 * c(1) / c(3) - 1,
        0.0193 * c(2) / c(4) - 1,
        0.0298 * c(1) / c(5) - 1,
        47.136 * c(2) ** 0.333 / c(10) * c(12)
        - 1.333 * c(8) * c(13) ** 2.1195
        + 62.08 * c(13) ** 2.1195 * c(8) ** 0.2 / (c(12) * c(10))
        - 1,
        0.056 * c(2) / c(6) - 1,
        2 / c(9) - 1,
        2 / c(10) - 1,
        c(12) / c(11) - 1,
    )
    return f, g, none(n)


def rc17(x):
    """Tension/compression spring design (case 1)."""
    c, n = columns(x), len(x)
    f = c(1) ** 2 * c(2) * (c(3) + 2)
    g = stack(
        n,
        1 - (c(2) ** 3 * c(3)) / (71785 * c(1) ** 4),
        (4 * c(2) ** 2 - c(1) * c(2)) / (12566 * (c(2) * c(1) ** 3 - c(1) ** 4)) + 1 / (5108 * c(1) ** 2) - 1,
        1 - 140.45 * c(1) / (c(2) ** 2 * c(3)),
        (c(1) + c(2)) / 1.5 - 1,
    )
    return f, g, none(n)


def rc18(x):
    """Pressure vessel design; plate thicknesses come in multiples of 1/16 inch."""
    c, n = columns(x), len(x)
    shell, head, radius, length = 0.0625 * mround(c(1)), 0.0625 * mround(c(2)), c(3), c(4)
    f = (
        0.6224 * shell * radius * length
        + 1.7781 * head * radius**2
        + 3.1661 * shell**2 * length
        + 19.84 * shell**2 * radius
    )
    g = stack(
        n,
        -shell + 0.0193 * radius,
        -head + 0.00954 * radius,
        -np.pi * radius**2 * length - 4 / 3 * np.pi * radius**3 + 1296000,
        length - 240,
    )
    return f, g, none(n)


def rc19(x):
    """Welded beam design."""
    c, n = columns(x), len(x)
    f = 1.10471 * c(1) ** 2 * c(2) + 0.04811 * c(3) * c(4) * (14 + c(2))
    load, length, deflection_max, modulus, shear_modulus, shear_max, stress_max = (
        6000,
        14,
        0.25,
        30e6,
        12e6,
        13600,
        30000,
    )
    critical = (
        4.013
        * modulus
        * np.sqrt(c(3) ** 2 * c(4) ** 6 / 30)
        / length**2
        * (1 - c(3) / (2 * length) * np.sqrt(modulus / (4 * shear_modulus)))
    )
    stress = 6 * load * length / (c(4) * c(3) ** 2)
    deflection = 6 * load * length**3 / (modulus * c(3) ** 2 * c(4))
    polar = 2 * (np.sqrt(2) * c(1) * c(2) * (c(2) ** 2 / 4 + (c(1) + c(3)) ** 2 / 4))
    radius = np.sqrt(c(2) ** 2 / 4 + (c(1) + c(3)) ** 2 / 4)
    moment = load * (length + c(2) / 2)
    secondary = moment * radius / polar
    primary = load / (np.sqrt(2) * c(1) * c(2))
    shear = np.sqrt(primary**2 + 2 * primary * secondary * c(2) / (2 * radius) + secondary**2)
    g = stack(n, shear - shear_max, stress - stress_max, c(1) - c(4), deflection - deflection_max, load - critical)
    return f, g, none(n)


def rc20(x):
    """Three-bar truss design."""
    c, n = columns(x), len(x)
    f = (2 * np.sqrt(2) * c(1) + c(2)) * 100
    g = stack(
        n,
        (np.sqrt(2) * c(1) + c(2)) / (np.sqrt(2) * c(1) ** 2 + 2 * c(1) * c(2)) * 2 - 2,
        c(2) / (np.sqrt(2) * c(1) ** 2 + 2 * c(1) * c(2)) * 2 - 2,
        1 / (np.sqrt(2) * c(2) + c(1)) * 2 - 2,
    )
    return f, g, none(n)


def rc21(x):
    """Multiple disk clutch brake design."""
    c, n = columns(x), len(x)
    mf, ms, iz, speed, tmax, s, delta, vsr_max, rho, pmax, mu, lmax, del_r = (
        3,
        40,
        55,
        250,
        15,
        1.5,
        0.5,
        10,
        0.0000078,
        1,
        0.6,
        30,
        20,
    )
    rsr = 2 / 3 * (c(2) ** 3 - c(1) ** 3) / (c(2) ** 2 * c(1) ** 2)
    vsr = np.pi * rsr * speed / 30
    area = np.pi * (c(2) ** 2 - c(1) ** 2)
    prz = c(4) / area
    w = np.pi * speed / 30
    mh = 2 / 3 * mu * c(4) * c(5) * (c(2) ** 3 - c(1) ** 3) / (c(2) ** 2 - c(1) ** 2)
    t = iz * w / (mh + mf)
    f = np.pi * (c(2) ** 2 - c(1) ** 2) * c(3) * (c(5) + 1) * rho
    g = stack(
        n,
        -c(2) + c(1) + del_r,
        (c(5) + 1) * (c(3) + delta) - lmax,
        prz - pmax,
        prz * vsr - pmax * vsr_max,
        vsr - vsr_max,
        t - tmax,
        s * ms - mh,
        -t,
    )
    return f, g, none(n)


def rc22(x):
    """Planetary gear train design; every variable is an integer, the last three index tables."""
    n = len(x)
    x = mround(np.abs(x))
    c = columns(x)
    n1, n2, n3, n4, n5, n6 = (c(k) for k in range(1, 7))
    p = np.array([3, 4, 5])[np.clip(c(7).astype(int), 1, 3) - 1]
    modules = np.array([1.75, 2, 2.25, 2.5, 2.75, 3.0])
    m1, m2 = modules[np.clip(c(8).astype(int), 1, 6) - 1], modules[np.clip(c(9).astype(int), 1, 6) - 1]
    i1, i2, ir = n6 / n4, n6 * (n1 * n3 + n2 * n4) / (n1 * n3 * (n6 - n4)), -(n2 * n6 / (n1 * n3))
    f = np.maximum(np.maximum(i1 - 3.11, i2 - 1.84), ir + 3.11)
    dmax, dlt = 220, 0.5
    cosine = ((n6 - n3) ** 2 + (n4 + n5) ** 2 - (n3 + n5) ** 2) / (2 * (n6 - n3) * (n4 + n5))
    beta = np.arccos(np.clip(cosine, -1, 1))
    clearance = (n3 + n5 + 2 + dlt) ** 2 - (
        (n6 - n3) ** 2 + (n4 + n5) ** 2 - 2 * (n6 - n3) * (n4 + n5) * np.cos(2 * np.pi / p - beta)
    )
    g = stack(
        n,
        m2 * (n6 + 2.5) - dmax,
        m1 * (n1 + n2) + m1 * (n2 + 2) - dmax,
        m2 * (n4 + n5) + m2 * (n5 + 2) - dmax,
        np.abs(m1 * (n1 + n2) - m2 * (n6 - n3)) - m1 - m2,
        -((n1 + n2) * np.sin(np.pi / p) - n2 - 2 - dlt),
        -((n6 - n3) * np.sin(np.pi / p) - n3 - 2 - dlt),
        -((n4 + n5) * np.sin(np.pi / p) - n5 - 2 - dlt),
        np.where(np.abs(cosine) > 1, 1e6, clearance),  # the planets cannot be placed: acos would be complex
        -(n6 - 2 * n3 - n4 - 4 - 2 * dlt),
        -(n6 - n4 - 2 * n5 - 4 - 2 * dlt),
    )
    return f, g, stack(n, np.fmod(n6 - n4, p))


def rc23(x):
    """Step-cone pulley design."""
    c, n = columns(x), len(x)
    d = [c(k) * 1e-3 for k in range(1, 5)]
    w = c(5) * 1e-3
    speed, speeds = 350, (750, 450, 250, 150)
    rho, a, mu, s, t = 7200, 3, 0.35, 1.75e6, 8e-3
    f = rho * w * np.pi / 4 * sum(di**2 * (1 + (ni / speed) ** 2) for di, ni in zip(d, speeds, strict=True))
    lengths = [
        np.pi * di / 2 * (1 + ni / speed) + (ni / speed - 1) ** 2 * di**2 / (4 * a) + 2 * a
        for di, ni in zip(d, speeds, strict=True)
    ]
    wraps = [np.pi - 2 * np.arcsin((ni / speed - 1) * di / (2 * a)) for di, ni in zip(d, speeds, strict=True)]
    ratios = [np.exp(mu * wrap) for wrap in wraps]
    powers = [
        s * t * w * (1 - np.exp(-mu * wrap)) * np.pi * di * ni / 60
        for wrap, di, ni in zip(wraps, d, speeds, strict=True)
    ]
    g = stack(n, *(-ratio + 2 for ratio in ratios), *(-power + 0.75 * 745.6998 for power in powers))
    h = stack(n, lengths[0] - lengths[1], lengths[0] - lengths[2], lengths[0] - lengths[3])
    return f, g, h


# RC24 robot gripper: the objective is the range of the gripping force over the actuator's stroke, which the
# organisers find with MATLAB's fminbnd. Impossible linkages make acos complex; such designs score 1e4.

STROKE = 99.9999


def _acos(z: complex) -> complex:
    """MATLAB's acos: arguments beyond ±1 return complex values with MATLAB's sign of the imaginary part."""
    if abs(z) > 1 or z.imag != 0:
        return 1j * abs(cmath.acos(z)) if z.real == abs(z) else cmath.acos(z)
    return complex(math.acos(z.real))


def _force(v: np.ndarray, z: complex, sign: int) -> complex:
    a, b, c, e, length = complex(v[0]), complex(v[1]), complex(v[2]), complex(v[3]), complex(v[5])
    reach = cmath.sqrt(e * e + (z - length) * (z - length))
    phi = cmath.atan(e / (length - z))
    alpha = _acos((a * a + reach * reach - b * b) / (2 * a * reach))
    beta = _acos((b * b + reach * reach - a * a) / (2 * b * reach))
    return sign * 100 * b * cmath.sin(alpha + beta) / (2 * c * cmath.cos(alpha + phi))


def _fminbnd(v: np.ndarray, sign: int, low: float = 0.0, high: float = STROKE, tol: float = 1e-4) -> complex:
    """MATLAB's fminbnd (golden section with parabolic steps), in complex arithmetic as MATLAB runs it here."""
    seps, golden = math.sqrt(2.220446049250313e-16), 0.5 * (3.0 - math.sqrt(5.0))
    a, b = complex(low), complex(high)
    v_ = w = xf = a + golden * (b - a)
    d = e = 0j
    fx = fv = fw = _force(v, xf, sign)
    count = 1
    xm = 0.5 * (a + b)
    tol1 = seps * abs(xf) + tol / 3
    tol2 = 2 * tol1
    while abs(xf - xm) > (tol2 - 0.5 * (b - a)).real:
        golden_step = True
        if abs(e) > tol1:
            golden_step = False
            r = (xf - w) * (fx - fv)
            q = (xf - v_) * (fx - fw)
            p = (xf - v_) * q - (xf - w) * r
            q = 2 * (q - r)
            if q.real > 0:
                p = -p
            q = abs(q)
            r, e = e, d
            if abs(p) < abs(0.5 * q * r) and p.real > (q * (a - xf)).real and p.real < (q * (b - xf)).real:
                d = p / q
                x = xf + d
                if (x - a).real < tol2 or (b - x).real < tol2:
                    d = tol1 * ((xm - xf) / abs(xm - xf) + float((xm - xf).real == 0))
            else:
                golden_step = True
        if golden_step:
            e = a - xf if xf.real >= xm.real else b - xf
            d = golden * e
        direction = (d / abs(d) if abs(d) else 0) + float(abs(d) == 0)
        x = xf + direction * max(abs(d), tol1)
        fu = _force(v, x, sign)
        count += 1
        if fu.real <= fx.real:
            if x.real >= xf.real:
                a = xf
            else:
                b = xf
            v_, fv, w, fw, xf, fx = w, fw, xf, fx, x, fu
        else:
            if x.real < xf.real:
                a = x
            else:
                b = x
            if fu.real <= fw.real or w == xf:
                v_, fv, w, fw = w, fw, x, fu
            elif fu.real <= fv.real or v_ in (xf, w):
                v_, fv = x, fu
        xm = 0.5 * (a + b)
        tol1 = seps * abs(xf) + tol / 3
        tol2 = 2 * tol1
        if count >= 500:
            break
    return fx


def _gripper(v: np.ndarray) -> tuple[float, list[float]]:
    a, b, c, e, ff, length, delta = (complex(value) for value in v)
    lowest, highest = _fminbnd(v, -1), _fminbnd(v, 1)  # min of −F and min of F
    total = -lowest - highest
    f = 1e4 if total.imag != 0 else -lowest.real - highest.real

    def angle(stroke: float) -> complex:
        reach = cmath.sqrt((length - stroke) ** 2 + e * e)
        cosine = (b * b + (length - stroke) ** 2 + e * e - a * a) / (2 * b * reach)
        if abs(cosine.real) > 1:
            return cmath.acos(cosine) - cmath.atan(e / (length - stroke))
        return complex(math.acos(cosine.real) - math.atan((e / (length - stroke)).real))

    beta_0, beta_m = angle(0.0), angle(STROKE)
    y_min = 2 * (e + ff + c * cmath.sin(beta_m + delta))
    y_max = 2 * (e + ff + c * cmath.sin(beta_0 + delta))
    values = [
        y_min - 50,
        -y_min,
        100 - y_max,
        y_max - 150,
        length**2 + e**2 - (a + b) ** 2,
        b**2 - (a - e) ** 2 - (length - STROKE) ** 2,
        STROKE - length,
    ]
    return f, [1e4 if abs(value.imag) > 1e-10 else value.real for value in values]


def rc24(x):
    """Robot gripper design."""
    results = [_gripper(v) for v in x]
    return np.array([f for f, _ in results]), np.array([g for _, g in results]), none(len(x))


def rc25(x):
    """Hydrostatic thrust bearing design."""
    c, n = columns(x), len(x)
    radius, recess, viscosity, flow = c(1), c(2), c(3), c(4)
    gamma, constant, exponent, c1, load, pmax, rise_max, hmin, gravity, speed = (
        0.0307,
        0.5,
        -3.55,
        10.04,
        101000,
        1000,
        50,
        0.001,
        386.4,
        750,
    )
    p = (np.log10(np.log10(8.122e6 * viscosity + 0.8)) - c1) / exponent
    rise = 2 * (10**p - 560)
    friction = 9336 * flow * gamma * constant * rise
    film = (2 * np.pi * speed / 60) ** 2 * 2 * np.pi * viscosity / friction * (radius**4 / 4 - recess**4 / 4) - 1e-5
    inlet = (6 * viscosity * flow / (np.pi * film**3)) * np.log(radius / recess)
    capacity = np.pi * inlet / 2 * (radius**2 - recess**2) / (np.log(radius / recess) - 1e-5)
    f = (flow * inlet / 0.7 + friction) / 12
    g = stack(
        n,
        load - capacity,
        inlet - pmax,
        rise - rise_max,
        hmin - film,
        recess - radius,
        gamma / (gravity * inlet) * (flow / (2 * np.pi * radius * film)) - 0.001,
        capacity / (np.pi * (radius**2 - recess**2) + 1e-5) - 5000,
    )
    return f, g, none(n)


def rc26(x):
    """Four-stage gear box; gear teeth are integers and widths and positions index tables."""
    n = len(x)
    x = mround(x)
    c = columns(x)
    np1, ng1, np2, ng2, np3, ng3, np4, ng4 = (c(k) for k in range(1, 9))
    widths = np.array([3.175, 5.715, 8.255, 12.7])
    b1, b2, b3, b4 = (widths[np.clip(c(k).astype(int), 1, 4) - 1] for k in range(9, 13))
    grid = np.array([12.7, 25.4, 38.1, 50.8, 63.5, 76.2, 88.9, 101.6, 114.3])
    xp1, xg1, xg2, xg3, xg4, yp1, yg1, yg2, yg3, yg4 = (
        grid[np.clip(c(k).astype(int), 1, 9) - 1] for k in range(13, 23)
    )
    c1 = np.sqrt((xg1 - xp1) ** 2 + (yg1 - yp1) ** 2)
    c2 = np.sqrt((xg2 - xp1) ** 2 + (yg2 - yp1) ** 2)
    c3 = np.sqrt((xg3 - xp1) ** 2 + (yg3 - yp1) ** 2)
    c4 = np.sqrt((xg4 - xp1) ** 2 + (yg4 - yp1) ** 2)
    cr_min, d_min, phi, w, jr, km, ko, lmax, sigma_h, sigma_n, w1, w_min, w_max, cp = (
        1.4,
        25.4,
        20 * np.pi / 180,
        55.9,
        0.2,
        1.6,
        1.5,
        127,
        3290,
        2090,
        5000,
        245,
        255,
        464,
    )
    pinions, gears, widths_, centres = (np1, np2, np3, np4), (ng1, ng2, ng3, ng4), (b1, b2, b3, b4), (c1, c2, c3, c4)
    f = (
        np.pi
        / 1000
        * sum(
            bi * ci**2 * (pi**2 + gi**2) / (pi + gi) ** 2
            for bi, ci, pi, gi in zip(widths_, centres, pinions, gears, strict=True)
        )
    )
    speeds = (
        366000 / (np.pi * w1),
        366000 * ng1 / (np.pi * w1 * np1),
        366000 * ng1 * ng2 / (np.pi * w1 * np1 * np2),
        366000 * ng1 * ng2 * ng3 / (np.pi * w1 * np1 * np2 * np3),
    )
    bending = [
        (sp + 2 * ci * pi / (pi + gi)) * ((pi + gi) ** 2 / (4 * bi * ci**2 * pi))
        - sigma_n * jr / (0.0167 * w * ko * km)
        for sp, ci, pi, gi, bi in zip(speeds, centres, pinions, gears, widths_, strict=True)
    ]
    contact = [
        (sp + 2 * ci * pi / (pi + gi)) * ((pi + gi) ** 3 / (4 * bi * ci**2 * gi * pi**2))
        - (sigma_h / cp) ** 2 * (np.sin(phi) * np.cos(phi)) / (0.0334 * w * ko * km)
        for sp, ci, pi, gi, bi in zip(speeds, centres, pinions, gears, widths_, strict=True)
    ]
    ratio = [
        cr_min * np.pi * np.cos(phi)
        - pi * np.sqrt(np.sin(phi) ** 2 / 4 + 1 / pi + (1 / pi) ** 2)
        - gi * np.sqrt(np.sin(phi) ** 2 / 4 + 1 / gi + (1 / gi) ** 2)
        + np.sin(phi) * (pi + gi) / 2
        for pi, gi in zip(pinions, gears, strict=True)
    ]
    pinion_size = [d_min - 2 * ci * pi / (pi + gi) for ci, pi, gi in zip(centres, pinions, gears, strict=True)]
    gear_size = [d_min - 2 * ci * gi / (pi + gi) for ci, pi, gi in zip(centres, pinions, gears, strict=True)]
    pinion_x, pinion_y = (xp1, xg2, xg3, xg4), (yp1, yg2, yg3, yg4)
    gear_x, gear_y = (xg1, xg2, xg3, xg4), (yg1, yg2, yg3, yg4)
    reach_p = [(pi + 2) * ci / (pi + gi) for ci, pi, gi in zip(centres, pinions, gears, strict=True)]
    reach_g = [(gi + 2) * ci / (pi + gi) for ci, pi, gi in zip(centres, pinions, gears, strict=True)]
    fit = [
        *(px + rp - lmax for px, rp in zip(pinion_x, reach_p, strict=True)),
        *(-px + rp for px, rp in zip(pinion_x, reach_p, strict=True)),
        *(py + rp - lmax for py, rp in zip(pinion_y, reach_p, strict=True)),
        *(-py + rp for py, rp in zip(pinion_y, reach_p, strict=True)),
        *(gx + rg - lmax for gx, rg in zip(gear_x, reach_g, strict=True)),
        *(-gx + rg for gx, rg in zip(gear_x, reach_g, strict=True)),
        *(gy + rg - lmax for gy, rg in zip(gear_y, reach_g, strict=True)),
        *(-gy + rg for gy, rg in zip(gear_y, reach_g, strict=True)),
    ]
    sums = [pi + gi for pi, gi in zip(pinions, gears, strict=True)]
    sums_misprinted = [
        np1 + ng1,
        np2 + ng2,
        np2 + ng3,
        np3 + ng4,
    ]  # the organisers' code pairs np2/ng3 and np3/ng4 in two rows
    table = [
        *(
            (0.945 * ci - s) * (bi - 5.715) * (bi - 8.255) * (bi - 12.70) * (-1)
            for ci, s, bi in zip(centres, sums, widths_, strict=True)
        ),
        *(
            (0.646 * ci - s) * (bi - 3.175) * (bi - 8.255) * (bi - 12.70)
            for ci, s, bi in zip(centres, sums, widths_, strict=True)
        ),
        *(
            (0.504 * ci - s) * (bi - 3.175) * (bi - 5.715) * (bi - 12.70) * (-1)
            for ci, s, bi in zip(centres, sums, widths_, strict=True)
        ),
        *(
            (0.0 * ci - s) * (bi - 3.175) * (bi - 5.715) * (bi - 8.255)
            for ci, s, bi in zip(centres, sums, widths_, strict=True)
        ),
        *(
            (-1.812 * ci + s) * (bi - 5.715) * (bi - 8.255) * (bi - 12.70) * (-1)
            for ci, s, bi in zip(centres, sums, widths_, strict=True)
        ),
        *(
            (-0.945 * ci + s) * (bi - 3.175) * (bi - 8.255) * (bi - 12.70)
            for ci, s, bi in zip(centres, sums, widths_, strict=True)
        ),
        *(
            (-0.646 * ci + s) * (bi - 3.175) * (bi - 5.715) * (bi - 12.70) * (-1)
            for ci, s, bi in zip(centres, sums_misprinted, widths_, strict=True)
        ),
        *(
            (-0.504 * ci + s) * (bi - 3.175) * (bi - 5.715) * (bi - 8.255)
            for ci, s, bi in zip(centres, sums, widths_, strict=True)
        ),
    ]
    output = w1 * (np1 * np2 * np3 * np4) / (ng1 * ng2 * ng3 * ng4)
    g = stack(n, *bending, *contact, *ratio, *pinion_size, *gear_size, *fit, *table, w_min - output, -w_max + output)
    g[np.isinf(g)] = 1e6
    return f, g, none(n)


# RC27 ten-bar truss: minimise the weight subject to lower bounds on the first three natural frequencies.

TRUSS_NODES = np.array([[18.288, 9.144], [18.288, 0], [9.144, 9.144], [9.144, 0], [0, 9.144], [0, 0]])
TRUSS_BARS = np.array([[3, 5], [1, 3], [4, 6], [2, 4], [3, 4], [1, 2], [4, 5], [3, 6], [2, 3], [1, 4]]) - 1
TRUSS_FIXED = [8, 9, 10, 11]  # both degrees of freedom of nodes 5 and 6


def rc27(x):
    """Ten-bar truss design with frequency constraints."""
    n = len(x)
    modulus, rho, added = 6.98e10, 2770, 454
    start, end = TRUSS_NODES[TRUSS_BARS[:, 0]], TRUSS_NODES[TRUSS_BARS[:, 1]]
    lengths = np.linalg.norm(end - start, axis=1)
    cosines = (end - start) / lengths[:, None]
    f = (rho * lengths * x).sum(axis=1)
    stiffness, mass = np.zeros((n, 12, 12)), np.zeros((n, 12, 12))
    lumped = np.array([[2, 0, 1, 0], [0, 2, 0, 1], [1, 0, 2, 0], [0, 1, 0, 2]]) / 6
    for bar, (i, j) in enumerate(TRUSS_BARS):
        transform = np.array([[*cosines[bar], 0, 0], [0, 0, *cosines[bar]]])
        unit = transform.T @ np.array([[1, -1], [-1, 1]]) @ transform / lengths[bar]
        index = np.ix_(range(n), [2 * i, 2 * i + 1, 2 * j, 2 * j + 1], [2 * i, 2 * i + 1, 2 * j, 2 * j + 1])
        stiffness[index] += (x[:, bar] * modulus)[:, None, None] * unit
        mass[index] += (rho * lengths[bar] * x[:, bar])[:, None, None] * lumped
    mass += added * np.eye(12)
    free = [k for k in range(12) if k not in TRUSS_FIXED]
    k, m = stiffness[:, free][:, :, free], mass[:, free][:, :, free]
    lower = np.linalg.cholesky(m)
    inverse = np.linalg.inv(lower)
    eigenvalues = np.linalg.eigvalsh(inverse @ k @ np.swapaxes(inverse, 1, 2))
    frequencies = np.sqrt(eigenvalues) / (2 * np.pi)
    g = stack(n, 7 / frequencies[:, 0] - 1, 15 / frequencies[:, 1] - 1, 20 / frequencies[:, 2] - 1)
    return f, g, none(n)


def rc28(x):
    """Rolling element bearing design."""
    c, n = columns(x), len(x)
    dm, db, balls, fi, fo, kd_min, kd_max, eps, e, chi = (
        c(1),
        c(2),
        mround(c(3)),
        c(4),
        c(5),
        c(6),
        c(7),
        c(8),
        c(9),
        c(10),
    )
    outer, inner, width = 160, 90, 30
    t = outer - inner - 2 * db
    phi = 2 * np.pi - 2 * np.arccos(
        (((outer - inner) * 0.5 - 0.75 * t) ** 2 + (0.5 * outer - 0.25 * t - db) ** 2 - (0.5 * inner + 0.25 * t) ** 2)
        / (2 * (0.5 * (outer - inner) - 0.75 * t) * (0.5 * outer - 0.25 * t - db))
    )
    gamma = db / dm
    fc = (
        37.91
        * (
            1
            + (1.04 * ((1 - gamma) / (1 + gamma)) ** 1.72 * (fi * (2 * fo - 1) / (fo * (2 * fi - 1))) ** 0.41)
            ** (10 / 3)
        )
        ** (-0.3)
        * (gamma**0.3 * (1 - gamma) ** 1.39 / (1 + gamma) ** (1 / 3))
        * (2 * fi / (2 * fi - 1)) ** 0.41
    )
    f = np.where(db > 25.4, 3.647 * fc * balls ** (2 / 3) * db**1.4, fc * balls ** (2 / 3) * db**1.8)
    g = stack(
        n,
        balls - 1 - phi / (2 * np.arcsin(db / dm)),
        kd_min * (outer - inner) - 2 * db,
        2 * db - kd_max * (outer - inner),
        chi * width - db,
        0.5 * (outer + inner) - dm,
        dm - (0.5 + e) * (outer + inner),
        eps * db - 0.5 * (outer - dm - db),
        0.515 - fi,
        0.515 - fo,
    )
    return f, g, none(n)


def rc29(x):
    """Gas transmission compressor design."""
    c, n = columns(x), len(x)
    f = (
        8.61e5 * c(1) ** 0.5 * c(2) * c(3) ** (-2 / 3) * c(4) ** (-1 / 2)
        + 3.69e4 * c(3)
        + 7.72e8 * c(1) ** (-1) * c(2) ** 0.219
        - 765.43e6 * c(1) ** (-1)
    )
    return f, stack(n, c(4) * c(2) ** (-2) + c(2) ** (-2) - 1), none(n)


# standard wire diameters as the organisers list them (0.0105 included)
WIRE = np.array([
    0.009, 0.0095, 0.0104, 0.0118, 0.0128, 0.0132, 0.014, 0.015, 0.0162, 0.0173, 0.018, 0.020, 0.023, 0.025, 0.028,
    0.032, 0.035, 0.041, 0.047, 0.054, 0.063, 0.072, 0.080, 0.092, 0.0105, 0.120, 0.135, 0.148, 0.162, 0.177, 0.192,
    0.207, 0.225, 0.244, 0.263, 0.283, 0.307, 0.331, 0.362, 0.394, 0.4375, 0.500,
])  # fmt: skip


def rc30(x):
    """Tension/compression spring design (case 2): integer coils and a wire diameter from a table."""
    c, n = columns(x), len(x)
    coils, diameter, wire = mround(c(1)), c(2), WIRE[np.clip(mround(c(3)).astype(int), 1, len(WIRE)) - 1]
    f = (np.pi**2 * diameter * wire**2 * (coils + 2)) / 4
    cf = (4 * diameter / wire - 1) / (4 * diameter / wire - 4) + 0.615 * wire / diameter
    k = (11.5e6 * wire**4) / (8 * coils * diameter**3)
    free_length = 1000 / k + 1.05 * (coils + 2) * wire
    preload = 300 / k
    g = stack(
        n,
        (8000 * cf * diameter) / (np.pi * wire**3) - 189000,
        free_length - 14,
        0.2 - wire,
        diameter - 3,
        3 - diameter / wire,
        preload - 6,
        preload + 700 / k + 1.05 * (coils + 2) * wire - free_length,
        1.25 - 700 / k,
    )
    return f, g, none(n)


def rc31(x):
    """Gear train design."""
    c, n = columns(x), len(x)
    return (1 / 6.931 - (c(1) * c(2)) / (c(3) * c(4))) ** 2, none(n), none(n)


def rc32(x):
    """Himmelblau's function."""
    c, n = columns(x), len(x)
    f = 5.3578547 * c(3) ** 2 + 0.8356891 * c(1) * c(5) + 37.293239 * c(1) - 40792.141
    g1 = 85.334407 + 0.0056858 * c(2) * c(5) + 0.0006262 * c(1) * c(4) - 0.0022053 * c(3) * c(5)
    g2 = 80.51249 + 0.0071317 * c(2) * c(5) + 0.0029955 * c(1) * c(2) + 0.0021813 * c(3) ** 2
    g3 = 9.300961 + 0.0047026 * c(3) * c(5) + 0.0012547 * c(1) * c(3) + 0.0019085 * c(3) * c(4)
    return f, stack(n, g1 - 92, -g1, g2 - 110, -g2 + 90, g3 - 25, -g3 + 20), none(n)


# RC33 topology optimisation: a 3 × 10 mesh of element densities (column-major, MATLAB's X = reshape(x, 10, 3))
# loaded at its free corner; minimise compliance. The constraints are the filtered sensitivities.

ELEMENTS_X, ELEMENTS_Y, PENALTY, RADIUS = 3, 10, 3, 1.5


@cache
def _element_stiffness() -> np.ndarray:
    modulus, nu = 206000000.0, 0.3
    k = [
        1 / 2 - nu / 6,
        1 / 8 + nu / 8,
        -1 / 4 - nu / 12,
        -1 / 8 + 3 * nu / 8,
        -1 / 4 + nu / 12,
        -1 / 8 - nu / 8,
        nu / 6,
        1 / 8 - 3 * nu / 8,
    ]
    order = [[0, 1, 2, 3, 4, 5, 6, 7], [1, 0, 7, 6, 5, 4, 3, 2], [2, 7, 0, 5, 6, 3, 4, 1], [3, 6, 5, 0, 7, 2, 1, 4]]
    order += [[4, 5, 6, 7, 0, 1, 2, 3], [5, 4, 3, 2, 1, 0, 7, 6], [6, 3, 4, 1, 2, 7, 0, 5], [7, 2, 1, 4, 3, 6, 5, 0]]
    return modulus / (1 - nu**2) * np.array(k)[np.array(order)]


@cache
def _mesh() -> tuple[np.ndarray, np.ndarray]:
    """Degrees of freedom of every element (elements in column-major order) and the mesh filter's weights."""
    dofs = []
    for elx in range(1, ELEMENTS_X + 1):
        for ely in range(1, ELEMENTS_Y + 1):
            n1, n2 = (ELEMENTS_Y + 1) * (elx - 1) + ely, (ELEMENTS_Y + 1) * elx + ely
            dofs.append(
                np.array([2 * n1 - 1, 2 * n1, 2 * n2 - 1, 2 * n2, 2 * n2 + 1, 2 * n2 + 2, 2 * n1 + 1, 2 * n1 + 2]) - 1
            )
    centres = np.array([(i, j) for i in range(ELEMENTS_X) for j in range(ELEMENTS_Y)])
    weights = np.maximum(0.0, RADIUS - np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=-1))
    weights[np.abs(centres[:, None, :] - centres[None, :, :]).max(axis=-1) > np.floor(RADIUS)] = 0.0
    return np.array(dofs), weights


def rc33(x):
    """Topology optimisation of a cantilever."""
    n = len(x)
    ke, (dofs, weights) = _element_stiffness(), _mesh()
    total = 2 * (ELEMENTS_X + 1) * (ELEMENTS_Y + 1)
    stiffness = np.zeros((n, total, total))
    for element, dof in enumerate(dofs):
        stiffness[:, dof[:, None], dof[None, :]] += (x[:, element] ** PENALTY)[:, None, None] * ke
    free = np.arange(2 * (ELEMENTS_Y + 1), total)
    load = np.zeros(total)
    load[-1] = -10000
    displacement = np.zeros((n, total))
    displacement[:, free] = np.linalg.solve(
        stiffness[:, free][:, :, free], np.broadcast_to(load[free], (n, len(free)))[..., None]
    )[..., 0]
    local = displacement[:, dofs]  # (n, elements, 8)
    energy = np.einsum("nei,ij,nej->ne", local, ke, local)
    f = (x**PENALTY * energy).sum(axis=1)
    sensitivity = -PENALTY * x ** (PENALTY - 1) * energy
    filtered = (x * sensitivity) @ weights.T / (x * weights.sum(axis=1))
    return f, filtered, none(n)
