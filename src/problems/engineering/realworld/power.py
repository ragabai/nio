"""RC34–RC44: power system problems. Bus voltages are decision variables, and the power-flow equations are the
equality constraints, as current mismatches (I = Y·V against the specified injections) and power balances."""

import numpy as np

from problems.engineering.realworld.common import data, none

DG_BUSES = np.array([34, 35, 36, 37, 38]) - 1  # distributed generators of the 38-bus network
OPF_GENERATORS = np.array([2, 13, 22, 23, 27]) - 1  # generators of the IEEE 30-bus network besides the slack bus


def _network(name: str) -> np.ndarray:
    return data()[f"{name}_G"] + 1j * data()[f"{name}_B"]


def _mismatch(current: np.ndarray, voltage: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Network current minus the current the specified injections draw, real and imaginary parts."""
    return current - np.conj((p + 1j * q) / voltage)


def _load(table: np.ndarray, voltage: np.ndarray) -> np.ndarray:
    """Voltage-dependent load: base × (|V| / V0)^exponent."""
    return table[:, 0] * (np.abs(voltage) / table[:, 4]) ** table[:, 5]


def rc34(x):
    """Single-phase distributed generation with reactive support for phase balancing at the main transformer."""
    n = len(x)
    y, p, q = _network("PS1"), data()["PS1_P"], data()["PS1_Q"]
    voltage = np.zeros((n, 30), dtype=complex)
    voltage[:, 0], voltage[:, 1], voltage[:, 2] = 1, np.exp(1j * 4 * np.pi / 3), np.exp(1j * 2 * np.pi / 3)
    voltage[:, 3:] = x[:, 0:27] + 1j * x[:, 27:54]
    psp, qsp, pdg, qdg = np.zeros((n, 30)), np.zeros((n, 30)), np.zeros((n, 30)), np.zeros((n, 30))
    psp[:, 3:], qsp[:, 3:] = x[:, 54:81], x[:, 81:108]
    buses = np.array([9, 16, 21, 24, 30]) - 1
    pdg[:, buses], qdg[:, buses] = x[:, 108:113], x[:, 113:118]
    current = voltage @ y.T
    mismatch = _mismatch(current, voltage, psp, qsp)
    f = np.abs(current[:, 0] + current[:, 1] + current[:, 2]) + np.abs(
        current[:, 0] + np.exp(1j * 4 * np.pi / 3) * current[:, 1] + np.exp(1j * 2 * np.pi / 3) * current[:, 2]
    )
    h = np.hstack([mismatch.real[:, 3:], mismatch.imag[:, 3:], (psp - pdg - p)[:, 3:], (qsp - qdg - q)[:, 3:]])
    return f, none(n), h


def _distribution(x, reactive: bool):
    n = len(x)
    y, p, q = _network("PS2"), data()["PS2_P"], data()["PS2_Q"]
    voltage = np.ones((n, 38), dtype=complex)
    voltage[:, 1:] = x[:, 0:37] + 1j * x[:, 37:74]
    psp, qsp, pdg, qdg = np.zeros((n, 38)), np.zeros((n, 38)), np.zeros((n, 38)), np.zeros((n, 38))
    psp[:, 1:], qsp[:, 1:] = x[:, 74:111], x[:, 111:148]
    pdg[:, DG_BUSES] = x[:, 148:153]
    if reactive:
        qdg[:, DG_BUSES] = x[:, 153:158]
    current = voltage @ y.T
    mismatch = _mismatch(current, voltage, psp, qsp)
    delta_p, delta_q = psp - pdg + _load(p, voltage), qsp - qdg + _load(q, voltage)
    h = np.hstack([mismatch.real[:, 1:], mismatch.imag[:, 1:], delta_p[:, 1:], delta_q[:, 1:]])
    slack = voltage[:, 0] * np.conj(current[:, 0])
    return slack, psp, qsp, h


def rc35(x):
    """Distributed generation sizing for active power loss minimisation."""
    slack, psp, _, h = _distribution(x, reactive=False)
    return slack.real + psp[:, 1:].sum(axis=1), none(len(x)), h


def rc36(x):
    """Distributed generation and capacitor sizing for reactive power loss minimisation."""
    slack, psp, qsp, h = _distribution(x, reactive=True)
    return 0.5 * (slack.real + psp[:, 1:].sum(axis=1)) + 0.5 * (slack.imag + qsp[:, 1:].sum(axis=1)), none(len(x)), h


def _optimal_flow(x):
    n = len(x)
    y, p, q = _network("PS11"), data()["PS11_P"], data()["PS11_Q"]
    voltage = np.ones((n, 30), dtype=complex)
    voltage[:, 1:] = x[:, 0:29] + 1j * x[:, 29:58]
    psp, qsp, pg, qg = np.zeros((n, 30)), np.zeros((n, 30)), np.zeros((n, 30)), np.zeros((n, 30))
    psp[:, 1:], qsp[:, 1:] = x[:, 58:87], x[:, 87:116]
    pg[:, OPF_GENERATORS], qg[:, OPF_GENERATORS] = x[:, 116:121], x[:, 121:126]
    current = voltage @ y.T
    mismatch = _mismatch(current, voltage, psp, qsp)
    h = np.hstack([mismatch.real[:, 1:], mismatch.imag[:, 1:], (psp - pg + p)[:, 1:], (qsp - qg + q)[:, 1:]])
    slack = (voltage[:, 0] * np.conj(current[:, 0])).real
    return slack, psp, pg, p, h


def _fuel(pg: np.ndarray, slack: np.ndarray) -> np.ndarray:
    pg = pg.copy()
    pg[:, 0] = slack
    units = np.array([1, 2, 13, 22, 23, 27]) - 1
    b, c = np.array([2, 1.75, 1, 3.25, 3, 3]), np.array([0.02, 0.0175, 0.0625, 0.00834, 0.025, 0.0025])
    return (b * pg[:, units] + c * pg[:, units] ** 2).sum(axis=1), pg


def rc37(x):
    """Optimal power flow: active power loss."""
    slack, psp, _, _, h = _optimal_flow(x)
    return slack + psp[:, 1:].sum(axis=1), none(len(x)), h


def rc38(x):
    """Optimal power flow: fuel cost."""
    slack, _, pg, _, h = _optimal_flow(x)
    return _fuel(pg, slack)[0], none(len(x)), h


def rc39(x):
    """Optimal power flow: active power loss and fuel cost."""
    slack, _, pg, p, h = _optimal_flow(x)
    cost, generation = _fuel(pg, slack)
    return cost + 0.75 * (generation - p).sum(axis=1), none(len(x)), h


def _admittance(frequency: np.ndarray) -> np.ndarray:
    """The islanded microgrid's bus admittance matrix, line reactances scaled by the frequency: (n × 38 × 38)."""
    lines = data()["PS14_linedata"]
    start, end = lines[:, 0].astype(int) - 1, lines[:, 1].astype(int) - 1
    resistance, reactance, shunt, tap = lines[:, 2], lines[:, 3], 1j * lines[:, 4], lines[:, 5]
    y = 1 / (resistance + 1j * reactance * frequency[:, None])  # (n, lines)
    n, size = len(frequency), int(max(start.max(), end.max())) + 1
    admittance = np.zeros((n, size, size), dtype=complex)
    for k in range(len(lines)):
        admittance[:, start[k], end[k]] -= y[:, k] / tap[k]
        admittance[:, end[k], start[k]] = admittance[:, start[k], end[k]]
    for k in range(len(lines)):
        admittance[:, start[k], start[k]] += y[:, k] / tap[k] ** 2 + shunt[k]
        admittance[:, end[k], end[k]] += y[:, k] + shunt[k]
    return admittance


def _islanded(x, droop_p: np.ndarray, droop_q: np.ndarray):
    n = len(x)
    p, q = data()["PS2_P"], data()["PS2_Q"]
    voltage = np.ones((n, 38), dtype=complex)
    voltage[:, 1:] = x[:, 0:37] + 1j * x[:, 37:74]
    frequency = x[:, 74]
    voltage[:, 0] = x[:, 75] + 1e-5
    pc, qc = np.zeros((n, 38)), np.zeros((n, 38))
    pc[:, DG_BUSES], qc[:, DG_BUSES] = droop_p, droop_q
    current = np.einsum("nij,nj->ni", _admittance(frequency), voltage)
    psp = pc * (1 - frequency[:, None]) - _load(p, voltage)
    qsp = qc * (1 - np.abs(voltage)) - _load(q, voltage)
    mismatch = _mismatch(current, voltage, psp, qsp)
    return voltage, current, psp, qsp, np.hstack([mismatch.real, mismatch.imag])


def rc40(x):
    """Microgrid power flow, islanded."""
    n = len(x)
    droop_p = np.broadcast_to(1 / np.array([5.102e-03, 1.502e-03, 4.506e-03, 2.253e-03, 2.253e-03]), (n, 5))
    droop_q = np.broadcast_to(1 / np.array([0.05, 0.03, 0.05, 0.01, 0.1]), (n, 5))
    voltage, current, psp, qsp, h = _islanded(x, droop_p, droop_q)
    vr, vm, ir, im = voltage.real, voltage.imag, current.real, current.imag
    f = ((psp - (vr * ir + vm * im)) ** 2).sum(axis=1) + ((qsp - (vm * ir - vr * im)) ** 2).sum(axis=1)
    return f, none(n), h


def rc41(x):
    """Microgrid power flow, grid-connected."""
    n = len(x)
    y, p, q = _network("PS2"), data()["PS2_P"], data()["PS2_Q"]
    voltage = np.ones((n, 38), dtype=complex)
    voltage[:, 1:] = x[:, 0:37] + 1j * x[:, 37:74]
    pdg, qdg = np.zeros(38), np.zeros(38)
    pdg[DG_BUSES], qdg[DG_BUSES] = 0.2, 0.18
    current = voltage @ y.T
    psp, qsp = pdg - _load(p, voltage), qdg - _load(q, voltage)
    mismatch = _mismatch(current, voltage, psp, qsp)
    vr, vm, ir, im = voltage.real, voltage.imag, current.real, current.imag
    f = ((psp - (vr * ir + vm * im))[:, 1:] ** 2).sum(axis=1) + ((qsp - (vm * ir - vr * im))[:, 1:] ** 2).sum(axis=1)
    return f, none(n), np.hstack([mismatch.real[:, 1:], mismatch.imag[:, 1:]])


def rc42(x):
    """Droop controller settings for active power loss minimisation in an islanded microgrid."""
    _, _, psp, _, h = _islanded(x, x[:, 76:81], x[:, 81:86])
    return psp.sum(axis=1), none(len(x)), h


def rc43(x):
    """Droop controller settings for reactive power loss minimisation in an islanded microgrid."""
    _, _, psp, qsp, h = _islanded(x, x[:, 76:81], x[:, 81:86])
    return 0.5 * (qsp.sum(axis=1) + psp.sum(axis=1)), none(len(x)), h


# RC44 wind farm layout: place 15 turbines on a 2 km square to maximise the expected power under a 24-direction
# wind rose (Jensen wake model, Weibull wind speeds). Pairs closer than five rotor radii violate a constraint.

WIND_SCALE = np.array([7, 5, 5, 5, 5, 4, 5, 6, 7, 7, 8, 9.5, 10, 8.5, 8.5, 6.5, 4.6, 2.6, 8, 5, 6.4, 5.2, 4.5, 3.9])
WIND_FREQUENCY = np.array([
    0.0003, 0.0072, 0.0237, 0.0242, 0.0222, 0.0301, 0.0397, 0.0268, 0.0626, 0.0801, 0.1025, 0.1445,
    0.1909, 0.1162, 0.0793, 0.0082, 0.0041, 0.0008, 0.0010, 0.0005, 0.0013, 0.0031, 0.0085, 0.0222,
])  # fmt: skip


def rc44(x):
    """Wind farm layout."""
    n, turbines = len(x), x.shape[1] // 2
    radius, kappa, shape, cut_in, rated, cut_out = 40, 0.01, 2, 3.5, 14, 25
    induction = 1 - np.sqrt(1 - 0.8)
    position = x.reshape(n, turbines, 2)
    offset = position[:, :, None, :] - position[:, None, :, :]  # [i, j]: downstream i relative to upstream j
    steps = int(np.floor((rated - cut_in) / 0.3))
    low = cut_in + np.arange(steps) * 0.3
    high = low + 0.3
    middle = (low + high) / 2
    curve = 1500 * np.exp(middle - 7.5) / (5 + np.exp(middle - 7.5))
    power = np.zeros(n)
    for direction, (scale, frequency) in enumerate(zip(WIND_SCALE, WIND_FREQUENCY, strict=True)):
        theta = np.deg2rad((direction + 0.5) * 15)
        along = np.cos(theta) * offset[..., 0] + np.sin(theta) * offset[..., 1]
        across = np.sqrt(
            np.maximum(offset[..., 0] ** 2 + offset[..., 1] ** 2 - along**2, 0)
        )  # MATLAB compares the real part
        wake = along * kappa + radius
        affected = (wake > across - radius) & (along > 0) & ~np.eye(turbines, dtype=bool)
        deficit = np.where(affected, induction / (1 + kappa * along / radius) ** 2, 0)
        speed = scale * (1 - np.sqrt((deficit**2).sum(axis=2)))  # (n, turbines)
        ratio_low, ratio_high = (low / speed[..., None]) ** shape, (high / speed[..., None]) ** shape
        turbine = (curve * (np.exp(-ratio_low) - np.exp(-ratio_high))).sum(axis=-1)
        turbine += 1500 * (np.exp(-((rated / speed) ** shape)) - np.exp(-((cut_out / speed) ** shape)))
        power += frequency * turbine.sum(axis=1)
    pairs = [
        (i, j) for i in range(turbines) for j in range(i + 1, turbines - 1)
    ]  # the organisers' loop stops one turbine short
    distance = np.stack([np.linalg.norm(position[:, i] - position[:, j], axis=1) for i, j in pairs], axis=1)
    return -power, 5 * radius - distance, none(n)
