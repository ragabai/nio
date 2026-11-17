"""Classic constrained engineering design problems, vectorised over leading axes (positions are the last axis).

Each design has a cost and constraints that hold when <= 0, every constraint divided by its limit, so one penalty
weight treats them alike. Discrete variables (plate thicknesses, numbers of teeth) are snapped by the design's
`discrete` function before evaluation.
"""

import numpy as np

THICKNESS = 0.0625  # pressure vessel plates come in multiples of 1/16 inch


def _columns(x: np.ndarray) -> list[np.ndarray]:
    return list(np.moveaxis(x, -1, 0))


def pressure_vessel_cost(x):
    shell, head, radius, length = _columns(x)
    return (
        0.6224 * shell * radius * length
        + 1.7781 * head * radius**2
        + 3.1661 * shell**2 * length
        + 19.84 * shell**2 * radius
    )


def pressure_vessel_constraints(x):
    shell, head, radius, length = _columns(x)
    volume = np.pi * radius**2 * length + 4 / 3 * np.pi * radius**3
    return np.stack(
        [0.0193 * radius / shell - 1, 0.00954 * radius / head - 1, 1 - volume / 1_296_000, length / 240 - 1], axis=-1
    )


def pressure_vessel_discrete(x):
    snapped = x.copy()
    snapped[..., :2] = np.round(x[..., :2] / THICKNESS) * THICKNESS
    return snapped


LOAD, LENGTH, MODULUS, SHEAR_MODULUS = 6000.0, 14.0, 30e6, 12e6  # welded beam: lb, in, psi, psi
SHEAR_LIMIT, STRESS_LIMIT, DEFLECTION_LIMIT = 13_600.0, 30_000.0, 0.25


def welded_beam_cost(x):
    weld, length, height, thickness = _columns(x)
    return 1.10471 * weld**2 * length + 0.04811 * height * thickness * (14 + length)


def welded_beam_constraints(x):
    """Coello's seven constraints (Coello 2000; as restated by Chehouri et al. 2016), whose optimum is 1.724852:
    deflection 4PL³/(E t³ b) and the cost limit 0.10471 h² + 0.04811 t b (14 + l) <= 5. CEC 2020 RC19 is another
    variant (deflection 6PL³/(E t² b), no cost limit, best 1.6702)."""
    weld, length, height, thickness = _columns(x)
    primary = LOAD / (np.sqrt(2) * weld * length)
    moment = LOAD * (LENGTH + length / 2)
    radius = np.sqrt(length**2 / 4 + ((weld + height) / 2) ** 2)
    inertia = 2 * (
        np.sqrt(2) * weld * length * (length**2 / 12 + ((weld + height) / 2) ** 2)
    )  # polar moment of the weld group
    secondary = moment * radius / inertia
    shear = np.sqrt(primary**2 + 2 * primary * secondary * length / (2 * radius) + secondary**2)
    stress = 6 * LOAD * LENGTH / (thickness * height**2)
    deflection = 4 * LOAD * LENGTH**3 / (MODULUS * height**3 * thickness)
    buckling = (
        4.013
        * MODULUS
        * np.sqrt(height**2 * thickness**6 / 36)
        / LENGTH**2
        * (1 - height / (2 * LENGTH) * np.sqrt(MODULUS / (4 * SHEAR_MODULUS)))
    )
    return np.stack(
        [
            shear / SHEAR_LIMIT - 1,
            stress / STRESS_LIMIT - 1,
            deflection / DEFLECTION_LIMIT - 1,
            weld / thickness - 1,
            LOAD / buckling - 1,
            0.125 / weld - 1,
            (0.10471 * weld**2 + 0.04811 * height * thickness * (14 + length)) / 5 - 1,
        ],
        axis=-1,
    )


def tension_spring_cost(x):
    wire, coil, turns = _columns(x)
    return (turns + 2) * coil * wire**2


def tension_spring_constraints(x):
    wire, coil, turns = _columns(x)
    return np.stack(
        [
            1 - coil**3 * turns / (71_785 * wire**4),
            (4 * coil**2 - wire * coil) / (12_566 * (coil * wire**3 - wire**4)) + 1 / (5108 * wire**2) - 1,
            1 - 140.45 * wire / (coil**2 * turns),
            (wire + coil) / 1.5 - 1,
        ],
        axis=-1,
    )


def speed_reducer_cost(x):
    face, module, teeth, shaft1, shaft2, diameter1, diameter2 = _columns(x)
    return (
        0.7854 * face * module**2 * (3.3333 * teeth**2 + 14.9334 * teeth - 43.0934)
        - 1.508 * face * (diameter1**2 + diameter2**2)
        + 7.4777 * (diameter1**3 + diameter2**3)
        + 0.7854 * (shaft1 * diameter1**2 + shaft2 * diameter2**2)
    )


def speed_reducer_constraints(x):
    face, module, teeth, shaft1, shaft2, diameter1, diameter2 = _columns(x)
    return np.stack(
        [
            27 / (face * module**2 * teeth) - 1,
            397.5 / (face * module**2 * teeth**2) - 1,
            1.93 * shaft1**3 / (module * teeth * diameter1**4) - 1,
            1.93 * shaft2**3 / (module * teeth * diameter2**4) - 1,
            np.sqrt((745 * shaft1 / (module * teeth)) ** 2 + 16.9e6) / (110 * diameter1**3) - 1,
            np.sqrt((745 * shaft2 / (module * teeth)) ** 2 + 157.5e6) / (85 * diameter2**3) - 1,
            module * teeth / 40 - 1,
            5 * module / face - 1,
            face / (12 * module) - 1,
            (1.5 * diameter1 + 1.9) / shaft1 - 1,
            (1.1 * diameter2 + 1.9) / shaft2 - 1,
        ],
        axis=-1,
    )


def speed_reducer_discrete(x):
    snapped = x.copy()
    snapped[..., 2] = np.round(x[..., 2])  # number of teeth on the pinion
    return snapped


TRUSS_LENGTH, TRUSS_LOAD, TRUSS_STRESS = 100.0, 2.0, 2.0


def three_bar_truss_cost(x):
    outer, middle = _columns(x)
    return (2 * np.sqrt(2) * outer + middle) * TRUSS_LENGTH


def three_bar_truss_constraints(x):
    outer, middle = _columns(x)
    denominator = np.sqrt(2) * outer**2 + 2 * outer * middle
    with np.errstate(divide="ignore", invalid="ignore"):  # a bar of zero area is infinitely stressed
        stresses = np.stack(
            [(np.sqrt(2) * outer + middle) / denominator, middle / denominator, 1 / (np.sqrt(2) * middle + outer)],
            axis=-1,
        )
    return np.nan_to_num(stresses * TRUSS_LOAD / TRUSS_STRESS - 1, nan=np.inf, posinf=np.inf)


def cantilever_beam_cost(x):
    return 0.0624 * np.sum(x, axis=-1)


def cantilever_beam_constraints(x):
    return (np.sum(np.array([61, 37, 19, 7, 1]) / x**3, axis=-1) - 1)[..., None]


GEAR_RATIO = 1 / 6.931


def gear_train_cost(x):
    driver1, driven1, driven2, driver2 = _columns(x)
    return (GEAR_RATIO - driven1 * driven2 / (driver1 * driver2)) ** 2


def gear_train_constraints(x):
    return np.zeros((*np.shape(x)[:-1], 0))  # bounds and integrality are the only restrictions


def gear_train_discrete(x):
    return np.round(x)


CLUTCH_SPEED, CLUTCH_INERTIA, CLUTCH_FRICTION, CLUTCH_STATIC = 250.0, 55.0, 3.0, 40.0  # rpm, kg·m², N·m, N·m
CLUTCH_DENSITY, CLUTCH_MU, CLUTCH_SAFETY = 0.0000078, 0.6, 1.5  # kg/mm³, friction coefficient, static safety
CLUTCH_GAP, CLUTCH_SPACING, CLUTCH_LENGTH = 20.0, 0.5, 30.0  # least ro − ri, gap per disc, most stack length (mm)
CLUTCH_PRESSURE, CLUTCH_SPEED_LIMIT, CLUTCH_TIME = 1.0, 10.0, 15.0  # MPa, m/s, s


def clutch_brake_discrete(x):
    return np.round(x)  # radii, thickness, force and surface count are integers in RC21


def clutch_brake_cost(x):
    inner, outer, thickness, _, surfaces = _columns(x)
    return np.pi * (outer**2 - inner**2) * thickness * (surfaces + 1) * CLUTCH_DENSITY


def clutch_brake_constraints(x):
    """The CEC 2020 RC21 constraints, normalised; like the competition's code, the sliding radius divides by
    ro²·ri² (so it is far below the textbook's), which leaves every speed constraint slack."""
    inner, outer, thickness, force, surfaces = _columns(x)
    sliding = 2 / 3 * (outer**3 - inner**3) / (outer**2 * inner**2)
    speed = np.pi * sliding * CLUTCH_SPEED / 30
    pressure = force / (np.pi * (outer**2 - inner**2))
    torque = 2 / 3 * CLUTCH_MU * force * surfaces * (outer**3 - inner**3) / (outer**2 - inner**2)
    time = CLUTCH_INERTIA * np.pi * CLUTCH_SPEED / 30 / (torque + CLUTCH_FRICTION)
    return np.stack(
        [
            (inner + CLUTCH_GAP) / outer - 1,
            (surfaces + 1) * (thickness + CLUTCH_SPACING) / CLUTCH_LENGTH - 1,
            pressure / CLUTCH_PRESSURE - 1,
            pressure * speed / (CLUTCH_PRESSURE * CLUTCH_SPEED_LIMIT) - 1,
            speed / CLUTCH_SPEED_LIMIT - 1,
            time / CLUTCH_TIME - 1,
            1 - torque / (CLUTCH_SAFETY * CLUTCH_STATIC),
            -time / CLUTCH_TIME,
        ],
        axis=-1,
    )


PISTON_LOAD, PISTON_ARM, PISTON_MOMENT, PISTON_PRESSURE = 10_000.0, 240.0, 1.8e6, 1500.0  # lb, in, lb·in, psi
PISTON_ANGLE = np.pi / 4  # the lever's swing, 45°


def _piston_lengths(x):
    height, base, _, position = _columns(x)
    closed = np.sqrt((position - base) ** 2 + height**2)
    opened = np.sqrt((position * np.sin(PISTON_ANGLE) + height) ** 2 + (base - position * np.cos(PISTON_ANGLE)) ** 2)
    return closed, opened


def piston_lever_cost(x):
    diameter = x[..., 2]
    closed, opened = _piston_lengths(x)
    return np.pi / 4 * diameter**2 * (opened - closed)  # the oil the piston sweeps


def piston_lever_constraints(x):
    height, base, diameter, position = _columns(x)
    closed, opened = _piston_lengths(x)
    arm = (
        np.abs(
            -position * (position * np.sin(PISTON_ANGLE) + height) + height * (base - position * np.cos(PISTON_ANGLE))
        )
        / closed
    )
    force = np.pi * PISTON_PRESSURE * diameter**2 / 4
    return np.stack(
        [
            1 - arm * force / (PISTON_LOAD * PISTON_ARM * np.cos(PISTON_ANGLE)),
            PISTON_LOAD * (PISTON_ARM - position) / PISTON_MOMENT - 1,
            1.2 * (opened - closed) / closed - 1,
            diameter / (2 * base) - 1,
        ],
        axis=-1,
    )
