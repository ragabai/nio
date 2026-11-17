"""The basic CEC functions, vectorised over a population (rows are positions).

They follow the organisers' reference C code operation for operation, including behaviour that defines the
published suites: the reference keeps a shared work buffer `y`, and Schaffer F7 and Lunacek bi-Rastrigin read
or write it. Every function therefore takes that buffer, `y` (rows × D), and treats it as the reference does.

Signature: `f(x, shift, rotation, shifted, rotated, y) -> values`, where `x` is (rows × n), `shift` holds at
least n values, `rotation` is n × n, and the flags say whether to subtract the shift and apply the rotation.
"""

import numpy as np

E = 2.7182818284590452353602874713526625
PI = 3.1415926535897932384626433832795029


def transform(x, shift, rotation, rate, shifted, rotated, y):
    """Shift, scale and rotate like the reference sr_func; writes the pre-rotation vector to `y` when rotating."""
    n = x.shape[-1]
    base = x - shift[:n] if shifted else x
    if rotated:
        y[:, :n] = base * rate
        return y[:, :n] @ rotation.T
    return base * rate


def sphere(x, shift, rotation, shifted, rotated, y):
    z = transform(x, shift, rotation, 1.0, shifted, rotated, y)
    return np.sum(z * z, axis=1)


def ellips(x, shift, rotation, shifted, rotated, y):
    z = transform(x, shift, rotation, 1.0, shifted, rotated, y)
    n = z.shape[1]
    with np.errstate(invalid="ignore"):  # a single coordinate gives 0/0 in the exponent, as in the reference
        weights = 10.0 ** (6.0 * np.arange(n) / (n - 1))
    return np.sum(weights * z * z, axis=1)


def bent_cigar(x, shift, rotation, shifted, rotated, y):
    z = transform(x, shift, rotation, 1.0, shifted, rotated, y)
    return z[:, 0] ** 2 + np.sum(1e6 * z[:, 1:] * z[:, 1:], axis=1)


def discus(x, shift, rotation, shifted, rotated, y):
    z = transform(x, shift, rotation, 1.0, shifted, rotated, y)
    return 1e6 * z[:, 0] ** 2 + np.sum(z[:, 1:] * z[:, 1:], axis=1)


def sum_diff_pow(x, shift, rotation, shifted, rotated, y):
    z = transform(x, shift, rotation, 1.0, shifted, rotated, y)
    return np.sum(np.abs(z) ** np.arange(1, z.shape[1] + 1), axis=1)


def zakharov(x, shift, rotation, shifted, rotated, y):
    z = transform(x, shift, rotation, 1.0, shifted, rotated, y)
    first = np.sum(z * z, axis=1)
    second = np.sum(0.5 * np.arange(1, z.shape[1] + 1) * z, axis=1)
    return first + second**2 + second**4


def rosenbrock(x, shift, rotation, shifted, rotated, y):
    z = transform(x, shift, rotation, 2.048 / 100.0, shifted, rotated, y) + 1.0
    head, tail = z[:, :-1], z[:, 1:]
    return np.sum(100.0 * (head * head - tail) ** 2 + (head - 1.0) ** 2, axis=1)


def rastrigin(x, shift, rotation, shifted, rotated, y):
    z = transform(x, shift, rotation, 5.12 / 100.0, shifted, rotated, y)
    return np.sum(z * z - 10.0 * np.cos(2.0 * PI * z) + 10.0, axis=1)


def step_rastrigin(x, shift, rotation, shifted, rotated, y):
    """The reference rounds a work buffer that the transform then overwrites, so this equals Rastrigin."""
    return rastrigin(x, shift, rotation, shifted, rotated, y)


def schaffer_f7(x, shift, rotation, shifted, rotated, y):
    """Reads the shared buffer instead of the rotated vector, as the reference does: the rotation has no effect,
    and inside a hybrid function it sees the first coordinates of the permuted vector rather than its own."""
    transform(x, shift, rotation, 1.0, shifted, rotated, y)
    n = x.shape[1]
    radius = np.sqrt(y[:, : n - 1] ** 2 + y[:, 1:n] ** 2)
    root = radius**0.5
    total = np.sum(root + root * np.sin(50.0 * radius**0.2) ** 2, axis=1)
    return total * total / (n - 1) / (n - 1)


def ackley(x, shift, rotation, shifted, rotated, y):
    z = transform(x, shift, rotation, 1.0, shifted, rotated, y)
    n = z.shape[1]
    first = -0.2 * np.sqrt(np.sum(z * z, axis=1) / n)
    second = np.sum(np.cos(2.0 * PI * z), axis=1) / n
    return E - 20.0 * np.exp(first) - np.exp(second) + 20.0


WEIERSTRASS_A, WEIERSTRASS_B, WEIERSTRASS_K = 0.5, 3.0, np.arange(21)


def weierstrass(x, shift, rotation, shifted, rotated, y):
    z = transform(x, shift, rotation, 0.5 / 100.0, shifted, rotated, y)
    amplitude = WEIERSTRASS_A**WEIERSTRASS_K
    frequency = 2.0 * PI * WEIERSTRASS_B**WEIERSTRASS_K
    terms = np.sum(amplitude * np.cos(frequency * (z[..., None] + 0.5)), axis=2)
    offset = np.sum(amplitude * np.cos(frequency * 0.5))
    return np.sum(terms, axis=1) - z.shape[1] * offset


def griewank(x, shift, rotation, shifted, rotated, y):
    z = transform(x, shift, rotation, 600.0 / 100.0, shifted, rotated, y)
    n = z.shape[1]
    return 1.0 + np.sum(z * z, axis=1) / 4000.0 - np.prod(np.cos(z / np.sqrt(1.0 + np.arange(n))), axis=1)


SCHWEFEL_SHIFT, SCHWEFEL_OFFSET = 4.209687462275036e002, 4.189828872724338e002


def schwefel(x, shift, rotation, shifted, rotated, y):
    z = transform(x, shift, rotation, 1000.0 / 100.0, shifted, rotated, y) + SCHWEFEL_SHIFT
    n = z.shape[1]
    above, below = z > 500, z < -500
    with np.errstate(invalid="ignore"):  # each branch is only kept where it is defined
        upper = -(500.0 - np.fmod(z, 500)) * np.sin(np.sqrt(500.0 - np.fmod(z, 500))) + ((z - 500.0) / 100) ** 2 / n
        lower = (
            -(-500.0 + np.fmod(np.abs(z), 500)) * np.sin(np.sqrt(500.0 - np.fmod(np.abs(z), 500)))
            + ((z + 500.0) / 100) ** 2 / n
        )
        middle = -z * np.sin(np.sqrt(np.abs(z)))
    return np.sum(np.where(above, upper, np.where(below, lower, middle)), axis=1) + SCHWEFEL_OFFSET * n


def katsuura(x, shift, rotation, shifted, rotated, y):
    z = transform(x, shift, rotation, 5.0 / 100.0, shifted, rotated, y)
    n = z.shape[1]
    powers = 2.0 ** np.arange(1, 33)
    scaled = powers * z[..., None]
    sums = np.sum(np.abs(scaled - np.floor(scaled + 0.5)) / powers, axis=2)
    product = np.prod((1.0 + np.arange(1, n + 1) * sums) ** (10.0 / n**1.2), axis=1)
    factor = 10.0 / n / n
    return product * factor - factor


def bi_rastrigin(x, shift, rotation, shifted, rotated, y):
    """Lunacek bi-Rastrigin; like the reference, it overwrites the start of the shared buffer `y`."""
    n = x.shape[1]
    mu0, d = 2.5, 1.0
    s = 1.0 - 1.0 / (2.0 * (n + 20.0) ** 0.5 - 8.2)
    mu1 = -(((mu0 * mu0 - d) / s) ** 0.5)
    y[:, :n] = (x - shift[:n]) if shifted else x.copy()  # copy first: x may be a slice of y
    y[:, :n] *= 10.0 / 100.0
    z = 2 * y[:, :n] * np.where(shift[:n] < 0.0, -1.0, 1.0)
    moved = z + mu0
    first = np.sum((moved - mu0) ** 2, axis=1)
    second = np.sum((moved - mu1) ** 2, axis=1) * s + d * n
    if rotated:
        y[:, :n] = z @ rotation.T
        waves = np.sum(np.cos(2.0 * PI * y[:, :n]), axis=1)
    else:
        waves = np.sum(np.cos(2.0 * PI * z), axis=1)
    return np.minimum(first, second) + 10.0 * (n - waves)


def grie_rosen(x, shift, rotation, shifted, rotated, y):
    z = transform(x, shift, rotation, 5.0 / 100.0, shifted, rotated, y) + 1.0
    nxt = np.roll(z, -1, axis=1)  # pairs (z_i, z_i+1) and finally (z_n, z_1)
    inner = 100.0 * (z * z - nxt) ** 2 + (z - 1.0) ** 2
    return np.sum(inner * inner / 4000.0 - np.cos(inner) + 1.0, axis=1)


def escaffer6(x, shift, rotation, shifted, rotated, y):
    z = transform(x, shift, rotation, 1.0, shifted, rotated, y)
    squares = z * z + np.roll(z, -1, axis=1) ** 2
    return np.sum(0.5 + (np.sin(np.sqrt(squares)) ** 2 - 0.5) / (1.0 + 0.001 * squares) ** 2, axis=1)


def happycat(x, shift, rotation, shifted, rotated, y):
    z = transform(x, shift, rotation, 5.0 / 100.0, shifted, rotated, y) - 1.0
    n = z.shape[1]
    r2, total = np.sum(z * z, axis=1), np.sum(z, axis=1)
    return np.abs(r2 - n) ** (2 * 1.0 / 8.0) + (0.5 * r2 + total) / n + 0.5


def hgbat(x, shift, rotation, shifted, rotated, y):
    z = transform(x, shift, rotation, 5.0 / 100.0, shifted, rotated, y) - 1.0
    n = z.shape[1]
    r2, total = np.sum(z * z, axis=1), np.sum(z, axis=1)
    return np.abs(r2**2.0 - total**2.0) ** (2 * 1.0 / 4.0) + (0.5 * r2 + total) / n + 0.5


def _levy(z):
    w = 1.0 + z / 4.0
    first = np.sin(PI * w[:, 0]) ** 2
    last = (w[:, -1] - 1) ** 2 * (1 + np.sin(2 * PI * w[:, -1]) ** 2)
    body = np.sum(
        (w[:, :-1] - 1) ** 2 * (1 + 10 * np.sin(PI * w[:, :-1] + 1) ** 2), axis=1
    )  # sin(πw + 1), as published
    return first + body + last


def levy(x, shift, rotation, shifted, rotated, y):
    """CEC 2017 and 2020 Levy: w = 1 + (z − 1) / 4."""
    return _levy(transform(x, shift, rotation, 1.0, shifted, rotated, y) - 1.0)


def levy_2022(x, shift, rotation, shifted, rotated, y):
    """CEC 2022 Levy: w = 1 + z / 4, which moves the optimum onto the shift vector."""
    return _levy(transform(x, shift, rotation, 1.0, shifted, rotated, y))


BASIC = {
    function.__name__: function
    for function in (
        sphere,
        ellips,
        bent_cigar,
        discus,
        sum_diff_pow,
        zakharov,
        rosenbrock,
        rastrigin,
        step_rastrigin,
        schaffer_f7,
        ackley,
        weierstrass,
        griewank,
        schwefel,
        katsuura,
        bi_rastrigin,
        grie_rosen,
        escaffer6,
        happycat,
        hgbat,
        levy,
        levy_2022,
    )
}
