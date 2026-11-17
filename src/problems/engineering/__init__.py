"""Nine classic constrained engineering designs, and the CEC 2020 real-world constrained suite (`realworld`).

    >>> from problems.engineering import DESIGNS, Penalized
    >>> objective = Penalized(DESIGNS["welded-beam"])  # cost + 1e6 · Σ normalised violations
    >>> objective.space.dimensions
    4

`Penalized(design)` is an objective NIO minimises over `design.space`; `evaluate(position)` gives the `Solution`.
"""

from problems.engineering import problems as p
from problems.engineering.design import Constrained, Design, Penalized, Solution

DESIGNS: dict[str, Design] = {
    design.code: design
    for design in (
        Design(
            "pressure-vessel",
            "Pressure Vessel",
            ("shell thickness", "head thickness", "inner radius", "length"),
            (p.THICKNESS, p.THICKNESS, 10, 10),
            (99 * p.THICKNESS, 99 * p.THICKNESS, 200, 200),
            p.pressure_vessel_cost,
            p.pressure_vessel_constraints,
            best=6059.714335,
            discrete=p.pressure_vessel_discrete,
        ),
        Design(
            "welded-beam",
            "Welded Beam",
            ("weld thickness", "weld length", "bar height", "bar thickness"),
            (0.1, 0.1, 0.1, 0.1),
            (2, 10, 10, 2),
            p.welded_beam_cost,
            p.welded_beam_constraints,
            best=1.724852,
        ),
        Design(
            "tension-spring",
            "Tension/Compression Spring",
            ("wire diameter", "coil diameter", "active coils"),
            (0.05, 0.25, 2),
            (2, 1.3, 15),
            p.tension_spring_cost,
            p.tension_spring_constraints,
            best=0.012665,
        ),
        Design(
            "speed-reducer",
            "Speed Reducer",
            (
                "face width",
                "teeth module",
                "pinion teeth",
                "shaft 1 length",
                "shaft 2 length",
                "shaft 1 diameter",
                "shaft 2 diameter",
            ),
            (2.6, 0.7, 17, 7.3, 7.8, 2.9, 5.0),
            (3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5),
            p.speed_reducer_cost,
            p.speed_reducer_constraints,
            best=2996.348165,  # with 7.8 <= shaft 2 length; 2994.471 needs shaft 2 down to 7.715
            discrete=p.speed_reducer_discrete,
        ),
        Design(
            "three-bar-truss",
            "Three-Bar Truss",
            ("outer bar area", "middle bar area"),
            (0, 0),
            (1, 1),
            p.three_bar_truss_cost,
            p.three_bar_truss_constraints,
            best=263.895843,
        ),
        Design(
            "cantilever-beam",
            "Cantilever Beam",
            ("section 1", "section 2", "section 3", "section 4", "section 5"),
            (0.01,) * 5,
            (100,) * 5,
            p.cantilever_beam_cost,
            p.cantilever_beam_constraints,
            best=1.339956,
        ),
        Design(
            "gear-train",
            "Gear Train",
            ("driver 1 teeth", "driven 1 teeth", "driven 2 teeth", "driver 2 teeth"),
            (12,) * 4,
            (60,) * 4,
            p.gear_train_cost,
            p.gear_train_constraints,
            best=2.700857e-12,
            discrete=p.gear_train_discrete,
        ),
        Design(
            "clutch-brake",
            "Multiple-Disc Clutch Brake",
            ("inner radius", "outer radius", "disc thickness", "actuating force", "friction surfaces"),
            (60, 90, 1, 0, 2),
            (80, 110, 3, 1000, 9),
            p.clutch_brake_cost,
            p.clutch_brake_constraints,
            best=0.2352424579,  # CEC 2020 RC21, whose formulation this follows
            discrete=p.clutch_brake_discrete,
        ),
        Design(
            "piston-lever",
            "Piston Lever",
            ("height", "base", "piston diameter", "lever position"),
            (0.05, 0.05, 0.05, 0.05),
            (500, 500, 120, 500),
            p.piston_lever_cost,
            p.piston_lever_constraints,
            best=1.057394,
        ),
    )
}


def get(code: str) -> Design:
    """A design of `DESIGNS` by code, such as "welded-beam", ignoring case and surrounding spaces."""
    wanted = code.strip().lower()
    for key, design in DESIGNS.items():
        if key == wanted:
            return design
    raise ValueError(f"Unknown design {code!r}. Choose from {', '.join(DESIGNS)}.")


__all__ = ["DESIGNS", "Constrained", "Design", "Penalized", "Solution", "get"]
