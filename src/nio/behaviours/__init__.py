"""The ibex behaviours of NIO.

Each behaviour proposes new positions for a group of ibex (`who`) from the herd of one season, as an array of shape
(len(who), D). The caller repairs the bounds, scores the proposals and accepts the better ones. Fission–fusion
(`nio.behaviours.fission`) is not a behaviour but the herd's structure: it splits the herd into subgroups, each with its
own leader, and the group behaviours (browse, follow_dominant, zigzag_climb) work within the ibex's own subgroup.

Every behaviour works on differences between positions or on the search space's bounds and span (the sentinel redraws
across the bounds), except two that use the structure of the search box on purpose:

* `home_range` pulls toward the centre of the box;
* `ridge_walk` steps along the line through the origin in the direction (1, …, 1), on which every coordinate is equal
  (the box's main diagonal only when every dimension has the same bounds).

Only `ridge_walk` breaks translation invariance.

`BEHAVIOURS` lists the behaviours by name. Its order is part of the algorithm: route preference draws an index into it.
"""

from nio.behaviours.browse import browse
from nio.behaviours.cliff import cliff_escape
from nio.behaviours.follow import follow_dominant
from nio.behaviours.home import home_range
from nio.behaviours.ledge import ledge_walk
from nio.behaviours.ridge import ridge_walk
from nio.behaviours.rut import rut
from nio.behaviours.sentinel import sentinel
from nio.behaviours.spring import return_to_spring
from nio.behaviours.trail import follow_trail
from nio.behaviours.zigzag import zigzag_climb

BEHAVIOURS = (
    "browse",
    "follow_dominant",
    "zigzag_climb",
    "cliff_escape",
    "sentinel",
    "return_to_spring",
    "rut",
    "home_range",  # pulls toward the centre of the box
    "ridge_walk",  # steps along (1, …, 1)
    "ledge_walk",
    "follow_trail",
)

__all__ = [
    "BEHAVIOURS",
    "browse",
    "cliff_escape",
    "follow_dominant",
    "follow_trail",
    "home_range",
    "ledge_walk",
    "return_to_spring",
    "ridge_walk",
    "rut",
    "sentinel",
    "zigzag_climb",
]
