"""Cloud task scheduling: makespan minimisation of independent tasks on heterogeneous machines.

The workloads are HCSP (1024 tasks on 32 machines, `hcsp("100")` to `hcsp("400")`) and GoCJ job sets (`gocj(size,
machines)`, with the machine speeds given). `Scheduling(workload, encoding)` is the objective:

* `priority`: a position in [0, 1]^tasks is sorted into a task order (random keys) and list-scheduled, each task going
  to the machine that frees up first;
* `assignment`: the integer part of each coordinate in [0, machines)^tasks is that task's machine.

`workload.bound`, the total work over the total speed, is a makespan no schedule can beat.
"""

from problems.scheduling.objective import Encoding, Scheduling
from problems.scheduling.schedule import Schedule, by_assignment, by_priority
from problems.scheduling.workload import GOCJ, HCSP, Workload, gocj, hcsp

__all__ = [
    "GOCJ",
    "HCSP",
    "Encoding",
    "Schedule",
    "Scheduling",
    "Workload",
    "by_assignment",
    "by_priority",
    "gocj",
    "hcsp",
]
