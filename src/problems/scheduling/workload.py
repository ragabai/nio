"""Workloads: independent tasks (lengths in million instructions) on heterogeneous machines (speeds in MIPS)."""

from dataclasses import dataclass

import numpy as np

from problems.paths import DATA

SCHEDULING = DATA / "scheduling"
HCSP = ("100", "200", "300", "400")
GOCJ = tuple(range(100, 1001, 50))


@dataclass(frozen=True, eq=False)
class Workload:
    """Independent tasks to run on heterogeneous machines."""

    name: str
    tasks: np.ndarray  # length of each task in million instructions
    machines: np.ndarray  # speed of each machine in MIPS

    def __post_init__(self):
        if self.tasks.ndim != 1 or self.machines.ndim != 1 or not len(self.tasks) or not len(self.machines):
            raise ValueError(f"{self.name}: tasks and machines must be non-empty 1-D arrays.")
        if np.any(self.tasks <= 0) or np.any(self.machines <= 0):
            raise ValueError(f"{self.name}: task lengths and machine speeds must be positive.")

    @property
    def size(self) -> int:
        """The number of tasks."""
        return len(self.tasks)

    @property
    def count(self) -> int:
        """Number of machines."""
        return len(self.machines)

    @property
    def bound(self) -> float:
        """Makespan if the work could be split perfectly across all machines; no schedule beats it."""
        return float(self.tasks.sum() / self.machines.sum())

    def runtime(self, task, machine) -> np.ndarray:
        """How long `task` takes on `machine`: its length over the machine's speed."""
        return self.tasks[task] / self.machines[machine]

    def loads(self, machine: np.ndarray) -> np.ndarray:
        """Busy time of every machine (rows × machines) for rows of task-to-machine assignments (rows × tasks)."""
        machine = np.atleast_2d(machine)
        loads = np.zeros((len(machine), self.count))
        np.add.at(loads, (np.arange(len(machine))[:, None], machine), self.runtime(np.arange(self.size), machine))
        return loads


def _read(path) -> np.ndarray:
    return np.loadtxt(path, dtype=float, ndmin=1)


def hcsp(name: str) -> Workload:
    """An HCSP instance: 1024 tasks on 32 machines, tasks and speeds paired by name."""
    if name not in HCSP:
        raise ValueError(f"Unknown HCSP instance '{name}'. Choose from {', '.join(HCSP)}.")
    root = SCHEDULING / "HCSP"
    return Workload(f"HCSP-{name}", _read(root / "tasks" / f"{name}.txt"), _read(root / "machines" / f"{name}.txt"))


def gocj(size: int, machines) -> Workload:
    """A GoCJ (Google-like realistic) job set; the dataset ships job sizes only, so machine speeds are given."""
    if size not in GOCJ:
        raise ValueError(f"Unknown GoCJ size {size}. Choose from {', '.join(map(str, GOCJ))}.")
    return Workload(f"GoCJ-{size}", _read(SCHEDULING / "GoCJ" / f"{size}.txt"), np.asarray(machines, dtype=float))
