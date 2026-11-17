"""A schedule, and the two decoders that build one: list scheduling by priority, or a given machine per task."""

import heapq
from dataclasses import dataclass

import numpy as np

from problems.scheduling.workload import Workload


@dataclass(frozen=True, eq=False)
class Schedule:
    """Where and when every task runs."""

    workload: Workload
    machine: np.ndarray  # machine index of each task
    start: np.ndarray  # start time of each task
    finish: np.ndarray  # finish time of each task

    @property
    def makespan(self) -> float:
        """When the last machine finishes."""
        return float(self.finish.max())

    @property
    def loads(self) -> np.ndarray:
        """Busy time of each machine."""
        return np.bincount(self.machine, weights=self.finish - self.start, minlength=self.workload.count)

    @property
    def utilization(self) -> np.ndarray:
        """Share of the makespan each machine is busy."""
        return self.loads / self.makespan

    def cost(self, prices: np.ndarray) -> float:
        """Total price of the busy time, with `prices` per machine per time unit."""
        return float(self.loads @ prices)

    @property
    def imbalance(self) -> float:
        """Relative spread of machine loads: 0 when every machine is equally busy."""
        loads = self.loads
        return float((loads.max() - loads.min()) / loads.mean())


def by_priority(workload: Workload, order: np.ndarray) -> Schedule:
    """List scheduling: take tasks in `order`, each on the machine that becomes free first (lowest index on ties)."""
    free = [(0.0, machine) for machine in range(workload.count)]
    machine, start, finish = (np.empty(workload.size, dtype=int), np.empty(workload.size), np.empty(workload.size))
    for task in order:
        time, chosen = heapq.heappop(free)
        machine[task], start[task] = chosen, time
        finish[task] = time + workload.runtime(task, chosen)
        heapq.heappush(free, (finish[task], chosen))
    return Schedule(workload, machine, start, finish)


def by_assignment(workload: Workload, machine: np.ndarray) -> Schedule:
    """Run each task on its given machine; a machine runs its tasks back to back in task order."""
    runtime = workload.runtime(np.arange(workload.size), machine)
    finish = np.empty(workload.size)
    for index in range(workload.count):
        tasks = np.flatnonzero(machine == index)
        finish[tasks] = np.cumsum(runtime[tasks])
    return Schedule(workload, machine, finish - runtime, finish)
