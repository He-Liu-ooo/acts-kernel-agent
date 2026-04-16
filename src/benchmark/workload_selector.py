"""Select representative workloads for iterative benchmarking.

SOL-ExecBench problems ship 7-48 workloads per problem.  Running all of
them every iteration is too slow, so we pick a small representative subset
(default 2-3) that spans the dynamic-axis range.
"""

from __future__ import annotations

from src.benchmark.problem import Workload


def select_workloads(
    workloads: list[Workload],
    count: int = 3,
) -> list[Workload]:
    """Pick *count* representative workloads from the full set.

    Strategy: sort by the product of all var-axis values (a rough proxy
    for problem size), then take evenly spaced samples so the subset
    covers small, medium, and large shapes.

    If the full set has *count* or fewer workloads, return all of them.
    """
    if len(workloads) <= count:
        return list(workloads)

    def _size_key(w: Workload) -> int:
        product = 1
        for v in w.axes.values():
            product *= max(v, 1)
        return product

    sorted_wl = sorted(workloads, key=_size_key)
    n = len(sorted_wl)
    step = (n - 1) / max(count - 1, 1)
    indices = [round(i * step) for i in range(count)]
    return [sorted_wl[i] for i in indices]
