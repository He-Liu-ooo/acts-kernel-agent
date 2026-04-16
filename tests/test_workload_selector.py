"""Tests for benchmark/workload_selector.py — representative workload selection."""

from src.benchmark.problem import Workload
from src.benchmark.workload_selector import select_workloads


def _make_workloads(batch_sizes: list[int]) -> list[Workload]:
    return [
        Workload(uuid=f"wl-{bs}", axes={"batch_size": bs})
        for bs in batch_sizes
    ]


def test_select_returns_all_when_count_exceeds_total():
    wls = _make_workloads([1, 2, 3])
    selected = select_workloads(wls, count=5)
    assert len(selected) == 3


def test_select_picks_evenly_spaced():
    wls = _make_workloads([1, 8, 16, 64, 128, 256, 512, 1024, 4096])
    selected = select_workloads(wls, count=3)
    assert len(selected) == 3
    sizes = [wl.axes["batch_size"] for wl in selected]
    # Should span small, medium, large
    assert sizes[0] < sizes[1] < sizes[2]


def test_select_single():
    wls = _make_workloads([42])
    selected = select_workloads(wls, count=3)
    assert len(selected) == 1
    assert selected[0].axes["batch_size"] == 42
