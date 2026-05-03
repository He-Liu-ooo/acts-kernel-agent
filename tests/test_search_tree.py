"""Focused tests for ``src/search/tree.py`` fields + checkpoint round-trip.

Broader SearchTree behavior (frontier / best_node / serialize) lives in
``tests/test_search.py``; this file collects the targeted assertions for
the ``iter_no`` field added by the search-tree-recording feature.
"""
from __future__ import annotations


def test_iter_no_threaded_through_add_child():
    from src.search.tree import SearchTree
    from src.kernels.kernel import Kernel, KernelSpec, KernelType
    spec = KernelSpec(name="t", kernel_type=KernelType.ELEMENTWISE,
                      flop_count=0, memory_bytes=0, input_shapes=[],
                      pytorch_reference="", t_sol_us=1.0)
    root_kernel = Kernel(spec=spec, source_code="")
    child_kernel = Kernel(spec=spec, source_code="")
    tree = SearchTree()
    root = tree.add_root(root_kernel)
    child = tree.add_child(root.id, child_kernel, "tiling", iter_no=3)
    assert child.iter_no == 3
    assert root.iter_no == -1  # root default


def test_legacy_checkpoint_load_defaults_iter_no(tmp_path):
    """Pre-iter_no checkpoints round-trip with iter_no = -1."""
    import json
    from src.search.tree import SearchTree
    legacy = {
        "next_id": 1,
        "nodes": {
            "0": {
                "id": 0, "parent_id": None, "children_ids": [],
                "action_applied": "", "depth": 0,
                "branch_quality": None, "score": None,
                "kernel": {
                    "spec": {"name": "t", "kernel_type": "elementwise",
                             "flop_count": 0, "memory_bytes": 0,
                             "input_shapes": [], "definition_path": None,
                             "pytorch_reference": "", "t_sol_us": 1.0},
                    "source_code": "", "num_warps": 4, "num_stages": 2,
                    "block_size": 128, "triton_kernel_name": "", "dps": False,
                },
                "profiling": None, "per_workload_latency_us": None,
                "consecutive_agent_failures": 0,
                # iter_no intentionally missing
            },
        },
    }
    p = tmp_path / "tree.json"
    p.write_text(json.dumps(legacy))
    tree = SearchTree.load(p)
    assert tree.get_node(0).iter_no == -1
