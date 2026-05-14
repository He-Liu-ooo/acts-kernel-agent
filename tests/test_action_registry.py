"""Action registry hardware-gating tests.

Covers the contract that `ActionRegistry.list_applicable()` filters
hardware-gated actions (Tier 5 arch-specific) by `HardwareSpec.compute_capability`,
so the Planner never sees H100-only actions on a sub-Hopper GPU.

Backstory: events.jsonl from run_20260514T035029_708161Z showed the Planner
selecting `t5_h100_wgmma` on an RTX 6000 Ada (cc 8.9) — the action declared
`preconditions=["compute_capability >= 9.0"]` as free text but nothing enforced
it. This test file pins the structured-gate contract that replaces the text.
"""

from __future__ import annotations

import pytest

from src.actions.registry import Action, ActionRegistry, ActionTier, build_default_registry
from src.config import HardwareSpec


def _make_action(
    action_id: str,
    tier: ActionTier,
    *,
    applicable_to: list[str] | None = None,
    min_cc: float | None = None,
) -> Action:
    return Action(
        id=action_id,
        tier=tier,
        name=action_id,
        description="",
        applicable_to=applicable_to or [],
        min_compute_capability=min_cc,
    )


def test_action_min_compute_capability_default_is_none() -> None:
    """Plain Tier-1 actions don't gate on hardware."""
    a = _make_action("t1_block_size_tuning", ActionTier.SIZING)
    assert a.min_compute_capability is None


def test_list_applicable_filters_by_kernel_type() -> None:
    """`applicable_to` filter still works after the registry refactor."""
    reg = ActionRegistry()
    reg.register(_make_action("only_matmul", ActionTier.SIZING, applicable_to=["matmul"]))
    reg.register(_make_action("universal", ActionTier.SIZING))

    matmul_actions = {a.id for a in reg.list_applicable("matmul")}
    softmax_actions = {a.id for a in reg.list_applicable("softmax")}

    assert "only_matmul" in matmul_actions and "universal" in matmul_actions
    assert "only_matmul" not in softmax_actions and "universal" in softmax_actions


def test_list_applicable_excludes_hw_gated_when_hardware_missing() -> None:
    """No HardwareSpec supplied → safer to exclude hardware-gated actions.

    Rationale: the prior text-precondition behavior allowed an LLM to pick
    `t5_h100_wgmma` on an unknown box. Default-deny is the safer floor.
    """
    reg = ActionRegistry()
    reg.register(_make_action("ungated", ActionTier.SIZING))
    reg.register(_make_action("hopper_only", ActionTier.ARCH_SPECIFIC, min_cc=9.0))

    ids = {a.id for a in reg.list_applicable("matmul", hardware=None)}
    assert ids == {"ungated"}


def test_list_applicable_ada_hardware_excludes_h100_actions() -> None:
    """RTX 6000 Ada (cc 8.9) → no H100 (cc 9.0+) actions selectable."""
    ada = HardwareSpec(name="RTX6000Ada", compute_capability=8.9)
    reg = ActionRegistry()
    reg.register(_make_action("ungated", ActionTier.SIZING))
    reg.register(_make_action("ampere_or_newer", ActionTier.ARCH_SPECIFIC, min_cc=8.0))
    reg.register(_make_action("hopper_only", ActionTier.ARCH_SPECIFIC, min_cc=9.0))

    ids = {a.id for a in reg.list_applicable("matmul", hardware=ada)}
    assert ids == {"ungated", "ampere_or_newer"}


def test_list_applicable_h100_hardware_includes_h100_and_a100() -> None:
    """H100 (cc 9.0) → both Hopper and Ampere-or-newer actions selectable."""
    h100 = HardwareSpec(name="H100_PCIe", compute_capability=9.0)
    reg = ActionRegistry()
    reg.register(_make_action("ungated", ActionTier.SIZING))
    reg.register(_make_action("ampere_or_newer", ActionTier.ARCH_SPECIFIC, min_cc=8.0))
    reg.register(_make_action("hopper_only", ActionTier.ARCH_SPECIFIC, min_cc=9.0))

    ids = {a.id for a in reg.list_applicable("matmul", hardware=h100)}
    assert ids == {"ungated", "ampere_or_newer", "hopper_only"}


def test_list_applicable_zero_compute_capability_treated_as_unknown() -> None:
    """`compute_capability == 0.0` (default-zero HardwareSpec) → same as no HW info."""
    unknown = HardwareSpec(name="placeholder")  # compute_capability defaults to 0.0
    reg = ActionRegistry()
    reg.register(_make_action("hopper_only", ActionTier.ARCH_SPECIFIC, min_cc=9.0))

    ids = {a.id for a in reg.list_applicable("matmul", hardware=unknown)}
    assert ids == set()  # default-deny when CC unknown


def test_default_registry_h100_actions_carry_min_cc() -> None:
    """The shipped Tier-5 actions declare their hardware floor."""
    reg = build_default_registry()
    assert reg.get("t5_h100_wgmma").min_compute_capability == 9.0
    assert reg.get("t5_h100_tma").min_compute_capability == 9.0
    assert reg.get("t5_hopper_cluster").min_compute_capability == 9.0
    assert reg.get("t5_a100_cp_async").min_compute_capability == 8.0


def test_default_registry_on_ada_filters_h100_actions() -> None:
    """End-to-end: the shipped registry + Ada hardware doesn't expose H100 actions.

    This is the regression-pinning test for the events.jsonl symptom.
    """
    ada = HardwareSpec(name="RTX6000Ada", compute_capability=8.9)
    reg = build_default_registry()
    ids = {a.id for a in reg.list_applicable("matmul", hardware=ada)}

    assert "t5_h100_wgmma" not in ids
    assert "t5_h100_tma" not in ids
    assert "t5_hopper_cluster" not in ids
    # a100_cp_async needs >= 8.0; Ada has 8.9 → included.
    assert "t5_a100_cp_async" in ids


def test_hardware_spec_loads_compute_capability_from_yaml(tmp_path) -> None:
    """`load_hardware_spec` reads `compute_capability` from the SOLAR YAML."""
    from src.config import load_hardware_spec

    yaml_path = tmp_path / "ada.yaml"
    yaml_path.write_text(
        "name: RTX6000Ada\n"
        "compute_capability: 8.9\n"
        "freq_GHz: 2.505\n"
    )
    spec = load_hardware_spec(yaml_path)
    assert spec.compute_capability == pytest.approx(8.9)


def test_hardware_spec_compute_capability_defaults_to_zero() -> None:
    """Backward-compat: existing HardwareSpec() calls keep working."""
    spec = HardwareSpec()
    assert spec.compute_capability == 0.0


def test_detect_hardware_reads_compute_capability_from_torch_props() -> None:
    """``detect_hardware()`` builds ``compute_capability`` from props.major/minor."""
    import sys
    from types import SimpleNamespace
    from unittest.mock import patch

    from src.config import detect_hardware

    props = SimpleNamespace(
        name="Test GPU",
        clock_rate=2_505_000,
        L2_cache_size=100_663_296,
        total_memory=51_539_607_552,
        major=8,
        minor=9,
    )
    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_properties=lambda i: props,
    )
    fake = SimpleNamespace(cuda=cuda)
    with patch.dict(sys.modules, {"torch": fake}):
        spec = detect_hardware()

    assert spec.compute_capability == pytest.approx(8.9)
