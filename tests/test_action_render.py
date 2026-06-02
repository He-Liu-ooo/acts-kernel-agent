"""Unit tests for the action-catalog render helpers (Planner menu + Coder
per-technique guidance). Pure string functions — Tier-1 torchless."""

from src.actions.registry import Action, ActionTier
from src.agents.llm_backend import render_action_menu, render_technique_guidance


def _full_action() -> Action:
    return Action(
        id="t2_demo",
        tier=ActionTier.MEMORY,
        name="Demo Tiling",
        description="Reuse operands across the inner loop.",
        preconditions=["memory_bound"],
        parameters={"BLOCK_K": "32"},
        guidance="Use tl.dot and num_stages.",
        anti_patterns=["Tiling data with no reuse."],
        expected_impact="Large on reuse-heavy kernels.",
    )


def test_render_action_menu_includes_all_fields():
    menu = render_action_menu([_full_action()])
    assert "t2_demo" in menu
    assert "Demo Tiling" in menu
    assert "tier 2" in menu
    assert "Reuse operands across the inner loop." in menu
    assert "memory_bound" in menu
    assert "BLOCK_K=32" in menu
    assert "Large on reuse-heavy kernels." in menu


def test_render_action_menu_omits_empty_optional_lines():
    a = Action(id="t1_bare", tier=ActionTier.SIZING, name="Bare",
               description="Just a description.")
    menu = render_action_menu([a])
    assert "t1_bare" in menu
    assert "when:" not in menu
    assert "knobs:" not in menu
    assert "impact:" not in menu


def test_render_action_menu_empty_list_returns_empty():
    assert render_action_menu([]) == ""


def test_render_technique_guidance_renders_guidance_and_antipatterns():
    g = render_technique_guidance(_full_action())
    assert "Use tl.dot and num_stages." in g
    assert "Tiling data with no reuse." in g


def test_render_technique_guidance_empty_when_no_guidance_or_antipatterns():
    a = Action(id="t1_bare", tier=ActionTier.SIZING, name="Bare",
               description="d")
    assert render_technique_guidance(a) == ""
