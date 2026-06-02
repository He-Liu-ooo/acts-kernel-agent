"""Presence tests for the standing Triton-SMEM guards in the agent system
prompts. Regression tripwire — these guards must not silently disappear."""

from pathlib import Path

_PROMPTS = Path(__file__).resolve().parent.parent / "src" / "prompts"


def test_coder_system_prompt_has_triton_smem_guard():
    text = (_PROMPTS / "coder" / "system.md").read_text()
    assert "tl.static_shared_memory" in text
    assert "implicit" in text.lower()
    assert "num_stages" in text


def test_planner_system_prompt_has_smem_triton_steer():
    text = (_PROMPTS / "planner" / "system.md").read_text()
    assert "no explicit shared-memory" in text.lower()
