"""SDK-trace context-manager helper.

Wraps openai-agents ``trace(...)`` with a Tier-1 ``nullcontext`` fallback
so search / baseline code stays harness-agnostic. Centralizes the SDK
import shim that previously lived in both ``src/search/orchestrator.py``
and ``src/benchmark/baseline_generator.py``.

Metadata shape — ``{"iter": <int>, "agent": <str>, **extra}`` — is the
contract the resource accumulator (``src/runtime/usage.py``) reads at
trace-close to bucket usage by ``(iter, agent)``.
"""

from __future__ import annotations

import contextlib
from typing import Any

from src.runtime.usage import AgentLabel

try:
    from agents import trace as _agents_trace
except ModuleNotFoundError:
    _agents_trace = None  # SDK absent (Tier-1 test venv); helper degrades to nullcontext.


def _coerce_agent_label(agent: AgentLabel | str) -> str:
    """Return the plain string form of *agent*.

    For ``AgentLabel`` members, returns ``agent.value`` (e.g. ``"coder"``)
    — NOT ``str(agent)`` which on Python 3.10's stdlib enum returns
    ``"AgentLabel.CODER"`` because the enum's ``__str__`` overrides the
    ``str`` base class. The bare-string path returns the input unchanged.

    Centralized so any future consumer (event emitters, etc.) can reuse
    the same coercion rule without re-deriving it.
    """
    if isinstance(agent, AgentLabel):
        return agent.value
    return str(agent)


def trace_span(
    workflow_name: str,
    *,
    iter_no: int,
    agent: AgentLabel | str,
    **extra_metadata: Any,
):
    """Wrap a ``Runner.run()`` call in an SDK trace tagged with iter / agent.

    Returns ``contextlib.nullcontext()`` when the SDK isn't installed.
    Coerces ``agent`` to its plain string form (the enum *value*, not
    ``repr``) so downstream consumers (the resource accumulator's
    ``on_trace_close`` reader, the report sidecar) see a JSON-serializable
    string in metadata.
    """
    if _agents_trace is None:
        return contextlib.nullcontext()
    metadata: dict[str, Any] = {
        "iter": iter_no,
        "agent": _coerce_agent_label(agent),
        **extra_metadata,
    }
    return _agents_trace(workflow_name=workflow_name, metadata=metadata)
