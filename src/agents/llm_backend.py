"""OpenAI Agents SDK integration — model configuration and runner utilities.

Provides model-swapping via OpenAIChatCompletionsModel: any OpenAI-compatible
API (DeepSeek, vLLM, Together, etc.) works by changing the base URL.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import ValidationError

    from src.eval.types import BottleneckType

try:
    from agents import (
        Agent,
        AsyncOpenAI,
        ModelSettings,
        OpenAIChatCompletionsModel,
        RunConfig,
        Runner,
        RunResult,
    )
    from openai.types.shared.reasoning import Reasoning

    _SDK_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover
    _SDK_AVAILABLE = False
    Reasoning = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# Transient OpenAI errors worth retrying. Permanent failures (auth, schema,
# programmer bugs) must NOT be retried — they waste wall-clock and hide the
# real cause. When openai isn't installed (SDK-absent test mode) the tuple
# is empty, so every exception propagates.
try:
    from openai import (  # noqa: I001  — optional dep
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )

    RETRIABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )
except ImportError:  # pragma: no cover
    RETRIABLE_EXCEPTIONS = ()


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for an LLM model endpoint."""

    model: str
    base_url: str
    api_key: str
    timeout: int = 300
    # When set, overrides every per-agent ``make_run_config(temperature=...)``
    # call. Required for reasoning-style models that reject anything other
    # than a fixed temperature (Moonshot Kimi-K2: only 1.0; OpenAI o1:
    # only 1.0; DeepSeek-Reasoner: only 1.0). When None, per-agent values
    # are used unchanged.
    force_temperature: float | None = None
    # When set, overrides ``make_run_config``'s default ``max_tokens=4096``.
    # Properties of the *model*, not the run — Kimi-K2 supports 256k,
    # DeepSeek-Reasoner typically caps lower. None keeps the historical
    # 4096 default for any caller / provider that hasn't opted in.
    max_tokens: int | None = None
    # When set, threaded through as ``ModelSettings(reasoning=Reasoning(effort=...))``
    # so the SDK forwards it as ``reasoning_effort`` to the provider.
    # Required for thinking-mode models (DeepSeek-v4-pro, OpenAI o-series).
    # ``"low" | "medium" | "high"`` — provider-defined.
    reasoning_effort: str | None = None
    # When set, threaded through as ``ModelSettings(extra_body=...)`` for
    # provider-specific request-body extensions. DeepSeek-v4-pro uses
    # ``{"thinking": {"type": "enabled"}}`` to turn on its thinking mode.
    # Opaque dict — passed verbatim into the chat completions request body.
    extra_body: dict | None = None


# Module-level overrides populated by ``load_model_config`` and consulted
# by ``make_run_config``. Kept here (not in ACTSConfig) because the
# constraints are properties of the *model*, not the run.
_FORCE_TEMPERATURE: float | None = None
_MAX_TOKENS_OVERRIDE: int | None = None
_REASONING_EFFORT_OVERRIDE: str | None = None
_EXTRA_BODY_OVERRIDE: dict | None = None


def load_model_config(path: Path) -> ModelConfig:
    """Load model configuration from a JSON file.

    Expected format::

        {
            "model": "deepseek-v4-pro",
            "url": "https://api.deepseek.com/v1",
            "api_key": "sk-...",                    # optional — see api-key
            "force_temperature": 1.0,               # optional
            "max_tokens": 393216,                   # optional
            "reasoning_effort": "high",             # optional — thinking models
            "extra_body": {"thinking": {"type": "enabled"}}  # optional
        }

    Keeping ``api_key`` out of the JSON is the recommended path so the
    config file can be committed without leaking a secret. The loader
    tries (in order) the JSON literal, ``$OPENAI_API_KEY``, then
    ``$DEEPSEEK_API_KEY`` — the last matches the env var the DeepSeek
    SDK examples use. If none supplies a key, ``ValueError`` is raised
    naming all three sources.
    """
    data = json.loads(path.read_text())
    api_key = (
        data.get("api_key")
        or os.environ.get("OPENAI_API_KEY", "")
        or os.environ.get("DEEPSEEK_API_KEY", "")
    )
    if not api_key:
        raise ValueError(
            f"No API key for model config {path!s}: set 'api_key' in the "
            "JSON or export $OPENAI_API_KEY / $DEEPSEEK_API_KEY."
        )
    force_temp = data.get("force_temperature")
    max_tokens_override = data.get("max_tokens")
    reasoning_effort_override = data.get("reasoning_effort")
    extra_body_override = data.get("extra_body")
    global _FORCE_TEMPERATURE, _MAX_TOKENS_OVERRIDE
    global _REASONING_EFFORT_OVERRIDE, _EXTRA_BODY_OVERRIDE
    _FORCE_TEMPERATURE = float(force_temp) if force_temp is not None else None
    _MAX_TOKENS_OVERRIDE = int(max_tokens_override) if max_tokens_override is not None else None
    _REASONING_EFFORT_OVERRIDE = (
        str(reasoning_effort_override) if reasoning_effort_override is not None else None
    )
    _EXTRA_BODY_OVERRIDE = (
        dict(extra_body_override) if extra_body_override is not None else None
    )
    if _FORCE_TEMPERATURE is not None:
        logger.info(
            "Model config %s pins temperature=%s for every agent (overrides "
            "per-agent values).",
            path, _FORCE_TEMPERATURE,
        )
    if _MAX_TOKENS_OVERRIDE is not None:
        logger.info(
            "Model config %s sets max_tokens=%s for every agent.",
            path, _MAX_TOKENS_OVERRIDE,
        )
    if _REASONING_EFFORT_OVERRIDE is not None:
        logger.info(
            "Model config %s sets reasoning_effort=%s for every agent.",
            path, _REASONING_EFFORT_OVERRIDE,
        )
    if _EXTRA_BODY_OVERRIDE is not None:
        logger.info(
            "Model config %s sets extra_body keys=%s for every agent.",
            path, sorted(_EXTRA_BODY_OVERRIDE.keys()),
        )
    return ModelConfig(
        model=data["model"],
        base_url=data["url"],
        api_key=api_key,
        timeout=data.get("timeout", 300),
        force_temperature=_FORCE_TEMPERATURE,
        max_tokens=_MAX_TOKENS_OVERRIDE,
        reasoning_effort=_REASONING_EFFORT_OVERRIDE,
        extra_body=_EXTRA_BODY_OVERRIDE,
    )


def create_model(config: ModelConfig) -> OpenAIChatCompletionsModel:
    """Create an OpenAIChatCompletionsModel from a ModelConfig.

    This is the single point where the LLM provider is configured.
    Swap providers by changing the ModelConfig.
    """
    client = AsyncOpenAI(
        base_url=config.base_url,
        api_key=config.api_key,
        timeout=config.timeout,
    )
    return OpenAIChatCompletionsModel(
        model=config.model,
        openai_client=client,
    )


async def run_agent(
    agent: Agent,
    prompt: str,
    run_config: RunConfig | None = None,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    retriable: tuple[type[BaseException], ...] = RETRIABLE_EXCEPTIONS,
    max_turns: int | None = None,
) -> RunResult | None:
    """Run an agent with retry on transient OpenAI errors only.

    Retries on rate limits, timeouts, connection errors, and 5xx responses
    with exponential backoff + ±25% jitter starting at ``initial_delay``.
    All other exceptions (auth, schema, programmer bugs) propagate
    immediately — retrying them wastes time and hides the real failure.
    Returns ``None`` only when every retriable attempt has been exhausted.

    *max_turns* bounds the agent's internal tool-use loop (used by the Coder
    to cap self-correction). When ``None``, the SDK default applies.

    *retriable* is exposed for tests so they can inject a synthetic
    exception class without requiring the openai package.
    """
    run_kwargs: dict = {"run_config": run_config}
    if max_turns is not None:
        run_kwargs["max_turns"] = max_turns
    for attempt in range(1, max_retries + 1):
        try:
            return await Runner.run(agent, prompt, **run_kwargs)
        except retriable as exc:
            if attempt == max_retries:
                logger.warning(
                    "LLM retries exhausted after %d attempts (%s): %s",
                    max_retries, type(exc).__name__, exc,
                )
                return None
            wait = initial_delay * (2 ** (attempt - 1)) * random.uniform(0.75, 1.25)
            logger.info(
                "LLM transient error on attempt %d/%d (%s): %s — retrying in %.2fs",
                attempt, max_retries, type(exc).__name__, exc, wait,
            )
            await asyncio.sleep(wait)
    return None


def make_run_config(
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> RunConfig:
    """Create a RunConfig with ModelSettings.

    Honors the module-level overrides populated by ``load_model_config``:
    ``_FORCE_TEMPERATURE`` (reasoning models that reject temp ≠ 1.0:
    Kimi-K2, o1, DeepSeek-Reasoner / -v4-pro), ``_MAX_TOKENS_OVERRIDE``
    (long-context models where 4096 truncates legitimate output),
    ``_REASONING_EFFORT_OVERRIDE`` (thinking-mode toggle for DeepSeek-v4-pro
    and OpenAI o-series), and ``_EXTRA_BODY_OVERRIDE`` (provider-specific
    request-body extensions, e.g. DeepSeek's ``{"thinking": {"type":
    "enabled"}}``).
    """
    effective_temperature = (
        _FORCE_TEMPERATURE if _FORCE_TEMPERATURE is not None else temperature
    )
    effective_max_tokens = (
        _MAX_TOKENS_OVERRIDE if _MAX_TOKENS_OVERRIDE is not None else max_tokens
    )
    reasoning_obj = (
        Reasoning(effort=_REASONING_EFFORT_OVERRIDE)
        if _REASONING_EFFORT_OVERRIDE is not None and Reasoning is not None
        else None
    )
    return RunConfig(
        model_settings=ModelSettings(
            temperature=effective_temperature,
            max_tokens=effective_max_tokens,
            reasoning=reasoning_obj,
            extra_body=_EXTRA_BODY_OVERRIDE,
        ),
    )


# ── submit-tool helpers (shared by Coder/Planner/Reviewer) ────────────────


SUBMIT_OK_SENTINEL = (
    "Submitted. Emit a brief plain-text confirmation now "
    "(no further tool calls) so the run can terminate."
)
"""Returned by every ``submit_*`` tool on success. The wording is what
makes the SDK's tool loop terminate cleanly — drift in this string
silently breaks the loop in whichever agent diverges."""


def format_submit_validation_error(tool_name: str, exc: "ValidationError") -> str:
    """Standard error string returned by submit_* tools on Pydantic
    validation failure. The SDK hands this back to the LLM as the
    tool-call response, prompting an in-loop retry within the existing
    turn budget.
    """
    return f"{tool_name} FAILED:\n{exc}"


def render_kernel_section(kernel_source: str) -> str:
    """Render a kernel source as a fenced markdown ``## Current kernel`` section.

    Triple backticks in the source are escaped so they cannot close the fence.
    """
    safe_source = kernel_source.replace("```", r"\`\`\`")
    return "## Current kernel\n```python\n" + safe_source + "\n```"


def render_run_context(bottleneck: BottleneckType) -> str:
    """Render the once-per-run context section shared by Planner and Reviewer
    prompts. Callers that may not have a bottleneck are expected to gate
    the call themselves rather than relying on a ``None`` return.
    """
    return f"## Run context\n- Bottleneck: {bottleneck.value}"
