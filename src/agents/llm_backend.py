"""OpenAI Agents SDK integration — model configuration and runner utilities.

Provides model-swapping via OpenAIChatCompletionsModel: any OpenAI-compatible
API (DeepSeek, vLLM, Together, etc.) works by changing the base URL.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

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

    _SDK_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover
    _SDK_AVAILABLE = False

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


def load_model_config(path: Path) -> ModelConfig:
    """Load model configuration from a JSON file.

    Expected format::

        {
            "model": "deepseek-chat",
            "url": "https://api.deepseek.com/v1",
            "api_key": "sk-..."
        }
    """
    data = json.loads(path.read_text())
    return ModelConfig(
        model=data["model"],
        base_url=data["url"],
        api_key=data["api_key"],
        timeout=data.get("timeout", 300),
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
) -> RunResult | None:
    """Run an agent with retry on transient OpenAI errors only.

    Retries on rate limits, timeouts, connection errors, and 5xx responses
    with exponential backoff + ±25% jitter starting at ``initial_delay``.
    All other exceptions (auth, schema, programmer bugs) propagate
    immediately — retrying them wastes time and hides the real failure.
    Returns ``None`` only when every retriable attempt has been exhausted.

    *retriable* is exposed for tests so they can inject a synthetic
    exception class without requiring the openai package.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return await Runner.run(agent, prompt, run_config=run_config)
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
    """Create a RunConfig with ModelSettings."""
    return RunConfig(
        model_settings=ModelSettings(
            temperature=temperature,
            max_tokens=max_tokens,
        ),
    )
