"""OpenAI Agents SDK integration — model configuration and runner utilities.

Provides model-swapping via OpenAIChatCompletionsModel: any OpenAI-compatible
API (DeepSeek, vLLM, Together, etc.) works by changing the base URL.
"""

from __future__ import annotations

import asyncio
import json
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
    delay: float = 3.0,
) -> RunResult | None:
    """Run an agent with retry logic.

    Retries on transient errors (rate limits, timeouts, server errors).
    Returns None if all retries are exhausted.

    Pattern from AccelOpt's retry_runner_safer.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return await Runner.run(agent, prompt, run_config=run_config)
        except Exception:
            if attempt == max_retries:
                return None
            await asyncio.sleep(delay)
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
