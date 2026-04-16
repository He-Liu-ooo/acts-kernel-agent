# LLM Backend — `src/agents/llm_backend.py`

Single integration point between ACTS and the OpenAI Agents SDK.

## Purpose

Isolates all SDK internals so the rest of the codebase never imports from `agents` directly. If the SDK is swapped, only this file changes.

```
planner.py / coder.py / evaluator.py
        ↓
    llm_backend.py      ← only file that imports SDK internals
        ↓
    OpenAI Agents SDK
```

## Provider Swapping

Any OpenAI-compatible API works by changing the model config JSON:

```json
{
    "model": "deepseek-chat",
    "url": "https://api.deepseek.com/v1",
    "api_key": "sk-..."
}
```

DeepSeek, vLLM, Together, OpenAI, etc. — all work via `OpenAIChatCompletionsModel`.

## Components

### ModelConfig

Frozen dataclass: `model`, `base_url`, `api_key`, `timeout`.

### load_model_config(path) -> ModelConfig

Reads the JSON config file.

### create_model(config) -> OpenAIChatCompletionsModel

Creates `AsyncOpenAI` client → wraps in `OpenAIChatCompletionsModel`. Single point for provider configuration.

### run_agent(agent, prompt, ...) -> RunResult | None

Async runner with retry logic (rate limits, timeouts, server errors). Pattern from AccelOpt's `retry_runner_safer`. Returns `None` if all retries exhausted.

### make_run_config(temperature, max_tokens) -> RunConfig

Factory for `RunConfig` + `ModelSettings`.

## SDK Guard

All agent files use `try/except` around SDK imports:

```python
try:
    from agents import Agent, OpenAIChatCompletionsModel
    _SDK_AVAILABLE = True
except ModuleNotFoundError:
    _SDK_AVAILABLE = False
```

This allows the placeholder pipeline to run without the SDK installed (`python -m src.pipeline.optimize` works even without `openai-agents`).
