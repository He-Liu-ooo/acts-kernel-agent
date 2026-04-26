# LLM Backend — `src/agents/llm_backend.py`

Single integration point between ACTS and the OpenAI Agents SDK.

## Purpose

Isolates all SDK internals so the rest of the codebase never imports from `agents` directly. If the SDK is swapped, only this file changes.

```
planner.py / coder.py / reviewer.py
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

Async runner with a **narrow** retry policy. Only retries a fixed tuple of transient openai exceptions (`RateLimitError`, `APITimeoutError`, `APIConnectionError`, `InternalServerError`). Every other exception (auth, schema, programmer bugs) propagates immediately — retrying them wastes wall-clock and hides the real cause.

- **Backoff**: exponential with ±25% jitter. Sleep duration = `initial_delay * 2^(attempt-1) * uniform(0.75, 1.25)`.
- **Logging**: named logger (`src.agents.llm_backend`). `logger.info` per transient retry; `logger.warning` when retries are exhausted — both include the exception class name so the orchestrator can diagnose.
- **Return value**: `RunResult` on success, `None` only after all retriable attempts are exhausted.
- **Test injection**: the `retriable` parameter is exposed so tests can pass a synthetic exception class without requiring the real `openai` package installed.
- **Optional `max_turns` kwarg**: when not `None`, forwarded to `Runner.run` to bound the SDK's internal tool-use loop. Used by `CoderAgent` to cap self-correction; Planner/Reviewer cap their `submit_*` tool loops at `max_turns=4` (= 2N+2 with N=1 in-band validation retry: one invalid submit + one corrected submit + one confirmation + one buffer turn).

### make_run_config(temperature, max_tokens) -> RunConfig

Factory for `RunConfig` + `ModelSettings`.

### render_kernel_section(kernel_source) -> str

Shared helper that renders a kernel source as a fenced `## Current kernel` markdown section. Triple backticks in the source are escaped so they cannot close the fence. Used by Planner, Reviewer, and Coder prompt assembly to avoid triplicating the fence+escape logic.

### render_run_context(bottleneck) -> str

Shared helper that renders the once-per-run `## Run context\n- Bottleneck: <x>` section consumed by Planner and Reviewer prompts. Takes a non-None `BottleneckType`; callers that may not have a bottleneck (e.g. Planner's first iteration in the placeholder path) gate the call themselves. Keeps the section header + field label in one place so future additions (hardware, workload id) don't drift between agents.

## Shared submit-tool helpers

Two module-level exports consumed by every `submit_*` tool — Coder's `submit_kernel` (`_make_submit_tool` in `coder.py`), Planner's `submit_plan` (`_make_submit_plan_tool` in `planner.py`), and Reviewer's `submit_review` (`_make_submit_review_tool` in `reviewer.py`). Centralising them keeps the success/failure protocol identical across agents so the SDK tool loop terminates the same way everywhere.

### SUBMIT_OK_SENTINEL (str)

Return value of every `submit_*` tool's success path. The wording is load-bearing: it instructs the LLM to emit a brief plain-text confirmation (no further tool calls), which is what makes the SDK's tool loop terminate cleanly. Drift in this string across agents would silently break the loop in whichever agent diverged, so the constant lives once in `llm_backend.py` and all three submit tools import it.

### format_submit_validation_error(tool_name, exc) -> str

Produces the standard error string `"<tool_name> FAILED:\n<exc>"` returned by `submit_*` tools when Pydantic validation rejects the LLM's payload. The SDK hands this string back to the LLM as the tool-call response, which triggers an in-loop retry (bounded by `max_turns`) without aborting the run. Sharing the format keeps the failure surface uniform so prompt instructions about "what a FAILED response looks like" stay valid for every agent.

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
