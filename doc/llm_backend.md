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

Any OpenAI-compatible API works by changing the model config JSON. Minimal non-thinking example (DeepSeek-chat / V3, no reasoning fields):

```json
{
    "model": "deepseek-chat",
    "url": "https://api.deepseek.com/v1",
    "api_key": "sk-..."
}
```

DeepSeek, vLLM, Together, OpenAI, etc. — all work via `OpenAIChatCompletionsModel`. The project ships a thinking-mode default instead; see ModelConfig below for the full shape.

## Components

### ModelConfig

Frozen dataclass. Required: `model`, `base_url`, `api_key`. Optional (each carries an `Optional` type, defaults to `None`/`300`):

- `timeout: int = 300` — HTTP request timeout in seconds.
- `force_temperature: float | None` — pins temperature for every agent (Kimi-K2, OpenAI o-series, DeepSeek-Reasoner / -v4-pro all reject temp ≠ 1.0).
- `max_tokens: int | None` — per-response output cap; overrides `make_run_config`'s 4096 default. See **Token budgets** below.
- `reasoning_effort: str | None` — `"low" | "medium" | "high"` for thinking-mode models (DeepSeek-v4-pro, OpenAI o-series). Threaded through as `ModelSettings(reasoning=Reasoning(effort=...))`.
- `extra_body: dict | None` — opaque provider-specific request-body extension. DeepSeek-v4-pro uses `{"thinking": {"type": "enabled"}}` to turn on its thinking mode.

Project default is `configs/models/deepseek.json` = DeepSeek-v4-pro thinking-mode (`force_temperature=1.0`, `reasoning_effort=high`, `extra_body={"thinking": {"type": "enabled"}}`, `max_tokens=393216`) — see **Token budgets** below.

### load_model_config(path) -> ModelConfig

Reads the JSON config file and populates four module-level globals consulted by `make_run_config`: `_FORCE_TEMPERATURE`, `_MAX_TOKENS_OVERRIDE`, `_REASONING_EFFORT_OVERRIDE`, `_EXTRA_BODY_OVERRIDE`. All four overrides are global — every agent (Coder / Planner / Reviewer) sees the same value; there is no per-agent override path today.

**API-key resolution order:** JSON literal `"api_key"` → `$OPENAI_API_KEY` → `$DEEPSEEK_API_KEY`. The last matches the env var the DeepSeek SDK examples use; tried so the same shell environment that runs the canonical client works without an ACTS-specific export. If none supplies a key, `ValueError` is raised naming all three sources.

Example JSON (DeepSeek-v4-pro with thinking enabled):

```json
{
  "model": "deepseek-v4-pro",
  "url": "https://api.deepseek.com/v1",
  "force_temperature": 1.0,
  "max_tokens": 393216,
  "reasoning_effort": "high",
  "extra_body": {"thinking": {"type": "enabled"}}
}
```

Keeping `api_key` out of the JSON is the recommended path so the file can be committed without leaking a secret — export it via `$DEEPSEEK_API_KEY` or `$OPENAI_API_KEY` instead.

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

Factory for `RunConfig` + `ModelSettings`. See **Token budgets** below for what `max_tokens` does and does not bound.

## Token budgets

ACTS only configures the **output** side. The path: `configs/models/<name>.json["max_tokens"]` → `load_model_config` sets module-global `_MAX_TOKENS_OVERRIDE` → `make_run_config` reads it (falling back to the hardcoded `4096` default) → `ModelSettings(max_tokens=...)` → `RunConfig` → `Runner.run` → openai-agents SDK → `AsyncOpenAI.chat.completions.create(max_tokens=...)` → JSON body field on `POST /v1/chat/completions`. The override is global (every agent), not per-agent.

**What `max_tokens` means:**

- Cap on output tokens, not an allocation. Model is free to stop earlier (`finish_reason: "stop"` / `"tool_calls"`); hitting the cap mid-generation returns `finish_reason: "length"` with partial output.
- Bounds **output only** — input prompt tokens are a separate budget.
- For reasoner models (DeepSeek-reasoner, o1, …) the cap covers `reasoning_content` + `content` + tool-call arguments combined. Reasoning alone can consume the entire budget, leaving `content=""` / `tool_calls=null` — observed in `runs/run_20260512T102234_149702Z` where `usage.output_tokens=4096`, `reasoning_tokens=4096`, and the Coder loop terminated with no `submit_kernel` call.
- Setting it above the provider's per-request ceiling returns `400 BadRequestError`. The error is not in `RETRIABLE_EXCEPTIONS`, so it propagates immediately and burns all baseline retries within seconds. Provider ceilings (as of 2026-05-13):
  - **DeepSeek-reasoner**: 65 536 output tokens; 128K context (input + output combined).
  - **DeepSeek-v4-pro**: 393 216 output tokens (384K); 1 048 576 context (1M, input + output combined). Headroom for thinking-mode `reasoning_content` is significantly larger than reasoner's, which is the reason the configured override jumped from 4 096 to 393 216 when ACTS switched defaults.

**Input side is not configured by ACTS.** No `max_input_tokens` / `max_prompt_tokens` knob exists anywhere in the codebase. The effective input limit is the model's context window, enforced server-side. Over-long prompts return a 400. The `ModelSettings.truncation` field (visible as `"truncation": null` in trace `model_config` blocks) would tell the server to auto-truncate the prompt on overflow; ACTS leaves it null, so the default behaviour (provider-specific; DeepSeek = reject, not truncate) applies.

## Multi-turn tool-call flow + reasoning_content

How the SDK threads tool calls between turns for the agents ACTS uses (Coder, Planner, Reviewer), and where reasoning-content handling lives. References below are to the version of `openai-agents` installed in `~/.venvs/acts_run_venv`.

### Per-turn structure

A "turn" is one Chat Completions API call. The `Runner` loop (`agents/run.py`) maintains a growing list of typed *items* that accumulates across turns:

| Item type | Source |
|---|---|
| `ChatCompletionUserMessageParam` / `SystemMessageParam` | initial input |
| `response_output_message` (assistant) | model's text + `tool_calls` from the previous turn |
| `tool_call_item` | tool the model invoked |
| `tool_output_item` | result returned from the tool by the SDK |
| `reasoning_item` | thinking trace from a reasoning model (one per assistant response) |

**Invariant — reasoning_item lifecycle.** Exactly one `reasoning_item` is produced per turn (Chat Completions allows at most one `message.reasoning_content` per response), and it is *output of* turn N, not *input to* it. The item enters the conversation history only on turn N+1, where the replay hook may attach it as `reasoning_content` on the replayed assistant message originally produced by turn N.

On each turn, `_prepare_turn_input_items` (`run_internal/run_loop.py:280`) folds the accumulated items and hands them to `OpenAIChatCompletionsModel.get_response` (`models/openai_chatcompletions.py:106`). The adapter calls `ChatCmplConverter.items_to_messages` (`models/chatcmpl_converter.py:480`) to transform the typed-item list into `ChatCompletionMessageParam[]` for the wire — this is the only place where assistant messages get reconstructed for replay.

### Adapter choice: Chat Completions only

OpenAI ships two API endpoints — `POST /v1/chat/completions` (legacy-stable, broad ecosystem support) and `POST /v1/responses` (newer, richer `output[]` envelope, can carry multiple reasoning blocks per response). The openai-agents SDK has one adapter per endpoint: `OpenAIChatCompletionsModel` and `OpenAIResponsesModel`.

ACTS uses `OpenAIChatCompletionsModel` because DeepSeek and Moonshot **do not implement `/v1/responses`** — they only expose the chat-completions surface. Verified against DeepSeek 2026-05-13:

```
POST https://api.deepseek.com/v1/chat/completions  → 200
POST https://api.deepseek.com/v1/responses         → 404
```

(`/v1/models` returned 200 in the same probe, so the 404 is path-routing, not auth.) Consequence: even though Responses-API would allow multiple reasoning blocks per turn, that's hypothetical for ACTS — we are structurally limited to the chat-completions invariant (one `reasoning_item` per turn) as long as we target these providers.

### The reasoning_content replay hook

The converter walks items and tracks `pending_reasoning_content: str | None` (`chatcmpl_converter.py:514`). When it encounters a `reasoning_item` from a previous turn, it builds a `ReasoningContentReplayContext` and asks a hook whether to replay (`chatcmpl_converter.py:802-815`):

```python
should_replay = (
    should_replay_reasoning_content(replay_context)
    if should_replay_reasoning_content is not None
    else default_should_replay_reasoning_content(replay_context)
)
```

If True, the reasoning summary is stashed; the next assistant message constructed gets it attached as `reasoning_content` (`chatcmpl_converter.py:534-535`) and goes to the provider in turn N+1.

### The default hook is DeepSeek-only

`models/reasoning_content_replay.py:39-51`:

```python
def default_should_replay_reasoning_content(context):
    if "deepseek" not in context.model.lower():
        return False
    # ...DeepSeek-internal origin check
```

Consequences:
- **`deepseek-reasoner` works out of the box.** The hook fires, `reasoning_content` rides the replayed assistant message, the API accepts the multi-turn cycle.
- **`kimi-k2.*` / other non-DeepSeek thinking models fail.** Hook returns False → `reasoning_content` stripped → Moonshot 400 with `"reasoning is enabled but reasoning_content is missing in assistant tool call message at index N"`. Observed 2026-05-12 at the second turn of the Coder's `translate()` loop.

### Enabling other thinking providers

The hook is parameterised — `OpenAIChatCompletionsModel.__init__` (`models/openai_chatcompletions.py:57`) accepts `should_replay_reasoning_content: ShouldReplayReasoningContent | None`. To enable Kimi-K2 (or any non-DeepSeek reasoner) you would:

1. Subclass `OpenAIChatCompletionsModel`, or wrap `create_model` in this file, to pass a custom hook that returns True for the target model.
2. Verify the provider's reasoning-summary shape matches the extractor at `chatcmpl_converter.py:818-825` — if not, the extracted text will be wrong/empty (silent failure) rather than a clean 400.

Not done; the deferred-work flag lives in PROCESS.md → Future.

### Non-thinking fallbacks

The shipped default (`configs/models/deepseek.json` = DeepSeek-v4-pro) is thinking-mode and rides the default replay hook; the alternatives below are *fallbacks*, not the default. For multi-turn-tool-use today, the supported provider/model pairs are:

- DeepSeek: `deepseek-chat` (V3, non-thinking) or `deepseek-reasoner` (R1, replay supported).
- Moonshot: `moonshot-v1-*` family (non-thinking). `kimi-k2.*` (the shape of `configs/models/kimi.json`) requires the custom hook above.
- OpenAI: GPT-5/4o (non-thinking). `o1*` reasoning models are likely supported by the default hook ID-matching, not verified.

### render_kernel_section(kernel_source) -> str

Shared helper that renders a kernel source as a fenced `## Current kernel` markdown section. Triple backticks in the source are escaped so they cannot close the fence. Used by Planner, Reviewer, and Coder prompt assembly to avoid triplicating the fence+escape logic.

### render_run_context(bottleneck) -> str

Shared helper that renders the once-per-run `## Run context\n- Bottleneck (this run): <x>` section consumed by Planner and Reviewer prompts. Takes a non-None `BottleneckType`; callers that may not have a bottleneck (e.g. Planner's first iteration in the placeholder path) gate the call themselves. Keeps the section header + field label in one place so future additions (hardware, workload id) don't drift between agents.

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
