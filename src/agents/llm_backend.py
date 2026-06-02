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

    from src.actions.registry import Action
    from src.config import HardwareSpec
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


def render_action_menu(actions: list["Action"]) -> str:
    """Render applicable actions as a compact Planner selection menu.

    One block per action: ``- <id> (<name>, tier <n>): <description>`` plus
    optional advisory lines (``when:`` preconditions, ``knobs:`` parameters,
    ``impact:`` expected_impact). Empty optional fields drop their line; an
    empty ``actions`` list yields "" (caller falls back to the bare-ID list).
    """
    if not actions:
        return ""
    blocks: list[str] = []
    for a in actions:
        tier_no = a.tier.value if hasattr(a.tier, "value") else a.tier
        lines = [f"- {a.id} ({a.name}, tier {tier_no}): {a.description}"]
        if a.preconditions:
            lines.append(f"    when: {', '.join(a.preconditions)}")
        if a.parameters:
            knobs = ", ".join(f"{k}={v}" for k, v in a.parameters.items())
            lines.append(f"    knobs: {knobs}")
        if a.expected_impact:
            lines.append(f"    impact: {a.expected_impact}")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def render_technique_guidance(action: "Action") -> str:
    """Render the SELECTED technique's implementation guidance for the Coder.

    Concatenates ``guidance`` and an ``anti_patterns`` bullet list; returns
    "" when both are empty (Coder omits the ``## Technique guidance`` section).
    """
    parts: list[str] = []
    if action.guidance:
        parts.append(action.guidance)
    if action.anti_patterns:
        parts.append(
            "Anti-patterns to avoid:\n"
            + "\n".join(f"- {ap}" for ap in action.anti_patterns)
        )
    return "\n\n".join(parts)


def _tensor_core_tile_for(
    compute_capability: float, dominant_dtype: str
) -> str | None:
    """Look up the canonical Tensor Core tile shape for an arch + dtype.

    Returns the per-arch native tile (e.g. ``"m16n16k16"`` for fp16 on
    Ampere/Ada) so the renderer can surface BLOCK_K guidance to the Coder.
    Returns ``None`` when:
    - ``compute_capability < 7.0`` (pre-Volta, no Tensor Cores),
    - the arch supports Tensor Cores but not for *this* dtype
      (e.g. fp8 on Volta/Turing/Ampere),
    - the arch is unknown (defensive default).

    Ties in ``dominant_dtype`` are slash-joined (e.g. ``"bf16/fp16"``) by
    the caller; this helper picks the alphabetical-first member, matching
    the rest of the rendering convention.
    """
    if compute_capability < 7.0:
        return None
    dtype = dominant_dtype.split("/")[0] if "/" in dominant_dtype else dominant_dtype

    # Volta + Turing: fp16 only on Tensor Cores.
    if 7.0 <= compute_capability < 8.0:
        if dtype == "fp16":
            return "m16n16k16"
        return None
    # Ampere: fp16/bf16 → m16n16k16, tf32 → m16n16k8.
    if 8.0 <= compute_capability < 8.9:
        if dtype in ("fp16", "bf16"):
            return "m16n16k16"
        if dtype == "tf32":
            return "m16n16k8"
        return None
    # Ada (sm_89): Ampere set + fp8 → m16n16k32.
    if compute_capability == 8.9:
        if dtype in ("fp16", "bf16"):
            return "m16n16k16"
        if dtype == "tf32":
            return "m16n16k8"
        if dtype == "fp8":
            return "m16n16k32"
        return None
    # Hopper (sm_90): WGMMA family — single prose descriptor.
    if compute_capability == 9.0:
        return "WGMMA m64×N×K (N multiple of 8; K=16 for fp16/bf16, 32 for fp8)"
    # Blackwell (sm_100+): tcgen05 placeholder — exact tile shapes are
    # still moving; render a pointer instead of risking a stale literal.
    if compute_capability >= 10.0:
        return "tcgen05 — see docs"
    return None


def render_run_context(
    bottleneck: "BottleneckType | None" = None,
    *,
    hardware: "HardwareSpec | None" = None,
    workload_shapes: list[tuple[int, ...]] | None = None,
) -> str:
    """Render the once-per-run context section shared by all three agent prompts.

    The ``hardware`` kwarg appends the hw-budget block under the bottleneck
    line: SMEM caps, SM count, max threads/block, L2 cache, peak FLOPS
    table, peak DRAM bandwidth, Tensor Core tile descriptor, and the
    per-Config SMEM rule. ``hardware=None`` or an empty-named
    ``HardwareSpec`` (no spec configured) falls back to bottleneck-only.

    ``bottleneck=None`` is supported for the baseline-generation path
    (``CoderAgent.translate``) where the once-per-run bottleneck has not
    yet been classified. The function still renders the hw block when
    ``hardware`` is supplied; the bottleneck line reads "not yet classified
    (baseline generation)". When BOTH bottleneck and hardware are None,
    returns the empty string so the caller can skip section emission
    without a guard.

    ``workload_shapes`` (when supplied + non-empty) appends a "Workload
    shapes:" line after the hw block. Up to 3 shapes render literally as
    tuples; more than 3 summarize as per-dim min-max ranges with the
    total count. Tuple shape is caller-defined (each element is whatever
    dim ordering the orchestrator passes — typically `(M, N, K)` for
    matmul-shape problems, derived from `Workload.axes.values()`).

    Per the hw-spec-injection spec (doc/specs/2026-05-24-coding-hw-spec-
    design.md §5.1), the rule line is rendered in NAMING-AGNOSTIC prose
    (``input_tile_elements_loaded_to_smem`` rather than ``BLOCK_M+BLOCK_N``)
    so the prompt doesn't presume any specific meta-param naming
    convention. The actual SMEM check (Phase B, lives in
    ``src/agents/coder.py::_make_compile_tool``) reads ptxas truth from
    ``CompiledKernel.metadata.shared`` — naming-free.
    """
    has_bottleneck = bottleneck is not None
    has_hw = hardware is not None and hardware.name
    if not has_bottleneck and not has_hw:
        return ""

    lines = ["## Run context"]
    if has_bottleneck:
        lines.append(f"- Bottleneck (this run): {bottleneck.value}")
    else:
        lines.append("- Bottleneck (this run): not yet classified (baseline generation)")
    if has_hw:
        # All non-zero dtype peaks render together, sorted descending by
        # value, with ties grouped (alphabetical, '/'-joined). Replaces
        # the prior single-dominant-dtype pick (spec §3 decision 8) — on
        # Ada/Hopper that picked fp8 as "dominant" and a bf16-workload
        # LLM would measure pct_peak against the wrong ceiling. Showing
        # every peak lets the LLM (and the Reviewer's pct_peak prose)
        # pick the right one for its workload dtype. Includes
        # Hopper/Blackwell low-precision Tensor Core peaks (fp8, nvfp4)
        # and tf32 — int8 is omitted because it's TOPS, not TFLOPS.
        dtype_peaks = {
            "fp32": hardware.peak_flops_fp32,
            "tf32": hardware.peak_flops_tf32,
            "bf16": hardware.peak_flops_bf16,
            "fp16": hardware.peak_flops_fp16,
            "fp8": hardware.peak_flops_fp8,
            "nvfp4": hardware.peak_flops_nvfp4,
        }
        cap = hardware.shared_mem_per_block_bytes
        per_sm = hardware.shared_mem_per_multiprocessor_bytes
        # CUDA convention: sm_<major><minor> as concatenated integers
        # (e.g. 8.9 → "sm_89", 9.0 → "sm_90"). Float repr "sm_8.9" is
        # not a valid arch identifier.
        sm_id = f"sm_{int(round(hardware.compute_capability * 10))}"
        lines.extend([
            f"- Hardware: {hardware.name} ({sm_id})",
            f"- Shared mem per block: {cap} B (~{cap // 1024} KB)",
            f"- Shared mem per SM: {per_sm} B (~{per_sm // 1024} KB)",
        ])
        # SM count + max threads/block — agents use these for grid sizing
        # and num_warps ceilings respectively. Omit each line when the
        # underlying HardwareSpec field is 0 (mirrors the zero-peak-FLOPS
        # omit policy from fix #7).
        if hardware.sm_count > 0:
            lines.append(f"- SM count: {hardware.sm_count}")
        if hardware.max_threads_per_block > 0:
            max_warps = hardware.max_threads_per_block // 32
            lines.append(
                f"- Max threads per block: {hardware.max_threads_per_block} "
                f"(warp size 32 → num_warps ≤ {max_warps})"
            )
        # L2 cache — Triton can't control L2 directly but knowing the
        # capacity informs tile-reuse reasoning (does the working set
        # fit in L2 across SMs?). Surfaced from SRAM_capacity.
        if hardware.SRAM_capacity > 0:
            l2_mib = hardware.SRAM_capacity // (1024 * 1024)
            lines.append(
                f"- L2 cache: {hardware.SRAM_capacity} B (~{l2_mib} MiB)"
            )
        # When the operator hasn't populated any MAC_per_cycle_* fields,
        # every peak is 0 — skip the line entirely rather than emit a
        # meaningless "0.0 TFLOPS" entry. Otherwise group ties (peaks
        # within 1e-3 TFLOPS are considered equal — guards against
        # float-equality false-misses from arithmetic in `peak_flops_*`
        # properties) and render highest-to-lowest.
        nonzero = [(d, p) for d, p in dtype_peaks.items() if p > 0]
        dominant_dtype: str | None = None
        if nonzero:
            by_value: dict[float, list[str]] = {}
            for d, p in nonzero:
                # Bucket by rounded value so bf16==fp16 ties resolve
                # even when derived from independent properties.
                key = round(p, 3)
                by_value.setdefault(key, []).append(d)
            entries = []
            sorted_peaks = sorted(by_value, reverse=True)
            for p in sorted_peaks:
                names = "/".join(sorted(by_value[p]))
                entries.append(f"{names}={p:.1f}")
            lines.append(
                f"- Peak FLOPS (TFLOPS): {' · '.join(entries)}"
            )
            # Dominant-dtype pick for the Tensor Core tile lookup only —
            # the slash-joined name of the highest-peak group (e.g.
            # "bf16/fp16" when those tie at Ada's TC peak). Renderer
            # passes this slug through to _tensor_core_tile_for, which
            # picks the alphabetical-first member internally.
            top = sorted_peaks[0]
            dominant_dtype = "/".join(sorted(by_value[top]))
        lines.append(
            f"- Peak DRAM bandwidth: {hardware.peak_memory_bandwidth_gb_s:.0f} GB/s"
        )
        # Tensor Core tile descriptor — derived from compute_capability +
        # dominant_dtype, omitted when the helper returns None (pre-Volta,
        # unsupported dtype on this arch, or unknown arch).
        if dominant_dtype is not None:
            tc_tile = _tensor_core_tile_for(
                hardware.compute_capability, dominant_dtype
            )
            if tc_tile is not None:
                lines.append(
                    f"- Tensor Core tile ({dominant_dtype}): {tc_tile}"
                )
        lines.append(
            f"- Per-Config shared-mem rule: num_stages × "
            f"(input_tile_elements_loaded_to_smem) × dtype_bytes ≤ {cap}"
        )

    # Workload shapes — appended after the hw block when the orchestrator
    # supplies the iter-time workload set. Short lists render literally;
    # long lists summarize as per-dim min-max ranges with the count.
    if workload_shapes:
        if len(workload_shapes) <= 3:
            shape_str = ", ".join(
                "(" + ", ".join(str(v) for v in s) + ")" for s in workload_shapes
            )
            lines.append(f"- Workload shapes: {shape_str}")
        else:
            # Per-dim min-max — uses the first shape's length as the dim
            # count; ragged shapes (mixed lengths) fall back to the
            # shortest common prefix to avoid index-out-of-range on
            # off-axis dims.
            min_len = min(len(s) for s in workload_shapes)
            ranges = []
            for dim_idx in range(min_len):
                vals = [s[dim_idx] for s in workload_shapes]
                ranges.append(f"{min(vals)}-{max(vals)}")
            lines.append(
                f"- Workload shapes: (N={len(workload_shapes)}) "
                f"ranges {{{', '.join(ranges)}}}"
            )
    return "\n".join(lines)
