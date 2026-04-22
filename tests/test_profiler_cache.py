"""Tests for ``profile_kernel`` façade + source-hash-keyed NCU cache.

Tier 1: GPU-free. Reuses the ``fake_ncu`` shell-script pattern from
``tests/test_profiler_subprocess.py`` (copied here rather than factored
into a shared conftest — a clean extraction seam can come at review
time once both files are stable).

Covers, per spec §4.3 and Task #4 acceptance criteria:
  * cache miss → subprocess run + parse + disk write
  * cache hit  → subprocess skipped, on-disk JSON rehydrated
  * key components: source hash + workload + mode + _METRIC_SET_VERSION
  * version bump busts all entries
  * atomic write: temp file + rename, no partial files on failure
  * degraded results are NEVER persisted (degraded is always re-tried)
  * analytical path re-runs every call — cache holds only the NCU piece
  * analytical failure raises ``ProfilerError`` regardless of cache state
  * ``ncu`` binary missing → degraded result, no cache write, no crash
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from conftest import rtx6000_ada_hardware as _rtx6000_ada
from src.config import HardwareSpec
from src.eval import profiler as profiler_mod
from src.eval.profiler import (
    NCUMetrics,
    ProfilerError,
    ProfilingResult,
    profile_kernel,
)
from src.eval.roofline import BottleneckType
from src.kernels.kernel import Kernel, KernelSpec, KernelType


# ── hardware / kernel fixtures ─────────────────────────────────────────────


@pytest.fixture
def sample_kernel() -> Kernel:
    # Source is imported by ``compile_kernel`` inside ``profile_kernel``,
    # so it must at least define the entrypoint. The body is a no-op — the
    # fake ncu never execs the driver, so the callable is never invoked.
    return Kernel(
        spec=KernelSpec(
            name="my_elementwise",
            kernel_type=KernelType.ELEMENTWISE,
            entrypoint="elementwise_add_kernel",
        ),
        source_code=(
            "# kernel source v1 — tier 1 stub; compile_kernel needs the "
            "entrypoint to resolve\n"
            "def elementwise_add_kernel(*args, **kwargs):\n"
            "    return None\n"
        ),
    )


@pytest.fixture
def sample_workload() -> dict:
    return {"uuid": "workload-0", "axes": {"N": 1024}, "inputs": {}}


def _identity_input_generator(seed: int = 0) -> tuple:
    return ()


# ── fake_ncu machinery (copied pattern from test_profiler_subprocess.py) ──


def _canned_csv() -> str:
    """Minimal but parser-valid NCU CSV: one kernel row per curated metric
    + all 18 stalls. ``elementwise_add_kernel`` is the entrypoint the
    sample_kernel fixture uses — parser does substring match on the
    ``Kernel Name`` column.

    Values are chosen so ``_parse_ncu_csv`` returns a fully-populated
    ``NCUMetrics`` (degraded=False, reason=None)."""
    header = (
        '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
    )
    rows: list[str] = []

    def row(metric: str, value: str) -> str:
        return f'"0","elementwise_add_kernel","{metric}","%","{value}"\n'

    # Curated required + optional.
    rows.append(row("sm__warps_active.avg.pct_of_peak_sustained_active", "55.0"))
    rows.append(row("lts__t_sector_hit_rate.pct", "72.5"))
    rows.append(row("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active", "0"))

    # All 18 stalls (values arbitrary — only ranking matters to parser).
    stall_values = {
        "barrier": "0",
        "branch_resolving": "5",
        "dispatch_stall": "10",
        "drain": "15",
        "imc_miss": "20",
        "lg_throttle": "25",
        "long_scoreboard": "80",   # dominant
        "math_pipe_throttle": "35",
        "membar": "40",
        "mio_throttle": "45",
        "misc": "50",
        "no_instruction": "55",
        "not_selected": "60",
        "selected": "65",
        "short_scoreboard": "70",  # runner-up
        "sleeping": "1",
        "tex_throttle": "2",
        "wait": "3",
    }
    for reason, val in stall_values.items():
        rows.append(row(f"smsp__average_warp_latency_issue_stalled_{reason}.pct", val))

    # NCU banner prefix the parser must skip.
    banner = (
        "==PROF== Connected to process 1 (/usr/bin/python3.10)\n"
        "ok\n"
        "==PROF== Disconnected from process 1\n"
    )
    return banner + header + "".join(rows)


@pytest.fixture
def fake_ncu(tmp_path, monkeypatch):
    """Install a shell script named ``ncu`` in ``tmp_path`` that emits the
    canned CSV and increments a counter file on every invocation.

    Yields a ``(counter_path, script_path)`` pair — tests read the counter
    file to assert invocation counts.
    """
    counter = tmp_path / "ncu_calls.txt"
    counter.write_text("0")
    script = tmp_path / "ncu"

    csv = _canned_csv()
    # Escape for embedding in a shell heredoc: nothing special needed
    # since the heredoc delimiter is quoted (``<<"EOF"`` → no expansion).
    body = (
        "#!/usr/bin/env bash\n"
        f"n=$(cat {counter})\n"
        f"echo $((n+1)) > {counter}\n"
        'cat <<"NCUEOF"\n'
        + csv
        + "NCUEOF\n"
    )
    script.write_text(body)
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ.get('PATH', '')}")
    # Bust the module-level ncu-discovery cache between tests so each
    # test sees the fake_ncu on its monkeypatched PATH.
    monkeypatch.setattr(profiler_mod, "_NCU_BINARY_CACHE", profiler_mod._UNSET, raising=False)

    return counter, script


@pytest.fixture(autouse=True)
def _reset_ncu_cache(monkeypatch):
    """Safety net — force ``_discover_ncu_binary`` to re-probe PATH for
    every test. Without this a cached miss/hit leaks across tests."""
    monkeypatch.setattr(profiler_mod, "_NCU_BINARY_CACHE", profiler_mod._UNSET, raising=False)


def _reads_counter(counter: Path) -> int:
    return int(counter.read_text().strip())


def _call_profile(
    kernel: Kernel,
    workload: dict,
    *,
    cache_dir: Path,
    latency_s: float = 1e-3,
    mode: str = "curated",
    hardware_spec: HardwareSpec | None = None,
) -> ProfilingResult:
    return profile_kernel(
        kernel,
        workload,
        _identity_input_generator,
        hardware_spec=hardware_spec or _rtx6000_ada(),
        flops=1_000_000,
        nbytes=4_000_000,
        latency_s=latency_s,
        mode=mode,
        timeout_s=10.0,
        cache_dir=cache_dir,
    )


# ── cache miss → full path ─────────────────────────────────────────────────


def test_cache_miss_invokes_ncu_and_writes_entry(
    tmp_path, fake_ncu, sample_kernel, sample_workload
):
    counter, _ = fake_ncu
    cache_dir = tmp_path / "cache"
    assert _reads_counter(counter) == 0

    result = _call_profile(sample_kernel, sample_workload, cache_dir=cache_dir)

    assert _reads_counter(counter) == 1, "NCU must be invoked exactly once on cache miss"
    assert isinstance(result, ProfilingResult)
    assert result.degraded is False
    assert result.ncu is not None
    assert result.ncu.sm_occupancy_pct == pytest.approx(55.0)
    assert result.ncu.warp_stall_dominant == "long_scoreboard"

    # A single cache file was written.
    entries = list(cache_dir.glob("*.json"))
    assert len(entries) == 1


# ── cache hit skips subprocess ─────────────────────────────────────────────


def test_cache_hit_skips_subprocess(tmp_path, fake_ncu, sample_kernel, sample_workload):
    counter, _ = fake_ncu
    cache_dir = tmp_path / "cache"

    first = _call_profile(sample_kernel, sample_workload, cache_dir=cache_dir)
    assert _reads_counter(counter) == 1
    assert first.ncu is not None

    # Second call with identical inputs — must rehydrate from disk.
    second = _call_profile(sample_kernel, sample_workload, cache_dir=cache_dir)
    assert _reads_counter(counter) == 1, "cache hit must not invoke NCU"
    assert second.ncu is not None
    assert second.ncu.sm_occupancy_pct == first.ncu.sm_occupancy_pct
    assert second.ncu.warp_stall_dominant == first.ncu.warp_stall_dominant
    assert second.degraded is False


# ── cache key components ───────────────────────────────────────────────────


def test_cache_key_source_hash_bust(tmp_path, fake_ncu, sample_kernel, sample_workload):
    """Different kernel source → different key → NCU re-runs."""
    counter, _ = fake_ncu
    cache_dir = tmp_path / "cache"

    _call_profile(sample_kernel, sample_workload, cache_dir=cache_dir)
    assert _reads_counter(counter) == 1

    altered = Kernel(
        spec=sample_kernel.spec,
        source_code=sample_kernel.source_code + "# tweaked\n",
    )
    _call_profile(altered, sample_workload, cache_dir=cache_dir)
    assert _reads_counter(counter) == 2


def test_cache_key_workload_bust(tmp_path, fake_ncu, sample_kernel, sample_workload):
    counter, _ = fake_ncu
    cache_dir = tmp_path / "cache"

    _call_profile(sample_kernel, sample_workload, cache_dir=cache_dir)
    assert _reads_counter(counter) == 1

    alt_workload = {"uuid": "workload-1", "axes": {"N": 2048}, "inputs": {}}
    _call_profile(sample_kernel, alt_workload, cache_dir=cache_dir)
    assert _reads_counter(counter) == 2


def test_cache_key_mode_bust(tmp_path, fake_ncu, sample_kernel, sample_workload):
    counter, _ = fake_ncu
    cache_dir = tmp_path / "cache"

    _call_profile(sample_kernel, sample_workload, cache_dir=cache_dir, mode="curated")
    assert _reads_counter(counter) == 1

    _call_profile(sample_kernel, sample_workload, cache_dir=cache_dir, mode="full")
    assert _reads_counter(counter) == 2


def test_cache_key_metric_set_version_bust(
    tmp_path, fake_ncu, sample_kernel, sample_workload, monkeypatch
):
    """Bumping ``_METRIC_SET_VERSION`` invalidates prior entries — the old
    key is unreachable from the new version string."""
    counter, _ = fake_ncu
    cache_dir = tmp_path / "cache"

    _call_profile(sample_kernel, sample_workload, cache_dir=cache_dir)
    assert _reads_counter(counter) == 1

    # Simulate a version bump. Any key computed under the new version is
    # a fresh namespace — previous files still exist on disk but their
    # filenames no longer match.
    monkeypatch.setattr(profiler_mod, "_METRIC_SET_VERSION", "v999")
    _call_profile(sample_kernel, sample_workload, cache_dir=cache_dir)
    assert _reads_counter(counter) == 2


# ── atomic writes ──────────────────────────────────────────────────────────


def test_atomic_write_uses_temp_and_replace(
    tmp_path, fake_ncu, sample_kernel, sample_workload, monkeypatch
):
    """If ``os.replace`` raises mid-write, no partial ``<key>.json`` file
    should remain in the cache dir — the tempfile is cleaned up."""
    counter, _ = fake_ncu
    cache_dir = tmp_path / "cache"

    real_replace = os.replace
    call_count = {"n": 0}

    def boom_replace(src, dst):
        call_count["n"] += 1
        # Simulate rename failure after the temp file exists.
        raise OSError("disk went away")

    monkeypatch.setattr(profiler_mod.os, "replace", boom_replace)

    # Profile call must still succeed end-to-end — cache-write failure is
    # best-effort and should not raise.
    result = _call_profile(sample_kernel, sample_workload, cache_dir=cache_dir)
    assert result.ncu is not None
    assert result.degraded is False

    # Rename was attempted exactly once.
    assert call_count["n"] == 1
    # No final cache entry was produced.
    final_json = list(cache_dir.glob("*.json"))
    assert final_json == [], f"partial file leaked: {final_json}"

    # Restore replace for the next assertion — a subsequent call writes cleanly.
    monkeypatch.setattr(profiler_mod.os, "replace", real_replace)
    _call_profile(sample_kernel, sample_workload, cache_dir=cache_dir)
    assert len(list(cache_dir.glob("*.json"))) == 1


# ── degraded results NEVER cached ──────────────────────────────────────────


def test_driver_degraded_result_not_cached(tmp_path, sample_kernel, sample_workload, monkeypatch):
    """Driver-side failure (non-zero exit) → degraded result → no cache
    write → second call re-invokes NCU."""
    call_count = {"n": 0}

    def fake_run_ncu(*args, **kwargs):
        call_count["n"] += 1
        return "", 3, True, "ncu_nonzero_exit:3"

    # Bypass PATH lookup so _discover_ncu_binary returns a non-None
    # "present" sentinel. Tests run in venvs where ncu may not exist.
    monkeypatch.setattr(profiler_mod, "_discover_ncu_binary", lambda: "/usr/bin/ncu")
    monkeypatch.setattr(profiler_mod, "_run_ncu", fake_run_ncu)

    cache_dir = tmp_path / "cache"
    first = _call_profile(sample_kernel, sample_workload, cache_dir=cache_dir)
    assert first.degraded is True
    assert first.ncu is None
    assert list(cache_dir.glob("*.json")) == [], "degraded driver result must NOT be cached"

    second = _call_profile(sample_kernel, sample_workload, cache_dir=cache_dir)
    assert second.degraded is True
    assert call_count["n"] == 2, "driver degradation must not be cached — second call re-runs"


def test_parser_degraded_result_not_cached(tmp_path, sample_kernel, sample_workload, monkeypatch):
    """Parser-side failure (garbage CSV) → degraded → no cache write."""
    call_count = {"n": 0}

    def fake_run_ncu(*args, **kwargs):
        call_count["n"] += 1
        return "garbage not a csv", 0, False, None  # driver-clean, parser-dirty

    monkeypatch.setattr(profiler_mod, "_discover_ncu_binary", lambda: "/usr/bin/ncu")
    monkeypatch.setattr(profiler_mod, "_run_ncu", fake_run_ncu)

    cache_dir = tmp_path / "cache"
    first = _call_profile(sample_kernel, sample_workload, cache_dir=cache_dir)
    assert first.degraded is True
    assert first.ncu is None
    assert list(cache_dir.glob("*.json")) == []

    second = _call_profile(sample_kernel, sample_workload, cache_dir=cache_dir)
    assert second.degraded is True
    assert call_count["n"] == 2


# ── analytical path always runs ────────────────────────────────────────────


def test_analytical_recomputed_on_cache_hit(
    tmp_path, fake_ncu, sample_kernel, sample_workload
):
    """Cache stores only NCUMetrics. ``result.analytical`` is recomputed
    from the caller's ``latency_s`` on every call — a different latency
    on the second call must yield a different ``achieved_tflops``."""
    counter, _ = fake_ncu
    cache_dir = tmp_path / "cache"

    first = _call_profile(
        sample_kernel, sample_workload, cache_dir=cache_dir, latency_s=1e-3
    )
    assert _reads_counter(counter) == 1

    second = _call_profile(
        sample_kernel, sample_workload, cache_dir=cache_dir, latency_s=2e-3
    )
    assert _reads_counter(counter) == 1, "second call must be a cache hit"

    # Achieved TFLOPs halves when latency doubles.
    assert second.analytical.achieved_tflops == pytest.approx(
        first.analytical.achieved_tflops / 2, rel=1e-9
    )
    # But NCU piece is the same (rehydrated).
    assert second.ncu.sm_occupancy_pct == first.ncu.sm_occupancy_pct


# ── analytical failures kill the branch ────────────────────────────────────


def test_profiler_error_on_bad_latency_even_with_cache(
    tmp_path, fake_ncu, sample_kernel, sample_workload
):
    cache_dir = tmp_path / "cache"
    with pytest.raises(ProfilerError):
        _call_profile(sample_kernel, sample_workload, cache_dir=cache_dir, latency_s=0.0)


def test_profiler_error_on_bad_latency_with_populated_cache(
    tmp_path, fake_ncu, sample_kernel, sample_workload
):
    """Populate cache first, then call again with bad latency: still raises."""
    cache_dir = tmp_path / "cache"
    _call_profile(sample_kernel, sample_workload, cache_dir=cache_dir, latency_s=1e-3)
    with pytest.raises(ProfilerError):
        _call_profile(sample_kernel, sample_workload, cache_dir=cache_dir, latency_s=-1.0)


# ── ncu binary not found ───────────────────────────────────────────────────


def test_ncu_binary_missing_returns_degraded_no_cache_write(
    tmp_path, sample_kernel, sample_workload, monkeypatch
):
    """Empty ``$PATH`` → ``_discover_ncu_binary`` returns None → profile
    returns a degraded ``ProfilingResult`` without raising and without
    writing any cache entry. The analytical side still populated."""
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))

    cache_dir = tmp_path / "cache"
    result = _call_profile(sample_kernel, sample_workload, cache_dir=cache_dir)

    assert isinstance(result, ProfilingResult)
    assert result.ncu is None
    assert result.degraded is True
    assert result.degraded_reason == "ncu_binary_not_found"
    assert result.analytical is not None  # analytical always computed
    # No cache entry was written.
    if cache_dir.exists():
        assert list(cache_dir.glob("*.json")) == []


# ── no cache_dir → still works ─────────────────────────────────────────────


def test_no_cache_dir_runs_ncu_every_call(tmp_path, fake_ncu, sample_kernel, sample_workload):
    """``cache_dir=None`` bypasses the cache entirely — every call invokes NCU."""
    counter, _ = fake_ncu

    first = profile_kernel(
        sample_kernel,
        sample_workload,
        _identity_input_generator,
        hardware_spec=_rtx6000_ada(),
        flops=1_000_000,
        nbytes=4_000_000,
        latency_s=1e-3,
        mode="curated",
        timeout_s=10.0,
        cache_dir=None,
    )
    assert _reads_counter(counter) == 1
    assert first.ncu is not None

    profile_kernel(
        sample_kernel,
        sample_workload,
        _identity_input_generator,
        hardware_spec=_rtx6000_ada(),
        flops=1_000_000,
        nbytes=4_000_000,
        latency_s=1e-3,
        mode="curated",
        timeout_s=10.0,
        cache_dir=None,
    )
    assert _reads_counter(counter) == 2, "cache_dir=None must never cache"


# ── problem_definition_path forwarding ─────────────────────────────────────


def test_problem_definition_path_forwarded_to_run_ncu(
    tmp_path, sample_kernel, sample_workload, monkeypatch
):
    """``profile_kernel`` must thread ``problem_definition_path`` through to
    ``_run_ncu`` — otherwise the driver can't rebuild inputs for real
    Coder outputs (which don't define ``make_inputs``)."""
    captured: dict = {}

    def fake_run_ncu(*args, **kwargs):
        captured.update(kwargs)
        return "", 0, True, "short-circuit"

    monkeypatch.setattr(profiler_mod, "_discover_ncu_binary", lambda: "/usr/bin/ncu")
    monkeypatch.setattr(profiler_mod, "_run_ncu", fake_run_ncu)

    problem_path = tmp_path / "definition.json"
    problem_path.write_text("{}")

    profile_kernel(
        sample_kernel,
        sample_workload,
        _identity_input_generator,
        hardware_spec=_rtx6000_ada(),
        flops=1_000_000,
        nbytes=4_000_000,
        latency_s=1e-3,
        mode="curated",
        timeout_s=10.0,
        cache_dir=None,
        problem_definition_path=problem_path,
    )

    assert captured.get("problem_definition_path") == problem_path
