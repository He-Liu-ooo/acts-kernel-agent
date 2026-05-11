"""Tier 1 tests for `--config`-driven gpu_index plumbing in src/pipeline/optimize.py.

The CLI shrank to three flags (`--config`, `--run-dir`, `--trace-dir`) on
2026-05-11; `--gpu-index`, `--reset-clocks`, and the `problem_path`
positional now live in the `.cfg` file. The module-top preparse opens
the cfg before any CUDA-aware import so `CUDA_VISIBLE_DEVICES` lands
in time.
"""

from __future__ import annotations

import os
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest


# ── config-path preparse ─────────────────────────────────────────────────────


def test_preparse_config_path_with_space():
    from src.pipeline.optimize import _preparse_config_path
    assert _preparse_config_path(["prog", "--config", "/tmp/x.cfg"]) == "/tmp/x.cfg"


def test_preparse_config_path_with_equals():
    from src.pipeline.optimize import _preparse_config_path
    assert _preparse_config_path(["prog", "--config=/tmp/x.cfg"]) == "/tmp/x.cfg"


def test_preparse_config_path_absent():
    from src.pipeline.optimize import _preparse_config_path
    assert _preparse_config_path(["prog"]) is None


def test_preparse_config_path_dangling_flag_returns_none():
    from src.pipeline.optimize import _preparse_config_path
    assert _preparse_config_path(["prog", "--config"]) is None


# ── gpu_index preparse from cfg ──────────────────────────────────────────────


def test_preparse_gpu_index_reads_from_cfg(tmp_path):
    from src.pipeline.optimize import _preparse_gpu_index
    cfg_file = tmp_path / "acts.cfg"
    cfg_file.write_text("hardware: { gpu_index = 3; };\n")
    assert _preparse_gpu_index(["prog", "--config", str(cfg_file)]) == "3"


def test_preparse_gpu_index_no_config_flag_defaults_zero():
    from src.pipeline.optimize import _preparse_gpu_index
    assert _preparse_gpu_index(["prog"]) == "0"


def test_preparse_gpu_index_cfg_missing_section_defaults_zero(tmp_path):
    from src.pipeline.optimize import _preparse_gpu_index
    cfg_file = tmp_path / "acts.cfg"
    cfg_file.write_text("search: { beam_width = 5; };\n")
    assert _preparse_gpu_index(["prog", "--config", str(cfg_file)]) == "0"


def test_preparse_gpu_index_cfg_missing_key_defaults_zero(tmp_path):
    from src.pipeline.optimize import _preparse_gpu_index
    cfg_file = tmp_path / "acts.cfg"
    cfg_file.write_text(
        'hardware: { arch_config_path = "configs/arch/RTX6000Ada.yaml"; };\n'
    )
    assert _preparse_gpu_index(["prog", "--config", str(cfg_file)]) == "0"


def test_preparse_gpu_index_cfg_path_does_not_exist_defaults_zero(tmp_path):
    """A nonexistent cfg path must not crash the preparse — argparse in
    main() handles the bad-path error with a clean message."""
    from src.pipeline.optimize import _preparse_gpu_index
    assert _preparse_gpu_index(["prog", "--config", str(tmp_path / "missing.cfg")]) == "0"


# ── validate_gpu_visible — unchanged surface (kept for runtime check) ────────


def test_validate_no_cuda_exits(capsys):
    from src.pipeline.optimize import _validate_gpu_visible
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    with patch.dict("sys.modules", {"torch": fake_torch}):
        with pytest.raises(SystemExit) as exc:
            _validate_gpu_visible(0, reset_only=False)
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "CUDA-capable PyTorch" in err


def test_validate_zero_visible_exits(capsys):
    from src.pipeline.optimize import _validate_gpu_visible
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.device_count.return_value = 0
    with patch.dict("sys.modules", {"torch": fake_torch}):
        with pytest.raises(SystemExit) as exc:
            _validate_gpu_visible(99, reset_only=False)
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "GPU 99 not found" in err
    assert "out of range" in err


def test_validate_more_than_one_visible_exits(capsys):
    from src.pipeline.optimize import _validate_gpu_visible
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.device_count.return_value = 2
    with patch.dict("sys.modules", {"torch": fake_torch}):
        with pytest.raises(SystemExit) as exc:
            _validate_gpu_visible(0, reset_only=False)
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "Env-handling bug" in err


def test_validate_happy_path_returns_none():
    from src.pipeline.optimize import _validate_gpu_visible
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.device_count.return_value = 1
    with patch.dict("sys.modules", {"torch": fake_torch}):
        assert _validate_gpu_visible(0, reset_only=False) is None


def test_validate_reset_only_uses_nvidia_smi():
    from src.pipeline.optimize import _validate_gpu_visible
    fake_proc = MagicMock()
    fake_proc.returncode = 0
    with patch("subprocess.run", return_value=fake_proc) as mock_run:
        assert _validate_gpu_visible(2, reset_only=True) is None
    args = mock_run.call_args[0][0]
    assert args == ["nvidia-smi", "--list-gpus", "-i", "2"]


def test_validate_reset_only_nvidia_smi_failure_exits(capsys):
    from src.pipeline.optimize import _validate_gpu_visible
    fake_proc = MagicMock()
    fake_proc.returncode = 6
    with patch("subprocess.run", return_value=fake_proc):
        with pytest.raises(SystemExit) as exc:
            _validate_gpu_visible(99, reset_only=True)
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "GPU 99 not found by nvidia-smi" in err


# ── argparse rejects retired flags ──────────────────────────────────────────


def test_main_rejects_gpu_index_flag(capsys):
    """`--gpu-index` retired 2026-05-11; lives in cfg `hardware.gpu_index`."""
    from src.pipeline import optimize
    with pytest.raises(SystemExit):
        optimize.main(["--gpu-index", "0"])


def test_main_rejects_reset_clocks_flag(capsys):
    """`--reset-clocks` retired 2026-05-11; lives in cfg `runtime.reset_clocks`."""
    from src.pipeline import optimize
    with pytest.raises(SystemExit):
        optimize.main(["--reset-clocks"])


def test_main_rejects_problem_positional(capsys):
    """`problem_path` positional retired 2026-05-11; lives in cfg `runtime.problem_path`."""
    from src.pipeline import optimize
    with pytest.raises(SystemExit):
        optimize.main(["placeholder"])


# ── end-to-end (GPU only) ────────────────────────────────────────────────────


@pytest.mark.gpu
def test_gpu_index_end_to_end_subprocess(tmp_path):
    """Subprocess: --config <tmp cfg> runs to completion on the dev box,
    even when the shell's CUDA_VISIBLE_DEVICES is bogus. Verifies the
    preparse + env override land before SOL imports."""
    cfg_file = tmp_path / "acts.cfg"
    cfg_file.write_text(
        "hardware: { gpu_index = 0; };\n"
        'runtime: { problem_path = "placeholder"; };\n'
    )
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "999"}
    proc = subprocess.run(
        [sys.executable, "-m", "src.pipeline.optimize",
         "--config", str(cfg_file),
         "--run-dir", str(tmp_path / "runs"),
         "--trace-dir="],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, (
        f"subprocess failed: rc={proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert any((tmp_path / "runs").iterdir()), "no run_<UTC>/ directory created"
