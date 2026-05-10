"""Tier 1 tests for --gpu-index CLI flag plumbing in src/pipeline/optimize.py."""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.optimize import _preparse_gpu_index


def test_preparse_default_when_flag_absent():
    assert _preparse_gpu_index(["python", "-m", "src.pipeline.optimize"]) == "0"


def test_preparse_space_form():
    assert _preparse_gpu_index(["prog", "--gpu-index", "3"]) == "3"


def test_preparse_equals_form():
    assert _preparse_gpu_index(["prog", "--gpu-index=2"]) == "2"


def test_preparse_ignores_unrelated_flags():
    argv = ["prog", "--run-dir", "/tmp/x", "--gpu-index", "1", "--trace-dir="]
    assert _preparse_gpu_index(argv) == "1"


def test_preparse_dangling_flag_falls_back_to_default():
    assert _preparse_gpu_index(["prog", "--gpu-index"]) == "0"


from src.pipeline.optimize import _nonneg_int


def test_nonneg_int_accepts_zero():
    assert _nonneg_int("0") == 0


def test_nonneg_int_accepts_positive():
    assert _nonneg_int("3") == 3


def test_nonneg_int_rejects_non_integer():
    with pytest.raises(argparse.ArgumentTypeError, match="non-negative integer"):
        _nonneg_int("foo")


def test_nonneg_int_rejects_negative():
    with pytest.raises(argparse.ArgumentTypeError, match="non-negative integer"):
        _nonneg_int("-1")


def test_nonneg_int_rejects_float_string():
    with pytest.raises(argparse.ArgumentTypeError):
        _nonneg_int("1.5")


from src.pipeline.optimize import _validate_gpu_visible


def test_validate_no_cuda_exits(capsys):
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    with patch.dict("sys.modules", {"torch": fake_torch}):
        with pytest.raises(SystemExit) as exc:
            _validate_gpu_visible(0, reset_only=False)
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "CUDA-capable PyTorch" in err


def test_validate_zero_visible_exits(capsys):
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
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.device_count.return_value = 1
    with patch.dict("sys.modules", {"torch": fake_torch}):
        assert _validate_gpu_visible(0, reset_only=False) is None


def test_validate_reset_only_uses_nvidia_smi():
    fake_proc = MagicMock()
    fake_proc.returncode = 0
    with patch("subprocess.run", return_value=fake_proc) as mock_run:
        assert _validate_gpu_visible(2, reset_only=True) is None
    args = mock_run.call_args[0][0]
    assert args == ["nvidia-smi", "--list-gpus", "-i", "2"]


def test_validate_reset_only_nvidia_smi_failure_exits(capsys):
    fake_proc = MagicMock()
    fake_proc.returncode = 6
    with patch("subprocess.run", return_value=fake_proc):
        with pytest.raises(SystemExit) as exc:
            _validate_gpu_visible(99, reset_only=True)
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "GPU 99 not found by nvidia-smi" in err


def test_main_tripwire_catches_preparse_argparse_desync(monkeypatch):
    """If the module-top _GPU_INDEX disagrees with args.gpu_index, the
    tripwire assertion fires."""
    from src.pipeline import optimize

    monkeypatch.setattr(optimize, "_GPU_INDEX", "5")
    monkeypatch.setattr(optimize, "_validate_gpu_visible", lambda *a, **kw: None)

    with pytest.raises(AssertionError, match="preparse/argparse desync"):
        optimize.main(argv=["--gpu-index", "0", "placeholder"])


import os
import subprocess
import sys


@pytest.mark.gpu
def test_gpu_index_end_to_end_subprocess(tmp_path):
    """Subprocess invocation: --gpu-index 0 placeholder runs to completion
    on the dev box, even when the shell's CUDA_VISIBLE_DEVICES is bogus.
    Verifies the preparse + env override land before SOL imports."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "999"}
    proc = subprocess.run(
        [sys.executable, "-m", "src.pipeline.optimize",
         "--gpu-index", "0",
         "--run-dir", str(tmp_path),
         "--trace-dir=",
         "placeholder"],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, (
        f"subprocess failed: rc={proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert any(tmp_path.iterdir()), "no run_<UTC>/ directory created"
