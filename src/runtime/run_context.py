"""Per-run lifecycle: run_<UTC>/ directory, stdlib-logging config, events
FH binding, SDK trace processor wire-up. See
``doc/specs/2026-04-22-logger-system-design.md`` §5.1.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import IO

from src.runtime import events
from src.runtime.timefmt import filename_ts

logger = logging.getLogger(__name__)

_LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"
_SILENCED_LOGGERS = ("httpx", "openai", "agents")


@dataclass
class RunContext:
    run_dir: Path | None
    events_path: Path | None
    log_path: Path | None
    traces_dir: Path | None
    started_at: datetime

    _events_fh: IO[str] | None = field(default=None, repr=False)
    _file_handler: logging.Handler | None = field(default=None, repr=False)
    _stream_handler: logging.Handler | None = field(default=None, repr=False)
    _trace_processor: object | None = field(default=None, repr=False)
    _closed: bool = field(default=False, repr=False)

    @property
    def trace_processor(self) -> object | None:
        """Public read-only view of the SDK trace processor (``None`` when
        capture is disabled or the SDK is absent). Exposed so pipeline
        code can surface ``trace_processor.path`` in reports without
        reaching into dataclass internals.
        """
        return self._trace_processor

    @classmethod
    def create(
        cls,
        root: Path | None = None,
        *,
        trace_dir: Path | str | None = None,
        capture_traces: bool = True,
    ) -> "RunContext":
        """Create a run directory and wire all per-invocation logging.

        ``capture_traces=False`` disables SDK trace capture entirely
        (the CLI's ``--trace-dir=`` kill switch). When enabled, traces
        land under ``<run_dir>/traces/`` by default; passing
        ``trace_dir=<path>`` overrides that with an explicit location.
        """
        root = Path(root) if root is not None else Path("./runs")
        run_dir = root / f"run_{filename_ts()}"
        traces_dir = run_dir / "traces"
        events_path = run_dir / "events.jsonl"
        log_path = run_dir / "run.log"

        # Guard the full file-backed setup, not just mkdir. Disk quota /
        # FD exhaustion / mid-setup permissions issues between mkdir and
        # FileHandler construction would otherwise propagate and abort
        # the run — violating the "best-effort diagnostics, never kill a
        # run" contract (Codex review 2026-04-23 Finding 3).
        events_fh: IO[str] | None = None
        file_handler: logging.Handler | None = None
        stream_handler: logging.Handler | None = None
        try:
            traces_dir.mkdir(parents=True, exist_ok=False)
            events_fh = events_path.open("w", buffering=1)
            events.bind(events_fh)
            log_path.touch()
            fmt = logging.Formatter(_LOG_FORMAT)
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(fmt)
            stream_handler = logging.StreamHandler(sys.stderr)
            stream_handler.setLevel(logging.INFO)
            stream_handler.setFormatter(fmt)
            root_logger = logging.getLogger()
            # Lower root level only if currently higher; don't stomp app overrides.
            if root_logger.level == logging.NOTSET or root_logger.level > logging.DEBUG:
                root_logger.setLevel(logging.DEBUG)
            root_logger.addHandler(file_handler)
            root_logger.addHandler(stream_handler)
            for name in _SILENCED_LOGGERS:
                logging.getLogger(name).setLevel(logging.WARNING)
        except OSError as exc:
            cls._cleanup_partial_setup(events_fh, file_handler, stream_handler)
            logging.basicConfig(level=logging.WARNING, stream=sys.stderr,
                                format=_LOG_FORMAT)
            logger.warning("run-dir setup failed (%s); logger degraded to stderr", exc)
            return cls(
                run_dir=None,
                events_path=None,
                log_path=None,
                traces_dir=None,
                started_at=datetime.now(timezone.utc),
            )

        trace_processor: object | None = None
        if capture_traces:
            # SDK gate is inside the helper so test patches of
            # ``_SDK_AVAILABLE`` continue to work the same way.
            trace_processor = cls._wire_trace_capture(
                trace_dir if trace_dir is not None else traces_dir
            )

        return cls(
            run_dir=run_dir,
            events_path=events_path,
            log_path=log_path,
            traces_dir=traces_dir,
            started_at=datetime.now(timezone.utc),
            _events_fh=events_fh,
            _file_handler=file_handler,
            _stream_handler=stream_handler,
            _trace_processor=trace_processor,
        )

    @staticmethod
    def _wire_trace_capture(target: Path | str) -> object | None:
        """Enable SDK trace capture at ``target`` when the openai-agents
        SDK is installed. Best-effort — returns ``None`` on SDK absent
        or any setup failure; diagnostic plumbing must not abort a run.
        """
        try:
            from src.agents.llm_backend import _SDK_AVAILABLE
        except Exception:
            return None
        if not _SDK_AVAILABLE:
            return None
        try:
            from src.agents.trace_processor import enable_local_trace_capture
            return enable_local_trace_capture(Path(target))
        except (ImportError, ModuleNotFoundError, RuntimeError) as exc:
            logger.warning("SDK trace capture unavailable (%s); continuing without traces/*.jsonl", exc)
        except Exception as exc:  # noqa: BLE001 — trace wiring must not abort the run
            logger.warning("trace capture setup failed: %s", exc)
        return None

    @staticmethod
    def _cleanup_partial_setup(
        events_fh: IO[str] | None,
        file_handler: logging.Handler | None,
        stream_handler: logging.Handler | None,
    ) -> None:
        """Undo whatever half-landed during a failed ``create()`` setup.

        Called from the OSError branch so the null-context fallback
        doesn't leave behind a bound events FH or orphan log handlers.
        Every teardown step is swallowed — cleanup must not raise out
        of a path that's already handling an exception.
        """
        events.unbind()
        if events_fh is not None:
            try:
                events_fh.close()
            except Exception:
                pass
        root_logger = logging.getLogger()
        for handler in (file_handler, stream_handler):
            if handler is not None:
                try:
                    root_logger.removeHandler(handler)
                    handler.close()
                except Exception:
                    pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        events.unbind()
        if self._events_fh is not None:
            try:
                self._events_fh.close()
            except Exception:
                pass
        if self._trace_processor is not None and hasattr(self._trace_processor, "shutdown"):
            try:
                self._trace_processor.shutdown()
            except Exception:
                pass
        root_logger = logging.getLogger()
        for handler in (self._file_handler, self._stream_handler):
            if handler is not None:
                try:
                    handler.flush()
                    root_logger.removeHandler(handler)
                    handler.close()
                except Exception:
                    pass
