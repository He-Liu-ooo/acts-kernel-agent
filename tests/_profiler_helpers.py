"""Shared bash bodies for the fake ``ncu`` shell scripts used by the
profiler tests.

The mock honors NCU 2025.x's two-subprocess contract (see JOURNAL
2026-05-08 + tests/test_profiler_cache.py): with ``-o <basename>``
(capture mode) the binary ``.ncu-rep`` is written and stdout reduces to
``==PROF==`` banners; with ``--import <rep>`` (extract mode) the CSV is
emitted to stdout. Without either flag the script falls back to
emitting CSV directly (legacy single-call mode for argv-only tests that
pre-date the rep-capture work).

Leading-underscore filename prevents pytest auto-collection.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def two_phase_fake_ncu_body(
    csv_payload: str,
    *,
    counter_path: Path | None = None,
) -> str:
    """Return a bash script body that mocks NCU 2025.x's two-subprocess
    contract.

    Parses argv for ``-o <basename>`` and ``--import <rep>``:

    * ``--import`` set → extract mode → emit ``csv_payload`` to stdout.
    * ``-o`` set (no ``--import``) → capture mode → write a non-empty
      marker ``.ncu-rep`` file at ``$out_basename.ncu-rep`` and emit
      banners-only stdout.
    * neither set → legacy mode → emit ``csv_payload`` to stdout.

    If ``counter_path`` is given, capture mode and legacy mode each
    increment the counter file; extract mode does not (so cache-hit
    tests can assert on the number of full profile runs without
    conflating the post-profile ``--import`` extract subprocess).
    """
    if counter_path is not None:
        counter_inc = (
            f"  n=$(cat {counter_path})\n"
            f"  echo $((n+1)) > {counter_path}\n"
        )
    else:
        counter_inc = ""

    return (
        "#!/usr/bin/env bash\n"
        "out_basename=\"\"\n"
        "import_rep=\"\"\n"
        "while [[ $# -gt 0 ]]; do\n"
        "  if [[ \"$1\" == \"-o\" ]]; then\n"
        "    out_basename=\"$2\"\n"
        "    shift 2\n"
        "    continue\n"
        "  fi\n"
        "  if [[ \"$1\" == \"--import\" ]]; then\n"
        "    import_rep=\"$2\"\n"
        "    shift 2\n"
        "    continue\n"
        "  fi\n"
        "  shift\n"
        "done\n"
        "if [[ -n \"$import_rep\" ]]; then\n"
        "  cat <<\"NCUEOF\"\n"
        + csv_payload
        + "NCUEOF\n"
        "elif [[ -n \"$out_basename\" ]]; then\n"
        + counter_inc
        + "  printf 'ncu-rep-marker' > \"$out_basename.ncu-rep\"\n"
        "  printf '==PROF== Connected to process 1\\n'\n"
        "  printf '==PROF== Disconnected from process 1\\n'\n"
        "else\n"
        + counter_inc
        + "  cat <<\"NCUEOF\"\n"
        + csv_payload
        + "NCUEOF\n"
        "fi\n"
    )


def force_ncu_discovery(
    monkeypatch: pytest.MonkeyPatch, *, fallback_present: bool
) -> None:
    """Force ``_discover_ncu_binary()`` to take the fallback path.

    Resets the module-level discovery cache, monkeypatches ``shutil.which``
    to return None (so the primary PATH lookup misses), and monkeypatches
    ``Path.is_file`` so the fallback path resolves either to True (the file
    is "present") or False (no ncu anywhere — degradation expected).

    Other ``Path.is_file`` callers in the same test see the original
    behavior — only the ``_NCU_FALLBACK_PATH`` literal is intercepted.
    """
    from src.eval import profiler as profiler_mod

    monkeypatch.setattr(
        profiler_mod, "_NCU_BINARY_CACHE", profiler_mod._UNSET, raising=False
    )
    monkeypatch.setattr(profiler_mod.shutil, "which", lambda _name: None)
    real_is_file = Path.is_file

    def fake_is_file(self):
        if str(self) == profiler_mod._NCU_FALLBACK_PATH:
            return fallback_present
        return real_is_file(self)

    monkeypatch.setattr(Path, "is_file", fake_is_file)
