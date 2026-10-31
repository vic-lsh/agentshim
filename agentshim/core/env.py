"""Capture of the user's interactive shell environment."""

from __future__ import annotations

import os
import subprocess
import threading

_INTERACTIVE_ENV_ATTEMPT_S = 10.0
"""The most time one attempt may take; each of the two attempts has all of it.

A shell that stalls, or an ``env -0`` that fails late, must not take the plain-``env`` retry's
time with it: a slow start (a loaded machine, a heavy rc file) is the case the retry is for.
The worst case of a capture is therefore two attempts, 20 seconds. A caller whose own limit is
shorter than that bounds its call itself.
"""

_lock = threading.Lock()
# One-slot cache keyed by _CACHE_KEY. A dict rather than a rebound module
# global so the reader does not need a `global` statement, and so a test can
# clear it by replacing the mapping.
_CACHE_KEY = "env"
_cache: dict[str, dict[str, str]] = {}


def interactive_env(*, refresh: bool = False) -> dict[str, str]:
    """Return the environment an interactive login shell would provide.

    Provider CLIs are usually installed by a shell rc file that a
    non-interactive process never sources, so PATH and credentials are read
    from ``bash -i``. A successful capture is run once per process and
    cached: it costs hundreds of milliseconds and the answer does not change.
    Each of the two attempts (``env -0``, then ``env`` for a shell without it)
    has its own 10s, so a call takes at most 20 seconds. Any failure falls
    back to ``os.environ`` for that call only: a failed capture is never
    cached, so the next call asks the shell again instead of keeping a
    machine's PATH out of the answer for the life of the process.
    """
    with _lock:
        cached = _cache.get(_CACHE_KEY)
        if cached is not None and not refresh:
            return dict(cached)
        captured = _probe()
        if captured is None:
            return os.environ.copy()
        _cache[_CACHE_KEY] = captured
        return dict(captured)


def _probe() -> dict[str, str] | None:
    """Run ``bash -i -c 'env -0'`` and parse its output into a mapping, or ``None`` on failure.

    ``env -0`` separates entries with NUL, which is the only separator a
    variable's own value cannot contain: a multi-line value (a private key, a
    shell function exported into the environment) split on newlines silently
    truncates that variable and turns the rest of it into junk keys. ``env
    -0`` is a GNU extension, so a shell whose ``env`` does not have it falls
    back to plain ``env`` and the line-based reading.
    """
    raw = _run_env("env -0")
    if raw is None:
        raw = _run_env("env")
    if raw is None:
        return None
    return _parse_env(raw) or None


def _run_env(command: str) -> str | None:
    """Run *command* in an interactive shell, or ``None`` if it did not work in its time."""
    try:
        # start_new_session detaches from the TTY so `bash -i` setting its
        # process group cannot stop us with SIGTTOU/SIGTTIN.
        # The command is one of this module's two literals, never caller input.
        completed = subprocess.run(  # noqa: S603
            ["/bin/bash", "-i", "-c", command],
            capture_output=True,
            text=True,
            check=False,
            start_new_session=True,
            timeout=_INTERACTIVE_ENV_ATTEMPT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def _parse_env(raw: str) -> dict[str, str]:
    """Split a dump of the environment into entries, NUL-separated if it is."""
    entries = raw.split("\0") if "\0" in raw else raw.splitlines()
    env: dict[str, str] = {}
    for entry in entries:
        if "=" in entry:
            key, value = entry.split("=", 1)
            env[key] = value
    return env
