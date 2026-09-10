"""Capture of the user's interactive shell environment."""

from __future__ import annotations

import os
import subprocess
import threading

_INTERACTIVE_ENV_TIMEOUT_S = 10.0

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
    from ``bash -i``. The subprocess is run once per process and cached: it
    costs hundreds of milliseconds and the answer does not change. Any
    failure, including the 10s timeout, falls back to ``os.environ``.
    """
    with _lock:
        cached = _cache.get(_CACHE_KEY)
        if cached is not None and not refresh:
            return dict(cached)
        probed = _probe()
        _cache[_CACHE_KEY] = probed
        return dict(probed)


def _probe() -> dict[str, str]:
    """Run ``bash -i -c 'env -0'`` once and parse its output into a mapping.

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
        return os.environ.copy()
    return _parse_env(raw) or os.environ.copy()


def _run_env(command: str) -> str | None:
    """Run *command* in an interactive shell, or ``None`` if it did not work."""
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
            timeout=_INTERACTIVE_ENV_TIMEOUT_S,
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
