"""Capture of the user's interactive shell environment."""

from __future__ import annotations

import os
import subprocess
import threading

_INTERACTIVE_ENV_TIMEOUT_S = 10.0

_lock = threading.Lock()
_cached: dict[str, str] | None = None


def interactive_env(*, refresh: bool = False) -> dict[str, str]:
    """Return the environment an interactive login shell would provide.

    Provider CLIs are usually installed by a shell rc file that a
    non-interactive process never sources, so PATH and credentials are read
    from ``bash -i``. The subprocess is run once per process and cached: it
    costs hundreds of milliseconds and the answer does not change. Any
    failure, including the 10s timeout, falls back to ``os.environ``.
    """
    global _cached
    with _lock:
        if _cached is not None and not refresh:
            return dict(_cached)
        _cached = _probe()
        return dict(_cached)


def _probe() -> dict[str, str]:
    try:
        # start_new_session detaches from the TTY so `bash -i` setting its
        # process group cannot stop us with SIGTTOU/SIGTTIN.
        completed = subprocess.run(
            ["/bin/bash", "-i", "-c", "env"],
            capture_output=True,
            text=True,
            check=False,
            start_new_session=True,
            timeout=_INTERACTIVE_ENV_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return os.environ.copy()
    if completed.returncode != 0:
        return os.environ.copy()

    env: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            env[key] = value
    return env or os.environ.copy()
