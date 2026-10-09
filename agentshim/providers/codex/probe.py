"""Codex readiness: ``codex login status``.

The command prints ``Logged in using ChatGPT`` (or ``... an API key``) and
exits 0, or prints ``Not logged in`` and exits 1. It reads ``$CODEX_HOME/auth.json``
only; it makes no model request. Startup noise (``WARNING: ...``) shares the
output and is ignored.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentshim.core.status import AuthCheck, AuthProbe, AuthState, ProbeOutput, ProbeSpec

if TYPE_CHECKING:
    from collections.abc import Mapping

LOGIN_HINT = "run `codex login`"

#: Keys ``codex exec`` accepts in place of a stored login. ``login status``
#: does not look at them, so with one set a "Not logged in" proves nothing.
_API_KEY_ENV = ("CODEX_API_KEY", "OPENAI_API_KEY")


def read_auth(output: ProbeOutput, env: Mapping[str, str]) -> AuthCheck:
    """Read ``codex login status``'s text."""
    lines = [line.strip() for line in output.text.splitlines() if line.strip()]
    for line in lines:
        if line.startswith("Logged in"):
            return AuthCheck(AuthState.KNOWN_OK, line)
    if any(line.startswith("Not logged in") for line in lines):
        if any(env.get(key) for key in _API_KEY_ENV):
            return AuthCheck(
                AuthState.UNKNOWN,
                "no stored login, but an API key is set in the environment, "
                "which `codex login status` does not report",
            )
        return AuthCheck(AuthState.FAILED, f"codex is not logged in; {LOGIN_HINT}")
    return AuthCheck(
        AuthState.UNKNOWN,
        f"unrecognised `codex login status` output (exit {output.returncode})",
    )


PROBE = ProbeSpec(auth=AuthProbe(argv=("login", "status"), read=read_auth))
