"""A fake CLI for readiness probes: ``--version`` and status commands, scripted."""

from __future__ import annotations

import json
from typing import NamedTuple

from agentshim.core.status import AuthState
from agentshim.providers import get_probe_spec

#: What each CLI's ``--version`` prints, as recorded from the installed CLIs.
_VERSION_TEXT = {
    "claude": "{v} (Claude Code)\n",
    "codex": "codex-cli {v}\n",
    "gemini": "{v}\n",
    "copilot": "GitHub Copilot CLI {v}.\nRun 'copilot update' to check for updates.\n",
    "opencode": "{v}\n",
}


class ProbeRun(NamedTuple):
    """What a scripted probe command prints."""

    stdout: str = ""
    stderr: str = ""
    returncode: int = 0


def _claude_auth(state: AuthState) -> ProbeRun:
    if state is AuthState.KNOWN_OK:
        body = {"loggedIn": True, "authMethod": "claude.ai", "apiProvider": "firstParty"}
        return ProbeRun(stdout=json.dumps(body, indent=2) + "\n")
    if state is AuthState.FAILED:
        body = {"loggedIn": False, "authMethod": "none", "apiProvider": "firstParty"}
        return ProbeRun(stdout=json.dumps(body, indent=2) + "\n", returncode=1)
    return ProbeRun(stdout="something new\n")


def _codex_auth(state: AuthState) -> ProbeRun:
    noise = "WARNING: failed to clean up stale arg0 temp dirs: Directory not empty (os error 39)\n"
    if state is AuthState.KNOWN_OK:
        return ProbeRun(stdout="Logged in using ChatGPT\n", stderr=noise)
    if state is AuthState.FAILED:
        return ProbeRun(stdout="Not logged in\n", stderr=noise, returncode=1)
    return ProbeRun(stdout="something new\n", stderr=noise)


_AUTH = {"claude": _claude_auth, "codex": _codex_auth}


def probe_run(provider: str, args: list[str], *, version: str, auth: AuthState) -> ProbeRun:
    """What *provider*'s CLI prints for *args*: its version, its status, or a usage error.

    *auth* picks what the status command prints: logged in, logged out, or
    (``UNKNOWN``) output the provider's reader does not recognise. It has no
    effect for a provider that has no status command.
    """
    spec = get_probe_spec(provider)
    if args == list(spec.version_argv):
        return ProbeRun(stdout=_VERSION_TEXT[provider].format(v=version))
    if spec.auth is not None and args == list(spec.auth.argv):
        return _AUTH[provider](auth)
    return ProbeRun(stderr="error: unrecognized command\n", returncode=2)
