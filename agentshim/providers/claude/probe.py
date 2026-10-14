"""Claude Code readiness: ``claude auth status``.

The command prints a JSON object (``loggedIn``, ``authMethod``, ...) and exits
1 when logged out. It reads local credentials and the environment
(``ANTHROPIC_API_KEY``, ``CLAUDE_CODE_OAUTH_TOKEN``) only; it makes no model
request.

``authMethod`` names the credential in use (Claude Code 2.1): ``claude.ai``
for the stored subscription login, ``api_key`` for ``ANTHROPIC_API_KEY``, and
``oauth_token`` for both ``ANTHROPIC_AUTH_TOKEN`` (a static token, often a
gateway's) and ``CLAUDE_CODE_OAUTH_TOKEN`` (a subscription token). Only the
environment the command ran with tells those last two apart.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

from agentshim.core.auth import CredentialKind
from agentshim.core.status import AuthCheck, AuthProbe, AuthState, ProbeOutput, ProbeSpec

if TYPE_CHECKING:
    from collections.abc import Mapping

LOGIN_HINT = "run `claude auth login` or set ANTHROPIC_API_KEY"


def read_auth(output: ProbeOutput, env: Mapping[str, str]) -> AuthCheck:
    """Read ``claude auth status``'s JSON."""
    try:
        data: Any = json.loads(output.stdout)
    except ValueError:
        data = None
    if not isinstance(data, dict):
        return AuthCheck(AuthState.UNKNOWN, _unrecognised(output))
    status = cast("dict[str, Any]", data)
    logged_in = status.get("loggedIn")
    if logged_in is True:
        method = status.get("authMethod")
        how = f" ({method})" if isinstance(method, str) and method != "none" else ""
        return AuthCheck(AuthState.KNOWN_OK, f"logged in{how}", _credential(method, env))
    if logged_in is False:
        return AuthCheck(AuthState.FAILED, f"claude is not logged in; {LOGIN_HINT}")
    return AuthCheck(AuthState.UNKNOWN, _unrecognised(output))


def _credential(method: object, env: Mapping[str, str]) -> CredentialKind | None:
    if method == "claude.ai":
        return CredentialKind.SUBSCRIPTION
    if method == "api_key":
        return CredentialKind.API_KEY
    if method == "oauth_token":
        bearer = bool(env.get("ANTHROPIC_AUTH_TOKEN", "").strip())
        subscription = bool(env.get("CLAUDE_CODE_OAUTH_TOKEN", "").strip())
        if bearer != subscription:
            return CredentialKind.API_KEY if bearer else CredentialKind.SUBSCRIPTION
    return None


def _unrecognised(output: ProbeOutput) -> str:
    return f"unrecognised `claude auth status` output (exit {output.returncode})"


PROBE = ProbeSpec(auth=AuthProbe(argv=("auth", "status"), read=read_auth))
