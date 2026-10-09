"""Claude Code readiness: ``claude auth status``.

The command prints a JSON object (``loggedIn``, ``authMethod``, ...) and exits
1 when logged out. It reads local credentials and the environment
(``ANTHROPIC_API_KEY``, ``CLAUDE_CODE_OAUTH_TOKEN``) only; it makes no model
request.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

from agentshim.core.status import AuthCheck, AuthProbe, AuthState, ProbeOutput, ProbeSpec

if TYPE_CHECKING:
    from collections.abc import Mapping

LOGIN_HINT = "run `claude auth login` or set ANTHROPIC_API_KEY"


def read_auth(output: ProbeOutput, env: Mapping[str, str]) -> AuthCheck:
    """Read ``claude auth status``'s JSON."""
    del env
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
        return AuthCheck(AuthState.KNOWN_OK, f"logged in{how}")
    if logged_in is False:
        return AuthCheck(AuthState.FAILED, f"claude is not logged in; {LOGIN_HINT}")
    return AuthCheck(AuthState.UNKNOWN, _unrecognised(output))


def _unrecognised(output: ProbeOutput) -> str:
    return f"unrecognised `claude auth status` output (exit {output.returncode})"


PROBE = ProbeSpec(auth=AuthProbe(argv=("auth", "status"), read=read_auth))
