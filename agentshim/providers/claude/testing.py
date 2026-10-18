"""Claude-specific fixtures used by the generic consumer test helpers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from agentshim.core.status import AuthState
from agentshim.providers._testing import config_mcp_servers

from .provider import MCP_CONFIG_FILENAME, MCP_SERVER_KEY

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from agentshim.execution.executor import CommandRequest


def probe_fixture(args: Sequence[str], *, version: str, auth: AuthState) -> tuple[str, str, int]:
    """Replay Claude's version and auth-status commands."""
    if list(args) == ["--version"]:
        return f"{version} (Claude Code)\n", "", 0
    if list(args) == ["auth", "status"]:
        if auth is AuthState.KNOWN_OK:
            body = {"loggedIn": True, "authMethod": "claude.ai", "apiProvider": "firstParty"}
            return json.dumps(body, indent=2) + "\n", "", 0
        if auth is AuthState.FAILED:
            body = {"loggedIn": False, "authMethod": "none", "apiProvider": "firstParty"}
            return json.dumps(body, indent=2) + "\n", "", 1
        return "something new\n", "", 0
    return "", "error: unrecognized command\n", 2


def installed_mcp_servers(request: CommandRequest, workspace: Path) -> dict[str, dict[str, Any]]:
    """Read Claude's temporary MCP config."""
    del request
    return config_mcp_servers(workspace, MCP_CONFIG_FILENAME, MCP_SERVER_KEY, _canonical)


def _canonical(raw: Mapping[str, Any]) -> dict[str, Any]:
    if "url" in raw:
        return {"url": raw["url"], "transport": raw.get("type", "http")}
    return {
        "command": raw.get("command", ""),
        "args": list(raw.get("args", ())),
        "env": dict(raw.get("env", {})),
    }
