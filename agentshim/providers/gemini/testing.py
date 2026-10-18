"""Gemini-specific fixtures used by the generic consumer test helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentshim.providers._testing import config_mcp_servers

from .provider import MCP_CONFIG_FILENAME, MCP_SERVER_KEY

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from agentshim.core.status import AuthState
    from agentshim.execution.executor import CommandRequest


def probe_fixture(args: Sequence[str], *, version: str, auth: AuthState) -> tuple[str, str, int]:
    """Replay Gemini's version command."""
    del auth
    if list(args) == ["--version"]:
        return f"{version}\n", "", 0
    return "", "error: unrecognized command\n", 2


def installed_mcp_servers(request: CommandRequest, workspace: Path) -> dict[str, dict[str, Any]]:
    """Read Gemini's temporary MCP config."""
    del request
    return config_mcp_servers(workspace, MCP_CONFIG_FILENAME, MCP_SERVER_KEY, _canonical)


def _canonical(raw: Mapping[str, Any]) -> dict[str, Any]:
    if "httpUrl" in raw:
        return {"url": raw["httpUrl"], "transport": "http"}
    if "url" in raw:
        return {"url": raw["url"], "transport": "sse"}
    return {
        "command": raw.get("command", ""),
        "args": list(raw.get("args", ())),
        "env": dict(raw.get("env", {})),
    }
