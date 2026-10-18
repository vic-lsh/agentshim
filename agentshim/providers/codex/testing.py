"""Codex-specific fixtures used by the generic consumer test helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentshim.core.status import AuthState

from .provider import parse_mcp_servers

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from agentshim.execution.executor import CommandRequest


def probe_fixture(args: Sequence[str], *, version: str, auth: AuthState) -> tuple[str, str, int]:
    """Replay Codex's version and login-status commands."""
    if list(args) == ["--version"]:
        return f"codex-cli {version}\n", "", 0
    if list(args) == ["login", "status"]:
        noise = (
            "WARNING: failed to clean up stale arg0 temp dirs: Directory not empty (os error 39)\n"
        )
        if auth is AuthState.KNOWN_OK:
            return "Logged in using ChatGPT\n", noise, 0
        if auth is AuthState.FAILED:
            return "Not logged in\n", noise, 1
        return "something new\n", noise, 0
    return "", "error: unrecognized command\n", 2


def installed_mcp_servers(request: CommandRequest, workspace: Path) -> dict[str, dict[str, Any]]:
    """Decode Codex MCP flags from a scripted command."""
    del workspace
    return parse_mcp_servers(request.argv)
