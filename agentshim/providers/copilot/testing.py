"""Copilot-specific fixtures used by the generic consumer test helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .provider import parse_mcp_servers

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from agentshim.core.status import AuthState
    from agentshim.execution.executor import CommandRequest


def probe_fixture(args: Sequence[str], *, version: str, auth: AuthState) -> tuple[str, str, int]:
    """Replay Copilot's version command."""
    del auth
    if list(args) == ["--version"]:
        return f"GitHub Copilot CLI {version}.\nRun 'copilot update' to check for updates.\n", "", 0
    return "", "error: unrecognized command\n", 2


def installed_mcp_servers(request: CommandRequest, workspace: Path) -> dict[str, dict[str, Any]]:
    """Decode Copilot MCP flags from a scripted command."""
    del workspace
    return parse_mcp_servers(request.argv)
