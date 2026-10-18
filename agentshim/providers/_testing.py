"""Small shared helpers for provider-owned consumer test adapters."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path


def config_mcp_servers(
    workspace: Path,
    filename: str,
    server_key: str,
    canonical: Callable[[Mapping[str, Any]], dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Read a provider's temporary MCP config during a scripted run."""
    target = workspace / filename
    if not target.exists():
        return {}
    config = json.loads(target.read_text())
    servers = config.get(server_key, {})
    return {name: canonical(raw) for name, raw in servers.items()}
