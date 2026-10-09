"""MCP servers as the ``config`` object of ``thread/start`` and ``thread/resume``.

Servers go in per thread rather than as ``-c`` flags on the process. Both
reach the same Codex setting, but the per-thread form is JSON (no TOML quoting),
carries HTTP headers, is re-sent with every resume so a resumed conversation
cannot silently lose its servers, and its start-up reports name the thread.
Codex does not fail a thread because a server could not start; it only says so
in ``mcpServer/startupStatus/updated``, which the transport watches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentshim.core.errors import ProviderCapabilityError
from agentshim.core.mcp import HttpMcpServer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from agentshim.core.mcp import McpServer

    from ._wire import JsonObject, JsonValue


def server_key(server: McpServer) -> str:
    """The ``mcp_servers`` table key for *server*: TOML keys are snake_case by convention."""
    return server.name.replace("-", "_")


def server_keys(servers: Sequence[McpServer]) -> frozenset[str]:
    """The keys Codex reports the servers under."""
    return frozenset(server_key(server) for server in servers)


def thread_config(servers: Sequence[McpServer]) -> JsonObject | None:
    """The ``config`` for a thread given *servers*, or ``None`` for none.

    Every server is ``required``, as in the one-shot path.

    Raises:
        ProviderCapabilityError: Two servers share a key, or one speaks SSE,
            which Codex cannot reach.
    """
    if not servers:
        return None
    table: dict[str, JsonValue] = {}
    for server in servers:
        key = server_key(server)
        if key in table:
            msg = f"MCP servers {server.name!r} and another both map to the codex key {key!r}"
            raise ProviderCapabilityError(msg)
        table[key] = _entry(server)
    return {"mcp_servers": table}


def _entry(server: McpServer) -> JsonObject:
    entry: JsonObject = {"required": True}
    if isinstance(server, HttpMcpServer):
        if server.transport != "http":
            msg = f"codex cannot reach MCP server {server.name!r} over {server.transport}"
            raise ProviderCapabilityError(msg)
        entry["url"] = server.url
        if server.headers:
            entry["http_headers"] = dict(server.headers)
    else:
        entry["command"] = server.command
        entry["args"] = list(server.args)
        if server.env:
            entry["env"] = dict(server.env)
    if server.startup_timeout_s is not None:
        entry["startup_timeout_sec"] = server.startup_timeout_s
    if server.tool_timeout_s is not None:
        entry["tool_timeout_sec"] = server.tool_timeout_s
    return entry
