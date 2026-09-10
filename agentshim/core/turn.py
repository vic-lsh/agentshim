"""The request and result types of a single turn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from .mcp import McpServer
    from .usage import ProviderUsage


@dataclass(frozen=True)
class OutputSchema:
    """A JSON Schema the provider should force its final answer to satisfy.

    Two directories are carried because a container-executed CLI resolves
    paths inside the container: the schema file is written to ``host_dir``
    but named to the CLI as ``cli_dir``. Providers that take the schema
    inline (Claude Code) use only ``schema``.
    """

    schema: Mapping[str, Any]
    host_dir: Path
    cli_dir: str | None = None


@dataclass(frozen=True)
class TurnRequest:
    """One prompt plus the per-turn overrides that shape how it is run."""

    prompt: str
    cwd: str | None = None
    timeout: float | None = None
    output_schema: OutputSchema | None = None
    reasoning_effort: str | None = None
    extra_args: Sequence[str] = ()
    env: Mapping[str, str] | None = None
    mcp_servers: Sequence[McpServer] = ()
    mcp_workspace: Path | None = None
    """Host directory that receives config-file MCP installs.

    Defaults to ``cwd``. A container-executed turn has no host ``cwd`` (the
    CLI runs at the container path), so it names the bind-mounted host
    workspace here instead; the same split ``OutputSchema`` makes with
    ``host_dir`` and ``cli_dir``.
    """


@dataclass(frozen=True)
class TurnResult:
    """Everything one turn produced."""

    text: str
    structured_output: Any | None
    session_id: str | None
    resumed: bool
    usage: ProviderUsage
    cost_usd: float | None
    duration_ms: int
    exit_code: int


def coerce_request(request: TurnRequest | str) -> TurnRequest:
    """Accept a bare prompt string wherever a ``TurnRequest`` is expected."""
    if isinstance(request, str):
        return TurnRequest(prompt=request)
    return request
