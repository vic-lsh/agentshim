"""The contract a provider package implements."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from .usage import ProviderUsage

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from .errors import AgentShimError, CliExitError
    from .events import AgentEvent
    from .mcp import McpServer
    from .profile import ProviderProfile


@dataclass(frozen=True)
class ArgvContext:
    """Everything ``build_argv`` may use.

    The prompt is deliberately absent: it always goes on stdin so an agent's
    own ``pkill -f`` cannot match the CLI by prompt text.
    """

    binary_path: str
    model: str | None
    env: Mapping[str, str]
    resume_session_id: str | None
    reasoning_effort: str | None
    schema_inline: str | None
    schema_path: str | None
    mcp_argv: Sequence[str] = ()
    extra_args: Sequence[str] = ()


@dataclass(frozen=True)
class ParsedTurn:
    """What a stream parser recovered from one run."""

    text: str = ""
    structured_output: Any | None = None
    session_id: str | None = None
    usage: ProviderUsage = field(default_factory=ProviderUsage)
    cost_usd: float | None = None
    error: str | None = None


class StreamParser(Protocol):
    """Turns one provider's stdout/stderr into events and a ``ParsedTurn``."""

    def feed_stdout(self, line: str) -> None: ...

    def feed_stderr(self, line: str) -> None: ...

    def finish(self) -> ParsedTurn: ...


class McpInstallation(Protocol):
    """The undo record for MCP servers installed for one turn."""

    argv: Sequence[str]

    def restore(self) -> None: ...


class Provider(Protocol):
    """Argv construction, stream parsing, and provider-specific options."""

    profile: ProviderProfile

    def build_argv(self, ctx: ArgvContext) -> list[str]: ...

    def new_parser(
        self,
        emit: Callable[[AgentEvent], None],
        *,
        expect_structured: bool,
    ) -> StreamParser: ...

    def install_mcp(self, workspace: Path | None, servers: Sequence[McpServer]) -> McpInstallation: ...

    def classify_exit(self, error: CliExitError, *, resumed: bool) -> AgentShimError: ...
