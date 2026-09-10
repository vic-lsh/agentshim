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

    def feed_stdout(self, line: str) -> None:
        """Consume one stdout line, emitting whatever events it carried.

        The line keeps the trailing newline the stream produced, and the call
        arrives on the thread that ran the turn, so a parser may hold plain
        unsynchronized state across lines.
        """
        ...

    def feed_stderr(self, line: str) -> None:
        """Consume one stderr line.

        Providers mix progress chatter and real failures on stderr, so which
        of the two a line is has to be decided per provider, here, rather than
        by the session.
        """
        ...

    def finish(self) -> ParsedTurn:
        """Report what the run produced, after the last line has been fed.

        Called once per run and before the exit code is inspected, so a parser
        must return its best partial reading of a failed run rather than raise.
        """
        ...


class McpInstallation(Protocol):
    """The undo record for MCP servers installed for one turn."""

    argv: Sequence[str]

    def restore(self) -> None:
        """Put the workspace back the way the turn found it.

        Must be idempotent and safe to call from a ``finally`` block, including
        when the install itself failed part way through.
        """
        ...


class Provider(Protocol):
    """Argv construction, stream parsing, and provider-specific options."""

    profile: ProviderProfile

    def build_argv(self, ctx: ArgvContext) -> list[str]:
        """Build the command line for one turn, ``argv[0]`` included.

        The prompt is never part of argv; the session always writes it to
        stdin, so nothing that inspects the process table can read it.
        """
        ...

    def new_parser(
        self,
        emit: Callable[[AgentEvent], None],
        *,
        expect_structured: bool,
    ) -> StreamParser:
        """Create a parser for one run, wired to publish events through *emit*.

        A parser is single-use. ``expect_structured`` says the caller asked for
        an output schema, which is what lets a parser tell a missing structured
        result apart from a turn that never wanted one.
        """
        ...

    def install_mcp(self, workspace: Path | None, servers: Sequence[McpServer]) -> McpInstallation:
        """Make *servers* reachable for one turn and return the undo record.

        ``workspace`` is ``None`` when the turn runs without a cwd, which a
        provider that installs servers by writing a config file has no way to
        satisfy and should reject.
        """
        ...

    def classify_exit(self, error: CliExitError, *, resumed: bool) -> AgentShimError:
        """Narrow a nonzero exit to the most specific error this CLI can justify.

        ``resumed`` says the turn passed a session id. CLIs rarely give a
        distinct exit code for a conversation that no longer exists, so this
        flag is usually the only evidence that resuming, not the prompt, failed.
        """
        ...
