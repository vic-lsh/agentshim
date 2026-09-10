"""The process-transport contract, independent of any provider."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence


class CommandHandle(Protocol):
    """Executor-neutral handle for a started command.

    Held so a caller can stop the command from outside the timeout path.
    """

    def terminate(self) -> None:
        """Ask the command to stop, leaving it a chance to exit on its own."""
        ...

    def kill(self) -> None:
        """Stop the command now, with no chance for it to clean up."""
        ...


@dataclass(frozen=True)
class CommandRequest:
    """One command to run.

    ``timeout`` is seconds of wall clock, or ``None`` for no limit.
    """

    argv: Sequence[str]
    stdin: str | None
    cwd: str | None
    env: Mapping[str, str]
    timeout: float | None


@dataclass(frozen=True)
class CommandResult:
    """A finished command. ``stdout``/``stderr`` repeat what the sink saw."""

    returncode: int
    stdout: str
    stderr: str


class CommandStreamSink(Protocol):
    """Receives lifecycle and line-oriented output from a running command.

    Lines keep the trailing newline the stream provided, matching
    ``TextIO.readline``. Every callback runs on the thread that called
    ``CommandExecutor.run``.
    """

    def started(self, handle: CommandHandle) -> None:
        """Receive the handle of the started command, before any output."""
        ...

    def stdout(self, line: str) -> None:
        """Receive one line of stdout."""
        ...

    def stderr(self, line: str) -> None:
        """Receive one line of stderr."""
        ...


class CommandExecutor(Protocol):
    """Controls binary lookup, validation, and streaming execution.

    Implement this to run provider CLIs somewhere other than the local host,
    such as inside a container or over a remote shell. agentshim keeps
    argv construction, parsing, session state, and event handling.
    """

    def find_binary(self, name: str, env: Mapping[str, str]) -> str:
        """Return the value to use as ``argv[0]``, or raise ``CliNotFoundError``."""
        ...

    def check_binary(self, path: str, env: Mapping[str, str], *, timeout: float) -> None:
        """Validate the executable, or raise ``CliCheckError``."""
        ...

    def run(self, request: CommandRequest, sink: CommandStreamSink) -> CommandResult:
        """Run *request*, streaming into *sink*, or raise ``CliTimeoutError``."""
        ...


class NullSink:
    """Sink that drops everything, for runs whose output is not parsed."""

    def started(self, handle: CommandHandle) -> None:
        """Ignore the handle."""

    def stdout(self, line: str) -> None:
        """Ignore the line."""

    def stderr(self, line: str) -> None:
        """Ignore the line."""


class CallbackCommandStreamSink:
    """Sink backed by plain callables, for callers without a sink class."""

    def __init__(
        self,
        *,
        on_stdout: Callable[[str], None],
        on_stderr: Callable[[str], None],
        on_started: Callable[[CommandHandle], None] | None = None,
    ) -> None:
        """Wire the sink to plain callables; only the output ones are required."""
        self._on_stdout = on_stdout
        self._on_stderr = on_stderr
        self._on_started = on_started

    def started(self, handle: CommandHandle) -> None:
        """Forward the handle, when the caller asked for one."""
        if self._on_started is not None:
            self._on_started(handle)

    def stdout(self, line: str) -> None:
        """Forward the line to ``on_stdout``."""
        self._on_stdout(line)

    def stderr(self, line: str) -> None:
        """Forward the line to ``on_stderr``."""
        self._on_stderr(line)
