"""Long-lived, bidirectional processes.

``CommandExecutor.run`` is one request, one result. A ``Process`` is the other
shape: started once, written to over time, and read from by the caller.

Output is *pulled*: the caller asks for the next item. There are no callbacks
on foreign threads, so a caller (or a simulation) decides when output is
consumed and in what interleaving.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@dataclass(frozen=True)
class SpawnRequest:
    """One long-lived process to start.

    ``cwd`` of ``None`` means the executor's own working directory.
    """

    argv: Sequence[str]
    cwd: str | None
    env: Mapping[str, str]


@dataclass(frozen=True)
class StdoutLine:
    """One line of stdout, with its trailing newline (as ``TextIO.readline``).

    The last line of a stream that ends without a newline has none.
    """

    text: str


@dataclass(frozen=True)
class StderrLine:
    """One line of stderr, with the same newline convention as ``StdoutLine``."""

    text: str


@dataclass(frozen=True)
class ProcessExited:
    """The process ended. Always the last item a process yields.

    Every line the process wrote before exiting is delivered first.
    ``returncode`` is negative when a signal ended it.
    """

    returncode: int


ProcessOutput = StdoutLine | StderrLine | ProcessExited


class Process(Protocol):
    """A started process, owned by whoever spawned it."""

    def write(self, data: str) -> None:
        """Write *data* to stdin.

        Raises ``ProcessClosedError`` when stdin is closed or the process is gone.
        """
        ...

    def close_stdin(self) -> None:
        """Close stdin (end of input). Idempotent."""
        ...

    def next_output(self, timeout: float | None) -> ProcessOutput | None:
        """Return the next output item, or ``None`` if *timeout* seconds pass with none.

        ``timeout=None`` blocks until there is an item. Items come in one
        deterministic order. After ``ProcessExited`` has been returned, every
        further call returns ``ProcessExited`` again.
        """
        ...

    def terminate(self) -> None:
        """Ask the process (and its group) to stop. Idempotent."""
        ...

    def kill(self) -> None:
        """Stop the process (and its group) now. Idempotent."""
        ...

    def wait(self, timeout: float | None) -> int | None:
        """Return the exit code, or ``None`` if it has not exited within *timeout*."""
        ...
