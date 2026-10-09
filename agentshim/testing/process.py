"""A scripted, reactive stand-in for a long-lived process."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from agentshim.core.errors import ProcessClosedError
from agentshim.execution.process import ProcessExited, ProcessOutput, StdoutLine

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

_SIGKILL_CODE = -9
_SIGTERM_CODE = -15


@dataclass(frozen=True)
class GateMarker:
    """A point in a peer's output that a closed gate holds back.

    Not a ``ProcessOutput``: consumers of a process never see it.
    """

    name: str


class ReplayGates:
    """Named gates a simulation closes to pause a conversation at a known point.

    Every gate starts open. A peer emits ``GateMarker(name)``; while that gate
    is closed, the ``FakeProcess`` delivers nothing past the marker.
    """

    def __init__(self) -> None:
        """Start with every gate open."""
        self._closed: set[str] = set()

    def close(self, name: str) -> None:
        """Hold back output past markers named *name*."""
        self._closed.add(name)

    def open(self, name: str) -> None:
        """Release output held at markers named *name*."""
        self._closed.discard(name)

    def is_open(self, name: str) -> bool:
        """Whether *name* lets output through."""
        return name not in self._closed


PeerOutput = ProcessOutput | GateMarker


class FakePeer(Protocol):
    """The far end of a ``FakeProcess``: reacts to what the process is sent.

    Each method returns the output the process produces in response, in
    order. A peer ends the process by returning ``ProcessExited``. A protocol
    fake (JSON-RPC, say) answers by request id from ``on_stdin``.
    """

    def on_start(self) -> Sequence[PeerOutput]:
        """Output produced as soon as the process starts."""
        ...

    def on_stdin(self, data: str) -> Sequence[PeerOutput]:
        """Output produced in response to *data* written to stdin."""
        ...

    def on_stdin_closed(self) -> Sequence[PeerOutput]:
        """Output produced when stdin is closed (a filter usually exits here)."""
        ...


class SilentPeer:
    """A peer that never writes and never exits on its own."""

    def on_start(self) -> Sequence[PeerOutput]:
        """Produce nothing."""
        return ()

    def on_stdin(self, data: str) -> Sequence[PeerOutput]:
        """Produce nothing."""
        del data
        return ()

    def on_stdin_closed(self) -> Sequence[PeerOutput]:
        """Produce nothing."""
        return ()


class EchoPeer:
    """Like ``cat``: echoes stdin to stdout line by line, exits 0 at end of input."""

    def on_start(self) -> Sequence[PeerOutput]:
        """Produce nothing."""
        return ()

    def on_stdin(self, data: str) -> Sequence[PeerOutput]:
        """Echo each line of *data* as stdout, keeping its newline."""
        return [StdoutLine(line) for line in data.splitlines(keepends=True)]

    def on_stdin_closed(self) -> Sequence[PeerOutput]:
        """Exit cleanly."""
        return [ProcessExited(0)]


class FakeProcess:
    """A ``Process`` driven by a ``FakePeer``, with no real waiting.

    ``next_output`` returns buffered items in order; when none is available (the
    buffer is empty, or a closed gate holds the next one) it returns ``None`` at
    once, standing in for a timeout. Once the peer exits, or the process is
    terminated or killed, ``ProcessExited`` follows and repeats. A kill or
    terminate discards undelivered output: the process died before anyone
    read it. Writes and stop calls are recorded for assertions.
    """

    def __init__(self, peer: FakePeer, *, gates: ReplayGates | None = None) -> None:
        """Start the process and buffer whatever the peer says on start."""
        self._peer = peer
        self._gates = gates if gates is not None else ReplayGates()
        self._buffer: deque[PeerOutput] = deque()
        self._exit_buffered: ProcessExited | None = None
        self._exit_delivered: ProcessExited | None = None
        self.writes: list[str] = []
        self.stdin_closed = False
        self.terminate_calls = 0
        self.kill_calls = 0
        self._emit(peer.on_start())

    def _emit(self, items: Iterable[PeerOutput]) -> None:
        for item in items:
            if self._exit_buffered is not None:
                return  # nothing may follow ProcessExited
            self._buffer.append(item)
            if isinstance(item, ProcessExited):
                self._exit_buffered = item

    def write(self, data: str) -> None:
        """Record *data* and let the peer react, or raise if stdin is unusable."""
        if self.stdin_closed:
            msg = "stdin is closed"
            raise ProcessClosedError(msg)
        if self._exit_buffered is not None:
            msg = "process has exited"
            raise ProcessClosedError(msg)
        self.writes.append(data)
        self._emit(self._peer.on_stdin(data))

    def close_stdin(self) -> None:
        """Tell the peer input ended, once."""
        if self.stdin_closed:
            return
        self.stdin_closed = True
        self._emit(self._peer.on_stdin_closed())

    def next_output(self, timeout: float | None) -> ProcessOutput | None:
        """Return the next deliverable item, or ``None`` (a simulated timeout)."""
        del timeout  # simulated: never actually waits
        if self._exit_delivered is not None:
            return self._exit_delivered
        while self._buffer:
            head = self._buffer[0]
            if isinstance(head, GateMarker):
                if not self._gates.is_open(head.name):
                    return None
                self._buffer.popleft()
                continue
            self._buffer.popleft()
            if isinstance(head, ProcessExited):
                self._exit_delivered = head
            return head
        return None

    def terminate(self) -> None:
        """Record the call and end the process with SIGTERM's code."""
        self.terminate_calls += 1
        self._stop(_SIGTERM_CODE)

    def kill(self) -> None:
        """Record the call and end the process with SIGKILL's code."""
        self.kill_calls += 1
        self._stop(_SIGKILL_CODE)

    def _stop(self, returncode: int) -> None:
        if self._exit_buffered is not None:
            return
        self._buffer.clear()
        self._emit([ProcessExited(returncode)])

    def wait(self, timeout: float | None) -> int | None:
        """Return the exit code once the exit is known, else ``None``."""
        del timeout  # simulated: never actually waits
        exited = self._exit_delivered or self._exit_buffered
        return None if exited is None else exited.returncode
