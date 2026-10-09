"""The client's end of the app-server pipe: framed JSON in, parsed messages out.

A ``Channel`` wraps one ``Process``. It numbers requests, remembers which are
unanswered, writes whole lines under a lock (the conversation's turn thread and
whichever thread interrupts it both write), and reads by pulling
``Process.next_output`` in slices so a deadline can be honoured. It starts no
threads of its own.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from typing import TYPE_CHECKING

from agentshim.core.errors import FailureKind, ProcessClosedError, TurnFailedError
from agentshim.core.events import Lifecycle, RawOutput, Stderr
from agentshim.execution.process import ProcessExited, StderrLine

from ._wire import CodexProtocolError
from .protocol import parse_server_message

if TYPE_CHECKING:
    from collections.abc import Callable

    from agentshim.core.clock import Clock
    from agentshim.core.events import AgentEvent
    from agentshim.execution.process import Process

    from .protocol import (
        ClientNotification,
        ClientRequestParams,
        ErrorResponse,
        Response,
        ServerMessage,
    )

#: Longest a single read blocks, so a deadline is noticed within this many seconds.
POLL_S = 1.0
#: Stderr lines kept to explain a process that died.
STDERR_TAIL_LINES = 20
#: Longest a read blocks when collecting a dead process's last words.
DRAIN_WAIT_S = 0.5

#: A message whose loss would leave a turn waiting for something that will not
#: come, so one that cannot be read fails the turn instead of being skipped.
_ESSENTIAL_METHODS = frozenset(
    {"turn/started", "turn/completed", "item/started", "item/completed", "error"}
)


class Expired(Exception):  # noqa: N818 - control flow inside the transport, not an error to callers
    """A deadline passed before the awaited message arrived."""


class Gone(Exception):  # noqa: N818 - control flow inside the transport, not an error to callers
    """The process exited, or can no longer be written to."""

    def __init__(self, returncode: int | None) -> None:
        """Record the exit code, when the process reported one."""
        self.returncode = returncode
        super().__init__(f"codex app-server exited with code {returncode}")


class Deadline:
    """A time limit on a wait, measured on the injected clock.

    A read that returns nothing after waiting *n* seconds counts those *n*
    seconds even if the clock did not move (a simulated process answers a
    silent read at once), so a limit is reached with a real clock, a fake one,
    or none that advances. Time passing without silence is the clock's alone.
    """

    def __init__(self, clock: Clock, limit: float | None) -> None:
        """Start counting now; ``None`` never expires."""
        self._clock = clock
        self._limit = limit
        self._began = clock.monotonic()
        self._silent = 0.0

    def elapsed(self) -> float:
        """Seconds counted so far."""
        return max(self._clock.monotonic() - self._began, self._silent)

    def expired(self) -> bool:
        """Whether the limit has been reached."""
        return self._limit is not None and self.elapsed() >= self._limit

    def next_wait(self) -> float:
        """How long the next read may block."""
        if self._limit is None:
            return POLL_S
        return min(POLL_S, max(self._limit - self.elapsed(), 0.0))

    def silence(self, seconds: float) -> None:
        """Count *seconds* of a read that returned nothing."""
        self._silent += seconds


class Channel:
    """One app-server process and the framing of its JSON lines."""

    def __init__(self, process: Process, emit: Callable[[AgentEvent], None]) -> None:
        """Wrap *process*; *emit* receives stderr lines and output that is not protocol."""
        self._process = process
        self._emit = emit
        self._lock = threading.Lock()
        self._next_id = 1
        self._pending: dict[int, str] = {}
        self._tail: deque[str] = deque(maxlen=STDERR_TAIL_LINES)
        self.returncode: int | None = None
        self._unwritable = False

    @property
    def process(self) -> Process:
        """The wrapped process."""
        return self._process

    def stderr_tail(self) -> str:
        """The last stderr lines, newline-joined."""
        return "\n".join(self._tail)

    def request(self, params: ClientRequestParams) -> int:
        """Send a request and return its id. Thread-safe.

        Raises:
            Gone: The process cannot take input.
        """
        with self._lock:
            request_id = self._next_id
            self._next_id += 1
            self._pending[request_id] = params.METHOD
            line = {"id": request_id, "method": params.METHOD, "params": params.to_wire()}
            self._write_line(line)
        return request_id

    def notify(self, message: ClientNotification) -> None:
        """Send a notification. Thread-safe."""
        with self._lock:
            self._write_line(message.to_wire())

    def reply(self, message: Response | ErrorResponse) -> None:
        """Send the reply to a server request. Thread-safe."""
        with self._lock:
            self._write_line(message.to_wire())

    def settle(self, request_id: int) -> str | None:
        """Forget request *request_id* and return its method, or ``None`` if not pending."""
        with self._lock:
            return self._pending.pop(request_id, None)

    def _write_line(self, wire: object) -> None:
        try:
            self._process.write(json.dumps(wire, separators=(",", ":")) + "\n")
        except ProcessClosedError as error:
            self._unwritable = True
            raise Gone(self.returncode) from error

    def read(self, deadline: Deadline) -> ServerMessage:
        """Return the next protocol message from the server.

        Stderr lines and output that is not JSON go to *emit* on the way.

        Raises:
            Expired: *deadline* passed first.
            Gone: The process exited.
            TurnFailedError: A message the turn cannot do without could not be read.
        """
        while True:
            if self.returncode is not None:
                raise Gone(self.returncode)
            if deadline.expired():
                raise Expired
            wait = deadline.next_wait()
            output = self._process.next_output(wait)
            if output is None:
                deadline.silence(wait)
            elif isinstance(output, StderrLine):
                self._on_stderr(output.text)
            elif isinstance(output, ProcessExited):
                self.returncode = output.returncode
                raise Gone(output.returncode)
            else:
                message = self._parse(output.text)
                if message is not None:
                    return message

    def poll(self) -> ServerMessage | None:
        """Return a message that is already waiting, or ``None`` when nothing is.

        Never blocks. Used between turns, when nobody is reading, to find out
        whether the process is still there.

        Raises:
            Gone: The process exited.
        """
        while True:
            if self.returncode is not None or self._unwritable:
                raise Gone(self.returncode)
            output = self._process.next_output(0.0)
            if output is None:
                return None
            if isinstance(output, StderrLine):
                self._on_stderr(output.text)
            elif isinstance(output, ProcessExited):
                self.returncode = output.returncode
                raise Gone(output.returncode)
            else:
                message = self._parse(output.text)
                if message is not None:
                    return message

    def drain(self) -> None:
        """Collect what a process that stopped taking input has left to say.

        A write to a dead process fails before anyone has read its last words;
        this reads them (stderr into the tail, then the exit code) so the error
        that follows can quote them. Stdout is dropped: nothing waits for it.
        """
        while self.returncode is None:
            output = self._process.next_output(DRAIN_WAIT_S)
            if output is None:
                return
            if isinstance(output, StderrLine):
                self._on_stderr(output.text)
            elif isinstance(output, ProcessExited):
                self.returncode = output.returncode

    def _on_stderr(self, text: str) -> None:
        stripped = text.rstrip("\n")
        if stripped:
            self._tail.append(stripped)
            self._emit(Stderr(stripped))

    def _parse(self, text: str) -> ServerMessage | None:
        stripped = text.strip()
        if not stripped:
            return None
        try:
            value = json.loads(stripped)
        except ValueError:
            self._emit(RawOutput(stripped))
            return None
        try:
            return parse_server_message(value)
        except CodexProtocolError as error:
            if _essential(value):
                msg = f"unreadable codex message: {error}"
                raise TurnFailedError(msg, kind=FailureKind.OTHER, detail=str(error)) from error
            self._emit(Lifecycle("unreadable_message", str(error)))
            return None


def _essential(value: object) -> bool:
    """Whether a turn would stall without *value*: a reply, or a turn-driving notification."""
    if not isinstance(value, dict):
        return False
    members: dict[object, object] = value  # pyright: ignore[reportUnknownVariableType]
    return "id" in members or members.get("method") in _ESSENTIAL_METHODS
