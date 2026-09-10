"""Typed agent events and the handlers that consume them.

Thread contract: ``on_event`` runs on the thread that called
``AgentSession.turn()``; the executor serializes reader-thread output before
the session sees it.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from typing import TextIO

    from .usage import ProviderUsage


@dataclass(frozen=True)
class RunStarted:
    """The CLI process is about to start."""

    argv: tuple[str, ...]


@dataclass(frozen=True)
class RunFinished:
    """The CLI process exited."""

    exit_code: int | None


@dataclass(frozen=True)
class SessionStarted:
    """The provider named the conversation."""

    session_id: str


@dataclass(frozen=True)
class AssistantText:
    """Assistant-facing message text."""

    text: str


@dataclass(frozen=True)
class Reasoning:
    """Thinking or reasoning text."""

    text: str


@dataclass(frozen=True)
class ToolCall:
    """A tool was invoked."""

    tool_id: str | None
    tool: str
    args: Mapping[str, Any] | str | None


@dataclass(frozen=True)
class ToolResult:
    """A tool finished."""

    tool_id: str | None
    tool: str
    stdout: str
    stderr: str
    exit_code: int | None
    duration_s: float | None


@dataclass(frozen=True)
class UsageReport:
    """Provider accounting for the turn so far."""

    usage: ProviderUsage
    cost_usd: float | None


@dataclass(frozen=True)
class Lifecycle:
    """Provider plumbing that is not model output."""

    kind: str
    detail: str


@dataclass(frozen=True)
class Stderr:
    """One stderr line."""

    text: str


@dataclass(frozen=True)
class RawOutput:
    """One stdout line that was not a provider event."""

    text: str


@dataclass(frozen=True)
class ProviderError:
    """The provider reported an error."""

    message: str


AgentEvent = (
    RunStarted
    | RunFinished
    | SessionStarted
    | AssistantText
    | Reasoning
    | ToolCall
    | ToolResult
    | UsageReport
    | Lifecycle
    | Stderr
    | RawOutput
    | ProviderError
)


@runtime_checkable
class AgentEventHandler(Protocol):
    """Receives every event of a turn, in order, on the caller's thread."""

    def on_event(self, event: AgentEvent) -> None: ...


class EventHandlerBase:
    """No-op handler; subclass and override ``on_event``."""

    def on_event(self, event: AgentEvent) -> None:
        """Ignore the event."""


class NullEventHandler(EventHandlerBase):
    """Handler that intentionally drops every event."""


class CompositeEventHandler:
    """Fan an event out to a fixed list of handlers, in order."""

    def __init__(self, handlers: Iterable[AgentEventHandler]) -> None:
        self.handlers: list[AgentEventHandler] = list(handlers)

    def on_event(self, event: AgentEvent) -> None:
        for handler in self.handlers:
            handler.on_event(event)


def compose_event_handlers(
    handler: AgentEventHandler | None,
    handlers: Sequence[AgentEventHandler] = (),
) -> AgentEventHandler:
    """Collapse the two constructor spellings into one handler.

    Returns a ``NullEventHandler`` when nothing was supplied so callers never
    branch on ``None``.
    """
    combined: list[AgentEventHandler] = []
    if handler is not None:
        combined.append(handler)
    combined.extend(handlers)
    if not combined:
        return NullEventHandler()
    if len(combined) == 1:
        return combined[0]
    return CompositeEventHandler(combined)


_TOOL_ARG_MAX_LEN = 200
_TOOL_OUTPUT_MAX_LINES = 10


def _truncate(value: object, max_len: int = _TOOL_ARG_MAX_LEN) -> str:
    text = str(value)
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def _truncate_lines(content: str, max_lines: int = _TOOL_OUTPUT_MAX_LINES) -> str:
    lines = content.splitlines()
    if len(lines) > max_lines * 2:
        return "\n".join([*lines[:max_lines], "... (truncated) ...", *lines[-max_lines:]])
    return content


class ConsoleEventHandler:
    """Render events as human-readable lines on a text stream.

    Console output is an event handler rather than something the session does
    so a caller can replace it or compose it with handlers of their own.
    """

    _BLUE = "\033[34m"
    _GREEN = "\033[32m"
    _DIM = "\033[2m"
    _RESET = "\033[0m"

    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        prefix: str = "[agent]",
        color: bool = True,
        show_lifecycle: bool = False,
    ) -> None:
        self._stream = stream if stream is not None else sys.stdout
        self._prefix = prefix
        self._color = color
        self._show_lifecycle = show_lifecycle
        self._at_line_start = True

    def _paint(self, text: str, code: str) -> str:
        if not self._color:
            return text
        return f"{code}{text}{self._RESET}"

    def _write(self, text: str) -> None:
        self._stream.write(text)

    def _newline_if_needed(self) -> None:
        if not self._at_line_start:
            self._write("\n")
            self._at_line_start = True

    def _line(self, text: str) -> None:
        self._newline_if_needed()
        self._write(f"{self._prefix} {text}\n")

    def _stream_text(self, content: str) -> None:
        """Write assistant text incrementally, prefixing each new line."""
        if not content:
            return
        parts = content.split("\n")
        for index, part in enumerate(parts):
            last = index == len(parts) - 1
            if part:
                if self._at_line_start:
                    self._write(f"{self._prefix} ")
                    self._at_line_start = False
                self._write(part)
            if not last:
                self._write("\n")
                self._at_line_start = True

    def on_event(self, event: AgentEvent) -> None:
        if isinstance(event, RunStarted):
            self._line(self._paint("$ " + " ".join(event.argv), self._DIM))
            self._write("=" * 80 + "\n")
        elif isinstance(event, RunFinished):
            self._newline_if_needed()
            self._write("=" * 80 + "\n")
        elif isinstance(event, AssistantText):
            self._stream_text(event.text)
        elif isinstance(event, Reasoning):
            self._line(self._paint(_truncate_lines(event.text), self._DIM))
        elif isinstance(event, ToolCall):
            self._line(self._paint(f"[Tool Use] {event.tool} {_truncate(event.args)}", self._BLUE))
        elif isinstance(event, ToolResult):
            output = event.stdout or event.stderr
            if output:
                self._line(self._paint(f"[Tool Result] {_truncate_lines(output)}", self._GREEN))
            else:
                self._line(self._paint(f"{event.tool} ran successfully", self._GREEN))
        elif isinstance(event, Stderr):
            self._line(f"[stderr] {event.text.rstrip()}")
        elif isinstance(event, ProviderError):
            self._line(f"[error] {event.message}")
        elif isinstance(event, Lifecycle):
            if self._show_lifecycle:
                self._line(self._paint(f"[{event.kind}] {event.detail}", self._DIM))
        elif isinstance(event, RawOutput):
            self._stream_text(event.text.rstrip("\n") + "\n")
        self._stream.flush()
