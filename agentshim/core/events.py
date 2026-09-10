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

    def on_event(self, event: AgentEvent) -> None:
        """React to one event.

        Called synchronously in the middle of a turn, so a slow handler slows
        the turn down. A handler that raises fails the turn it is watching.
        """
        ...


class EventHandlerBase:
    """No-op handler; subclass and override ``on_event``."""

    def on_event(self, event: AgentEvent) -> None:
        """Ignore the event."""


class NullEventHandler(EventHandlerBase):
    """Handler that intentionally drops every event."""


class CompositeEventHandler:
    """Fan an event out to a fixed list of handlers, in order."""

    def __init__(self, handlers: Iterable[AgentEventHandler]) -> None:
        """Snapshot the handlers to fan out to.

        The iterable is copied at construction, so mutating the sequence that
        was passed in does not change this handler; the copy is exposed as the
        public ``handlers`` list instead.
        """
        self.handlers: list[AgentEventHandler] = list(handlers)

    def on_event(self, event: AgentEvent) -> None:
        """Pass the event to each handler in turn.

        There is no error isolation: a handler that raises stops the ones
        after it, matching the single-handler case where the turn fails.
        """
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
    _RED = "\033[31m"
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
        """Set up a console renderer bound to one text stream.

        ``stream`` is resolved once, at construction, so a later reassignment
        of ``sys.stdout`` does not redirect an existing handler. ``color``
        controls ANSI escapes only, and ``show_lifecycle`` opts in to provider
        plumbing that is hidden by default because it is not model output.
        """
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
        """Render one event, then flush so a piped console stays live.

        Dispatch is by event family. Events with no console rendering, such as
        ``SessionStarted`` and ``UsageReport``, fall through and only flush.
        """
        if isinstance(event, (RunStarted, RunFinished)):
            self._render_run_boundary(event)
        elif isinstance(event, (AssistantText, Reasoning, RawOutput)):
            self._render_model_output(event)
        elif isinstance(event, (ToolCall, ToolResult)):
            self._render_tool(event)
        else:
            self._render_diagnostic(event)
        self._stream.flush()

    def _render_run_boundary(self, event: RunStarted | RunFinished) -> None:
        """Fence the CLI process with rules, so a transcript shows where it ran.

        The start rule follows the command line; the end rule only has to close
        a partial line first.
        """
        if isinstance(event, RunStarted):
            self._line(self._paint("$ " + " ".join(event.argv), self._DIM))
        else:
            self._newline_if_needed()
        self._write("=" * 80 + "\n")

    def _render_model_output(self, event: AssistantText | Reasoning | RawOutput) -> None:
        """Write what the model produced.

        Assistant text and unrecognized stdout stream incrementally, since they
        arrive in fragments. Reasoning is dimmed and clipped: it is context for
        the reader, not the answer.
        """
        if isinstance(event, AssistantText):
            self._stream_text(event.text)
        elif isinstance(event, Reasoning):
            self._line(self._paint(_truncate_lines(event.text), self._DIM))
        else:
            self._stream_text(event.text.rstrip("\n") + "\n")

    def _render_tool(self, event: ToolCall | ToolResult) -> None:
        """Summarize a tool call or its result on one line.

        Arguments and output are clipped: a console reader wants to know which
        tool ran and roughly what came back, not to read a whole file dump.
        """
        if isinstance(event, ToolCall):
            self._line(self._paint(f"[Tool Use] {event.tool} {_truncate(event.args)}", self._BLUE))
            return
        # A failed tool reports on ``stderr`` with a nonzero ``exit_code`` on
        # every provider, so the exit code is what decides the colour: reading
        # the streams instead painted a failure green whenever the CLI still
        # printed something on stdout.
        failed = bool(event.exit_code)
        colour = self._RED if failed else self._GREEN
        output = event.stdout or event.stderr
        if output:
            self._line(self._paint(f"[Tool Result] {_truncate_lines(output)}", colour))
        elif failed:
            self._line(self._paint(f"{event.tool} failed with exit {event.exit_code}", colour))
        else:
            self._line(self._paint(f"{event.tool} ran successfully", colour))

    def _render_diagnostic(self, event: AgentEvent) -> None:
        """Write the tagged non-output events, and drop the rest.

        Lifecycle chatter is suppressed unless the handler was asked for it,
        because on most providers it is far noisier than the model output.
        """
        if isinstance(event, Stderr):
            self._line(f"[stderr] {event.text.rstrip()}")
        elif isinstance(event, ProviderError):
            self._line(f"[error] {event.message}")
        elif isinstance(event, Lifecycle) and self._show_lifecycle:
            self._line(self._paint(f"[{event.kind}] {event.detail}", self._DIM))
