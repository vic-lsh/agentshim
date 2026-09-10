"""Turn Gemini CLI's ``stream-json`` stdout into agentshim events."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentshim.core.events import (
    AssistantText,
    ProviderError,
    RawOutput,
    SessionStarted,
    Stderr,
    ToolCall,
    ToolResult,
    UsageReport,
)
from agentshim.core.provider import ParsedTurn
from agentshim.core.stream import ToolTracker, parse_json_object
from agentshim.core.usage import ProviderUsage, TokenUsage

from .events import (
    ErrorEvent,
    GeminiFrame,
    InitEvent,
    MessageEvent,
    ResultEvent,
    ToolResultEvent,
    ToolUseEvent,
    parse_frame,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from typing import Any

    from agentshim.core.events import AgentEvent

PROVIDER_NAME = "gemini"

#: Severity of an ``error`` frame that ends the turn rather than annotating it.
FATAL_SEVERITY = "error"


def fold_stats(stats: Mapping[str, Any] | None, turns: int = 0) -> TokenUsage:
    """Normalize Gemini's ``result`` stats to the shared token counts.

    ``StreamJsonFormatter.convertToStreamStats`` reports ``input_tokens`` as
    the prompt total and ``cached`` as the part of it served from the context
    cache, so the counts are already nested rather than disjoint. The clamp
    keeps ``cached_input_tokens <= input_tokens`` even if a future build
    reports them the other way around.
    """
    if stats is None:
        return TokenUsage(turns=turns)
    input_tokens = _int(stats.get("input_tokens"))
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=_int(stats.get("output_tokens")),
        cached_input_tokens=min(_int(stats.get("cached")), input_tokens),
        turns=turns,
    )


def _int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    return 0


class GeminiStreamParser:
    """Stateful parser for one Gemini CLI run.

    No ``Reasoning`` event is emitted: ``runNonInteractive`` in Gemini CLI
    0.26.0 forwards only content, tool, error and result events to the
    stream formatter, and drops ``GeminiEventType.Thought`` entirely, so the
    stream carries no thinking text to report.
    """

    def __init__(
        self,
        emit: Callable[[AgentEvent], None],
        *,
        expect_structured: bool = False,
    ) -> None:
        """Start a parser that publishes events through ``emit``.

        Args:
            emit: Sink for every event this run produces.
            expect_structured: Recorded only to keep the parser substitutable;
                Gemini has no native output schema, so nothing reads it.
        """
        self._emit = emit
        self._expect_structured = expect_structured
        self._tools = ToolTracker()
        self._text: list[str] = []
        self._session_id: str | None = None
        self._usage = ProviderUsage(provider=PROVIDER_NAME)
        self._error: str | None = None

    def feed_stdout(self, line: str) -> None:
        """Consume one stdout line."""
        data = parse_json_object(line)
        if data is None:
            stripped = line.rstrip("\n")
            if stripped:
                self._emit(RawOutput(stripped))
            return
        frame = parse_frame(data)
        if frame is not None:
            self._handle(frame)

    def feed_stderr(self, line: str) -> None:
        """Consume one stderr line."""
        stripped = line.rstrip("\n")
        if stripped:
            self._emit(Stderr(stripped))

    def finish(self) -> ParsedTurn:
        """Report what the run produced.

        Assistant frames are deltas of one message, so the accumulated
        chunks are concatenated without a separator.
        """
        return ParsedTurn(
            text="".join(self._text),
            structured_output=None,
            session_id=self._session_id,
            usage=self._usage,
            cost_usd=None,
            error=self._error,
        )

    def _handle(self, frame: GeminiFrame) -> None:
        if isinstance(frame, InitEvent):
            if self._session_id is None and frame.session_id:
                self._session_id = frame.session_id
                self._emit(SessionStarted(frame.session_id))
        elif isinstance(frame, MessageEvent):
            self._message(frame)
        elif isinstance(frame, ToolUseEvent):
            self._tools.start(frame.tool_id, frame.tool)
            self._emit(ToolCall(frame.tool_id, frame.tool, frame.parameters))
        elif isinstance(frame, ToolResultEvent):
            self._tool_result(frame)
        elif isinstance(frame, ErrorEvent):
            self._error_frame(frame)
        else:
            self._result(frame)

    def _message(self, frame: MessageEvent) -> None:
        # The ``user`` frame is the CLI echoing back the prompt we sent on
        # stdin; reporting it as model output would double it up.
        if frame.role != "assistant" or not frame.content:
            return
        self._text.append(frame.content)
        self._emit(AssistantText(frame.content))

    def _tool_result(self, frame: ToolResultEvent) -> None:
        failed = frame.status == "error" or frame.error_message is not None
        message = frame.error_message or frame.output
        self._emit(
            ToolResult(
                tool_id=frame.tool_id,
                tool=self._tools.name(frame.tool_id),
                stdout="" if failed else frame.output,
                stderr=message if failed else "",
                exit_code=1 if failed else None,
                duration_s=self._tools.duration(frame.tool_id),
            )
        )

    def _error_frame(self, frame: ErrorEvent) -> None:
        self._emit(ProviderError(frame.message))
        if frame.severity == FATAL_SEVERITY:
            self._error = frame.message

    def _result(self, frame: ResultEvent) -> None:
        # One CLI invocation is one turn: Gemini's stats carry no count of
        # the tool-calling rounds the agent loop took.
        self._usage = ProviderUsage(
            tokens=fold_stats(frame.stats, turns=1),
            total_cost_usd=None,
            provider=PROVIDER_NAME,
            raw=frame.stats,
        )
        self._emit(UsageReport(self._usage, None))
        if frame.status == "error":
            self._error = frame.error_message or "gemini reported an error"
            self._emit(ProviderError(self._error))
