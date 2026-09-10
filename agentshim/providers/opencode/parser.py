"""Turn opencode's ``run --format json`` stdout into agentshim events."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from agentshim.core.events import (
    AssistantText,
    ProviderError,
    RawOutput,
    Reasoning,
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
    ReasoningEvent,
    StepFinishEvent,
    TextEvent,
    ToolEvent,
    parse_frame,
    session_id_of,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from typing import Any

    from agentshim.core.events import AgentEvent

    from .events import OpencodeFrame

PROVIDER_NAME = "opencode"

#: Tool state that means the call failed rather than produced output.
FAILED_TOOL_STATE = "error"


def fold_tokens(tokens: Mapping[str, Any] | None, turns: int = 1) -> TokenUsage:
    """Normalize one ``step-finish`` token block to the shared counts.

    opencode reports ``input`` as the tokens that were not served from the
    context cache, so the cache hits are added back in to make
    ``cached_input_tokens <= input_tokens`` hold. ``reasoning`` is part of
    the generated output, so it is both added to ``output_tokens`` and
    reported on its own.
    """
    if tokens is None:
        return TokenUsage(turns=turns)
    cache = _cache(tokens.get("cache"))
    write = _int(cache.get("write"))
    cached = _int(cache.get("read")) + write
    reasoning = _int(tokens.get("reasoning"))
    return TokenUsage(
        input_tokens=_int(tokens.get("input")) + cached,
        output_tokens=_int(tokens.get("output")) + reasoning,
        cached_input_tokens=cached,
        cache_write_input_tokens=write,
        reasoning_output_tokens=reasoning,
        turns=turns,
    )


def _cache(value: object) -> Mapping[str, Any]:
    if isinstance(value, dict):
        return cast("Mapping[str, Any]", value)
    return {}


def _int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    return 0


class OpencodeStreamParser:
    """Stateful parser for one opencode run.

    opencode publishes a tool part only once it is in a terminal state, so
    one frame carries both the call and its result; the tracker still pairs
    them so the tool name is resolved the same way as on every other
    provider, and the part's own ``time`` block supplies the real duration.
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
                opencode has no native output schema, so nothing reads it.
        """
        self._emit = emit
        self._expect_structured = expect_structured
        self._tools = ToolTracker()
        self._text: list[str] = []
        self._session_id: str | None = None
        self._tokens = TokenUsage()
        self._cost_usd: float | None = None
        self._raw_tokens: Mapping[str, Any] | None = None
        self._error: str | None = None

    def feed_stdout(self, line: str) -> None:
        """Consume one stdout line."""
        data = parse_json_object(line)
        if data is None:
            stripped = line.rstrip("\n")
            if stripped:
                self._emit(RawOutput(stripped))
            return
        self._session(data)
        frame = parse_frame(data)
        if frame is not None:
            self._handle(frame)

    def feed_stderr(self, line: str) -> None:
        """Consume one stderr line."""
        stripped = line.rstrip("\n")
        if stripped:
            self._emit(Stderr(stripped))

    def finish(self) -> ParsedTurn:
        """Report what the run produced."""
        return ParsedTurn(
            text="\n".join(self._text),
            structured_output=None,
            session_id=self._session_id,
            usage=self._usage(),
            cost_usd=self._cost_usd,
            error=self._error,
        )

    def _usage(self) -> ProviderUsage:
        return ProviderUsage(
            tokens=self._tokens,
            total_cost_usd=self._cost_usd,
            provider=PROVIDER_NAME,
            raw=self._raw_tokens,
        )

    def _session(self, data: Mapping[str, Any]) -> None:
        if self._session_id is not None:
            return
        session_id = session_id_of(data)
        if session_id is not None:
            self._session_id = session_id
            self._emit(SessionStarted(session_id))

    def _handle(self, frame: OpencodeFrame) -> None:
        if isinstance(frame, TextEvent):
            if frame.text:
                self._text.append(frame.text)
                self._emit(AssistantText(frame.text))
        elif isinstance(frame, ReasoningEvent):
            if frame.text:
                self._emit(Reasoning(frame.text))
        elif isinstance(frame, ToolEvent):
            self._tool(frame)
        elif isinstance(frame, StepFinishEvent):
            self._step_finish(frame)
        elif isinstance(frame, ErrorEvent):
            self._error = frame.message
            self._emit(ProviderError(frame.message))

    def _tool(self, frame: ToolEvent) -> None:
        self._tools.start(frame.tool_id, frame.tool)
        self._emit(ToolCall(frame.tool_id, frame.tool, frame.args))
        failed = frame.status == FAILED_TOOL_STATE
        message = frame.error_message or frame.output
        duration = (
            frame.duration_s
            if frame.duration_s is not None
            else self._tools.duration(frame.tool_id)
        )
        self._emit(
            ToolResult(
                tool_id=frame.tool_id,
                tool=self._tools.name(frame.tool_id),
                stdout="" if failed else frame.output,
                stderr=message if failed else "",
                exit_code=1 if failed else None,
                duration_s=duration,
            )
        )

    def _step_finish(self, frame: StepFinishEvent) -> None:
        step = fold_tokens(frame.tokens)
        self._tokens = self._tokens + step
        self._raw_tokens = frame.tokens if frame.tokens is not None else self._raw_tokens
        if frame.cost is not None:
            self._cost_usd = (self._cost_usd or 0.0) + frame.cost
        self._emit(UsageReport(self._usage(), self._cost_usd))
