"""Turn Claude Code's ``stream-json`` stdout into agentshim events."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

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
    AssistantMessage,
    ResultFrame,
    SystemInit,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    parse_frame,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from agentshim.core.events import AgentEvent

PROVIDER_NAME = "claude"


def fold_usage(usage: Mapping[str, Any] | None, turns: int = 0) -> TokenUsage:
    """Normalize Claude's usage mapping to the shared token counts.

    Anthropic reports ``cache_creation_input_tokens`` and
    ``cache_read_input_tokens`` as disjoint from ``input_tokens``; folding
    them in is what makes ``cached_input_tokens <= input_tokens`` hold on
    every provider.
    """
    if usage is None:
        return TokenUsage(turns=turns)
    created = _int(usage.get("cache_creation_input_tokens"))
    read = _int(usage.get("cache_read_input_tokens"))
    cached = created + read
    return TokenUsage(
        input_tokens=_int(usage.get("input_tokens")) + cached,
        output_tokens=_int(usage.get("output_tokens")),
        cached_input_tokens=cached,
        cache_write_input_tokens=created,
        turns=turns,
    )


def _int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    return 0


class ClaudeStreamParser:
    """Stateful parser for one Claude Code run."""

    def __init__(
        self,
        emit: Callable[[AgentEvent], None],
        *,
        expect_structured: bool = False,
    ) -> None:
        """Start a parser for one run.

        Set *expect_structured* when the turn asked for a schema: it enables
        the fallback that reads the payload out of the result text.
        """
        self._emit = emit
        self._expect_structured = expect_structured
        self._tools = ToolTracker()
        self._text: list[str] = []
        self._session_id: str | None = None
        self._structured: object | None = None
        self._final_text: str | None = None
        self._usage = ProviderUsage(provider=PROVIDER_NAME)
        self._cost_usd: float | None = None
        self._error: str | None = None

    def feed_stdout(self, line: str) -> None:
        """Consume one stdout line, emitting the events its frame implies.

        A line that is not a JSON object is surfaced as ``RawOutput`` rather
        than dropped, so a CLI that prints prose is still observable.
        """
        data = parse_json_object(line)
        if data is None:
            stripped = line.rstrip("\n")
            if stripped:
                self._emit(RawOutput(stripped))
            return
        frame = parse_frame(data)
        if frame is None:
            return
        self._handle(frame)

    def feed_stderr(self, line: str) -> None:
        """Emit one non-blank stderr line as a ``Stderr`` event."""
        stripped = line.rstrip("\n")
        if stripped:
            self._emit(Stderr(stripped))

    def finish(self) -> ParsedTurn:
        """Collect what the run produced, after its last line was fed."""
        text = self._final_text or "\n".join(self._text)
        return ParsedTurn(
            text=text,
            structured_output=self._structured,
            session_id=self._session_id,
            usage=self._usage,
            cost_usd=self._cost_usd,
            error=self._error,
        )

    def _handle(self, frame: object) -> None:
        if isinstance(frame, SystemInit):
            if self._session_id is None and frame.session_id:
                self._session_id = frame.session_id
                self._emit(SessionStarted(frame.session_id))
        elif isinstance(frame, AssistantMessage):
            self._assistant(frame)
        elif isinstance(frame, ToolResultBlock):
            self._tool_result(frame)
        elif isinstance(frame, ResultFrame):
            self._result(frame)

    def _assistant(self, frame: AssistantMessage) -> None:
        if frame.usage is not None:
            self._emit(
                UsageReport(
                    ProviderUsage(
                        tokens=fold_usage(frame.usage),
                        total_cost_usd=None,
                        provider=PROVIDER_NAME,
                        raw=frame.usage,
                    ),
                    None,
                )
            )
        for block in frame.blocks:
            if isinstance(block, TextBlock):
                self._text.append(block.text)
                self._emit(AssistantText(block.text))
            elif isinstance(block, ThinkingBlock):
                self._emit(Reasoning(block.text))
            else:
                self._tools.start(block.tool_id, block.tool)
                self._emit(ToolCall(block.tool_id, block.tool, block.args))

    def _tool_result(self, frame: ToolResultBlock) -> None:
        name = self._tools.name(frame.tool_id)
        self._emit(
            ToolResult(
                tool_id=frame.tool_id,
                tool=name,
                stdout="" if frame.is_error else frame.output,
                stderr=frame.output if frame.is_error else "",
                exit_code=1 if frame.is_error else None,
                duration_s=self._tools.duration(frame.tool_id),
            )
        )

    def _result(self, frame: ResultFrame) -> None:
        self._final_text = frame.text
        self._cost_usd = frame.total_cost_usd
        self._structured = self._structured_payload(frame)
        self._usage = ProviderUsage(
            tokens=fold_usage(frame.usage, turns=frame.num_turns or 0),
            total_cost_usd=frame.total_cost_usd,
            provider=PROVIDER_NAME,
            raw=frame.usage,
        )
        self._emit(UsageReport(self._usage, frame.total_cost_usd))
        if frame.is_error:
            self._error = frame.text or (frame.subtype or "claude reported an error")
            self._emit(ProviderError(self._error))

    def _structured_payload(self, frame: ResultFrame) -> object | None:
        if frame.structured_output is not None:
            return frame.structured_output
        if not self._expect_structured or not frame.text:
            return None
        # Older Claude Code builds put the schema-conformant payload in
        # ``result`` instead of a dedicated field.
        try:
            return json.loads(frame.text)
        except (json.JSONDecodeError, ValueError):
            return None
