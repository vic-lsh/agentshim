"""Turn the Copilot CLI's ``--output-format json`` stdout into agentshim events."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

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
    AssistantIntent,
    AssistantMessage,
    AssistantMessageDelta,
    ResultFrame,
    SessionError,
    SessionStart,
    ToolExecutionComplete,
    ToolExecutionStart,
    TurnEnd,
    UsageFrame,
    parse_frame,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from typing import Any

    from agentshim.core.events import AgentEvent

    from .events import CopilotFrame

PROVIDER_NAME = "copilot"


def fold_usage(frame: UsageFrame) -> TokenUsage:
    """Normalize one ``assistant.usage`` frame to the shared token counts.

    Copilot reports cache reads and cache writes disjoint from
    ``inputTokens``, and reasoning tokens disjoint from ``outputTokens``.
    Folding both in is what makes ``cached_input_tokens <= input_tokens``
    hold here as it does on every other provider, and keeps the output count
    comparable with providers that bill reasoning as output.
    """
    cached = frame.cache_read_tokens + frame.cache_write_tokens
    return TokenUsage(
        input_tokens=frame.input_tokens + cached,
        output_tokens=frame.output_tokens + frame.reasoning_tokens,
        cached_input_tokens=cached,
        cache_write_input_tokens=frame.cache_write_tokens,
        reasoning_output_tokens=frame.reasoning_tokens,
    )


class CopilotStreamParser:
    """Stateful parser for one Copilot CLI run.

    Known gap, verified against Copilot CLI 1.0.83: a run reports no billed
    token counts at all. The ``assistant.usage`` frame this parser folds into
    ``TokenUsage`` is not printed, and neither is ``assistant.message``'s
    ``outputTokens``, so ``ParsedTurn.usage.tokens`` is all zeros. What the
    CLI does print is a ``session.usage_checkpoint`` frame carrying
    ``totalPremiumRequests`` and ``totalNanoAiu`` (billing units, not
    tokens) plus prompt-cache diagnostics: ``prompt_tokens``,
    ``frontier_tokens``, ``tool_tokens`` and per-segment ``tokens``. Those
    describe how the prompt was assembled for the cache, not what the turn
    was charged, so reading them into ``input_tokens`` would report a number
    that is not the turn's usage. The frame is therefore left unparsed until
    the CLI prints real counts.
    ``tests/fixtures/copilot/usage_checkpoint_1_0_83.jsonl`` is a recording
    of such a run, and the fixture suite pins this behaviour.
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
                Copilot has no native output schema, so nothing reads it.
        """
        self._emit = emit
        self._expect_structured = expect_structured
        self._tools = ToolTracker()
        self._session_id: str | None = None
        self._final_text: str | None = None
        self._deltas: list[str] = []
        self._streamed: set[str] = set()
        self._tokens = TokenUsage()
        self._saw_usage_frame = False
        self._message_output_tokens = 0
        self._turns = 0
        self._raw: Mapping[str, Any] | None = None
        self._error: str | None = None

    def feed_stdout(self, line: str) -> None:
        """Consume one stdout line, emitting whatever events it carries."""
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
        """Return what the run produced once its output is exhausted."""
        return ParsedTurn(
            text=self._text(),
            structured_output=None,
            session_id=self._session_id,
            usage=self._usage(),
            cost_usd=None,
            error=self._error,
        )

    def _text(self) -> str:
        if self._final_text is not None:
            return self._final_text
        return "".join(self._deltas).strip()

    def _usage(self) -> ProviderUsage:
        # Without an ``assistant.usage`` frame the only token count the CLI
        # printed is the per-message output count.
        tokens = (
            self._tokens
            if self._saw_usage_frame
            else TokenUsage(output_tokens=self._message_output_tokens)
        )
        return ProviderUsage(
            tokens=dataclasses.replace(tokens, turns=self._turns),
            total_cost_usd=None,
            provider=PROVIDER_NAME,
            raw=self._raw,
        )

    def _handle(self, frame: CopilotFrame) -> None:
        if isinstance(frame, AssistantMessageDelta):
            self._delta(frame)
        elif isinstance(frame, AssistantMessage):
            self._message(frame)
        elif isinstance(frame, ToolExecutionStart):
            self._tool_call(frame)
        elif isinstance(frame, ToolExecutionComplete):
            self._tool_result(frame)
        else:
            self._handle_session(frame)

    def _handle_session(
        self,
        frame: SessionStart | AssistantIntent | TurnEnd | UsageFrame | SessionError | ResultFrame,
    ) -> None:
        if isinstance(frame, UsageFrame):
            self._usage_frame(frame)
        elif isinstance(frame, TurnEnd):
            self._turns += 1
        elif isinstance(frame, AssistantIntent):
            self._emit(Reasoning(frame.intent))
        elif isinstance(frame, SessionStart):
            self._start_session(frame.session_id)
        elif isinstance(frame, SessionError):
            self._session_error(frame)
        else:
            self._result(frame)

    def _delta(self, frame: AssistantMessageDelta) -> None:
        if not frame.delta_content:
            return
        if frame.message_id is not None:
            self._streamed.add(frame.message_id)
        self._deltas.append(frame.delta_content)
        self._emit(AssistantText(frame.delta_content))

    def _message(self, frame: AssistantMessage) -> None:
        self._message_output_tokens += frame.output_tokens
        if not frame.content:
            return
        self._final_text = frame.content
        # A message that was already streamed as deltas must not be emitted
        # twice; the complete frame only confirms what the deltas carried.
        if frame.message_id is None or frame.message_id not in self._streamed:
            self._emit(AssistantText(frame.content))

    def _tool_call(self, frame: ToolExecutionStart) -> None:
        self._tools.start(frame.tool_id, frame.tool)
        self._emit(ToolCall(frame.tool_id, frame.tool, frame.args))

    def _tool_result(self, frame: ToolExecutionComplete) -> None:
        failed = not frame.success
        exit_code = frame.exit_code
        if exit_code is None and failed:
            exit_code = 1
        self._emit(
            ToolResult(
                tool_id=frame.tool_id,
                tool=self._tools.name(frame.tool_id),
                stdout="" if failed else frame.output,
                stderr=(frame.error_message or frame.output) if failed else "",
                exit_code=exit_code,
                duration_s=self._tools.duration(frame.tool_id),
            )
        )

    def _usage_frame(self, frame: UsageFrame) -> None:
        self._saw_usage_frame = True
        self._tokens = self._tokens + fold_usage(frame)
        if frame.raw is not None:
            self._raw = dict(frame.raw)
        self._emit(
            UsageReport(
                ProviderUsage(
                    tokens=fold_usage(frame),
                    total_cost_usd=None,
                    provider=PROVIDER_NAME,
                    raw=frame.raw,
                ),
                None,
            )
        )

    def _session_error(self, frame: SessionError) -> None:
        message = frame.message or frame.error_type or "copilot reported an error"
        self._error = message
        self._emit(ProviderError(message))

    def _result(self, frame: ResultFrame) -> None:
        self._start_session(frame.session_id)
        if self._raw is None and frame.usage is not None:
            # No token counts, but the session totals still beat nothing when
            # a caller is diagnosing a run.
            self._raw = dict(frame.usage)
        # Copilot reports no per-turn cost, so the terminal report exists to
        # guarantee one usage event per turn.
        self._emit(UsageReport(self._usage(), None))

    def _start_session(self, session_id: str | None) -> None:
        if self._session_id is None and session_id:
            self._session_id = session_id
            self._emit(SessionStarted(session_id))
