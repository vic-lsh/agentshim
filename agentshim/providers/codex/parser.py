"""Turn Codex's ``codex exec --json`` stdout into agentshim events."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from agentshim.core.events import (
    AssistantText,
    Lifecycle,
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
    CommandItem,
    ErrorFrame,
    GenericItem,
    ItemStarted,
    MessageItem,
    ReasoningItem,
    ThreadStarted,
    TurnCompleted,
    TurnStarted,
    parse_frame,
    summarize_item,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from agentshim.core.events import AgentEvent

    from .events import CodexFrame, CodexItem

PROVIDER_NAME = "codex"

#: Codex reports every shell command under one item type, so one tool name.
COMMAND_TOOL = "execute"

#: Item status that means the call failed rather than produced output.
FAILED_ITEM_STATUS = "failed"


def fold_usage(frame: TurnCompleted, previous: TokenUsage) -> TokenUsage:
    """Add one ``turn.completed`` frame's counts to the running total.

    Codex's ``input_tokens`` already includes ``cached_input_tokens``, so
    unlike Claude nothing is folded in: adding them would double-count the
    cached prefix and break ``cached_input_tokens <= input_tokens``.
    """
    return previous + TokenUsage(
        input_tokens=frame.input_tokens,
        output_tokens=frame.output_tokens,
        cached_input_tokens=frame.cached_input_tokens,
        turns=1,
    )


class CodexStreamParser:
    """Stateful parser for one ``codex exec`` run."""

    def __init__(
        self,
        emit: Callable[[AgentEvent], None],
        *,
        expect_structured: bool = False,
    ) -> None:
        """Build a parser that emits through *emit*.

        Args:
            emit: Sink for every event this run produces.
            expect_structured: Whether the turn asked Codex for a schema, in
                which case the final message is decoded as JSON.
        """
        self._emit = emit
        self._expect_structured = expect_structured
        self._tools = ToolTracker()
        self._session_id: str | None = None
        # Codex has no terminal result frame: the last agent_message is the
        # answer, and with a schema it is the structured payload verbatim.
        self._final_text: str | None = None
        self._tokens = TokenUsage()
        self._usage = ProviderUsage(provider=PROVIDER_NAME)
        self._error: str | None = None

    def feed_stdout(self, line: str) -> None:
        """Parse one stdout line, emitting whatever events it carries."""
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
        """Emit one stderr line as a ``Stderr`` event."""
        stripped = line.rstrip("\n")
        if stripped:
            self._emit(Stderr(stripped))

    def finish(self) -> ParsedTurn:
        """Return everything the run produced."""
        return ParsedTurn(
            text=self._final_text or "",
            structured_output=self._structured_payload(),
            session_id=self._session_id,
            usage=self._usage,
            cost_usd=None,
            error=self._error,
        )

    def _handle(self, frame: CodexFrame) -> None:
        if isinstance(frame, ThreadStarted):
            self._thread_started(frame)
        elif isinstance(frame, TurnStarted):
            self._emit(Lifecycle("turn_started", ""))
        elif isinstance(frame, TurnCompleted):
            self._turn_completed(frame)
        elif isinstance(frame, ErrorFrame):
            self._error = frame.message or "codex reported an error"
            self._emit(ProviderError(self._error))
        elif isinstance(frame, ItemStarted):
            self._item_started(frame.item)
        else:
            self._item_completed(frame.item)

    def _thread_started(self, frame: ThreadStarted) -> None:
        if not frame.thread_id:
            return
        if self._session_id is None:
            self._session_id = frame.thread_id
            self._emit(SessionStarted(frame.thread_id))
        self._emit(Lifecycle("thread_started", frame.thread_id))

    def _turn_completed(self, frame: TurnCompleted) -> None:
        self._tokens = fold_usage(frame, self._tokens)
        self._usage = ProviderUsage(
            tokens=self._tokens,
            total_cost_usd=None,
            provider=PROVIDER_NAME,
            raw=frame.usage if frame.usage is not None else self._usage.raw,
        )
        if frame.usage is None:
            self._emit(Lifecycle("turn_completed", ""))
            return
        detail = (
            f"in={frame.input_tokens} cached={frame.cached_input_tokens} out={frame.output_tokens}"
        )
        self._emit(Lifecycle("turn_completed", detail))
        self._emit(UsageReport(self._usage, None))

    def _item_started(self, item: CodexItem) -> None:
        if isinstance(item, CommandItem):
            self._tools.start(item.item_id, COMMAND_TOOL)
            self._emit(ToolCall(item.item_id, COMMAND_TOOL, {"command": item.command}))
        elif isinstance(item, GenericItem):
            self._tools.start(item.item_id, item.kind)
            self._emit(ToolCall(item.item_id, item.kind, dict(item.fields)))
        # agent_message and reasoning carry their text only on completion.

    def _item_completed(self, item: CodexItem) -> None:
        if isinstance(item, MessageItem):
            self._final_text = item.text
            if item.text:
                self._emit(AssistantText(item.text))
        elif isinstance(item, ReasoningItem):
            if item.text:
                self._emit(Reasoning(item.text))
        elif isinstance(item, CommandItem):
            failed = item.status == FAILED_ITEM_STATUS or bool(item.exit_code)
            self._emit(
                self._result(item.item_id, COMMAND_TOOL, item.output, item.exit_code, failed=failed)
            )
        else:
            self._emit(
                self._result(
                    item.item_id,
                    item.kind,
                    summarize_item(item),
                    None,
                    failed=item.status == FAILED_ITEM_STATUS,
                )
            )

    def _result(
        self,
        item_id: str | None,
        tool: str,
        output: str,
        exit_code: int | None,
        *,
        failed: bool,
    ) -> ToolResult:
        """Build a tool result, routing a failure onto ``stderr``.

        Every provider reports a failed tool the same way: the message on
        ``stderr`` with a nonzero ``exit_code``, so a renderer cannot mistake
        it for success.
        """
        if failed and exit_code is None:
            exit_code = 1
        return ToolResult(
            tool_id=item_id,
            tool=tool,
            stdout="" if failed else output,
            stderr=output if failed else "",
            exit_code=exit_code,
            duration_s=self._tools.duration(item_id),
        )

    def _structured_payload(self) -> object | None:
        """Decode the final message, which a schema turn asked Codex to make JSON."""
        if not self._expect_structured or not self._final_text:
            return None
        try:
            return json.loads(self._final_text)
        except (json.JSONDecodeError, ValueError):
            return None
