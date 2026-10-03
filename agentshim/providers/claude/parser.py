"""Turn Claude Code's ``stream-json`` stdout into agentshim events."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, cast

from agentshim.core.events import (
    AssistantText,
    ProviderError,
    RawOutput,
    Reasoning,
    SessionStarted,
    SkillInvoked,
    SkillsDiscovered,
    Stderr,
    ToolCall,
    ToolResult,
    UsageReport,
)
from agentshim.core.provider import ParsedTurn
from agentshim.core.stream import ToolTracker, parse_json_object
from agentshim.core.usage import ProviderUsage, TokenUsage, normalized_usage

from .events import (
    AssistantMessage,
    ResultFrame,
    SystemInit,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    parse_frame,
)
from .failures import SCHEMA_RETRIES_SUBTYPE, classify_failure

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from agentshim.core.errors import FailureKind
    from agentshim.core.events import AgentEvent

PROVIDER_NAME = "claude"

#: The built-in tool through which Claude Code loads a skill.
SKILL_TOOL = "Skill"

#: The tool through which Claude Code submits a schema-constrained answer;
#: its error result is the validation failure of one submission.
STRUCTURED_OUTPUT_TOOL = "StructuredOutput"

#: The built-in tool through which Claude Code reads a file.
READ_TOOL = "Read"

#: ``.../skills/<name>/SKILL.md``: a skill's instructions, read directly.
_SKILL_FILE = re.compile(r"/skills/(?:[^/]+/)*?(?P<name>[^/]+)/SKILL\.md$")


def fold_usage(usage: Mapping[str, Any] | None, turns: int = 0) -> TokenUsage:
    """Normalize Claude's usage mapping to the shared token counts.

    Anthropic reports ``cache_creation_input_tokens`` (cache writes) and
    ``cache_read_input_tokens`` disjoint from ``input_tokens`` (uncached
    input); both are folded in so ``input_tokens`` is the total, as on every
    provider. The ``cache_creation`` breakdown gives the one-hour-TTL writes,
    which are billed above five-minute writes. Claude bills thinking as
    output and does not report it apart, so ``reasoning_output_tokens`` is 0.
    """
    if usage is None:
        return TokenUsage(turns=turns)
    created = _int(usage.get("cache_creation_input_tokens"))
    read = _int(usage.get("cache_read_input_tokens"))
    breakdown = usage.get("cache_creation")
    created_1h = (
        _int(cast("Mapping[str, Any]", breakdown).get("ephemeral_1h_input_tokens"))
        if isinstance(breakdown, dict)
        else 0
    )
    return normalized_usage(
        input_tokens=_int(usage.get("input_tokens")) + read + created,
        output_tokens=_int(usage.get("output_tokens")),
        cache_read_input_tokens=read,
        cache_write_input_tokens=created,
        cache_write_1h_input_tokens=created_1h,
        turns=turns,
    )


def _int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def skill_invocation(
    tool_id: str | None, tool: str, args: Mapping[str, Any] | str | None
) -> SkillInvoked | None:
    """Recognize a tool call that loads a skill.

    Claude Code loads a skill through its ``Skill`` tool. An agent can also
    read a ``SKILL.md`` with ``Read``, which loads the same instructions.
    """
    if not isinstance(args, dict):
        return None
    if tool == SKILL_TOOL:
        name = args.get("skill")
        if isinstance(name, str) and name:
            return SkillInvoked(name=name, tool_id=tool_id)
        return None
    if tool == READ_TOOL:
        path = args.get("file_path")
        if isinstance(path, str):
            match = _SKILL_FILE.search(path)
            if match is not None:
                return SkillInvoked(name=match["name"], source_path=path, tool_id=tool_id)
    return None


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
        # What the stream said about a failed API request, for classifying it.
        self._api_error: str | None = None
        self._api_error_text: str | None = None
        self._api_error_status: int | None = None
        self._subtype: str | None = None
        # The validation errors of the last rejected StructuredOutput call.
        self._schema_errors: str | None = None
        self._stderr: list[str] = []

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
            self._stderr.append(stripped)
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
            error_kind=self._error_kind(),
        )

    def _error_kind(self) -> FailureKind:
        """Classify the failure from the stream, or from stderr if it had none.

        A CLI that dies before writing a ``result`` frame leaves stderr as the
        only account of why, so stderr is read only then.
        """
        reported = [text for text in (self._error, self._api_error_text) if text]
        text = "\n".join(reported) if reported else "\n".join(self._stderr)
        return classify_failure(
            subtype=self._subtype,
            api_error=self._api_error,
            status=self._api_error_status,
            text=text,
        )

    def _handle(self, frame: object) -> None:
        if isinstance(frame, SystemInit):
            if self._session_id is None and frame.session_id:
                self._session_id = frame.session_id
                self._emit(SessionStarted(frame.session_id))
            if frame.skills is not None:
                self._emit(SkillsDiscovered(frame.skills))
        elif isinstance(frame, AssistantMessage):
            self._assistant(frame)
        elif isinstance(frame, ToolResultBlock):
            self._tool_result(frame)
        elif isinstance(frame, ResultFrame):
            self._result(frame)

    def _assistant(self, frame: AssistantMessage) -> None:
        if frame.error is not None:
            self._api_error = frame.error
            self._api_error_text = "\n".join(
                block.text for block in frame.blocks if isinstance(block, TextBlock)
            )
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
                skill = skill_invocation(block.tool_id, block.tool, block.args)
                if skill is not None:
                    self._emit(skill)

    def _tool_result(self, frame: ToolResultBlock) -> None:
        name = self._tools.name(frame.tool_id)
        if name == STRUCTURED_OUTPUT_TOOL and frame.is_error:
            self._schema_errors = frame.output
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
            self._api_error_status = frame.api_error_status
            self._subtype = frame.subtype
            self._error = self._error_text(frame)
            self._emit(ProviderError(self._error))

    def _structured_payload(self, frame: ResultFrame) -> object | None:
        # Only a turn that asked for a schema gets a payload: an unrequested
        # ``structured_output`` field must never replace the prose answer.
        if not self._expect_structured:
            return None
        if frame.structured_output is not None:
            return frame.structured_output
        if not frame.text:
            return None
        # Older Claude Code builds put the schema-conformant payload in
        # ``result`` instead of a dedicated field.
        try:
            return json.loads(frame.text)
        except (json.JSONDecodeError, ValueError):
            return None

    def _error_text(self, frame: ResultFrame) -> str:
        """Say what an error ``result`` frame reported.

        A turn that used up its schema retries is described by the validation
        errors of its last rejected submission, which only the
        ``StructuredOutput`` tool result carries. Otherwise an ``error_*``
        subtype carries its messages in ``errors`` and often no ``result``
        text at all, so both are read before falling back to the subtype,
        which is the only part that names some failures.
        """
        if frame.subtype == SCHEMA_RETRIES_SUBTYPE and self._schema_errors:
            return self._schema_errors
        return frame.text or "; ".join(frame.errors) or frame.subtype or "claude reported an error"
