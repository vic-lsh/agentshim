"""Typed view of the Copilot CLI's ``--output-format json`` frames.

Kept separate from the parser so the frame shapes are testable on their own
and so a change in Copilot's wire format has one place to land.

Every frame but ``result`` carries its fields under a ``data`` object; the
``result`` frame puts them at the top level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


@dataclass(frozen=True)
class SessionStart:
    """``session.start``: names the resumable conversation."""

    session_id: str | None


@dataclass(frozen=True)
class AssistantIntent:
    """``assistant.intent``: an ephemeral note of what the agent is about to do."""

    intent: str


@dataclass(frozen=True)
class AssistantMessage:
    """``assistant.message``: one complete assistant message."""

    message_id: str | None
    content: str
    output_tokens: int = 0


@dataclass(frozen=True)
class AssistantMessageDelta:
    """``assistant.message_delta``: one streamed chunk of a message."""

    message_id: str | None
    delta_content: str


@dataclass(frozen=True)
class TurnEnd:
    """``assistant.turn_end``: the model finished one turn of the exchange."""

    turn_id: str | None


@dataclass(frozen=True)
class UsageFrame:
    """``assistant.usage``: token accounting for one model request.

    ``raw`` keeps the payload the CLI printed so a caller can diagnose a
    normalization gap without re-parsing the stream.
    """

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    raw: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ToolExecutionStart:
    """``tool.execution_start``: a tool was invoked."""

    tool_id: str | None
    tool: str
    args: Mapping[str, Any] | str | None


@dataclass(frozen=True)
class ToolExecutionComplete:
    """``tool.execution_complete``: a tool finished, successfully or not."""

    tool_id: str | None
    success: bool
    output: str = ""
    error_message: str = ""
    exit_code: int | None = None


@dataclass(frozen=True)
class SessionError:
    """``session.error``: the CLI reported a session-level failure."""

    message: str
    error_type: str | None = None


@dataclass(frozen=True)
class ResultFrame:
    """The terminal frame: the session id, the exit code, and session totals.

    Its ``usage`` mapping carries premium-request and duration counters, not
    token counts, so it is kept only for diagnostics.
    """

    session_id: str | None
    exit_code: int | None = None
    usage: Mapping[str, Any] | None = None


CopilotFrame = (
    SessionStart
    | AssistantIntent
    | AssistantMessage
    | AssistantMessageDelta
    | TurnEnd
    | UsageFrame
    | ToolExecutionStart
    | ToolExecutionComplete
    | SessionError
    | ResultFrame
)


def parse_frame(data: Mapping[str, Any]) -> CopilotFrame | None:
    """Map one decoded stdout object to a frame, or ``None`` if uninteresting."""
    kind = data.get("type")
    if not isinstance(kind, str):
        return None
    if kind == "result":
        return _result(data)
    builder = _BUILDERS.get(kind)
    if builder is None:
        return None
    return builder(_mapping(data.get("data")) or {})


def _session_start(payload: Mapping[str, Any]) -> CopilotFrame | None:
    return SessionStart(session_id=_str_or_none(payload.get("sessionId")))


def _intent(payload: Mapping[str, Any]) -> CopilotFrame | None:
    intent = _str_or_none(payload.get("intent"))
    if intent is None:
        return None
    return AssistantIntent(intent=intent)


def _message(payload: Mapping[str, Any]) -> CopilotFrame | None:
    return AssistantMessage(
        message_id=_str_or_none(payload.get("messageId")),
        content=_str(payload.get("content")),
        output_tokens=_int(payload.get("outputTokens")),
    )


def _message_delta(payload: Mapping[str, Any]) -> CopilotFrame | None:
    return AssistantMessageDelta(
        message_id=_str_or_none(payload.get("messageId")),
        delta_content=_str(payload.get("deltaContent")),
    )


def _turn_end(payload: Mapping[str, Any]) -> CopilotFrame | None:
    return TurnEnd(turn_id=_str_or_none(payload.get("turnId")))


def _usage(payload: Mapping[str, Any]) -> CopilotFrame | None:
    return UsageFrame(
        model=_str(payload.get("model")),
        input_tokens=_int(payload.get("inputTokens")),
        output_tokens=_int(payload.get("outputTokens")),
        cache_read_tokens=_int(payload.get("cacheReadTokens")),
        cache_write_tokens=_int(payload.get("cacheWriteTokens")),
        reasoning_tokens=_int(payload.get("reasoningTokens")),
        raw=payload,
    )


def _tool_start(payload: Mapping[str, Any]) -> CopilotFrame | None:
    args = payload.get("arguments")
    return ToolExecutionStart(
        tool_id=_str_or_none(payload.get("toolCallId")),
        tool=_str_or_none(payload.get("toolName")) or "Tool",
        args=cast("Mapping[str, Any]", args) if isinstance(args, dict) else _str_or_none(args),
    )


def _tool_complete(payload: Mapping[str, Any]) -> CopilotFrame | None:
    output, exit_code = tool_output(payload.get("result"))
    error = _mapping(payload.get("error")) or {}
    return ToolExecutionComplete(
        tool_id=_str_or_none(payload.get("toolCallId")),
        success=bool(payload.get("success")),
        output=output,
        error_message=_str(error.get("message")),
        exit_code=exit_code,
    )


def _error(payload: Mapping[str, Any]) -> CopilotFrame | None:
    return SessionError(
        message=_str(payload.get("message")),
        error_type=_str_or_none(payload.get("errorType")),
    )


def _result(data: Mapping[str, Any]) -> CopilotFrame | None:
    return ResultFrame(
        session_id=_str_or_none(data.get("sessionId")),
        exit_code=_int_or_none(data.get("exitCode")),
        usage=_mapping(data.get("usage")),
    )


_BUILDERS: dict[str, Callable[[Mapping[str, Any]], CopilotFrame | None]] = {
    "session.start": _session_start,
    "assistant.intent": _intent,
    "assistant.message": _message,
    "assistant.message_delta": _message_delta,
    "assistant.turn_end": _turn_end,
    "assistant.usage": _usage,
    "tool.execution_start": _tool_start,
    "tool.execution_complete": _tool_complete,
    "session.error": _error,
}


def tool_output(result: object) -> tuple[str, int | None]:
    """Flatten a tool result payload into its text and its exit code.

    Copilot renders a tool result three ways: ``detailedContent`` (the full
    output), ``content`` (an abbreviated form), and a ``contents`` block list.
    The detailed form wins; the block list is joined only when neither string
    field carried anything. A ``terminal`` block also declares the exit code
    of the command that produced it.
    """
    payload = _mapping(result) or {}
    detailed = _str(payload.get("detailedContent"))
    output = detailed or _str(payload.get("content"))
    blocks, exit_code = _content_blocks(payload.get("contents"))
    if not output and blocks:
        output = "\n".join(blocks)
    return output, exit_code


def _content_blocks(contents: object) -> tuple[list[str], int | None]:
    if not isinstance(contents, list):
        return [], None
    rendered: list[str] = []
    exit_code: int | None = None
    for raw in cast("list[Any]", contents):
        item = _mapping(raw)
        if item is None:
            continue
        kind = item.get("type")
        if kind in ("text", "terminal"):
            text = _str(item.get("text"))
            if text:
                rendered.append(text)
            if kind == "terminal":
                exit_code = _int_or_none(item.get("exitCode"))
        elif kind == "resource_link":
            link = _resource_link(item)
            if link is not None:
                rendered.append(link)
    return rendered, exit_code


def _resource_link(item: Mapping[str, Any]) -> str | None:
    uri = _str_or_none(item.get("uri"))
    if uri is None:
        return None
    name = _str_or_none(item.get("title")) or _str_or_none(item.get("name")) or uri
    return f"{name}: {uri}"


def _str(value: object) -> str:
    return value if isinstance(value, str) else ""


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _mapping(value: object) -> Mapping[str, Any] | None:
    if isinstance(value, dict):
        return cast("Mapping[str, Any]", value)
    return None
