"""Typed view of Gemini CLI's ``stream-json`` frames.

Kept separate from the parser so the frame shapes are testable on their own
and so a change in Gemini's wire format has one place to land. The frames
mirror ``JsonStreamEvent`` in ``@google/gemini-cli-core`` 0.26.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


@dataclass(frozen=True)
class InitEvent:
    """``{"type":"init",...}``: names the conversation and the model."""

    session_id: str | None
    model: str | None = None


@dataclass(frozen=True)
class MessageEvent:
    """One ``message`` frame.

    The CLI echoes the prompt as a ``user`` frame and streams the model's
    reply as ``assistant`` frames with ``delta: true``, one per chunk.
    """

    role: str
    content: str
    delta: bool = False


@dataclass(frozen=True)
class ToolUseEvent:
    """A tool call the model requested."""

    tool_id: str | None
    tool: str
    parameters: Mapping[str, Any] | str | None


@dataclass(frozen=True)
class ToolResultEvent:
    """The outcome of one tool call."""

    tool_id: str | None
    status: str | None
    output: str
    error_message: str | None = None


@dataclass(frozen=True)
class ErrorEvent:
    """A problem the CLI reported mid-stream, with its severity."""

    severity: str
    message: str


@dataclass(frozen=True)
class ResultEvent:
    """The terminal frame: status, the fatal error if any, and stats."""

    status: str = "success"
    error_message: str | None = None
    stats: Mapping[str, Any] | None = None


GeminiFrame = InitEvent | MessageEvent | ToolUseEvent | ToolResultEvent | ErrorEvent | ResultEvent


def parse_frame(data: Mapping[str, Any]) -> GeminiFrame | None:
    """Map one decoded stdout object to a frame, or ``None`` if uninteresting."""
    builder = _BUILDERS.get(_text(data.get("type")))
    return None if builder is None else builder(data)


def _init(data: Mapping[str, Any]) -> GeminiFrame:
    return InitEvent(
        session_id=_str_or_none(data.get("session_id")), model=_str_or_none(data.get("model"))
    )


def _message(data: Mapping[str, Any]) -> GeminiFrame:
    return MessageEvent(
        role=_text(data.get("role")),
        content=_text(data.get("content")),
        delta=bool(data.get("delta")),
    )


def _tool_use(data: Mapping[str, Any]) -> GeminiFrame:
    parameters = data.get("parameters")
    return ToolUseEvent(
        tool_id=_str_or_none(data.get("tool_id")),
        tool=_text(data.get("tool_name")) or "Tool",
        parameters=cast("Mapping[str, Any]", parameters)
        if isinstance(parameters, dict)
        else _str_or_none(parameters),
    )


def _tool_result(data: Mapping[str, Any]) -> GeminiFrame:
    return ToolResultEvent(
        tool_id=_str_or_none(data.get("tool_id")),
        status=_str_or_none(data.get("status")),
        output=_text(data.get("output")),
        error_message=_error_message(data.get("error")),
    )


def _error(data: Mapping[str, Any]) -> GeminiFrame:
    return ErrorEvent(
        severity=_text(data.get("severity")) or "error", message=_text(data.get("message"))
    )


def _result(data: Mapping[str, Any]) -> GeminiFrame:
    return ResultEvent(
        status=_text(data.get("status")) or "success",
        error_message=_error_message(data.get("error")),
        stats=_mapping_or_none(data.get("stats")),
    )


_BUILDERS: dict[str, Callable[[Mapping[str, Any]], GeminiFrame]] = {
    "init": _init,
    "message": _message,
    "tool_use": _tool_use,
    "tool_result": _tool_result,
    "error": _error,
    "result": _result,
}


def _error_message(value: object) -> str | None:
    """Pull the human-readable message out of a Gemini error object."""
    if not isinstance(value, dict):
        return None
    message = cast("dict[str, Any]", value).get("message")
    return message if isinstance(message, str) and message else None


def _text(value: object) -> str:
    return value if isinstance(value, str) else ("" if value is None else str(value))


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _mapping_or_none(value: object) -> Mapping[str, Any] | None:
    if isinstance(value, dict):
        return cast("Mapping[str, Any]", value)
    return None
