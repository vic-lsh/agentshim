"""Typed view of opencode's ``run --format json`` frames.

Kept separate from the parser so the frame shapes are testable on their own
and so a change in opencode's wire format has one place to land. Every line
is ``{"type": ..., "timestamp": ..., "sessionID": ..., "part": {...}}``,
where the event type is the message part's own type with ``-`` replaced by
``_``; the exception is ``error``, which carries ``error`` instead of
``part``. The shapes mirror ``MessageV2`` in opencode 1.2.18.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


@dataclass(frozen=True)
class TextEvent:
    """One finished assistant text part."""

    text: str


@dataclass(frozen=True)
class ReasoningEvent:
    """One finished reasoning part; only emitted under ``--thinking``."""

    text: str


@dataclass(frozen=True)
class ToolEvent:
    """A tool part in a terminal state: the call and its outcome in one frame."""

    tool_id: str | None
    tool: str
    status: str | None
    args: Mapping[str, Any] | str | None
    output: str
    error_message: str | None = None
    duration_s: float | None = None


@dataclass(frozen=True)
class StepStartEvent:
    """A model round is starting. Carries nothing agentshim reports."""


@dataclass(frozen=True)
class StepFinishEvent:
    """A model round finished: its stop reason, cost and token counts."""

    reason: str | None = None
    cost: float | None = None
    tokens: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ErrorEvent:
    """A ``session.error`` the CLI forwarded."""

    message: str


OpencodeFrame = (
    TextEvent | ReasoningEvent | ToolEvent | StepStartEvent | StepFinishEvent | ErrorEvent
)

#: Tool states opencode emits; it publishes a tool part only once it is done.
TERMINAL_TOOL_STATES = ("completed", "success", "error")


def session_id_of(data: Mapping[str, Any]) -> str | None:
    """Return the conversation id every frame carries at the top level."""
    session_id = data.get("sessionID")
    return session_id if isinstance(session_id, str) and session_id else None


def parse_frame(data: Mapping[str, Any]) -> OpencodeFrame | None:
    """Map one decoded stdout object to a frame, or ``None`` if uninteresting."""
    builder = _BUILDERS.get(_text(data.get("type")))
    return None if builder is None else builder(data)


def _part(data: Mapping[str, Any]) -> Mapping[str, Any]:
    part = data.get("part")
    return cast("Mapping[str, Any]", part) if isinstance(part, dict) else {}


def _text_frame(data: Mapping[str, Any]) -> OpencodeFrame:
    return TextEvent(text=_text(_part(data).get("text")))


def _reasoning(data: Mapping[str, Any]) -> OpencodeFrame:
    return ReasoningEvent(text=_text(_part(data).get("text")))


def _tool(data: Mapping[str, Any]) -> OpencodeFrame | None:
    part = _part(data)
    state = part.get("state")
    if not isinstance(state, dict):
        return None
    fields = cast("Mapping[str, Any]", state)
    status = _str_or_none(fields.get("status"))
    if status not in TERMINAL_TOOL_STATES:
        return None
    args = fields.get("input")
    error = _str_or_none(fields.get("error"))
    return ToolEvent(
        tool_id=_str_or_none(part.get("callID")) or _str_or_none(part.get("id")),
        tool=_text(part.get("tool")) or "Tool",
        status=status,
        args=cast("Mapping[str, Any]", args) if isinstance(args, dict) else _str_or_none(args),
        output=_text(fields.get("output")),
        error_message=error or None,
        duration_s=_elapsed(fields.get("time")),
    )


def _step_start(data: Mapping[str, Any]) -> OpencodeFrame:
    del data
    return StepStartEvent()


def _step_finish(data: Mapping[str, Any]) -> OpencodeFrame:
    part = _part(data)
    return StepFinishEvent(
        reason=_str_or_none(part.get("reason")),
        cost=_float_or_none(part.get("cost")),
        tokens=_mapping_or_none(part.get("tokens")),
    )


def _error(data: Mapping[str, Any]) -> OpencodeFrame:
    error = _mapping_or_none(data.get("error")) or {}
    payload = _mapping_or_none(error.get("data")) or {}
    message = _str_or_none(payload.get("message")) or _str_or_none(error.get("name"))
    return ErrorEvent(message=message or "opencode reported an error")


_BUILDERS: dict[str, Callable[[Mapping[str, Any]], OpencodeFrame | None]] = {
    "text": _text_frame,
    "reasoning": _reasoning,
    "tool_use": _tool,
    "step_start": _step_start,
    "step_finish": _step_finish,
    "error": _error,
}


def _elapsed(value: object) -> float | None:
    """Convert a part's ``{"start": ms, "end": ms}`` block to seconds."""
    if not isinstance(value, dict):
        return None
    times = cast("Mapping[str, Any]", value)
    start = _float_or_none(times.get("start"))
    end = _float_or_none(times.get("end"))
    if start is None or end is None:
        return None
    return max(end - start, 0.0) / 1000.0


def _text(value: object) -> str:
    return value if isinstance(value, str) else ("" if value is None else str(value))


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _mapping_or_none(value: object) -> Mapping[str, Any] | None:
    if isinstance(value, dict):
        return cast("Mapping[str, Any]", value)
    return None
