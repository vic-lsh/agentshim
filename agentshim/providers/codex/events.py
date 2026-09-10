"""Typed view of Codex's ``codex exec --json`` frames.

Kept separate from the parser so the frame shapes are testable on their own
and so a change in Codex's wire format has one place to land.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Mapping

#: Item keys that describe the envelope rather than the call's arguments.
_ENVELOPE_KEYS = frozenset({"id", "type", "status"})

#: Where a completed generic item hides its output, most specific first.
_OUTPUT_KEYS = ("aggregated_output", "output", "text", "summary", "result", "error", "message")


@dataclass(frozen=True)
class ThreadStarted:
    """``{"type":"thread.started",...}``: names the resumable conversation."""

    thread_id: str | None


@dataclass(frozen=True)
class TurnStarted:
    """``{"type":"turn.started"}``: the model began working."""


@dataclass(frozen=True)
class TurnCompleted:
    """``{"type":"turn.completed","usage":{...}}``: one turn's token counts.

    Codex's ``input_tokens`` already includes ``cached_input_tokens``, so the
    two are never added together.
    """

    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    usage: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ErrorFrame:
    """``turn.failed`` or a top-level ``error``: the turn did not succeed."""

    message: str


@dataclass(frozen=True)
class CommandItem:
    """A ``command_execution`` item: a shell command Codex ran."""

    item_id: str | None
    command: str
    output: str = ""
    exit_code: int | None = None
    status: str | None = None


@dataclass(frozen=True)
class MessageItem:
    """An ``agent_message`` item: assistant-facing text."""

    item_id: str | None
    text: str


@dataclass(frozen=True)
class ReasoningItem:
    """A ``reasoning`` item: the model's thinking text."""

    item_id: str | None
    text: str


@dataclass(frozen=True)
class GenericItem:
    """Any other item (``mcp_tool_call``, ``file_change``, ``web_search``, ...).

    Codex keeps adding item types, so the parser treats an unknown one as a
    tool call whose arguments are whatever the item carried.
    """

    item_id: str | None
    kind: str
    fields: Mapping[str, Any]
    status: str | None = None


CodexItem = CommandItem | MessageItem | ReasoningItem | GenericItem


@dataclass(frozen=True)
class ItemStarted:
    """``{"type":"item.started","item":{...}}``: an item began."""

    item: CodexItem


@dataclass(frozen=True)
class ItemCompleted:
    """``{"type":"item.completed","item":{...}}``: an item finished."""

    item: CodexItem


CodexFrame = ThreadStarted | TurnStarted | TurnCompleted | ErrorFrame | ItemStarted | ItemCompleted


def parse_frame(data: Mapping[str, Any]) -> CodexFrame | None:
    """Map one decoded stdout object to a frame, or ``None`` if uninteresting."""
    kind = data.get("type")
    if kind in ("item.started", "item.completed"):
        item = _item(data.get("item"))
        if item is None:
            return None
        return ItemStarted(item) if kind == "item.started" else ItemCompleted(item)
    return _lifecycle_frame(kind, data)


def _lifecycle_frame(kind: object, data: Mapping[str, Any]) -> CodexFrame | None:
    if kind == "thread.started":
        return ThreadStarted(thread_id=_str_or_none(data.get("thread_id")))
    if kind == "turn.started":
        return TurnStarted()
    if kind == "turn.completed":
        return _turn_completed(data.get("usage"))
    if kind == "turn.failed":
        return ErrorFrame(_failure_message(data.get("error")))
    if kind == "error":
        return ErrorFrame(_text(data.get("message")))
    return None


def summarize_item(item: GenericItem) -> str:
    """Return the output a completed generic item carried, if any.

    Codex reports a generic item's result under a different key per item
    type, and a failed one reports an ``error`` payload instead, so the
    first populated key wins and non-strings are rendered as JSON.
    """
    for key in _OUTPUT_KEYS:
        value = item.fields.get(key)
        if value:
            return value if isinstance(value, str) else _json(value)
    return ""


def _json(value: object) -> str:
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)


def _turn_completed(usage_raw: object) -> TurnCompleted:
    usage = cast("Mapping[str, Any]", usage_raw) if isinstance(usage_raw, dict) else None
    fields: Mapping[str, Any] = usage if usage is not None else {}
    return TurnCompleted(
        input_tokens=_int(fields.get("input_tokens")),
        cached_input_tokens=_int(fields.get("cached_input_tokens")),
        output_tokens=_int(fields.get("output_tokens")),
        usage=usage,
    )


def _failure_message(error: object) -> str:
    if isinstance(error, dict):
        return _text(cast("Mapping[str, Any]", error).get("message"))
    return _text(error)


def _item(raw: object) -> CodexItem | None:
    if not isinstance(raw, dict):
        return None
    item = cast("dict[str, Any]", raw)
    kind = _str_or_none(item.get("type"))
    item_id = _str_or_none(item.get("id"))
    status = _str_or_none(item.get("status"))
    if kind == "agent_message":
        return MessageItem(item_id=item_id, text=_text(item.get("text")))
    if kind == "reasoning":
        return ReasoningItem(item_id=item_id, text=_text(item.get("text")))
    if kind == "command_execution":
        return CommandItem(
            item_id=item_id,
            command=_text(item.get("command")),
            output=_text(item.get("aggregated_output")),
            exit_code=_int_or_none(item.get("exit_code")),
            status=status,
        )
    return GenericItem(
        item_id=item_id,
        kind=kind or "item",
        fields={key: value for key, value in item.items() if key not in _ENVELOPE_KEYS},
        status=status,
    )


def _text(value: object) -> str:
    if isinstance(value, str):
        return value
    return "" if value is None else str(value)


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
