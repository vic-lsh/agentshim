"""Typed view of Claude Code's ``stream-json`` frames.

Kept separate from the parser so the frame shapes are testable on their own
and so a change in Claude's wire format has one place to land.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@dataclass(frozen=True)
class SystemInit:
    """``{"type":"system","subtype":"init",...}``: names the conversation."""

    session_id: str | None
    subtype: str | None


@dataclass(frozen=True)
class TextBlock:
    """A ``text`` content block: prose the assistant addressed to the user."""

    text: str


@dataclass(frozen=True)
class ThinkingBlock:
    """A ``thinking`` content block: reasoning the run chose to expose."""

    text: str


@dataclass(frozen=True)
class ToolUseBlock:
    """A ``tool_use`` content block: one tool call and the arguments it got.

    ``args`` is whatever the frame carried: a decoded object, an unparsed
    string while Claude is still streaming the arguments, or nothing.
    """

    tool_id: str | None
    tool: str
    args: Mapping[str, Any] | str | None


ContentBlock = TextBlock | ThinkingBlock | ToolUseBlock


@dataclass(frozen=True)
class AssistantMessage:
    """One assistant frame: its content blocks and its incremental usage."""

    blocks: Sequence[ContentBlock] = ()
    usage: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ToolResultBlock:
    """A ``user`` frame carrying the output of one tool call."""

    tool_id: str | None
    output: str
    is_error: bool


@dataclass(frozen=True)
class ResultFrame:
    """The terminal frame: final text, structured payload, usage, cost."""

    text: str = ""
    structured_output: Any | None = None
    num_turns: int | None = None
    usage: Mapping[str, Any] | None = None
    total_cost_usd: float | None = None
    duration_ms: int | None = None
    is_error: bool = False
    subtype: str | None = None


ClaudeFrame = SystemInit | AssistantMessage | ToolResultBlock | ResultFrame


def parse_frame(data: Mapping[str, Any]) -> ClaudeFrame | None:
    """Map one decoded stdout object to a frame, or ``None`` if uninteresting."""
    kind = data.get("type")
    if kind == "system":
        return SystemInit(
            session_id=_str_or_none(data.get("session_id")),
            subtype=_str_or_none(data.get("subtype")),
        )
    if kind == "assistant":
        return _assistant(data)
    if kind == "user":
        return _tool_result(data)
    if kind == "result":
        return ResultFrame(
            text=_text(data.get("result")),
            structured_output=data.get("structured_output"),
            num_turns=_int_or_none(data.get("num_turns")),
            usage=_mapping_or_none(data.get("usage")),
            total_cost_usd=_float_or_none(data.get("total_cost_usd")),
            duration_ms=_int_or_none(data.get("duration_ms")),
            is_error=bool(data.get("is_error")),
            subtype=_str_or_none(data.get("subtype")),
        )
    return None


def _assistant(data: Mapping[str, Any]) -> AssistantMessage | None:
    message = data.get("message")
    if not isinstance(message, dict):
        return None
    payload = cast("dict[str, Any]", message)
    raw_blocks = payload.get("content")
    blocks: list[ContentBlock] = []
    if isinstance(raw_blocks, list):
        for raw in cast("list[Any]", raw_blocks):
            block = _content_block(raw)
            if block is not None:
                blocks.append(block)
    usage = _mapping_or_none(payload.get("usage"))
    if not blocks and usage is None:
        return None
    return AssistantMessage(blocks=tuple(blocks), usage=usage)


def _content_block(raw: object) -> ContentBlock | None:
    if not isinstance(raw, dict):
        return None
    block = cast("dict[str, Any]", raw)
    kind = block.get("type")
    if kind == "text":
        return TextBlock(_text(block.get("text")))
    if kind == "thinking":
        return ThinkingBlock(_text(block.get("thinking")))
    if kind == "tool_use":
        args = block.get("input")
        return ToolUseBlock(
            tool_id=_str_or_none(block.get("id")),
            tool=_text(block.get("name")) or "Tool",
            args=cast("Mapping[str, Any]", args) if isinstance(args, dict) else _str_or_none(args),
        )
    return None


def _tool_result(data: Mapping[str, Any]) -> ToolResultBlock | None:
    message = data.get("message")
    if not isinstance(message, dict):
        return None
    raw_blocks = cast("dict[str, Any]", message).get("content")
    if not isinstance(raw_blocks, list):
        return None
    for raw in cast("list[Any]", raw_blocks):
        if not isinstance(raw, dict):
            continue
        block = cast("dict[str, Any]", raw)
        if block.get("type") != "tool_result":
            continue
        return ToolResultBlock(
            tool_id=_str_or_none(block.get("tool_use_id")),
            output=_tool_output(block.get("content")),
            is_error=bool(block.get("is_error")),
        )
    return None


def _tool_output(value: object) -> str:
    """Flatten a tool_result payload, which may be a string or a block list.

    Anything but plain text arrives as a list of content blocks. Rendering
    those with ``str()`` prints Python dict reprs, single quotes and ``True``
    included, into a result a caller displays or re-parses, so text blocks
    contribute their text and every other block is serialized as JSON.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_output_block(item) for item in cast("list[object]", value))
    return _output_block(value)


def _output_block(item: object) -> str:
    """Render one content block of a tool result."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        block = cast("dict[str, Any]", item)
        if block.get("type") == "text":
            return _text(block.get("text"))
    # ``default=str`` because a block may carry something JSON cannot encode
    # and a tool result is never worth failing the turn over.
    return json.dumps(item, ensure_ascii=False, default=str)


def _text(value: object) -> str:
    return value if isinstance(value, str) else ("" if value is None else str(value))


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


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
