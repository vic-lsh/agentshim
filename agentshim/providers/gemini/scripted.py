"""Canned Gemini CLI stream-json lines for tests.

Looked up by ``agentshim.testing.scripted_turn`` so a consumer can script a
turn without knowing Gemini's wire format. The lines are the real format:
they round-trip through the real parser.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from agentshim.core.usage import TokenUsage

#: Fixed so a scripted run is byte-identical between invocations.
TIMESTAMP = "2026-01-01T00:00:00.000Z"

MODEL = "gemini-2.5-pro"

_NO_STRUCTURED_OUTPUT = (
    "gemini has no native output schema; scripted_turn cannot produce structured output"
)


def scripted_lines(
    *,
    text: str = "",
    session_id: str | None = None,
    usage: TokenUsage | None = None,
    tool_calls: Sequence[tuple[str, Mapping[str, Any], str]] = (),
    structured_output: object | None = None,
) -> list[str]:
    """Build the stdout of one Gemini turn.

    ``tool_calls`` entries are ``(tool, args, output)`` and become a
    ``tool_use`` frame plus the matching ``tool_result`` frame, in the order
    the CLI emits them: every call of a round first, then every result.

    Args:
        text: The assistant reply, streamed as one delta frame.
        session_id: Conversation id for the ``init`` frame; omitted when None.
        usage: Token counts to report in the terminal ``result`` frame.
        tool_calls: Tool calls to script.
        structured_output: Must be None; Gemini has no native output schema.

    Returns:
        The stdout lines, each a JSON object with a trailing newline.

    Raises:
        ValueError: If ``structured_output`` is given.
    """
    if structured_output is not None:
        raise ValueError(_NO_STRUCTURED_OUTPUT)

    lines: list[str] = []
    if session_id is not None:
        lines.append(_line({"type": "init", "session_id": session_id, "model": MODEL}))
    if text:
        lines.append(
            _line({"type": "message", "role": "assistant", "content": text, "delta": True})
        )
    for index, (tool, args, _output) in enumerate(tool_calls):
        lines.append(
            _line(
                {
                    "type": "tool_use",
                    "tool_name": tool,
                    "tool_id": f"call_{index}",
                    "parameters": dict(args),
                }
            )
        )
    for index, (_tool, _args, output) in enumerate(tool_calls):
        lines.append(
            _line(
                {
                    "type": "tool_result",
                    "tool_id": f"call_{index}",
                    "status": "success",
                    "output": output,
                }
            )
        )
    lines.append(
        _line({"type": "result", "status": "success", "stats": _stats(usage, len(tool_calls))})
    )
    return lines


def _stats(usage: TokenUsage | None, tool_calls: int) -> dict[str, int]:
    """Render token counts the way ``convertToStreamStats`` does.

    ``cached`` is the part of ``input_tokens`` served from the context cache,
    and ``input`` is the remainder, so the two are nested and not disjoint.
    """
    if usage is None:
        return {
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cached": 0,
            "input": 0,
            "duration_ms": 0,
            "tool_calls": tool_calls,
        }
    return {
        "total_tokens": usage.input_tokens + usage.output_tokens,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cached": usage.cached_input_tokens,
        "input": max(usage.input_tokens - usage.cached_input_tokens, 0),
        "duration_ms": 1234,
        "tool_calls": tool_calls,
    }


def _line(payload: dict[str, Any]) -> str:
    return json.dumps({**payload, "timestamp": TIMESTAMP}) + "\n"
