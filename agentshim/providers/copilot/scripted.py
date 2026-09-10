"""Canned Copilot CLI JSON lines for tests.

Looked up by ``agentshim.testing.scripted_turn`` so a consumer can script a
turn without knowing Copilot's wire format. The lines are the real format,
modelled on recorded runs: they round-trip through the real parser.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from agentshim.core.usage import TokenUsage


def scripted_lines(
    *,
    text: str = "",
    session_id: str | None = None,
    usage: TokenUsage | None = None,
    tool_calls: Sequence[tuple[str, Mapping[str, Any], str]] = (),
    structured_output: object = None,
) -> list[str]:
    """Build the stdout of one Copilot turn.

    ``tool_calls`` entries are ``(tool, args, output)`` and become a
    ``tool.execution_start`` frame plus its ``tool.execution_complete``.
    ``structured_output`` is ignored: Copilot has no output-schema mode, so
    a scripted turn cannot pretend to produce one.
    """
    del structured_output
    lines: list[str] = []
    for index, (tool, args, output) in enumerate(tool_calls):
        call_id = f"tool-{index}"
        lines.append(
            _line(
                {
                    "type": "tool.execution_start",
                    "data": {"toolCallId": call_id, "toolName": tool, "arguments": dict(args)},
                }
            )
        )
        lines.append(
            _line(
                {
                    "type": "tool.execution_complete",
                    "data": {"toolCallId": call_id, "success": True, "result": {"content": output}},
                }
            )
        )

    output_tokens = (
        max(usage.output_tokens - usage.reasoning_output_tokens, 0) if usage is not None else 0
    )
    lines.append(
        _line(
            {
                "type": "assistant.message",
                "data": {
                    "messageId": "msg-0",
                    "content": text,
                    "toolRequests": [],
                    "outputTokens": output_tokens,
                },
            }
        )
    )
    if usage is not None:
        lines.append(
            _line({"type": "assistant.usage", "ephemeral": True, "data": _usage_payload(usage)})
        )

    turns = usage.turns if usage is not None else 1
    lines.extend(
        _line({"type": "assistant.turn_end", "data": {"turnId": str(turn)}})
        for turn in range(turns)
    )

    result: dict[str, Any] = {"type": "result", "exitCode": 0, "usage": {"premiumRequests": 0}}
    if session_id is not None:
        result["sessionId"] = session_id
    lines.append(_line(result))
    return lines


def _usage_payload(usage: TokenUsage) -> dict[str, Any]:
    """Undo the parser's folding to print the disjoint counts Copilot reports."""
    cached = usage.cached_input_tokens
    written = usage.cache_write_input_tokens
    reasoning = usage.reasoning_output_tokens
    return {
        "model": "gpt-5",
        "inputTokens": max(usage.input_tokens - cached, 0),
        "outputTokens": max(usage.output_tokens - reasoning, 0),
        "cacheReadTokens": max(cached - written, 0),
        "cacheWriteTokens": written,
        "reasoningTokens": reasoning,
    }


def _line(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload) + "\n"
