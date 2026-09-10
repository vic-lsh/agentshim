"""Canned opencode JSON event lines for tests.

Looked up by ``agentshim.testing.scripted_turn`` so a consumer can script a
turn without knowing opencode's wire format. The lines are the real format:
they round-trip through the real parser.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from agentshim.core.usage import TokenUsage

#: Fixed so a scripted run is byte-identical between invocations.
TIMESTAMP = 1789020428786

MESSAGE_ID = "msg_scripted"

_NO_STRUCTURED_OUTPUT = (
    "opencode has no native output schema; scripted_turn cannot produce structured output"
)


def scripted_lines(
    *,
    text: str = "",
    session_id: str | None = None,
    usage: TokenUsage | None = None,
    tool_calls: Sequence[tuple[str, Mapping[str, Any], str]] = (),
    structured_output: object | None = None,
) -> list[str]:
    """Build the stdout of one opencode turn.

    ``tool_calls`` entries are ``(tool, args, output)`` and become one
    completed ``tool_use`` frame each, which is how opencode publishes a
    tool part: only once it has finished.

    Args:
        text: The assistant reply, emitted as one finished text part.
        session_id: Conversation id echoed on every frame; omitted when None.
        usage: Token counts to report in the closing ``step_finish`` frame.
        tool_calls: Tool calls to script.
        structured_output: Must be None; opencode has no native output schema.

    Returns:
        The stdout lines, each a JSON object with a trailing newline.

    Raises:
        ValueError: If ``structured_output`` is given.
    """
    if structured_output is not None:
        raise ValueError(_NO_STRUCTURED_OUTPUT)

    lines = [_line("step_start", session_id, {"type": "step-start"}, index=0)]
    if text:
        lines.append(
            _line(
                "text",
                session_id,
                {"type": "text", "text": text, "time": {"start": TIMESTAMP, "end": TIMESTAMP}},
                index=1,
            )
        )
    for index, (tool, args, output) in enumerate(tool_calls):
        lines.append(
            _line(
                "tool_use",
                session_id,
                {
                    "type": "tool",
                    "callID": f"call_{index}",
                    "tool": tool,
                    "state": {
                        "status": "completed",
                        "input": dict(args),
                        "output": output,
                        "title": tool,
                        "metadata": {},
                        "time": {"start": TIMESTAMP, "end": TIMESTAMP + 1000},
                    },
                },
                index=index + 2,
            )
        )
    lines.append(
        _line(
            "step_finish",
            session_id,
            {"type": "step-finish", "reason": "stop", "cost": 0.01, "tokens": _tokens(usage)},
            index=len(tool_calls) + 2,
        )
    )
    return lines


def _tokens(usage: TokenUsage | None) -> dict[str, Any]:
    """Render token counts the way a ``step-finish`` part carries them.

    The parser adds the cache hits back into ``input`` and folds
    ``reasoning`` into the output total, so both are undone here to produce
    the disjoint counts opencode actually prints.
    """
    if usage is None:
        return {
            "total": 0,
            "input": 0,
            "output": 0,
            "reasoning": 0,
            "cache": {"read": 0, "write": 0},
        }
    write = usage.cache_write_input_tokens
    read = max(usage.cached_input_tokens - write, 0)
    reasoning = usage.reasoning_output_tokens
    return {
        "total": usage.input_tokens + usage.output_tokens,
        "input": max(usage.input_tokens - usage.cached_input_tokens, 0),
        "output": max(usage.output_tokens - reasoning, 0),
        "reasoning": reasoning,
        "cache": {"read": read, "write": write},
    }


def _line(kind: str, session_id: str | None, part: Mapping[str, Any], *, index: int) -> str:
    payload: dict[str, Any] = {"type": kind, "timestamp": TIMESTAMP}
    body = {"id": f"prt_{index}", "messageID": MESSAGE_ID, **part}
    if session_id is not None:
        payload["sessionID"] = session_id
        body = {**body, "sessionID": session_id}
    payload["part"] = body
    return json.dumps(payload) + "\n"
