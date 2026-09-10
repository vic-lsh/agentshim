"""Canned Claude Code stream-json lines for tests.

Looked up by ``agentshim.testing.scripted_turn`` so a consumer can script a
turn without knowing Claude's wire format. The lines are the real format:
they round-trip through the real parser.
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
    structured_output: object | None = None,
) -> list[str]:
    """Build the stdout of one Claude turn.

    ``tool_calls`` entries are ``(tool, args, output)`` and become a
    ``tool_use`` block plus the matching ``tool_result`` frame.
    """
    lines: list[str] = []
    if session_id is not None:
        lines.append(_line({"type": "system", "subtype": "init", "session_id": session_id}))

    blocks: list[dict[str, Any]] = []
    if text:
        blocks.append({"type": "text", "text": text})
    for index, (tool, args, _output) in enumerate(tool_calls):
        blocks.append(
            {"type": "tool_use", "id": f"toolu_{index}", "name": tool, "input": dict(args)}
        )
    if blocks:
        message: dict[str, Any] = {"role": "assistant", "content": blocks}
        lines.append(_line({"type": "assistant", "message": message}))

    for index, (_tool, _args, output) in enumerate(tool_calls):
        lines.append(
            _line(
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": f"toolu_{index}",
                                "content": output,
                            }
                        ],
                    },
                }
            )
        )

    result: dict[str, Any] = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": text,
        "num_turns": usage.turns if usage is not None else 1,
        "duration_ms": 1234,
        "total_cost_usd": 0.01,
        "usage": _usage_payload(usage),
    }
    if session_id is not None:
        result["session_id"] = session_id
    if structured_output is not None:
        result["structured_output"] = structured_output
    lines.append(_line(result))
    return lines


def resume_failure_lines(*, session_id: str | None = None) -> tuple[list[str], list[str], int]:
    """Build the stdout, stderr and exit code of a resumed turn Claude cannot continue.

    ``ClaudeProvider.classify_exit`` treats any nonzero exit of a resumed
    turn as ``SessionResumeError``: ``claude --resume`` gives no
    distinguishable exit code for a missing transcript, so a bare failure on
    stderr is enough to trigger it. ``session_id`` is folded into the message
    for realism only; the id ``SessionResumeError`` actually reports comes
    from the resumed turn's own argv, not from anything scripted here.

    Args:
        session_id: Conversation id to name in the scripted stderr message.

    Returns:
        ``(stdout, stderr, returncode)`` for a ``FakeRun``.
    """
    detail = f" {session_id}" if session_id else ""
    stderr = [f"Error: no conversation found to resume{detail}\n"]
    return [], stderr, 1


def _usage_payload(usage: TokenUsage | None) -> dict[str, int]:
    if usage is None:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
    # The parser folds cache tokens into input_tokens, so undo that here to
    # produce the disjoint counts Claude actually prints.
    cached = usage.cached_input_tokens
    created = usage.cache_write_input_tokens
    return {
        "input_tokens": max(usage.input_tokens - cached, 0),
        "output_tokens": usage.output_tokens,
        "cache_creation_input_tokens": created,
        "cache_read_input_tokens": max(cached - created, 0),
    }


def _line(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload) + "\n"
