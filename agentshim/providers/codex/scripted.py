"""Canned Codex ``--json`` lines for tests.

Looked up by ``agentshim.testing.scripted_turn`` so a consumer can script a
turn without knowing Codex's wire format. The lines are the real format:
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
    """Build the stdout of one Codex turn.

    ``tool_calls`` entries are ``(tool, args, output)``. Codex reports every
    shell command as a ``command_execution`` item, so ``args`` may carry a
    ``command``; without one the tool name stands in as the command.
    ``structured_output`` replaces the final message, which is where Codex
    puts a schema-conformant payload.

    Args:
        text: The assistant's final message.
        session_id: Thread id announced by ``thread.started``.
        usage: Token counts reported by ``turn.completed``.
        tool_calls: Commands to report as started-then-completed items.
        structured_output: Payload to serialize as the final message.

    Returns:
        One newline-terminated JSON line per Codex event.
    """
    lines: list[str] = []
    if session_id is not None:
        lines.append(_line({"type": "thread.started", "thread_id": session_id}))
    lines.append(_line({"type": "turn.started"}))

    for index, (tool, args, output) in enumerate(tool_calls):
        item_id = f"item_{index}"
        command = args.get("command")
        item: dict[str, Any] = {
            "id": item_id,
            "type": "command_execution",
            "command": command if isinstance(command, str) else tool,
        }
        lines.append(_line({"type": "item.started", "item": {**item, "status": "in_progress"}}))
        lines.append(
            _line(
                {
                    "type": "item.completed",
                    "item": {
                        **item,
                        "status": "completed",
                        "aggregated_output": output,
                        "exit_code": 0,
                    },
                }
            )
        )

    message = json.dumps(structured_output) if structured_output is not None else text
    if message:
        lines.append(
            _line(
                {
                    "type": "item.completed",
                    "item": {"id": "item_msg", "type": "agent_message", "text": message},
                }
            )
        )

    lines.append(_line({"type": "turn.completed", "usage": _usage_payload(usage)}))
    return lines


def _usage_payload(usage: TokenUsage | None) -> dict[str, int]:
    if usage is None:
        return {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
    # Codex's input_tokens already includes the cached prefix, so the counts
    # go out exactly as the parser will read them back.
    return {
        "input_tokens": usage.input_tokens,
        "cached_input_tokens": usage.cached_input_tokens,
        "output_tokens": usage.output_tokens,
    }


def _line(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload) + "\n"
