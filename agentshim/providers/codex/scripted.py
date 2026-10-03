"""Canned Codex ``--json`` lines for tests.

Looked up by ``agentshim.testing.scripted_turn`` so a consumer can script a
turn without knowing Codex's wire format. The lines are the real format:
they round-trip through the real parser.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from agentshim.core.errors import FailureKind

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from agentshim.core.usage import TokenUsage


# Mirrors scripted_turn: each option is an independent knob of the test double.
def scripted_lines(  # noqa: PLR0913
    *,
    text: str = "",
    session_id: str | None = None,
    usage: TokenUsage | None = None,
    tool_calls: Sequence[tuple[str, Mapping[str, Any], str]] = (),
    structured_output: object | None = None,
    skills_offered: Sequence[str] | None = None,
    skills_invoked: Sequence[str] = (),
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
        skills_offered: Must be None; the stream lists no offered skills.
        skills_invoked: Skills to load first, each as a shell read of its
            ``SKILL.md``.

    Returns:
        One newline-terminated JSON line per Codex event.
    """
    tool_calls = [
        *(
            ("execute", {"command": f"cat .agents/skills/{name}/SKILL.md"}, "")
            for name in skills_invoked
        ),
        *tool_calls,
    ]
    if skills_offered is not None:
        msg = "Codex does not list offered skills in its stream"
        raise ValueError(msg)
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


def resume_failure_lines(*, session_id: str | None = None) -> tuple[list[str], list[str], int]:
    """Build the stdout, stderr and exit code of a resumed turn with no rollout.

    ``CodexProvider.classify_exit`` is the one provider that names its
    failure explicitly: it matches "thread/resume failed" and "no rollout
    found" on stderr, so unlike the other providers a plain nonzero exit is
    not enough here; the message has to say so.

    Args:
        session_id: Thread id to name in the scripted stderr message.

    Returns:
        ``(stdout, stderr, returncode)`` for a ``FakeRun``.
    """
    thread = session_id or "unknown-thread"
    stderr = [f"thread/resume failed: no rollout found for thread {thread}\n"]
    return [], stderr, 1


#: The message Codex reports for a failed turn of each kind.
_FAILURES: dict[FailureKind, str] = {
    FailureKind.TRANSIENT: "exceeded retry limit, last status: 500 Internal Server Error",
    FailureKind.USAGE_LIMIT: (
        "You've hit your usage limit. Upgrade to Plus to continue using Codex "
        "(https://chatgpt.com/explore/plus), or try again in 2 hours."
    ),
    FailureKind.AUTH: "unexpected status 401 Unauthorized: Missing bearer authentication",
    FailureKind.OTHER: "Codex ran out of room in the model's context window.",
}


#: Codex sends ``--output-schema`` to the model as a strict response format,
#: so decoding is constrained to the schema and a turn never fails over it.
_NO_SCHEMA_FAILURE = (
    "codex never fails a turn over its output schema: decoding is constrained to the schema"
)


def failure_lines(
    kind: FailureKind, *, session_id: str | None = None
) -> tuple[list[str], list[str], int]:
    """Build the stdout, stderr and exit code of a turn that fails with *kind*.

    Codex reports the failure as an ``error`` event followed by
    ``turn.failed``, both carrying the same message. ``SCHEMA`` raises
    ``ValueError``: Codex has no such failure.

    Args:
        kind: The classification the scripted failure must produce.
        session_id: Thread id ``thread.started`` names, if any.

    Returns:
        ``(stdout, stderr, returncode)`` for a ``FakeRun``.
    """
    message = _FAILURES.get(kind)
    if message is None:
        raise ValueError(_NO_SCHEMA_FAILURE)
    lines: list[str] = []
    if session_id is not None:
        lines.append(_line({"type": "thread.started", "thread_id": session_id}))
    lines.append(_line({"type": "turn.started"}))
    lines.append(_line({"type": "error", "message": message}))
    lines.append(_line({"type": "turn.failed", "error": {"message": message}}))
    return lines, [], 1


def _usage_payload(usage: TokenUsage | None) -> dict[str, int]:
    if usage is None:
        return {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "cache_write_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_output_tokens": 0,
        }
    # Codex nests its counts the way agentshim normalizes them (input includes
    # cache reads and writes, output includes reasoning), so they go out
    # exactly as the parser will read them back.
    return {
        "input_tokens": usage.input_tokens,
        "cached_input_tokens": usage.cache_read_input_tokens,
        "cache_write_input_tokens": usage.cache_write_input_tokens,
        "output_tokens": usage.output_tokens,
        "reasoning_output_tokens": usage.reasoning_output_tokens,
    }


def _line(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload) + "\n"
