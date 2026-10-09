"""Codex over ``app-server`` against the real binary.

Skipped unless ``AGENTSHIM_E2E=1`` and ``codex`` is on PATH; the model comes
from ``AGENTSHIM_E2E_CODEX_MODEL`` (for example ``gpt-6-luna``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from agentshim import (
    Agent,
    ApprovalDenied,
    ApprovalPolicy,
    Continuity,
    NativePermissions,
    OutputSchema,
    Session,
    ToolCall,
    TransportKind,
    TurnInterrupted,
    TurnRequest,
)
from agentshim.testing import RecordingEventHandler

from tests.e2e.conftest import CODEX_MODEL_VAR, model_from_env, requires_cli

if TYPE_CHECKING:
    from pathlib import Path

    from agentshim import AgentEvent, Turn

pytestmark = [pytest.mark.e2e, requires_cli("codex")]

TURN_TIMEOUT_S = 300.0


def _agent(
    permissions: NativePermissions | None = None,
    approvals: ApprovalPolicy = ApprovalPolicy.DENY,
    handler: RecordingEventHandler | None = None,
) -> Agent:
    return Agent(
        "codex",
        transport=TransportKind.STREAM,
        model=model_from_env(CODEX_MODEL_VAR),
        permissions=permissions if permissions is not None else NativePermissions.bypass(),
        approvals=approvals,
        event_handlers=[handler] if handler is not None else [],
    )


def _run(session: Session, prompt: str, **request: object) -> Turn:
    ticket = session.prepare_turn(TurnRequest(prompt=prompt, timeout=TURN_TIMEOUT_S, **request))  # type: ignore[arg-type]
    return session.run(ticket)


def test_two_turns_share_one_thread_and_report_exact_usage(tmp_path: Path) -> None:
    with _agent().session(str(tmp_path)) as session:
        first = _run(session, "Remember the word 'juniper'. Reply with 'ok'.")
        second = _run(session, "What word did I ask you to remember? Reply with just the word.")
    assert first.conversation_id is not None
    assert first.conversation_id == second.conversation_id
    assert "juniper" in second.result.text.lower()
    assert first.result.usage.tokens.input_tokens > 0
    assert second.result.usage.increment_known
    assert second.result.usage.raw is not None
    assert first.result.usage.raw is not None
    summed = first.result.usage.tokens + second.result.usage.tokens
    assert summed.input_tokens == second.result.usage.raw["input_tokens"]


def test_a_thread_resumes_in_a_new_process(tmp_path: Path) -> None:
    with _agent().session(str(tmp_path)) as session:
        first = _run(session, "Remember the word 'juniper'. Reply with 'ok'.")
    assert first.conversation_id is not None
    with _agent().session(
        str(tmp_path), resume_id=first.conversation_id, previous_usage=first.result.usage
    ) as session:
        second = _run(session, "What word did I ask you to remember? Reply with just the word.")
    assert second.continuity is Continuity.CONTINUED
    assert second.result.resumed is True
    assert "juniper" in second.result.text.lower()
    assert second.result.usage.increment_known


def test_an_output_schema_returns_structured_output(tmp_path: Path) -> None:
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    with _agent().session(str(tmp_path)) as session:
        turn = _run(
            session,
            "What is 6 times 7? Answer in the schema.",
            output_schema=OutputSchema(schema, tmp_path),
        )
    assert turn.result.structured_output == {"answer": 42}


def test_an_interrupt_ends_a_running_command_and_the_thread_goes_on(tmp_path: Path) -> None:
    handler = RecordingEventHandler()
    with _agent(handler=handler).session(str(tmp_path)) as session:

        class InterruptOnTool:
            def on_event(self, event: AgentEvent) -> None:
                if isinstance(event, ToolCall) and "sleep" in str(event.args):
                    session.interrupt()

        ticket = session.prepare_turn(
            TurnRequest(prompt="Run the shell command: sleep 60", timeout=TURN_TIMEOUT_S)
        )
        interrupted = session.run(ticket, on_event=InterruptOnTool())
        assert interrupted.result.interrupted is True
        assert handler.of_type(TurnInterrupted)
        after = _run(session, "Reply with the single word pong.")
    assert "pong" in after.result.text.lower()


def test_a_write_under_read_only_with_deny_is_refused_and_the_turn_still_ends(
    tmp_path: Path,
) -> None:
    handler = RecordingEventHandler()
    agent = _agent(NativePermissions.read_only(), ApprovalPolicy.DENY, handler)
    with agent.session(str(tmp_path)) as session:
        turn = _run(
            session,
            "Use the shell to create a file named marker.txt containing the word x in the "
            "current directory. If you are not allowed to, say 'denied'.",
        )
    assert turn.result.interrupted is False
    assert not (tmp_path / "marker.txt").exists()
    # The sandbox stops the write; if the agent asked to go beyond it, the policy said no.
    for event in handler.of_type(ApprovalDenied):
        assert isinstance(event, ApprovalDenied)
