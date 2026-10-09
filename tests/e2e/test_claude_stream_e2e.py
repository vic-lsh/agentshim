"""``ClaudeStreamTransport`` against the real ``claude`` binary.

Skipped unless ``AGENTSHIM_E2E=1`` and ``claude`` is on PATH. Set
``AGENTSHIM_E2E_CLAUDE_MODEL`` (for example ``haiku``) to keep it cheap.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import pytest
from agentshim import (
    ApprovalPolicy,
    ClaudeStreamTransport,
    ConversationSpec,
    NativePermissions,
    OutputSchema,
    SessionResumeError,
    TurnInterrupted,
    TurnRequest,
)

from tests.e2e.conftest import CLAUDE_MODEL_VAR, model_from_env, requires_cli

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from agentshim import AgentEvent, Conversation

pytestmark = [pytest.mark.e2e, requires_cli("claude")]


def _ignore(event: AgentEvent) -> None:
    del event


def _spec(cwd: Path, **changes: object) -> ConversationSpec:
    base = ConversationSpec(
        cwd=str(cwd),
        model=model_from_env(CLAUDE_MODEL_VAR),
        permissions=NativePermissions.bypass(),
        approvals=ApprovalPolicy.DENY,
    )
    fields = {**base.__dict__, **changes}
    return ConversationSpec(**fields)  # type: ignore[arg-type]


@pytest.fixture
def transport() -> ClaudeStreamTransport:
    return ClaudeStreamTransport()


@pytest.fixture
def conversation(transport: ClaudeStreamTransport, tmp_path: Path) -> Iterator[Conversation]:
    opened = transport.open(_spec(tmp_path))
    yield opened
    opened.close()


def test_two_turns_in_one_process_keep_their_context(conversation: Conversation) -> None:
    first = conversation.turn(
        TurnRequest(prompt="Remember the word 'juniper'. Reply with 'ok'."), _ignore
    )
    second = conversation.turn(
        TurnRequest(prompt="What word did I ask you to remember? Reply with just the word."),
        _ignore,
    )
    assert second.session_id == first.session_id
    assert second.resumed is True
    assert "juniper" in second.text.lower()
    assert second.cost_usd is not None
    assert second.cost_usd >= 0


def test_a_conversation_resumes_across_processes(
    transport: ClaudeStreamTransport, tmp_path: Path
) -> None:
    first = transport.open(_spec(tmp_path))
    try:
        result = first.turn(TurnRequest(prompt="Remember 'juniper'. Reply 'ok'."), _ignore)
    finally:
        first.close()
    assert result.session_id is not None
    second = transport.open(_spec(tmp_path, resume_id=result.session_id))
    try:
        answer = second.turn(TurnRequest(prompt="What word? Reply with just the word."), _ignore)
    finally:
        second.close()
    assert "juniper" in answer.text.lower()


def test_an_output_schema_restarts_the_process_and_returns_the_payload(
    conversation: Conversation, tmp_path: Path
) -> None:
    first = conversation.turn(TurnRequest(prompt="Reply with: ok"), _ignore)
    schema = OutputSchema(
        {
            "type": "object",
            "properties": {"answer": {"type": "integer"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
        host_dir=tmp_path,
    )
    second = conversation.turn(TurnRequest(prompt="What is 2 + 2?", output_schema=schema), _ignore)
    assert second.structured_output == {"answer": 4}
    assert second.session_id == first.session_id


def test_interrupt_ends_the_turn_and_the_conversation_survives(conversation: Conversation) -> None:
    events: list[AgentEvent] = []
    timer = threading.Timer(4.0, conversation.interrupt)
    timer.start()
    try:
        interrupted = conversation.turn(
            TurnRequest(prompt="Run the bash command `sleep 120`, then say done."),
            events.append,
        )
    finally:
        timer.cancel()
    assert interrupted.interrupted is True
    assert any(isinstance(event, TurnInterrupted) for event in events)
    after = conversation.turn(TurnRequest(prompt="Reply with exactly: alive"), _ignore)
    assert "alive" in after.text.lower()
    assert after.session_id == interrupted.session_id


def test_a_refused_resume_is_a_session_resume_error_at_open(
    transport: ClaudeStreamTransport, tmp_path: Path
) -> None:
    with pytest.raises(SessionResumeError):
        transport.open(_spec(tmp_path, resume_id="11111111-2222-4333-8444-555555555555"))


def test_workspace_write_confines_native_writes(
    transport: ClaudeStreamTransport, tmp_path: Path
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    outside = tmp_path / "outside.txt"
    conversation = transport.open(_spec(work, permissions=NativePermissions.workspace_write()))
    try:
        conversation.turn(
            TurnRequest(prompt=f"Write the text hi to the file {outside} using your Write tool."),
            _ignore,
        )
    finally:
        conversation.close()
    assert not outside.exists()
