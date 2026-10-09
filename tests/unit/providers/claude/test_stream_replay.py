"""The real transport against real recorded frames (``tests/fixtures/claude_stream``)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from agentshim import (
    ApprovalPolicy,
    ClaudeStreamTransport,
    ConversationSpec,
    NativePermissions,
    OutputSchema,
    SessionResumeError,
    SkillsDiscovered,
    TurnInterrupted,
    TurnRequest,
)
from agentshim.testing import (
    ClaudeRecordedPeer,
    FakeClock,
    FakeExecutor,
    RecordingEventHandler,
    SequentialIds,
)

FIXTURES = Path(__file__).parents[2].joinpath("..", "fixtures", "claude_stream").resolve()


def _transport(name: str, tmp: list[ClaudeRecordedPeer] | None = None) -> ClaudeStreamTransport:
    stdout = FIXTURES.joinpath(f"{name}.stdout.jsonl").read_text(encoding="utf-8")

    def peer(request: object) -> ClaudeRecordedPeer:
        del request
        recorded = ClaudeRecordedPeer(stdout, stderr="Error: No conversation found\n")
        if tmp is not None:
            tmp.append(recorded)
        return recorded

    return ClaudeStreamTransport(
        executor=FakeExecutor([], peers=peer),
        env={},
        clock=FakeClock(),
        ids=SequentialIds(),
    )


def _ignore(event: object) -> None:
    del event


def _spec(**changes: object) -> ConversationSpec:
    base = ConversationSpec(
        cwd="/work",
        model=None,
        permissions=NativePermissions.bypass(),
        approvals=ApprovalPolicy.DENY,
    )
    return replace(base, **changes)  # type: ignore[arg-type]


def test_two_turns_in_one_process() -> None:
    conversation = _transport("a_two_turns").open(_spec())
    events = RecordingEventHandler()
    first = conversation.turn(TurnRequest(prompt="Reply with just: ok"), events.on_event)
    second = conversation.turn(TurnRequest(prompt="reply again"), events.on_event)
    assert first.text == "ok"
    assert second.text
    assert first.session_id == second.session_id == conversation.conversation_id
    assert first.interrupted is second.interrupted is False
    assert events.of_type(SkillsDiscovered)  # every turn repeats system/init


def test_the_cost_of_each_turn_is_the_difference_of_the_cumulative_totals() -> None:
    conversation = _transport("a_two_turns").open(_spec())
    first = conversation.turn(TurnRequest(prompt="a"), _ignore)
    second = conversation.turn(TurnRequest(prompt="b"), _ignore)
    assert first.cost_usd == pytest.approx(0.00148042)
    assert second.cost_usd == pytest.approx(0.00264091 - 0.00148042)


def test_usage_is_the_turns_own() -> None:
    conversation = _transport("a_two_turns").open(_spec())
    first = conversation.turn(TurnRequest(prompt="a"), _ignore)
    second = conversation.turn(TurnRequest(prompt="b"), _ignore)
    assert first.usage.tokens.input_tokens > 0
    assert second.usage.tokens.input_tokens > 0
    assert second.usage.tokens.turns == 1


def test_a_recorded_interrupt_is_an_interrupted_turn_and_the_conversation_goes_on() -> None:
    conversation = _transport("b_bash_interrupt").open(_spec())
    events = RecordingEventHandler()
    results = [
        conversation.turn(TurnRequest(prompt=prompt), events.on_event)
        for prompt in ("echo hi", "sleep 60", "alive?")
    ]
    assert [r.interrupted for r in results] == [False, True, False]
    assert len(events.of_type(TurnInterrupted)) == 1
    assert len({r.session_id for r in results}) == 1
    assert results[2].text


def test_a_recorded_structured_output_turn(tmp_path: Path) -> None:
    schema = OutputSchema(
        {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"]},
        host_dir=tmp_path,
    )
    conversation = _transport("c_schema").open(_spec())
    result = conversation.turn(TurnRequest(prompt="6*7?", output_schema=schema), _ignore)
    assert result.structured_output == {"answer": 42}


def test_a_recorded_refused_resume_is_a_session_resume_error_at_open() -> None:
    with pytest.raises(SessionResumeError) as raised:
        _transport("d_bad_resume").open(_spec(resume_id="11111111-2222-4333-8444-555555555555"))
    assert raised.value.session_id == "11111111-2222-4333-8444-555555555555"
    assert raised.value.returncode == 1
    assert "No conversation found" in raised.value.detail


def test_every_recorded_frame_is_handled_without_a_raw_output_leak() -> None:
    from agentshim import RawOutput  # noqa: PLC0415 - only this test needs it

    conversation = _transport("a_two_turns").open(_spec())
    events = RecordingEventHandler()
    conversation.turn(TurnRequest(prompt="a"), events.on_event)
    assert events.of_type(RawOutput) == []
