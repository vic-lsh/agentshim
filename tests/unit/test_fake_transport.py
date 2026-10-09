"""The scripted transport and checkpoint store, against their contracts and on their own terms."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
from agentshim import (
    ApprovalPolicy,
    Checkpoint,
    ConversationSpec,
    FailureKind,
    Lifecycle,
    NativePermissions,
    SessionResumeError,
    SessionStateError,
    TurnFailedError,
    TurnInterrupted,
    TurnRequest,
)
from agentshim.testing import (
    FakeCheckpointStore,
    FakeTransport,
    FakeTurn,
    fake_profile,
    resume_refused,
    turn_failed,
)
from agentshim.testing.contracts import (
    CheckpointStoreContract,
    ConversationContract,
    SteerableConversationContract,
    TransportContract,
)

if TYPE_CHECKING:
    from agentshim import CheckpointStore, Conversation, Transport


def _spec(**changes: object) -> ConversationSpec:
    base = ConversationSpec(
        cwd="/w",
        model=None,
        permissions=NativePermissions.bypass(),
        approvals=ApprovalPolicy.DENY,
    )
    return replace(base, **changes)  # type: ignore[arg-type]


def _ignore(event: object) -> None:
    del event


class TestFakeTransportContract(TransportContract):
    def make_transport(self) -> Transport:
        return FakeTransport()


class TestFakeConversationContract(ConversationContract):
    def make_conversation(self) -> Conversation:
        return FakeTransport().open(_spec())


class TestFakeSteerableConversationContract(SteerableConversationContract):
    def make_conversation(self) -> Conversation:
        transport = FakeTransport(
            [FakeTurn(events=(Lifecycle("working", "a long tool call"),))],
            profile=fake_profile(supports_steer=True),
        )
        return transport.open(_spec())


class TestFakeCheckpointStoreContract(CheckpointStoreContract):
    def make_store(self) -> CheckpointStore:
        return FakeCheckpointStore()


def test_the_script_is_consumed_in_order_across_conversations() -> None:
    transport = FakeTransport(
        [FakeTurn(text="a"), turn_failed(FailureKind.AUTH), FakeTurn(text="c")]
    )
    first = transport.open(_spec())
    second = transport.open(_spec())
    assert first.turn(TurnRequest(prompt="1"), _ignore).text == "a"
    with pytest.raises(TurnFailedError):
        second.turn(TurnRequest(prompt="2"), _ignore)
    assert first.turn(TurnRequest(prompt="3"), _ignore).text == "c"
    assert first.turn(TurnRequest(prompt="4"), _ignore).text == "ok"


def test_conversations_name_themselves_and_keep_the_name() -> None:
    conversation = FakeTransport().open(_spec())
    assert conversation.conversation_id is None
    result = conversation.turn(TurnRequest(prompt="1"), _ignore)
    assert result.session_id == conversation.conversation_id == "conv-1"
    assert result.resumed is False
    assert conversation.turn(TurnRequest(prompt="2"), _ignore).resumed is True


def test_a_resumed_conversation_starts_with_its_id() -> None:
    conversation = FakeTransport().open(_spec(resume_id="old"))
    assert conversation.conversation_id == "old"


def test_an_unknown_conversation_is_refused_at_the_first_turn() -> None:
    transport = FakeTransport(resumable={"known"})
    transport.open(_spec(resume_id="known")).turn(TurnRequest(prompt="x"), _ignore)
    with pytest.raises(SessionResumeError):
        transport.open(_spec(resume_id="gone")).turn(TurnRequest(prompt="x"), _ignore)


def test_an_unknown_conversation_can_be_refused_at_open() -> None:
    transport = FakeTransport(resumable=set(), refuse_at_open=True)
    with pytest.raises(SessionResumeError):
        transport.open(_spec(resume_id="gone"))


def test_a_scripted_refusal_is_a_resume_error() -> None:
    conversation = FakeTransport([resume_refused("x")]).open(_spec(resume_id="x"))
    with pytest.raises(SessionResumeError):
        conversation.turn(TurnRequest(prompt="x"), _ignore)


def test_interrupting_during_a_turn_marks_the_result() -> None:
    holder: list[Conversation] = []

    def interrupt() -> None:
        holder[0].interrupt()

    transport = FakeTransport([FakeTurn(during=interrupt)])
    events: list[object] = []
    conversation = transport.open(_spec())
    holder.append(conversation)
    result = conversation.turn(TurnRequest(prompt="x"), events.append)
    assert result.interrupted is True
    assert events == [TurnInterrupted()]


def test_a_closed_conversation_refuses_turns_and_close_is_counted() -> None:
    transport = FakeTransport()
    conversation = transport.open(_spec())
    conversation.close()
    conversation.close()
    with pytest.raises(SessionStateError):
        conversation.turn(TurnRequest(prompt="x"), _ignore)
    assert transport.conversations[0].close_calls == 2


def test_the_checkpoint_store_records_its_calls() -> None:
    store = FakeCheckpointStore()
    store.save("k", Checkpoint("c"))
    store.load("k")
    store.clear("k")
    assert store.calls == [("save", "k"), ("load", "k"), ("clear", "k")]
