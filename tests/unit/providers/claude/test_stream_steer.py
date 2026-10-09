"""Steering a running turn over the Claude stream transport, against the scripted CLI.

The fake CLI folds a mid-turn message into the turn (``injects_steer``) or queues
it as a turn of its own, the two behaviours verified live against Claude Code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from agentshim import (
    NoRunningTurnError,
    SteerableConversation,
    SteerConsumed,
    SteerDelivered,
    SteerRejected,
    TurnRequest,
)
from agentshim.providers.claude import ClaudeStreamTransport
from agentshim.testing import (
    ClaudePeerTurn,
    ClaudeStreamPeers,
    FakeClock,
    FakeExecutor,
    SequentialIds,
)

from tests.unit.providers.claude.test_stream_transport import rig, spec

if TYPE_CHECKING:
    from agentshim import AgentEvent, TurnResult

TEXT = "instead, say STEERED"


def _steered(
    script: list[ClaudePeerTurn], *, interrupt: bool = False, drops: bool = False
) -> tuple[SteerableConversation, TurnResult, list[AgentEvent], ClaudeStreamPeers]:
    if drops:
        peers = ClaudeStreamPeers(script, drops_queued_messages=True)
        transport = ClaudeStreamTransport(
            executor=FakeExecutor([], peers=peers.build),
            env={"PATH": "/bin"},
            clock=FakeClock(),
            ids=SequentialIds(),
        )
    else:
        rigged = rig(script)
        transport, peers = rigged.transport, rigged.peers
    conversation = transport.open(spec())
    assert isinstance(conversation, SteerableConversation)
    events: list[AgentEvent] = []
    sent: list[bool] = []

    def on_event(event: AgentEvent) -> None:
        events.append(event)
        if not sent:
            sent.append(True)
            conversation.steer(TEXT)
            if interrupt:
                conversation.interrupt()

    result = conversation.turn(TurnRequest(prompt="go"), on_event)
    return conversation, result, events, peers


def _steer_events(events: list[AgentEvent]) -> list[AgentEvent]:
    return [e for e in events if isinstance(e, (SteerDelivered, SteerConsumed, SteerRejected))]


def test_the_argv_asks_the_cli_to_echo_user_messages() -> None:
    rigged = rig()
    rigged.transport.open(spec())
    assert "--replay-user-messages" in rigged.argvs[0]


def test_a_message_folded_into_the_turn_ends_in_one_result() -> None:
    script = [ClaudePeerTurn(stall=True, injects_steer=True), ClaudePeerTurn(text="STEERED")]
    conversation, result, events, peers = _steered(script)
    assert result.text == "STEERED"
    assert _steer_events(events) == [SteerDelivered(TEXT), SteerConsumed(TEXT)]
    assert peers.peers[0].prompts == ["go", TEXT]
    assert len(peers.peers) == 1
    later = conversation.turn(TurnRequest(prompt="next"), lambda _e: None)
    assert later.text == "ok"


def test_a_queued_message_runs_inside_the_same_turn_call() -> None:
    script = [ClaudePeerTurn(text="first"), ClaudePeerTurn(text="second")]
    conversation, result, events, _ = _steered(script)
    assert result.text == "second"
    assert _steer_events(events) == [SteerDelivered(TEXT), SteerConsumed(TEXT)]
    later = conversation.turn(TurnRequest(prompt="next"), lambda _e: None)
    assert later.text == "ok"  # no result of the steered turn leaked into this one


def test_usage_of_a_chained_turn_is_the_sum_of_its_results() -> None:
    script = [ClaudePeerTurn(text="a", input_tokens=10), ClaudePeerTurn(text="b", input_tokens=7)]
    _, result, _, _ = _steered(script)
    assert result.usage.tokens.input_tokens == 17


def test_an_interrupt_also_stops_the_turn_a_queued_steer_starts() -> None:
    script = [ClaudePeerTurn(stall=True), ClaudePeerTurn(stall=True)]
    conversation, result, _, _ = _steered(script, interrupt=True)
    assert result.interrupted is True
    later = conversation.turn(TurnRequest(prompt="next"), lambda _e: None)
    assert later.interrupted is False


def test_a_queued_message_the_cli_never_starts_is_rejected_after_the_grace() -> None:
    script = [ClaudePeerTurn(stall=True)]
    _, result, events, _ = _steered(script, interrupt=True, drops=True)
    assert result.interrupted is True
    rejected = [e for e in events if isinstance(e, SteerRejected)]
    assert [e.text for e in rejected] == [TEXT]


def test_steer_with_no_turn_raises() -> None:
    conversation = rig().transport.open(spec())
    assert isinstance(conversation, SteerableConversation)
    with pytest.raises(NoRunningTurnError):
        conversation.steer("early")


def test_the_profile_declares_steering() -> None:
    assert rig().transport.profile.supports_steer is True
