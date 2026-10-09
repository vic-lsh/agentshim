"""``turn/steer`` over the Codex app-server transport, against the scripted server."""

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
from agentshim.providers.codex.app_server.protocol import TurnSteerParams
from agentshim.testing import AwaitSteer, CodexScript, RunCommand, Say

from tests.unit.providers.codex.app_server.harness import Rig, rig

if TYPE_CHECKING:
    from agentshim import Conversation, TurnResult

TEXT = "instead, say STEERED"


def _run(script: CodexScript) -> tuple[Rig, Conversation, TurnResult]:
    rigged = rig(script)
    conversation = rigged.open()
    assert isinstance(conversation, SteerableConversation)
    sent: list[bool] = []

    def steer_once() -> None:
        try:
            conversation.steer(TEXT)
        except NoRunningTurnError:
            return
        sent.append(True)

    def on_event(event: object) -> None:
        rigged.events.append(event)  # type: ignore[arg-type]
        if not sent:
            steer_once()

    result = conversation.turn(TurnRequest(prompt="go", timeout=60), on_event)
    return rigged, conversation, result


def test_a_steered_turn_continues_and_reports_delivery_then_consumption() -> None:
    script = CodexScript().turn(RunCommand("sleep 20"), AwaitSteer(then=(Say("STEERED"),)))
    rigged, _, result = _run(script)
    assert result.text == "STEERED"
    steer_events = [
        e for e in rigged.events if isinstance(e, (SteerDelivered, SteerConsumed, SteerRejected))
    ]
    assert steer_events == [SteerDelivered(TEXT), SteerConsumed(TEXT)]


def test_the_request_carries_the_running_turn_id_and_a_client_id() -> None:
    script = CodexScript().turn(AwaitSteer(then=(Say("ok"),)))
    _run(script)
    (request,) = script.requests("turn/steer")
    params = request.params
    assert isinstance(params, TurnSteerParams)
    assert params.expected_turn_id == "turn-1"
    assert params.client_user_message_id == "steer-1"
    assert script.violations == []


def test_a_refusal_is_reported_as_an_event_and_the_turn_is_unaffected() -> None:
    script = CodexScript(steer_refusal="turn cannot be steered").turn(
        Say("a"), AwaitSteer(then=(Say("b"),))
    )
    rigged = rig(script)
    conversation = rigged.open()
    assert isinstance(conversation, SteerableConversation)
    sent: list[bool] = []

    def on_event(event: object) -> None:
        rigged.events.append(event)  # type: ignore[arg-type]
        if not sent:
            try:
                conversation.steer(TEXT)
            except NoRunningTurnError:
                return
            sent.append(True)
            conversation.interrupt()

    result = conversation.turn(TurnRequest(prompt="go", timeout=60), on_event)
    assert result.interrupted is True
    rejected = [e for e in rigged.events if isinstance(e, SteerRejected)]
    assert [(e.text, e.reason) for e in rejected] == [(TEXT, "turn cannot be steered")]


def test_steer_with_no_turn_raises() -> None:
    conversation = rig().open()
    assert isinstance(conversation, SteerableConversation)
    with pytest.raises(NoRunningTurnError):
        conversation.steer("early")


def test_a_steer_after_the_turn_completed_raises() -> None:
    _, conversation, _ = _run(CodexScript().turn(AwaitSteer(then=(Say("ok"),))))
    assert isinstance(conversation, SteerableConversation)
    with pytest.raises(NoRunningTurnError):
        conversation.steer("late")


def test_the_profile_declares_steering() -> None:
    assert rig().transport.profile.supports_steer is True
