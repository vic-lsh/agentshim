"""``Agent(..., transport=TransportKind.STREAM)``: sessions over the long-lived Claude transport."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from agentshim import (
    Agent,
    ApprovalPolicy,
    ClaudeStreamTransport,
    Continuity,
    FailureKind,
    NativePermissions,
    OneShotTransport,
    OutputSchema,
    RetryPolicy,
    TransportKind,
    Turn,
    TurnFailedError,
    TurnRequest,
)
from agentshim.testing import (
    ClaudeApiError,
    ClaudePeerTurn,
    ClaudeStreamPeers,
    FakeClock,
    FakeConfinement,
    FakeExecutor,
    SequentialIds,
)

if TYPE_CHECKING:
    from pathlib import Path

    from agentshim import Session


def _agent(
    script: list[ClaudePeerTurn] | None = None,
    *,
    known_sessions: tuple[str, ...] = (),
) -> tuple[Agent, FakeExecutor, FakeClock]:
    peers = ClaudeStreamPeers(script or [], known_sessions=known_sessions)
    executor = FakeExecutor([], peers=peers.build)
    clock = FakeClock()
    agent = Agent(
        "claude",
        transport=TransportKind.STREAM,
        executor=executor,
        env={},
        permissions=NativePermissions.bypass(),
        approvals=ApprovalPolicy.DENY,
        clock=clock,
        ids=SequentialIds(),
        retry=RetryPolicy((30.0, 60.0)),
    )
    return agent, executor, clock


def _turn(
    session: Session, prompt: str = "go", *, output_schema: OutputSchema | None = None
) -> Turn:
    request = TurnRequest(prompt=prompt, output_schema=output_schema)
    return session.run(session.prepare_turn(request))


class TestWiring:
    def test_the_stream_kind_builds_the_stream_transport(self) -> None:
        agent, _, _ = _agent()
        assert isinstance(agent.transport, ClaudeStreamTransport)

    def test_one_shot_stays_the_default(self) -> None:
        agent = Agent(
            "claude",
            executor=FakeExecutor([]),
            env={},
            permissions=NativePermissions.bypass(),
            approvals=ApprovalPolicy.DENY,
        )
        assert isinstance(agent.transport, OneShotTransport)

    def test_a_ready_made_transport_rejects_a_transport_kind(self) -> None:
        agent, _, _ = _agent()
        with pytest.raises(ValueError, match="transport"):
            Agent(
                agent.transport,
                transport=TransportKind.STREAM,
                permissions=NativePermissions.bypass(),
                approvals=ApprovalPolicy.DENY,
            )

    def test_a_provider_without_a_stream_transport_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no stream transport"):
            Agent(
                "gemini",
                transport=TransportKind.STREAM,
                executor=FakeExecutor([]),
                permissions=NativePermissions.bypass(),
                approvals=ApprovalPolicy.DENY,
            )

    def test_the_confinement_wraps_the_executor_and_supplies_the_env(self) -> None:
        peers = ClaudeStreamPeers()
        executor = FakeExecutor([], peers=peers.build)
        confinement = FakeConfinement(env={"IS_SANDBOX": "1"})
        agent = Agent(
            "claude",
            transport=TransportKind.STREAM,
            executor=executor,
            confinement=confinement,
            permissions=NativePermissions.bypass(),
            approvals=ApprovalPolicy.DENY,
        )
        with agent.session("/host/work") as session:
            _turn(session)
        assert executor.spawns[0].env["IS_SANDBOX"] == "1"


class TestSessions:
    def test_turns_run_in_one_process_and_continue(self) -> None:
        agent, executor, _ = _agent()
        with agent.session("/work") as session:
            first = _turn(session)
            second = _turn(session)
        assert len(executor.spawns) == 1
        assert first.continuity is Continuity.CONTINUED
        assert second.continuity is Continuity.CONTINUED
        assert first.conversation_id == second.conversation_id

    def test_a_refused_resume_starts_a_fresh_conversation_and_reports_replaced(self) -> None:
        agent, executor, _ = _agent()
        with agent.session("/work", resume_id="gone") as session:
            turn = _turn(session)
        assert turn.continuity is Continuity.REPLACED
        assert turn.conversation_id != "gone"
        assert "--resume=gone" in executor.spawns[0].argv
        assert not any(arg.startswith("--resume") for arg in executor.spawns[1].argv)

    def test_a_known_conversation_is_continued(self) -> None:
        agent, executor, _ = _agent(known_sessions=("s-9",))
        with agent.session("/work", resume_id="s-9") as session:
            turn = _turn(session)
        assert turn.continuity is Continuity.CONTINUED
        assert turn.conversation_id == "s-9"
        assert len(executor.spawns) == 1

    def test_a_transient_api_error_is_retried_in_place(self) -> None:
        script = [ClaudePeerTurn(api_error=ClaudeApiError()), ClaudePeerTurn(text="recovered")]
        agent, executor, clock = _agent(script)
        with agent.session("/work") as session:
            turn = _turn(session)
        assert turn.result.text == "recovered"
        assert turn.continuity is Continuity.CONTINUED
        assert 30.0 in clock.waits
        assert len(executor.spawns) == 1

    def test_a_transient_error_that_ends_the_process_is_retried_by_resuming(self) -> None:
        script = [
            ClaudePeerTurn(api_error=ClaudeApiError(ends_process=True)),
            ClaudePeerTurn(text="recovered"),
        ]
        agent, executor, _ = _agent(script)
        with agent.session("/work") as session:
            turn = _turn(session)
        assert turn.result.text == "recovered"
        assert turn.continuity is Continuity.CONTINUED
        assert len(executor.spawns) == 2
        assert any(arg.startswith("--resume=") for arg in executor.spawns[1].argv)

    def test_a_non_transient_error_propagates_and_keeps_the_conversation(self) -> None:
        script = [
            ClaudePeerTurn(api_error=ClaudeApiError(status=401, kind="authentication_failed"))
        ]
        agent, _, _ = _agent(script)
        with agent.session("/work") as session, pytest.raises(TurnFailedError) as raised:
            _turn(session)
        assert raised.value.kind is FailureKind.AUTH

    def test_a_schema_change_restarts_with_resume_and_the_session_stays_continued(
        self, tmp_path: Path
    ) -> None:
        schema = OutputSchema(
            {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]},
            host_dir=tmp_path,
        )
        script = [ClaudePeerTurn(text="plain"), ClaudePeerTurn(structured_output={"n": 1})]
        agent, executor, _ = _agent(script)
        with agent.session("/work") as session:
            first = _turn(session)
            second = _turn(session, output_schema=schema)
        assert second.continuity is Continuity.CONTINUED
        assert second.conversation_id == first.conversation_id
        assert second.result.structured_output == {"n": 1}
        assert len(executor.spawns) == 2
        assert f"--resume={first.conversation_id}" in executor.spawns[1].argv

    def test_session_interrupt_ends_the_turn_and_the_session_goes_on(self) -> None:
        from agentshim import AssistantText  # noqa: PLC0415

        script = [ClaudePeerTurn(text="busy", stall=True), ClaudePeerTurn(text="next")]
        agent, executor, _ = _agent(script)
        with agent.session("/work") as session:
            ticket = session.prepare_turn(TurnRequest(prompt="go"))
            from agentshim import EventHandlerBase  # noqa: PLC0415

            class Interrupter(EventHandlerBase):
                def on_event(self, event: object) -> None:
                    if isinstance(event, AssistantText):
                        session.interrupt()

            interrupted = session.run(ticket, on_event=Interrupter())
            after = _turn(session)
        assert interrupted.result.interrupted is True
        assert after.result.text == "next"
        assert len(executor.spawns) == 1
