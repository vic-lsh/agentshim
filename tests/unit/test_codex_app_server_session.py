"""Sessions on the Codex app-server transport: recovery, renewal, checkpoints and wiring."""

from __future__ import annotations

import pytest
from agentshim import (
    Agent,
    ApprovalPolicy,
    Continuity,
    FailureKind,
    NativePermissions,
    OneShotTransport,
    SessionResumeError,
    TransportKind,
    TurnFailedError,
    TurnInterrupted,
    TurnRequest,
)
from agentshim.core.checkpoints import Checkpoint
from agentshim.providers.codex.app_server import CodexAppServerTransport
from agentshim.testing import (
    CodexScript,
    Fail,
    FakeCheckpointStore,
    FakeClock,
    FakeExecutor,
    Hang,
    RecordingEventHandler,
    Say,
    SequentialIds,
    Spend,
)

ENV = {"PATH": "/usr/bin:/bin", "HOME": "/home/tester"}


def _agent(script: CodexScript, clock: FakeClock | None = None, **options: object) -> Agent:
    arguments: dict[str, object] = {
        "transport": TransportKind.STREAM,
        "model": "fake-model",
        "executor": FakeExecutor([], peers=script.peer),
        "env": dict(ENV),
        "permissions": NativePermissions.bypass(),
        "approvals": ApprovalPolicy.DENY,
        "clock": clock if clock is not None else FakeClock(),
        "ids": SequentialIds(),
    }
    arguments.update(options)
    return Agent("codex", **arguments)  # type: ignore[arg-type]


def _run(session, prompt: str = "go"):  # noqa: ANN001, ANN202 - a test helper
    return session.run(session.prepare_turn(TurnRequest(prompt=prompt, timeout=60.0)))


# -- wiring


def test_the_app_server_transport_is_chosen_by_name_and_the_default_stays_one_shot() -> None:
    assert isinstance(_agent(CodexScript()).transport, CodexAppServerTransport)
    one_shot = Agent(
        "codex",
        executor=FakeExecutor([]),
        env=dict(ENV),
        permissions=NativePermissions.bypass(),
        approvals=ApprovalPolicy.DENY,
    )
    assert isinstance(one_shot.transport, OneShotTransport)


def test_the_agents_profile_is_the_app_server_one() -> None:
    modes = _agent(CodexScript()).profile.native_permission_modes
    assert len(modes) == 3


def test_a_provider_without_a_stream_transport_is_refused() -> None:
    with pytest.raises(ValueError, match="no stream transport"):
        Agent(
            "copilot",
            transport=TransportKind.STREAM,
            executor=FakeExecutor([]),
            env=dict(ENV),
            permissions=NativePermissions.bypass(),
            approvals=ApprovalPolicy.DENY,
        )


def test_a_ready_made_transport_cannot_also_be_given_a_transport_kind() -> None:
    transport = _agent(CodexScript()).transport
    with pytest.raises(ValueError, match="belong to the transport"):
        Agent(
            transport,
            transport=TransportKind.STREAM,
            permissions=NativePermissions.bypass(),
            approvals=ApprovalPolicy.DENY,
        )


def test_a_confinement_wraps_the_spawn_and_supplies_the_environment() -> None:
    from agentshim.testing import FakeConfinement  # noqa: PLC0415

    script = CodexScript()
    executor = FakeExecutor([], peers=script.peer)
    confinement = FakeConfinement(env={"CODEX_HOME": "/home/agent/.codex", "PATH": "/usr/bin"})
    agent = Agent(
        "codex",
        transport=TransportKind.STREAM,
        executor=executor,
        confinement=confinement,
        permissions=NativePermissions.bypass(),
        approvals=ApprovalPolicy.DENY,
        clock=FakeClock(),
        ids=SequentialIds(),
    )
    with agent.session("/work") as session:
        _run(session)
    assert list(executor.spawns[0].argv)[:3] == ["fake-confine", "codex", "app-server"]
    assert executor.spawns[0].env["CODEX_HOME"] == "/home/agent/.codex"
    assert confinement.wraps[-1][1] == "/work"  # the spawn, after the health check


# -- recovery


def test_a_refused_resume_starts_a_fresh_conversation_and_says_replaced() -> None:
    script = CodexScript()
    agent = _agent(script)
    with agent.session("/work", resume_id="gone") as session:
        turn = _run(session)
    assert turn.continuity is Continuity.REPLACED
    assert turn.conversation_id == "thread-1"
    assert script.spawned == 2  # the refused attempt, then the fresh one
    assert script.violations == []


def test_a_resume_the_server_accepts_continues() -> None:
    script = CodexScript()
    script.add_thread("old")
    with _agent(script).session("/work", resume_id="old") as session:
        turn = _run(session)
    assert turn.continuity is Continuity.CONTINUED
    assert turn.conversation_id == "old"


def test_a_transient_failure_is_retried_in_the_same_process_after_the_wait() -> None:
    script = CodexScript()
    script.turn(Fail("overloaded", "serverOverloaded"))
    script.turn(Say("second try"))
    clock = FakeClock()
    with _agent(script, clock).session("/work") as session:
        turn = _run(session)
    assert turn.result.text == "second try"
    assert turn.continuity is Continuity.CONTINUED
    assert clock.waits == [30.0]
    assert script.spawned == 1


def test_a_usage_limit_is_not_retried() -> None:
    script = CodexScript()
    script.turn(Fail("usage limit", "usageLimitExceeded"))
    clock = FakeClock()
    with (
        _agent(script, clock).session("/work") as session,
        pytest.raises(TurnFailedError) as caught,
    ):
        _run(session)
    assert caught.value.kind is FailureKind.USAGE_LIMIT
    assert clock.waits == []


def test_codex_is_renewed_after_two_turns_into_a_new_process() -> None:
    script = CodexScript()
    with _agent(script).session("/work") as session:
        first = _run(session)
        second = _run(session)
        third = _run(session)
    assert [t.continuity for t in (first, second, third)] == [
        Continuity.CONTINUED,
        Continuity.RESET,
        Continuity.CONTINUED,
    ]
    assert first.conversation_id == second.conversation_id != third.conversation_id
    assert script.spawned == 2
    assert script.closed >= 1  # the retired conversation's process was shut down


def test_a_heavy_turn_retires_the_conversation_early() -> None:
    script = CodexScript()
    script.turn(Spend(input_tokens=10_000_000))
    with _agent(script).session("/work") as session:
        assert _run(session).continuity is Continuity.RESET


def test_release_closes_the_process_and_the_next_turn_resumes_the_thread() -> None:
    script = CodexScript()
    script.turn(Spend(input_tokens=1000, output_tokens=10))
    script.turn(Spend(input_tokens=200, output_tokens=5))
    with _agent(script).session("/work") as session:
        first = _run(session)
        session.release()
        # Pinned, so renewal (two turns for Codex) does not retire it.
        second = session.run(session.prepare_turn(TurnRequest(prompt="go", timeout=60.0), pin=True))
    assert script.spawned == 2
    assert second.conversation_id == first.conversation_id
    assert second.continuity is Continuity.CONTINUED
    assert second.result.resumed is True
    # The session carried the first report across the restart, so the increment is exact.
    assert second.result.usage.increment_known
    assert second.result.usage.tokens.input_tokens == 200


def test_a_checkpoint_resumes_in_a_later_process_with_exact_usage() -> None:
    store = FakeCheckpointStore()
    script = CodexScript()
    script.turn(Spend(input_tokens=1000, output_tokens=10))
    script.turn(Spend(input_tokens=300, output_tokens=7))
    with _agent(script).session("/work", checkpoints=store, checkpoint_key="k") as session:
        first = _run(session)
    saved = store.load("k")
    assert isinstance(saved, Checkpoint)
    assert saved.conversation_id == first.conversation_id
    # A later process: a new agent, the same rollout store, the saved checkpoint.
    with _agent(script).session("/work", checkpoints=store, checkpoint_key="k") as session:
        second = _run(session)
    assert second.continuity is Continuity.CONTINUED
    assert second.result.usage.increment_known
    assert second.result.usage.tokens.input_tokens == 300


def test_an_interrupt_ends_the_turn_and_the_session_goes_on() -> None:
    script = CodexScript()
    script.turn(Say("working", "commentary"), Hang())
    script.turn(Say("next"))
    agent = _agent(script)
    handler = RecordingEventHandler()
    with agent.session("/work") as session:
        ticket = session.prepare_turn(TurnRequest(prompt="go", timeout=60.0))

        class Interrupter:
            def on_event(self, event: object) -> None:
                handler.on_event(event)  # type: ignore[arg-type]
                if type(event).__name__ == "AssistantText":
                    session.interrupt()

        turn = session.run(ticket, on_event=Interrupter())  # type: ignore[arg-type]
        assert turn.result.interrupted is True
        assert handler.of_type(TurnInterrupted)
        assert _run(session).result.text == "next"
        assert script.spawned == 1


def test_a_refused_resume_that_cannot_be_replaced_propagates() -> None:
    script = CodexScript()
    with _agent(script).session("/work", resume_id="gone") as session:
        ticket = session.prepare_turn(TurnRequest(prompt="go", timeout=60.0), pin=True)
        with pytest.raises(SessionResumeError):
            session.run(ticket)
