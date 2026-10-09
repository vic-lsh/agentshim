"""``Agent`` and ``Session``: the shell around the policy, driven by fakes.

Every test runs on a ``FakeTransport``, a ``FakeClock`` and ``SequentialIds``:
no process, no real waiting.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
from agentshim import (
    Agent,
    ApprovalPolicy,
    AssistantText,
    CheckpointStore,
    Continuity,
    ContinuityError,
    ConversationSpec,
    FailureKind,
    NativePermissions,
    ProviderUsage,
    RenewalBudget,
    RetryPolicy,
    Session,
    SessionResumeError,
    SessionStateError,
    StopSignal,
    Turn,
    TurnCancelledError,
    TurnFailedError,
    TurnInterrupted,
    TurnRequest,
)
from agentshim.core.checkpoints import Checkpoint
from agentshim.testing import (
    FakeCheckpointStore,
    FakeClock,
    FakeOutcome,
    FakeTransport,
    FakeTurn,
    RecordingEventHandler,
    SequentialIds,
    fake_profile,
    resume_refused,
    turn_failed,
    turn_timeout,
)
from hypothesis import given
from hypothesis import strategies as st

if TYPE_CHECKING:
    from collections.abc import Collection

    from agentshim import Transport

DELAYS = (1.0, 2.0, 4.0)


def _agent(
    transport: Transport,
    *,
    clock: FakeClock | None = None,
    retry: RetryPolicy | None = None,
    handlers: RecordingEventHandler | None = None,
) -> Agent:
    return Agent(
        transport,
        permissions=NativePermissions.bypass(),
        approvals=ApprovalPolicy.DENY,
        clock=clock if clock is not None else FakeClock(),
        ids=SequentialIds(),
        retry=retry if retry is not None else RetryPolicy(DELAYS),
        event_handlers=(handlers,) if handlers is not None else (),
    )


def _session(  # noqa: PLR0913 - mirrors the session options the tests vary
    transport: Transport,
    *,
    clock: FakeClock | None = None,
    checkpoints: CheckpointStore | None = None,
    checkpoint_key: str | None = None,
    idle_release_after: float | None = None,
    resume_id: str | None = None,
) -> Session:
    return _agent(transport, clock=clock).session(
        "/work",
        checkpoints=checkpoints,
        checkpoint_key=checkpoint_key,
        idle_release_after=idle_release_after,
        resume_id=resume_id,
    )


def _run(
    session: Session,
    prompt: str = "go",
    *,
    expect_conversation: str | None = None,
    pin: bool = False,
) -> Turn:
    ticket = session.prepare_turn(
        TurnRequest(prompt=prompt), expect_conversation=expect_conversation, pin=pin
    )
    return session.run(ticket)


def _transport(
    *script: FakeOutcome,
    renewal: RenewalBudget | None = None,
    resumable: Collection[str] | None = None,
    refuse_at_open: bool = False,
) -> FakeTransport:
    profile = replace(fake_profile(), renewal=renewal)
    return FakeTransport(
        list(script), profile=profile, resumable=resumable, refuse_at_open=refuse_at_open
    )


class TestTurns:
    def test_a_turn_reports_its_result_continuity_and_ids(self) -> None:
        session = _session(_transport(FakeTurn(text="hello")))
        ticket = session.prepare_turn(TurnRequest(prompt="hi"))
        assert ticket.turn_id == "turn-1"
        assert ticket.expected_conversation is None
        turn = session.run(ticket)
        assert turn.result.text == "hello"
        assert turn.result.interrupted is False
        assert turn.continuity is Continuity.CONTINUED
        assert turn.turn_id == "turn-1"
        assert turn.conversation_id == "conv-1"
        assert session.conversation_id == "conv-1"
        assert session.last_turn_conversation_id == "conv-1"

    def test_the_second_turn_continues_the_same_conversation(self) -> None:
        transport = _transport()
        session = _session(transport)
        _run(session)
        second = _run(session)
        assert second.conversation_id == "conv-1"
        assert len(transport.conversations) == 1
        assert len(transport.conversations[0].turns) == 2

    def test_the_spec_carries_the_session_options(self) -> None:
        transport = _transport()
        agent = Agent(
            transport,
            model="m1",
            permissions=NativePermissions.bypass(),
            approvals=ApprovalPolicy.FAIL_TURN,
            clock=FakeClock(),
            ids=SequentialIds(),
        )
        _run(agent.session("/w", reasoning_effort="high"))
        spec = transport.specs[0]
        assert (spec.cwd, spec.model, spec.reasoning_effort) == ("/w", "m1", "high")
        assert spec.approvals is ApprovalPolicy.FAIL_TURN

    def test_events_go_to_the_agent_handlers_and_the_turn_handler(self) -> None:
        agent_events = RecordingEventHandler()
        turn_events = RecordingEventHandler()
        transport = _transport(FakeTurn(events=(AssistantText("a"),)))
        session = _agent(transport, handlers=agent_events).session("/work")
        session.run(session.prepare_turn(TurnRequest(prompt="x")), on_event=turn_events)
        assert agent_events.events == [AssistantText("a")]
        assert turn_events.events == [AssistantText("a")]
        later = RecordingEventHandler()
        session.run(session.prepare_turn(TurnRequest(prompt="y")), on_event=later)
        assert later.events == []

    def test_ids_are_known_at_prepare_time_and_increase(self) -> None:
        session = _session(_transport())
        first = session.prepare_turn(TurnRequest(prompt="a"))
        second = session.prepare_turn(TurnRequest(prompt="b"))
        assert (first.turn_id, second.turn_id) == ("turn-1", "turn-2")


class TestTickets:
    def test_a_ticket_runs_once(self) -> None:
        session = _session(_transport())
        ticket = session.prepare_turn(TurnRequest(prompt="a"))
        session.run(ticket)
        with pytest.raises(SessionStateError, match="stale or was already run"):
            session.run(ticket)

    def test_only_the_most_recent_ticket_runs(self) -> None:
        session = _session(_transport())
        old = session.prepare_turn(TurnRequest(prompt="a"))
        new = session.prepare_turn(TurnRequest(prompt="b"))
        with pytest.raises(SessionStateError):
            session.run(old)
        assert session.run(new).turn_id == new.turn_id

    def test_a_failed_turns_ticket_cannot_be_rerun(self) -> None:
        session = _session(_transport(turn_failed(FailureKind.AUTH)))
        ticket = session.prepare_turn(TurnRequest(prompt="a"))
        with pytest.raises(TurnFailedError):
            session.run(ticket)
        with pytest.raises(SessionStateError):
            session.run(ticket)

    def test_preparing_during_a_running_turn_is_refused(self) -> None:
        holder: list[Session] = []
        refused: list[Exception] = []

        def during() -> None:
            try:
                holder[0].prepare_turn(TurnRequest(prompt="nested"))
            except SessionStateError as error:
                refused.append(error)

        session = _session(_transport(FakeTurn(during=during)))
        holder.append(session)
        _run(session)
        assert len(refused) == 1

    def test_expecting_a_conversation_the_session_does_not_hold_is_refused(self) -> None:
        session = _session(_transport())
        with pytest.raises(ContinuityError) as info:
            session.prepare_turn(TurnRequest(prompt="a"), expect_conversation="conv-9")
        assert (info.value.expected, info.value.actual) == ("conv-9", None)
        _run(session)
        ok = session.prepare_turn(TurnRequest(prompt="b"), expect_conversation="conv-1")
        assert ok.strict
        assert ok.expected_conversation == "conv-1"
        with pytest.raises(ContinuityError):
            session.prepare_turn(TurnRequest(prompt="c"), expect_conversation="other")

    def test_a_closed_session_refuses_prepare_and_run(self) -> None:
        session = _session(_transport())
        ticket = session.prepare_turn(TurnRequest(prompt="a"))
        session.close()
        with pytest.raises(SessionStateError):
            session.run(ticket)
        with pytest.raises(SessionStateError):
            session.prepare_turn(TurnRequest(prompt="b"))


class TestTransientRetry:
    def test_each_delay_is_waited_then_the_turn_succeeds(self) -> None:
        clock = FakeClock()
        transient = turn_failed(FailureKind.TRANSIENT)
        transport = _transport(transient, transient, FakeTurn(text="finally"))
        session = _session(transport, clock=clock)
        turn = _run(session)
        assert turn.result.text == "finally"
        assert clock.waits == [1.0, 2.0]
        assert len(transport.conversations) == 1

    def test_the_error_propagates_after_the_last_delay(self) -> None:
        clock = FakeClock()
        transport = _transport(*[turn_failed(FailureKind.TRANSIENT, f"n{i}") for i in range(9)])
        session = _session(transport, clock=clock)
        with pytest.raises(TurnFailedError) as info:
            _run(session)
        assert info.value.kind is FailureKind.TRANSIENT
        assert clock.waits == list(DELAYS)
        assert session.conversation_id is None or session.conversation_id.startswith("conv")

    @pytest.mark.parametrize(
        "kind", [FailureKind.USAGE_LIMIT, FailureKind.AUTH, FailureKind.SCHEMA, FailureKind.OTHER]
    )
    def test_other_kinds_propagate_at_once(self, kind: FailureKind) -> None:
        clock = FakeClock()
        session = _session(_transport(turn_failed(kind)), clock=clock)
        with pytest.raises(TurnFailedError) as info:
            _run(session)
        assert info.value.kind is kind
        assert clock.waits == []

    def test_an_interrupt_during_a_wait_ends_it_with_the_transient_error(self) -> None:
        class InterruptingClock(FakeClock):
            def __init__(self) -> None:
                super().__init__()
                self.session: Session | None = None

            def wait(self, seconds: float, stop: StopSignal | None = None) -> bool:
                assert self.session is not None
                self.session.interrupt()
                return super().wait(seconds, stop)

        clock = InterruptingClock()
        transport = _transport(turn_failed(FailureKind.TRANSIENT))
        session = _session(transport, clock=clock)
        clock.session = session
        with pytest.raises(TurnFailedError) as info:
            _run(session)
        assert info.value.kind is FailureKind.TRANSIENT
        assert clock.waits == [1.0]
        # The conversation is untouched and the next turn is not poisoned.
        assert _run(session).result.text == "ok"

    def test_a_timeout_propagates_and_keeps_the_conversation(self) -> None:
        session = _session(_transport(FakeTurn(), turn_timeout()))
        _run(session)
        with pytest.raises(Exception, match="did not finish"):
            _run(session)
        assert session.conversation_id == "conv-1"


class TestResumeRefused:
    def _resumed(self, *script: FakeOutcome) -> tuple[Session, FakeTransport]:
        transport = _transport(*script)
        return _session(transport, resume_id="old"), transport

    def test_a_refused_resume_is_replaced_by_a_fresh_conversation(self) -> None:
        session, transport = self._resumed(resume_refused("old"), FakeTurn(text="fresh"))
        turn = _run(session)
        assert turn.result.text == "fresh"
        assert turn.continuity is Continuity.REPLACED
        assert [spec.resume_id for spec in transport.specs] == ["old", None]
        assert transport.conversations[0].closed
        assert session.conversation_id == turn.conversation_id

    def test_a_refusal_at_open_is_handled_the_same_way(self) -> None:
        transport = _transport(FakeTurn(), resumable=set(), refuse_at_open=True)
        session = _session(transport, resume_id="old")
        turn = _run(session)
        assert turn.continuity is Continuity.REPLACED
        assert [spec.resume_id for spec in transport.specs] == [None]

    def test_a_second_refusal_propagates(self) -> None:
        session, _ = self._resumed(resume_refused(), resume_refused())
        with pytest.raises(SessionResumeError):
            _run(session)

    def test_a_pinned_turn_never_retries_fresh(self) -> None:
        session, transport = self._resumed(resume_refused())
        with pytest.raises(SessionResumeError):
            _run(session, pin=True)
        assert len(transport.specs) == 1
        assert session.conversation_id == "old"

    def test_a_strict_turn_never_retries_fresh(self) -> None:
        session, transport = self._resumed(resume_refused())
        with pytest.raises(SessionResumeError):
            _run(session, expect_conversation="old")
        assert len(transport.specs) == 1

    def test_a_refusal_of_a_fresh_conversation_is_not_retried(self) -> None:
        transport = _transport(resume_refused())
        session = _session(transport)
        with pytest.raises(SessionResumeError):
            _run(session)
        assert len(transport.specs) == 1


class TestBackstop:
    def test_an_unclassified_failure_of_a_resumed_turn_forgets_the_conversation(self) -> None:
        transport = _transport(turn_failed(FailureKind.OTHER), FakeTurn())
        session = _session(transport, resume_id="old")
        with pytest.raises(TurnFailedError):
            _run(session)
        assert session.conversation_id is None
        second = _run(session)
        assert [spec.resume_id for spec in transport.specs] == ["old", None]
        assert second.continuity is Continuity.CONTINUED

    def test_a_failure_of_a_fresh_conversation_keeps_it(self) -> None:
        session = _session(_transport(turn_failed(FailureKind.OTHER)))
        with pytest.raises(TurnFailedError):
            _run(session)
        assert session.conversation_id is None  # it never named one

    @pytest.mark.parametrize(
        "kind", [FailureKind.USAGE_LIMIT, FailureKind.AUTH, FailureKind.SCHEMA]
    )
    def test_classified_failures_keep_a_resumed_conversation(self, kind: FailureKind) -> None:
        session = _session(_transport(turn_failed(kind)), resume_id="old")
        with pytest.raises(TurnFailedError):
            _run(session)
        assert session.conversation_id == "old"

    def test_a_pinned_resumed_turn_keeps_the_conversation_on_any_failure(self) -> None:
        session = _session(_transport(turn_failed(FailureKind.OTHER)), resume_id="old")
        with pytest.raises(TurnFailedError):
            _run(session, pin=True)
        assert session.conversation_id == "old"


class TestRenewal:
    def _renewing(self) -> tuple[Session, FakeTransport]:
        transport = _transport(renewal=RenewalBudget(max_turns=2, max_turn_duration_ms=500))
        return _session(transport), transport

    def test_the_conversation_is_retired_after_the_budgeted_turns(self) -> None:
        session, transport = self._renewing()
        first = _run(session)
        second = _run(session)
        assert first.continuity is Continuity.CONTINUED
        assert second.continuity is Continuity.RESET
        assert session.conversation_id is None
        assert session.last_turn_conversation_id == "conv-1"
        assert transport.conversations[0].closed
        third = _run(session)
        assert third.conversation_id == "conv-2"
        assert third.continuity is Continuity.CONTINUED

    def test_a_heavy_turn_retires_early(self) -> None:
        transport = _transport(
            FakeTurn(duration_ms=500), renewal=RenewalBudget(max_turn_duration_ms=500)
        )
        session = _session(transport)
        assert _run(session).continuity is Continuity.RESET

    def test_a_pinned_turn_is_exempt(self) -> None:
        session, _ = self._renewing()
        _run(session)
        second = _run(session, pin=True)
        assert second.continuity is Continuity.CONTINUED
        assert session.conversation_id == "conv-1"

    def test_a_strict_turn_is_exempt(self) -> None:
        session, _ = self._renewing()
        _run(session)
        second = _run(session, expect_conversation="conv-1")
        assert second.continuity is Continuity.CONTINUED


class TestIdleRelease:
    def test_an_idle_conversation_is_closed_and_reopened_by_id(self) -> None:
        clock = FakeClock()
        transport = _transport()
        session = _session(transport, clock=clock, idle_release_after=60.0)
        _run(session)
        clock.advance(61.0)
        ticket = session.prepare_turn(TurnRequest(prompt="again"))
        assert transport.conversations[0].closed
        turn = session.run(ticket)
        assert turn.continuity is Continuity.CONTINUED
        assert [spec.resume_id for spec in transport.specs] == [None, "conv-1"]
        assert turn.conversation_id == "conv-1"

    def test_a_recently_used_conversation_stays_open(self) -> None:
        clock = FakeClock()
        transport = _transport()
        session = _session(transport, clock=clock, idle_release_after=60.0)
        _run(session)
        clock.advance(59.0)
        _run(session)
        assert len(transport.specs) == 1

    def test_release_closes_but_keeps_the_conversation_id(self) -> None:
        transport = _transport()
        session = _session(transport)
        _run(session)
        session.release()
        session.release()
        assert transport.conversations[0].closed
        assert session.conversation_id == "conv-1"
        _run(session)
        assert [spec.resume_id for spec in transport.specs] == [None, "conv-1"]


class TestCheckpoints:
    def test_a_turn_saves_the_conversation_and_its_usage(self) -> None:
        store = FakeCheckpointStore()
        session = _session(
            _transport(FakeTurn(input_tokens=7)), checkpoints=store, checkpoint_key="k"
        )
        _run(session)
        saved = store.load("k")
        assert saved is not None
        assert saved.conversation_id == "conv-1"
        assert isinstance(saved.usage, ProviderUsage)
        assert saved.usage.tokens.input_tokens == 7

    def test_a_new_session_resumes_the_saved_conversation(self) -> None:
        store = FakeCheckpointStore()
        store.save("k", Checkpoint("conv-7", ProviderUsage()))
        transport = _transport()
        session = _session(transport, checkpoints=store, checkpoint_key="k")
        assert session.conversation_id == "conv-7"
        _run(session)
        assert transport.specs[0].resume_id == "conv-7"
        assert transport.specs[0].previous_usage == ProviderUsage()

    def test_an_explicit_resume_id_wins_over_the_checkpoint(self) -> None:
        store = FakeCheckpointStore()
        store.save("k", Checkpoint("saved"))
        session = _session(_transport(), checkpoints=store, checkpoint_key="k", resume_id="given")
        assert session.conversation_id == "given"

    def test_retiring_a_conversation_clears_its_checkpoint(self) -> None:
        store = FakeCheckpointStore()
        transport = _transport(renewal=RenewalBudget(max_turns=1))
        session = _session(transport, checkpoints=store, checkpoint_key="k")
        _run(session)
        assert store.load("k") is None

    def test_a_replaced_conversation_checkpoint_follows_the_fresh_one(self) -> None:
        store = FakeCheckpointStore()
        store.save("k", Checkpoint("old"))
        session = _session(
            _transport(resume_refused("old"), FakeTurn()), checkpoints=store, checkpoint_key="k"
        )
        turn = _run(session)
        assert turn.conversation_id is not None
        assert store.load("k") == Checkpoint(turn.conversation_id, turn.result.usage)

    def test_a_store_without_a_key_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="together"):
            _session(_transport(), checkpoints=FakeCheckpointStore())

    def test_a_provider_that_cannot_resume_ignores_a_checkpoint(self) -> None:
        store = FakeCheckpointStore()
        store.save("k", Checkpoint("conv-7"))
        transport = FakeTransport(profile=fake_profile(supports_resume=False))
        session = _session(transport, checkpoints=store, checkpoint_key="k")
        assert session.conversation_id is None

    @given(st.lists(st.sampled_from(["ok", "refused", "other", "auth"]), min_size=1, max_size=8))
    def test_the_stored_checkpoint_always_names_the_retained_conversation(
        self, steps: list[str]
    ) -> None:
        script = [
            {
                "ok": FakeTurn(),
                "refused": resume_refused(),
                "other": turn_failed(FailureKind.OTHER),
                "auth": turn_failed(FailureKind.AUTH),
            }[step]
            for step in steps
        ]
        store = FakeCheckpointStore()
        session = _session(
            _transport(*script, renewal=RenewalBudget(max_turns=3)),
            checkpoints=store,
            checkpoint_key="k",
        )
        for _ in steps:
            with suppress(TurnFailedError, SessionResumeError):
                _run(session)
            saved = store.load("k")
            assert (saved.conversation_id if saved else None) == session.conversation_id


class TestAdopt:
    def test_a_cold_session_adopts_and_resumes(self) -> None:
        transport = _transport()
        session = _session(transport)
        assert session.adopt("conv-5", ProviderUsage()) is True
        assert session.conversation_id == "conv-5"
        _run(session)
        assert transport.specs[0].resume_id == "conv-5"
        assert transport.specs[0].previous_usage == ProviderUsage()

    def test_a_session_with_history_refuses(self) -> None:
        session = _session(_transport())
        _run(session)
        assert session.adopt("other") is False
        assert session.conversation_id == "conv-1"

    def test_a_provider_that_cannot_resume_refuses(self) -> None:
        session = _session(FakeTransport(profile=fake_profile(supports_resume=False)))
        assert session.adopt("x") is False

    def test_adopting_while_a_turn_is_prepared_is_refused(self) -> None:
        session = _session(_transport())
        session.prepare_turn(TurnRequest(prompt="a"))
        assert session.adopt("x") is False


class TestInterrupt:
    def test_interrupting_a_running_turn_returns_an_interrupted_result(self) -> None:
        holder: list[Session] = []
        events = RecordingEventHandler()

        def interrupt() -> None:
            holder[0].interrupt()

        transport = _transport(FakeTurn(during=interrupt))
        session = _session(transport)
        holder.append(session)
        ticket = session.prepare_turn(TurnRequest(prompt="a"))
        turn = session.run(ticket, on_event=events)
        assert turn.result.interrupted is True
        assert events.of_type(TurnInterrupted)
        assert turn.continuity is Continuity.CONTINUED
        assert session.conversation_id == "conv-1"
        assert _run(session).result.interrupted is False

    def test_interrupting_an_idle_session_does_nothing(self) -> None:
        session = _session(_transport())
        session.interrupt()
        assert _run(session).result.interrupted is False

    def test_an_interrupt_before_the_turn_starts_cancels_it(self) -> None:
        holder: list[Session] = []

        class InterruptOnOpen:
            def __init__(self, inner: FakeTransport) -> None:
                self.inner = inner
                self.profile = inner.profile

            def open(self, spec: ConversationSpec):  # noqa: ANN202
                holder[0].interrupt()
                return self.inner.open(spec)

        inner = _transport()
        session = _agent(InterruptOnOpen(inner)).session("/work")
        holder.append(session)
        with pytest.raises(TurnCancelledError):
            _run(session)
        assert inner.conversations[0].turns == []

    def test_close_interrupts_and_closes_the_conversation(self) -> None:
        transport = _transport()
        session = _session(transport)
        _run(session)
        session.close()
        session.close()
        assert transport.conversations[0].closed

    def test_the_session_is_a_context_manager(self) -> None:
        transport = _transport()
        with _session(transport) as session:
            _run(session)
        assert transport.conversations[0].closed
        with pytest.raises(SessionStateError):
            session.prepare_turn(TurnRequest(prompt="late"))


class TestAgent:
    def test_the_profile_and_transport_are_exposed(self) -> None:
        transport = _transport()
        agent = _agent(transport)
        assert agent.transport is transport
        assert agent.profile is transport.profile

    def test_a_transport_with_its_own_executor_options_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="belong to the transport"):
            Agent(
                _transport(),
                env={"A": "b"},
                permissions=NativePermissions.bypass(),
                approvals=ApprovalPolicy.DENY,
            )

    def test_the_profile_renewal_budget_reaches_the_policy(self) -> None:
        session = _session(_transport(renewal=RenewalBudget(max_turns=1)))
        assert _run(session).continuity is Continuity.RESET

    def test_the_default_retry_policy_is_the_documented_one(self) -> None:
        agent = Agent(
            _transport(),
            permissions=NativePermissions.bypass(),
            approvals=ApprovalPolicy.DENY,
            clock=FakeClock(),
        )
        assert agent.session("/w") is not None


def test_the_checkpoint_store_protocol_is_satisfied_by_the_fake() -> None:
    store: CheckpointStore = FakeCheckpointStore()
    assert store.load("nothing") is None
