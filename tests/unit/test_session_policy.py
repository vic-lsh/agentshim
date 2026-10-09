"""The session policy, as a pure function and as a model driven by random scripts.

Nothing here has a clock, a thread, or a conversation: the tests play the
shell's part, answering each command with an event, and check what the policy
did against definitions written independently of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from agentshim import (
    Continuity,
    FailureKind,
    PolicyConfig,
    RenewalBudget,
    RetryPolicy,
    SessionPolicy,
    SessionResumeError,
    SessionState,
    TurnCancelledError,
    TurnFailedError,
    TurnTimeoutError,
)
from agentshim.core.session_policy import (
    Adopt,
    AdoptVerdict,
    ClearCheckpoint,
    Close,
    CloseConversation,
    Command,
    Event,
    Execute,
    Fail,
    Failed,
    Finish,
    Open,
    Opened,
    Outcome,
    Prepare,
    RefusalReason,
    Refuse,
    Release,
    RunBegan,
    SaveCheckpoint,
    Start,
    Succeeded,
    Wait,
    WaitDone,
    outcome_of,
)
from hypothesis import given
from hypothesis import strategies as st

# -- the shell's side of the conversation, as a model -------------------------


@dataclass(frozen=True)
class Ok:
    input_tokens: int = 0
    duration_ms: int = 0
    interrupted: bool = False
    names_conversation: bool = True


@dataclass(frozen=True)
class Bad:
    outcome: Outcome


@dataclass(frozen=True)
class Attempt:
    """What the world does to one attempt: maybe fail to open, then succeed or fail."""

    open_fails: Outcome | None
    result: Ok | Bad


@dataclass(frozen=True)
class TurnPlan:
    gap: float
    strict: bool
    pinned: bool
    attempts: tuple[Attempt, ...]
    #: The interrupt arrives during this (0-based) retry wait, if any.
    stop_at_wait: int | None = None


def turn_plan(
    *attempts: Attempt, strict: bool = False, pinned: bool = False, gap: float = 0.0
) -> TurnPlan:
    """A turn with keyword flags, so call sites read."""
    return TurnPlan(gap=gap, strict=strict, pinned=pinned, attempts=attempts)


@dataclass
class World:
    """The effects of the commands, as a shell would leave them."""

    checkpoint: str | None = None
    live_id: str | None = None
    live: bool = False
    next_id: int = 0
    now: float = 0.0
    opened: list[str | None] = field(default_factory=list)
    waits_seen: int = 0
    stop_at_wait: int | None = None

    def fresh_id(self) -> str:
        self.next_id += 1
        return f"c{self.next_id}"


@dataclass
class TurnTrace:
    """What one turn did: the commands the policy issued and the events it was sent."""

    commands: list[Command] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    #: Each event with the commands the policy answered it with.
    steps: list[tuple[Event, tuple[Command, ...]]] = field(default_factory=list)
    checkpoint_after: str | None = None
    refused: RefusalReason | None = None
    finish: Finish | None = None
    failed: bool = False
    expected: str | None = None
    keeps: bool = False
    resumed: bool = False


def run_plan(
    policy: SessionPolicy, state: SessionState, world: World, plan: TurnPlan
) -> tuple[SessionState, TurnTrace]:
    """Play one turn against *policy*, answering commands from *plan*."""
    trace = TurnTrace()
    world.now += plan.gap
    expect = state.conversation_id if plan.strict else None
    trace.expected = state.conversation_id
    trace.keeps = expect is not None or plan.pinned
    attempts = list(plan.attempts)
    world.waits_seen = 0
    world.stop_at_wait = plan.stop_at_wait

    def send(event: Event) -> tuple[Command, ...]:
        nonlocal state
        trace.events.append(event)
        state, commands = policy.step(state, event)
        trace.commands.extend(commands)
        trace.steps.append((event, commands))
        return commands

    commands = send(Prepare(world.now, expect_conversation=expect, pin=plan.pinned))
    for command in commands:
        _effect(command, world)
        if isinstance(command, Refuse):
            trace.refused = command.reason
            trace.checkpoint_after = world.checkpoint
            return state, trace
    trace.resumed = state.conversation_id is not None
    commands = send(RunBegan(world.now))
    while True:
        answer: Event | None = None
        for command in commands:
            _effect(command, world)
            if isinstance(command, Finish):
                trace.finish = command
            elif isinstance(command, Fail):
                trace.failed = True
            else:
                answer = _answer(command, world, attempts) or answer
        if trace.finish is not None or trace.failed:
            trace.checkpoint_after = world.checkpoint
            return state, trace
        assert answer is not None, f"no way forward from {commands}"
        commands = send(answer)


def _effect(command: Command, world: World) -> None:
    if isinstance(command, CloseConversation):
        world.live = False
        world.live_id = None
    elif isinstance(command, SaveCheckpoint):
        world.checkpoint = command.conversation_id
    elif isinstance(command, ClearCheckpoint):
        world.checkpoint = None


def _next_attempt(attempts: list[Attempt]) -> Attempt:
    return attempts.pop(0) if attempts else Attempt(None, Ok())


def _answer(command: Command, world: World, attempts: list[Attempt]) -> Event | None:
    if isinstance(command, Open):
        attempt = _next_attempt(attempts)
        attempts.insert(0, attempt)
        world.opened.append(command.resume_id)
        if attempt.open_fails is not None:
            attempts.pop(0)
            return Failed(world.now, attempt.open_fails)
        world.live = True
        world.live_id = command.resume_id
        return Opened()
    if isinstance(command, Execute):
        attempt = _next_attempt(attempts)
        result = attempt.result
        if isinstance(result, Bad):
            return Failed(world.now, result.outcome, world.live_id)
        if result.names_conversation and world.live_id is None:
            world.live_id = world.fresh_id()
        return Succeeded(
            world.now,
            world.live_id if result.names_conversation else None,
            input_tokens=result.input_tokens,
            duration_ms=result.duration_ms,
            interrupted=result.interrupted,
        )
    if isinstance(command, Wait):
        stopped = world.stop_at_wait == world.waits_seen
        world.waits_seen += 1
        return WaitDone(world.now, stopped=stopped)
    return None


# -- strategies ----------------------------------------------------------------

# Refusals and overloads are the interesting ones: weight them.
outcomes = st.sampled_from(
    [*Outcome, Outcome.RESUME_REFUSED, Outcome.RESUME_REFUSED, Outcome.TRANSIENT, Outcome.OTHER]
)
oks = st.builds(
    Ok,
    input_tokens=st.integers(0, 100),
    duration_ms=st.integers(0, 100),
    interrupted=st.booleans(),
    names_conversation=st.booleans(),
)
attempts = st.builds(
    Attempt,
    open_fails=st.none() | outcomes,
    result=oks | st.builds(Bad, outcome=outcomes),
)
plans = st.builds(
    TurnPlan,
    gap=st.floats(0, 30),
    strict=st.booleans(),
    pinned=st.booleans(),
    attempts=st.lists(attempts, max_size=8).map(tuple),
    stop_at_wait=st.none() | st.integers(0, 3),
)
budgets = st.none() | st.builds(
    RenewalBudget,
    max_turns=st.none() | st.integers(1, 4),
    max_turn_input_tokens=st.none() | st.integers(1, 120),
    max_turn_duration_ms=st.none() | st.integers(1, 120),
)
configs = st.builds(
    PolicyConfig,
    retry=st.builds(RetryPolicy, delays=st.lists(st.floats(0, 100), max_size=4).map(tuple)),
    renewal=budgets,
    supports_resume=st.just(value=True),
    idle_release_after=st.none() | st.floats(1, 30),
)


@dataclass
class Played:
    policy: SessionPolicy
    world: World
    initial: str | None
    turns: list[tuple[SessionState, TurnTrace]]


def play(config: PolicyConfig, initial: str | None, turn_plans: list[TurnPlan]) -> Played:
    policy = SessionPolicy(config)
    state, _ = policy.step(policy.initial_state(), Start(initial))
    world = World(checkpoint=initial)
    played: list[tuple[SessionState, TurnTrace]] = []
    for plan in turn_plans:
        state, trace = run_plan(policy, state, world, plan)
        played.append((state, trace))
    return Played(policy, world, initial, played)


def scenarios() -> st.SearchStrategy[Played]:
    return st.builds(
        play, configs, st.none() | st.just("c0"), st.lists(plans, min_size=1, max_size=6)
    )


def completed(played: Played) -> list[tuple[SessionState, TurnTrace]]:
    return [(state, trace) for state, trace in played.turns if trace.refused is None]


# -- properties ---------------------------------------------------------------


def _refusals(trace: TurnTrace) -> list[tuple[Event, tuple[Command, ...]]]:
    return [
        (event, commands)
        for event, commands in trace.steps
        if isinstance(event, Failed) and event.outcome is Outcome.RESUME_REFUSED
    ]


def _fresh_retries(trace: TurnTrace) -> int:
    return sum(1 for _, commands in _refusals(trace) if Open(None) in commands)


@given(scenarios())
def test_a_turn_always_ends_in_exactly_one_finish_or_fail(played: Played) -> None:
    for state, trace in completed(played):
        assert (trace.finish is not None) != trace.failed
        assert state.turn is None


@given(scenarios())
def test_a_strict_or_pinned_turn_never_loses_its_conversation(played: Played) -> None:
    for state, trace in completed(played):
        if trace.keeps and trace.expected is not None:
            assert state.conversation_id == trace.expected
            assert ClearCheckpoint() not in trace.commands
            assert Open(None) not in trace.commands


@given(scenarios())
def test_at_most_one_fresh_retry_per_turn(played: Played) -> None:
    for _, trace in completed(played):
        assert _fresh_retries(trace) <= 1


@given(scenarios())
def test_a_fresh_retry_only_replaces_a_resumed_conversation_the_turn_may_lose(
    played: Played,
) -> None:
    for _, trace in completed(played):
        if _fresh_retries(trace):
            assert trace.resumed
            assert not trace.keeps


@given(scenarios())
def test_waits_follow_transient_failures_and_walk_the_configured_delays(played: Played) -> None:
    delays = played.policy.config.retry.delays
    for _, trace in completed(played):
        spent = 0
        for event, commands in trace.steps:
            waits = [c for c in commands if isinstance(c, Wait)]
            if waits:
                assert isinstance(event, Failed)
                assert event.outcome is Outcome.TRANSIENT
                assert waits == [Wait(delays[spent])]
                spent += 1
            if Open(None) in commands and isinstance(event, Failed):
                spent = 0
            assert spent <= len(delays)


@given(scenarios())
def test_a_transient_failure_gives_up_only_after_every_delay_was_spent(played: Played) -> None:
    delays = played.policy.config.retry.delays
    for _, trace in completed(played):
        spent = 0
        for event, commands in trace.steps:
            if any(isinstance(c, Wait) for c in commands):
                spent += 1
            elif isinstance(event, Failed) and event.outcome is Outcome.TRANSIENT:
                assert Fail() in commands
                assert spent == len(delays)
            if Open(None) in commands and isinstance(event, Failed):
                spent = 0


@given(scenarios())
def test_a_stopped_wait_ends_the_turn_with_the_error_and_keeps_the_conversation(
    played: Played,
) -> None:
    for state, trace in completed(played):
        for index, (event, commands) in enumerate(trace.steps):
            if isinstance(event, WaitDone) and event.stopped:
                assert commands[-1] == Fail()
                assert Execute() not in commands
                assert index == len(trace.steps) - 1
                if trace.expected is not None and not _refusals(trace):
                    assert state.conversation_id == trace.expected


@given(scenarios())
def test_the_checkpoint_follows_the_retained_conversation_at_every_turn_end(
    played: Played,
) -> None:
    for state, trace in completed(played):
        assert trace.checkpoint_after == state.conversation_id


@given(scenarios())
def test_continuity_matches_its_definition(played: Played) -> None:
    for state, trace in completed(played):
        finish = trace.finish
        if finish is None:
            continue
        replaced = _fresh_retries(trace) == 1
        retired = ClearCheckpoint() in trace.steps[-1][1]
        if replaced:
            assert finish.continuity is Continuity.REPLACED
        elif retired:
            assert finish.continuity is Continuity.RESET
            assert not trace.keeps
        else:
            assert finish.continuity is Continuity.CONTINUED
            assert state.conversation_id == finish.conversation_id


@given(scenarios())
def test_the_last_turn_conversation_survives_a_reset(played: Played) -> None:
    for state, trace in completed(played):
        if trace.finish is not None:
            assert state.last_turn_conversation_id == trace.finish.conversation_id


@given(scenarios())
def test_an_unclassified_failure_of_a_resumed_turn_forgets_the_conversation(
    played: Played,
) -> None:
    for state, trace in completed(played):
        last = trace.events[-1]
        if (
            trace.failed
            and isinstance(last, Failed)
            and last.outcome is Outcome.OTHER
            and trace.resumed
            and not trace.keeps
            and not _refusals(trace)
        ):
            assert state.conversation_id is None
            assert trace.checkpoint_after is None


@given(scenarios())
def test_classified_failures_and_timeouts_keep_the_conversation(played: Played) -> None:
    kept = {Outcome.TRANSIENT, Outcome.USAGE_LIMIT, Outcome.AUTH, Outcome.SCHEMA, Outcome.TIMEOUT}
    for state, trace in completed(played):
        last = trace.events[-1]
        if (
            trace.failed
            and isinstance(last, Failed)
            and last.outcome in kept
            and trace.expected is not None
            and not _refusals(trace)
        ):
            assert state.conversation_id == trace.expected


@given(scenarios())
def test_only_a_successful_uninterrupted_turn_can_be_followed_by_a_reset(played: Played) -> None:
    for _, trace in completed(played):
        finish = trace.finish
        if finish is not None and finish.continuity is Continuity.RESET:
            succeeded = [e for e in trace.events if isinstance(e, Succeeded)]
            assert succeeded
            assert not succeeded[-1].interrupted


@given(scenarios())
def test_without_a_renewal_budget_nothing_is_ever_reset(played: Played) -> None:
    if played.policy.config.renewal is None:
        for _, trace in completed(played):
            if trace.finish is not None:
                assert trace.finish.continuity is not Continuity.RESET


@given(scenarios())
def test_idle_release_closes_at_prepare_and_the_turn_reopens_by_id(played: Played) -> None:
    for _, trace in completed(played):
        _, prepare_commands = trace.steps[0]
        if CloseConversation() in prepare_commands:
            assert played.policy.config.idle_release_after is not None
            opens = [c for c in trace.commands if isinstance(c, Open)]
            assert opens
            assert opens[0].resume_id == trace.expected


# -- examples that pin the headline behaviours ---------------------------------


def _drive(config: PolicyConfig, plans_: list[TurnPlan], initial: str | None = None) -> Played:
    return play(config, initial, plans_)


def test_outcome_of_classifies_by_base_error_types() -> None:
    assert outcome_of(SessionResumeError([], 1, "s")) is Outcome.RESUME_REFUSED
    assert outcome_of(TurnFailedError("x", kind=FailureKind.AUTH)) is Outcome.AUTH
    assert outcome_of(TurnFailedError("x", kind=FailureKind.TRANSIENT)) is Outcome.TRANSIENT
    assert outcome_of(TurnTimeoutError(1.0)) is Outcome.TIMEOUT
    assert outcome_of(TurnCancelledError("x")) is Outcome.INTERRUPTED
    assert outcome_of(ValueError("x")) is Outcome.ERROR


def test_a_transient_failure_waits_each_delay_then_gives_up() -> None:
    config = PolicyConfig(retry=RetryPolicy(delays=(1.0, 2.0)))
    transient = Attempt(None, Bad(Outcome.TRANSIENT))
    played = _drive(config, [turn_plan(*[transient] * 3)])
    trace = played.turns[0][1]
    assert [c.seconds for c in trace.commands if isinstance(c, Wait)] == [1.0, 2.0]
    assert trace.failed


def test_a_refused_resume_is_replaced_once_and_reported() -> None:
    config = PolicyConfig()
    refuse = Attempt(None, Bad(Outcome.RESUME_REFUSED))
    played = _drive(config, [turn_plan(refuse)], initial="old")
    _, trace = played.turns[0]
    assert trace.finish is not None
    assert trace.finish.continuity is Continuity.REPLACED
    assert trace.finish.conversation_id != "old"
    assert played.world.opened == ["old", None]


def test_a_pinned_turn_propagates_a_refused_resume() -> None:
    refuse = Attempt(None, Bad(Outcome.RESUME_REFUSED))
    played = _drive(PolicyConfig(), [turn_plan(refuse, pinned=True)], initial="old")
    state, trace = played.turns[0]
    assert trace.failed
    assert state.conversation_id == "old"
    assert played.world.checkpoint == "old"


def test_renewal_after_the_configured_turns() -> None:
    config = PolicyConfig(renewal=RenewalBudget(max_turns=2))
    plan = turn_plan()
    played = _drive(config, [plan, plan, plan])
    continuity = [t.finish.continuity for _, t in played.turns if t.finish]
    assert continuity == [Continuity.CONTINUED, Continuity.RESET, Continuity.CONTINUED]


def test_a_heavy_turn_retires_the_conversation() -> None:
    config = PolicyConfig(renewal=RenewalBudget(max_turn_input_tokens=50))
    heavy = Attempt(None, Ok(input_tokens=50))
    played = _drive(config, [turn_plan(heavy)])
    assert played.turns[0][1].finish is not None
    assert played.turns[0][1].finish.continuity is Continuity.RESET
    assert played.world.checkpoint is None


def test_adopt_is_refused_with_newer_history_and_accepted_on_a_cold_start() -> None:
    policy = SessionPolicy(PolicyConfig())
    cold = policy.initial_state()
    state, commands = policy.step(cold, Adopt("x"))
    assert commands == (AdoptVerdict(accepted=True),)
    assert state.conversation_id == "x"
    state, commands = policy.step(state, Adopt("y"))
    assert commands == (AdoptVerdict(accepted=False),)
    assert state.conversation_id == "x"


def test_adopt_is_refused_when_the_provider_cannot_resume() -> None:
    policy = SessionPolicy(PolicyConfig(supports_resume=False))
    _, commands = policy.step(policy.initial_state(), Adopt("x"))
    assert commands == (AdoptVerdict(accepted=False),)


def test_prepare_refuses_a_conversation_the_session_does_not_hold() -> None:
    policy = SessionPolicy(PolicyConfig())
    state, _ = policy.step(policy.initial_state(), Start("a"))
    _, commands = policy.step(state, Prepare(0.0, expect_conversation="b"))
    assert commands == (Refuse(RefusalReason.CONTINUITY),)


def test_prepare_refuses_after_close_and_while_running() -> None:
    policy = SessionPolicy(PolicyConfig())
    state, _ = policy.step(policy.initial_state(), Prepare(0.0))
    state, _ = policy.step(state, RunBegan(0.0))
    _, commands = policy.step(state, Prepare(1.0))
    assert commands == (Refuse(RefusalReason.BUSY),)
    closed, _ = policy.step(policy.initial_state(), Close())
    _, commands = policy.step(closed, Prepare(1.0))
    assert commands == (Refuse(RefusalReason.CLOSED),)


def test_release_keeps_the_conversation_id() -> None:
    policy = SessionPolicy(PolicyConfig())
    state, _ = policy.step(policy.initial_state(), Start("a"))
    state, _ = policy.step(state, Prepare(0.0))
    state, _ = policy.step(state, RunBegan(0.0))
    state, _ = policy.step(state, Opened())
    state, _ = policy.step(state, Succeeded(0.0, "a"))
    released, commands = policy.step(state, Release())
    assert commands == (CloseConversation(),)
    assert released.conversation_id == "a"
    assert not released.live


def test_an_event_out_of_order_is_a_programming_error() -> None:
    policy = SessionPolicy(PolicyConfig())
    with pytest.raises(ValueError, match="phase"):
        policy.step(policy.initial_state(), Opened())


def test_negative_delays_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        RetryPolicy(delays=(1.0, -1.0))


def test_the_default_delays_are_the_documented_ones() -> None:
    assert RetryPolicy().delays == (30, 60, 120, 240, 480)
