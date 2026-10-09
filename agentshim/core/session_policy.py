"""The recovery policy of a session, as a pure state machine.

``SessionPolicy.step(state, event)`` returns the next state and the commands
the caller must carry out. Nothing here does I/O, reads a clock, or starts a
thread: the clock reading arrives inside an event, and every effect (open a
conversation, run a turn, wait, save a checkpoint) leaves as a command. The
shell in ``agentshim/session.py`` is the interpreter. That split is what lets
a property test drive random event sequences against the whole policy.

What the policy decides, per turn:

* A ``TRANSIENT`` failure is retried in place after each delay of the
  ``RetryPolicy``; the waits end early when the shell reports a stop.
* A refused resume replaces the conversation once, unless the turn must not
  lose its conversation (``strict`` or ``pinned``).
* An unclassified failure of a resumed turn forgets the conversation.
* After a successful turn that exhausts the provider's ``RenewalBudget`` the
  conversation is retired.
* An idle conversation is released at prepare time and reopened by id.
* A checkpoint is saved whenever a conversation is retained, and cleared when
  it is retired, replaced, or forgotten.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING

from .errors import (
    FailureKind,
    SessionResumeError,
    TurnCancelledError,
    TurnFailedError,
    TurnTimeoutError,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from .profile import RenewalBudget

#: Waits before each transient retry, in seconds: about fifteen minutes in all,
#: long enough to outlast a typical provider overload.
DEFAULT_RETRY_DELAYS: tuple[float, ...] = (30, 60, 120, 240, 480)


class Continuity(Enum):
    """How a turn related to the conversation the caller expected it to run in."""

    #: It ran in the expected conversation (or the session had none yet), and
    #: that conversation is retained.
    CONTINUED = "continued"
    #: It ran in the expected conversation, which was retired afterwards: the
    #: next turn starts cold.
    RESET = "reset"
    #: It did not run in the expected conversation: a fresh one replaced it
    #: during the turn.
    REPLACED = "replaced"


@dataclass(frozen=True)
class RetryPolicy:
    """How long to wait before each retry of a transient failure.

    ``delays`` has one entry per retry, so ``len(delays)`` retries follow the
    first attempt. An empty tuple never retries.
    """

    delays: tuple[float, ...] = DEFAULT_RETRY_DELAYS

    def __post_init__(self) -> None:
        """Freeze the delays and reject a negative or non-finite one."""
        delays = tuple(float(delay) for delay in self.delays)
        if any(not 0 <= delay < float("inf") for delay in delays):
            msg = f"retry delays must be finite and non-negative, got {self.delays!r}"
            raise ValueError(msg)
        object.__setattr__(self, "delays", delays)


class Outcome(Enum):
    """How an attempt failed, as far as the policy cares."""

    TRANSIENT = "transient"
    USAGE_LIMIT = "usage_limit"
    AUTH = "auth"
    SCHEMA = "schema"
    #: A failure the provider could not classify.
    OTHER = "other"
    #: The provider no longer has the conversation the attempt tried to resume.
    RESUME_REFUSED = "resume_refused"
    TIMEOUT = "timeout"
    #: An interrupt arrived before the attempt could start.
    INTERRUPTED = "interrupted"
    #: Anything that says nothing about the conversation (bad request, spawn failure).
    ERROR = "error"


_OUTCOME_OF_KIND = {
    FailureKind.TRANSIENT: Outcome.TRANSIENT,
    FailureKind.USAGE_LIMIT: Outcome.USAGE_LIMIT,
    FailureKind.AUTH: Outcome.AUTH,
    FailureKind.SCHEMA: Outcome.SCHEMA,
    FailureKind.OTHER: Outcome.OTHER,
}


def outcome_of(error: BaseException) -> Outcome:
    """Classify *error* by the base error types a transport raises.

    Only the base types are consulted, so a transport never has to subclass
    anything specific to its implementation.
    """
    if isinstance(error, SessionResumeError):
        return Outcome.RESUME_REFUSED
    if isinstance(error, TurnFailedError):
        return _OUTCOME_OF_KIND[error.kind]
    if isinstance(error, TurnTimeoutError):
        return Outcome.TIMEOUT
    if isinstance(error, TurnCancelledError):
        return Outcome.INTERRUPTED
    return Outcome.ERROR


@dataclass(frozen=True)
class PolicyConfig:
    """The fixed inputs of a policy: what the provider and the caller allow."""

    retry: RetryPolicy = RetryPolicy()
    renewal: RenewalBudget | None = None
    supports_resume: bool = True
    #: Seconds of inactivity after which the live conversation is released at
    #: the next prepare; ``None`` never releases.
    idle_release_after: float | None = None


class Phase(Enum):
    """Where the turn in flight is."""

    PREPARED = "prepared"
    OPENING = "opening"
    RUNNING = "running"
    WAITING = "waiting"


@dataclass(frozen=True)
class TurnState:
    """The turn in flight."""

    phase: Phase
    #: The conversation the caller expected: the one the session held at prepare.
    expected: str | None
    #: The caller required that exact conversation (``expect_conversation``).
    strict: bool
    #: The caller asked to keep the conversation whatever happens (``pin``).
    pinned: bool
    #: This attempt continues an existing conversation.
    resumed: bool = False
    retries: int = 0
    replaced: bool = False

    @property
    def keeps_conversation(self) -> bool:
        """Whether the turn must never retire, replace, or forget the conversation."""
        return self.strict or self.pinned


@dataclass(frozen=True)
class SessionState:
    """Everything the policy remembers. Immutable: ``step`` returns a new one."""

    #: The provider conversation the next turn continues, or ``None`` for a cold start.
    conversation_id: str | None = None
    #: A ``Conversation`` is open right now.
    live: bool = False
    #: Successful turns in the retained conversation.
    turns: int = 0
    #: Where the last completed turn ran; a retirement does not clear it.
    last_turn_conversation_id: str | None = None
    last_activity: float | None = None
    closed: bool = False
    turn: TurnState | None = None


# -- events -----------------------------------------------------------------


@dataclass(frozen=True)
class Start:
    """The session begins, possibly with a conversation to resume."""

    conversation_id: str | None


@dataclass(frozen=True)
class Adopt:
    """The caller offers a conversation to continue."""

    conversation_id: str


@dataclass(frozen=True)
class Prepare:
    """The caller is preparing a turn. *now* is the clock reading."""

    now: float
    expect_conversation: str | None = None
    pin: bool = False


@dataclass(frozen=True)
class RunBegan:
    """The caller started running the prepared turn."""

    now: float


@dataclass(frozen=True)
class Opened:
    """A conversation was opened."""


@dataclass(frozen=True)
class Succeeded:
    """An attempt finished and produced a result."""

    now: float
    #: The id on the result, or ``None`` when the provider did not name one.
    conversation_id: str | None
    input_tokens: int = 0
    duration_ms: int = 0
    interrupted: bool = False


@dataclass(frozen=True)
class Failed:
    """An attempt (or opening a conversation) failed."""

    now: float
    outcome: Outcome
    #: The id the conversation names after the failure, if it does.
    conversation_id: str | None = None


@dataclass(frozen=True)
class WaitDone:
    """A retry wait ended; *stopped* says an interrupt ended it early."""

    now: float
    stopped: bool


@dataclass(frozen=True)
class Release:
    """The caller asked to release the live conversation."""


@dataclass(frozen=True)
class Close:
    """The caller closed the session."""


Event = (
    Start | Adopt | Prepare | RunBegan | Opened | Succeeded | Failed | WaitDone | Release | Close
)


# -- commands ---------------------------------------------------------------


class RefusalReason(Enum):
    """Why the policy refused an event."""

    #: ``expect_conversation`` is not the conversation the session holds.
    CONTINUITY = "continuity"
    #: A turn is already running.
    BUSY = "busy"
    CLOSED = "closed"


@dataclass(frozen=True)
class Open:
    """Open a conversation, resuming *resume_id* when it is not ``None``."""

    resume_id: str | None


@dataclass(frozen=True)
class Execute:
    """Run the turn in the live conversation."""


@dataclass(frozen=True)
class Wait:
    """Wait *seconds* (cancellably), then report ``WaitDone``."""

    seconds: float


@dataclass(frozen=True)
class CloseConversation:
    """Close the live conversation."""


@dataclass(frozen=True)
class SaveCheckpoint:
    """Save a checkpoint for *conversation_id*."""

    conversation_id: str


@dataclass(frozen=True)
class ClearCheckpoint:
    """Clear the checkpoint."""


@dataclass(frozen=True)
class Finish:
    """The turn is done: report *continuity* and the conversation it ran in."""

    continuity: Continuity
    conversation_id: str | None


@dataclass(frozen=True)
class Fail:
    """The turn is over: raise the error of the latest failed attempt."""


@dataclass(frozen=True)
class Refuse:
    """Refuse the event, for *reason*."""

    reason: RefusalReason


@dataclass(frozen=True)
class AdoptVerdict:
    """Answer an ``Adopt``."""

    accepted: bool


Command = (
    Open
    | Execute
    | Wait
    | CloseConversation
    | SaveCheckpoint
    | ClearCheckpoint
    | Finish
    | Fail
    | Refuse
    | AdoptVerdict
)

Step = tuple[SessionState, tuple[Command, ...]]


class SessionPolicy:
    """The pure step function of a session. Holds only its configuration."""

    def __init__(self, config: PolicyConfig | None = None) -> None:
        """Bind the configuration; the policy itself keeps no state."""
        self.config = config if config is not None else PolicyConfig()
        self._handlers: dict[type, Callable[..., Step]] = {
            Start: self._on_start,
            Adopt: self._on_adopt,
            Prepare: self._on_prepare,
            RunBegan: self._on_run_began,
            Opened: self._on_opened,
            Succeeded: self._on_succeeded,
            Failed: self._on_failed,
            WaitDone: self._on_wait_done,
            Release: self._on_release,
            Close: self._on_close,
        }

    def initial_state(self) -> SessionState:
        """A session that holds no conversation and has done nothing."""
        return SessionState()

    def step(self, state: SessionState, event: Event) -> Step:
        """Apply *event* to *state*.

        Raises ``ValueError`` for an event that cannot happen in *state* (for
        example ``Opened`` with no turn running): the shell, not the caller of
        the library, is at fault.
        """
        return self._handlers[type(event)](state, event)

    # -- session-level events

    def _on_start(self, state: SessionState, event: Start) -> Step:
        cid = event.conversation_id if self.config.supports_resume else None
        return replace(state, conversation_id=cid, turns=0), ()

    def _on_adopt(self, state: SessionState, event: Adopt) -> Step:
        """Accept only when nothing newer is held: no turn, no live history."""
        held = state.conversation_id
        acceptable = (
            self.config.supports_resume
            and not state.closed
            and state.turn is None
            and not state.live
            and held in (None, event.conversation_id)
        )
        if not acceptable:
            return state, (AdoptVerdict(accepted=False),)
        # Adopting the id already held keeps its turn count toward renewal.
        turns = state.turns if held == event.conversation_id else 0
        return replace(state, conversation_id=event.conversation_id, turns=turns), (
            AdoptVerdict(accepted=True),
        )

    def _on_release(self, state: SessionState, event: Release) -> Step:
        del event
        if not state.live or _running(state) or state.conversation_id is None:
            # Without an id there is nothing to reopen by: closing would lose the history.
            return state, ()
        return replace(state, live=False), (CloseConversation(),)

    def _on_close(self, state: SessionState, event: Close) -> Step:
        del event
        commands: tuple[Command, ...] = (CloseConversation(),) if state.live else ()
        return replace(state, live=False, closed=True), commands

    # -- turn events

    def _on_prepare(self, state: SessionState, event: Prepare) -> Step:
        if state.closed:
            return state, (Refuse(RefusalReason.CLOSED),)
        if _running(state):
            return state, (Refuse(RefusalReason.BUSY),)
        expect = event.expect_conversation
        if expect is not None and expect != state.conversation_id:
            return state, (Refuse(RefusalReason.CONTINUITY),)
        commands: tuple[Command, ...] = ()
        live = state.live
        if live and state.conversation_id is not None and self._idle_due(state, event.now):
            live = False
            commands = (CloseConversation(),)
        turn = TurnState(
            Phase.PREPARED,
            expected=state.conversation_id,
            strict=expect is not None,
            pinned=event.pin,
        )
        return replace(state, live=live, turn=turn), commands

    def _idle_due(self, state: SessionState, now: float) -> bool:
        limit = self.config.idle_release_after
        return (
            limit is not None
            and state.last_activity is not None
            and now - state.last_activity >= limit
        )

    def _on_run_began(self, state: SessionState, event: RunBegan) -> Step:
        del event
        turn = _turn_in(state, Phase.PREPARED)
        resumed = state.conversation_id is not None
        if state.live:
            turn = replace(turn, phase=Phase.RUNNING, resumed=resumed)
            return replace(state, turn=turn), (Execute(),)
        turn = replace(turn, phase=Phase.OPENING, resumed=resumed)
        return replace(state, turn=turn), (Open(state.conversation_id),)

    def _on_opened(self, state: SessionState, event: Opened) -> Step:
        del event
        turn = _turn_in(state, Phase.OPENING)
        return replace(state, live=True, turn=replace(turn, phase=Phase.RUNNING)), (Execute(),)

    def _on_succeeded(self, state: SessionState, event: Succeeded) -> Step:
        turn = _turn_in(state, Phase.RUNNING)
        cid = event.conversation_id or state.conversation_id
        turns = state.turns if event.interrupted else state.turns + 1
        retire = (
            not event.interrupted
            and not turn.keeps_conversation
            and self._exceeds_budget(turns, event)
        )
        commands: list[Command] = []
        if retire:
            if state.live:
                commands.append(CloseConversation())
            commands.append(ClearCheckpoint())
            retained_id, live, turns = None, False, 0
        else:
            retained_id, live = (cid if self.config.supports_resume else None), state.live
            if retained_id is not None:
                commands.append(SaveCheckpoint(retained_id))
        if turn.replaced:
            continuity = Continuity.REPLACED
        elif retire:
            continuity = Continuity.RESET
        else:
            continuity = Continuity.CONTINUED
        commands.append(Finish(continuity, cid))
        done = replace(
            state,
            conversation_id=retained_id,
            live=live,
            turns=turns,
            last_turn_conversation_id=cid,
            last_activity=event.now,
            turn=None,
        )
        return done, tuple(commands)

    def _exceeds_budget(self, turns: int, event: Succeeded) -> bool:
        budget = self.config.renewal
        if budget is None:
            return False
        return (
            _reached(turns, budget.max_turns)
            or _reached(event.input_tokens, budget.max_turn_input_tokens)
            or _reached(event.duration_ms, budget.max_turn_duration_ms)
        )

    def _on_failed(self, state: SessionState, event: Failed) -> Step:
        turn = _turn_in(state, Phase.OPENING, Phase.RUNNING)
        if (
            event.conversation_id is not None
            and self.config.supports_resume
            and event.outcome is not Outcome.RESUME_REFUSED
        ):
            state = replace(state, conversation_id=event.conversation_id)
        if state.closed:
            return _give_up(state, event.now)
        if event.outcome is Outcome.TRANSIENT:
            return self._retry_transient(state, turn, event)
        if event.outcome is Outcome.RESUME_REFUSED:
            if turn.resumed and not turn.keeps_conversation:
                return _replace_conversation(state, turn)
            return _give_up(state, event.now)
        if event.outcome is Outcome.OTHER and turn.resumed and not turn.keeps_conversation:
            return _forget_conversation(state, event.now)
        return _give_up(state, event.now)

    def _retry_transient(self, state: SessionState, turn: TurnState, event: Failed) -> Step:
        delays = self.config.retry.delays
        if turn.retries >= len(delays):
            return _give_up(state, event.now)
        waiting = replace(turn, phase=Phase.WAITING, retries=turn.retries + 1)
        return replace(state, turn=waiting), (Wait(delays[turn.retries]),)

    def _on_wait_done(self, state: SessionState, event: WaitDone) -> Step:
        turn = _turn_in(state, Phase.WAITING)
        if event.stopped or state.closed:
            return _give_up(state, event.now)
        if state.live:
            return replace(state, turn=replace(turn, phase=Phase.RUNNING)), (Execute(),)
        # The attempt that failed never got a conversation (opening it failed).
        reopening = replace(turn, phase=Phase.OPENING)
        return replace(state, turn=reopening), (Open(state.conversation_id),)


def _running(state: SessionState) -> bool:
    return state.turn is not None and state.turn.phase is not Phase.PREPARED


def _turn_in(state: SessionState, *phases: Phase) -> TurnState:
    turn = state.turn
    if turn is None or turn.phase not in phases:
        names = "/".join(phase.value for phase in phases)
        msg = f"event needs a turn in phase {names}, but the turn is {turn!r}"
        raise ValueError(msg)
    return turn


def _reached(value: int, limit: int | None) -> bool:
    return limit is not None and value >= limit


def _give_up(state: SessionState, now: float) -> Step:
    """End the turn with its error, keeping whatever conversation is held."""
    commands: list[Command] = []
    if state.conversation_id is not None:
        commands.append(SaveCheckpoint(state.conversation_id))
    commands.append(Fail())
    return replace(state, turn=None, last_activity=now), tuple(commands)


def _forget_conversation(state: SessionState, now: float) -> Step:
    """End the turn with its error and drop the conversation it was continuing."""
    commands: list[Command] = []
    if state.live:
        commands.append(CloseConversation())
    commands += [ClearCheckpoint(), Fail()]
    forgotten = replace(
        state, conversation_id=None, live=False, turns=0, turn=None, last_activity=now
    )
    return forgotten, tuple(commands)


def _replace_conversation(state: SessionState, turn: TurnState) -> Step:
    """Drop the refused conversation and retry the turn once in a fresh one."""
    commands: list[Command] = []
    if state.live:
        commands.append(CloseConversation())
    commands += [ClearCheckpoint(), Open(None)]
    fresh = replace(turn, phase=Phase.OPENING, resumed=False, retries=0, replaced=True)
    return replace(state, conversation_id=None, live=False, turns=0, turn=fresh), tuple(commands)
