"""Scripted conversations, transports and checkpoint stores.

``FakeTransport`` plays the provider's side of a ``Transport``: each turn
follows the next entry of a script, so a test can say "overloaded twice, then
fine" or "the first conversation is gone" and drive a ``Session`` through it
without a process, a clock, or any waiting.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from agentshim.core.checkpoints import Checkpoint, InMemoryCheckpointStore
from agentshim.core.errors import (
    AgentShimError,
    FailureKind,
    ProviderCapabilityError,
    SessionResumeError,
    SessionStateError,
    TurnFailedError,
    TurnTimeoutError,
)
from agentshim.core.events import TurnInterrupted
from agentshim.core.turn import TurnResult
from agentshim.core.usage import ProviderUsage, TokenUsage
from agentshim.providers import get_provider

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Sequence

    from agentshim.core.conversation import Conversation, ConversationSpec
    from agentshim.core.events import AgentEvent
    from agentshim.core.permissions import NativeMode
    from agentshim.core.profile import ProviderProfile
    from agentshim.core.turn import TurnRequest


@dataclass(frozen=True)
class FakeTurn:
    """A scripted successful turn.

    ``conversation_id`` is the id the result names; by default the conversation
    keeps the id it has, or is given a fresh one on its first turn. ``during``
    runs in the middle of the turn on the calling thread: call
    ``session.interrupt()`` from it to interrupt deterministically.
    """

    text: str = "ok"
    conversation_id: str | None = None
    input_tokens: int = 0
    duration_ms: int = 0
    events: Sequence[AgentEvent] = ()
    during: Callable[[], None] | None = None


FakeOutcome = FakeTurn | AgentShimError


def turn_failed(kind: FailureKind = FailureKind.OTHER, detail: str = "failed") -> TurnFailedError:
    """A scripted failure of the given kind."""
    return TurnFailedError(f"turn failed ({kind.value}): {detail}", kind=kind, detail=detail)


def resume_refused(conversation_id: str = "gone") -> SessionResumeError:
    """A scripted refusal to resume a conversation."""
    return SessionResumeError((), 1, conversation_id, detail="conversation not found")


def turn_timeout(timeout: float = 1.0) -> TurnTimeoutError:
    """A scripted timeout."""
    return TurnTimeoutError(timeout)


def fake_profile(*, supports_resume: bool = True) -> ProviderProfile:
    """A profile for the fake provider: bypass-only, no renewal, resumable by default."""
    base = get_provider("claude").profile
    return replace(
        base, name="fake", display_name="Fake", binary="fake", supports_resume=supports_resume
    )


class FakeConversation:
    """A ``Conversation`` that follows its transport's script."""

    def __init__(self, transport: FakeTransport, spec: ConversationSpec) -> None:
        """Start with the spec's ``resume_id`` as the conversation id."""
        self._transport = transport
        self.spec = spec
        self._id = spec.resume_id
        self._resumed = spec.resume_id is not None
        self._lock = threading.Lock()
        self._running = False
        self._interrupted = False
        self.turns: list[TurnRequest] = []
        self.interrupts = 0
        self.close_calls = 0

    @property
    def conversation_id(self) -> str | None:
        """The conversation's id, once it has one."""
        return self._id

    @property
    def closed(self) -> bool:
        """Whether ``close`` has been called."""
        return self.close_calls > 0

    def turn(self, request: TurnRequest, emit: Callable[[AgentEvent], None]) -> TurnResult:
        """Follow the next scripted outcome."""
        if self.closed:
            msg = "conversation is closed"
            raise SessionStateError(msg)
        self.turns.append(request)
        resumed = self._id is not None
        if resumed and not self._transport.can_resume(self._id):
            raise resume_refused(self._id or "")
        outcome = self._transport.next_outcome()
        if isinstance(outcome, AgentShimError):
            raise outcome
        with self._lock:
            self._running = True
            self._interrupted = False
        try:
            if outcome.during is not None:
                outcome.during()
            for event in outcome.events:
                emit(event)
            with self._lock:
                interrupted = self._interrupted
            if interrupted:
                emit(TurnInterrupted())
        finally:
            with self._lock:
                self._running = False
        self._id = outcome.conversation_id or self._id or self._transport.new_conversation_id()
        return TurnResult(
            text=outcome.text,
            structured_output=None,
            session_id=self._id,
            resumed=resumed,
            usage=ProviderUsage(tokens=TokenUsage(input_tokens=outcome.input_tokens)),
            cost_usd=None,
            duration_ms=outcome.duration_ms,
            exit_code=0,
            interrupted=interrupted,
        )

    def interrupt(self) -> None:
        """Mark the running turn interrupted; with none running, do nothing."""
        with self._lock:
            if self._running:
                self._interrupted = True
                self.interrupts += 1

    def close(self) -> None:
        """Record the close; idempotent."""
        self.close_calls += 1


class FakeTransport:
    """A ``Transport`` whose conversations follow one script, in turn order.

    ``script`` is consumed one entry per turn across all conversations; a
    ``FakeTurn`` succeeds and an ``AgentShimError`` is raised. Once it runs out
    every turn succeeds. ``resumable`` limits which ids can be resumed (any, by
    default); resuming another raises ``SessionResumeError`` at the first turn,
    or at ``open`` with ``refuse_at_open``. Everything opened is recorded.
    """

    def __init__(
        self,
        script: Sequence[FakeOutcome] = (),
        *,
        profile: ProviderProfile | None = None,
        resumable: Collection[str] | None = None,
        refuse_at_open: bool = False,
    ) -> None:
        """Bind the script and the resume behavior."""
        self._profile = profile if profile is not None else fake_profile()
        self._script = list(script)
        self._resumable = None if resumable is None else frozenset(resumable)
        self._refuse_at_open = refuse_at_open
        self._lock = threading.Lock()
        self._ids = 0
        self.specs: list[ConversationSpec] = []
        self.conversations: list[FakeConversation] = []

    @property
    def profile(self) -> ProviderProfile:
        """The fake provider's profile."""
        return self._profile

    def open(self, spec: ConversationSpec) -> Conversation:
        """Record *spec* and start a conversation, rejecting an unsupported mode."""
        if spec.permissions.mode not in self._profile.native_permission_modes:
            mode: NativeMode = spec.permissions.mode
            msg = f"{self._profile.name} cannot enforce native permission mode {mode.value!r}"
            raise ProviderCapabilityError(msg)
        if (
            spec.resume_id is not None
            and self._refuse_at_open
            and not self.can_resume(spec.resume_id)
        ):
            raise resume_refused(spec.resume_id)
        conversation = FakeConversation(self, spec)
        with self._lock:
            self.specs.append(spec)
            self.conversations.append(conversation)
        return conversation

    def can_resume(self, conversation_id: str | None) -> bool:
        """Whether the fake provider still has *conversation_id*."""
        return self._resumable is None or conversation_id in self._resumable

    def next_outcome(self) -> FakeOutcome:
        """Take the next scripted outcome, or a plain success once the script is spent."""
        with self._lock:
            return self._script.pop(0) if self._script else FakeTurn()

    def new_conversation_id(self) -> str:
        """Name a new conversation: ``conv-1``, ``conv-2``, ..."""
        with self._lock:
            self._ids += 1
            return f"conv-{self._ids}"


class FakeCheckpointStore(InMemoryCheckpointStore):
    """An in-memory store that also records every call made to it."""

    def __init__(self) -> None:
        """Start empty with no calls recorded."""
        super().__init__()
        self.calls: list[tuple[str, str]] = []

    def load(self, key: str) -> Checkpoint | None:
        """Record and answer a load."""
        self.calls.append(("load", key))
        return super().load(key)

    def save(self, key: str, checkpoint: Checkpoint) -> None:
        """Record and store a save."""
        self.calls.append(("save", key))
        super().save(key, checkpoint)

    def clear(self, key: str) -> None:
        """Record and apply a clear."""
        self.calls.append(("clear", key))
        super().clear(key)
