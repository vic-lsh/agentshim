"""The session: a conversation that survives the failures conversations have.

A ``Session`` owns one logical conversation and keeps it going across a
provider's bad moments: overloads, a vanished conversation, a conversation
grown too large, a long idle gap, a process restart. The decisions are made by
the pure ``SessionPolicy`` (``agentshim/core/session_policy.py``); this module
is the thin shell that carries out its commands with a ``Transport``, a
``Clock``, an ``IdAllocator`` and a ``CheckpointStore``, and does nothing
else. Nothing here reads the wall clock or sleeps except through the injected
``Clock``.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from agentshim.core.checkpoints import Checkpoint
from agentshim.core.clock import StopSignal
from agentshim.core.errors import (
    AgentShimError,
    ContinuityError,
    SessionStateError,
    TurnCancelledError,
)
from agentshim.core.events import compose_event_handlers
from agentshim.core.mcp import StdioMcpServer
from agentshim.core.session_policy import (
    Adopt,
    AdoptVerdict,
    ClearCheckpoint,
    Close,
    CloseConversation,
    Execute,
    Fail,
    Failed,
    Finish,
    Open,
    Opened,
    Outcome,
    PolicyConfig,
    Prepare,
    RefusalReason,
    Refuse,
    Release,
    RunBegan,
    SaveCheckpoint,
    SessionPolicy,
    Start,
    Succeeded,
    Wait,
    WaitDone,
    outcome_of,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from agentshim.core.checkpoints import CheckpointStore
    from agentshim.core.clock import Clock
    from agentshim.core.conversation import Conversation, ConversationSpec, Transport
    from agentshim.core.events import AgentEvent, AgentEventHandler
    from agentshim.core.ids import IdAllocator
    from agentshim.core.mcp import McpServer
    from agentshim.core.session_policy import Command, Continuity, Event, SessionState
    from agentshim.core.turn import TurnRequest, TurnResult
    from agentshim.core.usage import ProviderUsage


@dataclass(frozen=True)
class TurnTicket:
    """A turn that has been prepared but not yet run.

    The id exists before dispatch so a caller can journal its intent to run the
    turn, and find it again after a crash.
    """

    turn_id: str
    request: TurnRequest
    #: The conversation the session held when the turn was prepared; the turn's
    #: continuity is judged against it.
    expected_conversation: str | None
    #: The caller required exactly that conversation (``expect_conversation``).
    strict: bool
    #: The caller asked to keep the conversation whatever happens (``pin``).
    pinned: bool


@dataclass(frozen=True)
class Turn:
    """A completed turn."""

    result: TurnResult
    continuity: Continuity
    turn_id: str
    #: The conversation the turn ran in. A retirement does not clear it.
    conversation_id: str | None


def map_mcp_servers(
    servers: Sequence[McpServer], agent_path: Callable[[str], str]
) -> tuple[McpServer, ...]:
    """Map the absolute paths in stdio MCP servers to where the agent sees them.

    A relative command or argument (a flag, a workspace-relative path) is left
    alone, as is a server reached over HTTP.
    """

    def mapped(value: str) -> str:
        return agent_path(value) if value.startswith("/") else value

    return tuple(
        replace(server, command=mapped(server.command), args=tuple(map(mapped, server.args)))
        if isinstance(server, StdioMcpServer)
        else server
        for server in servers
    )


class Session:
    """One logical conversation, kept alive by a recovery policy.

    Normally made by ``Agent.session``. One turn at a time: ``prepare_turn``
    then ``run``. ``interrupt`` may be called from any thread; everything else
    belongs to the thread driving the session.
    """

    # Every argument is an independent collaborator or option the Agent wires.
    def __init__(  # noqa: PLR0913
        self,
        transport: Transport,
        spec: ConversationSpec,
        *,
        clock: Clock,
        ids: IdAllocator,
        policy: PolicyConfig,
        event_handlers: Sequence[AgentEventHandler] = (),
        checkpoints: CheckpointStore | None = None,
        checkpoint_key: str | None = None,
        previous_usage: ProviderUsage | None = None,
        agent_path: Callable[[str], str] | None = None,
        host_cwd: str | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        """Bind a session to its collaborators and resume what it can.

        *spec* is the conversation template: its ``resume_id`` (with
        *previous_usage*) names a conversation to continue, and otherwise the
        checkpoint saved under *checkpoint_key* does. *policy* configures
        the recovery policy. When the agent is confined, *agent_path* maps host
        paths to the agent's view and *host_cwd* is the working directory on
        the host (``spec.cwd`` is the agent's view of it). Passing a store without a key, or the reverse, is an
        error rather than a session that silently never checkpoints.
        """
        if (checkpoints is None) != (checkpoint_key is None):
            msg = "checkpoints and checkpoint_key must be given together"
            raise ValueError(msg)
        self._transport = transport
        self._spec = spec
        self._clock = clock
        self._ids = ids
        self._handler = compose_event_handlers(None, event_handlers)
        self._store = checkpoints
        self._key = checkpoint_key
        self._agent_path = agent_path
        self._host_cwd = host_cwd if host_cwd is not None else spec.cwd
        self._log: Callable[[str], None] = log if log is not None else _discard
        self._policy = SessionPolicy(policy)
        self._lock = threading.Lock()
        self._state: SessionState = self._policy.initial_state()
        self._conversation: Conversation | None = None
        self._ticket: TurnTicket | None = None
        self._stop: StopSignal | None = None
        self._usage: dict[str, ProviderUsage] = {}
        resume_id = spec.resume_id
        usage = previous_usage if previous_usage is not None else spec.previous_usage
        if resume_id is None and checkpoints is not None and checkpoint_key is not None:
            saved = checkpoints.load(checkpoint_key)
            if saved is not None:
                resume_id, usage = saved.conversation_id, saved.usage
        if resume_id is not None and usage is not None:
            self._usage[resume_id] = usage
        self._step(Start(resume_id))

    @property
    def conversation_id(self) -> str | None:
        """The conversation the next turn continues, or ``None`` for a cold start."""
        with self._lock:
            return self._state.conversation_id

    @property
    def last_turn_conversation_id(self) -> str | None:
        """The conversation the last completed turn ran in; a retirement keeps it."""
        with self._lock:
            return self._state.last_turn_conversation_id

    def adopt(self, conversation_id: str, previous_usage: ProviderUsage | None = None) -> bool:
        """Continue a provider conversation from the next turn on.

        Returns ``False``, changing nothing, when the session already holds a
        conversation with newer history, a turn is prepared or running, or the
        provider cannot resume.
        """
        with self._lock:
            commands = self._step(Adopt(conversation_id))
        accepted = any(isinstance(c, AdoptVerdict) and c.accepted for c in commands)
        if accepted and previous_usage is not None:
            self._usage[conversation_id] = previous_usage
        return accepted

    def prepare_turn(
        self,
        request: TurnRequest,
        *,
        expect_conversation: str | None = None,
        pin: bool = False,
    ) -> TurnTicket:
        """Reserve a turn and name it, without running it.

        ``expect_conversation`` demands that this turn continue exactly that
        conversation: if the session holds another, or none, this raises
        ``ContinuityError`` and the turn is never run. ``pin`` keeps the
        conversation through renewal and never replaces a refused resume with a
        fresh one. A later ``prepare_turn`` supersedes this ticket.
        """
        now = self._clock.monotonic()
        with self._lock:
            commands = self._step(Prepare(now, expect_conversation, pin))
            refusal = next((c for c in commands if isinstance(c, Refuse)), None)
            if refusal is not None:
                raise self._refusal_error(refusal.reason, expect_conversation)
            ticket = TurnTicket(
                turn_id=self._ids.new_id("turn"),
                request=request,
                expected_conversation=self._state.conversation_id,
                strict=expect_conversation is not None,
                pinned=pin,
            )
            self._ticket = ticket
        self._perform_effects(commands)
        return ticket

    def _refusal_error(self, reason: RefusalReason, expected: str | None) -> AgentShimError:
        if reason is RefusalReason.CONTINUITY:
            return ContinuityError(expected, self._state.conversation_id)
        if reason is RefusalReason.BUSY:
            return SessionStateError("a turn is already running in this session")
        return SessionStateError("session is closed")

    def run(self, ticket: TurnTicket, *, on_event: AgentEventHandler | None = None) -> Turn:
        """Run the turn *ticket* names, to completion.

        *ticket* must be the most recent one prepared and not yet run, or this
        raises ``SessionStateError``. Events go to the agent's handlers and to
        *on_event*. Errors are the transport's: a failure the policy could not
        recover from is re-raised unchanged.
        """
        now = self._clock.monotonic()
        stop = StopSignal()
        with self._lock:
            if self._state.closed:
                msg = "session is closed"
                raise SessionStateError(msg)
            if ticket is not self._ticket:
                msg = f"turn ticket {ticket.turn_id!r} is stale or was already run"
                raise SessionStateError(msg)
            self._ticket = None
            self._stop = stop
            commands = self._step(RunBegan(now))
        try:
            return self._drive(ticket, on_event, stop, commands)
        finally:
            with self._lock:
                self._stop = None

    def interrupt(self) -> None:
        """End the running turn, keeping the conversation. Thread-safe.

        Also ends a retry wait. With no turn running it does nothing and leaves
        the next turn alone.
        """
        with self._lock:
            stop = self._stop
            conversation = self._conversation
        if stop is None:
            return
        stop.set()
        if conversation is not None:
            conversation.interrupt()

    def release(self) -> None:
        """Close the live conversation but keep its id, so the next turn resumes it."""
        with self._lock:
            commands = self._step(Release())
        self._perform_effects(commands)

    def close(self) -> None:
        """Interrupt any running turn and release everything. Idempotent."""
        with self._lock:
            if self._state.closed:
                return
            commands = self._step(Close())
            stop = self._stop
            conversation = self._conversation
        if stop is not None:
            stop.set()
        if conversation is not None:
            conversation.interrupt()
        self._perform_effects(commands)

    def __enter__(self) -> Session:
        """Return the session; it closes on exit."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the session."""
        self.close()

    # -- shell internals

    def _step(self, event: Event) -> tuple[Command, ...]:
        """Advance the policy. The caller holds the lock."""
        self._state, commands = self._policy.step(self._state, event)
        return commands

    def _apply(self, event: Event) -> tuple[Command, ...]:
        with self._lock:
            return self._step(event)

    def _drive(
        self,
        ticket: TurnTicket,
        on_event: AgentEventHandler | None,
        stop: StopSignal,
        commands: Sequence[Command],
    ) -> Turn:
        """Carry out the policy's commands, feeding back what happened, until it ends the turn.

        Each round of commands is some side effects followed by at most one
        command that needs an answer; the answer becomes the next event.
        """
        attempt = _Attempt(ticket, on_event, stop)
        while True:
            event: Event | None = None
            for command in commands:
                outcome = self._perform(command, attempt)
                if isinstance(outcome, Turn):
                    return outcome
                if outcome is not None:
                    event = outcome
            if event is None:
                msg = f"policy gave no command that continues the turn: {commands!r}"
                raise AssertionError(msg)
            commands = self._apply(event)

    def _perform(self, command: Command, attempt: _Attempt) -> Event | Turn | None:
        if isinstance(command, Open):
            return self._open(command.resume_id, attempt)
        if isinstance(command, Execute):
            return self._execute(attempt)
        if isinstance(command, Wait):
            self._log(f"retrying the turn in {command.seconds:g}s")
            stopped = self._clock.wait(command.seconds, attempt.stop)
            return WaitDone(self._clock.monotonic(), stopped)
        if isinstance(command, Finish):
            if attempt.result is None:
                msg = "policy finished a turn that produced no result"
                raise AssertionError(msg)
            return Turn(
                result=attempt.result,
                continuity=command.continuity,
                turn_id=attempt.ticket.turn_id,
                conversation_id=command.conversation_id,
            )
        if isinstance(command, Fail):
            if attempt.error is None:
                msg = "policy failed a turn that has no error"
                raise AssertionError(msg)
            raise attempt.error
        self._perform_effects((command,))
        return None

    def _open(self, resume_id: str | None, attempt: _Attempt) -> Event:
        try:
            conversation = self._transport.open(self._open_spec(resume_id))
        except AgentShimError as error:
            attempt.error = error
            return Failed(self._clock.monotonic(), outcome_of(error))
        except BaseException:
            self._apply(Failed(self._clock.monotonic(), Outcome.ERROR))
            raise
        with self._lock:
            closed = self._state.closed
            if not closed:
                self._conversation = conversation
        if closed:
            # close() ran while the transport was opening: nothing else holds
            # this conversation, so release it here instead of leaking it.
            conversation.close()
            attempt.error = TurnCancelledError("session closed while opening the conversation")
            return Failed(self._clock.monotonic(), Outcome.INTERRUPTED)
        return Opened()

    def _execute(self, attempt: _Attempt) -> Event:
        with self._lock:
            conversation = self._conversation
        if conversation is None:
            attempt.error = SessionStateError("session is closed")
            return Failed(self._clock.monotonic(), Outcome.ERROR)
        if attempt.stop.is_set():
            attempt.error = TurnCancelledError("turn interrupted before it started")
            return Failed(self._clock.monotonic(), Outcome.INTERRUPTED)
        request = self._request_for_agent(attempt.ticket.request)
        try:
            result = conversation.turn(request, self._emitter(attempt.on_event))
        except AgentShimError as error:
            attempt.error = error
            return Failed(self._clock.monotonic(), outcome_of(error), conversation.conversation_id)
        except BaseException:
            self._apply(Failed(self._clock.monotonic(), Outcome.ERROR))
            raise
        attempt.result = result
        cid = result.session_id or conversation.conversation_id
        if cid is not None and not result.interrupted:
            # An interrupted result carries no report, so it must not replace
            # the baseline the next turn's usage increment is measured from.
            self._usage[cid] = result.usage
        return Succeeded(
            self._clock.monotonic(),
            cid,
            input_tokens=result.usage.tokens.input_tokens,
            duration_ms=result.duration_ms,
            interrupted=result.interrupted,
        )

    def _emitter(self, on_event: AgentEventHandler | None) -> Callable[[AgentEvent], None]:
        """Return the callback that fans an event out to the agent's handlers and a turn's own."""

        def emit(event: AgentEvent) -> None:
            self._handler.on_event(event)
            if on_event is not None:
                on_event.on_event(event)

        return emit

    def _perform_effects(self, commands: Sequence[Command]) -> None:
        """Carry out the commands that are side effects only."""
        for command in commands:
            if isinstance(command, CloseConversation):
                self._close_conversation()
            elif self._store is not None and self._key is not None:
                self._perform_checkpoint(command, self._store, self._key)

    def _perform_checkpoint(self, command: Command, store: CheckpointStore, key: str) -> None:
        if isinstance(command, SaveCheckpoint):
            usage = self._usage.get(command.conversation_id)
            store.save(key, Checkpoint(command.conversation_id, usage))
        elif isinstance(command, ClearCheckpoint):
            store.clear(key)

    def _close_conversation(self) -> None:
        with self._lock:
            conversation, self._conversation = self._conversation, None
        if conversation is not None:
            conversation.close()

    def _request_for_agent(self, request: TurnRequest) -> TurnRequest:
        """Express a request's paths as the agent sees them, when confined."""
        if self._agent_path is None:
            return request
        schema = request.output_schema
        if schema is not None and schema.cli_dir is None:
            schema = replace(schema, cli_dir=self._agent_path(str(schema.host_dir)))
        workspace = request.mcp_workspace
        if workspace is None and request.cwd is None:
            workspace = Path(self._host_cwd)
        return replace(
            request,
            output_schema=schema,
            mcp_servers=map_mcp_servers(request.mcp_servers, self._agent_path),
            mcp_workspace=workspace,
        )

    def _open_spec(self, resume_id: str | None) -> ConversationSpec:
        usage = self._usage.get(resume_id) if resume_id is not None else None
        return replace(self._spec, resume_id=resume_id, previous_usage=usage)


class _Attempt:
    """What the current run has learned: its latest error and result."""

    def __init__(
        self, ticket: TurnTicket, on_event: AgentEventHandler | None, stop: StopSignal
    ) -> None:
        self.ticket = ticket
        self.stop = stop
        self.error: BaseException | None = None
        self.result: TurnResult | None = None
        self.on_event = on_event


def _discard(message: str) -> None:
    """Default log sink: drop the message."""
