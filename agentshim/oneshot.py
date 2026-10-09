"""The transitional transport: one CLI process per turn behind the ``Transport`` protocol.

``OneShotTransport`` wraps today's ``CliAgent`` / ``AgentSession`` so the
session tier can drive every provider before the long-lived transports exist.
It adds nothing to a turn: the same argv, the same events, the same result.
It goes away with the one-shot path.
"""

from __future__ import annotations

import threading
from dataclasses import replace
from typing import TYPE_CHECKING

from agentshim.agent import CliAgent
from agentshim.core.errors import (
    CliExitError,
    ProviderCapabilityError,
    SessionStateError,
)
from agentshim.core.events import TurnInterrupted
from agentshim.core.permissions import NativeMode
from agentshim.core.profile import McpMechanism
from agentshim.core.turn import TurnResult
from agentshim.core.usage import ProviderUsage

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from agentshim.agent import AgentSession
    from agentshim.core.conversation import Conversation, ConversationSpec
    from agentshim.core.events import AgentEvent
    from agentshim.core.profile import ProviderProfile
    from agentshim.core.provider import Provider
    from agentshim.core.turn import TurnRequest
    from agentshim.execution.executor import CommandExecutor


class _Router:
    """An event handler that forwards to whichever turn is running."""

    def __init__(self) -> None:
        self.emit: Callable[[AgentEvent], None] | None = None

    def on_event(self, event: AgentEvent) -> None:
        """Forward *event* to the running turn's callback, if there is one."""
        emit = self.emit
        if emit is not None:
            emit(event)


class OneShotTransport:
    """A ``Transport`` over per-turn CLI processes.

    Construction resolves the binary and runs the provider's health check once,
    like ``CliAgent``, so a broken install fails here and not mid-run. Only
    ``NativeMode.BYPASS`` is supported, which is how the one-shot path has
    always run.
    """

    # Each argument is an independent documented option, as on ``CliAgent``.
    def __init__(
        self,
        provider: str | Provider,
        *,
        executor: CommandExecutor | None = None,
        env: Mapping[str, str] | None = None,
        check_timeout: float = 15.0,
        log: Callable[[str], None] | None = None,
    ) -> None:
        """Check the install of *provider* on *executor* (the local host by default)."""
        self._agent = CliAgent(
            provider, executor=executor, env=env, check_timeout=check_timeout, log=log
        )

    @property
    def profile(self) -> ProviderProfile:
        """What the provider can do."""
        return self._agent.profile

    def open(self, spec: ConversationSpec) -> Conversation:
        """Start a conversation, resuming ``spec.resume_id`` on its first turn.

        Raises ``ProviderCapabilityError`` for a permission mode other than
        bypass, or for reasoning effort or MCP servers the provider cannot take.
        """
        profile = self.profile
        if spec.permissions.mode not in profile.native_permission_modes:
            msg = f"{profile.name} cannot enforce native permission mode {spec.permissions.mode.value!r}"
            raise ProviderCapabilityError(msg)
        if (
            spec.permissions.mode is not NativeMode.BYPASS
        ):  # pragma: no cover - profiles say bypass only
            msg = "the one-shot transport only runs with native permissions bypassed"
            raise ProviderCapabilityError(msg)
        if spec.reasoning_effort is not None and not profile.supports_reasoning_effort:
            msg = f"{profile.name} does not support reasoning effort"
            raise ProviderCapabilityError(msg)
        if spec.mcp_servers and profile.mcp is McpMechanism.NONE:
            msg = f"{profile.name} does not support MCP servers"
            raise ProviderCapabilityError(msg)
        router = _Router()
        agent = self._agent.derive(model=spec.model, event_handler=router)
        session = agent.start_session(
            cwd=spec.cwd,
            session_id=spec.resume_id,
            previous_usage=spec.previous_usage,
            skill_scope=spec.skill_scope,
            mcp_scope=spec.mcp_scope,
            config_scope=spec.config_scope,
        )
        return _OneShotConversation(session, router, spec)


class _OneShotConversation:
    """A ``Conversation`` that runs each turn as an ``AgentSession`` turn."""

    def __init__(self, session: AgentSession, router: _Router, spec: ConversationSpec) -> None:
        self._session = session
        self._router = router
        self._spec = spec
        self._lock = threading.Lock()
        self._closed = False
        self._interrupted = False

    @property
    def conversation_id(self) -> str | None:
        """The session's provider conversation id."""
        return self._session.session_id

    def turn(self, request: TurnRequest, emit: Callable[[AgentEvent], None]) -> TurnResult:
        """Run one turn, filling the spec's MCP servers and reasoning effort when absent.

        An interrupt that ends the process turns the failure it causes into a
        result with ``interrupted`` set, so the conversation survives it.
        """
        with self._lock:
            if self._closed:
                msg = "conversation is closed"
                raise SessionStateError(msg)
            self._interrupted = False
        self._router.emit = emit
        try:
            return self._session.turn(self._with_spec_defaults(request))
        except CliExitError as error:
            # A resumed turn the interrupt killed is classified as a refused
            # resume (any unclassified exit of a resumed turn is), so the
            # interrupt, not the classification, decides what happened.
            if not self._was_interrupted():
                raise
            emit(TurnInterrupted())
            return self._interrupted_result(error)
        finally:
            self._router.emit = None

    def _was_interrupted(self) -> bool:
        with self._lock:
            return self._interrupted

    def _with_spec_defaults(self, request: TurnRequest) -> TurnRequest:
        spec = self._spec
        if not request.mcp_servers and spec.mcp_servers:
            request = replace(request, mcp_servers=tuple(spec.mcp_servers))
        if request.reasoning_effort is None and spec.reasoning_effort is not None:
            request = replace(request, reasoning_effort=spec.reasoning_effort)
        return request

    def _interrupted_result(self, error: CliExitError) -> TurnResult:
        """Describe a turn the interrupt cut short; it has no parsed result to report."""
        return TurnResult(
            text="",
            structured_output=None,
            session_id=self._session.session_id,
            resumed=False,
            usage=ProviderUsage(),
            cost_usd=None,
            duration_ms=0,
            exit_code=error.returncode,
            interrupted=True,
        )

    def interrupt(self) -> None:
        """Stop the running process; the conversation id is kept."""
        with self._lock:
            self._interrupted = True
        self._session.cancel()

    def close(self) -> None:
        """Stop any running process. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._session.cancel()
