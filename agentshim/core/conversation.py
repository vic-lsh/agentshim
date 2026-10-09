"""The transport tier: one provider conversation over one live connection.

A ``Transport`` opens a ``Conversation``. A conversation runs turns and does
nothing else: it never retries, never decides to start over, and never reads a
clock to pace itself. Recovery is the session's job (``session_policy``).
Nothing here imports a provider, so a new provider is a new ``Transport`` and
nothing else changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol

from .profile import ConfigScope, McpScope, SkillScope

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from .events import AgentEvent
    from .mcp import McpServer
    from .permissions import ApprovalPolicy, NativePermissions
    from .profile import ProviderProfile
    from .turn import TurnRequest, TurnResult
    from .usage import ProviderUsage


class TransportKind(Enum):
    """Which way an ``Agent`` reaches a provider by name."""

    #: One CLI process per turn (today's behaviour, the default for now).
    ONE_SHOT = "one_shot"
    #: One long-lived process per conversation, over the provider's structured
    #: streaming protocol.
    STREAM = "stream"


@dataclass(frozen=True)
class ConversationSpec:
    """Everything fixed for the life of one conversation.

    ``permissions`` and ``approvals`` have no default: what the agent may do
    is always a decision of the caller. ``resume_id`` names a provider
    conversation to continue, and ``previous_usage`` is that conversation's
    last report, the baseline for a provider whose usage is cumulative.

    The three scopes are transitional: they exist for the one-shot transport
    and go away when it does.
    """

    cwd: str
    model: str | None
    permissions: NativePermissions
    approvals: ApprovalPolicy
    mcp_servers: Sequence[McpServer] = ()
    reasoning_effort: str | None = None
    resume_id: str | None = None
    previous_usage: ProviderUsage | None = None
    skill_scope: SkillScope = SkillScope.ALL
    mcp_scope: McpScope = McpScope.ALL
    config_scope: ConfigScope = ConfigScope.ALL


class Conversation(Protocol):
    """One provider conversation. One turn at a time."""

    @property
    def conversation_id(self) -> str | None:
        """The provider's id for this conversation, once it has named one.

        ``None`` before the first turn of a conversation that was not resumed.
        """
        ...

    def turn(self, request: TurnRequest, emit: Callable[[AgentEvent], None]) -> TurnResult:
        """Run one turn, calling *emit* for each event on the calling thread.

        Raises ``TurnFailedError`` (with its ``kind``), ``SessionResumeError``
        when the conversation it was asked to continue is gone,
        ``TurnTimeoutError``, or another ``AgentShimError``. Never retries. A
        turn ended by ``interrupt`` returns a result with ``interrupted`` set
        instead of raising.
        """
        ...

    def interrupt(self) -> None:
        """End the running turn and keep the conversation. Thread-safe.

        Does nothing when no turn is running.
        """
        ...

    def close(self) -> None:
        """Release the connection. Idempotent."""
        ...


class Transport(Protocol):
    """A way to reach one provider, producing conversations."""

    @property
    def profile(self) -> ProviderProfile:
        """What the provider behind this transport can do."""
        ...

    def open(self, spec: ConversationSpec) -> Conversation:
        """Start (or, with ``spec.resume_id``, resume) a conversation.

        Raises ``ProviderCapabilityError`` for a permission mode that is not in
        ``profile.native_permission_modes``. A refused resume may surface here
        or at the first turn, always as ``SessionResumeError``.
        """
        ...
