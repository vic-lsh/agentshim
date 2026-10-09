"""A transport on a fake Codex server, for the transport tests."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, TypeVar

from agentshim import (
    ApprovalPolicy,
    ConversationSpec,
    NativePermissions,
    TurnRequest,
    TurnResult,
)
from agentshim.providers.codex.app_server import CodexAppServerTransport
from agentshim.testing import CodexScript, FakeClock, FakeExecutor

if TYPE_CHECKING:
    from agentshim import AgentEvent, Conversation

_E = TypeVar("_E")

#: Bounds every turn a test runs, so a broken fake server fails the test with a
#: timeout (counted on the fake clock, instantly) instead of hanging it.
SAFETY_TIMEOUT_S = 60.0
ENV = {"PATH": "/usr/bin:/bin", "HOME": "/home/tester"}


def spec(**changes: object) -> ConversationSpec:
    """A bypass spec in ``/work``; override any field."""
    base = ConversationSpec(
        cwd="/work",
        model="fake-model",
        permissions=NativePermissions.bypass(),
        approvals=ApprovalPolicy.DENY,
    )
    return replace(base, **changes)  # type: ignore[arg-type]


@dataclass
class Rig:
    """A transport, the fake server behind it, and what the client saw."""

    script: CodexScript
    executor: FakeExecutor
    clock: FakeClock
    transport: CodexAppServerTransport
    events: list[AgentEvent] = field(default_factory=list)

    def open(self, **changes: object) -> Conversation:
        """Open a conversation on the transport."""
        return self.transport.open(spec(**changes))

    def turn(self, conversation: Conversation, prompt: str = "go", **request: object) -> TurnResult:
        """Run a turn, collecting its events in ``events``."""
        request.setdefault("timeout", SAFETY_TIMEOUT_S)
        return conversation.turn(TurnRequest(prompt=prompt, **request), self.events.append)  # type: ignore[arg-type]

    def of_type(self, kind: type[_E]) -> list[_E]:
        """The collected events that are instances of *kind*."""
        return [event for event in self.events if isinstance(event, kind)]


def rig(
    script: CodexScript | None = None,
    *,
    peers: object = None,
    **transport_options: object,
) -> Rig:
    """Build a transport on a fake executor serving *script* (a fresh one by default)."""
    script = script if script is not None else CodexScript()
    clock = FakeClock()
    factory: object = peers if peers is not None else script.peer
    executor = FakeExecutor([], peers=factory)  # type: ignore[arg-type]
    transport = CodexAppServerTransport(
        executor=executor,
        env=dict(ENV),
        clock=clock,
        **transport_options,  # type: ignore[arg-type]
    )
    return Rig(script, executor, clock, transport)
