"""``CodexAppServerTransport`` passes the transport and conversation contract suites."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentshim.testing import AwaitSteer, CodexScript, Say
from agentshim.testing.contracts import (
    ConversationContract,
    SteerableConversationContract,
    TransportContract,
)

from tests.unit.providers.codex.app_server.harness import rig, spec

if TYPE_CHECKING:
    from agentshim import Conversation, Transport


class TestCodexAppServerTransport(TransportContract):
    def make_transport(self) -> Transport:
        return rig().transport


class TestCodexAppServerConversation(ConversationContract):
    def make_conversation(self) -> Conversation:
        return rig().transport.open(spec())


class TestCodexAppServerSteerableConversation(SteerableConversationContract):
    def make_conversation(self) -> Conversation:
        return rig().transport.open(spec())

    def make_steerable_conversation(self) -> Conversation:
        script = CodexScript().turn(AwaitSteer(then=(Say("steered"),)))
        return rig(script).transport.open(spec())
