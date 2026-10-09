"""``CodexAppServerTransport`` passes the transport and conversation contract suites."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentshim.testing.contracts import ConversationContract, TransportContract

from tests.unit.providers.codex.app_server.harness import rig, spec

if TYPE_CHECKING:
    from agentshim import Conversation, Transport


class TestCodexAppServerTransport(TransportContract):
    def make_transport(self) -> Transport:
        return rig().transport


class TestCodexAppServerConversation(ConversationContract):
    def make_conversation(self) -> Conversation:
        return rig().transport.open(spec())
