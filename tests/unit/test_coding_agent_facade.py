from unittest.mock import MagicMock

import pytest

from agentshim import ClaudeCodeCodingAgent, CodingAgent
from agentshim.base import BaseCodingAgent, register_provider
from agentshim.cli_agent import CLICodingAgent
from agentshim.mcp_config import HttpMcpServer
from agentshim.trajectory import NullTrajectoryRecorder


@pytest.fixture
def mock_binaries(monkeypatch):
    monkeypatch.setattr(
        "agentshim.cli_agent.shutil.which",
        lambda cmd, path=None: f"/usr/local/bin/{cmd}",
    )
    monkeypatch.setattr(CLICodingAgent, "_check_cli", lambda self: None)


def test_coding_agent_instantiates_requested_provider(mock_binaries):
    agent = CodingAgent(provider="claude", model="test-model")
    assert isinstance(agent.backend, ClaudeCodeCodingAgent)
    assert agent.provider == "claude"
    assert agent.model == "test-model"


def test_coding_agent_delegates_start_session(mock_binaries):
    agent = CodingAgent(provider="claude", model="test-model")
    expected = object()
    agent.backend.start_session = MagicMock(return_value=expected)  # type: ignore[method-assign]
    assert agent.start_session(cwd="/tmp", timeout=12, silent=True) is expected
    agent.backend.start_session.assert_called_once_with(cwd="/tmp", timeout=12, silent=True)  # type: ignore[attr-defined]


def test_coding_agent_delegates_generate(mock_binaries):
    agent = CodingAgent(provider="claude", model="test-model")
    agent.backend.generate = MagicMock(return_value="done")  # type: ignore[method-assign]
    assert agent.generate("hi", cwd="/tmp", timeout=12, silent=True) == "done"
    agent.backend.generate.assert_called_once_with("hi", cwd="/tmp", timeout=12, silent=True)  # type: ignore[attr-defined]


def test_coding_agent_forwards_mutable_common_properties(mock_binaries):
    agent = CodingAgent(provider="claude", model="test-model")
    recorder = NullTrajectoryRecorder()
    handler = object()
    agent.recorder = recorder
    agent.event_handler = handler
    assert agent.backend.recorder is recorder
    assert agent.backend.event_handler is handler


def test_coding_agent_exposes_backend_attributes(mock_binaries):
    agent = CodingAgent(provider="claude", model="test-model")
    assert agent.binary_name == "claude"
    assert agent.claude_path == "/usr/local/bin/claude"


def test_coding_agent_identity_properties(mock_binaries):
    agent = CodingAgent(provider="claude", model="test-model")
    assert agent.readable_name == "Claude Code"
    assert agent.backend_class_name == "ClaudeCodeCodingAgent"


def test_coding_agent_passes_supported_optional_args(mock_binaries):
    servers = [HttpMcpServer(name="docs", url="http://localhost:9000/sse")]
    agent = CodingAgent(
        provider="claude",
        model="test-model",
        mcp_servers=servers,
        sandbox=True,
    )
    assert agent.backend.mcp_servers == servers
    assert agent.backend.sandbox is not None


def test_coding_agent_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown coding agent provider"):
        CodingAgent(provider="does-not-exist")


def test_coding_agent_rejects_unsupported_kwargs(mock_binaries):
    with pytest.raises(TypeError, match="does not accept argument"):
        CodingAgent(provider="claude", unsupported_option=True)


def test_register_provider_works_with_coding_agent():
    @register_provider("dummy-facade-provider")
    class DummyAgent(BaseCodingAgent):
        def __init__(self, model=None):
            self.model = model
            self.recorder = NullTrajectoryRecorder()
            self.event_handler = None

        def generate(self, prompt, cwd=None, timeout=300, silent=False):
            return f"dummy:{prompt}"

    agent = CodingAgent(provider="dummy-facade-provider", model="dummy-model")
    assert isinstance(agent.backend, DummyAgent)
    assert agent.generate("hello") == "dummy:hello"
