from unittest.mock import MagicMock

import pytest

from agentshim import (
    BaseAgentSession,
    ClaudeCodeCodingAgent,
    CodingAgent,
    CompositeEventHandler,
    get_provider_class,
    list_providers,
    register_provider,
)
from agentshim.base import BaseCodingAgent
from agentshim.cli_agent import CLICodingAgent
from agentshim.mcp_config import HttpMcpServer


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


def test_coding_agent_start_session_returns_portable_session_type(mock_binaries):
    agent = CodingAgent(provider="claude", model="test-model")
    session = agent.start_session()
    assert isinstance(session, BaseAgentSession)


def test_coding_agent_delegates_generate(mock_binaries):
    agent = CodingAgent(provider="claude", model="test-model")
    agent.backend.generate = MagicMock(return_value="done")  # type: ignore[method-assign]
    assert agent.generate("hi", cwd="/tmp", timeout=12, silent=True) == "done"
    agent.backend.generate.assert_called_once_with("hi", cwd="/tmp", timeout=12, silent=True)  # type: ignore[attr-defined]


def test_coding_agent_forwards_event_handler_property(mock_binaries):
    agent = CodingAgent(provider="claude", model="test-model")
    handler = object()
    agent.event_handler = handler
    assert agent.backend.event_handler is handler


def test_coding_agent_requires_explicit_backend_access_for_backend_attributes(mock_binaries):
    agent = CodingAgent(provider="claude", model="test-model")
    assert agent.backend.binary_name == "claude"
    assert agent.backend.claude_path == "/usr/local/bin/claude"


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


def test_coding_agent_passes_composable_event_handlers(mock_binaries):
    first = MagicMock()
    second = MagicMock()
    agent = CodingAgent(provider="claude", event_handlers=[first, second])

    assert isinstance(agent.backend.event_handler, CompositeEventHandler)
    assert agent.backend.event_handler.handlers == [first, second]


def test_coding_agent_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown coding agent provider"):
        CodingAgent(provider="does-not-exist")


@pytest.mark.parametrize("kwarg_name", ["location", "dspy_config"])
def test_coding_agent_rejects_non_portable_top_level_kwargs(mock_binaries, kwarg_name):
    with pytest.raises(TypeError, match=rf"unexpected keyword argument '{kwarg_name}'"):
        CodingAgent(provider="claude", **{kwarg_name: "value"})  # type: ignore[call-arg]


def test_register_provider_works_with_coding_agent_backend_kwargs():
    dspy_config = object()

    @register_provider("dummy-facade-provider-with-backend-kwargs")
    class DummyAgent(BaseCodingAgent):
        def __init__(self, model=None, location=None, dspy_config=None):
            self.model = model
            self.location = location
            self.dspy_config = dspy_config
            self.event_handler = None

        def generate(self, prompt, cwd=None, timeout=300, silent=False):
            return f"dummy:{prompt}"

    agent = CodingAgent(
        provider="dummy-facade-provider-with-backend-kwargs",
        model="dummy-model",
        backend_kwargs={"location": "us-west1", "dspy_config": dspy_config},
    )
    assert isinstance(agent.backend, DummyAgent)
    assert agent.backend.location == "us-west1"
    assert agent.backend.dspy_config is dspy_config
    assert agent.generate("hello") == "dummy:hello"


def test_coding_agent_rejects_backend_kwargs_overlapping_portable_args():
    with pytest.raises(ValueError, match="backend_kwargs must not override portable CodingAgent arguments: model"):
        CodingAgent(
            provider="claude",
            model="portable-model",
            backend_kwargs={"model": "backend-model"},
        )


def test_coding_agent_surfaces_backend_constructor_errors_for_backend_kwargs():
    @register_provider("dummy-facade-provider-with-strict-init")
    class DummyAgent(BaseCodingAgent):
        def __init__(self, model=None):
            self.model = model
            self.event_handler = None

        def generate(self, prompt, cwd=None, timeout=300, silent=False):
            return prompt

    with pytest.raises(TypeError, match="unexpected keyword argument 'location'"):
        CodingAgent(
            provider="dummy-facade-provider-with-strict-init",
            model="dummy-model",
            backend_kwargs={"location": "us-west1"},
        )


def test_get_provider_class_resolves_aliases():
    assert get_provider_class("claude") is ClaudeCodeCodingAgent
    assert get_provider_class("claude-code") is ClaudeCodeCodingAgent
    assert get_provider_class("anthropic") is ClaudeCodeCodingAgent


def test_list_providers_returns_canonical_names():
    providers = list_providers()
    assert "claude" in providers
    assert "codex" in providers
    assert "gemini" in providers
    assert "opencode" in providers
    assert "claude-code" not in providers
    assert "anthropic" not in providers


def test_register_provider_rejects_invalid_names():
    with pytest.raises(ValueError, match="invalid provider name"):
        register_provider("not valid")


def test_register_provider_rejects_non_agent_classes():
    with pytest.raises(TypeError, match="must inherit from BaseCodingAgent"):
        register_provider("not-an-agent")(object)


def test_register_provider_rejects_abstract_classes():
    class AbstractAgent(BaseCodingAgent):
        pass

    with pytest.raises(TypeError, match="must be concrete"):
        register_provider("abstract-provider")(AbstractAgent)


def test_register_provider_rejects_collisions():
    @register_provider("collision-provider-1")
    class FirstAgent(BaseCodingAgent):
        def generate(self, prompt, cwd=None, timeout=300, silent=False):
            return prompt

    with pytest.raises(ValueError, match="already registered"):

        @register_provider("collision-provider-1")
        class SecondAgent(BaseCodingAgent):
            def generate(self, prompt, cwd=None, timeout=300, silent=False):
                return prompt

    assert get_provider_class("collision-provider-1") is FirstAgent


def test_register_provider_can_overwrite_existing_registration():
    @register_provider("overwrite-provider")
    class FirstAgent(BaseCodingAgent):
        def generate(self, prompt, cwd=None, timeout=300, silent=False):
            return f"first:{prompt}"

    @register_provider("overwrite-provider", overwrite=True)
    class SecondAgent(BaseCodingAgent):
        def generate(self, prompt, cwd=None, timeout=300, silent=False):
            return f"second:{prompt}"

    assert get_provider_class("overwrite-provider") is SecondAgent
