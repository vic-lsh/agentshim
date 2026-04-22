import pytest

from agentshim.cli_agent import CLICodingAgent
from agentshim.gemini import GeminiCodingAgent
from agentshim.mcp_config import HttpMcpServer
from agentshim.opencode import OpencodeCodingAgent


@pytest.fixture
def mock_binaries(monkeypatch):
    """Mock binary discovery and CLI check."""
    monkeypatch.setattr(
        "agentshim.cli_agent.shutil.which",
        lambda cmd, path=None: f"/usr/local/bin/{cmd}",
    )
    monkeypatch.setattr(CLICodingAgent, "_check_cli", lambda self: None)


class TestGeminiMcpUnsupported:
    def test_no_error_without_mcp_servers(self, mock_binaries):
        agent = GeminiCodingAgent()
        assert agent.mcp_servers == []

    def test_raises_with_mcp_servers(self, mock_binaries):
        servers = [HttpMcpServer(name="srv", url="http://localhost:8080")]
        with pytest.raises(ValueError, match="GeminiCodingAgent does not support"):
            GeminiCodingAgent(mcp_servers=servers)


class TestOpencodeMcpUnsupported:
    def test_no_error_without_mcp_servers(self, mock_binaries):
        agent = OpencodeCodingAgent()
        assert agent.mcp_servers == []

    def test_raises_with_mcp_servers(self, mock_binaries):
        servers = [HttpMcpServer(name="srv", url="http://localhost:8080")]
        with pytest.raises(ValueError, match="OpencodeCodingAgent does not support"):
            OpencodeCodingAgent(mcp_servers=servers)
