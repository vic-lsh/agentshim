import pytest

from agentshim.cli_agent import CLICodingAgent
from agentshim.codex import CodexCodingAgent
from agentshim.mcp_config import HttpMcpServer, StdioMcpServer


@pytest.fixture
def mock_binaries(monkeypatch):
    """Mock binary discovery and CLI check."""
    monkeypatch.setattr(
        "agentshim.cli_agent.shutil.which",
        lambda cmd, path=None: f"/usr/local/bin/{cmd}",
    )
    monkeypatch.setattr(CLICodingAgent, "_check_cli", lambda self: None)


@pytest.fixture
def agent(mock_binaries):
    return CodexCodingAgent(model="test-model")


class TestCodexCommandConstruction:
    def test_command_base_flags(self, agent):
        cmd = agent._get_command("test")
        assert "exec" in cmd
        assert "--dangerously-bypass-approvals-and-sandbox" in cmd

    def test_command_omits_mcp_when_no_servers(self, agent):
        cmd = agent._get_command("test")
        assert "-c" not in cmd

    def test_mcp_http_server(self, mock_binaries):
        servers = [HttpMcpServer(name="srv", url="http://localhost:9000/sse")]
        agent = CodexCodingAgent(mcp_servers=servers)
        cmd = agent._get_command("test")
        assert "-c" in cmd
        idx = cmd.index("-c")
        assert cmd[idx + 1] == 'mcp_servers.srv.url="http://localhost:9000/sse"'

    def test_mcp_stdio_server(self, mock_binaries):
        servers = [StdioMcpServer(name="tool", command="npx", args=["-y", "pkg"])]
        agent = CodexCodingAgent(mcp_servers=servers)
        cmd = agent._get_command("test")
        c_values = [cmd[i + 1] for i, v in enumerate(cmd) if v == "-c"]
        assert 'mcp_servers.tool.command="npx"' in c_values
        assert 'mcp_servers.tool.args=["-y", "pkg"]' in c_values

    def test_mcp_stdio_server_with_env(self, mock_binaries):
        servers = [StdioMcpServer(name="t", command="cmd", env={"K1": "v1", "K2": "v2"})]
        agent = CodexCodingAgent(mcp_servers=servers)
        cmd = agent._get_command("test")
        c_values = [cmd[i + 1] for i, v in enumerate(cmd) if v == "-c"]
        assert 'mcp_servers.t.env.K1="v1"' in c_values
        assert 'mcp_servers.t.env.K2="v2"' in c_values

    def test_mcp_multiple_servers(self, mock_binaries):
        servers = [
            HttpMcpServer(name="a", url="http://a"),
            StdioMcpServer(name="b", command="cmd"),
        ]
        agent = CodexCodingAgent(mcp_servers=servers)
        cmd = agent._get_command("test")
        c_values = [cmd[i + 1] for i, v in enumerate(cmd) if v == "-c"]
        assert 'mcp_servers.a.url="http://a"' in c_values
        assert 'mcp_servers.b.command="cmd"' in c_values
