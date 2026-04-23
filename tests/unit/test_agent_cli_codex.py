import json
from unittest.mock import MagicMock

import pytest

from agentshim.cli_agent import CLICodingAgent
from agentshim.codex import CodexCodingAgent, CodexGenerationSession
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
        assert "--json" in cmd
        assert cmd[-1] == "-"

    def test_command_omits_mcp_when_no_servers(self, agent):
        cmd = agent._get_command("test")
        assert "-c" not in cmd
        assert cmd[-1] == "-"

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


def _session(event_handler=None) -> CodexGenerationSession:
    session = CodexGenerationSession(
        binary_name="codex",
        env={},
        log_prefix="[Codex]",
        cmd=["codex", "exec", "--json", "-"],
        logger=MagicMock(),
        silent=True,
        event_handler=event_handler,
    )
    return session


class TestCodexGenerationSession:
    def test_turn_completed_forwards_normalized_usage(self):
        handler = MagicMock()
        session = _session(event_handler=handler)
        session._process_stdout(
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": 1200,
                        "cached_input_tokens": 800,
                        "output_tokens": 150,
                    },
                }
            )
        )
        handler.on_usage.assert_called_once_with(
            {
                "input_tokens": 1200,
                "output_tokens": 150,
                "cache_read_input_tokens": 800,
                "cache_creation_input_tokens": 0,
            }
        )

    def test_turn_completed_without_usage_does_not_call_on_usage(self):
        handler = MagicMock()
        session = _session(event_handler=handler)
        session._process_stdout(json.dumps({"type": "turn.completed"}))
        handler.on_usage.assert_not_called()

    def test_legacy_event_handler_without_on_usage_still_works(self):
        class LegacyHandler:
            def on_thinking(self, text):
                pass

            def on_tool_call(self, tool, args=None):
                pass

            def on_tool_result(self, tool, stdout="", stderr="", exit_code=None, duration=None):
                pass

        session = _session(event_handler=LegacyHandler())
        session._process_stdout(
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": 10,
                        "cached_input_tokens": 2,
                        "output_tokens": 3,
                    },
                }
            )
        )
