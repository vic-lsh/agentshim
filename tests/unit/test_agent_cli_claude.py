import json
from unittest.mock import MagicMock

import pytest

from agentshim.claude import ClaudeCodeCodingAgent, ClaudeGenerationSession
from agentshim.cli_agent import CLICodingAgent
from agentshim.mcp_config import HttpMcpServer, StdioMcpServer
from agentshim.trajectory import NullTrajectoryRecorder


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
    """Create a ClaudeCodeCodingAgent with mocked binaries."""
    return ClaudeCodeCodingAgent(model="test-model")


class TestClaudeCodeCodingAgentInit:
    """Tests for ClaudeCodeCodingAgent initialization."""

    def test_binary_name_is_claude(self, agent):
        assert agent.binary_name == "claude"

    def test_binary_path_resolved(self, agent):
        assert agent.binary_path == "/usr/local/bin/claude"

    def test_claude_path_property(self, agent):
        """claude_path is a backward-compatible alias for binary_path."""
        assert agent.claude_path == agent.binary_path

    def test_model_stored(self, agent):
        assert agent.model == "test-model"

    def test_default_model_is_none(self, mock_binaries):
        agent = ClaudeCodeCodingAgent()
        assert agent.model is None

    def test_log_prefix(self, agent):
        assert agent._log_prefix == "[Claude]"

    def test_binary_not_found_raises_runtime_error(self, monkeypatch):
        monkeypatch.setattr(
            "agentshim.cli_agent.shutil.which",
            lambda cmd, path=None: None,
        )
        with pytest.raises(RuntimeError, match="claude binary not found"):
            ClaudeCodeCodingAgent()


class TestClaudeCommandConstruction:
    """Tests for _get_command method."""

    def test_command_includes_required_flags(self, agent):
        cmd = agent._get_command("test prompt")
        assert agent.binary_path in cmd
        assert "-p" in cmd
        assert "--dangerously-skip-permissions" in cmd
        assert "--output-format" in cmd
        assert "stream-json" in cmd
        assert "--verbose" in cmd

    def test_command_includes_model_when_set(self, agent):
        cmd = agent._get_command("test prompt")
        assert "--model" in cmd
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "test-model"

    def test_command_omits_model_when_none(self, mock_binaries):
        agent = ClaudeCodeCodingAgent(model=None)
        cmd = agent._get_command("test prompt")
        assert "--model" not in cmd

    def test_command_includes_prompt(self, agent):
        cmd = agent._get_command("deploy the app")
        assert "deploy the app" in cmd


class TestClaudeGenerationSession:
    """Tests for ClaudeGenerationSession event processing."""

    def _make_session(self, event_handler=None, recorder=None):
        return ClaudeGenerationSession(
            binary_name="claude",
            env={},
            log_prefix="[Claude]",
            cmd=["claude", "-p"],
            logger=MagicMock(),
            silent=True,
            recorder=recorder or NullTrajectoryRecorder(),
            event_handler=event_handler,
        )

    def test_process_stdout_parses_text_event(self):
        session = self._make_session()
        line = '{"type":"assistant","message":{"content":[{"type":"text","text":"hello"}]}}\n'
        session._process_stdout(line)
        assert "hello" in session.stdout_lines

    def test_process_stdout_parses_tool_use_event(self):
        session = self._make_session()
        line = (
            '{"type":"assistant","message":{"content":'
            '[{"type":"tool_use","name":"Bash","id":"t1","input":{"cmd":"ls"}}]}}\n'
        )
        session._process_stdout(line)
        assert "t1" in session.tool_map
        assert session.tool_map["t1"] == "Bash"

    def test_process_stdout_parses_tool_result_event(self):
        session = self._make_session()
        # Set up tool map first
        session.tool_map["t1"] = "Bash"
        session.tool_start_times["t1"] = 1000.0
        session.tool_args["t1"] = {"cmd": "ls"}

        line = (
            '{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"t1","content":"file1.txt"}]}}\n'
        )
        session._process_stdout(line)
        # Tool result was processed (recorder recorded it via NullTrajectoryRecorder)

    def test_process_stdout_parses_result_event(self):
        session = self._make_session()
        line = '{"type":"result","result":"all done"}\n'
        session._process_stdout(line)
        assert session.final_result == "all done"

    def test_process_stdout_handles_non_json(self):
        session = self._make_session()
        session._process_stdout("some plain text\n")
        assert "some plain text" in session.stdout_lines

    def test_process_stdout_skips_empty_lines(self):
        session = self._make_session()
        session._process_stdout("")
        assert session.stdout_lines == []

    def test_event_handler_on_thinking_called(self):
        handler = MagicMock()
        session = self._make_session(event_handler=handler)
        line = '{"type":"assistant","message":{"content":[{"type":"text","text":"thinking..."}]}}\n'
        session._process_stdout(line)
        handler.on_thinking.assert_called_once_with("thinking...")

    def test_event_handler_on_tool_call_called(self):
        handler = MagicMock()
        session = self._make_session(event_handler=handler)
        line = (
            '{"type":"assistant","message":{"content":'
            '[{"type":"tool_use","name":"Read","id":"t2","input":{"path":"/tmp"}}]}}\n'
        )
        session._process_stdout(line)
        handler.on_tool_call.assert_called_once_with("Read", {"path": "/tmp"})

    def test_assistant_usage_forwarded_to_event_handler(self):
        handler = MagicMock()
        session = self._make_session(event_handler=handler)
        line = (
            '{"type":"assistant","message":{"content":'
            '[{"type":"text","text":"ok"}],'
            '"usage":{"input_tokens":14000,"output_tokens":50,'
            '"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}}\n'
        )
        session._process_stdout(line)
        handler.on_usage.assert_called_once_with(
            {
                "input_tokens": 14000,
                "output_tokens": 50,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            }
        )

    def test_assistant_without_usage_does_not_call_on_usage(self):
        handler = MagicMock()
        session = self._make_session(event_handler=handler)
        line = (
            '{"type":"assistant","message":{"content":'
            '[{"type":"text","text":"ok"}]}}\n'
        )
        session._process_stdout(line)
        handler.on_usage.assert_not_called()

    def test_legacy_event_handler_without_on_usage_still_works(self):
        class LegacyHandler:
            def __init__(self):
                self.text_calls = []

            def on_thinking(self, text):
                self.text_calls.append(text)

            def on_tool_call(self, tool, args=None):
                pass

            def on_tool_result(self, tool, stdout="", stderr="", exit_code=None, duration=None):
                pass

        handler = LegacyHandler()
        session = self._make_session(event_handler=handler)
        line = (
            '{"type":"assistant","message":{"content":'
            '[{"type":"text","text":"hi"}],'
            '"usage":{"input_tokens":500,"output_tokens":10}}}\n'
        )
        session._process_stdout(line)
        assert handler.text_calls == ["hi"]

    def test_create_session_returns_claude_session(self, agent):
        session = agent._create_session(cmd=["claude", "-p"])
        assert isinstance(session, ClaudeGenerationSession)


class TestClaudeMcpConfig:
    """Tests for MCP server configuration in Claude agent."""

    def test_command_omits_mcp_when_no_servers(self, agent):
        cmd = agent._get_command("test")
        assert "--mcp-config" not in cmd
        assert "--strict-mcp-config" not in cmd

    def test_command_includes_mcp_flags_when_servers_set(self, mock_binaries):
        servers = [HttpMcpServer(name="test", url="http://localhost:8080")]
        agent = ClaudeCodeCodingAgent(mcp_servers=servers)
        cmd = agent._get_command("test")
        assert "--mcp-config" in cmd
        assert "--strict-mcp-config" in cmd

    def test_mcp_json_http_server(self, mock_binaries):
        servers = [HttpMcpServer(name="my-srv", url="http://localhost:9000/sse")]
        agent = ClaudeCodeCodingAgent(mcp_servers=servers)
        cmd = agent._get_command("test")
        idx = cmd.index("--mcp-config")
        config = json.loads(cmd[idx + 1])
        assert config == {"mcpServers": {"my-srv": {"type": "sse", "url": "http://localhost:9000/sse"}}}

    def test_mcp_json_http_server_with_headers(self, mock_binaries):
        servers = [
            HttpMcpServer(
                name="auth-srv",
                url="https://example.com/sse",
                headers={"Authorization": "Bearer xyz"},
            )
        ]
        agent = ClaudeCodeCodingAgent(mcp_servers=servers)
        cmd = agent._get_command("test")
        idx = cmd.index("--mcp-config")
        config = json.loads(cmd[idx + 1])
        assert config["mcpServers"]["auth-srv"] == {
            "type": "sse",
            "url": "https://example.com/sse",
            "headers": {"Authorization": "Bearer xyz"},
        }

    def test_mcp_json_stdio_server(self, mock_binaries):
        servers = [
            StdioMcpServer(
                name="tool",
                command="npx",
                args=["-y", "@some/pkg"],
                env={"KEY": "val"},
            )
        ]
        agent = ClaudeCodeCodingAgent(mcp_servers=servers)
        cmd = agent._get_command("test")
        idx = cmd.index("--mcp-config")
        config = json.loads(cmd[idx + 1])
        assert config == {
            "mcpServers": {
                "tool": {
                    "command": "npx",
                    "args": ["-y", "@some/pkg"],
                    "env": {"KEY": "val"},
                }
            }
        }

    def test_mcp_json_stdio_server_no_env(self, mock_binaries):
        servers = [StdioMcpServer(name="t", command="cmd")]
        agent = ClaudeCodeCodingAgent(mcp_servers=servers)
        cmd = agent._get_command("test")
        idx = cmd.index("--mcp-config")
        config = json.loads(cmd[idx + 1])
        assert "env" not in config["mcpServers"]["t"]

    def test_mcp_json_multiple_servers(self, mock_binaries):
        servers = [
            HttpMcpServer(name="http-srv", url="http://localhost:8080"),
            StdioMcpServer(name="stdio-srv", command="node", args=["server.js"]),
        ]
        agent = ClaudeCodeCodingAgent(mcp_servers=servers)
        cmd = agent._get_command("test")
        idx = cmd.index("--mcp-config")
        config = json.loads(cmd[idx + 1])
        assert "http-srv" in config["mcpServers"]
        assert "stdio-srv" in config["mcpServers"]
