import json
from unittest.mock import MagicMock

import pytest

from agentshim.cli_agent import CLICodingAgent
from agentshim.copilot import CopilotCodingAgent, CopilotGenerationSession
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
    return CopilotCodingAgent(model="gpt-5.4")


class TestCopilotCodingAgentInit:
    def test_binary_name_is_copilot(self, agent):
        assert agent.binary_name == "copilot"

    def test_binary_path_resolved(self, agent):
        assert agent.binary_path == "/usr/local/bin/copilot"

    def test_copilot_path_property(self, agent):
        assert agent.copilot_path == agent.binary_path

    def test_log_prefix(self, agent):
        assert agent._log_prefix == "[Copilot]"


class TestCopilotCommandConstruction:
    def test_command_includes_required_flags(self, agent):
        cmd = agent._get_command("test prompt")
        assert agent.binary_path in cmd
        assert "--output-format" in cmd
        assert "json" in cmd
        assert "--stream" in cmd
        assert "off" in cmd
        assert "--allow-all-tools" in cmd
        assert "--allow-all-paths" in cmd
        assert "--allow-all-urls" in cmd
        assert "-p" in cmd
        assert "test prompt" in cmd

    def test_command_includes_model_when_set(self, agent):
        cmd = agent._get_command("test prompt")
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "gpt-5.4"

    def test_command_omits_model_when_none(self, mock_binaries):
        cmd = CopilotCodingAgent(model=None)._get_command("test prompt")
        assert "--model" not in cmd

    def test_command_includes_resume_flag(self, agent):
        cmd = agent._get_command("test prompt", resume_session_id="abc-123")
        idx = cmd.index("--resume")
        assert cmd[idx + 1] == "abc-123"

    def test_command_omits_mcp_when_no_servers(self, agent):
        cmd = agent._get_command("test prompt")
        assert "--additional-mcp-config" not in cmd

    def test_mcp_json_http_server(self, mock_binaries):
        servers = [HttpMcpServer(name="srv", url="http://localhost:9000/sse")]
        cmd = CopilotCodingAgent(mcp_servers=servers)._get_command("test")
        idx = cmd.index("--additional-mcp-config")
        config = json.loads(cmd[idx + 1])
        assert config == {"mcpServers": {"srv": {"type": "sse", "url": "http://localhost:9000/sse"}}}

    def test_mcp_json_http_server_with_headers(self, mock_binaries):
        servers = [HttpMcpServer(name="srv", url="https://example.com/sse", headers={"Authorization": "Bearer x"})]
        cmd = CopilotCodingAgent(mcp_servers=servers)._get_command("test")
        idx = cmd.index("--additional-mcp-config")
        config = json.loads(cmd[idx + 1])
        assert config["mcpServers"]["srv"] == {
            "type": "sse",
            "url": "https://example.com/sse",
            "headers": {"Authorization": "Bearer x"},
        }

    def test_mcp_json_stdio_server(self, mock_binaries):
        servers = [StdioMcpServer(name="tool", command="npx", args=["-y", "@some/pkg"], env={"KEY": "val"})]
        cmd = CopilotCodingAgent(mcp_servers=servers)._get_command("test")
        idx = cmd.index("--additional-mcp-config")
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


class TestCopilotGenerationSession:
    def _make_session(self, event_handler=None, recorder=None):
        return CopilotGenerationSession(
            binary_name="copilot",
            env={},
            log_prefix="[Copilot]",
            cmd=["copilot", "--output-format", "json"],
            logger=MagicMock(),
            silent=True,
            recorder=recorder or NullTrajectoryRecorder(),
            event_handler=event_handler,
        )

    def test_process_stdout_parses_session_start(self):
        session = self._make_session()
        session._process_stdout(
            '{"type":"session.start","data":{"sessionId":"sid-1","version":1,"producer":"copilot-agent","copilotVersion":"1.0.34","startTime":"2026-01-01T00:00:00Z"}}\n'
        )
        assert session.session_id == "sid-1"

    def test_process_stdout_parses_message(self):
        session = self._make_session()
        session._process_stdout(
            '{"type":"assistant.message","data":{"messageId":"msg-1","content":"hello","outputTokens":5}}\n'
        )
        assert session.final_result == "hello"
        assert "hello" in session.stdout_lines
        assert session.usage.tokens.output_tokens == 5

    def test_process_stdout_parses_message_delta(self):
        session = self._make_session()
        session._process_stdout(
            '{"type":"assistant.message_delta","ephemeral":true,"data":{"messageId":"msg-1","deltaContent":"hel"}}\n'
        )
        session._process_stdout(
            '{"type":"assistant.message_delta","ephemeral":true,"data":{"messageId":"msg-1","deltaContent":"lo"}}\n'
        )
        assert "".join(session._streamed_text_chunks) == "hello"

    def test_process_stdout_parses_tool_lifecycle(self):
        session = self._make_session()
        session._process_stdout(
            '{"type":"tool.execution_start","data":{"toolCallId":"t1","toolName":"shell","arguments":{"command":"ls"}}}\n'
        )
        session._process_stdout(
            '{"type":"tool.execution_complete","data":{"toolCallId":"t1","success":true,"result":{"content":"ok","contents":[{"type":"terminal","text":"ok","exitCode":0}]}}}\n'
        )
        assert session.tool_map["t1"] == "shell"

    def test_process_stdout_parses_result_event_session_id(self):
        session = self._make_session()
        session._process_stdout('{"type":"result","sessionId":"sid-2","exitCode":0,"usage":{"premiumRequests":0}}\n')
        assert session.session_id == "sid-2"

    def test_process_stdout_parses_non_json(self):
        session = self._make_session()
        session._process_stdout("plain text\n")
        assert "plain text" in session.stdout_lines

    def test_event_handler_on_thinking_called_for_delta(self):
        handler = MagicMock()
        session = self._make_session(event_handler=handler)
        session._process_stdout(
            '{"type":"assistant.message_delta","ephemeral":true,"data":{"messageId":"msg-1","deltaContent":"thinking"}}\n'
        )
        handler.on_thinking.assert_called_once_with("thinking")

    def test_event_handler_on_tool_call_called(self):
        handler = MagicMock()
        session = self._make_session(event_handler=handler)
        session._process_stdout(
            '{"type":"tool.execution_start","data":{"toolCallId":"t1","toolName":"read","arguments":{"path":"/tmp"}}}\n'
        )
        handler.on_tool_call.assert_called_once_with("read", {"path": "/tmp"})

    def test_create_session_returns_copilot_session(self, agent):
        session = agent._create_session(cmd=["copilot", "--output-format", "json"])
        assert isinstance(session, CopilotGenerationSession)
