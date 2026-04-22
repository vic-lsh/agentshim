import io
import json
from unittest.mock import MagicMock, patch

import pytest
from loguru import logger

from agentshim.gemini import GeminiCodingAgent


@pytest.fixture
def mock_env():
    """Mock environment for agent initialization."""
    return {"PATH": "/usr/bin:/bin", "HOME": "/tmp", "USER": "test"}


@pytest.fixture
def mock_which():
    """Mock shutil.which to return a fake binary path."""
    with patch("shutil.which", return_value="/usr/bin/gemini"):
        yield


@pytest.fixture
def gemini_agent(mock_which, mock_env):
    """Create a GeminiCodingAgent instance with mocked environment."""
    with patch(
        "agentshim.cli_agent.get_interactive_env",
        return_value=mock_env,
    ):
        with patch("agentshim.cli_agent.CLICodingAgent._check_cli"):
            agent = GeminiCodingAgent()
            yield agent


@pytest.fixture
def mock_popen():
    """Mock subprocess.Popen."""
    with patch("subprocess.Popen") as mock:
        yield mock


def test_generate_stream_prefixing(gemini_agent, mock_popen):
    """Test that assistant output is correctly prefixed with [Gemini]."""
    mock_process = MagicMock()
    mock_process.returncode = 0

    # Streaming: "He", "llo\n", "Wor", "ld"
    stream_data = [
        json.dumps({"type": "message", "role": "assistant", "content": "He"}) + "\n",
        json.dumps({"type": "message", "role": "assistant", "content": "llo\n"}) + "\n",
        json.dumps({"type": "message", "role": "assistant", "content": "Wor"}) + "\n",
        json.dumps({"type": "message", "role": "assistant", "content": "ld"}) + "\n",
    ]

    mock_process.stdout.readline.side_effect = stream_data + [""]
    mock_process.stderr.readline.return_value = ""
    mock_process.wait.return_value = 0
    mock_popen.return_value = mock_process

    captured_stdout = io.StringIO()
    handler_id = logger.add(captured_stdout, format="{message}")
    try:
        result = gemini_agent.generate("Test")
    finally:
        logger.remove(handler_id)

    output = captured_stdout.getvalue()

    # Verify content accumulation
    assert result == "Hello\nWorld"

    # Verify prefixing
    assert "[Gemini] Hello" in output
    assert "[Gemini] World" in output


def test_tool_use_truncation_and_prefix(gemini_agent, mock_popen):
    """Test that tool parameters are truncated and prefixed."""
    mock_process = MagicMock()
    mock_process.returncode = 0

    long_params = {"data": "x" * 300}
    stream_data = [json.dumps({"type": "tool_use", "tool_name": "test_tool", "parameters": long_params}) + "\n"]

    mock_process.stdout.readline.side_effect = stream_data + [""]
    mock_process.stderr.readline.return_value = ""
    mock_process.wait.return_value = 0
    mock_popen.return_value = mock_process

    captured_stdout = io.StringIO()
    handler_id = logger.add(captured_stdout, format="{message}")
    try:
        gemini_agent.generate("Test")
    finally:
        logger.remove(handler_id)

    output = captured_stdout.getvalue()

    # Verify tool use prefix and truncation (semantic content, not ANSI codes)
    assert "[Gemini]" in output
    assert "[Tool Use]" in output
    assert "..." in output
    assert "x" * 300 not in output


def test_tool_result_empty_message(gemini_agent, mock_popen):
    """Test that empty tool results print a success message using the tool name."""
    mock_process = MagicMock()
    mock_process.returncode = 0

    tool_id = "test-tool-123"
    stream_data = [
        # First tool use maps ID to name
        json.dumps(
            {
                "type": "tool_use",
                "tool_name": "my_awesome_tool",
                "tool_id": tool_id,
                "parameters": {},
            }
        )
        + "\n",
        # Then empty result uses that ID
        json.dumps({"type": "tool_result", "tool_id": tool_id, "output": ""}) + "\n",
    ]

    mock_process.stdout.readline.side_effect = stream_data + [""]
    mock_process.stderr.readline.return_value = ""
    mock_process.wait.return_value = 0
    mock_popen.return_value = mock_process

    captured_stdout = io.StringIO()
    handler_id = logger.add(captured_stdout, format="{message}")
    try:
        gemini_agent.generate("Test")
    finally:
        logger.remove(handler_id)

    output = captured_stdout.getvalue()

    # Verify success message (semantic content, not ANSI codes)
    assert "[Gemini]" in output
    assert "my_awesome_tool ran successfully" in output


def test_generate_emits_ui_events(gemini_agent, mock_popen):
    """Test that UI events are emitted during generation."""
    mock_process = MagicMock()
    mock_process.returncode = 0

    tool_id = "test-tool-123"
    stream_data = [
        json.dumps({"type": "message", "role": "assistant", "content": "Thinking..."}) + "\n",
        json.dumps(
            {
                "type": "tool_use",
                "tool_name": "test_tool",
                "tool_id": tool_id,
                "parameters": {"k": "v"},
            }
        )
        + "\n",
        json.dumps({"type": "tool_result", "tool_id": tool_id, "output": "Result"}) + "\n",
    ]

    mock_process.stdout.readline.side_effect = stream_data + [""]
    mock_process.stderr.readline.return_value = ""
    mock_process.wait.return_value = 0
    mock_popen.return_value = mock_process

    # Attach mock event handler
    mock_ui = MagicMock()
    gemini_agent.event_handler = mock_ui

    gemini_agent.generate("Test")

    # Verify calls
    mock_ui.on_thinking.assert_called_with("Thinking...")
    mock_ui.on_tool_call.assert_called_with("test_tool", {"k": "v"})

    # Check tool result call
    mock_ui.on_tool_result.assert_called()
    call_args = mock_ui.on_tool_result.call_args[1]
    assert call_args["tool"] == "test_tool"
    assert call_args["stdout"] == "Result"
