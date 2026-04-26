import io
import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from loguru import logger

from agentshim import CompositeEventHandler, ConsoleEventHandler
from agentshim.claude import ClaudeCodeCodingAgent


@pytest.fixture
def mock_env():
    """Mock environment for agent initialization."""
    return {"PATH": "/usr/bin:/bin", "HOME": "/tmp", "USER": "test"}


@pytest.fixture
def mock_event_handler():
    """Mock AgentEventHandler."""
    return MagicMock()


@pytest.fixture
def claude_agent(mock_llm_subprocess, mock_env, mock_event_handler):
    """Create a ClaudeCodeCodingAgent instance with mocked environment."""
    mock_popen, mock_which = mock_llm_subprocess
    mock_which.return_value = "/usr/bin/claude"

    with patch(
        "agentshim.cli_agent.get_interactive_env",
        return_value=mock_env,
    ):
        with patch("agentshim.cli_agent.CLICodingAgent._check_cli"):
            agent = ClaudeCodeCodingAgent(
                event_handler=CompositeEventHandler(
                    [
                        ConsoleEventHandler(),
                        mock_event_handler,
                    ]
                )
            )
            cast("Any", agent).mock_event_handler = mock_event_handler
            yield agent


def test_parse_system_event(claude_agent, mock_llm_subprocess):
    """Test that system events are parsed and handled silently."""
    mock_popen, _ = mock_llm_subprocess
    mock_process = mock_popen.return_value
    mock_process.returncode = 0

    system_event = {
        "type": "system",
        "subtype": "init",
        "session_id": "test-123",
        "model": "claude-haiku-4-5-20251001",
    }

    stream_data = [json.dumps(system_event) + "\n"]

    mock_process.stdout.readline.side_effect = stream_data + [""]

    captured_stdout = io.StringIO()
    handler_id = logger.add(captured_stdout, format="{message}")
    try:
        claude_agent.generate("Test")
    finally:
        logger.remove(handler_id)

    # System events should be silent
    output = captured_stdout.getvalue()
    assert "system" not in output.lower()


def test_parse_text_streaming(claude_agent, mock_llm_subprocess):
    """Test that multiple text events accumulate correctly."""
    mock_popen, _ = mock_llm_subprocess
    mock_process = mock_popen.return_value
    mock_process.returncode = 0

    # Streaming: "Hello\n", "World"
    stream_data = [
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "Hello\n"}],
                    "role": "assistant",
                },
            }
        )
        + "\n",
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "World"}],
                    "role": "assistant",
                },
            }
        )
        + "\n",
    ]

    mock_process.stdout.readline.side_effect = stream_data + [""]

    captured_stdout = io.StringIO()
    handler_id = logger.add(captured_stdout, format="{message}")
    try:
        result = claude_agent.generate("Test")
    finally:
        logger.remove(handler_id)

    output = captured_stdout.getvalue()

    # Verify content accumulation
    assert result == "Hello\n\nWorld"

    # Verify prefixing
    assert "[Claude] Hello" in output
    assert "[Claude] World" in output


def test_parse_tool_use(claude_agent, mock_llm_subprocess):
    """Test that tool use events render with truncation and blue color."""
    mock_popen, _ = mock_llm_subprocess
    mock_process = mock_popen.return_value
    mock_process.returncode = 0

    long_params = {"data": "x" * 300}
    stream_data = [
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tool-123",
                            "name": "test_tool",
                            "input": long_params,
                        }
                    ],
                    "role": "assistant",
                },
            }
        )
        + "\n"
    ]

    mock_process.stdout.readline.side_effect = stream_data + [""]

    captured_stdout = io.StringIO()
    handler_id = logger.add(captured_stdout, format="{message}")
    try:
        claude_agent.generate("Test")
    finally:
        logger.remove(handler_id)

    output = captured_stdout.getvalue()

    # Verify tool use prefix and truncation (semantic content, not ANSI codes)
    assert "[Claude]" in output
    assert "[Tool Use]" in output
    assert "..." in output
    assert "x" * 300 not in output


def test_parse_tool_result(claude_agent, mock_llm_subprocess):
    """Test that tool results render with green color and emit events."""
    mock_popen, _ = mock_llm_subprocess
    mock_process = mock_popen.return_value
    mock_process.returncode = 0

    tool_id = "tool-456"
    stream_data = [
        # First tool use maps ID to name
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": tool_id,
                            "name": "read_file",
                            "input": {"path": "/test.txt"},
                        }
                    ],
                    "role": "assistant",
                },
            }
        )
        + "\n",
        # Then result uses that ID
        json.dumps(
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": "File contents here",
                            "is_error": False,
                        }
                    ],
                    "role": "user",
                },
            }
        )
        + "\n",
    ]

    mock_process.stdout.readline.side_effect = stream_data + [""]

    captured_stdout = io.StringIO()
    handler_id = logger.add(captured_stdout, format="{message}")
    try:
        claude_agent.generate("Test")
        claude_agent.mock_event_handler.on_tool_result.assert_called_once()
        call_args = claude_agent.mock_event_handler.on_tool_result.call_args
        assert call_args.kwargs["tool"] == "read_file"
        assert call_args.kwargs["stdout"] == "File contents here"
    finally:
        logger.remove(handler_id)

    output = captured_stdout.getvalue()

    # Verify tool result prefix (semantic content, not ANSI codes)
    assert "[Claude]" in output
    assert "[Tool Result]" in output
    assert "File contents here" in output


def test_tool_result_mapping(claude_agent, mock_llm_subprocess):
    """Test that tool_use_id correctly maps to tool names."""
    mock_popen, _ = mock_llm_subprocess
    mock_process = mock_popen.return_value
    mock_process.returncode = 0

    tool_id = "tool-789"
    stream_data = [
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": tool_id,
                            "name": "custom_tool",
                            "input": {},
                        }
                    ],
                    "role": "assistant",
                },
            }
        )
        + "\n",
        json.dumps(
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": "Success",
                            "is_error": False,
                        }
                    ],
                    "role": "user",
                },
            }
        )
        + "\n",
    ]

    mock_process.stdout.readline.side_effect = stream_data + [""]

    captured_stdout = io.StringIO()
    handler_id = logger.add(captured_stdout, format="{message}")
    try:
        claude_agent.generate("Test")
        # Verify the correct tool name was used
        call_args = claude_agent.mock_event_handler.on_tool_result.call_args
        assert call_args.kwargs["tool"] == "custom_tool"
    finally:
        logger.remove(handler_id)


def test_final_result_returned(claude_agent, mock_llm_subprocess):
    """Test that result event's 'result' field becomes return value."""
    mock_popen, _ = mock_llm_subprocess
    mock_process = mock_popen.return_value
    mock_process.returncode = 0

    final_summary = "This is the final result summary"
    stream_data = [
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "Intermediate text"}],
                    "role": "assistant",
                },
            }
        )
        + "\n",
        json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "result": final_summary,
                "duration_ms": 1000,
            }
        )
        + "\n",
    ]

    mock_process.stdout.readline.side_effect = stream_data + [""]

    result = claude_agent.generate("Test")

    # Result should be the summary from result event, not intermediate text
    assert result == final_summary


def test_prefix_handling(claude_agent, mock_llm_subprocess):
    """Test that [Claude] prefix appears on new lines correctly."""
    mock_popen, _ = mock_llm_subprocess
    mock_process = mock_popen.return_value
    mock_process.returncode = 0

    stream_data = [
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "Line 1"}],
                    "role": "assistant",
                },
            }
        )
        + "\n",
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "\nLine 2"}],
                    "role": "assistant",
                },
            }
        )
        + "\n",
    ]

    mock_process.stdout.readline.side_effect = stream_data + [""]

    captured_stdout = io.StringIO()
    handler_id = logger.add(captured_stdout, format="{message}")
    try:
        claude_agent.generate("Test")
    finally:
        logger.remove(handler_id)

    output = captured_stdout.getvalue()

    # Both lines should have prefix
    lines = output.strip().split("\n")
    prefixed_lines = [line for line in lines if line.startswith("[Claude]")]
    assert len(prefixed_lines) >= 2


def test_multiline_streaming(claude_agent, mock_llm_subprocess):
    """Test that text with newlines preserves prefix behavior."""
    mock_popen, _ = mock_llm_subprocess
    mock_process = mock_popen.return_value
    mock_process.returncode = 0

    stream_data = [
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "Line 1\nLine 2\nLine 3"}],
                    "role": "assistant",
                },
            }
        )
        + "\n"
    ]

    mock_process.stdout.readline.side_effect = stream_data + [""]

    captured_stdout = io.StringIO()
    handler_id = logger.add(captured_stdout, format="{message}")
    try:
        claude_agent.generate("Test")
    finally:
        logger.remove(handler_id)

    output = captured_stdout.getvalue()

    # Each line should have its own prefix
    assert "[Claude] Line 1" in output
    assert "[Claude] Line 2" in output
    assert "[Claude] Line 3" in output


def test_non_json_fallback(claude_agent, mock_llm_subprocess):
    """Test that non-JSON lines are logged as-is."""
    mock_popen, _ = mock_llm_subprocess
    mock_process = mock_popen.return_value
    mock_process.returncode = 0

    stream_data = [
        "This is not JSON\n",
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "Valid JSON"}],
                    "role": "assistant",
                },
            }
        )
        + "\n",
    ]

    mock_process.stdout.readline.side_effect = stream_data + [""]

    captured_stdout = io.StringIO()
    handler_id = logger.add(captured_stdout, format="{message}")
    try:
        claude_agent.generate("Test")
    finally:
        logger.remove(handler_id)

    output = captured_stdout.getvalue()

    # Non-JSON line should appear with prefix
    assert "[Claude] This is not JSON" in output
    # Valid JSON should also be processed
    assert "[Claude] Valid JSON" in output


def test_empty_tool_result(claude_agent, mock_llm_subprocess):
    """Test that empty tool results show success message."""
    mock_popen, _ = mock_llm_subprocess
    mock_process = mock_popen.return_value
    mock_process.returncode = 0

    tool_id = "tool-empty"
    stream_data = [
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": tool_id,
                            "name": "write_file",
                            "input": {"path": "/test.txt"},
                        }
                    ],
                    "role": "assistant",
                },
            }
        )
        + "\n",
        json.dumps(
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": "",
                            "is_error": False,
                        }
                    ],
                    "role": "user",
                },
            }
        )
        + "\n",
    ]

    mock_process.stdout.readline.side_effect = stream_data + [""]

    captured_stdout = io.StringIO()
    handler_id = logger.add(captured_stdout, format="{message}")
    try:
        claude_agent.generate("Test")
    finally:
        logger.remove(handler_id)

    output = captured_stdout.getvalue()

    # Empty result should show success message with tool name (semantic check)
    assert "[Claude]" in output
    assert "write_file ran successfully" in output


def test_tool_result_event_emission(claude_agent, mock_llm_subprocess):
    """Test that tool calls are recorded with correct args and duration."""
    mock_popen, _ = mock_llm_subprocess
    mock_process = mock_popen.return_value
    mock_process.returncode = 0

    tool_id = "tool-traj"
    tool_args = {"command": "ls -la", "timeout": 30}
    stream_data = [
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": tool_id,
                            "name": "bash",
                            "input": tool_args,
                        }
                    ],
                    "role": "assistant",
                },
            }
        )
        + "\n",
        json.dumps(
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": "total 0",
                            "is_error": False,
                        }
                    ],
                    "role": "user",
                },
            }
        )
        + "\n",
    ]

    mock_process.stdout.readline.side_effect = stream_data + [""]

    claude_agent.generate("Test")

    # Verify event emission
    claude_agent.mock_event_handler.on_tool_call.assert_called_once_with("bash", tool_args)
    claude_agent.mock_event_handler.on_tool_result.assert_called_once()
    call_args = claude_agent.mock_event_handler.on_tool_result.call_args[1]

    assert call_args["tool"] == "bash"
    assert call_args["stdout"] == "total 0"
    assert call_args["duration"] is not None
    assert call_args["duration"] >= 0


def test_parse_real_fixture(claude_agent, mock_llm_subprocess):
    """Test parsing the real fixture file."""
    # Fix path since we moved the test file to tests/unit/llm
    fixture_path = Path(__file__).parents[2] / "fixtures" / "claude" / "example_stream_json.txt"

    if not fixture_path.exists():
        pytest.skip(f"Fixture file not found at {fixture_path}")

    mock_popen, _ = mock_llm_subprocess
    mock_process = mock_popen.return_value
    mock_process.returncode = 0

    with open(fixture_path) as f:
        stream_lines = f.readlines()

    mock_process.stdout.readline.side_effect = stream_lines + [""]

    result = claude_agent.generate("Test")

    # Verify all lines parse without errors
    assert result is not None

    # The fixture has a result event at the end
    assert "AI-native infrastructure automation system" in result

    # Count tool results (should be 38 based on fixture)
    assert claude_agent.mock_event_handler.on_tool_result.call_count == 38

    # Count text events by checking if result contains multiple parts
    # The fixture should have processed text events successfully
    assert len(result) > 100  # Result should be substantial
