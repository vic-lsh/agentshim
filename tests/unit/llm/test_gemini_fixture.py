import io
from pathlib import Path
from unittest.mock import patch

import pytest
from loguru import logger

from agentshim.gemini import GeminiCodingAgent

# Define the fixture path relative to this test file or project root
FIXTURE_PATH = Path("tests/fixtures/gemini/example_stream_json.txt")


@pytest.fixture
def mock_env():
    """Mock environment for agent initialization."""
    return {"PATH": "/usr/bin:/bin", "HOME": "/tmp", "USER": "test"}


@pytest.fixture
def gemini_agent(mock_llm_subprocess, mock_env):
    """Create a GeminiCodingAgent instance with mocked environment."""
    mock_popen, mock_which = mock_llm_subprocess
    mock_which.return_value = "/usr/bin/gemini"

    with patch(
        "agentshim.cli_agent.get_interactive_env",
        return_value=mock_env,
    ):
        with patch("agentshim.cli_agent.CLICodingAgent._check_cli"):
            agent = GeminiCodingAgent()
            yield agent


def test_generate_from_fixture(gemini_agent, mock_llm_subprocess):
    """Test parsing a real stream dump from a fixture file."""

    if not FIXTURE_PATH.exists():
        pytest.skip(f"Fixture file not found at {FIXTURE_PATH}")

    # Read the fixture file
    with open(FIXTURE_PATH) as f:
        fixture_lines = f.readlines()

    # Mock the process output
    mock_popen, _ = mock_llm_subprocess
    mock_process = mock_popen.return_value
    mock_process.returncode = 0
    # readline side effect needs to return each line, then empty string to signal EOF
    mock_process.stdout.readline.side_effect = fixture_lines + [""]

    # Capture stdout to verify rendering
    captured_stdout = io.StringIO()
    handler_id = logger.add(captured_stdout, format="{message}")
    try:
        result = gemini_agent.generate("Test prompt")
    finally:
        logger.remove(handler_id)

    output = captured_stdout.getvalue()

    # Verify key interactions from the log

    # 1. Prefixing on text content
    assert "[Gemini] I will start by analyzing" in output

    # 2. Tool Use
    assert "[Gemini]" in output
    assert "[Tool Use] read_file" in output
    assert "docker-compose.yml" in output

    # 3. Tool Result -- empty output shows success message
    assert "read_file ran successfully" in output

    # 4. Another Tool Result with content
    assert "[Tool Result] Found 1 matching file(s)" in output

    # 5. Verify accumulation of content
    # The agent accumulates the 'content' fields from 'message' events
    assert "I will start by analyzing" in result
    assert "I will check the configuration" in result

    # 6. Verify non-JSON lines (YOLO mode) are printed
    assert "[Gemini] YOLO mode is enabled." in output
