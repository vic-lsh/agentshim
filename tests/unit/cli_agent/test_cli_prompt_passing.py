"""
Test that all CLI agents properly pass the prompt to their underlying CLI tools.

This test suite ensures that each agent type (Claude, Codex, Gemini, Opencode)
correctly passes the user's prompt either as a command-line argument or via stdin,
depending on the CLI tool's requirements.

Design:
- Claude and Opencode: Pass prompt as command-line argument
- Codex and Gemini: Pass prompt via stdin
"""

from unittest.mock import MagicMock, patch

import pytest

from agentshim.claude import ClaudeCodeCodingAgent
from agentshim.codex import CodexCodingAgent
from agentshim.copilot import CopilotCodingAgent
from agentshim.gemini import GeminiCodingAgent
from agentshim.opencode import OpencodeCodingAgent


@pytest.fixture
def mock_which():
    """Mock shutil.which to return a fake binary path."""

    def which_impl(name, path=None):
        return f"/usr/bin/{name}"

    return which_impl


@pytest.fixture
def mock_process():
    """Create a reusable mock process with standard behavior."""
    process = MagicMock()
    process.pid = 12345
    process.returncode = 0
    process.stdout.readline.side_effect = lambda: ""
    process.stderr.readline.side_effect = lambda: ""
    process.stdin.write = MagicMock()
    process.wait.return_value = 0
    process.poll.return_value = 0
    return process


@pytest.fixture
def command_tracker(mock_process):
    """Track subprocess calls and return captured commands and stdin writes."""
    captured_commands = []
    stdin_writes = []

    def track_write(data):
        stdin_writes.append(data)

    mock_process.stdin.write = track_write

    def track_popen(*args, **kwargs):
        captured_commands.append(args[0] if args else [])
        return mock_process

    return track_popen, captured_commands, stdin_writes


@pytest.fixture(
    params=[
        # (name, class, has_prompt_in_cmd)
        ("claude", ClaudeCodeCodingAgent, True),
        ("copilot", CopilotCodingAgent, True),
        ("codex", CodexCodingAgent, False),
        ("gemini", GeminiCodingAgent, False),
        ("opencode", OpencodeCodingAgent, True),
    ],
    ids=["claude", "copilot", "codex", "gemini", "opencode"],
)
def agent_info(request):
    """Parameterized fixture for all agent types with prompt passing info."""
    return request.param


# ============================================================================
# PROMPT PASSING IN COMMAND LINE TESTS
# ============================================================================


def test_agent_includes_prompt_in_command_when_required(agent_info, mock_which, command_tracker):
    """Test agents that require prompt in command line include it."""
    agent_name, agent_class, has_prompt_in_cmd = agent_info
    test_prompt = "Write a hello world function"
    track_popen, captured_commands, _ = command_tracker

    with patch("shutil.which", side_effect=mock_which):
        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            with patch("subprocess.Popen", side_effect=track_popen):
                agent = agent_class()
                agent.generate(test_prompt, silent=True)

    assert len(captured_commands) > 0
    command = captured_commands[0]

    if has_prompt_in_cmd:
        # For Claude and Opencode, prompt should be in the command
        cmd_str = " ".join(command)
        assert test_prompt in cmd_str, (
            f"{agent_name} should include prompt '{test_prompt}' in command, but got: {cmd_str}"
        )


def test_agent_passes_prompt_via_stdin(agent_info, mock_which, command_tracker):
    """Test all agents pass prompt via stdin."""
    agent_name, agent_class, _ = agent_info
    test_prompt = "Create a function to sort a list"
    track_popen, _, stdin_writes = command_tracker

    with patch("shutil.which", side_effect=mock_which):
        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            with patch("subprocess.Popen", side_effect=track_popen):
                agent = agent_class()
                agent.generate(test_prompt, silent=True)

    # All agents should write to stdin
    assert len(stdin_writes) > 0, f"{agent_name} should write prompt to stdin"
    assert test_prompt in stdin_writes[0], (
        f"{agent_name} should pass prompt '{test_prompt}' to stdin, but got: {stdin_writes}"
    )


# ============================================================================
# COMMAND CONSTRUCTION TESTS
# ============================================================================


def test_claude_command_structure(mock_which, command_tracker):
    """Test Claude agent constructs correct command with prompt."""
    test_prompt = "Write a test function"
    track_popen, captured_commands, _ = command_tracker

    with patch("shutil.which", side_effect=mock_which):
        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            with patch("subprocess.Popen", side_effect=track_popen):
                agent = ClaudeCodeCodingAgent()
                agent.generate(test_prompt, silent=True)

    assert len(captured_commands) > 0
    cmd = captured_commands[0]

    # Check expected flags
    assert "-p" in cmd, "Claude should use -p flag for print mode"
    assert "--dangerously-skip-permissions" in cmd
    assert "--output-format" in cmd
    assert "stream-json" in cmd
    assert "--verbose" in cmd

    # Check prompt is in command
    cmd_str = " ".join(cmd)
    assert test_prompt in cmd_str


def test_codex_command_structure(mock_which, command_tracker):
    """Test Codex agent constructs correct command."""
    test_prompt = "Write a test function"
    track_popen, captured_commands, _ = command_tracker

    with patch("shutil.which", side_effect=mock_which):
        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            with patch("subprocess.Popen", side_effect=track_popen):
                agent = CodexCodingAgent()
                agent.generate(test_prompt, silent=True)

    assert len(captured_commands) > 0
    cmd = captured_commands[0]

    # Check expected command structure
    assert "codex" in cmd[0]  # Binary name
    assert "exec" in cmd
    assert "--dangerously-bypass-approvals-and-sandbox" in cmd


def test_copilot_command_structure(mock_which, command_tracker):
    """Test Copilot agent constructs correct command with prompt."""
    test_prompt = "Write a test function"
    track_popen, captured_commands, _ = command_tracker

    with patch("shutil.which", side_effect=mock_which):
        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            with patch("subprocess.Popen", side_effect=track_popen):
                agent = CopilotCodingAgent()
                agent.generate(test_prompt, silent=True)

    assert len(captured_commands) > 0
    cmd = captured_commands[0]

    assert "--output-format" in cmd
    assert "json" in cmd
    assert "--allow-all-tools" in cmd
    assert "--allow-all-paths" in cmd
    assert "--allow-all-urls" in cmd
    assert "-p" in cmd
    assert test_prompt in cmd


def test_gemini_command_structure(mock_which, command_tracker):
    """Test Gemini agent constructs correct command."""
    test_prompt = "Write a test function"
    track_popen, captured_commands, _ = command_tracker

    with patch("shutil.which", side_effect=mock_which):
        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            with patch("subprocess.Popen", side_effect=track_popen):
                agent = GeminiCodingAgent()
                agent.generate(test_prompt, silent=True)

    assert len(captured_commands) > 0
    cmd = captured_commands[0]

    # Check expected flags
    assert "-y" in cmd  # YOLO mode
    assert "-o" in cmd
    assert "stream-json" in cmd


def test_opencode_command_structure(mock_which, command_tracker):
    """Test Opencode agent constructs correct command with prompt."""
    test_prompt = "Write a test function"
    track_popen, captured_commands, _ = command_tracker

    with patch("shutil.which", side_effect=mock_which):
        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            with patch("subprocess.Popen", side_effect=track_popen):
                agent = OpencodeCodingAgent()
                agent.generate(test_prompt, silent=True)

    assert len(captured_commands) > 0
    cmd = captured_commands[0]

    # Check expected command structure
    assert "opencode" in cmd[0]  # Binary name
    assert "run" in cmd
    assert "--format=json" in cmd

    # Check prompt is in command
    cmd_str = " ".join(cmd)
    assert test_prompt in cmd_str


# ============================================================================
# MODEL PARAMETER TESTS
# ============================================================================


def test_agent_includes_model_in_command_when_specified(agent_info, mock_which, command_tracker):
    """Test agents include --model flag when model is specified."""
    agent_name, agent_class, _ = agent_info
    test_model = "custom-model-v1"
    track_popen, captured_commands, _ = command_tracker

    with patch("shutil.which", side_effect=mock_which):
        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            with patch("subprocess.Popen", side_effect=track_popen):
                # OpencodeCodingAgent has a default model, so skip for this test
                if agent_class == OpencodeCodingAgent:
                    pytest.skip("Opencode has default model behavior")

                agent = agent_class(model=test_model)
                agent.generate("test prompt", silent=True)

    assert len(captured_commands) > 0
    cmd = captured_commands[0]

    # All agents should include --model flag when specified
    assert "--model" in cmd, f"{agent_name} should include --model flag"
    # Find the index of --model and check the next element is our model
    model_idx = cmd.index("--model")
    assert cmd[model_idx + 1] == test_model, (
        f"{agent_name} should use model '{test_model}', but got: {cmd[model_idx + 1]}"
    )


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


def test_agent_handles_multiline_prompts(agent_info, mock_which, command_tracker):
    """Test agents correctly handle prompts with multiple lines."""
    agent_name, agent_class, _ = agent_info
    test_prompt = """Write a function that:
1. Takes a list of numbers
2. Filters even numbers
3. Returns the sum"""
    track_popen, _, stdin_writes = command_tracker

    with patch("shutil.which", side_effect=mock_which):
        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            with patch("subprocess.Popen", side_effect=track_popen):
                agent = agent_class()
                agent.generate(test_prompt, silent=True)

    # Verify the full multiline prompt is passed via stdin
    assert len(stdin_writes) > 0
    assert test_prompt in stdin_writes[0], f"{agent_name} should pass full multiline prompt via stdin"


def test_agent_handles_special_characters_in_prompt(agent_info, mock_which, command_tracker):
    """Test agents handle prompts with special characters."""
    agent_name, agent_class, _ = agent_info
    test_prompt = "Fix bug in \"auth.js\" where user's password isn't validated"
    track_popen, _, stdin_writes = command_tracker

    with patch("shutil.which", side_effect=mock_which):
        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            with patch("subprocess.Popen", side_effect=track_popen):
                agent = agent_class()
                agent.generate(test_prompt, silent=True)

    # Verify prompt with special characters is passed correctly
    assert len(stdin_writes) > 0
    assert test_prompt in stdin_writes[0], f"{agent_name} should handle special characters in prompt"


def test_agent_handles_empty_prompt(agent_info, mock_which, command_tracker):
    """Test agents handle empty prompts gracefully."""
    agent_name, agent_class, _ = agent_info
    test_prompt = ""
    track_popen, _, stdin_writes = command_tracker

    with patch("shutil.which", side_effect=mock_which):
        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            with patch("subprocess.Popen", side_effect=track_popen):
                agent = agent_class()
                # Should not raise an exception
                agent.generate(test_prompt, silent=True)

    # Verify stdin write was attempted even with empty prompt
    assert len(stdin_writes) > 0


def test_claude_prompt_not_double_quoted(mock_which, command_tracker):
    """Test that Claude command does not wrap the prompt in extra quotes.

    Regression test for issue #20: subprocess.run with a list passes
    arguments directly, so wrapping in f'"{prompt}"' adds literal quote
    characters around the prompt.
    """
    test_prompt = "Deploy the application"
    track_popen, captured_commands, _ = command_tracker

    with patch("shutil.which", side_effect=mock_which):
        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            with patch("subprocess.Popen", side_effect=track_popen):
                agent = ClaudeCodeCodingAgent()
                agent.generate(test_prompt, silent=True)

    assert len(captured_commands) > 0
    cmd = captured_commands[0]

    # The prompt should appear as a bare element, not wrapped in quotes
    assert test_prompt in cmd, "Prompt should be in command list"
    assert f'"{test_prompt}"' not in cmd, "Prompt should not be wrapped in extra quotes"
